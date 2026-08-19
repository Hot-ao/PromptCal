"""
2단계: SimilarityHarness가 head에서 뽑는 텐서가 진짜 region-prompt 유사도 행렬인지 검증.

진단 우선 접근:
  (1) head 구조를 찍는다 — 타입, 하위 모듈, cv4(contrastive) 존재 여부.
  (2) head 출력과 cv4 각 레벨 출력을 hook으로 잡아 shape를 전부 출력.
  (3) cv4에서 유사도 행렬 [anchors, num_prompts]를 조립.
  (4) 그 행렬의 전역 argmax 클래스가 모델 정식 예측의 최고신뢰 탐지 클래스와 일치하는지 대조.
      일치하면 = "우리가 가로챈 게 진짜 이 detector의 의사결정" 증명.

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/02_dump_similarity.py \
        --model yolov8s-worldv2.pt --coco-root /data/taeho/coco_datasets --device 0
"""

import argparse
import glob
import os
import torch


def pick_image(coco_root, explicit):
    if explicit:
        return explicit
    cands = sorted(glob.glob(os.path.join(coco_root, "val2017", "*.jpg")))
    if not cands:
        raise FileNotFoundError(f"val2017 이미지 없음: {coco_root}/val2017")
    return cands[0]


def describe(name, t, indent="  "):
    if torch.is_tensor(t):
        print(f"{indent}{name}: Tensor {tuple(t.shape)} dtype={t.dtype}")
    elif isinstance(t, (list, tuple)):
        print(f"{indent}{name}: {type(t).__name__}(len={len(t)})")
        for j, e in enumerate(t):
            describe(f"{name}[{j}]", e, indent + "  ")
    elif isinstance(t, dict):
        print(f"{indent}{name}: dict(keys={list(t.keys())})")
    else:
        print(f"{indent}{name}: {type(t).__name__}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-worldv2.pt")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--image", default=None)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLOWorld
    model = YOLOWorld(args.model)
    core = model.model                 # nn.Module (DetectionModel/WorldModel)
    head = core.model[-1]              # 마지막 모듈 = head

    print("=" * 60)
    print("[head] type      :", type(head).__name__)
    print("[head] children  :", [n for n, _ in head.named_children()])
    print("[head] nc        :", getattr(head, "nc", "?"))
    print("[head] reg_max   :", getattr(head, "reg_max", "?"))
    print("[head] has cv4   :", hasattr(head, "cv4"),
          ("(contrastive head 존재 → pre-sigmoid 유사도 직접 캡처 가능)"
           if hasattr(head, "cv4") else "(cv4 없음 → head 출력에서 슬라이스 필요)"))
    print("=" * 60)

    captured = {}
    handles = []

    def head_hook(mod, inp, out):
        captured["head_out"] = out
    handles.append(head.register_forward_hook(head_hook))

    if hasattr(head, "cv4"):
        for i, sub in enumerate(head.cv4):
            def make(idx):
                def h(mod, inp, out):
                    captured[f"cv4_{idx}"] = out
                return h
            handles.append(sub.register_forward_hook(make(i)))

    img = pick_image(args.coco_root, args.image)
    print("[image]", img)

    # 정식 예측(참조용) — 이 forward가 위 hook들을 채운다.
    results = model.predict(img, device=args.device, verbose=False)
    r = results[0]

    for h in handles:
        h.remove()

    print("\n--- captured shapes ---")
    for k in captured:
        describe(k, captured[k])

    # ---- cv4에서 유사도 행렬 조립 ----------------------------------------
    cv4_keys = sorted(k for k in captured if k.startswith("cv4_"))
    sim = None
    if cv4_keys:
        # 각 레벨: [B, num_prompts, H, W] → [B, num_prompts, H*W]
        parts = []
        for k in cv4_keys:
            t = captured[k]
            if not torch.is_tensor(t):
                print(f"[warn] {k}가 텐서가 아님, 스킵")
                continue
            B, P, H, W = t.shape
            parts.append(t.reshape(B, P, H * W))
        cat = torch.cat(parts, dim=2)          # [B, num_prompts, total_anchors]
        sim = cat[0].transpose(0, 1).contiguous()  # [total_anchors, num_prompts]
        print(f"\n[assembled] region-prompt 유사도 행렬(cv4): {tuple(sim.shape)} "
              f"= [anchors, num_prompts]")

    if sim is None:
        print("\n[!] cv4가 없어 head 출력에서 직접 슬라이스가 필요함. "
              "위 head_out shape를 보고 harness._locate_cls_logits를 맞춰야 함.")
        return

    num_prompts = sim.shape[1]
    print(f"[assembled] num_prompts = {num_prompts} (COCO면 80이어야 정상)")
    print(f"[assembled] value range : min={sim.min():.3f} max={sim.max():.3f} "
          f"(pre-sigmoid logit)")

    # ---- 의미 검증: 전역 최고점 앵커의 클래스 == 최고신뢰 탐지 클래스? -------
    scores = sim.sigmoid()
    flat = scores.reshape(-1)
    best = torch.argmax(flat).item()
    best_anchor, best_cls = divmod(best, num_prompts)

    print("\n--- 의미 검증 ---")
    print(f"[sim ] 전역 최고점 앵커={best_anchor}, 클래스idx={best_cls}, "
          f"score={scores[best_anchor, best_cls]:.3f}")

    if r.boxes is not None and len(r.boxes) > 0:
        conf = r.boxes.conf
        top = torch.argmax(conf).item()
        pred_cls = int(r.boxes.cls[top].item())
        names = r.names
        print(f"[pred] 최고신뢰 탐지 클래스idx={pred_cls} "
              f"({names.get(pred_cls, '?')}), conf={conf[top]:.3f}")
        match = (best_cls == pred_cls)
        print(f"\n[VERDICT] argmax 일치: {'OK ✅  hook이 올바른 유사도 행렬을 캡처함' if match else 'MISMATCH ⚠️  슬라이스/조립 로직 점검 필요'}")
        if not match:
            print("  (참고: NMS/멀티객체 때문에 전역 1등과 최고신뢰 탐지가 다를 수 있음. "
                  "그럴 땐 상위 몇 개 클래스 집합이 겹치는지로 판단.)")
            top5_cls = torch.unique(scores.argmax(dim=1)).tolist()
            print(f"  sim 상위 클래스 집합(일부): {top5_cls[:10]}")
            print(f"  예측 클래스 집합: {sorted(set(int(c) for c in r.boxes.cls.tolist()))}")
    else:
        print("[pred] 탐지 결과가 비어 다른 이미지로 재시도 권장")


if __name__ == "__main__":
    main()
