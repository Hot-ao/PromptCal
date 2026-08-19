"""
v1 전환 관문: yolov8s-world (v1) head 구조 진단.

harness는 WorldDetect.cv4(contrastive head)에 hook을 걸어 유사도 행렬을 캡처한다.
v1은 head 구조가 v2와 다를 수 있으므로, harness가 v1에서 작동하는지 먼저 확인.

확인 항목:
  (1) v1 로드 + set_classes 되나
  (2) head 타입은? cv4가 있나? 있으면 개수/형태는?
  (3) cv4 hook으로 유사도 행렬 [anchors, P] 조립되나
  (4) 전역 argmax == 모델 정식 예측인가 (2단계 검증의 v1 재현)

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/00b_v1_head_diagnose.py \
        --coco-root /data/taeho/coco_datasets --device 0
"""

import argparse, glob, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_coco_names():
    import ultralytics, yaml
    from pathlib import Path
    d = yaml.safe_load(open(Path(ultralytics.__file__).parent/"cfg"/"datasets"/"coco.yaml"))
    return [d["names"][i] for i in range(len(d["names"]))]


def letterbox(im, new=640, color=(114,114,114)):
    h,w=im.shape[:2]; r=min(new/h,new/w); nh,nw=int(round(h*r)),int(round(w*r))
    im_r=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_LINEAR)
    top,left=(new-nh)//2,(new-nw)//2
    return cv2.copyMakeBorder(im_r,top,new-nh-top,left,new-nw-left,cv2.BORDER_CONSTANT,value=color)


def preprocess(path, imgsz, device):
    im=letterbox(cv2.imread(path),imgsz)
    im=np.ascontiguousarray(im[:,:,::-1].transpose(2,0,1))
    return torch.from_numpy(im).float().unsqueeze(0).to(device)/255.0


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-world.pt")   # ★ v1
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args=ap.parse_args()
    device=f"cuda:{args.device}" if args.device!="cpu" else "cpu"
    names=load_coco_names()

    from ultralytics import YOLOWorld
    print(f"[1] 로드: {args.model}")
    model=YOLOWorld(args.model)
    model.set_classes(names)
    print("    set_classes(80) OK")
    model.fuse()
    core=model.model
    head=core.model[-1] if hasattr(core,"model") else core[-1]
    print(f"\n[2] head 타입: {type(head).__name__}")
    print(f"    head 하위 모듈: {[n for n,_ in head.named_children()]}")
    has_cv4 = hasattr(head, "cv4")
    print(f"    cv4 존재: {has_cv4}")
    if has_cv4:
        print(f"    cv4 타입: {type(head.cv4).__name__}, 개수: {len(head.cv4)}")
        for i, sub in enumerate(head.cv4):
            print(f"      cv4[{i}]: {type(sub).__name__}")
    else:
        print("    ⚠️ cv4 없음 — harness 캡처 지점을 새로 찾아야 함.")
        # txt_feats나 다른 contrastive 관련 속성 탐색
        for attr in ["txt_feats","cv2","cv3","cv4","embed","emb"]:
            print(f"      has {attr}: {hasattr(head, attr)}")

    model.model.to(device).eval()

    # [3] cv4 hook으로 유사도 캡처 시도
    if has_cv4:
        print(f"\n[3] cv4 hook 유사도 캡처 시도")
        buf={}
        handles=[]
        for i,sub in enumerate(head.cv4):
            def mk(idx):
                def hook(_m,_i,o): buf[idx]=o
                return hook
            handles.append(sub.register_forward_hook(mk(i)))
        img=sorted(glob.glob(os.path.join(args.coco_root,"val2017","*.jpg")))[0]
        t=preprocess(img,args.imgsz,device)
        with torch.no_grad():
            res=model.model(t)
        for h in handles: h.remove()
        if buf:
            shapes=[tuple(buf[i].shape) for i in sorted(buf)]
            print(f"    cv4 출력 형태(레벨별): {shapes}")
            # 조립 [anchors, P]
            parts=[]
            for i in sorted(buf):
                B,P,H,W=buf[i].shape
                parts.append(buf[i].reshape(B,P,H*W))
            sim=torch.cat(parts,dim=2)[0].transpose(0,1)
            print(f"    조립된 유사도 행렬: {tuple(sim.shape)} (기대 [anchors, 80])")
            print(f"    value range: [{sim.min():.2f}, {sim.max():.2f}] (pre-sigmoid logit이어야)")
            # [4] 전역 argmax vs 모델 예측
            prob=sim.sigmoid(); maxp,amax=prob.max(-1)
            best=torch.argmax(maxp).item()
            print(f"\n[4] 최고 신뢰 앵커 → 클래스 '{names[int(amax[best])]}' (p={maxp[best]:.3f})")
            print("    (2단계처럼 모델 정식 예측과 일치하는지 육안 확인 필요)")
            print("\n" + "="*50)
            print(" v1 HARNESS 호환: OK ✅ — cv4 캡처·조립 정상")
            print(" → harness 그대로 사용 가능. baseline(37.4) 재현부터 진행.")
        else:
            print("    ⚠️ cv4 hook이 출력을 못 잡음 — forward 경로 확인 필요.")
    else:
        print("\n" + "="*50)
        print(" v1 HARNESS 호환: 수정 필요 ⚠️")
        print(" → cv4가 없으므로 harness 캡처 지점을 v1 구조에 맞게 재설계해야 함.")


if __name__ == "__main__":
    main()
