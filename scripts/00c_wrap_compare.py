"""
v1 AP 대폭 하락 원인 규명: v1 vs v2 래핑 대상 Conv2d 비교.

가설: v1은 71개, v2는 68개 래핑. 추가 3개가 cv4(ContrastiveHead) 등 민감한 위치라면
유사도 계산이 양자화되어 AP가 과하게 하락. 그 경우 해당 conv를 제외해야 공정.

각 모델에서 래핑 대상 Conv2d의 전체 경로를 나열하고, cv4/head 내부인지 표시.
v1에만 있는(=v2에 없는) conv를 골라냄.

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/00c_wrap_compare.py --device 0
"""

import argparse, sys, os
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_coco_names():
    import ultralytics, yaml
    from pathlib import Path
    d = yaml.safe_load(open(Path(ultralytics.__file__).parent/"cfg"/"datasets"/"coco.yaml"))
    return [d["names"][i] for i in range(len(d["names"]))]


def collect_conv_paths(module, prefix="", under_dfl=False):
    """Conv2d 경로 수집. DFL 하위는 under_dfl=True로 표시(wrap에서 제외되는 것)."""
    out = []
    is_dfl = type(module).__name__ == "DFL"
    for name, child in module.named_children():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Conv2d):
            out.append((path, under_dfl or is_dfl, type(module).__name__))
        else:
            out.extend(collect_conv_paths(child, path, under_dfl or is_dfl))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    names = load_coco_names()

    from ultralytics import YOLOWorld
    results = {}
    for tag, w in [("v1", "yolov8s-world.pt"), ("v2", "yolov8s-worldv2.pt")]:
        m = YOLOWorld(w); m.set_classes(names); m.fuse()
        convs = collect_conv_paths(m.model)
        # wrap 대상 = DFL 아래가 아닌 Conv2d
        wrapped = [(p, parent) for (p, ud, parent) in convs if not ud]
        results[tag] = wrapped
        print(f"\n[{tag}] 전체 Conv2d {len(convs)}개, 래핑 대상(DFL 제외) {len(wrapped)}개")
        # cv4/head 내부 conv 표시
        head_convs = [(p, par) for (p, par) in wrapped
                      if "cv4" in p or "ContrastiveHead" in par or ".23." in p or p.startswith("model.23")]
        print(f"     이 중 cv4/ContrastiveHead 관련: {len(head_convs)}개")
        for p, par in head_convs:
            print(f"       {p}  (parent={par})")

    # v1에만 있는 래핑 대상 (경로 기준)
    v1p = set(p for p, _ in results["v1"])
    v2p = set(p for p, _ in results["v2"])
    only_v1 = sorted(v1p - v2p)
    only_v2 = sorted(v2p - v1p)
    print("\n" + "="*60)
    print(f" v1에만 있는 래핑 conv ({len(only_v1)}개):")
    for p in only_v1:
        par = dict(results["v1"]).get(p, "?")
        flag = "  ⚠️ cv4/head!" if ("cv4" in p or "Contrastive" in str(par)) else ""
        print(f"   {p}  (parent={par}){flag}")
    print(f"\n v2에만 있는 래핑 conv ({len(only_v2)}개):")
    for p in only_v2:
        print(f"   {p}")
    print("="*60)
    print("\n판정:")
    print(" - v1 추가 conv가 cv4/ContrastiveHead면 → 유사도 계산이 양자화됨 =")
    print("   AP 대폭 하락의 원인. v2처럼 제외하면 공정 비교 가능.")
    print(" - 추가 conv가 일반 backbone/neck이면 → v1 본질적 취약성. 프레이밍 조정 필요.")


if __name__ == "__main__":
    main()
