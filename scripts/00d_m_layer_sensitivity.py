"""
m 양자화 −13.7 원인 규명: 레이어별 양자화 민감도.

방법: 한 번에 conv를 하나씩만 양자화하고 나머지는 FP로 두고 AP(빠른 근사) 측정.
     특정 레이어 하나가 AP를 크게 떨어뜨리면 그 레이어가 outlier(범인).
     손상이 고루 퍼져있으면 m 본질적 취약성.

빠른 근사 AP: 전체 val 대신 소수 이미지에서 FP 대비 예측 유지율(신뢰 영역 flip)로 프록시.
정확한 AP가 필요하면 --full로 pycocotools.

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/00d_m_layer_sensitivity.py \
        --model yolov8m-world.pt --coco-root /data/taeho/coco_datasets \
        --calib 64 --probe 100 --device 0
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness
from src.quant.fake_quant import QuantConv2d
from src.quant.quant_model import calibrate


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


def conv_paths(module, prefix="", under_dfl=False):
    out=[]
    is_dfl=type(module).__name__=="DFL"
    for name,child in module.named_children():
        path=f"{prefix}.{name}" if prefix else name
        if isinstance(child,nn.Conv2d) and not (under_dfl or is_dfl):
            out.append(path)
        elif not isinstance(child,nn.Conv2d):
            out.extend(conv_paths(child,path,under_dfl or is_dfl))
    return out


def get_module(root, path):
    m=root
    for p in path.split("."):
        m=m[int(p)] if p.isdigit() else getattr(m,p)
    return m

def set_module(root, path, val):
    parts=path.split("."); m=root
    for p in parts[:-1]:
        m=m[int(p)] if p.isdigit() else getattr(m,p)
    last=parts[-1]
    if last.isdigit(): m[int(last)]=val
    else: setattr(m,last,val)


def flip_proxy(h_fp, h_q, probe_tensors, conf=0.25):
    """FP 대비 양자화 모델의 confident-region top-1 flip률(낮을수록 손상 적음)."""
    tot=fl=0
    for i,t in enumerate(probe_tensors):
        sf=h_fp.run_image(t,i).sim; sq=h_q.run_image(t,i).sim
        p=sf.sigmoid(); mp,c=p.max(-1); m=mp>conf
        if m.sum()==0: continue
        idx=m.nonzero(as_tuple=True)[0]
        fl+=int((sq.argmax(-1)[idx]!=c[idx]).sum()); tot+=len(idx)
    return fl/max(tot,1)*100


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8m-world.pt")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--probe", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--topk", type=int, default=15, help="가장 민감한 레이어 상위 K개 출력")
    args=ap.parse_args()
    device=f"cuda:{args.device}" if args.device!="cpu" else "cpu"
    names=load_coco_names()

    from ultralytics import YOLOWorld
    imgs=sorted(glob.glob(os.path.join(args.coco_root,"val2017","*.jpg")))
    calib=[preprocess(p,args.imgsz,device) for p in imgs[:args.calib]]
    probe=[preprocess(p,args.imgsz,device) for p in imgs[args.calib:args.calib+args.probe]]

    # FP 레퍼런스
    fp=YOLOWorld(args.model); fp.set_classes(names); fp.fuse(); fp.model.to(device).eval()
    h_fp=SimilarityHarness(fp.model,device=device)

    paths=conv_paths(fp.model)
    print(f"[진단] {args.model}: wrap 대상 conv {len(paths)}개")
    print(f"[진단] 레이어 하나씩만 양자화 → confident flip 프록시 측정 ({args.probe} probe)\n")

    # 전체 양자화 기준선
    def build_full():
        m=YOLOWorld(args.model); m.set_classes(names); m.fuse()
        from src.quant.quant_model import wrap_convs
        wrap_convs(m.model,8,8); m.model.to(device).eval()
        calibrate(m.model,calib,device=device)
        return m
    full=build_full(); h_full=SimilarityHarness(full.model,device=device)
    base=flip_proxy(h_fp,h_full,probe); h_full.close()
    print(f"[기준] 전체 양자화 시 confident flip: {base:.2f}%\n")

    # 레이어별: 그 레이어 하나만 양자화
    results=[]
    for k,path in enumerate(paths):
        m=YOLOWorld(args.model); m.set_classes(names); m.fuse()
        orig=get_module(m.model,path)
        qc=QuantConv2d(orig,8,8); set_module(m.model,path,qc)
        m.model.to(device).eval()
        # 이 레이어만 calibrate
        qc.calibrating=True
        with torch.no_grad():
            for t in calib[:32]: m.model(t)
        qc.a_obs.freeze(); qc.calibrating=False; qc.quantized=True
        h_q=SimilarityHarness(m.model,device=device)
        f=flip_proxy(h_fp,h_q,probe); h_q.close()
        results.append((path,f))
        if (k+1)%10==0: print(f"  [{k+1}/{len(paths)}] 진행...")
    h_fp.close()

    results.sort(key=lambda x:-x[1])
    print(f"\n{'='*66}")
    print(f" 레이어별 단독 양자화 민감도 (confident flip %, 높을수록 치명적)")
    print(f"{'='*66}")
    print(f" 전체 양자화 기준선: {base:.2f}%")
    print(f"{'-'*66}")
    for path,f in results[:args.topk]:
        bar="█"*int(f/max(results[0][1],1e-9)*30)
        print(f" {f:>6.2f}%  {bar} {path}")
    print(f"{'='*66}")
    print("\n판정:")
    print(" - 소수 레이어가 압도적으로 높으면 → 그 레이어가 outlier(범인). 제외/per-channel로 해결.")
    print(" - 다수 레이어가 고루 높으면 → m 본질적 취약성. per-tensor W8A8 부적합.")


if __name__=="__main__":
    main()
