"""
method 확정 전 확인 2가지:
  (1) cv2 기전: cv2는 box regression이라 유사도(cv4) 경로가 아닌데 cv2-FP가 held-out을 -12% 낮췄다.
      → cv2-FP가 '실제로 유사도 행렬을 바꾸는가'를 직접 측정. FP naive와 cv2-FP의 sim 행렬을
        같은 이미지에서 비교해 최대/평균 절대차를 본다.
        · 차이 ≈ 0 이면: cv2는 유사도에 무관, held-out -12%는 measurement 노이즈/간접효과 → 방법에서 제외.
        · 차이 유의미면: cv2가 실제로 결정에 영향(공유 경로/재파라미터화 등) → 조사·반영.
  (2) cv3 activation bit knee: A8(=naive) ~ A13에서 held-out 회복이 어디서 포화되는가.
      → 최소 여유 bit 결정(효율 최적점).

학습 없음(빠름).

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/24_method_confirm.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --calib 32 --eval 300 --seeds 0 1 2 --device 0
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness
from src.quant.quant_model import wrap_convs, calibrate
from src.quant.fake_quant import QuantConv2d
from src.quant.pdquant import _find_head


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

def branch_convs(det_model, branch):
    head=_find_head(det_model); br=getattr(head,branch,None)
    return [] if br is None else [m for m in br.modules() if isinstance(m,QuantConv2d)]

def build(model_cls, w, names, device, calib, skip=None, bit_override=None):
    m=model_cls(w); m.set_classes(names); m.fuse()
    wrap_convs(m.model, 8, 8, skip_names=skip or set())
    if bit_override:
        for br,(wb,ab) in bit_override.items():
            for c in branch_convs(m.model, br):
                if wb is not None: c.w_bits=wb
                if ab is not None: c.a_obs.bits=ab
    m.model.to(device).eval()
    calibrate(m.model, calib, device=device)
    return m

def heldout_flip(h_fp, h_q, probe, H_eval, conf=0.25):
    Hm=torch.zeros(80,dtype=torch.bool); Hm[H_eval]=True
    tot=fl=0
    for i,t in enumerate(probe):
        sf=h_fp.run_image(t,i).sim; sq=h_q.run_image(t,i).sim
        prob=sf.sigmoid(); mp,c_fp=prob.max(-1); target=(mp>conf)&Hm[c_fp]
        if target.sum()==0: continue
        idx=target.nonzero(as_tuple=True)[0]
        fpK=sf.clone(); fpK[:,Hm]=-1e9; qK=sq.clone(); qK[:,Hm]=-1e9
        fl+=int((fpK.argmax(-1)[idx]!=qK.argmax(-1)[idx]).sum()); tot+=len(idx)
    return fl/max(tot,1)*100

def sim_delta(h_a, h_b, probe):
    """두 모델의 유사도 행렬 차이. 같은 이미지에서 max/mean 절대차(평균)."""
    mx=mn=0.0; n=0
    for i,t in enumerate(probe):
        sa=h_a.run_image(t,i).sim; sb=h_b.run_image(t,i).sim
        d=(sa-sb).abs()
        mx+=float(d.max()); mn+=float(d.mean()); n+=1
    return mx/max(n,1), mn/max(n,1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="yolov8s-world.pt")
    ap.add_argument("--coco-root",default="/data/taeho/coco_datasets")
    ap.add_argument("--calib",type=int,default=32)
    ap.add_argument("--eval",type=int,default=300)
    ap.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    ap.add_argument("--imgsz",type=int,default=640)
    ap.add_argument("--device",default="0")
    args=ap.parse_args()
    device=f"cuda:{args.device}" if args.device!="cpu" else "cpu"
    names=load_coco_names()

    from ultralytics import YOLOWorld
    imgs=sorted(glob.glob(os.path.join(args.coco_root,"val2017","*.jpg")))
    calib=[preprocess(p,args.imgsz,device) for p in imgs[:args.calib]]
    probe=[preprocess(p,args.imgsz,device) for p in imgs[args.calib:args.calib+args.eval]]

    print("[load] FP"); fp=YOLOWorld(args.model); fp.set_classes(names); fp.fuse(); fp.model.to(device).eval()
    h_fp=SimilarityHarness(fp.model,device=device)

    # ---- (1) cv2 기전: naive vs cv2-FP 의 유사도 행렬 차이 ----
    print("\n[1] cv2 기전 확인: cv2-FP가 유사도 행렬을 실제로 바꾸는가")
    nv=build(YOLOWorld,args.model,names,device,calib)                 # naive
    c2=build(YOLOWorld,args.model,names,device,calib,skip={"cv2"})    # cv2-FP
    h_nv=SimilarityHarness(nv.model,device=device); h_c2=SimilarityHarness(c2.model,device=device)
    mx,mn=sim_delta(h_nv,h_c2,probe[:50])
    # 참조 스케일: naive가 FP로부터 얼마나 벗어나는지(정상적 양자화 편차)
    rmx,rmn=sim_delta(h_fp,h_nv,probe[:50])
    print(f"    naive vs cv2-FP  : max|Δsim|≈{mx:.4f}  mean|Δsim|≈{mn:.5f}")
    print(f"    (참조) FP vs naive: max|Δsim|≈{rmx:.4f}  mean|Δsim|≈{rmn:.5f}")
    if mx < 1e-4:
        print("    → cv2-FP가 유사도를 거의 안 바꿈. held-out -12%는 아티팩트/간접효과 → 방법에서 cv2 제외.")
    else:
        print(f"    → cv2-FP가 유사도를 바꿈(참조 대비 {mx/max(rmx,1e-9)*100:.0f}%). cv2가 결정에 실제 영향 → 조사.")
    h_nv.close(); h_c2.close()

    # ---- (2) cv3 activation bit knee ----
    print("\n[2] cv3 activation bit knee (W8 고정, A만 스윕)")
    configs=[("naive (cv3 A8)", None)]
    for ab in [9,10,11,12,13]:
        configs.append((f"cv3 @ W8A{ab}", {"cv3":(8,ab)}))
    configs.append(("cv3-FP (상한)", "skip"))

    built=[]
    for label,bo in configs:
        if bo=="skip":
            m=build(YOLOWorld,args.model,names,device,calib,skip={"cv3"})
        else:
            m=build(YOLOWorld,args.model,names,device,calib,bit_override=bo)
        built.append((label, SimilarityHarness(m.model,device=device)))

    hdr=f"{'config':16s} |"
    for s in args.seeds: hdr+=f" seed{s:>2d} |"
    hdr+=f" {'mean':>6s} | {'Δ vs naive':>10s}"
    print(hdr); print("-"*76)
    base=None
    for label,h in built:
        vals=[]
        for s in args.seeds:
            rng=np.random.default_rng(s); H_eval=rng.permutation(80)[60:80].tolist()
            vals.append(heldout_flip(h_fp,h,probe,H_eval))
        mean=float(np.mean(vals)); base=mean if base is None else base
        d=(mean-base)/max(base,1e-9)*100
        row=f"{label:16s} |"+"".join(f" {v:>5.2f}% |" for v in vals)+f" {mean:>5.2f}% | {d:>+9.1f}%"
        print(row)
    for _,h in built: h.close()
    h_fp.close()
    print("-"*76)
    print("\n판독: A가 낮아질수록 회복이 언제 무너지는지 = 최소 여유 bit. 포화 지점이 method의 cv3 bit.")

if __name__=="__main__":
    main()
