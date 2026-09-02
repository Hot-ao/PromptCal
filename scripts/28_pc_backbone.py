"""
유저 아이디어 검증: head는 정밀도로 보호(학습 제외)하고, PromptCal(decision loss)을
backbone/neck에만 적용하면 held-out이 개선되나?

27번과의 차이: 27은 cv3@A11 위에 '전체' conv를 학습(head 포함) → anti-transfer 악화.
여기선 head(cv2·cv3 branch)를 학습에서 '빼고' backbone/neck만 학습. head 결정 geometry는
A11로 고정된 채, vocab-agnostic한 상류 feature scale만 PromptCal이 건드린다.

비교(같은 seed·같은 split):
  naive                 : 전부 W8A8
  Ours(cv3@A11)         : cv3 activation만 A11, 학습 없음 (현재 method)
  Ours + PC(backbone)   : cv3@A11 + PromptCal decision loss를 backbone/neck에만 (head 동결)
  [대조] Ours + PC(all) : cv3@A11 + PromptCal 전체(=27 재현, head 포함)

판정:
  - PC(backbone) < Ours (개선) → 유저 옳음. mixed-precision + PromptCal(backbone) 결합 method 성립.
  - PC(backbone) ≥ Ours (악화/무개선) → head를 빼도 backbone 학습조차 anti-transfer.
    → 더 강한 negative result, 학습 계열 종료 근거.

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/28_pc_backbone.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --calib 32 --eval 500 --seeds 0 1 2 --iters 150 --device 0
    # 전체(head 포함) 대조까지: --with-all
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness
from src.quant.quant_model import wrap_convs, calibrate
from src.quant.fake_quant import QuantConv2d
from src.quant.pdquant import _find_head
from src.quant.promptcal_v2 import (convert_to_learnable_scale, optimize_promptcal_v2,
                                    list_scale_convs, LearnableScaleQuantConv2d)

CV3_ABITS = 11


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

def head_convs(det):
    """head(cv2·cv3·cv4) 아래 conv 전체 — 학습에서 제외할 대상."""
    head=_find_head(det); out=[]
    for nm in ("cv2","cv3","cv4"):
        br=getattr(head,nm,None)
        if br is not None:
            out+=[m for m in br.modules() if isinstance(m,QuantConv2d)]
    return out

def cv3_convs(det):
    head=_find_head(det); br=getattr(head,"cv3",None)
    return [] if br is None else [m for m in br.modules() if isinstance(m,QuantConv2d)]

def build_static(model_cls, w, names, device, calib, ours_cv3):
    m=model_cls(w); m.set_classes(names); m.fuse()
    wrap_convs(m.model, 8, 8)
    if ours_cv3:
        for c in cv3_convs(m.model): c.a_obs.bits=CV3_ABITS
    m.model.to(device).eval(); calibrate(m.model, calib, device=device)
    return m

def build_pc(model_cls, w, names, device, calib, fp, removed, keep, iters, scope):
    """cv3@A11 + PromptCal. scope='backbone'이면 head 제외, 'all'이면 전체."""
    m=model_cls(w); m.set_classes(names); m.fuse()
    wrap_convs(m.model, 8, 8)
    for c in cv3_convs(m.model): c.a_obs.bits=CV3_ABITS       # head 정밀도 보호(A11)
    m.model.to(device).eval(); calibrate(m.model, calib, device=device)
    convert_to_learnable_scale(m.model)
    if scope=="backbone":
        scale_all=list_scale_convs(m.model)
        head_ids={id(c) for c in head_convs(m.model)}         # LearnableScaleQuantConv2d도 QuantConv2d 상속 → 매칭됨
        train=[c for c in scale_all if id(c) not in head_ids]
    else:
        train=None                                            # 전체(head 포함) = 27 재현
    optimize_promptcal_v2(m.model, fp.model, calib, device, removed, keep,
                          iters=iters, lr=3e-3, temp=1.0, train_convs=train, verbose=False)
    return m

def heldout_flip(h_fp, h_q, probe, H_eval, conf=0.25):
    Hm=torch.zeros(80,dtype=torch.bool); Hm[H_eval]=True; tot=fl=0
    for i,t in enumerate(probe):
        sf=h_fp.run_image(t,i).sim; sq=h_q.run_image(t,i).sim
        prob=sf.sigmoid(); mp,c_fp=prob.max(-1); target=(mp>conf)&Hm[c_fp]
        if target.sum()==0: continue
        idx=target.nonzero(as_tuple=True)[0]
        fpK=sf.clone(); fpK[:,Hm]=-1e9; qK=sq.clone(); qK[:,Hm]=-1e9
        fl+=int((fpK.argmax(-1)[idx]!=qK.argmax(-1)[idx]).sum()); tot+=len(idx)
    return fl/max(tot,1)*100

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="yolov8s-world.pt")
    ap.add_argument("--coco-root",default="/data/taeho/coco_datasets")
    ap.add_argument("--calib",type=int,default=32)
    ap.add_argument("--eval",type=int,default=500)
    ap.add_argument("--iters",type=int,default=150)
    ap.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    ap.add_argument("--with-all",action="store_true",help="head 포함 전체 학습(27 재현) 대조 추가")
    ap.add_argument("--imgsz",type=int,default=640)
    ap.add_argument("--device",default="0")
    args=ap.parse_args()
    device=f"cuda:{args.device}" if args.device!="cpu" else "cpu"
    names=load_coco_names()

    from ultralytics import YOLOWorld
    imgs=sorted(glob.glob(os.path.join(args.coco_root,"val2017","*.jpg")))
    calib=[preprocess(p,args.imgsz,device) for p in imgs[:args.calib]]
    probe=[preprocess(p,args.imgsz,device) for p in imgs[args.calib:args.calib+args.eval]]

    print("[build] FP"); fp=YOLOWorld(args.model); fp.set_classes(names); fp.fuse(); fp.model.to(device).eval()
    print("[build] naive"); nv=build_static(YOLOWorld,args.model,names,device,calib,ours_cv3=False)
    print("[build] Ours(cv3@A11)"); ou=build_static(YOLOWorld,args.model,names,device,calib,ours_cv3=True)
    # 학습 대상 개수 안내
    tmp=build_static(YOLOWorld,args.model,names,device,calib,ours_cv3=True); convert_to_learnable_scale(tmp.model)
    n_all=len(list_scale_convs(tmp.model)); n_head=len(head_convs(tmp.model)); del tmp
    print(f"    scale conv 총 {n_all}개 중 head {n_head}개 → backbone/neck {n_all-n_head}개만 학습")

    h_fp=SimilarityHarness(fp.model,device=device)
    h_nv=SimilarityHarness(nv.model,device=device)
    h_ou=SimilarityHarness(ou.model,device=device)

    cols=["naive","Ours(cv3@A11)","+PC(backbone)"]+(["+PC(all)"] if args.with_all else [])
    print("\n"+"="*90)
    print(" head 보호 + PromptCal(backbone만) vs 현재 method")
    print("="*90)
    hdr=f"{'seed':>5} |"+ "".join(f" {c:>14} |" for c in cols) + f" {'PC효과':>8}"
    print(hdr); print("-"*90)
    acc={c:[] for c in cols}
    for s in args.seeds:
        rng=np.random.default_rng(s); perm=rng.permutation(80)
        S=perm[:40].tolist(); H_cal=perm[40:60].tolist(); H_eval=perm[60:80].tolist()
        fn=heldout_flip(h_fp,h_nv,probe,H_eval); acc["naive"].append(fn)
        fo=heldout_flip(h_fp,h_ou,probe,H_eval); acc["Ours(cv3@A11)"].append(fo)
        pb=build_pc(YOLOWorld,args.model,names,device,calib,fp,H_cal,S,args.iters,"backbone")
        h_pb=SimilarityHarness(pb.model,device=device); fb=heldout_flip(h_fp,h_pb,probe,H_eval); h_pb.close()
        acc["+PC(backbone)"].append(fb)
        row=f"{s:>5} | {fn:>13.2f}% | {fo:>13.2f}% | {fb:>13.2f}% |"
        if args.with_all:
            pa=build_pc(YOLOWorld,args.model,names,device,calib,fp,H_cal,S,args.iters,"all")
            h_pa=SimilarityHarness(pa.model,device=device); fa=heldout_flip(h_fp,h_pa,probe,H_eval); h_pa.close()
            acc["+PC(all)"].append(fa); row+=f" {fa:>13.2f}% |"
        eff=(fb-fo)/max(fo,1e-9)*100      # +면 PC(backbone)이 Ours보다 나쁨
        row+=f" {eff:>+7.1f}%"
        print(row)
    h_fp.close(); h_nv.close(); h_ou.close()
    print("-"*90)
    mrow=f"{'mean':>5} |"+"".join(f" {np.mean(acc[c]):>13.2f}% |" for c in cols)
    eff=(np.mean(acc['+PC(backbone)'])-np.mean(acc['Ours(cv3@A11)']))/max(np.mean(acc['Ours(cv3@A11)']),1e-9)*100
    print(mrow+f" {eff:>+7.1f}%")
    print("="*90)
    print("\n판정:")
    print(" - +PC(backbone) < Ours(cv3@A11) 이고 seed 일관 → 유저 아이디어 성립(결합 method).")
    print(" - +PC(backbone) ≥ Ours → head를 빼도 backbone 학습조차 anti-transfer(더 강한 negative).")

if __name__=="__main__":
    main()
