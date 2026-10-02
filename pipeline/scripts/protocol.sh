# 확정 프로토콜(2026-10-02) 공통 인자. worker.sh가 source한다. 바꾸면 모든 결과가 바뀌므로 문서와 함께 갱신할 것.
#   head 포함, 첫·마지막 레이어 8bit, attention·contrastive matmul 8bit(attn_cls), head 마지막 conv 입력 A16,
#   결정적 실행, calibration train2017 앞 256장. 보호(P)·이전(M)·게이트 교환(G)은 조건 접미사로 고른다.
PROTOCOL="--deterministic --calib 256 --no-skip-head --first-last-bits 8 \
  --attn-quant attn_cls --last-abits 16 \
  --combined-stage1 brecq --combined-recon-iters 2000"
export CUDA_DEVICE_ORDER=PCI_BUS_ID YOLO_AUTOINSTALL=false OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
