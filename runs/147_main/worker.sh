#!/bin/bash
# 2026-09-30: 주 실험 대기열 작업자. 사용법: worker.sh <gpu> <cores>
# queue.tsv에서 아직 안 가져간 첫 줄을 flock으로 가져가(taken.log에 기록) 실행한다. 실패하면 failed.log에 남긴다.
# 실행 전마다 해당 GPU에 다른 사용자 프로세스가 있으면 비워질 때까지 기다린다(공유 서버 규칙).
cd /home/taeho/promptcal-ptq
D=runs/147_main; G=$1; C=$2
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 YOLO_AUTOINSTALL=false
COMMON="--model yolov8s-world.pt --deterministic --calib 256 --no-skip-head --first-last-bits 8 \
  --combined-stage1 brecq --combined-recon-iters 2000"
BUS=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' -v g=$G '$1==g{print $2}')
while true; do
  line=$(flock $D/queue.lock bash -c "while IFS= read -r l; do t=\${l%%\$'\t'*}; grep -qx \"\$t\" $D/taken.log 2>/dev/null || { echo \"\$t\" >> $D/taken.log; echo \"\$l\"; break; }; done < $D/queue.tsv")
  [ -z "$line" ] && { echo "[gpu$G] 대기열 비었음 $(date +%H:%M:%S)" >> $D/chain.log; exit 0; }
  tag=${line%%$'\t'*}; args=${line#*$'\t'}
  others() {  # 이 GPU에서 taeho가 아닌 사용자의 프로세스 수
    nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | grep -i "$BUS" | \
      awk -F', ' '{print $1}' | xargs -r -I{} ps -o user= -p {} | awk '$1 != "taeho"' | wc -l
  }
  while [ "$(others)" -gt 0 ]; do sleep 120; done
  echo "$tag start gpu$G $(date '+%m-%d %H:%M:%S')" >> $D/chain.log
  nice -n 5 taskset -c $C .venv/bin/python pipeline/run_comparison.py $COMMON $args --device $G > $D/$tag.log 2>&1
  rc=$?
  echo "$tag done rc=$rc gpu$G $(date '+%m-%d %H:%M:%S')" >> $D/chain.log
  [ $rc -ne 0 ] && echo "$tag rc=$rc" >> $D/failed.log
done
