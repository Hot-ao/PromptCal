#!/bin/bash
# GPU 하나를 맡는 대기열 작업자. 사용법: pipeline/scripts/worker.sh <run_dir> <gpu> <cpu_cores>
#   <run_dir>/queue.tsv 의 각 줄: "<태그>\t<run_comparison.py 인자>" (예: --model yolov8s-world.pt --w-bits 4 --a-bits 5
#   --conditions naive,brecq,brecq+PM,brecq+GPM --seed 0). 여러 작업자가 같은 대기열을 flock으로 나눠 가진다.
#   다른 사용자가 이 GPU를 쓰고 있으면 빌 때까지 기다린다(공유 서버 규칙). 로그: <run_dir>/<태그>.log, chain.log, failed.log
cd "$(dirname "$0")/../.."
D=$1; G=$2; C=$3
source pipeline/scripts/protocol.sh
BUS=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' -v g=$G '$1==g{print $2}')
others() {
  nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | grep -i "$BUS" | \
    awk -F', ' '{print $1}' | xargs -r -I{} ps -o user= -p {} | awk -v me="$USER" '$1 != me' | wc -l
}
while true; do
  line=$(flock $D/queue.lock bash -c "while IFS= read -r l; do t=\${l%%\$'\t'*}; grep -qx \"\$t\" $D/taken.log 2>/dev/null || { echo \"\$t\" >> $D/taken.log; echo \"\$l\"; break; }; done < $D/queue.tsv")
  [ -z "$line" ] && { echo "[gpu$G] 대기열 비었음 $(date '+%m-%d %H:%M:%S')" >> $D/chain.log; exit 0; }
  tag=${line%%$'\t'*}; args=${line#*$'\t'}
  while [ "$(others)" -gt 0 ]; do sleep 120; done
  echo "$tag start gpu$G $(date '+%m-%d %H:%M:%S')" >> $D/chain.log
  nice -n 5 taskset -c $C .venv/bin/python pipeline/run_comparison.py $PROTOCOL $args --device $G > $D/$tag.log 2>&1
  rc=$?
  echo "$tag done rc=$rc gpu$G $(date '+%m-%d %H:%M:%S')" >> $D/chain.log
  [ $rc -ne 0 ] && echo "$tag rc=$rc" >> $D/failed.log
done
