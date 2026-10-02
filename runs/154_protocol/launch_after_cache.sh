#!/bin/bash
# 보호 진단 캐시 3개가 다 생기면 GPU 0,4-7에 작업자를 띄운다(캐시를 여러 작업이 동시에 만들지 않도록).
cd /home/taeho/promptcal-ptq
for m in yolov8s-world yolov8s-worldv2 yolov8m-world; do
  until [ -f configs/protect_cache/${m}_n200_c256_img640_la16_aqattn_cls.json ]; do
    if ! pgrep -f "make_cache.py ${m}.pt" > /dev/null && [ ! -f configs/protect_cache/${m}_n200_c256_img640_la16_aqattn_cls.json ]; then
      echo "캐시 생성 실패: $m" >> runs/154_protocol/chain.log; exit 1; fi
    sleep 30; done
done
# taken.log는 m 기준선(B) 4개를 미리 표시해 둔 상태로 유지(작업자 B가 GPU 0·7에서 처리)
for spec in "4 0-23" "5 24-47" "6 48-71"; do set -- $spec; setsid runs/154_protocol/worker.sh $1 $2 > /dev/null 2>&1 & done
echo "작업자 시작 $(date '+%m-%d %H:%M:%S')" >> runs/154_protocol/chain.log
