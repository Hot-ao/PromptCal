#!/bin/bash
# 리더보드 자동 갱신: runs/*/chain.log에 완료(done) 기록이 늘어날 때마다 LEADERBOARD.html을 다시 만든다(1분 간격 확인).
# 켜기: nohup setsid pipeline/scripts/leaderboard_watch.sh > /dev/null 2>&1 &     끄기: ps로 PID를 찾아 kill
cd "$(dirname "$0")/../.."
last=""
while true; do
  sig=$(cat runs/*/chain.log 2>/dev/null | grep -c " done ")
  if [ "$sig" != "$last" ]; then
    .venv/bin/python pipeline/scripts/make_leaderboard.py >> runs/leaderboard_watch.log 2>&1 && last=$sig
  fi
  sleep 60
done
