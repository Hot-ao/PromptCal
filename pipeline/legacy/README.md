# pipeline/legacy — 더 이상 쓰지 않는 코드 (2026-10-02 정리)

| 파일 | 원래 용도 | 기각 이유 |
|---|---|---|
| `build_generic_vocab.py` | vocab-metric 재구성용 일반 어휘 생성 | vocab-metric 재구성(brecq_vm)이 W4A8 모든 변형에서 BRECQ보다 나빴다(09-28~29) |
| `diag_vocab_subspace.py` | 임베딩 부분공간 진단(vocab-metric 사전 분석) | 같은 방향의 진단이라 함께 은퇴 |
| `README_2026-09-29.md` | 09-29까지의 pipeline README(Combined 설계 시절 누적 로그) | 현재 README(`../README.md`)로 대체 |

이 스크립트들은 `run_comparison`을 `sys.path`로 불러온다. 다시 쓰려면 `pipeline/`에서 실행 경로를 맞춰야 한다. 경위는 `docs/PROMPTCAL_VOCAB_METRIC_2026-09-28.md`에 있다.
