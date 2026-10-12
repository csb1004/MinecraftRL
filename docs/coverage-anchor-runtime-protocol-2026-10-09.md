# KL 단일 변수 후보의 조건부 실제 대조 — 2026-10-09

662갱신 종료 후 전체 개발 측정/재로드/감사 완료 전에 고정한다.
기존 coverage 비교의 `diagnostic_eligible` 기준을 그대로 사용한다:
개발 두 시작점 각각 공격 해제+회전 확률0.5% 이상, 부모보다 첫 NLL1 이상 감소,
첫 회전 평균 증가, 계속 교사 NLL이 부모의110% 이내.
기존 strict KL≤.03 gate는 별도로 보고하며 기본 정책 승격 조건으로 유지한다.
위 진단 기준 미달 시4게임 모두 미실행으로 기록한다. 기준을 낮추지 않는다.

입장한 경우 기존 `validate_temporal_runtime.py`로 전체 개발 에피소드의
현재 후보 전체 영상 추론과 학습 cache 재생의 특징/기억/분포를 대조한다.
오차 기준을 통과한 뒤에만 다음4게임을 실행한다.

| 세계 | 고정 seed/행동 sampling seed | 순서 |
|---|---|---|
| 기존 eval-long | 2000003 | KL1 → KL0.1 |
| 기존 natural | 2000029 | KL0.1 → KL1 |

KL1은 `results/onset-coverage-training-20261009/checkpoint.pt`,
KL0.1은 `results/coverage-anchor-ablation-20261009/checkpoint.pt`다.
두 후보는 같은 masked 부모·자료·순서·662갱신이고 KL계수만 다르다.
기존 원목 embedding/wood-potential-v4, 최대180초, 실제 인벤토리 완료를 유지한다.
실제 planner/탐색 개입/스크립트 행동을 사용하지 않는 신경망 정책 비교다.
world archive/서버 상태/최초 화면/기억·sampling seed를 검사하며 픽셀 불일치가 있으면
학습의 인과 효과로 해석하지 않는다. 실패 결과·전체 궤적·최종 월드·STOP/해제/PID를 보존한다.
추론 중앙값/p95·전체 주기·GPU·획득량·실패 위치/단계를 기록한다.
최종5시드 미사용, 새 유료 API/Colab0, PPO갱신0, 기본 정책 승격0.

이4조건은 제어 종료 수정만 검사하는 별도 고정 응답 corner1회와 구분한다.
이미 끝난 배치를 반복하지 않으며 모든 새 결과는 새 폴더에 쓴다.
