# 마지막 기억층 burn-in 계산 검증

[사전 조건](bounded-burnin-protocol-2026-10-09.md)에 따라 Windows의 WSL Ubuntu-24.04,
GTX1060 3GB, 기존 STEVE-1 환경에서 수행했다. 클라우드 결과가 아니다.

공식 작은 transformer의 CPU5검사가 통과했다. 실제 GPU에서는 기존 masked 부모와
집중 BC 후보, 원본6시연, 시작/후반 window를 조합한24쌍을 비교했다.
모든 쌍에서 특징·정답 logprob·parameter/input gradient의 최대 차이는0이며 기억 상태도 동일했다.
burn-in은 gradient 없이 현재 가중치로 재생하고 학습 window8의 gradient 범위는 유지한다.

| 측정 | 결과 |
|---|---:|
| 검증 쌍 | 24 |
| 기존 window 계산 합계 | 99.263초 |
| 유효cache만 재생한 계산 합계 | 21.240초 |
| 쌍별 속도비 중앙값 | 4.520배 |
| 쌍별 속도비 범위 | 2.681–6.829배 |
| PyTorch peak allocated / reserved | 1643.394 / 1758 MiB |
| optimizer 갱신 / 게임 행동 | 0 / 0 |

동기화된 GPU 시간이며 실제 첫/후반 window의 forward/backward warm-up 뒤 측정했다.
전체 학습에는 모델 로드·특징 측정·저장 등의 비용이 있으므로 전체 학습이4.52배 빨라졌다고
해석하지 않는다. 정책 성능·샘플 효율 개선의 증거도 아니다.

적용 범위는 앞선 층이 고정되고 마지막 공식 clipped-causal transformer만 학습하는 현재 구조다.
128개 **층 입력**에는 앞선 층이 계산한 과거 정보가 이미 포함돼 있다. 전체 정책의 영상 기억을
128프레임으로 제한하지 않는다. 다른 recurrent/attention 구조는 구현에서 거부한다.

원본 특징·두 체크포인트·관련 코드 해시와 STOP 내용을 보존했다. 현재 onset/uniform 배치의
계산 경로는 그대로 유지했고 재실행하지 않았다. 후속 별도 학습에서 이 경로를 사용할 수 있다.

원본 측정: `results/bounded-burnin-verification-20261009/report.json`.
구현: `src/minecraftrl/bounded_burnin.py`, 검사: `tests/test_bounded_burnin.py`,
실제 GPU 확인: `scripts/verify_bounded_burnin.py`.
