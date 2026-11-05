# 공통 이동: 사람 시연 미니배치 구성 비교 — 2026-10-19

02:52 시작한 exec7705는 03:09 종료0으로 완료됐다. 2184갱신과 준비 검사 결과를 고정했다.
점프 준비 기준 미달로 새 게임은0이며 실제 게임 성능 향상은 검증하지 않았다.
상태는 `results/common-human-stratified-preparation-20261019/pipeline.json`과
`results/common-human-stratified-training-20261019/workflow.json`에 기록한다.

## 변경 근거와 통제

사람 BC 비중만 .10에서 .25로 높인 3ccc 후보는 사람 학습 자료의 점프33개 중 선택0이었다.
초기 actor의 학습 자료3개 미니배치에 대한 읽기 전용 미분 진단에서는, 희소한 사람 행의 가중치를
높이면 큰 gradient norm과 clipping이 함께 커졌다. 이 진단은 전체 학습 원인을 확정하지 않는다.
이번에는 매 배치에 사람 행을 일정하게 넣어 이 가설을 시험한다.

비교 대상은 `results/common-human-balance-training-20261019`의 비중.25 조건이다.
동일 d613 actor에서 출발하며 같은17452행(local17240/human212), 개발254local/98human,
학습률1e-4, seed14101, batch32, global KL.01, 추가 local KL1.0을 유지한다.
개발 자료는 gradient에 넣지 않는다. 추가로 검증한 사람311행은 이번 실험에 넣지 않는다.

변경 변수는 `--human-minibatch stratified-8`이다. 그룹별 shuffle/cycle로 local24/human8을
뽑고 원래 데이터 비율/배치 비율을 BC·global KL·local KL 각각에 적용한다.
따라서 기대 손실은 보존하지만 gradient clipping과 Adam의 실제 학습 경로까지 같다는 뜻은 아니다.
각 가상 epoch546updates, 총2184updates다. 총 방문 횟수는 local52416/human17472다.
각 자료를 정확히 네 번 순회한 학습으로 해석하지 않는다.

계획: `configs/common-human-stratified-training-v1.json`, SHA256
`d62a5177389eaf9f66c40daa79d1699208a12961e5a1ce64b0d7f6bf285cfeb9`.
uniform 기본 경로는 기존 RNG/연산을 유지한다. 새 모드는 혼합 BC continuation에 한정했다.
root/value/encoder와 행동 decoder는 변경하지 않는다. epoch마다 그룹별 gradient clipping과
실제 방문 횟수를 기록한다.

## 설치 검증과 진입 기준

staging에서 WSL CPU13개 검사를 통과했다(새7/기존혼합6).
기대 손실·gradient 보정, local KL의 human 제외, 샘플 순환/seed/예산 및 잘못된 입력을 검사했다.
`installed-verification.json`으로 검토한 trainer/test 바이트와 실제 설치를 확인했고,
다른353운영 파일·기본 actor·연구 명세는 그대로였다.
설치/검토/소스 동결 근거는 `results/common-human-stratified-preparation-20261019`에 보존한다.

학습 후 전문가 정답 감사, 거리 확률 진단, 사람 점프 선택 감사와 평지 mode 검사를 수행한다.
사전 게임 진입 기준은 train 점프 TP≥11/33, 비점프 FP≤35/179,
평지 near4/4+far4/4다. 준비 기준 통과는 게임 성공이 아니며, 실제 교정·개발 평가가 별도로 필요하다.
기본 정책 승격0, 최종5월드 미사용, API0이다.

## 결과와 다음 단계 — 03:10

checkpoint: `results/common-human-stratified-training-20261019/training/mlp/checkpoint.pt`, SHA256
`ed31b42a78b4d4107147518d1c4567d9083acb4b4c3ae6e72db134f36b0549b9`.
2184갱신, root/value 보존, reload 오차0, local52416/human17472방문을 확인했다.
`controlled-training-comparison.json`은 출발 actor/학습 전 지표·정답14파일 바이트·입력·예산 일치를
확인하며 운영 소스의 차이는 trainer뿐이다. 운영354파일의 실행 중 불변과
기본 actor/연구 명세 보존을 `evidence-freeze.json`에 고정했다.

| 지표 | 이전 uniform .25 | 이번 stratified .25 |
|---|---:|---:|
| 사람 train 전달 행동 NLL | 7.89726 | 3.21150 |
| 사람 development 전달 행동 NLL | 8.55553 | 5.82549 |
| local development 전달 행동 NLL | 1.04898 | 1.06649 |
| 사람 train 점프 TP / 정답 | 0/33 | 8/33 |
| 사람 train 비점프 FP / 정답 | 0/179 | 2/179 |
| 사람 development 점프 TP / 정답 | 0/19 | 0/19 |

학습 점프8개는 사전 최소11개에 미달한다. 평지 mode는 near4/4+far4/4로 통과했으나
전체 게임 진입은false다. 원래 확률 기준도 near.75908<.85로 미달이다.
손실 감소·학습 시연의 점프 증가를 자연 지형 통과나 일반화 성공으로 해석하지 않는다.
개발에서 점프 선택0이 유지되어 더 다양한 검증 시연에 대한 후속 비교를 진행한다.

각 epoch의 gradient clipping 전 평균 norm은10.511/9.113/9.034/8.717이며 모든2184배치에
clipping이 적용됐다. 이전 초기3배치 진단과 이 전체 학습 통계를 같은 표본처럼 비교하지 않는다.
샘플링과 사람 행 방문 횟수가 함께 바뀌므로 clipping만의 인과 효과라고 주장하지 않는다.
