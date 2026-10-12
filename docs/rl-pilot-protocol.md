# 첫 로컬 강화학습 파일럿

대상은 사전학습 STEVE-1의 **행동 출력층과 새 원목 채집 가치함수**다.
영상 encoder와 recurrent backbone은 고정한다. 전체 모델 미세조정으로 해석하지 않는다.
기존 공식 MineRL1.16.5·STEVE-1 격리 환경을 사용하며 API 제어/제작 보조를 사용하지 않는다.

## 과제와 데이터

빈 초기 인벤토리에서 서버가 원목 최소1개 획득을 확인하면 보상1과 과업 종료를 동시에 발생시킨다.
여러 물품이 한 관측에 들어오면 실제 개수를 그대로 기록한다. 정확1개만 줍는 과제로 해석하지 않는다.
보상은 한 번만 주며 목표 종료 이후 새 정책 행동은 금지한다.
클라이언트 예측 인벤토리·미리보기만으로 보상을 주지 않는다.
시간 제한180초는 truncated로 bootstrap하고, 목표 도달·환경 종료는 terminal로 bootstrap0.
취소·기술 오류·0행동 trajectory와 eval 분할 데이터는 학습에서 거부한다.

학습 개발 seed104729에서 rollout1개를 사용한다. 평가 seed는 사전 고정한
1000003,1000033,1000037,1000039,1000081이며 학습 전후 각각 새 월드에서 실행한다.
평가 결과를 보고 checkpoint·prompt·하이퍼파라미터를 고르거나 평가 데이터를 학습에 넣지 않는다.
텍스트 `collect wood`, guidance6과 동일한 저장 텍스트 embedding을 사용한다.

실제 전송 행동과 내부 sampled action, normalized log probability, value, first mask,
reward/terminal, frozen features를 함께 기록한다. 초기/최종 recurrent state도 저장한다.
backbone을 고정하므로 PPO 업데이트는 같은 특징에서 행동층을 정확히 다시 계산할 수 있다.
중간 recurrent state를 학습 중 역전파하거나 영상 backbone을 업데이트하는 실험은 아니다.

## 확률과 최적화

공식 STEVE-1은 conditional/unconditional log score를 `7*conditional-6*unconditional`로 합친다.
Gumbel sampling에는 상수 정규화가 필요 없지만 PPO log probability에는 필요하다.
guided score에 log_softmax를 적용하고 수집·재계산·학습 모두 같은 함수를 사용한다.
frozen backbone은 기존 autocast, 행동층/가치함수는 float32다.
비교 기준선도 이 수집기이므로 기존 혼합정밀도 upstream 추론 결과와 완전히 동일하다고 주장하지 않는다.

PPO clip0.2, gamma0.99, GAE lambda0.95, AdamW lr1e-5, weight_decay0,
gradient norm clip1, value MSE 계수0.5, 최초 행동층에 대한 KL 계수0.01.
entropy 보너스0, minibatch16, epoch2로 고정한다. 채집 skill reward와 RM accept는
수집 로그에 분리하고 human prior는 보상에 합산하지 않고 KL 정규화로 적용한다.
행동 확률 재계산의 허용 오차는 logp 절댓값0.0002다. 배치 GEMM 오차를 기록한다.

매 업데이트에서 loss·gradient의 유한성을 확인하고 실제 행동층 파라미터 변화,
optimizer state와 checkpoint 재로딩을 검사한다. 한 번의 성공 rollout에 대한 작은 업데이트는
학습 파이프라인의 실제 실행 증거이며 일반적인 능력 향상 증거가 아니다.

학습은 게임 종료 뒤 cached feature로 별도 수행한다. 이 단계의 GPU 사용량을
게임+전체 모델 동시 학습 사용량으로 해석하지 않는다. 모델·가중치·데이터는 Git 제외 경로에 둔다.

## 근거

- [PPO 원 논문](https://arxiv.org/abs/1707.06347)
- 고정 로컬 `external/steve1/steve1/embed_conditioned_policy.py`와 `VPT/lib/action_head.py`의 실제 sampler/logprob 구현
- [MineStudio 학습 경로](https://craftjarvis.github.io/MineStudio/online/trainer.html)는 후속 후보로 유지한다.
  이번 작은 행동층 학습에 환경 교체를 선행 조건으로 추가하지 않았다.

실행 코드는 `scripts/run_rl_pilot.py`, `scripts/train_rl_heads.py`, `scripts/evaluate_rl_heads.py`다.
실험 결과 경로와 완료 여부는 `rl-execution-ledger.md`에 기록한다.
