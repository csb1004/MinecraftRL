# 교사 시연 BC와 분리 비교 — 2026-10-08

## 고정 조건

- train: 기존 시드104743/104759 교사 기록 각205행동. 원본을 다시 수집/수정하지 않는다.
- development: 새 시드104761을 이번 BC용으로 예약, 벽 끝5·막힌 출발(6.5,3.5)인 새
  fixture에서 교사 시연1회. 기존 train 벽 끝은3이다. 개발 자료는 optimizer에 넣지 않는다.
  CLI의 eval 표시는 수집 경로용이며 이 월드는 최종 평가가 아닌 BC development다.
- 실제 비교: 기존 개발 long_wall2000003, natural2000029 각각 부모/후보1회, 제어≤180초.
  이 두 월드는 이미 반복 관측된 개발 평가이며 새로운 최종 일반화 검증이 아니다.
  최종5시드는 계속 미사용. 초기 서버/정책 메모리/픽셀 일치 여부를 별도로 기록한다.
- 부모: `results/recovery-gap-20261007/train-gap-epochs2/checkpoint.pt`.
- 목표: 런타임에서 쓰던 `results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy`
  그대로 사용하고 파일 SHA256을 고정한다. 새 CVAE 샘플을 생성하지 않는다.

## 학습 입력과 gate

교사의 privileged 서버 정보는 시연 선택·정합성 확인과 기존6차원 RM context에만 사용한다.
학생은 기존 영상+목표 embedding+RM context 경로를 쓴다. 따라서 영상 전용 조건은 아니다.
행동 전 RM context를 입력하고 전체 episode를 첫 프레임부터 순서대로 통과시킨다.
정체에서 빠져나오는 첫 행동을 warmup이라는 이유로 제거하지 않는다.

공식 action transformer/hierarchical mapping을 왕복해 버튼 손실0, 유한 camera±10을 요구한다.
연속 회전→이산 라벨 오차는 각 축 최대2.5도 이하로 제한하고 실제 최대/평균/p95를 남긴다.
train 두 시연 검사: 평균약.0251도, p95 .1145도, 최대2.1481도, 버튼 손실0.
이것은 양자화된 교사 행동의 closed-loop 재실행 결과가 아니다.

기존 BC optimizer 조건 유지: frozen vision/recurrent backbone, actor head만 AdamW1e−5,
2epoch/batch16/KL계수1/clip1/seed9202. 이번에는 기록된 RM 입력이 있으므로 actor의 RM6열도
학습한다. value는 그대로 보존한다. 데이터 세션 분리·source/frame 해시·STOP 이후 dispatch0
감사 후 특징을 만들며, 모델 reload 오차0과 개발NLL 비악화·train KL≤.03일 때 실제 비교한다.
불합격 후보는 기본 정책에 적용하지 않으며 미실행 평가는 실패 표본으로 세지 않는다.
BC optimizer를 PPO optimizer로 재개하지 않는다.

## 해석

교사 성공과 학생 성공을 분리한다. 최대3개 scripted 경로의 BC는 소규모 진단이다.
개발 loss 감소만으로 우회 능력을 인정하지 않는다. 실제 서버 획득/경로·실패 원인·중단을
검토한 뒤 후보 적용 여부를 판단하며 원본/후보를 모두 보존한다.
