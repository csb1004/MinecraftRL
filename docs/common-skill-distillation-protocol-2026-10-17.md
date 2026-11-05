# 공통 이동 head의 교정 학습 강도 비교

v2의 실제 학생 상태 정답 추가 뒤에도 카메라 top1정답률은39.6%, train NLL4.8395→4.6164였다.
최대16epoch 중3번째에서 평균KL.1 제한에 걸려2epoch/120갱신만 남았다. v2의 실제 개발 비교와
별개로, 같은 자료에서 교정 행동을 충분히 학습할 수 있는지 확인하는 후속 조건을 고정한다.

카메라39.6%는 **내부 head의 canonical label 정확도**이며 실제 시선 조작 정확도가 아니다.
공식 mapper는 camera meta-off일 때 camera head 출력을 무시한다. 이에 따라 같은 환경 명령으로
변환되는 latent 행동들의 확률을 합친 NLL과 greedy 변환 명령 일치율을 보조 지표로 추가한다.
기존 BC 손실/실행 sampler/old_logp/PPO는 이 진단 추가로 변경하지 않는다.

- 설정 `configs/common-navigation-v3.json`, 결과 `results/common-navigation-distill-20261017`.
- v2와 **동일한10개 train 실행, 동일한1,908개 정답**, 원래baseline에서 새 후보를 만든다.
  새 수집/개발 실패 자료 추가0. 교사 범위 밖511개/중립46개 제외를 유지한다.
- 학습률1e-5→1e-4, 평균KL한도.1→3.0, 최대16epoch, KL손실계수.01/gradient clip1/기존 가중치 유지.
  두 hyperparameter를 함께 바꾼 조건이며 어느 하나의 독립 효과로 주장하지 않는다.
  낮은 KL은 불확실한 기존 이동 분포를 유지시키므로, 명시적 expert 정답이 있는 BC에만 높은 한도를 허용한다.
  일반BC/PPO의 기본 제한은 유지한다. NaN/STOP 검사, 한도 초과 epoch rollback은 유지한다.
- 원목 actor·영상 backbone·recurrent memory·value는 동결한다. 바뀌는 것은 기존 이동 residual/projection이다.
- 새로운2개 development 시작 방향/목표 순서에서 원래baseline과 paired비교한다.
  최종5seed는 사용하지 않는다. 목표 도착/행동량은 실제 서버 판정이며 같은월드6목표는 서로 독립이 아니다.
- 새로운조건에서12/12 도착과 baseline 대비 효율 개선이 확인돼도 즉시 default로 교체하지 않는다.
  별도 samplingseed 및 앞선 공통개발 조건의 회귀 비교로 이어간다. 실패하면 상태별 정답 확률/행동/위치로 분석한다.
- v2와v3의 서로 다른 개발 과업 결과를 직접적인 인과 비교로 취급하지 않는다. 각자 동일 시작본baseline과 비교한다.
- 매 실행175초 제어 예산·실측180초 이하, STOP후0·입력해제·소유PID정리 확인.
- 지형회피/주변탐색/채굴/제작을 이 평지 결과만으로 구현했다고 주장하지 않는다. 유료 자원/게시0.

이 문서는 실행 전 조건이다. 실제 수행/완료는 각 workflow.json을 따른다.
