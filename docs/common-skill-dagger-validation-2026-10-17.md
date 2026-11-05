# 학생 방문 상태 감독 결과

후보는 기본 정책으로 채택하지 않았다. train 상태 정답을 추가하고 실제 가중치를 갱신했으나,
분리된2개 개발 월드의 합계는 기존11/12도착·771행동에서 후보10/12·783행동으로 악화됐다.
같은 월드의6목표는 순차적이고 서로 의존하므로12개의 독립 성공률 실험으로 취급하지 않는다.

근거: `results/common-navigation-dagger-20261017/{workflow,comparison,final-verification}.json`.
조건은 [실행 전 프로토콜](common-skill-dagger-protocol-2026-10-17.md)과 `configs/common-navigation-v2.json`이다.

## 수집과 학습

- 새 train2회에서 학생644행동, 각5/6도착. 교사/중립 행동0. 이는 학습 자료 수집 결과다.
- 이전train8회와 합쳐 사후 expert 정답1,908개: 학생 방문 상태1,036/기존교사 상태872.
  점프 높이 상태511개와 중립 인계46개는 이 grounded 평지 교사의 범위에서 제외했다.
- 기존 실행 행동/logp를 고치지 않고 별도 `bc/expert-labels/*.json`으로 정답을 보존했다.
  official mapping 왕복, 기록된 교사 행동과 동일성, 원본hash, train split을 다시 검사해 통과했다.
  `bc/expert-query-audit.json`은 별도 실행으로 만든 감사이며 시작된workflow의steps 목록에는 없다.
- 638,746개 이동 residual/projection 파라미터의 BC. AdamW lr1e-5/batch32/최대16epoch.
  epoch3에서 평균KL.1을 넘겨 그epoch60갱신을rollback했다. 실행180/보존120갱신, 최종KL.080021.
- canonical latent NLL4.839492→4.616378. 별도 development 교사136개 NLL2.809128→2.849522.
  새 상태의 내부camera head top1은39.6%다. 이는 실제 시선 조작 정확도와 다르다.
- 실제 actor tensor 변경, 원목actor/value 동결, 저장/reload 출력오차0 확인.
  학습176.960초/최대PyTorch allocated450.388MiB. 새PPO 갱신0.
- 후보SHA256 `5df577e0d1e6ec28fa3528df15054e052c751caf5349b249c3a53e3bbe76f0bf`.
  기존baseline `e033f1ef7902102569d01459d85cdd80252c141f4e1346bdb40832502e8acaea` 유지.

## 실제 개발 비교

| 조건 | 기존 도착/행동 | 후보 도착/행동 | 기존/후보 제어초 |
|---|---:|---:|---:|
| A, seed2000003 |5/6·423|5/6·315|54.354 / 41.865|
| B, seed2000029 |6/6·348|5/6·468|46.115 / 62.374|

두쌍의 초기PNG/서버pose/recurrent state/주요정책source가 같다. samplingseed17231 고정,
B에서는 후보를 먼저 실행했다. 같은 시작조건을 확인한 비교이며 소수seed의 결과다.
A에서 후보는 기존 첫목표 실패를 해결했지만 다른목표에서 실패했다. B에서는 도착과 효율이 모두 악화됐다.
가중치 변화나 낮아진loss만으로 공통 이동이 개선됐다고 주장하지 않는다.

평가 추론 중앙값48.718~49.330ms/p95 60.944~61.850ms, 제어주기 중앙값121.895~125.637ms/
p95 140.781~150.492ms. PyTorch peak allocated1915.001MiB/reserved2126MiB, 전체장치VRAM과 구분한다.
첫baseline 초기화와 일부 실행 시간에 CPU 정답재검증이 겹쳤으므로 작은 지연 차이를 모델의 인과효과로 해석하지 않는다.

실제6게임2,198행동, 제어최대62.374초, STOP후새추론/행동0, 입력키0, 각소유PID잔존0을 확인했다.
기존월드/가중치/원본궤적 보존, 연구명세SHA256 `4af3ad00c5bab9b9d08b2e23fe768e140944118aa85cf412c45a67bb032275ad` 유지.
최종평가시드 사용0, 유료API/Colab/게시0. 원목수집/제작/자연지형 성공 실험은 아니다.

## 다음 조치

목표 달성까지 이어가라는 사용자 지시에 따라 [후속 학습 강도 조건](common-skill-distillation-protocol-2026-10-17.md)을
실행한다. 같은1,908정답과원래baseline을 사용하며 lr/KL허용치를 조정한다. 기본가중치는 계속 유지한다.
또한 official mapper의camera meta-off가 무시하는 내부출력까지 오답으로 해석하지 않도록,
동일 환경명령의 확률 합산/greedy명령일치율 진단을 추가했다. 기존BC loss와실행 sampler/PPO는 바꾸지 않았다.
공식mapper 포함WSL관련30검사 통과. v2초기Windows 전체813검사(790통과/23환경제외)와 이후추가검사는 구분한다.
