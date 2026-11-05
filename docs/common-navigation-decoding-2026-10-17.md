# 같은 이동 가중치의 전달 명령 선택 검사 — 2026-10-17

목표 조건 학습 v4는 실제3/12로 미채택했다. 같은 입력 진단에서는 회전의 잠재 argmax가
확률적 선택보다 일관됐지만, 후보 자신이 만든 상태에서 전진 확률도 낮아졌다.
따라서 행동 선택만으로 해결된다고 가정하지 않고 작은 실제 calibration을 먼저 한다.

기존 실행 흐름의 평가 옵션 `delivered_mode`는 공식 mapper에서 같은 전달 명령이 되는
잠재 button/camera 조합의 확률을 합산한 후 가장 높은 명령을 고른다. camera off일 때
버려지는 camera 표본과 camera on의 중앙 camera 표본을 합산한다. 단순한 두 head의
독립 argmax와 다르다. 환경 좌표로 행동을 직접 만드는 규칙이나 교사를 실행하지 않는다.

명시적인 `navigation_trial_config_v2`/evaluate 전용이다. 실제 결정적 행동의 log probability는0,
원래 신경망의 전달 명령 제안 log probability는 `proposal_delivered_logp`로 따로 기록한다.
독립 감사가 선택한 명령·두 확률·공식 사상·기존 root RM/시점/안정 도착을 재생한다.
결정적 자료를 stochastic on-policy PPO 자료로 사용하려 하면 검증기가 거부한다.
기존 sample 방식은 그대로 기본이며 가중치/상위 RM을 바꾸지 않는다.

검사는 작은 분포의 전 조합 합산, multirow/동률/RNG 불변/0확률/공식 mapper 및 CPU-CUDA
일치와 실행 계약을 포함한다. Windows873검사(27환경별 제외), WSL관련70검사 통과.
로그는 `results/common-navigation-decoding-preparation-20261017`에 있다.

`configs/common-navigation-decoding-probe-v1.json`에 actor SHA와 원본 월드를 고정했다.
`train-b`(104743)의 새 복제에서6목표/12틱 안정 도착/제어180초 이하로 한 번 평가한다.
5/6 이상과 확률·STOP·입력 해제·소유PID 감사를 통과하면 별도 개발 비교를 준비한다.
결과는 `results/common-navigation-decoding-20261017`에 보존하며 완료 배치를 다시 실행하지 않는다.
학습용 월드 calibration은 일반화나 새로운 가중치 학습 성과가 아니다. optimizer/API0,
최종 시드 미사용, 자동 기본 승격 없음이다.

완료 결과: 3/6·471행동·제어55.521초로 5/6에 미달했다. 추론 중앙46.714/P95 58.062ms,
제어 주기 중앙110.652/P95 126.690ms, 프로세스 GPU peak allocated1,915MiB/reserved2,126MiB다.
471행동의 선택 명령과 제안 확률·값 재생 오차0, STOP 후 추가 행동0/해제 키0/잔여 소유PID0.
실패3목표 모두 반경에 들어가지 못했다. 마지막 목표는100행동 동안 전진0, 회전88회 중
43회의 회전 부호가 목표 방향과 반대였다. 따라서 mode 선택만으로 해결됐다고 보지 않는다.
기본 정책/기본 sample은 유지하고, [다음 train 상태 교정 학습](common-navigation-dagger-v5-2026-10-17.md)을 진행한다.
