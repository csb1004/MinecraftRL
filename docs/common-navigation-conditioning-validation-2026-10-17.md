# 거리 입력과 실제 명령 BC의 분리 비교 결과

`results/common-navigation-conditioning-20261017`의 두 모델은 같은 초기 v3/1,384train정답/
254monitor/난수/16epoch/학습률/KL 한도에서 각각704회 가중치 갱신을 완료했다.
root/value 보존·재로딩오차0·원본 행동 불변·expert 질의 재생 감사를 통과했다.
거리 변환 모델은 optimizer 전에1,638개 입력의 정책 출력 오차0을 확인했다.

| 조건 | monitor 실제 명령 NLL | 가까운 목표 비이동 확률 | 먼 목표 전진 확률 |
|---|---:|---:|---:|
| 기존 canonical/scale1 | 2.6828 | 0.5092 | 0.4126 |
| canonical/scale16 | 2.6528 | 0.4965 | 0.3762 |
| delivered/scale1 | 2.4554 | 0.7604 | 0.1753 |

확률은 사전에 적합성을 확인한4개 거의 정지한 **train** 상태의 반사실 목표 평균이다.
거리0.05/3블록,정면 방향이며 실제 게임 성공률이 아니다.
첫 후보는 가까운 목표 정지를 개선하지 못했고 두 번째는 먼 목표 이동도 지나치게 줄였다.
둘 다 사전 선택 gate에 미달했다. 새 실제 게임0이며 미실행을 실패 표본으로 세지 않았다.
기본 모델은 교체하지 않았다.

checkpoint SHA256:

- distance16: `237461aab188c345e203617fc2f4dc96f684c02413568cd1ef6afea9044a2bbf`
- delivered: `d12cc1b030f799bc387d138f7863aa9e927da890f57308c1442be5d58eedd33e`

실행기의 진단 입력 순서 오류도 보존했다. 원래 실행기는 B/C 자료로 진단하고 A/B 참조와
비교하려 해 선택 단계 assertion에서 종료됐다. 두 학습·감사는 이미 정상 완료돼 있었다.
`runner-at-launch.py`,원래workflow와 다른 자료의 진단을 보존하고 A/B로 맞춘 CPU 검사만
`paired-diagnostics`에 반영했다. 추가 optimizer0/실게임0이며 복구workflow completed=true다.
선택 근거는 `paired-diagnostics/selection.json`이다.

Windows853검사(828통과/25제외),WSL navigation52검사 통과 후 이 학습을 실행했다.
별도 linear 변환 검사에서 WSL float32 최대1.1921e-7 차이가 있었고 실제 MLP는 CPU/CUDA0이었다.
실제 MLP 학습의 변환 오차0 조건을 낮추지 않았다. 훈련 시간은 CPU 진단과 일부 겹쳐
속도 개선의 인과 비교로 사용하지 않는다.

후속은 [상대 목표를 짝지은 자료 보강](common-navigation-goal-variation-2026-10-17.md)이다.
