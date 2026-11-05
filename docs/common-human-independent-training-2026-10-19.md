# 독립 녹화 확대 BC — 2026-10-19

04:30 현재 학습 및 증거 고정을 마쳤다. 목적은 특정 월드 가드를 추가하는 대신 공통 이동 가중치에
다른 사람 녹화의 이동 시연을 연결하는 것이다. PPO 결과나 실제 게임 성공으로 보고하지 않는다.

## 사전 고정 조건

- 계획: `configs/common-human-independent-training-v1.json`, SHA256
  `5707ad0855099b783e99a8c89836120ffd4e2063b035801056628f922356231e`.
- 출발 정책 d6137dfa…482d, 로컬17240행, human629행. human 개발154행과 로컬 개발254행은 gradient 제외.
- 네 번의 가상 epoch, 각546갱신, 총2184. 배치24local/8human, 사람 BC 비중.25,
  global KL.01/local KL1, lr1e-4, seed14101, 새 AdamW. 보조 이동 BC는0.
- 대조는 broader957 정책. 녹화 수가 train2→4/dev1→2로 늘었다.
  표본 수에 따른 KL population fraction도 변하므로 순수 다양성 단일 효과로 주장하지 않는다.
- 기존523/98 및212/98 부분집합을 별도로 비교한다. 확대154개 개발 평균을 이전98개 평균과 직접 비교하지 않는다.
- 학습 점프TP≥34/100·오탐≤105/529, 기존523 TP≥25/74·오탐≤89/449,
  기존212 TP≥11/33·오탐≤35/179, 평지mode near4/far4를 사전 선언했다.
  통과해도 게임 성공이 아니며 실제 train 교정과 새 코드 기준 baseline/candidate 개발 비교가 필요하다.

## 자료 자격과 전이 진단

기존720frames를 유지하고200frames를 추가했다. frozen STEVE-1의 실제64저장화면 특징과
최대오차0, 공식 버튼 투영 손실0, 시간 정렬/전구간 화면 검수를 통과한 녹화만 사용한다.
특징 SHA `14e026b6462a8ba15b85291b6d951a662b46b24125ce5c01310a02a5d5820bb6`.
새 자료106train/56dev의 정답 좌표는 사후 reference이며 인간 의도/안전성/ground/stop 정답을
관측한 것으로 꾸미지 않았다. `goal[5]=1`은 목표 존재 상수이지 접지 관측이 아니다.

`results/common-human-independent-transfer-20261019/comparison.json`에서 기존d613/957/655
모두 새 자료의 점프·좌우 버튼 true-positive가0이었다. 새 dev56의 NLL은 각각
11.6061/5.5303/5.7667이다. 프레임 상관과 적은 독립 녹화 수를 고려해야 하며
기록된 인간 행동과 다르다는 사실만으로 실제 경로 실패라고 판정하지 않는다.

## 복원 지점

후보 SHA `4b451e4f5c2204f898b4245064d443e3c5e43bf901603e2072444f01edda8726`.
2184갱신,14정답파일 동일, 초기 actor 및 로컬 monitor 동일,root/value불변,reload0을 확인했다.
`evidence-freeze.json` verified=true,game_admitted=false다. 아래 모두 오프라인 모방 관측이다.

| 관측 묶음 | 점프 TP/정답 수 | 점프 오탐 | delivered NLL |
|---|---:|---:|---:|
| 전체 train629 | 13/100 | 3/529 | 4.27256 |
| 기존 train523 | 11/74 | 3/449 | 4.41950 |
| 새 train106 | 2/26 | 0/80 | 3.54754 |
| 전체 dev154 | 3/24 | 0/130 | 4.04569 |
| 기존 dev98 | 3/19 | 0/79 | 3.44951 |
| 새 dev56 | 0/5 | 0/51 | 5.08898 |

기존212행은8/33·오탐2/179다. 평지 mode near4/far4는 통과했으나 세 가지 점프 준비 기준은
미달이다. 게임 실행0/기본 승격0이며 미실행 조건을 실패 게임 표본으로 세지 않았다.
같은 기존523행에서 이전957은4/74·오탐0/NLL4.23089였다. 점프 모방 증가와 전체NLL 악화가
함께 있으므로 전반적 개선으로 주장하지 않는다. 좌우 이동 TP는 여전히0이다.

- 실행: `results/common-human-independent-training-preparation-20261019/pipeline.json`, exec59825.
- 학습: `results/common-human-independent-training-20261019/workflow.json` 및 `logs/training.log`.
- 특징 자격: `results/common-human-independent-features-preparation-20261019/feature-workflow.json`.
- 이 연구의 소스355개와 증거는 고정했다. 이후 시각 경로 비교는 별도 버전으로 진행한다.
- 기본 정책/연구 명세/최종5시드 보존. API/게임 실행0.
