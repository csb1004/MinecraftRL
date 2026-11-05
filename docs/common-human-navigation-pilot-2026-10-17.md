# 사람 이동 자료를 연결하는 가중치 학습 pilot

입력 계약과 로컬 자료 학습을 분리한 뒤, 실제 사람 영상의 행동을 소량 섞는 조건이다.
train은 기존6개 로컬 실행의5,979정답+사람 한 녹화36행동이며, 다른 녹화35행동은 모니터로만 사용한다.
이 작은 자료로 일반적 이동 능력을 입증한다고 주장하지 않는다. 상위 RM/원목 actor/기본 모델은 유지한다.

사람의 원래 의도는 없으므로 기록된 미래 위치를 사후 참고점으로만 준다. 공중 높이가 변하는 끝점,
첫 frame, 참고점1블록 이내는 제외하며 정지·도착·안전·접지를 정답으로 만들지 않는다.
원본120frame feature bundle은 `training_ready=false`/RM과goal=null 그대로 보존한다. 별도 계약과
독립 검사로 승인된36/35행동만 새 BC row로 만들며 PPO에는 넣지 않는다.

조건은8epoch·lr0.0001·batch32·전달 명령 BC·KL상한3이다. 동일한 ignored 입력 계약과 가중치를
이어받는다. loss 비중은 로컬90%/사람10%이고 사람 row를 복제해 표본 수를 늘리지 않는다.
이 비중은 가중 손실의 비중이며 gradient clipping 후 기울기 크기 비율을 뜻하지 않는다.
AdamW는 새로 시작한다. fixture 모니터254개와 사람 모니터35개를 합치지 않고 별도로 기록한다.

출발점은 `results/common-navigation-root-context-20261017/training/mlp/checkpoint.pt`,
SHA `e2955a352c331c1e652df374d3cec522cde99600dc957033171df76478a90e8d`다.
이전 후보는 실제 실행 준비 기준 미달이므로 성공한 기준 정책으로 표시하지 않는다.
같은4관측의 near4/4+far4/4를 통과해야만 실제 train calibration5/6을 수행한다.
그 다음 동일 decoder·입력 계약에서 출발/혼합 후보를 개발2월드로 비교한다. 최종 시드는 미사용이다.
사람 자료의 효과를 추가 optimizer 갱신 효과와 완전히 분리한 대조 실험이라고 주장하지 않는다.

## 실행 전 검사

원본 파일 SHA·정확한 행동 사상·선택 frame/목표/경계/녹화별 split·미관측 RM을 독립 재현했다.
Windows/WSL 수치 차이 최대1.11e-16, 목표 비교 허용오차1e-12이며 입력은 저장된 원본 값을 사용한다.
변조 목표·가짜 도착·RM 생성·split 변경·마스크 무시·중복·끝점 변경·잘못된 모델/계보9조건을 모두 거부했다.
실제 사람 특징의 CPU 역전파는 유한 gradient를 만들었고 optimizer0·가중치/root 불변을 확인했다.

실제 코드 연결 후 Windows883개 중856통과/27환경별 제외, WSL 관련77검사 통과.
검사 통과/손실 감소/사람 행동 일치는 실제 게임 성공이 아니다.

복원: `results/common-human-navigation-pilot-20261017/workflow.json`.
실험: `configs/common-human-navigation-pilot-v1.json`.
자료: `configs/common-human-navigation-bc-pilot-v1.json`.
준비 및 후속: `results/common-human-navigation-pilot-preparation-20261017`.

## 완료 결과 — 2026-10-17 18:47

8epoch·1,504갱신 완료. checkpoint SHA
`fbe9ec9c76034dbfb0d531c2b9f976cae7db1e75e901e69bcd791cdcba81d5d8`.
전체 train NLL0.80788, 사람 train9.23038→6.34199, 다른 녹화 사람 모니터17.94763→16.77094다.
기존 fixture 모니터는1.58318→1.61363으로 악화했다. 작은 두 사람 구간의 결과를 일반화하지 않는다.
root/value 보존·reload오차0·자료 독립 재현·기존12개 정답 sidecar 동일성을 확인했다.

평균 제안 확률 gate는 통과했으나 실제 사용할 mode 반응은 근거리4/4·원거리3/4로 미달했다.
평균만으로 실패 관측을 가리지 않는다. 실제 calibration/development 게임0, 기본 정책 교체0이다.
체크포인트와 실패를 보존하고 [전체 관측 목표 감독 조건](common-navigation-dense-goal-2026-10-17.md)을 분리했다.
