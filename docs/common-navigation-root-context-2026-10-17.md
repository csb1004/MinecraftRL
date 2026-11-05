# 공통 이동의 상위 RM 입력 분리 실험

사용자가 요청한 공통 스킬 학습을 위해, 이동 actor가 나무 수집용 RM6채널에 의존하지 않는 별도
조건을 만들었다. 상위 RM/원목 actor/시각2048채널/상대 목표10채널은 보존한다. 인코더에는 기존
wood embedding과 학습된 tail이 남으므로 완전히 과업 중립인 인코더라고 주장하지 않는다.

`navigation_refinement_v3`는 `root_context_mode`와 거리 scale을 필수로 기록한다. ignored 분기는
RM열을 linear 연산에서 제외하고, RM이 없는2048특징도 받는다. 값이0인 가상 RM을 만들지 않는다.
기존 checkpoint 경로는 원래 연산을 유지하고 잘못된 버전/입력 조합을 거부한다. 상위 RM 상태는
실제 게임에서 계속 기록하며 이 실험에서 navigation critic의 기존 가중치는 갱신하지 않는다.

먼저 v6 가중치를 그대로 가진 입력만 변경한 기준 checkpoint를 저장했다.
`results/common-navigation-root-context-preparation-20261017/input-only/checkpoint.pt`,
SHA `1fa58884aed97b7138dc0c3b38cd28b2dfd8992785918db9d2870853b487f8d6`.
50실제 train/monitor 입력에서 과거 모델의 log probability 재현 오차0, ignored RM값 변경/미제공의
출력 동일, actor/value/root 보존, reload0이다. 입력 변경으로 제안 분포의 평균KL0.006689,
최대0.072740이 생겼으므로 동일 함수라고 주장하지 않는다. 이 단계 optimizer/game0이다.

이 기준에서 같은6개 train/5,979정답으로16epoch·lr0.0001·batch32·전달 명령 BC를 실행한다.
사람 자료는 이번 학습에 넣지 않는다. 거리/정지 준비 기준은 기존4상태의 near4/4+far4/4 그대로다.
통과할 때만 train-b 안정 도착5/6을 확인한다. 이후 같은 입력 계약의 학습 전/후 모델을 별도 개발
월드에서 비교하며, 기본 자동 승격과 최종 시드 사용은 하지 않는다.

## 구현 확인

격리 모델 Windows/WSL 각15검사. 실제 코드 Windows878개 중851통과/27환경별 제외,
WSL navigation75통과. 전체 검사에서 controller의 메타데이터 초기값 누락을 찾아 생성자에서
초기화했고 실패 로그를 보존했다. 실제 actor의 구버전·v3 입력/루트 복귀도 검사했다.
구현 검사 통과는 게임 성공이 아니다.

복원: `results/common-navigation-root-context-20261017/workflow.json`.
고정 조건: `configs/common-navigation-root-context-v1.json`.

## 완료 결과 — 18:31

16epoch/2,992갱신 완료. checkpoint `e2955a352c331c1e652df374d3cec522cde99600dc957033171df76478a90e8d`.
동일 입력 조건에서 학습 전달NLL1.1341→0.8695, fixture 모니터1.4862→1.5832로 모니터는 악화했다.
root/value 보존·reload0·전체 정답 감사와12개 sidecar 불변을 확인했다. 기존 제안 확률 기준은
near0.79736으로 미달했고 mode도 near4/4·far3/4로 미달했다. 실제 calibration/개발 게임0이다.
입력 분리는 사람 RM을 만들지 않기 위한 계약으로 유지하되, 이동 실패 해결의 증거로 주장하지 않는다.
이 모델을 기본으로 교체하지 않았고 별도 사람 자료 혼합 pilot의 출발점으로만 사용한다.
