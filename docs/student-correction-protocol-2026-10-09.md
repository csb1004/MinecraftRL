# 학생 방문 정체 상태의 교정 시연 — 2026-10-09

사용자 승인된 다음 작업 단위다. 실제 게임 전 아래 조건을 고정한다.
이전 평가 궤적을 학습으로 옮기거나 원문 연구 목표를 수정하지 않는다.

- 학생: `results/temporal-bc-temporal-20261008/checkpoint.pt`, SHA256
  `8ce793b4f02af4218eb65d7e741ca3ac021d0caee1d891cc324722eeb288fdb9`.
  KL .03 gate 미통과였던 탐색 후보이며 기본 정책 승격은 하지 않는다.
- 목표: 기존 고정 wood embedding, v4 RM/context6, sampling seed 9701.
- 기존 보존 월드의 새 실행: train104743/104759 각각 far·long-corner(4회),
  development104761 far·corner(2회). 이 순서를 유지한다. 최종5시드 미사용.
- 각 조건 최대 제어180초, reset/시드 재선택 없이 한 에피소드에서 진행한다.
  student가 먼저 획득하거나 정체를 감지하지 못하면 교사 시연은 없는 것으로 기록한다.
- 기존 그래프 감지기의 search400/approach200/mine400/collect200 및 목표800tick을
  그대로 쓴다. 첫 정체 event 이후 학생 신규 추론을 중지하고 중립 입력1tick을 전송한다.
  서버 held_key_count0·정상 관측을 확인한 뒤 명시적 privileged 교사가 인계한다.
  STOP/terminal이면 교사를 시작하지 않는다. 전체 제한시간은 인계로 재설정하지 않는다.
- 교사는 실제 pose에서 같은 fixture의 통로로 진입한다. west이면 현재 z에서 x5.5로
  물러난 뒤 통로로, 이미 east이면 벽을 다시 건너지 않고 나무에 접근한다.
  반전 좌표를 반영한다. 지원 범위 밖/벽 내부/열린 GUI는 오류로 보존한다.
- 모든 화면·전송 행동·전후 서버 상태·RM context를 연속 보존한다.
  student prefix와 neutral은 문맥 전용(loss_eligible=false), teacher만 라벨(true)이다.
  임의로 prefix를 잘라 기억을 초기화하지 않는다. 신규 파일은 correction-rows.json이며
  기존 모든 행을 정답으로 보는 teacher 추출기에 직접 투입하지 않는다.
- PPO eligible=false/optimizer0, 학생 획득과 교사 도움 획득을 별도 집계한다.
  이번 단위는 수집·감사다. masked BC 추출/학습 연결 전 bc_training_ready=false를 유지한다.
- source checkpoint/월드/embedding/원문 해시, frame/action 연속성, 교체 시점과 neutral
  receipt, STOP 뒤 dispatch0, 종료 키0, 소유 PID0, GPU/지연/전체 제어 시간까지 감사한다.
  실행 실패는 보존하고 같은 완료 조건을 반복하지 않는다. 환경 오류로 배치가 중단되면
  원인을 먼저 고치고 새 산출물에 미실행 조건만 이어간다.

이는 한 번의 학생 방문 상태 보강이며 완전한 반복 DAgger나 정책 우회 성공이 아니다.
LLM 호출·실제 Colab·새 평가·모델 갱신은 이번 배치에 없다.
