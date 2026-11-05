# Common Step Curriculum Implementation Plan

**Goal:** 일반적인 한 블록 상승·하강·정지를 실제 학생 상태의 교정 시연으로 학습하고 자연 지형 전이를 확인한다.

**Architecture:** 기존 저장 월드 준비, NavigationTrial의 correction 모드, 공식 행동 매핑,
BC와 감사 흐름을 확장한다. 새 step_navigation 조건만 제한된 단차 교사를 사용한다.
자연 평가와 기존 평지 계약을 유지한다. 주 실행 흐름에서 순차 구현하며 서브에이전트는 사용하지 않는다.

**Tech Stack:** Windows PowerShell, WSL Ubuntu-24.04, 기존 isolated Python/torch/MineRL, nbtlib.

**Spec:** [단차 구성과 경계](../../common-step-curriculum-2026-10-23.md).

## 공통 제약

- 프로젝트 D:\DATA\Desktop\MinecraftRL, 원본 명세/가중치/월드 보존, gh 및 전역 환경 변경 없음.
- 실제 서버/client 상태로12틱 안정 도착. 교사와 학습 정책 성공 분리. 매 실행180초 이내.
- train/dev/final분리, 자연 평가 궤적을 BC/PPO에 넣지 않음. final5시드 미사용.
- 수동 STOP 보존. 완료 STOP만 기존 원자적 해제 경로 사용. 새 행동 금지/키 해제/소유PID 정리 검증.
- 유료/API/Colab/공개 게시 없음. 승인된 후속 구현·로컬 실행의 단계별 재승인 없음.

## 검토 초점

회전 후 플레이어 몸의 충돌, targetY가 실제 바닥과 다른 경우, 공중에서의 반복 jump,
평지 반사실 감독이 새 지형에 새어드는 경우, teacher완료를 learned완료로 합산하는 경우를 각각 검사한다.

## 1. 제한된 지형과 교사 계약

- [x] `src/minecraftrl/step_navigation.py`: 재질/회전/경계 검증, 블록·상대 목표 생성,
  train/development 분리, metadata/관측 바인딩, 제한된 step expert. 기존 `fixture_probe.navigate` 재사용.
- [x] `tests/test_step_navigation.py`: 잘못된 재질·회전·높이·원본kind·범위 거부,
  회전된 경계/몸통/목표 검증, 상승 접근/공중/평지/하강/도착 행동을 검증.
- [x] `rl_fixture.py`: 명시적 step_layout을 갖는 새kind만 지원. 기존9검사를 통과했고
  새ZIP/NBT의 블록과 시작/목표 머리 공간을 다시 읽어 검증. 원본 해시 불변을 확인.

## 2. 실제 수집 연결과 교사 자격

- [x] `navigation_refinement.local_targets`, `navigation_trial`, `run_navigation_refinement_trial.py`에
  새kind/goal_contract를 연결. 기존 조건은 그대로. trial의 teacher분기만 새교사 호출.
- [x] 평가 모드에서 목표 예산을 넘겨도 교사로 전환되지 않는 상태 흐름 검사 추가.
- [x] 새 준비 스크립트/고정 config로6train조건 생성. 먼저 작은 교사 자격 실행 후 전체 수집.
  성공/실패 원본을 남기고 데이터 수를 검사. 시작 상태 재생·STOP·입력 해제·PID 증거를 보존.
- [x] 수집 자격은6목표 실제도착·피해0. 미달하면 수집이 성공한 것으로 넘어가지 않는다.

## 3. 새 정답 연결과 실제 가중치 학습

- [x] `navigation_supervision`에 종류가 명시된 단차 expert만 연결한다. 기존 owner/action/feature
  원본 불변, 실제 teacher 행동과 offline label일치 및 공식 매핑 roundtrip을 확인한다.
- [x] 새 단차에는 평지 goal_variations를 적용하지 않고 제외 사유/원본을 학습 감사에 기록한다.
  기존 평지17,240과 사람2,109/355를 보존한다. 새자료 적격성과 중복/분리를 검증한다.
- [x] 학습에 쓰지 않을 새자료 유지KL을 구분하고 학습 범위/초기 함수·동결 텐서 보존 검사 유지.
  정확한 자료 수 확인 후 초기 actor, 혼합량, optimizer seed, 고정 updates와 선택 규칙을 새 config에 고정.
- [x] 가중치 업데이트→재로드·자료·gradient분리 감사→기존 행동 gate. 코드/자료 변경을 한 배치 중 섞지 않는다.

## 4. 전이와 후속 판단

- [ ] 기존 평지/자연 교정과 분리 개발을 진입 조건대로 실행한다. 초기화면 차이가 있으면
  짝 인과 비교로 보고하지 않는다. 결과에 따라 원인 진단과 후속 학습을 계속한다.
- [ ] logs/checkpoints/월드/코드 SHA와 실제지표/실패원인을 기록하고 progress/validation 문서를 갱신한다.
  아직 RM의 stochastic/도착 계약 통합이나 장기 전체 과업 완료를 주장하지 않는다.

이 계획은 새 교사 자료의 품질을 먼저 확인하는 순서다. 미확인 데이터 수·성공률을 전제하지 않으며,
수집 자격 확인 전 학습 실행 config를 만들어 완료된 것으로 취급하지 않는다.
