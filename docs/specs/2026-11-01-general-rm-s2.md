# S2 — 공통 목표 조건부 RL과 실전 대조 검증

사용자 승인: 2026-11-01, 일반화 개선안 구현 및 실전 검증. P1~P4 원본은 보존한다.
연구 명세/공식 STEVE-1/관측기/JAR는 바꾸지 않는다. 게임 입력은 학습 정책이 생성한다.

## 구현 계약

- 공통 목표: 대상 집합, inventory 증가 결과, 수량, caller 소유 완료 조건/선행 조건,
  현재 진행과 증거의 유무를 구조화한다. RM 노드 id를 정책 입력으로 쓰지 않는다.
- 현재 실행 capability는 wood_memory/dirt_fixed만 유지한다. 미지원 과업을 지원한다고 표시하지 않는다.
- actor/value에 공통 context를 실제 연결한다. 목표 대상은 기존 MineCLIP 목표 경로를 유지하고
  수량/결과/단계/진전/증거 유무/최근 조작을 새 context로 전달한다.
- 나무의 단계는 기존 고정 관측으로 결정한다. dirt는 세부 단계 관측이 없으므로 조작/결과 대기라는
  큰 의미만 표시하고 known/reachable은 unknown mask를 준다. 나무 관측을 dirt 관측으로 재사용하지 않는다.
- 선형 head와 작은 nonlinear residual adapter(64 hidden)를 비교한다. adapter 마지막 층은 0 초기화하여
  초기 행동은 공식 head와 같게 한다. backbone은 동결하며 새 actor/value만 학습한다.
- 각 구조에서 RM context 사용/제거 두 조건. 제거 조건도 대상의 기존 목표 임베딩은 유지한다.
  이 비교는 기존 목표 정보 전체의 제거가 아니라 추가 RM context의 효과를 측정한다.
- 진전 없는 상호작용은 server tick 기준 600tick, 파괴 후 미회수는 300tick 개발 한도.
  반복 행동만으로 실패시키지 않는다. 결과 없음/반복/관측 부족을 구분한 근거를 남긴다.
  이는 교정된 pretrained 이상 탐지기 완성이 아니라 고정 관측 기반 개발 판정이다.
- 정체 종료는 task budget의 terminal 실패로 새 S2 환경에서 명시한다. 외부 STOP/기술 오류는 학습 제외.
  1200행동 budget 및 wall 180초는 truncation이며 마지막 상태 value로 bootstrap한다.
- LLM용 failure packet에는 시도/증거/계약을 담는다. 기존 유료 호출 승인을 재사용하지 않고 API 0.
  LLM의 새 지식/온라인 자동 학습 완료로 해석하지 않는다.

## 학습과 2×2 대조

12개 새 train seed 81101001, 81101003, 81101007, 81101009, 81101013, 81101017,
81101019, 81101023, 81101029, 81101031, 81101037, 81101041.
목표 순서: wood1, dirt1, wood2, wood1, dirt2, wood2, wood1, dirt1, wood2, wood1, dirt2, wood2.
각 실행 최대 1200행동/180초. 월드 생성 seed 및 실제 초기 관측/모든 이벤트를 보존한다.
현재 RM 실행기는 시작 월드 ZIP 복원을 지원하지 않는다. 같은 seed의 생성 결과와 초기 관측을 대조하며,
서로 다른 초기 화면을 동일 시작 상태로 주장하지 않는다. ZIP 복원 비교는 별도 통합이 필요하다.

초기 actor가 동일한 네 조건은 새 수집 궤적을 공동 사용한다. **각 조건의 초기 actor가 실제 수집 확률을
최대오차 1e-4 이하로 재현해야** 이 공유를 허용한다. 과거 P4/개발/실패 진단 표본은 학습하지 않는다.
같은 궤적/seed/shuffle/optimizer 한도로 비교하며 실제 업데이트 수와 KL 조기 종료도 보고한다.
선형-off, 선형-on, nonlinear-off, nonlinear-on. 결과 후보는 독립 경로에 저장하고 자동 승격하지 않는다.
gamma .999, lambda 1, actor lr .0001, value lr .001, epochs 2, minibatch 32,
PPO clip .1, reference KL coefficient .1, target KL .02, actor/value 각각 grad clip 1,
optimizer seed 110101. 희소 상위 목표 완료당 1. 추가 버튼/거리 보상 0.
원본 행동 확률, advantage, phase coverage, actor/조건부 계층 gradient/변화량, 실제 train 행 수를 남긴다.
추가 RM 효과를 head 변경 자체와 구분하고 고정 특징에서 context 반사실 검사를 한다.

## 개발 실전 비교

초기 정책 + 네 후보, 다섯 seed(학습과 분리), 각 1회, 회전 순서로 실행한다.

| seed | 목표 | 구분 |
|---|---|---|
| 81102001 | wood1 | 새 월드 |
| 81102003 | dirt1 | 새 월드 |
| 81102007 | wood3 | 학습 밖 수량 |
| 81102009 | wood1 → dirt1 | 새 조합 |
| 81102013 | dirt1 → wood1 | 새 조합/순서 |

전체 완료·실제 인벤토리·개별 목표 진전·정체 이유·소요 tick/wall·추론/제어 지연·GPU 메모리·
단계별 경험·초기 화면 일치도를 기록한다. 미실행 목표는 실패 표본에 포함하지 않는다.
단일 optimizer seed 및 작은 개발 표본이므로 최종 일반화/통계적 우위 주장은 하지 않는다.
미관측 목표/도구/제작 일반화는 이 배치가 검증하지 않는다. 개발 결과를 보고 튜닝하면 새 평가가 필요하다.

## 중단/보존/실행 순서

구현·단위/통합 검사 → 원본/가중치/프로토콜 해시 잠금 → 사용자 승인 근거의 1회 STOP 보존 이동 →
train12 → probability gate 및 4조건 학습 → 개발25 → 실제 nonlinear-on STOP 진단 → 독립 로그 감사/보고.
STOP 진단은 10행동 후 요청, 새 dispatch0/입력 해제/소유PID0/정상 취소를 확인한다.
성능 실패는 재시도하지 않는다. 기술 오류는 증거와 수정 부록을 보존하고 해당 조건을 별도로 처리한다.
종료 시 STOP 유지. results/가중치는 Git 제외. 자동 GitHub 게시/유료 자원/Colab 실행 없음.

진행 상태는 docs/progress.md와 results/rm-general-s2-20261101의 manifest에 기록한다.
