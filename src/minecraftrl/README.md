# 패키지 지도

[전체 흐름](../../docs/handbook/architecture.md) · [개발 규칙](../../docs/handbook/development.md)

공용 Python 코드는 이 패키지에 둔다. `scripts/` 파일을 라이브러리처럼 import하는 새 의존성은 만들지 않는다.
과거 클래스/모듈 경로는 저장된 자료의 재현을 위해 유지한다. 파일이 남아 있다는 이유로 현재 실행 기능으로 간주하지 않는다.

| 영역 | 주요 모듈 | 역할/상태 |
|---|---|---|
| 진입·수명 | `cli_arguments`, `cli`, `runtime`, `processes` | 인수 선언, 모드 검증, 실행, STOP·소유 PID 정리 |
| 공식 정책·환경 | `policies`, `preflight`, `minerl_runtime`, `cuda_startup` | STEVE-1/VPT와 MineRL 경계 |
| 목표/계약 | `goal_spec`, `goal_queue`, `rm_program`, `execution_contract` | 같은 요청 의미, 상위 큐, 경계에서 우선순위 반영 |
| 조건·관측 | `rm_conditioning`, `semantic_goal`, `interaction_progress`, `logo_validators` | 목표 벡터/단계, 근거와 성공 판정 |
| 현재 E2 학습 정책 | `execution_learning`, `execution_returns`, `execution_training` | 현재 효과 context, 전체 과업 return, 보존/critic 보정 |
| 독립 학습기 | `training/adapter`, `training/validation`, `training/provenance` | 자료 admission, PPO, 구현 해시, 후보 저장 |
| 재사용 학습 기반 | `conditioned_head`, `rl_core`, `rl_opt`, `rl_policy`, `s1_learning`, `s2_learning`, `recovery_learning`, `effect_learning`, `effect_skill` | E2가 사용하는 공통 수학/부모 구현 및 앞선 비교 경로 |
| 월드/세션 | `world_store`, `policy_memory`, `tasks`, `scheduled_tasks`, `task_runtime`, `startup_control` | 공용 세션·저장·기억·초기화; E2와 실행 조건은 다름 |
| 과거 연구/재사용 후보 | `navigation_*`, `waypoint_*`, `step_*`, `human_*`, `planning_recovery`, `subgoal_*`, `dialogue*`, `recovery_broker` | 전용 자료/실행기가 폐기된 경로도 있음; E2의 자동 루프가 아님 |
| 명시적 GUI 실험 | `craft*`, `recipe_book`, `workshop*`, `tool_collection` | 보조 제어/과거 실험; 현재 학습 정책의 일반 제작 성공으로 세지 않음 |
| 배포 보조 | `artifact_*` | 파일 해시·릴리스 보조; 자동 게시하지 않음 |

Java 파일은 MineRL 관측·초기화 패치 소스다. 임의로 시스템 JAR에 적용하지 않는다.
관측기의 고정/privileged 여부는 각 프로토콜에서 정한다.

새 학습 능력을 추가할 때는 등록 이름만 늘리지 말고 실제 입력, 결과 판정, 학습 자료, 회귀 검사까지 연결한다.
E2 이후 연구에서 backbone/학습 경계를 바꾼다면 별도 명세와 비교 조건을 먼저 만든다.
