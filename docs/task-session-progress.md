# 과업 제어 통합 — 2026-10-04

권위: research-spec.md, design-decisions.md, 사용자가 승인한 후속 시나리오.
목표: 동일 MineRL 월드에서 원목 4개 요청 → 흙 4개로 수정 → 일시 중단/해제 → 재개 → 관측 기반 완료.

1. 과업/버전 및 관측 이벤트 기반 수집 RM: 테스트 먼저, 구현, 회귀 검사.
2. 비동기 명령 큐와 버전별 행동 차단, 동일 환경 세션, 기존 감독 프로세스 연결.
3. PowerShell 명령 발행 및 실제 STEVE-1 시나리오; 로그 검증과 문서화.

Ruling: 프로젝트는 Git 저장소가 아니며 사용자가 최종 D: 경로를 지정했다. worktree/자동 Git 초기화 없이 해당 폴더에서 작업하고 이 문서에 진행을 보존한다.
Ruling: 사용자 지침에 따라 서브에이전트 없이 메인 스레드에서 순차 작업/최종 리뷰한다.
Ruling: 이미 승인한 설계를 구체화한다. 구조화된 request/modify/pause/resume/cancel 명령을 먼저 연결하며 유료 LLM 호출·자유 문장 이해는 이번 완료 조건이 아니다.
Ruling: 수집 완료는 해당 revision 수락 시 인벤토리 기준 대비 목표군의 현재 총수량 증가 4개 이상이다. 이전 revision의 획득은 새 목표에 이월하지 않는다. pause/resume은 기준과 진행을 유지한다.
Ruling: 세션당 제어 벽시계 180초(중단 대기 포함), 초기화 600초 및 기존 종료 유예를 유지한다. 환경 reset은 단 한 번이며 정책 recurrent state/텍스트 목표만 revision 변경 또는 resume 때 재설정한다.
Pre-flight: 기존 supervisor는 첫 episode_started 이벤트로 제한 시간을 설정하므로 세션도 같은 이벤트를 첫 줄에 기록한다. 기존 정책 벤치마크 run_episode는 유지한다.

Task 1: complete — 상태/RM/명령 버스 구현. 신규 테스트에서 기능 부재 확인 후 통과. 목표 수정은 revision 증가 및 새 baseline, pause/resume은 baseline 유지. 오래된 revision/중복/위조 완료 명령 검증 포함.
Task 2: complete — 동일 월드 세션 및 파일 명령 입력 구현. 추론 도중 수정/취소, 일시 중단 중 무행동, 재개 후 인벤토리 완료 테스트 통과. 기존 episode/suite 실행은 별도 유지.
Ruling: 세션의 pause 입력 해제는 최종 shutdown 증거가 아니다. 강제 종료 요약 복구는 stop_acknowledged 이후의 input_release만 채택한다. 기존 오판정을 새 회귀 테스트에서 재현 후 수정했다.
검사: WSL 32개 전체 통과 (`results/task-session-final-preflight-tests.log`), PowerShell 명령 발행 기능 검사 통과. Windows 검사에서는 Linux 전용 3개를 제외했다.
Task 3 진행: 실제 STEVE-1 동일 월드 시나리오 seed 130363, 제어 예산 180초. 명령 발행/수정/중단/재개를 자동 검증하되 인벤토리나 정책 행동은 주입하지 않는다. 시나리오용 시드는 앞선 실행에서 흙 획득을 관측한 고정 시드이며 성공률 추정 표본이 아니다.
이전 마지막 STOP은 원래 진단 결과 폴더의 STOP.saved로 보존 이동했다.

Task 3 실게임 주요 시나리오 완료: `results/task-scenario-130363-20261003T201117656522Z`.
104.713초, 정책 step 860회, inference 862회(폐기 2회), env reset 1회. 원목 revision 1에서
흙 revision 2로 전환, pause/resume 수락, 실제 baseline 0 → dirt 4로 완료.
이전 revision의 수정 후 dispatch 0, pause 중 dispatch 0, 중립 입력 해제 확인,
worker exit 0 및 자식 잔존 0. 추론 중앙값/p95 43.07/52.15ms, 제어 주기 110.74/137.58ms.
추가 확인: 별도 세션에서 PowerShell request/cancel 명령의 실게임 수락과 종료를 검증 중.

Task 3: complete. PowerShell 취소 시험은 191 step 뒤 수락, 이후 행동 0, 중립 해제/자식 정리 확인.
정상 취소의 CLI exit 2를 사용자 예외로 표시하는 문제를 발견해 회귀 테스트 후 session에만
반환 의미를 수정했다. 실행 실패/강제 종료를 숨기지 않고 취소 상태 및 worker 코드는 보존한다.
실제 환경의 두 번째 취소 시험은 사용자 PowerShell exit 0으로 종료했다.
최신 WSL 전체 테스트 33개 통과. 세부 결과: `task-session-validation-2026-10-04.md`.
최종 메인 스레드 검토 완료. 미구현 범위는 대화 해석/범용 RM 컴파일/다중 과업 서비스 등
이번 승인 시나리오 밖의 후속 단계이며, 완료된 기능으로 주장하지 않는다.
