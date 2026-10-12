# 과업 수정·중단·재개 로컬 검증 — 2026-10-04

## 구현 범위

기존 STEVE-1 실행 환경에 과업 ID/revision, 구조화된 명령 입력, 수집용 관측 이벤트/RM,
실행 세대에 따른 오래된 행동 차단을 연결했다. 기존 VPT/STEVE-1 벤치마크 경로와
원본 결과는 보존했다. 실행 방법과 정확한 의미는 `task-session.md`에 있다.

## 동일 월드 실제 시나리오

Windows/WSL GTX 1060 3GB, 공식 STEVE-1 및 MineRL HumanSurvival, seed 130363.
결과 폴더: `results/task-scenario-130363-20261003T201117656522Z` (폴더 시각은 UTC).

| 단계 | 제어 시작 후 수락 시각 | 결과 |
|---|---:|---|
| 원목 4개 요청 | 0.037초 | gather/revision 1, running |
| 흙 4개로 수정 | 5.089초 | revision 2, 새 인벤토리 baseline 0 |
| 일시 중단 | 5.425초 | 입력 중립화 후 paused |
| 재개 | 7.682초 | 같은 revision/baseline을 유지하고 running |
| 실제 관측으로 완료 | 제어 구간 104.713초 내 | dirt 4, completed |

- 환경 reset **1회**, 정책 step 860회(원목 버전 3회 + 흙 버전 857회).
- 수정은 실제 4번째 추론 도중, pause는 실제 6번째 추론 도중 큐에 들어왔다.
  두 추론의 결과를 전송 전에 폐기했다. 모델 호출의 실제 시작/종료 시각과 대조했다.
- 수정 수락 이후 이전 revision의 정책 dispatch **0회**.
- pause 수락부터 resume 수락까지 정책 dispatch **0회**.
- 최종 과업은 revision 2, dirt baseline 0 → 관측 4로 완료했다.
  원목 4개 획득을 완료했다는 뜻이 아니라 원목 과업을 수행하다 목표를 변경한 시험이다.
- neutral_command_sent, worker exit 0, 소유 자식 프로세스 잔존 0개.
- 추론 중앙값/p95 **43.07/52.15ms**, 제어 주기 **110.74/137.58ms**.
- PyTorch peak allocated/reserved **2098.72/2144 MiB**; 장치 전체 표본 최대 **2796 MiB**.

`scenario-evaluation.json`의 scenario_passed=true와 `control-evidence.json`의 명령/호출
시각이 근거다. 인벤토리 및 행동을 시나리오 실행기에서 주입하지 않았다. 일반 성능 추정용
다중 시드 실험이 아니며 이전 15회 벤치마크의 표본에 추가하지 않는다. 세션의 로그 범위가
다르므로 이전 제어 주기보다 빨라진 것을 정책 성능 개선으로 주장하지 않는다.

## PowerShell 과업 취소

`results/session-steve1-20261003T201716547248Z`에서 PowerShell로 dirt 64 요청을 보내고
191개 정책 step 이후 cancel을 보냈다. 큐 수신→수락 45.56ms, 수락 이후 정책 dispatch
0회, 중립 입력 해제와 소유 자식 잔존 0개를 확인했다. 과업은 cancelled이며 수집 완료나
성공/실패 성능 표본으로 계산하지 않는다.

이때 정상 과업 취소에도 기존 CLI가 비완료 worker exit 2를 그대로 전달해 PowerShell이
예외를 표시했다. 원본을 보존하고, 과업 취소·중립 해제·자식 정리가 확인된 session에만
사용자 CLI가 0을 반환하도록 수정했다. task_completed/ execution_valid는 false로 유지한다.
강제 종료, 해제 실패, close 오류는 정상 취소로 처리하지 않는다.

`results/session-steve1-20261003T202246592263Z`에서 요청/취소를 미리 제출한 실제 환경
재시험은 reset 이후 정책 행동 0회로 취소됐고 PowerShell이 예외 없이 exit 0으로 종료했다.
내부 worker exit 2 및 취소 상태는 원본 summary에 그대로 보존했다. 이 재시험은 진행 중인
정책 취소의 성능 표본이 아니며 앞선 191-step 취소 시험을 대신하지 않는다.

## 검사와 수정

- WSL 하네스 **33개 통과** (`results/task-session-tests-after-cancel-fix.log`).
- 새 기능의 부재/잘못된 동작을 먼저 확인한 테스트 로그와 통과 로그를 results에 보존했다.
- PowerShell 송신기의 실제 JSON 발행·구문 검사, WSL 실행기의 실제 session/scenario 경로 확인.
- 예전 pause의 해제 이벤트를 최종 강제 종료의 해제 성공으로 오인하는 회귀를 재현했다.
  supervisor는 최종 stop_acknowledged 뒤의 input_release만 복구하도록 수정했다.
- 메인 스레드에서 최종 검토: revision 경합, pause 중 행동 차단, 중복 명령, 잘못된 상태·입력,
  관측 기반 완료, 중립 step 후 최신 관측 사용, 취소와 기술 실패의 분리, 기존 벤치마크 경로를 확인했다.
  사용자 지침에 따라 별도 서브에이전트를 사용하지 않았다.

## 범위의 한계

이번 구현은 수집 전용 RM과 명시적인 구조화 명령을 연결한 단계다. 자연어 대화의
CHAT/REQUEST/MODIFY 구분, 임의 RM compiler/validator, 다중 과업 지속 서비스, 재계획,
추가 스킬 및 persona/memory/social 평가는 후속 작업이다. 최신 Bukkit 서버 호환성과
인간다움·문화 이해를 입증하지 않는다. 유료 자원·Colab 연결·GitHub push는 수행하지 않았다.
