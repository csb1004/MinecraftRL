# 로컬 대화 실행 경계 검증 — 2026-10-04 KST

Windows PowerShell / WSL Ubuntu-24.04 / GTX 1060 3GB에서 실행했다. 유료 API, 클라우드,
Colab, API 봇 행동 대체는 사용하지 않았다. 아래는 기능 통합 시험이며 기존 정책 벤치마크
15회에 추가하거나 자연어 이해 성공률로 계산하지 않는다.

## 동일 월드 대화 제어

원본: `results/dialogue-scenario-130363-20261003T221958547365Z`.
seed 130363, 제어 한도 180초. 이 시드는 앞선 기능 시험에서도 사용한 시드다.

- 비실행 문장 6개: 인사, 미래 제안, 질문, 인용, 부정, 단독 ‘응’. 모두 명령 없음.
- PowerShell에서 추가한 한국어 질문은 QUESTION/no_action. 같은 메시지 ID 재전송은
  dialogue_duplicate 1회로 기록됐다.
- 한국어 원목 4개 요청 → 흙 4개 수정 → 잠시 중단 → 재개를 실제 worker가 수락했다.
- environment reset 1회. 제어 49.912초, 정책 step 367회(원목 revision 1: 3, 흙 revision 2: 364).
- 실제 dirt baseline 0 → 현재 5. 목표 4개 이상을 인벤토리 관측으로 충족했다.
- 추론 369회 중 과거 제어 상태의 2회 폐기(attempt 4, 6). 요청 전 정책 행동 0,
  수정 수락 후 이전 revision 행동 0, pause 수락부터 resume 수락까지 행동 0.
- 추론 지연 중앙값/p95 43.34/53.59ms, 전체 제어 주기 중앙값/p95 112.32/152.39ms.
- PyTorch 최대 allocated/reserved 2098.72/2144 MiB. 전체 GPU 샘플 최대 사용 2782 MiB
  (다른 프로세스 포함, 순간 peak를 놓칠 수 있음). OOM 없음.
- 수정 enqueue→ack 81.41ms, pause 79.49ms, resume 201.64ms.
- 종료 neutral_command_sent, worker exit 0, 감독이 추적한 자식 잔존 0.

`scenario-evaluation.json` 및 `control-evidence.json`에 판정과 시간 증거를 보존했다.
PowerShell 발행 성공은 수락과 다르다. 종료 무렵 추가 발행한 ID 상충 메시지 1개는 inbox
종료 뒤 남아 처리되지 않았으므로 그 건을 거부 검증 성공으로 계산하지 않는다.

## PowerShell 대화 취소

원본: `results/session-steve1-20261003T222414058524Z`.
PowerShell로 ‘흙 64개 모아줘’를 발행하고 실제 정책 행동 22회 뒤 ‘작업 취소해줘’를 발행했다.
취소 enqueue→ack 39.59ms, 수락 이후 새 정책 dispatch 0, 중립 해제와 잔존 자식 0을 확인했다.
제어 6.379초, 인벤토리 흙 순증가 0. 정상 취소로 task_completed=false, execution_valid=false,
내부 worker exit 2를 보존하고 사용자 PowerShell은 exit 0으로 반환했다.
이번 취소는 추론 도중 수락을 의도적으로 맞춘 시험이 아니며, 앞선 추론 중 STOP 시험과 구분한다.

같은 요청 ID/내용의 재전송은 dialogue_duplicate 1회, 같은 ID에 취소 문장으로 바꾼 입력은
message_id reused with different content로 거부 1회다. 별도 새 ID의 취소만 수락됐다.
`control-evidence.json`과 `dialogue-evidence.json`에 기록했다. 종료된 세션 및 프로젝트 results
밖의 경로에 대한 PowerShell 발행 거부도 실제 호출로 확인했다.

최종 프로젝트 관련 WSL Python/Java 프로세스 잔존 0을 확인했다
(`results/dialogue-final-process-check-20261004.json`). 마지막 STOP을 유지한다.

## 회귀와 범위

WSL 전체 39개 테스트 통과 (`results/dialogue-wsl-tests-fixed.log`). Windows는 35개 통과,
Linux 전용 4개 제외. PowerShell 두 스크립트와 WSL wrapper 구문 검사 통과.
메시지 중복/상충, 위조 필드, 수량/목표 제한, 오래된 revision, 의미 후보 불일치를 검사했다.

첫 WSL 검사에서 기존 자식 정리 테스트가 100ms startup 제한 때문에 PID 생성 전 종료됐다.
독립 재현에서도 동일했고 자식 Python import 측정은 247ms였다. 자식이 종료 핸들러 설치를
완료한 신호 뒤 제어 deadline을 시작하도록 테스트를 수정했다. 실제 실행기의 제한/종료
코드는 바꾸지 않았다. 첫 실패 및 재현 로그를 삭제하지 않았다.

이 결과는 제한된 한국어 문법에서의 실행 경계 검증이다. 자유 대화, LLM 연동, 대화 기억,
일반 semantic parser, 전 범위 intent 분류 정확도, 장기 과업/페르소나 기능은 미구현이다.
원문 연구 명세와 장기 목표는 유지한다. 기존 VPT에 자연어 지시 능력을 추가한 것도 아니다.
