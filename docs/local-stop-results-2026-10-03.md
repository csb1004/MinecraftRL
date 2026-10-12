# 실제 환경 중단 시험 — 2026-10-03

로컬 Windows/WSL GTX 1060 3GB, 고정 시드 104729에서 VPT와 STEVE-1을 각각 시험했다. 이 진단은 정상 180초 수집 실험의 성공/실패 분모에 포함하지 않는다.

| 정책 | STOP 단계 | 요청→ack ms | 입력 해제 | worker exit | 이후 새 정책 행동 | 잔존 자식 |
|---|---|---:|---|---:|---:|---:|
| vpt | initialization | 10292.57 | environment_not_initialized_no_policy_input | -9 | 0 | 0 |
| vpt | inference | 1539.11 | neutral_command_sent | 2 | 0 | 0 |
| vpt | step | 126.42 | neutral_command_sent | 2 | 0 | 0 |
| steve1 | initialization | 10173.92 | environment_not_initialized_no_policy_input | -9 | 0 | 0 |
| steve1 | inference | 244.86 | neutral_command_sent | 2 | 0 | 0 |
| steve1 | step | 110.66 | neutral_command_sent | 2 | 0 | 0 |

모든 행에서 STOP 요청이 실제 해당 단계 안에 있었음을 확인했다. 추론/step은 call_started_s/call_finished_s 사이에 파일 생성이 완료됐는지 대조했다. 요청 이후와 acknowledgment 이후 새 policy dispatch는 모두 0회였다. 중립 입력은 controller=cleanup으로 구분한다.

VPT step 및 STEVE-1 추론/step 시험은 두 번째 호출을 대상으로 하므로, 첫 정책 행동이 이미 전달된 상태에서 중단했다. VPT 추론 시험은 첫 호출을 대상으로 했다.

초기화 시험은 실제 Java 프로세스 시작 후 STOP을 요청했다. reset이 즉시 반환하지 않아 약 10초의 유예 후 감독 프로세스가 종료했다. 정책 행동은 시작되지 않았으며 neutral step은 해당되지 않았다. worker exit -9와 EOF 오류를 정상적인 환경 close 성공으로 해석하지 않는다.

이미 실행 중인 CUDA 추론이나 환경 호출 자체를 되돌리는 보장은 아니다. 완료된 호출 이후 새 정책 행동을 차단하고 입력 해제 및 소유 프로세스 종료를 확인했다. 단계당 소수 시험으로 모든 고장 상황의 최악 지연을 보장하지 않는다.

첫 VPT 추론 시험에서는 neutral step 성공 이벤트가 있었으나 후속 close가 유예를 넘겨 최종 summary가 유실됐다. 원본 summary/evaluation은 그대로 보존하고 shutdown-evidence-review.json에 증거를 대조했다. 감독 요약 복구를 수정한 뒤 회귀 포함 22개 하네스 검사가 통과했고 VPT 추론 재시험도 완료했다. 위 표에는 재시험을 사용하고 원래 진단도 aggregate의 all_attempts에 남겼다.

근거: `results/real-stop-aggregate.json` 및 아래 개별 결과 폴더. 각각 request, events, summary, stop-test-evaluation을 보존했다.

- `results/real-stop-vpt-initialization-20261003T120454134344Z`
- `results/real-stop-vpt-inference-20261003T121312659874Z`
- `results/real-stop-vpt-step-20261003T121553689586Z`
- `results/real-stop-steve1-initialization-20261003T121816523446Z`
- `results/real-stop-steve1-inference-20261003T122124460156Z`
- `results/real-stop-steve1-step-20261003T122458554502Z`
