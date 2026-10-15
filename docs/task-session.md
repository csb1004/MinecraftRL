# 동일 월드 과업 제어

STEVE-1의 검증된 화면 기반 정책 위에 구조화된 과업 제어를 연결한다. 일반 대화 해석,
임의 Reward Machine 컴파일러, 계획 재수립, Bukkit 연결은 이 경로에 포함하지 않는다.
이번 RM은 wood/dirt 수집 한 과업의 관측 기반 진행·완료 전이에 한정한다.

2026-10-15 추가: 명시적 `-TaskPriorities` 모드는 여러 수집 목표를 보관하고 미완료 기간,
실행 대기와 선행 조건으로 다음 목표를 선택한다. 기존 기본 모드는 유지한다.
[명령·시간·저장 계약](goal-aging.md)을 참조한다.

## 명령과 권한

명령은 command_id, task_id, op를 갖는다. request/modify만 goal(wood 또는 dirt)과
count(1–64)를 받는다. modify/pause/resume/cancel에는 expected_revision이 필요하다.
목표나 count를 수정하면 revision이 증가한다. pause/resume은 revision과 baseline을
유지하고 실행 세대만 바꿔 이전 추론 결과의 재전송을 차단한다.

명령 파일 제출은 수락을 뜻하지 않는다. events.jsonl의 command_acknowledged 또는
command_rejected가 결과다. 동일 command_id와 동일 내용은 한 번만 실행하며, 같은 ID에
다른 내용을 보내면 거부한다. 완료·보상·인벤토리 값을 명령으로 제공할 수 없다.
이 입력은 사용자가 명시적으로 실행을 요청한 구조화된 명령을 위한 경계다. 향후 대화
해석기가 만든 제안/일반 채팅을 이 폴더에 곧바로 게시해서는 안 되며 의도·commitment 판정을
거쳐야 한다. 현재 구조화 제어기는 원격 인증 API가 아니다. 후속 추가한 제한된 한국어 입력
경계는 [dialogue-interface.md](dialogue-interface.md)에 설명한다. 일반 LLM 대화 해석은 미구현이다.

명령 수신은 별도 스레드가 담당한다. 큐에 들어온 명령은 진행 중인 추론의 실행 세대를
무효화한다. 이미 env.step에 전달된 호출은 완료될 수 있지만, 명령 수락 전 입력을
중립화하고 이후 이전 버전 행동은 전달하지 않는다. 정책과 게임 호출은 한 스레드에서
순차 수행한다. 중립 step의 최신 화면/인벤토리를 후속 추론과 baseline에 사용한다.

## 관측과 완료

환경 reset은 세션당 한 번이다. 수집 목표는 revision을 수락한 시점의 실제 인벤토리 총량을
baseline으로 삼는다. 동일 목표군의 현재 총량이 baseline보다 count 이상 증가하면
inventory_target_reached 이벤트를 만들고 완료한다. wood는 오버월드 6종 원목이다.
이전 목표에서 얻은 아이템, 모델의 자기 보고, 아이템별 과거 최대값을 합산하지 않는다.

canonical 이벤트: session_ready, command_queued, command_acknowledged,
command_rejected, inventory_observed, inventory_target_reached, action_discarded,
policy_dispatch_started/finished, input_release, stop_acknowledged.
각 행동은 task_id/revision/generation으로 추적한다. 관측 sequence는 환경 관측 순서이며
추론·step의 실제 호출 시작/종료 시각도 기록한다.

pause는 정책 행동을 중지하고 중립 입력을 보낸다. 월드를 새로 생성하거나 저장을
되돌리지 않는다. Minecraft 전체 프로세스/시간을 OS 수준에서 얼린다는 보장은 아니다.
resume은 최신 중립 관측에서 정책 recurrent state/조건을 재설정하며 목표 진행은 유지한다.
completed/cancelled 뒤에는 세션을 닫는다. 여러 독립 과업을 영구 서비스로 유지하는
기능이나 프로세스 재시작 뒤 과업 복구는 아직 없다.

## PowerShell 사용

기존 STOP이 있다면 직전 종료 결과를 검토한 뒤 그 파일을 해당 결과 폴더에 보존 이동한다.
자동으로 STOP을 해제하지 않는다. 한 GPU에서 세션을 동시에 실행하지 않는다.

```powershell
.\scripts\Invoke-PolicyProbe.ps1 -Policy steve1 -Operation session -Seed 130363 -Seconds 180
```

출력된 session 경로를 Windows 경로로 지정하고, 다른 PowerShell에서 명령을 보낸다.
아래 경로의 시간 부분은 실제 출력값으로 바꾼다.

```powershell
$session = 'D:\DATA\Desktop\MinecraftRL\results\session-steve1-실제시간'
.\scripts\Send-TaskCommand.ps1 -SessionPath $session -Operation request -Goal wood -Count 4
.\scripts\Send-TaskCommand.ps1 -SessionPath $session -Operation modify -ExpectedRevision 1 -Goal dirt -Count 4
.\scripts\Send-TaskCommand.ps1 -SessionPath $session -Operation pause -ExpectedRevision 2
.\scripts\Send-TaskCommand.ps1 -SessionPath $session -Operation resume -ExpectedRevision 2
# 완료 전에 취소할 경우
.\scripts\Send-TaskCommand.ps1 -SessionPath $session -Operation cancel -ExpectedRevision 2
```

각 명령의 수락을 확인한 뒤 다음 명령을 보낸다. 송신기는 완성된 JSON을 임시 파일에서
원자적으로 이름 변경해 게시한다. 외부 프로그램/에이전트도 같은 구조화된 파일 인터페이스를
사용할 수 있으며 불완전한 JSON을 직접 쓰는 것은 지원하지 않는다.

## 실제 시나리오 및 측정

```powershell
.\scripts\Invoke-PolicyProbe.ps1 -Policy steve1 -Operation task-scenario -Seed 130363 -Seconds 180
```

원목 4개 요청 → 추론 도중 흙 4개로 수정 → 흙 제어 시작 후 일시 중단 → 2초 대기 → 재개를
실제 명령 파일로 수행한다. 이후 정책이 실제 흙 4개를 얻어야 scenario_passed가 true다.
정책 행동·월드·인벤토리는 시나리오에서 조작하지 않는다. 성공하지 않으면 false와 원본
로그를 남기며 API 행동으로 대체하지 않는다. 이 시험은 기존 5-seed 성능 표본과 구분한다.

초기화 제한 600초, 제어 벽시계 제한 180초(명령 대기/일시 중단/정책 재설정 포함),
종료 유예 10초를 유지한다. 추론은 CUDA synchronize로 측정하고 전체 제어 주기에는
프레임 저장·추론·step·관측 기록이 포함된다. 명령 대기와 정책 조건 재설정은 해당 주기
분포 밖이지만 전체 제어 시간에는 포함된다. VRAM은 PyTorch와 장치 전체 표본을 구분한다.
inventory_observed의 frame은 해당 행동에 사용한 사전 입력 화면이고 inventory는 행동 후
관측이다. 세션은 inventory/life/location만 기록하므로 fullstats를 기록했던 이전 정책
벤치마크와 제어 주기 수치를 직접 성능 개선으로 비교하지 않는다.

강제 종료 시 과거 pause의 input_release를 최종 해제 성공으로 오인하지 않는다.
감독 요약 복구에는 최종 stop_acknowledged 뒤에 기록된 input_release만 사용한다.
정상 과업 취소는 task.status=cancelled, task_completed=false, execution_valid=false로
남기되 입력 해제와 프로세스 종료가 확인되면 사용자 CLI는 성공 코드 0을 반환한다.
내부 worker의 2는 비완료 실행을 나타내며 그대로 보존한다. 강제 종료나 해제/close 실패는
이 정상 취소 처리에 포함하지 않는다.
