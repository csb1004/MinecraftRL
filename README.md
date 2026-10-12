# MinecraftRL

공개 Minecraft 영상 사전학습 정책(VPT·STEVE-1)에 과업 기억, Reward Machine(RM),
상위 계획과 학습된 하위 행동을 연결하는 로컬 연구 프로젝트다.
현재 초점은 **실제 게임에서 정체를 감지하고 새로운 경로로 탈출하는 능력**이다.
원래 [연구 명세 20개 항목](docs/research-spec.md)과 장기 목표를 유지한다.

## 현재 결과

최신 기록은 [진행 상황](docs/progress.md), [남은 작업 단위](docs/next-task-units.md)에서 확인한다.
2026-10-12 벽 정보 제거 비교는 [고정 프로토콜](docs/blind-recovery-protocol-2026-10-12.md)과
[실제 결과](docs/blind-recovery-validation-2026-10-12.md)에 정리한다.

벽 정보를 주지 않은 far 실행에서 정체 후 벽 횡단을 확인했지만 원목 획득은 실패했다.
외곽에서 도달 불가능한 목표를 다시 고르는 문제가 남아 있다. corner의 같은 조건은
원래 정책이 먼저 완료해 복구 미실행으로 분리했다. 네 게임4,599전이를 감사했다.

| 영역 | 확인한 범위 |
|---|---|
| 공개 정책 실행 | GTX 1060 3GB / Windows·WSL에서 공식 MineRL 계열 VPT·STEVE-1 실제 실행. [초기 VPT](docs/local-vpt-results-2026-10-03.md)·[STEVE-1](docs/local-steve1-results-2026-10-03.md) 결과 |
| 과업과 대화 | 목표·완료 이력을 보존하는 RM, 순차 수집, 대화 요청과 중단/재개. [과업 분해 계약](docs/task-decomposition.md) |
| 학습 | STEVE-1 action head·일부 표현/기억층의 로컬 PPO·BC 실험. 실패한 후보와 기존 기본 정책을 구분해 보존 |
| 상위 계획 | 정체·반복·실패 이력을 입력으로 하위 계획 제안/검증/실행. [실제 LLM 1회 검사](docs/subgoal-live-validation-2026-10-10.md)는 연결을 확인했지만 해당 게임의 원목 획득은 실패 |
| 공간 기억 | 방문 구역/방향, 실패 이력, 명명 장소. [실제 F3 좌표 판독](docs/spatial-exploration-validation-2026-10-10.md)은 별도 관측 진단이며 live planner의 좌표는 아직 서버 출처 |
| 이동 하위 행동 | 상대 목표를 입력받는 별도 학습 head. [10월 10일 결과](docs/waypoint-learning-validation-2026-10-10.md): 경로를 준 초기 직접 진단에서 far 획득, corner 실패. 정체 후 복구와 구분 |
| 종료 | STOP 뒤 새 정책 행동 금지, 중립 입력 해제, 해당 실행의 자식 프로세스 정리. 실행별 원시 로그로 감사 |

일반적인 우회·탐험 성능이나 인간다움은 입증하지 않았다. 알려진 fixture 경로, 규칙 기반
후보 선택, 실제 LLM 계획을 구분한다. 서버 좌표/RM 입력을 쓰는 조건은 영상 전용 조건이 아니다.
모델 로드·합성 테스트·교사 시연은 학습 정책 성공으로 세지 않는다. VPT 단독에 자연어 지시
수행 능력을 가정하지 않으며, MineRL 실행을 최신 Bukkit 호환성으로 해석하지 않는다.

## 저장소와 재현 범위

이 저장소는 코드·테스트·프로토콜·보고서의 첫 정리본이다. 작은 결과 자료는
[docs/reports](docs/reports/2026-10-12)에 포함한다. 비밀정보, 외부 소스 checkout,
공식/로컬 가중치, 전체 프레임·궤적·월드와 런타임은 제외한다. 원본들은 로컬 `results/`,
`weights/`, `.runtime/`, `external/`에 보존한다. **clone만으로 기존 학습 결과를 재실행할 수는 없다.**
각 보고서의 SHA256과 실행 소스 사본으로 어떤 로컬 자산을 사용했는지 추적한다.

| 경로 | 내용 |
|---|---|
| `src/minecraftrl/` | 정책 어댑터, RM, 학습/계획, 기록·중단·월드 관리 |
| `scripts/` | Windows/WSL 설정, 학습, 일회 실험, 전이 재생 감사 |
| `tests/` | 입력 계약·계획·보상·종료 회귀 검사 |
| `configs/` | 고정 공식 소스 커밋, 환경 후보, 모델 출처 |
| `docs/` | 연구 명세·설계 결정·날짜별 프로토콜과 성공/실패 기록 |
| `notebooks/` | Colab 준비본. 실제 클라우드 실행 결과는 아님 |

일회 실행 도구는 완료한 결과 경로를 덮어쓰지 않는다. 완료 배치를 그대로 재실행하지 말고
새 프로토콜·새 결과 경로를 사용한다. 이전 긴 README는 [변경 전 기록](README-history-2026-10-10.md)으로 보존했다.

## 로컬 설치와 실행

실제 프로젝트 경로는 `D:\DATA\Desktop\MinecraftRL`이다. Windows에서는 PowerShell,
Linux 정책 실행은 WSL Ubuntu-24.04를 사용한다. 시스템 Python/Java/드라이버 대신
프로젝트 안의 격리 환경을 구성한다. `pip install`만으로 MineRL/CUDA/Java가 준비되지는 않는다.

현재 연구 PC에는 설치가 완료돼 있다. 다음은 새 설치의 시작 절차이며,
과거 클라우드 전달 ZIP 검사와 이후 로컬 실게임 검증은 [설치 기록](docs/local-validation-2026-10-03.md)에 구분했다.

```powershell
Set-Location -LiteralPath 'D:\DATA\Desktop\MinecraftRL'
.\scripts\Test-LocalEnvironment.ps1
.\scripts\Get-OfficialSources.ps1 -Policy steve1
.\scripts\Invoke-PolicyProbe.ps1 -Policy steve1 -Operation setup
.\scripts\Invoke-PolicyProbe.ps1 -Policy steve1 -Operation download
.\scripts\Invoke-PolicyProbe.ps1 -Policy steve1 -Operation preflight
```

VPT는 `-Policy vpt`로 별도 환경을 사용한다. [공식 출처와 라이선스/제약](docs/policy-sources.md),
[소스 pin](configs/sources.lock.json)을 먼저 확인한다. 외부 모델·Minecraft 자산을 이 저장소에
재배포하지 않는다. 기존 환경을 자동 삭제하거나 시스템 설정을 바꾸지 않는다.
STEVE-1의 알려진 gym 메타데이터 충돌 예외는 설치 receipt에 남기며 `pip check` 전체 통과로 표현하지 않는다.

새 실행에서는 제어 시간을 최대 180초로 제한하고 관측/행동/인벤토리/추론 지연/메모리/종료를 기록한다.
실행 절차와 판정은 [실험 프로토콜](docs/experiment-protocol.md)을 따른다.

```powershell
.\scripts\Invoke-PolicyProbe.ps1 -Policy steve1 -Operation episode -Condition wood -Seconds 180
```

별도 PowerShell에서 중단할 수 있다.

```powershell
.\scripts\Stop-PolicyProbe.ps1
```

`results/STOP`은 래치다. 중단 당시 진행 중이던 한 step은 끝날 수 있지만 이후 새 정책 행동은
전달하지 않는다. 중립 입력 해제는 별도로 기록한다. 자동 실험 도구는 이전 **자동 종료** 표식을
보존한 뒤 새 실행을 시작하며 사용자 STOP을 자동 해제하지 않는다.

## 검사

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src'
python -m unittest discover -s tests -v
```

하네스 검사는 실게임 성공률과 별개다. 의존성/운영체제 때문에 제외된 검사는 결과에 표시한다.
실게임 감사 도구는 원본 궤적과 해당 가중치를 필요로 한다. Git 업로드 전에는 staged 내용의
제외 자산·용량·일반적인 비밀 패턴을 `scripts/check_publish_content.py`로 검사하고 파일 목록도 검토한다.

대화 API는 선택 기능이며 기본 정책 실행/이번 벽 비교에는 키가 필요 없다. 사용할 때는
[.env.example](.env.example)의 설정을 로컬 비밀 파일로 관리한다. 유료 API·Colab 실행은
실험별 승인 범위와 호출 예산을 따른다.
