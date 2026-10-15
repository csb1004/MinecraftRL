# MinecraftRL

공개 Minecraft 영상 사전학습 정책(VPT·STEVE-1)에 과업 기억, Reward Machine(RM),
상위 계획과 학습된 하위 행동을 연결하는 로컬 연구 프로젝트다.
현재 초점은 **실제 게임에서 정체를 감지하고 새로운 경로로 탈출하는 능력**이다.
원래 [연구 명세 20개 항목](docs/research-spec.md)과 장기 목표를 유지한다.

**처음 받는 사람은 [새 PC 설치·가중치 다운로드·실패 실험 실행 안내](docs/new-pc-setup.md)부터 시작한다.**
Git clone, Windows/WSL 준비, 격리 환경 설치, 공식 기본 모델 4개와 추가 학습 checkpoint,
월드/기록 복원, 해시 확인, 실행·중단 순서를 제공한다. 새 PC 전체 설치·게임 실행은 아직 미검증이며,
과거 실패 재현은 특정 Minecraft JAR 해시를 요구한다. 안내의 실행 전 확인 단계를 따른다.

## 현재 결과

최신 기록은 [진행 상황](docs/progress.md), [남은 작업 단위](docs/next-task-units.md)에서 확인한다.
2026-10-15 [미완료 목표 우선순위](docs/goal-aging.md)를 구현했다. 미완료/실행 대기/직접 서비스
시간을 분리하고 선행 작업에 긴급도를 상속한다. RM 선택·LLM 입력과 명시적 다중 수집 세션에
연결했으며, 이번 변경의 실제 게임 성능은 아직 검증하지 않았다.
최신 [보강 이동 후보의 RM 연결](docs/rm-refined-navigation-validation-2026-10-15.md)은
실제2게임2,056전이에서 이동1/4도달·도달 후 찾기 실패를 관측했다. 접근/root 재개는
미실행이며 원목0이다. 중지 후 재개하여 far/corner 전체 감사가 통과했다.
후속 [경로 방문 기반 탐험·막힘 감지](docs/exploration-coverage-validation-2026-10-15.md)는
새3게임3,230전이를 감사했다. motion corner에서는 막힌 방향 조기 종료→다른 목표 도착→
나무 찾기까지 진행했으나 접근은 실패했고 원목0이다. 일부 목표가 외벽 밖이라는 원인을
평가 전용 NBT에서 확인했다. 지도는 정책에 공급하지 않았으며 인과 성능 개선은 미입증이다.
별도 [근거리·역방향 교정](docs/near-navigation-validation-2026-10-15.md)은4게임1,176전이를 감사했다.
새 교사249정답을 추가하고 기존PPO head를BC로17갱신했지만, 개발 도착10/12→9/12·행동649→819로
악화해 후보를 채택하지 않았다. 조건부 RM 시험은 미실행이며 기본 정책을 유지한다.

앞선 [이동 교정·BC·짧은 PPO 비교](docs/navigation-refinement-validation-2026-10-14.md)는
13게임4,816전이를 감사했다. 알려진 목표의 개발 도착은 기존3/12, 선형BC9/12,
비선형BC9/12, 비선형BC+PPO10/12였다. PPO는643개의 실제 훈련 전이로84갱신했다.
이는 서버 좌표를 받는 이동 부분의 소규모 개발 결과다. 후속 RM 연결과 과제 조건이 다르며,
벽 우회·자연 탐험 일반화는 미검증이다. 기본 정책과 최종5시드는 유지한다. 새 API 호출은0회다.

앞선 [RM 계획과 학습된 이동 연결](docs/rm-navigation-integration-validation-2026-10-14.md)을 구현하고
고정 계획의 실제2게임1,936전이를 감사했다. 실패 이력→다른 목표 재계획→경유점 도달1회는
확인했지만 원목은 두 게임 모두0개다. 찾기 단계 누락 등3가지 오류를 수정했다.
수정 후 게임은 이동부터 실패해 찾기·접근은 미실행이며, 새 실제 LLM/API 호출은0회다.

2026-10-14 실패 방향 기억·LLM 후보 선택 비교는 [고정 프로토콜](docs/adaptive-recovery-protocol-2026-10-14.md)과
[실제 결과](docs/adaptive-recovery-validation-2026-10-14.md)에 정리한다.

6게임4,814전이를 감사했다. 복구 호출3게임 모두 국소 이탈, 그중2게임 벽 횡단을 확인했지만
복구 후 원목 획득은0/3이었다. 나머지3게임은 원래 정책이 먼저 완료해 복구 분모에서 제외했다.
실제 LLM2회·추정US$0.00745325를 사용했으며 두 선택이 규칙과 같아 추가 효과는 미입증이다.
수집 시험으로 중단한 이동을 실패로 잘못 전달한 오류는 수정·재생 검증했다.
후속 RM 통합 게임은 별도 경로의 검사이며 이 후보 선택 배치의 재실행 결과로 합치지 않는다.

| 영역 | 확인한 범위 |
|---|---|
| 공개 정책 실행 | GTX 1060 3GB / Windows·WSL에서 공식 MineRL 계열 VPT·STEVE-1 실제 실행. [초기 VPT](docs/local-vpt-results-2026-10-03.md)·[STEVE-1](docs/local-steve1-results-2026-10-03.md) 결과 |
| 과업과 대화 | 목표·완료 이력을 보존하는 RM, 순차 수집, 대화 요청과 중단/재개. [과업 분해 계약](docs/task-decomposition.md) |
| 학습 | STEVE-1 action head·일부 표현/기억층의 로컬 PPO·BC 실험. 실패한 후보와 기존 기본 정책을 구분해 보존 |
| 상위 계획 | 정체·반복·실패 이력을 입력으로 하위 계획 제안/검증/실행. [실제 LLM 1회 검사](docs/subgoal-live-validation-2026-10-10.md)는 연결을 확인했지만 해당 게임의 원목 획득은 실패 |
| 공간 기억 | 방문 구역/방향, 실패 이력, 명명 장소. [실제 F3 좌표 판독](docs/spatial-exploration-validation-2026-10-10.md)은 별도 관측 진단이며 live planner의 좌표는 아직 서버 출처 |
| 이동 하위 행동 | 상대 목표를 받는 별도 학습 head. [10월 14일 교정·PPO](docs/navigation-refinement-validation-2026-10-14.md): 개발 이동10/12도착, 원목0. 기존 RM capability로의 새 후보 승격·일반 탐험과 구분 |
| 종료 | STOP 뒤 새 정책 행동 금지, 중립 입력 해제, 해당 실행의 자식 프로세스 정리. 실행별 원시 로그로 감사 |

일반적인 우회·탐험 성능이나 인간다움은 입증하지 않았다. 알려진 fixture 경로, 규칙 기반
후보 선택, 실제 LLM 계획을 구분한다. 서버 좌표/RM 입력을 쓰는 조건은 영상 전용 조건이 아니다.
모델 로드·합성 테스트·교사 시연은 학습 정책 성공으로 세지 않는다. VPT 단독에 자연어 지시
수행 능력을 가정하지 않으며, MineRL 실행을 최신 Bukkit 호환성으로 해석하지 않는다.

## 저장소와 재현 범위

이 공개 저장소에는 코드·테스트·프로토콜·보고서를 보관한다. 추가 학습 가중치·시작 월드·
실패 기록은 [research-20261015 릴리스](https://github.com/csb1004/MinecraftRL/releases/tag/research-20261015)의
해시가 고정된 묶음으로 제공한다. **clone 후 자료 동기화와 별도 정책 환경 설치가 필요하다.**
비밀정보·Minecraft 실행 JAR·이를 포함한 wheel은 공개하지 않으며 기본 모델은 공식 배포처에서 받는다.
원본들은 로컬 `results/`, `weights/`, `.runtime/`, `external/`에 그대로 보존한다.

```powershell
.\scripts\Sync-Artifacts.ps1 -Tag research-20261015 -Experiment oct15-motion-corner
```

`oct12-fixture`, `oct12-visit-memory`, `all`도 선택할 수 있다. 기존 파일은 해시가 같으면
재사용하고 다르면 덮어쓰지 않는다. 다음 학습 결과는 변경된 묶음만 게시할 수 있다.
[받기·환경 준비·다음 버전 발행 방법](docs/artifact-sharing.md)을 참조한다.
현재 코드 커밋과 당시 실행의 `source-at-game`을 구분해 보존한다.

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

현재 연구 PC에는 설치가 완료돼 있다. 새 PC는 [설치 안내](docs/new-pc-setup.md)의 순서를 따른다.
설치기가 쓰는 환경 기록과 릴리스의 과거 기록에 동일 파일명이 있으므로 새 설치 기록을 별도
보존한 뒤 자료를 동기화한다. 기존 환경에 setup을 반복하지 않는다.
과거 클라우드 전달 ZIP 검사와 이후 로컬 실게임 검증은 [설치 기록](docs/local-validation-2026-10-03.md)에 구분했다.

VPT는 `-Policy vpt`로 별도 환경을 사용한다. [공식 출처와 라이선스/제약](docs/policy-sources.md),
[소스 pin](configs/sources.lock.json)을 먼저 확인한다. 외부 모델·Minecraft 자산을 이 저장소에
재배포하지 않는다. 기존 환경을 자동 삭제하거나 시스템 설정을 바꾸지 않는다.
STEVE-1의 알려진 gym 메타데이터 충돌 예외는 설치 receipt에 남기며 `pip check` 전체 통과로 표현하지 않는다.

새 실행에서는 제어 시간을 최대 180초로 제한하고 관측/행동/인벤토리/추론 지연/메모리/종료를 기록한다.
실행 절차와 판정은 [실험 프로토콜](docs/experiment-protocol.md)을 따른다.

```powershell
.\scripts\Invoke-PolicyProbe.ps1 -Policy steve1 -Operation episode -Condition wood -Seconds 180
```

위 명령은 기본 STEVE-1의 새 월드 실행이며 추가 학습 checkpoint나 벽 월드를 자동 사용하지 않는다.
보관된 실패 실험은 [새 PC 안내의 8단계](docs/new-pc-setup.md#8-기본-정책-확인과-보관된-실패-실험-실행)에 따라
`reproduction/Show-ReproductionCommand.ps1`로 당시 명령을 출력해 새 결과 경로에서 실행한다.

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

대화 API는 선택 기능이며 기본 정책과 규칙 선택 조건에는 키가 필요 없다. LLM 비교에 사용할 때는
[.env.example](.env.example)의 설정을 로컬 비밀 파일로 관리한다. 유료 API·Colab 실행은
실험별 승인 범위와 호출 예산을 따른다.
