# 새 PC 설치와 실패 실험 재현

대상: 기존 MinecraftRL 프로젝트가 없는 Windows PC. 공개 버전은
[research-20261015](https://github.com/csb1004/MinecraftRL/releases/tag/research-20261015)다.
아래 순서는 **코드 받기 → 환경 설치 → 설치 기록 보존 → 실험 자료 받기 → 기본 가중치 받기 → 점검 → 실행**이다.
명령은 별도 표시가 없으면 **PowerShell 7**에서 한 단계씩 실행한다. 오류가 나면 다음 단계로 넘어가지 않는다.

공개 자료의 다운로드·5,631개 파일 해시 검증은 완료했다. **새 PC에서 전체 환경을 새로 빌드하고
게임까지 실행하는 과정은 아직 검증하지 않았다.** 특히 과거 실패 재현 코드는 특정 Minecraft JAR
해시만 허용한다. 새 빌드의 해시가 다르면 7단계에서 중단되며, 호환성 검토가 필요하다.
이 안내에 명령이 있다는 사실을 설치·정책 성공으로 해석하지 않는다.

기록만 볼 사람은 1단계의 PowerShell·Windows Python·Git, 2단계, 5단계만 수행하면 된다.
WSL/GPU/기본 모델/API 키 없이 저장된 PNG와 JSON을 볼 수 있다. 나중에 게임도 실행하려면
아래의 **자료를 먼저 받은 경우** 항목에 따라 설치 기록 충돌을 먼저 처리한다.

## 1. 필요한 프로그램과 PC 조건

| 구분 | 준비할 내용 |
|---|---|
| Windows | x64, WSL2·WSLg를 사용할 수 있는 Windows 11 또는 Windows 10 build 19044 이상 |
| 터미널 | [PowerShell 7 설치](https://learn.microsoft.com/en-us/powershell/scripting/install/install-powershell-on-windows). Windows PowerShell 5.1과 구분한다. |
| Windows Python | [공식 설치](https://www.python.org/downloads/windows/), 3.9 이상. 자료 동기화용이며 연구 PC는 3.11.9를 사용했다. |
| Git | [Git for Windows](https://git-scm.com/install/windows). 공개 자료 받기에 GitHub 로그인이나 `gh`는 필요 없다. |
| Linux 환경 | WSL2의 `Ubuntu-24.04`. 이 안내와 보관된 실행 명령은 이 이름을 사용한다. |
| GPU | Windows/WSL 양쪽에서 NVIDIA GPU가 보이고 CUDA·OpenGL 렌더링이 동작해야 한다. |
| 저장 공간 | 릴리스 약 871MB + 기본 모델 약 1.642GB 외에 압축 해제본·환경·소스·빌드·캐시 공간이 필요하다. 수십 GB의 여유를 권장하며 측정된 최소 요구량은 아니다. |

기존 실행 PC는 GTX 1060 3GB, Windows RAM 약 23.9GiB, WSL 가시 RAM 약 11.6GiB였다.
이는 [관측한 환경](local-validation-2026-10-03.md)이며 모든 GPU의 호환성이나 최소 사양을 보장하지 않는다.
특히 설치에 고정된 PyTorch 2.0.1/CUDA 11.7과 새 GPU의 호환성은 별도 확인이 필요하다.
Windows 시스템 Python·Java를 정책 버전에 맞춰 바꿀 필요는 없다. 정책용 Python 3.10/JDK 8은 프로젝트 안에 설치한다.
Node.js·Docker·유료 API는 아래 세 실패 실험의 필수 항목이 아니다.

WSL이 없는 PC에서만 관리자 PowerShell로 다음을 실행한다. 재부팅이 요청되면 재부팅하고,
Ubuntu를 처음 열어 Linux 사용자명·암호를 만든다. 기존 작업 중인 WSL을 무조건 초기화하지 않는다.
[Microsoft WSL 설치 안내](https://learn.microsoft.com/en-us/windows/wsl/install)를 따른다.

```powershell
wsl --list --online
wsl --install -d Ubuntu-24.04
```

일반 PowerShell 7을 새로 열고 확인한다.

```powershell
$PSVersionTable.PSVersion
python --version
git --version
wsl --list --verbose
nvidia-smi
wsl -d Ubuntu-24.04 -- nvidia-smi
wsl -d Ubuntu-24.04 -- printenv DISPLAY
```

`Ubuntu-24.04`가 VERSION 2로 표시돼야 한다. WSLg/드라이버 준비는
[Microsoft GUI 안내](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps),
[NVIDIA WSL 안내](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)를 확인한다.
NVIDIA 드라이버는 Windows 쪽에서 제공하며 WSL에 Linux GPU 드라이버를 중복 설치하지 않는다.
CUDA toolkit 전체를 시스템에 설치하는 단계도 이 프로젝트 절차에는 없다.
`python`이 Microsoft Store를 열거나 명령을 찾지 못하면 Windows Python 설치/PATH를 먼저 확인한다.
이 스크립트들은 `py`가 아니라 `python` 명령을 사용한다.

Ubuntu에 `bash`, `curl`, `tar`, `bzip2`, `git`, `flock`, `realpath`, `sha256sum`이 필요하다.
누락된 경우에만 해당 Ubuntu의 기본 도구를 준비한다. 아래는 **받는 PC의 WSL 시스템 패키지를 변경하는 선택 단계**다.
전역 Python/Java 설치나 `sudo pip`는 하지 않는다. `mesa-utils`는 렌더러 확인용이다.

```powershell
wsl -d Ubuntu-24.04 -- sudo apt-get update
wsl -d Ubuntu-24.04 -- sudo apt-get install ca-certificates curl git bzip2 unzip zip util-linux build-essential mesa-utils
wsl -d Ubuntu-24.04 -- env GALLIUM_DRIVER=d3d12 glxinfo -B
```

`glxinfo`가 없다는 메시지는 진단 도구 누락이다. 반면 `DISPLAY`가 비었거나 렌더러가 `llvmpipe`라면
GPU 게임 렌더링 준비가 끝난 것으로 보지 않는다. WSLg를 사용하는 이 경로에서 먼저 해결한다.

## 2. 저장소를 처음 받기

현재 보관된 실행 명령의 기준 경로는 **`D:\DATA\Desktop\MinecraftRL`**,
WSL에서는 **`/mnt/d/DATA/Desktop/MinecraftRL`**이다. 이 안내는 같은 경로를 사용한다.
D:가 없거나 다른 경로를 써야 하면 과거 명령과 데이터 내부의 절대 경로까지 이식해야 하므로,
clone 위치만 바꾸는 것은 아직 지원된 재현 절차가 아니다.

```powershell
$ErrorActionPreference = 'Stop'
$project = 'D:\DATA\Desktop\MinecraftRL'
if (-not (Test-Path 'D:\')) { throw 'D: 드라이브가 필요합니다.' }
if (Test-Path -LiteralPath $project) { throw '기존 폴더는 보존하세요. 신규 clone 절차를 중단합니다.' }
New-Item -ItemType Directory -Path 'D:\DATA\Desktop' -Force | Out-Null
git clone --config core.autocrlf=false https://github.com/csb1004/MinecraftRL.git $project
if ($LASTEXITCODE -ne 0) { throw 'clone 실패' }
Set-Location -LiteralPath $project
git rev-parse HEAD
```

`core.autocrlf=false`는 이 clone의 설정이며 WSL용 `.sh`의 LF 줄바꿈을 유지한다.
최신 main은 설치 도구·안내를 제공한다. 릴리스의 코드 커밋은
`70d7f6202295643400c44010c13b09e86de120f7`이고, 과거 게임은 별도 `source-at-game` 사본으로 실행한다.
GitHub의 일반 **Download ZIP**이나 예전 `MinecraftRL-preflight-delivery.zip`만으로 가중치까지 설치되지는 않는다.

## 3. 프로젝트 안에 STEVE-1 환경 설치

신규 clone에서 **실험 자료를 동기화하기 전에** 설치한다. 이미 `.runtime/steve1`이 있으면
새 설치를 반복하지 말고 아래 오류 표를 확인한다. 기존 연구 PC에는 이 단계를 다시 실행하지 않는다.

```powershell
Set-Location -LiteralPath 'D:\DATA\Desktop\MinecraftRL'
if (Test-Path '.runtime/steve1') { throw '기존 환경이 있습니다. setup을 반복하지 마세요.' }
if (Test-Path 'reproduction/run-commands.json') { throw '자료를 먼저 받은 경우의 안내를 확인하세요.' }
.\scripts\Test-LocalEnvironment.ps1 -Distribution Ubuntu-24.04
.\scripts\Get-OfficialSources.ps1 -Policy steve1
.\scripts\Invoke-PolicyProbe.ps1 -Policy steve1 -Distribution Ubuntu-24.04 -Operation setup
```

설치기는 `configs/sources.lock.json`의 고정 소스를 받고, `.runtime/bin/micromamba`와
`.runtime/steve1`을 사용한다. Python 3.10, PyTorch 2.0.1, CUDA 11.7 런타임, JDK 8,
MKL 2024.0 및 MineRL·STEVE-1·MineCLIP 패키지를 구성한다.
conda 입력은 `configs/conda-steve1-linux-64.candidate.txt`, pip 제약은 `configs/pip-constraints.txt`다.
일부 pip 전이 의존성은 완전 고정되지 않아 과거 패키지 목록과 차이가 날 수 있다.

MineRL/MCP Minecraft 빌드를 포함하므로 시간이 걸릴 수 있다. 프로젝트·설치 환경은 D:에 두고
빌드 중간 파일과 Gradle 임시 캐시만 WSL `/tmp`를 사용한다. setup은 한 번에 하나만 실행한다.
로그는 `results/setup-steve1-<시각>.log`에 남는다. 오류가 났다면 로그를 보존하고 원인을 먼저 확인한다.
이 단계는 Minecraft나 학습 정책을 실행하지 않는다.

## 4. 새 PC의 설치 기록을 별도 보존

설치기가 쓰는 아래 세 파일은 릴리스에 포함된 **원래 연구 PC의 환경 기록**과 파일명이 같다.
새 기록을 별도 폴더로 옮긴 뒤 자료를 받아야 충돌·덮어쓰기를 피할 수 있다.
setup이 성공한 후 프로젝트 루트에서 이 블록을 한 번 실행한다. 파일 내용은 수정하지 않는다.

```powershell
# preserve-new-install-records
$ErrorActionPreference = 'Stop'
$root = (Get-Location).Path
if ($root -ne 'D:\DATA\Desktop\MinecraftRL') { throw '프로젝트 루트에서 실행하세요.' }
if (Test-Path 'reproduction/run-commands.json') { throw '이미 복원한 과거 기록을 새 설치 기록으로 옮기면 안 됩니다.' }
$names = @('setup-steve1.json', 'build-steve1.patch', 'pip-check-steve1.txt')
foreach ($name in $names) {
    if (-not (Test-Path -LiteralPath (Join-Path 'results' $name) -PathType Leaf)) { throw ('설치 기록 누락: ' + $name) }
}
$installRecord = Join-Path $root ('results/local-install-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $installRecord -ErrorAction Stop | Out-Null
foreach ($name in $names) {
    Move-Item -LiteralPath (Join-Path $root ('results/' + $name)) -Destination (Join-Path $installRecord $name) -ErrorAction Stop
}
foreach ($name in @('conda-steve1-explicit.txt', 'pip-steve1-freeze.txt', 'build-location-steve1.json')) {
    $source = Join-Path $root ('results/' + $name)
    if (Test-Path -LiteralPath $source) { Copy-Item -LiteralPath $source -Destination (Join-Path $installRecord $name) -ErrorAction Stop }
}
git rev-parse HEAD | Set-Content -LiteralPath (Join-Path $installRecord 'code-commit.txt') -Encoding utf8
Write-Host ('새 PC 설치 기록: ' + $installRecord)
```

이후 릴리스에서 받는 `results/setup-steve1.json` 등의 작성 머신은 원래 연구 PC다.
그 파일을 새 PC 설치 성공의 증거로 쓰지 않는다. 새 PC의 근거는 위 `local-install-*` 폴더와 새 로그다.

## 5. 추가 학습 가중치·시작 월드·실행 기록 받기

세 실패 실험을 모두 받는 예시다. GitHub 로그인·API 키 없이 작동하며, 이미 같은 파일이 있으면 재사용한다.
내용이 다른 기존 파일은 전체 복원 전에 충돌로 중단한다.

```powershell
.\scripts\Sync-Artifacts.ps1 -Tag research-20261015 -Experiment all
```

| 받는 파일/폴더 | 용도 |
|---|---|
| `results/onset-coverage-training-20261009/checkpoint.pt` | 추가 학습한 원목 수집 정책 |
| `results/waypoint-training-goal-focused-20261010/checkpoint.pt` | 10/12의 복구 이동 정책 |
| `results/navigation-refinement-20261014/ppo/mlp/checkpoint.pt` | 10/15 통합 실험의 보강 이동 정책 |
| `results/blind-recovery-20261012/worlds/far.zip`, `corner.zip` 및 JSON | 실험 시작에 불러온 월드 |
| `results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy` | 원래 목표 임베딩 |
| `results/goal-conditioning-diagnosis-20261009/explore_9711.npy`, `results/subgoal-skills-20261010/embeddings` | 탐험/하위 skill 임베딩 |
| `results/blind-recovery-20261012/far/fixture`, `far/visit_memory` | 10/12의 두 실행 기록 |
| `results/exploration-motion-20261015/game-corner` | 10/15 통합 실패 기록 |
| `reproduction/` | 당시 Python·보충 Java 소스, 실행 인수, 모델 해시·환경 안내 |

하나만 필요하면 `all` 대신 `oct12-fixture`, `oct12-visit-memory`, `oct15-motion-corner`를 고른다.
해당 실험에 필요한 정책·월드·기록 묶음이 함께 선택된다. 체크포인트 파일을 `weights/steve1`에
임의로 옮기거나 이름을 바꾸지 않는다. 실행 명령은 위 경로를 직접 참조한다.
전체 파일/크기/SHA256 목록은 릴리스의 `artifact-manifest.json`과 `.cache/release-artifacts`의 로컬 사본에 있다.

**기록만 보기:** 각 실험의 `run/frames/*.png`, `run/events.jsonl`, `run/summary.json`,
`work.json`, `source-at-game`을 확인한다. MP4가 없어도 PNG와 JSON으로 관찰할 수 있다.
이 파일들을 열거나 아래 명령을 출력하는 것만으로 게임이 실행되지는 않는다.

## 6. 공식 기본 모델 4개 받기 및 동일 파일 확인

추가 학습 checkpoint만으로는 실행되지 않는다. 기본 모델 4개도 필요하다.

```powershell
.\scripts\Invoke-PolicyProbe.ps1 -Policy steve1 -Distribution Ubuntu-24.04 -Operation download
```

| 최종 경로 | 역할 | 파일 크기 |
|---|---|---:|
| `weights/steve1/2x.model` | VPT 2x 구조 설정 | 4,369 bytes |
| `weights/steve1/steve1.weights` | STEVE-1 기본 정책 | 998,232,471 bytes |
| `weights/steve1/attn.pth` | MineCLIP | 634,321,445 bytes |
| `weights/steve1/steve1_prior.pt` | 텍스트 목표 prior | 9,474,805 bytes |

출처 URL/Google Drive ID는 `configs/models.json`에 있다. 설치 도구가 공식 배포처에서 내려받고
각 파일 옆에 `.receipt.json`을 만든다. **다운로더가 만든 새 해시 기록만으로 과거 모델과 동일하다고
판정하지 않는다.** 아래에서 릴리스에 고정된 SHA256·크기와 추가 비교한다. 모델을 역직렬화하지 않는다.

```powershell
# verify-base-models
$ErrorActionPreference = 'Stop'
$models = Get-Content -LiteralPath 'reproduction/base-models.json' -Raw | ConvertFrom-Json
foreach ($model in $models) {
    $file = Get-Item -LiteralPath $model.file -ErrorAction Stop
    $actual = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($file.Length -ne $model.bytes -or $actual -cne $model.sha256) { throw ('기본 모델 불일치: ' + $model.file) }
    Write-Host ('동일 파일 확인: ' + $model.file)
}
```

Google Drive 제한/네트워크 오류 또는 `.partial` 파일이 남으면 아래 오류 표를 따른다.
API 키나 로그인 쿠키를 다운로드 스크립트에 넣지 않는다.

## 7. 실행 전 환경과 Minecraft JAR 확인

새 clone에는 보통 STOP이 없다. 이미 `results/STOP`이 있으면 먼저 중단 이유와 직전 종료 기록을
확인한다. 설치·자료 받기를 이유로 자동 해제하지 않는다. 재실행을 선택한 뒤의 보존 방법은 9단계에 있다.

```powershell
.\scripts\Invoke-PolicyProbe.ps1 -Policy steve1 -Distribution Ubuntu-24.04 -Operation preflight
```

새 `results/preflight-steve1-<시각>/preflight.json`에서 `problems`가 비어 있고
`status`가 `ready_for_execution_attempt`인지 확인한다. 이는 실행 시도 준비 상태이며
실제 Minecraft 렌더링/정책 성공이나 아래 JAR 호환성 검사를 대신하지 않는다.

**과거 실패 실험을 실행하기 전에 반드시** 기본 JAR를 확인한다.

```powershell
# verify-historical-runtime
$ErrorActionPreference = 'Stop'
$jar = '.runtime/steve1/lib/python3.10/site-packages/minerl/MCP-Reborn/build/libs/mcprec-6.13.jar'
$expected = 'f7a1e0aa8bee8a0ff9479856e807be6d8aa30373db8934d5b093342690ae9a48'
$actual = (Get-FileHash -LiteralPath $jar -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host ('빌드한 기본 JAR SHA256: ' + $actual)
if ($actual -cne $expected) { throw '과거 실행 코드의 JAR 허용 해시와 다릅니다. 재현 실행을 중단하고 빌드 호환성 검토가 필요합니다.' }
```

`world_save_patch.py`는 이 해시를 검사한 뒤 저장/서버 관측용 Java 패치를 별도
`.runtime/world-save/` JAR로 만든다. 실행 중 자동 빌드되며 별도 배포된 게임 JAR를 받을 필요는 없다.
보충 Java 소스도 `reproduction/execution-source`에 포함했다.
과거 **패치 후** JAR SHA256은 `afeb88a9dd572bad19f25cf8d0a7346c08ef17ab5993e9de5b9da1ebee40688c`다.
기본 JAR와 패치 후 JAR를 혼동하지 않는다.

같은 소스라도 다른 머신의 빌드가 byte 단위로 같다는 보장은 없다. 위 검사 실패 시 원인은
아직 판정되지 않은 것이며, 해시 상수를 새 값으로 바꾸거나 검사를 제거해 통과시키지 않는다.
이 엄격한 허용 조건은 새 PC에서의 과거 실패 재현에 남아 있는 제한이다.

## 8. 기본 정책 확인과 보관된 실패 실험 실행

### 기본 STEVE-1만 간단히 확인할 때

```powershell
.\scripts\Invoke-PolicyProbe.ps1 -Policy steve1 -Operation episode -Condition wood -Seconds 30
```

이 명령은 **공식 기본 정책의 새 월드 실행**이다. 추가 학습 checkpoint·벽 월드를 자동으로
적용하지 않으므로 보관된 실패 재현과 구분한다. 다음 명령을 대신 실행한 것으로 기록하지 않는다.

### 추가 학습 정책과 벽 월드에서 실패를 재현할 때

| 실행 이름 | 시작 월드 | 이동 정책/계획 |
|---|---|---|
| `oct12-fixture` | `far.zip` | 기존 이동 checkpoint + 고정 경로 |
| `oct12-visit-memory` | `far.zip` | 기존 이동 checkpoint + 방문 기억 |
| `oct15-motion-corner` | `corner.zip` | 보강 이동 checkpoint + 하위 계획/coverage_motion |

공통 원목 checkpoint와 목표 임베딩은 5단계의 파일을 사용한다.
세 실행은 API 호출·학습 갱신을 하지 않는 평가 실행이다. LLM 키를 넣을 필요가 없다.
실시간 LLM 복구가 실행됐다고 해석하지 않는다.

```powershell
.\reproduction\Show-ReproductionCommand.ps1 -Run oct15-motion-corner
```

위 명령은 **실행할 PowerShell 한 줄을 출력만 한다.** 출력된 `& 'wsl.exe' ...` 한 줄을
검토한 뒤 직접 붙여 넣으면 게임이 시작된다. `Invoke-Expression`으로 출력 전체를 자동 실행하지 않는다.
`-Run`을 표의 다른 이름으로 바꾸면 다른 실패 실험의 명령을 출력한다.

출력 명령은 당시 Python 사본을 `PYTHONPATH`로 지정하고, 원래 월드·가중치·시드를 사용하며,
`results/reproduction-<실행이름>-<UTC시각>/run`이라는 새 결과 경로를 만든다.
제어 시간은 최대 180초이며 모델 로드·게임 시작·정리 시간은 별도다. 기존 실패 폴더를 출력 경로로 쓰지 않는다.
동일 조건이라도 GPU/빌드/의존성 차이 때문에 같은 프레임·행동 순서까지 보장하지 않는다.

## 9. 중단·결과 확인·다시 실행

별도 PowerShell에서 다음 명령으로 중단한다.

```powershell
Set-Location -LiteralPath 'D:\DATA\Desktop\MinecraftRL'
.\scripts\Stop-PolicyProbe.ps1
```

진행 중이던 한 step은 마무리될 수 있다. 새 결과의 `summary.json`, `events.jsonl`과
입력 해제·소유 자식 프로세스 정리 기록을 확인한다. 정리되기 전에 두 번째 게임을 시작하지 않는다.
`execution_valid=false`/`planning_paused`는 완료 성공이 아니며, 원목 획득은 실제 인벤토리로 판정한다.
과거 공개 기록 3개는 원목 0, `planning_paused`, `execution_valid=false`였다. 같은 결과를 보더라도 새 실행 기록은 별도로 남긴다.

중단이 정상 처리됐고 사용자가 새 실행을 결정한 경우에만 STOP을 보존 이동한다.
setup/download/preflight 명령을 실행하려는 이유만으로 아래 블록을 무조건 실행하지 않는다.

```powershell
# explicit-rearm-after-reviewed-shutdown
if (Test-Path -LiteralPath 'results/STOP') {
    $savedStop = Join-Path 'results' ('STOP-reviewed-' + [Guid]::NewGuid().ToString('N'))
    Move-Item -LiteralPath 'results/STOP' -Destination $savedStop -ErrorAction Stop
}
```

새 명령을 출력하여 새 결과 폴더로 실행한다. 종료 코드가 0이 아닌 경우 새 `summary.json`과
`worker.stderr.log` 등 실제 생성된 로그를 먼저 확인한다. 실패를 성공으로 바꾸거나 기존 결과를 지우지 않는다.

## 오류가 났을 때

| 증상 | 확인/조치 |
|---|---|
| `.ps1` 실행 정책 오류 | 코드 내용을 확인하고 조직 정책을 따른다. 필요한 개인 PC에서만 현재 창에 `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`를 적용할 수 있다. 시스템 전체 Bypass는 하지 않는다. |
| `bash`의 `\r`/`bad interpreter` 오류 | Git의 줄바꿈을 확인한다. 새 clone은 2단계의 `core.autocrlf=false`를 사용한다. 기존 수정 파일을 일괄 초기화하지 않는다. |
| WSL 배포판 없음/경로 해석 실패 | `wsl --list --verbose`, D: 경로와 `Ubuntu-24.04` 이름을 확인한다. Docker Desktop 배포판을 대신 사용하지 않는다. |
| `Environment exists` | 기존 환경을 삭제하지 않는다. 설치 실패 로그를 확인한 뒤 기존 Python이 정상일 때만 `-Operation resume-setup`으로 재시도한다. 새 빌드/설치가 발생하므로 기록 보존 절차도 다시 적용한다. |
| 설치 도중 빌드/다운로드 실패 | `results/setup-steve1-*.log`, `build-location-steve1.json`과 새 설치 기록을 보존한다. 인터넷·디스크·오류 난 단계부터 확인하고 후보 버전을 임의 최신화하지 않는다. |
| `iJIT_NotifyEvent` 오류 | MKL/PyTorch 호환성을 확인한다. 고정 후보는 MKL 2024.0이다. [기존 원인 분석](local-validation-2026-10-03.md)을 참고한다. |
| `pip check`의 gym 충돌 | 설치기는 MineCLIP/MineDojo의 `gym==0.21.0` 메타데이터 충돌만 기록된 예외로 처리한다. 다른 충돌은 정상 설치로 넘기지 않는다. |
| `torch CUDA unavailable` / CUDA OOM | Windows/WSL `nvidia-smi`, 해당 환경의 preflight, 다른 GPU 사용 프로그램과 여유 VRAM을 확인한다. 드라이버/torch를 무조건 교체하지 않는다. |
| 모델 다운로드에 Google Drive 제한/HTML/`.partial` | 다운로드를 성공으로 세지 않는다. 진행 중인 다운로더가 없는지 확인하고 실패한 `.partial`을 별도 이름으로 보존한 뒤 나중에 `download`를 재시도한다. 최종 파일과 `.receipt.json`을 임의 작성하지 않는다. |
| `artifact conflict` | 표시된 로컬 파일과 manifest의 해시를 비교한다. 다른 파일을 먼저 별도 보존하고 경로를 비운 뒤 동기화한다. 다운로드 옵션으로 덮어쓰지 않는다. |
| `unsupported_upstream_world_jar` | 7단계의 기본 JAR 해시 제한이다. setup 성공만으로 과거 실패 재현이 가능하다고 보지 않는다. 빌드 로그·소스 pin·새 JAR 해시를 전달해 호환성을 검토한다. |
| `results/STOP is latched` | 직전 종료/사용자 중단 상태다. 9단계에 따라 검토·명시적 재시작 후 보존 이동한다. |
| 기록상 나무 0개/`planning_paused` | 정책/계획 실패일 수 있다. 환경 초기화 실패와 구분하고 새 원시 기록을 보존한다. |

### 자료를 먼저 받은 경우

`Sync-Artifacts.ps1`로 기록을 먼저 받은 뒤 환경을 설치하면 setup이 아래 과거 환경 기록을 덮어쓸 수 있다.
`results/setup-steve1.json`, `results/build-steve1.patch`, `results/pip-check-steve1.txt`를
프로젝트 내부의 별도 **과거 기록 보관 폴더**로 먼저 보존 이동한다. SHA256도 기록한다.
그 뒤 3단계의 신규 clone 검사 블록 전체를 다시 붙이지 말고, `Get-OfficialSources.ps1 -Policy steve1`과
`Invoke-PolicyProbe.ps1 -Policy steve1 -Operation setup` 두 설치 명령을 실행한다.
새로 생성된 같은 이름의 세 파일을 별도의 **새 PC 설치 기록 폴더**에 보존한다.
마지막으로 과거 사본을 원래 경로에 되돌리고 SHA256이 동일한지 확인한다.
`resume-setup`도 같은 주의가 필요하다. 과거 파일과 새 설치 파일을 섞거나 원본 없이 덮어쓰지 않는다.
설치에 실패했으면 부분 생성된 새 기록도 보존하고 과거 사본을 복구한 뒤 원인을 조사한다.

## 검증 범위와 문의 시 보낼 자료

이 안내의 PowerShell 구문·저장소 경로·명령 인수, 기존 로컬 기본 모델/JAR의 해시를 점검했다.
세 실행 명령은 출력/인수 검사만 했다. [안내 검증 기록](reports/2026-10-15/new-pc-guide-verification.json)에 범위를 남겼다.
기존 PC에서의 점검은 **다른 PC의 새 설치 결과를 대신하지 않는다**.
별도 API 사용, Colab 연결, 새 학습, 과거 일회용 배치 재실행은 이 안내의 필수 단계가 아니다.
실패 문의에는 Git 커밋, 시도한 단계/명령, `windows-preflight-*`, 새 `local-install-*`,
`setup-steve1-*.log`, 새 `preflight.json` 또는 실행 `summary.json`·오류 로그를 전달한다.
`.env`, 토큰, 개인 로그인 정보는 보내지 않는다.

공유 방식은 [자료 동기화·다음 버전 발행](artifact-sharing.md), 모델 출처는
[정책 출처](policy-sources.md), 현재 결과와 한계는 [진행 상황](progress.md)을 참조한다.
