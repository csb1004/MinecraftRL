# 우선순위 목표 대기열과 독립 S1 학습 실행

최신 실행 결과는 [진행 기록](progress.md)을 따른다. 아래는 새 실행 경로의 사용법이며,
이미 끝난 날짜별 실험 드라이버를 다시 실행하는 안내가 아니다.

## 목표를 등록하는 방법

`configs/goals-wood-then-stone.json`의 `goal_queue_v1` 형식을 사용한다.
`priority`는 0~100이며 큰 값이 먼저다. 기존 스케줄러의 대기 시간 보정은 유지하지만,
진행 중인 목표는 완료·명시적 차단·취소 전까지 자동 선점하지 않는다.
선택 점수까지 같으면 등록 순서가 앞선 목표를 선택한다.
반드시 순서를 지켜야 하는 목표에는 `depends_on`으로 앞 목표 ID를 넣는다.
의존 목표는 먼저 등록해야 하며, 선행 목표의 실패·취소는 의존성을 해제하지 않는다.

`items`와 `count`는 목표 활성화 시점의 실제 서버 인벤토리 대비 새 획득량이다.
예를 들어 원목 2개를 가진 채 다음 원목 1개 목표를 시작했다면 총 3개가 되어야 완료한다.
단순히 아이템을 보유하는 조건과 새로 획득하는 과업은 다르다.

현재 executor는 `wood_memory`(여섯 종류 원목 전체)와 `dirt_fixed`(`dirt`)만 등록한다.
선언한 아이템 집합과 executor의 집합은 정확히 같아야 한다.
`requires`에는 시작 전에 보유해야 하는 도구 등 실제 아이템 조건을 넣는다.
알려지지 않은 skill 이름을 JSON에 추가해도 정책이 그 일을 할 수 있게 되지는 않는다.
돌 예시는 `unsupported_skill`로 보존하며, 목재→도구 제작→돌 수집의 자율 수행은 아직 미검증이다.
현재 지원되는 서로 다른 목표의 연속 실행 예시는 `configs/goals-wood-then-dirt.json`이다.

`goal-queue.json`은 전체 목표·우선순위·활성화 당시 수량·완료/차단 이력을 기록한다.
`program`은 최초 선언을 보존하며, 수정된 현재 우선순위는 `priorities`와 변경 이벤트에 기록한다.
`skill-progress.json`은 현재 하위 단계와 확인된 상호작용/회수 대기 상태를 기록한다.
차단 시 `learning-request.json`을 남기지만 LLM 호출이나 학습을 자동 실행하지 않는다.
이 파일들은 감사용이다. 새 월드에서 과거 완료 상태를 자동 복원하는 기능은 제공하지 않는다.

## 조건 임베딩 준비

설치와 공식 가중치는 [설치 안내](new-pc-setup.md)를 따른다. PowerShell에서 WSL 실행을 사용한다.
아래 예제의 `results/my-queue`는 **새로 정한** 실험 폴더다. 기존 결과 경로를 재사용하지 않는다.

```powershell
$projectLinux = '/mnt/d/DATA/Desktop/MinecraftRL'
$pythonArgs = @('-d','Ubuntu-24.04','--','env',
  'LD_LIBRARY_PATH=/usr/lib/wsl/lib','GALLIUM_DRIVER=d3d12',
  "PYTHONPATH=$projectLinux/src","XDG_CACHE_HOME=$projectLinux/.cache",
  "$projectLinux/.runtime/bin/micromamba",'run','-p',
  "$projectLinux/.runtime/steve1",'python')
& wsl.exe @pythonArgs "$projectLinux/scripts/prepare_logo_rm.py" `
  --root $projectLinux --proposal "$projectLinux/configs/logo-wood-validators-v1.json" `
  --output "$projectLinux/results/my-queue/goal-bank/wood" --encode
if ($LASTEXITCODE -ne 0) { throw 'Goal preparation failed' }
Copy-Item -LiteralPath 'D:\DATA\Desktop\MinecraftRL\weights\steve1\visual-prompts\dirt.npy' `
  -Destination 'D:\DATA\Desktop\MinecraftRL\results\my-queue\goal-bank\dirt.npy'
```

wood는 검증된 LOGO 조건 문구를 인코딩하고 dirt는 공식 영상 임베딩을 사용한다.
wood의 search와 root 조건은 다르지만 approach/mine/collect는 같은 root 임베딩을 공유한다.
현재 dirt 조건에는 검증된 세부 단계 관측기를 추가하지 않았으며 collect라는 큰 단계만 기록한다.

## 실제 실행 및 학습 자료 수집

공유 `results/STOP`이 있으면 실행하지 않는다. 중단 이유를 먼저 읽고, 명시적으로 재개하기로
결정한 작업에서만 원문·해시·재개 근거를 보존해 한 번 이동한다. 아래 명령은 STOP을 지우지 않는다.
실행 중 다른 학습/게임을 동시에 띄우지 않는다.

```powershell
& wsl.exe @pythonArgs -m minecraftrl.cli episode --root $projectLinux `
  --policy steve1 --condition wood --seed 123456 --limit-s 180 `
  --rm-conditioning goal-queue `
  --rm-program "$projectLinux/configs/goals-wood-then-dirt.json" `
  --rm-goal-bank "$projectLinux/results/my-queue/goal-bank" `
  --output "$projectLinux/results/my-queue/run-123456" `
  --stop-file "$projectLinux/results/STOP"
```

프로그램 경로를 stone 예시로 바꾸면 미지원 큐 차단을 진단할 수 있다.
학습 자료 수집에는 지원되는 과업으로 별도 manifest와
학습/개발/평가 시드를 사전에 고정한다. 수집 명령에 `--s1-record --s1-split train`을 추가하면
공식 image/goal/recurrent backbone 뒤 actor/value 양쪽에 6개 단계/진전 입력을 연결한다.
게임 중 optimizer update는 하지 않고, 실제 전달한 행동의 특징·확률·보상·종료 정보를
`s1-trajectory.pt`에 보존한다. actor의 추가 입력 열은 0으로 초기화한다.

## 게임과 분리된 학습 및 후보 평가

서로 다른 학습 seed의 정상 종료 자료를 둘 이상 지정한다. 단순 모델 로드·취소·강제 종료·
정체 차단·개발/평가 자료는 학습에 넣지 않는다. 아래 run 경로는 실제로 수집한 자료로 바꾼다.

```powershell
& wsl.exe @pythonArgs "$projectLinux/scripts/train_s1_heads.py" --root $projectLinux `
  --runs "$projectLinux/results/my-queue/train-a" "$projectLinux/results/my-queue/train-b" `
  --output "$projectLinux/results/my-queue/training" `
  --stop-file "$projectLinux/results/STOP"
```

이 trainer는 frozen 특징 위의 actor/value head만 PPO로 갱신하며 backbone은 GPU에 로드하지 않는다.
수집 당시 행동 확률 재현·유한한 비상수 advantage를 확인하지 못하면 후보를 만들지 않는다.
구체적인 학습 설정은 `training.json`에 보존한다. 이 작은 갱신으로 새로운 운동 스킬이
습득되거나 일반화된다고 보장하지 않는다. 상위 목표 완료 보상은 1, 중간 행동 보상은 0이다.

후보 평가는 동일 수집 경로에 `--s1-record --s1-split development`와
`--s1-checkpoint <새 training/checkpoint.pt>`를 지정한다. 같은 별도 개발 시드의 초기 head 조건과
비교하고, 조건에 맞춘 성능·기존 능력·STOP를 확인한 뒤 채택 여부를 결정한다.
후보는 자동 승격하지 않으며 공식 가중치를 덮어쓰지 않는다.

모든 `results`/새 가중치는 Git 제외 대상이다. 실행 당시 소스·명령·가중치 해시와 원본 로그를
보존하고, 공개가 필요하면 별도 릴리스 절차를 따른다.
