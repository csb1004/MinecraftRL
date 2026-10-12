# Windows/WSL 실제 실행 기록 — 2026-10-03

이 기록은 이전 Linux 클라우드 전달 검증과 구분한다. 실제 연결 머신은
`DESKTOP-V0HJPC2`, Windows 10.0.26200, PowerShell 7.6.5이며 최종 프로젝트는
`D:\DATA\Desktop\MinecraftRL`이다. 연구 명세 20개 항목과 장기 목표는 변경하지 않았다.

## 설치 및 기본 검사

- Windows 다운로드 폴더 `D:\DATA\Downloads`에는 후보가 없었다. 프로젝트 폴더의
  `MinecraftRL-preflight-delivery.zip` 한 개를 사용했다.
- ZIP SHA256: `A2CD84590425FDC992BD5E38C32DD246CD9180DDEB3AEC8412A3B3C0103235E4`.
- 모든 ZIP entry에 절대 경로/상위 탈출/ADS/중복/심볼릭 링크 검사를 먼저 실시하고
  별도 임시 폴더에 풀었다. START-HERE, 설치기, manifest를 검토했다.
- 57개 원본 해시와 설치 후 해시가 모두 일치했다. 기존 내용 충돌은 없었다.
  `results/windows-install-receipt.json`, `results/delivery-manifest.json`에 기록했다.
- Windows Python 3.11.9에서 테스트 13개 통과, Linux 전용 2개 skip.
  WSL 시스템 Python에서는 15개 모두 통과했다. 이는 하네스 검사이며 실게임 검증이 아니다.
- 원본 외부 소스는 고정 커밋으로 checkout했다. GitHub push/공개는 하지 않았다.

## 관측한 환경

- NVIDIA GeForce GTX 1060 3GB, 드라이버 560.94, Windows 점검 당시 여유 2194 MiB.
- Windows RAM 약 23.9 GiB. WSL 가시 RAM 약 11.6 GiB, swap 3 GiB.
- WSL2 Ubuntu 24.04.4, kernel 6.18.33.2-microsoft-standard-WSL2.
- WSL `nvidia-smi`에서 같은 GPU 확인. 이후 CUDA driver API로 context 생성,
  4096바이트 GPU 할당/memset/host readback을 실제 검증했다 (`results/wsl-cuda-driver.json`).
  이것만으로 PyTorch CUDA 연산을 입증하지 않는다.
- WSLg DISPLAY `:0`. 기본 GLX는 llvmpipe/Accelerated=no.
  프로젝트 안에 Ubuntu mesa-utils-bin을 다운로드/압축 해제해 진단했다. apt install이나
  시스템 패키지 변경은 하지 않았다.
- 프로세스 범위 `GALLIUM_DRIVER=d3d12`로 GLX가 D3D12 GTX 1060/Accelerated=yes,
  OpenGL 4.6을 보고했고 glxgears가 실제 프레임을 그렸다. 8초 timeout으로 진단을 종료했다.
  이 FPS는 Minecraft 성능 측정이 아니다. `.runtime`의 진단 도구와 `results/wsl-gl*`에 근거가 있다.

## 로컬에서 발견한 문제와 수정

1. `Invoke-PolicyProbe.ps1`에서 Windows 역슬래시 경로가 WSL 전달 중 제거됐다.
   원래 호출은 `wslpath: D:DATADesktopMinecraftRL`, exit 1을 재현했다.
   `/`로 정규화한 호출은 `/mnt/d/DATA/Desktop/MinecraftRL`, exit 0이었다.
   변환 후 빈 결과 검사보다 먼저 `.Trim()`을 호출하던 부분도 수정했다.
   수정 후 실제 setup이 프로젝트 내부 prefix로 진입했다.
2. WSLg의 기본 소프트웨어 렌더링을 관측했다. WSLg/D3D12 라이브러리가 있을 때
   실행 프로세스 트리에만 GALLIUM_DRIVER 기본값을 설정한다. 호출자가 지정한 값은 유지한다.
   시스템 환경 변수/드라이버/전역 설정은 수정하지 않았다.
3. VPT conda 후보 생성 뒤 첫 torch import에서 `iJIT_NotifyEvent` undefined symbol이
   발생했다 (`results/torch-cuda-vpt.json`). 후보의 MKL 2026.1과 구형 PyTorch의
   조합 문제였다. [PyTorch 공식 이슈](https://github.com/pytorch/pytorch/issues/123097)와
   로컬 ELF 심볼 검사를 확인했고 MKL 2024.0 파일에 필요한 심볼이 존재함을 확인했다.
   원래 torch/Python/NumPy를 유지하는 conda 변경 계획은
   `results/vpt-mkl-repair-plan.json`에 보존했다. MKL 2024.0 적용 뒤 torch 1.9.0,
   CUDA 11.1 import와 GPU 256×256 행렬 곱/CPU 결과 비교가 통과했다
   (`results/torch-cuda-vpt-repaired.json`, 최대 절대 오차 0.0001220703125).
   첫 연산 시간에는 CUDA 라이브러리 초기화가 포함돼 정책 지연으로 해석하지 않는다.
   검증한 conda explicit 목록과 YAML MKL pin을 갱신했다. 원본 후보는 results와 ZIP에 보존했다.
   STEVE-1에서도 MKL 2024.2.2와 torch 2.0.1의 동일 오류를 실제 재현한 뒤 MKL 2024.0으로
   조정했다. torch 2.0.1/CUDA 11.7 import 및 GPU 행렬 연산도 통과했다
   (`results/torch-cuda-steve1-repaired.json`). 두 정책의 torch/CUDA 기준은 유지했다.
4. micromamba가 기본 보조 캐시용으로 `/home/firefly/.mamba/pkgs` 빈 디렉터리도
   만들었다. 실제 패키지는 프로젝트 내부에 저장됐으며, 후속 호출에서는
   `CONDA_PKGS_DIRS`를 프로젝트 내부로 명시하도록 수정했다. 시스템 패키지/프로필은
   변경하지 않았다.
5. 전달 스크립트의 `micromamba list --explicit`은 실제 2.0.5에서 지원되지 않았다.
   `env export --explicit`으로 교체하고 실제 export를 확인했다.
6. gdown의 기본 설정은 홈의 쿠키 파일을 사용한다. 공개 모델 다운로드에는 필요 없어
   후속 실행부터 `use_cookies=False`로 제한했다. 최초 다운로드 프로세스는 수정 전
   설정으로 실행됐으며 이는 정책 환경/시스템 패키지 변경과는 별도인 도구 캐시 동작이다.
7. 두 환경 설치를 병행하면서 Forge의 공유 생성물 캐시에 겹치는 접근이 발생했다.
   VPT의 joined merge가 `ZipException` 및 누락된 output.jar로 실패했고, 당시 STEVE-1
   daemon은 동일 joined/stripServer/output.jar를 쓰고 있었다. 동시 캐시 접근이 원인으로
   판단되며 `results/mcp-first-build-error.txt`, `/proc` 파일 확인과 daemon 로그를 근거로 남겼다.
   task 소유 setup/Java 프로세스를 PID·생성시각과 경로로 확인한 뒤 종료했고 남은 프로세스는
   0개였다 (`results/setup-recovery-processes.json`). 기존 빌드와 joined 캐시는
   `.runtime/recovery/first-build-attempt`에 보존했다. 이후 setup에 `flock`을 추가했으며
   두 번째 setup이 exit 2로 거부되는 실제 검사를 통과했다 (`results/setup-lock-regression.log`).
   conda 환경을 보존하는 명시적 `resume-setup` 경로로 VPT만 재시도했다.
8. upstream setup_mcp.sh는 Gradle 실패 뒤에도 다음 명령으로 넘어갔다. 빌드 복사본에
   `set -euo pipefail`을 추가해 실패 즉시 중단하도록 했고 이 변경도 build patch에 기록한다.
9. 순차 재시도에서도 JAR의 작은 write와 많은 파일 metadata 접근이 WSL drvfs에서 매우
   느렸다. jstack과 `/proc`에서 해당 I/O를 확인했다. 프로젝트/설치 prefix는 D:에 유지하고
   Gradle 임시 캐시만 `/tmp/minecraftrl-gradle-h6m8074u`로 지정했다. 위치와 범위는
   `results/gradle-scratch.json`에 기록했다. 둘째 빌드 복사본과 종료 기록도 보존했다.
   실행 프로세스의 GRADLE_USER_HOME override를 허용하며 시스템 설정은 변경하지 않았다.
   Gradle 캐시 외에 MCP project task도 D:에 단일-byte JAR write를 수행했다.
   완료된 작업을 보존한 복사본을 `/tmp/minecraftrl-build-ntbokv19/minerl`에서 이어갔다.
   `results/mcp-build-scratch.json`, `results/resume-mcp-scratch.py`와 빌드 로그를 남겼다.
   이는 임시 컴파일 작업물이며 최종 프로젝트 및 패키지 설치 prefix는 D:에 유지된다.
   소유 빌드 프로세스 종료 뒤 WSL이 재시작되면서 이전 /tmp 캐시는 사라졌고,
   새 빌드에서 필요한 파일을 다시 받았다. 시스템 WSL 설정은 수정하지 않았다.
10. STEVE-1의 첫 모델 진단은 cuDNN이 `libcuda.so`를 찾지 못해 native 종료했다.
   WSL ldconfig는 libcuda.so.1만 등록하고 실제 libcuda.so는 /usr/lib/wsl/lib에 있었다.
   해당 경로를 실행 프로세스 LD_LIBRARY_PATH에 추가한 뒤 같은 진단이 통과했다.
11. `elapsed_control_s`가 환경 종료 시간을 포함했다. 제어 0.2초와 종료 12초를
    분리한 회귀 검사에서 기존 결과 12.2초를 재현했다. 제어 종료 시각을 정리 전에
    저장하고 `elapsed_cleanup_s`를 별도로 기록하도록 수정했으며 검사가 통과했다.
    실제 STOP 시점 대조를 위한 inference/dispatch 시작·종료 이벤트를 추가했다.
    변경 후 WSL 하네스 16개 테스트가 통과했다. 테스트의 합성 집계 출력은 실제
    정책 실험 표본이 아니다 (`results/wsl-unit-tests-after-timing-fix.log`).

## MineRL/MCP 실제 빌드 및 설치 완료

전체 MCP Gradle 빌드가 완료됐고, 생성 JAR는 465329753바이트,
SHA256 `f7a1e0aa8bee8a0ff9479856e807be6d8aa30373db8934d5b093342690ae9a48`이다.
JAR의 CRC와 Main.class, 성공한 빌드 receipt의 해시가 일치할 때만 pip metadata/wheel
단계에서 재사용하도록 했다. 실패한 중간 산출물은 성공 캐시로 취급하지 않는다.

cp39/cp310 wheel을 `.runtime/artifacts`에 보존하고 각각 `.runtime/vpt`,
`.runtime/steve1`에 설치했다. drvfs의 다수 작은 파일 설치 병목 때문에 Windows pip의
명시적 Linux target 설치를 사용했으며, 시스템 Python의 site-packages는 수정하지 않았다.
wheel에는 Python/Java 파일만 있고 native extension은 없다. 재현 명령은
`scripts/Install-MineRLWheel.ps1`, 검증 receipt는 `results/setup-*.json`과
`results/installed-packages-*.json`이다. 게임의 작업 폴더와 설치 환경은 D:에 유지한다.

VPT의 pip check는 통과했다. STEVE-1에는 공식 MineCLIP/MineDojo가 선언한 gym==0.21.0과
공식 VPT 계열 fork의 0.19.0 메타데이터 충돌이 남아 있다. 이 충돌을 숨기지 않으며,
이번 실행 범위가 동작하더라도 다른 MineDojo 과업 전체의 호환성을 의미하지 않는다.
cloudpickle 1.6.0 및 triton의 cmake/lit 의존성을 프로젝트 환경에 보완했다.

Java 시작 시에도 drvfs JAR 읽기 병목이 관측돼 동일 SHA의 불변 JAR만 WSL 임시 캐시에
복사한다. 각 실행의 `runtime-jar-cache.json`에 원본/캐시 위치와 해시를 남긴다.
공식 external checkout은 수정하지 않았으며 실행용 설치 복사본의 launcher만 패치한다.

실제 오류에서 추가한 수정은 초기화 전 neutral step 금지, 정리 시간과 제어 시간 분리,
기술 오류 발생 시 suite의 남은 시드 중단, 현재 실행의 Java 로그 별도 보존이다.
수집 실험 전 하네스 21개 검사는 통과했으나 실제 중단 보증은 아래 별도 실게임 시험으로 판단한다.

실제 VPT 추론 중 STOP에서 추가 기록 오류를 발견했다. 첫 추론 종료 뒤 행동 폐기와
neutral step 성공은 이벤트에 기록됐지만, 이후 환경 close가 10초 유예를 넘겨 worker가
종료되자 summary가 생성되지 않아 입력 해제가 미검증으로 표시됐다. 원본 결과는
`results/real-stop-vpt-inference-20261003T120753506313Z`에 그대로 보존했다.
감독 프로세스가 summary 부재 시 flush된 input_release 이벤트를 근거로 해제 결과만
복구하도록 수정했다. 강제 종료 상태와 execution_valid=false는 유지한다. 같은 상황을
실제 자식 프로세스 종료로 재현한 회귀 검사는 수정 전 실패했고, 수정 후 전체 22개 검사가
통과했다 (`results/stop-release-regression-red.log`, `results/wsl-tests-stop-release-fix.log`).

## 모델만의 합성 진단 (게임 표본에서 제외)

VPT 1x adapter와 고정 공식 VPT/MineRL 소스, 공식 가중치로 검정 이미지 8회 추론을
실행했다. MineRL 설치 전 첫 진입은 import 차단됐으며 공식 MineRL source를 Python
경로에 추가한 별도 진단에서 추론이 통과했다. 실제 게임 환경 객체 대신 공식 action
space/task 설정을 담은 fixture를 사용했고 환경 step은 호출하지 않았다.

- `results/vpt-model-smoke-with-source.json`: gameplay_validated=false, inventory=null.
- 추론 8회 중앙값 53.77ms, p95 5642.68ms (첫 CUDA 추론 포함).
- PyTorch peak allocated 811.55 MiB / reserved 1090 MiB, 모델 로딩 포함.
- 이 수치는 Minecraft 전체 제어 주기, 실제 화면 성능, 획득 또는 정책 성공률이 아니다.
- STEVE-1 추론용 4개 artifact의 다운로드/receipt와 MineCLIP 공식 MD5 검증을 완료했다.
- STEVE-1도 `collect wood` 텍스트 임베딩과 검정 이미지 8회 추론이 통과했다
  (`results/steve1-model-smoke-cuda-path.json`). peak allocated 2097.33 MiB /
  reserved 2144 MiB, 중앙값 46.22ms / p95 11086.50ms (첫 추론 포함).
  실제 게임 렌더러가 없는 진단이므로 로컬 rollout 수용성 또는 수집 성공의 증거가 아니다.

## 현재 진행 상태

사용자의 재개 요청에 따라 19:46 KST부터 실제 실험을 이어갔다. 이전 사용자 STOP은
results에 보존 이동했다. 두 conda 환경, 공식 가중치, PyTorch CUDA 연산,
MineRL/MCP 빌드 및 두 Python용 wheel 설치를 완료했다. 후속 회귀 검사까지 WSL 하네스
21개가 통과했다 (`results/wsl-tests-before-steve1.log`). 이후 실제 중단에서 발견한 요약
기록 오류까지 수정한 최신 전체 검사는 22개 통과했다 (`results/wsl-tests-stop-release-fix.log`).

VPT foundation 1x는 고정 시드 5개, 각 제어 예산 180초의 실제 MineRL 실행을 완료했다.
유효 표본 5/5, 총 5798 step이며 4/5에서 아이템 순증가를 관측했다. 자연어 지시 성공률이
아니다. 모두 중립 입력 전달과 소유 자식 프로세스 잔존 0개를 확인했다. 수치와 한계는
`local-vpt-results-2026-10-03.md`에 기록했다.

STEVE-1은 나무/흙 각 5회, 총 10회를 완료했다. 모두 유효 실게임이고 각 조건 5/5에서
목표 아이템 획득을 확인했다. 총 12721 step, 기술 실패 0, 취소 0, 미실행 0이다.
수치·해석 범위는 `local-steve1-results-2026-10-03.md`에 기록했다.
두 정책을 합쳐 15회 모두 시간 제한 이후 새 정책 dispatch 0, neutral_command_sent,
소유 자식 잔존 0개다. 고정 시드마다 세 조건의 초기 위치/인벤토리는 일치했으나
초기 화면 해시는 달랐다. 완전 결정론의 증거로 보지 않는다.

외부 STOP은 두 정책의 초기화/추론/step 6조건 모두 실제 단계 도중 요청한 것을 확인했다.
요청 및 acknowledgment 이후 새 정책 행동은 0회였고 소유 자식 잔존도 0개였다.
추론/step 중단에서는 중립 입력 전달을 확인했다. 결과는 `local-stop-results-2026-10-03.md`다.
초기화 중단에는 10초 유예 후 강제 종료가 필요했으며, 정책 입력이 시작되기 전이므로
중립 입력은 해당되지 않았다. 첫 추론 진단의 요약 기록 손실도 정상 종료로 숨기지 않았다.

최종 별도 프로세스 점검에서도 프로젝트 관련 WSL Python/Java 잔존 0개를 확인했다.
`results/final-process-check.json`에 기록했고 마지막 진단의 STOP은 유지한다. 실행 중인
실험은 없다. 후속 실행 전에 종료 결과를 검토하고 STOP을 보존 이동한다.

로컬 실게임에서 OOM은 관측하지 않았다. 따라서 자원 부족에 따른 Colab 이전은
필요하지 않았다. Colab 실제 연결/유료 자원 사용은 하지 않았다. 기존 노트북은
클라우드에서 실행 검증되지 않은 준비본으로 남아 있다.
