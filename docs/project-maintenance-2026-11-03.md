# 프로젝트 정리 결과 — 2026-11-03

다른 에이전트가 현재 상태와 실행 경로를 파악할 수 있도록 정리했다.
연구 설계·관측·정책 가중치·완료된 실험은 변경하지 않았다.
[시작 안내](handbook/agent-start.md), [구조](handbook/architecture.md), [검사 방법](handbook/development.md),
[작업 계획·결정 기록](superpowers/plans/2026-11-03-project-maintenance.md).

## 바뀐 구조

- README → 시작 안내 → 최신 진행/결과 → 코드/검사 순서로 진입점을 만들었다.
  `docs`, `src/minecraftrl`, `scripts`, `configs`, `tests`에 역할 지도를 추가했다.
- 공용 adapter 학습을 `minecraftrl.training`으로 분리했다. 기존 R1/E1/E2 스크립트는 얇은 진입점이다.
  자료 검증, 학습, 작업 소스 해시 기록을 각각 분리했다.
- CLI 인수 선언은 `cli_arguments.py`, 실행/프로세스 수명 관리는 기존 `cli.py`에 남겼다.
  `cli.parser`, `cli.sampling_seed`, `train_recovery_adapter.validate`는 호환 import를 제공한다.
- `Invoke-Checks.ps1`로 CPU/Windows 검사를 한 명령으로 실행하고 새 결과 폴더에 요약과 로그를 남긴다.
- 삭제된 옛 실행기 전용 검사는 [원본·사유](../tests/retired/README.md)를 남겨 현행 검사에서 분리했다.
  현행 runtime 재개·원자적 정책 기억 저장 검사는 유지했다.

학습 결과의 새로운 `source_code`와 학습 중 소스 변경 시 저장 거부는 의도한 추가다.
보상, GAE/PPO 수식, optimizer/RNG 순서, 기본값, checkpoint scope/정책 클래스 경로는 유지했다.
공식 backbone, 관측기, 목표 의미를 바꾸지 않았다. 과거 결과 파일에 새 metadata를 소급 적용하지 않는다.

## 검증

| 검사 | 결과 | 범위 |
|---|---|---|
| 정리 전 WSL 전체 | 1,128개, 오류 5, skip 38 | 삭제된 의존성과 잘못된 skip 발견 |
| 최종 WSL 전체 | 1,109개: 통과 1,091, skip 18, 오류 0 | CPU unit/fake 환경 |
| Windows | 20개 통과 | PowerShell 계약, 오프라인 NBT fixture |
| PowerShell 구문 | 22파일 통과 | 실행기를 실제로 실행한 것은 아님 |
| AST 비교 | 동일 | CLI, 자료 검증, metadata 추가 외 학습 계산/효과 순서 |
| 학습 CLI 도움말 | R1/E1/E2 동일 | 기존 플래그/기본값 호환 |
| 패키지 포함 | 통과 | setuptools build_py로 새 하위 패키지 포함 확인 |
| 보호 파일 해시 | 1,254개, 변경 0 | 공식 가중치, 원문, STOP, 최근 E2 증거 |
| 문서 링크/구문 | 통과 | 새 안내의 로컬 링크, Python 구문 |

WSL skip에는 Windows 도구/오프라인 NBT 환경 차이, CUDA 비활성화, 없는 실제 F3 캡처가 포함된다.
Windows 관련 항목은 Host 검사에서 별도 확인했다. 두 환경의 중복 검사는 고유 성공 수로 합산하지 않는다.
GPU/F3 실제 캡처를 이번에 검증하지 않았다.

전체 검사 수 감소는 버그를 숨긴 결과가 아니다. 폐기 실행기 전용 method 25개와 clone 보조 부분을
분리했고, import 오류로 로드조차 안 되던 현행 검사와 새 entrypoint 검사도 실행되도록 했다.
원본 7개 파일은 바이트 동일 사본으로 남았다. 잘못된 Torch skip 20개도 과거 검사로 명시했다.
첫 통합 검사에서 드러난 분석 검사의 실행 순서 의존성은 명시적 모듈 import로 고쳤다.

## 보존과 후속 작업

`results/project-maintenance-20261103`에 기준 740파일 사본, Git 상태, 정리 전후 로그,
AST/도움말/패키지/해시 검증과 최종 변경 명세를 남겼다. 기존 dirty worktree를 되돌리지 않았다.
역사 문서와 실험 폴더를 대규모 이동하면 고정 경로/해시가 깨지므로 위치를 유지했다.

게임·실제 정책 학습·유료 API·설치·다운로드·게시를 수행하지 않았다. `results/STOP`은 그대로다.
이 결과는 코드 정리의 호환성 검증이며 실제 정책 성능 재검증이 아니다.
현재 연구 병목과 후보 미승격 판단은 [E2 결과](execution-contract-e2-results-2026-11-03.md)를 그대로 따른다.
