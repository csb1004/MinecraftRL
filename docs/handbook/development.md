# 개발과 검사

[시작 안내](agent-start.md) · [구조](architecture.md) · [설치](../new-pc-setup.md)

## 환경을 구분한다

| 환경 | 사용 범위 | 기준 |
|---|---|---|
| Windows PowerShell | 설치 조정, 검사, STOP, 로컬 파일 작업 | 프로젝트 내부 스크립트 |
| Windows `.runtime/fixture-tools` | CPU 데이터·오프라인 NBT·PowerShell 계약 검사 | `configs/fixture-tools.txt` |
| WSL `.runtime/steve1` | STEVE-1/PyTorch와 주 회귀 검사 | `configs/environment-steve1.yml` (Python 3.10) |
| WSL `.runtime/vpt` | 별도 공식 VPT 환경 | `configs/environment-vpt.yml` |

기존 환경을 쓰고 시스템 Python/Java/드라이버를 변경하지 않는다. 설치가 없는 PC에서는
[설치 안내](../new-pc-setup.md)를 먼저 따른다. 검사 명령은 환경을 자동 생성하지 않는다.

## 한 번에 코드 검사

```powershell
Set-Location -LiteralPath 'D:\DATA\Desktop\MinecraftRL'
& .\scripts\Invoke-Checks.ps1
```

기존 WSL 환경의 전체 CPU unit 검사, Windows PowerShell 계약/오프라인 월드 검사,
PowerShell 파일 구문 검사를 실행한다. `results/checks-*/summary.json`과 로그에 결과를 남기고
실패 시 오류를 반환한다. 실패/skip은 각각 확인한다. GPU·실제 게임·모델 다운로드·API 호출은 하지 않는다.
`results/STOP` 존재/내용을 바꾸지 않고 전후 해시를 확인한다.

```powershell
# 변경 범위가 명확할 때만 한쪽 실행
& .\scripts\Invoke-Checks.ps1 -Profile Unit
& .\scripts\Invoke-Checks.ps1 -Profile Host
```

특정 검사는 WSL에서 `PYTHONPATH=<프로젝트의 WSL 경로>/src:<경로>/tests`,
`CUDA_VISIBLE_DEVICES=`를 지정하고 기존 환경의
`python -m unittest test_training_entrypoints test_execution_training -v`로 실행한다.
모듈명을 추측하지 말고 실제 `tests/test_*.py`에서 확인한다.

일반 unit 검사는 fake game/임시 디렉터리로 논리를 확인한다. 이 결과로 실제 STOP, GPU 추론,
게임 rollout이나 정책 성능이 재검증됐다고 보고하지 않는다. 필요한 실전 검사는 별도 프로토콜을 만든다.

## 변경 절차

1. 최신 진행 기록과 관련 고정 명세를 읽고 현재 Git 수정 상태를 보존한다.
2. 바꾸는 책임과 영향을 받는 모듈/검사를 정한다. 공용 코드는 `src`, 명령 진입점은 `scripts`에 둔다.
3. 공개 CLI·import·체크포인트 클래스 경로를 유지한다. 연구 의미를 바꾸면 새 조건/프로토콜로 분리한다.
4. 관련 검사 후 필요하면 전체 검사로 넓힌다. 삭제된 과거 실행기의 검사를 현재 검사로 속이지 않는다.
5. 실제 검증 범위와 결과 파일을 `docs/progress.md`에 남긴다.

`tests/retired`에는 삭제된 옛 실험 실행기에 의존하던 검사 원본을 텍스트로 보관한다.
[보관 사유](../../tests/retired/README.md)는 현재 성공 수에 포함하지 않는다.

## 독립 학습 코드 수정

학습 시작 명령은 [E2 사용법](../execution-contract-e2-usage.md)을 따른다.
`training.adapter.main(profile, argv)`는 인수를 받고, `train(args, profile)`가 실제 학습을 담당한다.
인수 도움말 확인만으로 학습을 시작하지 않는다. 유효한 새 train 자료·출력 경로·명시적 재개가 필요하다.

새 결과의 `source_code`는 Python/Java 구현과 학습 진입점의 상대 경로·SHA256이다.
학습 시작/저장 전 해시가 다르면 후보를 저장하지 않는다. 이것은 환경/공식 모델/월드의 해시를 대체하지 않는다.
기존 완료 배치의 `work.json`과 `source-at-game`은 당시 증거이므로 절대 최신 소스에 맞춰 덮어쓰지 않는다.

## 정리 범위

실험 경로와 대용량 자료는 이동/삭제하지 않는다. 문서는 [문서 지도](../README.md)로 현재/과거를 구분한다.
Git 제외 대상은 `.gitignore`를 따른다. 가중치, 비밀정보, 대형 원본을 코드 커밋에 넣지 않는다.
이 로컬 정리 작업은 커밋·푸시·릴리스 발행을 자동 수행하지 않는다.
