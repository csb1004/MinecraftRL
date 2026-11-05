# MinecraftRL

공식 STEVE-1을 출발점으로, **LLM이 제안한 지식을 RM과 실제 RL 추가 학습으로 연결하는 지속학습**을 연구한다.
장기 목표와 [원문 연구 명세 20개 항목](docs/research-spec.md)은 유지한다.

**처음 온 에이전트는 [작업 시작 안내](docs/handbook/agent-start.md)부터 읽는다.**
[최신 진행 기록](docs/progress.md) · [문서 지도](docs/README.md) ·
[코드 구조](docs/handbook/architecture.md) · [개발·검사 방법](docs/handbook/development.md)

## 현재 상태 — 2026-11-03

- 최신 완료 실험은 **E2 공통 실행 계약**이다. 로컬 정책 34회와 후보 학습 3회를 완료했다.
- 일반 개발 5조건은 초기 2/5, 보존 후보 4/5였지만 기존 탐색·회수 성공을 잃어 **후보를 승격하지 않았다**.
- wood/dirt의 획득·보유·보존 계약과 RL 입력/보상 연결을 구현했다. 임의의 RM 실행 능력은 미검증이다.
- 서버 관측을 사용한 개발 실험이다. 영상 전용 관측이나 자동 LLM→RL 학습 루프 완성을 뜻하지 않는다.
- **게임·학습 종료, `results/STOP` 유지.** 완료 배치를 다시 실행하거나 STOP을 자동 해제하지 않는다.
- 프로젝트 정리 완료: [시작 안내](docs/handbook/agent-start.md), 공용 학습 패키지, 표준 검사 명령을 마련했다.
  [정리 결과·검증 범위](docs/project-maintenance-2026-11-03.md)에 호환성과 보존 검사를 기록했다.

근거: [E2 결과와 한계](docs/execution-contract-e2-results-2026-11-03.md),
[E2 명세](docs/specs/2026-11-03-execution-contract-e2.md),
[실행·독립 학습 사용법](docs/execution-contract-e2-usage.md).

## 빠른 진입

실제 프로젝트는 `D:\DATA\Desktop\MinecraftRL`이다. Windows에서는 PowerShell을 사용하고,
정책 환경은 WSL의 격리 환경을 사용한다. `gh`를 사용하지 않는다.

이미 설치된 PC에서 게임 없이 코드 상태만 확인한다.

```powershell
Set-Location -LiteralPath 'D:\DATA\Desktop\MinecraftRL'
& .\scripts\Invoke-Checks.ps1
```

검사는 기존 WSL/Windows 환경만 사용하며 결과를 새 `results/checks-*`에 남긴다.
새 PC는 먼저 [공식 모델 설치 안내](docs/new-pc-setup.md)를 따른다.
저장소 코드만으로 과거 실험의 가중치·월드·실행 로그가 복원되지는 않는다.
정확한 재현에는 해당 실행의 별도 자료와 `source-at-game`이 필요하다.

## 폴더 역할

| 경로 | 역할 |
|---|---|
| [src/minecraftrl](src/minecraftrl/README.md) | 실행, RM, 관측, 정책, 학습 라이브러리 |
| [scripts](scripts/README.md) | 설치·검사·실행·독립 학습 진입점, 완료 실험 드라이버 |
| [tests](tests/README.md) | 현재 회귀 검사와 별도로 보관한 폐기 실행기 검사 |
| [configs](configs/README.md) | 공식 소스/모델 출처, 환경, 검증용 목표 예제 |
| [docs](docs/README.md) | 현재 안내, 고정 명세, 결과, 과거 연구 기록 |
| `results/` | 실행별 원본 증거·체크포인트·월드·소스 사본, Git 제외 |
| `weights/`, `external/`, `.runtime/` | 공식 가중치, upstream 소스, 격리 환경, Git 제외 |
| `data/`, `logs/`, `.cache/` | 로컬 자료·로그·캐시, Git 제외 |
| `notebooks/` | 별도 클라우드 실행 준비 자료; 실제 연결은 별도 승인 |

과거 문서는 날짜가 있는 연구 기록이다. 특히 2026-10-25에 폐기한 학습기/자료의
실행 명령은 현재 안내로 사용하지 않는다. 과거 경로는 재현 근거를 보존하기 위해 유지한다.
