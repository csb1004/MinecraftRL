# 새 에이전트 작업 시작 안내

이 파일은 재개 절차이고, 현재 결과의 기준 문서는 [progress.md](../progress.md)이다.
전체 대화나 과거 보고서를 처음부터 읽을 필요는 없다.

## 먼저 읽을 순서

1. [AGENTS.md](../../AGENTS.md): 경로, 승인 범위, 보존·STOP·성공 판정 규칙.
2. [progress.md](../progress.md)의 맨 위 최신 항목과 그 항목이 링크한 결과/프로토콜.
3. [구조 지도](architecture.md): 현재 실행 경로와 과거 모듈의 경계.
4. [개발·검사](development.md): 사용할 환경과 변경 검증 방법.
5. 과업을 변경한다면 [과업 분해](../task-decomposition.md),
   [목표 작성 규칙](../specs/2026-10-31-goal-condition-authoring-rules.md),
   [LOGO 판별 계약](../specs/2026-10-31-logo-validator-adapter.md)을 추가로 읽는다.

## 지금 이어받는 상태

2026-11-03 기준 최신 완료 연구는 [E2](../execution-contract-e2-results-2026-11-03.md)다.
공식 정책의 시각·순환 특징과 motor를 고정하고 작은 residual/value head를 학습했다.
일반 개발 5조건의 retained 후보는 4/5였지만 기존 성공 조건을 잃었으므로 승격하지 않았다.
다음 연구 쟁점은 공통 능력 전이, 목표 조건 활용, 기존 능력 보존이다.
이번 유지보수는 이 연구 쟁점을 해결하거나 새 후보를 학습하는 작업이 아니다.

| 구분 | 현재 의미 |
|---|---|
| 구현·실제 검증 | wood/dirt 큐, acquire/possess/preserve 계약, 독립 PPO, 원본 결과 감사, STOP |
| 제한 | 서버 관측 사용, 소규모 개발 비교, 현재 목표 중심 입력, 알려진 재현 차이 |
| 미완성 | 일반 RM 수행 능력, 고정 영상 이상 감지기, 자동 LLM 학습 요청/검증/승격, SQL 지식 확장 |
| 금지된 오해 | 모델 로드=정책 성공, 새 조건 입력=일반화 증명, 준비용 제어=정책 성능 |

`results/STOP`은 유지한다. 완료된 `Invoke-ExecutionContractE2.ps1` 등의 드라이버는 재개 명령이 아니다.
기존 `--rl-*` 명령은 2026-10-25 폐기 경로로 현재 CLI에서 차단한다.
제작 GUI·옛 LLM broker 코드가 남아 있어도 E2의 지원 스킬/자동 학습 루프라고 해석하지 않는다.

## 첫 확인은 읽기 전용으로

```powershell
Set-Location -LiteralPath 'D:\DATA\Desktop\MinecraftRL'
git status --short
Get-Content -LiteralPath .\results\STOP
Get-Content -LiteralPath .\docs\progress.md -TotalCount 85
```

실제 Windows 경로와 현재 연결된 머신을 확인한다. 권한 확대를 다른 PC 접속으로 해석하지 않는다.
이 작업 트리에는 기존 수정·삭제·미추적 연구 파일이 많다. `git reset`, `git clean`이나 삭제 파일
일괄 복원으로 정리하지 않는다. 자신의 작업 시작 상태와 수정 파일을 별도 기록한다.

## 증거를 찾는 위치

- 본 E2: `results/execution-contract-e2-20261103`.
- 회수 보충 비교: `results/execution-contract-e2-20261103-retention-check`.
- 각 배치의 `driver-completion.json`, `work.json`, `analysis.json`, `*-command.json`이 종료/소스/결과/명령의 근거다.
- `source-at-game/`은 당시 코드다. 현재 소스를 고친 뒤 과거 해시를 새 값으로 갱신하지 않는다.
- 독립 학습 산출물은 `checkpoint.pt`, `training.json`. `promoted=false` 후보를 자동으로 기본 정책에 연결하지 않는다.
- 위 로컬 경로가 없는 PC에서는 **자료 미보유**로 기록한다. 빈 폴더를 만들어 재현됐다고 보고하지 않는다.

## 작업 종료 시 인계

`docs/progress.md` 맨 위에 변경 이유, 실제 검사 결과, 미검증 범위, STOP/프로세스 상태,
다음 미해결 과제를 남긴다. 상세 산출물 경로와 실행 명령은 별도 결과 폴더를 링크한다.
기존 실험·원문 명세·공식 가중치를 보존하고, 개발·학습·최종 평가를 구분한다.
유료 API/클라우드/외부 게시의 승인 범위는 별도로 확인한다.
