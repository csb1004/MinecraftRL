# E2 실행 계약·학습 경로

[명세와 고정 개발 실험](specs/2026-11-03-execution-contract-e2.md).
실제 결과는 [완료된 비교·남은 병목](execution-contract-e2-results-2026-11-03.md)과 `docs/progress.md`에서 확인한다. 구현/검사 통과는 정책 성공을 뜻하지 않는다.
자연 수집만으로 회수 능력 보존을 판정하지 않도록, 후보 생성 전에 선언한 별도 회수 개발 비교는
`results/execution-contract-e2-20261103-retention-check/plan.json`에 둔다. 원래26회와 합쳐 성공률을 만들지 않는다.

## 목표 작성

사람과 이후 LLM은 같은 `goal_spec_v2`를 작성하고 같은 compiler를 통과한다. API 호출은 자동으로 발생하지 않는다.

```json
{
  "schema": "goal_spec_v2",
  "goals": [
    {"id":"wood","target":"wood","quantity":1,"priority":80,
     "depends_on":[],"requires":[],"effect":"acquire","preserve":[]},
    {"id":"dirt","target":"dirt","quantity":2,"priority":80,
     "depends_on":["wood"],"requires":[],"effect":"possess",
     "preserve":[{"target":"wood","quantity":1}]}
  ]
}
```

첫 목표는 활성화 뒤 새 원목 1개, 두 번째는 현재 흙 2개 이상 보유를 요구한다.
두 번째 목표 수행 중 원목은 1개 이상 유지해야 한다. 흙이 이미 충분하면 새 행동 없이 완료한다.
`preserve`는 해당 목표가 활성화된 동안의 제약이다. 이전 목표 자원을 자동으로 모두 보존하는 규칙은 아니다.
현재 등록 대상은 wood/dirt뿐이다. 제작·항해 등의 새 능력은 지원된 것처럼 선언하지 않는다.

PowerShell의 프로젝트 격리 Python으로 API용 스키마를 만들 수 있다.

```powershell
$env:PYTHONPATH='D:\DATA\Desktop\MinecraftRL\src'
& .runtime/fixture-tools/Scripts/python.exe -m minecraftrl.goal_spec request --version 2 --output results/my-e2-request.json
& .runtime/fixture-tools/Scripts/python.exe -m minecraftrl.goal_spec compile --input results/my-goals.json --output results/my-compiled-goals.json
```

출력 파일은 새 경로여야 한다. 기존 v1 작성/실행 의미는 변경하지 않는다.

## 실행과 독립 학습

게임은 기존 saved-start recovery 명령에 `--execution-learning`을 추가하고, v2 program,
`--recovery-variant nonlinear-on`, 검증된 world/role을 지정한다. 승인된 STOP 해제와 새 실험 계획 없이
예전 배치 드라이버를 다시 실행하지 않는다. 정확한 실행 인수는 배치의 `*-command.json`에 보존한다.

학습 진입점은 WSL의 기존 STEVE-1 환경에서 `scripts/train_execution_adapter.py`이다.
공용 구현은 `src/minecraftrl/training/adapter.py`, 자료 검증은 `training/validation.py`에 있다.
2026-11-03 코드 정리 후에도 기존 인수와 checkpoint scope는 유지한다.
필수 인수: `--root`, 새 `--output`, `--stop-file`, `--variant nonlinear-on`, `--runs <새 train 실행 폴더들>`.
선택 인수: `--critic-warmup-epochs 0..10`, `--retention-runs <별도 train 폴더들>`, `--parent <직전 checkpoint>`.

- PPO 자료는 현재 parent와 실제 행동 확률이 일치하는 새로운 train 궤적이어야 한다.
- 보존 자료는 PPO와 별도인 train 자료이며 공식 정책 분포에 대한 KL에만 사용한다.
- 개발/최종 평가 자료는 어느 학습 입력에도 사용할 수 없다.
- STOP이 있으면 학습도 시작/다음 update를 거부한다. 게임 중에는 가중치를 갱신하지 않는다.
- 출력 `checkpoint.pt`, `training.json`은 계보, source hashes, 목표별 비중, critic 보정과 KL 기록을 남긴다.
- 새 학습 산출물의 `source_code`는 학습 시작 시점의 작업 소스 트리 해시다. 저장 전 소스가 바뀌면 거부한다.
  기존 완료 실험의 산출물/소스 사본에는 이 필드를 소급 추가하지 않는다.
- 기본 모델과 순환망은 고정된다. E2는 전체 STEVE-1을 재학습하는 코드가 아니다.

## 실행 근거와 후속 연결

`execution-trajectory.pt`: 실제 dispatch된 행동의 특징·확률·value·실행 요청·서버 결과·전체 과업 보상.
`execution-events.jsonl`, `rm-rows.json`, `events.jsonl`: 요청→실제 상태 변화의 대응 근거.
`failure-evidence.json`: 원인이 확정되지 않은 정체를 포함한 보수적인 실패 분류. 실패 하나로 자동 학습하지 않는다.

후속 LLM/이상 감지기/SQL은 이 계약과 결과를 사용한다. `BoundaryUpdates.submit()`의 우선순위 변경은
실행 소유자가 `apply()`를 호출하는 확정 행동 경계에서만 반영한다. 대기 중 추론이 있으면 미룬다.
중복·오래된 revision은 거부하고 의미/보상/대상 변경은 새 버전의 별도 실행으로 분리한다.
현재 구현은 단일 실행 소유자의 mailbox API이며, 동시 모니터링 스레드나 LLM/SQL 서비스 완성을 뜻하지 않는다.

정책/value에 실제로 입력되는 것은 현재 목표/효과/보존 조건과 기존 단계 정보·전체 진전도이다.
미래 RM 그래프 전체를 인코딩한 입력은 아직 아니다. 새 순서·수량의 조합과 완전히 새로운 효과는
별도 평가해야 하며, 임의의 LLM RM을 즉시 수행하거나 미래 계획에 최적이라고 보장하지 않는다.
다음 목표의 준비 조건은 명시적인 실행 계약으로 전달해야 한다.

원본 서버 감사: `scripts/analyze_execution_contracts.py <batch>`.
해당 E2 배치의 CPU tensor/시작 상태 비교: `results/execution-contract-e2-20261103/audit_tensors.py`.
실제 STOP 검사는 별도 실행으로 성능 집계에서 제외한다.

이번 세 후보는 개발용으로 보존하며 기본 정책으로 승격하지 않았다. 완전한 재현에는 해당 배치의
고정 세계·입력·`source-at-game` 사본과 실행 인수를 사용한다. 실험 후 E2 설명용 두 필드의 변경은
`postrun-metadata-receipt.json`에 분리했고, 과거 실행 기록의 의미를 소급 변경하지 않았다.
