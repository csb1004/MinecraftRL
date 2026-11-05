# 목표별 감시 명세 M1+M2 구현 계획 (명세 검사기 + 오프라인 재생)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (AGENTS.md forbids automatic subagents). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 하위 목표별 감시 명세를 검사기로 검증하고, 기존 실행 기록에 적용했다면 점검이 언제 켜졌을지 계산한다. 기준은 개발 월드 기록으로만 정하고, 평가 월드 기록으로 감지 시점과 오탐을 잰다.

**Architecture:**
- `pipeline.monitor_spec`: 고정 신호 목록, 하위 목표 명세, 사람이 쓴 제약의 스키마와 검사를 담당한다.
- `pipeline.monitor_replay`: 기록된 `timeline.jsonl`·`telemetry.jsonl`을 단계별로 자른다. 기존 `rm.sample_from`과 `StallDetector`로 신호를 다시 계산하고, 개발 기록에서 시간 기준을 맞춘 뒤 평가 기록에 적용한다.
- `scripts/replay_monitor_spec.py`: 실행 진입점이다. 결과를 새 `results/` 폴더에 쓴다.
- 실시간 파이프라인(`session.py`, `monitor.py`)은 바꾸지 않는다.

**범위 밖 (M3 계획으로 미룸):**
- 실시간 점검 연결
- 제약 집행과 기한 실현성 검사
- LLM이 명세를 쓰는 API 호출
- 낭비 신호를 학습 보상에 넣는 일

이번 계획은 명세 형식과 검사, 기록 재생만 다룬다.

**Tech Stack:** Python 3.10 (WSL `.runtime/steve1`), unittest, 기존 `minecraftrl.pipeline`·`minecraftrl.stall_detection`.

**Spec:** `docs/specs/2026-11-05-goal-monitor-spec-plan.md` (§4 명세, §5 관측 목록, §7 낭비 신호, §8 제약, §10 M1·M2)

## Global Constraints

- 게임, GPU, API를 쓰지 않는다. 입력 `results/*` 기록은 읽기만 한다.
  출력은 새 폴더 `results/monitor-spec-replay-20261105/`에만 쓰고(Git 제외 대상), 이미 있으면 덮어쓰지 않는다.
- **기준 시간은 개발 월드(seed 130363) 기록으로만 정한다.** 평가 월드(seed 104729) 기록은 평가에만 쓴다.
  두 집합의 실행 폴더나 seed가 겹치면 실행을 거부한다.
- 명세에는 시간 숫자를 쓰지 않는다(시간은 기록에서 온다). 명세는 성공 조건(`done`, `success`)을 가질 수 없다(호출자 소유).
  신호는 고정 목록에서만 고르고 `exec/eval`은 쓰지 않는다.
- 제약의 `source`는 `human`/`plan`/`system`만 허용한다. `llm`은 거부한다(2026-11-05 사용자 결정).
- 사람과 LLM이 쓴 명세에 같은 검사를 적용한다. `author`는 기록만 한다.
- 기존 `FastMonitor`·`StallDetector`·`session.py`의 동작과 실시간 경로는 수정하지 않는다.
- 테스트 명령 `WSLTEST <modules>` = Git Bash에서
  `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -- env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/mnt/d/DATA/Desktop/MinecraftRL/src:/mnt/d/DATA/Desktop/MinecraftRL/tests /mnt/d/DATA/Desktop/MinecraftRL/.runtime/bin/micromamba run -p /mnt/d/DATA/Desktop/MinecraftRL/.runtime/steve1 python -m unittest <modules> -v`
  (`Invoke-Checks.ps1`은 이 PC의 PowerShell 5.1에서 테스트 전에 멈추므로 lane을 직접 실행한다.)
- 커밋은 새 브랜치 `monitor-spec-replay`에 Task마다 한다. `main` 병합과 푸시는 사용자가 말할 때만 한다.

## Review Focus

1. 텔레메트리 중간에 `server_available != 1`이거나 tick이 건너뛴 구간 → 재생이 그 공백을 넘어 정체 점검을 내지 않아야 한다 (Task 3 `test_observation_gap_resets_stall`).
2. `step_finished`가 없는 단계(실행이 중간에 끊김) → `unfinished`로 분류한다. 성공에도 문제에도 세지 않는다 (Task 2 `test_unfinished_step_kept_separate`).
3. 개발 기록에 완료 사례가 3개 미만인 하위 목표 → 기준을 만들지 않고 "기록 부족"으로 보고한다. 1–2개로 기준을 세우지 않는다 (Task 3 `test_fit_needs_min_records`).
4. 같은 실행이나 같은 seed가 개발·평가 양쪽에 들어감 → 스크립트가 종료 코드 2로 거부한다 (Task 4 `test_overlapping_runs_rejected`, `test_shared_seed_rejected`).
5. 명세가 없는 스킬(예: `clear_above`가 빠진 명세) → 충돌하지 않고 `no_spec`으로 보고한다 (Task 3 `test_step_without_spec_reported`).

---

### Task 1: 명세 스키마와 검사기

**Files:**
- Create: `src/minecraftrl/pipeline/monitor_spec.py`
- Create: `configs/monitor-specs/iron-demo-v1.json`
- Test: `tests/test_pipeline_monitor_spec.py`

**Interfaces:**
- Produces:
  - `ENDINGS = ('bounded', 'until_found')`, `UNITS = ('step', 'item', 'block')`
  - `SIGNALS: dict[str, tuple[str, tuple[str, ...]]]` — 이름 → (역할 `progress`/`anomaly`/`waste`, 허용 끝나는 방식).
    값은 아래 표 그대로 쓴다.
  - `class SpecError(ValueError)` — `.reasons: list[str]` (모든 위반을 한 번에 모은다)
  - `@dataclass(frozen=True) SubgoalSpec(key: str, ending: str, unit: str, found: str | None, progress: tuple[str, ...], anomaly: tuple[str, ...], waste: tuple[str, ...])`
  - `@dataclass(frozen=True) Constraint(id: str, form: str, scope: str, source: str, strength: str, predicate: dict, by: dict | None, during: dict | None, on_violation: str)`
  - `@dataclass(frozen=True) MonitorSpec(version: str, author: str, margin: float, subgoals: dict[str, SubgoalSpec], constraints: tuple[Constraint, ...])`
    - `.for_step(skill: str, target: str | None) -> SubgoalSpec | None` — `"skill:target"` 키를 먼저, 없으면 `"skill"` 키를 찾는다.
  - `parse_spec(raw: dict) -> MonitorSpec`, `load_spec(path) -> MonitorSpec` (JSON 중복 키 → `SpecError`)
  - `constraint_priority(c: Constraint) -> int` — system 0, human·hard 1, plan 2, soft 3

| 신호 | 역할 | 허용 |
|---|---|---|
| `objective_inventory_increased` | progress | 둘 다 |
| `new_position_cell`, `new_search_view` | progress | until_found |
| `rise` | progress | bounded |
| `no_progress`, `repeated_cycle` | anomaly | until_found |
| `stationary_repetition`, `harvest_without_drop`, `needed_tool_lost`, `low_health` | anomaly | 둘 다 |
| `blocked_overhead`, `climb_without_block` | anomaly | bounded |
| `moving_but_stuck`, `looking_down` | waste | 둘 다 |
| `revisiting_area` | waste | until_found |

제약 값:
- `form`: `always`/`deadline`/`conditional`
- `strength`: `hard`/`soft`
- `on_violation`: `warn`/`check`/`hold`/`mask`/`cost`/`report_violation`/`stop`/`refuse`
- 술어: `{"fact": F, "op": "ge"|"le"|"eq", "value": number, "item": str (fact가 inventory일 때만 필수)}`.
  `F`는 `inventory`/`health`/`day_time`/`elapsed_s`/`api_calls` 중 하나다.
- `deadline`은 `by`가 필수, `conditional`은 `during`이 필수다.
- 같은 사실(같은 item)에 대해 hard 제약끼리 `ge a`와 `le b`(a > b)가 있으면 모순으로 거부한다.

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_pipeline_monitor_spec.py`. 각 테스트는 최소 유효 명세 `VALID`(bounded 하나, until_found 하나)를 고쳐서 만든다.
  - `test_valid_spec_parses`: `parse_spec(VALID).for_step('mine_iron_text', 'iron_ore').ending == 'until_found'`. 이 테스트 안에서 `for_step('craft', 'stone_pickaxe')`가 `"craft:stone_pickaxe"` 키를 `"craft"`보다 먼저 고르는지도 확인한다.
  - `test_all_reasons_collected`: 미등록 신호와 잘못된 ending을 함께 넣으면 `SpecError.reasons`가 2개 이상이다.
  - `test_unknown_signal_rejected`, `test_role_mismatch_rejected`: `waste`에 `no_progress`를 넣으면 거부한다.
  - `test_signal_not_allowed_for_ending`: bounded에 `repeated_cycle`을 넣으면 거부한다.
  - `test_until_found_requires_found`, `test_bounded_forbids_found`
  - `test_time_numbers_rejected`: 하위 목표에 `best_s`, `check_s`, `expected_s`, `max_s` 중 하나라도 있으면 거부하고, 이유에 `"times come from records"`가 들어 있다.
  - `test_success_fields_rejected`: `done`이나 `success`가 있으면 거부하고, 이유에 `"caller owns success"`가 들어 있다.
  - `test_llm_constraint_rejected`: `source: "llm"`이면 거부하고, 이유에 `"2026-11-05"`가 들어 있다.
  - `test_unobservable_fact_unsupported`: `fact: "noise"`이면 거부하고, 이유에 `"unsupported"`가 들어 있다.
  - `test_deadline_requires_by`, `test_conditional_requires_during`
  - `test_contradicting_hard_constraints_rejected`: inventory iron_ore `ge 1`과 `le 0`, 둘 다 hard면 거부한다.
  - `test_duplicate_json_key_rejected`: `load_spec`에 같은 키가 두 번 나오는 파일을 주면 `SpecError`.
  - `test_same_checks_for_human_and_llm_author`: `author`만 바꾼 두 명세가 같은 결과(같은 통과 여부와 같은 `reasons`)를 낸다.
  - `test_priority_order`: system < human·hard < plan < soft 순서로 정렬된다.
  - `test_margin_range`: margin이 0 미만이거나 5 초과면 거부한다.
  - `test_shipped_iron_spec_parses`: `configs/monitor-specs/iron-demo-v1.json`이 통과하고, `wood_visual`·`mine_iron_text`가 until_found다.

- [ ] **Step 2: 실패 확인** — `WSLTEST test_pipeline_monitor_spec` → `ModuleNotFoundError: minecraftrl.pipeline.monitor_spec`

- [ ] **Step 3: 구현**
  - `monitor_spec.py`를 만든다. 검사는 모두 모아 `SpecError(reasons)` 하나로 던진다. 중복 키 검사는 `object_pairs_hook`으로 한다.
  - `configs/monitor-specs/iron-demo-v1.json`을 쓴다.
    - `version: "iron-demo-monitor-v1"`, `author: "human"`, `margin: 0.2`
    - 하위 목표:

      | 키 | ending | found | unit | anomaly | waste |
      |---|---|---|---|---|---|
      | `wood_visual` | until_found | `log` | item | `no_progress`, `repeated_cycle`, `stationary_repetition` | `moving_but_stuck`, `revisiting_area` |
      | `mine_iron_text` | until_found | `iron_ore` | item | `no_progress`, `repeated_cycle`, `stationary_repetition`, `needed_tool_lost` | `moving_but_stuck`, `revisiting_area`, `looking_down` |
      | `mine_stone_text` | bounded | – | item | `harvest_without_drop`, `needed_tool_lost` | `moving_but_stuck` |
      | `climb_out_text` | bounded | – | block | `blocked_overhead`, `climb_without_block` | (없음) |
      | `craft:wooden_pickaxe` | bounded | – | step | | |
      | `craft:stone_pickaxe` | bounded | – | step | | |
      | `craft` | bounded | – | step | | |
      | `equip` | bounded | – | step | | |
      | `clear_above` | bounded | – | step | | |

    - 제약 두 개를 현재 시스템 규칙 그대로 적는다.
      - 체력: `always`, `system`, `hard`, `{"fact": "health", "op": "ge", "value": 6.01}`, `hold`
        (현재 코드는 6 이하에서 hold하므로 "6 초과"를 뜻한다.)
      - 실행 상한: `always`, `system`, `hard`, `{"fact": "elapsed_s", "op": "le", "value": 900}`, `stop`

- [ ] **Step 4: 통과 확인** — `WSLTEST test_pipeline_monitor_spec` → 전부 OK

- [ ] **Step 5: 커밋** — `git checkout -b monitor-spec-replay` 후, 위 세 파일과 이 계획서를 커밋한다. 메시지: "Add monitor spec schema and validator".

### Task 2: 기록 불러오기와 단계 자르기

**Files:**
- Create: `src/minecraftrl/pipeline/monitor_replay.py`
- Test: `tests/test_pipeline_monitor_replay.py`

**Interfaces:**
- Consumes: `rm.sample_from(info, action, t)`, `viewer.read_jsonl(path)`
- Produces:
  - `@dataclass(frozen=True) StepRecord(run: str, id: str, skill: str, target: str | None, block: str | None, count: int, start_t: float, end_t: float, start_step: int, end_step: int | None, status: str)`
  - `@dataclass Run(name: str, path: Path, seed: int | None, steps: list[StepRecord], rows: list[dict])`
  - `load_run(run_dir) -> Run`
    - `seed`는 `request.json` 최상위 `seed`에서 읽는다.
    - 단계 경계는 `step_started`(`t`, `start_step`)와 `step_finished`(`t`, `step`, `status`)로 정한다.
    - 같은 `id`가 다시 시작되면 새 단계로 본다.
    - `step_finished`가 없으면 `status='unfinished'`, `end_t`는 timeline의 마지막 `t`로 둔다.
  - `sample_from_row(row: dict) -> Sample`
    - `sample_from({'workshop': {**row, 'server_inventory': row.get('inventory') or {}}, 'demo': row.get('demo') or {}}, row.get('action') or {}, row['t'])`
  - `policy_rows(run: Run, step: StepRecord) -> list[dict]`
    - `controller == 'policy'`이고 `start_step <= row['step'] < end_step`인 행 (`end_step`이 None이면 상한 없음)

- [ ] **Step 1: 실패하는 테스트 작성** — 임시 폴더에 `request.json`·`timeline.jsonl`·`telemetry.jsonl`을 쓰는 helper `write_run(dir, steps, rows, seed)`를 만든다.
  - `test_steps_from_timeline`: 시작·종료 두 쌍 → `StepRecord` 2개(`status`, `count`, `start_step`, `end_step` 일치)
  - `test_restarted_step_id_is_new_step`: 같은 `id`가 두 번 시작·종료 → 단계 2개
  - `test_unfinished_step_kept_separate`: 종료가 없는 단계 → `status == 'unfinished'`, `end_t ==` 마지막 timeline `t`
  - `test_seed_read_from_request`: `seed == 104729`
  - `test_sample_from_row_matches_live_adapter`: 행 하나를 `sample_from_row`에 넣은 결과가, 같은 값을 `workshop`/`demo` 형태로 `sample_from`에 넣은 결과와 같다(`position`, `inventory`, `hit_block`, `valid`)
  - `test_policy_rows_exclude_rule_and_other_steps`: `rule` 행과 다음 단계 행은 빠진다

- [ ] **Step 2: 실패 확인** — `WSLTEST test_pipeline_monitor_replay` → `ModuleNotFoundError`

- [ ] **Step 3: 구현** — `monitor_replay.py`에 위 네 가지를 만든다.

- [ ] **Step 4: 통과 확인** — `WSLTEST test_pipeline_monitor_replay` → OK

- [ ] **Step 5: 커밋** — "Load recorded pipeline runs for monitor replay"

### Task 3: 기준 시간 맞추기와 단계 재생

**Files:**
- Modify: `src/minecraftrl/pipeline/monitor_replay.py`
- Test: `tests/test_pipeline_monitor_replay.py`

**Interfaces:**
- Consumes: Task 1 `MonitorSpec`·`SubgoalSpec`, Task 2 `Run`·`StepRecord`·`policy_rows`·`sample_from_row`.
  기존 `StallDetector`·`StallSample`, `steps.block_matches`·`steps.item_count`도 쓴다.
- Produces:
  - `@dataclass(frozen=True) TimeEntry(n: int, best: float, worst: float)` — 단위당 초
  - `unit_count(step: StepRecord, sub: SubgoalSpec) -> int`: `step` → 1, `item`·`block` → `step.count`
  - `found_time(rows: list[dict], step: StepRecord) -> float | None`: `block_matches(hit_block, step.block or step.target)`가 처음 참인 행의 `t`
  - `fit_times(runs, spec, *, min_records=3) -> tuple[dict[str, TimeEntry], dict[str, int]]` — (기준표, 기록 부족 키 → 개수)
    - bounded: `completed` 단계의 `duration / unit_count`
    - until_found: `completed`이면서 찾은 단계의 `(end_t − found_t) / unit_count`를 `"<key>:after_found"`로 넣는다. 찾기 전 구간은 넣지 않는다.
  - `replay_step(run, step, sub, table, margin) -> dict`
    - 반환 키: `run, step, skill, status, ending, duration_s, found_s, check_s, check_reason, saved_s, waste, note`
    - bounded: 기준이 있으면 `limit = worst * (1 + margin) * unit_count`, `duration > limit`이면 `check_s = limit`, `check_reason = 'over_check_time'`
    - until_found:
      - 단계마다 새 `StallDetector`를 만든다. 찾기 전 정책 행을 `StallSample(stage='search', ...)`로 넣는다. live `FastMonitor.update`와 같은 필드(`inventory_count`, `block_removed`, `context_id`)를 쓴다.
      - 처음 `stalled`가 되고 그 `reason`이 `sub.anomaly`에 있으면 그 시각을 점검 후보로 둔다.
      - 찾은 뒤에는 `:after_found` 기준으로 bounded와 같은 시간 점검을 한다.
      - 둘 중 이른 것을 `check_s`로 쓴다.
    - `saved_s`: 점검이 켜진 경우 `duration_s − check_s`, 아니면 None
    - `waste`: `sub.waste`에 있는 이름만, 정책 행 가운데 비율(0–1)
      - `moving_but_stuck`: 이동 키를 누르고, 공격하지 않고, 최근 40 server tick 동안 수평 이동이 0.5블록 미만
      - `looking_down`: `pitch > 60`
      - `revisiting_area`: 직전 행보다 0.1블록 이상 움직였는데, 이미 방문한 2블록 격자 칸 안
    - `sub`가 None이면 `note = 'no_spec'`, 기준이 없으면 `note = 'no_records'`

- [ ] **Step 1: 실패하는 테스트 작성** (Task 2의 helper로 합성 기록을 만든다)
  - `test_bounded_check_fires_after_worst_plus_margin`: 개발 기록 completed 3개(10, 12, 20초, unit step) → 기준 `worst = 20`. 평가 단계 30초 → `check_s == 24.0`, `saved_s == 6.0`. 23초 단계는 점검하지 않는다.
  - `test_per_unit_normalisation`: block 단위 오르기(3칸 9초, 6칸 12초, 2칸 8초) → `worst == 4.0`(8/2). 5칸 단계의 limit은 `4 * 1.2 * 5 = 24`
  - `test_fit_needs_min_records`: completed 2개뿐 → 기준표에 없고 `insufficient[key] == 2`
  - `test_fit_ignores_non_completed`: `time_limit`·`anomaly_hold`는 기준에 들어가지 않는다
  - `test_until_found_has_no_time_check_before_found`: 계속 새 칸으로 움직이는 600초 탐색 → `check_s is None`
  - `test_until_found_stall_check`: 제자리에서 같은 행동만 하는 탐색 → 약 20초(400 tick) 뒤 `check_reason == 'stationary_repetition'`
  - `test_found_switches_to_bounded`: 30초에 조준선에 대상이 잡히고, after_found 기준 worst 5초, 단계는 60초 → `check_s == 36.0`
  - `test_observation_gap_resets_stall`: 15초 정체 → 공백(tick 50 건너뜀) → 15초 정체 → 점검 없음
  - `test_waste_fractions`: 이동 키를 누르고 제자리인 행 3/4 → `waste['moving_but_stuck'] == 0.75`. `pitch` 70인 행 1/4 → `looking_down == 0.25`
  - `test_step_without_spec_reported`: 명세에 없는 스킬 → `note == 'no_spec'`, 예외 없음

- [ ] **Step 2: 실패 확인** — `WSLTEST test_pipeline_monitor_replay` → 새 테스트가 `AttributeError`/`ImportError`로 실패

- [ ] **Step 3: 구현** — 위 Interfaces 그대로 만든다. 정체 판정은 `StallDetector`를 그대로 쓰고 다시 구현하지 않는다.

- [ ] **Step 4: 통과 확인** — `WSLTEST test_pipeline_monitor_replay test_pipeline_monitor test_stall_detection` → OK (기존 검사도 그대로 통과)

- [ ] **Step 5: 커밋** — "Fit check times from dev records and replay steps"

### Task 4: 평가 요약, 재현도 확인, 실행 스크립트

**Files:**
- Modify: `src/minecraftrl/pipeline/monitor_replay.py`
- Create: `scripts/replay_monitor_spec.py`
- Test: `tests/test_pipeline_monitor_replay.py`

**Interfaces:**
- Consumes: Task 3 `fit_times`, `replay_step`
- Produces:
  - `evaluate(runs, spec, table, margins: tuple[float, ...]) -> dict` — margin마다 결과를 만든다.
    - `problem`: `status == 'time_limit'`인 단계. 감지 수와 `saved_s` 중앙값
    - `success`: `completed` 단계, 그리고 until_found 중 찾았고 `time_limit`이 아닌 단계. 점검이 켜진 수가 오탐이다
    - `other`: 나머지. 개수만 센다
    - 스킬별 같은 표
    - 단계별 행: 명세의 margin 기준
  - `stall_fidelity(run) -> dict`
    - live와 같이 단계 전체 정책 행에 `stage='search'` 감지기를 돌린다.
    - timeline의 기록된 `monitor_event`(`no_progress`, `repeated_cycle`, `stationary_repetition`, `no_goal_progress`) 중 같은 단계에서 같은 종류가 ±2초 안에 다시 나오는 비율을 센다: `{'recorded': n, 'reproduced': m}`
  - `scripts/replay_monitor_spec.py main(argv) -> int`
    - 인수: `--spec`, `--fit DIR...`, `--eval DIR...`, `--margins 0.2,0.5,1.0`, `--min-records 3`, `--out DIR`
    - `timeline.jsonl`·`telemetry.jsonl`이 없는 폴더는 건너뛰고 목록에 남긴다.
    - 거부(종료 코드 2): 출력 폴더가 이미 있음, 개발·평가 폴더가 겹침, seed가 겹침
    - `report.json`에 쓰는 내용:
      - 입력 실행 목록(폴더, seed, 두 기록 파일의 SHA256)
      - `monitor_spec.py`·`monitor_replay.py`·명세 JSON의 SHA256
      - 기준표, 기록 부족 목록, 평가, 재현도
    - `report.md`에는 사람이 읽을 요약표를 쓴다.

- [ ] **Step 1: 실패하는 테스트 작성**
  - `test_evaluate_counts_problem_and_success`: 시간 초과 1개(감지됨), 완료 2개(1개 오탐) → `problem.detected == 1`, `success.false_checks == 1`
  - `test_fidelity_reproduces_recorded_event`: 합성 정체 구간과 그에 맞는 기록 이벤트 → `reproduced == recorded == 1`
  - `test_overlapping_runs_rejected`, `test_shared_seed_rejected`, `test_existing_out_dir_rejected`: `main([...]) == 2`이고 출력 폴더를 만들지 않는다
  - `test_main_writes_report`: 개발 기록 3개와 평가 기록 1개 → 종료 코드 0. `report.json`의 `inputs.fit`에 seed와 SHA256이 있고, `source`에 세 파일 해시가 있다

- [ ] **Step 2: 실패 확인** — `WSLTEST test_pipeline_monitor_replay` → 새 테스트 실패

- [ ] **Step 3: 구현** — 스크립트는 `sys.path`에 `src`를 넣는 기존 `scripts/*.py` 방식을 따르고, 공용 로직은 `monitor_replay.py`에 둔다.

- [ ] **Step 4: 통과 확인** — `WSLTEST test_pipeline_monitor_replay` → OK

- [ ] **Step 5: 커밋** — "Add monitor spec replay report and CLI"

### Task 5: 실제 기록 재생과 결과 기록

**Files:**
- Create: `docs/monitor-spec-replay-2026-11-05.md`
- Modify: `docs/progress.md`, `docs/specs/2026-11-05-goal-monitor-spec-plan.md` (§10 M1·M2 상태 한 줄씩)

- [ ] **Step 1: 실행** (WSL steve1 환경)
  - `--fit`: `results/llm-loop-20261104/*/`, `results/pit-flow-20261105/*/`, `results/dev-checks-20261105/*/`, `results/ceiling-knowledge-20261105/*/` (seed 130363, 18회)
  - `--eval`: `results/iron-demo-eval-20261105*/*/` (seed 104729, 25회)
  - `--out`: `results/monitor-spec-replay-20261105`

  Expected: 종료 코드 0, `report.json`과 `report.md`가 생긴다. 수치는 미리 정하지 않는다. 평가 결과를 보고 명세나 기준을 고쳐 다시 돌리지 않는다. 고친다면 새 명세 버전과 새 출력 폴더로 하고, 그 사실을 기록한다.

- [ ] **Step 2: 결과 문서** — `docs/monitor-spec-replay-2026-11-05.md`에 다음을 적는다.
  - margin별 감지 수와 오탐 수
  - 감지된 문제 단계에서 아낄 수 있었던 시간
  - 스킬별 표, 기록 부족 목록
  - 재현도(기록된 정체 이벤트 중 다시 계산해 나온 비율)
  - 한계: 사후 분석이고, 점검 뒤 무엇을 했으면 성공했을지는 알 수 없다. 개발 기록 18회뿐이다.

- [ ] **Step 3: 전체 파이프라인 검사** — `WSLTEST`로 `tests/test_pipeline_*.py` 모듈 전체와 `test_stall_detection`을 돌린다 → OK

- [ ] **Step 4: 커밋** — progress.md와 계획서 §10에 M1·M2 상태를 적고 커밋한다. 메시지: "Record monitor spec replay on Nov 5 records".
  푸시와 `main` 병합은 사용자에게 묻는다.
