# 대상 쪽으로 돌아서기 학습 — 구현 계획 (2026-11-07)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> 이 프로젝트는 AGENTS.md에 따라 서브에이전트를 자동으로 쓰지 않는다. 사용자가 따로 말하지 않으면 executing-plans(주 흐름에서 순차 실행)로 한다.

**Goal:** 철광석 단계에서 손 닿는 거리의 철광석 쪽으로 돌아서 조준하는 행동을, 고정된 STEVE-1 위의 작은 residual에 DAgger+BC로 학습시키고 실제 게임에서 공식 STEVE-1과 비교한다.

**Architecture:** 기록 재생용 단계 판정(`stage_replay`)에서 상태 기계를 떼어 내 실시간과 자료 구축이 같은 함수를 쓰게 한다. 게임 중에는 STEVE-1 행동 head를 감싼 `ReacquireHead`가 단계 입력을 받고 정책 잠재 벡터를 기록한다. 교사는 기록이 끝난 뒤 특권 좌표로 정답을 붙이고, 학습기는 residual만 갱신한다. 배치 실행기는 라운드 수집·개발 비교·최종 평가를 같은 probe 장면에서 돌린다.

**Tech Stack:** Python 3.10, PyTorch(WSL `.runtime/steve1`), 공식 STEVE-1/VPT 코드(`external/steve1`), MineRL 1.16.5 + 프로젝트 Java 관측 패치, unittest.

**Spec:** [docs/specs/2026-11-07-target-reacquire-design.md](../../specs/2026-11-07-target-reacquire-design.md)

## Global Constraints

- 성공은 서버 인벤토리(`iron_ore` +1)로만 판정한다. 화면·모델 출력·loss로 성공을 말하지 않는다.
- 공식 STEVE-1 가중치(영상 인코더, transformer, 행동 head)는 고정한다. 학습하는 것은 `ReacquireHead`의 adapter뿐이다.
- 교사는 게임 중 행동하지 않는다. 정답은 기록이 끝난 뒤 붙인다. 평가 실행에서 규칙·교사 행동 0.
- seed는 고정한다. 학습 전용 world seed `271828, 314159, 161803`. 개발 비교 `130363`. 최종 평가 `577215`(새 seed). `104729`는 쓰지 않는다.
  자료 구축기는 학습 전용 seed가 아닌 실행을 거부한다.
- 장면은 2026-11-04 철광석 probe와 같다: `configs/iron-demo-fixture-v2.json` + `give: stone_pickaxe 1`, 계획은 철광석 한 단계,
  `limit_s 180`, `policy_tool_lock true`, oversight 없음, `--frame-every 2`.
- 기존 파이프라인 기본 동작과 `configs/iron-demo.json`은 바꾸지 않는다. 새 동작은 새 설정 `configs/reacquire-v1.json`의 새 skill로만 켠다.
- STOP 계약: STOP 뒤 새 행동 0, 입력 해제(`release`=`neutral_command_sent`), 소유 PID 잔존 0을 매 실행 확인한다.
- 특권 관측(서버 좌표, fixture 좌표, 단계 입력, 교사)은 보고서에 밝힌다. 단계 입력 있는 학생을 영상 전용 결과라고 쓰지 않는다.
- 저장소 파일은 LF다(`.gitattributes` `* -text`). 파일은 Write/Edit로 고치고, host Python으로 쓸 때는 `write_bytes`나 `newline='\n'`.
- 유료 API 0, GitHub 공개 0. 커밋은 계획의 커밋 단계에서만 하고 push하지 않는다.
- 브랜치: `skill-stages-replay`의 HEAD에서 새 브랜치 `target-reacquire`를 만들어 작업한다(`stage_replay`에 의존). main 병합은 사용자 결정이다.
- `results/STOP`을 건드리지 않는다. 배치 STOP 파일은 배치 폴더 안의 `STOP`이다.

테스트 명령(이하 `WSLTEST <module>`):
`wsl.exe -d Ubuntu-24.04 -- env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/mnt/d/DATA/Desktop/MinecraftRL/src:/mnt/d/DATA/Desktop/MinecraftRL/tests /mnt/d/DATA/Desktop/MinecraftRL/.runtime/bin/micromamba run -p /mnt/d/DATA/Desktop/MinecraftRL/.runtime/steve1 python -m unittest <module> -v`

게임·GPU 명령(이하 `WSLGAME <args>`)은 2026-11-06 STEVE-1 단독 실행과 같은 환경 변수를 쓴다:
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<R>/src LD_LIBRARY_PATH=/usr/lib/wsl/lib GALLIUM_DRIVER=d3d12 XDG_CACHE_HOME=<R>/.cache`, `OPENAI_*` unset,
`<R>/.runtime/bin/micromamba run -p <R>/.runtime/steve1 python <args>`. `<R>`=`/mnt/d/DATA/Desktop/MinecraftRL`.

## Review Focus

1. **잠재 벡터와 상태의 한 칸 어긋남.** 정책 스텝 k의 잠재 벡터는 k번째 행동을 고를 때 계산된다. 그때 정책이 본 상태는 telemetry의 직전 행(k-1)이다. 정답은 k-1행 상태로 붙여야 한다(과업 5 테스트).
2. **단계의 첫 결정.** 단계 첫 행동의 직전 행은 준비(prepare) 행이다. 실시간 단계와 자료 구축의 단계가 이 경우에도 같아야 한다(과업 4·5 테스트).
3. **중간에 끝난 실행.** 시간 제한·STOP으로 끝나도 잠재 벡터 마지막 묶음이 저장돼야 하고, 기록 수가 정책 행 수와 다르면 자료 구축이 그 실행을 거부해야 한다(과업 4·5 테스트).
4. **섞인 분포의 정규화.** `(1+s)·조건 − s·무조건`은 정규화된 log 확률이 아니다. NLL과 KL은 섞은 뒤 log_softmax를 다시 해야 한다(과업 6 테스트).
5. **금지 seed.** 개발·평가·최종 seed 실행이 학습 자료에 들어가면 안 된다(과업 5 테스트).

---

### Task 1: 단계 상태 기계를 실시간과 재생이 공유하게 떼어 내기

**Files:**
- Modify: `src/minecraftrl/pipeline/stage_replay.py` (`_stage`, `step_stages`)
- Test: `tests/test_pipeline_stage_replay.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class StageView: stage: str; distance: float | None; nearest: tuple | None; eye: tuple`
  - `class StageTracker: __init__(self, geometry: Geometry, wanted: str, target_item: str, start_t: float)`;
    `apply_destroys(self, events: list[tuple[float, tuple, str]], s: Sample) -> list` (적용한 대상 계열 파괴만 반환);
    `update(self, s: Sample) -> StageView` (유효한 정책 표본마다 시간 순서로 한 번);
    `peek(self, s: Sample) -> StageView` (closing 이력에 넣지 않고 판정만);
    읽기 속성 `recent: deque[tuple[int, float | None]]`(최근 (tick, 거리) 이력).
  - `step_stages`는 동작을 바꾸지 않고 `StageTracker`로 다시 쓴다. 이미 있는 반환 dict는 그대로다.

- [ ] **Step 1: 실패하는 테스트 작성**

`StepStagesTest`에 추가한다. 행은 기존 `test_rows_split_into_stages_by_ticks`의 8행을 helper `walk_rows()`로 꺼내 함께 쓴다.

```python
def tracked(self, rows):
    run = load_run(write_run(self.root / 'R1', [row(1, controller='prepare')] + rows, timeline(), self.fixture))
    step = run.steps[0]
    tracker = StageTracker(Geometry(fixture_blocks(json.loads(self.fixture.read_text()), (0, 64, 0), 0.)),
                           'iron_ore', 'iron_ore', step.start_t)
    pending, out = deque(destroy_events(run.rows)), []
    for r in policy_rows(run, step):
        s = sample_from_row(r)
        due = []
        while pending and pending[0][0] <= r['t']: due.append(pending.popleft())
        tracker.apply_destroys(due, s)
        out.append(tracker.update(s).stage)
    return out

def test_tracker_stages_match_step_stages_row_by_row(self):
    self.assertEqual(self.tracked(walk_rows()),
                     ['far', 'far', 'closing', 'near', 'aimed', 'mining', 'collect', 'far'])

def test_peek_does_not_change_closing_history(self):
    rows = [sample_from_row(r) for r in walk_rows()]
    tracker = StageTracker(Geometry(fixture_blocks(json.loads(self.fixture.read_text()), (0, 64, 0), 0.)),
                           'iron_ore', 'iron_ore', 0.)
    tracker.update(rows[0]); tracker.update(rows[1])
    before = list(tracker.recent)                        # (tick, distance) 이력, 다가감 판정용
    self.assertEqual(tracker.peek(rows[2]).stage, 'closing')
    self.assertEqual(list(tracker.recent), before)
    self.assertEqual(tracker.update(rows[2]).stage, 'closing')
```
(마지막 행은 대상이 모두 부서져 남은 대상이 없으므로 '먼 곳'이다. 기존 테스트의 초 합계와 같은 판정이다.)

- [ ] **Step 2: 실패 확인**

Run: `WSLTEST test_pipeline_stage_replay`
Expected: FAIL — `NameError: StageTracker` (기존 19개는 통과).

- [ ] **Step 3: `StageTracker`, `StageView` 구현, `step_stages`를 그 위로 다시 쓰기**

`step_stages`의 행 루프 안에 있던 geometry 갱신(파괴 제거, 조준으로 확인된 놓인 블록 추가), 최근 거리 이력(CLOSING_TICKS), collect 상태, `_stage` 호출을 tracker로 옮긴다. 시간 누적·near 조준선·각도·구덩이·부순 높이 집계는 `step_stages`에 남긴다.

- [ ] **Step 4: 통과 확인**

Run: `WSLTEST test_pipeline_stage_replay`
Expected: PASS 21/21 (기존 19 + 새 2).

- [ ] **Step 5: 기존 보고서 재현 확인**

Run(WSL): `python scripts/replay_skill_stages.py --runs <r4 eval 입력 목록> --out results/skill-stages-20261107-r5-check/eval --root <R>` 후
`results/skill-stages-20261107-r4/eval/report.json`과 `steps`, `groups`, `geometry`를 비교한다(입력 목록은 r4 `report.json`의 `inputs[].dir`).
Expected: 세 키가 모두 같다. 다르면 리팩터링이 동작을 바꾼 것이다.

- [ ] **Step 6: 커밋**

```bash
git add docs/specs/2026-11-07-target-reacquire-design.md docs/superpowers/plans/2026-11-07-target-reacquire.md
git commit -m "Add the turn-to-target design and implementation plan"
git add src/minecraftrl/pipeline/stage_replay.py tests/test_pipeline_stage_replay.py
git commit -m "Share the stage state machine between replay and live use"
```

### Task 2: 교사 정답

**Files:**
- Create: `src/minecraftrl/reacquire_teacher.py`
- Test: `tests/test_reacquire_teacher.py`

**Interfaces:**
- Consumes: `StageView`(과업 1), `pipeline.rm.Sample`, `pipeline.steps.block_matches`.
- Produces:
  - `LABELLED_STAGES = ('near', 'aimed', 'mining')`, `MAX_TURN = 10.`
  - `aim_delta(eye: tuple, yaw: float, pitch: float, pos: tuple) -> tuple[float, float]` — 블록 중심을 향하는 (dpitch, dyaw), 도 단위. yaw 차이는 (-180, 180]로 감는다. 규약은 `stage_replay.view_angle`과 같다(yaw 0 = +z, 90 = −x, pitch 아래가 +). camera[0]은 pitch에, camera[1]은 yaw에 더해진다.
  - `in_view(dpitch: float, dyaw: float) -> bool` — `abs(dyaw) <= 50 and abs(dpitch) <= 35` (진단 전용, 내 결정: 16:9 화면 시야의 반각 근사).
  - `teacher_env_action(s: Sample, view: StageView, wanted: str) -> dict | None` — 정답이 없는 행은 None.
  - `policy_label(env_action: dict, mapper, transformer) -> dict` — 공식 `env2policy` → `from_factored`, 왕복 불일치면 `ValueError('label failed official mapping roundtrip')`. 반환: `buttons`, `camera`(정수 index), `camera_on`(bool: 왕복 복원된 env camera가 0이 아님), `decoded_camera`.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
EYE = (0.5, 1.5, 0.5)
def test_aim_delta_straight_ahead_is_zero(self):
    self.assertEqual(tuple(round(v, 6) for v in aim_delta(EYE, 0., 0., (0, 1, 3))), (0., 0.))
def test_aim_delta_target_to_minus_x_turns_right_90(self):
    self.assertAlmostEqual(aim_delta(EYE, 0., 0., (-3, 1, 0))[1], 90., places=6)
def test_aim_delta_wraps_yaw(self):
    self.assertAlmostEqual(aim_delta(EYE, 350., 0., (0, 1, 3))[1], 10., places=6)
def test_aim_delta_behind_is_plus_180(self):
    self.assertAlmostEqual(aim_delta(EYE, 0., 0., (0, 1, -3))[1], 180., places=6)
def test_aim_delta_block_below_looks_down(self):
    self.assertAlmostEqual(aim_delta(EYE, 0., 0., (0, 0, 1))[0], 45., places=6)
def test_near_not_aimed_turns_without_attack(self):
    a = teacher_env_action(sample(hit='grass_block'), StageView('near', 3., (0, 1, -3), EYE), 'iron_ore')
    self.assertEqual(list(a['camera']), [0., 10.]); self.assertEqual(a['attack'], 0)
    self.assertEqual(sum(a[k] for k in ('forward', 'back', 'left', 'right', 'jump')), 0)
def test_aimed_attacks_and_keeps_aim_on_the_hit_block(self):
    a = teacher_env_action(sample(hit='iron_ore', hit_pos=(0, 1, 3)), StageView('aimed', 3., (1, 1, 3), EYE), 'iron_ore')
    self.assertEqual(a['attack'], 1); self.assertEqual([round(v, 6) for v in a['camera']], [0., 0.])
def test_far_gui_and_invalid_rows_have_no_label(self):
    self.assertIsNone(teacher_env_action(sample(), StageView('far', 9., (0, 1, 9), EYE), 'iron_ore'))
    self.assertIsNone(teacher_env_action(sample(gui=True), StageView('near', 3., (0, 1, 3), EYE), 'iron_ore'))
    self.assertIsNone(teacher_env_action(sample(valid=False), StageView('near', 3., (0, 1, 3), EYE), 'iron_ore'))
@skip_without_steve1
def test_policy_label_roundtrips_turn_and_attack(self):
    mapper, transformer = official_mapping()
    label = policy_label(dict(env_noop(), camera=np.array([0., 10.]), attack=1), mapper, transformer)
    self.assertTrue(label['camera_on']); self.assertAlmostEqual(float(label['decoded_camera'][1]), 10., delta=1.)
```
(`sample(...)`은 `pipeline.rm.Sample`을 만드는 테스트 helper다. 눈 위치는 `position=(0.5, 1.5-1.62, 0.5)`로 맞춘다. `official_mapping`, `env_noop`, `skip_without_steve1`은 `tests/test_navigation_supervision.py`가 쓰는 방식을 따른다.)

- [ ] **Step 2: 실패 확인** — Run: `WSLTEST test_reacquire_teacher` / Expected: FAIL `ModuleNotFoundError: minecraftrl.reacquire_teacher`.

- [ ] **Step 3: 구현**

대상: `aimed`·`mining`이고 조준선(채굴이면 채굴 블록)이 대상 계열이면 그 위치, 아니면 `view.nearest`. camera는 `aim_delta`를 각 축 ±`MAX_TURN`으로 자른 값. attack은 `block_matches(s.hit_block, wanted)`. 나머지 버튼은 `steve1.VPT.lib.actions.Buttons.ALL` 전부 0(환경 noop과 같은 키 집합).

- [ ] **Step 4: 통과 확인** — Run: `WSLTEST test_reacquire_teacher` / Expected: PASS 9/9.

- [ ] **Step 5: 커밋** — `git add src/minecraftrl/reacquire_teacher.py tests/test_reacquire_teacher.py` → `git commit -m "Add the privileged turn-to-target teacher labels"`

### Task 3: 학생 head

**Files:**
- Create: `src/minecraftrl/reacquire_head.py`
- Test: `tests/test_reacquire_head.py`

**Interfaces:**
- Consumes: `stage_replay.STAGES`.
- Produces:
  - `STAGE_DIM = len(STAGES)` (7), `CHECKPOINT_SCHEMA = 'reacquire_head_v1'`
  - `class ReacquireHead(torch.nn.Module): __init__(self, base, width: int, *, use_stage: bool, hidden: int = 64)`
    - 속성 `stage: int | None` — 호출자가 매 행동 전에 넣는 `STAGES` index. None 또는 `use_stage=False`면 0 벡터.
    - 속성 `last_latent: torch.Tensor | None` — 마지막 forward의 `pi_h`(detach, CPU 복사 아님), 모양 `[B, T, width]`. B=2면 [조건, 무조건].
    - `forward(self, pi_h, mask=None)` → `base(pi_h + delta, mask=mask)`; `delta = adapter(cat(pi_h.float(), onehot)).to(pi_h.dtype)`.
    - `adapter = Sequential(Linear(width+7, 64), Tanh(), Linear(64, width))`, 마지막 층 0 초기화, 초기화는 `torch.random.fork_rng`와 seed 110201(실행 sampling RNG를 소비하지 않음).
    - `sample`, `logprob`, `entropy`, `kl_divergence`는 `base`에 넘긴다.
  - `install(policy, *, use_stage: bool, checkpoint: Path | None = None) -> ReacquireHead` — `policy.agent.policy.pi_head`를 감싸 바꾸고, base는 `requires_grad_(False)`.
  - `save_checkpoint(head, path, *, metadata: dict) -> str`(sha256 반환), checkpoint는 `schema`, `use_stage`, `stages`, `width`, `adapter` state_dict, `base_weights_sha256`, `metadata`.
    `install`은 `use_stage`, `stages`, `width`, `base_weights_sha256`가 다르면 `ValueError('incompatible reacquire checkpoint')`.

- [ ] **Step 1: 실패하는 테스트 작성** (가짜 base: `forward(x, mask=None)`가 Linear인 모듈, `sample`/`logprob` 있음)

```python
def test_zero_initialized_head_returns_the_base_output_exactly(self):
    head = ReacquireHead(base, 8, use_stage=True); head.stage = 2
    x = torch.randn(2, 1, 8); self.assertTrue(torch.equal(head(x), base(x)))
def test_stage_is_ignored_without_stage_input(self):
    head = trained_head(use_stage=False)   # adapter 마지막 층에 0 아닌 값을 넣은 head
    head.stage = 0; a = head(x); head.stage = 4
    self.assertTrue(torch.equal(a, head(x)))
def test_stage_changes_output_with_stage_input(self):
    head = trained_head(use_stage=True)
    head.stage = 0; a = head(x); head.stage = 4
    self.assertFalse(torch.equal(a, head(x)))
def test_last_latent_is_the_input_latent(self):
    head = ReacquireHead(base, 8, use_stage=True); head(x)
    self.assertTrue(torch.equal(head.last_latent, x))
def test_checkpoint_roundtrip_and_mismatch(self):
    sha = save_checkpoint(trained, path, metadata={})
    again = load_into(ReacquireHead(base, 8, use_stage=True), path, base_weights_sha256='b')
    self.assertTrue(torch.equal(again(x), trained(x)))
    with self.assertRaises(ValueError): load_into(ReacquireHead(base, 8, use_stage=False), path, base_weights_sha256='b')
def test_initialization_does_not_consume_the_global_rng(self):
    torch.manual_seed(5); a = torch.rand(1); torch.manual_seed(5); ReacquireHead(base, 8, use_stage=True)
    self.assertTrue(torch.equal(torch.rand(1), a))
```
(`load_into(head, path, *, base_weights_sha256)`는 `install`이 쓰는 내부 함수로 같은 파일에 둔다.)

- [ ] **Step 2: 실패 확인** — Run: `WSLTEST test_reacquire_head` / Expected: FAIL `ModuleNotFoundError`.
- [ ] **Step 3: 구현** — 위 Interfaces 그대로.
- [ ] **Step 4: 통과 확인** — Run: `WSLTEST test_reacquire_head` / Expected: PASS 6/6.
- [ ] **Step 5: 실제 STEVE-1 출력 동일성 확인(GPU, 게임 없음)**

`scripts/check_reacquire_head.py`를 만든다: `Steve1Policy`를 GPU로 만들고 기록된 프레임 64장(`results/iron-demo-probe-20261104-arena-v2/iron_ore-mine_iron_text/trial-2/frames/`의 앞 64장)과 `mine_iron_ore.npy`로, head 없는 정책과 `install(use_stage=True)` 정책의 `pd`(guidance 6)를 같은 hidden state로 비교한다.
Run: `WSLGAME scripts/check_reacquire_head.py --root <R>` / Expected: 모든 key의 최대 절대 차이 0.0, 출력에 `identical: true`.

- [ ] **Step 6: 커밋** — `git add src/minecraftrl/reacquire_head.py scripts/check_reacquire_head.py tests/test_reacquire_head.py` → `git commit -m "Add the zero-initialized stage-conditioned residual head"`

### Task 4: 게임 중 단계 입력과 잠재 벡터 기록

**Files:**
- Create: `src/minecraftrl/pipeline/stage_live.py`, `configs/reacquire-v1.json`
- Modify: `src/minecraftrl/pipeline/session.py` (`dispatch`, `run_policy`, `run` 종료부), `src/minecraftrl/pipeline/cli.py` (skill의 `reacquire` 블록으로 head 설치, fixture JSON을 session에 전달)
- Test: `tests/test_pipeline_stage_live.py`, `tests/test_pipeline_session.py`(추가)

**Interfaces:**
- Consumes: `StageTracker`, `StageView`, `fixture_blocks`, `Geometry`(과업 1), `ReacquireHead`, `install`(과업 3).
- Produces:
  - `class LiveStageFeed: __init__(self, fixture: dict, wanted: str, target_item: str)`
    - `observe(self, row: dict) -> None` — 기록되는 telemetry 행(dict, `t`·`step`·`controller` 포함)을 **모든 dispatch마다** 넣는다. fixture 원점은 `fixture_origin`과 같은 규칙으로 첫 적용 행에서 잡고, 파괴 이벤트는 `destroy_events`와 같은 규칙(유효 행 사이 파괴 수 증가)으로 만든다.
    - `start_step(self, start_t: float) -> None` — 정책 단계 시작 때 새 `StageTracker`.
    - `decision_view(self) -> StageView | None` — 다음 정책 행동이 볼 단계. 마지막 행이 정책 행이면 그 행의 `update` 결과, 단계 첫 결정이면 마지막 행의 `peek`. 원점이 아직 없으면 None.
    - 정책 행은 `observe` 안에서 `update`된다(행의 `controller == 'policy'`).
  - `class LatentLog: __init__(self, folder: Path, *, width: int, cond_scale: float, meta: dict, chunk: int = 256)`;
    `add(self, step: int, latent: torch.Tensor, stage: int | None) -> None`(latent `[2,1,width]` → fp16 `cond`, `uncond`);
    `close(self) -> dict` — 남은 행을 쓰고 `manifest.json`(width, cond_scale, chunk 파일명과 sha256, 행 수, meta) 작성.
    파일: `<output>/reacquire/latents-000.npz`… (키 `step` int32, `cond` float16 `[n,width]`, `uncond` float16, `stage` int8, −1은 None).
  - 설정: `configs/reacquire-v1.json` = `configs/iron-demo.json`과 같은 필드 + skill `mine_iron_reacquire`(임베딩·sha·cond_scale은 `mine_iron_text`와 같음) +
    `"reacquire": {"checkpoint": null, "use_stage": true, "record_latents": true}`. 배치 실행기(과업 7)가 arm마다 이 블록을 바꾼 사본을 만든다.
  - session: skill에 `reacquire`가 있으면 `run_policy`가 매 `policy.act` 전에 `head.stage = STAGES.index(view.stage)`(view가 None이면 None), 행동 뒤 `LatentLog.add(self.steps + 1, head.last_latent, head.stage)`. `run`의 `finally`에서 `LatentLog.close()`. 기록된 단계는 `stage` 행렬로만 남기고 telemetry 형식은 바꾸지 않는다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_feed_matches_replay_stages_on_a_recorded_style_run(self):
    rows = synthetic_rows_prepare_then_policy()   # 준비 행 20개 + 정책 행(먼 곳→…→회수), fixture 적용 표시 포함
    feed = LiveStageFeed(FIXTURE, 'iron_ore', 'iron_ore'); views = []
    for i, r in enumerate(rows):
        if r['controller'] == 'policy' and rows[i - 1]['controller'] != 'policy': feed.start_step(r['t'] - 1e-6)
        if r['controller'] == 'policy': views.append(feed.decision_view().stage)
        feed.observe(r)
    replay = [v.stage for _, v in replay_views(rows)]          # 과업 1의 tracker로 정책 행만 재생
    self.assertEqual(views[1:], replay[:-1])                   # 결정 k는 행 k-1의 단계
def test_first_decision_uses_peek_of_the_prepare_row(self):
    ...  # 첫 결정의 단계 = 마지막 준비 행을 peek한 단계, tracker closing 이력은 비어 있음
def test_latent_log_flushes_partial_chunk_on_close(self):
    log = LatentLog(tmp, width=4, cond_scale=6., meta={}, chunk=3)
    for k in range(1, 6): log.add(k, torch.ones(2, 1, 4) * k, 2)
    manifest = log.close()
    self.assertEqual(manifest['rows'], 5); self.assertEqual(len(manifest['chunks']), 2)
def test_session_records_one_latent_per_policy_action_and_closes_on_time_limit(self):
    ...  # tests/test_pipeline_session.py의 fake env/policy로: 정책 행 수 == 기록 행 수, time_limit으로 끝나도 manifest 존재
def test_session_without_reacquire_block_is_unchanged(self):
    ...  # 기존 skill로 돌리면 reacquire 폴더가 없고 telemetry 행이 이전과 같다
```

- [ ] **Step 2: 실패 확인** — Run: `WSLTEST test_pipeline_stage_live test_pipeline_session` / Expected: 새 테스트 FAIL, 기존 session 테스트 PASS.
- [ ] **Step 3: 구현** — 위 Interfaces 그대로. cli는 skill에 `reacquire`가 있을 때만 `install`을 부르고, 설치한 head의 checkpoint sha256·use_stage를 `summary.json`의 `reacquire` 항목에 남긴다.
- [ ] **Step 4: 통과 확인** — Run: `WSLTEST test_pipeline_stage_live test_pipeline_session` / Expected: PASS(전체).
- [ ] **Step 5: 커밋** — `git add src/minecraftrl/pipeline/stage_live.py src/minecraftrl/pipeline/session.py src/minecraftrl/pipeline/cli.py configs/reacquire-v1.json tests/test_pipeline_stage_live.py tests/test_pipeline_session.py` → `git commit -m "Feed live stages to the reacquire head and record policy latents"`

### Task 5: 학습 자료 구축

**Files:**
- Create: `src/minecraftrl/reacquire_data.py`, `scripts/build_reacquire_dataset.py`
- Test: `tests/test_reacquire_data.py`

**Interfaces:**
- Consumes: `LiveStageFeed`(과업 4), `teacher_env_action`, `policy_label`, `in_view`, `aim_delta`(과업 2), `monitor_replay.load_run`.
- Produces:
  - `TRAIN_SEEDS = (271828, 314159, 161803)`, `FORBIDDEN_SEEDS = (130363, 104729, 577215)`
  - `run_examples(run_dir: Path, mapper, transformer) -> dict` — 텐서: `cond`, `uncond` `[n,width]` fp16, `stage` int8, `labelled` bool,
    `label_buttons`, `label_camera` int64(정답 없는 행은 −1), `camera_on` bool, `in_view` bool, `step` int32, `cond_scale` float.
    - 정책 스텝 k의 잠재 벡터에는 telemetry에서 k보다 앞선 마지막 행(=정책이 본 상태)의 단계·정답을 붙인다. 이를 위해 `LiveStageFeed`에 기록 행을 순서대로 넣고 각 정책 행 직전에 `decision_view()`를 쓴다(실시간과 같은 함수).
    - 거부(`ValueError`): seed가 `TRAIN_SEEDS` 밖, manifest 없음, chunk sha256 불일치, 기록 행 수 ≠ 정책 행 수, 기록된 단계 ≠ 다시 계산한 단계(`'live stage differs from replay'`), `summary.json`의 `execution_valid`가 false.
  - `build_dataset(run_dirs, out_path, *, holdout: str, mapper, transformer) -> dict` — 실행별 예제를 합쳐 `out_path`(torch.save)에 저장. 요약: 실행별 행·정답 행, 단계별 정답 수, 화면 안/밖 정답 수, chunk sha256, holdout 실행 이름. 요약은 `<out>.summary.json`.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_label_uses_the_state_before_the_action(self):
    run = make_run(rows=[prepare(), policy(step=1, stage_state='far'), policy(step=2, stage_state='near_facing_away')],
                   latents=[1, 2], recorded_stages=['far', 'far'])
    ex = run_examples(run, *mapping())
    self.assertEqual(ex['stage'].tolist(), [STAGES.index('far'), STAGES.index('far')])  # 결정 2는 행 1(먼 곳)을 봤다
    self.assertEqual(ex['labelled'].tolist(), [False, False])
def test_near_row_gets_a_turn_label_for_the_next_latent(self):
    ...  # 행 2가 near(등 뒤 대상)면 latent 3에 camera yaw +10, attack 0 정답
def test_forbidden_and_unknown_seeds_are_refused(self):
    for seed in (130363, 104729, 577215, 12345):
        with self.assertRaises(ValueError): run_examples(make_run(seed=seed), *mapping())
def test_latent_count_must_match_policy_rows(self):
    with self.assertRaises(ValueError): run_examples(make_run(latents=[1]), *mapping())
def test_recorded_stage_must_match_recomputed_stage(self):
    with self.assertRaisesRegex(ValueError, 'live stage differs'): run_examples(make_run(recorded_stages=['near', 'far']), *mapping())
def test_holdout_run_is_kept_apart(self):
    summary = build_dataset([run_a, run_b], out, holdout=run_b.name, mapper=..., transformer=...)
    self.assertEqual(summary['holdout'], run_b.name)
```
(`mapping()`은 steve1이 없으면 skip하는 공식 mapper/transformer. 합성 실행 helper `make_run`은 이 테스트 파일에 둔다.)

- [ ] **Step 2: 실패 확인** — Run: `WSLTEST test_reacquire_data` / Expected: FAIL `ModuleNotFoundError`.
- [ ] **Step 3: 구현** — 위 Interfaces 그대로. CLI: `build_reacquire_dataset.py --runs DIR... --holdout NAME --out PATH --root R`.
- [ ] **Step 4: 통과 확인** — Run: `WSLTEST test_reacquire_data` / Expected: PASS 6/6.
- [ ] **Step 5: 커밋** — `git add src/minecraftrl/reacquire_data.py scripts/build_reacquire_dataset.py tests/test_reacquire_data.py` → `git commit -m "Build reacquire training examples from recorded latents and teacher labels"`

### Task 6: 학습기

**Files:**
- Create: `src/minecraftrl/reacquire_training.py`, `scripts/train_reacquire.py`
- Test: `tests/test_reacquire_training.py`

**Interfaces:**
- Consumes: 자료 파일(과업 5), `ReacquireHead`, `save_checkpoint`(과업 3).
- Produces:
  - `guided_logp(head, cond, uncond, stage, cond_scale) -> dict[str, Tensor]` — 조건·무조건 잠재 벡터 모두에 같은 단계로 head를 적용하고, key(`buttons`, `camera`)마다 `(1+s)·조건 − s·무조건`을 계산한 뒤 **log_softmax를 다시** 한다.
  - `bc_nll(logp, label_buttons, label_camera, camera_on) -> Tensor` — 행마다 buttons NLL, camera NLL은 `camera_on`인 행만.
  - `kl_to_official(logp_student, logp_official) -> Tensor` — 행마다 key 합 KL(student‖official).
  - `train(dataset_path, *, use_stage, out_dir, lr=1e-4, batch=32, epochs=16, kl_coef=.01, kl_limit=3., clip=1., seed=17, head_factory=None) -> dict`
    (`head_factory=None`이면 실제 STEVE-1 행동 head를 불러 `ReacquireHead`로 감싼다. 테스트는 가짜 head를 넘긴다.)
    - 손실 = 정답 행 평균 NLL + `kl_coef` × 정답 없는 행 평균 KL. AdamW, gradient clip 1. holdout 실행은 학습에 쓰지 않는다.
    - epoch마다 holdout의 정답 NLL과 전체 행 평균 KL을 잰다. 평균 KL이 `kl_limit`를 넘은 epoch는 되돌리고 학습을 끝낸다. 남은 마지막 epoch를 후보로 둔다.
    - 저장 전 검사: 학습 전 KL 0(0 초기화 확인), 저장/재로드 출력 차이 0, holdout NLL 감소. 감소하지 않으면 `candidate=false`로 보고만 하고 checkpoint 이름에 `-rejected`를 붙인다.
    - 보고서 `train-report.json`: epoch별 train/holdout NLL·KL, 되돌림, 갱신 수, 학습 초, peak 메모리, 자료 sha256, 소스 sha256(`reacquire_*.py`, `stage_replay.py`, `stage_live.py`), base 가중치 sha256.
  - CLI: `train_reacquire.py --dataset PATH --use-stage on|off --out DIR --root R`(STEVE-1 행동 head만 GPU에 올림).

- [ ] **Step 1: 실패하는 테스트 작성** (가짜 base head: key 두 개의 Linear logits)

```python
def test_guided_logp_is_normalized(self):
    logp = guided_logp(head, cond, uncond, stage, 6.)
    for v in logp.values(): self.assertTrue(torch.allclose(v.exp().sum(-1), torch.ones(v.shape[:-1]), atol=1e-5))
def test_uniform_guided_distribution_has_log_n_nll(self):
    logp = {'buttons': torch.full((1, 4), -math.log(4)), 'camera': torch.full((1, 3), -math.log(3))}
    self.assertAlmostEqual(bc_nll(logp, torch.tensor([1]), torch.tensor([0]), torch.tensor([False])).item(), math.log(4), places=6)
def test_zero_head_has_zero_kl_to_official(self):
    self.assertEqual(kl_to_official(guided_logp(zero_head, ...), guided_logp(zero_head, ...)).abs().max().item(), 0.)
def test_training_lowers_holdout_nll_on_a_learnable_toy_set(self):
    report = train(toy_dataset_path, use_stage=True, out_dir=tmp, epochs=4, head_factory=fake_head)
    self.assertLess(report['epochs'][-1]['holdout_nll'], report['initial']['holdout_nll']); self.assertTrue(report['candidate'])
def test_epoch_over_kl_limit_is_rolled_back(self):
    report = train(toy_dataset_path, use_stage=True, out_dir=tmp, epochs=4, lr=1., kl_limit=1e-6, head_factory=fake_head)
    self.assertEqual(report['rolled_back_epochs'], [1]); self.assertEqual(report['updates_kept'], 0)
def test_holdout_rows_are_never_trained_on(self):
    ...  # holdout 행의 잠재 벡터를 NaN으로 바꿔도 학습이 끝나고 train 손실이 유한하다(평가 때만 쓰므로 그 값만 NaN)
```
(`train`은 테스트용 `head_factory` 인자를 받는다. 기본값은 실제 STEVE-1 행동 head를 감싸는 함수다.)

- [ ] **Step 2: 실패 확인** — Run: `WSLTEST test_reacquire_training` / Expected: FAIL `ModuleNotFoundError`.
- [ ] **Step 3: 구현** — 위 Interfaces 그대로.
- [ ] **Step 4: 통과 확인** — Run: `WSLTEST test_reacquire_training` / Expected: PASS 6/6.
- [ ] **Step 5: 커밋** — `git add src/minecraftrl/reacquire_training.py scripts/train_reacquire.py tests/test_reacquire_training.py` → `git commit -m "Train the reacquire residual with guided BC and a KL leash"`

### Task 7: 배치 실행기와 비교

**Files:**
- Create: `scripts/run_reacquire_batch.py`, `src/minecraftrl/reacquire_compare.py`
- Test: `tests/test_reacquire_batch.py`, `tests/test_reacquire_compare.py`

**Interfaces:**
- Consumes: `configs/reacquire-v1.json`(과업 4), `stage_replay.build_report`(과업 1).
- Produces:
  - `run_reacquire_batch.py collect --batch DIR --arm NAME --checkpoint PATH|official --use-stage on|off --seeds S... --sampling N...`
    - arm마다 `DIR/configs/<arm>.json`(reacquire 블록만 바꾼 사본), `DIR/fixtures/iron-probe.json`(fixture v2 + `give stone_pickaxe 1`, 설명에 DEVELOPMENT PROBE VARIANT), `DIR/plans/iron.json`(`id iron, skill mine_iron_reacquire, kind policy, target iron_ore, count 1, block iron_ore`)을 만든다.
    - 실행: `python -m minecraftrl.pipeline.cli probe --root R --output DIR/<arm>/s<seed>-n<sampling> --config ... --plan-json ... --fixture ... --stop-file DIR/STOP --seed S --sampling-seed N --limit-s 180 --frame-every 2`.
    - `run_iron_demo_probes.py`와 같은 보호: 끝난 실행 건너뜀, `DIR/STOP`이면 중단, GPU 여유 2,150 MiB 미만이면 15초 간격 10분 대기 후 중단,
      `execution_valid`가 false면 `DIR/excluded/`로 옮기고 배치 중단. 결과 줄은 `DIR/results.jsonl`.
    - 첫 실행 전 `DIR/frozen-inputs.json`에 입력 sha256(설정·fixture·계획·임베딩·checkpoint·`pipeline/*.py`·`reacquire_*.py`·Java 관측)을 쓰고, 이후 실행마다 같은지 확인한다(다르면 중단).
    - `scripts`끼리 import하지 않는다. 필요한 보호 논리는 이 파일에 둔다.
  - `run_reacquire_batch.py compare --batch DIR --arms A B C --out DIR/compare` → `reacquire_compare.compare(...)`.
  - `reacquire_compare.compare(arm_dirs: dict[str, list[Path]], *, root) -> dict` — arm마다 성공 수/실행 수, 성공 실행 철광석 단계 게임 초(중앙값·목록), 실패 사유, `stage_replay` 단계 초 합(먼 곳·다가감·near·조준·채굴·회수), **돌아서기 비율 = near 초 / 단계 전체 초(모든 실행 합)**, near 시선 각도 분포, STOP 계약(release·잔존 PID).
  - `verdict(official: dict, student: dict) -> dict(better: bool, checks: {success_not_lower, reacquire_half_or_less, median_shorter})`.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_collect_refuses_dev_or_final_seeds_without_the_matching_mode(self):
    ...  # --mode train(기본)에서 seed 130363/577215는 거부, --mode dev는 130363만, --mode final은 577215만 허용
def test_arm_config_only_changes_the_reacquire_block(self):
    ...  # 생성한 arm 설정과 configs/reacquire-v1.json의 차이는 reacquire 블록뿐
def test_finished_trials_are_skipped_and_stop_halts(self): ...
def test_verdict_requires_all_three_checks(self):
    off = dict(success=6, runs=8, reacquire_share=.37, median_s=137.)
    self.assertTrue(verdict(off, dict(success=6, runs=8, reacquire_share=.18, median_s=120.))['better'])
    self.assertFalse(verdict(off, dict(success=5, runs=8, reacquire_share=.10, median_s=90.))['better'])
    self.assertFalse(verdict(off, dict(success=7, runs=8, reacquire_share=.19, median_s=140.))['better'])
def test_reacquire_share_pools_successful_and_failed_runs(self): ...
```

- [ ] **Step 2: 실패 확인** — Run: `WSLTEST test_reacquire_batch test_reacquire_compare` / Expected: FAIL `ModuleNotFoundError`.
- [ ] **Step 3: 구현** — 위 Interfaces 그대로. `collect`에 `--mode train|dev|final`(기본 train): train은 `TRAIN_SEEDS`만, dev는 `130363`만, final은 `577215`만 허용.
- [ ] **Step 4: 통과 확인** — Run: `WSLTEST test_reacquire_batch test_reacquire_compare` / Expected: PASS.
- [ ] **Step 5: 전체 unit lane** — Run: `WSLTEST discover -s tests` / Expected: 실패 0(기존 skip 18 외 새 skip은 steve1 없는 환경용뿐). 결과를 원장에 남긴다.
- [ ] **Step 6: 커밋** — `git add scripts/run_reacquire_batch.py src/minecraftrl/reacquire_compare.py tests/test_reacquire_batch.py tests/test_reacquire_compare.py` → `git commit -m "Add the reacquire batch runner and arm comparison"`

### Task 8: 실험 실행 (게임 54회, 학습 4회)

배치 폴더 `results/reacquire-20261107/`(Git 제외). 각 단계의 명령은 `WSLGAME scripts/run_reacquire_batch.py ...`. 실행 하나는 제어 180초 이하, 시작 포함 약 4–5분.

- [ ] **Step 1: 라운드 0 수집(공식)** — `collect --batch B/r0 --arm official --checkpoint official --use-stage on --seeds 271828 314159 161803 --sampling 1 2`
  Expected: 6개 `summary.json`, 모두 `execution_valid` true, `reacquire/manifest.json` 행 수 = 정책 행 수. 성공 수는 기록만 한다.
- [ ] **Step 2: 자료 r0, 학습 r1(단계 있음)** — `build_reacquire_dataset.py --runs B/r0/official/* --holdout s161803-n2 --out B/data/r0.pt`, `train_reacquire.py --dataset B/data/r0.pt --use-stage on --out B/train/r1-on`
  Expected: 정답 행 수천 개(실행당 수백), `candidate: true`. false면 멈추고 원인을 원장에 쓴다(교사·정렬 의심).
- [ ] **Step 3: 라운드 1 수집** — `collect --batch B/r1 --arm r1-on --checkpoint B/train/r1-on/checkpoint.pt --use-stage on --seeds (같음) --sampling 3 4` / Expected: 6개 유효.
- [ ] **Step 4: 자료 r0–1, 학습 r2** — holdout 같음 / Expected: `candidate: true`.
- [ ] **Step 5: 라운드 2 수집** — `--sampling 5 6`, checkpoint r2 / Expected: 6개 유효.
- [ ] **Step 6: 최종 학습 두 갈래** — 자료 r0–2로 `--use-stage on` → `B/train/final-on`, `--use-stage off` → `B/train/final-off` / Expected: 둘 다 `candidate: true`(아니면 그 갈래는 비교에서 빼고 이유를 쓴다).
- [ ] **Step 7: 개발 비교** — `collect --mode dev --batch B/dev --arm {official|final-on|final-off} ... --seeds 130363 --sampling 11 12 13 14`, 이어서 `compare --batch B/dev --arms official final-on final-off`
  Expected: 12개 유효 실행, `compare/report.json`. 단계 없는 갈래가 공식보다 성공이 적거나 돌아서기 비율이 높으면 최종 평가에서 뺀다(설계 §7).
- [ ] **Step 8: 최종 평가** — `collect --mode final --batch B/final ... --seeds 577215 --sampling 21 22 23 24 25 26 27 28`(arm 2–3개), `compare`
  Expected: arm당 8개 유효 실행, `verdict`. 결과가 어떻든 그대로 보고한다.
- [ ] **Step 9: 기록** — `docs/reacquire-results-2026-11-07.md`(조건, 특권 관측, 라운드별 자료·학습 수치, 개발·최종 표, verdict, 화면 안/밖 정답 일치율, STOP 계약, 한계), `docs/progress.md` 맨 위 항목. 커밋: `git add docs/reacquire-results-2026-11-07.md docs/progress.md` → `git commit -m "Record the reacquire training and comparison results"`.

## 실행하지 않는 것

- 기본 파이프라인·`configs/iron-demo.json` 변경, 원목·돌 적용, 영상 단계 판정, 탐험 확장(설계 §10), 전체 "철 캐줘" 흐름 평가.
- 실패한 후보를 다른 하이퍼파라미터로 바꿔 계속 돌리는 것. 바꾸려면 새 조건으로 원장에 남기고 사용자에게 알린다.
