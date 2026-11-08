# 여러 블록 캐기 학습 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 돌아서기·조준·채굴 residual을 철광석이 아닌 블록 6개로 DAgger 학습하고, 처음 보는 철광석에서 개발 비교한다.

**Architecture:** 기존 돌아서기 파이프라인(잠재 벡터 기록 → 특권 hold 교사 정답 → residual BC+KL)을 그대로 쓴다. 배치 실행기에 `--block`을 더해 경기장의 철광석 자리만 대상 블록으로 바꾸고, 자료 구축에 블록 기록·여러 보류·금지 블록을, 학습에 (블록, 단계) 균형과 최선 epoch 되돌리기를 더한다. 블록별 채굴 시도 집계로 R0 관문과 개발 판정을 계산한다.

**Tech Stack:** Python 3, PyTorch(WSL steve1 env), 기존 `minecraftrl.pipeline` probe, unittest.

**Spec:** `docs/specs/2026-11-08-multi-block-mining-design.md`

## Global Constraints

- 철광석(`iron_ore`) 실행은 학습 자료에 들어가면 안 된다. 자료 구축이 거부해야 한다.
- 학습 world seed는 271828, 314159, 161803만. 개발 130363, 최종 662607(미리 정함), 이미 쓴 104729·577215는 학습 금지.
- 철광석 갈래의 설정·fixture·계획 파일 내용은 앞선 실험(`run_reacquire_batch.py` 기존 동작)과 같아야 한다. 기존 CLI 기본값(`--block` 없음 = 철광석)과 실행 폴더 이름(`s<seed>-n<n>`)을 유지한다.
- 문구 틀은 "mine {블록}", cond_scale 6.0, prior seed 31031.
- 학습 기본값: lr 1e-4, batch 32, 최대 16 epoch, KL 계수 10, KL 한도 3, hold 교사, 단계 입력 있음, patience 2.
- 저장소 파일은 LF. 파일은 Write/Edit로 고친다. 유료 API 0, push 0. `results/STOP`을 건드리지 않는다.
- 테스트: `WSLTEST <module>` = `wsl.exe -d Ubuntu-24.04 -- env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/mnt/d/DATA/Desktop/MinecraftRL/src:/mnt/d/DATA/Desktop/MinecraftRL/tests /mnt/d/DATA/Desktop/MinecraftRL/.runtime/bin/micromamba run -p /mnt/d/DATA/Desktop/MinecraftRL/.runtime/steve1 python -m unittest <module> -v`
- 게임·GPU: `reacq.sh <script> <args>`(스크래치패드, 2026-11-06 실행과 같은 환경 변수).

## Review Focus

1. **철광석 누출.** 자료 구축에 철광석 실행이 섞이면 거부해야 한다(과업 3 테스트).
2. **철광석 설정 동일성.** `--block` 없이 만든 설정·fixture·계획이 기존과 바이트 단위로 같아야 한다(과업 2 테스트).
3. **reacquire 블록 하나.** 블록 설정에는 대상 블록 skill에만 `reacquire`가 있어야 한다. 둘이면 probe가 시작을 거부한다(과업 2 테스트).
4. **옛 자료 호환.** 보류가 정수 하나인 기존 자료도 학습이 읽어야 한다(과업 4 테스트).
5. **균형 가중치의 0 나누기.** 정답 행이 없는 배치·묶음에서 손실이 NaN이 되면 안 된다(과업 4 테스트).

---

### Task 1: 블록 문구와 등록 파일

**Files:**
- Modify: `scripts/prepare_demo_prompts.py` (BLOCK_PROMPTS, SETS['blocks'])
- Create: `configs/multi-block-v1.json`

- [ ] **Step 1:** `BLOCK_PROMPTS = {'mine_coal_ore': 'mine coal ore', 'mine_lapis_ore': 'mine lapis lazuli ore', 'mine_sand': 'mine sand', 'mine_clay': 'mine clay', 'mine_white_wool': 'mine white wool', 'mine_oak_planks': 'mine oak planks'}`를 더하고 `SETS['blocks']`에 넣는다.
- [ ] **Step 2:** `reacq.sh scripts/prepare_demo_prompts.py --root R --output R/results/multi-block-prompts --set blocks` → 6개 `.npy`와 manifest.
- [ ] **Step 3:** `configs/multi-block-v1.json`: `schema: multi_block_v1`, `held_out: ["iron_ore"]`, `blocks: {block: {item, skill, prompt, embedding, sha256}}`(학습 6개). skill 이름은 `mine_<block>_reacquire`. sha256은 manifest의 file_sha256.
- [ ] **Step 4:** 커밋 `Add the six training-block prompts and their registry`.

### Task 2: 배치 실행기 `--block`

**Files:**
- Modify: `scripts/run_reacquire_batch.py`
- Test: `tests/test_reacquire_batch.py`

**Interfaces:**
- Produces: `prepare(batch, arm, root, checkpoint, use_stage, block='iron_ore') -> (config, fixture, plan)`; `block_config(base, registry, block, checkpoint, use_stage) -> dict`; `block_fixture(base_fixture, block) -> dict`; `run_name(block, seed, n) -> str`(철광석 `s%d-n%d`, 그 외 `<block>-s%d-n%d`); `collect(..., block='iron_ore')`; CLI `--block`.

- [ ] **Step 1: 실패 테스트**
  - `test_iron_files_are_unchanged_by_the_block_option`: `prepare(..., block='iron_ore')`의 세 파일 내용이 기존 `arm_config`/fixture/plan과 같다.
  - `test_block_fixture_replaces_only_the_iron_ores`: coal_ore fixture의 `blocks`에서 iron_ore 두 개만 coal_ore로 바뀌고 나머지·`give`(stone_pickaxe 1)는 같다.
  - `test_block_config_has_one_reacquire_skill`: coal_ore 설정에서 `reacquire`를 가진 skill은 `mine_coal_ore_reacquire` 하나, `target_block='coal_ore'`, `target_items=['coal']`, cond_scale 6.0, embedding·sha256은 등록 파일 값.
  - `test_block_plan_and_run_names`: 계획 `[dict(id='coal_ore', skill='mine_coal_ore_reacquire', kind='policy', target='coal', count=1, block='coal_ore')]`, 실행 폴더 `official/coal_ore-s271828-n1`, frozen 파일 `frozen-official-coal_ore.json`.
  - `test_unknown_block_is_refused`.
- [ ] **Step 2:** `WSLTEST test_reacquire_batch` → 새 테스트 실패.
- [ ] **Step 3:** 구현. 철광석 경로는 기존 코드 그대로 두고 블록 경로만 더한다.
- [ ] **Step 4:** `WSLTEST test_reacquire_batch` → 모두 통과.
- [ ] **Step 5:** 커밋 `Let the reacquire batch runner swap the iron ores for another block`.

### Task 3: 자료 구축 — 블록 기록, 여러 보류, 금지 블록

**Files:**
- Modify: `src/minecraftrl/reacquire_data.py`, `scripts/build_reacquire_dataset.py`
- Test: `tests/test_reacquire_data.py`

**Interfaces:**
- Produces: `build_dataset(run_dirs, out_path, *, holdout, mapper, transformer, root, teacher='centre', forbidden_blocks=())`. `holdout`은 str 또는 str 목록. 저장 자료에 `block`(행별 int, `blocks` 목록의 index), `blocks`(정렬된 블록 이름), `holdout`(str 하나면 int, 목록이면 int 목록). summary에 `labelled_by_block`. CLI `--holdout NAME...`, `--forbid-block BLOCK`(반복).

- [ ] **Step 1: 실패 테스트**: 금지 블록 실행 → ValueError; 보류 두 개 → `holdout` 목록 두 개; 행별 `block` 기록.
- [ ] **Step 2:** 실패 확인. **Step 3:** 구현. **Step 4:** `WSLTEST test_reacquire_data` 통과.
- [ ] **Step 5:** 커밋 `Record each row's block, allow several holdout runs and refuse held-out blocks`.

### Task 4: 학습 — (블록, 단계) 균형과 최선 epoch

**Files:**
- Modify: `src/minecraftrl/reacquire_training.py`, `scripts/train_reacquire.py`
- Test: `tests/test_reacquire_training.py`

**Interfaces:**
- Produces: `train(..., balance=False, patience=None)`. `balance=True`: 정답 학습 행마다 가중치 1/(같은 (블록, 단계) 묶음의 정답 학습 행 수), 정답 학습 행 평균 1로 정규화; 배치 BC = Σw·nll/Σw. `block`이 없는 자료는 단계만으로 묶는다. `patience=k`: 보류 NLL이 k epoch 연속 최저를 못 넘으면 멈추고 최저 epoch로 되돌린다. report에 `best_epoch`, `stopped_early`, `balance`. 보류가 int 목록인 자료와 int 하나인 자료 모두 읽는다. CLI `--balance on|off`, `--patience N`.

- [ ] **Step 1: 실패 테스트**: 균형 가중치(묶음 크기 1과 3 → 가중치 비 3:1, 평균 1); patience 되돌리기(가짜 head로 보류 NLL이 오르면 최저 epoch 상태); int 목록 보류.
- [ ] **Step 2–4:** 실패 확인 → 구현 → `WSLTEST test_reacquire_training` 통과.
- [ ] **Step 5:** 커밋 `Balance the BC loss over block and stage groups and keep the best holdout epoch`.

### Task 5: 채굴 시도 집계, R0 관문, 개발 판정

**Files:**
- Create: `src/minecraftrl/mining_attempts.py`, `scripts/multi_block_report.py`
- Test: `tests/test_mining_attempts.py`

**Interfaces:**
- Produces: `attempts(rows, block) -> dict(on_rows, attack_rows, ends={'destroyed','released','left'}, lengths, streaks)`(정책 행만, 시도 = 서버 채굴 블록이 대상 계열인 연속 행); `destroyed_share(stats)`; `r0_gate(per_block) -> dict(pass, failing_blocks)`(파괴 비율 < 0.5 블록 2개 이상); `dev_verdict(official, iron_only, multi) -> dict(conditions, improved)`(설계 §9의 세 조건). CLI `survey --batch B --arm A --out F`, `dev --batch B --arms official hold-on multi --out F`.

- [ ] **Step 1: 실패 테스트**: 합성 telemetry로 파괴·뗌·벗어남 구분; 관문 경계(파괴 비율 0.5 정확히는 실패 블록 아님); 판정 세 조건 각각.
- [ ] **Step 2–4:** 실패 확인 → 구현 → `WSLTEST test_mining_attempts` 통과.
- [ ] **Step 5:** 커밋 `Count mining attempts per block for the R0 gate and the iron verdict`.

### Task 6: 오프라인 철광석 관문 스크립트

**Files:**
- Create: `scripts/offline_hold_gate.py`

- [ ] **Step 1:** 앞선 `hold_probability.py` 방식으로 학생·공식의 p_hold를 계산한다: (a) `results/reacquire-20261107/data/r0-2-hold.pt`의 공식 실행 철광석 채굴·조준 hold 정답 행(학습에 안 쓰는 철광석 기록, 평가 전용), (b) 새 자료의 보류 실행 블록별 채굴 hold 행. 출력 JSON과 진행 조건(학생 철광석 채굴 평균 p_hold ≥ 공식).
- [ ] **Step 2:** 커밋 `Add the offline iron hold gate`.

### Task 7: 실험 (게임·학습, 결과 폴더 `results/multi-block-20261108/` = B)

- [ ] R0: 블록 6개 × `collect --block X --arm official --checkpoint official --use-stage on --seeds 271828 314159 161803 --sampling 1 --batch B/r0`.
- [ ] R0 관문: `multi_block_report.py survey --batch B/r0 --arm official`. 실패하면 멈추고 보고.
- [ ] 자료 r0 → 학습 r1(`--balance on --patience 2 --kl-coef 10`), 보류 = 블록마다 R0 실행 1개(seed 161803).
- [ ] R1: 학생 r1로 블록 6개 × seed 3개 × sampling 2. 자료 r0–1 → 학습 r2(보류 = R1 seed 161803 실행).
- [ ] R2: 학생 r2로 sampling 3. 자료 r0–2 → 학습 final(보류 = R2 seed 161803 실행).
- [ ] 오프라인 관문(과업 6). 실패하면 개발 게임 없이 분석.
- [ ] 개발 비교: `--block` 없이(철광석) `--mode dev --seeds 130363 --sampling 1 2 3 4`, 갈래 official / hold-on(`results/reacquire-20261107/train/hold-on/checkpoint.pt`) / multi(final).
- [ ] `multi_block_report.py dev` + `run_reacquire_batch.py compare`. 결과 문서 `docs/multi-block-results-2026-11-08.md`, `docs/progress.md` 갱신, 커밋.
