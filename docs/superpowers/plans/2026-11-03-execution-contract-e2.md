# Execution Contract E2 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans sequentially. User AGENTS.md forbids automatic subagents and requires this actual project path. The user has approved implementation; do not ask for plan reapproval.

**Goal:** Connect a common executable goal contract to policy/value inputs, root-aligned RL, transition/retention experiments and typed failure/revision boundaries.

**Architecture:** Add an opt-in E2 path over the existing queue and saved-start runtime. Preserve E1/v1 semantics. E2 adds acquire/possess and active inventory invariants, explicit existing goal embeddings in actor/value context, whole-program returns, and a bounded independent trainer. Future LLM/monitor/SQL consume the same contracts; no live API/KB implementation is claimed.

**Tech Stack:** Python unittest/PyTorch, existing MineRL/STEVE-1 WSL environment, Windows PowerShell.

**Spec:** `docs/rm-rl-bottleneck-review-2026-11-03.md`, user acceptance of its four concrete changes.

## Global Constraints

- Work only in D:\DATA\Desktop\MinecraftRL; PowerShell, no gh. Keep unrelated dirty changes.
- Preserve original research specification, weights, worlds and completed pilots. No automatic promotion.
- Fixed observer; inventory evidence remains privileged. No scripted policy replacement.
- Actual server evidence determines success. STOP, release and owned PID cleanup remain mandatory.
- No API, paid resources, cloud connections or publication. No production observer retraining.
- No subagents. Plan/ledger stay in this file instead of skill helper worktrees/commits.

## Review Focus

- Already satisfied possession goals dispatch no extra action and earn no fictitious training transition.
- Preservation violation on the same tick as acquisition cannot count as goal/program success.
- Priority-only changes cannot mutate semantic contracts or apply to a pending inference.
- Whole-program returns cross subgoal boundaries; true terminals and truncation bootstrap differ.
- Retention samples cannot enter on-policy PPO, and development/evaluation never train.

### Task 1: Common semantics and safe revision/failure boundaries

Files: new `src/minecraftrl/execution_contract.py`, `tests/test_execution_contract.py`; minimal edits `goal_spec.py`, `goal_queue.py`, `rm_program.py`.

Interfaces: `compile_spec` accepts explicit `goal_spec_v2`; v1 unchanged. Each v2 goal has `effect=acquire|possess` and `preserve=[{target,quantity}]` in addition to v1 fields. `execution_request(queue, phase, task)` contains caller-owned effect, target, preconditions, invariants and meaning hash. `BoundaryUpdates.submit/apply` only supports validated priority changes at committed boundaries; semantic changes require a new run. `classify_failure` returns evidence category and permitted next steps, never automatic training authorization.

- [x] Write failing tests for acquire/possess, reset completion, simultaneous violation, unknown fields, copied contracts, stale/duplicate/deferred updates and ambiguous failures.
- [x] Run native unittest; confirm missing functionality fails.
- [x] Implement strict v2 compilation and queue semantics, leaving legacy defaults intact.
- [x] Pass new tests and related goal/queue/interaction regressions. Ledger source scope.

### Task 2: Runtime and root-aligned learning evidence

Files: new `execution_learning.py`, `execution_returns.py`, `tests/test_execution_learning.py`; edits `cli.py`, `rm_conditioning.py`, STOP probe.

Interfaces: opt-in `--execution-learning`; separate scope/archive. Existing 512-D goal vector plus structured effect/invariant context feeds both actor and value. Reuse frozen official motor and shared residual form. Rows retain root reward and add program reward, potential, training reward, request/outcome and fixed observation metadata. Root-aligned GAE does not stop at individual acquisitions. Versioned potential coefficient .2, gamma .999, terminal potential zero; timeout bootstraps.

- [x] RED: target/context wiring, preservation state, root return across goals, shaping telescoping, termination vs truncation, stale/corrupt evidence rejection.
- [x] Implement runtime request/evidence and post-commit classification/boundary application.
- [x] GREEN: tensor/CPU tests and existing E1/legacy finalization/STOP unit checks.

### Task 3: Independent learning with calibration and retention

Files: new `scripts/train_execution_adapter.py`, focused tests; bounded extension to `train_recovery_adapter.py` profile hooks.

Interfaces: E2 profile supplies actor, trace validation and whole-program returns. Optional critic-only warmup records calibrated baseline separately; actor stays unchanged. Optional train-only retention runs provide official-policy KL anchors, excluded from PPO. Existing E1 defaults remain identical. All knobs/checkpoint schema and teacher identities logged.

- [x] RED: development rejection, stale parent PPO rejection, retention exclusion, critic-only actor identity, whole-program target usage.
- [x] Implement explicit profile options and meaningful validation; test regression in WSL.
- [x] GREEN: no-update replay and synthetic optimizer smoke; do not count these as game success.

### Task 4: Frozen local pilot and actual STOP

Files: fresh ignored results/protocol and bounded PowerShell driver; reusable audit support as needed.

- [x] Freeze train/development roles, world/source/model hashes, goal ordering, budgets and comparison candidates before games. Max 180s control/run, 5 fixed development seeds; paired initial/candidate starts.
- [x] Collect balanced new train data with both target orders and dedicated train retention starts. Use additional train world preparation if needed; setup never counts as policy success.
- [x] Train sequential controlled candidates (base E2, calibration, calibration+retention) from the same fresh behavior data. Do not sweep settings until a success appears.
- [x] Evaluate fixed conditions; audit actual inventory, request/context, termination, source preservation, GPU/latency and STOP/release/owned children. No promotion if preservation or goal adherence regresses.

### Task 5: Review and handoff

- [x] Sequential source/diff/contract review, relevant regression checks after fixes.
- [x] Document interfaces, independent commands, actual outcomes and remaining limitations. Update progress; final STOP latched.

## Execution ledger

- Planning: approved user intent mapped to E2; no fresh design/permission question needed. Sequential main-thread implementation.
- Preflight: tasks 1→2 share canonical semantics; tasks 2→3 share context/return schema; task 4 uses the same frozen implementation. Explicit v2/E2 avoids reinterpreting old v1/E1 artifacts.
- Task 1: complete. Native 40 related checks passed (one tensor test deferred to WSL). V2 explicit acquire/possess/preserve, copied requests, deferred priority-only updates and conservative failure taxonomy.
- Task 2: implemented. Whole-program return, instant-successor handoff, nonfinite/corrupt evidence and raw server row checks pass. V2 authoring schema is shared by humans and future API callers; v1 default remains intact.
- Task 3: implemented. RED observed before helpers/schema existed, then GREEN. E2 adds critic-only calibration, separate train-only official-KL anchors and equal observed target/effect objective mass without duplicate PPO visits. WSL final relevant suite: 118 passed in 5.237s. No game success is inferred from these tests.
- Task 4 complete: new neutral train starts81113001/81113003; actual CUDA matrix operation and rendered Minecraft frames verified. PPO2,373actions/3of4 programs complete; separate KL508actions, wood0/dirt1 success. Three candidates each150PPO updates, calibrated variants750critic-only updates, replay errors0 and official motor preserved.
- General development5cases: initial/plain/calibrated2of5, retained4of5. Retained regressed on wood→dirt exploration. Its wood→held case completed both nodes at tick340 after258actions, with no successor action. Predeclared pickup supplement8runs: initial1of2/plain2of2/calibrated1of2/retained1of2, with different success domains. No candidate promotion.
- Full audit:34policy runs/16,600actions, raw server/tensor/context mismatch0;544sampled probability rows reproduced. Actual STOP after10actions: no dispatch after request/ack, keys0 and owned children0. Neutral preparation and STOP excluded from performance. Both one-shot drivers completed; final STOP remains.
- Task 5 complete: sequential implementation/contract review, result/usage/progress/README updated. After frozen comparisons, corrected only E2 descriptive goal and phase-context labels. Whole-file AST equals snapshot after removing those two metadata keywords. Focused WSL regressions52passed, skipped0; historical artifacts unchanged. Diff whitespace and Git artifact exclusion checked. Final review408current files(407original hashes plus1declared metadata change),345source snapshots, identical runtime JAR across batches. No API/cloud/publish.
- Evidence: results/execution-contract-e2-20261103/{analysis.json,tensor-audit.json,stop-probe-audit.json,driver-completion.json,postrun-metadata-receipt.json,final-review.json}; supplement counterpart completion/audits. Original scope and four-stage OPEN decisions remain. Do not rerun completed batches.
