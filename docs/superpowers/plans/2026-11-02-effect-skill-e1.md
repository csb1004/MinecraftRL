# E1 implementation plan

Goal: connect canonical root goals, verified reusable effect requests, and independent RL training.
Architecture: GoalQueue → effect request/committed outcome → conditioned policy and segmented PPO.
Tech: Python, existing PyTorch/STEVE-1/MineRL WSL runtime, PowerShell supervisor.
Spec: `docs/specs/2026-11-02-effect-skill-e1.md`.

Constraints: original Windows workspace; sequential implementation; preserve unrelated dirty files,
official weights, source research spec and archived runs; no repeated approval, paid API, push, or automatic promotion.

1. Add pure effect ledger and counterexample tests. Verify partial credit, no re-pick exploitation,
   identity, actual inventory evidence, and skill/episode boundary separation.
2. Add opt-in E1 policy/environment using existing saved-world and motor paths. Wire request context,
   committed outcome/events, canonical hashes, and multiple root goals. Keep R1 default unchanged.
3. Extract configurable checks from existing adapter trainer and add separate E1 CLI. Validate effects
   from their evidence, reset GAE at skill boundaries, reject non-train or incompatible records.
4. Run focused regression tests, source/input snapshot, fixed small actual pilot with independent
   train/development data and actual STOP. Review evidence, write results and update progress.

Review focus: root completion unchanged; no reward by stale/inferred screen state; no credit for
drop/re-pick; no future state in inputs; no effects from uncommitted actions; no bootstrap across
completed skills; unchanged legacy behavior; data eligibility and checkpoint compatibility.

## Execution ledger

- Planning: complete. Implementation uses test-first checks in main thread; no subagents per AGENTS.md.
- Ruling: reuse the R1 residual actor form, extend only its structured request inputs. FiLM is deferred
  to avoid adding an unnecessary architecture intervention before validating the new learning boundary.
- Ruling: use existing recovery CLI controls plus explicit E1 opt-in; avoid a second run-config parser.
- Tasks 1–3: complete. Test-first credit/GAE, runtime handoff, conditioning, evidence rejection; legacy
  finalization regression found and fixed. WSL focused regression: 79 tests pass (4.664s test time).
- Pre-game review: sequential author review per AGENTS.md; checked root/skill boundaries, train isolation,
  zero-initialized frozen motor, dispatch-only events, parent replay and backward-compatible options.
  A separate reviewer was not used. Source/data snapshots precede all actual games.
- Task 4: fixed 14-run pilot + one independent STOP probe prepared; no results claimed yet.
- Task 4 primary: complete 2026-11-03 01:22 KST. New train 1,417 rows, 90 PPO updates; frozen motor
  unchanged and replay error 0. All 14 games/5,907 rows pass raw and tensor/effect audits. Actual STOP passes.
- Primary development: initial 4/5, candidate 3/5; no promotion. Candidate wood3 improves but wood1
  regresses. Candidate dirt→wood timed out amid 45s inference stall; user confirmed temporarily running
  another game. A separate two-run fixed diagnostic pair is authorized by standing local verification
  scope and frozen before execution. Do not replace original outcomes or perform a success-driven loop.
- Same-code actual STOP proof is reused for this supplemental pair; each added run still verifies
  release/no postcompletion actions/owned PID cleanup. No redundant additional STOP game.
- Task 4 supplemental/final: complete 01:36 KST. Two additional runs/1,073 actions pass all audits,
  source/input/official identities and original artifact hashes preserved. Initial succeeds in 473 actions;
  candidate fails dirt after 600, acquires incidental wood2, and never executes the queued wood goal.
  Same initial pose/PNG/features, normal candidate inference (median 57.46ms; max 245.58ms).
  This confirms policy retention/goal adherence remains unresolved, separate from external GPU contention.
- Final review: sequential author review (`results/effect-skill-e1-20261102/review.md`), not an independent
  reviewer. No further code defects found in audited traces. Main task implementation and validation complete;
  learned candidate remains experimental and unpromoted. STOP retained, no paid/API/cloud/publication action.
