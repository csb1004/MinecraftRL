# E2: executable contracts and whole-program learning

Status: implementation and fixed local development pilot, 2026-11-03. User approved the four proposals in the RM→RL bottleneck review. Research scope and four-stage OPEN decisions are unchanged.

## Common contract

`goal_spec_v2` adds explicit `effect=acquire|possess` and active `preserve=[{target,quantity}]` to registered wood/dirt contracts. Acquisition counts new inventory since activation; possession counts actual held stock. Already satisfied possession dispatches no action and earns no invented transition. Preservation is checked before success, including the success tick. Missing entry requirements block dispatch. Unknown targets remain unsupported.

The same compiler and `goal_spec request --version 2` schema serve humans and a future LLM. Execution requests preserve target, effect, quantity, prerequisites, invariants, source hash and fixed condition identity. Existing bank/prompt selection is unchanged; natural collection does not yet dynamically switch to a pickup embedding. The goal vector is the same existing 512-D vector, now also directly available to actor/value alongside structured effect and inventory constraints.

Information condition is privileged server inventory plus existing masked fixed phase evidence. No new detector or hidden item coordinates enter the actor. This is not a video-only experiment. The official STEVE-1 backbone, recurrent core and motor remain frozen; only the common residual adapter and value network train.

## Learning objective and controls

E2 records original subgoal reward for audit and separately learns whole-program completion. Return crosses acquisitions and goal transitions. Shaping is `0.2 * (0.999 * Phi(next) - Phi(current))`, with Phi mean completed/quantity progress across goals and zero potential at true terminal. Timeout retains its final potential and value bootstrap. No repeatedly earned inventory bonus is added. This does not guarantee useful signal if no task succeeds.

Fresh on-policy rows have equal total objective weight per observed target/effect group; PPO visits are not duplicated. This weighting applies to value loss, critic warmup, advantage normalization and clipped actor objective. Report group counts and weights. Common PPO: 2 epochs, minibatch 32, actor LR .0001, value LR .001, clip .1, old KL stop .02, official KL .1, gamma .999, lambda 1. All three candidates use exactly the same fresh data and weighting.

1. `plain`: no critic warmup or additional retention data.
2. `calibrated`: 10 critic-only epochs against fixed rollout returns, original behavior probabilities preserved and actor fingerprint unchanged.
3. `retained`: same calibration plus .1 official-policy KL on two separate train-only pickup rollouts. Those rows never enter PPO or its return statistics.

Learning is a separate bounded script. No updates or LLM calls during play. No automatic promotion. With insufficient/invalid learning signal, preserve evidence and report failure; do not sweep until success.

## Frozen local pilot

Batch `results/execution-contract-e2-20261103`. Freeze code, inputs, tests, official weights, research specification and old E1 completion receipts. No prior experiment rerun or overwrite.

- Prepare two new train starts, seeds 81113001 and 81113003, using 20 neutral steps without learned policy. Preparation is not performance data.
- Four train episodes: both wood→dirt and dirt→wood. Seed 81113001 uses acquire quantities 1→2; seed 81113003 uses possess 2→1. Second goals preserve the first target's required stock.
- Two separate train retention episodes: existing pickup wood-far/dirt-turned fixtures, target quantity 1, maximum 400 actions.
- Five development seeds 81112001/003/007/009/013 respectively: acquire wood3; possess dirt3; wood1→dirt1 preserving wood1; dirt1→wood1 preserving dirt1; acquire wood1→possess wood1 (no extra successor action).
- Every development case compares initial official policy with three candidates, same saved world and sampling seed, fixed rotated model order. 20 development runs; 26 policy runs total. Maximum 180s control/run, natural action budget1200. Known development worlds, not sealed final evaluation or proof of generalization.
- Separate actual STOP after 10 committed actions, retained candidate and wood64, excluded from performance. After request/ack: no new dispatch, release verified and owned children gone.

Audit actual raw server inventory/ticks/action against contract records and tensor rows. Record per-goal attempted/unexecuted status, whole-program completion, preservation violations, latency median/p95, whole control cycle, VRAM, cleanup and reasons. Compare paired start pose/image/features and disclose mismatches. Technical failures are not policy failures; no unexecuted goal counts as a failed sample.

## Future pipeline boundary

`BoundaryUpdates` accepts priority-only requests at committed boundaries. Pending inference defers application; stale/duplicate identities reject, semantic changes require a fresh versioned run. It is a single-owner mailbox API, not a new concurrent monitor/LLM service. Failure records distinguish uncertainty, resources, contracts, preconditions, preservation and unresolved stalls; none automatically authorizes training or proves lack of capability.

## Decision after pilot

Separate structural correctness from behavior improvement. Candidate-vs-initial task adherence and completed-program evidence decide whether additional experience/capacity is justified. Calibration and retention comparisons diagnose causes; this small pilot cannot establish retention across all skills. No success-rate target or automatic checkpoint promotion. Maintain STOP at end.
