# Common result recovery R1

User approved the proposed common recovery improvement and real validation. This is a new opt-in experiment; S2 data/checkpoints and the original research specification remain unchanged.

## Decisions and implementation units

1. `recovery_learning.py` reuses S2 dispatch, cancellation, authoritative goal queue and trajectory lifecycle. A versioned context adds phase/pending validity; unknown dirt fine phases are zero/masked. A requested pickup skill is an intent, not proof of a visible/located item. Wood harvest evidence never supplies the dropped entity's location. Existing observation/JAR stays fixed.
2. `recovery_fixture.py` creates immutable offline world copies with varied dropped wood/dirt, pose, distance and optional step/occluder. Construction gives no policy credit. Start archives/hash, initial server inventory/pose, frame differences and fixed 60-step neutral startup are recorded. No initial automatic pickups are credited. Neutral world preparation has no model actions or reward.
3. Freeze the official backbone AND original motor head; train only a small residual adapter and separate value function. Explicit additional-RM off/on candidates share the same initial behavior only after log-probability replay admission. Subsequent PPO updates require fresh trajectories from exactly that candidate checkpoint. New train data only; no S2/development reuse.
4. Fixed goal embeddings remain target-specific. Pickup requests use `pick up wood` / `pick up dirt`; collection keeps the existing bank. The higher-level queue and target conditioning remain in off. No actor coordinates, fixture labels, item positions or ground-truth visibility are introduced. No external LLM/API calls.
5. Reward is the existing one-time authoritative inventory goal completion (1); no distance/button/look bonus and no separate duplicate pickup reward. Goal activation baseline is preserved until completion; actual already-held/startup items earn nothing. Fixtures start empty, one goal of one new item; pickup/put-down loops cannot repeat terminal reward. Valid task failures train; STOP/technical failures do not.

## Bounded development protocol

- Preparation/diagnostic runs are separate from measured performance. New saved natural starts and fixtures only. No historical deleted world/weight restoration.
- Two training rounds: initial mixed collection/pickup batch shared under exact identical initial behavior; second round collected separately from each updated off/on candidate. Six starts per round/variant: four pickup starts (wood/dirt, varied view/distance) plus two whole-collection anchors. Total 18 train games (6 shared + 6 per candidate). Pickup up to 400 actions/90 seconds; collection up to 1200 actions/180 seconds. Seed/fixture manifest freezes before the measured batch.
- Curriculum round 1: nearby, visible or side-facing items; round 2: increased distance, turned view, modest obstacle/step. This tests a manually specified curriculum, not automatic curriculum discovery.
- PPO gamma .999, lambda 1, minibatch 32, two epochs, actor lr .0001, value lr .001, clip .1, target KL .02, reference KL .1, separate grad clip 1, optimizer seed 111101. KL gate remains enabled; record actual optimized rows/updates, not merely admitted rows. Reference is frozen initial official behavior, distinct from the old behavior used by PPO.
- Development: five held-out pickup layouts plus five saved natural collection starts, each initial/off/on, rotated order; 30 games. Same archive and policy sampling seed within each comparison. Train/dev layouts and natural seeds disjoint. This is development, not a sealed final evaluation; novel item-kind generalization is untested.
- Same protocol for all arms; no retry of valid failures. Technical failure is retained, diagnosed, and any retry receives a new id and a written amendment. No outcome-driven checkpoint selection: use round 2 candidate (or mark absent if training could not produce it).
- Evaluate authoritative acquisition, ticks/time, infer median/p95, full control median/p95, GPU memory, failure reasons, start-state differences, and whole-task regression. Actual STOP once after performance; no new actions after visible request/ack, key release and owned PID cleanup. Keep STOP latched at end.

## Verification and rejection criteria

`tests/test_recovery_learning.py`: unknown vs false evidence; pickup intent vs sensed existence; item-block evidence separation; zero-init base parity; only adapter/value updates; off invariant; checkpoint/source/variant/split replay admission; terminal reward once and no post-completion dispatch; bounded task failure vs truncation/STOP.

`tests/test_recovery_fixture.py`: immutable source, safe archive, exact seed/hash/player/item setup, collision checks, split binding, no actor access to layout metadata. Existing S1/S2/queue/STOP regressions remain required.

Actual diagnostic must confirm loaded archive/pose, zero target baseline, valid tick progression and unchanged observer JAR before the measured batch. Exact pixel equivalence is measured, not assumed from seed/archive equality. Item rendering/physics and recurrent warmup can differ; report rather than hide deviations. If frozen visual features cannot support recovery, do not claim an adapter fixes perception; use failure evidence to scope a later encoder experiment.

No automatic model promotion, public release, paid resources or new vocabulary for unimplemented stone/crafting skills. Review source and freeze it before the measured batch; preserve any later diagnostic fix separately.
