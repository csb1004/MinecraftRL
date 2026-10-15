# MinecraftRL research snapshot — 2026-10-15

This research prerelease publishes the current planner/recovery/priority-scheduling
code and reproducible evidence for the archived Minecraft wood-collection failures.

- October 12: both complete `far/fixture` and `far/visit_memory` records, original
  wood/waypoint checkpoints, start worlds, and goal embedding.
- October 15: the latest integrated wood-goal `coverage_motion` corner failure,
  its refined navigation checkpoint, skill embeddings, and exact Python snapshot.
- Content-addressed bundles, experiment manifests, SHA256 verification, resumable
  draft publication, and non-overwriting PowerShell artifact synchronization.
- Current code includes the previously local navigation/refinement and bounded
  goal-priority aging work. Aging's existing validation is software-only.

All three archived games ended `planning_paused`, acquired no wood, and recorded
`execution_valid=false`. They are privileged-server development diagnostics, not
successful policy benchmarks. No new training or game rollout was performed for
this release. The later near-navigation BC candidate was not adopted.

Download using `scripts/Sync-Artifacts.ps1 -Tag research-20261015 -Experiment all`.
See `docs/artifact-sharing.md` for per-experiment synchronization, official base
model downloads, local MineRL build instructions, and future publishing.

Minecraft JARs, game-containing wheels, upstream source mirrors and base-model
copies are deliberately absent from the public assets. Local original delivery
ZIPs are preserved separately. Exact hashes and official model locations remain
documented. The original research specification and archived evidence are retained.

Operational verification: verify remote release asset digests, download/restore
the published assets into a project-local verification directory, repeat sync to
confirm zero new downloads, and check that the STOP latch remains unchanged.
