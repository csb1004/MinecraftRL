# SDD ledger — plan: docs/first-rl-stage.md

Approved task: real wood reward, first gradient/optimizer update, bounded pilot and before/after evaluation.
Ruling: work in the required D: workspace; no Git initialization/worktree or subagents (user instructions).
Pre-flight: collector must store probability of exactly the action dispatched; terminal reward and GAE masks must agree.
Ruling: first pilot freezes STEVE-1 vision/recurrent backbone and trains the existing action head plus a zero-initialized wood value head.
This is explicitly head-only RL, not whole-model fine-tuning. Frozen features permit exact head recomputation without truncated recurrent gradients and keep the first experiment small. Full-backbone training remains unmeasured.
Ruling: normalize CFG logits before log probability; upstream sampling accepts unnormalized logits but its reported guided log_prob cannot directly be used for PPO ratios.
Ruling: use existing pinned MineRL/STEVE-1 environment; no mandatory MineStudio migration for the first bounded head update.
Train development seed104729; held-out evaluation seeds1000003,1000033,1000037,1000039,1000081. Evaluation trajectories never enter training.
Protocol: fixed text embedding and prompt collect wood, guidance6; sparse first server-confirmed log reward1; gamma.99 lambda.95; clip.2; AdamW1e-5; KL-to-initial-head coefficient.01; two epochs, minibatch16; one training rollout for the first pilot; no adaptive selection using evaluation results.
Initial state (superseded below): core tests written; implementation and real execution pending.

Task 1 complete: reward/GAE tests failed on missing implementation, then passed. Additional rejection of eval/cancelled/nonfinite data and server-vs-client/terminal dispatch tests failed then passed. PPO normalization/clipping test failed then passed in WSL.
Task 2 complete: real train rollout `results/rl-train-104729-20261005T022109534595Z`,159 transitions, server terminal reward1,24.741s. Final observed logs2; threshold is at least1, no exact-count claim. No post-accepting policy action.
Task 3 complete: `results/rl-head-training-20261005`,20 optimizer updates,17,955,387 trainable parameters, actor max delta0.000194434, checkpoint reload logp error0, optimizer state restored. Head training Torch peak779.08MiB. Behavior logp recomputation max error0.000139236 (<0.0002), finite losses/gradients.
Task 4 complete: evaluation `results/rl-evaluation-20261005T022639139802Z`, immutable checkpoint, five held-out seeds, before/after each,180s control limit. Both5/5 success, four faster and one slower. Initial frame hashes differ in four pairs; same seeds do not establish identical initial observations. No general improvement claim.

Task 5 audit complete: `results/rl-pilot-audit.json` verified actual dispatch equality, server reward/terminal alignment, no post-target/ack dispatch, immutable actor/value states, optimizer20 steps and no evaluation data used in training.
Main-thread review (user prohibits subagents): checked normalized CFG probabilities, terminal vs truncated GAE, eval rejection, nonfinite guards, CPU checkpoint staging and stop latching. Final current-source test results and process state go in `results/rl-final-verification.json`.
Final verification complete: Windows150 passed/4 skipped; WSL154 passed. WSL project runtimes0, Windows Python/Java0, STOP latched, checkpoint and research-spec hashes unchanged. Evidence saved in `results/rl-final-verification.json`; detailed interpretation in `docs/rl-pilot-validation-2026-10-05.md`.
Ruling: first train summary has generic target_observed:false because target_items was omitted, while server_target_reached:true and reward_sum1 are correct. Added target_items for later runs; preserve original train summary and use audited server reward for the learning result.
Ruling: no Colab connection needed for this head-only pilot; full-backbone training is not tested or inferred feasible from the779MiB head optimizer phase.
