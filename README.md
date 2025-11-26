# Terrarium Mind – Stage 2.5

Prototype with neural hemisphere cores, bridge, and DQN-style learning on simple tasks in a tiny gridworld. Stage 0.5 demo remains available.

## Setup

- Requires Python 3.10+ and `uv`.
- Create venv: `uv venv .venv` then `source .venv/bin/activate`.
- Install deps: `uv pip install -r requirements.txt` (torch, numpy, wandb).
- (Optional) pytest discovery is configured via `setup.cfg` to include the repo root on `PYTHONPATH`.

## Run the Stage 0.5 demo (random policy)

```bash
python -m terrarium.scripts.stage0_demo
```

Config is defined in `terrarium/config.py` (`RunConfig`). Adjust seeds, episodes, env size, backend choice, and logging cadence there.

# Run Stage 1 training (DQN)

```bash
python -m terrarium.scripts.stage1_train
```

Key config knobs (in `RunConfig`): `num_episodes`, `max_steps_per_episode`, `hidden_dim`, `bridge_dim`, `epsilon_*`, `gamma`, `lr`, `batch_size`, `train_start`, `priority_alpha/beta`, `curiosity_epsilon_scale`.

# Run Stage 2.5 training (2.5D env + new tasks)

```bash
python -m terrarium.scripts.stage2_train
```

Stage 2.5 uses the object-centric environment with mirrors/peers and tasks: goto_mirror, touch_object, examine_reflection, follow_peer, social_gaze, novel_object_investigation. Observations are structured egocentric dicts; training runs via a World + Runtime wrapper and logs per-task success rates plus energy/fatigue/time-since-* metrics.

## What you will see

- Per-episode summary with reward, mean valence/arousal, and ASCII grid snapshot (`A` agent, `O` objects, `M` mirror).
- wandb run (offline by default) logging episode metrics and config; switch to online by setting `wandb_mode="online"` in `RunConfig`.
- Final line with the number of stored transitions in the replay buffer.
- Stage 1 wandb metrics: episode reward/length, task success flags, per-task success_rate (rolling window), q_loss, valence/arousal/prediction_error means, epsilon trace.
- Stage 2.5 wandb metrics: Stage 1 set plus success rates for new tasks, mean energy/fatigue, time-since-{reward,social,reflection}, valence_positive_fraction, and richer observations (no ASCII grid).
