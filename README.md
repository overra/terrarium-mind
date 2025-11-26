# Terrarium Mind – Stage 0.5 (Prototype)

Minimal headless Python scaffold for a split-core organism in a tiny gridworld, now with run config, backend shim, and wandb logging.

## Setup

- Requires Python 3.10+ and `uv`.
- Create venv: `uv venv .venv` then `source .venv/bin/activate`.
- Install deps: `uv pip install -r requirements.txt` (torch, numpy, wandb).

## Run the demo

```bash
python -m terrarium.scripts.stage0_demo
```

Config is defined in `terrarium/config.py` (`RunConfig`). Adjust seeds, episodes, env size, backend choice, and logging cadence there.

## What you will see

- Per-episode summary with reward, mean valence/arousal, and ASCII grid snapshot (`A` agent, `O` objects, `M` mirror).
- wandb run (offline by default) logging episode metrics and config; switch to online by setting `wandb_mode="online"` in `RunConfig`.
- Final line with the number of stored transitions in the replay buffer.
