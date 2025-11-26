# Terrarium Mind – Stage 0 (Prototype)

Minimal headless Python scaffold for a split-core organism in a tiny gridworld.

## Setup

- Requires Python 3.10+.
- No external dependencies for Stage 0. Optionally create a virtualenv for isolation.

## Run the demo

```bash
python -m terrarium.scripts.stage0_demo
```

## What you will see

- Step-by-step logs showing chosen actions, rewards, novelty, valence, and arousal.
- ASCII snapshots of the grid after each episode (`A` agent, `O` reward objects, `M` mirror tile).
- A final line with the number of stored transitions in the replay buffer.
