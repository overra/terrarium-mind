# Repository Guidelines

## Project Structure & Module Organization
- Core package lives under `terrarium/`: `env/` (gridworld and stage2 world), `organism/` (slot cores, emotion, policy, expression), `simulation.py` (legacy runner), `replay.py`, `plasticity.py`, `utils/` (signals), `world.py` (wrapper), `runtime.py`, `metabolism.py`, and `scripts/` (entry points).
- Demos: `terrarium/scripts/stage0_demo.py` (grid) and training scripts under `terrarium/scripts/`.
- Tests: under `tests/` mirroring module paths (see current fixtures for env, cores, replay, policy).

## Build, Test, and Development Commands
- Environment: Python 3.10+. Use `uv` for env + tooling.
- Create/activate venv: `uv venv .venv` then `source .venv/bin/activate` (or `./.venv/Scripts/activate` on Windows). Sync deps with `uv pip install -r requirements.txt` (torch, numpy, wandb, pytest).
- Stage 0.5 demo: `python -m terrarium.scripts.stage0_demo` (config-driven run, wandb offline by default).
- Stage 1 training: `python -m terrarium.scripts.stage1_train` (grid env, DQN, prioritized replay).
- Stage 2.5 training: `python -m terrarium.scripts.stage2_train` (object-centric env, World/Runtime wrapper, slot cores, energy/time signals, expanded metrics).
- Tests: `uv run pytest` from repo root.

## Coding Style & Naming Conventions
- Language: Python with type hints and concise docstrings.
- Indentation: 4 spaces; keep lines readable (<100 chars preferred).
- Naming: snake_case for modules/functions/vars; PascalCase for classes; UPPER_SNAKE_CASE for constants.
- Comments: brief intent-level comments only where code isn’t obvious.

## Testing Guidelines
- Framework: `pytest`.
- Naming: `tests/<module_path>/test_<unit>.py`; test functions `test_<behavior>()`.
- Scope: prefer small, deterministic unit tests (env step/reset, novelty/error utilities, replay add/sample). Add regression tests when fixing bugs, and write/update tests alongside new features. Run `uv run pytest` regularly, especially before merging.

## Commit & Pull Request Guidelines
- Commits: write imperative, concise messages (e.g., `Add replay buffer priorities`, `Wire demo runner logging`).
- Pull Requests: include a short summary of changes, any breaking behavior, and how to run affected commands (`python -m ...`, `pytest`). Add screenshots or logs only when UI/output changes are notable (for now, copy console snippets).
- Link issues/tickets when applicable; keep PRs scoped to a coherent change.

## Architecture Overview (Stage 2.5)
- World: headless 2.5D env with objects/peers/mirrors and tasks; wrapped by `terrarium/world.py`.
- Organism: slot-based hemispheres + bridge, emotion engine producing latent `E_t` (drives, affect), expression head, policy/Q-network.
- Runtime/Training: `Runtime` + `RLTrainer` wire world, organism, replay, plasticity, metabolism; logs to wandb (success rates, valence/arousal, energy/fatigue, time-since-*).
