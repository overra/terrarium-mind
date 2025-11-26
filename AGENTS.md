# Repository Guidelines

## Project Structure & Module Organization
- Core package lives under `terrarium/`: `env/` (gridworld), `organism/` (split cores, emotion, policy, expression), `simulation.py` (runner), `replay.py`, `plasticity.py`, `utils/` (signals), and `scripts/` (entry points).
- Demo: `terrarium/scripts/stage0_demo.py`.
- Tests: not present yet; place future tests under `tests/` mirroring module paths (e.g., `tests/env/test_gridworld.py`).

## Build, Test, and Development Commands
- Environment: Python 3.10+. Use `uv` for env + tooling.
- Create/activate venv: `uv venv .venv` then `source .venv/bin/activate` (or `./.venv/Scripts/activate` on Windows). Sync deps with `uv pip install -r requirements.txt` (torch, numpy, wandb).
- Stage 0.5 demo: `python -m terrarium.scripts.stage0_demo` (config-driven run, wandb logging offline by default).
- Stage 1 training: `python -m terrarium.scripts.stage1_train` (DQN with neural cores/bridge, prioritized replay, emotion-modulated epsilon).
- Tests: install deps then run `uv run pytest` from repo root.

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

## Architecture Overview (Stage 0)
- Environment: headless 2D grid with `reset()/step(action)` returning structured observations.
- Organism: left/right cores + bridge, emotion engine producing latent `E_t`, random policy head, and expression head.
- Simulation: `SimulationRunner` wires env + organism + plasticity + replay, storing transitions for future learning. Designed to extend toward remote clients (e.g., r3f/WebSocket) later.
