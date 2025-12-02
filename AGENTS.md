# Repository Guidelines

## Project Structure & Module Organization

Core package lives under `terrarium/`:

### Environment
- `env/gridworld.py` – Stage 1 grid environment
- `env/world.py` – Stage 2 2.5D environment with objects, peers, mirrors, screens

### Organism
- `organism/organism.py` – Top-level container wiring all components
- `organism/slot_core.py` – `HemisphereSlotCore` with FiLM modulation
- `organism/cores.py` – `Bridge` connecting hemispheres
- `organism/emotion.py` – `EmotionEngine` producing 8-dim E_t latent
- `organism/vision.py` – `VisionEncoder` (retina) and `CameraVisionEncoder`
- `organism/audio.py` – `AudioEncoder` for binaural input
- `organism/attachment.py` – `AttachmentCore` for peer bonding
- `organism/memory.py` – `SalientMemory` for episodic storage
- `organism/world_model.py` – `PredictiveHead` for auxiliary prediction loss
- `organism/policy.py` – `EpsilonGreedyPolicy`
- `organism/q_network.py` – `QNetwork` for action values
- `organism/expression.py` – `ExpressionHead` for social signals

### Training & Runtime
- `training.py` – `RLTrainer` with full training loop and wandb logging
- `qlearning.py` – `QTrainer` for TD loss computation
- `replay.py` – `ReplayBuffer` with prioritized sampling
- `plasticity.py` – `PlasticityController` for priority computation

### World & Server
- `world.py` – `World` wrapper with retina/camera/audio rendering
- `world_server.py` – `WorldServer` orchestrating agents
- `agents.py` – `AgentClient` protocol and `OrganismClient` implementation
- `runtime.py` – `Runtime` wrapper

### Supporting Systems
- `metabolism.py` – `MetabolicCore` tracking energy/fatigue
- `homeostasis.py` – Homeostatic reward computation and `HomeostasisTracker`
- `config.py` – `RunConfig` dataclass

### Utilities
- `utils/signals.py` – `compute_novelty`, `compute_prediction_error`
- `vis/retina_logging.py` – `retina_to_image` for wandb logging
- `vis/world_viewer.py` – Top-down rendering and video generation

### Scripts
- `scripts/stage0_demo.py` – Random policy demo
- `scripts/stage1_train.py` – DQN on grid
- `scripts/stage2_train.py` – Full 2.5D training
- `scripts/long_train.py` – Extended training with slow epsilon decay
- `scripts/exist.py` – Persistent world server mode

### Tests
- `tests/` – 24 test files covering all major components

## Environment Setup

- Python 3.10+. Use `uv` for env + tooling.
- Create/activate venv: `uv venv .venv` then `source .venv/bin/activate` (or `./.venv/Scripts/activate` on Windows).
- Sync deps: `uv pip install -r requirements.txt` (torch, numpy, wandb, pytest).

## Run Commands

```bash
# Stage 0.5 demo (random policy, wandb offline)
python -m terrarium.scripts.stage0_demo

# Stage 1 training (grid env, DQN)
python -m terrarium.scripts.stage1_train

# Stage 2 training (2.5D env, slot cores, emotion modulation)
python -m terrarium.scripts.stage2_train

# Long training (extended runs, both vision modes)
python -m terrarium.scripts.long_train

# Exist mode (persistent server, optional learning)
python -m terrarium.scripts.exist [--steps N] [--learn] [--log-interval N]

# Tests
uv run pytest
```

## Coding Style & Naming Conventions

- Language: Python with type hints and concise docstrings.
- Indentation: 4 spaces; keep lines readable (<100 chars preferred).
- Naming: snake_case for modules/functions/vars; PascalCase for classes; UPPER_SNAKE_CASE for constants.
- Comments: brief intent-level comments only where code isn't obvious.

## Testing Guidelines

- Framework: `pytest`.
- Naming: `tests/test_<unit>.py`; test functions `test_<behavior>()`.
- Scope: prefer small, deterministic unit tests. Run `uv run pytest` regularly, especially before merging.
- Coverage areas: env step/reset, vision modes, emotion engine, replay buffer, slot cores, homeostasis, memory, audio, attachment, predictive head, Q-learning stability.

## Commit & Pull Request Guidelines

- Commits: write imperative, concise messages (e.g., `Add camera vision encoder`, `Wire homeostasis intrinsic reward`).
- Pull Requests: include a short summary of changes, any breaking behavior, and how to run affected commands.
- Link issues/tickets when applicable; keep PRs scoped to a coherent change.

## Architecture Overview (Stage 7.5)

### World
- Headless 2.5D env with objects, peers, mirrors, screens
- 12 tasks including social (gaze, follow, caregiver) and perceptual (sound, vision discrimination)
- Egocentric observations with relative positions/orientations
- Wrapped by `World` class providing retina, camera, and audio rendering

### Organism
- **Dual hemispheres**: `HemisphereSlotCore` with slots for self/objects/peers/reflections
- **Bridge**: Limited-bandwidth MLP with residual connections
- **FiLM modulation**: Emotion (E_t), vision, and audio modulate slot hidden states
- **Emotion engine**: 8-dim latent (valence, arousal, tiredness, social_satiation, curiosity, safety, sleep_urge, confusion)
- **Multi-modal perception**: Retina (7-channel grid), Camera (RGB pinhole), Audio (binaural)
- **Attachment**: EMA-tracked peer bonding values
- **Salient memory**: Stores emotionally significant moments above salience threshold
- **Predictive head**: Auxiliary loss predicting next emotion/core state
- **Expression head**: Converts internal state to social signals (smile_frown, posture_tension, gaze_direction)

### Metabolism & Homeostasis
- **MetabolicCore**: Energy depleted by actions/arousal/learning; fatigue accumulates; sleep restores
- **Homeostasis**: Intrinsic rewards for keeping E_t within target bands; chronic overload penalties

### Learning
- **DQN** with prioritized experience replay
- **Priority** based on novelty, prediction error, reward, arousal
- **Emotion-modulated exploration**: Curiosity scales epsilon
- **Sleep consolidation**: Extra replay steps during sleep action
- **Optional observational learning**: Imitation loss on demo transitions

### Client/Server
- **WorldServer**: Orchestrates world and multiple agent clients
- **OrganismClient**: Full RL loop with replay, learning, metabolism
- **Snapshots**: `get_snapshot()` provides world state for viewers/teachers
