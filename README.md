# Terrarium Mind – Stage 7.5

A biologically-inspired artificial organism that learns to navigate and interact with a simulated 2.5D world. The architecture integrates dual-hemisphere neural processing, emotion-driven learning, multi-modal perception (vision + audio), social attachment, metabolism, and homeostatic regulation.

## Setup

- Requires Python 3.10+ and `uv`.
- Create venv: `uv venv .venv` then `source .venv/bin/activate`.
- Install deps: `uv pip install -r requirements.txt` (torch, numpy, wandb, pytest).
- (Optional) pytest discovery is configured via `setup.cfg` to include the repo root on `PYTHONPATH`.

## Run Commands

### Stage 0.5 Demo (random policy)

```bash
python -m terrarium.scripts.stage0_demo
```

### Stage 1 Training (DQN on grid)

```bash
python -m terrarium.scripts.stage1_train
```

### Stage 2 Training (2.5D environment)

```bash
python -m terrarium.scripts.stage2_train
```

Full 2.5D environment with objects, peers, mirrors, screens, and all tasks. Uses retina vision by default.

### Long Training (extended runs)

```bash
python -m terrarium.scripts.long_train
```

Uses `epsilon_mode="long_train"` for gradual epsilon decay over many episodes. Enables both vision modes (retina + camera).

### Exist Mode (persistent world server)

```bash
python -m terrarium.scripts.exist [--steps N] [--learn] [--log-interval N] [--seed N]
```

Runs a long-lived world with a single organism client. Intended for future viewer/teacher agents. Supports optional online learning.

### Tests

```bash
uv run pytest
```

## Architecture Overview

### Dual-Hemisphere Brain

- **Left/Right `HemisphereSlotCore`**: Slot-based attention over entity types (self, objects, peers, reflections)
- **Bridge**: Limited-bandwidth MLP connecting hemispheres with residual modulation
- **FiLM Modulation**: Emotion, vision, and audio signals modulate slot processing via learned gain/bias

### Emotion Engine (E_t)

8-dimensional latent vector driving behavior:

| Index | Dimension | Description |
|-------|-----------|-------------|
| 0 | valence | Pleasant ↔ unpleasant |
| 1 | arousal | Calm ↔ keyed up |
| 2 | tiredness | Body depletion from energy/fatigue |
| 3 | social_satiation | Socially full ↔ lonely |
| 4 | curiosity_drive | Urge to explore |
| 5 | safety_drive | Urge to avoid threat |
| 6 | sleep_urge | Compulsion to sleep/rest |
| 7 | confusion_level | Cognitive uncertainty / model mismatch |

### Multi-Modal Perception

- **Retina**: 7-channel egocentric grid (occupancy, objects, peers, mirrors, screens, intensity, motion)
- **Camera**: Pinhole-projected RGB image with depth-based rendering
- **Audio**: Binaural (left/right) loudness from peers, screens, and sound sources

### Supporting Systems

- **Metabolism**: Energy/fatigue budget affected by actions, arousal, learning load; restored by sleep
- **Homeostasis**: Intrinsic rewards for maintaining E_t dimensions within target bands
- **Attachment Core**: Tracks social bonds with peer entities via EMA-updated values
- **Salient Memory**: Separate episodic store for emotionally/cognitively significant moments
- **Predictive Head**: Auxiliary loss predicting next emotion/core state given action

## Vision Modes

Configure via `vision_mode` in `RunConfig`:

| Mode | Description | Use Case |
|------|-------------|----------|
| `retina` | 7-channel egocentric grid | Default, fast, symbolic |
| `camera` | RGB pinhole projection | More realistic visual input |
| `both` | Fused retina + camera | Maximum information |

## Task List

| Task | Description | Config Flag |
|------|-------------|-------------|
| `goto_mirror` | Navigate to mirror surface | Always enabled |
| `touch_object` | Touch nearest object | Always enabled |
| `examine_reflection` | Face mirror at close range | Always enabled |
| `follow_peer` | Maintain following distance from peer | Always enabled |
| `social_gaze` | Hold gaze on peer for 2+ steps | Always enabled |
| `novel_object_investigation` | Approach unseen object | Always enabled |
| `cooperative_goal` | Agent + peer both reach goal zone | Always enabled |
| `vision_object_discrim` | Touch glowing object (visual discrimination) | `enable_vision_object_discrim` |
| `go_to_sound` | Navigate to sound source | `enable_go_to_sound` |
| `stay_with_caregiver` | Maintain proximity to caregiver | `enable_stay_with_caregiver` |
| `explore_and_return` | Explore far then return to caregiver | `enable_explore_and_return` |
| `move_to_target` | Navigate to specific target position | `enable_move_to_target` |

## Key Config Options

Config is defined in `terrarium/config.py` (`RunConfig`). Key options:

| Option | Default | Description |
|--------|---------|-------------|
| `num_episodes` | 50 | Training episodes |
| `max_steps_per_episode` | 60 | Max steps per episode |
| `vision_mode` | `"retina"` | Vision mode: `retina`, `camera`, or `both` |
| `enable_head_yaw` | `False` | Enable independent head orientation |
| `use_homeostasis` | `True` | Enable homeostatic intrinsic rewards |
| `use_predictive_head` | `True` | Enable auxiliary prediction loss |
| `use_salient_memory` | `True` | Enable salient episodic memory |
| `use_body_variation` | `False` | Randomize body parameters |
| `epsilon_mode` | `"dev"` | Epsilon schedule: `dev` or `long_train` |
| `sleep_replay_multiplier` | 3 | Extra replay steps during sleep |
| `log_retina` | `False` | Log retina images to wandb |
| `log_camera` | `False` | Log camera images to wandb |
| `log_topdown_video` | `False` | Log top-down episode videos |

## What You Will See

- **Per-episode summary**: reward, task success, energy/fatigue levels
- **wandb metrics** (online or offline):
  - Episode: reward, length, task success, per-task success rates (rolling window)
  - Emotion: mean valence/arousal/tiredness/social_satiation/curiosity/safety/sleep_urge/confusion
  - Metabolism: mean energy, mean fatigue
  - Time signals: time_since_reward, time_since_social_contact, time_since_reflection
  - Sleep: sleep_fraction, avg_sleep_length
  - Training: q_loss, prediction_error, epsilon
  - Vision: mean_intensity, mean_motion (if retina logged)
  - Memory: salient memory size
- **Final line**: number of stored transitions in replay buffer
