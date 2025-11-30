from terrarium.organism.emotion import EmotionEngine


def test_sleep_urge_increases_with_time_and_fatigue() -> None:
    eng = EmotionEngine()
    eng.reset()
    state = eng.update(
        reward=0.0,
        novelty=0.0,
        prediction_error=0.0,
        intero_signals={"time_since_sleep": 1.0, "energy": 0.2, "fatigue": 0.8},
    )
    drives = eng.drives_dict()
    assert "sleep_urge" in drives
    assert drives["sleep_urge"] > 0.2  # baseline was 0.2; expect increase
    # latent should be 8-dim
    assert len(state.latent) == 8


def test_tiredness_and_social_satiation_and_confusion_layout() -> None:
    eng = EmotionEngine()
    eng.reset()
    # low energy/high fatigue -> high tiredness
    st = eng.update(
        reward=0.0,
        novelty=0.0,
        prediction_error=0.0,
        intero_signals={"energy": 0.1, "fatigue": 0.9, "time_since_social_contact": 0.0},
    )
    tired_idx = 2
    social_idx = 3
    assert st.latent[tired_idx] > 0.4
    social_before = st.latent[social_idx]
    confusion_idx = 7
    assert social_before > 0.4  # social satiation starts mid and no deprivation
    # deprivation increases loneliness -> lower social_satiation
    confusion_before = st.latent[confusion_idx]
    st2 = eng.update(
        reward=0.0,
        novelty=0.0,
        prediction_error=0.5,
        intero_signals={"energy": 0.5, "fatigue": 0.2, "time_since_social_contact": 1.0},
    )
    assert st2.latent[social_idx] < social_before
    # confusion increases with prediction error
    assert st2.latent[confusion_idx] > confusion_before
