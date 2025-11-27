from terrarium.organism.emotion import EmotionEngine


def test_sleep_drive_increases_with_time_and_fatigue() -> None:
    eng = EmotionEngine()
    eng.reset()
    state = eng.update(
        reward=0.0,
        novelty=0.0,
        prediction_error=0.0,
        intero_signals={"time_since_sleep": 1.0, "energy": 0.2, "fatigue": 0.8},
    )
    drives = eng.drives_dict()
    assert "sleep_drive" in drives
    assert drives["sleep_drive"] > 0.2  # baseline was 0.2; expect increase
    # latent should include sleep drive (len >= 6)
    assert len(state.latent) >= 6
