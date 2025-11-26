"""Expression head for translating internal state into observable signals."""

from __future__ import annotations

from typing import Dict, Sequence


class ExpressionHead:
    """Maps core affect into a compact expression vector."""

    def generate(self, emotion_latent: Sequence[float], facing: str) -> Dict[str, float | str]:
        valence = float(emotion_latent[0]) if emotion_latent else 0.0
        arousal = float(emotion_latent[1]) if len(emotion_latent) > 1 else 0.0
        expression = {
            "smile_frown": max(-1.0, min(1.0, valence)),
            "posture_tension": max(0.0, min(1.0, (arousal + 1) / 2)),
            "gaze_direction": facing,
        }
        return expression
