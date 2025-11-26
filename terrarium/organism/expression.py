"""Expression head for translating internal state into observable signals."""

from __future__ import annotations

from typing import Dict, Sequence


class ExpressionHead:
    """Maps core affect into a compact expression vector."""

    def generate(
        self,
        emotion_latent: Sequence[float],
        facing: str,
        drives: Dict[str, float] | None = None,
        gaze_target: str | None = None,
    ) -> Dict[str, float | str]:
        valence = float(emotion_latent[0]) if emotion_latent else 0.0
        arousal = float(emotion_latent[1]) if len(emotion_latent) > 1 else 0.0
        safety = drives.get("safety_drive", 0.5) if drives else 0.5
        tension = max(0.0, min(1.0, (abs(arousal) + (1 - safety)) / 2))
        expression = {
            "smile_frown": max(-1.0, min(1.0, valence)),
            "posture_tension": tension,
            "gaze_direction": gaze_target or facing,
        }
        return expression
