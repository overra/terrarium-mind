from terrarium.organism.expression import ExpressionHead


def test_expression_orientation_used_for_gaze() -> None:
    head = ExpressionHead()
    orientation = 1.57
    expr = head.generate([0.1, 0.2], orientation, drives={"safety_drive": 0.5}, gaze_target=None)
    assert expr["gaze_direction"] == orientation
