import torch

from terrarium.organism.world_model import PredictiveHead


def test_predictive_head_shapes_and_training() -> None:
    batch = 4
    emotion_dim = 8
    core_dim = 16
    action_dim = 5
    head = PredictiveHead(emotion_dim, core_dim, action_dim, hidden_dim=32)
    emo = torch.randn(batch, emotion_dim)
    core = torch.randn(batch, core_dim)
    actions = torch.eye(action_dim)[:batch]
    pred_emo, pred_core = head(emo, core, actions)
    assert pred_emo.shape == (batch, emotion_dim)
    assert pred_core.shape == (batch, core_dim)

    target_emo = torch.zeros_like(pred_emo)
    target_core = torch.zeros_like(pred_core)
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.SGD(head.parameters(), lr=0.1)
    loss1 = loss_fn(pred_emo, target_emo) + loss_fn(pred_core, target_core)
    loss1.backward()
    opt.step()
    pred_emo2, pred_core2 = head(emo, core, actions)
    loss2 = loss_fn(pred_emo2, target_emo) + loss_fn(pred_core2, target_core)
    assert loss2.item() <= loss1.item()
