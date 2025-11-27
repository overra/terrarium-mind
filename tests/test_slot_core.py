import torch

from terrarium.organism.slot_core import HemisphereSlotCore


def test_slot_core_shapes_and_update() -> None:
    slot_dim = 8
    obj_slots = 2
    peer_slots = 1
    refl_slots = 1
    input_dim = 6
    core = HemisphereSlotCore(slot_dim, obj_slots, peer_slots, refl_slots, input_dim, emotion_dim=2, vision_dim=4)
    batch = 1
    total_slots = 1 + obj_slots + peer_slots + refl_slots
    hidden = torch.zeros(batch, total_slots, slot_dim)

    self_feat = torch.randn(batch, input_dim)
    obj_feats = torch.randn(batch, obj_slots, input_dim)
    peer_feats = torch.randn(batch, peer_slots, input_dim)
    refl_feats = torch.randn(batch, refl_slots, input_dim)

    emotion = torch.randn(batch, 2)
    vision = torch.randn(batch, 4)
    slots, summary = core(self_feat, obj_feats, peer_feats, refl_feats, hidden, emotion, vision)
    assert slots.shape == (batch, total_slots, slot_dim)
    assert summary.shape == (batch, slot_dim)
    # Ensure some change from zero hidden on at least one element
    assert (slots.detach() - hidden).abs().sum().item() > 0
    # Emotion modulation path exists
    assert hasattr(core, "emotion_mlp")
