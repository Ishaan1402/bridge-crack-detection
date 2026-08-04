import pytest
import torch

from src.models.checkpoint import inspect_state_dict, load_unet_checkpoint
from src.models.unet import UNet


def _save_with_legacy_names(model: UNet, path: str) -> None:
    """Save a state dict using the old ``up.*`` / ``final.*`` key names."""
    state = {}
    for key, value in model.state_dict().items():
        if key.startswith("up_transposes."):
            key = "up." + key[len("up_transposes."):]
        elif key.startswith("final_conv."):
            key = "final." + key[len("final_conv."):]
        state[key] = value
    torch.save(state, path)


def test_load_narrow_checkpoint_with_legacy_names(tmp_path):
    """The published narrow [32,64,128,256] weights (up/final naming) must load."""
    model = UNet(in_channels=3, out_channels=1, features=[32, 64, 128, 256])
    path = tmp_path / "narrow_legacy.pth"
    _save_with_legacy_names(model, path)

    loaded, features = load_unet_checkpoint(str(path), torch.device("cpu"))

    assert features == [32, 64, 128, 256]
    assert loaded.encoder[0].conv[0].out_channels == 32
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), f"Mismatch on {key}"


def test_load_wide_checkpoint_with_current_names(tmp_path):
    """Current-naming wide [64,128,256,512] checkpoints must still load."""
    model = UNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512])
    path = tmp_path / "wide_current.pth"
    torch.save(model.state_dict(), path)

    loaded, features = load_unet_checkpoint(str(path), torch.device("cpu"))

    assert features == [64, 128, 256, 512]
    assert loaded.encoder[0].conv[0].out_channels == 64


def test_inspect_rejects_garbage():
    with pytest.raises(ValueError, match="Unexpected checkpoint format"):
        inspect_state_dict({"not": "a checkpoint"})


def test_inspect_rejects_unknown_width():
    state = {"encoder.0.conv.0.weight": torch.zeros(128, 3, 3, 3)}
    with pytest.raises(ValueError, match="Unrecognized first encoder width"):
        inspect_state_dict(state)


def test_load_checkpoint_wrapped_in_state_dict(tmp_path):
    """Checkpoints saved as {'state_dict': ..., 'epoch': ...} must load too."""
    model = UNet(in_channels=3, out_channels=1, features=[32, 64, 128, 256])
    path = tmp_path / "wrapped.pth"
    torch.save({"state_dict": model.state_dict(), "epoch": 5, "val_dice": 0.747}, path)

    loaded, features = load_unet_checkpoint(str(path), torch.device("cpu"))

    assert features == [32, 64, 128, 256]
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), f"Mismatch on {key}"
