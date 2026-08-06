import torch

from src.models.checkpoint import load_unet_checkpoint
from src.models.losses import BCEDiceLoss
from src.models.unet import UNet


def test_default_unet_has_no_extra_keys():
    """Default flags must keep the exact published-checkpoint architecture."""
    model = UNet(features=[32, 64, 128, 256])
    keys = set(model.state_dict())
    assert not any(".se." in k for k in keys)
    assert not any(k.startswith("deep_heads.") for k in keys)
    assert "bottleneck_dropout.weight" not in keys  # Identity has no params


def test_se_adds_channel_attention_keys():
    model = UNet(features=[32, 64, 128, 256], se=True)
    keys = set(model.state_dict())
    assert "encoder.0.se.fc.1.weight" in keys
    assert "decoder.0.se.fc.1.weight" in keys


def test_forward_deep_returns_aligned_aux_heads():
    model = UNet(features=[32, 64, 128, 256], se=True, deep_supervision=True)
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        logits, aux = model.forward_deep(x)
    assert logits.shape == (1, 1, 128, 128)
    assert len(aux) == 3
    for a in aux:
        assert a.shape == logits.shape


def test_bcedice_loss_with_aux():
    torch.manual_seed(0)
    criterion = BCEDiceLoss(bce_weight=0.5, aux_weight=0.25)
    logits = torch.randn(2, 1, 64, 64)
    targets = (torch.rand(2, 1, 64, 64) > 0.95).float()
    aux = [torch.randn(2, 1, 64, 64) for _ in range(3)]

    loss_main = criterion(logits, targets)
    loss_aux = criterion(logits, targets, aux)

    assert loss_main.ndim == 0
    assert loss_aux.ndim == 0
    assert loss_aux.item() > loss_main.item()  # auxiliary terms add loss


def test_interpolate_upsample_mode():
    """interpolate mode uses 1x1 up-convs, not ConvTranspose."""
    default = UNet(features=[32, 64, 128, 256])
    interp = UNet(features=[32, 64, 128, 256], upsample_mode="interpolate")

    assert "up_transposes.0.weight" in default.state_dict()
    assert "up_transposes.0.weight" not in interp.state_dict()
    assert "up_convs.0.weight" in interp.state_dict()

    interp.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        logits = interp(x)
    assert logits.shape == (1, 1, 128, 128)


def test_loader_falls_back_to_interpolate(tmp_path):
    """An interpolate-mode checkpoint must load via the auto-fallback."""
    model = UNet(features=[32, 64, 128, 256], se=True, deep_supervision=True,
                 upsample_mode="interpolate")
    path = tmp_path / "v3_interp.pth"
    torch.save(model.state_dict(), path)

    loaded, features = load_unet_checkpoint(str(path), torch.device("cpu"))

    assert features == [32, 64, 128, 256]
    assert loaded.upsample_mode == "interpolate"
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), f"Mismatch on {key}"
