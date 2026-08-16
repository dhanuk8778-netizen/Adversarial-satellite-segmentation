import torch

from src.models.unet import UNet


def test_unet_output_shape():
    model = UNet(in_channels=6, num_classes=6, base_filters=8)
    x = torch.randn(2, 6, 64, 64)
    out = model(x)
    assert out.shape == (2, 6, 64, 64)


def test_unet_odd_input_size():
    """U-Net with skip connections must handle non-power-of-2 sizes via padding."""
    model = UNet(in_channels=6, num_classes=6, base_filters=8)
    x = torch.randn(1, 6, 65, 65)
    out = model(x)
    assert out.shape == (1, 6, 65, 65)


def test_unet_param_count_positive():
    model = UNet(in_channels=6, num_classes=6, base_filters=8)
    assert model.num_parameters() > 0


def test_unet_backward_pass():
    model = UNet(in_channels=6, num_classes=6, base_filters=8)
    x = torch.randn(2, 6, 32, 32, requires_grad=True)
    y = torch.randint(0, 6, (2, 32, 32))
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(loss)
