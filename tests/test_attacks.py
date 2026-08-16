import torch

from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.models.unet import UNet


def _tiny_model():
    return UNet(in_channels=6, num_classes=6, base_filters=4)


def test_pgd_stays_within_eps_ball():
    model = _tiny_model()
    # clamp to the attack's valid input range: the eps-ball projection is
    # only guaranteed relative to an already-in-range clean image (an
    # out-of-range clean pixel would itself move under the final box clip,
    # which is a data-validity issue, not an attack-budget violation)
    images = torch.clamp(torch.randn(2, 6, 32, 32), -3.0, 3.0)
    labels = torch.randint(0, 6, (2, 32, 32))
    eps = 8 / 255

    adv = pgd_attack(model, images, labels, eps=eps, num_iter=5)
    delta = (adv - images).abs()
    assert delta.max().item() <= eps + 1e-5


def test_pgd_respects_clip_bounds():
    model = _tiny_model()
    images = torch.randn(2, 6, 32, 32) * 5  # push near/over clip bounds
    labels = torch.randint(0, 6, (2, 32, 32))
    adv = pgd_attack(model, images, labels, eps=8 / 255, num_iter=3, clip_min=-3.0, clip_max=3.0)
    assert adv.max().item() <= 3.0 + 1e-5
    assert adv.min().item() >= -3.0 - 1e-5


def test_pgd_output_shape_matches_input():
    model = _tiny_model()
    images = torch.randn(3, 6, 16, 16)
    labels = torch.randint(0, 6, (3, 16, 16))
    adv = pgd_attack(model, images, labels, eps=8 / 255, num_iter=2)
    assert adv.shape == images.shape


def test_fgsm_stays_within_eps_ball():
    model = _tiny_model()
    images = torch.clamp(torch.randn(2, 6, 32, 32), -3.0, 3.0)
    labels = torch.randint(0, 6, (2, 32, 32))
    eps = 8 / 255
    adv = fgsm_attack(model, images, labels, eps=eps)
    delta = (adv - images).abs()
    assert delta.max().item() <= eps + 1e-5


def test_pgd_attack_does_not_track_gradients_on_output():
    model = _tiny_model()
    images = torch.randn(1, 6, 16, 16)
    labels = torch.randint(0, 6, (1, 16, 16))
    adv = pgd_attack(model, images, labels, eps=8 / 255, num_iter=2)
    assert not adv.requires_grad


def test_pgd_increases_loss_on_undefended_model():
    """Sanity check: PGD should generally raise the loss relative to clean
    input on a randomly-initialized (undefended) model, more than 50% of
    the time across a few trials, confirming gradients point the right way."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    model = _tiny_model()
    wins = 0
    trials = 5
    for _ in range(trials):
        images = torch.randn(2, 6, 24, 24)
        labels = torch.randint(0, 6, (2, 24, 24))
        with torch.no_grad():
            clean_loss = F.cross_entropy(model(images), labels).item()
        adv = pgd_attack(model, images, labels, eps=8 / 255, num_iter=10)
        with torch.no_grad():
            adv_loss = F.cross_entropy(model(adv), labels).item()
        if adv_loss >= clean_loss:
            wins += 1
    assert wins >= trials * 0.5
