import pytest
import torch

from veomni.optim.lr_scheduler import build_lr_scheduler


def test_wsd_warms_up_holds_then_decays_linearly_to_min():
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=2e-4)
    scheduler = build_lr_scheduler(
        optimizer, 100, lr=2e-4, lr_decay_style="wsd", lr_warmup_ratio=0.1, lr_min=1e-6, lr_wsd_decay_ratio=0.2
    )
    lrs = []
    for _ in range(101):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert lrs[0] == 0 and lrs[5] == pytest.approx(1e-4)
    assert all(lr == pytest.approx(2e-4) for lr in lrs[10:81])
    assert lrs[90] == pytest.approx(2e-4 - (2e-4 - 1e-6) * 0.5)
    assert lrs[100] == pytest.approx(1e-6)


def test_wsd_rejects_decay_longer_than_post_warmup():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    with pytest.raises(ValueError, match="fit after warmup"):
        build_lr_scheduler(optimizer, 10, lr=1.0, lr_decay_style="wsd", lr_warmup_ratio=0.5, lr_wsd_decay_ratio=0.8)
