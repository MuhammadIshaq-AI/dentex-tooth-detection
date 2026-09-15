import numpy as np
import torch
import torch.nn as nn

from dentex.labels import Boxes
from dentex.uncertainty.matching import average_precision
from dentex.uncertainty.mc_dropout import (
    MCDropout2d, _inference_network, fuse_passes, has_dropout, inject_dropout, set_mc,
)


class FakeDetect(nn.Module):
    def __init__(self):
        super().__init__()
        self.cv2 = nn.ModuleList([nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.Conv2d(4, 4, 1)) for _ in range(2)])
        self.cv3 = nn.ModuleList([nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.Conv2d(4, 2, 1)) for _ in range(2)])

    def forward(self, x):
        return torch.cat([self.cv3[0](x), self.cv2[0](x)], 1)


class FakeDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Identity(), FakeDetect())

    def forward(self, x):
        return self.model(x)


class _Backend:  # plain object, like Ultralytics' PyTorchBackend
    def __init__(self, model):
        self.model = model


class FakeAutoBackend(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backend = _Backend(model)

    @property
    def model(self):
        return self.backend.model


def test_inject_and_toggle_dropout():
    net = FakeDetectionModel().eval()
    assert inject_dropout(net, 0.5) == 4 and has_dropout(net)
    x = torch.rand(1, 4, 8, 8)
    with torch.no_grad():
        assert torch.allclose(net(x), net(x))
        set_mc(net, True)
        assert not torch.allclose(net(x), net(x))


def test_inference_network_reaches_through_autobackend():
    net = FakeDetectionModel()
    inject_dropout(net)
    wrapper = FakeAutoBackend(net)
    assert not has_dropout(wrapper)  # the bug: modules() does not see the backend's network
    assert _inference_network(wrapper) is net
    assert _inference_network(FakeAutoBackend(FakeDetectionModel())) is None


def test_fuse_passes_frequency_votes_and_entropy():
    box = np.array([[10, 10, 50, 50]], float)
    far = np.array([[200, 200, 240, 240]], float)
    passes = [Boxes(np.vstack([box, far]), np.array([1, 0]), np.array([0.9, 0.3])) for _ in range(3)]
    passes += [Boxes(box + 1, np.array([1]), np.array([0.8]))]  # far box missed in this pass
    fused, stats = fuse_passes(passes, num_classes=2)
    order = np.argsort(-fused.conf)
    assert len(fused.cls) == 2 and fused.cls[order].tolist() == [1, 0]
    assert np.allclose(stats["freq"][order], [1.0, 0.75])
    assert np.isclose(fused.conf[order][0], (0.9 * 3 + 0.8) / 4)
    assert ((stats["entropy"] >= 0) & (stats["entropy"] <= 1)).all()
    assert stats["entropy"][order][0] < stats["entropy"][order][1]  # confident, consistent box is less uncertain

    empty, s = fuse_passes([Boxes.empty(with_conf=True)] * 3)
    assert len(empty.cls) == 0 and len(s["freq"]) == 0


def test_average_precision_known_case():
    # 2 GT; ranked TP, FP, TP -> P/R points (1, .5), (.67, 1) -> AP = .5*1 + .5*.667
    assert np.isclose(average_precision([0.9, 0.8, 0.7], [1, 0, 1], 2), 0.5 + 0.5 * 2 / 3)
    assert average_precision([], [], 3) == 0.0
    assert np.isnan(average_precision([0.5], [0], 0))


def test_mcdropout_is_dropout2d():
    assert issubclass(MCDropout2d, nn.Dropout2d)
