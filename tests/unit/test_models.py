import pytest
import torch
from torch import nn

from xrayml.models import build_model
from xrayml.models.layers import Bottleneck, DenseLayer, MBConv, bn_to_gn, conv_macs, freeze_bn
from xrayml.registry import MODELS


def test_registered_models():
    assert {"timm_classifier", "fasterrcnn", "retinanet", "unet"} <= set(MODELS.names())


def test_classifier_outputs_logits():
    m = build_model("timm_classifier", backbone="resnet10t").eval()
    out = m(torch.rand(3, 1, 64, 64))
    assert out.shape == (3, 1)


def test_unet_shape_and_groupnorm():
    m = build_model("unet", base=4, norm="group")
    assert not any(isinstance(x, nn.BatchNorm2d) for x in m.modules())
    assert m(torch.rand(2, 1, 32, 32)).shape == (2, 1, 32, 32)


def test_detector_train_and_eval_contract():
    m = build_model("fasterrcnn", image_size=64)
    m.train()
    losses = m(
        [torch.rand(1, 64, 64)], [{"boxes": torch.tensor([[5.0, 5, 30, 20]]), "labels": torch.tensor([1])}]
    )
    assert set(losses) == {"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"}
    m.eval()
    with torch.no_grad():
        out = m([torch.rand(1, 64, 64)])[0]
    assert set(out) == {"boxes", "labels", "scores"}


def test_bottleneck_projection_shortcut():
    b = Bottleneck(64, 32, stride=2)
    assert b(torch.rand(1, 64, 16, 16)).shape == (1, 128, 8, 8)
    assert isinstance(Bottleneck(128, 32).shortcut, nn.Identity)


def test_mbconv_skip_only_when_shapes_match():
    assert MBConv(16, 16).use_skip and not MBConv(16, 24).use_skip and not MBConv(16, 16, stride=2).use_skip
    assert MBConv(16, 24, stride=2)(torch.rand(2, 16, 8, 8)).shape == (2, 24, 4, 4)


def test_dense_layer_grows_channels_by_growth_rate():
    assert DenseLayer(64, growth=32)(torch.rand(1, 64, 8, 8)).shape == (1, 96, 8, 8)


def test_grouped_and_depthwise_parameter_counts():
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    full = count(nn.Conv2d(128, 128, 3, bias=False))
    assert full == 147_456
    assert count(nn.Conv2d(128, 128, 3, groups=32, bias=False)) == full // 32
    assert count(nn.Conv2d(128, 128, 3, groups=128, bias=False)) == 1_152


def test_conv_macs_matches_revision_example():
    assert conv_macs(56, 56, 64, 64, 3) == pytest.approx(115.6e6, rel=1e-3)  # ~116M MACs


def test_bn_to_gn_and_freeze_bn():
    net = nn.Sequential(nn.Conv2d(1, 24, 3), nn.BatchNorm2d(24), nn.Sequential(nn.BatchNorm2d(24)))
    frozen = freeze_bn(nn.Sequential(nn.BatchNorm2d(8)))
    assert not frozen[0].training and not frozen[0].weight.requires_grad
    bn_to_gn(net)
    assert not any(isinstance(x, nn.BatchNorm2d) for x in net.modules())
    assert all(m.num_channels % m.num_groups == 0 for m in net.modules() if isinstance(m, nn.GroupNorm))


def test_gradcam_heatmap_range():
    from xrayml.explain.gradcam import GradCAM

    m = build_model("timm_classifier", backbone="resnet10t")
    cam = GradCAM(m, m.gradcam_target_layer())
    h = cam(torch.rand(2, 1, 64, 64))
    cam.remove()
    assert h.shape == (2, 64, 64) and float(h.min()) >= 0 and float(h.max()) <= 1 + 1e-6
