from functools import cache
import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnext import Block, LayerNorm


class ConvNeXtIsotropic(nn.Module):
    r"""ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf
        Isotropic ConvNeXts (Section 3.3 in paper)

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depth (tuple(int)): Number of blocks. Default: 18.
        dims (int): Feature dimension. Default: 384
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depth=18,
        dim=384,
        drop_path_rate=0.0,
        layer_scale_init_value=0,
        head_init_scale=1.0,
    ):
        super().__init__()

        self.stem = nn.Conv2d(in_chans, dim, kernel_size=16, stride=16)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                Block(dim=dim, layer_scale_init_value=layer_scale_init_value)
                for i in range(depth)
            ]
        )

        self.norm = LayerNorm(dim, eps=1e-6)  # final norm layer
        self.head = nn.Linear(dim, num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            # trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x, block=None):
        x = self.stem(x)
        if block == -1:
            return x
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if block == i:
                return x
        return self.norm(
            x.mean([-2, -1])
        )  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x, block=None):
        x = self.forward_features(x, block)
        if block is not None:
            return x
        x = self.head(x)
        return x


# @register_model
def convnext_isotropic_small(pretrained=False, **kwargs):
    model = ConvNeXtIsotropic(depth=18, dim=384, **kwargs)
    if pretrained:
        url = (
            "https://dl.fbaipublicfiles.com/convnext/convnext_iso_small_1k_224_ema.pth"
        )
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


# @register_model
def convnext_isotropic_base(pretrained=False, **kwargs):
    model = ConvNeXtIsotropic(depth=18, dim=768, **kwargs)
    if pretrained:
        url = "https://dl.fbaipublicfiles.com/convnext/convnext_iso_base_1k_224_ema.pth"
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


# @register_model
def convnext_isotropic_large(pretrained=False, **kwargs):
    model = ConvNeXtIsotropic(depth=36, dim=1024, layer_scale_init_value=1e-6, **kwargs)
    if pretrained:
        url = (
            "https://dl.fbaipublicfiles.com/convnext/convnext_iso_large_1k_224_ema.pth"
        )
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@cache
def get_convnext(size="small"):
    if size == "small":
        model = convnext_isotropic_small(pretrained=True)

        w_out = model.blocks[17].pwconv2.weight.detach().T
        head = model.head.weight.detach().T
        model.neuron_to_logit = w_out @ head

        w_outs = []
        w_ins = []
        for layer_id in range(18):
            summed_dwconv = (
                model.blocks[layer_id].dwconv.weight[:, 0].detach().sum(dim=(-1, -2))
            )
            w_in = model.blocks[layer_id].pwconv1.weight.detach()
            w_in = torch.einsum("nd,d->nd", w_in, summed_dwconv)
            w_out = model.blocks[layer_id].pwconv2.weight.detach().T
            w_ins.append(w_in)
            w_outs.append(w_out)
        w_ins = torch.stack(w_ins, dim=0)
        w_outs = torch.stack(w_outs, dim=0)
        model.w_ins = w_ins
        model.w_outs = w_outs
        model.normed_w_ins = w_ins / w_ins.norm(dim=-1, keepdim=True)
        model.normed_w_outs = w_outs / w_outs.norm(dim=-1, keepdim=True)

        return model
    if size == "base":
        model = convnext_isotropic_base(pretrained=True)
        w_out = model.blocks[17].pwconv2.weight.detach().T
        head = model.head.weight.detach().T
        model.neuron_to_logit = w_out @ head
    if size == "large":
        model = convnext_isotropic_large(pretrained=True)
        # w_out = model.blocks[35].pwconv2.weight.detach().T
        # head = model.head.weight.detach().T
        # model.neuron_to_logit = (w_out @ head)
    return model
