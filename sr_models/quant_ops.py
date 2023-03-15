import torch
import torch.nn as nn
from sr_models.quant_conv_lsq import BaseConv

import genotypes as gt


OPS = {
    "zero": lambda C_in, C_out, bits, C_fixed, stride, affine, shared, quant_noise=False: Zero(
        stride, zero=0
    ),
    "skip_connect": lambda C_in, C_out, bits, C_fixed, stride, affine, shared, quant_noise=False: Identity(
        shared, quant_noise=quant_noise
    ),
    "conv_5x1_1x5": lambda C_in, C_out, bits, C_fixed, stride, affine, shared, quant_noise=False: FacConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        5,
        stride,
        2,
        affine=affine,
        shared=shared,
        quant_noise=quant_noise,
    ),
    "conv_3x1_1x3": lambda C_in, C_out, bits, C_fixed, stride, affine, shared, quant_noise=False: FacConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        3,
        stride,
        1,
        affine=affine,
        shared=shared,
        quant_noise=quant_noise,
    ),
    "simple_3x3": lambda C_in, C_out, bits, C_fixed, stride, affine, shared, quant_noise=False: SimpleConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        3,
        stride,
        affine=affine,
        shared=shared,
        quant_noise=quant_noise,
    ),
    "simple_1x1": lambda C_in, C_out, bits, C_fixed, stride, affine, shared, quant_noise=False: SimpleConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        1,
        stride,
        affine=affine,
        shared=shared,
        quant_noise=quant_noise,
    ),
    "simple_5x5": lambda C_in, C_out, bits, C_fixed, stride, affine, shared, quant_noise=False: SimpleConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        5,
        stride,
        affine=affine,
        shared=shared,
        quant_noise=quant_noise,
    ),
    "3x3_increase_2": lambda C_in, C_out, bits, C_fixed, stride, affine, shared, quant_noise=False: IncreaseConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        3,
        stride,
        affine=affine,
        shared=shared,
        quant_noise=quant_noise,
        increase_rate=2,
    ),
    "1x1_increase_2": lambda C_in, C_out, bits, C_fixed, stride, affine, shared, quant_noise=False: IncreaseConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        1,
        stride,
        affine=affine,
        shared=shared,
        quant_noise=quant_noise,
        increase_rate=2,
    ),
    "5x5_increase_2": lambda C_in, C_out, bits, C_fixed, stride, affine, shared, quant_noise=False: IncreaseConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        5,
        stride,
        affine=affine,
        shared=shared,
        quant_noise=quant_noise,
        increase_rate=2,
    ),
    "3x3_decrease_2": lambda C_in, C_out, bits, C_fixed, stride, affine, shared, quant_noise=False: DecreaseConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        3,
        stride,
        affine=affine,
        shared=shared,
        quant_noise=quant_noise,
        decrease_rate=2,
    ),
    "1x1_decrease_2": lambda C_in, C_out, bits, C_fixed, stride, affine, shared, quant_noise=False: DecreaseConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        1,
        stride,
        affine=affine,
        shared=shared,
        quant_noise=quant_noise,
        decrease_rate=2,
    ),
    "5x5_decrease_2": lambda C_in, C_out, bits, C_fixed, stride, affine, shared, quant_noise=False: DecreaseConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        5,
        stride,
        affine=affine,
        shared=shared,
        quant_noise=quant_noise,
        decrease_rate=2,
    ),
    "simple_3x3_grouped_full": lambda C_in, C_out, bits, C_fixed, stride, affine, shared, quant_noise=False: SimpleConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        3,
        stride,
        groups=C_in,
        affine=affine,
        shared=shared,
        quant_noise=quant_noise,
    ),
    "simple_5x5_grouped_full": lambda C_in, C_out, bits, C_fixed, stride, affine, shared, quant_noise=False: SimpleConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        5,
        stride,
        groups=C_in,
        affine=affine,
        shared=shared,
        quant_noise=quant_noise,
    ),
    "simple_1x1_grouped_2": lambda C_in, C_out, bits, C_fixed, stride, affine, shared, quant_noise=False: SimpleConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        1,
        stride,
        groups=2,
        affine=affine,
        shared=shared,
        quant_noise=quant_noise,
    ),
    "simple_3x3_grouped_2": lambda C_in, C_out, bits, C_fixed, stride, affine, shared, quant_noise=False: SimpleConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        3,
        stride,
        groups=2,
        affine=affine,
        shared=shared,
        quant_noise=quant_noise,
    ),
    "simple_5x5_grouped_2": lambda C_in, C_out, bits, C_fixed, stride, affine, shared, quant_noise=False: SimpleConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        5,
        stride,
        groups=2,
        affine=affine,
        shared=shared,
        quant_noise=quant_noise,
    ),
    "simple_3x3_d2": lambda C_in, C_out, bits, C_fixed, stride, affine, shared, quant_noise=False: SimpleConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        3,
        stride,
        affine=affine,
        shared=shared,
        quant_noise=quant_noise,
        dilation=2,
    ),
    "simple_3x3_grouped_3_d2": lambda C_in, C_out, bits, C_fixed, stride, affine, shared, quant_noise=False: SimpleConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        3,
        stride,
        groups=3,
        affine=affine,
        shared=shared,
        quant_noise=quant_noise,
        dilation=2
    ),
    "growth2_3x3": lambda C_in, C_out, bits, C_fixed, stride, affine, shared, quant_noise=False: GrowthConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        3,
        stride,
        groups=1,
        affine=affine,
        shared=shared,
        growth=2,
        quant_noise=quant_noise,
    ),
    "growth1_1x1": lambda C_in, C_out, bits, C_fixed, stride, affine, shared, quant_noise=False: GrowthConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        1,
        stride,
        groups=1,
        affine=affine,
        shared=shared,
        growth=1,
        quant_noise=quant_noise,
    ),
    "growth1_3x3": lambda C_in, C_out, bits, C_fixed, stride, affine, shared, quant_noise=False: GrowthConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        3,
        stride,
        groups=1,
        affine=affine,
        shared=shared,
        growth=1,
        quant_noise=quant_noise,
    ),
    "growth1_5x5": lambda C_in, C_out, bits, C_fixed, stride, affine, shared, quant_noise=False: GrowthConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        5,
        stride,
        1,
        groups=1,
        affine=affine,
        shared=shared,
        growth=1,
        quant_noise=quant_noise,
    ),
}

class IncreaseConv(BaseConv):
    """Standard conv
    ReLU - Conv - BN
    """

    def __init__(
        self,
        C_in,
        C_out,
        bits,
        C_fixed,
        kernel_size,
        stride,
        dilation=1,
        groups=1,
        affine=False,
        shared=False,
        quant_noise=False,
        increase_rate=2
    ):
        super().__init__(shared=shared, quant_noise=quant_noise)
        self.increase_rate = increase_rate
        self.net = nn.Sequential(
            # nn.BatchNorm2d(C_in),
            self.conv_func(
                in_channels=C_in * increase_rate,
                out_channels=C_out,
                bits=bits,
                kernel_size=kernel_size,
                stride=stride,
                padding="same",
                dilation=dilation,
                groups=groups,
                bias=affine,
            )
        )

    def forward(self, x):
        return self.net(x.repeat(1, self.increase_rate, 1, 1)) * torch.sum(self.alphas)

class DecreaseConv(BaseConv):
    """Standard conv
    ReLU - Conv - BN
    """

    def __init__(
        self,
        C_in,
        C_out,
        bits,
        C_fixed,
        kernel_size,
        stride,
        dilation=1,
        groups=1,
        affine=False,
        shared=False,
        quant_noise=False,
        decrease_rate=2
    ):
        super().__init__(shared=shared, quant_noise=quant_noise)
        assert C_out % decrease_rate == 0, f"Wrong parameters C_out={C_out}, decrease_rate={decrease_rate}"
        self.decrease_rate = decrease_rate
        self.net = nn.Sequential(
            # nn.BatchNorm2d(C_in),
            self.conv_func(
                in_channels=C_in,
                out_channels=C_out // decrease_rate,
                bits=bits,
                kernel_size=kernel_size,
                stride=stride,
                padding="same",
                dilation=dilation,
                groups=groups,
                bias=affine,
            )
        )

    def forward(self, x):
        return self.net(x).repeat(1, self.decrease_rate, 1, 1) * torch.sum(self.alphas)

class GrowthConv(BaseConv):
    """Standard conv -C_in -> C_in*growth -> C_in"""

    def __init__(
        self,
        C_in,
        C_out,
        bits,
        C_fixed,
        kernel_size,
        stride,
        dilation=1,
        groups=1,
        affine=False,
        growth=2,
        shared=False,
        quant_noise=False,
    ):
        super().__init__(shared=shared, quant_noise=quant_noise)
        self.net = nn.Sequential(
            # nn.BatchNorm2d(C_in),
            self.conv_func(
                in_channels=C_in,
                out_channels=C_fixed * growth,
                bits=bits,
                kernel_size=kernel_size,
                stride=stride,
                padding="same",
                dilation=dilation,
                groups=groups,
                bias=affine,
            ),
            # nn.BatchNorm2d(C_in),
            self.conv_func(
                in_channels=C_fixed * growth,
                out_channels=C_out,
                bits=bits,
                kernel_size=kernel_size,
                stride=stride,
                padding="same",
                dilation=dilation,
                groups=groups,
                bias=affine,
            ),
            # nn.ReLU(),
            # nn.BatchNorm2d(C_out, affine=True),
        )

    def forward(self, x):
        return self.net(x) * torch.sum(self.alphas)

class SimpleConv(BaseConv):
    """Standard conv
    ReLU - Conv - BN
    """

    def __init__(
        self,
        C_in,
        C_out,
        bits,
        C_fixed,
        kernel_size,
        stride,
        dilation=1,
        groups=1,
        affine=False,
        shared=False,
        quant_noise=False,
        padding="same"
    ):
        super().__init__(shared=shared, quant_noise=quant_noise)
        self.net = nn.Sequential(
            # nn.BatchNorm2d(C_in),
            self.conv_func(
                in_channels=C_in,
                out_channels=C_out,
                bits=bits,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=affine,
            )
        )

    def forward(self, x):
        return self.net(x) * torch.sum(self.alphas)

class FacConv(BaseConv):
    """Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """

    def __init__(
        self,
        C_in,
        C_out,
        bits,
        C_fixed,
        kernel_length,
        stride,
        padding,
        affine=True,
        shared=False,
        quant_noise=False,
    ):
        super().__init__(shared=shared, quant_noise=quant_noise)
        self.net = nn.Sequential(
            # nn.BatchNorm2d(C_in),
            self.conv_func(
                in_channels=C_in,
                out_channels=C_fixed,
                bits=bits,
                kernel_size=(kernel_length, 1),
                stride=stride,
                padding=(padding, 0),
                dilation=1,
                bias=False,
            ),
            self.conv_func(
                in_channels=C_fixed,
                out_channels=C_out,
                bits=bits,
                kernel_size=(1, kernel_length),
                stride=stride,
                padding=(0, padding),
                dilation=1,
                bias=False,
            ),
        )

    def forward(self, x):
        return self.net(x) * torch.sum(self.alphas)


def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        # per data point mask; assuming x in cuda.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)

    return x


class Identity(BaseConv):
    def __init__(self, shared, quant_noise):
        super().__init__(shared=shared, quant_noise=quant_noise)

    def forward(self, x):
        return x * torch.sum(self.alphas)


class Zero(BaseConv):
    def __init__(self, stride, zero=1e-25, shared=True):
        super().__init__(shared=shared)
        self.stride = stride
        self.zero = zero

    def forward(self, x):
        if self.stride == 1:
            return x * self.zero

        # re-sizing by stride
        return (
            x[:, :, :: self.stride, :: self.stride]
            * self.zero
            * torch.sum(self.alphas)
        )


class MixedOp(nn.Module):
    """Creates mixed operation of specific block type."""

    def __init__(
        self,
        C_in,
        C_out,
        bits,
        C_fixed,
        gene_type,
        stride=1,
        quant_noise=False,
        primitives=None,
    ):
        super().__init__()
        self._ops = nn.ModuleList()
        self.bits = bits
        primitives = gt.PRIMITIVES_SR if primitives is None else primitives
        for primitive in primitives[gene_type]:
            func = OPS[primitive](
                C_in,
                C_out,
                bits,
                C_fixed,
                stride,
                affine=False,
                shared=True,
                quant_noise=quant_noise,
            )
            self._ops.append(func)

    def forward(self, x, alpha_vec):
        """
        Args:
            x: input
            weights: weight for each operation
        """

        outs = []
        # print("ALPHA SHAPE:", alpha_vec.shape)
        for alphas, op in zip(alpha_vec.chunk(len(self._ops)), self._ops):
            # print("ALPHA BIT:", alphas.shape)
            if (alphas != 0).any():
                op.set_alphas(alphas)
                outs.append(op(x))
            # print("ALPHA OUTS:", outs[-1].shape)
        return sum(outs)

    def fetch_weighted_info(self, alpha_vec):
        flops = 0
        memory = 0
        for alphas, op in zip(alpha_vec.chunk(len(self._ops)), self._ops):
            op.set_alphas(alphas)
            f, m = op.fetch_info()
            flops += f
            memory += m

        return flops, memory


if __name__ == "__main__":
    # TEST FLOPS TRACKING

    print("Get blocks weights for 4 bits")
    random_image = torch.randn(3, 36, 256, 256)

    C = 36
    # Keep stride 1 for all but GrowCo
    stride = 1

    PRIMITIVES_SR = [
        "skip_connect",  # identity
        "conv_5x1_1x5",
        "conv_3x1_1x3",
        "simple_3x3",
        "simple_1x1",
        "simple_5x5",
        "simple_1x1_grouped_full",
        "simple_3x3_grouped_full",
        "simple_5x5_grouped_full",
        "simple_1x1_grouped_3",
        "simple_3x3_grouped_3",
        "simple_5x5_grouped_3",
        "DWS_3x3",
        "DWS_5x5",
        "growth2_5x5",
        "growth2_3x3",
        "decenc_3x3_4",
        "decenc_3x3_2",
        "decenc_5x5_2",
        "decenc_5x5_8",
        "decenc_3x3_8",
        "decenc_3x3_4_g3",
        "decenc_3x3_2_g3",
        "decenc_5x5_2_g3",
        "decenc_5x5_8_g3",
        "decenc_3x3_8_g3",
    ]

    names = []
    flops = []
    for i, primitive in enumerate(PRIMITIVES_SR):
        func = OPS[primitive](C, C, [8], C, stride, affine=False)
        conv = func

        x = conv(random_image)
        flops.append(conv.fetch_info()[0])
        names.append(primitive)
        print(i + 1, primitive, f"FLOPS: {conv.fetch_info()[0]:.2e}")

    max_flops = max(flops)
    flops_normalized = [f / max_flops for f in flops]
    names_sorted = [
        n
        for f, n in sorted(
            zip(flops_normalized, names), key=lambda pair: pair[0]
        )
    ]
    flops_normalized = sorted(flops_normalized)
    print("\n## SORTED AND NORMALIZED ##\n")
    for n, f in zip(names_sorted, flops_normalized):
        print(n, "FLOPS:", round(f, 5))
