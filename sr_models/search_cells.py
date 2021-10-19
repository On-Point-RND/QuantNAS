""" CNN cell for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
from sr_models import ops_flops as ops


class Residual(nn.Module):
    def __init__(self, skip, body):
        super().__init__()
        self.skip = skip
        self.body = body

    def forward(self, x, s_weights, b_weights):
        return (self.skip(x, s_weights) + self.body(x, b_weights)) * 0.2 + x


class SharedBlock(nn.Module):
    def __init__(self, c_fixed, c_init, num_layers, gene_type="head", scale=4):
        super().__init__()

        self.net = nn.ModuleList()
        self.name = gene_type
        for i in range(num_layers):
            (
                c_in,
                c_out,
            ) = (c_fixed, c_fixed)

            if i == 0 and gene_type == "head":
                c_in = c_init
            elif i + 1 == num_layers and gene_type == "tail":
                c_out = c_init
            elif i == 0 and gene_type == "tail":
                c_in = c_init

            elif gene_type == "upsample":
                c_in = c_fixed
                c_out = 3 * (scale ** 2)
            else:
                c_in = c_fixed
                c_out = c_fixed

            self.net.append(ops.MixedOp(c_in, c_out, c_fixed, gene_type))

    def forward(self, x, alphas):
        print(self.name)
        for layer, a_w in zip(self.net, alphas):
            x = layer(x, a_w)
        return x

    def fetch_info(self, x, alphas):
        flops = 0
        memory = 0
        for layer in self.net:
            flops += layer.fetch_weighted_flops(x, alphas)
            memory += layer.fetch_weighted_memory(x, alphas)
        return flops, memory


class SearchArch(nn.Module):
    def __init__(self, c_init, c_fixed, scale, arch_pattern, body_cells):
        """
        Args:
            body_cells: # of intermediate body blocks
            c_fixed: # of channels to work with
            c_init:  # of initial channels, usually 3
            scale: # downsampling scale

            arch_pattern : {'head':2, 'body':4, 'tail':3, 'skip'=1, 'upsample'=1}
        """

        super().__init__()
        self.body_cells = body_cells
        self.c_fixed = c_fixed  # 32, 64 etc
        self.c_init = c_init

        # Generate searchable network with shared weights
        self.head = SharedBlock(
            c_fixed, c_init, arch_pattern["head"], gene_type="head"
        )

        self.body = nn.ModuleList()
        for _ in range(body_cells):
            b = SharedBlock(
                c_fixed, c_init, arch_pattern["body"], gene_type="body"
            )
            s = SharedBlock(
                c_fixed, c_init, arch_pattern["skip"], gene_type="skip"
            )
            self.body.append(Residual(s, b))

        self.upsample = SharedBlock(
            c_fixed, c_init, arch_pattern["upsample"], gene_type="upsample"
        )
        self.pixel_up = nn.PixelShuffle(scale)

        self.tail = SharedBlock(
            c_fixed, c_init, arch_pattern["tail"], gene_type="tail"
        )

    def forward(self, x, alphas):
        x = self.head(x, alphas["head"])
        for cell in self.body:
            x = cell(x, alphas["body"], alphas["skip"])
        x = self.upsample(self.upsample(x, alphas["upsample"])) + self.tail(
            x, alphas["tail"]
        )
        return x

    def fetch_weighted_flops_and_memory(self, alphas):
        total_flops = 0
        total_memory = 0

        for func, name in [
            (self.head, "head"),
            (self.skip, "skip"),
            (self.tail, "tail"),
            (self.upsample, "upsample"),
        ]:
            f, m = func.fetch_info(alphas[name])
            total_flops += f
            total_memory += m

        for cell in self.body:
            f, m = cell.fetch_info(alphas["body"])
            total_flops += f
            total_memory += m

        return total_flops, total_memory