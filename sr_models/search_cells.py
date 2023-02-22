""" CNN cell for architecture search """
import torch.nn as nn
import torch
from sr_models import quant_ops as ops
from sr_models.ADN import AdaptiveNormalization as ADN
from sr_models.RFDN.block import ESA


def summer(values, increments):
    return (v + i for v, i in zip(values, increments))


class Residual(nn.Module):
    def __init__(self, skip, body, c_out, skip_mode=True):
        super(Residual, self).__init__()
        self.skip = skip
        self.cum_channels = nn.Conv2d((c_out // 2) * (len(body.net)), c_out, 1)
        self.body = body
        self.esa = ESA(c_out)

    def forward(self, x, b_weights, s_weights):
        def func(x):
            return self.body_split(x, b_weights, s_weights) 

        return self.esa(func(x)) 

    def body_split(self, x, b_alphas, s_alphas):
        splits = []
        for i in range(len(self.body.net)):
            if i < len(self.body.net) - 1:
                splits += [self.skip.net[i](x, s_alphas[i])]
                x = x + self.body.net[i](x, b_alphas[i])
            else:
                x = self.body.net[i](x, b_alphas[i]) 
        splits += [x]
        output = self.cum_channels(torch.cat(splits, dim=1))
        return output

    def fetch_info(self, b_weights, s_weights):
        flops = 0
        memory = 0
        for layer, weights in zip(
            (self.body, self.skip), (b_weights, s_weights)
        ):

            flops, memory = summer((flops, memory), layer.fetch_info(weights))
        return flops, memory


class CommonBlock(nn.Module):
    def __init__(
        self,
        c_fixed,
        c_init,
        bits,
        num_layers,
        gene_type="tail",
        scale=4,
        quant_noise=False,
        primitives=None,
    ):
        """
        Creates list of blocks of specific gene_type.
        """
        super(CommonBlock, self).__init__()

        self.net = nn.ModuleList()
        self.name = gene_type
        for i in range(num_layers):
            (
                c_in,
                c_out,
            ) = (c_fixed, c_fixed)

            # if i == 0 and gene_type == "head":
            #     c_in = c_init
            if gene_type == "tail":
                c_out = c_init
                c_in = c_init
            elif gene_type == "upsample":
                c_in = c_fixed
                c_out = 3 * (scale**2)
            elif (gene_type == "skip") or (i == num_layers - 1 and gene_type == "body"):
                c_out = c_fixed // 2
            else:
                c_in = c_fixed
                c_out = c_fixed

            self.net.append(
                ops.MixedOp(
                    c_in,
                    c_out,
                    bits,
                    c_fixed,
                    gene_type,
                    quant_noise=quant_noise,
                    primitives=primitives,
                )
            )

    def forward(self, x, alphas):
        for layer, a_w in zip(self.net, alphas):
            x = layer(x, a_w)
        return x

    def fetch_info(self, alphas):
        flops = 0
        memory = 0
        for layer, weight in zip(self.net, alphas):
            flops, memory = summer(
                (flops, memory), layer.fetch_weighted_info(weight)
            )
        return flops, memory


class SearchArch(nn.Module):
    def __init__(
        self,
        c_init,
        c_fixed,
        bits,
        scale,
        arch_pattern,
        body_cells,
        quant_noise=False,
        skip_mode=True,
        primitives=None,
    ):
        """
        SuperNet.
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
        self.skip_mode = skip_mode
        self.primitives = primitives
        # Generate searchable network with shared weights
        # self.head = CommonBlock(
        #     c_fixed,
        #     c_init,
        #     bits,
        #     arch_pattern["head"],
        #     gene_type="head",
        #     quant_noise=quant_noise,
        #     primitives=primitives,
        # )

        self.body = nn.ModuleList()
        for _ in range(body_cells):
            b = CommonBlock(
                c_fixed,
                c_init,
                bits,
                arch_pattern["body"],
                gene_type="body",
                quant_noise=quant_noise,
                primitives=primitives,
            )
            s = CommonBlock(
                c_fixed,
                c_init,
                bits,
                arch_pattern["body"] - 1,
                gene_type="skip",
                quant_noise=quant_noise,
                primitives=primitives,
            )
            self.body.append(Residual(s, b, c_out=c_fixed, skip_mode=skip_mode))

        self.upsample = CommonBlock(
            c_fixed,
            c_init,
            bits,
            arch_pattern["upsample"],
            gene_type="upsample",
            quant_noise=quant_noise,
            primitives=primitives,
        )
        self.pixel_up = nn.PixelShuffle(scale)

        self.tail = CommonBlock(
            c_fixed,
            c_init,
            bits,
            arch_pattern["tail"],
            gene_type="tail",
            quant_noise=quant_noise,
            primitives=primitives,
        )

        self.adn_one = ADN(36, skip_mode=skip_mode)
        self.adn_two = ADN(3, skip_mode=skip_mode)

        self.c = nn.Conv2d(self.c_fixed * body_cells, self.c_fixed, 1, padding="same")
        self.c2 = nn.Conv2d(self.c_fixed, self.c_fixed, 3, padding="same")

    def forward(self, x, alphas):

        x = x.repeat(1, self.c_fixed // 3, 1, 1)
        head_skip = x

        def func_body(x):
            concat_skips = []
            for i in range(len(self.body)):
                x = self.body[i](x, alphas["body"][i], alphas["skip"][i])
                concat_skips += [x]
            concat_skips = torch.cat(concat_skips, dim=1)
            x = torch.nn.functional.leaky_relu(self.c(concat_skips), negative_slope=0.05)
            return self.c2(x)

        x = func_body(x) + head_skip
        x = self.pixel_up(self.upsample(x, alphas["upsample"]))

        out = x + self.tail(x, alphas["tail"])
        return out

    def fetch_weighted_flops_and_memory(self, alphas):
        flops = 0
        memory = 0
        for func, name in [
            # (self.head, "head"),
            (self.tail, "tail"),
            (self.upsample, "upsample"),
        ]:

            f, m = func.fetch_info(alphas[name])
            flops += f
            memory += m

        for i in range(len(self.body)):
            f, m = self.body[i].fetch_info(alphas["body"][i], alphas["skip"][i])
            flops += f
            memory += m

        return flops, memory
