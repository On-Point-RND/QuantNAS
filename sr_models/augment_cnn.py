""" CNN for network augmentation """
import torch
import torch.nn as nn
import genotypes as gt
from sr_models.quant_conv_lsq import QAConv2d
from sr_models.ADN import AdaptiveNormalization as ADN
from sr_models.RFDN.block import ESA
from .quant_ops import OPS

def summer(values, increments):
    return (v + i for v, i in zip(values, increments))

SUPPORT_CONV_BIT = 8

class Residual(nn.Module):
    def __init__(self, skip, body, c_out, skip_mode=True):
        super().__init__()
        self.skip = skip
        self.cum_channels = nn.Conv2d((c_out // 2) * (len(body)), c_out, 1) #OPS["simple_1x1"]((c_out // 2) * (len(body)), c_out, [SUPPORT_CONV_BIT], None, 1, False, shared=False, quant_noise=False) 
        self.body = body
        self.skip_mode = skip_mode
        self.adn = ADN(c_out, skip_mode=skip_mode)
        # self.esa = ESA(c_out, [8], shared=False)

    def forward(self, x):
        def func(x):
            return self.body_split(x)

        return self.adn(x, func, x) 

    def body_split(self, x):
        splits = []
        for i in range(len(self.body)):
            if i < len(self.body) - 1:
                splits += [self.skip[i](x)]
                x = x + self.body[i](x)
            else:
                x = self.body[i](x) 
        splits += [x]
        output = self.cum_channels(torch.cat(splits, dim=1))
        return output


class AugmentCNN(nn.Module):
    """Searched CNN model for final training"""

    def __init__(
        self, c_in, c_fixed, scale, genotype, blocks=4, skip_mode=True
    ):

        """
        Args:
            input_size: size of height and width (assuming height = width)
            C_in: # of input channels
            C: # of starting model channels
        """
        super().__init__()
        self.skip_mode = skip_mode
        self.c_fixed = c_fixed
        # self.head = gt.to_dag_sr(
        #     self.c_fixed, genotype.head, gene_type="head", c_in=c_in
        # )

        self.body = nn.ModuleList()
        for i in range(blocks):
            b = gt.to_dag_sr(
                self.c_fixed, genotype.body[i], gene_type="body", c_in=c_in
            )
            s = gt.to_dag_sr(
                self.c_fixed, genotype.skip[i], gene_type="skip", c_in=c_in
            )
            assert len(genotype.skip[i]) == len(genotype.body[i]) - 1
            self.body.append(Residual(s, b, c_out=self.c_fixed, skip_mode=skip_mode))

        upsample = gt.to_dag_sr(
            self.c_fixed, genotype.upsample, gene_type="upsample"
        )

        self.upsample = nn.Sequential(upsample, nn.PixelShuffle(scale))
        self.tail = gt.to_dag_sr(
            self.c_fixed, genotype.tail, gene_type="tail", c_in=c_in
        )
        self.quant_mode = True

        self.adn_one = ADN(36, skip_mode=skip_mode)
        self.adn_two = ADN(3, skip_mode=skip_mode)
        self.c = nn.Conv2d(self.c_fixed * blocks, self.c_fixed, 1, padding="same")
        self.c2 = nn.Conv2d(self.c_fixed, self.c_fixed, 3, padding="same")
        # self.c = OPS["simple_1x1"](self.c_fixed * blocks, self.c_fixed, [SUPPORT_CONV_BIT], self.c_fixed, 1, False, shared=False, quant_noise=False)
        # self.c2 = OPS["simple_3x3"](self.c_fixed, self.c_fixed, [SUPPORT_CONV_BIT], self.c_fixed, 1, False, shared=False, quant_noise=False)

    def forward(self, x):

        x = x.repeat(1, self.c_fixed // 3, 1, 1)
        head_skip = x

        def func(x):
            concat_skips = []
            for cell in self.body:
                x = cell(x)
                concat_skips += [x]
            concat_skips = torch.cat(concat_skips, dim=1)
            x = torch.nn.functional.leaky_relu(self.c(concat_skips), negative_slope=0.05)
            # x = self.c(concat_skips)
            return self.c2(x)

        x = self.upsample(func(x) + head_skip) 
        return x + self.tail(x)

    def set_fp(self):
        if self.quant_mode == True:
            for m in self.modules():
                if isinstance(m, QAConv2d):
                    m.set_fp()
            self.quant_mode = False

    def set_quant(self):
        if self.quant_mode == False:
            for m in self.modules():
                if isinstance(m, QAConv2d):
                    m.set_quant()
            self.quant_mode = True

    def fetch_info(self):
        sum_flops = 0
        sum_memory = 0
        for m in self.modules():
            if isinstance(m, QAConv2d):
                b, m = m._fetch_info()
                sum_flops += b
                sum_memory = m
        return (sum_flops, sum_memory)
