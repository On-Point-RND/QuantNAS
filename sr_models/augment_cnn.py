""" CNN for network augmentation """
import torch
import torch.nn as nn
import genotypes as gt
from sr_models.quant_conv_lsq import QAConv2d
from sr_models.ADN import AdaptiveNormalization as ADN


def summer(values, increments):
    return (v + i for v, i in zip(values, increments))


class Residual(nn.Module):
    def __init__(self, skip, body, skip_mode=True):
        super().__init__()
        self.skip = skip
        self.body = body
        self.skip_mode = skip_mode

        self.adn = ADN(36, skip_mode=skip_mode)

    def forward(self, x):
        def func(x):
            return self.skip(x) + self.body(x)

        return self.adn(x, func, x)

    def fetch_weighted_info(self):
        flops = 0
        memory = 0
        for layer in (self.skip, self.body):
            flops, memory = summer((flops, memory), layer.fetch_info())
        return flops, memory


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
        self.head = gt.to_dag_sr(
            self.c_fixed, genotype.head, gene_type="head", c_in=c_in
        )

        self.body = nn.ModuleList()
        for _ in range(blocks):
            b = gt.to_dag_sr(
                self.c_fixed, genotype.body, gene_type="body", c_in=c_in
            )
            s = gt.to_dag_sr(
                self.c_fixed, genotype.skip, gene_type="skip", c_in=c_in
            )
            self.body.append(Residual(s, b, skip_mode=skip_mode))

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

    def forward(self, x):

        init = self.head(x)
        x = init

        def func(x):
            # xs = 0
            for cell in self.body:
                x = cell(x)
                # xs += x
            return x

        x = self.upsample(self.adn_one(x, func, init))
        #tail = self.adn(_two(x, self.tail, x)
        return self.tail(x) + x

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
