""" CNN for found architechture """

import torch
import torch.nn as nn
import genotypes as gt
from sr_models.quant_conv_lsq import QAConv2d
from sr_models.ADN import AdaptiveNormalization as ADN


class ResidualSplitter(nn.Module):
    def __init__(self, body, f_fixed, skip_mode=True):
        super().__init__()
        self.body = body
        self.f_fixed = f_fixed
        self.skip_mode = skip_mode
        self.adn = ADN(36, skip_mode=skip_mode)

    def forward(self, x):
        def func(z):
            distilled_list = []
            for i, c in enumerate(self.body):
                if (i + 1) != len(self.body):
                    # device  = x.device
                    z = c(z)

                    # if one before last
                    if (i + 1) == len(self.body) - 1:
                        distilled_list.append(z)
                    else:
                        distilled, z = torch.split(
                            z, (self.f_fixed // 2, self.f_fixed // 2), dim=1
                        )
                        distilled_list.append(distilled)

            z = torch.cat(distilled_list, dim=1)
            return c(z)

        return self.adn(x, func, x)

    def fetch_weighted_info(self):

        flops, memory = self.body.fetch_info()
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

        body = []
        for _ in range(blocks):
            b = gt.to_dag_sr(
                self.c_fixed, genotype.body, gene_type="body", c_in=c_in
            )

        body.append(ResidualSplitter(b, self.c_fixed, skip_mode=skip_mode))
        self.body = nn.Sequential(*body)

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

        x = self.adn_one(x, self.body, init)
        x = self.upsample(x)
        tail = self.adn_two(x, self.tail, x)
        return tail

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
        return sum_flops, sum_memory
