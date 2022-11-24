import torch
import torch.nn as nn


def mean_channels(F):
    assert F.dim() == 4
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert F.dim() == 4
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(
        2, keepdim=True
    ) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


class AdaptiveNormalization(nn.Module):
    def __init__(self, filters, skip_mode=False):
        super().__init__()

        self.skip_mode = skip_mode
        if not skip_mode:
            self.bn = nn.BatchNorm2d(filters)
            self.phi = nn.Conv2d(1, 1, 1, 1, 0, bias=True)
            self.phi.weight.data.fill_(1.5)
            self.phi.bias.data.fill_(0)

        reduction = 3
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(filters, filters // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters // reduction, filters, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def skip_f(self, x, func, skip):
        x_nm = func(x)
        out = x_nm + skip
        return out

    def adn_f(self, x, func, skip):
        s = torch.std(skip, dim=[1, 2, 3], keepdim=True)
        self.s = self.phi(s)
        x_nm = self.bn(x)
        x_nm = func(x_nm)
        out = x_nm * self.s + skip
        return out

    def cca_f(self, x, func, skip):
        x = func(x)
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y + skip

    def forward(self, x, func, skip):
        if self.skip_mode:
            return self.skip_f(x, func, skip)
        else:
            return self.cca_f(x, func, skip)
