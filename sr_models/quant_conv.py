# Quant module adapted from:
# https://github.com/zhaoweicai/EdMIPS/blob/master/models/quant_module.py

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

gaussian_steps = {
    1: 1.5957691216057308,
    2: 1.0069544482241686,
    3: 0.6148642700335044,
    4: 0.3643855200347445,
    8: 0.0365160781639106,
    16: 0.000197312280995046,
}

hwgq_steps = {
    1: 1.0576462792297525,
    2: 0.6356366866203315,
    3: 0.3720645813370479,
    4: 0.21305606790772952,
    8: 0.020300567823662602,
    16: 9.714825915156693e-05,
}


class _gauss_quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, step, bit):
        lvls = 2 ** bit / 2
        alpha = x.std().item()
        step *= alpha
        y = (torch.round(x / step + 0.5) - 0.5) * step
        thr = (lvls - 0.5) * step
        y = y.clamp(min=-thr, max=thr)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class _gauss_quantize_resclaed_step(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, step, bit):
        lvls = 2 ** bit / 2
        y = (torch.round(x / step + 0.5) - 0.5) * step
        thr = (lvls - 0.5) * step
        y = y.clamp(min=-thr, max=thr)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class _hwgq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, step):
        y = torch.round(x / step) * step
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class HWGQ(nn.Module):
    def __init__(self, bit=2):
        super(HWGQ, self).__init__()
        self.bit = bit
        if bit < 32:
            self.step = hwgq_steps[bit]
        else:
            self.step = None

    def forward(self, x):
        if self.bit >= 32:
            return nn.functional.relu(x)
        lvls = float(2 ** self.bit - 1)
        clip_thr = self.step * lvls
        y = x.clamp(min=0.0, max=clip_thr)
        return _hwgq.apply(y, self.step)


class QuantConv(nn.Conv2d):

    """
    Flops are computed for square kernel
    FLOPs = 2 x Cin x Cout x k**2 x Wout x Hout / groups
    We use 2 because 1 for multiplocation and 1 for addition

    Hout = Hin + 2*padding[0] - dilation[0] x (kernel[0]-1)-1
          --------------------------------------------------- + 1
                                stride
    Wout same as above


    NOTE: We do not account for bias term


    """

    def __init__(self, **kwargs):
        super(QuantConv, self).__init__(**kwargs)
        self.kernel = self.to_tuple(self.kernel_size)
        self.stride = self.to_tuple(self.stride)
        self.padding = self.to_tuple(self.padding)
        self.dilation = self.to_tuple(self.dilation)
        # complexities
        # FLOPs = 2 x Cin x Cout x k**2 x Wout x Hout / groups
        self.param_size = (
            2
            * self.in_channels
            * self.out_channels
            * self.kernel[0]
            * self.kernel[1]
            / self.groups
        )  # * 1e-6  # stil unsure why we use 1e-6
        self.register_buffer("flops", torch.tensor(0, dtype=torch.float))
        self.register_buffer("memory_size", torch.tensor(0, dtype=torch.float))

    def to_tuple(self, value):
        if type(value) == int:
            return (value, value)
        if type(value) == tuple:
            return value

    def forward(self, input_x, bit):
        """
        BATCH x C x W x H

        """
        # get the same device to avoid errors
        device = input_x.device

        c_in, w_in, h_in = input_x.shape[1], input_x.shape[2], input_x.shape[3]

        w_out = self.compute_out(w_in, "w")
        h_out = self.compute_out(h_in, "h")

        tmp = torch.tensor(c_in * w_in * h_in, dtype=torch.float).to(device)
        self.memory_size.copy_(tmp)
        tmp = torch.tensor(
            self.param_size * w_out * h_out, dtype=torch.float
        ).to(device)
        self.flops.copy_(tmp)
        del tmp

        if bit == 32:
            quant_weight = self.weight
        else:
            step = gaussian_steps[bit]
            quant_weight = _gauss_quantize.apply(self.weight, step, bit)

        out = out = F.conv2d(
            input_x,
            quant_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        return out

    def compute_out(self, input_size, spatial="w"):

        if spatial == "w":
            idx = 0
        if spatial == "h":
            idx = 1
        return int(
            (
                input_size
                + 2 * self.padding[idx]
                - self.dilation[idx] * (self.kernel[idx] - 1)
                - 1
            )
            / self.stride[idx]
            + 1
        )

    def _fetch_info(self):
        return self.flops.item(), self.memory_size.item()


# USE Instead of CNN + ReLU Block for final quantized model
class QAConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(QAConv2d, self).__init__()
        self.bit = kwargs.pop("bits")[0]
        self.bit_orig = self.bit
        self.activ = HWGQ(self.bit)
        self.conv = QuantConv(**kwargs)

    def set_fp(self, bit=32):
        self.activ = HWGQ(bit)
        self.bit = bit

    def set_quant(self):
        self.activ = HWGQ(self.bit_orig)
        self.bit = self.bit_orig

    def forward(self, input):
        out = self.activ(input)
        out = self.conv(out, self.bit)
        return out

    def _fetch_info(self):
        f, m = self.conv._fetch_info()
        return f * self.bit, m * self.bit


class SharedQAConv2d(nn.Module):
    def __init__(self, **kwargs):
        super(SharedQAConv2d, self).__init__()
        self.bits = kwargs.pop("bits")
        self.acts = [HWGQ(bit) for bit in self.bits]
        self.conv = QuantConv(**kwargs)
        self.alphas = [1] * len(self.bits)

    def forward(self, input):
        outs = []
        for alpha, bit, act in zip(self.alphas, self.bits, self.acts):
            out = act(input)
            out = alpha * self.conv(out, bit)
        outs.append(out)
        return sum(outs)

    def _fetch_info(self):
        bit_ops, mem = 0, 0
        b, m = self.conv.fetch_info()
        for alpha, bit in zip(self.bits, self.alphas):
            bit_ops += alpha * b * bit
            mem += alpha * m * bit
        return bit_ops, mem


class BaseConv(nn.Module):
    def __init__(self, *args, **kwargs):
        shared = kwargs.pop("shared")
        super(BaseConv, self).__init__()
        if shared:
            self.conv_func = SharedQAConv2d
        else:
            self.conv_func = QAConv2d

    def fetch_info(self):
        sum_flops = 0
        sum_memory = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                f, mem = m._fetch_info()
                sum_flops += f
                sum_memory += mem

        return sum_flops, sum_memory

