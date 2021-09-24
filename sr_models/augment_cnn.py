""" CNN for network augmentation """
import torch
import torch.nn as nn
import torch.nn.functional as F
from sr_models import ops_flops as ops
import genotypes as gt


class AugmentCNN(nn.Module):
    """Augmented CNN model"""

    def __init__(self, c_init, repeat_factor, genotype):

        """
        Args:
            input_size: size of height and width (assuming height = width)
            C_in: # of input channels
            C: # of starting model channels
        """
        super().__init__()

        self.c_fixed = c_init * repeat_factor
        self.repeat_factor = repeat_factor

        self.dag = gt.to_dag_sr(self.c_fixed, genotype.normal)

        self.pixelup = nn.Sequential(
            nn.PixelShuffle(int(repeat_factor ** (1 / 2))), nn.PReLU()
        )

    def forward(self, x):

        state_zero = torch.repeat_interleave(x, self.repeat_factor, 1)
        self.assertion_in(state_zero.shape)

        s_cur = state_zero
        for op in self.dag:
            s_cur = op(s_cur)

        self.assertion_in(s_cur.shape)
        x = self.pixelup(s_cur)
        # x_residual = self.pixelup(state_zero)
        return x

    def drop_path_prob(self, p):
        """Set drop path probability"""
        for module in self.modules():
            if isinstance(module, ops.DropPath_):
                module.p = p

    def assertion_in(self, size_in):
        assert int(size_in[1]) == int(
            self.c_fixed
        ), f"Input size {size_in}, does not match fixed channels {self.c_fixed}"
