""" CNN cell for architecture search """
import torch
import torch.nn as nn
from sr_models import ops_flops as ops


def is_square(apositiveint):
    x = apositiveint // 2
    seen = set([x])
    while x * x != apositiveint:
        x = (x + (apositiveint // x)) // 2
        if x in seen:
            return False
        seen.add(x)
    return True


class SearchArch(nn.Module):
    """Cell for search
    Each edge is mixed and continuous relaxed.
    """

    def __init__(self, n_nodes, c_init, repeat_factor):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            C_fixed: # of channels to work with
            C_init:  # of initial channels, usually 3
        """

        super().__init__()
        self.n_nodes = n_nodes
        self.c_fixed = c_init * repeat_factor
        self.repeat_factor = repeat_factor
        self.c_init = c_init

        # Used for soft edge experiments to stabilize training after warm up
        assert is_square(repeat_factor), "Repear factor should be a square of N"

        # generate dag
        self.dag = nn.ModuleList()
        print("INITIALIZING MODEL's DAG")
        for _ in range(self.n_nodes):
            self.dag.append(ops.MixedOp(self.c_fixed, 1))

        self.pixelup = nn.Sequential(
            nn.PixelShuffle(int(repeat_factor ** (1 / 2))), nn.PReLU()
        )

    def assertion_in(self, size_in):
        assert int(size_in[1]) == int(
            self.c_fixed
        ), f"Input size {size_in}, does not match fixed channels {self.c_fixed}"

    def forward(self, x, w_dag):
        state_zero = torch.repeat_interleave(x, self.repeat_factor, 1)
        self.assertion_in(state_zero.shape)
        s_cur = state_zero

        for i, (edge, alphas) in enumerate(zip(self.dag, w_dag)):
            s_cur = edge(s_cur, alphas)

        self.assertion_in(s_cur.shape)

        out = self.pixelup(s_cur)
        x_residual = self.pixelup(state_zero)

        return out + x_residual

    def fetch_weighted_flops_and_memory(self, w_dag):
        total_flops = 0
        total_memory = 0

        for edge, alphas in zip(self.dag, w_dag):
            total_flops += edge.fetch_weighted_memory(alphas)
            total_memory += edge.fetch_weighted_memory(alphas)

        return total_flops, total_memory
