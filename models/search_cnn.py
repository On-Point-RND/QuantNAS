""" CNN for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.search_cells import SearchCell
import genotypes as gt
import logging


# Weighted Soft Edge in forward
# Gumbel Final flops are not correct

from models.gumbel_top2 import gumbel_top2k


class SearchCNN(nn.Module):
    """ Search CNN model """

    def __init__(
        self, C_in, C, n_classes, n_layers, n_nodes=4, stem_multiplier=3
    ):
        """
        Args:
            C_in: # of input channels
            C: # of starting model channels
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
            stem_multiplier
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers
        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False), nn.BatchNorm2d(C_cur)
        )

        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers // 3, 2 * n_layers // 3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            cell = SearchCell(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)

        self.edge_n = nn.ParameterList()
        self.edge_r = nn.ParameterList()

        for i in range(n_nodes):
            self.edge_n.append(nn.Parameter(torch.ones(i + 2)))
            self.edge_r.append(nn.Parameter(torch.ones(i + 2)))

    def forward(self, x, weights_normal, weights_reduce):
        s0 = s1 = self.stem(x)

        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        return logits

    def fetch_weighted_flops_and_memory(self, weights_normal, weights_reduce):
        total_flops = 0
        total_memory = 0

        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            flops, memory = cell.fetch_weighted_flops_and_memory(weights)
            total_flops += flops
            total_memory += memory

        return total_flops, total_memory


class SearchCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """

    def __init__(
        self,
        C_in,
        C,
        n_classes,
        n_layers,
        criterion,
        n_nodes=4,
        stem_multiplier=3,
        device_ids=None,
        use_soft_edge=False,
        alpha_selector="softmax",
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.criterion = criterion
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids

        # initialize architect parameters: alphas
        self.n_ops = len(gt.PRIMITIVES)

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        self.alphaselector = AlphaSelector(
            name=alpha_selector, use_soft_edge=use_soft_edge
        )
        self.softmax = AlphaSelector(
            name="softmax", use_soft_edge=use_soft_edge
        )

        self.use_soft_edge = use_soft_edge

        for i in range(n_nodes):
            self.alpha_normal.append(
                nn.Parameter(torch.ones(i + 2, self.n_ops) / self.n_ops)
            )
            self.alpha_reduce.append(
                nn.Parameter(torch.ones(i + 2, self.n_ops) / self.n_ops)
            )

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if "alpha" in n:
                self._alphas.append((n, p))

        self.net = SearchCNN(
            C_in, C, n_classes, n_layers, n_nodes, stem_multiplier
        )

    def forward(self, x, temperature=1, stable=False):

        if stable:
            func = self.softmax
        else:
            func = self.alphaselector

        weights_normal = [
            func(alpha, edge_w, temperature, dim=-1)
            for alpha, edge_w in zip(self.alpha_normal, self.net.edge_n)
        ]
        weights_reduce = [
            func(alpha, edge_w, temperature, dim=-1)
            for alpha, edge_w in zip(self.alpha_reduce, self.net.edge_r)
        ]

        out = self.net(x, weights_normal, weights_reduce)
        (flops, mem) = self.net.fetch_weighted_flops_and_memory(
            weights_normal, weights_reduce
        )
        return out, (flops, mem)

    def forward_current_best(self, x):

        weights_normal = [
            self.get_max(self.prod(a, self.net.edge_n[i]))
            for i, a in enumerate(self.alpha_normal)
        ]

        weights_reduce = [
            self.get_max(self.prod(a, self.net.edge_r[i]))
            for i, a in enumerate(self.alpha_reduce)
        ]

        return self.net(x, weights_normal, weights_reduce)

    def print_alphas(self, logger, temperature):

        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha, edge in zip(self.alpha_normal, self.net.edge_n):
            logger.info(self.alphaselector(alpha, edge, temperature, dim=-1))

        logger.info("\n# Alpha - reduce")
        for alpha, edge in zip(self.alpha_reduce, self.net.edge_r):
            logger.info(self.alphaselector(alpha, edge, temperature, dim=-1))
        logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def print_edges(self, logger):

        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### EDGES #######")
        logger.info("# EDGE W  - normal")
        for edge in self.net.edge_n:
            logger.info(F.softmax(edge, dim=-1))

        logger.info("\n# EDGE W - reduce")
        for edge in self.net.edge_r:
            logger.info(F.softmax(edge, dim=-1))
        logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        alpha_normal = [
            self.prod(a, self.net.edge_n[i])
            for i, a in enumerate(self.alpha_normal)
        ]
        alpha_reduce = [
            self.prod(a, self.net.edge_r[i])
            for i, a in enumerate(self.alpha_reduce)
        ]

        gene_normal = gt.parse(alpha_normal, k=2)
        gene_reduce = gt.parse(alpha_reduce, k=2)
        concat = range(2, 2 + self.n_nodes)  # concat all intermediate nodes

        return gt.Genotype(
            normal=gene_normal,
            normal_concat=concat,
            reduce=gene_reduce,
            reduce_concat=concat,
        )

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p

    def fetch_weighted_flops_and_memory(
        self,
    ):

        return self.net.fetch_weighted_flops_and_memory(
            self.get_weights_normal(F.softmax),
            self.get_weights_reduce(F.softmax),
        )

    def get_weights_normal(self, FN):
        return [
            FN(self.prod(a, self.net.edge_n[i]))
            for i, a in enumerate(self.alpha_normal)
        ]

    def get_weights_reduce(self, FN):
        return [
            FN(self.prod(a, self.net.edge_n[i]))
            for i, a in enumerate(self.alpha_reduce)
        ]

    def prod(self, vector, edge):
        if self.use_soft_edge:
            return (vector.T * F.softmax(edge)).T

        else:
            return vector

    def get_max(self, alpha, k=2, keep_weight=False):
        values, indices = alpha[:, :-1].max(1)
        ones = (values.unsqueeze(1) == alpha).type(torch.int)
        zero_rows = [
            i for i in range(alpha.shape[0]) if not i in values.topk(k).indices
        ]
        ones[zero_rows] = 0
        if keep_weight:
            return alpha * ones.detach()
        else:
            return ones.detach()

    def fetch_current_best_flops_and_memory(self):
        return self.net.fetch_weighted_flops_and_memory(
            self.get_weights_normal(self.get_max),
            self.get_weights_reduce(self.get_max),
        )


class AlphaSelector:
    def __init__(self, name="softmax", use_soft_edge=False):
        assert name in ["softmax", "gumbel", "gumbel2k"]
        self.name = name
        self.use_soft_edge = use_soft_edge

    def prod(self, vector, edge):
        if self.use_soft_edge:
            return (vector.T * F.softmax(edge)).T

        else:
            return vector

    def __call__(self, vector, edge, temperature=1, dim=-1):

        if self.name == "gumbel":
            return self.prod(F.gumbel_softmax(vector, temperature, dim), edge)

        if self.name == "softmax":
            return self.prod(F.softmax(vector, dim), edge)

        if self.name == "gumbel2k":
            return self.prod(gumbel_top2k(vector, temperature, dim), edge)