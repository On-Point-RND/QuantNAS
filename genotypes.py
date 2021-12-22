""" Genotypes
    - Genotype: normal/reduce gene + normal/reduce cell output connection (concat)
    - gene: discrete ops information (w/o output connection)
    - dag: real ops (can be mixed or discrete, but Genotype has only discrete information itself)
"""
from collections import namedtuple
import torch.nn as nn
from sr_models import quant_ops as ops_sr

Genotype_SR = namedtuple("Genotype_SR", "head body tail skip upsample")


body = [
    "simple_3x3",
    "simple_5x5",
    "simple_3x3_grouped_3",
    "simple_5x5_grouped_3",
    "decenc_3x3_2",
    "decenc_5x5_2",
    'simple_1x1_grouped_3',
    'simple_1x1', 
]

head = [
    "simple_3x3",
    "simple_5x5",
    "growth2_5x5",
    "growth2_3x3",
    "simple_3x3_grouped_3",
    "simple_5x5_grouped_3",
    'simple_1x1_grouped_3',
    'simple_1x1', 
]

tail = [
    "simple_3x3",
    "simple_5x5",
    "growth2_5x5",
    "growth2_3x3",
    "simple_3x3_grouped_3",
    "simple_5x5_grouped_3",
    'simple_1x1_grouped_3',
    'simple_1x1', 
]

upsample = [
    "conv_5x1_1x5",
    "conv_3x1_1x3",
    "simple_3x3",
    "simple_5x5",
    "growth2_5x5",
    "growth2_3x3",
    "decenc_3x3_2",
    "decenc_5x5_2",
    "simple_3x3_grouped_3",
    "simple_5x5_grouped_3",
    'simple_1x1_grouped_3',
    'simple_1x1', 
]

skip = [
    "decenc_3x3_2",
    "decenc_5x5_2",
    "simple_3x3",
    "simple_5x5",
]

PRIMITIVES_SR = {
    "head": head,
    "body": body,
    "skip": skip,
    "tail": tail,
    "upsample": upsample,
}


PRIMITIVES_SR = {
    "head": head,
    "body": body,
    "skip": skip,
    "tail": tail,
    "upsample": upsample,
}



def from_str(s):
    genotype = eval(s)
    return genotype


def to_dag_sr(C_fixed, gene, gene_type, c_in=3, c_out=3, scale=4):
    """generate discrete ops from gene"""
    dag = []
    for i, (op_name, bit) in enumerate(gene):
        C_in, C_out, = (
            C_fixed,
            C_fixed,
        )
        if i == 0 and gene_type == "head":
            C_in = c_in
        elif i + 1 == len(gene) and gene_type == "tail":
            C_out = c_out
        elif i == 0 and gene_type == "tail":
            C_in = c_in

        elif gene_type == "upsample":
            C_in = C_fixed
            C_out = 3 * (scale ** 2)
        else:
            C_in = C_fixed
            C_out = C_fixed

        print(gene_type, op_name, C_in, C_out, C_fixed, bit)
        op = ops_sr.OPS[op_name](
            C_in, C_out, [bit], C_fixed, 1, affine=False, shared=False
        )
        dag.append(op)
    return nn.Sequential(*dag)


def parse_sr(alpha, name, bits=[2]):
    gene = []
    for edges in alpha:
        best_bit = 0
        best_op = 0
        best_val = 0
        n_ops = len(edges) // len(bits)
        for op_idx, edge in enumerate(edges.chunk(n_ops)):
            max_val = edge.max()
            bit_idx = edge.argmax()
            if max_val > best_val:
                best_val = max_val
                best_op = op_idx
                best_bit = bit_idx.item()

        prim = PRIMITIVES_SR[name][best_op]
        gene.append((prim, bits[best_bit]))
    return gene
