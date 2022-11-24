from functools import reduce
from typing import Any, Dict, List, Tuple
from numpy import isin
import numpy as np
import pandas as pd

import torch
import torch.fx
import torch.nn as nn

# from utils import same_device, print_table

import sys
from os import path
import os
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from sr_models.quant_conv_lsq import QuantConv
# from bnn.ops import BinarizerBase
# from bnn.layers.linear import Linear as BinLinear
# from bnn.layers.conv import Conv2d as BinConv2d, Conv1d as BinConv1d



def _count_convNd(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    r"""Estimates the number of FLOPs in conv layer

    .. warning::
        Currently it ignore the padding

    :param node_string: an onnx node defining a convolutional layer

    :return: number of FLOPs
    :rtype: `int`
    """
    kernel_size = list(module.kernel_size)
    in_channels = module.in_channels
    out_channels = module.out_channels

    filters_per_channel = out_channels // module.groups
    conv_per_position_flops = reduce(lambda x, y: x * y, kernel_size) * \
        in_channels * filters_per_channel

    active_elements_count = output.shape[0] * reduce(lambda x, y: x * y, output.shape[2:])

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_ops = 0
    if module.bias is not None:
        bias_ops = out_channels * active_elements_count

    total_ops = overall_conv_flops + bias_ops
    # if isinstance(module, BinConv2d) or isinstance(module, BinConv1d):
    #     if (
    #         (module.bconfig is None) or 
    #         isinstance(module.activation_pre_process, nn.Identity) or 
    #         isinstance(module.weight_pre_process, nn.Identity)
    #     ):
    #         return 0, total_ops
    #     else:
    #         return total_ops, 0
    return 0, total_ops


def _count_expert_conv2d(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    r"""Estimates the number of FLOPs in EXPERT conv layer

    .. warning::
        Currently it ignore the padding

    :param node_string: an onnx node defining a convolutional layer

    :return: number of FLOPs
    :rtype: `int`
    """

    #kx*ky*in_ch*out_ch//groups  *  batch* out_h * out_w    +    (out_ch * batch* out_h * out_w)

    input = args[0]
    weight = args[1]

    batch_size = input.shape[0]
    output_dims = list(output.shape[-2:])

    kernel_dims = list(weight.shape[-2:])
    in_channels = input.shape[1]
    out_channels = output.shape[1]
    groups = kwargs["groups"]

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims)) * \
        in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if args[2] is not None:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    if module.__name__ == "expert_binary_conv2d":
        return overall_flops, 0
    elif module.__name__ == "expert_conv2d":
        return 0, 0


def _count_relu(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    r"""Estimates the number of FLOPs of a  ReLU activation.
    The function will count the comparison operation as a FLOP.

    :param node_string: an onnx node defining a ReLU op

    :return: number of FLOPs
    :rtype: `int`
    """
    total_ops = 2 * output.numel()  # also count the comparison
    return 0, total_ops

def _count_sigmoid(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    total_ops = 4 * output.numel() # y = 1 / (1 + exp(-x))
    return 0, total_ops


def _count_avgpool(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    r"""Estimates the number of FLOPs of an Average Pooling layer.

    :param node_string: an onnx node defining an average pooling layer

    :return: number of FLOPs
    :rtype: `int`
    """
    out_ops = output.numel()

    kernel_size = [module.kernel_size] * \
        (output.dim() - 2) if isinstance(module.kernel_size, int) else module.kernel_size

    ops_add = reduce(lambda x, y: x * y, kernel_size) - 1
    ops_div = 1
    total_ops = (ops_add + ops_div) * out_ops
    return 0, total_ops


def _count_globalavgpool(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    r"""Estimates the number of FLOPs of an Average Pooling layer.

    :param node_string: an onnx node defining an average pooling layer

    :return: number of FLOPs
    :rtype: `int`
    """
    inp = args[0]

    ops_add = reduce(lambda x, y: x * y, [inp.shape[-2], inp.shape[-1]]) - 1
    ops_div = 1
    total_ops = (ops_add + ops_div) * output.numel()
    return 0, total_ops


def _count_maxpool(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    r"""Estimates the number of FLOPs of a Max Pooling layer.

    :param node_string: an onnx node defining a max pooling layer

    :return: number of FLOPs
    :rtype: `int`
    """
    out_ops = output.numel()

    kernel_size = [module.kernel_size] * \
        (output.dim() - 2) if isinstance(module.kernel_size, int) else module.kernel_size
    ops_add = reduce(lambda x, y: x * y, kernel_size) - 1
    total_ops = ops_add * out_ops
    return 0, total_ops


def _count_bn(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    r"""Estimates the number of FLOPs of a Batch Normalisation operation.

    :param node_string: an onnx node defining a batch norm op

    :return: number of FLOPs
    :rtype: `int`
    """
    total_ops = output.numel() * 2
    return 0, total_ops


def _count_linear(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    r"""Estimates the number of a GEMM or linear layer.

    :param node_string: an onnx node defining a GEMM or linear layer

    :return: number of FLOPs
    :rtype: `int`
    """
    bias_ops = 0
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            bias_ops = output.shape[-1]
    total_ops = args[0].numel() * output.shape[-1] + bias_ops
    # if isinstance(module, BinLinear):
    #     if (
    #         (module.bconfig is None) or 
    #         isinstance(module.activation_pre_process, nn.Identity) or 
    #         isinstance(module.weight_pre_process, nn.Identity)
    #     ):
    #         return 0, total_ops
    #     else:
    #         return total_ops, 0

    return 0, total_ops


def _count_add_mul(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    r"""Estimates the number of FLOPs of a summation op.

    :param node_string: an onnx node defining a summation op

    :return: number of FLOPs
    :rtype: `int`
    """
    if type(output) == int:
        return 0, 1
    return 0, output.numel() * len(args)


def _undefined_op(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    r"""Default case for undefined or free (in terms of FLOPs) operations

    :param node_string: an onnx node

    :return: always 0
    :rtype: `int`
    """
    return 0, 0

class MyTracer(torch.fx.Tracer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str) -> bool:
        res = (
            (m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)) or
            isinstance(m, QuantConv)
            # isinstance(m, BinarizerBase) or 
            # isinstance(m, BinConv1d) or isinstance(m, BinConv2d) or isinstance(m, BinLinear)
        )
        return res

class ProfilingInterpreter(torch.fx.Interpreter):
    def __init__(self, mod: torch.nn.Module, custom_ops: Dict[str, Any] = {}):
        tracer = MyTracer()#param_shapes_constant=True)
        traced = tracer.trace(mod)
        gm = torch.fx.GraphModule(mod, traced)#, concrete_args={"x": input})
        super().__init__(gm)
        # print("Proxy done")
        self.custom_ops = custom_ops

        self.bitops_flops: Dict[torch.fx.Node, float] = {}
        self.parameters: Dict[torch.fx.Node, float] = {}

        self.not_implemented = set()

    def run_node(self, n: torch.fx.Node) -> Any: #TODO не меняет ничего, просто раскидал данные
        return_val = super().run_node(n)
        if isinstance(return_val, Tuple) and (n.op == "call_module" or n.op == "call_function" or n.op == "call_method"):
            if n in self.bitops_flops:
                raise ValueError(f"Triyng to overwrite nodes values. {n}")
            self.bitops_flops[n] = ((return_val[1], return_val[2]), return_val[3])
            self.parameters[n] = return_val[4]
            return_val = return_val[0]

        return return_val

    def call_module(self, target: 'Target', args: Tuple[Any], kwargs: Dict[str, Any]) -> Any:
        # Execute the method and return the result
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        output = submod(*args, **kwargs)

        if submod in self.custom_ops:
            count_ops_funct = self.custom_ops[submod]
        else:
            count_ops_funct = self.count_operations(submod)
        bitops, flops = count_ops_funct(submod, output, args, kwargs)
        current_params = sum(p.numel() for p in submod.parameters())

        return output, bitops, flops, submod.__class__.__name__, current_params

    def call_function(self, target: 'Target', args: Tuple[Any], kwargs: Dict[str, Any]) -> Any:
        assert not isinstance(target, str)

        # Execute the function and return the result
        output = target(*args, **kwargs)

        bitops, flops = self.count_operations(target.__name__)(target, output, args, kwargs)

        return output, bitops, flops, target.__name__, 0
        
    def call_method(self, target : 'Target', args : Tuple[Any], kwargs : Dict[str, Any]) -> Any:
        # args[0] is the `self` object for this method call
        self_obj, *args_tail = args
        # Execute the method and return the result
        assert isinstance(target, str)
        output = getattr(self_obj, target)(*args_tail, **kwargs)
        bitops, flops = self.count_operations(target)(target, output, args, kwargs)

        return getattr(self_obj, target)(*args_tail, **kwargs), bitops, flops, target, 0

    def count_operations(self, module: Any) -> Any:
        ignore = [
            "getitem",
            "zeros_like",
            "ones_like",
            "getattr",
            "eq",
            "flatten",
            "cat",
            "sign",
            "argmax",
            "softmax",
            "size",
            "squeeze",
            "view",
            "permute",
            "expand_as",
            "fill_",
            "view_as",
            "type_as",
            
        ]
        if isinstance(module, torch.nn.modules.conv._ConvNd) or isinstance(module, QuantConv):
            return _count_convNd
        # elif isinstance(module, nn.ReLU) or isinstance(module, nn.PReLU):
        #     return _count_relu
        # elif isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        #     return _count_bn
        # elif isinstance(module, torch.nn.modules.pooling._MaxPoolNd):
        #     return _count_maxpool
        # elif isinstance(module, torch.nn.modules.pooling._AvgPoolNd):
        #     return _count_avgpool
        # elif isinstance(module, torch.nn.modules.pooling._AdaptiveAvgPoolNd) or 'adaptive_avg_pool2d' == module:
        #     return _count_globalavgpool
        # elif isinstance(module, torch.nn.Linear):
        #     return _count_linear
        # elif 'add' == module or 'mul' == module:
        #     return _count_add_mul
        # elif 'matmul' == module:
        #     return _count_linear
        # elif isinstance(module, torch.nn.modules.activation.Sigmoid) or 'sigmoid' == module:
        #     return _count_sigmoid
        elif module == "expert_conv2d" or module == "expert_binary_conv2d":
            return _count_expert_conv2d
        else:
            if not module in ignore:
                old_len = len(self.not_implemented)
                if type(module) != str:
                    self.not_implemented.add(module.__class__)
                else:
                    self.not_implemented.add(module)
                if old_len < len(self.not_implemented):
                    print("ALERT ALERT ALERT|UNDEFINED MODULE:", module)
            return _undefined_op # TODO: Может быть проблема тут, если он квантизующие модули не сможет распознать


def count_ops_fx(model: torch.nn.Module,
                 input: torch.Tensor,
                 custom_ops: Dict[Any,
                                  Any] = {},
                 ignore_layers: List[str] = [],
                 print_readable: bool = True,
                 verbose: bool = True,
                 *args):
    r"""Estimates the number of FLOPs of an :class:`torch.nn.Module`

    :param model: the :class:`torch.nn.Module`
    :param input: a N-d :class:`torch.tensor` containing the input to the model
    :param custom_ops: :class:`dict` containing custom counting functions. The keys represent the name
    of the targeted aten op, while the value a lambda or callback to a function returning the number of ops.
    This can override the ops present in the package.
    :param ignore_layers: :class:`list` containing the name of the modules to be ignored.
    :param print_readable: boolean, if True will print the number of FLOPs. default is True
    :param verbose: boolean, if True will print all the non-zero OPS operations from the network

    :return: number of FLOPs
    :rtype: `int`
    """
    model, input = same_device(model, input)

    # Place the model in eval mode, required for some models
    model_status = model.training
    model.eval()

    tracer = ProfilingInterpreter(model, custom_ops=custom_ops)
    tracer.run(input) # TODO I assume all magic comes from here

    ops = 0
    all_data = []

    for name, current_ops in tracer.flops.items():
        model_status = model.training

        if any(name.name == ign_name for ign_name in ignore_layers):
            continue

        ops += current_ops

        if current_ops and verbose: # TODO kinda dangerous to rely on verbose in order to save data
            all_data.append(['{}'.format(name), current_ops])

    if print_readable:
        if verbose:
            print_table(all_data)
        print("Input size: {0}".format(tuple(input.shape)))
        print("{:,} FLOPs or approx. {:,.2f} GFLOPs".format(ops, ops / 1e+9))

    if model_status:
        model.train()

    return ops, all_data
