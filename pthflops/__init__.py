# from ops_jit import count_ops_jit
# try:
import sys
from os import path
sys.path.append(path.dirname(path.abspath(__file__)))

from ops_fx import count_ops_fx, ProfilingInterpreter
force_jit = False
import torch
import pandas as pd
# print('Able to import torch.fx.')
# except Exception as e:
#     force_jit = True
#     print(e)
#     print('Unable to import torch.fx, you pytorch version may be too old.')

__version__ = '0.4.2'

def count_Flops(model, input_size=256, class_names=False):
    inp = torch.rand(1,3,input_size,input_size, device=next(model.parameters()).device)
    tracer = ProfilingInterpreter(model)
    tracer.run(inp)
    columns = [f"BitOps({input_size}x{input_size})", f"Flops({input_size}x{input_size})"]
    if class_names:
        columns += ["class"]
    df = pd.DataFrame(columns=columns)
    for k, v in tracer.bitops_flops.items():
        if k.name in df.index:
            raise ValueError(f"Trying to overwrite nodes values {k.name}")
        if sum(v[0]) > 0:
            value = v[0] + (v[1],) if class_names else v[0]
            df.loc[k.name] = value
    print(df.sum())
    return df.sum()[f"Flops({input_size}x{input_size})"]



def count_ops(model, input, mode='fx', custom_ops={}, ignore_layers=[], print_readable=True, verbose=True, *args):
    if 'fx' == mode and not force_jit:
        return count_ops_fx(
            model,
            input,
            custom_ops=custom_ops,
            ignore_layers=ignore_layers,
            print_readable=print_readable,
            verbose=verbose,
            *args) # TODO this branch
    # elif 'jit' == mode or force_jit:
    #     if force_jit:
    #         print("FX is unsupported on your pytorch version, falling back to JIT")
    #     return count_ops_jit(
    #         model,
    #         input,
    #         custom_ops=custom_ops,
    #         ignore_layers=ignore_layers,
    #         print_readable=print_readable,
    #         verbose=verbose,
    #         *args)
    else:
        raise ValueError('Unknown mode selected.')
