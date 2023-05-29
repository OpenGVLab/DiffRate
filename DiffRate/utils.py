'''
This file is modified based on https://github.com/facebookresearch/ToMe/blob/main/tome/utils.py
'''

import time
from typing import List, Tuple, Union

import math
import torch
from tqdm import tqdm

class STE_Min(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in1, x_in2, x_in3=math.inf):
        x = min(x_in1, x_in2, x_in3)
        return x
    
    @staticmethod
    def backward(ctx, g):
        return None, g, g
    
class STE_Ceil(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in):
        x = torch.ceil(x_in)
        return x
    
    @staticmethod
    def backward(ctx, g):
        return g, None
    
    
ste_ceil = STE_Ceil.apply
ste_min = STE_Min.apply


def benchmark(
    model: torch.nn.Module,
    device: torch.device = 0,
    input_size: Tuple[int] = (3, 224, 224),
    batch_size: int = 64,
    runs: int = 40,
    throw_out: float = 0.25,
    use_fp16: bool = False,
    verbose: bool = False,
) -> float:
    """
    Benchmark the given model with random inputs at the given batch size.

    Args:
     - model: the module to benchmark
     - device: the device to use for benchmarking
     - input_size: the input size to pass to the model (channels, h, w)
     - batch_size: the batch size to use for evaluation
     - runs: the number of total runs to do
     - throw_out: the percentage of runs to throw out at the start of testing
     - use_fp16: whether or not to benchmark with float16 and autocast
     - verbose: whether or not to use tqdm to print progress / print throughput at end

    Returns:
     - the throughput measured in images / second
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)
    is_cuda = torch.device(device).type == "cuda"

    model = model.eval().to(device)
    input = torch.rand(batch_size, *input_size, device=device)
    if use_fp16:
        input = input.half()

    warm_up = int(runs * throw_out)
    total = 0
    start = time.time()

    with torch.autocast(device.type, enabled=use_fp16):
        with torch.no_grad():
            for i in tqdm(range(runs), disable=not verbose, desc="Benchmarking"):
                if i == warm_up:
                    if is_cuda:
                        torch.cuda.synchronize()
                    total = 0
                    start = time.time()

                model(input)
                total += batch_size

    if is_cuda:
        torch.cuda.synchronize()

    end = time.time()
    elapsed = end - start

    throughput = total / elapsed

    if verbose:
        print(f"Throughput: {throughput:.2f} im/s")

    return throughput

