

import math
from typing import Callable, Tuple

import torch
import torch.nn as nn



def get_merge_func(metric: torch.Tensor, kept_number: int, class_token: bool = True):
    with torch.no_grad():
        metric = metric/metric.norm(dim=-1, keepdim=True)
        unimportant_tokens_metric = metric[:, kept_number:]
        compress_number = unimportant_tokens_metric.shape[1]
        important_tokens_metric = metric[:,:kept_number]
        similarity = unimportant_tokens_metric@important_tokens_metric.transpose(-1,-2)
        if class_token:
            similarity[..., :, 0] = -math.inf
        node_max, node_idx = similarity.max(dim=-1)
        dst_idx = node_idx[..., None]
    def merge(x: torch.Tensor, mode="mean", training=False) -> torch.Tensor:
        src = x[:,kept_number:]
        dst = x[:,:kept_number]
        n, t1, c = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, compress_number, c), src, reduce=mode) 
        if training:
            return torch.cat([dst, src], dim=1)
        else:
            return dst
    return merge, node_max

def uncompress(x, source):
    '''
    input: 
        x: [B, N', C]
        source: [B, N', N]
        size: [B, N', 1]
    output:
        x: [B, N, C]
        source: [B, N, N]
        size: [B, N, 1]
    '''
    index = source.argmax(dim=1)
    # print(index)
    uncompressed_x = torch.gather(x, dim=1, index=index.unsqueeze(-1).expand(-1,-1,x.shape[-1]))
    return uncompressed_x

def tokentofeature(x):
    B, N, C = x.shape
    H = int(N ** (1/2))
    x = x.reshape(B, H, H, C)
    return x
