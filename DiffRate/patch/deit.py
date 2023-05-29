# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from typing import Tuple

import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer
import torch.nn as nn

# import DiffRate.ddp as ddp
from DiffRate.ddp import DiffRate
from DiffRate.merge import get_merge_func

from DiffRate.utils import ste_min





class DiffRateBlock(Block):
    """
    Modifications:
     - Apply DiffRate between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """
    def introduce_diffrate(self,patch_number, prune_granularity, merge_granularity):
        self.prune_ddp = DiffRate(patch_number,prune_granularity)
        self.merge_ddp = DiffRate(patch_number,merge_granularity)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        size = self._diffrate_info["size"]
        mask = self._diffrate_info["mask"]
        x_attn, attn = self.attn(self.norm1(x), size, mask=self._diffrate_info["mask"])
        x = x + self.drop_path(x_attn)

        # importance metric
        cls_attn = attn[:, :, 0, 1:]
        cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
        _, idx = torch.sort(cls_attn, descending=True)
        cls_index = torch.zeros((B,1), device=idx.device).long()
        idx = torch.cat((cls_index, idx+1), dim=1)
        
        # sorting
        x = torch.gather(x, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        self._diffrate_info["size"] = torch.gather(self._diffrate_info["size"], dim=1, index=idx.unsqueeze(-1))
        mask = torch.gather( mask, dim=1, index=idx)
        if self._diffrate_info["trace_source"]:
            self._diffrate_info["source"] = torch.gather(self._diffrate_info["source"], dim=1, index=idx.unsqueeze(-1).expand(-1, -1, self._diffrate_info["source"].shape[-1]))

        
        if self.training:
            # pruning, pruning only needs to generate masks during training
            last_token_number = mask[0].sum().int()
            prune_kept_num = self.prune_ddp.update_kept_token_number()      # expected prune compression rate, has gradiet
            self._diffrate_info["prune_kept_num"].append(prune_kept_num)
            if prune_kept_num < last_token_number:        # make sure the kept token number is a decreasing sequence
                prune_mask = self.prune_ddp.get_token_mask(last_token_number)
                mask = mask * prune_mask.expand(B, -1)

            mid_token_number = min(last_token_number, int(prune_kept_num)) # token number after pruning
                
            # merging
            merge_kept_num = self.merge_ddp.update_kept_token_number()
            self._diffrate_info["merge_kept_num"].append(merge_kept_num)

            if merge_kept_num < mid_token_number:
                merge_mask = self.merge_ddp.get_token_mask(mid_token_number)
                x_compressed, size_compressed = x[:, mid_token_number:], self._diffrate_info["size"][:,mid_token_number:]
                merge_func, node_max = get_merge_func(metric=x[:, :mid_token_number].detach(), kept_number=int(merge_kept_num))
                x = merge_func(x[:,:mid_token_number],  mode="mean", training=True)
                # optimize proportional attention in ToMe by considering similarity
                size = torch.cat((self._diffrate_info["size"][:, :int(merge_kept_num)],self._diffrate_info["size"][:, int(merge_kept_num):mid_token_number]*node_max[..., None]),dim=1)
                size = size.clamp(1)
                size = merge_func(size,  mode="sum", training=True)
                x = torch.cat([x, x_compressed], dim=1)
                self._diffrate_info["size"] = torch.cat([size, size_compressed], dim=1)
                mask = mask * merge_mask

            self._diffrate_info["mask"] = mask
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            
        else:
            # pruning
            prune_kept_num = self.prune_ddp.kept_token_number
            x = x[:, :prune_kept_num]
            self._diffrate_info["size"] = self._diffrate_info["size"][:, :prune_kept_num]
            if self._diffrate_info["trace_source"]:
                self._diffrate_info["source"] = self._diffrate_info["source"][:, :prune_kept_num]
                
            
            # merging
            merge_kept_num = self.merge_ddp.kept_token_number
            if merge_kept_num < prune_kept_num:
                merge,node_max = get_merge_func(x.detach(), kept_number=merge_kept_num)
                x = merge(x,mode='mean')
                # optimize proportional attention in ToMe by considering similarity
                self._diffrate_info["size"] = torch.cat((self._diffrate_info["size"][:, :merge_kept_num],self._diffrate_info["size"][:, merge_kept_num:]*node_max[..., None] ),dim=1)
                self._diffrate_info["size"] = merge(self._diffrate_info["size"], mode='sum')
                if self._diffrate_info["trace_source"]:
                    self._diffrate_info["source"] = merge(self._diffrate_info["source"], mode="amax")

            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
                

                


class DiffRateAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, N = policy.size()
        B, H, N, N = attn.size()
        attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None, mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]
        
        if self.training:
            attn = self.softmax_with_policy(attn, mask)
        else:
            attn = attn.softmax(dim=-1)
            
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return attention map as well here
        return x, attn


def make_diffrate_class(transformer_class):
    class DiffRateVisionTransformer(transformer_class):
        def forward(self, x, return_flop=True) -> torch.Tensor:
            B = x.shape[0]
            self._diffrate_info["size"] = torch.ones([B,self.patch_embed.num_patches+1,1], device=x.device)
            self._diffrate_info["mask"] =  torch.ones((B,self.patch_embed.num_patches+1),device=x.device)
            self._diffrate_info["prune_kept_num"] = []
            self._diffrate_info["merge_kept_num"] = []
            if self._diffrate_info["trace_source"]:
                self._diffrate_info["source"] = torch.eye(self.patch_embed.num_patches+1, device=x.device)[None, ...].expand(B, self.patch_embed.num_patches+1, self.patch_embed.num_patches+1)
            x = super().forward(x)
            if return_flop:
                if self.training:
                    flops = self.calculate_flop_training()
                else:
                    flops = self.calculate_flop_inference()
                return x, flops
            else:
                return x
        
        def parameters(self, recurse=True):
            # original network parameter
            params = []
            for n, m in self.named_parameters():
                if n.find('ddp') > -1:
                    continue
                params.append(m)
            return iter(params)    
        
        def arch_parameters(self):
            params = []
            for n, m in self.named_parameters():
                if n.find('ddp') > -1:
                    params.append(m)
            return iter(params)    

        def get_kept_num(self):
            prune_kept_num = []
            merge_kept_num = []
            for block in self.blocks:
                prune_kept_num.append(int(block.prune_ddp.kept_token_number))
                merge_kept_num.append(int(block.merge_ddp.kept_token_number))
            return prune_kept_num, merge_kept_num
                

        def set_kept_num(self, prune_kept_numbers, merge_kept_numbers):
            assert len(prune_kept_numbers) == len(self.blocks) and len(merge_kept_numbers) == len(self.blocks)
            for block, prune_kept_number, merge_kept_number in zip(self.blocks, prune_kept_numbers, merge_kept_numbers):
                block.prune_ddp.kept_token_number = prune_kept_number
                block.merge_ddp.kept_token_number = merge_kept_number
        
        def calculate_flop_training(self):
            C = self.embed_dim
            patch_number = float(self.patch_embed.num_patches)
            N = torch.tensor(patch_number+1, device=self.blocks[0].prune_ddp.selected_probability.device)
            flops = 0
            patch_embedding_flops = N*C*(self.patch_embed.patch_size[0]*self.patch_embed.patch_size[1]*3)
            classifier_flops = C*self.num_classes
            with torch.cuda.amp.autocast(enabled=False):
                for prune_kept_number, merge_kept_number in zip(self._diffrate_info["prune_kept_num"],self._diffrate_info["merge_kept_num"]):
                    # translate fp16 to fp32 for stable training
                    prune_kept_number = prune_kept_number.float()     
                    merge_kept_number = merge_kept_number.float()
                    mhsa_flops = 4*N*C*C + 2*N*N*C
                    flops += mhsa_flops
                    N = ste_min(N, prune_kept_number, merge_kept_number)
                    ffn_flops = 8*N*C*C
                    flops += ffn_flops
            flops += patch_embedding_flops
            flops += classifier_flops
            return flops

        def calculate_flop_inference(self):
            C = self.embed_dim
            patch_number = float(self.patch_embed.num_patches)
            N = torch.tensor(patch_number+1, device=self.blocks[0].prune_ddp.selected_probability.device)
            flops = 0
            patch_embedding_flops = N*C*(self.patch_embed.patch_size[0]*self.patch_embed.patch_size[1]*3)
            classifier_flops = C*self.num_classes
            with torch.cuda.amp.autocast(enabled=False):
                for block in (self.blocks):
                    prune_kept_number = block.prune_ddp.kept_token_number
                    merge_kept_number = block.merge_ddp.kept_token_number
                    mhsa_flops = 4*N*C*C + 2*N*N*C
                    flops += mhsa_flops
                    N = ste_min(N, prune_kept_number, merge_kept_number)
                    ffn_flops = 8*N*C*C
                    flops += ffn_flops
            flops += patch_embedding_flops
            flops += classifier_flops
            return flops
        

    return DiffRateVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False,prune_granularity=1, merge_granularity=1
):
    """
    Applies DiffRate to this transformer.
    """
    DiffRateVisionTransformer = make_diffrate_class(model.__class__)

    model.__class__ = DiffRateVisionTransformer
    model._diffrate_info = {
        "size": None,
        "mask": None,           # only for training
        "source": None,
        "class_token": model.cls_token is not None,
        "trace_source": trace_source,
    }

    block_index = 0
    non_compressed_block_index = [0]
    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = DiffRateBlock
            if block_index in non_compressed_block_index:
                module.introduce_diffrate(model.patch_embed.num_patches, model.patch_embed.num_patches+1, model.patch_embed.num_patches+1)
            else:
                module.introduce_diffrate(model.patch_embed.num_patches, prune_granularity, merge_granularity)
            block_index += 1
            module._diffrate_info = model._diffrate_info
        elif isinstance(module, Attention):
            module.__class__ = DiffRateAttention
