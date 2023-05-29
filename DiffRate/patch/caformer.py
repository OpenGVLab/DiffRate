from typing import Tuple

import torch

from caformer import Attention, SepConv, MetaFormerBlock, MetaFormer, Downsampling

import torch.nn as nn

from DiffRate.ddp import DiffRate
from DiffRate.merge import get_merge_func
from DiffRate.patch.deit import DiffRateAttention

from DiffRate.utils import ste_min
from DiffRate.merge import tokentofeature, uncompress

import pdb

    
class DiffRateSepConv(SepConv):
    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        
        x = tokentofeature(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.flatten(-2).transpose(1,2)

        x = self.act2(x)
        x = self.pwconv2(x)
        return x
    
class DiffRateDownsampling(Downsampling):
    def forward(self, x):
        x = self.pre_norm(x)
        # to feature
        if len(x.shape) == 3:
            if self.training:
                mask = self._diffrate_info["mask"]
                last_token_number = mask[0].sum().int()
                x_ = uncompress(x[:,:last_token_number], self._diffrate_info["source"][:,:last_token_number])
                B, N, C = x_.shape
                x_sort = torch.zeros((B,N,C),device=x.device)
                x_sort = x_sort.scatter_reduce(1, self._diffrate_info["index"].unsqueeze(-1).expand(B, N, C), x,reduce='sum')
                mask_sort = torch.zeros_like(mask)
                mask_sort = mask_sort.scatter_reduce(1, self._diffrate_info["index"], mask,reduce='sum')
                x = mask_sort.unsqueeze(-1)*x_sort + (1-mask_sort.unsqueeze(-1)) * x_
                # x = x_
            else:
                x = uncompress(x, self._diffrate_info["source"])
            x = tokentofeature(x)
            
        if self.pre_permute:
            # if take [B, H, W, C] as input, permute it to [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.flatten(-2).transpose(1,2) # [B, C, H, W] -> [B, N, C]

        # refine
        B, N, C = x.shape
        self._diffrate_info["source"] = torch.eye(N, device=x.device)[None, ...].expand(B, N, N)
        self._diffrate_info["size"] = torch.ones([B,N,1], device=x.device)
        self._diffrate_info["mask"] = torch.ones((B,N),device=x.device)
        self._diffrate_info["index"] = torch.arange((N), device=x.device).unsqueeze(0).expand(B,N).long()
        
        x = self.post_norm(x)
        return x


class DiffRateMetaFormerBlock(MetaFormerBlock):
    def introduce_diffrate(self,patch_number, merge_granularity):
        self.merge_ddp = DiffRate(patch_number,merge_granularity)
    def forward(self, x):
        if isinstance(self.token_mixer, DiffRateAttention):
            size = self._diffrate_info["size"]
            mask = self._diffrate_info["mask"]
            x_token_mixer, attn = self.token_mixer(self.norm1(x),size,mask)

            metric = attn.mean(dim=(1,2))
            _, idx = torch.sort(metric, descending=True)
            x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(x_token_mixer)
            )

            # token compression, only consider merging in Hierarchical architecture
            ## sort
            x = torch.gather(x, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
            self._diffrate_info["size"] = torch.gather(self._diffrate_info["size"], dim=1, index=idx.unsqueeze(-1))
            self._diffrate_info["source"] = torch.gather(self._diffrate_info["source"], dim=1, index=idx.unsqueeze(-1).expand(-1, -1, self._diffrate_info["source"].shape[-1]))
            self._diffrate_info["index"] = torch.gather(self._diffrate_info["index"], dim=1, index=idx)

            ## merge
            if self.training:
                last_token_number = mask[0].sum().int()
                merge_kept_num = self.merge_ddp.update_kept_token_number()
                self._diffrate_info["merge_kept_num"].append(merge_kept_num)
                if merge_kept_num < last_token_number:
                    merge_mask = self.merge_ddp.get_token_mask(last_token_number)
                    x_compressed, size_compressed, source_compressed = x[:, last_token_number:], self._diffrate_info["size"][:,last_token_number:], self._diffrate_info["source"][:,last_token_number:]
                    merge_func, node_max = get_merge_func(metric=x[:, :last_token_number].detach(), kept_number=int(merge_kept_num))
                    x = merge_func(x[:,:last_token_number],  mode="mean", training=True)
                    # optimize proportional attention in ToMe by considering similarity
                    size = self._diffrate_info["size"][:, :last_token_number]
                    size = merge_func(size,  mode="sum", training=True)
                    x = torch.cat([x, x_compressed], dim=1)
                    self._diffrate_info["size"] = torch.cat([size, size_compressed], dim=1)
                    source = merge_func(self._diffrate_info["source"][:, :last_token_number], mode="amax", training=True)
                    self._diffrate_info["source"] = torch.cat([source, source_compressed], dim=1)
                    mask = mask * merge_mask

                self._diffrate_info["mask"] = mask

            else:
                merge_kept_num = self.merge_ddp.kept_token_number
                if merge_kept_num < x.shape[1]:
                    merge,node_max = get_merge_func(x.detach(), kept_number=merge_kept_num)

                    x = merge(x,mode='mean')
                    self._diffrate_info["size"] = merge(self._diffrate_info["size"], mode='sum')
                    self._diffrate_info["source"] = merge(self._diffrate_info["source"], mode="amax")
                

        else:
            x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(
                    self.token_mixer(self.norm1(x))
                )
            )
            
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )
        return x

        
        
    
def make_diffrate_class(transformer_class):
    class DiffRateMetaformer(transformer_class):
        def forward(self, x, return_flop=True) -> torch.Tensor:
            B = x.shape[0]
            self._diffrate_info["size"] = torch.ones([B,3136,1], device=x.device)
            self._diffrate_info["mask"] =  torch.ones((B,3136),device=x.device)
            self._diffrate_info["prune_kept_num"] = []
            self._diffrate_info["merge_kept_num"] = []
            self._diffrate_info["source"] = torch.eye(3136, device=x.device)[None, ...].expand(B, 3136, 3136)
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
            merge_kept_num = []
            for module in self.modules():
               if isinstance(module, DiffRateMetaFormerBlock) and isinstance(module.token_mixer, DiffRateAttention):
                    merge_kept_num.append(int(module.merge_ddp.kept_token_number))
            return None, merge_kept_num
                

        def set_kept_num(self, prune_kept_numbers, merge_kept_numbers):
            # only support for token merging for hierarchical architectures
            assert len(merge_kept_numbers) == self.depths[-2]+self.depths[-1]
            index = 0
            for module in self.modules():
                if isinstance(module, DiffRateMetaFormerBlock) and isinstance(module.token_mixer, DiffRateAttention):
                    module.merge_ddp.kept_token_number = merge_kept_numbers[index]
                    module.merge_ddp.kept_token_candidate = nn.Parameter(torch.tensor([float(merge_kept_numbers[index])]), requires_grad=False)   # for finetune
                    module.merge_ddp.selected_probability = nn.Parameter(torch.zeros((1)))   
                    index += 1 
                    
        def calculate_flop_training(self):
            # only support for CAFormer
            downsample_stride = [4,2,2,2]
            downsample_kernel = [7,3,3,3]
            sepconv_kernel = 7
            cur_reso = 224
            flops = 0.
            attention_index = 0
            for i in range(len(self.depths)):
                cur_reso = cur_reso/downsample_stride[i]
                N = cur_reso**2
                input_channel = 3 if i==0 else self.dims[i-1]
                C = self.dims[i]
                downsample_flops = (N*C)*(downsample_kernel[i]*downsample_kernel[i]*input_channel)
                flops += downsample_flops
                if i < 2:       # ConvFormer
                    block_flop = 12*N*C*C + sepconv_kernel*sepconv_kernel*N*C*2
                    flops += self.depths[i]*block_flop
                else:           # TransFormer
                    N = torch.tensor(N, device=self._diffrate_info["merge_kept_num"][attention_index].device)
                    for j in range(self.depths[i]):
                        merge_kept_number = self._diffrate_info["merge_kept_num"][attention_index].float()
                        mhsa_flops = 4*N*C*C + 2*N*N*C
                        flops += mhsa_flops
                        N = ste_min(N, merge_kept_number)
                        ffn_flops = 8*N*C*C
                        flops += ffn_flops
                        attention_index += 1
            classifier_flops = self.dims[-1]*self.num_classes*8     # MLP classifier head
            flops += classifier_flops
            return flops

        def calculate_flop_inference(self):
            # only support for CAFormer
            downsample_stride = [4,2,2,2]
            downsample_kernel = [7,3,3,3]
            sepconv_kernel = 7
            cur_reso = 224
            flops = 0.
            for i in range(len(self.depths)):
                cur_reso = cur_reso/downsample_stride[i]
                N = cur_reso**2
                input_channel = 3 if i==0 else self.dims[i-1]
                C = self.dims[i]
                downsample_flops = (N*C)*(downsample_kernel[i]*downsample_kernel[i]*input_channel)
                flops += downsample_flops
                if i < 2:       # ConvFormer
                    block_flop = 12*N*C*C + sepconv_kernel*sepconv_kernel*N*C*2
                    flops += self.depths[i]*block_flop
                else:           # TransFormer
                    for j in range(self.depths[i]):
                        merge_kept_number = self.stages[i][j].merge_ddp.kept_token_number
                        mhsa_flops = 4*N*C*C + 2*N*N*C
                        flops += mhsa_flops
                        N = ste_min(N, merge_kept_number)
                        ffn_flops = 8*N*C*C
                        flops += ffn_flops
            classifier_flops = self.dims[-1]*self.num_classes*8     # MLP classifier head
            flops += classifier_flops
            return flops
            
            
            
    
    return DiffRateMetaformer

def apply_patch(
    model: MetaFormer,prune_granularity=1, merge_granularity=1
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
    }

    block_index = 0
    depths = model.depths
    token_number = [3136]*depths[0] + [784]*depths[1] + [196]*depths[2] + [49]*depths[3]
    depth = sum(depths)
    non_compressed_block_index = []
    for i in range(depths[3]):
        non_compressed_block_index.append(depth-i-1)
    non_compressed_block_index.append(depths[0]+depths[1])
    non_compressed_block_index.append(depths[0]+depths[1]+depths[2]-1)
    for module in model.modules():
        if isinstance(module, MetaFormerBlock):
            module.__class__ = DiffRateMetaFormerBlock
            if isinstance(module.token_mixer,Attention):
                if block_index in non_compressed_block_index:
                    module.introduce_diffrate(token_number[block_index], token_number[block_index]+1)
                elif isinstance(module.token_mixer,Attention):
                    module.introduce_diffrate(token_number[block_index],  merge_granularity)
            block_index += 1
            module._diffrate_info = model._diffrate_info
        elif isinstance(module, Attention):
            module.__class__ = DiffRateAttention
            module._diffrate_info = model._diffrate_info
        elif isinstance(module, SepConv):
            module.__class__ = DiffRateSepConv
            module._diffrate_info = model._diffrate_info
        elif isinstance(module, Downsampling):
            module.__class__ = DiffRateDownsampling
            module._diffrate_info = model._diffrate_info





