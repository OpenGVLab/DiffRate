import torch.nn as nn


class Prune(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, kept_number):
        if self.training:
            return x
        else:
            return x[:, :kept_number]