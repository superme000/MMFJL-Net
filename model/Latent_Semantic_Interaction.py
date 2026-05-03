import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentSemanticInteraction(nn.Module):
    def __init__(self, channels, norm_type='l2'): 
        super(LatentSemanticInteraction, self).__init__()
        self.channels = channels
        self.norm_type = norm_type
        self.maxpool = nn.MaxPool3d(2)
        self.avgpool = nn.AvgPool3d(2)
        
        if self.norm_type == 'LN':
            self.layer_norm = nn.LayerNorm(channels)

    def forward(self, x, y):
        y = self.maxpool(y) + self.avgpool(y)

        x_flat = x.view(x.size(0), self.channels, -1)
        y_flat = y.view(y.size(0), self.channels, -1)
        
        interaction = torch.einsum('bci,bcj->bcij', x_flat, y_flat)
        interaction = interaction.contiguous().view(x.size(0), self.channels, -1)
        pooled = interaction.sum(dim=2)
        if self.norm_type == 'l2':
            return F.normalize(pooled, p=2, dim=-1) 
        elif self.norm_type == 'LN':
            return self.layer_norm(pooled)
        else:
            return pooled