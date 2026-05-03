import torch
import timm
import torch.nn.functional as F
from torch import nn
from einops import rearrange, einsum
from natten.functional import na3d_av

def downsample3d(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm3d(out_dim),
    )        

class DynamicConvBlock(nn.Module):
    def __init__(self,
                 dim=64,
                 ctx_dim=32,
                 kernel_size=7,
                 smk_size=5,
                 num_heads=2,
                 mlp_ratio=4,
                 res_scale=True,
                 is_first=False,
                 is_last=False,
                 use_gemm=False,
                 **kwargs):
        
        super().__init__()
        
        ctx_dim = ctx_dim
        self.kernel_size = kernel_size
        self.res_scale = res_scale
        self.use_gemm = use_gemm
        self.smk_size = smk_size
        self.num_heads = num_heads * 2
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.is_first = is_first
        self.is_last = is_last

        self.weight_query = nn.Sequential(
            nn.Conv3d(dim, dim//2, kernel_size=1, bias=False),
            nn.BatchNorm3d(dim//2),
        )
         
        self.weight_key = nn.Sequential(
            nn.AdaptiveAvgPool3d(7),
            nn.Conv3d(ctx_dim, dim//2, kernel_size=1, bias=False),
            nn.BatchNorm3d(dim//2),
        )
        
        self.weight_proj = nn.Conv2d(343, kernel_size**3 + smk_size**3, kernel_size=1)
        
        self.get_rpb()


    def get_rpb(self):
        self.rpb_size1 = 2 * self.smk_size - 1
        self.rpb1 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size1, self.rpb_size1, self.rpb_size1))
        self.rpb_size2 = 2 * self.kernel_size - 1
        self.rpb2 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size2, self.rpb_size2, self.rpb_size2))
        nn.init.zeros_(self.rpb1)
        nn.init.zeros_(self.rpb2)
    
    @torch.no_grad()
    def generate_idx(self, kernel_size):
        rpb_size = 2 * kernel_size - 1
        idx_d = torch.arange(0, kernel_size)
        idx_h = torch.arange(0, kernel_size)
        idx_w = torch.arange(0, kernel_size)
        idx_k = ((idx_d.unsqueeze(-1).unsqueeze(-1) * rpb_size**2) +
                (idx_h.unsqueeze(-1) * rpb_size) + idx_w).view(-1)
        return (idx_d, idx_h, idx_w, idx_k)

    def apply_rpb(self, attn, rpb, depth, height, width, kernel_size, idx_d, idx_h, idx_w, idx_k):
        num_repeat_d = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_h = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_w = torch.ones(kernel_size, dtype=torch.long)

        num_repeat_d[kernel_size // 2] = max(1, depth - (kernel_size - 1))
        num_repeat_h[kernel_size // 2] = max(1, height - (kernel_size - 1))
        num_repeat_w[kernel_size // 2] = max(1, width - (kernel_size - 1))
        bias_dhw = (idx_d.repeat_interleave(num_repeat_d).unsqueeze(-1).unsqueeze(-1) * (kernel_size**2)) + \
                (idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * kernel_size) + \
                idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_dhw.unsqueeze(-1) + idx_k
        bias_idx = bias_idx.reshape(-1, int(kernel_size ** 3))
        bias_idx = torch.flip(bias_idx, [0])

        rpb = torch.flatten(rpb, 1, 3)[:, bias_idx]
        rpb = rpb.reshape(1, int(self.num_heads), int(depth), int(height), int(width), int(kernel_size ** 3))
        return attn + rpb

    def _forward_inner(self, x, h_x):
        B, C, D, H, W = x.shape
        B, C_h, D_h, H_h, W_h = h_x.shape

        x_f = torch.cat([x, h_x], dim=1)

        query, key = torch.split(x_f, split_size_or_sections=[C, C_h], dim=1)
        query = self.weight_query(query) * self.scale
        key = self.weight_key(key)
        query = rearrange(query, 'b (g c) d h w -> b g c (d h w)', g=self.num_heads)
        key = rearrange(key, 'b (g c) d h w -> b g c (d h w)', g=self.num_heads)
        weight = torch.einsum('b g c n, b g c l -> b g n l', query, key)
        weight = rearrange(weight, 'b g n l -> b l g n').contiguous()
        weight = self.weight_proj(weight)
        weight = rearrange(weight, 'b l g (d h w) -> b g d h w l', d=D, h=H, w=W)

        attn1, attn2 = torch.split(weight, split_size_or_sections=[self.smk_size ** 3, self.kernel_size ** 3], dim=-1)
        rpb1_idx = self.generate_idx(self.smk_size)
        rpb2_idx = self.generate_idx(self.kernel_size)
        attn1 = self.apply_rpb(attn1, self.rpb1, D, H, W, self.smk_size, *rpb1_idx)
        attn2 = self.apply_rpb(attn2, self.rpb2, D, H, W, self.kernel_size, *rpb2_idx)
        attn1 = torch.softmax(attn1, dim=-1)
        attn2 = torch.softmax(attn2, dim=-1)
        value = rearrange(x, 'b (m g c) d h w -> m b g d h w c', m=2, g=self.num_heads)

        x1 = na3d_av(attn1, value[0], kernel_size=self.smk_size,dilation=1)
        x2 = na3d_av(attn2, value[1], kernel_size=self.kernel_size,dilation=1)

        x = torch.cat([x1, x2], dim=1)
        x = rearrange(x, 'b g d h w c -> b (g c) d h w', d=D, h=H, w=W)

        return x
    
    def forward(self, x, h_x):
        x = self._forward_inner(x, h_x)
        return x

class MultiLevelFusion(nn.Module):
    def __init__(self, channels=[16, 32, 64,128]):
        super().__init__()
        
        self.fusion_blocks = nn.ModuleList([
            DynamicConvBlock(
                dim=ch,          
                ctx_dim=ch,      
                kernel_size=7,  
                smk_size=3,
                num_heads=2,
                mlp_ratio=2,
                is_first=(i==0)
            ) for i, ch in enumerate(channels)
        ])
        self.embed1 = nn.Sequential(
                downsample3d(channels[0], channels[1]),
                nn.GELU()
            )
        self.embed2 = nn.Sequential(
                downsample3d(channels[2], channels[2]),
                nn.GELU()
            )
        self.embed3 = nn.Sequential(
                downsample3d(channels[3], channels[3]),
                nn.GELU()
            )
    def forward(self, mri_features, pet_features):

        fused_features = []
        
        for i, (mri, pet) in enumerate(zip(mri_features, pet_features)):
            fused = self.fusion_blocks[i](mri, pet)
            fused_features.append(fused)
        
        x1 = self.embed1(fused_features[0])
        x1_2 = torch.cat([x1, fused_features[1]], dim=1)
        x2 = self.embed2(x1_2)
        out = torch.cat([x2, fused_features[2]], dim=1)
        out = self.embed3(out)

        return out
if __name__ == '__main__':
    model = MultiLevelFusion().cuda()
    x1 = torch.randn(1, 16, 32,32,32).cuda()
    x2 = torch.randn(1,32,16,16,16).cuda()
    x3 = torch.randn(1,64,8,8,8).cuda()    
    y = model([x1,x2,x3],[x1,x2,x3])
    print(y.shape)