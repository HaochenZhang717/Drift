import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# this file only provides the 2 modules used in VQVAE
__all__ = ['Encoder', 'Decoder']

"""
References: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/model.py
"""


# Helper activation function: Swish
def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=16):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))
        # return self.conv(F.interpolate(x, scale_factor=2, mode='linear'))


class Downsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, pad=(0, 1), mode='constant', value=0))


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None,
                 dropout):  # conv_shortcut=False,  # conv_shortcut: always False in VAE
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        # Main processing path
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity()
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # Shortcut connection if channel sizes differ
        self.nin_shortcut = nn.Conv1d(in_channels, self.out_channels, kernel_size=1) \
            if in_channels != self.out_channels else nn.Identity()

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)  # Swish activation
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return self.nin_shortcut(x) + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.C = in_channels

        self.norm = Normalize(in_channels)
        self.qkv = nn.Conv1d(in_channels, 3 * in_channels, kernel_size=1, stride=1, padding=0)
        self.w_ratio = int(in_channels) ** (-0.5)
        self.proj_out = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Compute Q, K, V projections
        qkv = self.qkv(self.norm(x))  # [B, 3C, L]
        q, k, v = qkv.chunk(3, dim=1)  # Each [B, C, L]

        # Reshape for matrix multiplication
        q = q.permute(0, 2, 1)  # [B, L, C]
        k = k  # [B, C, L]
        v = v  # [B, C, L]

        # Compute attention scores
        attn_weights = torch.bmm(q, k) * self.w_ratio  # [B, L, L]
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention to values
        h = torch.bmm(v, attn_weights)  # [B, C, L]

        # Final projection and residual connection
        return x + self.proj_out(h)

def make_attn(in_channels, using_sa=True):
    return AttnBlock(in_channels) if using_sa else nn.Identity()


class Encoder(nn.Module):
    def __init__(
            self, *,
            ch=128,  # Base channel count
            ch_mult=(1, 2, 4, 8),  # Channel multipliers per resolution stage
            num_res_blocks=2,  # Number of residual blocks per stage
            dropout=0.0,  # Dropout probability
            in_channels=6,  # Input channels (e.g., for 3-channel features)
            z_channels,  # Latent space channels
            double_z=False,  # Whether to output double channels (μ and σ for VAE)
            using_sa=True,  # Use self-attention in last stage
            using_mid_sa=True  # Use self-attention in middle block
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.downsample_ratio = 2 ** (self.num_resolutions - 1) # Total reduction factor
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # Initial convolution
        self.conv_in = torch.nn.Conv1d(in_channels, self.ch, kernel_size=3, padding=1)
        # Downsampling stages
        in_ch_mult = (1,) + tuple(ch_mult)
        # print("ch_mult",ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample2x(block_in)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv1d(block_in, z_channels, 3, stride=1, padding=1)

    def forward(self, x):
        # downsampling
        x = x.transpose(1, 2) # B,L,C -> B,C,L
        h = self.conv_in(x) # 128*32*24
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # Middle processing
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # Final output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        # print("h",h.shape)
        # Output shape: [B, z_channels*2, L//downsample_ratio]
        return h


class Decoder(nn.Module):
    def __init__(
        self, *,
        ch=128,                  # Base channel count
        ch_mult=(1, 2, 4, 8),    # Channel multipliers (reverse of encoder)
        num_res_blocks=2,        # Number of residual blocks per stage
        dropout=0.0,
        in_channels=6,           # Output channels (matches input)
        z_channels,              # Latent space channels
        using_sa=True,           # Use self-attention in first stage
        using_mid_sa=True        # Use self-attention in middle block
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # z to block_in
        self.conv_in = torch.nn.Conv1d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample2x(block_in)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        # Project latent to initial channels
        h = self.conv_in(z)

        # Middle processing
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        h = h.transpose(1, 2)  # B,C,L -> B,L,C
        return h
    

class CrossAttnBlock(nn.Module):
    def __init__(self, query_channels, guide_channels, heads=4):
        super().__init__()
        #  guide_channels != query_channels1x1
        if guide_channels != query_channels:
            self.guide_proj = nn.Conv1d(guide_channels, query_channels, kernel_size=1)
        else:
            self.guide_proj = nn.Identity()
        self.attn = nn.MultiheadAttention(embed_dim=query_channels, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(query_channels)

    def forward(self, x, guide):
        guide_aligned = self.guide_proj(guide.permute(0, 2, 1))
        #  (B, L, C)
        q  = x.permute(0, 2, 1)          # (B, L_q, C_q)
        kv = guide_aligned.permute(0, 2, 1)  # (B, L_g, C_q)
        attn_out, _ = self.attn(query=q, key=kv, value=kv)
        out = self.norm(attn_out + q)
        return out.permute(0, 2, 1)      # (B, C_q, L_q)


class DummyCross(nn.Module):
    def forward(self, x, guide):
        return x  #

class SpectralFeatureExtractor(nn.Module):
    def __init__(self,
                 in_channels,
                 feat_dim,
                 n_fft=24,
                 low_freq: int = 1,
                 factor: float = 1.0):
        super().__init__()
        self.n_fft = n_fft
        self.low_freq = low_freq
        self.factor = factor
        max_f = n_fft // 2 + 1 - low_freq
        max_topk = int(factor * math.log(max_f))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * max_topk * 2, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, x):
        B, C, T = x.shape
        spec = torch.fft.rfft(x, n=self.n_fft, dim=2)  #  FFT
        spec = spec[:, :, self.low_freq:]
        F_bins = spec.size(2)
        topk = max(1, int(self.factor * math.log(F_bins)))
        mag = spec.abs().reshape(B * C, F_bins)
        vals, idx = torch.topk(mag, topk, dim=1, largest=True, sorted=True)
        batch_chan = torch.arange(B * C, device=x.device).unsqueeze(1)
        sel_spec = spec.reshape(B * C, F_bins)[batch_chan, idx]
        amp   = sel_spec.abs()
        phase = sel_spec.angle()
        feats = torch.cat([amp, phase], dim=1)
        feats = feats.view(B, C * topk * 2)
        out = self.mlp(feats)
        return out.unsqueeze(1)

class GuidedEncoder(nn.Module):
    def __init__(
            self, *,
            ch=128,  # Base channel count
            ch_mult=(1, 2, 4, 8),  # Channel multipliers per resolution stage
            num_res_blocks=2,  # Number of residual blocks per stage
            dropout=0.0,  # Dropout probability
            in_channels=6,  # Input channels (e.g., for 3-channel features)
            z_channels,  # Latent space channels
            double_z=False,  # Whether to output double channels (μ and σ for VAE)
            using_sa=True,  # Use self-attention in last stage
            using_mid_sa=True,  # Use self-attention in middle block
            num_heads=4,
            cross_attn_level=[1]
            # cross_attn_level=[]
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.downsample_ratio = 2 ** (self.num_resolutions - 1) # Total reduction factor
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.cross_attn_level = cross_attn_level  # 

        # Initial convolution
        self.conv_in = torch.nn.Conv1d(in_channels, self.ch, kernel_size=3, padding=1)
        self.spec_extractor=SpectralFeatureExtractor(in_channels=in_channels, n_fft=24, feat_dim=z_channels)
        # Downsampling stages
        in_ch_mult = (1,) + tuple(ch_mult)
        # print("ch_mult",ch_mult)
        self.down = nn.ModuleList()
        self.cross = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            crosses = nn.ModuleList()

            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))

                # 
                if i_level in self.cross_attn_level:
                    crosses.append(CrossAttnBlock(
                        query_channels=block_out,
                        guide_channels=z_channels,
                        heads=num_heads
                    ))
                else:
                    crosses.append(DummyCross())  # 


            down = nn.Module()
            down.block = block
            down.attn = attn
            down.cross   = crosses

            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample2x(block_in)
            self.down.append(down)
            self.cross.insert(0, crosses)
            

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv1d(block_in, z_channels, 3, stride=1, padding=1)

    def forward(self, x):
        # downsampling
        x = x.transpose(1, 2) # B,L,C -> B,C,L
        h = self.conv_in(x) # 128*32*24
        # 
        if self.cross_attn_level is not None:
            spec_feats = self.spec_extractor(x)  # [B, C, L] -> [B, feat_dim]
        for i_level in range(self.num_resolutions):
            # print("h.shape", h.shape)
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                h = self.down[i_level].cross[i_block](h, spec_feats)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # Middle processing
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # Final output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        # print("h",h.shape)
        # Output shape: [B, z_channels*2, L//downsample_ratio]
        return h

class GuidedDecoder(nn.Module):
    def __init__(
        self, *,
        ch=128,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        dropout=0.0,
        in_channels=6,
        z_channels,
        using_sa=True,
        using_mid_sa=True,
        num_heads=4,
        cross_attn_level=[0]  # 
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.cross_attn_level = cross_attn_level  # 

        # initial conv projecting z to feature map
        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.conv_in = nn.Conv1d(z_channels, block_in, kernel_size=3, padding=1)

        # middle layers
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # prepare upsampling structures
        self.up = nn.ModuleList()
        self.cross = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            blocks = nn.ModuleList()
            atts   = nn.ModuleList()
            crosses = nn.ModuleList()
            block_out = ch * ch_mult[i_level]

            for _ in range(self.num_res_blocks + 1):
                blocks.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                # optional self-attention only at highest resolution
                if i_level == self.num_resolutions - 1 and using_sa:
                    atts.append(make_attn(block_out, True))
                else:
                    atts.append(nn.Identity())
                # 
                if i_level in self.cross_attn_level:
                    crosses.append(CrossAttnBlock(
                        query_channels=block_out,
                        guide_channels=in_channels,
                        heads=num_heads
                    ))
                else:
                    crosses.append(DummyCross())  # 
                block_in = block_out

            up_module = nn.Module()
            up_module.block   = blocks
            up_module.attn    = atts
            up_module.cross   = crosses
            if i_level != 0:
                up_module.upsample = Upsample2x(block_in)
            self.up.insert(0, up_module)
            # self.cross.insert(0, crosses)
        # output layers
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv1d(block_in, in_channels, kernel_size=3, padding=1)

    def forward(self, z, recon_coarse):
        # z: (B, C, L_high), recon_coarse: (B, C, L_low)
        h = self.conv_in(z)

        # middle processing
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling with cross-attention at every block
        for i_level in reversed(range(self.num_resolutions)):
            level = self.up[i_level]
            for j, block in enumerate(level.block):
                h = block(h)
                h = level.attn[j](h)
                h = level.cross[j](h, recon_coarse)
            if hasattr(level, 'upsample'):
                h = level.upsample(h)

        # finalize and return (B, L, C)
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h.transpose(1, 2)


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network"""
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed forward
        ff_output = self.linear2(F.gelu(self.linear1(x)))
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x