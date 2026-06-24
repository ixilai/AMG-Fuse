import torch
import torch.nn as nn
import numbers
from einops import rearrange
import torch.nn.functional as F

Conv2d = nn.Conv2d


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention_histogram(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class SpaAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(SpaAttention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C / 2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 2 / 3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 3 / 4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 4 / 5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

# LayerNorm related code can be inserted here if needed.
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

## Dynamic-range Histogram Self-Attention (DHSA)

class Attention_histogram(nn.Module):
    def __init__(self, dim, num_heads, bias, ifBox=True):
        super(Attention_histogram, self).__init__()
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_dwconv = Conv2d(dim * 5, dim * 5, kernel_size=3, stride=1, padding=1, groups=dim * 5, bias=bias)
        self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        _, _, hw = x.shape
        return x[:, :, t_pad[0]:hw - t_pad[1]]

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def normalize(self, x):
        mu = x.mean(-2, keepdim=True)
        sigma = x.var(-2, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5)  # * self.weight + self.bias

    def reshape_attn(self, q, k, v, ifBox):
        b, c = q.shape[:2]
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)
        hw = q.shape[-1] // self.factor
        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b,
                        head=self.num_heads)
        out = self.unpad(out, t_pad)
        return out

    def forward(self, x):
        b, c, h, w = x.shape
        x_sort, idx_h = x[:, :c // 2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x[:, :c // 2] = x_sort
        qkv = self.qkv_dwconv(self.qkv(x))
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)  # b,c,x,x

        v, idx = v.view(b, c, -1).sort(dim=-1)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        out1 = self.reshape_attn(q1, k1, v, True)
        out2 = self.reshape_attn(q2, k2, v, False)

        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w)
        out = out1 * out2
        out = self.project_out(out)
        out_replace = out[:, :c // 2]
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        out[:, :c // 2] = out_replace
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                   groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2,
                                   groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1,
                                     groups=hidden_features, bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2,
                                     groups=hidden_features, bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)

        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_3, x2_5], dim=1)

        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))

        x = torch.cat([x1, x2], dim=1)

        x = self.project_out(x)

        return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class MSAKNet(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=2,
                 heads=[2, 4, 6, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False,  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 use_restormer = True
                 ):

        super(MSAKNet, self).__init__()

        self.patch_embed_vi = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed_ir = OverlapPatchEmbed(1, dim)
        self.patch_embed_f = OverlapPatchEmbed(inp_channels, dim)

                #                      LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
                #                      LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.encoder_level1_vi = nn.Sequential(ResBlock(dim), nn.ReLU())
        self.encoder_level1_ir = nn.Sequential(ResBlock(dim), nn.ReLU())
        self.encoder_level1_f = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2_vi = Downsample(dim)  ## From Level 1 to Level 2
        self.down1_2_ir = Downsample(dim)  ## From Level 1 to Level 2
        self.down1_2_f = Downsample(dim)  ## From Level 1 to Level 2

                #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
                #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level2_vi = nn.Sequential(ResBlock(dim * 2 ** 1), nn.ReLU())
        self.encoder_level2_ir = nn.Sequential(ResBlock(dim * 2 ** 1), nn.ReLU())
        self.encoder_level2_f = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3_vi = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.down2_3_ir = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.down2_3_f = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3

                #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
                #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.encoder_level3_vi = nn.Sequential(ResBlock(dim * 2 ** 2), nn.ReLU())
        self.encoder_level3_ir = nn.Sequential(ResBlock(dim * 2 ** 2), nn.ReLU())
        self.encoder_level3_f = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4_vi = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.down3_4_ir = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.down3_4_f = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4

                #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
                #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.latent_vi = nn.Sequential(ResBlock(dim * 2 ** 3), nn.ReLU())
        self.latent_ir = nn.Sequential(ResBlock(dim * 2 ** 3), nn.ReLU())
        self.latent_f = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3_vi = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.up4_3_ir = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.up4_3_f = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3

        self.reduce_chan_level3_vi = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.reduce_chan_level3_ir = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.reduce_chan_level3_f = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)

                #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
                #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.decoder_level3_vi = nn.Sequential(ResBlock(dim * 2 ** 2), nn.ReLU())
        self.decoder_level3_ir = nn.Sequential(ResBlock(dim * 2 ** 2), nn.ReLU())
        self.decoder_level3_f = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2_vi = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.up3_2_ir = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.up3_2_f = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2

        self.reduce_chan_level2_vi = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level2_ir = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level2_f = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)


                #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
                #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.decoder_level2_vi = nn.Sequential(ResBlock(dim * 2 ** 1), nn.ReLU())
        self.decoder_level2_ir = nn.Sequential(ResBlock(dim * 2 ** 1), nn.ReLU())
        self.decoder_level2_f = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1_vi = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.up2_1_ir = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.up2_1_f = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

                #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
                #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level1_vi = nn.Sequential(ResBlock(dim * 2 ** 1), nn.ReLU())
        self.decoder_level1_ir = nn.Sequential(ResBlock(dim * 2 ** 1), nn.ReLU())
        self.decoder_level1_f = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

                #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
                #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.refinement_vi = nn.Sequential(ResBlock(dim * 2 ** 1), nn.ReLU())
        self.refinement_ir = nn.Sequential(ResBlock(dim * 2 ** 1), nn.ReLU())
        self.refinement_f = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output_vi = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_ir = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_f = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.mask_processors = nn.ModuleList([
            self._make_mask_processor(dim),
            self._make_mask_processor(dim * 2 ** 1),
            self._make_mask_processor(dim * 2 ** 2),
            self._make_mask_processor(dim * 2 ** 3),
            self._make_mask_processor(dim * 2 ** 2),
            self._make_mask_processor(dim * 2 ** 1),
            self._make_mask_processor(dim * 2 ** 1),
            self._make_mask_processor(out_channels)
        ])

        # Restormer restoration branch for degradation-aware learning
        self.use_restormer = use_restormer
        if self.use_restormer:
            self.res = Restormer()
            # Note: Restormer checkpoint should be pre-trained for specific weather conditions
            # (e.g., derain, desnow, dehaze). Load the appropriate checkpoint during training.
            # The checkpoint path should be provided externally (not hardcoded).
            # For inference without TDAS, set use_restormer=False
            self.res.eval()
            for param in self.res.parameters():
                param.requires_grad = False

        self.CA_processors = nn.ModuleList([ChannelFusionAttention(dim, num_head=heads[0], bias=bias),
                                 ChannelFusionAttention(dim * 2 ** 1, num_head=heads[1], bias=bias),
                                 ChannelFusionAttention(dim * 2 ** 2, num_head=heads[2], bias=bias),
                                 ChannelFusionAttention(dim * 2 ** 3, num_head=heads[3], bias=bias),
                                 ChannelFusionAttention(dim * 2 ** 2, num_head=heads[2], bias=bias),
                                 ChannelFusionAttention(dim * 2 ** 1, num_head=heads[1], bias=bias),
                                 ChannelFusionAttention(dim * 2 ** 1, num_head=heads[0], bias=bias),
                                 ChannelFusionAttention(dim * 2 ** 1, num_head=heads[0], bias=bias),
                                 self._make_mask_processor(out_channels)
                                 ])


    def forward(self, vi, ir):

        vi_enc_1 = self.patch_embed_vi(vi)
        ir_enc_1 = self.patch_embed_ir(ir)

        out_enc_level1_vi = self.encoder_level1_vi(vi_enc_1)
        out_enc_level1_ir = self.encoder_level1_ir(ir_enc_1)

        out_enc_level1_fuse = out_enc_level1_vi + out_enc_level1_ir

        out_enc_level1_vi, out_enc_level1_ir, out_enc_level1_fuse, _ = self._fuse(out_enc_level1_vi, out_enc_level1_ir, out_enc_level1_fuse,  self.mask_processors[0], self.CA_processors[0])


        inp_enc_level2_vi = self.down1_2_vi(out_enc_level1_vi)
        inp_enc_level2_ir = self.down1_2_ir(out_enc_level1_ir)
        inp_enc_level2_fuse = self.down1_2_f(out_enc_level1_fuse)


        out_enc_level2_vi = self.encoder_level2_vi(inp_enc_level2_vi)
        out_enc_level2_ir = self.encoder_level2_ir(inp_enc_level2_ir)
        out_enc_level2_fuse = self.encoder_level2_f(inp_enc_level2_fuse)

        out_enc_level2_vi, out_enc_level2_ir, out_enc_level2_fuse, _ = self._fuse(out_enc_level2_vi, out_enc_level2_ir, out_enc_level2_fuse, self.mask_processors[1],self.CA_processors[1])

        inp_enc_level3_vi = self.down2_3_vi(out_enc_level2_vi)
        inp_enc_level3_ir = self.down2_3_ir(out_enc_level2_ir)
        inp_enc_level3_fuse = self.down2_3_f(out_enc_level2_fuse)


        out_enc_level3_vi = self.encoder_level3_vi(inp_enc_level3_vi)
        out_enc_level3_ir = self.encoder_level3_ir(inp_enc_level3_ir)
        out_enc_level3_fuse = self.encoder_level3_f(inp_enc_level3_fuse)

        out_enc_level3_vi, out_enc_level3_ir, out_enc_level3_fuse, _ = self._fuse(out_enc_level3_vi, out_enc_level3_ir, out_enc_level3_fuse, self.mask_processors[2],self.CA_processors[2])

        inp_enc_level4_vi = self.down3_4_vi(out_enc_level3_vi)
        inp_enc_level4_ir = self.down3_4_ir(out_enc_level3_ir)
        inp_enc_level4_fuse = self.down3_4_f(out_enc_level3_fuse)


        latent_vi = self.latent_vi(inp_enc_level4_vi)
        latent_ir = self.latent_ir(inp_enc_level4_ir)
        latent_fuse = self.latent_f(inp_enc_level4_fuse)

        latent_vi, latent_ir, latent_fuse, _ = self._fuse(latent_vi, latent_ir, latent_fuse, self.mask_processors[3],self.CA_processors[3])

        inp_dec_level3_vi = self.up4_3_vi(latent_vi)
        inp_dec_level3_ir = self.up4_3_ir(latent_ir)
        inp_dec_level3_fuse = self.up4_3_f(latent_fuse)


        inp_dec_level3_vi = torch.cat([inp_dec_level3_vi, out_enc_level3_vi], 1)
        inp_dec_level3_ir = torch.cat([inp_dec_level3_ir, out_enc_level3_ir], 1)
        inp_dec_level3_fuse = torch.cat([inp_dec_level3_fuse, out_enc_level3_fuse], 1)



        inp_dec_level3_vi = self.reduce_chan_level3_vi(inp_dec_level3_vi)
        inp_dec_level3_ir = self.reduce_chan_level3_ir(inp_dec_level3_ir)
        inp_dec_level3_fuse = self.reduce_chan_level3_f(inp_dec_level3_fuse)


        out_dec_level3_vi = self.decoder_level3_vi(inp_dec_level3_vi)
        out_dec_level3_ir = self.decoder_level3_ir(inp_dec_level3_ir)
        out_dec_level3_fuse = self.decoder_level3_f(inp_dec_level3_fuse)

        out_dec_level3_vi, out_dec_level3_ir, out_dec_level3_fuse, _ = self._fuse(out_dec_level3_vi, out_dec_level3_ir, out_dec_level3_fuse, self.mask_processors[4],self.CA_processors[4])

        inp_dec_level2_vi = self.up3_2_vi(out_dec_level3_vi)
        inp_dec_level2_ir = self.up3_2_ir(out_dec_level3_ir)
        inp_dec_level2_fuse = self.up3_2_f(out_dec_level3_fuse)


        inp_dec_level2_vi = torch.cat([inp_dec_level2_vi, out_enc_level2_vi], 1)
        inp_dec_level2_ir = torch.cat([inp_dec_level2_ir, out_enc_level2_ir], 1)
        inp_dec_level2_fuse = torch.cat([inp_dec_level2_fuse, out_enc_level2_fuse], 1)


        out_dec_level2_vi = self.reduce_chan_level2_vi(inp_dec_level2_vi)
        out_dec_level2_ir = self.reduce_chan_level2_ir(inp_dec_level2_ir)
        out_dec_level2_fuse = self.reduce_chan_level2_f(inp_dec_level2_fuse)

        out_dec_level2_vi = self.decoder_level2_vi(out_dec_level2_vi)
        out_dec_level2_ir = self.decoder_level2_ir(out_dec_level2_ir)
        out_dec_level2_fuse = self.decoder_level2_f(out_dec_level2_fuse)

        out_dec_level2_vi, out_dec_level2_ir, out_dec_level2_fuse, _ = self._fuse(out_dec_level2_vi, out_dec_level2_ir, out_dec_level2_fuse, self.mask_processors[5],self.CA_processors[5])

        inp_dec_level1_vi = self.up2_1_vi(out_dec_level2_vi)
        inp_dec_level1_ir = self.up2_1_ir(out_dec_level2_ir)
        inp_dec_level1_fuse = self.up2_1_f(out_dec_level2_fuse)


        inp_dec_level1_vi = torch.cat([inp_dec_level1_vi, out_enc_level1_vi], 1)
        inp_dec_level1_ir = torch.cat([inp_dec_level1_ir, out_enc_level1_ir], 1)
        inp_dec_level1_fuse = torch.cat([inp_dec_level1_fuse, out_enc_level1_fuse], 1)

        out_dec_level1_vi = self.decoder_level1_vi(inp_dec_level1_vi)
        out_dec_level1_ir = self.decoder_level1_ir(inp_dec_level1_ir)
        out_dec_level1_fuse = self.decoder_level1_f(inp_dec_level1_fuse)

        out_dec_level1_vi, out_dec_level1_ir, out_dec_level1_fuse, _ = self._fuse(out_dec_level1_vi, out_dec_level1_ir, out_dec_level1_fuse, self.mask_processors[6],self.CA_processors[6])

        out_dec_level1_vi = self.refinement_vi(out_dec_level1_vi)
        out_dec_level1_ir = self.refinement_ir(out_dec_level1_ir)
        out_dec_level1_fuse = self.refinement_f(out_dec_level1_fuse)

        out_dec_level1_vi = self.output_vi(out_dec_level1_vi)
        out_dec_level1_ir = self.output_ir(out_dec_level1_ir)
        out_dec_level1_fuse = self.output_f(out_dec_level1_fuse)

        _, _, _, out_mask = self._sigfuse(out_dec_level1_vi, out_dec_level1_ir, out_dec_level1_fuse, self.mask_processors[7],self.CA_processors[7])

        Fuse = out_dec_level1_fuse

        ir_3c = ir.repeat(1, 3, 1, 1)
        mask = (Fuse - vi) / (vi - ir_3c + Fuse + 1e-8)
        mask = torch.clamp(mask, 0, 1)
        mask = torch.sigmoid((mask - 0.5) * 5)

        res_image = mask * vi

        if self.use_restormer:
            final = self.res(res_image)
        else:
            final = res_image

        return  out_dec_level1_vi, Fuse, out_dec_level1_ir, final, res_image


    def _make_mask_processor(self, channels):
        return nn.Sequential(ResBlock(channels), nn.Sigmoid())

    def _fuse(self, vi, ir, fuse, mask_module, CA_module,  iters=1):
        fea_vi, fea_ir = vi, ir
        eps = 1e-8

        for _ in range(iters):
            mask = (fuse - fea_ir) / (fea_vi - fea_ir + fuse + eps)
            mask = torch.clamp(mask, 0, 1)
            mask = torch.sigmoid((mask - 0.5) * 5)
            mask = mask_module(mask)
            fea_vi, fea_ir, fuse_out = CA_module(vi, ir, mask, fuse)
            fuse = fuse_out + fuse
            fea_vi = fea_vi + vi
            fea_ir = fea_ir + ir


        return fea_vi, fea_ir, fuse, mask

    def _sigfuse(self, vi, ir, fuse, mask_module, CA_module,  iters=1):
        fea_vi, fea_ir = vi, ir
        eps = 1e-8

        for _ in range(iters):
            mask = (fuse - fea_ir) / (fea_vi - fea_ir + fuse + eps)
            mask = torch.clamp(mask, 0, 1)
            mask = torch.sigmoid((mask - 0.5) * 5)
            mask = mask_module(mask)


        return fea_vi, fea_ir, fuse, mask




class ChannelFusionAttention(nn.Module):
    def __init__(self, dim, num_head, bias):
        super(ChannelFusionAttention, self).__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1), requires_grad=True)

        self.q_vi = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv_vi = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.q_ir = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv_ir = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)


        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.project_out_vi = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out_ir = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out_f = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


        reduction = 16
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim*2, dim//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//reduction, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y, mask, fuse):
        # x -> q, y -> kv
        assert x.shape == y.shape, 'The shape of feature maps from image and features are not equal!'

        b, c, h, w = x.shape

        x = x * mask
        y = y * (1-mask)

        q_vi = self.q_dwconv_vi(self.q_vi(x))
        q_ir = self.q_dwconv_ir(self.q_ir(y))

        kv = self.kv_dwconv(self.kv(fuse))
        k, v = kv.chunk(2, dim=1)

        q_vi = rearrange(q_vi, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        q_ir = rearrange(q_ir, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_head)

        q_vi = torch.nn.functional.normalize(q_vi, dim=-1)
        q_ir = torch.nn.functional.normalize(q_ir, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn_vi = q_vi @ k.transpose(-2, -1) * self.temperature
        attn_vi = attn_vi.softmax(dim=-1)

        attn_ir = q_ir @ k.transpose(-2, -1) * self.temperature
        attn_ir = attn_ir.softmax(dim=-1)

        out_vi = attn_vi @ v
        out_ir = attn_ir @ v

        out_vi = rearrange(out_vi, 'b head c (h w) -> b (head c) h w', head=self.num_head, h=h, w=w)
        out_ir = rearrange(out_ir, 'b head c (h w) -> b (head c) h w', head=self.num_head, h=h, w=w)

        cat = torch.cat([out_vi, out_ir], dim=1)   # B,2C,H,W
        alpha = self.se(cat)              # B,2,1,1
        a_vi, a_ir = alpha[:,0:1], alpha[:,1:2]  # each [B,1,1,1]

        out_vi_F = a_vi * out_vi
        out_ir_F = a_ir * out_ir

        fuse = out_vi_F + out_ir_F

        out_vi = self.project_out_vi(out_vi)
        out_ir = self.project_out_ir(out_ir)
        fuse = self.project_out_f(fuse)
        return out_vi, out_ir, fuse

class Restormer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=2,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)


        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1