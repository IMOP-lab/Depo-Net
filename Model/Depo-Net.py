from .unet_parts import *
from .repvit import *
import timm
from purfMamba.models.util import *

class RepViTBlock(nn.Module):
    def __init__(self,in1, inp, hidden_dim, oup, kernel_size=3, stride=2, use_se=0, use_hs=0):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and inp == oup
        assert(hidden_dim == 2 * inp)
        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
                ))
        else:
            assert(self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
                ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))
      
class DimensionMatchingLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DimensionMatchingLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x = self.conv(x)
        return x 

class DSC(nn.Module):
    def __init__(self, feature_channels, output_dim):
        super(DSC, self).__init__()
        self.feature_channels = feature_channels
        self.output_dim = output_dim
        self.conv3x3 = nn.Conv2d(feature_channels * 2, feature_channels, kernel_size=3, padding=1)
        self.conv1x1 = nn.Conv2d(feature_channels, output_dim, kernel_size=1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(output_dim, output_dim // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim // 8, output_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, e1, y1):
        concat_features = torch.cat((e1, y1), dim=1)
        g = self.conv3x3(concat_features)
        g = self.conv1x1(g)
        m = self.channel_attention(g)
        t_weighted = e1 * m
        f_weighted = y1 * m
        z_af = t_weighted + f_weighted
        return z_af
      
class HastFormer(nn.Module):
    def __init__(self, channels, num_heads, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32, dynamic_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = 1 / (self.head_dim ** 0.5)
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.dynamic_weights = nn.Parameter(torch.Tensor(num_heads, self.head_dim // dynamic_ratio, self.head_dim))
        nn.init.xavier_uniform_(self.dynamic_weights)
        self.d = max(L, channels // reduction)
        self.DSC64 = DSC(channels, channels)
        self.convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(channels, channels, kernel_size=k, padding=k//2, groups=group)),
                ('bn', nn.BatchNorm2d(channels)),
                ('relu', nn.ReLU())
            ])) for k in kernels
        ])
        self.fc = nn.Linear(channels, self.d)
        self.fcs = nn.ModuleList([nn.Linear(self.d, channels) for _ in kernels])
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x_fft = torch.fft.fftn(x, dim=(-2, -1))
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))
        xfft_real=x_fft_shifted.real
        xfft_imag=x_fft_shifted.imag
        batch_size, channels, height, width = x.size()
        low_freq = x[:, :, 0::2, 0::2]
        high_freq = x[:, :, 1::2, 1::2]
        xdwt =self.cwf64(low_freq,high_freq)
        xdwt=F.interpolate(xdwt, scale_factor=2, mode='bilinear', align_corners=True)
        q = xfft_real
        k = xfft_imag
        v = xdwt
        bsz, ch, ht, wd = q.shape
        q = q.view(bsz, self.num_heads, self.head_dim, ht * wd).permute(0, 1, 3, 2)
        k = k.view(bsz, self.num_heads, self.head_dim, ht * wd).permute(0, 1, 3, 2)
        v = v.view(bsz, self.num_heads, self.head_dim, ht * wd).permute(0, 1, 3, 2)   
        scores = torch.einsum('bnid,bnjd->bnij', q, k) * self.scale
        attention_probs = torch.softmax(scores, dim=-1)
        context = torch.einsum('bnij,bnjd->bnid', attention_probs, v)
        context = context.permute(0, 1, 3, 2).contiguous().view(bsz, -1, ht, wd)    
        output = self.output_proj(context)
        conv_outs = [conv(output) for conv in self.convs]
        feats = torch.stack(conv_outs, dim=0)
        U = torch.sum(feats, dim=0)
        S = U.mean([-2, -1]) 
        Z = self.fc(S)
        weights = [fc(Z).view(bsz, ch, 1, 1) for fc in self.fcs]
        attention_weights = self.softmax(torch.stack(weights, dim=0))
        V = torch.einsum('nbcwh,nbcwh->bcwh', attention_weights, feats)
        return context   


class SLC(nn.Module):       
    def __init__(self, in_channels, channel_in, out_channels, scale_aware_proj=False, r=4):
        super(SLC, self).__init__()
        self.scale_aware_proj = scale_aware_proj
        inter_channels = int(out_channels // r)
        self.scene_encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )
        self.content_encoder = nn.Sequential(
            nn.Conv2d(channel_in, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.feature_reencoder = nn.Sequential(
            nn.Conv2d(channel_in, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.local_att = nn.Sequential(
            nn.Conv2d(out_channels * 2, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 2, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, scene_feature, features):
        scene_feat = self.scene_encoder(scene_feature)
        content_feat = self.content_encoder(features)
        desired_height, desired_width = scene_feat.size()[2], scene_feat.size()[3]
        content_feat = nn.functional.interpolate(content_feat, size=(desired_height, desired_width), mode='bilinear', align_corners=False)
        combined_feat = torch.cat((scene_feat, content_feat), dim=1)  
        local_feat = self.local_att(combined_feat)
        global_feat = self.global_att(combined_feat)
        attention_feat = self.sigmoid(local_feat + global_feat) 
        reencoded_feature = self.feature_reencoder(features)
        enhanced_feature = reencoded_feature * attention_feat + reencoded_feature * (1 - attention_feat)
        return enhanced_feature

class SFSM(nn.Module):
    def __init__(self, in_channels):
        super(SFSM, self).__init__()
        self.in_channels = in_channels
        self.learnable_filter = nn.Parameter(torch.randn(1, in_channels, 1, 1))
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.global_local_fusion = GLF(in_channels, in_channels, in_channels)
        # self.dualpool=DualPathPooling(in_channels)
    def forward(self, x):
        device = x.device  
        x_dft = torch.fft.fft2(x)
        high_mask = self.create_high_mask(x_dft.size(), device) 
        low_mask = self.create_low_mask(x_dft.size(), device)   
        high_freq = x_dft * high_mask
        low_freq = x_dft * low_mask
        low_freq = low_freq * self.learnable_filter.to(device) 
        high_freq_spatial = torch.fft.ifft2(high_freq).real
        low_freq_spatial = torch.fft.ifft2(low_freq).real
        concatenated = torch.cat([high_freq_spatial, low_freq_spatial], dim=1)
        conv_out = self.conv(concatenated)
        attention_map = self.sigmoid(conv_out)
        attention_output = x * attention_map
        max_out = self.max_pool(x)
        SLC_result = self.SLC(x, max_out)   
        gap_out = self.gap(SLC_result)
        combined = self.fc(gap_out)
        combined = self.sigmoid(combined)
        x1 = x* combined
        output = x1 + attention_output
        return output
    def create_high_mask(self, size, device):
        mask = torch.ones(size, dtype=torch.cfloat, device=device) 
        h, w = size[-2], size[-1]
        mask[:, :, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0
        return mask
    def create_low_mask(self, size, device):
        mask = torch.zeros(size, dtype=torch.cfloat, device=device)
        h, w = size[-2], size[-1]
        mask[:, :, h // 4:3 * h // 4, w // 4:3 * w // 4] = 1
        return mask
      
class Depo_Net(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Depo_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        def create_hastformer(channels):
            return HastFormer(
                channels=channels,num_heads=8,
                kernels=[1, 3, 5, 7], reduction=16,
                group=1, L=32,dynamic_ratio=4 )
        self.HastFormer = create_hastformer(128)
        self.HastFormer256 = create_hastformer(256)
        self.HastFormer512 = create_hastformer(512)
        self.HastFormer1024 = create_hastformer(1024)

        self.dim_match_layer1= DimensionMatchingLayer(96, 64)
        self.dim_match_layer2 = DimensionMatchingLayer(96, 128)
        self.dim_match_layer3= DimensionMatchingLayer(192, 256)
        self.dim_match_layer4= DimensionMatchingLayer(384, 512)
        self.dim_match_layer5= DimensionMatchingLayer(768, 1024)  

        self.DSC64 = DSC(64, 64)
        self.DSC128 = DSC(128, 128)
        self.DSC256 = DSC(256, 256)
        self.DSC512 = DSC(512, 512)
        self.DSC1024 = DSC(1024, 1024)
      
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.repvit = repvit_m1_1(pretrained=True, num_classes=1000)
        self.maxvit = timm.create_model(
            'maxxvit_rmlp_small_rw_256.sw_in1k',
            pretrained=False,
            features_only=True,
        )
        self.purfMamba64 = purfMamba(hidden_dim=64)       
        self.purfMamba128 = purfMamba(hidden_dim=128)
        self.purfMamba256 = purfMamba(hidden_dim=256)       
        self.purfMamba512 = purfMamba(hidden_dim=512)
        self.purfMamba1024 = purfMamba(hidden_dim=1024)    

        self.SFSM64=SFSM(64)
        self.SFSM128=SFSM(128)
        self.SFSM256=SFSM(256)
        self.SFSM512=SFSM(512)
        self.SFSM1024=SFSM(1024)     
    def forward(self, x):
        y1,y2=self.repvit(x)
        maxvit_features = self.maxvit(x)
        e1, e2, e3, e4, e5 = maxvit_features[:5] 
        e1=self.dim_match_layer1(e1)
        e2=self.dim_match_layer2(e2)
        e3=self.dim_match_layer3(e3)
        e4=self.dim_match_layer4(e4)
        e5=self.dim_match_layer5(e5)
        y1[1]=self.SFSM64(y1[1])
        y1[2]=self.SFSM128(y1[2])
        y1[3]=self.SFSM256(y1[3])
        y1[4]=self.SFSM512(y1[4])
        y1[5]=self.SFSM1024(y1[5])
        v1= self.DSC64(y1[1],e1)   # 64 112 112
        v2= self.DSC128(y1[2],e2)  # 128, 56, 56
        v2=self.HastFormer(v2)
        v3= self.DSC256(y1[3],e3)  # 256, 28, 28
        v3=self.HastFormer256(v3)
        v4= self.DSC512(y1[4],e4)  # 512, 14, 14
        v4=self.HastFormer512(v4)
        v5= self.DSC1024(y1[5],e5) # 1024, 7, 7
        v5 = self.purfMamba1024(v5)
        x = self.up1(v5, v4)
        x=self.purfMamba512(x)
        x = self.up2(x, v3)   
        x=self.purfMamba256(x)       
        x = self.up3(x, v2)  
        x=self.purfMamba128(x)     
        x = self.up4(x, v1) 
        x=self.purfMamba64(x)          
        x=F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        logits = self.outc(x)  
        return logits
