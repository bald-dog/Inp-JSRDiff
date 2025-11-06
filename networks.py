import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# 你原有的网络定义（如CBAM等）
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_att = self.sigmoid_channel(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(spatial_att))
        x = x * spatial_att
        return x

# 添加一个UNet网络用于边缘恢复
# class EdgeRestoreUNet(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, init_features=32):
#         super(EdgeRestoreUNet, self).__init__()
        
#         features = init_features
#         self.encoder1 = self._block(in_channels, features, name="enc1")
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder2 = self._block(features, features * 2, name="enc2")
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder3 = self._block(features * 2, features * 4, name="enc3")
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder4 = self._block(features * 4, features * 8, name="enc4")
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")

#         self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
#         self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
#         self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
#         self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
#         self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
#         self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
#         self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
#         self.decoder1 = self._block(features * 2, features, name="dec1")

#         self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)
        
#         # 添加CBAM注意力机制以提高边缘恢复效果
#         self.cbam_bottleneck = CBAM(features * 16)
#         self.cbam_dec4 = CBAM(features * 8)
#         self.cbam_dec3 = CBAM(features * 4)
#         self.cbam_dec2 = CBAM(features * 2)
#         self.cbam_dec1 = CBAM(features)

#     def forward(self, x):
#         enc1 = self.encoder1(x)
#         enc2 = self.encoder2(self.pool1(enc1))
#         enc3 = self.encoder3(self.pool2(enc2))
#         enc4 = self.encoder4(self.pool3(enc3))

#         bottleneck = self.bottleneck(self.pool4(enc4))
#         bottleneck = self.cbam_bottleneck(bottleneck)

#         dec4 = self.upconv4(bottleneck)
#         dec4 = torch.cat((dec4, enc4), dim=1)
#         dec4 = self.decoder4(dec4)
#         dec4 = self.cbam_dec4(dec4)

#         dec3 = self.upconv3(dec4)
#         dec3 = torch.cat((dec3, enc3), dim=1)
#         dec3 = self.decoder3(dec3)
#         dec3 = self.cbam_dec3(dec3)

#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat((dec2, enc2), dim=1)
#         dec2 = self.decoder2(dec2)
#         dec2 = self.cbam_dec2(dec2)

#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((dec1, enc1), dim=1)
#         dec1 = self.decoder1(dec1)
#         dec1 = self.cbam_dec1(dec1)

#         return torch.sigmoid(self.conv(dec1))

#     def _block(self, in_channels, features, name):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(features),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(features),
#             nn.ReLU(inplace=True)
#         )



# 添加SE注意力模块
# class SEAttention(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SEAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y

# 残差块替代普通卷积块
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(residual)
        out = self.relu(out)
        return out

# # 改进的EdgeRestoreUNet
# class EdgeRestoreUNet(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, init_features=64):
#         super(EdgeRestoreUNet, self).__init__()
        
#         features = init_features
        
#         # 编码器使用残差块
#         self.encoder1 = ResBlock(in_channels, features)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         self.encoder2 = ResBlock(features, features * 2)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         self.encoder3 = ResBlock(features * 2, features * 4)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         self.encoder4 = ResBlock(features * 4, features * 8)
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

#         # 瓶颈层使用空洞卷积增大感受野
#         self.bottleneck = nn.Sequential(
#             ResBlock(features * 8, features * 16, dilation=2),
#             ResBlock(features * 16, features * 16, dilation=4)
#         )
        
#         # 注意力模块结合SE和CBAM
#         self.cbam_bottleneck = CBAM(features * 16, reduction=8)
#         self.se_bottleneck = SEAttention(features * 16, reduction=8)

#         # 解码器路径
#         self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
#         self.decoder4 = ResBlock(features * 8 * 2, features * 8)
#         self.cbam_dec4 = CBAM(features * 8)
        
#         self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
#         self.decoder3 = ResBlock(features * 4 * 2, features * 4)
#         self.cbam_dec3 = CBAM(features * 4)
        
#         self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
#         self.decoder2 = ResBlock(features * 2 * 2, features * 2)
#         self.cbam_dec2 = CBAM(features * 2)
        
#         self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
#         self.decoder1 = ResBlock(features * 2, features)
#         self.cbam_dec1 = CBAM(features)

#         # 多尺度特征融合
#         self.multiscale_conv1 = nn.Conv2d(features, features, kernel_size=1)
#         self.multiscale_conv2 = nn.Conv2d(features * 2, features, kernel_size=1)
#         self.multiscale_conv3 = nn.Conv2d(features * 4, features, kernel_size=1)
#         self.multiscale_fuse = nn.Conv2d(features * 3, features, kernel_size=1)
        
#         # 最终输出层
#         self.final_conv = nn.Sequential(
#             nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
#             nn.BatchNorm2d(features // 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(features // 2, out_channels, kernel_size=1),
#             nn.Sigmoid()
#         )
        
#         # 边缘增强层
#         self.edge_enhance = nn.Sequential(
#             nn.Conv2d(out_channels, 16, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # 编码器路径
#         enc1 = self.encoder1(x)
#         enc2 = self.encoder2(self.pool1(enc1))
#         enc3 = self.encoder3(self.pool2(enc2))
#         enc4 = self.encoder4(self.pool3(enc3))

#         # 瓶颈层 + 双重注意力
#         bottleneck = self.bottleneck(self.pool4(enc4))
#         bottleneck = self.cbam_bottleneck(bottleneck)
#         bottleneck = self.se_bottleneck(bottleneck)

#         # 解码器路径
#         dec4 = self.upconv4(bottleneck)
#         dec4 = torch.cat((dec4, enc4), dim=1)
#         dec4 = self.decoder4(dec4)
#         dec4 = self.cbam_dec4(dec4)

#         dec3 = self.upconv3(dec4)
#         dec3 = torch.cat((dec3, enc3), dim=1)
#         dec3 = self.decoder3(dec3)
#         dec3 = self.cbam_dec3(dec3)

#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat((dec2, enc2), dim=1)
#         dec2 = self.decoder2(dec2)
#         dec2 = self.cbam_dec2(dec2)

#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((dec1, enc1), dim=1)
#         dec1 = self.decoder1(dec1)
#         dec1 = self.cbam_dec1(dec1)

#         # 多尺度特征融合
#         ms_feat1 = self.multiscale_conv1(dec1)
#         ms_feat2 = self.multiscale_conv2(F.interpolate(dec2, size=dec1.shape[2:], mode='bilinear', align_corners=False))
#         ms_feat3 = self.multiscale_conv3(F.interpolate(dec3, size=dec1.shape[2:], mode='bilinear', align_corners=False))
        
#         ms_fused = torch.cat([ms_feat1, ms_feat2, ms_feat3], dim=1)
#         ms_fused = self.multiscale_fuse(ms_fused)
        
#         # 最终输出
#         out = self.final_conv(ms_fused)
        
#         # 边缘增强
#         edge_enhanced = self.edge_enhance(out)
        
#         # 残差连接增强边缘
#         final_output = out + edge_enhanced
        
#         return torch.clamp(final_output, 0, 1)




# 添加Transformer相关模块
# class MultiHeadSelfAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, dropout=0.1):
#         super(MultiHeadSelfAttention, self).__init__()
#         self.num_heads = num_heads
#         self.dim = dim
#         self.head_dim = dim // num_heads
#         assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
#         self.qkv = nn.Linear(dim, dim * 3, bias=False)
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(dropout)
        
#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # B, num_heads, N, head_dim
        
#         attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # B, num_heads, N, N
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
        
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

# class TransformerBlock(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0.1):
#         super(TransformerBlock, self).__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = MultiHeadSelfAttention(dim, num_heads=num_heads, dropout=dropout)
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, int(dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(int(dim * mlp_ratio), dim),
#             nn.Dropout(dropout)
#         )
        
#     def forward(self, x):
#         x = x + self.attn(self.norm1(x))
#         x = x + self.mlp(self.norm2(x))
#         return x

# class ConvToTransformer(nn.Module):
#     def __init__(self, in_channels, embed_dim):
#         super(ConvToTransformer, self).__init__()
#         self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        
#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.proj(x)  # B, embed_dim, H, W
#         x = x.flatten(2).transpose(1, 2)  # B, H*W, embed_dim
#         return x, (H, W)

# class TransformerToConv(nn.Module):
#     def __init__(self, embed_dim, out_channels):
#         super(TransformerToConv, self).__init__()
#         self.proj = nn.Conv2d(embed_dim, out_channels, kernel_size=1)
        
#     def forward(self, x, size):
#         H, W = size
#         B, N, C = x.shape
#         x = x.transpose(1, 2).reshape(B, C, H, W)
#         x = self.proj(x)
#         return x

# # 改进的边缘感知模块
# class EdgeAwareModule(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(EdgeAwareModule, self).__init__()
#         self.edge_detect = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.edge_enhance = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.relu = nn.LeakyReLU(0.2, inplace=True)
        
#     def forward(self, x):
#         edges = self.edge_detect(x)
#         edges = self.relu(edges)
#         edges = self.edge_enhance(edges)
#         return edges

# # 改进的混合Transformer-UNet架构
# class EdgeRestoreUNet(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, init_features=32):
#         super(EdgeRestoreUNet, self).__init__()
        
#         features = init_features
        
#         # 编码器使用残差块
#         self.encoder1 = ResBlock(in_channels, features)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         self.encoder2 = ResBlock(features, features * 2)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         self.encoder3 = ResBlock(features * 2, features * 4)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         self.encoder4 = ResBlock(features * 4, features * 8)
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         # 瓶颈层使用Transformer
#         bottleneck_dim = features * 8
#         self.bottleneck_conv = ResBlock(features * 8, bottleneck_dim)
        
#         # Transformer相关组件
#         self.conv_to_transformer = ConvToTransformer(bottleneck_dim, bottleneck_dim)
        
#         # Transformer块
#         self.transformer_blocks = nn.ModuleList([
#             TransformerBlock(dim=bottleneck_dim, num_heads=8, mlp_ratio=4.0, dropout=0.1)
#             for _ in range(3)  # 使用3个Transformer块
#         ])
        
#         self.transformer_to_conv = TransformerToConv(bottleneck_dim, bottleneck_dim)
        
#         # 瓶颈层后的注意力
#         self.cbam_bottleneck = CBAM(bottleneck_dim, reduction=8)
        
#         # 解码器路径
#         self.upconv4 = nn.ConvTranspose2d(bottleneck_dim, features * 8, kernel_size=2, stride=2)
#         self.decoder4 = ResBlock(features * 8 * 2, features * 8)
#         self.cbam_dec4 = CBAM(features * 8)
        
#         self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
#         self.decoder3 = ResBlock(features * 4 * 2, features * 4)
#         self.cbam_dec3 = CBAM(features * 4)
        
#         self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
#         self.decoder2 = ResBlock(features * 2 * 2, features * 2)
#         self.cbam_dec2 = CBAM(features * 2)
        
#         self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
#         self.decoder1 = ResBlock(features * 2, features)
#         self.cbam_dec1 = CBAM(features)
        
#         # 边缘感知模块
#         self.edge_aware = EdgeAwareModule(features, features)
        
#         # 多尺度特征融合
#         self.multiscale_conv1 = nn.Conv2d(features, features, kernel_size=1)
#         self.multiscale_conv2 = nn.Conv2d(features * 2, features, kernel_size=1)
#         self.multiscale_conv3 = nn.Conv2d(features * 4, features, kernel_size=1)
#         self.multiscale_fuse = nn.Conv2d(features * 3, features, kernel_size=1)
        
#         # 最终输出层
#         self.final_conv = nn.Sequential(
#             nn.Conv2d(features * 2, features, kernel_size=3, padding=1),
#             nn.BatchNorm2d(features),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(features, out_channels, kernel_size=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # 编码器路径
#         enc1 = self.encoder1(x)
#         enc2 = self.encoder2(self.pool1(enc1))
#         enc3 = self.encoder3(self.pool2(enc2))
#         enc4 = self.encoder4(self.pool3(enc3))
        
#         # 瓶颈层
#         bottleneck = self.bottleneck_conv(self.pool4(enc4))
        
#         # 转换为Transformer处理
#         trans_features, size = self.conv_to_transformer(bottleneck)
        
#         # 通过Transformer块
#         for block in self.transformer_blocks:
#             trans_features = block(trans_features)
            
#         # 转回卷积特征
#         bottleneck = self.transformer_to_conv(trans_features, size)
        
#         # 应用CBAM注意力
#         bottleneck = self.cbam_bottleneck(bottleneck)
        
#         # 解码器路径
#         dec4 = self.upconv4(bottleneck)
#         dec4 = torch.cat((dec4, enc4), dim=1)
#         dec4 = self.decoder4(dec4)
#         dec4 = self.cbam_dec4(dec4)
        
#         dec3 = self.upconv3(dec4)
#         dec3 = torch.cat((dec3, enc3), dim=1)
#         dec3 = self.decoder3(dec3)
#         dec3 = self.cbam_dec3(dec3)
        
#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat((dec2, enc2), dim=1)
#         dec2 = self.decoder2(dec2)
#         dec2 = self.cbam_dec2(dec2)
        
#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((dec1, enc1), dim=1)
#         dec1 = self.decoder1(dec1)
#         dec1 = self.cbam_dec1(dec1)
        
#         # 边缘感知处理
#         edge_features = self.edge_aware(dec1)
        
#         # 多尺度特征融合
#         ms_feat1 = self.multiscale_conv1(dec1)
#         ms_feat2 = self.multiscale_conv2(F.interpolate(dec2, size=dec1.shape[2:], mode='bilinear', align_corners=False))
#         ms_feat3 = self.multiscale_conv3(F.interpolate(dec3, size=dec1.shape[2:], mode='bilinear', align_corners=False))
        
#         ms_fused = torch.cat([ms_feat1, ms_feat2, ms_feat3], dim=1)
#         ms_fused = self.multiscale_fuse(ms_fused)
        
#         # 组合边缘特征和多尺度特征
#         combined_features = torch.cat([ms_fused, edge_features], dim=1)
        
#         # 最终输出
#         output = self.final_conv(combined_features)
        
#         #return torch.clamp(output, 0, 1)
#         return output

# 从EdgeConnect项目导入的EdgeGenerator实现
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        初始化网络权重
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        # 使用非就地操作
        return x + self.conv_block(x)  # 使用+而不是+=

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module
# def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
#     """添加epsilon值的谱归一化，防止除零错误"""
#     return torch.nn.utils.spectral_norm(module, name, n_power_iterations, eps, dim)
class EdgeGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, use_spectral_norm=True, init_weights=True):
        super(EdgeGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        #x = torch.sigmoid(x)
        return x
    

class EdgeRestoreUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(EdgeRestoreUNet, self).__init__()
        # EdgeGenerator期望3通道输入: 灰度图(1) + 边缘(1) + 掩码(1)
        self.generator = EdgeGenerator(residual_blocks=8, use_spectral_norm=True)
        # 添加判别器: 输入通道为2 (灰度图+边缘)
        self.discriminator = Discriminator(in_channels=2, use_sigmoid=True)
        
        # 初始化为训练模式
        self.training = True
        
        # 损失函数设置
        self.l1_loss = nn.L1Loss()
        self.adversarial_loss = AdversarialLoss(type='nsgan')
        
        # 损失权重
        self.adversarial_weight = 0.1
        self.fm_weight = 10.0  # 特征匹配损失权重
        
    def forward(self, x):
        """
        x: 输入张量, 形状为[B,3,H,W]
           通道0: 灰度图像
           通道1: 已知区域的边缘
           通道2: 掩码 (1表示未知区域)
        """
        # 前向传播仅使用生成器
        edge_pred = self.generator(x)
        return edge_pred
    

    def process(self, images, edges, masks, outputs=None):
        """EdgeConnect风格的处理函数，计算所有损失
        
        Args:
            images: 灰度图像 [B,1,H,W]
            edges: 真实边缘图 [B,1,H,W]
            masks: 掩码 [B,1,H,W] (1表示未知区域)
            outputs: 预生成的边缘图，如果为None，则重新生成
        
        Returns:
            gen_loss: 生成器损失
            dis_loss: 判别器损失
            logs: 日志信息
        """
        # 边缘掩码处理
        edges_masked = edges * (1 - masks)
        images_masked = images * (1 - masks) + masks
        
        # 生成器输入和输出
        inputs = torch.cat((images_masked, edges_masked, masks), dim=1)
        
        # 如果没有提供预生成的输出，则生成
        if outputs is None:
            outputs = self.generator(inputs)
        
        # 仅在训练模式下计算损失
        if self.training:
            # 关闭自动混合精度以避免BCELoss问题
            with torch.cuda.amp.autocast(enabled=False):
                # 确保输入张量为float32
                images_f32 = images.float()
                edges_f32 = edges.float()
                outputs_f32 = outputs.float()
                masks_f32 = masks.float()
                
                # 判别器损失
                dis_input_real = torch.cat((images_f32, edges_f32), dim=1)
                dis_input_fake = torch.cat((images_f32, outputs_f32), dim=1)
                
                # 确保使用detach()避免生成器梯度流向判别器
                dis_real, dis_real_feat = self.discriminator(dis_input_real)
                dis_fake, dis_fake_feat = self.discriminator(dis_input_fake.detach())
                
                dis_real_loss = self.adversarial_loss(dis_real, True, True)
                dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
                dis_loss = (dis_real_loss + dis_fake_loss) / 2.0
                
                # 生成器对抗损失
                gen_input_fake = torch.cat((images_f32, outputs_f32), dim=1)
                gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)
                gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
                
                # 生成器特征匹配损失
                gen_fm_loss = 0.0
                for i in range(len(dis_real_feat)):
                    # 确保使用clone()避免修改原始特征
                    real_feat = dis_real_feat[i].detach().clone()
                    gen_fm_loss = gen_fm_loss + self.l1_loss(gen_fake_feat[i], real_feat)
                gen_fm_loss = gen_fm_loss * self.fm_weight
                
                # 像素级L1损失 (仅在掩码区域)
                gen_l1_loss = self.l1_loss(outputs_f32 * masks_f32, edges_f32 * masks_f32) * 10.0
                
                # 总生成器损失
                gen_loss = gen_gan_loss + gen_fm_loss + gen_l1_loss
                
                # 日志信息
                logs = [
                    ("l_d", dis_loss.item()),
                    ("l_g", gen_gan_loss.item()),
                    ("l_fm", gen_fm_loss.item()),
                    ("l_l1", gen_l1_loss.item()),
                ]
            
            return gen_loss, dis_loss, dis_loss, logs
        else:
            return outputs, None, None, None
    
    def train(self, mode=True):
        """设置训练/评估模式"""
        self.training = mode
        self.generator.train(mode)
        self.discriminator.train(mode)
        return self
    
    def eval(self):
        """设置评估模式"""
        return self.train(False)


# 添加对抗损失函数类
class AdversarialLoss(nn.Module):
    def __init__(self, type='nsgan'):
        super(AdversarialLoss, self).__init__()
        self.type = type
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        
        if type == 'nsgan':
            # 替换 BCELoss 为 BCEWithLogitsLoss，它在混合精度训练中更安全
            self.criterion = nn.BCEWithLogitsLoss()
        elif type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif type == 'hinge':
            self.criterion = nn.ReLU()
    
    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()
        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            return self.criterion(outputs, labels)


# 添加判别器类
class Discriminator(BaseNetwork):
    def __init__(self, in_channels=2, use_sigmoid=False, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )
        
        if init_weights:
            self.init_weights()
    
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        
        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)
            
        return outputs, [conv1, conv2, conv3, conv4, conv5]
    
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out