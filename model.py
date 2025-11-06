import torch, types
from torch import nn
import torch.nn.functional as F
from utils import *
from backprop import RevModule, VanillaBackProp, RevBackProp
from forward import MyUNet2DConditionModel_SD_v1_5_forward, \
                    MyCrossAttnDownBlock2D_SD_v1_5_forward, \
                    MyCrossAttnUpBlock2D_SD_v1_5_forward, \
                    MyResnetBlock2D_SD_v1_5_forward, \
                    MyTransformer2DModel_SD_v1_5_forward
from mamba_ssm import Mamba
from networks import CBAM
y = None
A = None
AT = None
ATy = None
alpha_bar = None
use_amp = None
unet = None
edge_prior = None  # 添加边缘先验全局变量
# 原始
# class Injector(nn.Module):
#     def __init__(self, nf, r, T):
#         super().__init__()
#         self.f2i = nn.ModuleList([nn.Sequential(
#             nn.PixelShuffle(r),
#             nn.Conv2d(nf//(r*r), 1, 1),
#         ) for _ in range(T)])
#         self.i2f = nn.ModuleList([nn.Sequential(
#             nn.Conv2d(4, nf//(r*r), 1),  # 修改输入通道数从3到4，以包含边缘先验
#             nn.PixelUnshuffle(r),
#         ) for _ in range(T)])

#     def forward(self, x_in):
#         x = self.f2i[t-1](x_in)
#         # 添加edge_prior作为额外的通道
#         x = torch.cat([x, AT(A(x)), ATy, edge_prior], dim=1)
#         return x_in + self.i2f[t-1](x)

# 修改后的Injector类，添加了通道注意力机制
class Injector(nn.Module):
    def __init__(self, nf, r, T):
        super().__init__()
        self.f2i = nn.ModuleList([nn.Sequential(
            nn.PixelShuffle(r),
            nn.Conv2d(nf//(r*r), 1, 1),
        ) for _ in range(T)])
        self.i2f = nn.ModuleList([nn.Sequential(
            nn.Conv2d(4, nf//(r*r), 1),  # 修改输入通道数从3到4，以包含边缘先验
            nn.PixelUnshuffle(r),
        ) for _ in range(T)])
        
        # 添加通道注意力机制
        self.channel_attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # 全局平均池化
                nn.Flatten(),
                nn.Linear(4, 8),          # 扩展特征
                nn.ReLU(inplace=True),
                nn.Linear(8, 4),          # 映射回原始通道数
                nn.Sigmoid()              # 输出0-1之间的值作为权重
            ) for _ in range(T)
        ])

    def forward(self, x_in):
        x = self.f2i[t-1](x_in)
        # 添加edge_prior作为额外的通道
        x_cat = torch.cat([x, AT(A(x)), ATy, edge_prior], dim=1)
        
        # 应用通道注意力机制
        b, c, h, w = x_cat.shape
        attention_weights = self.channel_attention[t-1](x_cat).view(b, c, 1, 1)
        x_attended = x_cat * attention_weights  # 对拼接后的特征应用注意力
        
        # 将注意力处理后的结果输入到i2f，并添加残差连接
        return x_in + self.i2f[t-1](x_attended)
# class Injector(nn.Module):
#     def __init__(self, nf, r, T):
#         super().__init__()
#         self.f2i = nn.ModuleList([nn.Sequential(
#             nn.PixelShuffle(r),
#             nn.Conv2d(nf//(r*r), 1, 1),
#         ) for _ in range(T)])
#         self.i2f = nn.ModuleList([nn.Sequential(
#             nn.Conv2d(4, nf//(r*r), 1),  # 修改输入通道数从3到4，以包含边缘先验
#             nn.PixelUnshuffle(r),
#         ) for _ in range(T)])
        
#         # 替换为通道+空间组合注意力
#         self.dual_attention = nn.ModuleList([
#             SpatioChannelAttention(channels=4) for _ in range(T)
#         ])

#     def forward(self, x_in):
#         x = self.f2i[t-1](x_in)
#         # 添加edge_prior作为额外的通道
#         x_cat = torch.cat([x, AT(A(x)), ATy, edge_prior], dim=1)
        
#         # 应用通道+空间注意力
#         x_attended = self.dual_attention[t-1](x_cat)
        
#         # 将注意力处理后的结果输入到i2f，并添加残差连接
#         return x_in + self.i2f[t-1](x_attended)

# # 新增通道+空间联合注意力模块
# class SpatioChannelAttention(nn.Module):
#     def __init__(self, channels, reduction_ratio=8):
#         super().__init__()
        
#         # 通道注意力子模块
#         self.channel_attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False),
#             nn.Sigmoid()
#         )
        
#         # 空间注意力子模块
#         self.spatial_attention = nn.Sequential(
#             # 先分别计算平均池化和最大池化特征，然后拼接
#             nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
        
#     def forward(self, x):
#         # 应用通道注意力
#         channel_weights = self.channel_attention(x)
#         x_channel = x * channel_weights
        
#         # 应用空间注意力
#         # 计算通道维度上的平均值和最大值，并拼接
#         avg_out = torch.mean(x_channel, dim=1, keepdim=True)
#         max_out, _ = torch.max(x_channel, dim=1, keepdim=True)
#         spatial_features = torch.cat([avg_out, max_out], dim=1)
        
#         spatial_weights = self.spatial_attention(spatial_features)
#         x_spatial = x_channel * spatial_weights
        
#         return x_spatial
# class Injector(nn.Module):
#     """简化的动态特征注入器
    
#     保留核心功能，降低内存占用和计算复杂度。
#     """
#     def __init__(self, nf, r, T):
#         super().__init__()
#         # 基础特征提取和投影层
#         self.f2i = nn.ModuleList([nn.Sequential(
#             nn.PixelShuffle(r),
#             nn.Conv2d(nf//(r*r), 1, 1),
#         ) for _ in range(T)])
        
#         # 特征融合和投影回原始空间
#         self.i2f = nn.ModuleList([nn.Sequential(
#             nn.Conv2d(4, nf//(r*r), 1),  # 4个通道: 原始特征, 掩码操作, ATy, 边缘先验
#             nn.PixelUnshuffle(r),
#         ) for _ in range(T)])
        
#         # 减少交叉注意力计算量，使用较小的投影维度和更少的头数
#         self.feature_projection = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(4, 32),  # 从80减少到32
#                 nn.ReLU(inplace=True)
#             ) for _ in range(T)
#         ])
        
#         # 使用更小的交叉注意力
#         self.cross_attention = nn.ModuleList([
#             nn.MultiheadAttention(
#                 embed_dim=32,  # 从80减少到32
#                 num_heads=2,   # 从4减少到2
#                 batch_first=True
#             ) for _ in range(T)
#         ])
        
#         # 轻量级注意力机制
#         self.lightweight_attention = nn.ModuleList([
#             nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Conv2d(4, 4, 1),
#                 nn.Sigmoid()
#             ) for _ in range(T)
#         ])
        
#     def forward(self, x_in):
#         x = self.f2i[t-1](x_in)
        
#         # 获取掩码信息 (从全局变量)
#         mask = A(torch.ones_like(x)).squeeze(1)
        
#         # 拼接特征
#         x_cat = torch.cat([x, AT(A(x)), ATy, edge_prior], dim=1)  # [B, 4, H, W]
#         b, c, h, w = x_cat.shape
        
#         # 1. 轻量级注意力 - 代替多个注意力机制
#         att_weights = self.lightweight_attention[t-1](x_cat)
#         refined_features = x_cat * att_weights
        
#         # 2. 交叉注意力处理 - 仅当有掩码和非掩码区域时应用
#         if torch.sum(mask) > 0 and torch.sum(1-mask) > 0:
#             # 将特征重塑为序列形式
#             x_flat = refined_features.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
#             mask_flat = mask.view(b, -1)  # [B, H*W]
            
#             # 使用简化的交叉注意力逻辑
#             attended_features = []
#             for batch_idx in range(b):
#                 # 限制处理的像素数量，防止OOM
#                 max_pixels = 1024  # 限制处理的最大像素数
                
#                 # 分离已知区域和未知区域的特征
#                 known_indices = (~mask_flat[batch_idx].bool()).nonzero().squeeze(1)
#                 unknown_indices = mask_flat[batch_idx].bool().nonzero().squeeze(1)
                
#                 # 如果像素太多，随机采样
#                 if len(known_indices) > max_pixels:
#                     perm = torch.randperm(len(known_indices))
#                     known_indices = known_indices[perm[:max_pixels]]
                
#                 if len(unknown_indices) > max_pixels:
#                     perm = torch.randperm(len(unknown_indices))
#                     unknown_indices = unknown_indices[perm[:max_pixels]]
                
#                 if len(known_indices) > 0 and len(unknown_indices) > 0:
#                     known_feats = x_flat[batch_idx, known_indices, :]
#                     unknown_feats = x_flat[batch_idx, unknown_indices, :]
                    
#                     # 投影特征到较小维度
#                     known_feats_proj = self.feature_projection[t-1](known_feats)
#                     unknown_feats_proj = self.feature_projection[t-1](unknown_feats)
                    
#                     # 使用交叉注意力
#                     attended_unknown, _ = self.cross_attention[t-1](
#                         unknown_feats_proj, known_feats_proj, known_feats_proj
#                     )
                    
#                     # 简化回原始维度的方式
#                     attended_unknown_orig = attended_unknown[:, :4]
                    
#                     # 将结果放回原处
#                     batch_features = x_flat[batch_idx].clone()
#                     batch_features[unknown_indices] = attended_unknown_orig.to(batch_features.dtype)
#                     attended_features.append(batch_features)
#                 else:
#                     attended_features.append(x_flat[batch_idx])
            
#             # 重塑回原始形状
#             if attended_features:
#                 cross_attention_refined = torch.stack(attended_features, dim=0).permute(0, 2, 1).view(b, c, h, w)
#                 # 混合原始特征和注意力处理后的特征
#                 refined_features = 0.7 * refined_features + 0.3 * cross_attention_refined
        
#         # 将特征投影回原始空间
#         return x_in + self.i2f[t-1](refined_features)
class MultiLevelSpatioChannelAttention(nn.Module):
    """多层次空间-通道交互自适应注意力模块
    
    该模块通过多头交互机制，同时在通道和空间维度上建立长期依赖关系，
    实现了特征的自适应重校准和增强。
    """
    def __init__(self, channels, reduction_ratio=2, num_heads=4, temperature=30):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = temperature
        
        # 通道注意力分支 - 多头设计
        self.channel_groups = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels//reduction_ratio, 1, bias=False),
                nn.BatchNorm2d(channels//reduction_ratio),  # 使用BatchNorm2d代替LayerNorm
                nn.GELU(),
                nn.Conv2d(channels//reduction_ratio, channels, 1, bias=False),
            ) for _ in range(num_heads)
        ])
        
        # 空间注意力分支
        self.spatial_pool = nn.Conv2d(channels, 1, kernel_size=1)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # 交叉维度信息融合
        self.cross_dim_mixing = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),  # 使用BatchNorm2d代替LayerNorm
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        
        # 注意力调节参数
        self.channel_gamma = nn.Parameter(torch.zeros(1))
        self.spatial_gamma = nn.Parameter(torch.zeros(1))
        self.mix_gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        
        # 多头通道注意力
        channel_atts = []
        for head in self.channel_groups:
            att = head(x)
            channel_atts.append(att.sigmoid())
            
        # 融合多头通道注意力
        channel_att = torch.stack(channel_atts, dim=1)
        channel_att = torch.mean(channel_att, dim=1)
        
        # 空间注意力
        spatial_avg = torch.mean(x, dim=1, keepdim=True)
        spatial_max, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_attn(torch.cat([spatial_avg, spatial_max], dim=1))
        
        # 交叉维度混合
        mixed_features = self.cross_dim_mixing(x)
        
        # 综合应用三种注意力机制
        refined_features = x * (1 + self.channel_gamma * channel_att) * \
                           (1 + self.spatial_gamma * spatial_att) + \
                           self.mix_gamma * mixed_features
        
        return refined_features
class Step(RevModule):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def body(self, x):
        with torch.cuda.amp.autocast(enabled=use_amp, cache_enabled=False):
            global t
            t = self.t
            cur_alpha_bar = alpha_bar[t]
            prev_alpha_bar = alpha_bar[t-1]
            e = F.pixel_shuffle(unet(F.pixel_unshuffle(x, 2)), 2) # 0. Noise Estimation (epsilon)
            x = (x - (1 - cur_alpha_bar).pow(0.5) * e) / cur_alpha_bar.pow(0.5) # 1. Denoising
            x = x - AT(A(x) - y) # 2. RND
            return prev_alpha_bar.pow(0.5) * x + (1 - prev_alpha_bar).pow(0.5) * e # 3. DDIM Sampling

class Net(nn.Module):
    def __init__(self, T, unet):
        super().__init__()
        del unet.time_embedding, unet.mid_block
        unet.down_blocks = unet.down_blocks[:-2]
        unet.down_blocks[-1].downsamplers = None
        unet.up_blocks = unet.up_blocks[2:]
        self.body = nn.ModuleList([Step(T-i) for i in range(T)])
        self.input_help_scale_factor = nn.Parameter(torch.tensor([1.0]))
        self.merge_scale_factor = nn.Parameter(torch.tensor([0.0]))
        self.alpha = nn.Parameter(torch.full((T,), 0.5))
        self.unet = unet
        self.unet_add_down_rev_modules_and_injectors(T)
        self.unet_add_up_rev_modules_and_injectors(T)
        self.unet_remove_resnet_time_emb_proj()
        self.unet_remove_cross_attn()
        self.unet_set_inplace_to_true()
        self.unet_replace_forward_methods()

    def unet_add_down_rev_modules_and_injectors(self, T):
        self.unet.down_blocks[0].register_module("injectors", nn.ModuleList([Injector(320, 2, T) for _ in range(4)]))
        self.unet.down_blocks[1].register_module("injectors", nn.ModuleList([Injector(640, 4, T) for _ in range(4)]))
        for i in range(2):
            self.unet.down_blocks[i].register_module("rev_module_lists", nn.ModuleList([]))
            self.unet.down_blocks[i].register_parameter("input_help_scale_factor", nn.Parameter(torch.ones(1,)))
            self.unet.down_blocks[i].register_parameter("merge_scale_factors", nn.Parameter(torch.zeros(2,)))
            for j in range(2):
                rev_module_list = nn.ModuleList([])
                if self.unet.down_blocks[i].resnets[j].in_channels == self.unet.down_blocks[i].resnets[j].out_channels:
                    rev_module_list.append(RevModule(self.unet.down_blocks[i].resnets[j]))
                rev_module_list.append(RevModule(self.unet.down_blocks[i].injectors[2*j]))
                rev_module_list.append(RevModule(self.unet.down_blocks[i].attentions[j]))
                rev_module_list.append(RevModule(self.unet.down_blocks[i].injectors[2*j+1]))
                self.unet.down_blocks[i].rev_module_lists.append(rev_module_list)

    def unet_add_up_rev_modules_and_injectors(self, T):
        self.unet.up_blocks[0].register_module("injectors", nn.ModuleList([Injector(640, 4, T) for _ in range(6)]))
        self.unet.up_blocks[1].register_module("injectors", nn.ModuleList([Injector(320, 2, T) for _ in range(6)]))
        for i in range(2):
            self.unet.up_blocks[i].register_parameter("input_help_scale_factor", nn.Parameter(torch.ones(1,)))
            self.unet.up_blocks[i].register_parameter("merge_scale_factor", nn.Parameter(torch.zeros(1,)))
            rev_module_list = nn.ModuleList([])
            for j in range(3):
                if j > 0:
                    rev_module_list.append(RevModule(self.unet.up_blocks[i].resnets[j]))
                rev_module_list.append(RevModule(self.unet.up_blocks[i].injectors[2*j]))
                rev_module_list.append(RevModule(self.unet.up_blocks[i].attentions[j]))
                rev_module_list.append(RevModule(self.unet.up_blocks[i].injectors[2*j+1]))
            self.unet.up_blocks[i].register_module("rev_module_list", rev_module_list)

    def unet_replace_forward_methods(self):
        from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D
        from diffusers.models.unets.unet_2d_blocks import CrossAttnUpBlock2D
        from diffusers.models.resnet import ResnetBlock2D
        from diffusers.models.transformers.transformer_2d import Transformer2DModel
        def replace_forward_methods(module):
            if isinstance(module, CrossAttnDownBlock2D):
                module.forward = types.MethodType(MyCrossAttnDownBlock2D_SD_v1_5_forward, module)
            elif isinstance(module, CrossAttnUpBlock2D):
                module.forward = types.MethodType(MyCrossAttnUpBlock2D_SD_v1_5_forward, module)
            elif isinstance(module, ResnetBlock2D):
                module.forward = types.MethodType(MyResnetBlock2D_SD_v1_5_forward, module)
            elif isinstance(module, Transformer2DModel):
                module.forward = types.MethodType(MyTransformer2DModel_SD_v1_5_forward, module)
        self.unet.apply(replace_forward_methods)
        self.unet.forward = types.MethodType(MyUNet2DConditionModel_SD_v1_5_forward, self.unet)

    def unet_remove_resnet_time_emb_proj(self):
        from diffusers.models.resnet import ResnetBlock2D
        def ResnetBlock2D_remove_time_emb_proj(module):
            if isinstance(module, ResnetBlock2D):
                module.time_emb_proj = None
        self.unet.apply(ResnetBlock2D_remove_time_emb_proj)

    def unet_remove_cross_attn(self):
        from diffusers.models.attention import BasicTransformerBlock
        def BasicTransformerBlock_remove_cross_attn(module):
            if isinstance(module, BasicTransformerBlock):
                module.attn2 = module.norm2 = None
        self.unet.apply(BasicTransformerBlock_remove_cross_attn)
    
    def unet_set_inplace_to_true(self):
        def set_inplace_to_true(module):
            if isinstance(module, nn.Dropout) or isinstance(module, nn.SiLU):
                module.inplace = True
        self.unet.apply(set_inplace_to_true)

    # 在模型的forward方法中，修改以下内容（只显示修改的部分）:

    # def forward(self, y_, A_, AT_, edge_prior_=None, use_amp_=True):
    #     global y, A, AT, unet, ATy, alpha_bar, use_amp, edge_prior
    #     y, A, AT, unet, use_amp = y_, A_, AT_, self.unet, use_amp_
    #     edge_prior = edge_prior_  # 设置全局边缘先验变量
    #     alpha_bar = torch.cat([torch.ones(1, device=y.device), self.alpha.cumprod(dim=0)])
    #     # 对于inpainting，y就是带掩码的图像，直接用于初始化x
    #     x = y
    #     ATy = x  # 对于inpainting，ATy就是y
    #     x = alpha_bar[-1].pow(0.5) * torch.cat([x, self.input_help_scale_factor * x], dim=1)
    #     x = RevBackProp.apply(x, self.body)
    #     return x[:, :1] + self.merge_scale_factor * x[:, 1:]
    
    # def forward(self, y_, A_, AT_, edge_prior_=None, use_amp_=True, use_steps=None):
    #     global y, A, AT, unet, ATy, alpha_bar, use_amp, edge_prior
    #     y, A, AT, unet, use_amp = y_, A_, AT_, self.unet, use_amp_
    #     edge_prior = edge_prior_  # 设置全局边缘先验变量
    #     alpha_bar = torch.cat([torch.ones(1, device=y.device), self.alpha.cumprod(dim=0)])
    #     # 对于inpainting，y就是带掩码的图像，直接用于初始化x
    #     x = y
    #     ATy = x  # 对于inpainting，ATy就是y
    #     x = alpha_bar[-1].pow(0.5) * torch.cat([x, self.input_help_scale_factor * x], dim=1)
        
    #     # 新增：根据use_steps参数决定使用多少步骤
    #     if use_steps is not None and 1 <= use_steps < len(self.body):
    #         # 使用指定数量的步骤
    #         print(f"使用 {use_steps}/{len(self.body)} 步进行推理")
    #         # 从高噪声到低噪声的前use_steps个步骤
    #         x = RevBackProp.apply(x, nn.ModuleList(self.body[:use_steps]))
    #     else:
    #         # 使用所有步骤
    #         print(f"使用全部 {len(self.body)} 步进行推理")
    #         x = RevBackProp.apply(x, self.body)
        
    #     return x[:, :1] + self.merge_scale_factor * x[:, 1:]
    # def forward(self, y_, A_, AT_, edge_prior_=None, use_amp_=True, use_steps=None):
    #     global y, A, AT, unet, ATy, alpha_bar, use_amp, edge_prior
    #     y, A, AT, unet, use_amp = y_, A_, AT_, self.unet, use_amp_
    #     edge_prior = edge_prior_  # 设置全局边缘先验变量
        
    #     # 计算完整的alpha_bar
    #     alpha_bar = torch.cat([torch.ones(1, device=y.device), self.alpha.cumprod(dim=0)])
        
    #     # 对于inpainting，y就是带掩码的图像，直接用于初始化x
    #     x = y
    #     ATy = x  # 对于inpainting，ATy就是y
    #     x = alpha_bar[-1].pow(0.5) * torch.cat([x, self.input_help_scale_factor * x], dim=1)
        
    #     # 根据use_steps参数决定使用多少步骤
    #     total_steps = len(self.body)
    #     if use_steps is not None and 1 <= use_steps < total_steps:
    #         # 正确实现步骤跳跃采样
    #         # 均匀选择时间步骤
    #         step_indices = torch.linspace(0, total_steps-1, use_steps).round().long()
            
    #         # 创建一个新的步骤序列
    #         selected_steps = []
    #         for idx in step_indices:
    #             selected_steps.append(self.body[idx])
            
    #         # 使用选择的步骤构建新的模块列表
    #         custom_body = nn.ModuleList(selected_steps)
    #         print(f"使用 {use_steps}/{total_steps} 步进行扩散推理")
    #         print(f"选择的步骤索引: {step_indices.tolist()}")
            
    #         # 使用自定义步骤进行推理
    #         x = RevBackProp.apply(x, custom_body)
    #     else:
    #         # 使用所有步骤
    #         print(f"使用全部 {total_steps} 步进行推理")
    #         x = RevBackProp.apply(x, self.body)
        
    #     return x[:, :1] + self.merge_scale_factor * x[:, 1:]

    def forward(self, y_, A_, AT_, edge_prior_=None, use_amp_=True, use_steps=None):
        global y, A, AT, unet, ATy, alpha_bar, use_amp, edge_prior
        y, A, AT, unet, use_amp = y_, A_, AT_, self.unet, use_amp_
        edge_prior = edge_prior_  # 设置全局边缘先验变量
        
        # 计算完整的alpha_bar
        alpha_bar = torch.cat([torch.ones(1, device=y.device), self.alpha.cumprod(dim=0)])
        
        # 对于inpainting，y就是带掩码的图像，直接用于初始化x
        x = y
        ATy = x  # 对于inpainting，ATy就是y
        x = alpha_bar[-1].pow(0.5) * torch.cat([x, self.input_help_scale_factor * x], dim=1)
        
        # 根据use_steps参数决定使用多少步骤
        total_steps = len(self.body)
        if use_steps is not None and 1 <= use_steps < total_steps:
            # 倒置索引选择：从高到低（从末尾到开始）
            step_indices = torch.linspace(total_steps-1, 0, use_steps).round().long()
            
            # 创建一个新的步骤序列
            selected_steps = []
            for idx in step_indices:
                selected_steps.append(self.body[idx])
            
            # 使用选择的步骤构建新的模块列表
            custom_body = nn.ModuleList(selected_steps)
            print(f"使用 {use_steps}/{total_steps} 步进行扩散推理（倒序选择）")
            print(f"选择的步骤索引: {step_indices.tolist()}")
            
            # 使用自定义步骤进行推理
            x = RevBackProp.apply(x, custom_body)
        else:
            # 使用所有步骤
            print(f"使用全部 {total_steps} 步进行推理")
            x = RevBackProp.apply(x, self.body)
        
        return x[:, :1] + self.merge_scale_factor * x[:, 1:]