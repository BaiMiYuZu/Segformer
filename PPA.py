import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def build_norm(norm_type, num_features):
    norm_type = norm_type.lower()
    if norm_type == 'bn':
        return nn.BatchNorm2d(num_features)
    if norm_type == 'syncbn':
        return nn.SyncBatchNorm(num_features)
    if norm_type in ('none', 'identity'):
        return nn.Identity()
    raise ValueError(f'Unsupported norm_type: {norm_type}')


# ==========================================
# 1. 基础卷积块 (conv_block)
# ==========================================
class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_features,
            out_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.norm = build_norm(norm_type, out_features)

        if activation:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

# ==========================================
# 2. 高效通道注意力 (ECA)
# ==========================================
class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        # 动态计算卷积核大小
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        # 降维、一维卷积、升维
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# ==========================================
# 3. 空间注意力模块 (SpatialAttentionModule)
# ==========================================
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x

# ==========================================
# 4. 局部/全局 Patch 感知注意力 (LocalGlobalAttention)
# ==========================================
class LocalGlobalAttention(nn.Module):
    def __init__(self, channels, patch_size):
        super(LocalGlobalAttention, self).__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

        dim = patch_size * patch_size
        # FFN 用于计算空间维度的概率分布
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.softmax = nn.Softmax(dim=-1)

        # 任务相关的特征选择 (Task Embedding)
        self.task_embedding = nn.Parameter(torch.randn(1, channels, 1))
        self.channel_selection = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        # 为了防止 H 或 W 不能被 patch_size 整除进行 pad
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            _, _, H_pad, W_pad = x.shape
        else:
            H_pad, W_pad = H, W

        # 分块 (Unfold) -> (B, C * p^2, L)
        patches = self.unfold(x)
        L = patches.shape[-1] # L = (H_pad // p) * (W_pad // p)
        patches = patches.view(B, C, p * p, L) # (B, C, p^2, L)

        # 1. 通道维度平均求空间概率分布
        mean_patches = patches.mean(dim=1).transpose(1, 2) # (B, L, p^2)
        attn = self.ffn(mean_patches) # (B, L, p^2)
        attn = self.softmax(attn).transpose(1, 2).unsqueeze(1) # (B, 1, p^2, L)

        # 加权特征
        weighted_patches = patches * attn # (B, C, p^2, L)

        # 2. 特征选择 (Feature Selection) - Token & Channel Selection
        # 聚合每个 Patch 的特征
        tokens = weighted_patches.sum(dim=2) # (B, C, L)

        # 计算与 Task Embedding 的余弦相似度
        norm_tokens = F.normalize(tokens, p=2, dim=1)
        norm_task = F.normalize(self.task_embedding, p=2, dim=1)
        sim = torch.bmm(norm_task.transpose(1, 2).expand(B, -1, -1), norm_tokens) # (B, 1, L)

        # 相似度加权 (Token Selection)
        selected_tokens = tokens * sim # (B, C, L)

        # 通道选择 (Channel Selection)
        out_features = self.channel_selection(selected_tokens) # (B, C, L)

        # 3. 还原回 2D 空间维度并插值回原图大小
        out = out_features.view(B, C, H_pad // p, W_pad // p)
        if pad_h > 0 or pad_w > 0:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        else:
            # 如果没有 pad，可以直接放大到原分辨率（论文中的重构逻辑）
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        return out

# ==========================================
# 5. 主模块：并行块感知注意力 (PPA)
# ==========================================
class PPA(nn.Module):
    def __init__(self,
                 in_features,
                 filters,
                 norm_type='bn',
                 dropout=0.1) -> None:
        super().__init__()

        # PW Conv 用于对齐输入特征
        self.skip = conv_block(
            in_features=in_features,
            out_features=filters,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type,
            activation=False)

        # 串行卷积分支 (Serial convolution branch)
        self.c1 = conv_block(
            in_features=filters,
            out_features=filters,
            kernel_size=(3, 3),
            padding=(1, 1),
            norm_type=norm_type,
            activation=True)
        self.c2 = conv_block(
            in_features=filters,
            out_features=filters,
            kernel_size=(3, 3),
            padding=(1, 1),
            norm_type=norm_type,
            activation=True)
        self.c3 = conv_block(
            in_features=filters,
            out_features=filters,
            kernel_size=(3, 3),
            padding=(1, 1),
            norm_type=norm_type,
            activation=True)

        # 局部与全局分支 (Patch-Aware)
        self.lga2 = LocalGlobalAttention(filters, 2)
        self.lga4 = LocalGlobalAttention(filters, 4)

        # 注意力融合层
        self.cn = ECA(filters)
        self.sa = SpatialAttentionModule()

        # 尾部处理
        self.bn1 = build_norm(norm_type, filters)
        self.drop = nn.Dropout2d(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 统一经过 Skip (PW Conv) 进行维度变换/对齐
        x_skip = self.skip(x)

        # 分支 1: 局部感知 (p=2)
        x_lga2 = self.lga2(x_skip)

        # 分支 2: 全局感知 (p=4)
        x_lga4 = self.lga4(x_skip)

        # 分支 3: 串行卷积 (修复了之前 c1 接收 x 而不是 x_skip 的潜在问题)
        x1 = self.c1(x_skip)
        x2 = self.c2(x1)
        x3 = self.c3(x2)

        # 多分支融合
        x_fused = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4

        # 注意力增强与收尾
        x_out = self.cn(x_fused)
        x_out = self.sa(x_out)
        x_out = self.drop(x_out)
        x_out = self.bn1(x_out)
        x_out = self.relu(x_out)

        return x_out
