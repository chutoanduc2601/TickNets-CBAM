"""
Script xuất hình ảnh kiến trúc của 3 mô hình TickNet-small:
  1. SE Baseline
  2. CBAM-Local 
  3. CBAM-Hook (Đề xuất)

Dùng torchinfo.summary() để xuất bảng tóm tắt kiến trúc dạng text,
và matplotlib để vẽ sơ đồ khối kiến trúc trực quan cho luận văn.

Chạy trên Kaggle/Colab hoặc máy local (cần PyTorch).
Output: Lưu vào thư mục OUTPUT_DIR dưới dạng file PNG.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ============================================================================
# OUTPUT DIRECTORY
# ============================================================================
# Tren Kaggle: OUTPUT_DIR = '/kaggle/working/architecture_diagrams'
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'architecture_diagrams')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# KIEN TRUC MO HINH (COPY TU NOTEBOOK - PHIEN BAN CHUAN LUAN VAN)
# ============================================================================

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class HSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=True) / 6.0

class HSigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3.0, inplace=True) / 6.0

def get_activation(activation):
    if activation == "relu":
        return nn.ReLU(inplace=True)
    elif activation == "relu6":
        return nn.ReLU6(inplace=True)
    elif activation == "swish":
        return Swish()
    elif activation == "hswish":
        return HSwish()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "hsigmoid":
        return HSigmoid()
    else:
        raise NotImplementedError(f"Activation {activation} not implemented")

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False, use_bn=True, activation="relu"):
        super().__init__()
        self.use_bn = use_bn
        self.use_activation = (activation is not None)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels)
        if self.use_activation:
            self.activation = get_activation(activation)
    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_activation:
            x = self.activation(x)
        return x

def conv1x1_block(in_channels, out_channels, stride=1, groups=1, bias=False, use_bn=True, activation="relu"):
    return ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, groups=groups, bias=bias, use_bn=use_bn, activation=activation)

def conv3x3_block(in_channels, out_channels, stride=1, bias=False, use_bn=True, activation="relu"):
    return ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=bias, use_bn=use_bn, activation=activation)

def conv3x3_dw_blockAll(channels, stride=1, use_bn=True, activation="relu", padding=1, dilation=1):
    return ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, stride=stride, padding=padding, groups=channels, dilation=dilation, use_bn=use_bn, activation=activation)

class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=1, bias=True)
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return x
    def init_params(self):
        nn.init.xavier_normal_(self.conv.weight, gain=1.0)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
    def forward(self, x):
        squeeze_avg = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att = self.mlp(squeeze_avg)
        scale = torch.sigmoid(channel_att).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class SE(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(SE, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
    def forward(self, x):
        return self.ChannelGate(x)

class CBAMChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CBAMChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, max(in_planes // ratio, 8), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_planes // ratio, 8), in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class CBAMSpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(CBAMSpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_in = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_in)
        return self.sigmoid(x_out)

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, kernel_size=3, use_residual=False):
        super(CBAM, self).__init__()
        self.use_residual = use_residual
        self.ChannelAttention = CBAMChannelAttention(gate_channels, reduction_ratio)
        self.SpatialAttention = CBAMSpatialAttention(kernel_size=kernel_size)
    def forward(self, x):
        x_att = x * self.ChannelAttention(x)
        x_att = x_att * self.SpatialAttention(x_att)
        if self.use_residual:
            return x + x_att
        return x_att

class FR_PDP_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, attention_type='se', reduction=16, spatial_kernel=3):
        super().__init__()
        self.Pw1 = conv1x1_block(in_channels=in_channels, out_channels=in_channels, use_bn=False, activation=None)
        self.Dw = conv3x3_dw_blockAll(channels=in_channels, stride=stride)
        self.Pw2 = conv1x1_block(in_channels=in_channels, out_channels=out_channels, groups=1)
        self.PwR = conv1x1_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attention_type = attention_type
        if attention_type == 'se':
            self.attention = SE(out_channels, reduction)
        elif attention_type == 'cbam':
            self.attention = CBAM(out_channels, reduction, kernel_size=spatial_kernel)
        else:
            self.attention = nn.Identity()
    def forward(self, x):
        residual = x
        x = self.Pw1(x)
        x = self.Dw(x)
        x = self.Pw2(x)
        x = self.attention(x)
        if self.stride == 1 and self.in_channels == self.out_channels:
            x = x + residual
        else:
            residual = self.PwR(residual)
            x = x + residual
        return x

# PHIEN BAN CHUAN TU NOTEBOOK (co cifar check cho Pre-GAP)
class TickNetSmall(nn.Module):
    def __init__(self, num_classes, attention_mode='se_only', cbam_reduction=8, cbam_spatial_kernel=3, cifar=True):
        super().__init__()
        init_conv_channels = 32
        backbone1_channels = [[128], [64, 128], [256, 512, 128]]
        backbone2_channels = [[64, 128, 256], [512]]
        if cifar:
            self.in_size = (32, 32)
            init_conv_stride = 1
            strides_b1 = [1, 1, 2]
            strides_b2 = [2, 2]
        else:
            self.in_size = (224, 224)
            init_conv_stride = 2
            strides_b1 = [2, 1, 2]
            strides_b2 = [2, 2]
        self.attention_mode = attention_mode
        self.cifar = cifar
        self.data_bn = nn.BatchNorm2d(num_features=3)
        self.init_conv = conv3x3_block(in_channels=3, out_channels=init_conv_channels, stride=init_conv_stride)
        hook_kernel = cbam_spatial_kernel
        self.backbone1 = nn.Sequential()
        in_ch = init_conv_channels
        unit_idx = 0
        for stage_id, stage_channels in enumerate(backbone1_channels):
            stage = nn.Sequential()
            for u_id, unit_channels in enumerate(stage_channels):
                stride = strides_b1[stage_id] if u_id == 0 else 1
                blk_att = 'cbam' if attention_mode == 'cbam_local' else 'se'
                block = FR_PDP_block(in_ch, unit_channels, stride, attention_type=blk_att,
                                     reduction=16, spatial_kernel=cbam_spatial_kernel)
                stage.add_module(f"unit{u_id + 1}", block)
                in_ch = unit_channels
                unit_idx += 1
                # HOOKING POINT 1: Sau khoi FR-PDP thu 5 (512 kenh)
                if unit_idx == 5 and attention_mode == 'cbam_hook':
                    stage.add_module("cbam_junction", CBAM(gate_channels=512, reduction_ratio=cbam_reduction, kernel_size=hook_kernel, use_residual=True))
            self.backbone1.add_module(f"stage{stage_id + 1}", stage)
        self.backbone2 = nn.Sequential()
        unit_idx_b2 = 0
        for stage_id, stage_channels in enumerate(backbone2_channels):
            stage = nn.Sequential()
            for u_id, unit_channels in enumerate(stage_channels):
                stride = strides_b2[stage_id] if u_id == 0 else 1
                blk_att = 'cbam' if attention_mode == 'cbam_local' else 'se'
                block = FR_PDP_block(in_ch, unit_channels, stride, attention_type=blk_att,
                                     reduction=16, spatial_kernel=cbam_spatial_kernel)
                stage.add_module(f"unit{u_id + 1}", block)
                in_ch = unit_channels
                unit_idx_b2 += 1
                # HOOKING POINT 2: Sau khoi FR-PDP thu 10 (512 kenh)
                if unit_idx_b2 == 4 and attention_mode == 'cbam_hook':
                    stage.add_module("cbam_tail", CBAM(gate_channels=512, reduction_ratio=cbam_reduction, kernel_size=hook_kernel, use_residual=True))
            self.backbone2.add_module(f"stage{stage_id + 4}", stage)
        self.final_conv_channels = 1024
        self.final_conv = conv1x1_block(in_channels=in_ch, out_channels=self.final_conv_channels, activation="relu")
        # Pre-GAP chi cho anh lon (theo notebook)
        if attention_mode == 'cbam_hook' and not cifar:
            self.cbam_pre_gap = CBAM(gate_channels=1024, reduction_ratio=cbam_reduction, kernel_size=hook_kernel, use_residual=True)
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = Classifier(in_channels=self.final_conv_channels, num_classes=num_classes)
        self.init_params()

    def init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.classifier.init_params()

    def forward(self, x):
        x = self.data_bn(x)
        x = self.init_conv(x)
        x = self.backbone1(x)
        x = self.backbone2(x)
        x = self.final_conv(x)
        if self.attention_mode == 'cbam_hook' and not self.cifar:
            x = self.cbam_pre_gap(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# VE SO DO KIEN TRUC TRUC QUAN (MATPLOTLIB)
# ============================================================================

def draw_block(ax, x, y, w, h, label, color, text_color='white', fontsize=7,
               alpha=0.95, edgecolor='#333333', linewidth=1.0):
    """Ve mot khoi hinh chu nhat bo goc voi nhan ben trong."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.03",
                         facecolor=color, edgecolor=edgecolor,
                         linewidth=linewidth, alpha=alpha, zorder=2)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold', zorder=3,
            fontfamily='monospace')

def draw_arrow(ax, x1, y1, x2, y2, color='#555555', lw=1.2):
    """Ve mui ten noi giua 2 khoi."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw),
                zorder=1)


def draw_single_model(ax, model_name, attention_mode, title_color):
    """
    Ve so do kien truc cho 1 mo hinh TickNet-small.
    Chinh xac theo notebook: 10 khoi FR-PDP (6 backbone1 + 4 backbone2).
    Chuoi kenh: 128->64->128->256->512->128 | 64->128->256->512
    """
    # Kenh backbone1: [128], [64, 128], [256, 512, 128] = 6 khoi
    # Kenh backbone2: [64, 128, 256], [512] = 4 khoi
    # Tong: 10 khoi FR-PDP
    
    channels_all = [128, 64, 128, 256, 512, 128, 64, 128, 256, 512]
    # Backbone1: indices 0-5 (6 khoi), Backbone2: indices 6-9 (4 khoi)
    
    # Mau sac
    COL_INPUT   = '#78909C'
    COL_STEM    = '#5C6BC0'
    COL_FRPDP_SE   = '#1565C0'    # FR-PDP voi SE
    COL_FRPDP_CBAM = '#E65100'    # FR-PDP voi CBAM Local
    COL_CBAM_HOOK  = '#C62828'    # CBAM Hook point (do dam)
    COL_FINAL   = '#6A1B9A'
    COL_GAP     = '#2E7D32'
    COL_CLASS   = '#37474F'
    
    BW = 0.72
    BH = 0.026
    BH_SMALL = 0.020
    cx = 0.5
    y = 0.96
    gap = 0.036
    gap_small = 0.030
    
    # Mau FR-PDP tuy attention_mode
    if attention_mode == 'cbam_local':
        frpdp_color = COL_FRPDP_CBAM
        att_label = "CBAM"
    else:
        frpdp_color = COL_FRPDP_SE
        att_label = "SE"
    
    # --- Tieu de ---
    ax.text(cx, 1.02, model_name, ha='center', va='bottom', fontsize=9.5,
            fontweight='bold', color=title_color)
    
    # --- Input ---
    draw_block(ax, cx, y, BW*0.55, BH_SMALL, 'Input  3xHxW', COL_INPUT, fontsize=6)
    y_prev = y - BH_SMALL/2
    y -= gap
    
    # --- Stem Conv ---
    draw_arrow(ax, cx, y_prev, cx, y + BH/2)
    draw_block(ax, cx, y, BW*0.7, BH, 'Stem Conv3x3 -> 32ch', COL_STEM, fontsize=6)
    y_prev = y - BH/2
    y -= gap
    
    # --- Backbone 1: 6 khoi FR-PDP (index 1-6) ---
    # Label ngoac cho backbone1
    bb1_y_top = y
    for i in range(6):
        ch = channels_all[i]
        draw_arrow(ax, cx, y_prev, cx, y + BH/2)
        label = f'FR-PDP-{i+1}  ({att_label}, r=16)\n{ch}ch'
        draw_block(ax, cx, y, BW, BH*1.05, label, frpdp_color, fontsize=5.3)
        y_prev = y - BH*1.05/2
        
        # HOOKING POINT 1: Sau khoi FR-PDP thu 5 (512ch)
        if i == 4 and attention_mode == 'cbam_hook':
            y -= gap_small
            draw_arrow(ax, cx, y_prev, cx, y + BH/2)
            draw_block(ax, cx, y, BW*0.82, BH*1.1,
                       '* CBAM Hook-1\nCAM+SAM (r=8, residual)',
                       COL_CBAM_HOOK, fontsize=5.3)
            y_prev = y - BH*1.1/2
        
        y -= gap_small
    
    # --- Backbone 2: 4 khoi FR-PDP (index 7-10) ---
    for i in range(4):
        ch = channels_all[6 + i]
        draw_arrow(ax, cx, y_prev, cx, y + BH/2)
        label = f'FR-PDP-{i+7}  ({att_label}, r=16)\n{ch}ch'
        draw_block(ax, cx, y, BW, BH*1.05, label, frpdp_color, fontsize=5.3)
        y_prev = y - BH*1.05/2
        
        # HOOKING POINT 2: Sau khoi FR-PDP thu 10 (512ch, index 9 = i==3)
        if i == 3 and attention_mode == 'cbam_hook':
            y -= gap_small
            draw_arrow(ax, cx, y_prev, cx, y + BH/2)
            draw_block(ax, cx, y, BW*0.82, BH*1.1,
                       '* CBAM Hook-2\nCAM+SAM (r=8, residual)',
                       COL_CBAM_HOOK, fontsize=5.3)
            y_prev = y - BH*1.1/2
        
        y -= gap_small
    
    # --- Final Conv 1x1 ---
    draw_arrow(ax, cx, y_prev, cx, y + BH/2)
    draw_block(ax, cx, y, BW*0.75, BH, 'Final Conv1x1 -> 1024ch', COL_FINAL, fontsize=6)
    y_prev = y - BH/2
    y -= gap
    
    # --- GAP ---
    draw_arrow(ax, cx, y_prev, cx, y + BH_SMALL/2)
    draw_block(ax, cx, y, BW*0.6, BH_SMALL, 'Global Avg Pool', COL_GAP, fontsize=6)
    y_prev = y - BH_SMALL/2
    y -= gap
    
    # --- Classifier ---
    draw_arrow(ax, cx, y_prev, cx, y + BH_SMALL/2)
    draw_block(ax, cx, y, BW*0.55, BH_SMALL, 'Classifier (FC)', COL_CLASS, fontsize=6)
    
    # --- Thong tin tham so ---
    model = TickNetSmall(num_classes=10, attention_mode=attention_mode,
                         cbam_reduction=8, cbam_spatial_kernel=3, cifar=True)
    params = count_parameters(model)
    ax.text(cx, y - 0.04, f'Params: {params:,}', ha='center', va='top',
            fontsize=7, color='#333', fontfamily='monospace')
    del model
    
    # --- Cau hinh truc ---
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.06, 1.08)
    ax.set_aspect('auto')
    ax.axis('off')


def plot_architecture_comparison():
    """Ve so do so sanh kien truc 3 mo hinh canh nhau."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 20))
    fig.patch.set_facecolor('#FAFAFA')
    
    fig.suptitle('So sanh kien truc 3 mo hinh TickNet-small\n'
                 'De tai: Nghien cuu cai tien TickNets bang co che attention CBAM',
                 fontsize=14, fontweight='bold', y=0.98, color='#1a1a1a')
    
    models = [
        ('Mo hinh 1\nSE Baseline', 'se_only', '#1565C0'),
        ('Mo hinh 2\nCBAM-Local', 'cbam_local', '#E65100'),
        ('Mo hinh 3\nCBAM-Hook (De xuat)', 'cbam_hook', '#C62828'),
    ]
    
    for ax, (name, mode, color) in zip(axes, models):
        ax.set_facecolor('#FAFAFA')
        draw_single_model(ax, name, mode, color)
    
    # --- Chu thich (Legend) ---
    legend_elements = [
        mpatches.Patch(facecolor='#78909C', edgecolor='#333', label='Input'),
        mpatches.Patch(facecolor='#5C6BC0', edgecolor='#333', label='Stem Conv'),
        mpatches.Patch(facecolor='#1565C0', edgecolor='#333', label='FR-PDP + SE'),
        mpatches.Patch(facecolor='#E65100', edgecolor='#333', label='FR-PDP + CBAM (Local)'),
        mpatches.Patch(facecolor='#C62828', edgecolor='#333', label='* CBAM Hooking Point'),
        mpatches.Patch(facecolor='#6A1B9A', edgecolor='#333', label='Final Conv 1x1'),
        mpatches.Patch(facecolor='#2E7D32', edgecolor='#333', label='Global Avg Pool'),
        mpatches.Patch(facecolor='#37474F', edgecolor='#333', label='Classifier'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=9, frameon=True, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, 0.005))
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    
    out_path = os.path.join(OUTPUT_DIR, 'architecture_comparison_3_models.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print(f"[OK] Saved: {out_path}")
    return out_path


def plot_single_model_detailed(model_name, attention_mode, filename):
    """Ve so do kien truc chi tiet cho 1 mo hinh rieng le."""
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 20))
    fig.patch.set_facecolor('white')
    
    color_map = {
        'se_only': '#1565C0',
        'cbam_local': '#E65100',
        'cbam_hook': '#C62828',
    }
    
    draw_single_model(ax, model_name, attention_mode, color_map[attention_mode])
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Saved: {out_path}")
    return out_path


def plot_architecture_table():
    """Tao bang so sanh kien truc 3 mo hinh duoi dang hinh anh."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    col_labels = ['Thanh phan', 'SE Baseline', 'CBAM-Local', 'CBAM-Hook (De xuat)']
    
    model_se = TickNetSmall(10, 'se_only', 8, 3, cifar=True)
    model_local = TickNetSmall(10, 'cbam_local', 8, 3, cifar=True)
    model_hook = TickNetSmall(10, 'cbam_hook', 8, 3, cifar=True)
    
    table_data = [
        ['Stem Conv', '3x3, s=1, 32ch', '3x3, s=1, 32ch', '3x3, s=1, 32ch'],
        ['Backbone 1\n(6 khoi FR-PDP)',
         '128->64->128\n->256->512->128\n(SE, r=16)',
         '128->64->128\n->256->512->128\n(CBAM, r=16)',
         '128->64->128\n->256->512->128\n(SE, r=16)'],
        ['Hooking Point 1\n(sau FR-PDP #5, 512ch)', '---', '---', '* CBAM\nCAM+SAM, r=8\n+ Residual'],
        ['Backbone 2\n(4 khoi FR-PDP)',
         '64->128->256\n->512\n(SE, r=16)',
         '64->128->256\n->512\n(CBAM, r=16)',
         '64->128->256\n->512\n(SE, r=16)'],
        ['Hooking Point 2\n(sau FR-PDP #10, 512ch)', '---', '---', '* CBAM\nCAM+SAM, r=8\n+ Residual'],
        ['Final Conv 1x1', '512->1024', '512->1024', '512->1024'],
        ['Global Avg Pool', '1024->1x1', '1024->1x1', '1024->1x1'],
        ['Classifier', 'FC -> num_classes', 'FC -> num_classes', 'FC -> num_classes'],
        ['Tong tham so\n(CIFAR-10)', f'{count_parameters(model_se):,}',
         f'{count_parameters(model_local):,}', f'{count_parameters(model_hook):,}'],
    ]
    
    del model_se, model_local, model_hook
    
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    
    # Header colors
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#1565C0')
        cell.set_text_props(color='white', fontweight='bold', fontsize=10)
    
    # Row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            if i % 2 == 0:
                cell.set_facecolor('#E3F2FD')
            else:
                cell.set_facecolor('#FFFFFF')
            if j == 3 and '*' in str(table_data[i-1][j]):
                cell.set_facecolor('#FFEBEE')
                cell.set_text_props(color='#C62828', fontweight='bold')
    
    for i in range(1, len(table_data) + 1):
        table[i, 0].set_text_props(fontweight='bold', fontsize=8)
    
    ax.set_title('Bang so sanh kien truc chi tiet 3 mo hinh TickNet-small',
                 fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'architecture_table.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Saved: {out_path}")
    return out_path


# ============================================================================
# XUAT TORCHINFO SUMMARY
# ============================================================================

def export_model_summary():
    """Xuat bang tom tat kien truc 3 mo hinh bang torchinfo."""
    try:
        from torchinfo import summary as torchinfo_summary
        has_torchinfo = True
    except ImportError:
        has_torchinfo = False
        print("[INFO] torchinfo not installed. Run: pip install torchinfo")

    configs = [
        ('SE Baseline (se_only)', 'se_only'),
        ('CBAM-Local (cbam_local)', 'cbam_local'),
        ('CBAM-Hook (cbam_hook)', 'cbam_hook'),
    ]

    for name, mode in configs:
        model = TickNetSmall(num_classes=10, attention_mode=mode, cbam_reduction=8, cbam_spatial_kernel=3, cifar=True)
        params = count_parameters(model)
        print("=" * 80)
        print(f"  MODEL: {name}  |  Params: {params:,}")
        print("=" * 80)
        if has_torchinfo:
            torchinfo_summary(model, input_size=(1, 3, 32, 32),
                            col_names=["input_size", "output_size", "num_params"],
                            depth=3, verbose=0)
        else:
            print(model)
        print("\n")
        del model


def export_torchviz_graphs():
    """Xuat computation graph bang torchviz (can graphviz)."""
    try:
        from torchviz import make_dot
    except ImportError:
        print("[INFO] torchviz not installed. Run: pip install torchviz")
        return
    try:
        import graphviz
    except ImportError:
        print("[INFO] graphviz not installed. Run: pip install graphviz + system graphviz")
        return
    
    configs = [
        ('se_baseline', 'se_only', 'SE Baseline'),
        ('cbam_local', 'cbam_local', 'CBAM-Local'),
        ('cbam_hook', 'cbam_hook', 'CBAM-Hook'),
    ]
    for fname, mode, title in configs:
        model = TickNetSmall(num_classes=10, attention_mode=mode,
                             cbam_reduction=8, cbam_spatial_kernel=3, cifar=True)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        y = model(x)
        dot = make_dot(y, params=dict(model.named_parameters()),
                       show_attrs=False, show_saved=False)
        dot.attr(label=title, labelloc='t', fontsize='16')
        out_path = os.path.join(OUTPUT_DIR, f'computation_graph_{fname}')
        dot.render(out_path, format='png', cleanup=True)
        print(f"[OK] Computation graph: {out_path}.png")
        del model, x, y


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  EXPORT ARCHITECTURE DIAGRAMS - 3 TICKNET-SMALL MODELS")
    print("  Section 3.3.4 - Ba mo hinh doi chung thuc nghiem")
    print("=" * 70)
    print(f"Output: {OUTPUT_DIR}\n")
    
    # 1. Text summary
    print("\n--- 1. Model architecture summary ---")
    export_model_summary()
    
    # 2. Side-by-side comparison
    print("\n--- 2. Architecture comparison diagram ---")
    plot_architecture_comparison()
    
    # 3. Individual model diagrams
    print("\n--- 3. Individual model diagrams ---")
    plot_single_model_detailed('Mo hinh 1: SE Baseline', 'se_only', 'arch_1_SE_Baseline.png')
    plot_single_model_detailed('Mo hinh 2: CBAM-Local', 'cbam_local', 'arch_2_CBAM_Local.png')
    plot_single_model_detailed('Mo hinh 3: CBAM-Hook (De xuat)', 'cbam_hook', 'arch_3_CBAM_Hook.png')
    
    # 4. Architecture table image
    print("\n--- 4. Architecture comparison table ---")
    plot_architecture_table()
    
    # 5. Computation graph (optional)
    print("\n--- 5. Computation graph (torchviz) ---")
    export_torchviz_graphs()
    
    print("\n" + "=" * 70)
    print(f"  DONE! All images saved to: {OUTPUT_DIR}")
    print("=" * 70)
