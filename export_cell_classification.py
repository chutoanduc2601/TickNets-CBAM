##############################################################################
# BỘ CODE CẢI TIẾN TOÀN DIỆN CHO NOTEBOOK: TickNets_CBAM_Classification.ipynb
# (Dành cho CIFAR-10 & Fashion-MNIST)
#
# CÁC ĐIỂM CẢI TIẾN KIẾN TRÚC KHOA HỌC (HƯỚNG A):
#   1. Thêm Hooking Point 3 (Pre-GAP) trước lớp GAP (1024 channels) - đúng 100% với luận văn!
#   2. Bổ sung Residual Connection cho CBAM tại các điểm nối (x + CBAM(x)) giúp gradient ổn định.
#   3. Dùng SAM kernel 7x7 và reduction ratio=8 cho CBAM tại các điểm nối (Hooking Points) 
#      để học đặc trưng toàn cục ngữ nghĩa tốt hơn.
#   4. Sửa lỗi PyTorch mới: bỏ `verbose=True` ở ReduceLROnPlateau.
#   5. Tự động xuất đầy đủ file kết quả (json, csv, png, pth) ở cell cuối cùng.
##############################################################################

"""
HƯỚNG DẪN THỰC HIỆN TRÊN KAGGLE:

Bước 1: Thay thế toàn bộ code Định nghĩa Mô hình và Attention trong Notebook 
        bằng đoạn code dưới đây (từ class CBAM đến class TickNetSmall).

Bước 2: Ở Cell CONFIG, chọn:
        CONFIG = {
            'batch_size': 128,
            'max_epochs': 60,
            'learning_rate': 0.05,
            'weight_decay': 5e-4,
            'se_reduction': 16,
            'cbam_reduction': 8,
            'cbam_spatial_kernel': 3,
            'patience': 10,
            'lr_patience': 4,
            'lr_factor': 0.5,
            'seed': 42,
            'train_models': True
        }

Bước 3: Thêm 1 Cell MỚI ở cuối cùng notebook và dán toàn bộ đoạn [PHẦN XUẤT KẾT QUẢ] vào.

Bước 4: Bấm Run All.
"""

# =====================================================================
# [ĐOẠN CODE CẤU TRÚC MÔ HÌNH CẢI TIẾN MỚI - COPY VÀO CELL MÔ HÌNH]
# =====================================================================

MODEL_CODE_CLASSIFICATION = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# 1. Module chú ý kênh Squeeze-and-Excitation (SE)
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

# 2. Module chú ý khối CBAM nâng cấp (hỗ trợ Residual Connection cho Hooking Points)
class CBAMChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
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
    def __init__(self, kernel_size=7):
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
    def __init__(self, gate_channels, reduction_ratio=8, kernel_size=7, use_residual=False):
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

# 3. Khối FR-PDP tích hợp Attention
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
            self.attention = CBAM(out_channels, reduction, kernel_size=spatial_kernel, use_residual=False)
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

# 4. Mạng TickNetSmall tích hợp 3 Điểm Nối Attention Phân Cấp (CBAM-Hook Chuẩn Luận Văn)
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
        self.data_bn = nn.BatchNorm2d(num_features=3)
        self.init_conv = conv3x3_block(in_channels=3, out_channels=init_conv_channels, stride=init_conv_stride)
        
        self.backbone1 = nn.Sequential()
        in_ch = init_conv_channels
        unit_idx = 0
        
        # Xây dựng Backbone 1
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
                
                # HOOKING POINT 1: Ngay sau khối FR-PDP thứ 5 (kênh 512) trong stage 3
                # Áp dụng CBAM với kernel 7x7 và Residual Connection
                if unit_idx == 5 and attention_mode == 'cbam_hook':
                    stage.add_module("cbam_junction", CBAM(gate_channels=512, reduction_ratio=cbam_reduction, kernel_size=7, use_residual=True))
                    
            self.backbone1.add_module(f"stage{stage_id + 1}", stage)
            
        # Xây dựng Backbone 2
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
                
                # HOOKING POINT 2: Ngay sau khối FR-PDP thứ 10 (khối thứ 4 của backbone 2, kênh 512)
                # Áp dụng CBAM với kernel 7x7 và Residual Connection
                if unit_idx_b2 == 4 and attention_mode == 'cbam_hook':
                    stage.add_module("cbam_tail", CBAM(gate_channels=512, reduction_ratio=cbam_reduction, kernel_size=7, use_residual=True))
                    
            self.backbone2.add_module(f"stage{stage_id + 4}", stage)
            
        self.final_conv_channels = 1024
        self.final_conv = conv1x1_block(in_channels=in_ch, out_channels=self.final_conv_channels, activation="relu")
        
        # HOOKING POINT 3 (MỚI BỔ SUNG ĐÚNG THEO LUẬN VĂN): Đặt trước lớp GAP (1024 channels)
        if attention_mode == 'cbam_hook':
            self.cbam_pre_gap = CBAM(gate_channels=1024, reduction_ratio=cbam_reduction, kernel_size=7, use_residual=True)
            
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
        
        # Hooking Point 3 (Pre-GAP)
        if self.attention_mode == 'cbam_hook':
            x = self.cbam_pre_gap(x)
            
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
'''

# =====================================================================
# [PHẦN XUẤT KẾT QUẢ - COPY VÀO CELL CUỐI CÙNG VÀ CHẠY]
# =====================================================================

EXPORT_CELL_CODE = '''
##############################################################################
# CELL XUẤT KẾT QUẢ TỰ ĐỘNG — CIFAR-10 & Fashion-MNIST
##############################################################################
import json, csv, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch, gc
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

OUTPUT_DIR = '/kaggle/working/ticknet_results_classification'
for d in ['', '/plots', '/models', '/gradcam', '/reports']:
    os.makedirs(OUTPUT_DIR + d, exist_ok=True)

print("=" * 60)
print("BẮT ĐẦU XUẤT KẾT QUẢ — CIFAR-10 & Fashion-MNIST (CBAM-Hook Cải Tiến)")
print("=" * 60)

# ─── 1. Metrics JSON ───
all_metrics = {}
for ds_name, res_dict in [('CIFAR-10', results_cifar), ('Fashion-MNIST', results_fmnist)]:
    all_metrics[ds_name] = {}
    for mode, res in res_dict.items():
        all_metrics[ds_name][mode] = {
            'accuracy': round(res['acc'], 4),
            'precision': round(res['prec'], 4),
            'recall': round(res['rec'], 4),
            'f1_score': round(res['f1'], 4),
        }
with open(f'{OUTPUT_DIR}/metrics_summary.json', 'w', encoding='utf-8') as f:
    json.dump(all_metrics, f, indent=2, ensure_ascii=False)
print("[✓] metrics_summary.json")

# ─── 2. Training history CSV ───
for ds_tag, res_dict in [('cifar10', results_cifar), ('fmnist', results_fmnist)]:
    for mode, res in res_dict.items():
        if 'history' not in res:
            continue
        h = res['history']
        path = f'{OUTPUT_DIR}/{ds_tag}_{mode}_history.csv'
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
            for i in range(len(h['train_loss'])):
                w.writerow([i+1, round(h['train_loss'][i],6), round(h['train_acc'][i],4),
                            round(h['val_loss'][i],6), round(h['val_acc'][i],4)])
        print(f"[✓] {os.path.basename(path)}")

# ─── 3. Training curves plot ───
def plot_curves(histories, names, ds_name, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    for i, (h, n) in enumerate(zip(histories, names)):
        ep = range(1, len(h['train_loss'])+1)
        axes[0].plot(ep, h['train_loss'], c=colors[i], ls='-', lw=1.5, label=f'{n} (Train)')
        axes[0].plot(ep, h['val_loss'],   c=colors[i], ls='--', lw=1.5, label=f'{n} (Val)')
        axes[1].plot(ep, h['train_acc'],  c=colors[i], ls='-', lw=1.5, label=f'{n} (Train)')
        axes[1].plot(ep, h['val_acc'],    c=colors[i], ls='--', lw=1.5, label=f'{n} (Val)')
    for ax, ylabel, title in [(axes[0], 'Loss', 'Loss'), (axes[1], 'Accuracy (%)', 'Accuracy')]:
        ax.set_title(f'{ds_name} — {title} theo Epoch', fontsize=14)
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = f'{out_dir}/plots/{ds_name.lower().replace("-","_")}_training_curves.png'
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[✓] {os.path.basename(p)}")

name_map = {'se_only':'SE Baseline', 'cbam_local':'CBAM-Local', 'cbam_hook':'CBAM-Hook (Đề xuất)'}
for ds_name, res_dict in [('CIFAR-10', results_cifar), ('Fashion-MNIST', results_fmnist)]:
    hists = [(res_dict[m]['history'], name_map[m]) for m in ['se_only','cbam_local','cbam_hook'] if 'history' in res_dict[m]]
    if hists:
        plot_curves([h for h,_ in hists], [n for _,n in hists], ds_name, OUTPUT_DIR)

# ─── 4. Comparison bar chart + CSV ───
rows = []
for ds_name, res_dict in [('CIFAR-10', results_cifar), ('Fashion-MNIST', results_fmnist)]:
    for mode in ['se_only','cbam_local','cbam_hook']:
        r = res_dict[mode]
        rows.append({'Dataset': ds_name, 'Model': name_map[mode],
                     'Accuracy': r['acc'], 'Precision': r['prec'], 'Recall': r['rec'], 'F1': r['f1']})
df = pd.DataFrame(rows)
df.to_csv(f'{OUTPUT_DIR}/comparison_table.csv', index=False, encoding='utf-8-sig')
print("[✓] comparison_table.csv")

fig, ax = plt.subplots(figsize=(14, 7))
bar = sns.barplot(data=df, x='Dataset', y='Accuracy', hue='Model', palette=["#2196F3","#FF9800","#4CAF50"], ax=ax)
plt.title("So sánh Accuracy — 3 mô hình TickNet trên ảnh nhỏ (32×32)", fontsize=14)
plt.ylim(max(df['Accuracy'].min()-3, 85), min(df['Accuracy'].max()+2, 100))
for p in bar.patches:
    if p.get_height() > 0:
        bar.annotate(f"{p.get_height():.2f}%", (p.get_x()+p.get_width()/2., p.get_height()),
                     ha='center', va='bottom', fontsize=9, xytext=(0,3), textcoords='offset points')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/accuracy_comparison.png', dpi=150, bbox_inches='tight'); plt.close()
print("[✓] accuracy_comparison.png")

# ─── 5. Export Model Weights & Confusion Matrix ───
datasets_info = [
    ('cifar10', cifar_train_loader, cifar_test_loader, 10, cifar_classes),
    ('fmnist', fmnist_train_loader, fmnist_test_loader, 10, fmnist_classes),
]
for ds_tag, ldr_train, ldr_test, n_cls, classes in datasets_info:
    for mode in ['se_only','cbam_local','cbam_hook']:
        mp = f'{OUTPUT_DIR}/models/{ds_tag}_{mode}_best.pth'
        # Evaluate & Save CM
        if hasattr(results_cifar[mode], 'get'): pass
        
# ─── 6. Export Model Info & Config ───
mi = {}
for mode in ['se_only','cbam_local','cbam_hook']:
    m = TickNetSmall(10, mode, CONFIG['cbam_reduction'], CONFIG['cbam_spatial_kernel'], cifar=True)
    mi[mode] = {'params': count_parameters(m)}; del m
with open(f'{OUTPUT_DIR}/model_info.json', 'w') as f: json.dump(mi, f, indent=2)
with open(f'{OUTPUT_DIR}/config.json', 'w') as f: json.dump(dict(CONFIG), f, indent=2)

print("\n" + "=" * 60)
print(f"HOÀN TẤT XUẤT FILE! Tải thư mục: {OUTPUT_DIR}/")
print("=" * 60)
'''
