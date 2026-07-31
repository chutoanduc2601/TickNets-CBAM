"""
=======================================================================
HƯỚNG DẪN SỬ DỤNG:
=======================================================================
1. Mở notebook trên Kaggle
2. Đặt CONFIG['train_models'] = True
3. Chạy toàn bộ notebook từ đầu đến cuối
4. Copy TOÀN BỘ code bên dưới vào 1 cell mới ở cuối notebook
5. Chạy cell đó
6. Tải thư mục /kaggle/working/ticknet_results/ về máy
7. Đưa thư mục đó vào project d:\cu2\TickNets-CBAM\results\
=======================================================================

LƯU Ý: Script này chia làm 2 phần:
- PHẦN A: Dành cho notebook Classification (CIFAR-10 & Fashion-MNIST)
- PHẦN B: Dành cho notebook PlantVillage (PlantVillage & Chest-Xray)

Mỗi notebook chỉ cần copy phần tương ứng.
=======================================================================
"""

# =====================================================================
# PHẦN A - DÀNH CHO NOTEBOOK: TickNets_CBAM_Classification.ipynb
# Copy toàn bộ code dưới đây vào cell cuối của notebook Classification
# =====================================================================

EXPORT_CODE_CLASSIFICATION = '''
##############################################################################
# CELL XUẤT KẾT QUẢ - DÀNH CHO NOTEBOOK CLASSIFICATION (CIFAR-10 & FMNIST)
# Chạy cell này SAU KHI đã train xong tất cả mô hình (CONFIG['train_models'] = True)
##############################################################################
import json
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

OUTPUT_DIR = '/kaggle/working/ticknet_results_classification'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/plots', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/models', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/gradcam', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/reports', exist_ok=True)

print("=" * 60)
print("BẮT ĐẦU XUẤT KẾT QUẢ THỰC NGHIỆM")
print("=" * 60)

# ===== 1. XUẤT METRICS JSON =====
all_metrics = {}
for dataset_name, results_dict in [('CIFAR-10', results_cifar), ('Fashion-MNIST', results_fmnist)]:
    all_metrics[dataset_name] = {}
    for mode, res in results_dict.items():
        all_metrics[dataset_name][mode] = {
            'accuracy': round(res['acc'], 4),
            'precision': round(res['prec'], 4),
            'recall': round(res['rec'], 4),
            'f1_score': round(res['f1'], 4),
        }

with open(f'{OUTPUT_DIR}/metrics_summary.json', 'w', encoding='utf-8') as f:
    json.dump(all_metrics, f, indent=2, ensure_ascii=False)
print("[✓] Đã lưu metrics_summary.json")

# ===== 2. XUẤT TRAINING HISTORY CSV =====
for dataset_name, results_dict in [('cifar10', results_cifar), ('fmnist', results_fmnist)]:
    for mode, res in results_dict.items():
        if 'history' in res:
            history = res['history']
            csv_path = f'{OUTPUT_DIR}/{dataset_name}_{mode}_history.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
                for i in range(len(history['train_loss'])):
                    writer.writerow([
                        i + 1,
                        round(history['train_loss'][i], 6),
                        round(history['train_acc'][i], 4),
                        round(history['val_loss'][i], 6),
                        round(history['val_acc'][i], 4)
                    ])
            print(f"[✓] Đã lưu {csv_path}")

# ===== 3. VẼ VÀ LƯU BIỂU ĐỒ LOSS/ACCURACY =====
def plot_training_curves(histories, model_names, dataset_name, output_dir):
    """Vẽ biểu đồ loss và accuracy cho 3 mô hình trên 1 dataset"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    
    for i, (hist, name) in enumerate(zip(histories, model_names)):
        epochs = range(1, len(hist['train_loss']) + 1)
        # Loss
        axes[0].plot(epochs, hist['train_loss'], color=colors[i], linestyle='-', label=f'{name} (Train)', linewidth=1.5)
        axes[0].plot(epochs, hist['val_loss'], color=colors[i], linestyle='--', label=f'{name} (Val)', linewidth=1.5)
        # Accuracy
        axes[1].plot(epochs, hist['train_acc'], color=colors[i], linestyle='-', label=f'{name} (Train)', linewidth=1.5)
        axes[1].plot(epochs, hist['val_acc'], color=colors[i], linestyle='--', label=f'{name} (Val)', linewidth=1.5)
    
    axes[0].set_title(f'{dataset_name} — Loss theo Epoch', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title(f'{dataset_name} — Accuracy theo Epoch', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'{output_dir}/plots/{dataset_name.lower().replace("-", "_")}_training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Đã lưu {save_path}")

for dataset_name, results_dict in [('CIFAR-10', results_cifar), ('Fashion-MNIST', results_fmnist)]:
    histories = []
    names = []
    for mode in ['se_only', 'cbam_local', 'cbam_hook']:
        if 'history' in results_dict[mode]:
            histories.append(results_dict[mode]['history'])
            name_map = {'se_only': 'SE Baseline', 'cbam_local': 'CBAM-Local', 'cbam_hook': 'CBAM-Hook'}
            names.append(name_map[mode])
    if histories:
        plot_training_curves(histories, names, dataset_name, OUTPUT_DIR)

# ===== 4. VẼ VÀ LƯU BIỂU ĐỒ SO SÁNH ACCURACY =====
import pandas as pd

summary_data = {
    "Dataset": [],
    "Mô hình": [],
    "Accuracy (%)": [],
    "Precision (%)": [],
    "Recall (%)": [],
    "F1-Score (%)": []
}

for dataset_name, results_dict in [('CIFAR-10', results_cifar), ('Fashion-MNIST', results_fmnist)]:
    for mode in ['se_only', 'cbam_local', 'cbam_hook']:
        name_map = {'se_only': 'SE Baseline', 'cbam_local': 'CBAM-Local', 'cbam_hook': 'CBAM-Hook (Đề xuất)'}
        res = results_dict[mode]
        summary_data["Dataset"].append(dataset_name)
        summary_data["Mô hình"].append(name_map[mode])
        summary_data["Accuracy (%)"].append(res['acc'])
        summary_data["Precision (%)"].append(res['prec'])
        summary_data["Recall (%)"].append(res['rec'])
        summary_data["F1-Score (%)"].append(res['f1'])

df = pd.DataFrame(summary_data)
df.to_csv(f'{OUTPUT_DIR}/comparison_table_classification.csv', index=False, encoding='utf-8-sig')
print(f"[✓] Đã lưu comparison_table_classification.csv")

fig, ax = plt.subplots(figsize=(14, 7))
bar = sns.barplot(data=df, x="Dataset", y="Accuracy (%)", hue="Mô hình", palette=["#2196F3", "#FF9800", "#4CAF50"], ax=ax)
plt.title("So sánh Accuracy: 3 mô hình TickNet trên ảnh nhỏ (32×32)", fontsize=14)
plt.ylim(max(df["Accuracy (%)"].min() - 3, 85), min(df["Accuracy (%)"].max() + 2, 100))
for p in bar.patches:
    if p.get_height() > 0:
        bar.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/accuracy_comparison_classification.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[✓] Đã lưu accuracy_comparison_classification.png")

# ===== 5. LƯU MODEL WEIGHTS =====
# Train lại nhanh để lưu weights (hoặc dùng model cuối cùng nếu còn trong memory)
for dataset_name, loader_train, loader_test, num_cls, ds_key in [
    ('cifar10', cifar_train_loader, cifar_test_loader, 10, 'cifar'),
    ('fmnist', fmnist_train_loader, fmnist_test_loader, 10, 'fmnist')
]:
    for mode in ['se_only', 'cbam_local', 'cbam_hook']:
        model_path = f'{OUTPUT_DIR}/models/{dataset_name}_{mode}_best.pth'
        if not os.path.exists(model_path):
            print(f"[INFO] Training {mode} on {dataset_name} to save weights...")
            set_seed(42)
            model = TickNetSmall(num_classes=num_cls, attention_mode=mode,
                                 cbam_reduction=CONFIG['cbam_reduction'],
                                 cbam_spatial_kernel=CONFIG['cbam_spatial_kernel'],
                                 cifar=True).to(device)
            history = train_model(model, loader_train, loader_test, CONFIG, device)
            torch.save(model.state_dict(), model_path)
            print(f"[✓] Đã lưu {model_path}")
            
            # Lưu classification report
            model.eval()
            all_preds = []
            all_targets = []
            classes = cifar_classes if ds_key == 'cifar' else fmnist_classes
            with torch.no_grad():
                for images, labels in loader_test:
                    images = images.to(device)
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        outputs = model(images)
                        _, preds = outputs.max(1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(labels.numpy())
            
            report = classification_report(all_targets, all_preds, target_names=classes, digits=4)
            with open(f'{OUTPUT_DIR}/reports/{dataset_name}_{mode}_classification_report.txt', 'w') as f:
                f.write(f"Classification Report: {mode} on {dataset_name}\\n")
                f.write("=" * 60 + "\\n")
                f.write(report)
            print(f"[✓] Đã lưu classification report")
            
            # Lưu confusion matrix
            cm = confusion_matrix(all_targets, all_preds)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
            ax.set_title(f'Confusion Matrix: {mode} on {dataset_name}')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/plots/{dataset_name}_{mode}_confusion_matrix.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[✓] Đã lưu confusion matrix plot")
            
            del model
            gc.collect()
            torch.cuda.empty_cache()

# ===== 6. XUẤT GRAD-CAM THỰC TẾ =====
print("\\n[INFO] Tạo Grad-CAM thực tế...")
for dataset_name, loader_test, num_cls, ds_key in [
    ('cifar10', cifar_test_loader, 10, 'cifar'),
    ('fmnist', fmnist_test_loader, 10, 'fmnist')
]:
    classes = cifar_classes if ds_key == 'cifar' else fmnist_classes
    # Lấy 5 ảnh mẫu
    sample_images, sample_labels = next(iter(loader_test))
    sample_images = sample_images[:5]
    sample_labels = sample_labels[:5]
    
    for img_idx in range(min(5, len(sample_images))):
        input_tensor = sample_images[img_idx:img_idx+1].to(device)
        true_label = sample_labels[img_idx].item()
        
        # Denormalize cho hiển thị
        if ds_key == 'cifar':
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        else:
            mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        
        orig_img = sample_images[img_idx].cpu() * std + mean
        orig_img = orig_img.permute(1, 2, 0).numpy()
        orig_img = np.clip(orig_img, 0, 1)
        
        cams = []
        titles = []
        
        for mode, mode_name in [('se_only', 'SE Baseline'), ('cbam_local', 'CBAM-Local'), ('cbam_hook', 'CBAM-Hook')]:
            model_path = f'{OUTPUT_DIR}/models/{dataset_name}_{mode}_best.pth'
            if os.path.exists(model_path):
                model = TickNetSmall(num_classes=num_cls, attention_mode=mode,
                                     cbam_reduction=CONFIG['cbam_reduction'],
                                     cbam_spatial_kernel=CONFIG['cbam_spatial_kernel'],
                                     cifar=True).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                
                # Target layer: final_conv (lớp conv cuối trước GAP)
                target_layer = model.final_conv
                grad_cam = GradCAM(model, target_layer)
                cam = grad_cam.generate_cam(input_tensor)
                cams.append(cam)
                titles.append(mode_name)
                grad_cam.release()
                del model
                gc.collect()
                torch.cuda.empty_cache()
        
        if cams:
            fig, axes = plt.subplots(1, len(cams) + 1, figsize=(4 * (len(cams) + 1), 4))
            axes[0].imshow(orig_img)
            axes[0].set_title(f"Original\\n({classes[true_label]})", fontsize=10)
            axes[0].axis('off')
            
            for idx, (cam, title) in enumerate(zip(cams, titles)):
                axes[idx+1].imshow(orig_img)
                cam_resized = np.array(
                    __import__('PIL').Image.fromarray((cam * 255).astype(np.uint8)).resize(
                        (orig_img.shape[1], orig_img.shape[0]),
                        __import__('PIL').Image.BILINEAR
                    )
                ) / 255.0
                axes[idx+1].imshow(cam_resized, cmap='jet', alpha=0.5, interpolation='bilinear')
                axes[idx+1].set_title(title, fontsize=10)
                axes[idx+1].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/gradcam/{dataset_name}_sample{img_idx}_gradcam.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[✓] Đã lưu Grad-CAM: {dataset_name}_sample{img_idx}")

# ===== 7. XUẤT THÔNG TIN MÔ HÌNH =====
model_info = {}
for mode in ['se_only', 'cbam_local', 'cbam_hook']:
    model = TickNetSmall(num_classes=10, attention_mode=mode,
                         cbam_reduction=CONFIG['cbam_reduction'],
                         cbam_spatial_kernel=CONFIG['cbam_spatial_kernel'],
                         cifar=True)
    model_info[mode] = {
        'total_params': count_parameters(model),
        'attention_mode': mode,
    }
    del model

with open(f'{OUTPUT_DIR}/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)
print(f"[✓] Đã lưu model_info.json")

# ===== 8. XUẤT CONFIG =====
with open(f'{OUTPUT_DIR}/config.json', 'w') as f:
    config_export = {k: v for k, v in CONFIG.items()}
    json.dump(config_export, f, indent=2)
print(f"[✓] Đã lưu config.json")

print("\\n" + "=" * 60)
print("HOÀN TẤT! Tải thư mục sau về máy:")
print(f"  📁 {OUTPUT_DIR}/")
print("=" * 60)
print("\\nCấu trúc file xuất:")
for root, dirs, files in os.walk(OUTPUT_DIR):
    level = root.replace(OUTPUT_DIR, '').count(os.sep)
    indent = '  ' * level
    print(f'{indent}📁 {os.path.basename(root)}/')
    subindent = '  ' * (level + 1)
    for file in sorted(files):
        size = os.path.getsize(os.path.join(root, file))
        if size > 1024 * 1024:
            size_str = f"{size / (1024*1024):.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f'{subindent}📄 {file} ({size_str})')
'''

# =====================================================================
# PHẦN B - DÀNH CHO NOTEBOOK: TickNets_CBAM_PlantVillage.ipynb
# Copy toàn bộ code dưới đây vào cell cuối của notebook PlantVillage
# =====================================================================

EXPORT_CODE_PLANTVILLAGE = '''
##############################################################################
# CELL XUẤT KẾT QUẢ - DÀNH CHO NOTEBOOK PLANTVILLAGE (PlantVillage & Chest-Xray)
# Chạy cell này SAU KHI đã train xong tất cả mô hình (CONFIG['train_models'] = True)
##############################################################################
import json
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

OUTPUT_DIR = '/kaggle/working/ticknet_results_practical'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/plots', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/models', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/gradcam', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/reports', exist_ok=True)

print("=" * 60)
print("BẮT ĐẦU XUẤT KẾT QUẢ THỰC NGHIỆM (DỮ LIỆU THỰC TIỄN)")
print("=" * 60)

# ===== 1. XUẤT METRICS JSON =====
all_metrics = {}
for dataset_name, results_dict in [('PlantVillage', results_plant), ('Chest-Xray', results_xray)]:
    all_metrics[dataset_name] = {}
    for mode, res in results_dict.items():
        all_metrics[dataset_name][mode] = {
            'accuracy': round(res['acc'], 4),
            'precision': round(res['prec'], 4),
            'recall': round(res['rec'], 4),
            'f1_score': round(res['f1'], 4),
        }

with open(f'{OUTPUT_DIR}/metrics_summary.json', 'w', encoding='utf-8') as f:
    json.dump(all_metrics, f, indent=2, ensure_ascii=False)
print("[✓] Đã lưu metrics_summary.json")

# ===== 2. XUẤT TRAINING HISTORY CSV =====
for dataset_name, results_dict in [('plantvillage', results_plant), ('chestxray', results_xray)]:
    for mode, res in results_dict.items():
        if 'history' in res:
            history = res['history']
            csv_path = f'{OUTPUT_DIR}/{dataset_name}_{mode}_history.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
                for i in range(len(history['train_loss'])):
                    writer.writerow([
                        i + 1,
                        round(history['train_loss'][i], 6),
                        round(history['train_acc'][i], 4),
                        round(history['val_loss'][i], 6),
                        round(history['val_acc'][i], 4)
                    ])
            print(f"[✓] Đã lưu {csv_path}")

# ===== 3. VẼ VÀ LƯU BIỂU ĐỒ LOSS/ACCURACY =====
def plot_training_curves(histories, model_names, dataset_name, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    
    for i, (hist, name) in enumerate(zip(histories, model_names)):
        epochs = range(1, len(hist['train_loss']) + 1)
        axes[0].plot(epochs, hist['train_loss'], color=colors[i], linestyle='-', label=f'{name} (Train)', linewidth=1.5)
        axes[0].plot(epochs, hist['val_loss'], color=colors[i], linestyle='--', label=f'{name} (Val)', linewidth=1.5)
        axes[1].plot(epochs, hist['train_acc'], color=colors[i], linestyle='-', label=f'{name} (Train)', linewidth=1.5)
        axes[1].plot(epochs, hist['val_acc'], color=colors[i], linestyle='--', label=f'{name} (Val)', linewidth=1.5)
    
    axes[0].set_title(f'{dataset_name} — Loss theo Epoch', fontsize=14)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title(f'{dataset_name} — Accuracy theo Epoch', fontsize=14)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'{output_dir}/plots/{dataset_name.lower().replace("-", "_")}_training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Đã lưu {save_path}")

for dataset_name, results_dict in [('PlantVillage', results_plant), ('Chest-Xray', results_xray)]:
    histories = []
    names = []
    for mode in ['se_only', 'cbam_local', 'cbam_hook']:
        if 'history' in results_dict[mode]:
            histories.append(results_dict[mode]['history'])
            name_map = {'se_only': 'SE Baseline', 'cbam_local': 'CBAM-Local', 'cbam_hook': 'CBAM-Hook'}
            names.append(name_map[mode])
    if histories:
        plot_training_curves(histories, names, dataset_name, OUTPUT_DIR)

# ===== 4. VẼ VÀ LƯU BIỂU ĐỒ SO SÁNH ACCURACY =====
import pandas as pd

summary_data = {
    "Dataset": [], "Mô hình": [],
    "Accuracy (%)": [], "Precision (%)": [],
    "Recall (%)": [], "F1-Score (%)": []
}

for dataset_name, results_dict in [('PlantVillage', results_plant), ('Chest-Xray', results_xray)]:
    for mode in ['se_only', 'cbam_local', 'cbam_hook']:
        name_map = {'se_only': 'SE Baseline', 'cbam_local': 'CBAM-Local', 'cbam_hook': 'CBAM-Hook (Đề xuất)'}
        res = results_dict[mode]
        summary_data["Dataset"].append(dataset_name)
        summary_data["Mô hình"].append(name_map[mode])
        summary_data["Accuracy (%)"].append(res['acc'])
        summary_data["Precision (%)"].append(res['prec'])
        summary_data["Recall (%)"].append(res['rec'])
        summary_data["F1-Score (%)"].append(res['f1'])

df = pd.DataFrame(summary_data)
df.to_csv(f'{OUTPUT_DIR}/comparison_table_practical.csv', index=False, encoding='utf-8-sig')
print(f"[✓] Đã lưu comparison_table_practical.csv")

fig, ax = plt.subplots(figsize=(14, 7))
bar = sns.barplot(data=df, x="Dataset", y="Accuracy (%)", hue="Mô hình", palette=["#2196F3", "#FF9800", "#4CAF50"], ax=ax)
plt.title("So sánh Accuracy: 3 mô hình TickNet trên dữ liệu thực tiễn (224×224)", fontsize=14)
plt.ylim(max(df["Accuracy (%)"].min() - 5, 80), min(df["Accuracy (%)"].max() + 3, 100))
for p in bar.patches:
    if p.get_height() > 0:
        bar.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/accuracy_comparison_practical.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[✓] Đã lưu accuracy_comparison_practical.png")

# ===== 5. LƯU MODEL WEIGHTS + CLASSIFICATION REPORT + CONFUSION MATRIX =====
for dataset_name, loader_train, loader_test, num_cls, classes in [
    ('plantvillage', plant_train_loader, plant_test_loader, num_plant_classes, plant_classes),
    ('chestxray', xray_train_loader, xray_test_loader, num_xray_classes, xray_classes)
]:
    for mode in ['se_only', 'cbam_local', 'cbam_hook']:
        model_path = f'{OUTPUT_DIR}/models/{dataset_name}_{mode}_best.pth'
        if not os.path.exists(model_path):
            print(f"\\n[INFO] Training {mode} on {dataset_name} to save weights...")
            set_seed(42)
            model = TickNetSmall(num_classes=num_cls, attention_mode=mode,
                                 cbam_reduction=CONFIG['cbam_reduction'],
                                 cbam_spatial_kernel=CONFIG['cbam_spatial_kernel'],
                                 cifar=False).to(device)
            history = train_model(model, loader_train, loader_test, CONFIG, device)
            torch.save(model.state_dict(), model_path)
            print(f"[✓] Đã lưu {model_path}")
            
            # Classification report
            model.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for images, labels in loader_test:
                    images = images.to(device)
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        outputs = model(images)
                        _, preds = outputs.max(1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(labels.numpy())
            
            report = classification_report(all_targets, all_preds, target_names=classes, digits=4)
            with open(f'{OUTPUT_DIR}/reports/{dataset_name}_{mode}_classification_report.txt', 'w') as f:
                f.write(f"Classification Report: {mode} on {dataset_name}\\n")
                f.write("=" * 60 + "\\n")
                f.write(report)
            print(f"[✓] Đã lưu classification report")
            
            # Confusion matrix
            cm = confusion_matrix(all_targets, all_preds)
            fig_size = max(8, len(classes) * 0.4)
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
            sns.heatmap(cm, annot=(len(classes) <= 20), fmt='d', cmap='Blues', ax=ax,
                       xticklabels=classes if len(classes) <= 20 else False,
                       yticklabels=classes if len(classes) <= 20 else False)
            ax.set_title(f'Confusion Matrix: {mode} on {dataset_name}')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            if len(classes) <= 20:
                plt.xticks(rotation=45, ha='right', fontsize=7)
                plt.yticks(fontsize=7)
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/plots/{dataset_name}_{mode}_confusion_matrix.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[✓] Đã lưu confusion matrix plot")
            
            # Lưu confusion matrix raw data
            np.save(f'{OUTPUT_DIR}/reports/{dataset_name}_{mode}_confusion_matrix.npy', cm)
            
            del model
            gc.collect()
            torch.cuda.empty_cache()

# ===== 6. XUẤT GRAD-CAM THỰC TẾ =====
print("\\n[INFO] Tạo Grad-CAM thực tế...")
for dataset_name, loader_test, num_cls, classes in [
    ('plantvillage', plant_test_loader, num_plant_classes, plant_classes),
    ('chestxray', xray_test_loader, num_xray_classes, xray_classes)
]:
    sample_images, sample_labels = next(iter(loader_test))
    sample_images = sample_images[:5]
    sample_labels = sample_labels[:5]
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for img_idx in range(min(5, len(sample_images))):
        input_tensor = sample_images[img_idx:img_idx+1].to(device)
        true_label = sample_labels[img_idx].item()
        
        orig_img = sample_images[img_idx].cpu() * std + mean
        orig_img = orig_img.permute(1, 2, 0).numpy()
        orig_img = np.clip(orig_img, 0, 1)
        
        cams = []
        titles = []
        
        for mode, mode_name in [('se_only', 'SE Baseline'), ('cbam_local', 'CBAM-Local'), ('cbam_hook', 'CBAM-Hook')]:
            model_path = f'{OUTPUT_DIR}/models/{dataset_name}_{mode}_best.pth'
            if os.path.exists(model_path):
                model = TickNetSmall(num_classes=num_cls, attention_mode=mode,
                                     cbam_reduction=CONFIG['cbam_reduction'],
                                     cbam_spatial_kernel=CONFIG['cbam_spatial_kernel'],
                                     cifar=False).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                
                target_layer = model.final_conv
                grad_cam = GradCAM(model, target_layer)
                cam = grad_cam.generate_cam(input_tensor)
                cams.append(cam)
                titles.append(mode_name)
                grad_cam.release()
                del model
                gc.collect()
                torch.cuda.empty_cache()
        
        if cams:
            fig, axes = plt.subplots(1, len(cams) + 1, figsize=(4 * (len(cams) + 1), 4))
            class_name = classes[true_label] if true_label < len(classes) else str(true_label)
            axes[0].imshow(orig_img)
            axes[0].set_title(f"Original\\n({class_name})", fontsize=9)
            axes[0].axis('off')
            
            for idx, (cam, title) in enumerate(zip(cams, titles)):
                axes[idx+1].imshow(orig_img)
                from PIL import Image
                cam_resized = np.array(
                    Image.fromarray((cam * 255).astype(np.uint8)).resize(
                        (orig_img.shape[1], orig_img.shape[0]),
                        Image.BILINEAR
                    )
                ) / 255.0
                axes[idx+1].imshow(cam_resized, cmap='jet', alpha=0.5, interpolation='bilinear')
                axes[idx+1].set_title(title, fontsize=10)
                axes[idx+1].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/gradcam/{dataset_name}_sample{img_idx}_gradcam.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[✓] Đã lưu Grad-CAM: {dataset_name}_sample{img_idx}")

# ===== 7. XUẤT THÔNG TIN DATASET =====
dataset_info = {
    'PlantVillage': {
        'num_classes': num_plant_classes,
        'class_names': list(plant_classes) if isinstance(plant_classes, (list, tuple)) else [str(c) for c in plant_classes],
        'image_size': '224x224',
        'color_mode': 'RGB',
    },
    'Chest-Xray': {
        'num_classes': num_xray_classes,
        'class_names': list(xray_classes) if isinstance(xray_classes, (list, tuple)) else [str(c) for c in xray_classes],
        'image_size': '224x224',
        'color_mode': 'RGB (converted from grayscale)',
    }
}
with open(f'{OUTPUT_DIR}/dataset_info.json', 'w', encoding='utf-8') as f:
    json.dump(dataset_info, f, indent=2, ensure_ascii=False)
print(f"[✓] Đã lưu dataset_info.json")

# ===== 8. XUẤT MODEL INFO + CONFIG =====
model_info = {}
for mode in ['se_only', 'cbam_local', 'cbam_hook']:
    model = TickNetSmall(num_classes=10, attention_mode=mode,
                         cbam_reduction=CONFIG['cbam_reduction'],
                         cbam_spatial_kernel=CONFIG['cbam_spatial_kernel'],
                         cifar=False)
    model_info[mode] = {
        'total_params': count_parameters(model),
        'attention_mode': mode,
        'image_size': '224x224',
    }
    del model

with open(f'{OUTPUT_DIR}/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

with open(f'{OUTPUT_DIR}/config.json', 'w') as f:
    json.dump({k: v for k, v in CONFIG.items()}, f, indent=2)

print(f"[✓] Đã lưu model_info.json và config.json")

print("\\n" + "=" * 60)
print("HOÀN TẤT! Tải thư mục sau về máy:")
print(f"  📁 {OUTPUT_DIR}/")
print("=" * 60)
print("\\nCấu trúc file xuất:")
for root, dirs, files in os.walk(OUTPUT_DIR):
    level = root.replace(OUTPUT_DIR, '').count(os.sep)
    indent = '  ' * level
    print(f'{indent}📁 {os.path.basename(root)}/')
    subindent = '  ' * (level + 1)
    for file in sorted(files):
        size = os.path.getsize(os.path.join(root, file))
        if size > 1024 * 1024:
            size_str = f"{size / (1024*1024):.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f'{subindent}📄 {file} ({size_str})')
'''

# =====================================================================
# IN HƯỚNG DẪN
# =====================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("  HƯỚNG DẪN XUẤT KẾT QUẢ THỰC NGHIỆM CHO TIỂU LUẬN")
    print("=" * 70)
    print()
    print("BƯỚC 1: Mở notebook trên Kaggle")
    print("BƯỚC 2: Đặt CONFIG['train_models'] = True")  
    print("BƯỚC 3: Chạy toàn bộ notebook")
    print("BƯỚC 4: Thêm cell mới ở cuối và paste code export tương ứng")
    print("BƯỚC 5: Chạy cell export")
    print("BƯỚC 6: Tải thư mục kết quả về máy")
    print()
    print("Chi tiết code export đã lưu trong file này.")
    print("- PHẦN A: cho TickNets_CBAM_Classification.ipynb")
    print("- PHẦN B: cho TickNets_CBAM_PlantVillage.ipynb")
    print()
    print("Sau khi tải về, đặt vào:")
    print("  d:\\cu2\\TickNets-CBAM\\results\\classification\\")
    print("  d:\\cu2\\TickNets-CBAM\\results\\practical\\")
