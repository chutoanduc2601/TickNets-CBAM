##############################################################################
# CELL XUẤT KẾT QUẢ — NOTEBOOK: TickNets_CBAM_PlantVillage.ipynb
# 
# HƯỚNG DẪN:
#   1. Đặt CONFIG['train_models'] = True ở cell CONFIG
#   2. Chạy toàn bộ notebook từ đầu
#   3. Copy TOÀN BỘ code trong file này vào 1 cell MỚI ở cuối notebook
#   4. Chạy cell đó
#   5. Tải thư mục /kaggle/working/ticknet_results_practical/ về
#   6. Đặt vào d:\cu2\TickNets-CBAM\results\practical\
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

OUTPUT_DIR = '/kaggle/working/ticknet_results_practical'
for d in ['', '/plots', '/models', '/gradcam', '/reports']:
    os.makedirs(OUTPUT_DIR + d, exist_ok=True)

print("=" * 60)
print("BẮT ĐẦU XUẤT KẾT QUẢ — PlantVillage & Chest-Xray")
print("=" * 60)

# ─── 1. Metrics JSON ───
all_metrics = {}
for ds_name, res_dict in [('PlantVillage', results_plant), ('Chest-Xray', results_xray)]:
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
for ds_tag, res_dict in [('plantvillage', results_plant), ('chestxray', results_xray)]:
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

name_map = {'se_only':'SE Baseline', 'cbam_local':'CBAM-Local', 'cbam_hook':'CBAM-Hook'}
for ds_name, res_dict in [('PlantVillage', results_plant), ('Chest-Xray', results_xray)]:
    hists = [(res_dict[m]['history'], name_map[m]) for m in ['se_only','cbam_local','cbam_hook'] if 'history' in res_dict[m]]
    if hists:
        plot_curves([h for h,_ in hists], [n for _,n in hists], ds_name, OUTPUT_DIR)

# ─── 4. Comparison bar chart + CSV ───
rows = []
for ds_name, res_dict in [('PlantVillage', results_plant), ('Chest-Xray', results_xray)]:
    for mode in ['se_only','cbam_local','cbam_hook']:
        nm = {'se_only':'SE Baseline','cbam_local':'CBAM-Local','cbam_hook':'CBAM-Hook (Đề xuất)'}
        r = res_dict[mode]
        rows.append({'Dataset': ds_name, 'Model': nm[mode],
                     'Accuracy': r['acc'], 'Precision': r['prec'], 'Recall': r['rec'], 'F1': r['f1']})
df = pd.DataFrame(rows)
df.to_csv(f'{OUTPUT_DIR}/comparison_table.csv', index=False, encoding='utf-8-sig')
print("[✓] comparison_table.csv")

fig, ax = plt.subplots(figsize=(14, 7))
bar = sns.barplot(data=df, x='Dataset', y='Accuracy', hue='Model', palette=["#2196F3","#FF9800","#4CAF50"], ax=ax)
plt.title("So sánh Accuracy — 3 mô hình TickNet trên dữ liệu thực tiễn (224×224)", fontsize=14)
plt.ylim(max(df['Accuracy'].min()-5, 80), min(df['Accuracy'].max()+3, 100))
for p in bar.patches:
    if p.get_height() > 0:
        bar.annotate(f"{p.get_height():.2f}%", (p.get_x()+p.get_width()/2., p.get_height()),
                     ha='center', va='bottom', fontsize=9, xytext=(0,3), textcoords='offset points')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/accuracy_comparison.png', dpi=150, bbox_inches='tight'); plt.close()
print("[✓] accuracy_comparison.png")

# ─── 5. Model weights, classification reports, confusion matrices ───
datasets_info = [
    ('plantvillage', plant_train_loader, plant_test_loader, num_plant_classes, plant_classes),
    ('chestxray', xray_train_loader, xray_test_loader, num_xray_classes, xray_classes),
]
for ds_tag, ldr_train, ldr_test, n_cls, classes in datasets_info:
    for mode in ['se_only','cbam_local','cbam_hook']:
        mp = f'{OUTPUT_DIR}/models/{ds_tag}_{mode}_best.pth'
        if os.path.exists(mp):
            print(f"[SKIP] {os.path.basename(mp)} already exists")
            continue
        print(f"\n[TRAIN] {mode} on {ds_tag}...")
        set_seed(42)
        model = TickNetSmall(num_classes=n_cls, attention_mode=mode,
                             cbam_reduction=CONFIG['cbam_reduction'],
                             cbam_spatial_kernel=CONFIG['cbam_spatial_kernel'],
                             cifar=False).to(device)
        train_model(model, ldr_train, ldr_test, CONFIG, device)
        torch.save(model.state_dict(), mp)
        print(f"[✓] {os.path.basename(mp)}")

        # Evaluate
        model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for imgs, lbs in ldr_test:
                imgs = imgs.to(device)
                with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                    _, preds = model(imgs).max(1)
                all_p.extend(preds.cpu().numpy()); all_t.extend(lbs.numpy())

        # Report
        cls_list = list(classes) if isinstance(classes, (list, tuple)) else [str(c) for c in classes]
        rpt = classification_report(all_t, all_p, target_names=cls_list, digits=4)
        with open(f'{OUTPUT_DIR}/reports/{ds_tag}_{mode}_report.txt', 'w') as f:
            f.write(rpt)
        print(f"[✓] {ds_tag}_{mode}_report.txt")

        # Confusion matrix
        cm = confusion_matrix(all_t, all_p)
        np.save(f'{OUTPUT_DIR}/reports/{ds_tag}_{mode}_cm.npy', cm)
        fig_sz = max(8, len(cls_list) * 0.4)
        show_labels = len(cls_list) <= 20
        fig, ax = plt.subplots(figsize=(fig_sz, fig_sz))
        sns.heatmap(cm, annot=show_labels, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=cls_list if show_labels else False,
                   yticklabels=cls_list if show_labels else False)
        ax.set_title(f'Confusion Matrix: {name_map[mode]} — {ds_tag}')
        ax.set_ylabel('True'); ax.set_xlabel('Predicted')
        if show_labels:
            plt.xticks(rotation=45, ha='right', fontsize=7)
            plt.yticks(fontsize=7)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/plots/{ds_tag}_{mode}_cm.png', dpi=150, bbox_inches='tight'); plt.close()
        print(f"[✓] {ds_tag}_{mode}_cm.png")

        del model; gc.collect(); torch.cuda.empty_cache()

# ─── 6. Grad-CAM ───
print("\n[INFO] Generating Grad-CAM...")
mean_img = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
std_img = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

for ds_tag, ldr_test, n_cls, classes in [
    ('plantvillage', plant_test_loader, num_plant_classes, plant_classes),
    ('chestxray', xray_test_loader, num_xray_classes, xray_classes),
]:
    imgs, lbs = next(iter(ldr_test))
    imgs, lbs = imgs[:5], lbs[:5]
    cls_list = list(classes) if isinstance(classes, (list, tuple)) else [str(c) for c in classes]

    for ii in range(min(5, len(imgs))):
        inp = imgs[ii:ii+1].to(device)
        orig = np.clip((imgs[ii].cpu()*std_img+mean_img).permute(1,2,0).numpy(), 0, 1)
        cams, titles = [], []
        for mode, mn in [('se_only','SE'), ('cbam_local','CBAM-Local'), ('cbam_hook','CBAM-Hook')]:
            mp = f'{OUTPUT_DIR}/models/{ds_tag}_{mode}_best.pth'
            if not os.path.exists(mp): continue
            m = TickNetSmall(n_cls, mode, CONFIG['cbam_reduction'], CONFIG['cbam_spatial_kernel'], cifar=False).to(device)
            m.load_state_dict(torch.load(mp, map_location=device)); m.eval()
            gc_obj = GradCAM(m, m.final_conv)
            cam = gc_obj.generate_cam(inp)
            cams.append(cam); titles.append(mn)
            gc_obj.release(); del m; gc.collect(); torch.cuda.empty_cache()

        if cams:
            fig, axes = plt.subplots(1, len(cams)+1, figsize=(4*(len(cams)+1), 4))
            label_name = cls_list[lbs[ii].item()] if lbs[ii].item() < len(cls_list) else str(lbs[ii].item())
            axes[0].imshow(orig); axes[0].set_title(f"Original\n({label_name})", fontsize=9); axes[0].axis('off')
            for idx, (c, t) in enumerate(zip(cams, titles)):
                axes[idx+1].imshow(orig)
                cr = np.array(Image.fromarray((c*255).astype(np.uint8)).resize(
                    (orig.shape[1], orig.shape[0]), Image.BILINEAR)) / 255.0
                axes[idx+1].imshow(cr, cmap='jet', alpha=0.5); axes[idx+1].set_title(t); axes[idx+1].axis('off')
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/gradcam/{ds_tag}_s{ii}.png', dpi=150, bbox_inches='tight'); plt.close()
            print(f"[✓] gradcam/{ds_tag}_s{ii}.png")

# ─── 7. Dataset info, Model info, Config ───
dataset_info = {}
try:
    dataset_info['PlantVillage'] = {
        'num_classes': num_plant_classes,
        'class_names': list(plant_classes) if isinstance(plant_classes, (list,tuple)) else [str(c) for c in plant_classes],
        'image_size': '224x224', 'color_mode': 'RGB',
        'train_size': len(plant_train_loader.dataset),
        'test_size': len(plant_test_loader.dataset),
    }
except: pass
try:
    dataset_info['Chest-Xray'] = {
        'num_classes': num_xray_classes,
        'class_names': list(xray_classes) if isinstance(xray_classes, (list,tuple)) else [str(c) for c in xray_classes],
        'image_size': '224x224', 'color_mode': 'RGB (from grayscale)',
        'train_size': len(xray_train_loader.dataset),
        'test_size': len(xray_test_loader.dataset),
    }
except: pass
with open(f'{OUTPUT_DIR}/dataset_info.json', 'w', encoding='utf-8') as f:
    json.dump(dataset_info, f, indent=2, ensure_ascii=False)
print("[✓] dataset_info.json")

mi = {}
for mode in ['se_only','cbam_local','cbam_hook']:
    m = TickNetSmall(10, mode, CONFIG['cbam_reduction'], CONFIG['cbam_spatial_kernel'], cifar=False)
    mi[mode] = {'params': count_parameters(m), 'image_size': '224x224'}; del m
with open(f'{OUTPUT_DIR}/model_info.json', 'w') as f: json.dump(mi, f, indent=2)
with open(f'{OUTPUT_DIR}/config.json', 'w') as f: json.dump(dict(CONFIG), f, indent=2)
print("[✓] model_info.json, config.json")

print("\n" + "=" * 60)
print(f"HOÀN TẤT! Tải thư mục: {OUTPUT_DIR}/")
print("=" * 60)
for root, dirs, files in os.walk(OUTPUT_DIR):
    lvl = root.replace(OUTPUT_DIR, '').count(os.sep)
    print('  '*lvl + f'📁 {os.path.basename(root)}/')
    for f in sorted(files):
        sz = os.path.getsize(os.path.join(root, f))
        s = f"{sz/(1024*1024):.1f}MB" if sz>1048576 else f"{sz/1024:.1f}KB" if sz>1024 else f"{sz}B"
        print('  '*(lvl+1) + f'📄 {f} ({s})')
