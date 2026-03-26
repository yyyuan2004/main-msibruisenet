import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from data.dataset import MSIDataset
from data.split import get_data_splits
from data.augment import get_val_transforms
from model.model import build_model

ckpt_path = "outputs/band_attn_seed43/checkpoints/best_model.pth"
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
cfg = checkpoint["config"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(cfg).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 加载 val 数据
splits = get_data_splits(cfg["data"]["data_dir"], cfg["data"]["image_dir"], seed=43)
val_dataset = MSIDataset(
    splits["val"], data_dir=cfg["data"]["data_dir"],
    image_dir=cfg["data"]["image_dir"], mask_dir=cfg["data"]["mask_dir"],
    transform=get_val_transforms(cfg), num_classes=2,
)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

all_weights = []
with torch.no_grad():
    for images, masks, stems in val_loader:
        images = images.to(device)
        w = model.band_attention.get_weights(images)
        all_weights.append(w)

avg_w = np.concatenate(all_weights, axis=0).mean(axis=0)
print("Average band weights across val set:")
for i, w in enumerate(avg_w):
    print(f"  Band {i+1}: {w:.4f}")

# 画图
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(1, len(avg_w)+1), avg_w, color='steelblue')
ax.set_xlabel('Band Index')
ax.set_ylabel('Learned Weight')
ax.set_title('InputBandSE: Average Band Importance')
ax.set_xticks(range(1, len(avg_w)+1))
ax.set_ylim(0, 1)
fig.savefig('band_weights.png', dpi=150)
print("Saved band_weights.png")
