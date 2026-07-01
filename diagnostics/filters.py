import torch
from matplotlib import pyplot as plt
from src.model import MyNet
from src.config import CHECKPOINT_FILE, OUTPUT_DIR

model = MyNet()
checkpoint = torch.load(CHECKPOINT_FILE)
model.load_state_dict(checkpoint['model_state_dict'])

#extract data 
filters = model.conv1.weight.data.clone()
filters = (filters - filters.min()) / (filters.max() - filters.min())  # normalize to 0-1

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(filters[i].permute(1, 2, 0))  # [3,3,3] → [3,3,3] RGB
    ax.axis('off')
    ax.set_title(f'filter {i}')

plt.suptitle('conv1 learned filters')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'filters.png')
plt.show()