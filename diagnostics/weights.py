import torch
from matplotlib import pyplot as plt
from src.model import MyNet
from src.config import CHECKPOINT_FILE, OUTPUT_DIR

model = MyNet()
checkpoint = torch.load(CHECKPOINT_FILE)
model.load_state_dict(checkpoint['model_state_dict'])

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, name, param in zip(axes,
    ['conv1', 'conv2', 'fc1', 'fc2'],
    [model.conv1.weight, model.conv2.weight, model.fc1.weight, model.fc2.weight]):
    ax.hist(param.data.view(-1).detach(), 50)
    ax.set_title(f'{name} weights')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'weights.png')
plt.show()