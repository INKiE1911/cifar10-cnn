import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from matplotlib import pyplot as plt
from src.model import MyNet
from src.config import NORM_MEAN, NORM_STD, DATA_DIR, BATCH_SIZE, CHECKPOINT_FILE, OUTPUT_DIR

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

#loading model,no model.eval()
model = MyNet()
checkpoint = torch.load(CHECKPOINT_FILE)
model.load_state_dict(checkpoint['model_state_dict'])

#batch
images, labels = next(iter(trainloader))

#set up loss
criterion = nn.CrossEntropyLoss()

out = model(images)
loss = criterion(out, labels)
loss.backward()

#histograms for gradients

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, name, param in zip(axes, 
    ['conv1', 'conv2', 'fc1', 'fc2'],
    [model.conv1.weight, model.conv2.weight, model.fc1.weight, model.fc2.weight]):
    if param.grad is not None:
        ax.hist(param.grad.view(-1).detach(), 50)
    ax.set_title(f'{name} gradients')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'gradients.png')
plt.show()