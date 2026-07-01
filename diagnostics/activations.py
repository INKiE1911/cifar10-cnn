import torch
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt
from src.model import MyNet
from src.config import NORM_MEAN, NORM_STD, DATA_DIR, BATCH_SIZE, CHECKPOINT_FILE, OUTPUT_DIR

# data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])
testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)

# model
model = MyNet()
checkpoint = torch.load(CHECKPOINT_FILE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

#batch
images, _ = next(iter(testloader)) 
with torch.no_grad():
    x = model.conv1(images)
    x = model.batch_norm_conv1(x)
    a1 = torch.relu(x)               # conv1 activations

    x = model.pool(a1)
    x = model.conv2(x)
    x = model.batch_norm_conv2(x)
    a2 = torch.relu(x)               # conv2 activations

    x = model.pool(a2)
    x = model.flatten(x)
    x = model.batch_norm_fc1(x)
    a3 = torch.relu(model.fc1(x))    # fc1 activations

#dead neurons
# dead neurons — neurons that never fire (always zero)
dead_conv1 = (a1.sum(dim=(0, 2, 3)) == 0).sum().item()
dead_conv2 = (a2.sum(dim=(0, 2, 3)) == 0).sum().item()
dead_fc1 = (a3.sum(dim=0) == 0).sum().item()

print(f"conv1: {dead_conv1} dead")
print(f"conv2: {dead_conv2} dead")
print(f"fc1:   {dead_fc1} dead")

#histograms
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(a1.view(-1), 50)
axes[0].set_title('conv1 activations')

axes[1].hist(a2.view(-1), 50)
axes[1].set_title('conv2 activations')

axes[2].hist(a3.view(-1), 50)
axes[2].set_title('fc1 activations')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'activations.png')
plt.show()