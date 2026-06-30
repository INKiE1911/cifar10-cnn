import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from src.model import MyNet
from src.config import DATA_DIR, BATCH_SIZE, NORM_MEAN, NORM_STD, CHECKPOINT_FILE, NUM_CLASSES

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN,   # CIFAR-10 channel means
                         NORM_STD),  # channel stds
])

testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE
)

model = MyNet()

#loading checkpoint
checkpoint = torch.load(CHECKPOINT_FILE)
model.load_state_dict(checkpoint['model_state_dict'])



correct = 0
total = 0
model.eval()
#overall test accuracy 
with torch.no_grad():
    for images, labels in testloader:
        out = model(images)
        _, predicted = torch.max(out, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"test accuracy: {100 * correct / total:.2f}%")

#per_class accuracy
class_correct = [0] * NUM_CLASSES
class_total = [0] * NUM_CLASSES

with torch.no_grad():
    for images, labels in testloader:
        out = model(images)
        _, predicted = torch.max(out, 1)
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1

for i in range(NUM_CLASSES):
    print(f"{testset.classes[i]:>12}: {100 * class_correct[i] / class_total[i]:.1f}%")