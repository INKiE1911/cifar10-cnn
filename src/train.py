import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from src.model import MyNet
from src.config import NORM_MEAN, NORM_STD, DATA_DIR, BATCH_SIZE, LEARNING_RATE, EPOCHS, CHECKPOINT_FILE

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN,   # CIFAR-10 channel means
                         NORM_STD),  # channel stds
])

trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size = BATCH_SIZE, shuffle = True
)

model = MyNet() #initializing the model
#initializing weights and applying kaiming normalisation
nn.init.kaiming_normal_(model.conv1.weight)
nn.init.kaiming_normal_(model.conv2.weight)
nn.init.kaiming_normal_(model.fc1.weight)
nn.init.kaiming_normal_(model.fc2.weight)

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

#training loop
losses = []
for epoch in range(EPOCHS):
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(trainloader)
    print(f"epoch {epoch}, avg loss: {avg_loss:.4f}")
    losses.append(avg_loss)

#saving checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, CHECKPOINT_FILE)