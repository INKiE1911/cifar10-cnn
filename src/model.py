from src.config import NUM_CLASSES
import torch
import torch.nn as nn

KERNEL_SIZE = 3 # 3x3 Window, modest enough for our 32x32 sized image
INPUT_CHANNELS = 3 #R,G,B
CONV1_FILTERS = 32 #number of KERNEL_SIZE x KERNEL_SIZE x INPUT_CHANNEL(RGB) filters, here  3x3x3 = 27 learnable weights
CONV2_FILTERS = 64 
FC1_SIZE = 256 #number of neurons in first linear layer, inputs to it would be CONV2_FILTERS x 8 x 8 , here 8 comes after pooling 2 times 32(CONV1_FILTERS) / 2 then / 2 = 8 therefore 4096 inputs

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(INPUT_CHANNELS , CONV1_FILTERS ,kernel_size=KERNEL_SIZE, padding=1) #padding keeps spatial size same, without it images shrink by 2 pixels because of conv layer
        self.conv2 = nn.Conv2d(CONV1_FILTERS, CONV2_FILTERS, kernel_size=KERNEL_SIZE, padding=1)
        self.flatten = nn.Flatten()
        self.batch_norm_conv1 = nn.BatchNorm2d(CONV1_FILTERS)
        self.batch_norm_conv2 = nn.BatchNorm2d(CONV2_FILTERS)
        self.batch_norm_fc1 = nn.BatchNorm1d(CONV2_FILTERS * 8 * 8)
        self.pool = nn.MaxPool2d(2) # shrinks by 2
        self.fc1 = nn.Linear(CONV2_FILTERS * 8 * 8, FC1_SIZE)
        self.fc2 = nn.Linear(FC1_SIZE, NUM_CLASSES)
    def forward(self,x):
        x = self.conv1(x)
        x = self.batch_norm_conv1(x)
        x = torch.relu(x)
        x = self.pool(x) # (CONV1_FILTERS,32,32) -> (CONV1_FILTERS,16,16)
        x = self.conv2(x) # (CONV1_FILTERS,16,16) -> (CONV2_FILTERS,16,16)
        x = self.batch_norm_conv2(x)
        x = torch.relu(x)
        x = self.pool(x) # (CONV2_FILTERS,16,16) -> (CONV2_FILTERS,8,8)
        x = self.flatten(x) # (CONV2_FILTERS,8,8) -> [4096]
        x = self.batch_norm_fc1(x)
        x = self.fc1(x) #[4096] -> [FC1_SIZE]
        x = torch.relu(x)
        x = self.fc2(x) #[FC1_SIZE] -> [NUM_CLASSES]
        return x 