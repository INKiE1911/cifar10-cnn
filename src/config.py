from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
OUTPUT_DIR = BASE_DIR / 'outputs'

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
NORM_MEAN = (0.4914, 0.4822, 0.4465) #calculated from data
NORM_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10 #airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
CHECKPOINT_FILE = CHECKPOINT_DIR / 'cifar10_cnn.pth'
ONNX_FILE = OUTPUT_DIR / 'cifar10_cnn.onnx'