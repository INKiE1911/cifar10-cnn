import onnxruntime
import torch
from src.model import MyNet
from src.config import CHECKPOINT_FILE, ONNX_FILE


model = MyNet()

#loading checkpoint
checkpoint = torch.load(CHECKPOINT_FILE)
model.load_state_dict(checkpoint['model_state_dict'])

#setting model to eval mode
model.eval()
dummy_input = torch.randn(1, 3, 32, 32)

#export
torch.onnx.export(model, (dummy_input,), ONNX_FILE)

# verify
session = onnxruntime.InferenceSession(str(ONNX_FILE))
result = session.run(None, {'x': dummy_input.numpy()})
print(f"ONNX output: {result[0]}")
print("Export verified successfully")