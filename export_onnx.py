import torch
import torch.nn as nn
from train_mvp_model import LipReadingGRU
import os

# To bypass the strict shape checking in newer PyTorch ONNX exporters,
# we can use TorchScript to freeze the logic before exporting.
def export_to_onnx(model_path="mvp_lip_model.pth", onnx_path="mvp_lip_model.onnx"):
    input_dim = 84
    hidden_dim = 64
    num_classes = 5
    
    model = LipReadingGRU(input_dim, hidden_dim, num_classes)
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 30, input_dim)

    print("Converting to TorchScript first...")
    scripted_model = torch.jit.script(model)

    print("Exporting TorchScript to ONNX...")
    torch.onnx.export(
        scripted_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"✅ Success! Model exported to: {onnx_path}")

if __name__ == "__main__":
    try:
        export_to_onnx()
    except Exception as e:
        print(f"❌ Final attempt failed: {e}")
