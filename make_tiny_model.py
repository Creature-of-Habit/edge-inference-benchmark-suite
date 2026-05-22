# make_tiny_model.py
import torch
import torch.nn as nn

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 2) # Takes 4 numbers, outputs 2 numbers

    def forward(self, x):
        return self.layer(x)

# Create the model instance
model = TinyModel()
model.eval()

# Create dummy input data shape [Batch Size=1, Features=4]
dummy_input = torch.randn(1, 4)

# Export the graph to an ONNX file
torch.onnx.export(
    model, 
    dummy_input, 
    "tiny_model.onnx",
    input_names=["input_layer"],
    output_names=["output_layer"]
)
print("Success! 'tiny_model.onnx' has been generated.")