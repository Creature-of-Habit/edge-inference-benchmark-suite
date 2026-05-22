# download_model.py
import urllib.request
from pathlib import Path

# Direct link to the official ONNX Model Zoo repository
model_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx"
output_path = Path("tiny_model.onnx")

print("Connecting to GitHub Model Zoo...")
urllib.request.urlretrieve(model_url, output_path)
print("Download Complete! 'tiny_model.onnx' has been saved to your directory.")