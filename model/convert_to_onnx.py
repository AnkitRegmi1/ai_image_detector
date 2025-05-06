"""
convert_to_onnx.py
------------------
Exports the PyTorch EfficientNet‑B0 (2‑class) weights that live **in the same
folder as this script** to an ONNX file under  browser_extension/onnx_model.onnx.

Run from project root:
    python model/convert_to_onnx.py
"""

import torch, onnx
from torchvision.models import efficientnet_b0
from pathlib import Path

IMG_SIZE = 224
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

# -----------------------------------------------------------
# NEW: always load the weights that sit right next to THIS file
# -----------------------------------------------------------
WEIGHTS = Path(__file__).with_name("8best_model.pth")    # <— added line

net = efficientnet_b0(weights=None)
net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, 2)

net.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))   # <— uses WEIGHTS
net.eval()

out = Path(__file__).parents[1] / "browser_extension/onnx_model.onnx"

torch.onnx.export(
    net,
    dummy,
    out,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}},   # ← changed dyn_axes → dynamic_axes
    opset_version=17,
)

print("✅  ONNX exported ➜", out)
