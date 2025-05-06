import torch
from pathlib import Path
from preprocessing import prepare

MODEL_PATH = Path(__file__).with_name("8best_model.pth")

# --- EfficientNet‑B0 backbone ---
from torchvision.models import efficientnet_b0
net = efficientnet_b0(weights=None)
in_feats = net.classifier[1].in_features       # 1280
net.classifier[1] = torch.nn.Linear(in_feats, 2)

net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
net.eval()

@torch.no_grad()
def predict(path: str, thresh: float = 0.9):
    x = prepare(path)
    p_ai = torch.softmax(net(x), 1)[0, 1].item()
    return {"file": path,
            "ai_probability": p_ai,
            "label": "ai" if p_ai >= thresh else "human"}
