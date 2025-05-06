# model/fine_tune_efficientnet.py
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.optim as optim

# ───────── config ─────────
ROOT       = Path(__file__).parents[1]          # project root
VAL_DIR    = ROOT / "VAL"                       # your mixed human/ai folder
CKPT_IN    = ROOT / "model" / "8best_model.pth" # original good weights
CKPT_OUT   = ROOT / "model" / "fine_tuned_model.pth"
EPOCHS     = 4          # full‑frame touch‑up – keep small
BATCH_SIZE = 16
LR         = 2e-4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ───────── data ───────────
tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
ds   = datasets.ImageFolder(VAL_DIR, tf)
dl   = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# ───────── model ──────────
net = models.efficientnet_b0(weights=None)
net.classifier[1] = nn.Linear(net.classifier[1].in_features, 2)
net.load_state_dict(torch.load(CKPT_IN, map_location="cpu"), strict=True)
net.to(DEVICE)

crit = nn.CrossEntropyLoss()
opt  = optim.AdamW(net.parameters(), lr=LR)

# ───────── train  ──────────
for ep in range(1, EPOCHS+1):
    net.train(); total=correct=0; loss_sum=0.
    for x,y in dl:
        x,y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = net(x); loss = crit(out,y); loss.backward(); opt.step()
        preds = out.argmax(1)
        correct += (preds==y).sum().item(); total += y.size(0)
        loss_sum += loss.item()*y.size(0)
    acc = 100*correct/total
    print(f"Epoch {ep}/{EPOCHS}  loss {loss_sum/total:.4f}  acc {acc:.2f}%")

torch.save(net.state_dict(), CKPT_OUT)
print(f"✅  Fine‑tuned weights saved → {CKPT_OUT}")
