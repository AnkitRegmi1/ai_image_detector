# -----------------------------------------------------------
# Converts any image file to a 1 × 3 × 224 × 224 tensor
# using Resize(256) + CenterCrop(224) + ToTensor + Normalize.
# -----------------------------------------------------------

from pathlib import Path
from PIL import Image
from torchvision import transforms

IMG_SIZE = 224           # final side length fed to EfficientNet‑B0
_SHORTER_SIDE = 256      # keep aspect ratio; shorter side → 256 px

_MEAN = [0.485, 0.456, 0.406]   # ImageNet mean
_STD  = [0.229, 0.224, 0.225]   # ImageNet std

_preproc = transforms.Compose([
    transforms.Resize(_SHORTER_SIDE),          # preserve aspect ratio
    transforms.CenterCrop(IMG_SIZE),           # grab the middle square
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

def prepare(path: str | Path):
    """
    Load an image (RGB) → 4‑D tensor (1×3×224×224) ready for model inference.
    
    Usage:
        x = prepare("/path/to/photo.jpg")   # torch.Size([1, 3, 224, 224])
    """
    img = Image.open(path).convert("RGB")
    return _preproc(img).unsqueeze(0)
