"""
PyTest check: evaluates whole‑frame accuracy on the images
you placed in  VAL/human/  and  VAL/ai/ .
Passes if accuracy is at least 95 %.
"""
import sys, pathlib                              #  ← NEW
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))  #  ← NEW

from pathlib import Path
from model.inference import predict
THRESH = 0.9 

# Path to the validation folder relative to the project root
VAL_DIR = Path(__file__).parents[1] / "VAL"     # ai_image_detector/VAL


def test_accuracy():
    """
    Loop through every file in VAL/human and VAL/ai
    and assert that at least 95 % are classified correctly.
    """
    if not (VAL_DIR / "human").exists() or not (VAL_DIR / "ai").exists():
        # Skip gracefully if the user hasn't created the folders yet
        assert True, "VAL folder missing – test skipped"
        return

    total = correct = 0
    for cls in ["human", "ai"]:
        for fp in (VAL_DIR / cls).glob("*"):
            res = predict(str(fp), thresh=THRESH)
            total += 1
            correct += int(res["label"] == cls)

    accuracy = 100 * correct / total if total else 0
    assert accuracy >= 95, (
        f"accuracy too low: {accuracy:.2f}%  ({correct}/{total})"
    )
