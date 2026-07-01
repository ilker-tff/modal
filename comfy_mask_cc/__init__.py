"""Connected-component mask filter for try-on garment isolation.

Segformer/SAM2 garment masks sometimes carry small disconnected islands
(skin mislabelled as "Pants" on swimwear, stray specks). This node keeps
only the large connected components and drops the islands — fully automatic,
no coordinates. The threshold is RELATIVE to the largest component so it
generalises across garment types and framings (a one-piece dress = 1 big
component kept; a two-piece bikini = 2 comparable components kept; a 1%
skin speck = dropped).
"""

import numpy as np
import torch

try:
    import cv2

    def _label(arr):
        n, lab = cv2.connectedComponents(arr)
        return n, lab
except Exception:  # pragma: no cover - cv2 ships with ComfyUI deps
    from scipy import ndimage

    def _label(arr):
        lab, n = ndimage.label(arr)
        return n + 1, lab


class KeepLargeMaskComponents:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "min_area_ratio": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "filter"
    CATEGORY = "mask"

    def filter(self, mask, min_area_ratio):
        out = []
        for m in mask:
            arr = (m.detach().cpu().numpy() > 0.5).astype(np.uint8)
            n, lab = _label(arr)
            keep = np.zeros_like(arr)
            areas = [(int((lab == i).sum()), i) for i in range(1, n)]
            if areas:
                largest = max(a for a, _ in areas)
                thr = largest * float(min_area_ratio)
                for area, i in areas:
                    if area >= thr:
                        keep[lab == i] = 1
            out.append(torch.from_numpy(keep.astype(np.float32)))
        return (torch.stack(out, dim=0),)


NODE_CLASS_MAPPINGS = {"KeepLargeMaskComponents": KeepLargeMaskComponents}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KeepLargeMaskComponents": "Keep Large Mask Components"
}
