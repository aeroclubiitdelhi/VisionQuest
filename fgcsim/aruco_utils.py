"""ArUco helpers shared by the world generator and the renderer."""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import cv2
import numpy as np

PAPER_WHITE = 238
INK_BLACK = 28
MARGIN_FRAC = 0.25   # white paper around the black square, as a fraction of its side


def get_dictionary(name: str):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary {name!r}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


@lru_cache(maxsize=None)
def marker_patch(dict_name: str, marker_id: int, cell_px: int = 16) -> Tuple[np.ndarray, int]:
    """Grayscale printed-marker image with a white paper margin.

    Returns (patch, side_px): patch is square; side_px is the width in pixels of
    the black square inside it (this is what the marker's physical size refers to).
    """
    d = get_dictionary(dict_name)
    cells = d.markerSize + 2
    side = cells * cell_px
    img = cv2.aruco.generateImageMarker(d, int(marker_id), side, borderBits=1)
    img = np.where(img > 127, PAPER_WHITE, INK_BLACK).astype(np.uint8)
    m = int(round(MARGIN_FRAC * side))
    patch = cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_CONSTANT, value=PAPER_WHITE)
    patch.setflags(write=False)
    return patch, side


# Wrong-dictionary IDs that a DICT_4X4_50 detector (almost) never mis-reads as a valid ID
# (0-34): at most 2 valid mis-reads in ~2,600 small, tilted, blurred, noisy, JPEG-compressed
# views. Produced by tools/find_safe_fake_ids.py. A fixed table keeps worlds identical on
# every machine.
SAFE_FAKE_IDS = {
    "DICT_5X5_50": (11, 21, 22, 27, 33, 36, 37, 43),
    "DICT_6X6_50": (5, 18, 23),
    "DICT_APRILTAG_36h11": (1, 13, 14, 19, 27, 41),
}
