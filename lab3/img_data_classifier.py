import random
from dataclasses import dataclass

import numpy as np

MAX_CLASSES = 32
MAX_NUM_POINTS = 2**19


@dataclass
class GroundTruthClass:
    lbl: int
    definition_points: np.ndarray


def get_random_pts(w: int, h: int, count_pts: int = None) -> np.ndarray:
    if count_pts is None:
        count_pts = random.randint(1, MAX_NUM_POINTS + 1)
    x_top_idx = np.random.randint(0, w + 1)
    y_top_idx = np.random.randint(0, h + 1)
    x_row = np.random.randint(x_top_idx, size=(count_pts, 1))
    y_row = np.random.randint(y_top_idx, size=(count_pts, 1))
    # Concatenate x_row and y_row to create an array of shape (count_pts, 2)
    return np.hstack((x_row, y_row))

