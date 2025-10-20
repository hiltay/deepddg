from typing import Optional
import numpy as np
from .const import *


def rasa2acc20(rasa: np.ndarray, length: Optional[int] = None):
    length = len(list(rasa)) if length is None else length
    acc_20 = [
        (rasa == 0).sum() / length,
        *[(rasa >= i).sum() / length for i in ACC20_THRESHOLDS[1:]],
    ]
    return acc_20
