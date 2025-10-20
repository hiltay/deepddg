from decimal import Decimal
from typing import Iterable

import numpy as np


def cal_charge(input_res: str, input_pH: float, input_pKa: float): # pylint: disable=invalid-name
    charge_rules = {
        "R": 1,
        "K": 1,
        "H": 1,
        "D": -1,
        "E": -1,
        "C": -1,
        "Y": -1,
        "NH": 1,
        "COOH": -1
    }
    charge = charge_rules.get(input_res, 0)
    if charge == 0:
        return 0
    if charge > 0:
        return charge / (1 + np.power(10, (input_pH - input_pKa)))
    else:
        return charge / (1 + np.power(10, (input_pKa - input_pH)))

def accuracy_float_sum(input_list: Iterable[float]):
    return float(sum(Decimal(str(i)) if i is not None else 0 for i in input_list))
