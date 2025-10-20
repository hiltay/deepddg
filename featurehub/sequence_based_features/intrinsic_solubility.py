import numpy as np
from .const import *


def is_alternating_pattern(local_seq):
    return not (
        2 in local_seq
        or local_seq[0] == local_seq[1]
        or not ((local_seq[0] == local_seq[2] == local_seq[4]) and (local_seq[1] == local_seq[3]))
    )
    
def ipat(sequence):
    polar_clf = [AMINO_ACIDS_POLAR_CLF.get(aa, 0) for aa in sequence]
    ipat = np.zeros(len(sequence))
    for i in range(len(polar_clf) - 4):
        window = polar_clf[i:i+5]
        if is_alternating_pattern(window):
            ipat[i:i+5] = 1
    return ipat

def igk(sequence):
    charges = np.array([abs(AMINO_ACIDS_CHARGE.get(aa, 0)) for aa in sequence])
    seq_len = len(sequence)
    positions = np.arange(seq_len)[:, None]
    window_positions = np.arange(-5, 6)[None, :]
    all_positions = positions + window_positions
    mask = (all_positions >= 0) & (all_positions < seq_len)
    all_positions = np.clip(all_positions, 0, seq_len - 1)
    weights = np.exp(-np.power(window_positions, 4) / 200)
    window_charges = charges[all_positions]
    masked_charges = window_charges * mask
    igk_values = np.sum(weights * masked_charges, axis=1)
    return igk_values

def residues_intrinsic_solubility(sequence, norm=True) -> np.ndarray:
    igk_arr = igk(sequence)
    ipat_arr = ipat(sequence)
    si_arr = np.zeros(len(sequence))
    for i in range(len(sequence)):
        si_arr[i] = sum(SI_DICT.get(key, 0) for key in sequence[max(0, i - 3) : i + 4]) / 7
    si_arr = si_arr + A_PAT * ipat_arr + A_GK * igk_arr + 2.17876545
    if norm:
        si_arr = (si_arr - 0.01043561) / 0.7393881
    return si_arr

def intrinsic_solubility(sequence, norm=True) -> float:
    si_arr = residues_intrinsic_solubility(sequence)
    sp_arr = np.zeros(len(si_arr))
    too_low = si_arr < SP_THLOW
    too_high = si_arr > SP_THUP
    sp_arr[too_low] = SP_WLOW * (si_arr[too_low] - SP_THLOW)
    sp_arr[too_high] = SP_WUP * (si_arr[too_high] - SP_THUP)
    return np.mean(sp_arr) / SP_Y + SP_B