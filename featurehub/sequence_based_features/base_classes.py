import warnings

warnings.filterwarnings("ignore")
import math
from collections import Counter
from decimal import Decimal
from typing import Dict, Iterable

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from .const import *
from .utils import *


class BaseSequenceFeatures:
    def __init__(self, pH: float = 7.0, mass_standard: str = "expasy"):
        print("init BaseSequenceFeatures")
        self.pH = pH
        self.mass_table = AMINO_ACIDS_MASS[mass_standard]

    def amino_acid_counter(self, sequence: str):
        return {aa: sequence.count(aa) for aa in AMINO_ACIDS}

    def aaindex_features(self, sequence: str):
        return {
            f"AAIndex_{i}": np.nanmean([AAINDEX_MATRIX[j][i] for j in sequence])
            for i in range(AAINDEX_LENGTH)
        }

    def amino_acid_frequency(self, counter: Counter):
        seq_length = accuracy_float_sum(counter.values())
        return {
            f"Freq_{aa}": counter.get(aa) / seq_length if aa in counter else 0
            for aa in AMINO_ACIDS
        }

    def sequence_complexity(self, amino_acid_frequency: Iterable[float]):
        return -1000 * accuracy_float_sum(
            i * math.log2(i) for i in amino_acid_frequency if i > 0
        )

    def molecular_weight(self, counter: Counter):
        return accuracy_float_sum(
            self.mass_table[aa] * aa_count for aa, aa_count in counter.items()
        )

    def mass_length_ratio(self, molecular_weight: float, seq_length: int):
        return molecular_weight / seq_length

    def count_nonpolar_hydrophobic_residues(self, counter: Counter):
        return sum(counter[i] for i in NONPOLAR_HYDROPHOBIC_RESIDUES)

    def count_polar_hydrophobic_residues(self, counter: Counter):
        return sum(counter[i] for i in POLAR_HYDROPHOBIC_RESIDUES)

    def count_uncharged_polar_hydrophilic_residues(self, counter: Counter):
        return sum(counter[i] for i in UNCHARGED_POLAR_HYDROPHILIC_RESIDUES)

    def count_charged_polar_hydrophilic_residues(self, counter: Counter):
        return sum(counter[i] for i in CHARGED_POLAR_HYDROPHILIC_RESIDUES)

    def count_small_sized_residues(self, counter: Counter):
        return sum(counter[i] for i in SMALL_SIZED_RESIDUES)

    def count_large_sized_residues(self, counter: Counter):
        return sum(counter[i] for i in LARGE_SIZED_RESIDUES)

    def count_basic_residues(self, counter: Counter):
        return sum(counter[i] for i in BASIC_RESIDUES)

    def count_aromatic_residues(self, counter: Counter):
        return sum(counter[i] for i in AROMATIC_RESIDUES)

    def count_turn_forming_residues(self, counter: Counter):
        return sum(counter[i] for i in TURN_FORMING_RESIDUES)

    def alipathis_index(self, counter: Counter):
        return (
            counter["A"] + 2.9 * counter["V"] + 3.9 * (counter["I"] + counter["L"])
        ) / sum(counter.values())

    def net_charge(self, first_res: str, last_res: str, counter: Counter):
        return sum(
            [
                cal_charge("COOH", self.pH, AMINO_ACIDS_PKA[last_res]),
                cal_charge("NH", self.pH, AMINO_ACIDS_PKB[first_res]),
                *[
                    cal_charge(aa, self.pH, AMINO_ACIDS_PKX[aa]) * aa_count
                    for aa, aa_count in counter.items()
                    if aa_count > 0
                ],
            ]
        )

    def absolute_charge(self, counter: Counter):
        return abs(
            (counter["R"] + counter["K"] - counter["D"] - counter["E"])
            / sum(counter.values())
            - 0.03
        )


class pseAAC:
    """see [http://www.csbio.sjtu.edu.cn/bioinf/PseAAC/PseAAReadme.htm]"""

    def __init__(self, w: float = 0.05, l: int = 5) -> None:
        print("init pseAAC")
        self.w = w
        self.l = max(l, 0)

    def pseaac(self, seq: str):
        if len(seq) < self.l:
            print(
                f"Warning: sequence length {len(seq)} is shorter than l {self.l}! Use sequence length instead."
            )
            print("this might cause error in downstream analysis.")
            this_l = len(seq)
        else:
            this_l = self.l
        seq_length = len(seq)
        AA_freq = np.array([seq.count(i) for i in AMINO_ACIDS])
        norm_freq = (AA_freq - AA_freq.min()) / (AA_freq.max() - AA_freq.min())

        matrix_J = np.zeros([seq_length, seq_length])
        for i, aa_i in enumerate(seq):
            for j, aa_j in enumerate(seq[i + 1 :]):
                matrix_J[i, i + 1 + j] = (
                    1
                    / 3
                    * (
                        math.pow((PSEAAC_DICT[aa_j][0] - PSEAAC_DICT[aa_i][0]), 2)
                        + math.pow((PSEAAC_DICT[aa_j][1] - PSEAAC_DICT[aa_i][1]), 2)
                        + math.pow((PSEAAC_DICT[aa_j][2] - PSEAAC_DICT[aa_i][2]), 2)
                    )
                )
        tao_lst = np.zeros(this_l)
        for j in range(this_l):
            for i in range(seq_length - j - 1):
                tao_lst[j] += matrix_J[i, i + j + 1]
            tao_lst[j] /= seq_length - j - 1
        denominator = norm_freq.sum() + self.w * tao_lst.sum()
        res = np.hstack([norm_freq / denominator, self.w * tao_lst / denominator])
        return res


class ProteinAnalysisExtended(ProteinAnalysis):
    def flexibility_extended(self):
        seq_length = len(self.sequence)
        total_flex_score = Decimal("0.")
        for resid in range(seq_length):
            subseq = self.sequence[max(0, resid - 4) : resid + 5]
            sub_flex_corr_vals = FLEX_CORR_VALS[
                max(0, 4 - resid) : seq_length + 4 - resid
            ]
            sub_flex_scales = (AMINO_ACIDS_FLEX_SCALE[cur_aa] for cur_aa in subseq)
            proportion = FLEX_CORR_VALS_SUM / sum(sub_flex_corr_vals)
            cur_flex_score = accuracy_float_sum(
                flex_scale * corr_val * proportion
                for flex_scale, corr_val in zip(sub_flex_scales, sub_flex_corr_vals)
            )
            total_flex_score += Decimal(str(cur_flex_score))
        return (total_flex_score / seq_length).__float__()

    def gravy_extended(self):
        return (
            accuracy_float_sum(GRAVY_KD_SCALE[aa] for aa in self.sequence) / self.length
        )

    def instability_index_extended(self):
        score = 0.0
        for i in range(self.length - 1):
            this, next = self.sequence[i : i + 2]
            dipeptide_value = DIWV_EXT[this][next]
            score += dipeptide_value
        return (10.0 / self.length) * score


class AminoAcidDescriptor:
    def __init__(self, mass_standard: str = "expasy") -> None:
        self.mass_table = AMINO_ACIDS_MASS[mass_standard]
        self.descriptor = {aa: self.describe(aa) for aa in AMINO_ACIDS}

    def describe(self, amino_acid):
        assert (
            amino_acid in AMINO_ACIDS
        ), f"amino acid {amino_acid} not in {AMINO_ACIDS}"
        return dict(
            mass=self.mass_table[amino_acid],
            acidic=int(amino_acid in ACIDIC_RESIDUES),
            basic=int(amino_acid in BASIC_RESIDUES),
            large_sized=int(amino_acid in LARGE_SIZED_RESIDUES),
            small_sized=int(amino_acid in SMALL_SIZED_RESIDUES),
            non_polar_hydrophobic=int(amino_acid in NONPOLAR_HYDROPHOBIC_RESIDUES),
            polar_hydrophobic=int(amino_acid in POLAR_HYDROPHOBIC_RESIDUES),
            uncharged_polar_hydrophilic=int(
                amino_acid in UNCHARGED_POLAR_HYDROPHILIC_RESIDUES
            ),
            charged_polar_hydrophilic=int(
                amino_acid in CHARGED_POLAR_HYDROPHILIC_RESIDUES
            ),
            hard_charged=int(amino_acid in HARD_CHARGED_RESIDUES),
            soft_charged=int(amino_acid in SOFT_CHARGED_RESIDUES),
            aromatic=int(amino_acid in AROMATIC_RESIDUES),
            turn_forming=int(amino_acid in TURN_FORMING_RESIDUES),
            hydrophobicity_scale=HYDROPHOBICITY_SCALE[amino_acid],
            charge_scale=CHARGE_SCALE.get(amino_acid, 0),
            sasa_scale=SASA_SCALE[amino_acid],
            george_dsasa_scale=GEORGE_DSASA_SCALE[amino_acid],
            pka=AMINO_ACIDS_PKA[amino_acid],
            pkb=AMINO_ACIDS_PKB[amino_acid],
            pkx=AMINO_ACIDS_PKX[amino_acid],
            flex_scale=AMINO_ACIDS_FLEX_SCALE[amino_acid],
            gravy_kd_scale=GRAVY_KD_SCALE[amino_acid],
            **{f"diwv_{aa}": DIWV_EXT[amino_acid][aa] for aa in AMINO_ACIDS},
            **{f"reverse_diwv_{aa}": DIWV_EXT[aa][amino_acid] for aa in AMINO_ACIDS},
            hydr=AMINO_ACIDS_HYDR[amino_acid],
            alpha=AMINO_ACIDS_ALPHA[amino_acid],
            beta=AMINO_ACIDS_BETA[amino_acid],
            si=SI_DICT[amino_acid],
        )
