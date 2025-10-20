import math
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import IsoelectricPoint
from .base_classes import *
from .intrinsic_solubility import intrinsic_solubility, residues_intrinsic_solubility


class Sequence(ProteinAnalysis):
    sequence: str
    pH: float = 7.0
    mass_standard: str = "expasy",
    pseAAC: dict = {
        "w": 0.005,
        "l": 5,
    }
    def __init__(
        self, 
        sequence: str, 
        pH: float = 7.0, 
        mass_standard: str = "expasy", 
        pseAAC: dict = {
            "w": 0.005,
            "l": 5,
        },
    ):
        """传统方法对序列进行描述

        Args:
            pH (float, optional): 就是pH, 不会有人不知道pH吧, 不会吧. Defaults to 7.0.
            mass_standard (str, optional): 氨基酸分子量标准. Defaults to "expasy". choices: ["expasy", "mascot", "monoisotopic"]
            w (float, optional): pseAAC需要使用的常数之一. Defaults to 0.005.
            l (int, optional): pseAAC需要使用的常数之一. Defaults to 5.
            round_level (int, optional): 保留小数点后的位数. Defaults to 4.
        """
        self.sequence = sequence
        self.pH = pH
        self.mass_table = AMINO_ACIDS_MASS[mass_standard]
        self.pseAAC = pseAAC
        self.aa_counter = {aa: sequence.count(aa) for aa in AMINO_ACIDS}
        self.length = len(sequence)
        self._calculate_features()
        
    def _calculate_features(self):
        # 计算AAIndex特征
        self.aaindex_features = {f"AAIndex_{i}": np.nanmean([AAINDEX_MATRIX[j][i] for j in self.sequence if j in AAINDEX_MATRIX]) for i in range(AAINDEX_LENGTH)}
        
        # 计算氨基酸频率
        self.aa_frac = {f"Freq_{aa}": self.aa_counter.get(aa, 0) / self.length for aa in AMINO_ACIDS}
        
        # 计算序列复杂度
        self.sequence_complexity = -1000 * accuracy_float_sum(i * math.log2(i) for i in self.aa_frac.values() if i > 0)
        
        # 计算分子量
        self.molecular_weight = accuracy_float_sum(self.mass_table[aa] * aa_count for aa, aa_count in self.aa_counter.items())
        
        # 计算质量长度比
        self.mass_length_ratio = self.molecular_weight / self.length
        
        # 计算各种残基计数
        self.count_nonpolar_hydrophobic_residues = sum(self.aa_counter[i] for i in NONPOLAR_HYDROPHOBIC_RESIDUES)
        self.fraction_nonpolar_hydrophobic_residues = self.count_nonpolar_hydrophobic_residues / self.length
        self.count_polar_hydrophobic_residues = sum(self.aa_counter[i] for i in POLAR_HYDROPHOBIC_RESIDUES)
        self.fraction_polar_hydrophobic_residues = self.count_polar_hydrophobic_residues / self.length
        self.count_uncharged_polar_hydrophilic_residues = sum(self.aa_counter[i] for i in UNCHARGED_POLAR_HYDROPHILIC_RESIDUES)
        self.fraction_uncharged_polar_hydrophilic_residues = self.count_uncharged_polar_hydrophilic_residues / self.length
        self.count_charged_polar_hydrophilic_residues = sum(self.aa_counter[i] for i in CHARGED_POLAR_HYDROPHILIC_RESIDUES)
        self.fraction_charged_polar_hydrophilic_residues = self.count_charged_polar_hydrophilic_residues / self.length
        self.count_small_sized_residues = sum(self.aa_counter[i] for i in SMALL_SIZED_RESIDUES)
        self.fraction_small_sized_residues = self.count_small_sized_residues / self.length
        self.count_large_sized_residues = sum(self.aa_counter[i] for i in LARGE_SIZED_RESIDUES)
        self.fraction_large_sized_residues = self.count_large_sized_residues / self.length
        self.count_basic_residues = sum(self.aa_counter[i] for i in BASIC_RESIDUES)
        self.fraction_basic_residues = self.count_basic_residues / self.length
        self.count_aromatic_residues = sum(self.aa_counter[i] for i in AROMATIC_RESIDUES)
        self.fraction_aromatic_residues = self.count_aromatic_residues / self.length
        self.count_turn_forming_residues = sum(self.aa_counter[i] for i in TURN_FORMING_RESIDUES)
        self.fraction_turn_forming_residues = self.count_turn_forming_residues / self.length
        
        # 计算Alipathis指数
        self.alipathis_index = (self.aa_counter["A"] + 2.9 * self.aa_counter["V"] + 3.9 * (self.aa_counter["I"] + self.aa_counter["L"])) / self.length

        # 计算绝对电荷
        self.absolute_charge = abs((self.aa_counter["R"] + self.aa_counter["K"] - self.aa_counter["D"] - self.aa_counter["E"]) / self.length - 0.03)
        self.intrinsic_solubility = float(intrinsic_solubility(self.sequence))
        self.residues_intrinsic_solubility = residues_intrinsic_solubility(self.sequence)
        
    @property
    def pseaac(self):
        w = self.pseAAC.get("w", 0.005)
        l = self.pseAAC.get("l", 5)
        if len(self.sequence) < l:
            print(f"Warning: sequence length {len(self.sequence)} is shorter than l {l}! Use sequence length instead.")
            print("this might cause error in downstream analysis.")
            this_l = len(self.sequence)
        else:
            this_l = l
        AA_freq = np.array([self.sequence.count(i) for i in AMINO_ACIDS])
        norm_freq = (AA_freq - AA_freq.min()) / (AA_freq.max() - AA_freq.min())

        matrix_J = np.zeros([self.length, self.length])
        for i, aa_i in enumerate(self.sequence):
            if aa_i not in PSEAAC_DICT:
                continue
            for j, aa_j in enumerate(self.sequence[i + 1 :]):
                if aa_j not in PSEAAC_DICT:
                    continue
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
            for i in range(self.length - j - 1):
                tao_lst[j] += matrix_J[i, i + j + 1]
            tao_lst[j] /= self.length - j - 1
        denominator = norm_freq.sum() + w * tao_lst.sum()
        res = np.hstack([norm_freq / denominator, w * tao_lst / denominator])
        return {f"pseaac_{i}": v for i, v in enumerate(res)}

    @property
    def flexibility(self):
        total_flex_score = Decimal("0.")
        for resid in range(self.length):
            subseq = self.sequence[max(0, resid - 4) : resid + 5]
            sub_flex_corr_vals = FLEX_CORR_VALS[max(0, 4 - resid) : self.length + 4 - resid]
            sub_flex_scales = (AMINO_ACIDS_FLEX_SCALE.get(cur_aa, 0) for cur_aa in subseq)
            proportion = FLEX_CORR_VALS_SUM / sum(sub_flex_corr_vals)
            cur_flex_score = accuracy_float_sum(
                flex_scale * corr_val * proportion for flex_scale, corr_val in zip(sub_flex_scales, sub_flex_corr_vals)
            )
            total_flex_score += Decimal(str(cur_flex_score))
        return (total_flex_score / self.length).__float__()

    @property
    def gravy(self):
        return accuracy_float_sum(GRAVY_KD_SCALE.get(aa, 0) for aa in self.sequence) / self.length

    @property
    def instability_index(self):
        score = 0.0
        for i in range(self.length - 1):
            _this, _next = self.sequence[i : i + 2]
            dipeptide_value = DIWV_EXT.get(_this, {}).get(_next, 0)
            score += dipeptide_value
        return (10.0 / self.length) * score
    
    @property
    def isoelectric_point(self):
        """Calculate the isoelectric point.

        Uses the module IsoelectricPoint to calculate the pI of a protein.
        """
        ie_point = IsoelectricPoint.IsoelectricPoint(self.sequence, self.aa_counter)
        return ie_point.pi()

    @property
    def net_charge(self):
        """Calculate the charge of a protein at given pH."""
        charge = IsoelectricPoint.IsoelectricPoint(self.sequence, self.aa_counter)
        return charge.charge_at_pH(self.pH)

    @property
    def secondary_structure_fraction(self):
        """Calculate fraction of helix, turn and sheet.

        Returns a list of the fraction of amino acids which tend
        to be in Helix, Turn or Sheet, according to Haimov and Srebnik, 2016;
        Hutchinson and Thornton, 1994; and Kim and Berg, 1993, respectively.

        Amino acids in helix: E, M, A, L, K.
        Amino acids in turn: N, P, G, S, D.
        Amino acids in sheet: V, I, Y, F, W, L, T.

        Note that, prior to v1.82, this method wrongly returned
        (Sheet, Turn, Helix) while claiming to return (Helix, Turn, Sheet).

        Returns a tuple of three floats (Helix, Turn, Sheet).
        """
        helix = sum(self.aa_frac[f"Freq_{r}"] for r in "EMALK")
        turn = sum(self.aa_frac[f"Freq_{r}"] for r in "NPGSD")
        sheet = sum(self.aa_frac[f"Freq_{r}"] for r in "VIYFWLT")

        return {
            "helix": helix,
            "turn": turn,
            "sheet": sheet,
        }

    @property
    def molar_extinction_coefficient(self):
        """Calculate the molar extinction coefficient.

        Calculates the molar extinction coefficient assuming cysteines
        (reduced) and cystines residues (Cys-Cys-bond)
        """
        mec_reduced = self.aa_frac[f"Freq_W"] * 5500 + self.aa_frac[f"Freq_Y"] * 1490
        mec_cystines = mec_reduced + (self.aa_frac[f"Freq_C"] // 2) * 125
        return {
            "mec_reduced": mec_reduced,
            "mec_cystines": mec_cystines,
        }


    def sequence_dict(self):
        return dict(
            length=self.length,
            molecular_weight=self.molecular_weight,
            **self.aa_counter,
            **self.aa_frac,
            **self.aaindex_features,
            sequence_complexity=self.sequence_complexity,
            mass_length_ratio=self.mass_length_ratio,
            fraction_nonpolar_hydrophobic_residues=self.fraction_nonpolar_hydrophobic_residues,
            count_nonpolar_hydrophobic_residues=self.count_nonpolar_hydrophobic_residues,
            fraction_polar_hydrophobic_residues=self.fraction_polar_hydrophobic_residues,
            count_polar_hydrophobic_residues=self.count_polar_hydrophobic_residues,
            fraction_uncharged_polar_hydrophilic_residues=self.fraction_uncharged_polar_hydrophilic_residues,
            count_uncharged_polar_hydrophilic_residues=self.count_uncharged_polar_hydrophilic_residues,
            fraction_charged_polar_hydrophilic_residues=self.fraction_charged_polar_hydrophilic_residues,
            count_charged_polar_hydrophilic_residues=self.count_charged_polar_hydrophilic_residues,
            fraction_small_sized_residues=self.fraction_small_sized_residues,
            count_small_sized_residues=self.count_small_sized_residues,
            fraction_large_sized_residues=self.fraction_large_sized_residues,
            count_large_sized_residues=self.count_large_sized_residues,
            fraction_basic_residues=self.fraction_basic_residues,
            count_basic_residues=self.count_basic_residues,
            fraction_aromatic_residues=self.fraction_aromatic_residues,
            count_aromatic_residues=self.count_aromatic_residues,
            fraction_turn_forming_residues=self.fraction_turn_forming_residues,
            count_turn_forming_residues=self.count_turn_forming_residues,
            alipathis_index=self.alipathis_index,
            net_charge=self.net_charge,
            absolute_charge=self.absolute_charge,
            instability_index=self.instability_index,
            flexibility=self.flexibility,
            gravy=self.gravy,
            isoelectric_point=self.isoelectric_point,
            charge_at_pH=self.net_charge,
            **self.secondary_structure_fraction,
            **self.molar_extinction_coefficient,
            **self.pseaac,
            intrinsic_solubility=self.intrinsic_solubility,
        )
    
    def residues_dict(self):
        return {
            "residues_intrinsic_solubility": self.residues_intrinsic_solubility,
        }