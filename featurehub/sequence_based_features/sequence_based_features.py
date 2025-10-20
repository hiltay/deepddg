import warnings
from .base_classes import *
from .intrinsic_solubility import intrinsic_solubility, residues_intrinsic_solubility


class SequenceAnalysis(BaseSequenceFeatures, pseAAC):
    """see [https://pubmed.ncbi.nlm.nih.gov/30525483/]"""

    def __init__(
        self, pH: float = 7.0, mass_standard: str = "expasy", w: float = 0.005, l: int = 5, round_level: int = 4
    ):
        """传统方法对序列进行描述

        Args:
            pH (float, optional): 就是pH, 不会有人不知道pH吧, 不会吧. Defaults to 7.0.
            mass_standard (str, optional): 氨基酸分子量标准. Defaults to "expasy". choices: ["expasy", "mascot", "monoisotopic"]
            w (float, optional): pseAAC需要使用的常数之一. Defaults to 0.005.
            l (int, optional): pseAAC需要使用的常数之一. Defaults to 5.
            round_level (int, optional): 保留小数点后的位数. Defaults to 4.
        """
        warnings.warn("featurehub.SequenceAnalysis will be deprecated in the future, use featurehub.Sequence instead", DeprecationWarning)
        BaseSequenceFeatures.__init__(self, pH, mass_standard)
        pseAAC.__init__(self, w, l)
        print("init SequenceAnalysis")
        self.round_level = round_level
    
    def generate_features(self, seq: str):
        protein_anal_ext = ProteinAnalysisExtended(seq)
        seq_length = len(seq)
        counter = self.amino_acid_counter(seq)
        molecular_weight = self.molecular_weight(counter)
        amino_acid_frequency = self.amino_acid_frequency(counter)
        count_nonpolar_hydrophobic_residues = self.count_nonpolar_hydrophobic_residues(counter)
        count_polar_hydrophobic_residues = self.count_polar_hydrophobic_residues(counter)
        count_uncharged_polar_hydrophilic_residues = self.count_uncharged_polar_hydrophilic_residues(counter)
        count_charged_polar_hydrophilic_residues = self.count_charged_polar_hydrophilic_residues(counter)
        count_small_sized_residues = self.count_small_sized_residues(counter)
        count_large_sized_residues = self.count_large_sized_residues(counter)
        count_basic_residues = self.count_basic_residues(counter)
        count_aromatic_residues = self.count_aromatic_residues(counter)
        count_turn_forming_residues = self.count_turn_forming_residues(counter)
        camsol_solubility = float(intrinsic_solubility(seq))
        camsol_residue_contribution = residues_intrinsic_solubility(seq)
        sequence_dict = dict(
            seq_length=seq_length,
            molecular_weight=molecular_weight,
            **counter,
            **amino_acid_frequency,
            # some traditional features
            **self.aaindex_features(seq),
            camsol_solubility=camsol_solubility,
            sequence_complexity=self.sequence_complexity(amino_acid_frequency.values()),
            mass_length_ratio=self.mass_length_ratio(molecular_weight, seq_length),
            fraction_nonpolar_hydrophobic_residues=count_nonpolar_hydrophobic_residues / seq_length,
            count_nonpolar_hydrophobic_residues=count_nonpolar_hydrophobic_residues,
            fraction_polar_hydrophobic_residues=count_polar_hydrophobic_residues / seq_length,
            count_polar_hydrophobic_residues=count_polar_hydrophobic_residues,
            fraction_uncharged_polar_hydrophilic_residues=count_uncharged_polar_hydrophilic_residues / seq_length,
            count_uncharged_polar_hydrophilic_residues=count_uncharged_polar_hydrophilic_residues,
            fraction_charged_polar_hydrophilic_residues=count_charged_polar_hydrophilic_residues / seq_length,
            count_charged_polar_hydrophilic_residues=count_charged_polar_hydrophilic_residues,
            fraction_small_sized_residues=count_small_sized_residues / seq_length,
            count_small_sized_residues=count_small_sized_residues,
            fraction_large_sized_residues=count_large_sized_residues / seq_length,
            count_large_sized_residues=count_large_sized_residues,
            fraction_basic_residues=count_basic_residues / seq_length,
            count_basic_residues=count_basic_residues,
            fraction_aromatic_residues=count_aromatic_residues / seq_length,
            count_aromatic_residues=count_aromatic_residues,
            fraction_turn_forming_residues=count_turn_forming_residues / seq_length,
            alipathis_index=self.alipathis_index(counter),
            net_charge=self.net_charge(seq[0], seq[-1], counter),
            absolute_charge=self.absolute_charge(counter),
            # Features from Biopython ProteinAnalysisExtended
            pa_instability_index=protein_anal_ext.instability_index_extended(),
            pa_flexibility=protein_anal_ext.flexibility_extended(),
            pa_gravy=protein_anal_ext.gravy_extended(),
            pa_isoelectric_point=protein_anal_ext.isoelectric_point(),
            pa_charge_at_pH=protein_anal_ext.charge_at_pH(pH=self.pH),
            **dict(
                zip(
                    ["pa_helix", "pa_turn", "pa_sheet"],
                    protein_anal_ext.secondary_structure_fraction(),
                )
            ),
            **dict(
                zip(
                    ["pa_mec_reduced", "pa_mec_cystines"],
                    protein_anal_ext.molar_extinction_coefficient(),
                )
            ),
            # pseaac features
            **{f"pseaac_{i}": v for i, v in enumerate(self.pseaac(seq))},
        )
        sequence_dict = {k: round(v, self.round_level) for k, v in sequence_dict.items()}
        return {
            "sequence_level": sequence_dict,
            "residue_level": {"camsol_residue_contribution": camsol_residue_contribution},
        }


if __name__ == "__main__":
    import numpy as np

    print(CURDIR)
    input_sequence = "".join(np.random.choice(list(AMINO_ACIDS), 30))
    print(input_sequence)
    seq_sa = SequenceAnalysis(pH=7)
    seq_features = seq_sa.generate_features(seq=input_sequence)
    print(seq_features.__len__())
    print(seq_features)
