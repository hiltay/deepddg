import warnings
from .const import (
    AMINO_ACIDS,
    AMINO_ACIDS_MASS,
    ACIDIC_RESIDUES,
    BASIC_RESIDUES,
    LARGE_SIZED_RESIDUES,
    SMALL_SIZED_RESIDUES,
    NONPOLAR_HYDROPHOBIC_RESIDUES,
    POLAR_HYDROPHOBIC_RESIDUES,
    UNCHARGED_POLAR_HYDROPHILIC_RESIDUES,
    CHARGED_POLAR_HYDROPHILIC_RESIDUES,
    HARD_CHARGED_RESIDUES,
    SOFT_CHARGED_RESIDUES,
    AROMATIC_RESIDUES,
    TURN_FORMING_RESIDUES,
    HYDROPHOBICITY_SCALE,
    CHARGE_SCALE,
    SASA_SCALE,
    GEORGE_DSASA_SCALE,
    AMINO_ACIDS_PKA,
    AMINO_ACIDS_PKB,
    AMINO_ACIDS_PKX,
    AMINO_ACIDS_FLEX_SCALE,
    GRAVY_KD_SCALE,
    DIWV_EXT,
    AMINO_ACIDS_HYDR,
    AMINO_ACIDS_ALPHA,
    AMINO_ACIDS_BETA,
    SI_DICT,
)


class AminoAcid:
    def __init__(self, amino_acid, mass_standard: str = "expasy") -> None:
        self.amino_acid = amino_acid
        self.mass_table = AMINO_ACIDS_MASS[mass_standard]
        self.descriptor = {aa: self.describe(aa) for aa in AMINO_ACIDS}

    @property
    def describe(self):
        assert (
            self.amino_acid in AMINO_ACIDS
        ), f"amino acid {self.amino_acid} not in {AMINO_ACIDS}"
        return dict(
            mass=self.mass_table[self.amino_acid],
            acidic=int(self.amino_acid in ACIDIC_RESIDUES),
            basic=int(self.amino_acid in BASIC_RESIDUES),
            large_sized=int(self.amino_acid in LARGE_SIZED_RESIDUES),
            small_sized=int(self.amino_acid in SMALL_SIZED_RESIDUES),
            non_polar_hydrophobic=int(self.amino_acid in NONPOLAR_HYDROPHOBIC_RESIDUES),
            polar_hydrophobic=int(self.amino_acid in POLAR_HYDROPHOBIC_RESIDUES),
            uncharged_polar_hydrophilic=int(
                self.amino_acid in UNCHARGED_POLAR_HYDROPHILIC_RESIDUES
            ),
            charged_polar_hydrophilic=int(
                self.amino_acid in CHARGED_POLAR_HYDROPHILIC_RESIDUES
            ),
            hard_charged=int(self.amino_acid in HARD_CHARGED_RESIDUES),
            soft_charged=int(self.amino_acid in SOFT_CHARGED_RESIDUES),
            aromatic=int(self.amino_acid in AROMATIC_RESIDUES),
            turn_forming=int(self.amino_acid in TURN_FORMING_RESIDUES),
            hydrophobicity_scale=HYDROPHOBICITY_SCALE[self.amino_acid],
            charge_scale=CHARGE_SCALE.get(self.amino_acid, 0),
            sasa_scale=SASA_SCALE[self.amino_acid],
            george_dsasa_scale=GEORGE_DSASA_SCALE[self.amino_acid],
            pka=AMINO_ACIDS_PKA[self.amino_acid],
            pkb=AMINO_ACIDS_PKB[self.amino_acid],
            pkx=AMINO_ACIDS_PKX[self.amino_acid],
            flex_scale=AMINO_ACIDS_FLEX_SCALE[self.amino_acid],
            gravy_kd_scale=GRAVY_KD_SCALE[self.amino_acid],
            **{f"diwv_{aa}": DIWV_EXT[self.amino_acid][aa] for aa in AMINO_ACIDS},
            **{
                f"reverse_diwv_{aa}": DIWV_EXT[aa][self.amino_acid]
                for aa in AMINO_ACIDS
            },
            hydr=AMINO_ACIDS_HYDR[self.amino_acid],
            alpha=AMINO_ACIDS_ALPHA[self.amino_acid],
            beta=AMINO_ACIDS_BETA[self.amino_acid],
            si=SI_DICT[self.amino_acid],
        )
