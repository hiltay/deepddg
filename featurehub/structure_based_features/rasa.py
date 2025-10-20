import numpy as np
from Bio.PDB.Model import Model
from Bio.PDB.SASA import ShrakeRupley

from ..sequence_based_features.const import AMINO_ACIDS_3TO1
from .const import RASA_SCALE


class RelativeASA:
    def __init__(self, rasa_scale: str = "Tien") -> None:
        """
        Args:
            rasa_scale (str, optional): _description_. Defaults to "Tien". choose from ["Tien", "Rose", "Zhengfeng", "DSSP"]. if "DSSP" is chosen, the DSSP will be used to calculate the relative ASA
        """
        self.scale_dict = RASA_SCALE[rasa_scale]

    def get_rasa(self, input_model: Model) -> np.ndarray:
        """use ShrakeRupley to estimate the relative ASA of each residue, then scale them by specific scale dict

        Args:
            pdb_file (str): _description_

        Returns:
            list: _description
        """
        shrake_rupley = ShrakeRupley()
        shrake_rupley.compute(input_model, level="R")
        unexpected = []
        for i in input_model.get_residues():
            if i.resname not in AMINO_ACIDS_3TO1:
                unexpected.append(
                    f"{i.parent.id}:{''.join(str(j).strip() for j in i.id)}_{i.resname}"
                    )
        if unexpected:
            print("Warning: ", ", ".join(unexpected), "are not supported. \
                they will be removed from the calculation. \
                this may cause some unexpected results. \
                You might want to remove these residues\
                before using this function.")
        scaled_rasa = np.array(
            [
                res.sasa / self.scale_dict[AMINO_ACIDS_3TO1[res.resname]]
                for res in input_model.get_residues() 
                if res.resname in AMINO_ACIDS_3TO1
            ]
        ).astype(float)
        return scaled_rasa
