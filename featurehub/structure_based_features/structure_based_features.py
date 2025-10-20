from collections import Counter
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from Bio.PDB import DSSP, PDBParser

from ..sequence_based_features.const import AMINO_ACIDS_3TO1
from .const import *
from .rasa import RelativeASA
from .rasa2acc20 import rasa2acc20


class StructureAnalysis:
    def __init__(self, dssp: str = "mkdssp", round_level: int = 4, rasa_scale: str = "Tien"):
        """利用dssp和一些简单的统计方法计算蛋白质的结构特征.

        Args:
            dssp (str, optional): 可以手动指定一个环境变量或是绝对路径. Defaults to "mkdssp".
            round_level (int, optional): 保留小数点后的位数. Defaults to 4.
        """
        print("init StructureAnalysis")
        self.dssp = dssp
        self.round_level = round_level
        self.rasa_scale = rasa_scale
        self.rasa_dict = RASA_SCALE.get(rasa_scale, None)
        if self.rasa_dict is not None:
            self.rasa_tool = RelativeASA(rasa_scale=rasa_scale)
        else:
            print(f"rasa_scale {rasa_scale} is not supported, use DSSP to calculate the relative ASA")

    def generate_ss3_ss8_acc20(self, input_dssp_df):
        length = input_dssp_df.shape[1]
        ss8 = input_dssp_df.loc["Secondary_structure"].to_numpy()
        ss8_counter = {k: (ss8 == k).sum() for k in SS8_TYPES}
        ss8_fraction = {k: v / length for k, v in ss8_counter.items()}
        ss3 = np.array(["E" if ss in "BE" else "H" if ss in "GHI" else "C" for ss in ss8])
        ss3_counter = {k: (ss3 == k).sum() for k in SS3_TYPES}
        ss3_fraction = {k: v / length for k, v in ss3_counter.items()}
        rasa = input_dssp_df.loc["Relative_ASA"].values
        acc_20 = rasa2acc20(rasa=rasa, length=length)
        return {
            **{f"structure_analysis_ss8_{k}": v for k, v in ss8_counter.items()},
            **{f"structure_analysis_ss8_frac_{k}": round(v, self.round_level) for k, v in ss8_fraction.items()},
            **{f"structure_analysis_ss3_{k}": v for k, v in ss3_counter.items()},
            **{f"structure_analysis_ss3_frac_{k}": round(v, self.round_level) for k, v in ss3_fraction.items()},
            **{f"structure_analysis_acc20_{i}": round(v, self.round_level) for i, v in enumerate(acc_20)},
        }

    def generate_features(self, pdb_path: str, frame: int = 0, target_chains: Optional[Iterable[str]] = None) -> dict:
        """生成特征的主函数.

        Args:
            pdb_path (str): pdb文件的路径.
            frame (int): 使用结构文件中的第X帧. Defaults to 0.
            target_chains (Iterable[str]): 目标的链. Defaults to None.

        Returns:
            dict: 返回一个字典, 包含了三个层次的特征, 分别是结构层次, 链层次, 残基层次.
        """
        model = PDBParser(QUIET=True).get_structure("input_pdb", pdb_path)[frame]
        dssp_df = pd.DataFrame(DSSP(model, pdb_path, dssp=self.dssp).property_dict, index=DSSP_HEADERS)
        for i, res in enumerate(model.get_residues()):
            cur_key = (res.parent.id, res.id)
            if cur_key not in dssp_df:
                dssp_df.insert(loc=i, column=cur_key, value=["NA", AMINO_ACIDS_3TO1.get(res.resname, "X"), "-", *["NA"]*11])
        dssp_df.loc["id"] = range(dssp_df.shape[1])
        dssp_df.replace("NA", np.nan, inplace=True)
        target_chains = target_chains if target_chains is not None else list(set(i[0] for i in dssp_df.keys()))
        if self.rasa_dict is not None:
            dssp_df.loc["Relative_ASA"] = self.rasa_tool.get_rasa(model).round(self.round_level)
        residue_dict = {
            chain_id: {k: dssp_df.loc[k, chain_id].values for k in DSSP_HEADERS} for chain_id in target_chains
        }
        for chain_id in residue_dict:
            residue_dict[chain_id]["Relative_ASA"] = (
                residue_dict[chain_id]["Relative_ASA"].astype(float).round(self.round_level)
            )
        chain_dict = {
            target_chain: self.generate_ss3_ss8_acc20(dssp_df[target_chain]) for target_chain in target_chains
        }
        structure_dict = self.generate_ss3_ss8_acc20(dssp_df)
        return {
            "structure_level": structure_dict,
            "chain_level": chain_dict,
            "residue_level": residue_dict,
        }


if __name__ == "__main__":
    pdb_path = "/home/guolvjun/data/ddg/fixed_final_pdbs/1A2P.pdb"
    sa = StructureAnalysis()
    print(sa.generate_features(pdb_path))
