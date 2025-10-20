import warnings

warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser # type: ignore
from featurehub.sequence_based_features.const import AMINO_ACIDS_1TO3
from pdbfixer import PDBFixer
from openmm.app.pdbfile import PDBFile
from autogluon.tabular import TabularPredictor
import pandas as pd
import os
import json
from pathlib import Path
from .utils import *
from .feature_generator import FeatureGenerator


class MacroDDG:
    def __init__(
        self,
        input_json_filepath: str,
        output_dir: str,
        model_ckpt: str,
        thread: int = os.cpu_count(),
    ):
        self.feature_generator = FeatureGenerator()
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.input_df = pd.DataFrame(get_mutation_info(json.load(open(input_json_filepath))))
        self.ddg_model = TabularPredictor.load(model_ckpt, require_py_version_match=False)
        self.thread = thread
        pdb_path = self.input_df["pdb_path"].values[0]
        self.pdb_name = Path(pdb_path).stem
        self.ori_pdb_path = str(Path(output_dir, f"{self.pdb_name}.pdb"))
        self.resnum_mapping_path = f"{self.output_dir}/resnum_mapping.json"
        self.resnum_mapping_reverse_path = f"{self.output_dir}/resnum_mapping_reverse.json"
        self.resnum_mapping, self.resnum_mapping_reverse = standardize_resnum(
            pdb_path,
            self.ori_pdb_path,
            self.resnum_mapping_path,
            self.resnum_mapping_reverse_path,
        )
        roughly_fix_pdb(self.ori_pdb_path, self.ori_pdb_path)

    def run(self):
        feature_df = self.feature_generator.make_ddg_features(self.input_df)
        self.input_df["mutcode"] = self.input_df[["chain_id", "ori_aa", "mut_pos", "mut_aa"]].apply(
            lambda x: f"{x['chain_id']}:{x['ori_aa']}{x['mut_pos']}{x['mut_aa']}",
            axis=1,
        )
        self.input_df[["chain_id", "ori_aa", "mut_aa", "mut_pos"]].apply(lambda x: self.get_mutated_pdb(*x), axis=1)
        pred = self.ddg_model.predict(feature_df)
        self.input_df["pdb_path"] = self.input_df.apply(
            lambda x: Path(
                self.output_dir,
                f"{Path(x['pdb_path']).stem}_{x['ori_aa']}_{x['mut_pos']}_{x['mut_aa']}.pdb",
            ).as_posix(),
            axis=1,
        )
        out = pd.concat(
            [
                self.input_df.drop(columns=["ori_seq", "mut_seq", "f_mutcode", "pos_from_0"]),
                pred,
            ],
            axis=1,
        ).copy()
        out.rename(columns={"label": "ddg"}, inplace=True)
        out = out.to_dict("records")
        json.dump(out, open(Path(self.output_dir, "output.json"), "w"))
        restore_resnum(self.ori_pdb_path, self.ori_pdb_path, self.resnum_mapping_path)
        return out

    def get_mutated_pdb(self, chain_id, ori_aa, mut_aa, mut_pos):
        # 检查输入文件
        structure = PDBParser(QUIET=True).get_structure("input_pdb", self.ori_pdb_path)
        cur_child_dict = {"".join(str(i).strip() for i in k): v for k, v in structure[0][chain_id].child_dict.items()}
        mut_pos = str(mut_pos)
        standard_mut_pos = self.resnum_mapping_reverse[mut_pos]
        if standard_mut_pos not in cur_child_dict:
            raise ResidNotFoundError(self.ori_pdb_path, mut_pos, chain_id)
        if cur_child_dict[standard_mut_pos].get_resname() != AMINO_ACIDS_1TO3[ori_aa]:
            raise PDBResidError(self.ori_pdb_path, ori_aa, mut_pos, chain_id)
        # 获得突变后的pdb
        letter3_mutcode = f"{AMINO_ACIDS_1TO3[ori_aa]}-{standard_mut_pos}-{AMINO_ACIDS_1TO3[mut_aa]}"
        mut_pdb_path = Path(self.output_dir, f"{self.pdb_name}_{ori_aa}_{mut_pos}_{mut_aa}.pdb").as_posix()
        fixer = PDBFixer(filename=self.ori_pdb_path)
        fixer.applyMutations([letter3_mutcode], chain_id)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(True)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        PDBFile.writeFile(fixer.topology, fixer.positions, open(mut_pdb_path, "w"), keepIds=True)
        restore_resnum(mut_pdb_path, mut_pdb_path, self.resnum_mapping_path)
        fixer = PDBFixer(filename=mut_pdb_path)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(True)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        PDBFile.writeFile(fixer.topology, fixer.positions, open(mut_pdb_path, "w"), keepIds=True)
        assert Path(mut_pdb_path).exists(), "something wrong when applying mutation"
