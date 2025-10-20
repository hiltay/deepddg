import pandas as pd
from biopandas.pdb import PandasPdb
import json
from Bio.PDB import PDBParser # type: ignore
from featurehub.sequence_based_features.const import AMINO_ACIDS_3TO1
from .custom_exceptions import *


def standardize_resnum(
    in_pdb_path: str,
    out_pdb_path: str,
    resnum_mapping_savepath: str,
    resnum_mapping_reverse_savepath: str,
):
    ppdb = PandasPdb().read_pdb(in_pdb_path)
    df = ppdb.df["ATOM"]
    resnum_mapping = {}
    resnum_mapping_reverse = {}
    resid = 1
    for res_num, insertion in df[["residue_number", "insertion"]].values:
        if f"{res_num}{insertion}" not in resnum_mapping_reverse:
            resnum_mapping[str(resid)] = f"{res_num}::{insertion}"
            resnum_mapping_reverse[f"{res_num}{insertion}"] = str(resid)
            resid += 1
    df["residue_number"] = df[["residue_number", "insertion"]].apply(
        lambda x: resnum_mapping_reverse["".join(map(str, x))], axis=1
    )
    df["insertion"] = ""
    ppdb.df["ATOM"] = df
    ppdb.to_pdb(out_pdb_path)
    json.dump(resnum_mapping, open(resnum_mapping_savepath, "w"))
    json.dump(
        resnum_mapping_reverse,
        open(resnum_mapping_reverse_savepath, "w"),
    )
    return resnum_mapping, resnum_mapping_reverse


def restore_resnum(in_pdb_path: str, out_pdb_path: str, resnum_mapping_path: str):
    out_pdb_path = in_pdb_path
    ppdb = PandasPdb().read_pdb(in_pdb_path)
    df = ppdb.df["ATOM"]
    resnum_mapping = json.load(open(resnum_mapping_path, "r"))
    df[["residue_number", "insertion"]] = df["residue_number"].apply(
        lambda x: pd.Series(resnum_mapping[str(x)].split("::"))
    )
    ppdb.df["ATOM"] = df
    ppdb.to_pdb(out_pdb_path)
    return out_pdb_path


def get_fixed_mutcode(ori_seq, mut_seq):
    for i in range(len(ori_seq)):
        if ori_seq[i] != mut_seq[i]:
            return f"{ori_seq[i]}{i}{mut_seq[i]}"
    return None


def get_mutation_info(input_json):
    pdb_path = input_json["pdb_path"]
    mutcodes = input_json["mutcodes"]
    mut_info = []
    for mutcode in mutcodes:
        chain_id, mutcode = mutcode.split(":")
        ori_aa, mut_pos, mut_aa = mutcode[0], str(mutcode[1:-1]), mutcode[-1]
        if ori_aa == mut_aa:
            raise NotAMutationException(mutcode)
        mut_pos = "".join(mut_pos)
        struct = PDBParser().get_structure("test", pdb_path)
        if chain_id not in struct[0].child_dict.keys():
            raise ChainNotInPDBException(chain_id, pdb_path)
        target_chain = struct[0][chain_id]
        ori_id2aa_dict = {
            "".join(str(i).strip() for i in k): AMINO_ACIDS_3TO1.get(v.resname, "X")
            for k, v in target_chain.child_dict.items()
        }
        ori_seq = "".join(ori_id2aa_dict.values())
        if mut_pos not in ori_id2aa_dict:
            raise ResidNotFoundError(pdb_path, mut_pos)
        if ori_id2aa_dict[mut_pos] != ori_aa:
            raise PDBResidError(pdb_path, ori_aa, mut_pos, chain_id)
        mut_id2aa_dict = ori_id2aa_dict.copy()
        mut_id2aa_dict[mut_pos] = mut_aa
        mut_seq = "".join(mut_id2aa_dict.values())
        f_mutcode = get_fixed_mutcode(ori_seq, mut_seq)
        mut_info.append(
            {
                "pdb_path": pdb_path,
                "chain_id": chain_id,
                "ori_aa": ori_aa,
                "mut_pos": mut_pos,
                "mut_aa": mut_aa,
                "f_mutcode": f_mutcode,
                "ori_seq": ori_seq,
                "mut_seq": mut_seq,
            }
        )
    return mut_info


def get_mutcode(ori_seq, mut_seq):
    for i, (ori_aa, mut_aa) in enumerate(zip(ori_seq, mut_seq)):
        if ori_aa != mut_aa:
            return ori_aa, int(i), mut_aa


def roughly_fix_pdb(in_pdb_path, out_pdb_path):
    with open(in_pdb_path, "r") as f:
        open_file = f.readlines()
    save_file = open(out_pdb_path, "w")
    for line in open_file:
        if line.startswith("ATOM") or line.startswith("TER"):
            if line[17:20] not in AMINO_ACIDS_3TO1:
                line = line[:17] + "UNK" + line[20:]
            save_file.write(line)
    save_file.close()
