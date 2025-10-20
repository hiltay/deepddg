import uuid
import warnings
import re
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from Bio.Data.IUPACData import extended_protein_letters
from tempfile import TemporaryDirectory
from macroddg.utils import roughly_fix_pdb
warnings.filterwarnings("ignore")
from dp.launching.typing import BaseModel, Field
from dp.launching.typing import InputFilePath, OutputDirectory
from dp.launching.typing import String
from pydantic import ValidationError, root_validator


class IOOptions(BaseModel):
    output_dir: OutputDirectory = Field(
        default=f"work_dir/output/{uuid.uuid4().hex}"
    )  # default will be override after online


class InputInfo(BaseModel):
    """_summary_"""

    input_pdb: InputFilePath = Field(
        ...,
        max_file_count=1,
        ftypes=["pdb"],
        description="Upload your file. Currently support .pdb.",
    )
    mutations: String = Field(
        format="multi-line",
        description="""
Mutations in the format of 'chain:[wildtype]residue_number[mutant]'. e.g. 'A:R7V' or 'A:V502N'. 
Multiple Mutations should be separated by a comma, space, or newline. e.g. 'A:R7V, A:V502N' / 'A:R7V A:V502N' / 'A:R7V\nA:V502N'
Note: We currently only support single mutations. Multiple mutations will be treated as separate single mutations.
    """,
    )

    @root_validator()
    def check_valid_input(cls, values):
        with TemporaryDirectory() as tmpdir:
            input_pdb = values.get("input_pdb")
            roughly_fix_pdb(input_pdb, f"{tmpdir}/input.pdb")
            input_pdb = f"{tmpdir}/input.pdb"
            mutations = re.split("[, \n]+", values.get("mutations").strip())
            structure = PDBParser(QUIET=True).get_structure("input_pdb", input_pdb)

            errors = []
            
            for mutation in mutations:
                try:
                    chain_id, ori_aa, res_id, new_res = re.match(r"(\S+):([A-Z])(\d+)([A-Z])", mutation).groups()
                except AttributeError:
                    errors.append(f"突变格式不正确: {mutation}")
                    continue
                if chain_id not in structure[0].child_dict:
                    errors.append(f"无效的链ID: {chain_id}. 输入的链ID: {structure[0].child_dict.keys()}")
                    continue
                children_dict = {
                    "".join(str(i).strip() for i in k): v
                    for k, v in structure[0].child_dict[chain_id].child_dict.items()
                }
                if new_res not in extended_protein_letters:
                    errors.append(f"无效的氨基酸: {new_res}. 支持的氨基酸: {extended_protein_letters}")
                if chain_id not in structure[0].child_dict:
                    errors.append(f"无效的链ID: {chain_id}. 输入的链ID: {structure[0].child_dict.keys()}")
                if str(res_id) not in children_dict:
                    errors.append(f"无效的残基ID: {res_id}. 输入的残基ID: {children_dict.keys()}")
                if str(res_id) in children_dict and seq1(children_dict[str(res_id)].resname) != ori_aa:
                    errors.append(f"原氨基酸与输入的不一致: {ori_aa} != {seq1(children_dict[str(res_id)].resname)}")
            
            if errors:
                raise ValidationError(errors, cls)
            
            return values


class GlobalOptions(IOOptions, InputInfo):
    ...
