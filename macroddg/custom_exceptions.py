class PDBResidError(Exception):
    def __init__(self, pdb_path: str, ori_aa: str, mut_pos: str, chain_id: str, *args: object):
        super().__init__(*args)
        print(f"PDBResidError: {chain_id}_{mut_pos} is not {ori_aa} in {pdb_path}")


class ResidNotFoundError(Exception):
    def __init__(self, pdb_path: str, mut_pos: str, chain_id: str, *args: object):
        super().__init__(*args)
        print(f"ResidNotFoundError: {chain_id}_{mut_pos} is not found in {pdb_path}")


class NotAMutationException(Exception):
    def __init__(self, mutcode, *args: object):
        super().__init__(*args)
        print(f"{mutcode} is not a mutation")


class ChainNotInPDBException(Exception):
    def __init__(self, chain_id, pdb_path, *args: object):
        super().__init__(*args)
        print(f"chain {chain_id} not in {pdb_path}")
