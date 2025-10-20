import pandas as pd
import torch
from tqdm import tqdm
from featurehub.sequence_based_features.base_classes import AminoAcidDescriptor
from featurehub.sequence_based_features.intrinsic_solubility import residues_intrinsic_solubility
from featurehub import ESMPSSM
from .utils import *
import numpy as np
from typing import List, Tuple


class FeatureGenerator:
    def __init__(self, device: str = "auto", batch_size: int = 8) -> None:
        self.aa_descriptor = AminoAcidDescriptor()
        # self.camsol = Camsol()
        
        # 设备选择
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.batch_size = batch_size
        print(f"使用设备: {self.device}")
        
        # 初始化ESM模型并移到指定设备
        self.esm_pssm = ESMPSSM("models/esm2_t33_650M_UR50D")
        if hasattr(self.esm_pssm, 'model'):
            self.esm_pssm.model = self.esm_pssm.model.to(self.device)
            self.esm_pssm.model.eval()  # 设置为评估模式

    def get_seq_desc(self, sequence, target_position, prefix: str = "", window_side_length: int = 3):
        seq_length = len(sequence)
        camsol_res = residues_intrinsic_solubility(sequence)
        return pd.Series(
            {
                f"{prefix}camsol_residue_contribution_{idx}": (camsol_res[i] if 0 <= i < seq_length else 0)
                for idx, i in enumerate(
                    range(
                        target_position - window_side_length,
                        target_position + window_side_length + 1,
                    )
                )
            }
        )

    @torch.no_grad()
    def get_window_pssm(self, input_seq: str, target_position: int, prefix: str = "", window_side_length: int = 3):
        """单个序列的PSSM计算 - 为了向后兼容保留"""
        return self._get_single_window_pssm(input_seq, target_position, prefix, window_side_length)
    
    @torch.no_grad()
    def _get_single_window_pssm(self, input_seq: str, target_position: int, prefix: str = "", window_side_length: int = 3):
        """单个序列的PSSM计算"""
        input_dict = self.esm_pssm.tokenizer(input_seq, return_tensors="pt").to(self.device)
        seq_out = self.esm_pssm.model(**input_dict).logits
        seq_out_softmax = torch.softmax(seq_out, dim=-1)
        
        result = {}
        for i, pos in enumerate(range(target_position - window_side_length, target_position + window_side_length + 1)):
            if 0 <= pos < len(input_seq):
                token_id = self.esm_pssm.tokenizer.convert_tokens_to_ids(input_seq[pos])
                score = seq_out_softmax[0][pos + 1][token_id].item()
            else:
                score = 0
            result[f"{prefix}esm_pssm_{i}"] = score
        
        return pd.Series(result)

    @torch.no_grad()
    def get_batch_window_pssm(self, sequences: List[str], target_positions: List[int], 
                             prefixes: List[str], window_side_length: int = 5) -> List[pd.Series]:
        """批量处理PSSM计算以提高GPU利用率"""
        if len(sequences) == 0:
            return []
        
        results = []
        
        # 分批处理
        for i in tqdm(range(0, len(sequences), self.batch_size)):
            batch_seqs = sequences[i:i + self.batch_size]
            batch_positions = target_positions[i:i + self.batch_size]
            batch_prefixes = prefixes[i:i + self.batch_size]
            
            # Tokenize整个批次
            try:
                # 使用padding来处理不同长度的序列
                inputs = self.esm_pssm.tokenizer(
                    batch_seqs, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=2048  # 最大长度限制
                ).to(self.device)
                
                # 批量推理
                outputs = self.esm_pssm.model(**inputs)
                logits = outputs.logits
                seq_out_softmax = torch.softmax(logits, dim=-1)
                
                # 为每个序列提取窗口特征
                for j, (seq, pos, prefix) in enumerate(zip(batch_seqs, batch_positions, batch_prefixes)):
                    result = {}
                    for k, window_pos in enumerate(range(pos - window_side_length, pos + window_side_length + 1)):
                        if 0 <= window_pos < len(seq):
                            token_id = self.esm_pssm.tokenizer.convert_tokens_to_ids(seq[window_pos])
                            # 注意：tokenizer会在序列开头添加CLS token，所以位置要+1
                            score = seq_out_softmax[j][window_pos + 1][token_id].item()
                        else:
                            score = 0
                        result[f"{prefix}esm_pssm_{k}"] = score
                    
                    results.append(pd.Series(result))
                    
            except Exception as e:
                print(f"批量处理出错，回退到单个处理: {e}")
                # 如果批量处理失败，回退到单个处理
                for seq, pos, prefix in zip(batch_seqs, batch_positions, batch_prefixes):
                    results.append(self._get_single_window_pssm(seq, pos, prefix, window_side_length))
        
        return results

    def make_ddg_features(self, input_df: pd.DataFrame):
        """
        Args:
            input_df (pd.DataFrame): 必须含有ori_seq, mut_seq
        """
        input_df[["ori_aa", "pos_from_0", "mut_aa"]] = input_df.apply(
            lambda x: pd.Series(get_mutcode(x["ori_seq"], x["mut_seq"])), axis=1
        )
        
        print("正在计算氨基酸描述符...")
        tqdm.pandas(desc="ori aa desc")
        ori_aa_desc = input_df["ori_aa"].progress_apply(
            lambda x: pd.Series({f"ori_aa_desc_{k}": v for k, v in self.aa_descriptor.descriptor[x].items()})
        )
        tqdm.pandas(desc="mut aa desc")
        mut_aa_desc = input_df["mut_aa"].progress_apply(
            lambda x: pd.Series({f"mut_aa_desc_{k}": v for k, v in self.aa_descriptor.descriptor[x].items()})
        )

        print("正在计算序列描述符...")
        tqdm.pandas(desc="ori seq desc")
        ori_seq_desc = input_df[["ori_seq", "pos_from_0"]].progress_apply(
            lambda x: self.get_seq_desc(*x, prefix="ori_"), axis=1
        )
        tqdm.pandas(desc="mut seq desc")
        mut_seq_desc = input_df[["mut_seq", "pos_from_0"]].progress_apply(
            lambda x: self.get_seq_desc(*x, prefix="mut_"), axis=1
        )

        print("正在使用GPU批量计算PSSM特征...")
        
        # 批量处理原始序列的PSSM
        ori_sequences = input_df["ori_seq"].tolist()
        ori_positions = input_df["pos_from_0"].tolist()
        ori_prefixes = ["ori_"] * len(ori_sequences)
        
        print(f"处理 {len(ori_sequences)} 个原始序列...")
        ori_window_pssm_list = self.get_batch_window_pssm(
            ori_sequences, ori_positions, ori_prefixes, window_side_length=5
        )
        if len(ori_window_pssm_list) == len(input_df):
            ori_window_pssm = pd.concat(ori_window_pssm_list, axis=1).T
            ori_window_pssm.index = input_df.index
        else:
            # 如果长度不匹配，使用DataFrame构造方式确保索引正确
            ori_window_pssm = pd.DataFrame(ori_window_pssm_list, index=input_df.index)
        
        # 批量处理突变序列的PSSM
        mut_sequences = input_df["mut_seq"].tolist()
        mut_positions = input_df["pos_from_0"].tolist()
        mut_prefixes = ["mut_"] * len(mut_sequences)
        
        print(f"处理 {len(mut_sequences)} 个突变序列...")
        mut_window_pssm_list = self.get_batch_window_pssm(
            mut_sequences, mut_positions, mut_prefixes, window_side_length=5
        )
        if len(mut_window_pssm_list) == len(input_df):
            mut_window_pssm = pd.concat(mut_window_pssm_list, axis=1).T
            mut_window_pssm.index = input_df.index
        else:
            # 如果长度不匹配，使用DataFrame构造方式确保索引正确
            mut_window_pssm = pd.DataFrame(mut_window_pssm_list, index=input_df.index)
        
        print("特征计算完成，正在合并结果...")
        return pd.concat(
            [
                ori_aa_desc,
                mut_aa_desc,
                ori_seq_desc,
                mut_seq_desc,
                ori_window_pssm,
                mut_window_pssm,
            ],
            axis=1,
        )
