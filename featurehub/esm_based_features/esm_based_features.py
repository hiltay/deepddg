from typing import Iterable, Optional
import os
from tqdm import tqdm
import torch
import importlib

from .const import *


class InitESM:
    def __init__(
        self,
        model_type, 
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        device: str = "cpu",
        round_level: int = 4,
    ):
        """利用EsmForMaskedLM进行类似PSSM的预测
        Args:
            model_type (EsmPreTrainedModel): 模型类型. choose from [EsmForMaskedLM, EsmModel]
            model_name (str): 模型名称. Defaults to "facebook/esm2_t6_8M_UR50D".
            device (str): 期望使用的设备, 序列数量比较少且逐个预测的时候, 用cpu其实更快. Defaults to "cpu".
            round_level (int): 保留小数点后的位数. Defaults to 4.
        """
        assert (
            model_name in AVAILABLE_ESM_MODELS
        ), f"Model name {model_name} is not available. Please choose from {AVAILABLE_ESM_MODELS}"
        self.model_name = model_name
        self.round_level = round_level
        print("Loading ESM model...")
        try:
            from transformers import AutoTokenizer
            self.model = model_type.from_pretrained(model_name).to(device)
        except OSError:
            import transformers
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            importlib.reload(transformers)
            self.model = model_type.from_pretrained(model_name).to(device)
        print("Loading ESM tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer_vocab = self.tokenizer.get_vocab()
        self.tokenizer_reversed_vocab = {v: k for k, v in self.tokenizer.get_vocab().items()}
        print("Init done.")


class ESMPSSM(InitESM):
    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        device: str = "cpu",
        round_level: int = 8,
    ):
        from transformers import EsmForMaskedLM
        super().__init__(
            model_type=EsmForMaskedLM,
            model_name=model_name,
            device=device,
            round_level=round_level,
        )
        self.mask_token = self.tokenizer.mask_token

    def mask_seq(self, input_seq: str, target_position: int):
        """将序列中的某个位置用mask_token替换
        Args:
            input_seq (str): _description_
            target_position (int): _description_
        """
        return f"{input_seq[:target_position]}{self.mask_token}{input_seq[target_position+1:]}"

    @torch.inference_mode()
    def make_pssm(
        self,
        input_seq: str,
        target_positions: Optional[Iterable[int]] = None,
        target_amino_acids: Optional[Iterable[str]] = None,
    ):
        """利用ESMForMaskedLM进行类似PSSM的预测, 获取某个位置的残基分布概率, 或是某些特定残基的概率
        实际上因为ESM的一些特性, 不推荐在做inference的时候使用batch, 因为原理和实测中attention_mask都并不如预期一般工作.
        Args:
            input_seq (str): 输入序列
            target_positions (Iterable[int], optional): 残基序号(从0计数). Defaults to None.
            target_amino_acids (Iterable[str], optional): 目标残基. Defaults to None.
        Returns:
            _type_: Dict[位置(int), Dict[残基(str), 概率(float)]]
        """
        target_tokens = (
            target_amino_acids if target_amino_acids is not None else list(self.tokenizer.get_vocab().keys())
        )
        token_with_ids = [(token, self.tokenizer_vocab[token]) for token in target_tokens]
        target_positions = target_positions if target_positions is not None else range(len(input_seq))
        all_masked_seqs = [self.mask_seq(input_seq, i) for i in target_positions]
        input_dict = self.tokenizer(all_masked_seqs, return_tensors="pt").to(self.model.device)
        output = {}
        with torch.no_grad():
            for input_id, attention_mask, target_position in tqdm(zip(input_dict.input_ids, input_dict.attention_mask, target_positions), total=len(input_dict.input_ids)):
                model_out = self.model(
                    input_ids = input_id.reshape(1, -1),
                    attention_mask = attention_mask.reshape(1, -1),
                ).logits
                proba_matrix = torch.softmax(model_out[0, target_position + 1], dim=-1).cpu()
                output[target_position] = {target_aa: proba_matrix[target_id].item() for target_aa, target_id in token_with_ids}
        return output


class ESMEmbedding(InitESM):
    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        device: str = "cpu",
        round_level: int = 8,
    ):
        from transformers import EsmModel
        super().__init__(
            model_type=EsmModel,
            model_name=model_name,
            device=device,
            round_level=round_level,
        )

    @torch.inference_mode()
    def make_embedding(self, input_seq: str):
        """利用ESMModel对序列进行embedding
        实际上因为ESM的一些特性, 不推荐在做inference的时候使用batch, 因为原理和实测中attention_mask都并不如预期一般工作.
        Args:
            input_seq (str): 输入序列
        """
        input_dict = self.tokenizer(input_seq, return_tensors="pt").to(self.model.device)
        out = self.model(**input_dict)["pooler_output"][0].cpu()
        return {f"{self.model_name}_{i}": round(v.item(), self.round_level) for i, v in enumerate(out)}


if __name__ == "__main__":
    import numpy as np

    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYX"
    seq = "".join(np.random.choice(list(AMINO_ACIDS), 30))
    print(seq)
    esm_embeding = ESMEmbedding()
    seq_embedding = esm_embeding.make_embedding(seq)
    print(seq_embedding.__len__())
    print(seq_embedding)
