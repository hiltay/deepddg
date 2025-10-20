from transformers import EsmForMaskedLM

model = EsmForMaskedLM.from_pretrained("models/esm2_t33_650M_UR50D")
print(model)