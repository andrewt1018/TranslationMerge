# ta_merge_marian.py

import torch
from transformers import MarianMTModel

base_name = "Helsinki-NLP/opus-mt-en-mul"
ja_dir = "models/en_ja_mul_ft"
zh_dir = "models/en_zh_mul_ft"
out_dir = "models/ta_en_ja_zh"

base = MarianMTModel.from_pretrained(base_name).state_dict()
ja   = MarianMTModel.from_pretrained(ja_dir).state_dict()
zh   = MarianMTModel.from_pretrained(zh_dir).state_dict()

print("merging...")

merged = {}
for k in base.keys():
    delta_ja = ja[k] - base[k]
    delta_zh = zh[k] - base[k]
    merged[k] = base[k] + 0.5 * (delta_ja + delta_zh)

model = MarianMTModel.from_pretrained(base_name)
model.load_state_dict(merged)
model.save_pretrained(out_dir)
