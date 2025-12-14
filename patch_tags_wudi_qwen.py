import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

merged_dir = "models/qwen_wudi_ja_zh_v2"
ja_dir     = "models/qwen_en_ja_ft/checkpoint-50000"
out_dir    = "models/qwen_wudi_v2_tagpatched"

tok = AutoTokenizer.from_pretrained(merged_dir, use_fast=True)
id_ja = tok.encode("<ja>", add_special_tokens=False)[0]

merged = AutoModelForCausalLM.from_pretrained(merged_dir, device_map="cpu", dtype=torch.float32)
ja     = AutoModelForCausalLM.from_pretrained(ja_dir,     device_map="cpu", dtype=torch.float32)

with torch.no_grad():
    # Patch input embedding row
    merged.get_input_embeddings().weight[id_ja].copy_(
        ja.get_input_embeddings().weight[id_ja]
    )

    out_emb_m = merged.get_output_embeddings()
    out_emb_j = ja.get_output_embeddings()
    if out_emb_m is not None and out_emb_j is not None:
        out_emb_m.weight[id_ja].copy_(out_emb_j.weight[id_ja])

merged.save_pretrained(out_dir)
tok.save_pretrained(out_dir)
print("Saved:", out_dir)
