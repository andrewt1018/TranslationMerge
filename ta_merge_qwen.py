#!/usr/bin/env python3
"""
Simple task-vector average baseline for merging two fine-tuned causal LMs:

  theta_merge = theta_base + alpha * 0.5 * ((theta_A - theta_base) + (theta_B - theta_base))

Handles common vocab-size mismatches (embeddings / lm_head) by padding along dim0.
Loads models via AutoModelForCausalLM.from_pretrained (no torch.load on dirs).

Example:
  python merge_avg_qwen.py \
    --base_model_dir Qwen/Qwen2.5-0.5B \
    --expert_a_dir models/qwen_en_ja_ft \
    --expert_b_dir models/qwen_en_zh_ft \
    --out_dir models/qwen_avg_merged \
    --alpha 1.0 \
    --non_linear average \
    --vocab_merge avg_experts
"""

import argparse
import os
from typing import Dict

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_dir", type=str, required=True)
    p.add_argument("--expert_a_dir", type=str, required=True)
    p.add_argument("--expert_b_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--alpha", type=float, default=1.0, help="Scaling on averaged task vector (default 1.0).")

    p.add_argument("--non_linear", type=str, default="average",
                   choices=["base", "average", "expert_a", "expert_b"],
                   help="Policy for non-2D params (biases, layernorm, etc.).")

    p.add_argument("--vocab_merge", type=str, default="avg_experts",
                   choices=["avg_experts", "copy_a", "copy_b", "keep_base"],
                   help="How to merge embeddings/lm_head if vocab sizes differ.")

    p.add_argument("--skip_substr", type=str, default="",
                   help="If non-empty, skip any param name containing this substring (keep base).")
    return p.parse_args()


def load_state_dict_autocausal(model_dir_or_id: str) -> Dict[str, torch.Tensor]:
    model = AutoModelForCausalLM.from_pretrained(
        model_dir_or_id,
        device_map="cpu",
        dtype=torch.float32,
    )
    sd = model.state_dict()
    del model
    return sd


def is_vocab_matrix(name: str) -> bool:
    n = name.lower()
    return ("embed_tokens.weight" in n) or ("lm_head.weight" in n)


def pad_rows_to(t: torch.Tensor, V: int) -> torch.Tensor:
    if t.shape[0] == V:
        return t
    out = torch.zeros((V, t.shape[1]), dtype=t.dtype)
    out[: t.shape[0]] = t
    return out


def merge_vocab_rows(base_t: torch.Tensor, a_t: torch.Tensor, b_t: torch.Tensor, mode: str) -> torch.Tensor:
    assert base_t.ndim == a_t.ndim == b_t.ndim == 2
    if not (base_t.shape[1] == a_t.shape[1] == b_t.shape[1]):
        raise RuntimeError(f"Hidden dim mismatch for vocab matrix: base {base_t.shape}, A {a_t.shape}, B {b_t.shape}")

    V = max(base_t.shape[0], a_t.shape[0], b_t.shape[0])
    base_p = pad_rows_to(base_t, V)
    a_p = pad_rows_to(a_t, V)
    b_p = pad_rows_to(b_t, V)

    if mode == "avg_experts":
        return (0.5 * (a_p + b_p)).to(base_t.dtype)
    if mode == "copy_a":
        return a_p.to(base_t.dtype)
    if mode == "copy_b":
        return b_p.to(base_t.dtype)
    if mode == "keep_base":
        return base_p.to(base_t.dtype)
    raise ValueError(f"Unknown vocab_merge mode: {mode}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading BASE:", args.base_model_dir)
    base_sd = load_state_dict_autocausal(args.base_model_dir)

    print("Loading EXPERT A:", args.expert_a_dir)
    a_sd = load_state_dict_autocausal(args.expert_a_dir)

    print("Loading EXPERT B:", args.expert_b_dir)
    b_sd = load_state_dict_autocausal(args.expert_b_dir)

    keys = sorted(set(base_sd.keys()) & set(a_sd.keys()) & set(b_sd.keys()))
    if not keys:
        raise RuntimeError("No overlapping parameter keys among base/expert A/expert B.")

    merged_sd: Dict[str, torch.Tensor] = {}
    shape_fixed_vocab = 0
    shape_skipped = 0
    kept_base = 0

    for k in keys:
        bt = base_sd[k]
        at = a_sd[k]
        ct = b_sd[k]

        if args.skip_substr and (args.skip_substr in k):
            merged_sd[k] = bt
            kept_base += 1
            continue

        # Shape mismatch handling (mostly vocab)
        if bt.shape != at.shape or bt.shape != ct.shape:
            if bt.ndim == 2 and at.ndim == 2 and ct.ndim == 2 and is_vocab_matrix(k):
                if bt.shape[1] == at.shape[1] == ct.shape[1]:
                    merged_sd[k] = merge_vocab_rows(bt, at, ct, mode=args.vocab_merge)
                    shape_fixed_vocab += 1
                    continue

            merged_sd[k] = bt  
            shape_skipped += 1
            continue

        # Non-2D params: policy
        if bt.ndim != 2:
            if args.non_linear == "base":
                merged_sd[k] = bt
            elif args.non_linear == "average":
                merged_sd[k] = (0.5 * (at.to(torch.float32) + ct.to(torch.float32))).to(bt.dtype)
            elif args.non_linear == "expert_a":
                merged_sd[k] = at
            elif args.non_linear == "expert_b":
                merged_sd[k] = ct
            else:
                merged_sd[k] = bt
            continue

        # 2D weights: task-vector average
        tau_a = at.to(torch.float32) - bt.to(torch.float32)
        tau_b = ct.to(torch.float32) - bt.to(torch.float32)
        tau_avg = 0.5 * (tau_a + tau_b)

        merged = bt.to(torch.float32) + args.alpha * tau_avg
        merged_sd[k] = merged.to(bt.dtype)

    print("---- Summary ----")
    print(f"shape_fixed_vocab={shape_fixed_vocab}")
    print(f"shape_skipped_kept_base={shape_skipped}")
    print(f"kept_base_by_skip={kept_base}")
    print(f"written_keys={len(merged_sd)} / common_keys={len(keys)}")

    print("Saving merged model...")
    config = AutoConfig.from_pretrained(args.base_model_dir)
    merged_model = AutoModelForCausalLM.from_config(config)

    missing, unexpected = merged_model.load_state_dict(merged_sd, strict=False)
    if missing:
        print("WARNING missing keys (first 30):", missing[:30])
    if unexpected:
        print("WARNING unexpected keys (first 30):", unexpected[:30])

    merged_model.save_pretrained(args.out_dir)

    try:
        tok = AutoTokenizer.from_pretrained(args.expert_b_dir, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.base_model_dir, use_fast=True)
    tok.save_pretrained(args.out_dir)

    print("✅ Saved avg-merged model to:", args.out_dir)


if __name__ == "__main__":
    main()
