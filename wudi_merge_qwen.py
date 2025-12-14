#!/usr/bin/env python3
"""
WUDI-merge two fine-tuned Qwen2.5-0.5B causal-LM checkpoints (e.g., en->ja + en->zh) into one merged model.

What it does:
- Loads base + expert A + expert B with AutoModelForCausalLM (CPU by default).
- Gets their state_dicts.
- For matching-shape 2D tensors (linear weights), runs WUDI optimization on task vectors tau = expert - base.
- For vocab matrices with mismatched vocab sizes, merges by padding rows and averaging/copying.
- For other mismatched tensors, keeps base (safe fallback).
- For non-2D params, uses a policy: base / average / expert_a / expert_b.

Usage:
  python wudi_merge_qwen_autocausal.py \
    --base_model_dir Qwen/Qwen2.5-0.5B \
    --expert_a_dir models/qwen_en_ja_ft \
    --expert_b_dir models/qwen_en_zh_ft \
    --out_dir models/qwen_wudi_merged \
    --device cuda \
    --steps 200 --lr 1e-2 --omega 1e-3 \
    --non_linear base \
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

    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])

    # WUDI solver hyperparams
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--omega", type=float, default=1e-3)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Which tensors get WUDI
    p.add_argument(
        "--include_embeddings",
        action="store_true",
        help="If set, include embedding matrices in WUDI (not recommended if vocab differs).",
    )

    # Non-2D params merge policy
    p.add_argument("--non_linear", type=str, default="base",
                   choices=["base", "average", "expert_a", "expert_b"])

    # Skip tensors by substring match
    p.add_argument("--skip_substr", type=str, default="",
                   help="If non-empty, skip params whose name contains this substring.")

    # Vocab mismatch handling
    p.add_argument("--vocab_merge", type=str, default="avg_experts",
                   choices=["avg_experts", "copy_a", "copy_b", "keep_base"],
                   help="How to merge embeddings/lm_head if vocab sizes differ.")

    p.add_argument("--verbose_every", type=int, default=50)
    return p.parse_args()


def torch_dtype(name: str):
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def is_vocab_matrix(name: str) -> bool:
    n = name.lower()
    # Common patterns across many causal LMs (Qwen2.5 should match embed_tokens / lm_head)
    return ("embed_tokens.weight" in n) or ("lm_head.weight" in n)


def is_embedding_weight(name: str) -> bool:
    n = name.lower()
    return ("embed" in n and n.endswith(".weight")) or ("wte" in n and n.endswith(".weight"))


def is_linear_weight(name: str, t: torch.Tensor, include_embeddings: bool) -> bool:
    if t.ndim != 2:
        return False
    if (not include_embeddings) and is_embedding_weight(name):
        return False
    return True


def frob_norm_sq(x: torch.Tensor) -> torch.Tensor:
    return (x * x).sum()


@torch.no_grad()
def safe_to(x: torch.Tensor, device: str, dtype: torch.dtype) -> torch.Tensor:
    return x.to(device=device, dtype=dtype, non_blocking=True)


def load_state_dict_autocausal(model_dir_or_id: str) -> Dict[str, torch.Tensor]:
    """
    Robustly loads a HF causal LM checkpoint directory or hub id by instantiating the model.
    This avoids torch.load(directory) issues and works with sharded + safetensors models.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_dir_or_id,
        torch_dtype=torch.float32,   
        device_map="cpu",            
        low_cpu_mem_usage=False,   
    )
    sd = model.state_dict()
    del model
    return sd

def pad_rows_to(t: torch.Tensor, V: int) -> torch.Tensor:
    """Pad a (V0, D) matrix along dim0 up to V with zeros."""
    if t.shape[0] == V:
        return t
    out = torch.zeros((V, t.shape[1]), dtype=t.dtype)
    out[: t.shape[0]] = t
    return out


def merge_vocab_rows(base_t: torch.Tensor, a_t: torch.Tensor, b_t: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Merge vocab-sized matrices (V, D) when V differs.
    We pad all to Vmax, then choose:
      - avg_experts: 0.5*(a+b)
      - copy_a/copy_b: pick that expert
      - keep_base: keep base (padded)
    """
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

def wudi_merge_one_matrix(
    tau_a: torch.Tensor,
    tau_b: torch.Tensor,
    omega: float,
    steps: int,
    lr: float,
    max_grad_norm: float,
    device: str,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Two-task WUDI objective per 2D tensor:
      sum_i w_i * || (tau_m - tau_i) tau_i^T ||_F^2 + omega * ||tau_m - tau_i||_F^2
    where w_i = 1 / (||tau_i||_F^2 + eps).
    """
    eps = 1e-12

    ta = safe_to(tau_a, device, torch.float32)
    tb = safe_to(tau_b, device, torch.float32)

    wa = 1.0 / (ta.norm(p="fro") ** 2 + eps)
    wb = 1.0 / (tb.norm(p="fro") ** 2 + eps)

    tm = torch.nn.Parameter(0.5 * (ta + tb))
    opt = torch.optim.Adam([tm], lr=lr)

    for s in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)

        da = tm - ta
        db = tm - tb

        Ia = da @ ta.transpose(0, 1)  # (out,out)
        Ib = db @ tb.transpose(0, 1)  # (out,out)

        loss = (
            wa * frob_norm_sq(Ia) + omega * frob_norm_sq(da) +
            wb * frob_norm_sq(Ib) + omega * frob_norm_sq(db)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_([tm], max_grad_norm)
        opt.step()

        if verbose and (s == 1 or s % 50 == 0 or s == steps):
            print(f"      step {s:4d}/{steps}  loss={loss.item():.6e}")

    return tm.detach().cpu()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    out_dtype = torch_dtype(args.dtype)

    print("Loading BASE (AutoModelForCausalLM):", args.base_model_dir)
    base_sd = load_state_dict_autocausal(args.base_model_dir)

    print("Loading EXPERT A (AutoModelForCausalLM):", args.expert_a_dir)
    a_sd = load_state_dict_autocausal(args.expert_a_dir)

    print("Loading EXPERT B (AutoModelForCausalLM):", args.expert_b_dir)
    b_sd = load_state_dict_autocausal(args.expert_b_dir)

    # Work on common keys only
    keys = sorted(set(base_sd.keys()) & set(a_sd.keys()) & set(b_sd.keys()))
    if not keys:
        raise RuntimeError("No overlapping parameter keys among base/expert A/expert B.")

    merged_sd: Dict[str, torch.Tensor] = {}
    linear_merged = 0
    shape_fixed_vocab = 0
    shape_skipped = 0
    skipped = 0

    print(f"Common keys: {len(keys)}")
    print(f"Settings: non_linear={args.non_linear}, include_embeddings={args.include_embeddings}, vocab_merge={args.vocab_merge}")
    if args.skip_substr:
        print(f"skip_substr: '{args.skip_substr}'")

    for k in keys:
        if args.skip_substr and (args.skip_substr in k):
            merged_sd[k] = base_sd[k]
            skipped += 1
            continue

        bt = base_sd[k]
        at = a_sd[k]
        ct = b_sd[k]

        # Shape mismatch handling
        if bt.shape != at.shape or bt.shape != ct.shape:
            if bt.ndim == 2 and at.ndim == 2 and ct.ndim == 2 and is_vocab_matrix(k):
                # Typical mismatch: vocab size (dim0) differs
                if bt.shape[1] == at.shape[1] == ct.shape[1]:
                    print(f"[shape-fix vocab] {k}: base{tuple(bt.shape)} A{tuple(at.shape)} B{tuple(ct.shape)} -> {args.vocab_merge}")
                    merged_sd[k] = merge_vocab_rows(bt, at, ct, mode=args.vocab_merge)
                    shape_fixed_vocab += 1
                    continue

            # Unknown mismatch: safest is to keep base
            print(f"[shape-skip] {k}: base{tuple(bt.shape)} A{tuple(at.shape)} B{tuple(ct.shape)} -> keep base")
            merged_sd[k] = bt
            shape_skipped += 1
            continue

        # Non-linear params policy
        if not is_linear_weight(k, bt, include_embeddings=args.include_embeddings):
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

        # WUDI for 2D tensors
        if linear_merged % max(1, args.verbose_every) == 0:
            print(f"[{linear_merged}] WUDI param: {k} shape={tuple(bt.shape)}")

        tau_a = at.to(torch.float32) - bt.to(torch.float32)
        tau_b = ct.to(torch.float32) - bt.to(torch.float32)

        tau_m = wudi_merge_one_matrix(
            tau_a=tau_a,
            tau_b=tau_b,
            omega=args.omega,
            steps=args.steps,
            lr=args.lr,
            max_grad_norm=args.max_grad_norm,
            device=device,
            verbose=(linear_merged % max(1, args.verbose_every) == 0),
        )

        merged = bt.to(torch.float32) + tau_m.to(torch.float32)
        merged_sd[k] = merged.to(bt.dtype)

        linear_merged += 1

    print("---- Summary ----")
    print(f"linear_merged={linear_merged}")
    print(f"shape_fixed_vocab={shape_fixed_vocab}")
    print(f"shape_skipped_kept_base={shape_skipped}")
    print(f"skipped_by_substr={skipped}")
    print(f"written_keys={len(merged_sd)} / common_keys={len(keys)}")

    # Build merged model from BASE config and load weights
    print("Building merged model from base config and saving...")
    config = AutoConfig.from_pretrained(args.base_model_dir)
    merged_model = AutoModelForCausalLM.from_config(config)

    missing, unexpected = merged_model.load_state_dict(merged_sd, strict=False)
    if missing:
        print("WARNING missing keys (first 30):", missing[:30])
    if unexpected:
        print("WARNING unexpected keys (first 30):", unexpected[:30])

    merged_model.save_pretrained(args.out_dir)

    # Save tokenizer: prefer expert tokenizer (may contain added tags)
    try:
        tok = AutoTokenizer.from_pretrained(args.expert_a_dir, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.base_model_dir, use_fast=True)
    tok.save_pretrained(args.out_dir)

    print(f"Saved WUDI-merged model to: {args.out_dir}")


if __name__ == "__main__":
    main()
