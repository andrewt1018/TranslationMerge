# wudi_merge.py (safer v2)
import argparse
from typing import List, Dict

import torch
from transformers import MarianMTModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="Helsinki-NLP/opus-mt-en-mul",
        help="Shared pretrained checkpoint (θ0)",
    )
    parser.add_argument(
        "--expert_dirs",
        nargs="+",
        required=True,
        help="List of fine-tuned model dirs (e.g. models/en_ja_mul_ft models/en_zh_mul_ft)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save the merged model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for WUDI optimization (cuda or cpu)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of Adam steps per linear layer",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-6,  # much smaller than before
        help="Adam learning rate for WUDI",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm for τ_m",
    )
    return parser.parse_args()


def get_state_dict(model_or_path: str) -> Dict[str, torch.Tensor]:
    model = MarianMTModel.from_pretrained(model_or_path)
    return model.state_dict()


def is_linear_weight(key: str, tensor: torch.Tensor) -> bool:
    # any 2D ".weight" tensor is treated as a linear matrix
    return key.endswith(".weight") and tensor.ndim == 2


def is_frozen_param(key: str) -> bool:
    """
    Parameters we DON'T want to touch:
      - shared embeddings (model.shared.weight)
      - encoder/decoder embeddings
      - layer norms
      - final logits bias
    """
    key_lower = key.lower()
    if "embed" in key_lower:
        return True
    if "layer_norm" in key_lower or "layernorm" in key_lower or "ln_" in key_lower:
        return True
    if "final_logits_bias" in key_lower:
        return True
    # shared LM head in Marian
    if "model.shared" in key_lower:
        return True
    return False


def build_task_vectors(
    base_sd: Dict[str, torch.Tensor],
    expert_sds: List[Dict[str, torch.Tensor]],
    linear_keys: List[str],
) -> Dict[str, torch.Tensor]:
    num_tasks = len(expert_sds)
    task_vecs = {}
    for k in linear_keys:
        base_param = base_sd[k]
        diffs = []
        for sd in expert_sds:
            diffs.append(sd[k] - base_param)
        task_vecs[k] = torch.stack(diffs, dim=0)  # (T, out, in)
    return task_vecs


def wudi_merge_layer(
    tau: torch.Tensor,
    steps: int,
    lr: float,
    max_grad_norm: float,
    device: str,
) -> torch.Tensor:
    """
    Memory-efficient, conservative WUDI for a single linear layer.

    tau: (T, out_dim, in_dim) task vectors τ_i,l
    Returns: τ_m,l (out_dim, in_dim)
    """
    tau = tau.to(device)
    num_tasks, out_dim, in_dim = tau.shape

    # weights 1 / ||τ_i||_F^2
    norms = tau.view(num_tasks, -1).norm(dim=1)  # (T,)
    weights = 1.0 / (norms**2 + 1e-12)

    # Precompute G_tau_i = τ_i^T τ_i  (T, in, in)
    G_tau = []
    with torch.no_grad():
        for i in range(num_tasks):
            B = tau[i]                        # (out, in)
            G_tau_i = B.transpose(0, 1) @ B   # (in, in)
            G_tau.append(G_tau_i)
    G_tau = torch.stack(G_tau, dim=0)

    # Initialize τ_m as the MEAN of τ_i (not sum)
    tau_m = tau.mean(dim=0).detach().clone()
    tau_m = torch.nn.Parameter(tau_m)

    optimizer = torch.optim.Adam([tau_m], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        loss = torch.zeros((), device=device)

        for i in range(num_tasks):
            delta = tau_m - tau[i]                  # (out, in)
            G_delta = delta.transpose(0, 1) @ delta # (in, in)
            frob_sq = (G_delta * G_tau[i]).sum()
            loss_i = weights[i] * frob_sq
            loss = loss + loss_i

        loss.backward()
        # clip τ_m gradients
        torch.nn.utils.clip_grad_norm_([tau_m], max_grad_norm)
        optimizer.step()

    return tau_m.detach().cpu()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading base model:", args.base_model)
    base_sd = get_state_dict(args.base_model)

    print("Loading expert models:", args.expert_dirs)
    expert_sds = [get_state_dict(path) for path in args.expert_dirs]
    num_tasks = len(expert_sds)
    assert num_tasks >= 2, "Need at least two experts for merging."

    base_keys = set(base_sd.keys())
    for i, sd in enumerate(expert_sds):
        assert set(sd.keys()) == base_keys, f"State dict mismatch for expert {i}"

    linear_keys = [
        k for k, v in base_sd.items()
        if is_linear_weight(k, v) and not is_frozen_param(k)
    ]
    print(f"Identified {len(linear_keys)} linear weight tensors for WUDI.")

    task_vecs = build_task_vectors(base_sd, expert_sds, linear_keys)

    merged_sd: Dict[str, torch.Tensor] = {}

    # 1) frozen parameters: keep base as-is
    for k, v in base_sd.items():
        if is_frozen_param(k):
            merged_sd[k] = v.clone()

    # 2) WUDI on linear weights (except frozen)
    for k in linear_keys:
        print(f"  WUDI layer: {k}")
        tau_l = task_vecs[k]
        tau_m_l = wudi_merge_layer(
            tau_l,
            steps=args.steps,
            lr=args.lr,
            max_grad_norm=args.max_grad_norm,
            device=str(device),
        )
        merged_sd[k] = base_sd[k] + tau_m_l

    # 3) For other params (non-frozen, non-linear), use small averaged update
    for k, base_param in base_sd.items():
        if k in merged_sd:
            continue
        if is_frozen_param(k):
            continue

        # biases or 1D: very small update
        if base_param.ndim == 1:
            merged_param = base_param.clone()
            for sd in expert_sds:
                merged_param += 0.25 * (sd[k] - base_param) / num_tasks
        else:
            # small average update for leftover weights
            merged_param = base_param.clone()
            for sd in expert_sds:
                merged_param += 0.5 * (sd[k] - base_param) / num_tasks

        merged_sd[k] = merged_param

    print("Saving merged model to", args.output_dir)
    merged_model = MarianMTModel.from_pretrained(args.base_model)
    merged_model.load_state_dict(merged_sd)
    merged_model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
