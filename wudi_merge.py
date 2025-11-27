# wudi_merge.py
import argparse
from typing import List, Dict

import torch
import torch.nn as nn
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
        help="List of fine-tuned model dirs (e.g. en_ja_mul_ft en_zh_mul_ft)",
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
        help="Number of Adam steps per linear layer (paper: ~100–300)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Adam learning rate for WUDI",
    )
    return parser.parse_args()


def get_state_dict(model_or_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a MarianMTModel state_dict from a HF name or local dir.
    """
    model = MarianMTModel.from_pretrained(model_or_path)
    return model.state_dict()


def is_linear_weight(key: str, tensor: torch.Tensor) -> bool:
    """
    Heuristic: treat any 2D 'weight' tensor as a linear layer weight.
    This will include attention and FFN projections, which is what we want.
    """
    return key.endswith(".weight") and tensor.ndim == 2


def build_task_vectors(
    base_sd: Dict[str, torch.Tensor],
    expert_sds: List[Dict[str, torch.Tensor]],
    linear_keys: List[str],
) -> Dict[str, torch.Tensor]:
    """
    For each linear layer key, stack task vectors for all experts.

    Returns:
      task_vecs[key]: shape (num_tasks, out_dim, in_dim)
    """
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
    steps: int = 200,
    lr: float = 1e-5,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Memory-efficient WUDI for a single linear layer.

    tau: (T, out_dim, in_dim) task vectors τ_i,l
    Returns: τ_m,l (out_dim, in_dim)
    """
    tau = tau.to(device)
    num_tasks, out_dim, in_dim = tau.shape

    # Weights 1 / ||τ_i||_F^2
    norms = tau.view(num_tasks, -1).norm(dim=1)  # (T,)
    weights = 1.0 / (norms**2 + 1e-12)

    # Precompute G_tau_i = τ_i^T τ_i  (these don't depend on τ_m)
    G_tau = []
    with torch.no_grad():
        for i in range(num_tasks):
            B = tau[i]                        # (out, in)
            G_tau_i = B.transpose(0, 1) @ B   # (in, in)
            G_tau.append(G_tau_i)
    G_tau = torch.stack(G_tau, dim=0)         # (T, in, in)

    # Initialize τ_m as sum of τ_i (as in Algorithm 1)
    tau_m = tau.sum(dim=0).detach().clone()
    tau_m = torch.nn.Parameter(tau_m)

    optimizer = torch.optim.Adam([tau_m], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        loss = torch.zeros((), device=device)

        for i in range(num_tasks):
            # Δ_i = τ_m - τ_i
            delta = tau_m - tau[i]                  # (out, in)
            # G_Δ_i = Δ_i^T Δ_i
            G_delta = delta.transpose(0, 1) @ delta # (in, in)
            # Frobenius inner product: Tr(G_Δ_i G_τ_i) = sum_ij G_Δ_i * G_τ_i
            frob_sq = (G_delta * G_tau[i]).sum()
            loss_i = weights[i] * frob_sq
            loss = loss + loss_i

        loss.backward()
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

    # Sanity: ensure all state_dicts have same keys
    base_keys = set(base_sd.keys())
    for i, sd in enumerate(expert_sds):
        assert set(sd.keys()) == base_keys, f"State dict mismatch for expert {i}"

    # Identify linear-layer weights
    linear_keys = [
        k for k, v in base_sd.items()
        if is_linear_weight(k, v)
    ]
    print(f"Identified {len(linear_keys)} linear weight tensors for WUDI.")

    # Build task vectors τ_i,l for each linear layer
    task_vecs = build_task_vectors(base_sd, expert_sds, linear_keys)

    merged_sd = {}

    # 1) WUDI on all linear layer weights
    for k in linear_keys:
        tau_l = task_vecs[k]  # (T, out, in)
        tau_m_l = wudi_merge_layer(
            tau_l,
            steps=args.steps,
            lr=args.lr,
            device=str(device),
        )
        merged_sd[k] = base_sd[k] + tau_m_l  # W_m,l = W0,l + τ_m,l

    # 2) For all other params (biases, embeddings, layer norms),
    #    use simple task arithmetic: θm = θ0 + average_i(θi - θ0)
    other_keys = [k for k in base_sd.keys() if k not in linear_keys]

    for k in other_keys:
        base_param = base_sd[k]
        merged_param = base_param.clone()
        for sd in expert_sds:
            merged_param += (sd[k] - base_param) / num_tasks
        merged_sd[k] = merged_param

    # Save merged model
    print("Saving merged model to", args.output_dir)
    merged_model = MarianMTModel.from_pretrained(args.base_model)
    merged_model.load_state_dict(merged_sd)
    merged_model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
