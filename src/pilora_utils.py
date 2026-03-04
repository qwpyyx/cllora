# coding=utf-8
"""
PiLoRA utilities for your Federated UIE-LoRA framework.

What this file provides:
1) Reference (previous-task) LoRA snapshot:
   - extract_pilora_ref_from_model(model)
   - save_pilora_ref(adapter_dir, model)
   - load_pilora_ref(model_name_or_path)

2) Move reference to device:
   - move_ref_to_device(ref_cpu, device)

3) PiLoRA orthogonal regularization implemented as *gradient injection* (DDP-friendly):
   - add_pilora_ortho_grads_(model, ref, lambda_ortho, ...)

Design goals:
- Lightweight, self-contained.
- Compatible with DDP (no graph changes; adds gradients after backward).
- Works for Seq2Seq (no prototype/DCE parts).
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Union

import torch


# -------------------------
# Key normalization helpers
# -------------------------
def canon_name(name: str) -> str:
    """Strip DDP 'module.' prefix to keep keys stable across wrapped/unwrapped models."""
    return name[7:] if name.startswith("module.") else name


def _is_lora_weight_key(k: str) -> bool:
    return ("lora_" in k) and k.endswith("weight")


# -------------------------
# Reference snapshot IO
# -------------------------
def extract_pilora_ref_from_model(
    model,
    *,
    include_A: bool = True,
    include_B: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Extract a *CPU* snapshot of LoRA matrices from a model, to serve as PiLoRA reference.

    Returns:
        Dict[canonical_param_name -> CPU tensor clone]
    """
    ref: Dict[str, torch.Tensor] = {}
    for n, p in model.named_parameters():
        cn = canon_name(n)
        if not _is_lora_weight_key(cn):
            continue
        if (include_A and "lora_A" in cn) or (include_B and "lora_B" in cn):
            ref[cn] = p.detach().cpu().clone()
    return ref


def extract_pilora_ref_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    *,
    include_A: bool = True,
    include_B: bool = True,
) -> Dict[str, torch.Tensor]:
    """Same as extract_pilora_ref_from_model, but from a state_dict."""
    ref: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        ck = canon_name(k)
        if not _is_lora_weight_key(ck):
            continue
        if (include_A and "lora_A" in ck) or (include_B and "lora_B" in ck):
            ref[ck] = v.detach().cpu().clone()
    return ref


def save_pilora_ref(
    adapter_dir: str,
    model_or_state: Union[torch.nn.Module, Dict[str, torch.Tensor]],
    *,
    filename: str = "pilora_ref.pt",
) -> str:
    """
    Save PiLoRA reference to <adapter_dir>/<filename>.
    The saved object is a Dict[str, Tensor] on CPU.

    Returns: saved file path.
    """
    os.makedirs(adapter_dir, exist_ok=True)
    path = os.path.join(adapter_dir, filename)

    if isinstance(model_or_state, dict):
        ref = extract_pilora_ref_from_state_dict(model_or_state)
    else:
        ref = extract_pilora_ref_from_model(model_or_state)

    torch.save(ref, path)
    return path


def load_pilora_ref(
    model_name_or_path: str,
    *,
    filename: str = "pilora_ref.pt",
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Load PiLoRA reference:
    - If model_name_or_path is a directory, read <dir>/<filename>
    - If it's a .pt file, read it directly

    Returns:
        Dict[str, Tensor] on CPU, or None if not found/invalid.
    """
    if not model_name_or_path:
        return None

    # Directory case (typical: adapter folder)
    if os.path.isdir(model_name_or_path):
        p = os.path.join(model_name_or_path, filename)
        if os.path.isfile(p):
            obj = torch.load(p, map_location="cpu")
            if isinstance(obj, dict):
                return {canon_name(k): v.detach().cpu() for k, v in obj.items()}
            return None

    # Direct file case
    if os.path.isfile(model_name_or_path) and model_name_or_path.endswith(".pt"):
        obj = torch.load(model_name_or_path, map_location="cpu")
        if isinstance(obj, dict):
            return {canon_name(k): v.detach().cpu() for k, v in obj.items()}
        return None

    return None


# -------------------------
# Device move
# -------------------------
def move_ref_to_device(
    ref_cpu: Optional[Dict[str, torch.Tensor]],
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    """Move PiLoRA reference weights onto the training device."""
    if ref_cpu is None:
        return None
    return {k: v.to(device, non_blocking=True) for k, v in ref_cpu.items()}


# -------------------------
# PiLoRA orthogonal regularization (gradient injection)
# -------------------------
@torch.no_grad()
def add_pilora_ortho_grads_(
    model,
    ref: Dict[str, torch.Tensor],
    lambda_ortho: float,
    *,
    use_delta: bool = True,
    reg_on_A: bool = True,
    reg_on_B: bool = True,
    normalize: bool = False,
    eps: float = 1e-8,
    grad_scale: float = 1.0,
) -> Dict[str, float]:
    """
    In-place add PiLoRA orthogonal-regularization gradients to LoRA parameters.

    This is implemented as **post-backward gradient injection** (DDP-friendly).
    It does NOT modify forward graph.

    Reference:
      Encourage current task increment Δ to be orthogonal to reference weights W_ref.

    Shapes:
      A: (r, in)   B: (out, r)

    Loss definitions (low-rank friendly):
      For A:
        L_A = || ΔA · A_ref^T ||_F^2
        dL/dΔA = 2 (ΔA · A_ref^T) · A_ref

      For B:
        L_B = || B_ref^T · ΔB ||_F^2
        dL/dΔB = 2 B_ref · (B_ref^T · ΔB)

    Args:
      lambda_ortho: strength of orthogonal regularization.
      use_delta: if True, Δ = W - W_ref; else Δ = W (regularize absolute W).
      reg_on_A/reg_on_B: choose which matrices to regularize.
      normalize: if True, scale grad by a norm-based factor to reduce scale sensitivity.
      grad_scale: recommended to use 1/grad_accum_steps to keep magnitude stable.

    Returns:
      Dict with small stats for logging: {"proxy_sum":..., "num_tensors":..., "num_applied":...}
    """
    stats = {"proxy_sum": 0.0, "num_tensors": float(len(ref) if ref else 0), "num_applied": 0.0}

    if lambda_ortho <= 0.0 or (ref is None) or (len(ref) == 0):
        return stats

    lam = float(lambda_ortho) * float(grad_scale)

    for n, p in model.named_parameters():
        if p.grad is None or (not p.requires_grad):
            continue

        cn = canon_name(n)
        if cn not in ref:
            continue
        if not _is_lora_weight_key(cn):
            continue

        is_A = ("lora_A" in cn)
        is_B = ("lora_B" in cn)
        if (is_A and not reg_on_A) or (is_B and not reg_on_B) or ((not is_A) and (not is_B)):
            continue

        W_ref = ref[cn].to(p.device, non_blocking=True)
        W_cur = p.data

        # Only handle 2D weights
        if W_cur.ndim != 2 or W_ref.ndim != 2:
            continue

        Delta = (W_cur - W_ref) if use_delta else W_cur

        if normalize:
            denom = (W_ref.norm() * (Delta.norm() + eps) + eps)
        else:
            denom = 1.0

        if is_A:
            # G = ΔA A_ref^T   -> (r, r)
            G = Delta @ W_ref.t()
            grad = 2.0 * (G @ W_ref) / denom  # (r, in)
            stats["proxy_sum"] += float(G.abs().sum().item())
        else:
            # G = B_ref^T ΔB   -> (r, r)
            G = W_ref.t() @ Delta
            grad = 2.0 * (W_ref @ G) / denom  # (out, r)
            stats["proxy_sum"] += float(G.abs().sum().item())

        p.grad.add_(grad * lam)
        stats["num_applied"] += 1.0

    return stats
