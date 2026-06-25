"""Compression baselines migrated from the old FLM-TopK framework.

This module operates on LoRA update dictionaries used by federated_uie_lora.py.
The sign convention is the same as the current framework: delta[k] = global[k] - local[k].
Returned tensors keep the same shapes/keys and can be aggregated directly.
"""

import copy
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim


def pq_loss(x, xb, parameter: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(parameter):
        parameter = torch.as_tensor(parameter, dtype=torch.float32)
    return x / (2 * ((2 ** xb - 1) ** 2)) * torch.sum(parameter)


def pq_quan(parameters: torch.Tensor, centroids: int) -> torch.Tensor:
    """Probabilistic quantization used by the old framework, with zero-range guard."""
    if parameters.numel() == 0:
        return parameters.clone()
    device = parameters.device
    work = parameters.detach().float()
    ans = torch.zeros_like(work)
    abs_para = work.abs()
    p_max = torch.max(abs_para)
    p_min = torch.min(abs_para)
    if centroids <= 1 or torch.isclose(p_max, p_min):
        return work.to(parameters.dtype)
    interval = (p_max - p_min) / float(centroids)
    if interval.abs().item() < 1e-12:
        return work.to(parameters.dtype)
    left = ((abs_para - p_min) / interval).int().float() * interval + p_min
    right = left + interval
    denom = torch.clamp(right - left, min=1e-12)
    probability = (abs_para - left) / denom
    probability = torch.clamp(probability, min=0.0, max=1.0)
    probability[probability < 1e-5] = 0
    seed = torch.rand(work.shape, device=device)
    ans[seed < probability] = right[seed < probability]
    ans[seed >= probability] = left[seed >= probability]
    ans = ans * torch.sign(work)
    return ans.to(parameters.dtype)


def _safe_log2_int(x: int) -> int:
    return int(math.ceil(math.log(max(int(x), 2), 2)))


def _calculate_n(costs: List[int], budget_bits: int) -> int:
    """Keep the old FedComp behavior: choose the largest prefix that stays below budget."""
    n = 0
    current_sum = 0
    for c in costs:
        current_sum += int(c)
        if current_sum >= budget_bits:
            break
        n += 1
    return n


def _split_even(total: int, parts: int) -> List[int]:
    parts = max(1, min(int(parts), max(int(total), 1)))
    base = total // parts
    sizes = [base] * parts
    left = total - sum(sizes)
    if left > 0:
        for i in range(parts - left, parts):
            sizes[i] += 1
    return [s for s in sizes if s > 0]


@dataclass
class CompressionStats:
    method: str
    packet_num: int
    nonzero: int
    total_numel: int
    extra: Dict[str, float]


class BaselineCompressor:
    """Old-framework baselines adapted to dict-shaped LoRA updates.

    method aliases accepted by compress():
      - flasc / topk       -> pure TopK
      - compeft / topk_pq  -> global TopK + probabilistic quantization
      - fedcomp            -> row-vector FedComp + 4-bit residual
      - flm_topk / block_opt -> FLM-TopK optimized block-size version
    """

    def __init__(
        self,
        *,
        packet_num: int,
        packet_bytes: int = 1500,
        blocks: int = 1024,
        bit: int = 18,
        min_bit: int = 4,
        topk_method: str = "gradient",
        lora_rank: int = 8,
        flm_opt_max_iter: int = 40,
        flm_max_blocks: int = 256,
        flm_disable_optim: bool = False,
    ) -> None:
        self.packet_num = max(1, int(packet_num))
        self.packet_size = int(packet_bytes) * 8
        self.blocks = max(1, int(blocks))
        self.max_bit = int(bit)
        self.min_bit = int(min_bit)
        self.topk_method = str(topk_method)
        self.lora_rank = int(lora_rank)
        # FLM-TopK/block_opt can be very expensive on 8B/14B LLM LoRA updates.
        # These knobs keep the baseline usable while making the approximation explicit
        # in baseline_compression_history.json. Set max_iter=1000 and max_blocks=0
        # to recover the old, much slower search behavior.
        self.flm_opt_max_iter = max(0, int(flm_opt_max_iter))
        self.flm_max_blocks = int(flm_max_blocks)
        self.flm_disable_optim = bool(flm_disable_optim)
        self.beta_1 = 0.85
        self.beta_2 = 0.85
        self.T_previous = {}
        self.U_previous = {}

        self.trainable_parameters = 0
        self.d = 0
        self.s = 1
        self.packet_min = 1
        self.packet_max = 1

    # ---------- public API ----------
    def compress(
        self,
        method: str,
        delta: Dict[str, torch.Tensor],
        param: Optional[Dict[str, torch.Tensor]],
        lora_keys: Iterable[str],
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], CompressionStats]:
        method = self._normalize_method(method)
        keys = [k for k in lora_keys if k in delta]
        flat_delta, flat_param, meta = self._flatten(delta, param, keys)
        total_numel = int(flat_delta.numel())
        if total_numel == 0:
            empty_stats = CompressionStats(method, self.packet_num, 0, 0, {})
            return {k: delta[k] for k in keys}, None, empty_stats

        self.trainable_parameters = total_numel

        residual_flat = None
        if method == "flasc":
            compressed_flat = self._compress_flasc(flat_delta)
            extra = {}
        elif method == "compeft":
            # ComPEFT is treated as a global TopK + probabilistic quantization baseline.
            # It does not use block partitioning; block optimization is reserved for FLM-TopK.
            compressed_flat = self._compress_compeft(flat_delta, flat_param)
            extra = {
                "bit": float(self.max_bit),
                "index_bits": float(self.s),
                "selected_values": float(getattr(self, "_last_compeft_k", 0)),
                "uses_blocks": 0.0,
            }
        elif method == "flm_topk":
            compressed_flat, new_blocks = self._compress_flm_topk_block_opt(flat_delta, flat_param)
            extra = {
                "init_blocks": float(self.blocks),
                "optimized_blocks": float(new_blocks),
                "optimized_blocks_unclipped": float(getattr(self, "_last_flm_new_blocks_unclipped", new_blocks)),
                "flm_opt_max_iter": float(self.flm_opt_max_iter),
                "flm_max_blocks": float(self.flm_max_blocks),
                "flm_disable_optim": 1.0 if self.flm_disable_optim else 0.0,
            }
        elif method == "fedcomp":
            compressed_flat, residual_flat = self._compress_fedcomp(delta, keys)
            extra = {}
        else:
            raise ValueError(f"Unknown compression baseline method: {method}")

        compressed = self._unflatten(compressed_flat, meta)
        residual = self._unflatten(residual_flat, meta) if residual_flat is not None else None
        nnz = int(sum(torch.count_nonzero(v).item() for v in compressed.values()))
        stats = CompressionStats(method, self.packet_num, nnz, total_numel, extra)
        return compressed, residual, stats

    @staticmethod
    def _normalize_method(method: str) -> str:
        method = str(method).lower()
        aliases = {
            "topk": "flasc",
            "flasc": "flasc",
            "topk_pq": "compeft",
            "compeft": "compeft",
            "fedcomp": "fedcomp",
            "block_opt": "flm_topk",
            "flm_topk": "flm_topk",
            "flm-topk": "flm_topk",
        }
        return aliases.get(method, method)

    # ---------- flatten / unflatten ----------
    def _flatten(self, delta, param, keys):
        flats = []
        p_flats = []
        meta = []
        for k in keys:
            t = delta[k].detach().cpu()
            flats.append(t.reshape(-1).float())
            if param is not None and k in param:
                p_flats.append(param[k].detach().cpu().reshape(-1).float())
            else:
                p_flats.append(torch.zeros(t.numel(), dtype=torch.float32))
            meta.append((k, tuple(t.shape), int(t.numel()), t.dtype))
        return torch.cat(flats), torch.cat(p_flats), meta

    def _unflatten(self, flat: Optional[torch.Tensor], meta):
        if flat is None:
            return None
        out = {}
        idx = 0
        for k, shape, numel, dtype in meta:
            out[k] = flat[idx:idx + numel].view(shape).to(dtype=dtype).cpu()
            idx += numel
        return out

    # ---------- common FLM-TopK helpers ----------
    def _prepare_pid_params(self, d_for_pid: int, value_bits: Optional[int] = None):
        value_bits = self.max_bit if value_bits is None else int(value_bits)
        self.d = max(1, int(d_for_pid))
        self.s = _safe_log2_int(self.d)
        self.packet_min = max(1, int(self.packet_size / max(1, value_bits + self.s)))
        self.packet_max = max(1, int(self.packet_size / max(1, self.min_bit + self.s)))

    def _generate_packet_sizes(self, k: int, packet_num: int) -> np.ndarray:
        packet_num = max(1, int(packet_num))
        if k < packet_num:
            return np.full(packet_num, self.packet_min, dtype=int)
        sizes = np.random.randint(self.packet_min, self.packet_max + 1, size=packet_num)
        diff = int(k - sizes.sum())
        guard = 0
        while diff != 0 and guard < 100000:
            guard += 1
            changed = False
            for i in range(packet_num):
                if diff == 0:
                    break
                if diff > 0 and sizes[i] < self.packet_max:
                    inc = min(self.packet_max - sizes[i], diff)
                    sizes[i] += inc
                    diff -= inc
                    changed = True
                elif diff < 0 and sizes[i] > self.packet_min:
                    dec = min(sizes[i] - self.packet_min, -diff)
                    sizes[i] -= dec
                    diff += dec
                    changed = True
            if not changed:
                break
        return sizes

    def _calculate_norm(self, grad: torch.Tensor) -> torch.Tensor:
        sq = grad.detach().float().abs().pow(2)
        norm = sq.sum()
        return torch.clamp(norm, min=1e-12)

    def _topk_grad(self, blocks: int, grad: torch.Tensor, param: torch.Tensor):
        n = int(grad.numel())
        blocks = max(1, min(int(blocks), n, self.packet_num))
        block_sizes = _split_even(n, blocks)
        list_grad = list(torch.split(grad, block_sizes, dim=0))
        list_param = list(torch.split(param, block_sizes, dim=0))

        base_packets = self.packet_num // blocks
        list_packet_num = [base_packets] * blocks
        left = self.packet_num - sum(list_packet_num)
        if left > 0:
            for i in range(blocks - left, blocks):
                list_packet_num[i] += 1
        list_packet_num = [max(1, x) for x in list_packet_num]
        list_k = [max(1, int(x * self.packet_max)) for x in list_packet_num]

        init_value, init_indices = [], []
        for i in range(blocks):
            if self.topk_method == "gradient":
                score = list_grad[i].abs()
            elif self.topk_method == "graproduct":
                score = (list_param[i] * list_grad[i]).abs()
            elif self.topk_method == "graproduct_2":
                score = (list_param[i] * list_grad[i]).pow(2).abs()
            elif self.topk_method == "adalora":
                gp = list_param[i] * list_grad[i]
                T_current = self.beta_1 * self.T_previous.get(i, torch.zeros_like(gp)) + (1 - self.beta_1) * gp
                U_current = self.beta_2 * self.U_previous.get(i, torch.zeros_like(gp)) + (1 - self.beta_2) * (gp - T_current).abs()
                self.T_previous[i] = T_current
                self.U_previous[i] = U_current
                score = (T_current * U_current).abs()
            else:
                raise ValueError(f"Unknown topk_method={self.topk_method}")
            k = min(list_k[i], score.numel())
            if k <= 0:
                init_value.append(torch.empty(0, dtype=score.dtype))
                init_indices.append(torch.empty(0, dtype=torch.long))
            else:
                v, ind = torch.topk(score, k=k, largest=True)
                init_value.append(v)
                init_indices.append(ind)
        return init_value, init_indices, list_grad, list_param, list_packet_num

    def _apply_cvlc(self, init_value, init_indices: torch.Tensor, grad: torch.Tensor, packet_num: int, norm: torch.Tensor):
        if grad.numel() == 0 or packet_num <= 0 or init_value.numel() == 0:
            return torch.zeros_like(grad), 0.0
        abs_grad = grad.abs()
        sort_grad, _ = torch.sort(abs_grad, descending=True)
        sort_grad = sort_grad * sort_grad

        best_loss = float("inf")
        best_ks: List[int] = []
        best_bs: List[int] = []
        best_k = 0
        k_max = min(int(self.packet_max * packet_num), int(grad.numel()))
        k_min = min(int(self.packet_min * packet_num), k_max)
        if k_max <= 0:
            return torch.zeros_like(grad), 0.0
        interval = max(1, int(max(1, k_max - k_min) / max(1, self.max_bit)))

        for k in range(k_max, max(k_min, 0), -interval):
            ks = self._generate_packet_sizes(k, packet_num)
            ks.sort()
            # Clamp generated sizes to feasible selected-k budget.
            if int(ks.sum()) > k_max:
                overflow = int(ks.sum() - k_max)
                for i in range(len(ks)):
                    take = min(max(0, ks[i] - 1), overflow)
                    ks[i] -= take
                    overflow -= take
                    if overflow <= 0:
                        break
            ks = np.array([int(x) for x in ks if int(x) > 0], dtype=int)
            if len(ks) == 0:
                continue
            bs = [int(self.packet_size / max(1, ks_i) - self.s) for ks_i in ks]
            if len(bs) == 0 or bs[0] < self.min_bit:
                continue

            sparse_tail_start = min(int(ks.sum()), sort_grad.numel())
            spar_loss = torch.sum(sort_grad[sparse_tail_start:]) / norm
            quan_loss = torch.tensor(0.0)
            acc = 0
            for i in range(len(ks)):
                quan_loss += pq_loss(int(ks[i]), int(bs[i]), sort_grad[acc:acc + int(ks[i])]) / norm
                acc += int(ks[i])
            loss = spar_loss + quan_loss

            last_loss = None
            for _ in range(max(1, self.max_bit)):
                kL = 0
                for i in range(len(ks) - 1):
                    kR = int(kL + ks[i] + ks[i + 1])
                    x, y = int(ks[i]), int(ks[i + 1])
                    xb, yb = int(bs[i]), int(bs[i + 1])
                    fx_loss = (pq_loss(x, xb, sort_grad[kL:kL + x]) + pq_loss(y, yb, sort_grad[kL + x:kR])) / norm

                    for xb_try in range(self.min_bit, self.max_bit):
                        x_try = int(self.packet_size / max(1, self.s + xb_try))
                        y_try = kR - kL - x_try
                        if x_try <= 0 or y_try <= 0 or y_try <= self.s:
                            continue
                        yb_try = int(self.packet_size / max(1, y_try) - self.s)
                        if yb_try < self.min_bit or yb_try > self.max_bit:
                            continue
                        tmp_loss = (pq_loss(x_try, xb_try, sort_grad[kL:kL + x_try]) + pq_loss(y_try, yb_try, sort_grad[kL + x_try:kR])) / norm
                        if tmp_loss < fx_loss:
                            fx_loss = tmp_loss
                            ks[i], ks[i + 1] = x_try, y_try
                            bs[i], bs[i + 1] = xb_try, yb_try
                    kL += int(ks[i])

                quan_loss = torch.tensor(0.0)
                acc = 0
                for i in range(len(ks)):
                    quan_loss += pq_loss(int(ks[i]), int(bs[i]), sort_grad[acc:acc + int(ks[i])]) / norm
                    acc += int(ks[i])
                loss = spar_loss + quan_loss
                loss_val = float(loss.detach().cpu().item())
                if last_loss is not None and abs(loss_val - last_loss) < 1e-12:
                    break
                last_loss = loss_val

            loss_val = float(loss.detach().cpu().item())
            if loss_val < best_loss:
                best_loss = loss_val
                best_k = min(int(np.sum(ks)), init_value.numel())
                best_ks = [int(x) for x in ks]
                best_bs = [int(x) for x in bs]

        if best_k <= 0:
            # Conservative fallback: keep as many top values as packet budget allows.
            best_k = min(k_max, init_value.numel())
            best_ks = [best_k]
            best_bs = [self.max_bit]

        _, final_indices = torch.topk(init_value, k=best_k)
        final_index = init_indices[final_indices]
        new_grad = torch.zeros_like(grad)
        new_grad[final_index] = grad[final_index]

        acc = 0
        for ks_i, bs_i in zip(best_ks, best_bs):
            if acc >= best_k:
                break
            take = min(int(ks_i), best_k - acc)
            indices = final_index[acc:acc + take]
            new_grad[indices] = pq_quan(new_grad[indices], 2 ** int(bs_i))
            acc += take
        return new_grad, best_loss

    # ---------- baseline methods ----------
    def _compress_flasc(self, grad: torch.Tensor) -> torch.Tensor:
        self._prepare_pid_params(grad.numel(), value_bits=32)
        k = int(self.packet_size / max(1, 32 + self.s) * self.packet_num)
        k = max(1, min(k, grad.numel()))
        _, indices = torch.topk(grad.abs(), k=k)
        out = torch.zeros_like(grad)
        out[indices] = grad[indices]
        return out

    def _compress_compeft(self, grad: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        """Global TopK + probabilistic quantization baseline.

        Unlike FLM-TopK/block_opt, ComPEFT is not block-partitioned here.
        Under a packet budget, each selected value consumes one quantized value
        plus one global index.  We therefore choose a global TopK set and then
        probabilistically quantize the selected values.
        """
        del param  # ComPEFT here is magnitude-based and does not need current parameters.
        self._prepare_pid_params(grad.numel(), value_bits=self.max_bit)
        k = int(self.packet_size / max(1, self.max_bit + self.s) * self.packet_num)
        k = max(1, min(k, grad.numel()))
        self._last_compeft_k = int(k)

        _, indices = torch.topk(grad.abs(), k=k)
        out = torch.zeros_like(grad)
        out[indices] = grad[indices]
        if indices.numel() > 0:
            out[indices] = pq_quan(out[indices], 2 ** max(1, self.max_bit - 1))
        return out

    def _block_optim_only(self, grad: torch.Tensor, param: torch.Tensor, block_size: float, norm: torch.Tensor, value_bit: int) -> int:
        if self.flm_disable_optim or self.flm_opt_max_iter <= 0:
            return max(1, int(round(block_size)))
        k_min = self.packet_size / max(1.0, (math.log(max(self.d, 2), 2) + 32.0))
        block_size_t = torch.tensor([float(block_size)], dtype=torch.float32, requires_grad=True)
        optimizer = optim.Adam([block_size_t], lr=100)
        min_block_size = max(float(self.d) / float(self.packet_num), float(k_min), 1.0)
        max_iter = max(1, int(self.flm_opt_max_iter))
        last_block = None
        best_block_size = max(min_block_size, float(block_size))
        min_loss = float("inf")

        for _ in range(max_iter):
            optimizer.zero_grad()
            bs = max(float(block_size_t.detach().item()), 1.0)
            self.packet_max = max(1, int(self.packet_size / max(1, value_bit + 1 + int(math.log(max(bs, 2), 2)))))
            block = max(1, min(int(self.d / bs), grad.numel(), self.packet_num))
            try:
                init_value, _, list_grad, _, _ = self._topk_grad(block, grad, param)
            except Exception:
                break
            if not grad.requires_grad:
                grad = grad.clone().detach().requires_grad_(True)
            if not param.requires_grad:
                param = param.clone().detach().requires_grad_(True)
            list_grad_req = [g.clone().detach().requires_grad_(True) for g in list_grad]

            sparse_errors = []
            for idx, g in enumerate(list_grad_req):
                keep = init_value[idx].numel()
                sg, _ = torch.sort(g.abs(), descending=True)
                sg = sg * sg
                sparse_errors.append(torch.sum(sg[keep:]) / norm)

            avg_r = torch.tensor(float(self.packet_num), dtype=torch.float32) / (float(self.d) / block_size_t)
            k_mid = int((self.packet_max + self.packet_min) / 2)
            quan_errors = []
            for g in list_grad_req:
                sg, _ = torch.sort(g.abs(), descending=True)
                sg = sg * sg
                quan_errors.append(avg_r * (pq_loss(k_mid, value_bit, sg[:k_mid]) / norm))

            loss = sum(sparse_errors) + sum(quan_errors)
            if not loss.requires_grad:
                loss = loss.clone().detach().requires_grad_(True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                block_size_t.clamp_(min=min_block_size)

            loss_val = float(loss.detach().cpu().item())
            if loss_val < min_loss:
                min_loss = loss_val
                best_block_size = float(block_size_t.detach().item())
            curr_block = int(float(block_size_t.detach().item()))
            if last_block == curr_block:
                break
            last_block = curr_block
        return max(1, int(round(best_block_size)))

    def _compress_flm_topk_block_opt(self, grad: torch.Tensor, param: torch.Tensor) -> Tuple[torch.Tensor, int]:
        # Initial state follows old do_compress(method='block_opt').
        self.d = int(self.trainable_parameters)
        self.s = _safe_log2_int(self.d)
        self.packet_min = max(1, int(self.packet_size / max(1, self.max_bit + self.s)))
        self.packet_max = max(1, int(self.packet_size / max(1, self.min_bit + self.s)))

        block_cap = self.flm_max_blocks if self.flm_max_blocks and self.flm_max_blocks > 0 else self.packet_num
        init_blocks = max(1, min(self.blocks, grad.numel(), self.packet_num, int(block_cap)))
        block_size = float(self.d) / float(init_blocks)
        norm = self._calculate_norm(grad)
        value_bit = 10
        new_block_size = self._block_optim_only(grad, param, block_size, norm, value_bit)
        new_blocks_unclipped = max(1, round(self.d / max(1, new_block_size)))
        new_blocks = max(1, min(new_blocks_unclipped, grad.numel(), self.packet_num, int(block_cap)))
        self._last_flm_new_blocks_unclipped = int(new_blocks_unclipped)

        self._prepare_pid_params(math.ceil(self.trainable_parameters / new_blocks), value_bits=self.max_bit)
        init_value, init_indices, list_grad, list_param, list_packet_num = self._topk_grad(new_blocks, grad, param)
        out = torch.zeros_like(grad)
        start = 0
        for b in range(new_blocks):
            param_size = list_param[b].numel()
            opt, _ = self._apply_cvlc(init_value[b], init_indices[b], list_grad[b], list_packet_num[b], norm)
            out[start:start + param_size] = opt
            start += param_size
        return out, new_blocks

    def _split_tensor_to_vectors(self, tensor: torch.Tensor, index: int):
        if tensor.dim() == 1:
            tensor2 = tensor.unsqueeze(1)
            squeeze_back = True
        else:
            tensor2 = tensor
            squeeze_back = False
        new_tensor = torch.zeros_like(tensor2, dtype=torch.long)
        vectors = [tensor2[i, :].unsqueeze(0) for i in range(tensor2.shape[0])]
        for i in range(tensor2.shape[0]):
            new_tensor[i, :] = index
            index += 1
        if squeeze_back:
            new_tensor = new_tensor.squeeze(1)
        return vectors, new_tensor, index

    def _infer_fedcomp_vector_len(self, vector_list: List[torch.Tensor]) -> int:
        widths = [int(v.size(1)) for v in vector_list if v.dim() == 2]
        if len(widths) == 0:
            return max(1, self.lora_rank)
        non_rank = [w for w in widths if w != self.lora_rank]
        return max(non_rank) if len(non_rank) > 0 else max(widths)

    def _compress_fedcomp(self, delta: Dict[str, torch.Tensor], keys: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Reproduce old FedComp at tensor-row granularity, but infer hidden size instead of hard-coding 4096.
        all_vectors = []
        mark_dict = {}
        index = 0
        for k in keys:
            vectors, mark, index = self._split_tensor_to_vectors(delta[k].detach().cpu().float(), index)
            mark_dict[k] = mark
            all_vectors.extend(vectors)

        vector_len = self._infer_fedcomp_vector_len(all_vectors)
        budget_bits = self.packet_size * self.packet_num
        scored = []
        for idx, v in enumerate(all_vectors):
            l1 = torch.mean(torch.abs(v)).item()
            if int(v.size(1)) == self.lora_rank:
                index_len = vector_len
            else:
                index_len = self.lora_rank
            cost = 32 * int(v.size(1)) + 5 + _safe_log2_int(index_len)
            scored.append((idx, l1, cost))
        scored.sort(key=lambda x: x[1], reverse=True)
        costs = [x[2] for x in scored]
        k_num = _calculate_n(costs, budget_bits)
        selected = set(x[0] for x in scored[:k_num])

        flat_new = []
        flat_res = []
        for k in keys:
            mark = mark_dict[k].numpy()
            arr = delta[k].detach().cpu().float().numpy()
            arr_res = copy.deepcopy(arr)
            mask = np.isin(mark, list(selected))
            arr[~mask] = 0
            arr_res[mask] = 0
            flat_new.append(torch.from_numpy(arr).reshape(-1))
            flat_res.append(torch.from_numpy(arr_res).reshape(-1))
        new_flat = torch.cat(flat_new) if flat_new else torch.tensor([])
        res_flat = torch.cat(flat_res) if flat_res else torch.tensor([])
        # Old framework quantizes the residual with 4-bit PQ before replay.
        res_flat = pq_quan(res_flat, 2 ** max(1, 4 - 1))
        return new_flat, res_flat


def apply_residual_to_lora_state(
    global_state_cpu: Dict[str, torch.Tensor],
    residual_cpu: Dict[str, torch.Tensor],
    lora_keys: Iterable[str],
) -> Dict[str, torch.Tensor]:
    """FedComp residual replay: initialize local LoRA as global - residual."""
    out = {k: v.detach().cpu().clone() for k, v in global_state_cpu.items()}
    for k in lora_keys:
        if k in residual_cpu and k in out:
            out[k] = out[k] - residual_cpu[k].to(dtype=out[k].dtype)
    return out
