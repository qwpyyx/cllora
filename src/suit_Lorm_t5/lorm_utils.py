# src/lorm_utils.py
import torch
import torch.nn as nn
import logging
import os

logger = logging.getLogger(__name__)


class LoRMTracker:
    """
    LoRM 专用：用于在训练后计算输入数据的 Gram 矩阵 (X^T * X)。
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.gram_matrices = {}
        self.hooks = []

    def _get_hook(self, layer_name):
        def hook(module, input, output):
            x = input[0].detach()
            if x.dim() == 3:
                x = x.view(-1, x.shape[-1])

            x = x.to(torch.float32)
            gram_diag = (x * x).sum(dim=0).cpu()  # shape: [Dim]

            if layer_name not in self.gram_matrices:
                self.gram_matrices[layer_name] = gram_diag
            else:
                self.gram_matrices[layer_name] += gram_diag

        return hook

    def register_hooks(self):
        self.hooks = []
        self.gram_matrices = {}
        target_keyword = "lora_A"
        for name, module in self.model.named_modules():
            if target_keyword in name and isinstance(module, nn.Linear):
                clean_name = name.replace(".default", "")
                h = module.register_forward_hook(self._get_hook(clean_name))
                self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_grams(self):
        return self.gram_matrices


class LormGlobalState:
    def __init__(self):
        # 存储累积的 Gram 矩阵: Sum(G_t)
        self.sum_G = {}
        # 存储累积的 Numerator: Sum(DeltaW_t * G_t)
        self.sum_WG = {}

    def update(self, model, current_task_grams, device="cpu", eps=1e-6):
        """
        Temporal aggregation (Eq.9) 的对角近似版本：
        - 若 G_task 是 1D（diag(X^T X) 向量 g），则：
            WG = DeltaW * g[None, :]
            Sum_G 累积的是向量 g
        - 若 G_task 是 2D（完整 Gram），则保持旧逻辑：
            WG = DeltaW @ G
            Sum_G 累积矩阵 G
        """
        logger.info("Updating LoRM Global State (Temporal Aggregation)...")

        # scaling = alpha / r（保持你原来的逻辑）
        scaling = 1.0
        if hasattr(model, "peft_config") and "default" in model.peft_config:
            pc = model.peft_config["default"]
            if hasattr(pc, "lora_alpha") and hasattr(pc, "r"):
                scaling = pc.lora_alpha / pc.r
                logger.info(f"LoRM: Applied scaling factor {scaling} (alpha={pc.lora_alpha}, r={pc.r})")

        params = {n: p for n, p in model.named_parameters() if "lora" in n}

        for gram_key, G_task in current_task_grams.items():
            if torch.isnan(G_task).any():
                logger.warning(f"⚠️ NaN detected in Gram for {gram_key}, skipping update for this layer.")
                continue

            # 你的 tracker 用 clean_name 去掉了 ".default"，这里仍旧沿用你的 key 拼法
            key_A = gram_key + ".default.weight"
            key_B = key_A.replace("lora_A", "lora_B")

            # 兼容：有些实现可能不是 default 命名
            if key_A not in params:
                key_A_alt = gram_key + ".weight"
                if key_A_alt in params:
                    key_A = key_A_alt
                    key_B = key_A.replace("lora_A", "lora_B")

            if key_A not in params or key_B not in params:
                continue

            A = params[key_A].detach().to(device).float()  # (r, in)
            B = params[key_B].detach().to(device).float()  # (out, r)

            # DeltaW = B @ A * scaling  => (out, in)
            DeltaW = torch.matmul(B, A) * scaling

            G = G_task.to(device).float()

            # ---------- diag 版 ----------
            if G.dim() == 1:
                g = G  # (in,)
                # 稳健：避免出现全 0 导致后面除 0（本层仍可合并，但影响会被 eps 控制）
                g = torch.clamp(g, min=0.0)

                WG = DeltaW * g.unsqueeze(0)  # (out, in)

                if gram_key not in self.sum_G:
                    self.sum_G[gram_key] = g.detach().cpu()
                    self.sum_WG[gram_key] = WG.detach().cpu()
                else:
                    self.sum_G[gram_key] += g.detach().cpu()
                    self.sum_WG[gram_key] += WG.detach().cpu()

            # ---------- full Gram 兼容版 ----------
            elif G.dim() == 2:
                WG = torch.matmul(DeltaW, G)  # (out, in)

                if gram_key not in self.sum_G:
                    self.sum_G[gram_key] = G.detach().cpu()
                    self.sum_WG[gram_key] = WG.detach().cpu()
                else:
                    self.sum_G[gram_key] += G.detach().cpu()
                    self.sum_WG[gram_key] += WG.detach().cpu()

            else:
                logger.warning(f"Unexpected Gram dim={G.dim()} for {gram_key}, skipping.")
                continue

    def merge_and_apply(self, model, device="cpu", eps=1e-6):
        """
        根据 Eq.9 计算最终 DeltaW_final 并合并到 backbone。
        - diag Gram（1D）：DeltaW_final[:, j] = Sum_WG[:, j] / (Sum_g[j] + eps)
        - full Gram（2D）：DeltaW_final = Sum_WG @ pinv(Sum_G)
        """
        logger.info("Applying LoRM Global State to Backbone...")

        updates_count = 0
        with torch.no_grad():
            for gram_key, Sum_G in self.sum_G.items():
                if gram_key not in self.sum_WG:
                    continue

                Sum_WG = self.sum_WG[gram_key].to(device).float()
                Sum_G_t = Sum_G.to(device).float()

                # ---------- diag 版 ----------
                if Sum_G_t.dim() == 1:
                    denom = Sum_G_t + eps  # (in,)
                    denom = torch.clamp(denom, min=eps)
                    DeltaW_final = Sum_WG / denom.unsqueeze(0)  # (out, in)

                # ---------- full Gram 兼容版 ----------
                elif Sum_G_t.dim() == 2:
                    try:
                        G_inv = torch.linalg.pinv(Sum_G_t, rcond=1e-5)
                    except Exception as e:
                        logger.error(f"SVD did not converge for {gram_key}, skipping merge. Error: {e}")
                        continue
                    DeltaW_final = torch.matmul(Sum_WG, G_inv)

                else:
                    logger.warning(f"Unexpected Sum_G dim={Sum_G_t.dim()} for {gram_key}, skipping.")
                    continue

                if torch.isnan(DeltaW_final).any():
                    logger.error(f"NaN generated in DeltaW_final for {gram_key}, skipping apply.")
                    continue

                # 找到 backbone Linear 层（沿用你的逻辑）
                linear_path = gram_key.replace(".lora_A", "")
                parts = linear_path.split('.')
                module = model
                found = True
                for p in parts:
                    if hasattr(module, p):
                        module = getattr(module, p)
                    else:
                        found = False
                        break
                if not found:
                    continue

                target_linear = module
                if hasattr(module, "base_layer"):
                    target_linear = module.base_layer

                if hasattr(target_linear, "weight"):
                    DeltaW_final = DeltaW_final.to(target_linear.weight.device).type(target_linear.weight.dtype)
                    target_linear.weight.data += DeltaW_final
                    updates_count += 1

        logger.info(f"LoRM: Merged {updates_count} layers into backbone.")

    def save(self, path):
        torch.save({'sum_G': self.sum_G, 'sum_WG': self.sum_WG}, path)

    def load(self, path):
        if os.path.exists(path):
            data = torch.load(path, map_location='cpu')
            self.sum_G = data.get('sum_G', {})
            self.sum_WG = data.get('sum_WG', {})
            logger.info(f"Loaded LoRM Global State from {path}")


def lorm_aggregate(client_updates, global_model, target_matrix="A", device="cpu", eps=1e-6):
    """
    LoRM 聚合（diag 版）：
    - client["grams"][name] 为 diag(X^T X) 向量 (Dim,)
    - 聚合 A：A_M[:,j] = (Σ_i A_i[:,j] * g_i[j]) / (Σ_i g_i[j])
    - 聚合 B：B_M = (Σ_i B_i G'_i) (Σ_i G'_i)^-1, 其中 G'_i = A diag(g_i) A^T (r×r)
    """
    aggregated_updates = {}
    global_params = {k: v.to(device) for k, v in global_model.state_dict().items() if "lora" in k}

    def _right_solve(num, den):
        # num: (out, r), den: (r, r). 计算 num @ inv(den) 的稳定做法
        I = torch.eye(den.shape[0], device=den.device, dtype=den.dtype)
        den_reg = den + eps * I
        return torch.linalg.solve(den_reg.T, num.T).T  # => num @ inv(den)

    all_keys = [k for k in global_params.keys() if "lora" in k]
    lora_A_keys = [k for k in all_keys if "lora_A" in k]

    for key_A in lora_A_keys:
        key_B = key_A.replace("lora_A", "lora_B")
        gram_key_base = key_A.replace(".weight", "").replace(".default", "")

        # ========== 聚合 A ==========
        if target_matrix == "A":
            num = None  # (r, d)
            den = None  # (d,)
            count = 0

            for client in client_updates:
                if key_A not in client["state_dict"]:
                    continue

                matched = next((k for k in client["grams"] if gram_key_base in k), None)
                if not matched:
                    continue

                A_i = client["state_dict"][key_A].to(device).to(torch.float64)  # (r, d)

                G_i = client["grams"][matched].to(device)
                # 兼容：如果你某次仍然传了 full Gram，就自动取对角线
                if G_i.dim() == 2:
                    g = torch.diagonal(G_i).to(torch.float64)  # (d,)
                else:
                    g = G_i.to(torch.float64)  # (d,)

                if num is None:
                    num = torch.zeros_like(A_i, dtype=torch.float64)
                    den = torch.zeros_like(g, dtype=torch.float64)

                num += A_i * g.unsqueeze(0)  # 列加权
                den += g
                count += 1

            if count > 0:
                A_merged = num * (1.0 / (den + eps)).unsqueeze(0)  # (r, d)
                aggregated_updates[key_A] = A_merged.to(torch.float32).cpu()

        # ========== 聚合 B ==========
        elif target_matrix == "B":
            if key_B not in all_keys:
                continue

            A_fixed = global_params[key_A].to(torch.float64)  # (r, d)

            num = None  # (out, r)
            den = None  # (r, r)
            count = 0

            for client in client_updates:
                if key_B not in client["state_dict"]:
                    continue

                matched = next((k for k in client["grams"] if gram_key_base in k), None)
                if not matched:
                    continue

                B_i = client["state_dict"][key_B].to(device).to(torch.float64)  # (out, r)

                G_i = client["grams"][matched].to(device)
                if G_i.dim() == 2:
                    # full Gram fallback：G' = A G A^T
                    G_prime = A_fixed @ G_i.to(torch.float64) @ A_fixed.T  # (r, r)
                else:
                    # diag 版：G' = A diag(g) A^T = (A * g) @ A^T
                    g = G_i.to(torch.float64)  # (d,)
                    G_prime = (A_fixed * g.unsqueeze(0)) @ A_fixed.T  # (r, r)

                if num is None:
                    num = torch.zeros((B_i.shape[0], A_fixed.shape[0]), device=device, dtype=torch.float64)
                    den = torch.zeros((A_fixed.shape[0], A_fixed.shape[0]), device=device, dtype=torch.float64)

                num += B_i @ G_prime
                den += G_prime
                count += 1

            if count > 0:
                B_merged = _right_solve(num, den)  # (out, r)
                aggregated_updates[key_B] = B_merged.to(torch.float32).cpu()

    return aggregated_updates