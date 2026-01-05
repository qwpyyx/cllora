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

            # [Fix 1] 强制转 float32 防止半精度溢出
            x = x.float()

            # 计算 Gram: X^T * X
            gram = torch.matmul(x.t(), x).cpu()

            if layer_name not in self.gram_matrices:
                self.gram_matrices[layer_name] = gram
            else:
                self.gram_matrices[layer_name] += gram

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

    def update(self, model, current_task_grams, device="cpu"):
        """
        在任务结束时，将当前任务的 DeltaW 和 Gram 并入全局历史。
        [Fix 2] 包含 Scaling Factor
        """
        logger.info("Updating LoRM Global State (Temporal Aggregation)...")

        # 获取 Scaling Factor (alpha / r)
        scaling = 1.0
        if hasattr(model, "peft_config") and "default" in model.peft_config:
            pc = model.peft_config["default"]
            if hasattr(pc, "lora_alpha") and hasattr(pc, "r"):
                scaling = pc.lora_alpha / pc.r
                logger.info(f"LoRM: Applied scaling factor {scaling} (alpha={pc.lora_alpha}, r={pc.r})")

        params = {n: p for n, p in model.named_parameters() if "lora" in n}

        for gram_key, G_task in current_task_grams.items():
            # 检查 Gram 是否含有 NaN
            if torch.isnan(G_task).any():
                logger.warning(f"⚠️ NaN detected in Gram matrix for {gram_key}, skipping update for this layer.")
                continue

            key_A = gram_key + ".default.weight"
            key_B = key_A.replace("lora_A", "lora_B")

            if key_A not in params or key_B not in params:
                continue

            A = params[key_A].detach().to(device).float()
            B = params[key_B].detach().to(device).float()
            G = G_task.to(device).float()

            # [Fix 2] DeltaW = B @ A * scaling
            # 必须乘 scaling，否则合并进 Backbone 的权重太小
            DeltaW = torch.matmul(B, A) * scaling

            WG = torch.matmul(DeltaW, G)

            # 累积到全局状态
            if gram_key not in self.sum_G:
                self.sum_G[gram_key] = G.cpu()
                self.sum_WG[gram_key] = WG.cpu()
            else:
                self.sum_G[gram_key] += G.cpu()
                self.sum_WG[gram_key] += WG.cpu()

    def merge_and_apply(self, model, device="cpu"):
        """
        根据 Equation 9 计算最终权重，并直接合并到 Backbone 中。
        [Fix 3] 使用 pinv 替代 inverse 防止崩溃
        """
        logger.info("Applying LoRM Global State to Backbone...")

        updates_count = 0
        with torch.no_grad():
            for gram_key, Sum_G in self.sum_G.items():
                if gram_key not in self.sum_WG: continue

                Sum_WG = self.sum_WG[gram_key].to(device).float()
                Sum_G = Sum_G.to(device).float()

                # [Fix 3] 使用伪逆 (pseudo-inverse) 替代 inverse
                # 对于不满秩或接近奇异的矩阵，pinv 更加稳定且不会抛出错误
                # 即使 Sum_G 包含 0 或很小的值，pinv 也能给出一个最小范数解
                # 这种方法在 RegMean 论文实现中也很常见
                try:
                    # 使用 hermitian=True (对称矩阵) 可以加速，但 G 是 X^TX 必然对称
                    # rcond=1e-5 忽略极小的奇异值
                    G_inv = torch.linalg.pinv(Sum_G, rcond=1e-5)
                except Exception as e:
                    logger.error(f"SVD did not converge for {gram_key}, skipping merge. Error: {e}")
                    continue

                DeltaW_final = torch.matmul(Sum_WG, G_inv)

                # 检查结果是否 NaN
                if torch.isnan(DeltaW_final).any():
                    logger.error(f"NaN generated in DeltaW_final for {gram_key}, skipping apply.")
                    continue

                # 找到 Backbone Linear 层
                linear_path = gram_key.replace(".lora_A", "")
                parts = linear_path.split('.')
                module = model
                found = True
                for p in parts:
                    if hasattr(module, p):
                        module = getattr(module, p)
                    else:
                        found = False;
                        break

                if found:
                    target_linear = module
                    if hasattr(module, "base_layer"):
                        target_linear = module.base_layer

                    if hasattr(target_linear, "weight"):
                        # 合并到 Backbone
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


# lorm_aggregate 保持不变 (空间聚合不需要 scaling，因为是在 LoRA 空间内操作)
def lorm_aggregate(client_updates, global_model, target_matrix="A", device="cpu"):
    aggregated_updates = {}
    global_params = {k: v.to(device) for k, v in global_model.state_dict().items() if "lora" in k}

    # 空间聚合仍使用带扰动的 inverse，这通常足够且比 pinv 快
    def safe_inverse(m):
        return torch.inverse(m + 1e-4 * torch.eye(m.shape[0], device=m.device))

    all_keys = [k for k in global_params.keys() if "lora" in k]
    lora_A_keys = [k for k in all_keys if "lora_A" in k]

    for key_A in lora_A_keys:
        key_B = key_A.replace("lora_A", "lora_B")
        gram_key_base = key_A.replace(".weight", "").replace(".default", "")

        if target_matrix == "A":
            num, den = 0, 0
            count = 0
            for client in client_updates:
                if key_A not in client['state_dict']: continue
                matched = next((k for k in client['grams'] if gram_key_base in k), None)
                if not matched: continue

                A_i = client['state_dict'][key_A].to(device).float()
                G_i = client['grams'][matched].to(device).float()
                num += A_i @ G_i
                den += G_i
                count += 1

            if count > 0:
                aggregated_updates[key_A] = (num @ safe_inverse(den)).cpu()

        elif target_matrix == "B":
            if key_B not in all_keys: continue
            A_fixed = global_params[key_A].float()
            num, den = 0, 0
            count = 0
            for client in client_updates:
                if key_B not in client['state_dict']: continue
                matched = next((k for k in client['grams'] if gram_key_base in k), None)
                if not matched: continue

                B_i = client['state_dict'][key_B].to(device).float()
                G_i = client['grams'][matched].to(device).float()
                G_prime = A_fixed @ G_i @ A_fixed.T
                num += B_i @ G_prime
                den += G_prime
                count += 1

            if count > 0:
                aggregated_updates[key_B] = (num @ safe_inverse(den)).cpu()

    return aggregated_updates