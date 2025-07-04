import copy
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.mab_optimizer import UCB1
import logging

logger = logging.getLogger(__name__)
# 该模块实现基于多臂老虎机搜索阈值并动态选择压缩率的联邦持续学习算法


class MABFedCL:
    """Federated continual learning module using MAB search for tau and dynamic k."""

    def __init__(self, model, args, accelerator, tau=0.5, topk_ratio=0.1):
        self.g_model = model            # 全局模型
        self.args = args                # 参数配置
        self.accelerator = accelerator  # accelerate 对象
        self.tau = tau                  # 当前阈值
        self.topk_ratio = topk_ratio    # 当前压缩比例
        self.historical_grad = None     # 历史梯度均值
        self.alpha = 100                  # softplus 系数
        # 惩罚系数
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda_orth = args.lambda_orth
        # 奇异值
        self.k = args.orthogonal_k
        # 正交基
        self.Q = None


    # 保存模型权重的辅助函数
    def _get_state_dict(self, model):
        state = {}
        for name, param in model.named_parameters():
            state[name] = param.detach().clone()
        return state

    # 记录客户端最后完成的任务编号
    def update_last_task(self, idx, current_task):
        import os
        client_dir = os.path.join(
            self.args.base_dir,
            f"seq_{self.args.idrandom}_seed{self.args.seed}",
            str(self.args.baseline),
            str(self.args.dataset),
            f"topK_{str(self.args.topk_ratio)}",
            f"client_idx_{idx}"
        )
        os.makedirs(client_dir, exist_ok=True)
        last_task_path = os.path.join(client_dir, 'last_task.txt')
        with open(last_task_path, 'w') as f:
            f.write(str(current_task))

    # 计算一个批次的梯度向量
    def _compute_batch_grad(self, model, batch):
        batch = {k: v.to(self.accelerator.device) if hasattr(v, "to") else v for k, v in batch.items()}
        outputs = model(**batch, restrict_label=True)
        loss = outputs.loss
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
        grad_vec = torch.cat([g.detach().flatten() for g in grads]).to(self.accelerator.device)
        return grad_vec

    # 计算当前梯度与历史梯度的余弦相似度
    def _compute_phi(self, g_curr):
        if self.historical_grad is None:
            return 1.0
        g_curr_unit = g_curr / (torch.norm(g_curr) + 1e-8)
        hist_unit = self.historical_grad / (torch.norm(self.historical_grad) + 1e-8)
        hist_unit = hist_unit.to(g_curr_unit.device)
        return torch.dot(g_curr_unit, hist_unit)


    # 对 LoRA 层的更新做 TopK 压缩
    def global_topk_compress_lora(self, delta_model, k_ratio, model):
        lora_layer_names = []
        if self.args.is_peft:
            for name, param in model.named_parameters():
                if 'lora' in name.lower() and param.requires_grad:
                    lora_layer_names.append(name)
        else:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    lora_layer_names.append(name)
        all_updates = []
        for name, tensor in delta_model.items():
            if name in lora_layer_names:
                all_updates.append(tensor.view(-1))
        if all_updates:
            all_updates = torch.cat(all_updates, dim=0)
            total_elements = all_updates.numel()
            k_num = max(int(total_elements * k_ratio), 1)
            k_num = min(k_num, total_elements)
            topk_values = torch.topk(torch.abs(all_updates), k_num, sorted=False)[0]
            global_threshold = topk_values.min()
            compressed_delta = {}
            for name, tensor in delta_model.items():
                if name in lora_layer_names:
                    mask = torch.ge(torch.abs(tensor), global_threshold)
                    compressed_delta[name] = tensor * mask.float()
                else:
                    compressed_delta[name] = tensor
        else:
            compressed_delta = delta_model
        return compressed_delta

    # 执行两阶段的本地训练流程并返回压缩后的模型差分
    def local_training(self, model, train_loader, optimizer, lr_scheduler, idx,
                       current_output_dir, historical_grad=None, local_ep=1,
                       current_task=0, global_tau=None, search_only=False, Q=None):
        if self.accelerator.is_main_process:
            logger.info("***** Running training in Local Client *****")
            logger.info(f"Client idx = {idx},  training size = {train_loader.total_dataset_length}")
            logger.info(f"Batch Size = {self.args.local_bs}, Local Epoch = {self.args.local_ep}")

        initial_state = self._get_state_dict(model)
        if self.accelerator.is_main_process:
            print('#' * 100)
            print("Begin Local Training!")

        first_batch = next(iter(train_loader))

        # 加载历史主子空间
        if Q is not None:
            self.Q = Q.to(self.accelerator.device)

        if historical_grad is not None:
            self.historical_grad = historical_grad.to(self.accelerator.device)
        else:
            self.historical_grad = self._compute_batch_grad(model, first_batch)

        # 第一阶段：利用多臂老虎机搜索最佳 tau
        if global_tau is None:
            tau_values = [float(x) for x in self.args.tau_candidates.split(',')]
            mab = UCB1(tau_values)

            mab_loader = iter(train_loader)

            for rnd in range(self.args.mab_rounds):
                try:
                    mab_batch = next(mab_loader)
                except StopIteration:
                    mab_loader = iter(train_loader)
                    mab_batch = next(mab_loader)
                grad_tmp = self._compute_batch_grad(model, mab_batch)
                phi_static = self._compute_phi(grad_tmp)
                if isinstance(phi_static, torch.Tensor):
                    phi_static = phi_static.item()


                arm = mab.select_arm(rnd + 1)
                tau_try = tau_values[arm]
                reward = max(phi_static - tau_try, 0.0)
                mab.update(arm, reward)
            self.tau = tau_values[mab.best_arm()]
        else:
            self.tau = global_tau

        if search_only:
            model = self.accelerator.unwrap_model(model)
            model.cpu()
            return None, None, 0, None, None, self.tau

        gradients = []
        phi_list = []

        for ep in range(local_ep):
            progress_bar = tqdm(range(len(train_loader)), disable=not self.accelerator.is_local_main_process)
            for _, batch in enumerate(train_loader):
                g_new = self._compute_batch_grad(model, batch)
                phi_val = self._compute_phi(g_new)
                phi_list.append(phi_val.detach())
                gradients.append(g_new.detach())
                # phi_loss = F.softplus(self.alpha * (self.tau - phi_val)) / self.alpha
                batch = {k: v.to(self.accelerator.device) if hasattr(v, "to") else v for k, v in batch.items()}
                outputs = model(**batch, restrict_label=True)
                task_loss = outputs.loss

                # total_loss = task_loss + self.lambda1 * phi_loss

                lora_params = [p for n, p in model.named_parameters() if 'lora_A' in n]
                L_orth = 0.0
                if self.Q is not None and lora_params:
                    g_vec = torch.cat([p.view(-1) for p in lora_params])
                    proj = torch.matmul(self.Q.t(), g_vec)
                    L_orth = self.lambda_orth * torch.sum(proj ** 2)

                total_loss = task_loss + L_orth

                optimizer.zero_grad()
                self.accelerator.backward(total_loss)
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                progress_bar.update(1)
                progress_bar.set_description('Train Iter (Epoch=%3d,loss=%5.3f, lambda=%5.3f)' % (ep, total_loss.item(), self.lambda1))

        self.accelerator.wait_for_everyone()
        phi_avg = torch.mean(torch.stack(phi_list))
        self.historical_grad = torch.mean(torch.stack(gradients), dim=0).detach()

        # 第二阶段：在确定 tau 后动态选择 k
        # k_values = [float(x) for x in self.args.k_candidates.split(',')]
        # best_score = -1e9
        # best_k = self.topk_ratio
        # for k_val in k_values:
        #     f_val = max(self.tau - phi_avg.item(), 0.0)
        #     score = phi_avg.item() - self.lambda1 * f_val - self.lambda2 * k_val
        #     if score > best_score:
        #         best_score = score
        #         best_k = k_val
        # self.topk_ratio = best_k


        save_dict = {"historical_avg_grad": self.historical_grad}
        output_file_path = os.path.join(current_output_dir, 'historical_avg_grad.pt')
        self.accelerator.save(save_dict, output_file_path)
        logger.info(f"Local historical_grad saved to {output_file_path}")

        self.update_last_task(idx, current_task)

        current_state = self._get_state_dict(model)
        delta_model = {key: current_state[key] - initial_state[key] for key in current_state}

        if self.accelerator.is_main_process:
            print(f'Doing compress by TopK, ratio is {self.topk_ratio}')

        delta_model_compressed = self.global_topk_compress_lora(delta_model, self.topk_ratio, model)

        model = self.accelerator.unwrap_model(model)
        model.cpu()

        total_samples = getattr(train_loader, 'total_dataset_length', len(train_loader.dataset))

        return None, None, total_samples, delta_model_compressed, phi_avg, self.tau

    # 服务器端聚合所有客户端的更新
    def server_aggregate(self, client_updates, client_prototypes, client_readouts, client_sample_counts, client_tau):
        if self.accelerator.is_main_process:
            logger.info("***** Begin to aggregate on Server *****")

        total_samples = sum(client_sample_counts.values())
        tau_global = sum(client_tau[cid] * client_sample_counts[cid] for cid in client_tau) / total_samples
        aggregated_delta = {}
        for cid, delta in client_updates.items():
            weight = client_sample_counts[cid] / total_samples
            for name, value in delta.items():
                aggregated_delta[name] = aggregated_delta.get(name, 0) + weight * value
        corrected_delta = {}
        for key, value in aggregated_delta.items():
            new_key = key[len("module."):] if key.startswith("module.") else key
            corrected_delta[new_key] = value
        for name, param in self.g_model.named_parameters():
            if name in corrected_delta:
                param.data += corrected_delta[name].to(param.device)

        return tau_global, None, None, self.g_model

    def aggregate_tau(self, client_tau, client_sample_counts):
        """Aggregate tau values from clients to get a global tau."""
        total_samples = sum(client_sample_counts.values())
        tau_global = sum(client_tau[cid] * client_sample_counts[cid] for cid in client_tau) / total_samples
        self.tau = tau_global
        return self.tau