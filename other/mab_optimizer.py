# src/mab_optimizer.py
import math
import torch
class UCB1:
    """基于 UCB1 的简单多臂老虎机优化器——全用 float"""

    def __init__(self, arms):
        self.arms = list(arms)
        self.counts = [0] * len(arms)
        self.values = [0.0] * len(arms)

    def select_arm(self, t):
        # 还没拉过的臂先试一次
        for i, c in enumerate(self.counts):
            if c == 0:
                return i

        # 全部用 Python float 列表来计算 UCB
        ucb_vals = [
            self.values[i] + math.sqrt(2 * math.log(t) / self.counts[i])
            for i in range(len(self.arms))
        ]
        # 直接返回最大值的索引
        return max(range(len(ucb_vals)), key=lambda i: ucb_vals[i])

    def update(self, arm_idx, reward):
        # 确保 reward 是纯 float
        if isinstance(reward, torch.Tensor):
            reward = reward.item()

        # 增量更新平均奖励
        self.counts[arm_idx] += 1
        n = self.counts[arm_idx]
        value = self.values[arm_idx]
        self.values[arm_idx] = ((n - 1) / float(n)) * value + reward / float(n)

    def best_arm(self):
        # self.values 全是 float，可以直接：
        return max(range(len(self.values)), key=lambda i: self.values[i])
