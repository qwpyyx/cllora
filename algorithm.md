### 算法流程

#### 1.**符号**

- 全局模型 $w$；节点 $j$；LoRA 层 $\ell=1,\dots,L$
- 当前批 Fisher（对角/EMA）：$F_{\text{curr}}^{(\ell)}$
- 旧任务 Fisher 与最优参：$\bar F^{(\ell)}$, $\bar\theta^{(\ell)}$（每个任务结束时由 state 维护）
- 学习率：基础 $\eta_0$，冲突层步长 $\eta^{(\ell)}_B$
- 当前任务$t$，任务集合$1,2,...,T_{task}$
- 任务 $t$ 的轮数 $R_t$，本地 epoch $E$，每轮本地步数 $T$（epoch×batch），batch $\mathcal{B}$
- 半径阈值 $R$，误差常数 $\beta\!\ge\!1,\ \sigma,\tau\!\ge\!0$
- 通信预算 $P$，每层上传成本 $c_\ell$（按“包”计整数）

#### 2.算法流程

对每个任务$t={1,2,...,T_{task}}$:

**服务器**

对每一轮$r=1,...,R_t$:

​	对被选择的$S_r$集合内的节点并行做：

​		上传参数$\mu^j_{r} = \text{clientupdate}(w_{r})$ 

​	FedAvg 聚合当前轮的模型：$w_{r+1} \leftarrow w_{r} - \frac{1}{|S_r|}\sum_{j\in S_r}\mu^j_{r}$

​	广播 $w_{r+1}$ 到下一轮集合 $S_{r+1}$

return $w^T$

****

**客户端clientupdate**

接受服务器的参数$\theta \leftarrow w_r$，并获取每一层的$\theta^{(\ell)}$

初始化每一层的当前fisher信息：对每一层$l$，令$F_{curr}^l=0$

获取过去任务的表征信息：

$\bar F^{(\ell)} \leftarrow \text{state}.\bar F^{(\ell)}$ （表征所有过去任务的Fisher信息)

$\bar\theta^{(\ell)} \leftarrow \text{state}.\bar B^{(\ell)} \oslash \bar F^{(\ell)}$ (表征所有过去任务的最优参数信息)

初始化整轮统计量（为step-3服务）：

$r^2_{\ell}\ \gets\ \|\theta^{(\ell)}-\bar\theta^{(\ell)}\|^2_{\bar F^{(\ell)}},\quad r^2_{\ell,\text{start}}\gets r^2_{\ell}$

$B^{\text{round}}_\ell\gets0,\quad \text{conf}_\ell\gets0 $

对每个local epoch $1,2，...，E$：

​	对每个batch  $B \in \mathcal{B}$：

​		对于每个LoRA层 $\ell$:

​			对所有层获取当前batch的梯度$g_B^{(\ell)}$

​			**/* ---- 在线获取对角近似Fisher**

​			$F_{\mathrm{batch}}^{(\ell)} \leftarrow g_B^{(\ell)} \odot g_B^{(\ell)}$ （对角近似）

​			$F_{\mathrm{curr}}^{(\ell)} \leftarrow \alpha F_{\mathrm{curr}}^{(\ell)} + (1-\alpha) F_{\mathrm{batch}}^{(\ell)}$ （EMA平滑）

​			**/* ---- Step 2需要**

​			$v_B^{(\ell)}\gets g_B^{(\ell)}\oslash\big(F_{\text{curr}}^{(\ell)}\big)$

​			$a_B^{(\ell)}=(v_B^{(\ell)})^\top \bar F^{(\ell)} v_B^{(\ell)},\quad b_B^{(\ell)}=(v_B^{(\ell)})^\top \bar F^{(\ell)}\big(\theta^{(\ell)}-\bar\theta^{(\ell)}\big)$

​			$\Delta_{\ell}= R^2 - r^2_{\ell}\quad(\text{半径余量})$

​			如果$\Delta_{\ell} < \tau$，学习率为0并跳过该层更新

​			**/* ---- Step 2 对冲突参数调整学习率**

​			如果 $b_B^{(\ell)}<0$，说明新旧任务方向存在冲突，该层即认定为冲突层，用带误差的安全闭式
$$
\eta^{(\ell)}_B =\frac{\,b_B^{(\ell)}-\sigma+\sqrt{(b_B^{(\ell)}-\sigma)^2+\beta\,a_B^{(\ell)}(\Delta_\ell-\tau)}\,}{\beta\,a_B^{(\ell)}}, \quad \eta^{(\ell)}_B\leftarrow \min\{\eta^{(\ell)}_B,\eta_0\}.
$$
​			并记录 $\text{conf}_\ell \leftarrow \text{conf}_\ell +1$

​			否则$\eta_{B}^{(\ell)} \leftarrow \{\eta_0,\sqrt{(\Delta_t - \tau)/a_B^{(\ell)}}\}$

​			**/* ---- Step 3 累计“收益/遗忘/半径/冲突频度”**

​			预测收益（新任务，二阶）

$$
Q_B^{(\ell)}= (g_B^{(\ell)})^\top\!\big(F_{\text{curr}}^{(\ell)}\big)^{-1}\!g_B^{(\ell)},\quad B^{\text{round}}_\ell\!+\!=\max\!\left\{0,\ \Big(\eta_B^{(\ell)}-\tfrac12(\eta_B^{(\ell)})^2\Big)\,Q_B^{(\ell)}\right\}.
$$
​			更新马氏半径（旧任务）
$$
r^2_{\ell}\ \leftarrow\ r^2_{\ell} - 2\eta_B^{(\ell)} b_B^{(\ell)} + (\eta_B^{(\ell)})^2 a_B^{(\ell)} .
$$
​			更新参数：$\theta^{(\ell)} \leftarrow \theta^{(\ell)} - \eta_{B}^{(\ell)}\, v_B^{(\ell)}$

****

**/* ---- Step 3 选择关键层集合并上传（本地该轮训练已经完成）**

- **整轮遗忘代价**
  $$
  F^{\text{round}}_\ell=\max\left\{0,\ \tfrac12\big(r^2_{\ell}-r^2_{\ell,\text{start}}\big)\right\}.
  $$

- **冲突频度**：$p^{\text{round}}_\ell=\text{conf}_\ell/T$。

- **新任务收益**：$B^{\text{round}}_\ell$

- **构建优化问题**：
  $$
  \max_{z\in\{0,1\}^L}\ \sum_\ell z_\ell\Big(B_\ell^{\text{round}}-\lambda\,p_\ell^{\text{round}}F_\ell^{\text{round}}\Big) \quad\text{s.t. }\sum_\ell z_\ell c_\ell\le P.
  $$
  标准0-1背包问题，可以用DP求解

- **得到上传的层集合S**：若$\ell \in S$则上传该层增量，否则置零

****

训练完该轮后$\theta_j^\ast \leftarrow \theta$，$\mu_{full} \leftarrow w_r - \theta_j^* $

逐层判断$\mu^{(\ell)} \leftarrow \mu_{full}^{(\ell)}$ ，若$\ell \in S$，否则$\mu^{(\ell)} \leftarrow 0$ 

返回$\mu_r^j = \mu$

****

**任务结束后**：

更新每一层的旧任务表征，对每一层$\ell$累计$F_{\mathrm{r}}^{(\ell)} \leftarrow F_{\mathrm{curr}}^{(\ell)}$

假设整个任务总共全局聚合了$R_t$轮，任务结束后，对R轮$F_{\mathrm{task}}^{(\ell)}$求平均$F_{\mathrm{task}}^{(\ell)} = \frac{1}{R_t}\sum_{r=1}^{R_t} F_{\mathrm{r}}^{(\ell)}$

任务结束后，更新表征所有旧任务的Fisher信息和参数信息

$\text{state}.\bar F^{(\ell)} \leftarrow \gamma\,\text{state}.\bar F^{(\ell)} + F_{\mathrm{task}}^{(\ell)}$

$\text{state}.\bar B^{(\ell)} \leftarrow \gamma\,\text{state}.\bar B^{(\ell)} + \big(F_{\mathrm{task}}^{(\ell)} \odot \theta^{\ast(\ell)}\big)$ 