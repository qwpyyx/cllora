# 基于几何理论的自适应学习率稀疏上传策略

#### **背景**

1. 联邦大模型下的持续学习为达到新任务可塑性和旧任务稳定性，通常需要**更多轮的全局通信**才能达到平衡，在大模型参数规模大，下游任务数量多的场景下带来巨大的通信开销
2. 主流方法除上传更新参数外，通常需要**额外传递信息**来获得平衡，进一步加剧通信成本，造成通信瓶颈。

![image-20250905143424851](C:\Users\Wenqi\AppData\Roaming\Typora\typora-user-images\image-20250905143424851.png)

#### **目的**

在联邦大模型持续学习中，达到**新旧任务平衡**同时**提高通信效率**。

#### **别人怎么解决的**

- 纯持续学习方法

1. 完全正交更新（O-LoRA）：为不同任务强制 LoRA 更新彼此正交，意在消除任务间干扰、避免遗忘。

2. 完全共享（标准 LoRA）：不做任何保护或约束，每个新任务直接更新（覆盖）上一任务参数。
3. 重放/Replay：为后续任务保留上一任务的部分样本，在新任务训练时“回放”以稳住旧知识。
4. 重要性正则（EWC/等）：估计参数重要性（如Fisher矩阵），对“重要参数”施加二次惩罚、限制其偏移。
5. 梯度投影（GEM/等）：基于梯度方向相容性进行投影更新，确保对旧任务损失不升高。

- 联邦持续学习方法

  常见范式是在上述机制上“加联邦”：如客户端本地做 EWC/投影/重放 等，再把更新上传到服务器聚合；也有做“额外信息”传递（例如重要性统计、原型/表征、掩码等）来缓解遗忘与异构性。但这些额外通信用于“换来稳定性”的做法，往往会**进一步加剧通信成本**，在大模型与多任务场景下易形成瓶颈

#### **别人方法的问题**

1. “二极对立”：完全正交 vs 完全共享 

   完全正交（O-LoRA） 把任务子空间彼此割裂，**牺牲可共享表征**，在任务相似度不低时会抑制正迁移，新任务效果反而变差（“一刀切”地忽略相似性）

   完全共享（标准 LoRA） 则走向另一端：**不保护旧任务**、大范围覆盖易触发灾难性遗忘，同样属于“一刀切”。

2. 只看“重要性”或只看“方向”的偏颇度量

   EWC 类只关注“哪些参数重要”，但**不关心当下更新方向是否与旧知识冲突**；而 GEM 类恰好相反，强调“方向不冲突”，但**忽略不同维度的重要性差异**。这两类方法都属于“单指标视角”。

3. 重放/回放的现实约束

   样本级回放受**隐私合规与存储**/带宽限制掣肘；用表征/原型替代虽可缓解，但仍会带来**额外通信与系统复杂度**

4. 通信与可扩展性

   为稳住旧任务，许多联邦方法需要**额外上传/下发**“统计量、掩码或原型”等辅助信息，进一步加剧通信开销

总结：

| **方法/特性**             |         旧任务稳定          | 新任务可塑                  | 通信高效         | 无需额外数据                                | 隐私安全         | 适配大模型           |
| ------------------------- | :-------------------------: | --------------------------- | ---------------- | ------------------------------------------- | ---------------- | -------------------- |
| **完全正交 (O-LoRA)**     |              ✓              | ✗ （惩罚抑制新任务）        | ✓                | ✓                                           | ✓                | ✓                    |
| **完全共享 (标准LoRA)**   |   ✗（完全覆盖旧任务参数）   | ✓                           | ✓                | ✓                                           | ✓                | ✓                    |
| **正则化 (EWC等)**        | ✗（只考虑重要性不考虑方向） | ✓                           | ✗ (需传重要性)   | ✓                                           | ✓                | ✗（计算参数Hessian） |
| **梯度投影 (GEM等)**      |              ✓              | ✗（只考虑方向不考虑重要性） | ✗ (需传梯度矩阵) | ✗（需要额外存储旧任务数据以表征旧任务信息） | ✓                | ✗ (投影计算复杂)     |
| **重放/回放**             |              ✓              | ✓                           | ✓                | ✗（需要额外存储样本）                       | ✗ (表征泄露风险) | ✗ (存储瓶颈)         |
| **本文方法 (几何自适应)** |            **✓**            | **✓**                       | **✓**            | **✓**                                       | **✓**            | **✓**                |

现有路线要么在**共享**与**隔离**之间“二选一”，要么在**重要性**与**方向**之间“二选一”；再叠加**联邦通信开销**与**隐私约束**，使得联邦大模型的持续学习难以同时兼顾**新任务可塑性、旧任务稳定性**与**通信效率**。

****

#### **我们实验发现**

1. **适度共享优于“完全正交/完全共享”的二元极端**：**这表明并非所有参数共享都会引发遗忘**。

2. **适当稀疏既能达到好的性能，又能防止遗忘**：**这表明遗忘源于高度稀疏的参数冲突**，造成灾难性遗忘的“有害更新”并非均匀分布，而是高度集中在少量参数维度上。因此**共享应该是选择性的，问题不在“要不要共享”，而在共享谁**。

   ![image-20250909175342674](C:\Users\Wenqi\AppData\Roaming\Typora\typora-user-images\image-20250909175342674.png)

| 方案            | 上传比例    | (均匀分布)预期遗忘度 | **实际遗忘度 (exact_match)** | 结论                               |
| :-------------- | :---------- | :------------------- | :--------------------------- | :--------------------------------- |
| `newAlg-03`     | 100% (基准) | -0.30                | **-0.30**                    | 基准                               |
| `newAlg-03-k01` | 10%         | **≈ -0.03**          | **-0.48**                    | **实际遗忘 >> 预期遗忘**           |
| `newAlg-03-k03` | 30%         | **≈ -0.09**          | **+0.30**                    | **实际遗忘 ≠ 预期遗忘 (甚至变好)** |
| `newAlg-03-k05` | 50%         | **≈ -0.15**          | **-0.27**                    | **实际遗忘 ≈ 基准遗忘**            |

上述发现共同指向一个结论：**理想策略应为“选择性共享”**——在绝大多数参数上鼓励共享以促进学习与高效通信，仅对少数高冲突参数进行修复调整。

然而现有方法要么仅基于重要性（EWC 类）一刀切地约束被估为重要的参数，**不区分当前更新是利是害**；要么仅基于**方向相容性**（GEM类）对所有不相容方向一概投影，**不区分不同维度的风险权重**。**因此这两种单一度量无法精准定位这少数真正有害的维度**。

为此我们引入**“冲突参数”**的定义：在**重要性加权的度量下**，其当前更新将使**旧任务损失即时上升**的参数。这一定义同时融合了“重要性”与“方向”两个维度。基于此，我们的方案分为三步：

**①识别**：利用黎曼几何，精准识别出高冲突参数。

**②调整**：构造旧任务盆地并为冲突参数计算安全更新步长，在激发新任务可塑性的同时，严格保障旧任务稳定性。

**③通信优化**：在有限带宽下，将上传决策构建为收益-代价优化问题，优先上传对**新任务收益高且对旧任务冲突小**的参数，最终同时实现**稳定性、可塑性与通信效率**的三角平衡。

#### **难点**：

1. 如何在同时考虑参数重要性和方向的背景下，识别冲突参数——step1
2. 如何调整这些冲突参数，来防止旧任务遗忘？——step2
3. 如何达到“稳定-可塑-通信优化”三者的平衡？——step3

****

### step1—识别冲突参数

**两个步骤：**

1. 欧氏-Fisher-分布-黎曼递进链条，说明在黎曼流形下分析冲突的必要性
2. 用二阶泰勒展开把“是否伤害旧任务”的判定量推出来

**步骤1—为什么必须在黎曼度量下做冲突判定**

**①、判定需求：同时衡量“方向 × 重要性”且不依赖坐标**

- 欧氏内积默认各维等权，且对**重参数化**（同一模型不同参数化/缩放/LoRA 分解尺度）不稳定，导致“是否冲突”随坐标改变。

**②、切换到分布尺度：KL 的局部几何给出正确的“度量”**

- 我们关心的不是“参数走了多远”，而是“**模型分布改变了多少**”。

- 对任意可微参数化分布 $P_\theta$，小扰动 $\Delta\theta$ 引起的分布偏移由 KL 的二阶展开刻画：
  $$
  \mathrm{KL}\!\big(P_\theta\Vert P_{\theta+\Delta\theta}\big) \;=\;\tfrac12\,\Delta\theta^\top F(\theta)\,\Delta\theta\;+\;o(\|\Delta\theta\|^2).
  $$
  其中$F(\theta):=\mathbb{E}_{x\sim P_\theta}\!\big[\nabla_\theta\log P_\theta(x)\,\nabla_\theta\log P_\theta(x)^\top\big]$是信息矩阵。这说明$F(\theta)$是 KL 的本征二次型：**它天然编码“重要性”，并诱导出与分布变化对齐的内积$\langle u,v\rangle_F := u^\top F(\theta)v$**

**③、坐标无关：Fisher的重参数化不变性**

- 该度量对可逆可微重参数化 $\phi=\psi(\theta)$ 满足协变关系 $F_\phi=J^\top F_\theta J$，从而 $\Delta\phi^\top F_\phi\Delta\phi=\Delta\theta^\top F_\theta\Delta\theta$（**重参数化不变**）。即局部“长度/角度”保持不变——**冲突判定不随参数化改变**。这正是我们需要的性质	

**④、 Fisher 到黎曼：参数空间成为统计流形，冲突在该度量下定义。**

-   当 $\theta\mapsto F(\theta)$ 随位置平滑变化时，$\langle\cdot,\cdot\rangle_{F(\theta)}$ 构成**黎曼度量场**（Fisher统计流形）。
    
-   于是，整个理论应在这个**坐标无关且能表达重要性的度量**下进行——这就是我们后续分析与算法操作的统一几何。
    

> 本质上是**Fisher度量下的局部信任域问题**，而非欧氏几何中的启发式约束。

**步骤2—如何在黎曼流形下判断参数是否冲突？**

常用假设：

**G1（局部二阶光滑）**  $L_{\text{old}}$ 在 $\theta$ 邻域二阶可微，Hessian-Lipschitz:  
$$
\|\nabla^2 L_{\text{old}}(\theta + \Delta) - \nabla^2 L_{\text{old}}(\theta)\| \leq \rho \|\Delta\|.
$$

*用途*：保证泰勒余项 $O(\alpha^3\|v\|^3)$ 可控，用于二阶近似。  

**G2（旧任务几何代理）** 
旧任务在当前的梯度与曲率可用历史 Fisher近似：  
$$
\nabla L_{\text{old}}(\theta) \approx \bar{F}(\theta)(\theta - \hat{\theta}), 
\qquad 
\nabla^2 L_{\text{old}}(\theta) \approx \bar{F}(\theta),
$$

其中 $\bar{F}$ 采用对角/块对角估计 + EMA 平滑 + damping $(+\lambda I)$。  

*用途*：给出可计算的 $S, Q$ 及逐参 $s_i$。

****

-   旧任务合并损失 $L_{\text{old}}(\theta)$ 在当前点 $\theta$ 二阶可微；$\bar\theta$ 是旧任务的近似极小点。
    
-   记旧任务的**Fisher$\bar{F}(\theta)$ **与旧梯度代理 $g_{\text{old}}:=\bar{F}(\theta)\,(\theta-\bar\theta)$（根据梯度的泰勒展开，在极小点附近成立）。
    
-   令 $v$ 为任意候选更新方向，下面推导对任意 $v$ 都成立。

$$
\begin{aligned} \Delta L_{\text{old}}(\alpha) &:= L_{\text{old}}(\theta-\alpha v)-L_{\text{old}}(\theta)\\ &\approx -\alpha\,\underbrace{\nabla L_{\text{old}}(\theta)^\top v}_{(A)}\; +\;\tfrac{\alpha^2}{2}\,\underbrace{v^\top H_{\text{old}}(\theta)\,v}_{(B)} . \end{aligned}
$$

在旧任务极小点邻域用 $ \nabla L_{\text{old}}(\theta)\approx \bar{F}(\theta)(\theta-\bar\theta)$、$H_{\text{old}}(\theta)\approx \bar{F}(\theta)$ 得

$$
(A)\approx v^\top \bar{F}(\theta-\bar\theta),\qquad (B)\approx v^\top \bar{F} v .
$$
**定义两个与度量相关的标量：**


$$
\boxed{\;S:=\,v^\top \bar{F}(\theta-\bar\theta),\qquad Q:=v^\top \bar{F} v\ \ge 0\;}
$$

于是


$$
\boxed{\;\Delta L_{\text{old}}(\alpha)\ \approx\ -\alpha\,S\ +\ \tfrac{\alpha^2}{2}\,Q.\;}
$$

可以看出，当S<0时，$\Delta L_{\text{old}}(\alpha)$一定上升，因为Q非负，因此**只需要判断每个参数的$S_i$是否小于0就能判断该参数是否是冲突参数**。（$v=F_{curr}^{-1}g_{new}$）

****

### step2—构建旧任务盆地，设计安全更新步长

目标：

1. 将旧任务知识抽象成一个参数空间中围绕其最优解$\bar{\theta}$的安全盆地（定义旧任务盆地）
2. 对于冲突参数，任何新任务更新步长都必须保证参数在更新后仍停留在该盆地内（设计安全步长）

前提：

1.为什么只处理S<0而不处理S>0

- 当S>0，沿-v的无穷小步会降低旧任务损失，而S<0时，无穷小步就会**升高**旧损失、
- 当S>0时， 始终存在一个非零步长区间能保证不遗忘（$\alpha_{safe}=\frac{2S}{Q}$).
- 新旧任务同方向时做投影或者缩放，使得损失下降变缓慢，反而降低正迁移能力。

2.为什么要按层聚合来判断冲突，而不是按逐参数级别来判断？

在稀疏上传，小噪声的背景下，假设模型按层划分：总参量 $P=\sum_{\ell=1}^{L} n_\ell$。

- 计算复杂度下降，由逐参数的${O(P)+O(P\log k)}$降到按层聚合的${O(P)+O(L\log L)}$

- 统计稳定性好（零均值，有限二阶矩的常见假设下）

  - **方差下降**
    $$
    \ \mathrm{Var}(\bar S_\ell\mid\mathcal H)\ \le\ \frac{\sigma_{\max,\ell}^2}{n_{\mathrm{eff},\ell}}\ ,\qquad n_{\mathrm{eff},\ell}:=\frac{n_\ell}{1+(n_\ell-1)\rho_{\max,\ell}}\in(0,n_\ell]\ .
    $$
     $\rho_{\max,\ell}<1$，$n_{\mathrm{eff},\ell}$ 随 $n_\ell$ 增长，**层聚合的方差必随之下降**；$\rho\!\to\!1$ 时最坏“持平”，跟逐参数方差一致。

  - **SNR上升**
    $$
    \ \mathrm{SNR}_\ell\ \ge\ \frac{|\bar\mu_\ell|}{\sigma_{\max,\ell}}\ \sqrt{\,n_{\mathrm{eff},\ell}\,}\ .
    $$
    与逐参数单点 $\mathrm{SNR}_i=|\mu_i|/\sigma_i\le |\bar\mu_\ell|/\sigma_{\max,\ell}$ 相比，**层聚合至少获得 $\sqrt{n_{\mathrm{eff},\ell}}$ 量级的提升**

  - **错判概率随$n_{e,l}$下降**
    $$
    \Pr(\text{错判}) \ \le\ \frac{\mathrm{Var}(S_\ell\mid\mathcal H)}{M_\ell^2} \ \le\ \frac{\sigma_{\max,\ell}^2\,n_\ell^2/n_{\mathrm{eff},\ell}}{(n_\ell\bar\mu_\ell)^2} \ =\ \ \Big(\frac{\sigma_{\max,\ell}}{|\bar\mu_\ell|}\Big)^2\ \frac{1}{n_{\mathrm{eff},\ell}}\ .
    $$
    对比逐参数：单点错判 $\Pr(\mathrm{sign}(s_i)\neq\mathrm{sign}(\mu_i))\le (\sigma_i/|\mu_i|)^2$，与 $n_\ell$ 无关，也就是逐参数判定不具备这种随层宽衰减的性质；而层聚合的错判**随 $n_{\mathrm{eff},\ell}$ 至少按 $1/n$ 衰减**

- 系统实现更友好

  - **逐参数稀疏**：需要传大量**索引**（$\approx k\log_2 P$ bit），并造成非连续内存访问与碎片化通信，吞吐明显受损
  - **按层/块稀疏**：上传的是**连续张量块**，天然适配NCCL/all-reduce、压缩与量化；端到端带宽与延迟更可控。

****

#### （1）构建旧任务盆地

目标：构造一个围绕$\bar{\theta}$的“安全盆地”集合，使得留在集合就能保证旧任务变化受控。

对于$S_{\ell}$小于0的冲突情况，不改变梯度方向的话旧损一定上升，我们希望旧任务损失的变化/增加量不超过$\varepsilon$。在考虑KL分布的情况下，就是KL散度<=$\varepsilon$。我们在 **不重放旧数据** 的前提下，需要一个“旧任务的可计算代理“，用来定义”旧任务盆地“。由于没有旧数据，因此无法直接用旧任务损失$L_{old}(\theta)$，于是我们引入分布层面的量，用仅需$\bar{\theta},\bar{F}$的二次型来替代。

##### ①构造KL球（分布视角）

旧任务代表点$\bar{\theta}$ ; Fisher度量 $g(\vartheta)=F(\vartheta)$。记$\Delta\theta=\theta-\bar{\theta}$、$\bar{F}=F(\bar{\theta})$。

一、首先将“KL散度<=$\varepsilon$"的集合近似等价为“马氏距离<=$R$"的集合
$$
2\operatorname{KL}(p_{\bar{\theta}}\parallel p_\theta)=\Delta\theta^\top\bar{F}\Delta\theta+O(\|\Delta\theta\|^3).
$$
给出“分布不变预算 $\varepsilon$”对应的**KL椭球**:
$$
\mathcal{E}_R:=\{\theta:\Delta\theta^\top\bar{F}\Delta\theta\leq R^2\},\quad R^2:=2\varepsilon.
$$
于是，在无重放的条件下，我们获得一个**可计算的旧知识约束**，只依赖$\bar{\theta},\bar{F}$.

****

接下来需要建立几何解释，把KL球与黎曼球对齐。因为我们本质上是从黎曼流形的角度来分析，因此我们需要构建黎曼球，这个球才是我们从黎曼流形角度得到的真正的旧任务盆地。同时我们要给出黎曼球和KL球之间的误差（因为黎曼球可能无法计算），所以我们要知道从黎曼球落回可计算的KL球会带来多大的误差。

##### ②构建黎曼球（几何视角）

在 Fisher 流形上定义黎曼距离$d_g$​,旧任务盆地的黎曼球：
$$
\mathcal{B}_R^{\mathrm{geo}}:=\{\theta:\:d_g(\theta,\bar{\theta})\leq R\}.
$$
利用法向坐标/比较定理，小半径下有严格夹逼:
$$
\psi_{+K}(R)\:\Delta\theta^\top\bar{F}\:\Delta\theta\:\leq\:d_g^2(\theta,\bar{\theta})\:\leq\:\psi_{-\kappa}(R)\:\Delta\theta^\top\bar{F}\:\Delta\theta,
$$
其中$\psi_{-\kappa}(R)=1+\frac{\kappa R^{2}}{3}+O(\kappa^{2}R^{4})\geq1$, $\psi_{+K}(R)=1-\frac{KR^{2}}{3}+O(K^{2}R^{4})\leq1$。

可以得出结论：

在半径不超过R的邻域，用KL球近似黎曼球的相对误差为$O(\kappa R^2)+O(KR^2)$，其中$\kappa, K$为最大/最小截面曲率，只要R足够小且曲率不爆炸，这个误差就是可控的。这给统计近似的KL球升级为几何意义：KL球是黎曼球的二阶刻度

****

#### （2）设计安全更新步长

当层被判定为冲突层时，应该限制其更新步长，使其一步之后仍然在旧任务盆地内，因此需要找到最大安全学习率。

首先从黎曼流形角度推导出安全上界和离开下界。考虑到可计算问题，应该从黎曼视角落回可计算的拉普拉斯视角（KL），并考虑近似误差。

**前提符号**

- 旧任务基点：$\bar{\theta}$；当前位置：$\theta$
- Fisher度量：$g(\vartheta)=F(\vartheta),\bar{F}:=F(\bar{\theta})$。
- 半径：
  - 几何：$r_\mathrm{geo}^2=d_g^2(\theta,\bar{\theta});$
  - 拉普拉斯：$r^2=(\theta-\bar{\theta})^\top\bar{F}(\theta-\bar{\theta})$。
- 方向与系数(按层):
  - 更新方向：$v$ (用当前 Fisher 的自然梯度得到：$v=F_{\mathrm{curr}}^{-1}g_{\mathrm{new}}$)。
  - 几何二次项：$\tilde{a}=\|v\|_{F_{(\theta)}}^2$；拉普拉斯二次项：$a=v^\top\bar{F}v.$
  - 线性项 (外/内判定) : $b=v^\top\bar{F}(\theta-\bar{\theta})$。
    导数$\frac d{dn}r^2(\eta)|_{\eta=0}=-2b$。冲突当且仅当$b<0$ (初始朝外)。

**黎曼视角**

沿$\gamma(\eta)=\exp_\theta(-\eta v)$,由$S(\eta)=\frac12d_g^2(\gamma(\eta),\bar{\theta})$的 Hessian 比较$\psi_{+K}\tilde{a}\leq\bar{S^{\prime\prime}}(\eta)\leq\psi_{-\kappa}\tilde{a}$，通过两次积分可以得到

安全上界（充分条件）：让上界不超过阈值
$$
\eta_{\mathrm{safe}}^{\mathrm{geo}}=\frac{-\tilde{b}+\sqrt{\tilde{b}^{2}+\psi_{-\kappa}\tilde{a}\Delta}}{\psi_{-\kappa}\tilde{a}}
$$
离开下界（必要条件）：让下界超过阈值
$$
\eta_{\mathrm{leave}}^{\mathrm{geo}}=\frac{-\tilde{b}+\sqrt{\tilde{b}^2+\psi_{+K}\tilde{a}\Delta}}{\psi_{+K}\tilde{a}}
$$
其中$\tilde{b}=\langle v,\mathrm{Log}_\theta(\bar{\theta})\rangle_{F(\theta)}\mathrm{、}\Delta=R^2-r_\mathrm{geo°}^2$而$r_{\mathrm{geo}}^2=d_g^2(\theta,\bar{\theta}),\tilde{a}=\|v\|_{F(\theta)}^{2}$

**KL近似视角（可计算）**

用$d_g^2\approx r^2,-\tilde{b}\to b,\tilde{a}\to a,\psi_\pm\to1$，可得拉普拉斯下的精确解：
$$
{\eta_{\max}=\frac{b+\sqrt{b^2+a(R^2-r^2)}}{a}}
$$
通过单调性可以证明：
$$
\eta_{\mathrm{geo}}^{\mathrm{safe}}\leq\eta_{\mathrm{max}}\leq\eta_{\mathrm{geo}}^{\mathrm{leave}}
$$
**误差修正**

主要源自距离畸变的相对误差（黎曼球和KL球）和Log线性化的一阶小偏差

- 距离畸变(相对误差) $\to$乘法因子 $\beta=\psi_{-\kappa}(R)\geq1;$
- Log线性化(一阶小偏差)$\to$线性项微调 $\sigma$(取保守号)。

$$
\eta_{\mathrm{safe}}=\frac{b-\sigma+\sqrt{(b-\sigma)^{2}+\beta a\left(R^{2}-r^{2}\right)}}{\beta a}
$$

$$
\eta_{\mathrm{leave}}=\frac{b+\sigma+\sqrt{(b+\sigma)^2+\tilde{\beta}a(R^2-r^2)}}{\tilde{\beta}a}
$$

其中$\beta=\psi_{-\kappa}(R)\ge 1,\tilde{\beta}=\psi_{+K}(R)\leq1。$当$\beta=1,\sigma=0$时，回到$\eta_{\max}$

****

#### **step3—在有限通信资源下，上传最大“收益”的层**

目的：有限通信开销下，通过选择新任务收益高且对旧任务冲突小的层上传，提高通信效率

问题：设计优化问题求解，包括新任务收益，旧任务遗忘

**（1）计算新任务“增益”**

令当前点$\theta$，更新轨迹取测地线$\gamma(\eta)=Exp_{\theta}(-\eta v_{\ell})$，对新任务对数似然$L_{new}(\theta)$做黎曼泰勒：
$$
L_{\text{new}}(\gamma(\eta)) = L_{\text{new}}(\theta) - \eta \langle \nabla L_{\text{new}}(\theta), v_\ell \rangle_{F(\theta)} + \frac{\eta^2}{2} \operatorname{Hess} L_{\text{new}}(\theta)[v_\ell, v_\ell] + R_3
$$
其中$v_\ell = \left( F_{\text{curr}}^{(\ell)}  \right)^{-1} g_\ell, \quad g_\ell := \nabla_{\theta} L_{\text{new}}(\theta)$，若假设Hessian的Lipschitz有界，则可以推出 $|R_3| \leq \frac{L_H}{6} \eta^3 \|v_\ell\|_{F(\theta)}^3$，令$v_\ell^\top F_{\text{curr}} v_\ell = Q_\ell$，可以得到一步下降的损失：
$$
\Delta L_{\text{new}}^{(\ell)} := L_{\text{new}}(\gamma(\eta_\ell)) - L_{\text{new}}(\theta) = - \left( \eta_\ell - \frac{1}{2}\eta_\ell^2 \right) Q_\ell + R_3
$$
把下降的损失幅度当做增益，可以得到**新任务增益为**
$$
{B_{\ell}=\max\{0,(\eta_{\ell}-\frac{1}{2}\eta_{\ell}^{2})Q_{\ell}\}}
$$
取正值是因为如果沿该层方向一步对新任务是“负收益”（不降反升），本就不该上传这一层。累计收益为：
$$
B_\ell^\mathrm{round}=\sum_{t=1}^TB_{\ell,t}.
$$
**（2）计算旧任务遗忘程度**

定义旧任务遗忘程度是**该层的冲突频度跟旧任务的遗忘代价的乘积**

**遗忘代价：**

令旧任务在$\bar{\theta}$附近用二次能量刻画为：$\Phi(\theta)\:\triangleq\:\frac{1}{2}\|\theta-\bar{\theta}\|_F^2\:=\:\frac{1}{2}(\theta-\bar{\theta})^\top\bar{F}\:(\theta-\bar{\theta})$，其中$\bar{F}=F(\bar{\theta})$为旧任务的Fisher。令本层更新为$\theta^+=\theta-\eta v$。记$r^2:=\|\theta-\bar{\theta}\|_F^2$，$ a:=v^\top\bar{F}v\:(>0)$，$b:=v^\top\bar{F}(\theta-\bar{\theta})$，则一步更新后的能量变化为：
$$
\begin{aligned}
\Delta\Phi(\eta) & :=\Phi(\theta^+)-\Phi(\theta) \\
 & =\frac{1}{2}\|(\theta-\bar{\theta})-\eta v\|_\bar{F}^2-\frac{1}{2}\|\theta-\bar{\theta}\|_\bar{F}^2 \\
 & =\frac{1}{2}\left(r^2-2\eta b+\eta^2a-r^2\right) \\
 & =
-\eta b+\frac{1}{2}\eta^{2}a.\\
&=
\frac{1}{2}(r^2(\eta)-r^2)
\end{aligned}
$$
可以得出：**遗忘代价就是“到盆地中心的马氏半径平方的增量的一半”**

因此我们定义**代价**为：
$$
F_\ell:=\max\{0,\Delta\Phi_\ell(\eta_\ell)\}=\max\left\{0,-\eta_\ell b_\ell+\frac{1}{2}\eta_\ell^2a_\ell\right\}
$$
在训练过程中同步维护
$$
r_{t+1}^2=r_t^2-2\eta_tb_t+\eta_t^2a_t,\quad a_t=v_t^\top\bar{F}v_t,b_t=v_t^\top\bar{F}(\theta_t-\bar{\theta}),
$$
可得累计代价：
$$
F_\ell^\mathrm{round}=\max\{0,\frac{1}{2}(r_T^2-r_0^2)\}
$$
**冲突频度：**

记录所有batch中该层被确定为冲突层的次数，与总迭代轮数的比：
$$
p_\ell^\mathrm{round}=\frac{1}{T}\sum_{t=1}^T\mathbf{1}[b_t<0]\quad\in[0,1].
$$
**（3）通信成本**

一层传输量为$c_{\ell}$，总共传输字节为$\mathcal{P}$，有$\sum_{\ell}z_{\ell}c_\ell \le \mathcal{P}$

**（4）优化问题**
$$
\max_{z\in\{0,1\}^{L}}\sum_{\ell}z_{\ell}\left(B_{\ell}^{\mathrm{round}}-\lambda p_{\ell}^{\mathrm{round}}F_{\ell}^{\mathrm{round}}\right)\quad\mathrm{s.t.}\sum_{\ell}z_{\ell}c_{\ell}\leq \mathcal{P}.
$$
标准0-1背包问题，可以用DP求解，复杂度$O(L\mathcal{P})$，内存$O(\mathcal{P})$。
