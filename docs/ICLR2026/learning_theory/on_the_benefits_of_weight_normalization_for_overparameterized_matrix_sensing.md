---
title: >-
  [论文解读] On the Benefits of Weight Normalization for Overparameterized Matrix Sensing
description: >-
  [ICLR 2026][学习理论][权重归一化] 本文给出了权重归一化（WN）在过参数化矩阵 sensing 上的首个理论刻画：把矩阵变量按"方向（Stiefel 流形）+ 幅度（对称矩阵）"解耦、配合黎曼梯度下降后，可以在有限样本下做到**线性收敛**（相比无 WN 的次线性下界是指数级加速），而且过参数化程度越高，迭代复杂度和样本复杂度反而**多项式地下降**。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "优化"
  - "权重归一化"
  - "矩阵sensing"
  - "过参数化"
  - "黎曼优化"
  - "鞍点逃逸"
---

# On the Benefits of Weight Normalization for Overparameterized Matrix Sensing

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=JIMM5YLShy](https://openreview.net/forum?id=JIMM5YLShy)  
**代码**: 待确认  
**领域**: 学习理论 / 优化  
**关键词**: 权重归一化, 矩阵sensing, 过参数化, 黎曼优化, 鞍点逃逸

## 一句话总结
本文给出了权重归一化（WN）在过参数化矩阵 sensing 上的首个理论刻画：把矩阵变量按"方向（Stiefel 流形）+ 幅度（对称矩阵）"解耦、配合黎曼梯度下降后，可以在有限样本下做到**线性收敛**（相比无 WN 的次线性下界是指数级加速），而且过参数化程度越高，迭代复杂度和样本复杂度反而**多项式地下降**。

## 研究背景与动机
**领域现状**：归一化（BatchNorm / LayerNorm / WeightNorm）是现代深度网络的标配，实践中能稳定训练、提升泛化。其中权重归一化（WN）把参数解耦成"方向"和"幅度"分别优化，最近因为能无缝接入 LoRA（DoRA 等 PEFT 方法）而重新受到关注。但 WN 为什么有效，理论上一直没说清楚——已有结果只覆盖到过参数化最小二乘的隐式正则、对角线性网络等较简单的场景。

**现有痛点**：要刻画 WN 的好处，需要一个既非凸、损失地形又足够丰富的"试金石"问题。过参数化矩阵 sensing 正是这样一个经典 testbed：目标是从线性测量 $y_i = \mathrm{Tr}(M_i^\top A)$ 中恢复一个低秩 PSD 矩阵 $A \in \mathbb{S}^m_+$，用 Burer-Monteiro 分解 $Y Y^\top \approx A$（$Y \in \mathbb{R}^{m\times r}$）参数化。由于事先不知道真实秩 $r_A$，实践中取 $r > r_A$ 的过参数化设置以保证精确恢复。

**核心矛盾**：过参数化在这里是把双刃剑。已有工作（Xiong et al., 2024）证明，即使在无穷样本（population）下，**普通梯度下降（GD）的收敛速度也存在一个次线性下界** $\Omega(1/t)$——比已知秩时取 $r=r_A$ 的线性速率指数级地慢。也就是说，过参数化非但不帮 GD，反而把它拖慢；而且无 WN 时还必须把随机初始化的幅度精心控制（往往要正比于 $1/\kappa$），否则连收敛都难保证。

**本文目标**：证明 WN 能绕开这个次线性下界，并量化"过参数化越多，收敛越快、样本越省"这件反直觉的事。

**切入角度**：作者注意到 WN 的方向变量天然被约束在光滑流形上（球面 / Stiefel 流形），这正好对应黎曼优化的框架。于是把标量版 WN 推广到矩阵变量，用黎曼梯度下降（RGD）来优化方向，看看解耦后的损失地形会不会变"良性"。

**核心 idea**：用极分解把 $Y = X\tilde\Theta$ 拆成 Stiefel 流形上的方向 $X$ 和对称矩阵幅度 $\Theta$，对 $X$ 跑 RGD、对 $\Theta$ 跑 GD，从而把无 WN 时的次线性收敛**升级成线性收敛**，并让过参数化从"敌人"变成"朋友"。

## 方法详解

### 整体框架
本文不是提出一个新算法去刷 benchmark，而是给"WN + 黎曼优化"在过参数化矩阵 sensing 上建立一套收敛理论。整体逻辑是三步走：**(1) 重参数化**——把原始 Burer-Monteiro 目标 $\min_Y \frac14\|\mathcal{M}(YY^\top)-y\|^2$ 通过极分解改写成方向-幅度解耦的 WN 形式；**(2) 优化器**——方向 $X$ 落在 Stiefel 流形上用 RGD（投影到切空间 + 极收缩 retraction），幅度 $\Theta$ 在对称矩阵空间用普通 GD，交替更新；**(3) 收敛分析**——证明这套迭代会经历"鞍点阶段 → 线性收敛阶段"两个相位，并把过参数化程度 $r$ 作为可调旋钮，量化它对迭代/样本复杂度的多项式改善。

具体地，原始目标里 $Y\in\mathbb{R}^{m\times r}$ 用极分解写成 $Y = X\tilde\Theta$，其中 $X \in \mathrm{St}(m,r) = \{X\in\mathbb{R}^{m\times r}\mid X^\top X = I_r\}$ 是方向（$r$ 维子空间的正交基），$\tilde\Theta \in \mathbb{S}^r_+$ 是幅度。再做两步简化：把 $\tilde\Theta\tilde\Theta^\top$ 合并成单个矩阵 $\Theta\in\mathbb{S}^r_+$，并把 PSD 约束**松弛成仅对称** $\Theta\in\mathbb{S}^r$。这个松弛在过参数化区间下不改变全局目标，却避免了优化 PSD 锥所需的 SVD / 矩阵指数，大幅省算力。最终目标变成

$$\min_{X,\Theta}\; f(X,\Theta) := \tfrac14\big\|\mathcal{M}(X\Theta X^\top)-y\big\|^2,\quad \text{s.t. } X\in\mathrm{St}(m,r),\ \Theta\in\mathbb{S}^r.$$

### 关键设计

**1. 矩阵版权重归一化：用极分解把"方向"与"幅度"解耦**

标量 WN 把向量拆成单位方向加一个标量长度，本文要把它推广到矩阵变量。痛点在于：直接对 $Y$ 跑 GD，损失地形布满鞍点、又缺全局光滑性，过参数化还会把 GD 拖到次线性。作者用极分解 $Y = X\tilde\Theta$ 把这两个角色分离——$X$ 是 Stiefel 流形上的正交基（几何上就是一个 $r$ 维子空间的"朝向"），$\tilde\Theta$ 是这个子空间里的"伸缩量"。关键的工程化处理是把 $\tilde\Theta\tilde\Theta^\top$ 并成对称矩阵 $\Theta$ 并松弛掉 PSD 约束：这一步让幅度变量住在线性空间 $\mathbb{S}^r$ 里、可以用最朴素的 GD 更新，省掉 PSD 投影的 SVD 开销，同时在 $r>r_A$ 时不损失全局最优。解耦之后，方向变量被约束在光滑流形上，正好把问题搬进黎曼优化的成熟工具箱——这是后面所有好处的几何根源。

**2. 方向走 RGD、幅度走 GD 的交替更新**

解耦完要解决"怎么在流形上迭代"。对方向 $X$，先把欧氏梯度投影到 Stiefel 流形在 $X_t$ 处的切空间得到黎曼梯度 $G_t$（形式上 $G_t = (I-X_tX_t^\top)\nabla_X G_t + \tfrac{X_t}{2}(X_t^\top \nabla G_t - \nabla G_t^\top X_t)$，⚠️ 具体投影式以原文 (3) 为准），再用**极收缩（polar retraction）** 把更新后的点拉回流形：

$$X_{t+1} = (X_t - \eta G_t)\,(I_r + \eta^2 G_t^\top G_t)^{-1/2},$$

其中 $\eta$ 是步长。极收缩在这里只为理论简洁，作者实验里验证了 QR、Cayley 等其他收缩在数值上几乎等价。对幅度 $\Theta$ 则直接用 GD：$\Theta_{t+1} = \Theta_t - \mu K_t$，$K_t := \nabla_\Theta f(X_{t+1},\Theta_t)$。这个更新还顺带保证了对称约束的可行性——只要 $\Theta_0\in\mathbb{S}^r$，后续所有 $\Theta_t$ 都自动落在 $\mathbb{S}^r$ 里。整个流程记为 Algorithm 1（即 RGD）。区别于无 WN 的 GD：这里方向被流形约束"扶住"了，不会乱跑，幅度又能独立缩放，二者配合才换来后面的线性速率。

**3. 两阶段收敛证明：先逃鞍点、再线性收敛**

这是全文的理论主菜（Theorem 3.2）。在随机初始化 $X_0\sim\mathrm{St}(m,r)$、测量算子满足 $(r+r_A+1,\delta)$-RIP 的条件下，重构误差 $\|X_t\Theta_t X_t^\top - A\|_F$ 的轨迹被一个 burn-in 时间 $t_0$ 切成两段：

- **鞍点阶段**（$1\le t\le t_0$）：误差单调下降但看似缓慢，因为迭代要穿越/逃离若干鞍点；$t_0$ 有上界 $O\!\big(\frac{\kappa^4 m^4 r^4 r_A^2}{(r-r_A)^8}\big)$。
- **线性收敛阶段**（$t\ge t_0+1$）：误差以线性速率收敛到零，$\|X_t\Theta_t X_t^\top - A\|_F \le 3\big(1 - \tfrac{c_3(r-r_A)^4}{\kappa^4 m^2 r^2 r_A}\big)^{t-t_0}$。

证明的几何直觉来自方向-幅度解耦：用对齐矩阵 $\Phi_t := U^\top X_t$（$U$ 是 $A$ 的紧凑 SVD 左奇异向量），其奇异值正是 $\mathrm{span}(U)$ 与 $\mathrm{span}(X_t)$ 主夹角的余弦。证明的核心是 $\mathrm{Tr}(\Phi_t\Phi_t^\top)\to r_A$，即两个子空间最终对齐。第一阶段 $\mathrm{Tr}(\Phi_t\Phi_t^\top)$ 从近 0 单调爬到 $r_A-0.5$（每步至少涨 $O(\frac{(r-r_A)^8}{\kappa^4 m^4 r^4 r_A})$，对应多项式时间逃鞍）；越过 $r_A-0.5$ 后，对齐误差 $r_A - \mathrm{Tr}(\Phi_t\Phi_t^\top)$ 线性降到 0。把这条线性速率换算成迭代复杂度，就是相对无 WN 次线性下界 $\Omega(1/t)$ 的指数级加速。

**4. 让过参数化从"敌人"变"朋友"**

最反直觉的结论：加参数不但不慢，反而更快、更省样本。设 $r = p\,r_A$（$p>1$），burn-in 时间可改写为 $O\!\big(\frac{\kappa^4 m^4 p^4}{(p-1)^8 r_A^2}\big)$，随 $p$ 多项式下降；线性阶段速率 $\exp\!\big(-O(\frac{(p-1)^4 r_A}{\kappa^4 m^2 p^2}t)\big)$ 也随 $p$ 增大而变快。具体两档对比：轻度过参数化 $r=r_A+c$（$c=O(1)$）时速率约 $\exp(-O(\frac{t}{\kappa^4 m^2 r_A^3}))$；提到 $r=c\,r_A$ 时改善到 $\exp(-O(\frac{r_A t}{\kappa^4 m^2}))$——指数项里最多可获 $O(r_A^4)$ 的改善。统计侧同理：高斯设计下 RIP 以高概率成立要求样本量 $n = O\!\big(\frac{\kappa^4 m^7 r^9 r_A^2}{(r-r_A)^{12}}\big)$，**也随 $r$ 增大而多项式下降**，最多可省 $O(r_A^{12})$ 的因子。这与无 WN 的 GD 形成鲜明对照：后者过参数化只会拖慢，前者却把它变成可调的加速旋钮。

### 损失函数 / 训练策略
优化目标即式 (2) 的非凸最小二乘 $f(X,\Theta)=\frac14\|\mathcal{M}(X\Theta X^\top)-y\|^2$。算法关键超参由 Theorem 3.2 给定：方向步长 $\eta = O\!\big(\frac{(r-r_A)^4}{\kappa^2 m^2 r^2 r_A}\big)$、幅度步长 $\mu=2$、RIP 常数 $\delta = O\!\big(\frac{(r-r_A)^6}{\kappa^2 m^3 r^4 r_A}\big)$。初始化采用真·随机：$X_0 = Z_0(Z_0^\top Z_0)^{-1/2}$（$Z_0$ 各元 i.i.d. $\mathcal{N}(0,1)$），$\|\Theta_0\|\le 2$——这比无 WN 要求"小幅度、正比 $1/\kappa$"的精细初始化宽松得多。

## 实验关键数据

实验全部围绕"验证理论"，用合成数据（$A=U\Sigma U^\top$，高斯测量保证 RIP）和真实图像重构两类任务。核心是 WN+RGD 与无 WN 的 GD 在平方重构误差-迭代曲线上的对比。

### 主实验：WN 的线性收敛 vs GD 的次线性

| 实验设置 | 关键参数 | 现象 | 结论 |
|----------|----------|------|------|
| 不同条件数 $\kappa$（Fig 2a） | $m=10, r=5, r_A=3, n=60000,\ \kappa\in\{50,75,100\}$ | 经过鞍点阶段后 WN+RGD 线性收敛到 0，**与 $\kappa$ 无关**；GD 进入次线性后误差大得多 | 印证 Theorem 3.2 的线性速率 |
| 不同过参数化 $r$（Fig 2b） | $m=300, r_A=5, \kappa=10,\ r\in\{50,75,100\}, n=50000$ | WN 下 $r$ 越大收敛越快、鞍点平台越短；GD 反而随 $r$ 略微变慢 | 印证"过参数化是朋友" |
| 满秩 $r=m$（Fig 2c） | — | WN 在满秩情形收敛极快 | WN 对过参数化极端情形稳健 |

### 鞍点-到-鞍点动力学（Fig 1）

| 观测量 | 现象 | 对应理论 |
|--------|------|----------|
| 平方重构误差（1a） | 出现多级平台，每个平台 = 一个鞍点 | Lemma 4.1 |
| 梯度范数（1b） | 在鞍点处骤降数个量级 | 鞍点的小梯度刻画 |
| $\|X_t\Theta_t X_t^\top - A_\ell\|_F^2$（1c） | 依次贴近最佳 rank-$\ell$ 近似 $A_\ell$ | 增量/序贯学习 |
| $\sigma_i(\Phi_t\Phi_t^\top)$、$\sigma_i(\Theta_t)$（1d–1f） | 方向与幅度都呈序贯学习模式 | $\mathrm{Tr}(\Phi_t\Phi_t^\top)\to r_A$ |

### 关键发现
- **鞍点本质 = 增量学习**：Lemma 4.1 证明 population loss 的鞍点恰好对应 $A$ 的最佳 rank-$\ell$ 近似 $A_\ell$（$X\Theta X^\top = A_\ell$ 且 $\mathrm{Tr}(X^\top UU^\top X)=\ell$），算法是"逐秩"地把 $A_\ell$ 学出来直到恢复完整 $A$，每次逃鞍就是离开某个 $A_\ell$ 的邻域。
- **过参数化加速逃鞍**：burn-in 时间上界 $O(\frac{\kappa^4 m^4 r^4 r_A^2}{(r-r_A)^8})$ 随 $r$ 增大而下降，实验里表现为更短的平台、更早进入线性收敛。
- **收缩方式不敏感**：极收缩只为证明方便，QR / Cayley 等收缩数值表现几乎一致（Fig 3a）；含噪测量下方法依然稳健（Fig 3b）。

## 亮点与洞察
- **把 WN 翻译成黎曼优化**：方向变量天然住在 Stiefel 流形上这一观察，把一个"为什么归一化有用"的经验问题，转成了流形上有成熟工具的收敛分析问题——这是全文最关键的视角转换。
- **"过参数化越多越快"被量化**：以往直觉是过参数化要付出计算/收敛代价，本文却给出 burn-in 时间和样本复杂度都随 $r$ **多项式下降**的明确公式（最多 $O(r_A^4)$ 速率、$O(r_A^{12})$ 样本改善），把模糊的"WN 友好"做成了可算的旋钮。
- **PSD→对称的松弛 trick 可迁移**：把 $\tilde\Theta\tilde\Theta^\top$ 合并并松弛 PSD 约束，在过参数化下不损全局最优却省掉 SVD，这个"用过参数化换约束松弛"的思路在其他流形/锥约束优化里也可能复用。

## 局限与展望
- **局限于矩阵 sensing**：结论建立在对称低秩 PSD 矩阵 sensing + RIP 假设之上，能否推广到非对称、非 PSD、或一般深度网络仍是开放问题；WN 接入 LoRA 的实际收益本文只是动机，未做端到端验证。
- **复杂度常数偏大**：速率与样本复杂度里出现 $m^7 r^9$、$\kappa^4$ 这类高次因子，作为理论刻画足够，但与实际可用的常数还有距离；这些上界是否紧也待考。
- **依赖随机初始化与精确 RIP**：分析是高概率结论，对极端病态（超大 $\kappa$）或 RIP 不成立的测量算子未覆盖；实验规模也偏小（$m$ 最大 300）。
- **改进思路**：把方向-幅度解耦 + RGD 的框架推广到非凸张量分解、或带噪/缺失测量的鲁棒矩阵补全，验证"过参数化是朋友"是否在更广问题上成立。

## 相关工作与启发
- **vs 无 WN 的 GD（Stöger & Soltanolkotabi 2021；Xiong et al. 2024）**：他们证明随机初始化 GD 只能早停得到常数误差、或最佳也只有次线性下界 $\Omega(1/t)$，且过参数化会拖慢收敛、初始化幅度需精细控制；本文用 WN+RGD 做到线性收敛、随机初始化、过参数化反而加速——这是直接的指数级改进。
- **vs WN 的已有理论（Wu et al. 2020；Chou et al. 2024；Cisneros-Velarde et al. 2025）**：前人分析 WN 在过参数化最小二乘的隐式正则、对角线性网络的隐式偏置、或 Hessian 谱范数与泛化；本文首次在非凸的矩阵 sensing 上刻画 WN **如何利用过参数化换取更快收敛**，是收敛速率层面的新结果。
- **vs 黎曼优化（Absil et al. 2008；Boumal 2023）**：本文沿用标准 RGD（切空间投影 + retraction）的记号与工具，贡献不在优化器本身，而在把它和 WN 的解耦结构对接、并完成针对矩阵 sensing 的两阶段收敛证明。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次证明归一化能从过参数化中获益，视角与结论都新。
- 实验充分度: ⭐⭐⭐⭐ 合成+图像实验充分印证理论，但规模偏小、纯为验证而非应用。
- 写作质量: ⭐⭐⭐⭐ 理论叙述清晰、两阶段几何直觉讲得好，公式密度高但结构分明。
- 价值: ⭐⭐⭐⭐ 为 WN / DoRA 类 PEFT 方法提供了难得的理论支撑，testbed 结论有借鉴意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Testing Fourier Sparsity via Implicit Sensing](testing_fourier_sparsity_via_implicit_sensing.md)
- [\[ICLR 2026\] In-Context Algorithm Emulation in Fixed-Weight Transformers](in-context_algorithm_emulation_in_fixed-weight_transformers.md)
- [\[ICLR 2026\] Gradient Descent Dynamics of Rank-One Matrix Denoising](gradient_descent_dynamics_of_rank-one_matrix_denoising.md)
- [\[ICLR 2026\] Improved High-Dimensional Estimation with Langevin Dynamics and Stochastic Weight Averaging](improved_high-dimensional_estimation_with_langevin_dynamics_and_stochastic_weigh.md)
- [\[ICLR 2026\] Closed-form $\ell_r$ norm scaling with data for overparameterized linear regression and diagonal linear networks under $\ell_p$ bias](closed-form_ell_r_norm_scaling_with_data_for_overparameterized_linear_regression.md)

</div>

<!-- RELATED:END -->
