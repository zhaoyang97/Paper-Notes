# A Theoretical Framework for Grokking: Interpolation followed by Riemannian Norm Minimisation

**会议**: NEURIPS2025  
**arXiv**: [2505.20172](https://arxiv.org/abs/2505.20172)  
**代码**: 无  
**领域**: others  
**关键词**: grokking, weight decay, gradient flow, Riemannian optimization, implicit regularization, two-timescale dynamics  

## 一句话总结

本文从纯优化角度严格证明了 grokking 现象的成因：带小 weight decay 的梯度流在 $\lambda\to 0$ 极限下呈现两阶段动力学——先快速收敛到训练损失的临界流形 $\mathcal{M}$，再在 $t\approx 1/\lambda$ 时沿流形做黎曼梯度流以最小化 $\ell_2$ 范数，从而延迟实现泛化。

## 研究背景与动机

1. **Grokking 现象**：Power et al. (2022) 首次报道——训练损失迅速降至零，测试损失却在很长一段 plateau 后才突然下降。这一现象在 modular addition、图像分类、矩阵分解等任务中均被观测到。
2. **现有解释不足**：先前工作（Liu et al. 2022, Lyu et al. 2023）将 grokking 归因于 lazy → rich regime 的转变，但缺乏对慢漂移阶段的严格优化理论刻画，且依赖特定架构（齐次参数化、大初始化）的假设。
3. **Weight decay 的角色存疑**：grokking 通常在有 weight decay 时更明显、更早出现，但也有无 weight decay 的报道（分类隐式偏置）。理论上缺乏对 weight decay 如何驱动 norm 下降→泛化提升的严格分析。
4. **插值流形上的漂移**：Li et al. (2021) 利用 Katzenberger 框架研究了 SGD 噪声驱动的漂移，但那是随机效应；本文发现的是**确定性**机制（正则化驱动）。
5. **技术缺口**：已有分析要么将 Katzenberger 结果作为黑箱，要么仅处理初始化已在流形附近的情况。本文需要从任意初始化出发，刻画快慢阶段的完整衔接。
6. **回归 vs. 分类**：在回归任务中，没有 weight decay 就无法观测到 grokking（无界动力学无隐式偏置），这进一步说明 weight decay 的核心作用。

## 方法详解

### 整体框架

考虑足够光滑的损失函数 $F:\mathbb{R}^d\to\mathbb{R}_+$，正则化梯度流：

$$\dot{w}^\lambda(t) = -\nabla F(w^\lambda(t)) - \lambda w^\lambda(t)$$

在 $\lambda\to 0$ 极限下，证明轨迹 $w^\lambda(t)$ 可分解为两个耦合动力学阶段。

### 三个核心设计

**1. 快动力学（Proposition 1）**

- 在任意有限时间区间 $[0,T]$ 上，正则化梯度流 $w^\lambda$ 一致收敛到无正则化梯度流 $w^{\mathrm{GF}}$
- 技术工具：Grönwall 不等式，得到误差界 $\|w^\lambda(t)-w^{\mathrm{GF}}(t)\|\leq \lambda t e^{ct}\sup_{K}\|w\|$
- 数学含义：训练损失快速降至零，但此时参数 norm 大、泛化差（类似 lazy/NTK regime）

**2. 慢动力学（Proposition 2）**

- 时间重标 $\tilde{w}^\lambda(t)=w^\lambda(t/\lambda)$ 后，在 $[\varepsilon,T]$ 上 $\tilde{w}^\lambda$ 一致收敛到黎曼梯度流 $\tilde{w}^\circ$：
$$\dot{\tilde{w}}^\circ(t)=-\mathrm{grad}_{\mathcal{M}}\ell_2(\tilde{w}^\circ(t)), \quad \tilde{w}^\circ(0)=\Phi(w_0)$$
- 其中 $\mathrm{grad}_{\mathcal{M}}\ell_2(w)=P_{\mathrm{Ker}(\nabla^2 F(w))}(w)$ 是 $\ell_2$ 范数在临界流形 $\mathcal{M}$ 上的黎曼梯度
- 关键引理：流映射 $\Phi$ 的微分 $D\Phi_w$ 恰好是 Hessian 零空间的正交投影（利用 Li et al. 2021 的结果）
- 证明核心：$D\Phi(w)\cdot\nabla F(w)=0$，因此正则化项中的 $\frac{1}{\lambda}\nabla F$ 被消去，只保留 norm 梯度

**3. 快慢衔接（Lemma 2）**

- 构造衔接时间 $t(\lambda)=-\frac{\lambda\ln\lambda}{2c}$，使得 $\tilde{w}^\lambda(t(\lambda))\to\Phi(w_0)$
- 保证慢动力学的初始条件从无正则化流的极限点 $\Phi(w_0)\in\mathcal{M}$ 出发
- 技术上确保两阶段在极限下无缝对接

### 关键假设与损失函数

- **Assumption 1**（正则性）：$F$ 是 $\mathcal{C}^3$ 且在 o-minimal 结构中可定义（这保证有界梯度流收敛），涵盖多项式、指数等所有常见架构
- **Assumption 2**（Morse-Bott 性质）：临界流形 $\mathcal{M}$ 是光滑子流形，Hessian 的非零特征值下界为 $\eta>0$
- 这两个假设是标准的：覆盖所有使用可微激活函数的神经网络和 Transformer

### 收敛终点刻画

- **Proposition 3**：$w^\lambda$ 的极限点包含在 $\min_{w\in\mathcal{M}}\|w\|_2^2$ 的 KKT 点中
- **Proposition 4**：若黎曼流收敛到严格局部极小，则 $\lim_{\lambda\to 0}\lim_{t\to\infty}w^\lambda(t)=w^\star$
- Grokking 的本质 = 两个极限不可交换：$\lim_{\lambda\to 0}\lim_{t\to\infty}\neq\lim_{t\to\infty}\lim_{\lambda\to 0}$

## 实验

| 实验设定 | 训练损失 | 测试损失 | Grokking | 核心发现 |
|---|---|---|---|---|
| 线性回归 $F(w)=\|Xw-y\|^2$ | 快速→0 | plateau→下降 | ✓ | 慢阶段收敛到 $w^\star=X^+y$（最小范数解），可解析求解 |
| 低秩矩阵补全 ($20\times20$, rank 3) | 快速→0 @ $t\approx 1$ | 高plateau→$t\approx 10^2$开始下降 | ✓ | 慢阶段最小化 $\|U\|_F^2+\|V\|_F^2$（等价于核范数），奇异值从全部大→仅3个非零 |
| 双层 ReLU 网络 ($m=100$, $n=10$) | 快速→0 @ $t_2=1$ | 高plateau→$t_3\approx 10^5$下降 | ✓ | norm 缓慢降低促进简单函数（少 kink），$\lambda=10^{-3}$ |
| 对角线性网络 | 快速→0 | plateau→下降 | ✓ | 慢阶段促进稀疏估计器 |

**关键发现：**

1. **Grokking 转变并非突然的**：在线性时间尺度上，下降持续约 $O(1/\lambda)$ 时间，与 plateau 长度可比；只有在对数时间尺度上才显得"突然"
2. **初始化尺度的作用**：大初始化→$\Phi(w_0)$ norm 大→grokking 明显；小初始化→已是低 norm→无 grokking
3. **矩阵补全实验**验证了理论预测：前3个奇异值收敛到真实值 $\sigma_1^\star,\sigma_2^\star,\sigma_3^\star$，其余趋近于零
4. ReLU 网络虽不满足 $\mathcal{C}^3$ 假设，实验仍完美符合理论预测的两阶段行为

## 亮点

1. **理论优雅**：将 grokking 统一为纯优化现象（无需统计假设），核心结论清晰——快阶段=无正则化流，慢阶段=流形上黎曼 norm 最小化
2. **证明自足**：避免了 Katzenberger (1990) 的重型随机微分方程工具，给出基于 Falconer (1983) 的简洁确定性证明
3. **泛化性强**：假设温和（$\mathcal{C}^3$ + o-minimal + Morse-Bott），不限定特定架构或初始化分布
4. **洞察深刻**："grokking 不突然"的观点纠正了文献中的常见误解；对 SAM 等其他正则化的延伸分析有启发
5. **快慢衔接完整**：$t(\lambda)=-\lambda\ln\lambda/(2c)$ 的衔接时间构造填补了先前工作的技术空白

## 局限性

1. **仅处理回归**：假设无正则化流有界，排除了分类任务中参数发散的情况，而分类正是 grokking 最初被发现的场景
2. **渐近分析 $\lambda\to 0$**：实际中 $\lambda$ 是固定值（如 $10^{-3}$），论文未给出有限 $\lambda$ 下的定量保证（仅在附录给出启发式讨论）
3. **需要 $\mathcal{C}^3$ 光滑性**：排除了 ReLU 等不可微激活（虽然实验表明结论仍成立）
4. **Morse-Bott 假设**：要求临界流形处 Hessian 非零特征值有正下界，退化点的处理留作 open problem
5. **纯优化无泛化保证**：norm 减小→泛化提升只是经验性的关联，并未从理论上证明测试损失的下降
6. **离散时间/SGD**：分析基于连续梯度流，SGD 的噪声效应和有限学习率未被涵盖

## 相关工作

- **Power et al. (2022)**：首创 grokking 术语，Transformer + modular addition + weight decay
- **Lyu et al. (2023)**：grokking = lazy→rich regime 转变，齐次参数化 + 大初始化，无慢漂移刻画
- **Kumar et al. (2024)**：类似 lazy→rich 视角，强调无 weight decay 也可 grokking
- **Liu et al. (2022, OmniGrok)**：提出直觉性解释（weight decay 驱动 norm 下降），但缺乏理论证明——本文正式化了这一直觉
- **Li et al. (2021)**：SGD 达到插值流形后的噪声驱动漂移（Katzenberger 框架），本文发现的是**确定性正则化驱动**的漂移
- **Chizat et al. (2020)**：infinite-width 两层网络的隐式偏置，早期已观察到延迟泛化
- **Fatkullin & Vanden-Eijnden (2010)**：小随机扰动下能量景观中的漂移 SDE，本文给出更简洁的确定性版本

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将 grokking 从特定架构/任务的经验观察提升为通用的两时间尺度优化定理，黎曼梯度流视角新颖
- 实验充分度: ⭐⭐⭐ — 线性回归/矩阵补全/ReLU/对角网络覆盖广，但均为合成任务，缺少真实数据集和大规模实验
- 写作质量: ⭐⭐⭐⭐⭐ — 行文极其清晰，直觉+形式化+实验的三重对照出色，sketch of proof 非常易读
- 价值: ⭐⭐⭐⭐ — 为理解 weight decay 的深层作用提供了坚实的理论基础，对 implicit regularization 和 SAM 等方向有广泛启发
