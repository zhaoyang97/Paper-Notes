---
title: >-
  [论文解读] RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization
description: >-
  [ICML 2026][优化/理论][Preconditioning] 本文基于 Transformer 层级 Hessian 的「行块对角占优」结构，把 Muon 优化器里昂贵的 Newton-Schulz 正交化换成一次行级 $\ell_2$ 归一化，将每步预条件复杂度从 $\mathcal{O}(mn\min(m,n))$ 降到 $\mathcal{O}(mn)$，在 GPT-2 / LLaMA 预训练上 wall-clock 提速 13–44×、ppl 不降反略升。
tags:
  - "ICML 2026"
  - "优化/理论"
  - "Preconditioning"
  - "Muon"
  - "Newton-Schulz"
  - "行归一化"
  - "Transformer"
---

# RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization

**会议**: ICML 2026  
**arXiv**: [2603.20527](https://arxiv.org/abs/2603.20527)  
**代码**: 论文正文提到「Our code is available at this link」  
**领域**: 优化算法 / LLM 预训练  
**关键词**: Preconditioning, Muon, Newton-Schulz, 行归一化, Transformer Hessian

## 一句话总结
本文基于 Transformer 层级 Hessian 的「行块对角占优」结构，把 Muon 优化器里昂贵的 Newton-Schulz 正交化换成一次行级 $\ell_2$ 归一化，将每步预条件复杂度从 $\mathcal{O}(mn\min(m,n))$ 降到 $\mathcal{O}(mn)$，在 GPT-2 / LLaMA 预训练上 wall-clock 提速 13–44×、ppl 不降反略升。

## 研究背景与动机
**领域现状**：Adam/AdamW 一类对角预条件器虽然便宜，但忽略参数间相关性；K-FAC、Shampoo 等用 Kronecker 分解结构捕捉矩阵级曲率；最近的 Muon 用 Newton-Schulz 迭代 $D_t \approx (V_tV_t^\top)^{-1/2}V_t$ 隐式实现 $H^{-1}$ 而不需要显式求逆，已成为大模型预训练里 AdamW 的强力竞争者。

**现有痛点**：Muon 每步要做 5 次矩阵乘的 Newton-Schulz 多项式逼近，复杂度 $\mathcal{O}(mn\min(m,n))$，对宽矩阵（$m,n$ 都很大）开销迅速变成训练瓶颈 —— GPT-2 1.5B 每 100 步光预条件就要 36.65 秒。

**核心矛盾**：Muon 是按「对 $V_tV_t^\top$ 做完整谱重整」设计的，但近期工作（Zhang et al., Dong et al.）发现 Transformer 层级 Hessian 实际上行块对角占优 —— 即只有对角块（同一行内交互）显著，跨行交互几乎可忽略；这意味着 Muon 花了大量算力去拟合一个根本「几乎为对角」的结构。

**本文目标**：构造一个与 Muon 同复杂度的等价近似，但只保留行级对角块，从而在不损失优化质量的前提下把复杂度压到线性级。

**切入角度**：作者从 K-FAC 形式 $H_{\text{MUON}}=(V_tV_t^\top)^{1/2}\otimes I_n$ 出发，假设只需保留对角元 $\operatorname{diag}(V_tV_t^\top)$，并实测 Transformer 训练中 Gram 矩阵 $V_tV_t^\top$ 的「对角/非对角幅值比」$r_{\min},r_{\text{avg}},r_{\max}$ 长期保持 > 1 且随模型增大持续上升，验证了上述假设。

**核心 idea**：把 Newton-Schulz $(V_tV_t^\top)^{-1/2}V_t$ 替换为简单的「行向量除以行 $\ell_2$ 范数」—— 这等价于用 $(\operatorname{diag}(V_tV_t^\top))^{-1/2}\otimes I_n$ 作为预条件器，恰好对应 Hessian 的行块对角近似。

## 方法详解

### 整体框架
RMNP 与 Muon 在算法骨架上几乎一模一样：每步 (i) 取小批量梯度 $G_t=\nabla f(W_t;\xi^t)$；(ii) 维护一阶动量 $V_t=\beta V_{t-1}+(1-\beta)G_t$；(iii) 预条件得到下降方向 $D_t$；(iv) 更新 $W_{t+1}=W_t-\eta_t D_t$。区别只在第 (iii) 步：Muon 用 5 次 Newton-Schulz 迭代 $D_t=\operatorname{NS}_5(V_t)\approx(V_tV_t^\top)^{-1/2}V_t$；RMNP 用 $D_t=\operatorname{RN}(V_t)=(\operatorname{diag}(V_tV_t^\top))^{-1/2}V_t$，即直接对动量矩阵的每一行 $V_{t,i:}$ 做 $V_{t,i:}/\|V_{t,i:}\|_2$。整体上 RMNP 沿用 Muon 的混合策略 —— 矩阵参数用 RMNP，非矩阵参数（embedding/biases/norm）继续用 AdamW，并配合两套学习率 $\text{lr}_{\text{AdamW}}$ 与 $\text{lr}_{\text{Matrix}}$。

### 关键设计

**1. 行级 $\ell_2$ 归一化预条件器：用一次逐行除范数等价实现"按 Hessian 对角块缩放"**

Muon 的瓶颈在于它按"对 $V_tV_t^\top$ 做完整谱重整"设计，而 Newton–Schulz 多项式逼近要 $\mathcal{O}(mn\min(m,n))$。RMNP 的出发点是 Muon 的 K-FAC 形式 $H_{\text{MUON}}=(V_tV_t^\top)^{1/2}\otimes I_n$：既然只需保留对角块，那就丢掉所有非对角块得到 $H_{\text{RMNP}}=(\operatorname{diag}(V_tV_t^\top))^{1/2}\otimes I_n$，其逆预条件作用在动量 $V_t$ 上恰好是

$$\big[D_t\big]_{i,:}=\frac{V_{t,i:}}{\sqrt{(V_tV_t^\top)_{ii}}}=\frac{V_{t,i:}}{\|V_{t,i:}\|_2},$$

也就是标准的行 $\ell_2$ 归一化。整套实现只剩"按行求平方和、开方、除"三个 op、不含任何矩阵乘，复杂度从 $\mathcal{O}(mn\min(m,n))$ 直接降到 $\mathcal{O}(mn)$，却仍保留了 row-wise（而非 element-wise）的矩阵级自适应。它和 LMO 框架下的 row-normalized 优化器（SRON、SCALE、SWAN、Mano、MOGA）形式一致，但 RMNP 是从 Hessian 结构推出来的，而非 worst-case norm。

**2. 基于 Hessian 行块占优的合理性验证：把"两者等价"从直觉变成可度量的实证现象**

"扔掉非对角块"凭什么不掉精度？作者把这件事做成可观测的指标：对 Gram 矩阵 $V_tV_t^\top$ 定义逐行对角占优比 $r_i\triangleq(V_tV_t^\top)_{ii}/(\frac{1}{m-1}\sum_{j\ne i}|(V_tV_t^\top)_{ij}|)$，聚合成 $r_{\text{avg}},r_{\min},r_{\max}$，在 GPT-2 Small/Medium/Large 和 LLaMA 60M/130M/350M 上跟踪整个训练。结果是 warm-up 后这三个指标全部稳定在 $>1$，且模型越大对角占优越显著（GPT-2 Small 上 $\bar r_{\text{avg}}\approx4.9$、$\bar r_{\max}\approx60$）。这一步的意义在于补上传统 LMO/steepest-descent 分析回答不了的问题——它们只能给 worst-case 保证、解释不了"为什么这种 norm 对神经网络好"，而从真实损失景观结构出发就能说清：Transformer 层级 Hessian 本就行块对角占优，Muon 花在"跨行修正"上的算力其实是在拟合一个几乎为对角的结构。

**3. 非凸收敛性的几何匹配证明：在与算法几何对齐的光滑性下拿到和 Muon 同阶的复杂度**

要说服人"便宜不等于劣等"，得证明 RMNP 在同等理论尺度下不掉精度。作者引入混合范数 $\|W\|_{1,2}=\sum_i\|W_{i,:}\|_2$ 和 $\|W\|_{\infty,2}=\max_i\|W_{i,:}\|_2$（满足 $|\langle A,B\rangle|\le\|A\|_{1,2}\|B\|_{\infty,2}$），给出三种光滑性 + 准则组合下的收敛：Theorem 5.5 在 Frobenius-Lipschitz、以 $\|\nabla f\|_F$ 为准则下是 $\mathcal{O}(m^2 L_F\sigma^2\Delta\epsilon^{-4})$；Theorem 5.7 换 $\|\nabla f\|_{1,2}$ 准则仍是 $\mathcal{O}(m^2)$；最关键的 Theorem 5.9 在 $L_{\infty,2}$-光滑下给出 $\mathcal{O}(mL_{\infty,2}\sigma^2\Delta\epsilon^{-4})$ 的 $\mathcal{O}(m)$ 维度依赖——与 Muon 在核范数光滑下的最优复杂度一致、达到非凸随机优化的 minimax 下界。之所以专挑 $\|\cdot\|_{\infty,2}$ 光滑性来证，是因为它恰好和 RMNP 的行归一化几何对齐，这正是 RMNP 能在便宜的同时维持精度的理论根因。

### 损失函数 / 训练策略
标准 LLM 预训练 CE loss。优化器侧：cosine annealing schedule + 10% warmup；AdamW 部分 $\beta=(0.9,0.95)$，weight decay 0.1；矩阵部分单独搜 $\text{lr}_{\text{Matrix}}$。仅对矩阵参数应用 RMNP，embedding / lm-head / biases / layer-norm 仍走 AdamW。

## 实验关键数据

### 主实验

| 模型 | 数据 | Muon ppl | RMNP ppl | RMNP 相对 AdamW |
|------|------|----------|----------|-----------------|
| GPT-2 Small (125M) | OpenWebText 5B tok | -- | $\Delta$=-0.04 | -1.37 |
| GPT-2 Medium (355M) | OpenWebText 10B tok | -- | -0.07 | -1.49 |
| GPT-2 Large (770M) | OpenWebText 20B tok | -- | -0.24 | -0.84 |
| LLaMA-60M | C4 1B tok | -- | -0.63 | -4.33 |
| LLaMA-130M | C4 2B tok | -- | -0.28 | -1.10 |
| LLaMA-350M | C4 6B tok | -- | -0.02 | -- |

**预条件 wall-clock 时间（100 步, 单 RTX Pro 6000, batch 16）**

| 模型规模 | Muon (s) | RMNP (s) | 加速 |
|----------|----------|----------|------|
| 60M | 1.480 | 0.115 | 12.9× |
| 125M | 2.975 | 0.201 | 14.8× |
| 355M | 7.380 | 0.401 | 18.4× |
| 770M | 27.070 | 0.611 | 44.3× |
| 1.3B | 30.570 | 0.783 | 39.0× |
| 1.5B | 36.650 | 0.855 | 42.9× |

### 消融实验

| 配置 | 现象 | 说明 |
|------|------|------|
| 完整 RMNP (行 $\ell_2$) | ppl 与 Muon 持平甚至略低 | 主结果 |
| 仅看对角占优指标 $r_i$ | $r_{\min}>1$ 持续整个训练 | 行块对角占优假设成立 |
| 模型放大（60M→1.5B） | $r_{\text{avg}}, r_{\max}$ 持续上升 | 越大模型越偏对角，RMNP 越合理 |
| 2× 训练 budget | 优势保持 | RMNP 不只是早期更快 |
| 同时应用到 LM-head / Embedding | 见 D.4 | 提供进一步效率空间 |

### 关键发现
- 复杂度差距随模型放大而拉大：60M 上 Muon 预条件只 1.48s，RMNP 提速 12.9×；到 1.5B 时 Muon 涨到 36.65s 而 RMNP 仍 < 1s，提速 42.9×。在 ≥1B 模型上 Newton-Schulz 已经成为 end-to-end 训练的真实瓶颈。
- ppl 不仅没掉，反而在多数 scale 上小幅优于 Muon —— 说明对 Transformer 而言 Newton-Schulz 的「跨行」修正可能是无效甚至有害的过拟合。
- 三个理论 Theorem 一起证明：RMNP 在 $\|\cdot\|_{\infty,2}$ 光滑这种「与算法几何匹配」的设定下能拿到 $\mathcal{O}(m)$ 维度复杂度，与 Muon 的核范数分析对偶。

## 亮点与洞察
- 用 Hessian 结构反向指导优化器设计 —— 不依赖 worst-case norm，而是看「神经网络真实长什么样」，这是相比 LMO 框架推导 row-norm 工作（SCALE/SWAN/Mano/MOGA）的关键论证升级。
- 行 $\ell_2$ 归一化两行代码就能替掉 Muon 里几十行 Newton-Schulz —— 易用性上几乎零成本，可直接 drop-in。
- 理论部分给出三种 norm 组合下的统一收敛分析，特别是 $\|\cdot\|_{\infty,2}$ smoothness + $\|\cdot\|_{1,2}$ 准则的几何匹配，为「矩阵级优化器选哪种 norm」提供了模板。
- $r_{\min},r_{\text{avg}},r_{\max}$ 这种逐行对角占优度量可作为「该不该用 row-norm 优化器」的诊断工具迁移到其他架构。

## 局限与展望
- 实验主要集中在 GPT-2 与小型 LLaMA（最高 1.5B），尚未在主流 70B+ 上验证；理论的几何假设是否在 MoE、Mamba 等架构上仍成立未明。
- 行块对角占优是「Transformer 现象」，对 CNN / GNN 的 Hessian 是否同样成立、RMNP 是否还能 drop-in 仍待回答。
- 实验仅在预训练上做，没有覆盖 SFT/RLHF 等后训练阶段。
- 没有处理嵌入层 / LM-head 这种非方形且行数极大的矩阵的最佳归一化轴选择，附录里有初步消融但未给统一推荐。

## 相关工作与启发
- **vs Muon**: 思想一致（矩阵级自适应），但 RMNP 显式利用 Transformer Hessian 结构，把预条件操作从 $\mathcal{O}(mn\min(m,n))$ 砍到 $\mathcal{O}(mn)$。
- **vs Shampoo / K-FAC**: 两者都是 Kronecker 分解的对角块近似，但需要显式构造并求逆 $L,R$；RMNP 仅靠隐式动量统计绕过显式矩阵。
- **vs SCALE / SWAN / Mano / MOGA**: 同样是 row/column 归一化，但前人是从 LMO/steepest descent 的 worst-case 视角推导；RMNP 从 Hessian 结构推导，并给出第一份「跟 Muon 同阶」的非凸收敛证明。
- 启发：对其他「实现复杂但结构上很冗余」的优化器（如 Shampoo），可用类似「测一下 Hessian / Gram 是否真的稠密」的实证手段去寻找廉价等价物。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把行归一化与 Transformer Hessian 结构挂钩，并给出与 Muon 等价的理论复杂度
- 实验充分度: ⭐⭐⭐⭐ GPT-2 + LLaMA 多 scale + 预条件 wall-clock + 对角占优度量，覆盖完整
- 写作质量: ⭐⭐⭐⭐ 算法图清晰、动机一句话讲透；理论部分较密集需慢读
- 价值: ⭐⭐⭐⭐⭐ 大模型预训练可直接 drop-in，省 13–44× 预条件时间，工程价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] On the Provable Suboptimality of Momentum SGD in Nonstationary Stochastic Optimization](on_the_provable_suboptimality_of_momentum_sgd_in_nonstationary_stochastic_optimi.md)
- [\[ICML 2026\] Learning-Augmented Scalable Linear Assignment Problem Optimization via Neural Dual Warm-Starts](learning-augmented_scalable_linear_assignment_problem_optimization_via_neural_du.md)
- [\[AAAI 2026\] ECPv2: Fast, Efficient, and Scalable Global Optimization of Lipschitz Functions](../../AAAI2026/optimization/ecpv2_fast_efficient_and_scalable_global_optimization_of_lipschitz_functions.md)
- [\[ACL 2025\] ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting](../../ACL2025/optimization/scalebio_bilevel_data_reweighting.md)
- [\[ICLR 2026\] High Probability Bounds for Non-Convex Stochastic Optimization with Momentum](../../ICLR2026/optimization/high_probability_bounds_for_non-convex_stochastic_optimization_with_momentum.md)

</div>

<!-- RELATED:END -->
