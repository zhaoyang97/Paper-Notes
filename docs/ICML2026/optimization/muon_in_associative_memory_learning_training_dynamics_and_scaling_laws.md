---
title: >-
  [论文解读] Muon in Associative Memory Learning: Training Dynamics and Scaling Laws
description: >-
  [ICML2026][优化/理论][Muon 优化器] 本文在带 softmax 检索和分层频谱的线性关联记忆模型上，对 Muon 进行收敛速率与缩放律的理论刻画：相对 GD，Muon 在无噪声情形获得指数级加速，在幂律频谱噪声情形将损失收敛律从 $\tilde{\Omega}(T^{-(1-1/\beta)})$ 提升到 $\tilde{\mathcal{O}}(T^{-2})$，并把这一加速归因于矩阵符号算子等价于一个自适应任务对齐的隐式预条件子。
tags:
  - "ICML2026"
  - "优化/理论"
  - "Muon 优化器"
  - "关联记忆"
  - "矩阵符号算子"
  - "缩放律"
  - "训练动力学"
---

# Muon in Associative Memory Learning: Training Dynamics and Scaling Laws

**会议**: ICML2026  
**arXiv**: [2602.05725](https://arxiv.org/abs/2602.05725)  
**代码**: 未公开  
**领域**: 优化  
**关键词**: Muon 优化器, 关联记忆, 矩阵符号算子, 缩放律, 训练动力学  

## 一句话总结
本文在带 softmax 检索和分层频谱的线性关联记忆模型上，对 Muon 进行收敛速率与缩放律的理论刻画：相对 GD，Muon 在无噪声情形获得指数级加速，在幂律频谱噪声情形将损失收敛律从 $\tilde{\Omega}(T^{-(1-1/\beta)})$ 提升到 $\tilde{\mathcal{O}}(T^{-2})$，并把这一加速归因于矩阵符号算子等价于一个自适应任务对齐的隐式预条件子。

## 研究背景与动机

**领域现状**：现代 LLM 大规模预训练中，矩阵参数优化器已经从 SGD/Adam/AdamW 逐步过渡到由 Jordan 等人提出的 Muon。Muon 在 dense Transformer 和 MoE 等架构上反复表现出在大规模训练 regime 比 AdamW 更高的算力和数据效率，因而被工程界迅速接纳。

**现有痛点**：现有理论文献几乎都把 Muon 当成"标准 stochastic 优化"问题去推一个收敛上界（Bernstein 把 Muon 视为算子范数下的最速下降，后续工作给出梯度范数收敛速率），但这种静态的 worst-case bound 无法解释 Muon 在真实预训练中为什么"更快、更均衡"，更没有给出 Muon 自己的 neural scaling law。

**核心矛盾**：Muon 在矩阵参数上做的是 $\mathrm{msgn}(\mathbf{G})=\mathbf{U}\,\mathrm{sgn}(\boldsymbol{\Sigma})\,\mathbf{V}^\top$ 这一谱归一化，它的本质效应是在低频长尾任务上"放大"步长；而 GD 的有效步长正比于知识频率 $p_j$，因此 GD 的尾部任务收敛 $\sim 1/(p_j t)$ 极慢。要解释 Muon 的优势，必须脱离静态 bound，转而刻画动态训练轨迹中"频率分量学得多快"。

**本文目标**：(1) 在 noiseless 和 label-noise 两种 associative memory 设定下，分别推 Muon 与 GD 的逐子任务损失曲线与总损失曲线；(2) 在幂律频谱下推出 Muon 的 optimization scaling law，并与 GD 的下界对比；(3) 给出一个解释 Muon 加速的机制视角。

**切入角度**：作者用 associative memory 作为可解析的代理模型 —— 知识被组织为 $K$ 个正交 query-answer 对 $(\mathbf{E}_j,\widetilde{\mathbf{E}}_j)$，按 $M$ 组分层频率 $\tilde p_i$ 出现；模型是单矩阵 $\mathbf{W}\in\mathbb{R}^{K\times K}$ 的 softmax 检索。这一框架既忠实地模拟 Transformer 的 factual recall（Geva、Meng 一系工作的实验背书），又把梯度结构清晰拆成"频率 × 残差 × 关联"三项，可以在闭式上跟踪 Muon 的 SVD 演化。

**核心 idea**：Muon 的矩阵符号操作在任务表征基下近似等于单位阵 $\mathbf{I}_K$（即 $\mathrm{msgn}(\mathbf{G}_t)\approx \mathbf{I}_K$），它把 GD 中按频率倾斜的方向偏置"夷平"成各向同性更新，于是高频/低频组以同一速率学习，进而把幂律积分换成 $\mathcal{O}(T^{-2})$ 的快速衰减。

## 方法详解

本文是纯理论刻画，没有提出新算法，"方法详解"对应理论框架的搭建和关键证明思路。

### 整体框架
分析对象是关联记忆下的最小化问题。给定 $K$ 个正交等范数嵌入 $(\mathbf{E}_j,\widetilde{\mathbf{E}}_j)$、频率结构 $p_j=\tilde p_i/C$（$M$ 个频率组，每组 $C=K/M$ 个条目），label noise 水平 $\alpha\in[0,1)$ 诱导出条件分布 $p_{i\mid j}=(1-\alpha)\mathbb{1}[i=j]+\alpha/K$。线性 softmax 模型 $\hat p_{i\mid j}(\mathbf{W})=\frac{\exp(\widetilde{\mathbf{E}}_i^\top \mathbf{W}\mathbf{E}_j)}{\sum_k \exp(\widetilde{\mathbf{E}}_k^\top \mathbf{W}\mathbf{E}_j)}$ 通过最小化 cross-entropy $\mathcal{L}(\mathbf{W})=\mathbb{E}_{\mathcal{D}_\alpha}[-\log\hat p_{i\mid j}(\mathbf{W})]$ 来存储知识。两种优化器分别是 $\mathbf{W}_{t+1}=\mathbf{W}_t-\eta\nabla\mathcal{L}$（GD）和 $\mathbf{W}_{t+1}=\mathbf{W}_t-\eta\,\mathrm{msgn}(\nabla\mathcal{L})$（Muon，省去 momentum，与 Spectral GD 等价），都从零初始化。整套分析在任务表征空间里追踪 $\widehat{\mathbf{W}}_t=\widetilde{\mathbf{E}}^\top \mathbf{W}_t \mathbf{E}$ 与对应梯度 $\mathbf{G}_t=\widetilde{\mathbf{E}}^\top \nabla\mathcal{L} \mathbf{E}$ 的演化。

### 关键设计

**1. 梯度结构分解 + 频率瓶颈刻画：定位 GD 慢在低频长尾**

要解释 Muon 为什么快，先得看清 GD 慢在哪。softmax 模型的梯度可写成三因子相乘

$$\nabla\mathcal{L}(\mathbf{W})=\sum_{i,j} p_j\,(\hat p_{i\mid j}-p_{i\mid j})\,\widetilde{\mathbf{E}}_i\mathbf{E}_j^\top,$$

即"查询频率 $p_j$ × 预测残差 × 嵌入关联"，这是后续所有定理的基础。沿第 $j$ 个分量 GD 的有效步长正比于 $p_j$，于是无噪声下逐子任务损失 $\mathcal{L}_j^{\mathrm{GD}}(t)\eqsim 1/(p_j t)$、总损失 $\eqsim K/t$（Theorem 4.1），长尾类被这个 $p_j$ 因子锁死。Muon 的 $\mathrm{msgn}$ 做谱归一化恰好把 $p_j$ 剥掉，所有子任务以同一指数速率收敛 $\mathcal{L}_j^{\mathrm{Muon}}(t)\eqsim Ke^{-(1+o_K(1))t}$（Theorem 4.2）；要把损失降到目标精度，GD 需 $\mathcal{O}(1/\epsilon)$ 步，Muon 只需 $\mathcal{O}(\log(1/\epsilon))$ 步。

**2. 三阶段动力学 + Muon 缩放律 $\tilde{\mathcal{O}}(T^{-2})$：写出 Muon 自己的 neural scaling law**

有了无噪声的指数收敛，再加 label noise 才能对上真实预训练。Muon 子任务损失呈两阶段：下降阶段 $\sim Ke^{-\eta t}+\eta t$、振荡阶段 $\sim\eta^2+\mathcal{L}_j^\ast$，临界时间 $T_j^\ast=\Theta(\log K/\eta)$（Theorem 5.1）。选 $\eta=\Theta(\log K/T)$ 让下降项 $Ke^{-\eta T}$ 与振荡项 $\eta^2$ 取最优平衡，得 $\mathcal{L}^{\mathrm{Muon}}(T)-\mathcal{L}^\ast\lesssim(\log K/T)^2$（Theorem 5.8）。同一幂律频谱 $\tilde p_i\propto i^{-\beta}$（$\beta>1$）下，GD 逐子任务 $\gtrsim e^{-\eta p_j T}\log K$，对 $j$ 求和并用积分近似 $\int_1^M z^{-\beta}e^{-z^{-\beta}T}\mathrm{d}z\approx T^{-(1-1/\beta)}$ 给下界 $\tilde\Omega(T^{-(1-1/\beta)})$（Theorem 5.7）。两者相比，Muon 的缩放指数 $-2$ 不依赖 $\beta$，而 GD 的 $-(1-1/\beta)$ 在 $\beta\to 1$ 时退化到 0，把总损失降到同一精度 Muon 比 GD 快 $\Omega(C)$ 倍（组大小）——这正是大规模预训练里 Muon 损失-算力曲线更陡的形式化解释。

**3. 预条件视角：$\mathrm{msgn}\approx\mathbf{I}_K$ 是自动找到任务表征基的隐式对齐**

加速的机制可以讲得更透：在任务表征空间里 $\mathrm{msgn}(\mathbf{G}_t)\approx\mathbf{I}_K$，即 Muon 实际在做 $\widehat{\mathbf{W}}_t\approx t\mathbf{I}_K$ 的各向同性对齐更新。归纳证明 Muon 从 $\mathbf{W}_0=\mathbf{0}$ 出发保留频率组诱导的块对称结构（Proposition 6.1），残差分解为 $\mathbf{P}-\widehat{\mathbf{P}}_t=\mathbf{R}_t^+-\mathbf{R}_t^-$，在 $M(C-1)$ 维组内对比子空间上 $\mathrm{msgn}$ 退化为单位阵，仅 $M$ 维块均值方向贡献至多 $M/C$ 偏离，故 $\|\mathrm{msgn}(\mathbf{P}-\widehat{\mathbf{P}}_t)-\mathbf{I}_K\|_{\max}\le 1/C+M/C=o_K(1)$。再对照理想化的 TRA-SignGD（更新 $\widehat{\mathbf{W}}_{t+1}=\widehat{\mathbf{W}}_t-\eta\,\mathrm{sgn}(\mathbf{G}_t)$），Theorem 6.3 证明它用 $\eta$ 能匹配 Muon 用 $2\eta$ 的所有结论。区别就此清楚：SignGD 要在原坐标做符号、必须 oracle 知道未知的 $\mathbf{E},\widetilde{\mathbf{E}}$ 才能对齐，而 Muon 借 SVD 自动找到这组任务表征基、无需 oracle——优势精确归因到 SVD 的自动对齐能力，而非笼统的"矩阵优化更强"。

### 损失函数 / 训练策略
所有理论结果都基于零初始化 $\mathbf{W}_0=\mathbf{0}_{K\times K}$、constant learning rate $\eta$；scaling law 章节取 $\eta=\Theta(\log K/T)$；GD 的稳定性条件 $\eta p_1\lesssim 1$ 由不动点 Jacobian 的 linear stability 给出（Proposition 5.4）。

## 实验关键数据

实验只是 sanity check，用合成长尾分类和 LLaMA-style 预训练来验证理论预测。

### 主实验

| 设定 | 频谱 / 数据 | Muon 行为 | GD 行为 |
|------|-------------|----------|---------|
| Noiseless associative memory | $K$ 个正交知识，分 $M$ 组 | 各子任务同步指数收敛，$\mathcal{L}^{\mathrm{Muon}}\eqsim K e^{-t}$ | 子任务速率 $\propto p_j$，总损失 $\eqsim K/t$，低频组卡死 |
| Noisy power-law spectrum $\tilde p_i\propto i^{-\beta}$ | $\beta>1$ | $\mathcal{L}^{\mathrm{Muon}}(T)-\mathcal{L}^\ast\lesssim (\log K/T)^2$ | $\mathcal{L}^{\mathrm{GD}}(T)-\mathcal{L}^\ast\gtrsim \log K/T^{1-1/\beta}$ |
| LLaMA-style pre-training | 真实长尾文本 | 长尾分类准确率显著高，scaling 曲线斜率更陡 | 收敛慢，长尾类欠学习 |

### 消融实验

| 配置 | 行为 | 说明 |
|------|------|------|
| GD（baseline） | 频率敏感 $1/(p_j t)$ | 低频组卡死，scaling 受 $\beta$ 限制 |
| Normalized GD (NGD) | 比 GD 快但仍不均衡 | 说明加速不仅是步长归一化，必须用矩阵-sign |
| SignGD（原坐标） | 不能利用任务结构 | 需要 oracle $\mathbf{E},\widetilde{\mathbf{E}}$ 才能匹配 Muon |
| TRA-SignGD（理想化对齐） | 用 $\eta$ 匹配 Muon 用 $2\eta$ | 验证 Muon 的优势来自"自动找到任务表征基"这一隐式预条件 |
| Muon（无 momentum） | 同时实现指数加速 + $\tilde{\mathcal{O}}(T^{-2})$ scaling | 矩阵 sign 把 $p_j$ 因子从有效步长里剥掉 |

### 关键发现
- Muon 加速可拆成两部分：谱归一化把更新尺度抹平（NGD 已能做到一部分），以及 $\mathrm{msgn}$ 沿任务表征基的隐式对齐（NGD 做不到），后者才是 $\Omega(C)$ 倍加速的来源。
- label noise 引入的 oscillation 项 $\eta^2$ 与下降项 $Ke^{-\eta T}+\eta T$ 之间的 trade-off 决定了最优 $\eta=\Theta(\log K/T)$，这一调度自然给出了 Muon 的 scaling exponent $-2$。
- GD 的 scaling exponent $-(1-1/\beta)$ 随幂律指数 $\beta$ 趋近 1 而退化到 0，对长尾尤其差；Muon 的 exponent 不依赖 $\beta$，是它在真实语料上 scaling 更陡的根本原因。

## 亮点与洞察
- 第一次给 Muon 写了"自己的" neural scaling law，而不是套用 SGD/SignGD 的 worst-case bound 二次加工；缩放指数 $-2$ 与 GD 的 $-(1-1/\beta)$ 形成了非平凡的对比，把"Muon 在大规模训练时更有效"这条工程经验落到了形式化的渐近律上。
- 关联记忆 + 块对称归纳 + 任务表征空间这套技术组合很可复用：只要梯度有"频率 × 残差 × 关联"的可分解结构，任何"按 SVD 做 sign"的优化器都可以套这个框架去算缩放律和加速倍数，对未来 spectral 方法（Shampoo、SOAP、Spectral GD）都是现成模板。
- TRA-SignGD 这个理想化对照体设计巧妙：它在保留 SignGD 形式的前提下"借给"它任务表征基，证明 Muon 与"对齐后的 SignGD"等价，等于把 Muon 的优势精确归因到 SVD 的自动对齐能力上，而不是模糊地说"矩阵优化更强"。

## 局限与展望
- 模型是单矩阵线性 softmax，没有非线性、没有 MLP、没有 multi-head；从这里推出的 scaling law 严格只覆盖 factual recall 子任务，不能直接外推到生成 token-level 的 perplexity。
- 嵌入做了正交等范数假设（虽然作者声称可松弛到 near-orthogonal），现实预训练的 token embedding 远没有这么干净；power-law 频谱也只是 Zipf 的一阶近似。
- Muon 的实际实现包含 momentum（Newton-Schulz 迭代估计 $\mathrm{msgn}$），本文为了纯化分析把 momentum 砍掉，与生产环境的 Muon 存在 gap；momentum-free 版本同时也是 Spectral GD。
- 可改进方向：把这套块对称分析推广到 (i) 多层 Transformer 的逐层关联记忆叠加，(ii) MoE 的 expert 路由频率，(iii) momentum + 学习率调度下的 scaling law；都是直接的工程化延伸。

## 相关工作与启发
- **vs Bernstein & Newhouse 2024 / Li & Hong 2025 / Pethick et al. 2025**：他们给的是 Muon 的 worst-case stochastic 收敛 bound，本文给的是 problem-specific 训练动力学闭式 + 缩放律，能定量解释加速倍数。
- **vs Wang et al. 2025b（heavy-tailed associative memory）**：那篇是实验工作，发现 Muon 在尾部类强；本文把这一现象证明成 $\Omega(C)$ 倍加速并给出机制（隐式预条件）。
- **vs Kunstner & Bach 2025 / Kim et al. 2026（SignSGD scaling law）**：把 SignGD 类方法的 scaling 分析框架从 bigram 推广到关联记忆，并新增了 $\mathrm{msgn}$ 与 $\mathrm{sgn}$ 在"是否需要 oracle 对齐"上的关键区分。
- **vs Vasudeva et al. 2025（Muon 在 Gaussian mixture 的泛化）**：那篇关注泛化，本文关注优化动力学与 scaling，互补。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Muon 的第一条 neural scaling law，且揭示"$\mathrm{msgn}\approx \mathbf{I}_K$"的隐式对齐机制
- 实验充分度: ⭐⭐⭐ 主要靠合成 + LLaMA 小规模训练验证，缺少大规模 ablation
- 写作质量: ⭐⭐⭐⭐ 定理排版清晰，preconditioning view 一节把直觉讲透
- 价值: ⭐⭐⭐⭐⭐ 直接给后续 spectral 优化器的理论分析提供模板，工程上佐证选 Muon 的 scaling 收益

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Learning Quadratic Neural Networks in High Dimensions: SGD Dynamics and Scaling Laws](../../NeurIPS2025/optimization/learning_quadratic_neural_networks_in_high_dimensions_sgd_dynamics_and_scaling_l.md)
- [\[NeurIPS 2025\] Functional Scaling Laws in Kernel Regression: Loss Dynamics and Learning Rate Schedules](../../NeurIPS2025/optimization/functional_scaling_laws_in_kernel_regression_loss_dynamics_and_learning_rate_sch.md)
- [\[ICML 2026\] Balancing Learning Rates Across Layers: Exact Two-Step Dynamics and Optimal Scaling in Linear Neural Networks](balancing_learning_rates_across_layers_exact_two-step_dynamics_and_optimal_scali.md)
- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](../../NeurIPS2025/optimization/emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [\[ICML 2026\] Mirror Mean-Field Langevin Dynamics](mirror_mean-field_langevin_dynamics.md)

</div>

<!-- RELATED:END -->
