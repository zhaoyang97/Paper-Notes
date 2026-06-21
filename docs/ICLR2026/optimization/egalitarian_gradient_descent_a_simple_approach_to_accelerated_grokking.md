---
title: >-
  [论文解读] Egalitarian Gradient Descent: A Simple Approach to Accelerated Grokking
description: >-
  [ICLR 2026][优化/理论][Grokking] 论文把 grokking 的长平台归因到梯度谱严重不均衡，并提出 EGD：在不改变梯度主方向的前提下把各奇异方向更新速度拉平，从而将“先记忆后泛化”的延迟显著压缩到很少 epoch。 领域现状：grokking 现象在 modular arithmetic、spars…
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "Grokking"
  - "梯度谱归一化"
  - "Fisher 预条件"
  - "随机SVD"
  - "泛化延迟"
---

# Egalitarian Gradient Descent: A Simple Approach to Accelerated Grokking

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=wCnHeql3ow](https://openreview.net/forum?id=wCnHeql3ow)  
**代码**: https://github.com/asahebpa/Egalitarian-Gradient-Descent (官方)  
**领域**: optimization  
**关键词**: Grokking, 梯度谱归一化, Fisher 预条件, 随机SVD, 泛化延迟  

## 一句话总结
论文把 grokking 的长平台归因到梯度谱严重不均衡，并提出 EGD：在不改变梯度主方向的前提下把各奇异方向更新速度拉平，从而将“先记忆后泛化”的延迟显著压缩到很少 epoch。

## 研究背景与动机
**领域现状**：grokking 现象在 modular arithmetic、sparse parity 等任务上已被反复观察到，典型曲线是训练集很快到接近 100%，测试集却长时间停在随机水平，随后突然跃升。已有解释包括 kernel escape、表示竞争、数值稳定性边缘等，但对“为什么会出现超长平台”缺乏一个可操作、可干预的统一优化视角。

**现有痛点**：工程上最直接的问题不是最终能不能泛化，而是泛化来得太晚。即便最终测试精度很高，训练过程也要浪费大量迭代在平台期。像 Grokfast 这类方法能加速，但引入历史梯度缓冲和调参负担，不够轻量。

**核心矛盾**：作者认为关键矛盾是梯度在不同主方向上的尺度差异过大，导致优化在“快方向”迅速收敛，在“慢方向”极慢推进；而后者往往正对应决定泛化跃迁的结构性特征。换句话说，模型不是学不会，而是某些必要方向推进速度被谱条件数拖慢。

**本文目标**：
1. 给出一个可解析 toy setup，证明平台长度与各向异性参数（如 $\varepsilon$）直接相关；
2. 构造一个尽量简单、可直接替换 SGD 的更新规则，让主方向推进速度一致；
3. 在 parity/modular arithmetic 及更实际数据上验证“更快 grok 且不降最终性能”。

**切入角度**：论文从梯度矩阵的奇异值分解出发，不改奇异向量（方向信息），只改奇异值（速度信息）。如果把每个主方向的步长“均平化”，就能减少 ill-conditioned dynamics 造成的滞后。

**核心 idea**：把每层梯度 $G$ 变换为 $\tilde G=(GG^\top)^{-1/2}G$，使所有奇异值归一，从而在主方向上实现“egalitarian（平权）”更新速度。

## 方法详解

### 整体框架
EGD 可看作“对梯度做谱归一化后再更新参数”的插件式步骤。一次训练迭代中，前向、反向与常规训练完全一致，仅在参数更新前对每层梯度矩阵做变换。输入是原始梯度 $G\in\mathbb{R}^{m\times p}$，输出是重标度后的 $\tilde G$，再交给原优化器（SGD/Adam 等）执行步进。

这个框架不依赖具体任务结构，论文主打的是“优化动力学”而非模型结构创新，因此它在 MLP/CNN/Transformer 上都可插入。作者也强调，EGD 在检测到已 grok 后可以关闭，后续退回 vanilla 更新以降低额外计算。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
	A[反向传播得到每层梯度 G] --> B[计算 Fisher 近似 F = GG^T]
	B --> C[谱归一化: ˜G = F^{-1/2}G]
	C --> D[用 ˜G 替换原梯度更新参数]
	D --> E[监控验证集: 若已 grok 可关闭 EGD]
```

### 关键设计

**1. 梯度平权化：保持方向，统一速度**

核心变换是
$$
	ilde G=(GG^\top)^{-1/2}G.
$$
若 $G=USV^\top$，则 $\tilde G=UV^\top$。这意味着左右奇异向量保持不变（仍沿原先“有意义”的主方向优化），但奇异值全部变为 1。作者的观点是：grokking 平台的根因并非“方向错了”，而是“某些方向太慢”；因此只校正速度尺度而不破坏方向结构，是最小侵入且有效的改法。

**2. 与自然梯度的关系：不是同一个算法，但共享几何动机**

论文给出关系式
$$
	ilde G = F^{1/2}\bar G,\quad \bar G=F^{-1}G,
$$
其中 $\bar G$ 是自然梯度形式。EGD 不是直接做 NGD，而是“白化版”更新：它不会像 NGD 那样直接按 $F^{-1}$ 强烈重权，而是以奇异值拉平为目标。这一设计在实践里更稳定，也更贴近“让每个主方向等速前进”的目标。

**3. 工程可落地：精确 SVD + 近似 RSVD 双路径**

EGD 的主要额外成本是每步 SVD。作者给出两条缓解路径：一是达到目标验证精度后关掉 EGD；二是用 randomized SVD 近似，利用梯度常见的低秩结构减少耗时。实验显示，适当 rank 下 RSVD 往往在 wall-clock 上优于 full SVD，同时仍明显快于 vanilla SGD 的 grok 时刻。

### 一个完整示例
以 modular addition（$p=97$）为例：vanilla SGD 通常先在很早阶段把训练精度拉高，但测试精度要经过大量 epoch 才跳升；EGD 将梯度谱等化后，慢方向不再被压制，测试曲线在早期就开始同步上升。

从“动力学”角度看，这相当于把原本近似
$$
A\approx\begin{bmatrix}1-\eta m_2 & 0\\0&1-\eta\varepsilon\end{bmatrix}
$$
的各向异性收敛（慢模态由 $\varepsilon\ll1$ 决定）替换成更接近统一衰减的形式，平台长度不再被 $1/\varepsilon$ 级别拖住。

### 损失函数 / 训练策略
论文主体实验使用标准任务配置：
- Sparse parity：两层 ReLU 网络，hinge loss + weight decay；
- Modular arithmetic：两层 ReLU 网络，cross-entropy + weight decay；
- 对比方法包含 vanilla SGD、EGD、以及更简化的 column normalization。

重要的是，EGD 不要求新增训练调度器或复杂历史缓存；其超参负担主要来自 RSVD 的截断秩选择，而 full SVD 版本几乎是“零新超参”。

## 实验关键数据

### 主实验
论文主结论在三类任务上高度一致：EGD 显著提前 grok，且最终精度不低于基线。

| 任务 | 现象（Vanilla SGD） | 现象（EGD） | 结论 |
|---|---|---|---|
| Modular Addition ($p=79,97,127$) | 长平台后突增 | 很少 epoch 内快速跃升 | 平台大幅缩短 |
| Modular Multiplication ($p=79,97,127$) | 训练先收敛，测试滞后明显 | 测试几乎同步上升 | 泛化延迟显著缓解 |
| Sparse Parity ($(n,k)=(400,2),(100,3),(50,4)$) | 延迟 grok 明显 | 早期即进入高测试精度 | 对困难组合任务同样有效 |

### 消融实验
附录给出 modular addition 到 95% 精度的效率对比（含 wall-clock），可概括为下表。

| 方法 | 到 95% 所需 epoch（相对 Vanilla） | 到 95% 时间（相对 Vanilla） | 额外代价 |
|---|---:|---:|---|
| Vanilla SGD | 1.0x | 1.0x | 无 |
| EGD (SVD) | 约 45x-53x 更少 | 约 10x-14x 更快 | 每步 SVD |
| EGD (RSVD, 合适 rank) | 约 23x-45x 更少 | 常优于 full SVD | 需选 rank |
| Column Norm | 约 12x-16x 更少 | 有时最快 | 加速明显但通常弱于 EGD |

### 关键发现
- 最稳定的结论是“epoch 维度上的大幅提前 grok”，这与论文的谱动力学解释直接对齐。
- 在 wall-clock 维度，full SVD 并非总最优；RSVD 可在速度与效果间给出更实用折中。
- 一个有价值的工程观察是：即便简化到 column normalization，也能明显优于 vanilla，说明“降低梯度谱不均衡”本身就是有效方向。

## 亮点与洞察
- 从“现象描述”推进到“可干预机理”。论文把 grokking 平台与梯度谱条件数联系起来，并给出可执行更新规则，而不是停留在后验解释。
- 方向与速度解耦的视角很实用。保留奇异向量、只重标度奇异值，使方法兼顾稳定性与泛化需求。
- 理论与工程闭环做得较完整。toy model 给出可解析结论，主实验与附录再证明该结论在更复杂场景并非失效。
- 与 Grokfast 的关系讲得清楚。EGD 在“抑制快方向、提升慢方向影响”上与其有共同归纳偏置，但实现更轻、记忆开销更低。

## 局限与展望
- 计算成本仍是现实约束。即使 RSVD 缓解了开销，在大模型高频更新场景下每步谱分解依然昂贵。
- 理论主干目前依赖简化设置。对深层非线性网络的严格收敛与泛化界还不完善，论文也把这列为后续方向。
- 论文强调“加速 grok”，但对不同数据噪声、标签污染、极端小 batch 的鲁棒性分析仍不充分。
- 未来可探索与 AdamW、Muon、学习率重热等策略的联合调度，形成“前期 EGD 拉起泛化、后期常规优化精修”的混合范式。

## 相关工作与启发
- **vs Grokfast (Lee et al., 2024)**：Grokfast 通过低通滤波放大慢梯度分量，效果强但依赖历史缓冲与超参；EGD 直接做谱归一化，更简洁、更省内存，且有更明确的“等速主方向”解释。
- **vs Natural Gradient (Amari, 1998)**：两者都与 Fisher 几何相关，但 EGD 目标是奇异值均衡，不是标准 NGD 的全逆预条件，实践重心在“消除方向速度不平等”。
- **vs Muon 系列**：Muon 与 EGD 都在更新几何上趋向正交/等谱，但 Muon 偏经验工程路线；EGD 从 grokking 机制推导而来，解释性更强。
- **对实践的启发**：当训练曲线出现“训练早收敛、测试长平台”时，与其只调学习率/正则，不如直接检查梯度谱并考虑做低成本谱归一化。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 通过“梯度平权化”解释并干预 grokking，思路简洁且有辨识度。  
- 实验充分度: ⭐⭐⭐⭐☆ 主任务证据扎实，附录含效率与更现实场景；但仍以中小规模问题为主。  
- 写作质量: ⭐⭐⭐⭐☆ 理论-算法-实验链条清晰，和相关工作的定位较到位。  
- 价值: ⭐⭐⭐⭐⭐ 对“如何缩短泛化延迟”给出低侵入、可复用的优化插件，实践潜力高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] On the Convergence Direction of Gradient Descent](on_the_convergence_direction_of_gradient_descent.md)
- [\[ICLR 2026\] On the Convergence Behavior of Preconditioned Gradient Descent Toward the Rich Learning Regime](on_the_convergence_behavior_of_preconditioned_gradient_descent_toward_the_rich_l.md)
- [\[ICLR 2026\] Corner Gradient Descent](corner_gradient_descent.md)
- [\[ICLR 2026\] Fast Data Mixture Optimization via Gradient Descent](fast_data_mixture_optimization_via_gradient_descent.md)
- [\[ICLR 2026\] Provably Accelerated Imaging with Restarted Inertia and Score-based Image Priors](provably_accelerated_imaging_with_restarted_inertia_and_score-based_image_priors.md)

</div>

<!-- RELATED:END -->
