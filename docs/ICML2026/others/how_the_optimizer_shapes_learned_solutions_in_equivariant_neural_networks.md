---
title: >-
  [论文解读] How the Optimizer Shapes Learned Solutions in Equivariant Neural Networks
description: >-
  [ICML2026 (Workshop on Weight-Space Symmetries)][Muon] 本文系统比较 Muon 与 Adam 在等变/几何网络（EGNN、DGCNN、PointNet、GotenNet、GINE）上的训练效果，发现 Muon 在 3D 点云任务上稳定优于 Adam，且收敛到的解在 Hessian 曲率、损失景观局部光滑度、权重/表征谱秩三个维度上都呈现显著不同的结构性差异——把"优化器选择"重新定位为等变网络训练里被严重忽视的一个 inductive bias。
tags:
  - "ICML2026 (Workshop on Weight-Space Symmetries)"
  - "Muon"
  - "Adam"
  - "等变性"
  - "Hessian"
  - "谱秩"
  - "ModelNet40"
---

# How the Optimizer Shapes Learned Solutions in Equivariant Neural Networks

**会议**: ICML2026 (Workshop on Weight-Space Symmetries)  
**arXiv**: [2605.27662](https://arxiv.org/abs/2605.27662)  
**代码**: 无  
**领域**: 等变神经网络 / 优化器 / 损失景观分析  
**关键词**: Muon, Adam, 等变性, Hessian, 谱秩, ModelNet40

## 一句话总结
本文系统比较 Muon 与 Adam 在等变/几何网络（EGNN、DGCNN、PointNet、GotenNet、GINE）上的训练效果，发现 Muon 在 3D 点云任务上稳定优于 Adam，且收敛到的解在 Hessian 曲率、损失景观局部光滑度、权重/表征谱秩三个维度上都呈现显著不同的结构性差异——把"优化器选择"重新定位为等变网络训练里被严重忽视的一个 inductive bias。

## 研究背景与动机

**领域现状**：等变神经网络通过把几何对称性直接写进架构（如 EGNN 的 $E(n)$ 等变、DGCNN 的动态 k-NN 几何图、PointNet 的对称池化）来获得 inductive bias，是几何深度学习的主流路线。但实验中人们越来越承认：硬等变约束让优化变得非常困难，损失景观中存在大量临界点甚至 spurious minima（Xie & Smidt 2025），等变网络在 scale 上还经常打不过更宽松的对手（Xie et al. 2025; Brehmer et al. 2025）。

**现有痛点**：针对这个问题，社区给出的回答几乎都是"放松架构约束"——approximate equivariance（Wang et al. 2022）、relaxed equivariance（Pertigkiozoglou et al. 2024; Manolache et al. 2025; Elhag et al. 2025）等等。所有这些工作都把**架构**当成优化问题的中心，而把**优化器**当成黑盒。

**核心矛盾**：但 Pascanu et al. (2025) 等近期研究强调：不同优化器不仅收敛速度不同，更会引导网络收敛到*定性不同*的解。如果等变网络训练难是个优化问题，那为什么没人换个优化器试试？

**本文目标**：在不动架构的前提下，把 Adam 替换成 Muon（Jordan et al. 2024 提出的、用 Newton-Schulz 迭代把动量正交化的新优化器），测一测优化器单独的作用有多大，并分析它把解推到了损失景观的什么地方。

**切入角度**：Muon 的核心机制是对 2D 参数的动量缓存做正交化更新，从设计上就会"提拔小而重要的方向"，与 Adam 自适应学习率的"自动均衡"形成对比；作者推测这种谱级别的差异会与等变约束发生有趣的相互作用。

**核心 idea**：换优化器（Adam → Muon）就能在等变网络上稳定提点；并且 Muon 的解在 Hessian 谱、损失局部几何、权重/表征谱秩上都更"展开"，提示我们用"防止谱集中"作为等变优化器的设计原则。

## 方法详解

本文不是新算法论文，而是经验研究 + 损失景观分析。

### 整体框架

研究流程分三步：(1) 在 ModelNet40 / ModelNet40-C（3D 点云分类）和 QM9 / Peptides-func / ZINC（分子学习）上对 Adam 与 Muon 做严格的网格搜索 + 4 seed 重复，得到统一的精度比较；(2) 用 Hessian 估计（power iteration 算 top eigenvalue，Hutchinson 估计算 trace）和 2D 损失切片（Li et al. 2018 风格）刻画两类解的局部几何；(3) 用 stable rank 与 effective rank 两个统计量描述训练后权重矩阵与中间表征的谱结构。

### 关键设计

**1. 架构覆盖：横跨三档等变强度，确保效应不是单一架构的偶然**

如果只在一个架构上换优化器看到提升，很难排除"那只是这个网络的特性"。作者刻意把三档等变强度都纳进比较：EGNN（显式 $E(n)$ 等变）当最硬等变端，DGCNN（动态 k-NN 图、仅置换等变 + 局部几何）当中等档，PointNet（对称池化、完全置换不变）当最弱端，另用 GotenNet（$E(3)$ 等变 Transformer）跑分子任务、GINE（置换等变消息传递）跑图任务。为保公平，Muon 用作者默认设置（Newton-Schulz 迭代次数、谱缩放常数），Adam 用同样的 (lr × wd) 网格搜索范围。这么设计是因为等变强弱直接决定参数空间的隐藏对称多寡（Xie & Smidt 2025 指出隐参数对称会撕裂损失景观）——若优化器效应真和等变性耦合，跨档对比就该看到不同程度的提升，而结果也确实呈"等变越强、Muon 越受益"。

**2. 损失局部几何刻画：Hessian 摘要 + 2D 切片双管齐下，避免单维度误导**

要回答"Muon 把解推到了景观的什么地方"，单看一种视角容易上当。作者对每个 4-seed 训练好、最接近平均精度的 checkpoint 同时做两件事：用 autograd Hessian-vector product 跑 power iteration 算最大特征值 $\lambda_{\max}$、用 Hutchinson（Rademacher 探针）估计 trace；再按 Li et al. (2018) 在两个滤波归一化方向上画 2D 损失等高线看局部形状。两者必须一起看是有道理的——损失切片只是低维投影、容易误导，而 Hessian 摘要又会被 Dinh et al. (2017) 指出的"参数对称下 sharpness 不是函数级不变量"污染。把两者并置，才能同时呈现"Muon 解的局部曲率反而更大"与"切片上看起来更光滑"这对看似矛盾的事实，从而避免单维度结论的偏差。

**3. 谱结构分析：stable rank 与 effective rank 量化谱有多集中**

最后要验证"优化器是一种 spectral inductive bias"这个猜想，就得量化奇异值分布的集中程度。作者对每个权重矩阵 $W$（以及每层中间激活特征矩阵）算两个量：stable rank $\|W\|_F^2/\|W\|_2^2=\sum_i\sigma_i^2/\sigma_1^2$，effective rank $\exp(H(p))$（$p_i=\sigma_i/\sum_j\sigma_j$，$H$ 为 Shannon 熵），两者都落在 $[1,\mathrm{rank}(W)]$，谱越均匀值越大；表征则在中间层做点级 mean-pool、最后一层用各架构原生池化。这个度量直指要害：梯度下降被广泛报告有"低秩隐式偏好"（Arora et al. 2019），而 Muon 的正交化动量恰恰对小奇异方向做 rescale。若观察到 Muon 训练后的权重与表征谱真的更展开，就给"优化器作为 spectral inductive bias"提供了直接证据，也与 Dong et al. (2021) 关于"纯注意力指数级掉到 rank-1"的退化失败模式形成呼应。

### 训练协议
对每个数据集 × 优化器组合做 (learning rate, weight decay) 网格搜索，选出最优配置后用 4 个 seed 重复，按 best-checkpoint 报告均值 ± std。这避免了把"Muon 帮你少调一次参"误读为"Muon 本质更强"的混淆。

## 实验关键数据

### 主实验：ModelNet40 与 ModelNet40-C 分类精度

| 设置 | 架构 | Adam | Muon | $\Delta$ |
|------|------|------|------|----------|
| Clean | EGNN | 76.91 ± 0.94 | **82.08 ± 0.36** | **+5.17** |
| Clean | PointNet | 84.53 ± 0.70 | **87.21 ± 0.39** | **+2.67** |
| Clean | DGCNN | 87.10 ± 0.69 | **89.06 ± 0.17** | **+1.96** |
| Corrupted | EGNN | 65.76 ± 0.95 | **70.12 ± 0.10** | **+4.36** |
| Corrupted | PointNet | 72.85 ± 1.05 | **75.87 ± 0.28** | **+3.02** |
| Corrupted | DGCNN | 75.26 ± 1.63 | **77.84 ± 0.27** | **+2.58** |

Muon 在三种几何归纳偏置不同的架构上一致提点，且 std 普遍更小（说明 Muon 的解更稳定）；最硬等变的 EGNN 提升最大（+5.17%），最弱等变的 DGCNN 提升最小（+1.96%），呈现出"等变越强、Muon 越受益"的趋势。QM9（GotenNet）上 Muon 在 11/12 个目标上更优（如 r2 从 0.4320 降到 0.2310），但 GINE 在 Peptides-func / ZINC 上 Muon 优势消失甚至变差，说明优势主要集中在 3D $SE(3)$-style 等变任务上。

### 二级实验：ModelNet40 checkpoint 上的 Hessian 估计

| 度量 | 架构 | Adam | Muon | 比值 |
|------|------|------|------|------|
| Top eigenvalue | EGNN | 27.14 | 128.83 | **4.75×** |
| Top eigenvalue | PointNet | 32.75 | 714.49 | **21.82×** |
| Top eigenvalue | DGCNN | 12.14 | 136.23 | **11.22×** |
| Trace | EGNN | 402.37 | 1472.78 | **3.66×** |
| Trace | PointNet | 482.61 | 7362.05 | **15.25×** |
| Trace | DGCNN | 184.47 | 1218.74 | **6.61×** |

Muon 解的 Hessian 曲率（$\lambda_{\max}$ 与 trace）反而比 Adam 大 4–22 倍——这直接否定了"Muon 收敛到更平坦的解"这种最直观的解释。

### 关键发现
- **"光滑切片 vs 高曲率"的悖论**：2D 损失切片上 Muon 周围明显比 Adam 更平滑（PointNet 案例最显著），但 Hessian 数值告诉我们这只是低维投影错觉；按 Dinh et al. (2017) 的警告，参数对称重参数化下 sharpness 本就不是函数级不变量，因此优化器改变的是"checkpoint 在景观里的位置"而非"那个位置的本质曲率"。
- **谱结构反低秩偏好**：Adam 留下集中谱（低秩隐式偏好），Muon 的权重 stable rank 与 effective rank 在所有 ModelNet40 架构的多数层上更高，EGNN 上每一层都更高；表征谱也呈现同样趋势，EGNN 最后一层的 effective rank 比值约 2×。这与 Jordan et al. (2024) 设计 Muon 的初衷（rescale "rare directions"）一致。
- **3D vs 图的差异**：所有显著提升都在 3D 点云 / 分子（涉及 $SE(3)$ 等变），而图的纯置换等变任务上 Muon 没优势，提示优化器与几何 inductive bias 的相互作用是任务/对称类型相关的，而非通用 free lunch。
- **EGNN 提升最大**：最硬等变的 EGNN 提点幅度最大（+5.17% Clean、+4.36% Corrupted），与 Xie & Smidt (2025) 关于"隐参数对称把损失景观切成多个区域"的发现呼应——Muon 似乎能更可靠地走到更优的那块区域。

## 亮点与洞察
- **把"优化器"重新放回等变网络研究的中心**：整个 relaxed equivariance 文献都在改架构，本文反向操作（架构不动只换优化器）就拿到 +2~5% 稳定提升，提示后续社区不应再忽视优化器选择这一独立维度。
- **"高曲率 + 光滑切片"是个值得追的现象**：单维度 sharpness 度量已被 Dinh et al. (2017) 否定，本文揭示"切片光滑度"与"Hessian 曲率"完全可以反向变动，给后续设计"对参数对称鲁棒"的 sharpness/flatness 度量提供了具体实证案例。
- **谱秩可能是等变优化的关键指标**：Muon 在权重和表征两个层级都让谱展开，且与精度提升正相关；这把"优化器设计原则"从"自适应学习率/动量"具体到了"防止谱集中"，是个可以指导新优化器设计的可操作目标。
- **任务/对称类型敏感性**：3D 等变赢、图等变没赢，说明"等变 + 优化器"组合需要按对称群类型分类讨论，是个明确可延伸的研究方向（不同对称群应该需要不同的优化器谱行为）。

## 局限与展望
- **只是 workshop short paper**：实验范围聚焦在 ModelNet40 + 少量分子数据，scale 离 Brehmer et al. (2025) 讨论的"等变是否在 scale 上仍 matter"还差一个量级。
- **机制分析停留在 post-hoc**：所有谱、Hessian 分析都在 checkpoint 上做，没有跟踪训练动力学；"Muon 何时何地把解推开"的过程性证据缺失。
- **优化器超参公平性**：尽管做了 (lr, wd) 网格搜索，但 Muon 还有 Newton-Schulz 迭代次数、谱缩放常数等独立超参没在搜索范围内，可能仍存在调参敏感性。
- **图任务上 Muon 失效未解释**：Peptides-func / ZINC 上 Muon 没优势，作者承认观察但没给机制猜想，是否与置换等变的 GIN 特殊优化几何有关、还是数据规模问题，尚待后续工作。
- **无因果链**：谱秩、Hessian、精度三者只有相关性观察，没有干预实验（如人为约束谱秩看是否影响精度）来证明因果方向。

## 相关工作与启发
- **vs Pertigkiozoglou et al. (2024) / Manolache et al. (2025) / Elhag et al. (2025)**：他们都通过 relaxing equivariance 改变训练问题来帮 SGD 找到更好极小点，本文则不改训练目标只换优化器；两条路线是正交互补的，原则上可以叠加使用。
- **vs Xie & Smidt (2025)**：他们指出等变损失景观因为隐参数对称而被切成多区，本文经验地展示了 Muon 似乎更倾向走到"好那块"区域，提供了 Xie & Smidt 理论框架的一个实证 corollary。
- **vs Jordan et al. (2024)**：Muon 原论文是通用优化器，本文把它具体应用到等变网络场景，并报告了"orthogonalized momentum 防低秩"这条机制在等变模型上同样成立的证据。
- **vs Dinh et al. (2017)**：本文的"高 Hessian + 光滑切片"是 Dinh et al. 关于"sharpness 在参数对称下不可靠"论断的一个具体新案例，适合作为后续 sharpness-aware 度量改进研究的对照样本。
- **vs Arora et al. (2019)**：他们论证梯度法的低秩隐式偏好，本文实证 Muon 的正交化动量打破了这一偏好，是对该研究线的优化器侧反例。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把 Muon 系统地放进等变网络做对比、并把谱秩 + Hessian 摆到一起分析，是首次
- 实验充分度: ⭐⭐⭐ 4 seed + 网格搜索做得规范，但 scale 偏小，图任务结论不强
- 写作质量: ⭐⭐⭐⭐ 短文写得紧凑，结论与不确定性都坦诚交代
- 价值: ⭐⭐⭐⭐ 把"优化器"重新引入等变网络研究议程，给后续优化器设计指出明确方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Identifiable Equivariant Networks are Layerwise Equivariant](identifiable_equivariant_networks_are_layerwise_equivariant.md)
- [\[ICML 2025\] Permutation Equivariant Neural Networks for Symmetric Tensors](../../ICML2025/others/permutation_equivariant_neural_networks_for_symmetric_tensors.md)
- [\[NeurIPS 2025\] Learning (Approximately) Equivariant Networks via Constrained Optimization](../../NeurIPS2025/others/learning_approximately_equivariant_networks_via_constrained_optimization.md)
- [\[ICML 2026\] On the Epistemic Uncertainty of Overparametrized Neural Networks](on_the_epistemic_uncertainty_of_overparametrized_neural_networks.md)
- [\[NeurIPS 2025\] On Universality Classes of Equivariant Networks](../../NeurIPS2025/others/on_universality_classes_of_equivariant_networks.md)

</div>

<!-- RELATED:END -->
