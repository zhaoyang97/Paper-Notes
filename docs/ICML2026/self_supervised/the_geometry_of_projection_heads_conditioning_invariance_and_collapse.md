---
title: >-
  [论文解读] The Geometry of Projection Heads: Conditioning, Invariance and Collapse
description: >-
  [ICML 2026][自监督学习][投影头] 本文从黎曼几何视角把自监督学习中的投影头分析为可训练的度量张量，证明其作用是动态白化优化景观、用光滑激活的负曲率逃脱坍缩鞍点、并沿数据增强方向诱导度量奇异性——三件事一起解释了"训练时需要、推理时丢弃"这一长期谜团。 经典现象：自监督学习（SSL）的"边训练边丢弃"——训练时加…
tags:
  - "ICML 2026"
  - "自监督学习"
  - "投影头"
  - "黎曼几何"
  - "表示坍缩"
  - "不变性"
---

# The Geometry of Projection Heads: Conditioning, Invariance and Collapse

**会议**: ICML 2026  
**arXiv**: [2605.17180](https://arxiv.org/abs/2605.17180)  
**代码**: 待确认  
**领域**: 自监督学习 / 表示学习理论  
**关键词**: 投影头, 自监督学习, 黎曼几何, 表示坍缩, 不变性

## 一句话总结
本文从黎曼几何视角把自监督学习中的投影头分析为可训练的度量张量，证明其作用是动态白化优化景观、用光滑激活的负曲率逃脱坍缩鞍点、并沿数据增强方向诱导度量奇异性——三件事一起解释了"训练时需要、推理时丢弃"这一长期谜团。

## 研究背景与动机

**经典现象**：自监督学习（SSL）的"边训练边丢弃"——训练时加 MLP 投影头 $h_\phi$，推理时只用骨干 $f_\theta$，把投影头扔掉。这看似矛盾：如果投影头训练必要，为什么推理不用它？

**现有解释的局限**：既有工作用信息瓶颈、维度坍缩防御等做事后描述，但缺乏机制性理解——为什么非线性头能同时滤除信息、加快收敛、逃脱坍缩鞍点？

**核心观察**：投影头本质上是表示流形上的度量学习。对比损失的刚性不变性约束（如颜色不变）会强制网络"毁灭"某些信息，投影头通过改变表示空间的局部几何来吸收这种破坏，保护骨干。

**核心矛盾**：损失函数对增强方向的强约束 vs 骨干表示需要保留下游任务所需的信息丰富度，二者矛盾必须通过中间层吸收。

**本文目标**：用黎曼几何工具把投影头建模为作用在骨干表示流形上的动态度量张量，导出三大几何作用并实证验证。

**核心 idea**：投影头是"一次性预处理器"——白化优化景观；在光滑激活下注入负曲率逃脱坍缩；沿增强方向诱导度量奇异性把增强无关信息推出表示空间。

## 方法详解

### 整体框架
设骨干 $f_\theta: \mathcal{X} \to \mathcal{Z}$，投影头 $h_\phi: \mathcal{Z} \to \mathcal{H}$。定义**有效 Hessian** $H_{\text{eff}}(z) = J_h(z)^\top \nabla_h^2 \mathcal{L} J_h(z) + \sum_i [\nabla_h \mathcal{L}]_i \nabla_z^2 h_i(z)$；第一项是 Gauss-Newton 拉回度量，第二项是投影头内禀曲率驱动的交互项。**增强切空间** $\mathcal{V}_{\text{aug}}(z)$ 是连续增强参数无穷小变化张成的方向——投影头必须在这些方向压缩轨道。

### 关键设计

**1. 线性头视角下的全局 Mahalanobis 白化：投影头本质是度量学习**

要理解投影头干了什么，先从最简单的线性头入手。当 $h(z) = W z$ 时，相似度 $\langle h(z_i), h(z_j) \rangle = z_i^\top (W^\top W) z_j$，等于学了一个全局度量 $M = W^\top W$——投影头从一开始就是在做度量学习。Theorem 3.1 证明存在线性头让有效 Hessian 在 $r$ 维子空间上同构于单位矩阵，即对损失相关子空间做隐式白化，这解释了为什么"加个线性头"就比无头方案收敛更快。

但线性头是全局固定变换，没法适应优化轨迹上不断变化的几何。Proposition 3.3 把这个局限说死：当损失的内禀几何有非零 Riemann 曲率时，不存在任何全局常数线性变换能处处让有效 Hessian 既非退化又各向同性。换言之，非线性头不是锦上添花，而是几何上必需的。

**2. 非线性头的轨迹线性化 + 容量阈值：深度宽度是拓扑必然**

既然线性头不够，非线性头到底强在哪？Theorem 3.2 给出：对任意光滑非自交的优化轨迹 $\gamma(t)$，都存在一个 MLP 头让诱导的有效 Hessian 沿整条轨迹 $\epsilon$-各向同性——它能学到随状态变化的度量，把弯曲的优化景观一路捋直。

但这要求头有足够容量。Proposition 3.4 量化近似误差上界 $\|H_{\text{eff}}^\phi - H_{\text{eff}}^*\|_2 \leq 2 L M \epsilon + M \epsilon^2$，Corollary 3.5 进一步给出阈值：要维持各向同性的条件数，必须 $\epsilon < \lambda_{\min}(H_{\text{eff}}^*) / (2 L M)$，一旦近似误差越过这个阈值，坍缩点就会从可逃脱变成稳定。这意味着"头要多深多宽"不是经验玄学，而是拓扑必然——容量不够就越界，坍缩就被锁死。

**3. 光滑激活通过内禀曲率注入负特征值，逃脱坍缩**

非对比 SSL（BYOL、SimSiam）没有显式负样本却不坍缩，机制一直是谜。本文指出关键在激活函数的曲率。在坍缩配置 $z^*$（所有输入映到常数）下，线性头的交互项 $M(z^*) = 0$（因为 $\nabla^2 \text{linear} = 0$），有效 Hessian 仍是半正定，坍缩成了非排斥的临界区，跑不出来。换成光滑非线性头（Swish、GELU），$\nabla^2 h$ 非零；而损失 Hessian $G(z^*)$ 在高维表示空间的零空间往往非平凡（内禀秩 $r < d$），交互项 $M(z^*)$ 就在这些方向上产生负特征值，把稳定最小值变成严格鞍点——再由 Lee 等的非凸优化理论保证梯度下降几乎必然逃出严格鞍点。

这就把"非对比 SSL 为何不坍缩"归因到几何曲率而非随机噪声，也顺带预言 ReLU 头（$\nabla^2 \text{ReLU} = 0$ 几乎处处）没有这个保证，必须靠离散动力学和 BatchNorm 才行。

**4. 沿增强方向诱导度量奇异性：把"训练用、推理丢"说成理论必然**

前三点解释了头怎么帮训练，但还欠一个问题：既然头对训练这么关键，为什么推理时反而要把它扔掉（即 SimCLR 发现的 guillotine effect）？这一点回答的是标题里的"Invariance"。Proposition 5.2 给出关键：当光滑头真的实现了对连续增强的局部不变性，拉回度量 $G(z) = J_h(z)^\top J_h(z)$ 必然在增强切空间 $\mathcal{V}_{\text{aug}}$ 上奇异——$v^\top G(z) v = 0,\ \forall v \in \mathcal{V}_{\text{aug}}$。也就是说投影头是一个**几何低通滤波器**，把骨干空间 $\mathcal{Z}$ 里沿增强轨道的有限距离一律压成嵌入空间 $\mathcal{H}$ 中的零，等价于把 $\mathcal{Z}$ 换成商空间 $\mathcal{Z}/\mathcal{T}$、把整条轨道坍成一个等价类——满足了 SSL 目标，却是**不可逆的信息丢失**。

Theorem 5.3（信息层级）用 Fisher 信息矩阵把这笔损失量化到秩上：$\text{rank}(\mathcal{I}_{h(z)}) \leq \text{rank}(\mathcal{I}_z) - \dim(\mathcal{V}_{\text{aug}})$，投影输出携带的信息秩比骨干整整少了一个增强维度。因为骨干处在这个奇异度量的**上游**，它保留了数据流形的完整维度（拿色彩做增强训练时，骨干仍留着分猫品种要用的颜色信息，被滤掉的只是头的输出）。所以"丢弃投影头"不是经验玄学，而是要拿回被不变性学习滤掉的可分信息的理论必然——这正是后面消融里"线性探针在骨干上比在头输出上高 14.72 个点"的几何根源。

## 实验关键数据

### 主实验：Hessian 追踪 + 激活函数效应

| 激活 | 初始化 | 条件数行为 | $\lambda_{\min} < 0$ 注入 | 逃脱坍缩 |
|------|-------|----------|------------------------|---------|
| Swish (光滑) | 正常 | 快速峰值后平台 | 是 | ✓ 快速 |
| Swish (伪坍缩) | 坍缩-like | 暴力峰值 $\rho_s = 0.609$ | 是 | ✓ 机制性 |
| ReLU | 正常 | 缓慢、无负特征值 | 否 | ✗ 失败 |
| ReLU | 伪坍缩 | 静态振荡 | 否 | ✗ 需 BN / 大 LR |
| 线性 | 伪坍缩 | 缓慢漂移 | 否 | ✓ 最终逃脱但慢 |

光滑激活主动注入负特征值触发"拓扑相变"，驱动表示方差剧增；ReLU 无此机制，在无 BatchNorm 的连续梯度流下陷入坍缩。

### 消融实验：轨道压缩 + 信息纠缠

| 指标 | 骨干 $z$ | 投影头 $h(z)$ | 比例 | 说明 |
|------|--------|-------------|------|------|
| 轨道均方展度 ($\times 10^{-2}$) | 2.25 ± 1.07 | 0.10 ± 0.06 | 22.5× 压缩 | Prop 5.2 验证：$\mathcal{V}_{\text{aug}}$ 方向度量奇异性 |
| 轨道内距离 $D_{\text{intra}}$ | 0.211 ± 0.045 | 0.044 ± 0.011 | 4.76× 缩 | 增强方向被目标性压缩 |
| 类间距离 $D_{\text{inter}}$ | 0.432 ± 0.052 | 0.111 ± 0.014 | 3.89× 缩 | 语义结构相对保持 |
| $D_{\text{inter}} / D_{\text{intra}}$ | 2.04 ± 0.52 | 2.50 ± 0.72 | 1.22× ↑ | 选择性压缩 |
| 线性探针精度 | 52.27% | 37.55% | -14.72 | 线性不变性的信息代价 |
| MLP 探针精度 | 55.46% | 43.56% | -11.90 | MLP-线性 gap 翻倍：信息纠缠 |

直接验证核心理论：投影头学到的度量 $G(z) = J_h(z)^\top J_h(z)$ 选择性地在增强切空间诱导度量奇异性，同时相对保持语义聚类。MLP 探针优势大幅增长说明信息未被擦除而是被非线性纠缠。

### 关键发现
- **ReLU vs 光滑激活的本质差异**：光滑激活的非零 $\nabla^2 h$ 是逃脱坍缩的关键，ReLU 因二阶导几乎处处为零失效。
- **22.5× 轨道压缩**：直接证明投影头在 $\mathcal{V}_{\text{aug}}$ 方向诱导接近奇异的度量，但语义距离仅缩 3.89×——选择性而非全局压缩。
- **MLP-线性 gap 翻倍**：投影头输出的下游线性可分性下降但 MLP 可分性恢复，说明信息被非线性纠缠而非擦除——支持"丢弃投影头"的最优性。

## 亮点与洞察
- **几何统一框架**：用黎曼度量张量统一解释投影头的三大作用（白化、逃脱坍缩、诱导奇异性），比信息论或优化论单视角更深入。
- **ReLU 理论鸿沟**：发现光滑激活和 ReLU 在连续梯度流下的本质不同，解释为什么实践中用 Swish / GELU 替代 ReLU。
- **度量奇异性的几何解释**：Theorem 5.1–5.3 量化"为什么丢弃投影头"——头诱导 $\mathcal{V}_{\text{aug}}$ 方向度量退化，使骨干表示对下游任务更优。
- **容量阈值定量化**：Proposition 3.4 把"头深 / 宽到多少"从经验启发上升为拓扑定理。

## 局限与展望
- 理论证投影头存在表达容量优化景观，但 SGD 动力学如何到达这些配置仍未知。
- 对 ViT 自注意力诱导的数据相关度量的扩展尚缺。
- 离散增强（翻转等）无光滑群结构，无穷小建模不适用。
- 假设投影头快速平衡，但实际训练中头与骨干的时间尺度耦合未必满足。

## 相关工作与启发
- **vs 信息瓶颈（Tishby）**：两者都说头滤除无关变量；本文补充几何机制（度量奇异性），从"什么被滤除"到"如何滤除"。
- **vs 维度坍缩防御**（Jing 2022；Tian 2021）：前人聚焦 BatchNorm / 梯度停止；本文证明光滑激活的内禀曲率本身就足以逃脱坍缩。
- **vs 自然梯度下降**（Amari 1998）：头学到的动态度量正是自然梯度的几何基础。
- **vs 显式白化**（VICReg、Barlow Twins）：显式白化靠损失约束，隐式白化靠投影头度量；本文统一二者的几何本质。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐  从黎曼几何重新诠释投影头，引入度量张量 / 轨道压缩 / 曲率注入等概念体系，对 SSL 理论是颠覆性创新。
- 实验充分度: ⭐⭐⭐⭐  Hessian 追踪、轨道可视化、基础模型验证完整；离散增强和大规模数据的边界案例缺。
- 写作质量: ⭐⭐⭐⭐⭐  定理陈述精确，证明思路清晰，可视化直观有力。
- 价值: ⭐⭐⭐⭐⭐  解决 SSL 两大经典谜团，为算法设计（激活选择、头深度）提供几何依据。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Why Prototypes Collapse: Diagnosing and Preventing Partial Collapse in Prototypical Self-Supervised Learning](../../ICLR2026/self_supervised/why_prototypes_collapse_diagnosing_and_preventing_partial_collapse_in_prototypic.md)
- [\[ICML 2026\] LimiX-2M: Mitigating Low-Rank Collapse and Attention Bottlenecks in Tabular Foundation Models](limix-2m_mitigating_low-rank_collapse_and_attention_bottlenecks_in_tabular_found.md)
- [\[ICML 2026\] Provable Accuracy Collapse in Embedding-Based Representations under Dimensionality Mismatch](provable_accuracy_collapse_in_embedding-based_representations_under_dimensionali.md)
- [\[CVPR 2026\] Teaching DINOv3 About Partial 3D Geometry: A Self-Supervised Geometry-Aware Approach](../../CVPR2026/self_supervised/teaching_dinov3_about_partial_3d_geometry_a_self-supervised_geometry-aware_appro.md)
- [\[CVPR 2026\] Reframing Long-Tailed Learning via Loss Landscape Geometry](../../CVPR2026/self_supervised/reframing_long-tailed_learning_via_loss_landscape_geometry.md)

</div>

<!-- RELATED:END -->
