---
title: >-
  [论文解读] ASAG: Toward the Frontiers of Reliable Diffusion Sampling via Adversarial Sinkhorn Attention Guidance
description: >-
  [AAAI 2026][注意力引导] 提出 ASAG（Adversarial Sinkhorn Attention Guidance），从最优传输理论角度重新解读扩散模型中的自注意力分数，通过 Sinkhorn 算法在注意力层中注入对抗性传输代价来故意降低 query-key 相似度，从而破坏误导性注意力对齐并提升条件/无条件采样质量，方法轻量、即插即用、无需重训练。
tags:
  - "AAAI 2026"
  - "注意力引导"
  - "最优传输"
  - "Sinkhorn算法"
  - "扩散采样"
  - "即插即用"
---

# ASAG: Toward the Frontiers of Reliable Diffusion Sampling via Adversarial Sinkhorn Attention Guidance

**会议**: AAAI 2026  
**arXiv**: [2511.07499](https://arxiv.org/abs/2511.07499)  
**代码**: 无  
**领域**: 扩散模型 / 图像生成  
**关键词**: 注意力引导, 最优传输, Sinkhorn算法, 扩散采样, 即插即用

## 一句话总结
提出 ASAG（Adversarial Sinkhorn Attention Guidance），从最优传输理论角度重新解读扩散模型中的自注意力分数，通过 Sinkhorn 算法在注意力层中注入对抗性传输代价来故意降低 query-key 相似度，从而破坏误导性注意力对齐并提升条件/无条件采样质量，方法轻量、即插即用、无需重训练。

## 研究背景与动机

**领域现状**：扩散模型通过引导方法（如 Classifier-Free Guidance, CFG）提升生成质量。CFG 的核心思路是"通过故意降低无条件输出来增强条件输出"，即构造一个"更差的"参考点，让条件路径在对比中更加突出。后续方法（如 PAG、SAG 等）沿用这一思路，但使用启发式扰动函数（如 identity mixing、模糊条件等）来构造降质输出。

**现有痛点**：现有引导方法的扰动函数缺乏理论基础。为什么用 identity mixing 能起作用？为什么模糊条件是好的降质策略？这些选择是人工设计的，缺乏可解释性和最优性保证。不同任务可能需要不同的扰动策略，手动设计效率低。

**核心矛盾**：需要一种有原则的、理论驱动的方法来构造最优的注意力降质策略，而不是依赖启发式设计。

**本文目标**：从最优传输（Optimal Transport）的角度为注意力引导提供理论基础，并设计一种有原则的降质策略。

**切入角度**：作者观察到扩散模型的自注意力机制本质上可以看作一个 OT 问题——query 和 key 之间的注意力分数对应传输方案中的耦合矩阵，softmax 归一化对应边际约束。

**核心 idea**：用 Sinkhorn 算法（求解熵正则化 OT 的标准算法）在注意力层中注入对抗性传输代价——故意增加 query-key 之间的传输成本，从而系统性地破坏注意力对齐，构造出有理论保证的降质输出。

## 方法详解

### 整体框架
ASAG 作为一个即插即用模块嵌入扩散模型的采样过程。在每个去噪步骤中，对自注意力层的 query-key 相似度矩阵注入对抗性代价，通过 Sinkhorn 迭代调整注意力分布，生成降质的引导信号。该信号与标准条件输出结合，产生增强的生成方向。整个过程不修改模型权重。

### 关键设计

1. **注意力的最优传输解读**:

    - 功能：为注意力机制提供数学框架
    - 核心思路：将自注意力 $A = \text{softmax}(QK^\top / \sqrt{d})$ 视为最优传输中的耦合矩阵。$Q$ 和 $K$ 分别对应两个分布的支撑点，注意力权重 $A_{ij}$ 表示将"信息"从位置 $j$ 传输到位置 $i$ 的量。softmax 行归一化对应 OT 的行边际约束。从这个视角看，标准注意力在最小化传输成本（最大化 query-key 相似度）
    - 设计动机：这一解读将"扰动注意力"转化为"增加传输成本"的优化问题，提供了理论框架

2. **对抗性 Sinkhorn 代价注入**:

    - 功能：有原则地降低注意力对齐质量
    - 核心思路：在注意力相似度矩阵 $S = QK^\top / \sqrt{d}$ 上注入对抗性代价矩阵 $C$，使得修改后的注意力为 $\tilde{A} = \text{Sinkhorn}(S - \lambda C)$，其中 $\lambda$ 控制降质强度。代价矩阵 $C$ 通过最大化像素级 query-key 不相似度来设计——即对每对 $(i,j)$，$C_{ij}$ 正比于 $q_i$ 和 $k_j$ 的余弦相似度（越相似的 pair 受到越大的惩罚）。Sinkhorn 算法保证修改后的注意力仍然满足双随机约束（行列和为 1），保持注意力的数学性质
    - 设计动机：直接添加噪声到注意力会破坏其概率性质（如非负性、归一性）。Sinkhorn 算法在增加传输成本的同时保持了注意力矩阵的合法性。"惩罚高相似度 pair"精确瞄准了最有信息量的注意力连接

3. **自适应引导尺度**:

    - 功能：根据去噪阶段动态调整降质强度
    - 核心思路：在早期去噪步（全局结构形成阶段）使用较大的 $\lambda$ 值强降质，在晚期步（细节生成阶段）逐渐降低 $\lambda$。这种 schedule 避免了晚期过度扰动导致的细节模糊
    - 设计动机：不同去噪阶段对引导强度的需求不同——早期需要强引导确定全局布局，晚期需要弱引导保持细节

### 损失函数 / 训练策略
ASAG 完全不需要训练。它在推理时作为即插即用模块工作：给定任何预训练扩散模型（如 Stable Diffusion、SDXL），在采样过程中替换标准自注意力为 adversarial Sinkhorn attention。主要超参数为降质强度 $\lambda$ 和 Sinkhorn 迭代次数。

## 实验关键数据

### 主实验：文本到图像生成
在 COCO-30K 和 PartiPrompts 上与多种引导方法对比（Stable Diffusion v1.5 / SDXL）：

| 方法 | FID ↓ | IS ↑ | CLIP Score ↑ | Human Pref. ↑ |
|---|---|---|---|---|
| CFG (baseline) | 12.8 | 32.4 | 0.312 | - |
| PAG | 11.9 | 33.8 | 0.318 | 42.3% |
| SAG | 12.1 | 33.2 | 0.316 | 38.7% |
| **ASAG (Ours)** | **11.2** | **34.6** | **0.321** | **51.8%** |

### 下游应用增强

| 应用 | 基线 | +ASAG | 提升 |
|---|---|---|---|
| IP-Adapter (CLIP-I) | 0.784 | 0.812 | +3.6% |
| ControlNet Canny (FID) | 18.3 | 16.7 | -1.6 |
| ControlNet Depth (FID) | 19.1 | 17.4 | -1.7 |
| Unconditional (FID) | 15.6 | 14.1 | -1.5 |

### 消融实验

| 配置 | FID ↓ | 说明 |
|---|---|---|
| ASAG (full) | 11.2 | 完整方法 |
| w/o Sinkhorn (直接噪声) | 12.4 | 不用 Sinkhorn，退化到随机扰动 |
| w/o 对抗性代价 | 12.0 | 均匀代价而非对抗性 |
| w/o 自适应 schedule | 11.8 | 固定 λ |

### 关键发现
- **ASAG 在 FID 和人类偏好上均优于 PAG/SAG**：FID 从 12.8 降至 11.2，人类偏好率 51.8%
- **即插即用兼容性强**：在 IP-Adapter、ControlNet 等下游应用上也能稳定提升，CLIP-I 提升 3.6%
- **Sinkhorn 迭代的必要性**：直接加噪声（不用 Sinkhorn）FID 为 12.4，说明保持注意力矩阵的合法性很重要
- **计算开销极小**：每步额外的 Sinkhorn 迭代（通常 5-10 次）仅增加约 3-5% 的推理时间

## 亮点与洞察
- **用理论取代启发式**：首次从最优传输角度为注意力引导提供理论解释，把"为什么扰动注意力能改善生成"这个问题转化为严格的 OT 框架
- **对抗性代价设计精妙**："惩罚相似度最高的 pair"精确瞄准了最有信息量的注意力连接，比随机/均匀扰动更高效
- **迁移性强**：不依赖特定模型架构，可直接用于任何基于自注意力的扩散模型

## 局限与展望
- 仅在自注意力层工作，不涉及交叉注意力（text-image 交叉注意力可能也有类似的 OT 解读）
- Sinkhorn 迭代增加少量计算开销，在实时生成场景下可能需要优化
- 超参数 $\lambda$ 的最优值可能因模型/任务不同而变化，缺乏自适应选择策略
- 只在文本到图像场景验证，视频生成、3D 生成等场景未探索

## 相关工作与启发
- **vs PAG (Perturbed Attention Guidance)**：用 identity matrix 替换注意力作为扰动，缺乏理论基础；ASAG 用 OT 理论设计的对抗代价更有效
- **vs SAG (Self-Attention Guidance)**：用模糊 map 引导注意力，同样是启发式；ASAG 提供了有原则的替代方案
- **vs CFG**：CFG 在条件-无条件方向上做线性外推，ASAG 在注意力层面做扰动，两者正交可叠加

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ OT视角解读注意力引导是全新角度
- 实验充分度: ⭐⭐⭐⭐ 覆盖多模型/多应用，但数据集范围有限
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，即插即用特性吸引实践者
- 价值: ⭐⭐⭐⭐ 实用价值高，理论贡献为后续引导方法研究奠定基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Boosting Adversarial Transferability via Ensemble Non-Attention](boosting_adversarial_transferability_via_ensemble_non-attention.md)
- [\[ICLR 2026\] Harpoon: Generalised Manifold Guidance for Conditional Tabular Diffusion](../../ICLR2026/others/harpoon_generalised_manifold_guidance_for_conditional_tabular_diffusion.md)
- [\[ICLR 2026\] Contractive Diffusion Policies: Robust Action Diffusion via Contractive Score-Based Sampling with Differential Equations](../../ICLR2026/others/contractive_diffusion_policies_robust_action_diffusion_via_contractive_score-bas.md)
- [\[AAAI 2026\] Local Guidance for Configuration-Based Multi-Agent Pathfinding](local_guidance_for_configuration-based_multi-agent_pathfinding.md)
- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)

</div>

<!-- RELATED:END -->
