---
title: >-
  [论文解读] Gemstones: A Model Suite for Multi-Faceted Scaling Laws
description: >-
  [NeurIPS 2025][预训练][缩放律] 开源由超过4000个检查点（覆盖50M-2B参数、多种宽度-深度比）组成的Gemstones模型套件，通过系统实验揭示缩放律对模型选择、学习率调度、冷却策略等设计选择高度敏感，并提出基于凸包的新拟合方法提升稀疏采样下的缩放律稳定性。 领域现状： 缩放律是LLM训练预算分配的关…
tags:
  - "NeurIPS 2025"
  - "预训练"
  - "缩放律"
  - "宽度-深度比"
  - "计算最优性"
  - "模型设计"
  - "凸包拟合"
---

# Gemstones: A Model Suite for Multi-Faceted Scaling Laws

**会议**: NeurIPS 2025  
**arXiv**: [2502.06857](https://arxiv.org/abs/2502.06857)  
**代码**: [GitHub](https://github.com/mcleish7/gemstone-scaling-laws)  
**领域**: 缩放律 / 模型架构  
**关键词**: 缩放律, 宽度-深度比, 计算最优性, 模型设计, 凸包拟合

## 一句话总结

开源由超过4000个检查点（覆盖50M-2B参数、多种宽度-深度比）组成的Gemstones模型套件，通过系统实验揭示缩放律对模型选择、学习率调度、冷却策略等设计选择高度敏感，并提出基于凸包的新拟合方法提升稀疏采样下的缩放律稳定性。

## 研究背景与动机

**领域现状**: 缩放律是LLM训练预算分配的关键指导工具，试图找到给定FLOP预算下的最优参数量与训练token数的平衡关系。Kaplan et al. (2020) 和 Hoffmann et al. (2022, Chinchilla) 给出了截然不同的处方，社区对其差异原因争论不休。**现有痛点**: 已有的缩放律研究通常将模型设计简化为单一维度（参数量），固定宽度-深度比在极小范围内，使用单一学习率调度策略，且训练数据集和训练token数有限，这些限制性设计选择可能根本性地影响了最终的缩放律系数。**核心矛盾**: 如果缩放律对实验设计高度敏感，那么在不同设置下拟合出的缩放律对实际从业者的指导价值就存在疑问——从业者往往采用与缩放律生成者完全不同的架构和超参数配置。**本文目标**: 通过构建迄今为止最全面的开源模型检查点套件，系统量化模型形状（宽度vs深度）、学习率、冷却策略等因素对缩放律的影响程度，并提出更鲁棒的拟合方法。**切入角度**: 基于Gemma架构的缩小版本，在二维空间（宽度×深度）而非传统的一维空间（参数量）中进行大规模实验。**核心idea**: 缩放律的处方取决于你如何设计实验和选择模型——这种敏感性必须被量化和理解，而非被忽视。

## 方法详解

### 整体框架

Gemstones项目包含三个核心部分：(1) 设计并训练覆盖多种宽度-深度比的大规模模型套件；(2) 提出基于凸包的新拟合方法以改进传统的binning方法；(3) 通过系统消融实验量化各设计选择对缩放律系数的影响。所有22种不同配置的模型均在Dolma 1.7数据集上训练350B token，每2B token保存一次检查点，并进行冷却和学习率消融实验。

### 关键设计

1. **多维度模型套件设计**:

    - 功能：构建覆盖50M、100M、500M、1B、2B五个参数级别的模型族，每个级别包含3-5种不同宽度-深度配置
    - 核心思路：模型宽度从256到3072、深度从3到80层，共2222种配置、11种宽度值和18种深度值。在参数量容差±5%内搜索可行模型集合，使模型分布在二维的宽度-深度空间中而非传统的一维参数量线上
    - 设计动机：已有工作（包括Chinchilla、Pythia等）的模型几乎都沿着固定的宽度-深度线分布（见Figure 2），无法分离宽度和深度各自的影响。Gemstones通过在二维空间采样，首次支持对模型形状的独立分析

2. **凸包拟合法（Convex Hull Approach）**:

    - 功能：替代Chinchilla的binning方法，提供更适合稀疏、多维模型分布的缩放律拟合方案
    - 核心思路：对所有模型的FLOP-Loss曲线计算下凸包 $\mathcal{H} = \text{ConvexHull}(\{(\text{FLOP}_i, L_i)\})$，仅使用凸包顶点拟合缩放律线 $\text{parameters} = c \cdot \text{compute}^{\text{exponent}}$。凸包天然排除了次优模型（位于凸包上方的点），使拟合不受这些噪声点干扰
    - 设计动机：Chinchilla的Approach 1使用每量级250个等对数间距的FLOP bin，在每个bin中选最优模型。但当模型在2D宽度-深度空间采样时，许多bin中的"最优"模型实际上是全局次优的。凸包方法仅关注真正的Pareto前沿，对模型选择策略的变化更稳定

3. **参数化缩放律拟合（Approach 3增强版）**:

    - 功能：拟合参数化公式 $L(p,T) = A/p^{\alpha} + B/T^{\beta} + \varepsilon$ 以预测最优参数量-token比
    - 核心思路：使用L-BFGS优化Huber损失（$\delta=10^{-4}$），在经验log loss和模型预测之间最小化，采用多个随机初始化（按Besiroglu et al., 2024）以避免局部最优
    - 设计动机：Approach 3相比Approach 1更适合外推预测，且对个别数据点不敏感。通过在不同子集上拟合并比较系数稳定性，可以定量评估设计选择的影响

### 损失函数 / 训练策略

所有模型使用AdamW优化器（$\beta_1=0.9, \beta_2=0.95$, weight decay=0.1），线性warmup 80M token后保持恒定学习率。学习率根据模型大小按可扩展方案调整以支持跨尺度超参数迁移。批大小固定为4M token，上下文长度2048。冷却消融在每10B token的检查点上进行，冷却时额外训练已见token的10%将学习率线性衰减至0。所有模型在AMD MI250X GPU上通过张量并行训练。

## 实验关键数据

### 主实验

缩放律斜率（参数量∝compute^exponent）对设计选择的敏感性分析：

| 拟合配置 | 令牌范围 | 冷却 | LR消融 | 嵌入 | 斜率 | Δ(vs基线) |
|---------|---------|------|-------|------|------|----------|
| Hoffmann原始 | - | ✗ | ✗ | ✓ | 0.513 | - |
| Approach 1 (基线) | 全部 | ✗ | ✗ | ✓ | 0.458 | - |
| Approach 1 | ≤100B | ✗ | ✗ | ✓ | 0.499 | +0.041 |
| Approach 1 | >120B | ✗ | ✗ | ✓ | **0.799** | **+0.341** |
| Approach 1 | 全部 | ✗ | ✓ | ✓ | 0.513 | +0.055 |
| Approach 1 | 全部 | ✓ | ✗ | ✓ | 0.597 | +0.139 |
| Approach 3 (基线) | 全部 | ✗ | ✗ | ✓ | 0.697 | - |
| Approach 3 | ≤100B | ✗ | ✗ | ✓ | 0.699 | +0.002 |
| Approach 3 | >120B | ✗ | ✗ | ✓ | 0.752 | +0.055 |
| Approach 3 Chinchilla采样 | 全部 | ✗ | ✗ | ✓ | 0.632 | -0.065 |

### 消融实验

宽度-深度对验证损失和基准精度的影响（1B级别模型，相近FLOP预算）：

| 模型配置(宽度×深度) | 类型 | 200B步精度 | 350B步精度 | 趋势 |
|-------------------|------|----------|----------|------|
| 2560×8 | 宽浅 | 较低 | 较低 | 时间效率高 |
| 1792×18 | 中等 | 中等 | 中等 | 平衡 |
| 1280×36 | 深窄 | **最高** | **最高** | FLOP效率高 |

基准预测模型拟合质量：

| 预测方法 | 公式 | 拟合质量 |
|---------|------|---------|
| Error预测 | $\text{Err}(L)=\epsilon - k\cdot\exp(-\gamma L)$ | 紧密 |
| Accuracy预测 | $\text{Acc}(L)=\frac{a}{1+e^{-k(L-L_0)}}+b$ | 有噪声（因形状变化） |
| 最可预测基准 | ARC, HellaSwag, MMLU | 小计算尺度下最稳定 |

### 关键发现

1. **缩放律极其脆弱**: 仅改变5个模型的选择（Chinchilla缩减采样）就能改变斜率0.065-0.100，而Approach 1中仅切换token范围（≤100B vs >120B）斜率变化高达0.341
2. **宽度-深度存在明显权衡**: 深模型以FLOP计更优（更低损失），但宽模型以GPU时间计更优（张量并行下通信开销更低）
3. **嵌入参数是否计入是Kaplan-Hoffmann分歧主因**: 与Pearce and Song (2024)和Porian et al. (2024)的结论一致
4. **验证集分布不影响模型排序**: 在Dolma、FineWeb、FineWeb-Edu、DCLM上的损失仅是y轴偏移，相对排序完全一致
5. **Approach 3比Approach 1更稳定**: Approach 3的斜率变化（Δ<0.07）远小于Approach 1（Δ最高0.34）

## 亮点与洞察

- **凸包拟合是解决稀疏采样问题的优雅方案**——当模型不再沿一维参数量线密集分布时，传统binning方法选出的"最优"模型可能是全局次优的，凸包天然解决了这个问题
- **"深model赢FLOP、宽model赢时间"的发现有重大实践意义**——从业者应根据自己的并行策略（有无pipeline parallelism）选择最优模型形状，而非盲目遵循以FLOP为单位的缩放律
- **仅靠5个模型就能显著改变缩放律斜率**——这直接质疑了以往基于少量模型拟合缩放律的可靠性，即使满足了"55个模型足够"的经验法则
- **4000+开源检查点的基础设施价值巨大**——从业者仅需在自己的硬件上跑少量step记录时间，即可将Gemstones的GPU Hours分析转化为自己硬件的分析

## 局限与展望

- 固定了Transformer扩展因子、词表大小、批大小等超参数，未探索这些维度的变化
- 仅使用张量并行（无pipeline parallelism），深模型的时间劣势可能在其他并行策略下不成立
- 仅在Dolma数据上训练，数据分布的影响未量化
- 最大模型仅2B参数，到更大规模的外推能力需验证
- 未探索MoE等稀疏架构，以及混合架构（如Mamba+Transformer）的缩放行为

## 相关工作与启发

- **vs Chinchilla (Hoffmann et al., 2022)**: Chinchilla在固定宽度-深度线上训练约400个模型——Gemstones在2D空间训练2222种配置，并量化证明Chinchilla的处方对模型选择高度敏感
- **vs Pythia (Biderman et al., 2023)**: Pythia也是开源模型套件但在已下架的Pile上训练且宽度-深度单一——Gemstones在可用数据上训练且覆盖多种形状
- **vs Gemma 2 (Team et al., 2024b)**: Gemma 2报告了9B级别深模型优于宽模型但细节稀少——Gemstones以开放、可复现的方式系统验证了这一结论
- **vs Alabdulmohsin et al. (2024)**: 他们研究了encoder-decoder ViT的宽深缩放律——Gemstones将类似分析扩展到decoder-only LLM
- **启发**: 缩放律应被视为有条件的经验观察而非普遍规律，从业者在使用时必须考虑自身设置与缩放律生成条件的匹配程度

## 评分

- 新颖性: ⭐⭐⭐⭐ 凸包拟合方法新颖，多维缩放律分析首次系统化，但核心仍是实验研究
- 实验充分度: ⭐⭐⭐⭐⭐ 4000+检查点、多维消融、多验证集、基准预测，极其全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，关键发现的呈现方式直观
- 价值: ⭐⭐⭐⭐⭐ 开源数据的基础设施价值极高，"缩放律脆弱性"的结论对社区有警示意义
---
title: >-
  [论文解读] Gemstones: A Model Suite for Multi-Faceted Scaling Laws
description: >-
  [NeurIPS 2025][宽度-深度比] Gemstones开源4000+检查点数据集（至2B参数），系统研究宽度-深度-训练代币在缩放律中的影响，揭示缩放律对设计选择的高度敏感性。
tags:
  - NeurIPS 2025
  - 宽度-深度比
  - 计算最优性
  - 扩展律
  - 模型设计
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training](power_lines_scaling_laws_for_weight_decay_and_batch_size_in_llm_pre-training.md)
- [\[ACL 2025\] Training Dynamics Underlying Language Model Scaling Laws: Loss Deceleration and Zero-Sum Learning](../../ACL2025/llm_pretraining/training_dynamics_underlying_language_model_scaling_laws_loss_deceleration_and_z.md)
- [\[ICML 2026\] Explaining Data Mixing Scaling Laws](../../ICML2026/llm_pretraining/explaining_data_mixing_scaling_laws.md)
- [\[NeurIPS 2025\] Scaling Embedding Layers in Language Models](scaling_embedding_layers_in_language_models.md)
- [\[ICML 2026\] Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos](../../ICML2026/llm_pretraining/dropout_universality_scaling_laws_and_optimal_scheduling_at_the_edge-of-chaos.md)

</div>

<!-- RELATED:END -->
