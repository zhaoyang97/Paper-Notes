---
title: >-
  [论文解读] Instance-dependent Noisy-label Learning with Graphical Model Based Noise-rate Estimation
description: >-
  [ECCV 2024][LLM评测][噪声标签学习] 本文提出一种基于概率图模型的噪声率估计方法，可自动估计训练集标签噪声率，并利用估计值指导样本选择策略的课程设计，可无缝集成到 DivideMix、InstanceGM 等 SOTA 噪声标签学习方法中，在合成和真实世界基准上提升其分类精度。 领域现状：深度学习模型因高容量…
tags:
  - "ECCV 2024"
  - "LLM评测"
  - "噪声标签学习"
  - "实例依赖噪声"
  - "噪声率估计"
  - "概率图模型"
  - "样本选择"
---

# Instance-dependent Noisy-label Learning with Graphical Model Based Noise-rate Estimation

**会议**: ECCV 2024  
**arXiv**: [2305.19486](https://arxiv.org/abs/2305.19486)  
**代码**: 无  
**领域**: LLM评测  
**关键词**: 噪声标签学习, 实例依赖噪声, 噪声率估计, 概率图模型, 样本选择

## 一句话总结

本文提出一种基于概率图模型的噪声率估计方法，可自动估计训练集标签噪声率，并利用估计值指导样本选择策略的课程设计，可无缝集成到 DivideMix、InstanceGM 等 SOTA 噪声标签学习方法中，在合成和真实世界基准上提升其分类精度。

## 研究背景与动机

**领域现状**：深度学习模型因高容量特性，在噪声标签下极易过拟合，尤其是实例依赖噪声（IDN）——这种噪声由样本自身的歧义性引起（如形态相似的猫和狗被标注错误），是最现实也最具挑战性的噪声类型。当前最成功的 IDN 学习方法通常包含一个样本选择阶段，将训练样本分为"干净"和"噪声"两组。

**现有痛点**：(1) 样本选择依赖一个"课程"函数 $R(t)$，定义每个训练轮次中被分为干净样本的比例。现有课程要么是预定义的固定函数（如 Co-teaching 的线性递减），要么是基于任意聚类阈值的（如 DivideMix 的 GMM 拟合），都不考虑训练集的实际噪声率。(2) 如果课程选择过多样本为干净（实际含噪声），导致过拟合；选择过少（丢弃干净样本），导致欠拟合。(3) 标签转移矩阵估计方法试图恢复成对标签转移概率，但在高噪声率和大类别数下不稳定，且与样本选择方法是不同的技术路线。

**核心矛盾**：样本选择课程的质量直接决定噪声标签学习的效果，但现有课程设计不利用从训练集可以获得的自然信号——噪声率。如图1所示，当将 DivideMix 的样本选择替换为基于真实噪声率 $\epsilon=50\%$ 的固定比例选择时，精度提升了约 6%。这表明噪声率信息对样本选择至关重要。

**切入角度**：设计一个概率图模型，在训练过程中同时估计噪声率 $\epsilon$ 和训练模型参数，并将估计的噪声率用于构建更有效的样本选择课程。该方法可作为插件集成到任意基于样本选择的 SOTA 方法中。

**核心 idea**：通过概率图模型从训练数据中自动估计标签噪声率，用估计值替代预定义课程来指导样本选择，提升现有噪声标签学习方法的效果。

## 方法详解

### 整体框架

整体框架分为两个互相耦合的组件：(1) 概率图模型——建模噪声标签的生成过程，通过 EM 算法迭代估计噪声率 $\epsilon$、干净标签分类器 $\theta_y$ 和噪声标签分类器 $\theta_{\hat{y}}$。(2) 样本选择与下游模型训练——利用估计的噪声率 $\epsilon^{(t)}$ 构建课程 $R(t) = 1 - \epsilon^{(t)}$，指导 SOTA 噪声标签学习方法（如 DivideMix）的样本选择过程。两个组件在训练中联合优化。

### 关键设计

1. **噪声标签生成的概率图模型**:

    - 功能：建模从数据到噪声标签的生成过程，估计全局噪声率
    - 核心思路：将噪声标签生成建模为三步过程——(1) 采样数据 $x \sim p(X)$；(2) 采样干净标签 $y \sim \text{Cat}(Y; f_{\theta_y}(x))$；(3) 采样噪声标签 $\hat{y} \sim \text{Cat}(\hat{Y}; \epsilon \cdot f_{\theta_{\hat{y}}}(x) + (1-\epsilon) \cdot y)$。其中 $\epsilon$ 即为全局噪声率。通过最大化对数似然 $\max_{\theta_y, \theta_{\hat{y}}, \epsilon} \mathbb{E}_{(x_i, \hat{y}_i) \sim \mathcal{D}} [\ln p(\hat{y}_i | x_i; \theta_y, \theta_{\hat{y}}, \epsilon)]$，使用 EM 算法迭代估计所有参数
    - 设计动机：将噪声率作为图模型的可学习参数，使其能从数据中自动推断，无需手动设定。EM 的 E 步估计干净标签后验，M 步更新模型参数和噪声率

2. **基于噪声率的样本选择课程**:

    - 功能：构建更精确的干净/噪声样本划分标准
    - 核心思路：课程函数定义为 $R(t) = 1 - \epsilon^{(t)}$，即在第 $t$ 个训练轮次，将排序后损失最小的 $\lfloor R(t) \times N \rfloor$ 个样本视为干净样本，其余为噪声样本。排序标准可以是损失值（DivideMix）、到特征空间主特征向量的距离（FINE）或 KNN 分数（SSR）。与预定义课程不同，$R(t)$ 随噪声率估计值动态变化
    - 设计动机：消融实验（图1a）表明，使用正确噪声率 $\epsilon=0.5$ 的固定选择比 DivideMix 原始方案提升 ~6%。但真实噪声率未知，因此需要估计。估计过程也解决了可辨识性问题——通过约束干净标签分类器

3. **与 SOTA 方法的无缝集成**:

    - 功能：作为通用插件提升任意基于样本选择的噪声标签学习方法
    - 核心思路：将 SOTA 方法的干净标签分类器作为图模型中的 $f_{\theta_y}$，保持其原有架构和超参数不变。在 M 步中，引入额外的样本选择约束项 $L(\theta_y, \epsilon^{(t)})$（即基于估计噪声率的交叉熵损失），与原始图模型目标联合优化：$\theta_y^{(t+1)}, \theta_{\hat{y}}^{(t+1)}, \epsilon^{(t+1)} = \arg\max Q(\cdot) - \lambda L(\theta_y, \epsilon^{(t)})$。超参数 $\lambda = 1$
    - 设计动机：设计为即插即用的方式，不改变基线方法的核心架构，最大限度降低集成成本。已成功集成 DivideMix、C2D、InstanceGM、FINE、SSR、CC 等6种方法

### 损失函数 / 训练策略

总损失包含两部分：(1) 概率图模型的 ELBO 最大化目标（包括干净标签后验估计和噪声标签似然）；(2) SOTA 方法自身的损失（如 DivideMix 的半监督学习损失），通过估计的噪声率指导样本选择。训练时先对干净标签分类器进行热身（warm-up），然后联合训练图模型和下游分类器。$\epsilon$ 通过 sigmoid 激活函数的可学习参数实现，使用 SGD 优化。

## 实验关键数据

### 主实验

CIFAR-100 上 IDN 噪声实验：

| 方法 | 噪声率0.2 | 噪声率0.3 | 噪声率0.4 | 噪声率0.5 |
|------|----------|----------|----------|----------|
| DivideMix | 77.03 | 76.33 | 70.80 | 58.61 |
| DivideMix + Ours | **77.42** | **77.21** | **72.41** | **64.02** |
| InstanceGM | 79.69 | 79.21 | 78.47 | 77.19 |
| InstanceGM + Ours | 79.61 | **79.40** | **79.52** | **77.76** |

red mini-ImageNet 上真实噪声实验：

| 方法 | 噪声率0.4 | 噪声率0.6 | 噪声率0.8 |
|------|----------|----------|----------|
| DivideMix | 46.72 | 43.14 | 34.50 |
| DivideMix + Ours | **50.70** | **45.11** | **37.44** |
| InstanceGM | 52.24 | 47.96 | 39.62 |
| InstanceGM + Ours | **56.61** | **51.40** | **43.83** |

### 消融实验

| 配置 | CIFAR-100 IDN 0.5 (Acc%) | 说明 |
|------|-------------------------|------|
| DivideMix 原始 | 58.61 | 基线，使用 GMM 样本选择 |
| DivideMix + 理想噪声率 ($\epsilon$=0.5) | 64.44 | 上界，假设已知真实噪声率 |
| 图模型 + 预训练 DivideMix | 52.31 | 不联合训练效果差 |
| 联合训练但不用估计 $\epsilon$ 做选择 | 56.30 | 噪声率估计对选择至关重要 |
| DivideMix + Ours（完整方法）| 64.02 | 接近理想上界 |

### 关键发现

- 完整方法（64.02%）非常接近理想情况（64.44%），表明噪声率估计准确
- 估计的噪声率与真实值合理一致（如 IDN 0.5 时 DivideMix 估计为 0.53）
- 在 90% 的实验配置中，集成本文方法都能提升基线性能
- 本文方法在高噪声率下提升更明显（如 DivideMix 在 0.5 IDN 上提升 +5.41%）
- 训练时间开销很小（DivideMix 基线约 18.7h，加入本文方法约 20.3h）

## 亮点与洞察

- **填补研究空白**：首个将噪声率估计直接用于样本选择课程的方法
- **即插即用**：可与 6 种不同的 SOTA 方法集成，通用性强
- **动机清晰**：通过简单的假设已知噪声率实验，令人信服地论证了噪声率对样本选择的重要性
- **估计质量验证**：不仅验证分类精度，还展示了估计噪声率与真实值的接近程度

## 局限与展望

- 全局噪声率估计可能对类别不均衡场景不够精细（不同类可能有不同噪声率）
- 噪声率参数由 sigmoid 单一标量建模，未考虑类别依赖的噪声率差异
- 可探索实例级噪声率估计而非全局噪声率
- 在更大规模数据集（如 ImageNet 全集）上的验证尚不充分

## 相关工作与启发

- **DivideMix**：基于 GMM 的样本选择 + 半监督学习，本文的主要基线
- **InstanceGM**：基于图模型的噪声标签学习，但未建模噪声率
- **Co-teaching**：样本选择课程的先驱工作，使用预定义线性递减函数
- **FINE**：基于特征空间特征向量距离的样本选择标准
- 启发：噪声率是噪声标签学习中未被充分利用的信号，概率图模型是建模标签噪声生成过程的自然选择

## 评分

- 新颖性: ⭐⭐⭐（噪声率估计用于课程设计的想法有价值，但方法本身较增量）
- 实验充分度: ⭐⭐⭐⭐⭐（合成+4个真实数据集，6种SOTA方法集成，详细消融）
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐（对噪声标签学习领域有实用贡献）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] MERLiN: Single-Shot Material Estimation and Relighting for Photometric Stereo](merlin_single-shot_material_estimation_and_relighting_for_photometric_stereo.md)
- [\[ECCV 2024\] Versatile Incremental Learning: Towards Class and Domain-Agnostic Incremental Learning](versatile_incremental_learning_towards_class_and_domain-agnostic_incremental_lea.md)
- [\[AAAI 2026\] DiCaP: Distribution-Calibrated Pseudo-labeling for Semi-Supervised Multi-Label Learning](../../AAAI2026/llm_evaluation/dicap_distribution-calibrated_pseudo-labeling_for_semi-supervised_multi-label_le.md)
- [\[ECCV 2024\] Image-Feature Weak-to-Strong Consistency: An Enhanced Paradigm for Semi-Supervised Learning](image-feature_weak-to-strong_consistency_an_enhanced_paradigm_for_semi-supervise.md)
- [\[ICML 2026\] Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning](../../ICML2026/llm_evaluation/agent_world_model_infinity_synthetic_environments_for_agentic_reinforcement_lear.md)

</div>

<!-- RELATED:END -->
