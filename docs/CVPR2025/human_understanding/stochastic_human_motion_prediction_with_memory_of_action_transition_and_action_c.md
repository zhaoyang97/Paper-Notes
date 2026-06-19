---
title: >-
  [论文解读] Stochastic Human Motion Prediction with Memory of Action Transition and Action Characteristic
description: >-
  [CVPR 2025][人体理解][随机人体动作预测] 本文针对动作驱动的随机人体运动预测中动作过渡不平滑和动作特征难以学习两大挑战，提出软过渡动作库（STAB）和动作特征库（ACB）两个记忆模块，配合自适应注意力调整（AAA）策略进行特征融合，在 GRAB、NTU、BABEL、HumanAct12 四个数据集上达到 SOTA。
tags:
  - "CVPR 2025"
  - "人体理解"
  - "随机人体动作预测"
  - "动作驱动"
  - "记忆库"
  - "动作过渡"
  - "CVAE"
---

# Stochastic Human Motion Prediction with Memory of Action Transition and Action Characteristic

**会议**: CVPR 2025  
**arXiv**: [2507.04062](https://arxiv.org/abs/2507.04062)  
**代码**: [https://hyqlat.github.io/STABACB.github.io/](https://hyqlat.github.io/STABACB.github.io/)  
**领域**: 人体理解 / 动作预测  
**关键词**: 随机人体动作预测, 动作驱动, 记忆库, 动作过渡, CVAE

## 一句话总结

本文针对动作驱动的随机人体运动预测中动作过渡不平滑和动作特征难以学习两大挑战，提出软过渡动作库（STAB）和动作特征库（ACB）两个记忆模块，配合自适应注意力调整（AAA）策略进行特征融合，在 GRAB、NTU、BABEL、HumanAct12 四个数据集上达到 SOTA。

## 研究背景与动机

**领域现状**：动作驱动的随机人体运动预测旨在根据观测到的历史动作序列和指定的未来动作标签，生成多样化的未来运动序列。这一任务在虚拟现实、人机交互等领域有广泛应用。现有方法（如 WAT）已经提出了基于动作标签控制未来运动的框架，但仍存在明显不足。

**现有痛点**：（1）生成平滑的过渡动作非常困难，因为不同动作的过渡速度差异很大，例如从"热身"到"喝水"的过渡与从"热身"到"投掷"的过渡完全不同；（2）动作特征难以学习，因为某些动作之间存在高度相似性，比如"喝水"和"举手"的部分动作非常接近，导致模型难以精细区分。

**核心矛盾**：现有方法没有充分利用动作过渡信息和动作特征信息——过渡信息对于生成平滑的动作衔接至关重要，而动作特征信息是区分相似动作的关键依据。这两类信息在预测的不同时间段起不同作用：初期需要更多过渡信息，后期需要更多动作特征信息。

**本文目标**：设计专门的记忆机制来存储和检索动作过渡信息和动作特征，并在预测过程中自适应地利用这两类信息。

**切入角度**：借鉴记忆库（memory bank）的思想，将动作过渡特征和动作特征分别存储在两个结构化的记忆库中，通过 key-value 机制进行检索。同时设计了软搜索方法来处理观测动作分类的不确定性。

**核心 idea**：用两个记忆库分别存储动作过渡模式和动作固有特征，配合自适应权重调整在预测不同时段动态切换关注重点，以实现更平滑的过渡和更准确的动作生成。

## 方法详解

### 整体框架

模型包含两个主要部分：动作识别模块（ARM）和运动预测模块（MPM）。ARM 基于 GRU 对输入的观测运动序列进行动作分类。MPM 基于条件 VAE（CVAE）结构进行运动预测，内部整合了 STAB 和 ACB 两个记忆库以及 AAA 融合策略。训练流程分两步：先训练 ARM 到收敛，然后冻结 ARM 参数再训练 MPM。输入为历史姿态序列 $\mathbf{X} \in \mathbb{R}^{K \times N}$ 和目标动作标签 $\mathbf{a}$，输出为预测的未来运动 $\hat{\mathbf{Y}} \in \mathbb{R}^{K \times T}$。

### 关键设计

1. **软过渡动作库（STAB）**:

    - 功能：存储不同动作之间的过渡信息，帮助生成平滑自然的动作衔接
    - 核心思路：STAB 是一个以（过去动作, 未来动作）为索引的 key-value 记忆结构。每个元素 $\mathbf{S}^{\hat{a}_p, a_f}$ 包含 M 个 (key, value) 元组。检索分两步：第一步用动作标签索引找到对应元素；第二步计算编码器输出 query 与各 key 的相似度，选择最相似的 value 并用相似度加权。创新之处在于"软搜索"：不使用 ARM 的 top-1 分类结果，而是取 top-k 个分类结果，用各自的 softmax 概率加权多个分支的检索结果进行融合，即 $\mathbf{F}_{st} = \sum_{j=1}^{k} w_p^{(j)} \cdot \mathbf{F}_{st}^{(j)}$
    - 设计动机：由于不同动作存在相似部分，观测序列可能被分类为多个候选类别。软搜索让模型关注观测动作的多种可能性，避免因单一分类错误导致检索到错误的过渡模式

2. **动作特征库（ACB）**:

    - 功能：存储每种动作的固有运动特征，为长序列预测提供更细致的动作语义信息
    - 核心思路：ACB 以未来动作标签 $a_f$ 为索引，结构与 STAB 类似但只用单一标签索引。每个动作对应 N 个 (key, value) 元组，同样通过相似度搜索和加权检索得到动作特征 $\mathbf{F}_{ac}$。与 STAB 相比，ACB 不关注过渡信息，而是聚焦于目标动作本身的运动模式
    - 设计动机：仅有过渡信息不足以在较长的预测时间段内保持动作准确性。ACB 提供了动作的"身份证"，帮助模型在预测后期生成符合目标动作特征的运动

3. **自适应注意力调整（AAA）**:

    - 功能：在预测的不同时间步动态调整 STAB 和 ACB 输出特征的融合比例
    - 核心思路：使用参数 $\alpha$ 对两个库的输出进行加权融合：$\mathbf{F} = \frac{\alpha}{1+\alpha} \mathbf{F}_{st} + \frac{1}{1+\alpha} \mathbf{F}_{ac}$。$\alpha$ 的值基于 ARM 对已预测帧的分类交叉熵损失动态计算——当分类准确（意味着已完成过渡）时 $\alpha$ 降低（更关注动作特征），反之 $\alpha$ 较高（更关注过渡信息）。为避免剧烈波动，采用 running-mean 方法平滑 $\alpha$ 的变化，并设置时间阈值 $\tau$，只在预测超过 $\tau$ 步后才开始调整
    - 设计动机：预测初期需要关注过渡信息以确保动作衔接平滑，预测后期需要关注动作特征以确保生成准确的目标动作。这种时变的权重分配符合人类运动的物理规律

### 损失函数 / 训练策略

ARM 使用交叉熵损失训练 500 个 epoch，对大于阈值 $\tau$ 的每一帧计算 CE loss。MPM 训练 500 个 epoch，损失包含 CVAE 的重建损失 $\mathcal{L}_{rec}$、KL 散度损失 $\mathcal{L}_{KL}$，以及对预测运动序列的 ARM 分类交叉熵损失。两阶段均使用 ADAM 优化器，初始学习率 0.002。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 | WAT (SOTA) | 提升 |
|--------|------|------|------------|------|
| GRAB | Acc↑ | **95.23** | 92.6 | +2.63 |
| GRAB | FID_tr↓ | **43.39** | 44.59 | -1.20 |
| GRAB | Div_w↑ | **1.14** | 1.10 | +0.04 |
| NTU | Acc↑ | **80.50** | 76.0 | +4.50 |
| NTU | FID_tr↓ | **65.11** | 72.18 | -7.07 |
| BABEL | Acc↑ | **55.37** | 49.6 | +5.77 |
| BABEL | FID_tr↓ | **20.35** | 22.54 | -2.19 |
| HumanAct12 | Acc↑ | **61.57** | 59.0 | +2.57 |
| HumanAct12 | FID_tr↓ | **112.85** | 129.95 | -17.10 |

### 消融实验

| 配置 | Acc↑ | FID_tr↓ | Div_w↑ | 说明 |
|------|------|---------|--------|------|
| Full model | **95.23** | **43.39** | **1.14** | 完整模型 |
| w/o AAA | 91.93 | 44.00 | 1.10 | 去掉自适应调整，Acc下降3.3% |
| w/o STAB | 92.18 | 43.97 | 1.11 | 去掉过渡库，Acc下降3.05% |
| w/o ACB | 93.45 | 43.61 | 1.13 | 去掉动作特征库，影响最小 |
| w/o RM (running-mean) | 90.84 | 48.99 | 1.11 | 去掉平滑，FID显著上升 |

### 关键发现

- **Running-mean 对稳定训练至关重要**：去掉 running-mean 后 FID 从 43.39 恶化到 48.99，说明不平滑的 $\alpha$ 变化会严重干扰训练过程
- **top-k=2 是软搜索的最佳设置**：k=1 时没有软搜索效果差，k=3/4 虽然多样性更高但准确率反而下降，说明过多候选会引入噪声
- **STAB 和 ACB 互补**：STAB 主要影响过渡质量和准确率，ACB 主要影响生成精度但对多样性影响小
- **在 NTU 和 BABEL 数据集上提升最大**：这两个数据集包含更多不同动作的组合，说明本方法在复杂动作过渡场景中优势更明显

## 亮点与洞察

- **软搜索机制很巧妙**：利用 ARM 的 top-k 分类概率加权多条检索路径，优雅地处理了动作分类的不确定性。这种思路可以迁移到任何需要基于分类结果检索记忆的场景
- **自适应时变融合**：AAA 策略利用分类损失作为"过渡完成度"的指标来动态调整权重，将一个难以直接建模的时间依赖关系转化为一个可观测的信号
- **记忆库结构化设计**：将动作过渡和动作特征分离存储的思路，避免了单一记忆库中信息混淆的问题，可以迁移到其他需要利用多种先验知识的生成任务

## 局限与展望

- **计算开销增加**：引入记忆库后 FPS 从 2.76 降至 1.98，GPU 显存从 2262MB 增至 2837MB，可通过并行检索优化
- **仅基于 SMPL 姿态参数**：不考虑手部、面部等细粒度运动，限制了在精细交互场景中的应用
- **软搜索的 k 值需要手动调优**：不同数据集可能需要不同的 k 值，缺乏自适应确定 k 的方法
- 可以探索将记忆库与扩散模型结合，利用扩散过程的去噪能力进一步提升生成质量

## 相关工作与启发

- **vs WAT**: WAT 是本文的直接 baseline，提出了动作驱动的随机预测范式但缺乏过渡建模。本文在 WAT 架构上增加记忆库机制，全面超越其性能
- **vs DLow**: DLow 关注生成多样性但不考虑动作语义。本文通过 STAB 和 ACB 在保持多样性的同时显著提升了语义准确性
- **vs ACTOR**: ACTOR 使用 Transformer 生成特定动作但不处理过渡。本文专注于从一个动作到另一个动作的自然过渡，填补了这一空白

## 评分

- 新颖性: ⭐⭐⭐ 记忆库的设计比较直觉，核心贡献在于问题分解和工程组合
- 实验充分度: ⭐⭐⭐⭐ 四个数据集全面验证，消融实验细致，包含 k 值敏感性分析
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，图示直观，方法描述有条理
- 价值: ⭐⭐⭐ 在细分领域（动作驱动预测）有价值，但领域应用面相对窄

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] SimMotionEdit: Text-Based Human Motion Editing with Motion Similarity Prediction](simmotionedit_text-based_human_motion_editing_with_motion_similarity_prediction.md)
- [\[CVPR 2026\] Gaussian-Mixture Latent Flow for Stochastic 3D Human Motion Prediction](../../CVPR2026/human_understanding/gaussian-mixture_latent_flow_for_stochastic_3d_human_motion_prediction.md)
- [\[ECCV 2024\] Bridging the Gap Between Human Motion and Action Semantics via Kinematic Phrases](../../ECCV2024/human_understanding/bridging_the_gap_between_human_motion_and_action_semantics_via_kinematic_phrases.md)
- [\[CVPR 2025\] Few-Shot Personalized Scanpath Prediction](few-shot_personalized_scanpath_prediction.md)
- [\[CVPR 2026\] Action Motifs: Self-Supervised Hierarchical Representation of Human Body Movements](../../CVPR2026/human_understanding/action_motifs_self-supervised_hierarchical_representation_of_human_body_movement.md)

</div>

<!-- RELATED:END -->
