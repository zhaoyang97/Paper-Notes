---
title: >-
  [论文解读] Temporal Alignment-Free Video Matching for Few-Shot Action Recognition
description: >-
  [CVPR 2025][视频理解][少样本动作识别] 本文提出 TEAM（TEmporal Alignment-free Matching），通过固定数量的可学习模式令牌（pattern tokens）以交叉注意力聚合视频特征，消除了对预定义时序单元和暴力对齐的依赖，在 FSAR 任务上实现了更灵活、高效的视频匹配，并在多个 benchmark 上达到 SOTA。
tags:
  - "CVPR 2025"
  - "视频理解"
  - "少样本动作识别"
  - "时序无对齐匹配"
  - "模式令牌"
  - "元学习"
  - "特征聚合"
---

# Temporal Alignment-Free Video Matching for Few-Shot Action Recognition

**会议**: CVPR 2025  
**arXiv**: [2504.05956](https://arxiv.org/abs/2504.05956)  
**代码**: [https://github.com/leesb7426/TEAM](https://github.com/leesb7426/TEAM)  
**领域**: 视频理解  
**关键词**: 少样本动作识别, 时序无对齐匹配, 模式令牌, 元学习, 特征聚合

## 一句话总结

本文提出 TEAM（TEmporal Alignment-free Matching），通过固定数量的可学习模式令牌（pattern tokens）以交叉注意力聚合视频特征，消除了对预定义时序单元和暴力对齐的依赖，在 FSAR 任务上实现了更灵活、高效的视频匹配，并在多个 benchmark 上达到 SOTA。

## 研究背景与动机

**领域现状**：少样本动作识别（FSAR）旨在仅用少量标注视频学习新类别。主流方法基于度量学习，采用帧级或元组级的时序对齐来度量 support 和 query 视频之间的距离。OTAM 等帧级方法在帧间寻找最佳匹配，TRX 等元组级方法在子序列单元间进行对齐。

**现有痛点**：(1) **灵活性不足**：帧级和元组级方法依赖预定义的对齐单元（帧或固定长度元组），当动作持续时间和速度变化较大时难以适应；(2) **效率问题**：对齐代价与帧数呈二次关系增长，因为需要计算 support 和 query 所有单元之间的成对相似度；(3) 现有 FSAR benchmark 多为裁剪过的短期视频，掩盖了这一问题。

**核心矛盾**：对齐方法在预定义单元与真实动作的多样持续时间之间存在根本矛盾——固定窗口无法适配无限变化的动作模式。

**本文目标**：(1) 消除动作表示中对预定义时序单元的需求；(2) 消除视频匹配中的暴力对齐过程。

**切入角度**：与其将视频分割成帧或元组然后逐一对齐，不如用一组固定数量的全局可学习模式令牌来"吸收"每个视频的判别性特征。匹配时直接比较对应的模式令牌即可，无需任何对齐步骤。

**核心 idea**：用交叉注意力将视频帧特征聚合到固定的模式令牌中，每个 token 学习一种全局判别性模式；匹配时做 token-wise 比较，消除时序对齐的二次复杂度。

## 方法详解

### 整体框架

给定 support 和 query 视频，均匀采样 T=8 帧，通过图像特征提取器（ResNet-50 或 ViT-B）独立提取每帧特征。随后定义 M 个可学习模式令牌 $P = [P_1, P_2, ..., P_M]$，通过双互补聚合方式生成实例令牌 $P^+$ 和排他令牌 $P^-$。对 support 集的令牌进行适应性调整后，通过令牌间相似度计算分类概率。

### 关键设计

1. **实例模式令牌（Instance Pattern Tokens）$P^+$**:

    - 功能：聚合视频中与类别判别相关的正面特征，捕捉同类视频间的共享模式
    - 核心思路：每个模式令牌 $P_m$ 作为 query，视频帧特征 $F$ 作为 key 和 value，通过交叉注意力聚合信息：$\bar{P}_m^+ = P_m + \text{CA}(P_m, F, F)$，随后经 MLP 得到 $P_m^+$。匹配时对应 token 间用余弦距离度量：$\text{PD}(P_n^S, P^Q) = \sum_{m=1}^{M} -d(P_{n,m}^{S+}, P_m^{Q+})$
    - 设计动机：模式令牌不受帧数或速度限制，捕捉全局判别模式而非依赖对齐确定的特定帧匹配，既灵活又高效

2. **排他模式令牌（Exclusive Pattern Tokens）$P^-$**:

    - 功能：编码"异质性"——即本类视频中不存在的、属于其他类别的特征模式
    - 核心思路：与实例令牌的残差加法相反，使用减法：$\bar{P}_m^- = P_m - \text{CA}(P_m, F, F)$，直接排除与当前类相关的信息，使得排他令牌接近其他类的实例令牌。分类时综合考虑实例和排他令牌的互补证据，用最近类距离 $\text{ND}$ 计算排他概率
    - 设计动机：有些视频可能缺少某类的典型判别特征，仅依靠实例令牌可能分类失败。排他令牌通过"不是什么"来辅助判断，提供互补的分类视角

3. **支持集令牌适应（Adaptation of Support Pattern Tokens）**:

    - 功能：针对每个 episode 中的新类别组成，调整 support 令牌以建立更清晰的类间边界
    - 核心思路：计算类间实例令牌的余弦相似度 $E_{n,o,m}^+ = P_{n,m}^{S+} \cdot P_{o,m}^{S+}$，作为语义纠缠程度的指标。适应后的实例令牌通过增强自身信息、抑制共享信息获得：$\tilde{P}_{n,o,m}^{S+} = P_m + (1 + E_{n,o,m}^+)\text{CA}(P_m, F_n^S, F_n^S) - E_{n,o,m}^+\text{CA}(P_m, F_o^S, F_o^S)$，然后对所有其他类平均。排他令牌类似处理但方向相反
    - 设计动机：全局训练的模式令牌对基类判别力强，但对新类可能不够精细。适应过程通过显式移除类间共享信息，在每个 episode 中动态优化决策边界

### 损失函数 / 训练策略

最终损失为实例和排他两部分的交叉熵之和：$\mathcal{L} = \mathcal{L}^+ + \mathcal{L}^-$。推理时，分类概率融合两种距离：$p(y^Q = n) = \text{softmax}(\text{PD}(\hat{P}^S, P^Q) + \text{ND}(\hat{P}^S, P^Q); n)$。

使用 SGD 训练 10,000 轮。模式令牌数量 M 在不同数据集/设置上为 50-80 个（如 Kinetics 1-shot 用 60，5-shot 用 80）。采用原型概念（prototype）处理 many-shot 场景。

## 实验关键数据

### 主实验（5-way，ResNet-50 骨干）

| 方法 | HMDB51 1-shot | Kinetics 1-shot | UCF101 1-shot |
|------|-------------|----------------|--------------|
| OTAM | 54.5 | 72.2 | 79.9 |
| MoLo | 60.8 | 74.0 | 86.0 |
| GgHM | 61.2 | 74.9 | 85.2 |
| **TEAM** | **62.8** | **75.1** | **87.2** |

使用 ViT-B 骨干时在所有数据集上进一步提升（HMDB51: 70.9，Kinetics: 83.3，UCF101: 94.5）。

### 消融实验

| $P^+$ | $P^-$ | $\hat{P}$ | E | HMDB51 | Kinetics | UCF101 |
|-------|-------|----------|---|--------|----------|--------|
| ✓ | | | | 61.8 | 74.6 | 86.7 |
| ✓ | ✓ | | | 62.5 | 75.0 | 86.8 |
| ✓ | ✓ | ✓ | | 62.2 | 74.8 | 87.1 |
| ✓ | ✓ | ✓ | ✓ | **62.8** | **75.1** | **87.2** |

### 关键发现

- 仅使用实例令牌（行 a）就已超越多数对齐方法，验证了无对齐匹配的核心优势
- 排他令牌提供稳定的互补提升（+0.4-0.7%）
- 无纠缠度控制的适应会导致性能下降（行 c vs b），说明控制适应幅度的重要性
- 跨域评估（Kinetics→HMDB51/UCF101）中 TEAM 一致性超越所有方法，展现强泛化能力
- 模式令牌数量在 40-80 范围内对性能不敏感，鲁棒性强
- 效率优势显著：相比帧级对齐的 $O(T^2)$ 复杂度，TEAM 仅需 $O(M)$（如图 7 所示时间大幅减少）

## 亮点与洞察

- **根本性简化**：将时序对齐问题转化为特征聚合问题，用固定 token 数替代可变帧数匹配，概念简洁且效果好
- **互补双令牌设计**：实例令牌捕捉"是什么"、排他令牌编码"不是什么"，形成更鲁棒的类边界
- **自适应纠缠控制**：根据类间相似度动态调节适应强度，避免过度适应

## 局限与展望

- 在 SSv2-Small 等时序复杂度极高的数据集上，TEAM 略逊于使用额外点跟踪器（Point Tracker）的 TATs 方法
- 模式令牌的数量需要按数据集手动调优（虽然不太敏感）
- 当前方法仅使用帧级全局特征，未建模更细粒度的时空注意力
- 未探索与大规模预训练视觉模型（如 CLIP、VideoMAE）的结合

## 相关工作与启发

- DETR 的 object query、BLIP-2 的 Q-Former 等 token-based 聚合方法提供了相似的设计范式
- TEAM 的创新在于双互补聚合方式和 episode 级适应，这些思路可迁移到其他小样本学习场景
- 无对齐匹配的思路可能启发其他需要序列匹配的领域（如文本匹配、时间序列分类）

## 评分

- **新颖性**: 8/10 — 将帧/元组对齐转化为 token 聚合的思路简洁且有效
- **实验充分度**: 8/10 — 覆盖 4 个数据集、多骨干、多设置，消融详尽
- **写作质量**: 8/10 — 图示直观，方法描述清晰
- **价值**: 7/10 — FSAR 领域内的实用改进，但影响范围相对有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] TAMT: Temporal-Aware Model Tuning for Cross-Domain Few-Shot Action Recognition](tamt_temporal-aware_model_tuning_for_cross-domain_few-shot_action_recognition.md)
- [\[AAAI 2026\] Task-Specific Distance Correlation Matching for Few-Shot Action Recognition](../../AAAI2026/video_understanding/task-specific_distance_correlation_matching_for_few-shot_action_recognition.md)
- [\[CVPR 2025\] Video-Panda: Parameter-efficient Alignment for Encoder-free Video-Language Models](video-panda_parameter-efficient_alignment_for_encoder-free_video-language_models.md)
- [\[CVPR 2026\] MPL: Match-guided Prototype Learning for Few-shot Action Recognition](../../CVPR2026/video_understanding/mpl_match-guided_prototype_learning_for_few-shot_action_recognition.md)
- [\[ICCV 2025\] Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition](../../ICCV2025/video_understanding/beyond_label_semantics_language-guided_action_anatomy_for_few-shot_action_recogn.md)

</div>

<!-- RELATED:END -->
