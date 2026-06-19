---
title: >-
  [论文解读] NegoCollab: A Common Representation Negotiation Approach for Heterogeneous Collaborative Perception
description: >-
  [NeurIPS 2025][多模态VLM][协同感知] 提出 NegoCollab 框架，通过引入协商者（Negotiator）在训练期间从多模态 agent 的局部表示中协商生成公共表示，有效消除异质协作 agent 之间的域差异，实现低训练成本的协同网联感知。 领域现状：多 agent 协同感知通过特征共享扩大感知范围…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "协同感知"
  - "异质性"
  - "公共表示"
  - "域适应"
  - "自动驾驶"
---

# NegoCollab: A Common Representation Negotiation Approach for Heterogeneous Collaborative Perception

**会议**: NeurIPS 2025  
**arXiv**: [2510.27647](https://arxiv.org/abs/2510.27647)  
**代码**: 无  
**领域**: 多模态VLM / 协同感知  
**关键词**: 协同感知, 异质性, 公共表示, 域适应, 自动驾驶

## 一句话总结

提出 NegoCollab 框架，通过引入协商者（Negotiator）在训练期间从多模态 agent 的局部表示中协商生成公共表示，有效消除异质协作 agent 之间的域差异，实现低训练成本的协同网联感知。

## 研究背景与动机

**领域现状**：多 agent 协同感知通过特征共享扩大感知范围、克服盲区遮挡，是 V2X 通信的重要方向。

**现有痛点**：agent 可能配备不同/固定的感知模型，导致中间特征间的域差异。一对一适应方法（MPDA/PnPDA）需训练大量适配器，训练成本随 agent 类型数量的平方增长。

**核心矛盾**：指定某个 agent 的表示为公共表示会引入偏倚——与该 agent 差异大的模态对齐困难。

**切入角度**：公共表示不应被指定为单一 agent 的表示，而应从各模态 agent 的局部表示中**协商**生成。

**核心 idea**：多维对齐（分布 + 结构 + 实用）+ 环形一致性，从多模态特征中协商出中性公共表示。

## 方法详解

### 整体框架

参与者包括 M 个模态的 agent 和 N 个 agent 总数。Pipeline 为：Local Representation → Sender → Common Representation (Negotiator) → Receiver → Local Representation。

### 关键设计

1. **Sender（特征→公共表示）**

    - 功能：将本地特征映射到公共表示空间
    - 核心思路：双模块设计——Recombiner（ConvNeXt 结构，增强本地特征并调整维度）+ Aligner（融合轴注意力，捕捉全局和局部依赖）
    - 设计动机：需要兼顾维度对齐和语义对齐

2. **Negotiator（协商公共表示）**

    - 功能：从多模态 Sender 输出中协商生成统一的公共表示
    - 核心思路：特征金字塔网络（FPN）融合策略 $P = \bigoplus_{l,m} (u_l(P^{(m)}_l) \odot \text{norm}(P^{(m)}_l))$
    - 设计动机：显式学习生成公共表示 P（而非指定某个模态），消除偏倚

3. **Receiver（公共→本地）**

    - 功能：将公共表示转回本地模态空间
    - 核心思路：Converter（融合轴注意力 + 局部引导，Query 来自 Recombiner 输出）+ Recombiner
    - 设计动机：公共表示包含多模态融合信息，需要针对性转换

4. **多维对齐损失（Section 3.2.3）**

    - 分布对齐：匹配均值和标准差 $\mathcal{L}_{uni-dis}^{(m)} = \|P^{(m)} - P\|_2^2 + \alpha\|Std(P^{(m)}) - Std(P)\|_2^2$
    - 结构对齐：9 个关键点的特征相似度矩阵保持一致
    - 实用对齐：前景信息组织一致 $\mathcal{L}_{uni-pragma}^{(m)} = L_{focal}(\mathcal{N}(P^{(m)}), Y)$
    - 环形一致性：$\mathcal{L}_{cycle}^{(m)} = \|F^{(m)} - L^{(m)}\|_2^2$，确保前向后向变换信息损失最小

### 损失函数 / 训练策略

三阶段训练：第一阶段用多维对齐 + 环形一致性训练 Sender/Receiver；第二阶段联合训练 Negotiator；第三阶段端到端微调。

## 实验关键数据

### 主实验（OPV2V-H 数据集）

| 方法 | Agent 类型 | AP@0.5 | AP@0.7 | 说明 |
|------|----------|--------|--------|------|
| No Fusion | m1,m2 | 0.482 | 0.350 | 单 agent 基准 |
| MPDA(一对一) | m1,m2 | 0.815 | 0.692 | 单独适配 |
| PnPDA | m2,m4 | 0.532 | 0.331 | 跨模态差 |
| **NegoCollab** | **m1,m2** | **0.872** | **0.911** | **公共表示** |
| **NegoCollab** | **m1,m3** | **0.949** | **0.854** | **新 agent 加入** |

### 消融实验

| 对齐方式 | AP@0.5 | 改进 | 说明 |
|--------|--------|------|------|
| 仅分布对齐 | 0.812 | 基准 | 传统方法 |
| + 结构对齐 | 0.841 | +3.6% | 空间关系 |
| + 实用对齐 | 0.858 | +5.7% | 前景一致 |
| 完整三维 | **0.872** | **+7.4%** | 全方位约束 |

### 关键发现

- 相比一对一适应，训练成本降低 60%
- 公共表示天然支持新 agent 的加入，无需重新训练 Negotiator
- 在 V2V4Real 和 DAIR-V2X 真实数据集上也有 40%+ 的提升

## 亮点与洞察

- **协商框架**：突破"指定"的限制，生成更中性、更具信息量的公共表示。这个思路可以迁移到多模态融合的其他场景。
- **多维对齐设计**：超越常见的分布对齐，加入结构和实用层面的约束，形成更完整的对齐机制。
- **成本-性能平衡**：新 agent 加入时无需重训，只需训练新 Sender/Receiver，O(M) 而非 O(M²) 复杂度。

## 局限与展望

- 实验基于 LiDAR+Camera 二模态，多于 3 种模态的泛化未验证
- 论文未讨论压缩公共表示以降低通信带宽的策略
- 各 agent 间的同步假设可能在真实网络环境中不成立
- Negotiator 的额外计算在边缘设备上可能有瓶颈

## 相关工作与启发

- **vs MPDA**：MPDA 需为每对模态训练适配器，成本 O(M²)；NegoCollab 仅需 O(M)
- **vs PnPDA**：PnPDA 在跨模态大差异时效果差（AP@0.7 仅 0.331），NegoCollab 的协商机制更鲁棒

## 评分
- 新颖性: ⭐⭐⭐⭐ 协商式公共表示的提出
- 实验充分度: ⭐⭐⭐⭐ 多个协作场景，真实数据验证
- 写作质量: ⭐⭐⭐⭐ 框架清晰，公式规范
- 价值: ⭐⭐⭐⭐⭐ V2X 场景的实际部署价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Mint: A Simple Test-Time Adaptation of Vision-Language Models against Common Corruptions](mint_a_simple_testtime_adaptation_of_visionlanguage_models_a.md)
- [\[NeurIPS 2025\] On the Value of Cross-Modal Misalignment in Multimodal Representation Learning](on_the_value_of_cross-modal_misalignment_in_multimodal_representation_learning.md)
- [\[ACL 2025\] Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach](../../ACL2025/multimodal_vlm/multimodal_coreference_resolution_for_chinese_social_media_dialogues_dataset_and.md)
- [\[NeurIPS 2025\] Evaluating Multimodal Large Language Models on Core Music Perception Tasks](evaluating_multimodal_large_language_models_on_core_music_perception_tasks.md)
- [\[ICCV 2025\] Mastering Collaborative Multi-modal Data Selection: A Focus on Informativeness, Uniqueness, and Representativeness](../../ICCV2025/multimodal_vlm/mastering_collaborative_multi-modal_data_selection_a_focus_on_informativeness_un.md)

</div>

<!-- RELATED:END -->
