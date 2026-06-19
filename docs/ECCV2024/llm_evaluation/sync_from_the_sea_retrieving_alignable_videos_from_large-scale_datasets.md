---
title: >-
  [论文解读] Sync from the Sea: Retrieving Alignable Videos from Large-Scale Datasets
description: >-
  [ECCV 2024][LLM评测][视频对齐] 提出可对齐视频检索（Alignable Video Retrieval, AVR）任务，通过 DRAQ 对齐质量指标从大规模视频数据库中识别并检索出最适合与查询视频进行时序对齐的视频，同时提出特征上下文化方法提升对齐性能。 时序视频对齐旨在同步两个视频中的关键事件（如动作阶段…
tags:
  - "ECCV 2024"
  - "LLM评测"
  - "视频对齐"
  - "视频检索"
  - "时序对齐"
  - "Dynamic Time Warping"
  - "自监督特征"
---

# Sync from the Sea: Retrieving Alignable Videos from Large-Scale Datasets

**会议**: ECCV 2024  
**arXiv**: [2409.01445](https://arxiv.org/abs/2409.01445)  
**代码**: [有](https://daveishan.github.io/avr-webpage/)  
**领域**: LLM评测  
**关键词**: 视频对齐, 视频检索, 时序对齐, Dynamic Time Warping, 自监督特征

## 一句话总结

提出可对齐视频检索（Alignable Video Retrieval, AVR）任务，通过 DRAQ 对齐质量指标从大规模视频数据库中识别并检索出最适合与查询视频进行时序对齐的视频，同时提出特征上下文化方法提升对齐性能。

## 研究背景与动机

时序视频对齐旨在同步两个视频中的关键事件（如动作阶段转换、物体交互等），在视频编辑、音频轨迁移、示例驱动时序重映射等应用中具有重要价值。然而现有方法存在两个核心假设限制：

**假设视频对已给定**：现有方法只关注如何对齐已知的视频对，忽略了"如何找到适合对齐的视频"这一前置问题

**限于行为良好的动作类**：如棒球挥杆等具有固定动作阶段序列的类别，忽视了一般性视频中的巨大变异性

以"切菠萝"为例，同一动作类别的不同视频可能有完全不同的执行方式，仅知道动作类别不足以判断两个视频是否可对齐。因此，基于动作识别的检索方法不足以识别可对齐视频对，需要专门的对齐质量评估和重排方案。

## 方法详解

### 整体框架

AVR 管道包含三个阶段：

| 阶段 | 目标 | 方法 |
|------|------|------|
| 1. 候选检索 | 从大规模数据库中获取 top-k 候选 | 基于 clip 级别特征的 k-NN 检索 |
| 2. 重排序 | 从候选中识别最可对齐的视频 | 使用 DRAQ 指标重排 |
| 3. 时序对齐 | 将最佳匹配与查询视频对齐 | 基于上下文化特征的 DTW |

系统基于时间自监督预训练的视频表征（NMS 特征），对整个视频和逐帧特征进行编码。

### 关键设计

**上下文化帧级特征**：将每帧特征与截止到该时刻的累计特征均值拼接，为帧特征注入全局时间上下文。具体地，对于有 $T$ 帧的视频，上下文化特征为：

$$\bar{f}_j^{(i)} = f_j^{(i)} \oplus \frac{1}{T} \sum_{t=1}^{j} f_t^{(i)}$$

进一步通过零中心化标准化每个 clip 的特征。这种设计使帧特征不仅捕获当前场景（如人的姿态），还编码该时刻在整个动作序列中的位置（是开头还是结尾）。该方法通用、无需额外训练，可应用于任何帧级特征。

**DRAQ：动态相对对齐质量评估**。直接使用 DTW 的最优路径代价 $D(n,m)$ 来评估对齐质量会被外观相似性主导，而非时序可对齐性。DRAQ 通过将最优对齐代价与随机对齐代价的比值来消除外观偏置：

$$\text{DRAQ} = \frac{D(n,m)}{\text{Cost}_{\text{random}}}$$

随机路径的生成策略：从 $(n,m)$ 出发，以位置比例 $P_{\text{up}} = i/(i+j)$, $P_{\text{left}} = j/(i+j)$ 采样方向，使随机路径偏向对角线并增加"挑战性"。生成 $k$ 条随机路径取平均代价。

DRAQ 越低说明最优对齐相对随机对齐提升越大，即两个视频越可能有意义地对齐。由于代价矩阵 $C$ 只需计算一次，随机路径采样高效，DRAQ 相比 DTW 几乎无额外开销。

### 损失函数 / 训练策略

本方法**无需额外训练**——直接使用预训练的视频特征表示。核心创新在推理阶段的特征设计和重排算法：

- 视频表示采用 NMS 自监督预训练特征
- 候选检索使用余弦相似度的近似最近邻
- 对齐使用标准 DTW 算法
- DRAQ 使用 $k=10$ 条随机路径

## 实验关键数据

### 主实验（表格）

**AVR 环形一致性评估（PennAction↺, Penn⇄UCF, Kinetics700↺）**：

| 候选来源 | 特征 | DRAQ重排 | PennAction FPE↓ | PennAction CPE↓ | Kinetics FPE↓ | Kinetics CPE↓ |
|---------|------|----------|----------------|----------------|---------------|---------------|
| NMS检索 | NMS | ✗ | 13.4 | 1.32 | 22.7 | 0.86 |
| NMS检索 | NMS | ✓ | **9.5** | **0.20** | **0.5** | **0.0** |
| Oracle | NMS | ✗ | 24.7 | 1.70 | 35.3 | 1.08 |
| Oracle | NMS | ✓ | **7.8** | **0.33** | **0.3** | **0.01** |

### 消融实验（表格）

**特征上下文化对对齐质量的影响（PennAction APA%↑）**：

| 特征 | 无上下文化 Avg | 有上下文化 Avg | 无上下文化 Top-DRAQ | 有上下文化 Top-DRAQ |
|------|--------------|---------------|-------------------|-------------------|
| BYOL | 0.769 | 提升 | 0.814 | 提升 |
| CARL | 0.826 | 提升 | 0.856 | 提升 |
| NMS | 基准 | 提升 | 基准 | 提升 |

### 关键发现

1. **DRAQ 重排效果显著**：在所有数据集和特征组合下，DRAQ 重排都大幅降低了环形一致性误差。Kinetics 上 CPE 从 0.86 降至 0.0
2. **特征上下文化普遍有效**：为任意帧级特征添加累计上下文都能提升对齐性能
3. **DRAQ 优于 DTW 代价作为重排指标**：相对量度消除了外观偏置
4. **NMS 特征表现最优**：时间自监督预训练的特征天然适合对齐任务
5. **跨数据集对齐（Penn⇄UCF）更具挑战**：因为动作类别不匹配且检索集有限

## 亮点与洞察

- **问题定义的贡献**：首次定义 AVR 任务，将视频对齐从"已知视频对"扩展到"大规模检索+对齐"的实际场景
- **方法的简洁优雅**：DRAQ 和特征上下文化都是即插即用的轻量方案，不需要额外训练
- **评估协议的创新**：提出基于环形一致性（Cycle Consistency）的 AVR 评估方法，避免了密集标注的昂贵成本
- **对现有基准的反思**：指出 PennAction 上的代理指标可通过位置编码"作弊"，提出更直接的 Aligned Phase Agreement 指标

## 局限与展望

1. 跨数据集对齐（不同动作类间的语义对齐）效果有限
2. DRAQ 的随机路径采样策略仍有优化空间（如学习式采样）
3. 上下文化特征仅用简单的累计均值，可以探索更复杂的时间聚合
4. 未考虑部分对齐（partial alignment）场景，即两视频仅部分可对齐
5. 大规模检索的效率（特别是 DRAQ 需要对每个候选计算 DTW）可进一步优化

## 相关工作与启发

- DTW 在视频对齐中的经典地位被本文重新审视并增强
- 自监督时间特征（NMS）的成功表明，时间敏感的预训练对对齐至关重要
- 环形一致性评估思想来源于视觉对应和光流领域，被巧妙适配到 AVR

## 评分

| 维度 | 评分 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4 | 首次提出 AVR 任务，DRAQ 指标新颖实用 |
| 技术深度 | 3.5 | 方法简洁但不复杂，核心贡献在问题定义和评估协议 |
| 实验充分性 | 4 | 3 个数据集、多种特征、多种设置下的全面评估 |
| 实用性 | 4 | 即插即用、无需训练，直接适用于大规模视频数据库 |
| 总体 | 4 | 定义了一个重要且实际的新任务，提出了简洁有效的基线方案 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] PetFace: A Large-Scale Dataset and Benchmark for Animal Identification](petface_a_large-scale_dataset_and_benchmark_for_animal_identification.md)
- [\[ICML 2025\] PhantomWiki: On-Demand Datasets for Reasoning and Retrieval Evaluation](../../ICML2025/llm_evaluation/phantomwiki_on-demand_datasets_for_reasoning_and_retrieval_evaluation.md)
- [\[ACL 2025\] HellaSwag-Pro: A Large-Scale Bilingual Benchmark for Evaluating the Robustness of LLMs in Commonsense Reasoning](../../ACL2025/llm_evaluation/hellaswag-pro_a_large-scale_bilingual_benchmark_for_evaluating_the_robustness_of.md)
- [\[ICCV 2025\] HiERO: Understanding the Hierarchy of Human Behavior Enhances Reasoning on Egocentric Videos](../../ICCV2025/llm_evaluation/hiero_understanding_the_hierarchy_of_human_behavior_enhances_reasoning_on_egocen.md)
- [\[ICML 2026\] Authority, Truth, and Citation Bias: A Large-Scale Multi-Domain Benchmark for Studying Epistemic Susceptibility in Large Language Models](../../ICML2026/llm_evaluation/authority_truth_and_citation_bias_a_large-scale_multi-domain_benchmark_for_study.md)

</div>

<!-- RELATED:END -->
