---
title: >-
  [论文解读] Ranked from Within: Ranking Large Multimodal Models Without Labels
description: >-
  [ICML 2025][多模态VLM][无监督模型排名] 系统研究能否在无标签场景下预测 LMM 的相对性能，评估 47 个 SOTA LMM 在 9 个 VQA 基准上的表现，发现基于 softmax 分布的不确定性指标能提供稳健的无监督模型排名（与真实排名 Spearman 相关 $\rho=0.92$）。
tags:
  - "ICML 2025"
  - "多模态VLM"
  - "无监督模型排名"
  - "不确定性估计"
  - "VQA"
  - "模型选择"
  - "LMM评测"
---

# Ranked from Within: Ranking Large Multimodal Models Without Labels

**会议**: ICML 2025  
**arXiv**: [2412.06461](https://arxiv.org/abs/2412.06461)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 无监督模型排名, 不确定性估计, VQA, 模型选择, LMM评测

## 一句话总结

系统研究能否在无标签场景下预测 LMM 的相对性能，评估 47 个 SOTA LMM 在 9 个 VQA 基准上的表现，发现基于 softmax 分布的不确定性指标能提供稳健的无监督模型排名（与真实排名 Spearman 相关 $\rho=0.92$）。

## 研究背景与动机

### 1. LMM 模型选择的难题

大量 LMM（LLaVA、InstructBLIP、Qwen-VL 等）不断涌现，用户面对新数据/任务时如何高效选择？传统做法等同于"出一套考试题并批改"——需要标注数据。

### 解决思路

**本文目标**：为每个新应用场景标注评测数据耗费大量资源。许多用户（尤其工业部署）可能只有无标签数据。需要"不用改卷就能排名"的方法。

### 3. 跨基准排名的不稳定性

模型在一个基准上的排名不能可靠预测在另一个基准上的排名（论文实验验证）——不能用旧基准排名来选新部署场景的模型。

### 4. 核心 Insight

LMM 生成答案时每个 token 都有 softmax 概率分布。利用这些概率信号可以评估模型的不确定性——不确定性越低的模型通常性能越好。

## 方法详解

### 整体框架

1. 在目标数据上对 $M$ 个 LMM 各自生成回答（无需标签）
2. 从生成过程中提取不确定性信号（softmax 概率、自一致性等）
3. 计算每个模型的代理排名分数 $s_m$
4. 用 Spearman/Kendall 相关系数衡量代理分数与真实性能的对应

### 关键设计

#### 1. 三类无监督排名信号

**概率基排名（最有效）**：
- Token-level 概率：每个生成 token 的 softmax 最大值
- Sequence-level 概率：整个回答序列的联合概率
- 计算方式：对所有测试样本的 token 概率取均值

**自一致性排名**：
- 多次采样同一问题的回答，一致性越高 → 模型越有信心
- 成本高（需多次推理）

**标注代理集排名**：
- 用少量标注数据的性能作为代理
- 但跨域迁移不稳定（核心发现）

#### 2. 评估协议

- 47 个 LMM：涵盖 LLaVA、InstructBLIP、Qwen-VL 等不同框架
- 不同视觉编码器（CLIP、SigLIP）和语言模型（Vicuna、LLaMA）
- 9 个基准：ScienceQA、MMMU、ChartQA、TextVQA、RealWorldQA 等
- 覆盖推理、OCR、空间理解等多样任务

## 实验关键数据

### 主实验：排名方法对比

| 排名方法 | Spearman $\rho$（均值） | 跨基准稳定性 | 计算成本 |
|---------|------------------|------------|--------|
| 基准 A → 基准 B（跨域代理） | ~0.65 | 低 | 需标注 |
| 自一致性 | ~0.78 | 中 | 高 |
| Sequence-level 概率 | ~0.88 | 中高 | 低 |
| **Token-level softmax 概率** | **0.92** | **高** | **低** |

### 跨基准相关性分析

| 基准对 | 性能排名相关 $\rho$ | 说明 |
|--------|----------|------|
| ScienceQA ↔ MMMU | ~0.82 | 同为推理任务，相关较高 |
| ScienceQA ↔ TextVQA | ~0.51 | 推理 vs OCR，相关低 |
| ChartQA ↔ RealWorldQA | ~0.43 | 不同能力维度 |
| 全基准均值 | ~0.60 | **用一个基准预测另一个不可靠** |

### 关键发现

- 基于 softmax 概率的排名在几乎所有基准上与真实性能高度相关（$\rho>0.85$）
- 跨基准排名迁移不稳定——模型在一个任务的优势不能保证在另一个任务上也优
- 文本 prompt 相似度比图像特征相似度更能预测跨数据集的模型性能相关性
- 概率基方法对闭式（选择题）和开放式（自由文本）任务都有效，但开放式稍弱
- 自一致性方法成本高（需 5-10 次采样）且对小模型不稳定

## 亮点与洞察

- **实用性极强**：用户只需对目标数据跑一次前向传播，无需标注即可选模型
- **反直觉发现**：用旧基准排名选新场景的模型不可靠——这挑战了社区常见做法
- **不确定性即质量信号**：模型"知道自己不知道什么"，softmax 概率是最佳无监督代理
- **规模化验证**：47 模型 × 9 基准确保结论的统计可靠性

## 局限与展望

- 仅测试 VQA 任务，对更开放的生成任务（对话、摘要）的适用性待验证
- softmax 概率可能受 temperature/top-p 等采样参数影响，论文未消融
- 部分模型（如 GPT-4V）无法获取 logits，方法不适用于闭源 API 模型
- 可探索组合多种信号（概率 + 自一致性）的混合排名策略

## 相关工作与启发

- **vs Accuracy-on-the-Line (Miller et al. 2021)**：AoL 假设 ID 精度预测 OOD 精度，本文发现在 LMM 场景下这不可靠
- **vs 不确定性估计方法（Kuhn et al. 2023）**：semantic entropy 用于每个预测的置信度，本文将不确定性用于模型间排名
- **vs LMM 基准评测**：传统方法需标注，本文提供标注无关的替代方案

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统研究 LMM 无标签排名，insight 清晰
- 实验充分度: ⭐⭐⭐⭐⭐ 47 模型 × 9 基准的规模化验证
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰、实验设计严谨
- 价值: ⭐⭐⭐⭐⭐ 对 LMM 部署实践有直接指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Streamline Without Sacrifice — Squeeze out Computation Redundancy in LMM](streamline_without_sacrifice_--_squeeze_out_computation_redundancy_in_lmm.md)
- [\[ICML 2025\] Robust Multimodal Large Language Models Against Modality Conflict](robust_multimodal_large_language_models_against_modality_conflict.md)
- [\[CVPR 2025\] PARC: A Quantitative Framework Uncovering the Symmetries within Vision Language Models](../../CVPR2025/multimodal_vlm/parc_a_quantitative_framework_uncovering_the_symmetries_within_vision_language_m.md)
- [\[CVPR 2026\] Towards Multimodal Domain Generalization with Few Labels](../../CVPR2026/multimodal_vlm/towards_multimodal_domain_generalization_with_few_labels.md)
- [\[NeurIPS 2025\] GeoRanker: Distance-Aware Ranking for Worldwide Image Geolocalization](../../NeurIPS2025/multimodal_vlm/georanker_distance-aware_ranking_for_worldwide_image_geolocalization.md)

</div>

<!-- RELATED:END -->
