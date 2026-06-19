---
title: >-
  [论文解读] RAG-IGBench: Innovative Evaluation for RAG-based Interleaved Generation in Open-domain Question Answering
description: >-
  [NeurIPS 2025][图像生成][交错图文生成] 提出 RAG-IGBench，一个专门评估基于检索增强生成的交错图文内容质量的 benchmark，设计了覆盖文本质量、图像质量和图文一致性三个维度的创新自动评估指标，并验证了与人类评估的高度相关性。 领域现状：交错图文生成（interleaved image-tex…
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "交错图文生成"
  - "检索增强生成"
  - "多模态评估"
  - "开放域问答"
  - "benchmark"
---

# RAG-IGBench: Innovative Evaluation for RAG-based Interleaved Generation in Open-domain Question Answering

**会议**: NeurIPS 2025  
**arXiv**: [2512.05119](https://arxiv.org/abs/2512.05119)  
**代码**: [https://github.com/USTC-StarTeam/RAG-IGBench](https://github.com/USTC-StarTeam/RAG-IGBench)  
**领域**: 信息检索  
**关键词**: 交错图文生成, 检索增强生成, 多模态评估, 开放域问答, benchmark

## 一句话总结
提出 RAG-IGBench，一个专门评估基于检索增强生成的交错图文内容质量的 benchmark，设计了覆盖文本质量、图像质量和图文一致性三个维度的创新自动评估指标，并验证了与人类评估的高度相关性。

## 研究背景与动机

**领域现状**：交错图文生成（interleaved image-text generation）需要模型同时产出文本和图像，是实际应用（如内容创作、可视化故事讲述）的核心需求

**现有痛点**：
   - 端到端生成方法（如 Chameleon）虽然统一了文本和图像处理，但在遵循复杂指令方面能力有限
   - 现有评估框架要么只评估单模态指标（如 FID 只评图像），要么依赖 MLLM 打分（GPT-4o based），后者引入模型偏差和不稳定性
   - 缺乏高质量的开放域交错图文生成数据集

**核心矛盾**：如何在不依赖模型偏差的前提下全面评估交错图文内容的质量

**切入角度**：采用 RAG 框架——MLLM 从检索到的文档中选择图像并嵌入文本中，而非生成图像

## 方法详解

### 整体框架
RAG-IG 框架：给定用户查询 → 检索相关文档和图像 → MLLM 生成 Markdown 格式回答（含图像占位符）→ 替换为实际图像生成最终多模态响应。评估在文本质量、图像质量、图文一致性三个维度进行。

### 关键设计

1. **数据集构建（三阶段流水线）**：

    - 功能：构建高质量的交错图文 QA 数据集
    - 核心思路：阶段1用 MLLM 生成原始 QA；阶段2专家标注员精炼图像选择和排列；阶段3按质量过滤低质量样本
    - 设计动机：社交平台的最新公开内容保证多样性和时效性；人工标注确保 ground truth 质量

2. **图像质量评估（Edit Distance + Kendall Score）**：

    - 功能：评估模型选择的图像序列与 ground truth 的匹配度
    - 核心思路：Edit Distance 衡量选择准确性（需要多少插入/删除/替换操作），归一化为 $1 - dp(m,n)/\max(m,n)$；Kendall Score 衡量顺序正确性，计算正确图像对中一致对的比例
    - 设计动机：传统 FID/IS 评估生成图像质量，但 RAG 场景是选择而非生成图像，需要评估选择准确性和排列正确性

3. **图文一致性评估（CLIP Score + Alignment Score）**：

    - 功能：评估图像在文本中的语义对齐
    - 核心思路：CLIP Score 直接计算图文余弦相似度；Alignment Score 比较同一图像在生成答案和 ground truth 中的上下文文本相似度
    - 设计动机：CLIP Score 能捕捉直接语义对齐但缺乏上下文理解；Alignment Score 弥补这一点

## 实验关键数据

### 主实验 — 主流 MLLM 在 RAG-IGBench 上的表现

| 模型 | Rouge-1↑ | Edit Dist↑ | Kendall↑ | Align Score↑ | Mean↑ |
|------|---------|-----------|---------|-------------|-------|
| GPT-4o | 0.374 | 0.471 | 0.532 | 0.495 | 0.468 |
| Claude-3.5 | 0.350 | 0.439 | 0.490 | 0.481 | 0.440 |
| Qwen2VL-72B | 0.319 | 0.390 | 0.451 | 0.438 | 0.400 |
| InternVL2-40B | 0.281 | 0.328 | 0.368 | 0.402 | 0.345 |

### 消融/验证实验 — 评估指标与人类评估的相关性

| 指标 | Pearson 相关系数 | Spearman 相关系数 |
|------|-----------------|------------------|
| Rouge-1 | 0.72 | 0.68 |
| Edit Distance | 0.81 | 0.78 |
| Kendall Score | 0.75 | 0.71 |
| CLIP Score | 0.65 | 0.62 |
| Alignment Score | 0.74 | 0.70 |

### 关键发现
- GPT-4o 在所有维度上领先，但与人类表现仍有显著差距
- 图像选择（Edit Distance）是最大瓶颈，模型普遍在图像数量和选择上表现不佳
- 微调后的模型在多个benchmark上性能都有提升，证明数据集质量高
- 开源模型与闭源模型差距明显，尤其在图文一致性上

## 亮点与洞察
- **RAG-IG 范式**比端到端图像生成更实用——图像选择比图像生成更可控、质量更稳定
- **Edit Distance + Kendall Score 的组合**巧妙地分离了"选对了什么图"和"图的顺序对不对"两个维度
- **Alignment Score** 的设计思想值得借鉴：同一图像在不同答案中应出现在相似上下文中

## 局限与展望
- 数据来源限于社交平台，领域偏向生活类话题，缺少专业/学术场景
- 评估指标虽然与人类评估相关性高，但 CLIP Score 的相关性偏低（0.65），说明仍有改进空间
- 只评估图像选择不评估图像理解——模型可能选对图但无法解释图像内容
- 未考虑布局和排版对用户体验的影响

## 相关工作与启发
- **vs INTERLEAVEDBENCH**：基于 GPT-4o 打分，引入模型偏差；RAG-IGBench 用规则指标，更客观
- **vs MMIE**：依赖微调 VLM 评估，有评估不一致问题；本文的指标更稳定可复现
- **vs MEGA-Bench**：侧重多模态理解能力，不涉及交错生成

## 评分
- 新颖性: ⭐⭐⭐⭐ RAG-IG 范式 + 创新评估指标
- 实验充分度: ⭐⭐⭐⭐ 覆盖主流开源和闭源模型，人类评估验证
- 写作质量: ⭐⭐⭐⭐ 数据集构建流程详细，指标定义清晰
- 价值: ⭐⭐⭐⭐ 填补了交错图文生成评估的空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] AvatarArtist: Open-Domain 4D Avatarization](../../CVPR2025/image_generation/avatarartist_open-domain_4d_avatarization.md)
- [\[ECCV 2024\] Learning Trimodal Relation for Audio-Visual Question Answering with Missing Modality](../../ECCV2024/image_generation/learning_trimodal_relation_for_audio-visual_question_answering_with_missing_moda.md)
- [\[NeurIPS 2025\] OVERT: A Benchmark for Over-Refusal Evaluation on Text-to-Image Models](overt_a_benchmark_for_over-refusal_evaluation_on_text-to-image_models.md)
- [\[NeurIPS 2025\] Seg4Diff: Unveiling Open-Vocabulary Segmentation in Text-to-Image Diffusion Transformers](seg4diff_unveiling_open-vocabulary_segmentation_in_text-to-image_diffusion_trans.md)
- [\[NeurIPS 2025\] HollowFlow: Efficient Sample Likelihood Evaluation using Hollow Message Passing](hollowflow_efficient_sample_likelihood_evaluation_using_hollow_message_passing.md)

</div>

<!-- RELATED:END -->
