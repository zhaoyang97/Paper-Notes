---
title: >-
  [论文解读] Document Haystacks: Vision-Language Reasoning Over Piles of 1000+ Documents
description: >-
  [多模态VLM] 提出 DocHaystack 和 InfoHaystack 两个大规模文档检索基准（每个问题对应 1000+ 文档），以及 V-RAG——一个视觉中心的检索增强生成框架，在 Recall@1 上比最佳基线提升 9%-11%。 领域现状 大型多模态模型（LMMs）在视觉-语言理解上取得了显著进展…
tags:
  - "多模态VLM"
---

# Document Haystacks: Vision-Language Reasoning Over Piles of 1000+ Documents

| 属性 | 值 |
|------|------|
| 会议 | CVPR 2025 |
| arXiv | [2411.16740](https://arxiv.org/abs/2411.16740) |
| 代码 | [GitHub](https://github.com/Vision-CAIR/dochaystacks) |
| 领域 | 人体理解 / 文档理解 / 多模态推理 |
| 关键词 | document retrieval, visual QA, RAG, large multimodal model, benchmark |

## 一句话总结

提出 DocHaystack 和 InfoHaystack 两个大规模文档检索基准（每个问题对应 1000+ 文档），以及 V-RAG——一个视觉中心的检索增强生成框架，在 Recall@1 上比最佳基线提升 9%-11%。

## 研究背景与动机

### 领域现状

大型多模态模型（LMMs）在视觉-语言理解上取得了显著进展，但在处理大规模图像/文档集合时仍面临困难。现有多图像 VQA 基准规模有限，每个问题最多配对 ~30 张图像，远不能反映真实场景需求。

### 现有痛点

1. **基准规模不足**：RetVQA、WebQA 等基准每个问题仅对应 ≤30 张图像，无法模拟真实的大规模文档检索场景
2. **答案歧义问题**：已有的 DocVQA、InfographicVQA 数据集中大量"通用问题"（如"表格编号是什么？"）可被多个文档回答，导致评估不可靠
3. **LMMs 上下文长度受限**：当前 LMMs 无法直接处理数百乃至上千张高分辨率文档图像
4. **检索方法精度不足**：单一视觉编码器难以全面捕获文档中的文字、符号、图表等多尺度信息

### 本文目标

1. 建立每个问题对应 **1000 个文档** 的大规模文档检索基准，且保证答案唯一性
2. 设计有效的视觉检索框架，使 LMMs 能在成百上千个文档中进行检索和推理

### 切入角度与核心 idea

通过三阶段数据过滤管线（LLM 过滤通用问题 → 人工审核 → 过滤通用知识问题）确保基准质量；提出 V-RAG，组合多个视觉编码器（CLIP + SigLIP + OpenCLIP）的集成检索 + LMM-Filter 二阶段过滤。

## 方法详解

### 整体框架

V-RAG 分为三步：(1) 视觉编码器集成——用 CLIP、SigLIP、OpenCLIP 三个编码器计算问题-文档相似度并取平均，选出 top-m 文档；(2) LMM-Filter 模块——用 LMM 逐一判断每个候选文档是否能回答该问题，剔除不相关文档；(3) LMM-VQA 模块——将 top-k 相关文档与问题输入 LMM 生成最终答案。

### 关键设计 1：三阶段数据过滤管线（基准构建）

- **功能**：确保基准中每个问题在整个文档集中有唯一答案
- **核心思路**：
    - Step 1：用 GPT-4o 过滤"通用问题"（可被多个文档回答的问题）
    - Step 2：人工审核，验证唯一标识符（人名、日期、标题等）存在性，并用 OCR + 全文搜索验证答案在其他文档中不出现
    - Step 3：过滤"通用知识问题"——GPT-4o 无需图像即可回答的问题（DocVQA 中 26.4%、InfographicVQA 中 54.9% 的问题可被 GPT-4o 不看图直接回答）
- **设计动机**：大规模检索基准的核心挑战不在于规模本身，而在于答案歧义性。没有严格过滤的基准无法可靠评估模型

### 关键设计 2：视觉编码器集成

- **功能**：综合多种视觉编码器的互补能力提高检索精度
- **核心思路**：对每个问题-文档对，分别用 CLIP (ViT-L/14@336)、SigLIP (ViT-SO400M/14@384)、OpenCLIP (ConvNeXt-XXL@1024) 计算余弦相似度 $Sim_c$, $Sim_s$, $Sim_o$，取平均得到 $Sim_{avg}$
- **设计动机**：不同编码器各有所长——ConvNeXt 处理高分辨率强，CLIP 处理文本描述强，SigLIP 的全局匹配更稳定。实验验证表明三者集成优于任意单一编码器

### 关键设计 3：LMM-Filter 二阶段过滤

- **功能**：利用 LMM 的推理能力进一步精炼检索结果
- **核心思路**：对 top-m 候选文档，逐一配对问题输入 LMM（LLaVA-OneVision），提示"这张图片能否回答该问题？只回答是/否"。仅保留"是"的文档
- **设计动机**：视觉编码器的相似度匹配是浅层语义，LMM 能做更深层的问题-文档关联推理。两阶段互补，粗筛高效、精筛准确

## 实验关键数据

### 检索结果 (Recall@1)

| 方法 | DocH-100 | DocH-1000 | InfoH-100 | InfoH-1000 |
|------|----------|-----------|-----------|------------|
| BM25 (OCR) | 63.30 | 56.88 | 56.77 | 38.71 |
| CLIP | 46.79 | 23.85 | 69.68 | 45.81 |
| OpenCLIP | 58.72 | 34.86 | 72.26 | 53.55 |
| **V-RAG** | **81.65** | **66.06** | **79.35** | **64.52** |

V-RAG 在 DocHaystack-1000 上 Recall@1 比最佳单一编码器（OpenCLIP）高 +31.2 个百分点。

### VQA 结果

| 方法 | DocH-100 | DocH-1000 | InfoH-100 | InfoH-1000 |
|------|----------|-----------|-----------|------------|
| GPT-4o (直接) | 27.52 | - | 23.87 | - |
| GPT-4o+V-RAG | 81.65 | 66.97 | 65.16 | 56.77 |
| Qwen2-VL-f.t.+V-RAG | **86.24** | **73.39** | **67.10** | **60.00** |

GPT-4o 直接处理 200 张文档准确率仅 23.85%，加 V-RAG 后飙升至 72.48%（+48.63%）。

### 消融实验

| CLIP | SigLIP | OpenCLIP | VLM-filter | DocH-1000 R@1 |
|------|--------|----------|------------|---------------|
| ✓ | | | | 23.85 |
| | | ✓ | | 34.86 |
| ✓ | ✓ | ✓ | | 56.88 |
| ✓ | ✓ | ✓ | ✓ | **66.06** |

编码器集成贡献 +22 个百分点，LMM-Filter 再贡献 +9 个百分点。

### 关键发现

- InfographicVQA 中 54.9% 的问题可被 GPT-4o 不看图回答，暴露语言偏见严重
- LLaVA-OneVision 无法在 100 张文档以上的场景运行（上下文长度限制）
- 微调 Qwen2-VL（加 1-10 个干扰图训练）可进一步提升鲁棒性约 4-7 个百分点
- 问题类型分布：DocHaystack 侧重表格/列表，InfoHaystack 侧重图表/文本

## 亮点与洞察

1. **基准设计理念深刻**：三阶段过滤管线确保答案唯一性，尤其"通用知识过滤"这一步揭示了现有基准的语言偏见问题
2. **V-RAG 的工程智慧**：不训练新模型、不改架构，纯模块组合（编码器集成 + LMM 过滤）就获得巨大提升
3. **1000 文档规模**：首次将多图像检索推到千级别，暴露了当前 LMMs 长上下文能力的短板
4. **编码器互补效应显著**：三编码器集成比最强单编码器高 30+ 个百分点

## 局限性

1. 最终保留数据较少（DocVQA 109 题 / InfographicVQA 155 题），testbed 规模偏小
2. V-RAG 的 LMM-Filter 需要对每个候选文档做一次 LMM 推理（top-60），延迟不低
3. "大海捞针"场景在实际问题分布上可能过于人工——真实场景的问题分布更复杂
4. 基准仅覆盖英文文档，多语种文档检索场景未涉及

## 相关工作与启发

- **RetVQA**（ECCV 2022）：每个问题配 ≤30 张图像的小规模检索基准
- **MIRAGE**（ICML 2024）：CLIP 检索器 + LMM 推理，V-RAG 用多编码器集成大幅超越
- **RAG 在 NLP 中的成功**：V-RAG 将 RAG 思想系统化地应用于视觉文档检索
- **启发**：未来大规模多模态推理可能需要"分层检索"策略——粗筛用轻量编码器、精筛用重量级 LMM

## 评分

⭐⭐⭐⭐ — 基准设计严谨、方法思路清晰、实验全面。V-RAG 本身技术含量不算前沿（主要是工程组合），但基准的贡献和暴露的问题（LMM 长上下文弱点、语言偏见）很有价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models](../../ICCV2025/multimodal_vlm/physics_context_builders_a_modular_framework_for_physical_reasoning_in_vision-la.md)
- [\[ACL 2025\] Can Vision Language Models Understand Mimed Actions?](../../ACL2025/multimodal_vlm/can_vision_language_models_understand_mimed_actions.md)
- [\[ICCV 2025\] MaTVLM: Hybrid Mamba-Transformer for Efficient Vision-Language Modeling](../../ICCV2025/multimodal_vlm/matvlm_hybrid_mamba-transformer_for_efficient_vision-language_modeling.md)
- [\[ACL 2025\] Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions](../../ACL2025/multimodal_vlm/teaching_vlm_ask_ambiguity.md)
- [\[ICCV 2025\] Training-Free Personalization via Retrieval and Reasoning on Fingerprints](../../ICCV2025/multimodal_vlm/training-free_personalization_via_retrieval_and_reasoning_on_fingerprints.md)

</div>

<!-- RELATED:END -->
