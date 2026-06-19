---
title: >-
  [论文解读] BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models
description: >-
  [ICCV 2025][多模态VLM][MLLM] 通过分析 LLM 浅层对视觉嵌入的语义精炼过程，提出 BASIC 方法，利用 LLM 内部精炼后的视觉嵌入作为监督信号，从方向对齐和语义分布两个维度直接指导视觉投射器生成更好的初始视觉嵌入。 当前主流 MLLM（如 LLaVA、InternVL、Qwen2-VL）采用"视觉…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "MLLM"
  - "Visual Embedding"
  - "Self-Distillation"
  - "Modality Alignment"
  - "LLM Interpretability"
---

# BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models

**会议**: ICCV 2025  
**arXiv**: [2508.06895](https://arxiv.org/abs/2508.06895)  
**代码**: 无  
**领域**: Multimodal VLM / Vision-Language Alignment  
**关键词**: MLLM, Visual Embedding, Self-Distillation, Modality Alignment, LLM Interpretability

## 一句话总结

通过分析 LLM 浅层对视觉嵌入的语义精炼过程，提出 BASIC 方法，利用 LLM 内部精炼后的视觉嵌入作为监督信号，从方向对齐和语义分布两个维度直接指导视觉投射器生成更好的初始视觉嵌入。

## 研究背景与动机

当前主流 MLLM（如 LLaVA、InternVL、Qwen2-VL）采用"视觉编码器-视觉投射器-LLM"架构。关键瓶颈在于：**视觉嵌入仅被当作上下文线索**，训练时只对文本 token 施加自回归监督，而**完全缺乏对视觉嵌入的直接监督**。

这种不对称监督带来两个问题：
1. 未能充分利用视觉数据中的丰富信息
2. 限制了视觉-语言表示的细粒度对齐能力

替代方案各有不足：
- 离散视觉 token（Chameleon/SEED-LLaMA）：离散化导致信息损失
- L2 回归下一位置嵌入（Emu1/2）：图像生成虽好，但视觉理解落后
- 额外 VM-head（VW-LMM）：需要复杂四阶段训练管线

作者的关键发现：**LLM 的浅层会自动精炼视觉嵌入**——最初匹配到无意义文本 token 的视觉嵌入，在浅层逐步对齐到有意义的 token。这些精炼后的嵌入可以作为高质量的监督信号。

## 方法详解

### 整体框架

BASIC 在标准 MLLM 训练流程上增加两个直接视觉监督损失，利用 LLM 浅层输出的精炼视觉嵌入监督投射器生成的初始视觉嵌入。无需额外模型或标注数据。

### 关键设计

1. **视觉感知过程分析**

    - 对每个初始视觉嵌入 $\boldsymbol{v}_i$，计算其与 LLM 词汇表所有 token 嵌入的余弦相似度，找到最匹配的 token
    - 发现：部分初始嵌入已能匹配到语义相关的 token（如 "clock"、"white"）
    - 用匹配的 token 嵌入替换视觉嵌入后，LLM 仍能生成与原图高度一致的描述 → 说明 LLM 通过解读视觉嵌入中的文本概念来理解图像
    - 跨层追踪发现关键模式：
        - **浅层**（1~16/32）：无意义匹配逐步变为有意义匹配
        - **深层**（16~32/32）：趋向匹配特殊结束 token `</s>`

2. **监督视觉嵌入的构建**

    - 加权聚合 LLM 浅层（前 $k$ 层）的精炼视觉嵌入：
    $\hat{V} = \sum_{i=1}^{k} w_i \tilde{V}_i, \quad w_i = \frac{i^2}{\sum_{i=1}^{k} i^2}$
    - 权重随层数二次递增：越深的浅层精炼越好，给予更大权重
    - 基于注意力的监督强度：不同图像 patch 重要性不同
    $a_i = \frac{1}{kn} \sum_{h=1}^{k} \sum_{j=1}^{n} a_{h,j,i}$
    - $a_i$ 为文本 token 对第 $i$ 个图像 patch 的平均注意力分数

3. **方向对齐监督（Directional Alignment Supervision）**

    - 将初始嵌入与监督嵌入投影到单位超球面上，最小化角距离：
    $\mathcal{L}_{das} = \sum_{i=1}^{m} a_i \left\| \frac{\boldsymbol{v}_i}{\|\boldsymbol{v}_i\|_2} - \frac{\hat{\boldsymbol{v}}_i}{\|\hat{\boldsymbol{v}}_i\|_2} \right\|_2^2$
    - 消除幅度影响，专注于语义方向的对齐

4. **语义分布监督（Semantic Distribution Supervision）**

    - 将视觉嵌入与整个词汇表做内积得到 logits 向量，反映全局语义关联
    - 最小化初始嵌入和监督嵌入的 logits 分布之间的 KL 散度：
    $P = \text{softmax}(\hat{V}E^\top), \quad Q = \text{softmax}(VE^\top)$
    $\mathcal{L}_{sds} = \sum_{i=1}^{m} a_i \text{KL}(\boldsymbol{p}_i \| \boldsymbol{q}_i)$
    - 确保视觉嵌入在整个词汇空间上的语义分布一致性

### 损失函数 / 训练策略

总损失：$\mathcal{L} = \mathcal{L}_{lm} + \lambda_1 \mathcal{L}_{das} + \lambda_2 \mathcal{L}_{sds}$

- $\lambda_1 = 1$，$\lambda_2 = 0.01$
- 预训练阶段：仅投射器可训练，LLaVA-558k 数据
- 指令微调阶段：投射器 + LLM 可训练，LLaVA-665k 数据
- 7B 模型使用前 16/32 层，13B 模型使用前 20/40 层

## 实验关键数据

### 主实验

| 方法 | LLM | VQAv2 | GQA | SQA-I | MMB-EN | MMB-CN | MM-Vet | VizWiz |
|------|-----|-------|-----|-------|--------|--------|--------|--------|
| LLaVA-1.5 | Vicuna-7B | 78.5 | 62.0 | 66.8 | 64.3 | 58.3 | 31.1 | 50.0 |
| **BASIC** | Vicuna-7B | **79.2** | **63.5** | **70.6** | **68.8** | **62.1** | **33.8** | **52.5** |
| LLaVA-1.5 | Vicuna-13B | 80.0 | 63.3 | 71.6 | 67.7 | 63.6 | 36.1 | 53.6 |
| **BASIC** | Vicuna-13B | **80.6** | **64.6** | **73.1** | **69.6** | **64.9** | **37.2** | **55.8** |

7B 模型在 SQA-I 上提升 **+3.8**，MMB-EN 上提升 **+4.5**，MMB-CN 上提升 **+3.8**。

### 消融实验

| $\mathcal{L}_{das}$ | $\mathcal{L}_{sds}$ | VQAv2 | GQA | SQA-I | MMB-EN | MM-Vet |
|---|---|-------|-----|-------|--------|--------|
| ✗ | ✗ | 78.5 | 62.0 | 66.8 | 64.3 | 31.1 |
| ✓ | ✗ | 78.9 | 63.0 | 68.5 | 68.6 | 33.1 |
| ✗ | ✓ | 79.1 | 63.3 | 68.1 | 68.0 | 32.5 |
| ✓ | ✓ | **79.2** | **63.5** | **70.6** | **68.8** | **33.8** |

跨模型泛化（均在 LLaVA 架构上对比）：

| VE + LLM | LLaVA MMB-EN | BASIC MMB-EN | 提升 |
|----------|-------------|-------------|------|
| CLIP-L + Gemma-2B | 54.0 | 55.8 | +1.8 |
| CLIP-L + Phi3-3.8B | 68.7 | 70.2 | +1.5 |
| CLIP-L + Mistral-7B | 70.0 | 72.1 | +2.1 |
| SigLIP + Vicuna-7B | 68.0 | 69.6 | +1.6 |
| SigLIP + Vicuna-13B | 69.5 | 70.8 | +1.3 |

层权重和监督强度消融：

| 设置 | VQAv2 | MMB-EN | MM-Vet |
|------|-------|--------|--------|
| $w_i$ 递减 | 72.9 | 57.4 | 27.8 |
| $w_i$ 恒定 | 73.2 | 57.9 | 28.3 |
| **$w_i$ 递增** | **73.4** | **58.4** | **28.7** |
| 等权监督 | 73.0 | 57.6 | 27.6 |
| **注意力加权** | **73.4** | **58.4** | **28.7** |

### 关键发现

- 两种监督损失各自有效，组合使用效果更好
- 使用 LLM 下半部分层（1~l/2）的精炼嵌入最优；引入深层嵌入反而有害（深层趋于预测 `</s>`）
- 层权重二次递增优于递减和恒定，证实精炼过程逐层增强
- 注意力加权监督优于等权，说明不同 patch 的重要性差异需要被考虑
- BASIC 在 TextVQA 上略有下降（-0.2），因为监督信号侧重"语义概念"，可能模糊细小文本信息
- 有意义初始嵌入比例从 74/576 提升到 217/576（BASIC vs LLaVA，30 张图统计）

## 亮点与洞察

- **自我蒸馏视角的独特应用**：将 LLM 浅层作为"教师"指导输入层的投射器，无需外部模型
- **可解释性驱动的方法设计**：先分析 LLM 各层如何处理视觉嵌入，再基于发现设计方法
- **零额外开销**：不需要额外的监督模型或人工标注，且适用于任何"VE-Projector-LLM"架构
- **从方向和分布两个视角的互补设计**：方向对齐调整整体语义方向，分布监督确保词汇空间的关联一致性

## 局限与展望

- 在 TextVQA 上轻微下降，说明语义层面的监督可能不利于需要精确识别细小文本的场景
- 监督信号来自冻结 LLM 的浅层输出，指令微调阶段 LLM 权重变化可能影响监督质量
- 仅在 LLaVA-1.5 框架上验证，未涉及 InternVL、Qwen2-VL 等更新架构的实际集成
- 未探索动态调整 $k$（监督层数）的策略

## 相关工作与启发

- 与 Emu1/2 的对比：Emu 用 L2 回归下一位置嵌入来统一建模，BASIC 用浅层精炼嵌入做方向/分布监督
- 与 VW-LMM 的对比：VW-LMM 需要额外 VM-head 和四阶段训练，BASIC 更简洁
- Logit Lens 思想的巧妙应用：用 LLM 头投影隐藏状态来解释中间表示
- 跨层自蒸馏的思路可推广到其他模块化架构的对齐问题

## 评分

- 新颖性: ⭐⭐⭐⭐ 从 LLM 内部精炼过程获取监督信号的思路很有启发
- 实验充分度: ⭐⭐⭐⭐⭐ 8 个 benchmark、5 种 LLM、2 种 VE、详细消融
- 写作质量: ⭐⭐⭐⭐⭐ 分析透彻，图表精美，逻辑清晰
- 价值: ⭐⭐⭐⭐ 即插即用的 MLLM 增强方法，实用且通用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Visual-Oriented Fine-Grained Knowledge Editing for MultiModal Large Language Models](visual-oriented_fine-grained_knowledge_editing_for_multimodal_large_language_mod.md)
- [\[CVPR 2026\] Unleashing the Intrinsic Visual Representation Capability of Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/unleashing_the_intrinsic_visual_representation_capability_of_multimodal_large_la.md)
- [\[ICCV 2025\] CapeLLM: Support-Free Category-Agnostic Pose Estimation with Multimodal Large Language Models](capellm_support-free_category-agnostic_pose_estimation_with_multimodal_large_lan.md)
- [\[ICCV 2025\] FALCON: Resolving Visual Redundancy and Fragmentation in High-resolution Multimodal Large Language Models via Visual Registers](falcon_resolving_visual_redundancy_and_fragmentation_in_high.md)
- [\[ICCV 2025\] AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models](aigi_holmes_towards_explainable_and_generalizable_ai_generated_image_detection_via_mllm.md)

</div>

<!-- RELATED:END -->
