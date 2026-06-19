---
title: >-
  [论文解读] VisFocus: Prompt-Guided Vision Encoders for OCR-Free Dense Document Understanding
description: >-
  [ECCV 2024][LLM评测][OCR] VisFocus提出了一种提示引导的视觉编码方法用于OCR-free文档理解：通过将用户提示（prompt）直接注入视觉编码器的patch merging层（ViLMA层），配合局部掩码提示建模（LMPM）预训练任务，使视觉编码器学会聚焦于与提示相关的文本区域，在多个文档VQA基准上达到同规模SOTA。
tags:
  - "ECCV 2024"
  - "LLM评测"
  - "OCR"
  - "文档理解"
  - "提示感知视觉编码"
  - "Transformer"
  - "视觉-语言交互"
---

# VisFocus: Prompt-Guided Vision Encoders for OCR-Free Dense Document Understanding

**会议**: ECCV 2024  
**arXiv**: [2407.12594](https://arxiv.org/abs/2407.12594)  
**代码**: 无  
**领域**: LLM评测  
**关键词**: OCR-free, 文档理解, 提示感知视觉编码, Swin Transformer, 视觉-语言交互

## 一句话总结
VisFocus提出了一种提示引导的视觉编码方法用于OCR-free文档理解：通过将用户提示（prompt）直接注入视觉编码器的patch merging层（ViLMA层），配合局部掩码提示建模（LMPM）预训练任务，使视觉编码器学会聚焦于与提示相关的文本区域，在多个文档VQA基准上达到同规模SOTA。

## 研究背景与动机

视觉文档理解（VDU）旨在从PDF或文档图像中提取有意义的信息，涵盖DocVQA、ChartQA、信息图理解等任务。

**现有痛点**：

**OCR依赖的弊端**：传统方法依赖外部OCR模型提取文本，增加延迟和计算成本，且OCR错误会传播到下游模型

**OCR-free方法的局限**：现有OCR-free方法（Donut、Dessurt等）虽然绕过了OCR，但用户查询仅输入到语言模型，视觉编码器对整篇文档"无差别"编码。对于密集文档，大量视觉token被空白区域、图表和无关文本占用，真正与查询相关的内容未获得足够关注

**核心矛盾**：视觉特征的提取独立于用户查询，导致视觉表示对于具体问题来说是次优的。这在密集文档中尤为严重——文档越长，无关信息越多，模型性能下降越明显。

**切入角度**：类比人类阅读文档的方式——我们不会逐字阅读，而是先扫描关键词找到与问题相关的区域，再精读。VisFocus将这种"选择性扫描"能力注入视觉编码器，让它在下采样过程中就考虑用户提示。

## 方法详解

### 整体框架
VisFocus基于"视觉编码器 + 投影层 + 语言模型"的标准OCR-free架构。视觉编码器采用SwinV2，语言模型采用T5。核心改进在于：(1) 用ViLMA层替换Swin的所有patch merging层，引入提示-视觉交互；(2) 设计LMPM预训练任务引导模型聚焦相关文本。训练分三阶段：LtR → LMPM → 下游微调。

传统方法中视觉编码器 $\mathcal{M}_V(X)$ 独立编码文档图像，prompt仅输入语言模型。VisFocus改为 $\mathcal{M}_V^p(\mathbf{p}, X)$，使视觉编码产生prompt感知的特征。整体公式：
$$\mathcal{M}(\mathbf{p}, X) = \mathcal{M}_L(\mathbf{p}, \mathcal{M}_V^p(\mathbf{p}, X))$$

### 关键设计

1. **ViLMA层（Vision-Language Merging Attention）**:

    - 功能：替换SwinV2中的patch merging层，在视觉特征下采样过程中引入prompt信息交互
    - 核心思路：在标准Swin patch merging的2×2邻域拼接（$L \times c \to L/4 \times 4c$）与线性投影（$L/4 \times 4c \to L/4 \times 2c$）之间，插入一个多头交叉注意力层。视觉特征作为query，提示嵌入作为key和value：
    $\tilde{F} = \hat{F} + \text{Norm}(\text{MHCA}(\hat{F}, \text{emb}(\mathbf{p})))$
    - 提示编码：使用冻结的语言编码器将用户提示转换为上下文感知的嵌入表示
    - 位置选择：ViLMA层放置在SwinV2的每个stage末尾（共4个stage），替换全部原始patch merging层
    - 设计动机：patch merging是信息聚合的关键步骤，在此处引入提示信号能让特征压缩过程有选择性地保留与提示相关的信息

2. **LMPM预训练任务（Localized Masked Prompt Modeling）**:

    - 功能：引导视觉编码器学会聚焦于文档中与提示相关的特定文本区域
    - 核心思路：从文档OCR文本中随机采样一个局部文本片段，对其中部分token进行掩码（借用T5的denoising目标），将被掩码的片段作为"提示"输入ViLMA层，模型需要预测被掩码的token。由于掩码token对应文档中特定位置的文本，模型必须学会将视觉注意力聚焦到该区域
    - 损失函数：$\mathcal{L}_{LMPM} = \mathcal{L}_{CE}(\mathcal{M}(\mathbf{s}, X), Y_{LMPM})$
    - 设计动机：仅有ViLMA架构能让视觉编码器"看到"提示，但不能保证它会"利用"提示来聚焦。LMPM通过显式训练信号强制模型建立提示与视觉区域的关联

3. **Prompt Dropout策略**:

    - 功能：防止语言模型"补偿"视觉编码器的聚焦能力不足
    - 核心思路：在LMPM训练阶段，以概率 $\rho$ 将提示拼接到语言模型输入，否则省略。这迫使视觉编码器独立发展聚焦能力，不能依赖语言模型来完成MLM任务
    - 设计动机：如果提示始终输入语言模型，语言模型可能直接利用提示信息完成掩码预测，而视觉编码器不需要真正学会聚焦。类似Dropout的思想，强制各组件独立学习

4. **三阶段训练流程**:

    - Stage I: **Learn to Read (LtR)** — 使用标准Swin（非ViLMA）在IDL数据集上训练文档阅读能力，预测文档中所有文字的光栅扫描顺序
    - Stage II: **LMPM** — 引入ViLMA层（随机初始化），在局部掩码提示建模任务上训练聚焦能力
    - Stage III: **Fine-tuning** — 在下游VQA任务上微调

### 损失函数 / 训练策略
- LtR阶段：标准交叉熵损失，预测文档OCR文本
- LMPM阶段：交叉熵损失 + Prompt Dropout（概率 $\rho$）
- 微调阶段：下游任务特定损失（如VQA的交叉熵）
- 优化器：AdamW + cosine annealing + warmup
- 输入分辨率：1536×768高分辨率
- 训练设备：8×A100 GPU，bfloat精度
- 模型变体：VisFocus-S（SwinV2-Small + T5-Small, 132M参数）、VisFocus-B（SwinV2-Small + T5-Base, 295M参数）

## 实验关键数据

### 主实验

| 方法 | 参数量 | DocVQA (ANLS) | InfoVQA (ANLS) | ChartQA (RA) | OCR-VQA (EM) | AI2D (EM) |
|------|--------|---------------|----------------|--------------|--------------|-----------|
| Donut | 176M | 67.5 | 11.6 | 41.8 | 66.0 | - |
| Baseline-S | 110M | 67.0 | 24.7 | 49.3 | 66.6 | 42.7 |
| **VisFocus-S** | 132M | **68.6** (+1.6) | **28.5** (+3.8) | **53.0** (+3.7) | **67.3** (+0.7) | 42.6 |
| Pix2Struct-B | 282M | 72.1 | **38.2** | 56.0 | 69.4 | 40.9 |
| Baseline-B | 273M | 71.7 | 26.8 | 52.5 | 66.9 | 45.6 |
| **VisFocus-B** | 295M | **72.9** (+1.2) | 31.9 (+5.1) | **57.1** (+4.6) | **70.0** (+3.1) | **47.8** (+2.2) |

### 消融实验

| 配置 | DocVQA (ANLS) | ChartQA (RA) | 说明 |
|------|---------------|--------------|------|
| Baseline-B | 70.9 | 52.5 | 无ViLMA、无LMPM |
| + ViLMA | 71.3 (+0.4) | 54.7 (+2.2) | 仅架构改进 |
| + LMPM | 71.8 (+0.5) | 55.7 (+1.0) | 加聚焦预训练 |
| + Prompt Dropout (完整VisFocus-B) | **72.2** (+0.4) | **57.1** (+1.4) | 完整方法 |

| ViLMA层位置 | DocVQA | ChartQA | 说明 |
|------------|--------|---------|------|
| Baseline (无) | 70.9 | 52.5 | — |
| VF-Early [1,2] | 71.0 | 54.1 | 仅浅层 |
| VF-Mid [2,3] | 71.3 | 54.4 | 中间层 |
| VF-Late [3,4] | 71.6 | 55.3 | 深层效果更好 |
| **VF-All [1,2,3,4]** | **72.2** | **57.1** | 全部替换最佳 |

| Prompt注入方式 | DocVQA | ChartQA | 说明 |
|---------------|--------|---------|------|
| Baseline (仅LM) | 70.9 | 52.5 | 标准方法 |
| Render (Pix2Struct方式) | 70.6 | 52.2 | 渲染到图像反而降低 |
| **ViLMA (本文)** | **71.3** | **54.7** | 语义级交互更优 |

### 关键发现
- ViLMA层和LMPM预训练任务存在**协同效应**：单独使用各有提升，组合后提升更大
- 深层ViLMA层贡献更大，但全部替换效果最佳（ChartQA上+4.6）
- Prompt Dropout对ChartQA提升尤为显著（+1.4），有效迫使视觉编码器独立学习聚焦
- VisFocus的优势随**文档密度增加而增大**：400词文档提升+0.7，800词文档提升+2.3
- 渲染提示到图像上（Pix2Struct方式）在本文实验设置下反而降低性能
- ViLMA引入的额外参数量相对模型总参数量很小（约一个数量级的差异）
- 注意力可视化显示LMPM训练后模型能关注与查询**语义相关**的词（如查询"diameter"时聚焦于"under-ream"和"180 degrees"），而非仅字面匹配

## 亮点与洞察
- 将"选择性阅读"的人类认知策略转化为模型设计，动机自然合理
- ViLMA层的设计位置选择精准——patch merging是信息压缩的瓶颈，在此引入提示信号最有效
- LMPM预训练任务设计巧妙——用文档自身文本构造提示，无需额外标注，且天然适配文档聚焦任务
- Prompt Dropout策略简单但有效，借鉴Dropout思想解决了语言模型可能"旁路"视觉编码器的问题
- 文档密度分析实验有力证明了聚焦机制在密集文档上的价值递增特性

## 局限与展望
- 聚焦能力主要针对文本区域，对包含信息图、图表、图片的文档效果有限（InfoVQA上与Pix2Struct仍有差距）
- 未探索超越文本的prompt感知预训练任务（如引导视觉编码器关注图表中与查询相关的视觉区域）
- 模型规模仍较小（<300M参数），与数十亿参数的大型VLM相比有天然劣势
- LMPM预训练依赖OCR标注作为监督信号，某种意义上仍未完全摆脱OCR
- 三阶段训练流程较复杂，能否端到端训练达到同等效果值得探索

## 相关工作与启发
- 与Pix2Struct的对比有启发性：后者通过渲染提示到图像实现视觉编码器的提示感知，但限制了语义利用能力。VisFocus通过跨注意力在语义层面交互，效果更好
- 与Donut/Dessurt形成方法演进：从"视觉编码独立于提示"到"提示引导视觉编码"
- 与concurrent work QA-ViT对比：后者也在自注意力层注入prompt，但缺少LMPM预训练导致VDU性能较低，验证了预训练任务的必要性
- 启发：在多模态模型中，尽早引入跨模态交互（而非仅在后端融合）对于需要精细定位的任务至关重要

## 评分
- 新颖性: ⭐⭐⭐⭐ 将prompt注入视觉编码器的patch merging层思路新颖，ViLMA+LMPM配合设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 5个数据集、详细消融（组件拆解、位置分析、密度分析、注入方式对比）
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示直观，动机阐述到位
- 价值: ⭐⭐⭐⭐ 为OCR-free文档理解提供了有效的提示感知编码范式，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding](../../ACL2025/llm_evaluation/kitab-bench_a_comprehensive_multi-domain_benchmark_for_arabic_ocr_and_document_u.md)
- [\[ECCV 2024\] SIGMA: Sinkhorn-Guided Masked Video Modeling](sigma_sinkhorn-guided_masked_video_modeling.md)
- [\[ECCV 2024\] OGNI-DC: Robust Depth Completion with Optimization-Guided Neural Iterations](ogni-dc_robust_depth_completion_with_optimization-guided_neural_iterations.md)
- [\[ECCV 2024\] Learn from the Learnt: Source-Free Active Domain Adaptation via Contrastive Sampling and Visual Persistence](learn_from_the_learnt_source-free_active_domain_adaptation_via_contrastive_sampl.md)
- [\[ACL 2025\] MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark](../../ACL2025/llm_evaluation/mmlu-cf_a_contamination-free_multi-task_language_understanding_benchmark.md)

</div>

<!-- RELATED:END -->
