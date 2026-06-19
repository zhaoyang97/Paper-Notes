---
title: >-
  [论文解读] Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis
description: >-
  [ICCV 2025][多模态VLM][多模态数据合成] 提出Oasis方法，仅需输入图像（无需任何文本提示）即可诱导MLLM自回归生成高质量多模态指令跟随数据，配合精细的指令质量控制机制，合成50万数据给LLaVA-NeXT带来平均3.1%的全面性能提升，且超越其他合成方法。 多模态大语言模型(MLLM)的成功高度依赖大规…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "多模态数据合成"
  - "指令跟随数据"
  - "质量控制"
  - "LLaVA"
  - "MLLM"
---

# Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis

**会议**: ICCV 2025  
**arXiv**: [2503.08741](https://arxiv.org/abs/2503.08741)  
**代码**: [https://github.com/Letian2003/MM_INF](https://github.com/Letian2003/MM_INF)  
**领域**: AI Safety / 多模态数据合成  
**关键词**: 多模态数据合成, 指令跟随数据, 质量控制, LLaVA, MLLM

## 一句话总结

提出Oasis方法，仅需输入图像（无需任何文本提示）即可诱导MLLM自回归生成高质量多模态指令跟随数据，配合精细的指令质量控制机制，合成50万数据给LLaVA-NeXT带来平均3.1%的全面性能提升，且超越其他合成方法。

## 研究背景与动机

多模态大语言模型(MLLM)的成功高度依赖大规模训练数据，但存在三大瓶颈：

**数据不可用**：顶级MLLM的训练数据因隐私原因不公开

**采集成本高**：多模态数据标注昂贵且劳动密集

**现有合成方法的局限**：
   - 固定流水线和不变提示限制了数据多样性
   - 质量控制不足，难以生成真正提升模型表征能力的高质量数据
   - 复杂框架需要大量人工设计数据模式和提示

核心洞察：受Magpie（纯文本无提示合成）启发，既然MLLM的自回归特性可以生成多样化输出，能否仅给图像就让MLLM自动生成指令数据？

## 方法详解

### 整体框架

Oasis管线包含三个步骤：
1. **数据合成**：用"hooking prompt"诱导MLLM生成指令
2. **数据分类**：LLM筛选指令跟随数据，过滤纯描述数据
3. **指令质量控制**：多维度评分过滤低质量指令

### 关键设计

1. **Hooking Prompt数据合成**

   传统MLLM输入由四部分组成：预查询模板 + 视觉内容 + 指令 + 后查询模板。Oasis的核心操作是**移除指令部分和后查询模板**，仅保留预查询模板和图像：

    $\text{Inst} = \Theta(\text{vision})$

   而非传统的 $\text{Resp} = \Theta(\text{vision}, \text{instruction})$。

   由于仅输入图像，MLLM会基于自身知识库自回归生成多样化指令。无人工文本提示意味着：
    - 生成指令不受固定提示的偏差限制
    - 自然覆盖46种语言（LLaVA-NeXT仅有英语）
    - 根动词和名词对象分布更自然多样

2. **数据分类 (Data Categorization)**

   生成数据中约49.9%为描述型(caption)、50.1%为指令跟随型。使用LLM作为分类器（few-shot）区分两类：
    - 指令跟随数据保留，提取指令
    - 描述型数据先过滤，后可用规则筛选+LLM清洗回收250K高质量caption

3. **指令质量控制 (Instruction Quality Control)**

   总结高质量指令的四个特征维度，各维度1-5分评分：
    - **可解性 (Solvability)**: 图像是否提供了回答问题的充分信息
    - **清晰度 (Clarity)**: 问题是否精确传达意图
    - **幻觉 (Hallucination)**: 问题内容与图像实际内容的对齐度
    - **无意义 (Nonsense)**: 语法正确性和语义连贯性

   前3项由MLLM评估（需视觉信息），第4项由LLM评估（语言质量判断更敏感）。高质量指令的通过率约50.9%。

### 损失函数 / 训练策略

采用LLaVA-NeXT标准两阶段训练：
- **预训练**：LLaVA-Pretrain-558K，仅训练随机初始化的投影器
- **微调**：LLaVA-NeXT官方SFT数据 + Oasis合成数据，全参数可训练

训练细节：AdamW优化器，cosine学习率调度，warmup比例0.03，batch size 128。预训练LR=1e-3，微调LR=1e-5。

合成工具：Qwen2.5-VL-72B-Instruct（MLLM）+ Qwen2.5-72B-Instruct（LLM），图像源自Cambrian-10M。

## 实验关键数据

### 主实验

14个基准上LLaVA-NeXT的性能对比（Vicuna-7B-v1.5骨干）：

| 方法 | MMBench | MME | MMStar | MMVet | DocVQA | TextVQA | OCRBench | 平均 |
|------|---------|-----|--------|-------|--------|---------|----------|------|
| Baseline | 64.2/54.4 | 1482/291 | 37.1 | 28.0 | 71.7 | 63.4 | 52.9 | 53.0 |
| +LLaVA(上采样) | 64.8/54.9 | 1461/353 | 37.6 | 34.3 | 67.8 | 64.0 | 52.6 | 53.7 |
| +DenseFusion | 67.4/56.2 | 1523/333 | 37.8 | 30.2 | 69.2 | 65.4 | 55.4 | 54.3 |
| +Cambrian | 66.8/56.6 | 1504/329 | 37.8 | 32.4 | 73.8 | 63.7 | 52.3 | 54.9 |
| +MMEvol | 63.6/53.8 | 1503/316 | 32.3 | 34.9 | 64.7 | 62.8 | 51.7 | 51.6 |
| **+Oasis** | **65.6/56.7** | **1532/357** | **38.0** | **37.2** | **76.0** | **66.1** | **55.0** | **56.1** |

Oasis在三种骨干上均带来显著提升：Vicuna +3.1%、Qwen2.5 +1.8%、Llama3 +3.2%。

### 消融实验

**指令质量控制的效果**（200K数据对比）：

| 配置 | DocVQA | InfoVQA | TextVQA | 平均 |
|------|--------|---------|---------|------|
| 有质量控制 | **74.8** | **38.7** | **64.5** | **54.9** |
| 无质量控制 | 67.7 | 31.4 | 63.6 | 53.9 |

质量控制带来整体1%提升，DocVQA和InfoVQA各提升7%以上。

**响应质量控制无效**：NLL采样和MLLM评分均导致性能下降（-0.7%和-1.6%），说明高质量指令本身就能从SOTA MLLM获取好响应。

**数据规模扩展**（在100K LLaVA基础上叠加）：

| Oasis数据量 | 平均分 | 提升 |
|-------------|--------|------|
| 0 | 46.5 | - |
| 150K | 46.6 | +0.1 |
| 300K | 47.7 | +1.2 |
| 500K | 51.7 | **+5.2** |

### 关键发现

- **Oasis数据更长更多样**：指令平均长度76.8 vs LLaVA-NeXT的45.2，响应长度71.2 vs 34.2
- **多语言覆盖**：自动覆盖46种语言，是首个如此多语言的合成多模态数据集
- **根动词分布无偏差**：LLaVA-NeXT中"answer question"占大比例，Oasis更均匀自然
- **领域特定能力**：OCR领域实验中，添加70K OCR域Oasis数据后DocVQA提升3.3%、ChartQA提升3.5%
- **Caption回收有价值**：被过滤的caption数据经清洗后回收250K，12/16指标超过baseline
- **指令质量控制是关键**：但响应质量控制反而有害，高质量指令自然诱导高质量响应

## 亮点与洞察

1. **极致简洁的方法设计**：仅需一张图片，无需任何文本提示设计，利用MLLM自身知识生成多样化指令
2. **质量控制洞察深刻**：发现指令质量是核心（控制指令即控制了数据质量），响应质量控制反而引入偏差
3. **领域可控性**：图像来源直接决定生成数据的领域，无需修改流水线即可生产特定领域数据
4. **可扩展性强**：数据规模与图像数量线性增长，300K→500K仍有4%的显著提升

## 局限与展望

- 依赖Qwen2.5-VL-72B这样的强MLLM作为数据生成器，小模型生成的数据质量未验证
- 质量控制中的评分标准需要人工设计，不同领域可能需要调整
- 未探索更大规模（如百万级）合成数据的scaling law
- Oasis数据主要在LLaVA-NeXT架构上验证，其他架构的泛化性待确认

## 相关工作与启发

- 与Magpie的文本版无提示合成理念一脉相承，是其在多模态领域的自然扩展
- "指令质量 > 响应质量"的发现对数据合成领域有重要参考价值
- 图像即领域的特性使得构建针对性数据集变得极为容易

## 评分

- **新颖性**: ⭐⭐⭐⭐ 思路极其简洁，"移除文本提示"的idea虽简单但有效
- **实验充分度**: ⭐⭐⭐⭐⭐ 14个基准、3种骨干、5种合成方法对比、详细消融
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，数据属性分析充分，可视化丰富
- **实用价值**: ⭐⭐⭐⭐⭐ 方法简单、可复现、数据和代码开源，实际应用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Distraction is All You Need for Multimodal Large Language Model Jailbreaking](../../CVPR2025/multimodal_vlm/distraction_is_all_you_need_for_multimodal_large_language_model_jailbreaking.md)
- [\[CVPR 2026\] Foundation Encoders Are All You Need for Preference-Aware Personalization](../../CVPR2026/multimodal_vlm/foundation_encoders_are_all_you_need_for_preference-aware_personalization.md)
- [\[ICCV 2025\] Effective Training Data Synthesis for Improving MLLM Chart Understanding](effective_training_data_synthesis_for_improving_mllm_chart_understanding.md)
- [\[ICCV 2025\] MMOne: Representing Multiple Modalities in One Scene](mmone_representing_multiple_modalities_in_one_scene.md)
- [\[ICCV 2025\] AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models](aigi_holmes_towards_explainable_and_generalizable_ai_generated_image_detection_via_mllm.md)

</div>

<!-- RELATED:END -->
