---
title: >-
  [论文解读] Words or Vision: Do Vision-Language Models Have Blind Faith in Text?
description: >-
  [CVPR 2025][多模态VLM][模态偏好] 本文发现VLM存在"盲目信任文本"现象——当视觉与文本输入不一致时，模型系统性地偏向文本（即使文本是错误的），通过构建包含Match/Corruption/Irrelevance三类文本变体的benchmark评估了10个VLM，分析了5个影响因素，并证明SFT+文本增强可有效缓解，同时从理论上解释了该现象源于纯文本与多模态训练数据的不平衡。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "模态偏好"
  - "文本偏差"
  - "VLM鲁棒性"
  - "多模态不一致"
  - "安全风险"
---

# Words or Vision: Do Vision-Language Models Have Blind Faith in Text?

**会议**: CVPR 2025  
**arXiv**: [2503.02199](https://arxiv.org/abs/2503.02199)  
**代码**: [https://github.com/d-ailin/blind-faith-in-text](https://github.com/d-ailin/blind-faith-in-text)  
**领域**: 多模态VLM  
**关键词**: 模态偏好, 文本偏差, VLM鲁棒性, 多模态不一致, 安全风险

## 一句话总结

本文发现VLM存在"盲目信任文本"现象——当视觉与文本输入不一致时，模型系统性地偏向文本（即使文本是错误的），通过构建包含Match/Corruption/Irrelevance三类文本变体的benchmark评估了10个VLM，分析了5个影响因素，并证明SFT+文本增强可有效缓解，同时从理论上解释了该现象源于纯文本与多模态训练数据的不平衡。

## 研究背景与动机

VLM在视觉中心任务上表现出色，但在RAG、多模态agent等实际应用中，文本输入经常与视觉信息不一致甚至误导。现有VLM评估主要是vision-centric的，文本仅作为question输入，缺乏对模型处理多模态不一致性能力的评估。核心矛盾：**VLM建立在预训练语言模型之上，纯文本训练数据远多于多模态数据（$N \gg M$），这导致模型天然偏向文本模态**。切入角度：通过系统性地向视觉任务注入三类文本变体（匹配/篡改/无关），量化模型的模态偏好。核心idea：VLM对文本的"盲信"是一个结构性问题，源于训练数据的模态不平衡。

## 方法详解

### 整体框架

构建包含4个域（VQAv2、DocVQA、MathVista、品牌识别）的评估benchmark，每个样本扩展为3种文本变体。用GPT-4o生成Match和Corruption文本，用WikiText随机抽取Irrelevance文本。在视觉与文本答案不一致的样本上，分析模型跟随哪个模态。定义Text Preference Ratio (TPR)量化文本偏好程度。评估10个VLM并分析5个影响因素。

### 关键设计

1. **三类文本变体构建**:
    - 功能：全面评估VLM在不同类型文本干扰下的行为
    - 核心思路：Match文本 $T_m$ 提供正确答案的描述，Corruption文本 $T_c$ 提供误导性描述（看文本会得到错误答案），Irrelevance文本 $T_{irr}$ 提供与任务无关的Wikipedia段落。三种变体+仅图像base组成完整评估
    - 设计动机：只有Corruption无法区分"忽略文本"和"正确使用文本"——模型可能学会简单拒绝所有文本。加入Match确保模型需要判断文本质量而非一律拒绝

2. **Text Preference Ratio (TPR) 指标**:
    - 功能：量化模型在视觉与文本答案不一致时的模态偏好
    - 核心思路：在 $\hat{Y}_{img} \neq \hat{Y}_{txt}$ 的样本中，计算 $\text{TPR} = \frac{p_{txt}}{p_{txt} + p_{img}}$，其中 $p_{txt}$ 是最终答案与纯文本答案一致的比例
    - 设计动机：仅看准确率无法区分"模型因信任错误文本而错"和"模型自身就不会"。TPR直接衡量模态偏好的方向和强度

3. **SFT文本增强缓解方案**:
    - 功能：通过有监督微调减少文本偏差
    - 核心思路：收集1000个样本，包含纯文本、原始VQA、Match/Corruption/Irrelevance各200个，用LoRA微调3个epoch。确保包含text-only数据以保持语言能力
    - 设计动机：指令调节效果有限（TPR仅降2.6%），而SFT直接在多种文本条件下训练模型，使其学会在各种文本质量下正确决策

### 损失函数 / 训练策略

SFT使用标准的next-token prediction损失，学习率 $1.0 \times 10^{-4}$，cosine decay，warmup ratio 0.1，LoRA高效微调。关键在于**数据组成**：必须包含text-only数据（否则模型会过度拒绝所有文本）和所有三种文本变体。

## 实验关键数据

### 主实验（Corruption下的性能影响）

| 模型 | VQAv2 Base | VQAv2 Corr | Norm↑ | TPR↓ | DocVQA Base | DocVQA Corr | Norm↑ |
|------|-----------|-----------|-------|------|-----------|-----------|-------|
| GPT-4o | 78.39 | 70.75 | 90.25 | 27.09 | 85.00 | 73.60 | 86.59 |
| Claude Sonnet | 66.88 | 68.17 | **101.93** | **9.58** | 87.00 | **84.60** | **97.24** |
| LLaVA-NeXT-7B | 79.45 | 28.69 | 36.10 | 85.52 | 53.60 | 10.00 | 18.60 |
| Qwen2-VL-7B | 85.51 | 50.79 | 59.41 | 29.22 | 90.50 | 57.50 | 63.63 |
| Molmo-7B-D | 76.33 | 49.29 | 64.50 | 59.40 | 74.00 | 38.40 | 51.90 |

开源模型的Corruption下降远大于闭源模型。LLaVA-NeXT-7B在VQAv2上仅保留36.1%的原始性能。

### SFT效果（VQAv2域内）

| 模型 | Base↑ | Match↑ | Corr↑ | Irr↑ | Macro↑ |
|------|-------|--------|-------|------|--------|
| LLaVA-NeXT-7B 原始 | 79.45 | 92.32 | 28.69 | 79.43 | 66.81 |
| + Instruction | 79.45 | 92.25 | 34.27 | 78.15 | 68.22 |
| + **SFT** | 77.48 | 87.56 | **71.25** | 77.32 | **78.71** |
| Qwen2-VL-7B 原始 | 85.51 | 92.76 | 50.79 | 83.70 | 75.75 |
| + **SFT** | 84.18 | 87.01 | **82.72** | 84.00 | **84.58** |

SFT使LLaVA的Corruption准确率从28.69%提升到71.25%，Macro提升12%。

### 关键发现

- **盲目信任文本是普遍现象**：几乎所有VLM在Corruption和Match下的TPR都>50%，即使错误文本也被偏信
- **开源模型文本偏差远强于闭源模型**：Claude Sonnet最鲁棒（VQAv2 TPR仅9.58%），LLaVA-NeXT-7B最差（85.52%）
- **指令效果有限**：Focus on Image指令仅将Qwen的TPR从16.8%降至14.2%
- **模型规模有限帮助**：7B→34B减少文本偏差但效果饱和
- **token顺序显著影响**：文本token放在图像token前会加剧文本偏差
- **文本相关性正相关**：BM25检索出的高相关性文本（即使无用）更容易影响模型
- **安全风险**：品牌识别中，Molmo-7B-D在HTML篡改下性能从87.44%降至41.44%，钓鱼网站可利用此漏洞

## 亮点与洞察

- **揭示了VLM的结构性盲点**：文本偏差不是个别模型的bug，而是基于LLM构建VLM的系统性后果
- **理论解释深刻**：$N \gg M$（纯文本数据远多于多模态数据）导致纯文本损失上界小而多模态损失上界大，从信息论角度解释了偏差根源
- **实用安全警示**：品牌识别任务直接说明了文本偏差的安全威胁——恶意HTML注入可绕过基于VLM的钓鱼检测
- **低成本缓解方案**：仅1000个SFT样本+LoRA即可大幅改善，具有很高的实用性

## 局限与展望

- 评估仅限分类/简短答案任务，未涉及长文本生成场景
- SFT虽有效但会略微降低base性能（LLaVA从79.45→77.48），如何做到零代价缓解尚需研究
- 理论分析基于较强假设（如有界损失、ERM收敛），实际训练更复杂
- 未探索在预训练阶段平衡纯文本与多模态数据比例的效果

## 相关工作与启发

- **vs 幻觉研究**: 幻觉关注"模型编造不存在的内容"，本文关注"模型在矛盾信息中选择了错误模态"
- **vs 多模态RAG**: 本文的发现直接警示RAG场景——检索到的"相关但错误"文本会严重误导VLM
- **vs 文本鲁棒性研究**: NLP领域有大量文本对抗攻击研究，本文将其扩展到多模态不一致场景

## 评分

- 新颖性: ⭐⭐⭐⭐ "盲信文本"现象的系统性揭示和量化是新颖贡献，理论解释有深度
- 实验充分度: ⭐⭐⭐⭐ 10个模型x4个域x3种变体+因素分析+SFT验证，较为全面
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，指标设计合理，逻辑链完整
- 价值: ⭐⭐⭐⭐ 对VLM安全部署有重要警示意义，SFT缓解方案实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Do Vision-Language Models Have Internal World Models? Towards an Atomic Evaluation](../../ACL2025/multimodal_vlm/do_vision-language_models_have_internal_world_models_towards_an_atomic_evaluatio.md)
- [\[CVPR 2025\] Vision-Language Models Do Not Understand Negation](vision-language_models_do_not_understand_negation.md)
- [\[CVPR 2025\] It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data](its_a_blind_match_towards_vision-language_correspondence_without_parallel_data.md)
- [\[CVPR 2025\] PARC: A Quantitative Framework Uncovering the Symmetries within Vision Language Models](parc_a_quantitative_framework_uncovering_the_symmetries_within_vision_language_m.md)
- [\[CVPR 2025\] FastVLM: Efficient Vision Encoding for Vision Language Models](fastvlm_efficient_vision_encoding_for_vision_language_models.md)

</div>

<!-- RELATED:END -->
