---
title: >-
  [论文解读] MathCoder-VL: Bridging Vision and Code for Enhanced Multimodal Mathematical Reasoning
description: >-
  [ACL 2025][多模态VLM][多模态数学推理] 提出利用代码作为跨模态对齐的监督信号，构建860万图像-代码对数据集ImgCode-8.6M和300万多模态数学指令微调数据集MM-MathInstruct-3M，训练的MathCoder-VL在开源模型中达到多模态数学推理SOTA，在几何问题上超越GPT-4o和Claude 3.5 Sonnet。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "多模态数学推理"
  - "图像到代码"
  - "跨模态对齐"
  - "数据合成"
  - "几何问题求解"
---

# MathCoder-VL: Bridging Vision and Code for Enhanced Multimodal Mathematical Reasoning

**会议**: ACL 2025  
**arXiv**: [2505.10557](https://arxiv.org/abs/2505.10557)  
**作者**: Ke Wang, Junting Pan, Linda Wei, Aojun Zhou, Weikang Shi, Zimu Lu, Han Xiao, Yunqiao Yang, Houxing Ren, Mingjie Zhan, Hongsheng Li (MMLab, CUHK)
**代码**: [https://github.com/mathllm/MathCoder](https://github.com/mathllm/MathCoder)  
**领域**: 多模态VLM  
**关键词**: 多模态数学推理, 图像到代码, 跨模态对齐, 数据合成, 几何问题求解

## 一句话总结

提出利用代码作为跨模态对齐的监督信号，构建860万图像-代码对数据集ImgCode-8.6M和300万多模态数学指令微调数据集MM-MathInstruct-3M，训练的MathCoder-VL在开源模型中达到多模态数学推理SOTA，在几何问题上超越GPT-4o和Claude 3.5 Sonnet。

## 研究背景与动机

### 问题背景
大型多模态模型（LMM）在数学推理方面仍远未达到理想水平，尤其在需要从图像中提取数学信息进行推理的任务中表现不佳。核心瓶颈在于两个方面：(1) 数学相关的视觉-文本跨模态对齐精度不足；(2) 缺乏多样化的数学图形生成能力来扩展训练数据。

### 已有工作的不足
- 传统图像-文本描述数据集（如ShareGPT4V、LAION-5B）主要面向自然场景，忽略了数学图形中的精细细节
- 自然语言描述无法完整传达图像中的所有数学信息（如精确角度、线段关系等），且无法保证正确性
- 现有多模态数学数据合成方法（Math-LLaVA、MammoTH-VL等）主要聚焦于文本多样性，图像多样性严重滞后
- MAVIS等工作虽然用代码生成几何图形，但仅包含人工设计的三种类型，缺乏多样性

### 核心动机
代码天然编码了生成对应图像所需的全部信息，代码与图像之间存在严格的一一对应关系。利用这一特性，可以：(1) 通过代码-图像对实现精确的跨模态对齐；(2) 通过修改代码参数自动合成多样化的新数学图形。

## 方法详解

### 关键设计1：迭代式图像到代码模型（FigCodifier）与ImgCode-8.6M

**数据收集**：收集300万数学相关图像，来源包括：
- DaTikZ训练集：119K图像-TikZ代码对（种子数据）
- K12题库：157万张数学题目图像，覆盖19个学科
- 数学教科书：8K本PDF提取的202K图像
- ArXiv：45K带TikZ代码的图像 + 681K无代码图像
- 开源数据集：MathV360K、MultiMath等

**迭代训练流程**（Model-in-the-Loop）：
1. 用119K种子数据在InternVL-Chat-V1-2-40B上训练初始图像到代码模型
2. 用该模型将300万收集的图像翻译为代码，执行代码渲染新图像
3. 仅保留成功生成的⟨Image^C, Code⟩对，加入数据集
4. 在扩大的数据集上重新训练模型（后续切换为InternVL2-8B以平衡性能和成本）
5. 反复迭代，最终得到FigCodifier模型

**TikZ到Python转换**：利用GPT-4o mini将TikZ代码翻译为Python代码，进一步扩展数据多样性，额外生成310万图像-Python对。

**数据清洗**：五步清洗流程——代码验证（仅保留可运行代码）、去重（移除4.4%）、质量过滤（移除3.7%低质数据）、代码长度过滤、图像质量检查（移除近全白图像约0.5%）。最终保留430万TikZ对 + 430万Python对 = ImgCode-8.6M。

代码成功率从初始的46.5%提升至最终的TikZ 81.2%、Python 84.5%。

### 关键设计2：MM-MathInstruct-3M数据集构建

**K12-2M**：
- 从460万数学题中，区分数学图形和公式（按尺寸），将公式转为LaTeX文本
- 用GPT-4o mini将简单解答扩展为详细的Step-by-Step CoT解答
- 最终得到200万包含实际图像的样本

**合成新图像的数学问题（New-1M）**：
1. 用FigCodifier（温度0.7）将K12-2M中的157万原始图像转换为代码，生成110万新图像-代码对
2. 用Qwen2.5-72B-Instruct基于新图像生成K12难度的数学推理问题
3. 用Qwen2.5-Math-72B-Instruct和Qwen2.5-72B-Instruct分别独立求解
4. 仅保留两个模型答案一致的样本（通过率51%），确保解答正确性
5. 清洗去重后得到100万新样本

### 关键设计3：两阶段训练

**阶段一——图像到代码中间训练**：
- 使用ImgCode-8.6M训练视觉编码器和MLP投影层
- 冻结LLM骨干，保留通用语言能力
- 目的：增强视觉编码器对数学视觉特征的提取能力

**阶段二——数学指令微调**：
- 使用MM-MathInstruct-3M（K12-2M + New-1M）全参数微调
- 目的：提升多模态数学问题求解能力

## 实验关键数据

### 实验1：主要结果对比

| 模型 | 参数量 | MATH-Vision | MathVerse | MathVista(GPS) | GAOKAO-MM | We-Math(S3) |
|------|--------|------------|-----------|---------------|-----------|-------------|
| GPT-4o | - | 30.4 | 50.8 | 64.7 | - | 43.6 |
| Claude 3.5 Sonnet | - | 37.9 | 49.0 | 64.4 | - | - |
| InternVL2-8B | 8B | 20.0 | 35.9 | 62.0 | 32.5 | 35.2 |
| InternVL2-76B | 76B | 23.6 | 42.8 | 67.8 | 41.2 | 49.1 |
| MAVIS-7B | 7B | 19.2 | 35.2 | 64.1 | - | 34.6 |
| MathGLM-Vision-9B | 9B | 19.2 | 44.2 | 64.4 | - | - |
| **MathCoder-VL-8B** | **8B** | **26.1** | **46.5** | **73.6** | **51.2** | **52.1** |
| Δ vs Base | | +6.1 | +10.6 | +11.6 | +18.7 | +16.9 |

MathCoder-VL-8B在所有6个指标上均达到同尺寸开源模型SOTA。在MathVista(GPS)上超越GPT-4o 8.9%、Claude 3.5 Sonnet 9.2%。

### 实验2：几何子集详细对比

| 模型 | Angle | Area | Length | 平均 |
|------|-------|------|--------|------|
| GPT-4o | 17.3 | 29.8 | 30.1 | 25.7 |
| InternVL2-8B | 20.8 | 22.4 | 20.5 | 21.2 |
| Math-LLaVA-13B | 20.2 | 18.4 | 17.6 | 18.7 |
| **MathCoder-VL-8B** | **48.6** | **32.2** | **32.1** | **37.6** |

在MATH-Vision平面几何子集上，MathCoder-VL-8B平均37.6%，超越GPT-4o的25.7%达11.9个百分点，角度子集更是达到48.6%。

### 实验3：消融实验

| 图像到代码中间训练 | 微调数据 | MATH-Vision | MathVerse | MathVista(GPS) | GAOKAO-MM |
|:--:|:--:|:--:|:--:|:--:|:--:|
| ✗ | K12-2M | 20.3 | 27.2 | 45.7 | 30.0 |
| ✓ | K12-2M | 22.0 | 33.0 | 64.4 | 33.8 |
| ✓ | K12-2M+New-1M | 21.7 | 35.4 | 66.4 | 37.5 |

中间训练带来MathVista(GPS) +18.7%的巨幅提升；合成新图像数据进一步在MathVerse上+2.4%、GAOKAO-MM上+3.7%。

## 关键发现

1. **代码是比自然语言更好的跨模态对齐信号**：代码与图像之间的对应关系精确且完整，不会遗失细节或引入错误
2. **中间训练对视觉理解影响最大**：在Vision-Only的MathVerse子集上提升高达11.0%，表明代码训练显著增强了纯视觉信息处理
3. **多步推理能力突出**：在We-Math三步问题上达到52.1%，超越GPT-4o的43.6%和InternVL2-76B的49.1%
4. **合成图像数据提升泛化性**：新图像带来的多样性使模型在不同于传统数学题的任务上也有显著提升（MathVista整体+5.0%）
5. **迭代训练显著提升代码生成质量**：TikZ代码成功率从46.5%提升至81.2%

## 亮点

- **创新的跨模态对齐范式**：首次系统性地利用代码作为图像-文本对齐的桥梁，解决了自然语言描述在数学领域的精度不足问题
- **大规模数据引擎**：860万图像-代码对是迄今最大的图像-代码数据集，且通过迭代方式自动扩展
- **首个图像合成驱动的数学数据集**：MM-MathInstruct-3M是首个同时包含新问题和新合成图像的高质量多模态数学数据集
- **几何理解能力突破**：在几何问题上以8B参数量大幅超越GPT-4o和Claude 3.5 Sonnet
- **实用的两阶段训练**：中间训练冻结LLM保留语言能力，微调阶段全量训练提升推理能力，设计合理

## 局限性

- **学科覆盖有限**：仅关注数学，未纳入物理、化学等其他STEM学科
- **语言单一**：数据集全部为英文，虽然在中文GAOKAO-MM上表现不错，但未专门构建中文数据
- **模型规模受限**：仅训练2B和8B模型，未探索更大模型的潜力
- **未使用强化学习**：未在后训练阶段使用GRPO等RL方法，可能遗漏进一步提升空间
- **代码生成依赖**：核心pipeline依赖图像到代码的转换质量，对非常复杂的图形可能仍有局限

## 与相关工作的对比

- **Math-LLaVA (Shi et al., 2024)**：通过图像复杂度分类增强问题，但不生成新图像，MathCoder-VL在MATH-Vision上+10.4%
- **MAVIS (Zhang et al., 2025)**：用代码生成几何和函数图像但仅三种人工设计类型，MathCoder-VL通过FigCodifier自动生成多样图像，在MATH-Vision上+6.9%
- **MathGLM-Vision (Yang et al., 2024)**：9B参数在MathVerse上44.2%，MathCoder-VL-8B达到46.5%且在其他指标全面超越
- **Multimath (Peng et al., 2024)**：7B模型在MathVista GPS上66.8%，MathCoder-VL-8B达73.6%
- **Math-PUMA (Zhuang et al., 2024)**：关注几何图形理解但依赖外部数据，MathCoder-VL在几何子集上大幅领先（平均37.6% vs 13.2%）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 利用代码作为跨模态对齐信号是非常巧妙的思路，迭代式数据引擎也是创新贡献
- 实验充分度: ⭐⭐⭐⭐ — 六个基准全面评测，消融实验详尽，但缺乏失败案例分析
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，方法和实验阐述充分，图表设计直观
- 价值: ⭐⭐⭐⭐⭐ — 数据、模型、代码全部开源，860万级数据集对社区贡献巨大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] The Role of Visual Modality in Multimodal Mathematical Reasoning: Challenges and Insights](the_role_of_visual_modality_in_multimodal_mathematical_reasoning_challenges_and_.md)
- [\[ACL 2025\] MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale](mammoth_vl_multimodal_reasoning.md)
- [\[ACL 2025\] We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?](wemath_knowledge_reasoning.md)
- [\[ACL 2025\] ChartCoder: Advancing Multimodal Large Language Model for Chart-to-Code Generation](chartcoder_chart_to_code.md)
- [\[CVPR 2026\] CodeMMR: Bridging Natural Language, Code, and Image for Unified Retrieval](../../CVPR2026/multimodal_vlm/codemmr_bridging_natural_language_code_and_image_for_unified_retrieval.md)

</div>

<!-- RELATED:END -->
