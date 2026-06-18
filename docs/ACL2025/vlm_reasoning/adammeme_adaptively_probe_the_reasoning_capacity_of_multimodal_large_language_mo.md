---
title: >-
  [论文解读] AdamMeme: Adaptively Probe the Reasoning Capacity of Multimodal Large Language Models on Harmfulness
description: >-
  [ACL 2025][多模态VLM][meme理解] 提出AdamMeme——一个基于多智能体协作的自适应评估框架，通过迭代生成更具挑战性的meme样本来探测多模态大语言模型(mLLM)在有害内容理解上的推理能力和特定弱点。 社交媒体时代，多模态meme（表情包/梗图）已成为网络传播的重要载体。许多meme包含隐含的仇恨、歧…
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "meme理解"
  - "有害性检测"
  - "多模态评估"
  - "多智能体"
  - "自适应探测"
---

# AdamMeme: Adaptively Probe the Reasoning Capacity of Multimodal Large Language Models on Harmfulness

**会议**: ACL 2025  
**arXiv**: [2507.01702](https://arxiv.org/abs/2507.01702)  
**代码**: [https://github.com/viczxchen/AdamMeme](https://github.com/viczxchen/AdamMeme)  
**领域**: 多模态 / VLM  
**关键词**: meme理解, 有害性检测, 多模态评估, 多智能体, 自适应探测

## 一句话总结

提出AdamMeme——一个基于多智能体协作的自适应评估框架，通过迭代生成更具挑战性的meme样本来探测多模态大语言模型(mLLM)在有害内容理解上的推理能力和特定弱点。

## 研究背景与动机

社交媒体时代，多模态meme（表情包/梗图）已成为网络传播的重要载体。许多meme包含隐含的仇恨、歧视或误导性内容，需要AI系统能够准确理解和识别其有害性。多模态大语言模型（mLLM）如GPT-4V、LLaVA等在视觉语言理解上取得了显著进展，但它们在理解meme的**隐含有害性**方面的能力如何，需要系统性评估。

现有meme有害性评估基准存在几个关键局限：

**静态数据集**：评估基于固定的数据集（如Hateful Memes），无法跟上网络meme的动态演变

**模型无关性**：所有模型使用相同的测试集，无法针对特定模型进行深度探测

**仅关注准确率**：只看对/错比例，无法提供模型弱点的细粒度分析

**缺乏挑战性**：简单样本占多数，难以区分模型间的真实能力差异

核心矛盾是：meme的有害性判断需要复杂的多模态推理（理解图文交互、文化隐喻、讽刺反语等），而现有评估方法无法充分挖掘模型在这方面的缺陷。

本文的切入角度是**自适应评估**：不是用固定的题目考所有模型，而是针对每个模型的弱点，动态生成更具挑战性的测试样本，类似于"自适应考试"的思想。核心idea：通过多智能体协作框架，让一个agent负责生成挑战性meme，另一个agent负责评估目标模型的表现，迭代协作逐步暴露模型的推理盲区。

## 方法详解

### 整体框架

AdamMeme是一个三阶段迭代管道：
1. **有害性挖掘（Harmfulness Mining）**：从种子数据集中挖掘具有特定有害类型的meme，并利用LLM生成"误信陈述"（misbelief statement）来构造更细粒度的评估维度
2. **模型评分（Model Scoring）**：让目标mLLM对挖掘出的meme进行有害性判断，通过对比参考答案评估模型表现
3. **迭代精炼（Iterative Refinement）**：根据模型的表现反馈，自适应地生成更具针对性的挑战样本，暴露模型的特定弱点

整个流程可多轮迭代，每轮将模型的错误模式反馈给挖掘agent，生成更加针对性的测试数据。

### 关键设计

1. **有害性挖掘Agent（Harmfulness Mining）**:

    - 功能：从原始meme数据中识别有害性属性，并系统性地分解有害性的细粒度维度
    - 核心思路：首先使用OCR-SAM工具将meme中的文字擦除，获得"纯图像"版本。然后让LLM分析原始meme的有害性来源——是图像本身、文字本身、还是图文交互产生的。对每个有害维度生成misbelief statement（如"女性不应该从事科技工作"），作为评估参考标准
    - 设计动机：meme的有害性往往不是显式的，而是通过图文间的隐含关系和文化背景传达。分离图像和文字可以帮助识别有害性的来源，而misbelief statement为评估提供了明确的参考基准

2. **多维度模型评分（Model Scoring）**:

    - 功能：从多个角度评估目标mLLM对meme有害性的理解能力
    - 核心思路：设计多层次的评估问题：(a) 二分类——是否有害？(b) 有害性类型——属于哪种有害类型？(c) 推理解释——为什么有害？通过对比模型输出和参考答案（含misbelief statement），使用评估agent打分。评分不仅看结果正确性，还评估推理链的合理性
    - 设计动机：仅看最终判断无法反映模型的理解深度。一个模型可能"蒙对"答案但推理链完全错误，多维度评分能更准确地捕捉模型的真实推理能力

3. **迭代精炼Agent（Iterative Refinement）**:

    - 功能：根据模型的错误模式，自适应地调整评估数据的难度和分布
    - 核心思路：分析模型在前一轮评估中的错误案例，提取错误模式（如"无法识别讽刺""忽视图文对比""对特定文化隐喻不敏感"），然后有针对性地从种子数据池中挖掘或生成更多类似的挑战样本。随着迭代进行，评估集对目标模型越来越"难"
    - 设计动机：静态评估只能提供"快照"，而自适应迭代可以持续探测模型的能力边界。类似于渗透测试的思想，通过不断施压找到系统的薄弱环节

### 损失函数 / 训练策略

AdamMeme是一个**评估框架**（evaluation framework），不涉及模型训练和损失函数。其核心策略是多智能体间的信息流：
- Mining Agent → Scoring Agent：传递挖掘出的meme和参考答案
- Scoring Agent → Refinement Agent：传递模型的错误模式分析
- Refinement Agent → Mining Agent：传递需要重点探测的有害性维度

## 实验关键数据

### 数据集

使用三个公开meme数据集：
- **MAMI**：多模态厌女检测数据集
- **HarM**：来自MOMENTA项目的有害meme数据
- **FHM**：Facebook Hateful Memes Challenge数据集

### 主实验

| 目标模型 | 初轮准确率 | 迭代后准确率 | 弱点维度数 | 暴露的主要弱点 |
|----------|-----------|-------------|-----------|---------------|
| GPT-4V | 较高 | 显著下降 | 多 | 文化隐喻理解 |
| LLaVA-1.5 | 中等 | 明显下降 | 多 | 讽刺反语识别 |
| InstructBLIP | 较低 | 进一步下降 | 多 | 图文交互推理 |
| MiniGPT-4 | 较低 | 持续下降 | 多 | 隐含偏见识别 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无迭代精炼 | 基线 | 相当于静态评估，区分度有限 |
| 1轮迭代 | 显著提升区分度 | 开始暴露模型间的能力差异 |
| 3轮迭代 | 最优 | 模型弱点充分暴露，继续迭代边际收益递减 |
| 无文字擦除 | 降低 | 无法区分有害性来源（图像vs文字vs交互） |
| 无misbelief statement | 降低 | 评估参考不够精确，评分一致性下降 |
| 单agent vs 多agent | 多agent更优 | 专业分工提升了挖掘和评估的质量 |

### 关键发现
- **不同mLLM有不同的弱点模式**：GPT-4V在文化敏感内容上表现较弱，而LLaVA在讽刺类meme上推理能力不足
- **迭代精炼有效**：经过3轮迭代后，AdamMeme能将模型的有效准确率降低显著百分点，暴露出静态评估无法发现的弱点
- **图文交互是最大难点**：所有模型在需要理解图文对比/矛盾关系的meme上表现最差
- **模型规模不决定一切**：中等规模的开源模型在某些有害性维度上可能优于大规模闭源模型

## 亮点与洞察

- **自适应评估范式**：突破了静态benchmark的局限，为mLLM评估引入了动态、个性化的思路。这种"考试自适应"的评估范式可以迁移到其他NLP/CV任务的评估中
- **多智能体分工**：Mining、Scoring、Refinement三个agent各司其职，信息流清晰，是一个可复用的多智能体协作框架模板
- **Misbelief Statement设计**：将隐含有害性显式化为"错误信念陈述"，不仅提供了评估基准，也有助于理解有害性的本质
- **图文分离分析**：通过OCR-SAM擦除文字创建纯图像版本，是分析图文交互有害性的有效方法

## 局限与展望

- **评估agent的偏见**：用LLM（如GPT-4）作为评估agent，其本身可能存在偏见，影响评分的客观性
- **文化局限性**：主要基于英文meme数据集，对中文、日文等其他语言和文化的meme覆盖不足
- **生成质量**：迭代精炼依赖agent的生成/挑选能力，如果agent本身无法生成足够高质量的挑战样本，评估的天花板受限
- **计算开销**：多轮迭代需要大量API调用（多个LLM交互），成本较高
- **缺少人工验证**：自动化评估的可靠性需要更多人工标注来验证

## 相关工作与启发

- **vs Hateful Memes Challenge（Facebook）**: HMC是静态benchmark+人类标注基线，AdamMeme是动态自适应评估框架，能持续挖掘模型弱点
- **vs MM-SafetyBench**: MM-SafetyBench关注安全对齐和越狱攻击，AdamMeme专注于有害性*理解*能力的评估，是评估维度的差异
- **vs 对抗样本方法（adversarial attack）**: 对抗攻击只关注"骗过模型"，AdamMeme不仅找到弱点样本还提供细粒度的能力分析报告

## 评分

- 新颖性: ⭐⭐⭐⭐ 自适应评估范式在meme领域属首次，多agent设计新颖
- 实验充分度: ⭐⭐⭐⭐ 多模型多数据集对比，迭代分析详细
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，图示丰富
- 价值: ⭐⭐⭐⭐ 为mLLM评估提供了新思路，方法可迁移

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning](benchmarking_and_improving_large_vision-language_models_for_fundamental_visual_g.md)
- [\[ACL 2025\] Can Multimodal Large Language Models Understand Spatial Relations?](spatialmqa_mllm_spatial_relations.md)
- [\[ACL 2025\] AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models](alignmmbench_evaluating_chinese_multimodal_alignment_in_large_vision-language_mo.md)
- [\[ACL 2025\] Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?](token_pruning_in_multimodal_large_language_models_are_we_solving_the_right_probl.md)
- [\[ACL 2025\] GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art](godbench_a_benchmark_for_multimodal_large_language_models_in_video_comment_art.md)

</div>

<!-- RELATED:END -->
