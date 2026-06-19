---
title: >-
  [论文解读] CHEER-Ekman: Fine-grained Embodied Emotion Classification
description: >-
  [ACL 2025][机器人][具身情感分类] 本文提出CHEER-Ekman数据集，将CHEER数据集的二元具身情感标注扩展为Ekman六类基础情绪，并采用基于LLM的自动Best-Worst Scaling（BWS）技术实现无需任务特定训练的细粒度情感分类，性能超越有监督BERT。 领域现状： 情绪不仅是抽象的心理状态…
tags:
  - "ACL 2025"
  - "机器人"
  - "具身情感分类"
  - "Ekman情绪"
  - "Best-Worst Scaling"
  - "大语言模型"
  - "提示工程"
---

# CHEER-Ekman: Fine-grained Embodied Emotion Classification

**会议**: ACL 2025  
**arXiv**: [2506.01047](https://arxiv.org/abs/2506.01047)  
**代码**: [https://github.com/menamerai/cheer-ekman](https://github.com/menamerai/cheer-ekman)  
**领域**: 机器人  
**关键词**: 具身情感分类, Ekman情绪, Best-Worst Scaling, 大语言模型, 提示工程

## 一句话总结

本文提出CHEER-Ekman数据集，将CHEER数据集的二元具身情感标注扩展为Ekman六类基础情绪，并采用基于LLM的自动Best-Worst Scaling（BWS）技术实现无需任务特定训练的细粒度情感分类，性能超越有监督BERT。

## 研究背景与动机

**领域现状**: 情绪不仅是抽象的心理状态，还与身体体验深度交织——快乐时微笑、恐惧时心跳加速。这种"具身情感"（embodied emotion）在NLP中较少被研究，现有工作主要关注显式情感分类和情感分析。

**现有痛点**: CHEER数据集（Zhuang et al., 2024）虽然提供了7300个人工标注的身体部位表达情感句子，但只支持二元分类（是否包含具身情感），无法区分具体情绪类型——例如心跳加速是恐惧还是兴奋。

**核心矛盾**: 细粒度情感标注需要大量人工成本，而直接用LLM做zero-shot分类在指令遵循上表现不稳定，容易输出错误结果。

**本文目标**: （1）构建细粒度具身情感分类数据集；（2）找到无需任务特定训练即可有效分类的方法。

**切入角度**: 将Ekman六类基础情绪（Joy, Sadness, Anger, Disgust, Fear, Surprise）引入具身情感识别，并借鉴情感强度标注中的自动BWS技术来替代直接分类。

**核心 idea**: 通过Best-Worst Scaling的比较判断机制，让LLM在无监督条件下实现超越BERT的细粒度具身情感分类。

## 方法详解

### 整体框架

本文包含三个核心组件：（1）通过提示简化和CoT提升LLM的具身情感检测能力；（2）构建CHEER-Ekman数据集；（3）采用BWS框架进行情感分类。

### 关键设计

#### 1. 提示简化策略（Prompt Simplification）

- **功能**: 将原始技术性提示语改写为简单日常语言
- **核心思路**: 对比base prompt（Zhuang et al., 2024原始提示）与simplified prompt（降低句法和词汇复杂度）在Llama-3.1和DeepSeek-R1上的效果。采用logit概率比较"True"和"False"的方式获取确定性输出
- **设计动机**: 技术定义可能对LLM造成理解障碍，简化语言可减少潜在困惑

#### 2. Chain-of-Thought (CoT) 提示

- **功能**: 通过多步推理引导模型理解身体-情感关系
- **核心思路**: 设计三种CoT变体：2-step（评估情感因果和无意识表达）、3-step（增加身体部位识别）、simplified 2-step（语言简化的2-step）
- **设计动机**: 显式因果推理可帮助8B小模型达到接近70B大模型的性能

#### 3. CHEER-Ekman数据集构建

- **功能**: 为CHEER数据集的1350个正样本标注Ekman六类情绪
- **核心思路**: 招募2名标注者，提供句子、相关身体部位和上下文（最多3个前导句），选择最匹配的情绪。Cohen's Kappa一致性为0.64
- **情绪分布**: Fear 24.7%, Joy 21.2%, Sadness 19.3%, Surprise 13.3%, Disgust 12.5%, Anger 9.0%

#### 4. Best-Worst Scaling (BWS) 情感分类

- **功能**: 通过比较判断而非直接分类来预测情绪
- **核心思路**: 向LLM呈现4个句子的元组，要求识别最能和最不能代表某种Ekman情绪的实例。通过公式 $\frac{\#Best - \#Worst}{\#Total}$ 计算每个句子在每种情绪上的得分，选择最高分情绪作为预测
- **设计动机**: 比较判断比直接分类更稳定，避免LLM不遵循指令的问题。测试了从 $2N$ 到 $72N$ 的元组数量

### 损失函数/训练策略

本文为无训练方法，不涉及损失函数。BWS的关键超参数是元组数量，实验发现 $36N$ 达到最佳性能，之后出现平台效应。

## 实验关键数据

### 主实验

**具身情感检测任务**（二元分类，CHEER数据集，7300句）：

| 模型 | Macro F1 | EE F1 | Neutral F1 |
|------|----------|-------|------------|
| Llama-70B (base prompt) | 37.2 | 35.3 | 39.0 |
| Llama-70B (simple prompt) | 66.7 | 52.8 | 80.6 |
| DeepSeek-70B (base prompt) | 32.6 | 33.7 | 31.5 |
| DeepSeek-70B (simple prompt) | 74.2 | 58.9 | 89.5 |
| GPT-3.5 (base prompt) | 70.2 | 53.5 | 86.9 |
| BERT (fine-tuned) | 83.5 | 72.6 | 94.4 |

**具身情感分类任务**（六类分类，CHEER-Ekman数据集，1350句）：

| 模型 | Macro F1 | Joy | Sadness | Fear | Anger | Disgust | Surprise |
|------|----------|-----|---------|------|-------|---------|----------|
| Llama-8B (zero-shot) | 31.6 | 39.4 | 43.6 | 26.6 | 32.2 | 19.1 | 28.5 |
| DeepSeek-8B (zero-shot) | 28.4 | 43.3 | 35.7 | 33.1 | 23.1 | 14.8 | 20.2 |
| BWS 36N (Llama-8B) | **50.6** | 66.7 | 64.7 | 48.0 | 53.2 | 22.0 | 48.9 |
| BERT (supervised) | 49.6 | 68.2 | 57.5 | 50.1 | 30.2 | 56.1 | 35.7 |

### 消融实验

**CoT对8B模型的提升效果**：

| 模型 | Macro F1 |
|------|----------|
| DeepSeek-8B (2-step) | 52.2 |
| DeepSeek-8B (3-step) | 57.4 |
| DeepSeek-8B (2-step-simple) | 67.5 |
| Llama-8B (2-step) | 53.4 |
| Llama-8B (3-step) | 54.8 |
| Llama-8B (2-step-simple) | 60.1 |

**BWS元组数量消融**：

| 元组数 | Macro F1 |
|--------|----------|
| 4N | 41.8 |
| 12N | 44.6 |
| 36N | 50.6 |
| 48N | 49.8 |
| 72N | 49.5 |

### 关键发现

1. 简化提示使Llama-70B的F1提升了29.5个点，DeepSeek-70B提升了41.6个点
2. CoT使8B模型与70B模型的差距缩小到6.7个F1点
3. BWS with 36N元组达到50.6 F1，超越有监督BERT的49.6
4. 错误分析显示93.3%为假阳性，主要来自隐喻表达（41%）、功能性动作（42%）和无动作身体部位引用（17%）

## 亮点与洞察

1. **反直觉发现**: 简化日常语言的提示比技术定义的提示效果更好（F1提升近30点），说明LLM在理解领域术语时存在显著困难
2. **比较优于分类**: BWS通过比较机制绕过了LLM直接分类时的不稳定性，是一种巧妙的工程方案
3. **小模型潜力**: CoT+简化提示使8B参数模型接近70B效果，展示了提示工程的巨大价值
4. **数据集价值**: CHEER-Ekman填补了具身情感细粒度分类的空白

## 局限与展望

1. 数据集仅1350句，规模较小，可能存在正样本偏差
2. 简化提示可能导致对特定措辞的过拟合，忽略更微妙的隐喻表达
3. BWS高元组数量带来显著计算开销，可扩展性受限
4. 受限于context window，未能实现few-shot设置
5. 可探索更细粒度的情绪分类体系（如27类情绪）

## 相关工作与启发

- **Bagdon et al. (2024)**: 自动BWS进行情感强度标注的先驱工作，本文将其扩展到分类任务
- **CHEER (Zhuang et al., 2024)**: 原始具身情感检测数据集，本文的基础
- **Ekman (1992)**: 六类基础情绪框架，提供了分类学基础
- **启发**: BWS的比较判断思路可推广到其他LLM难以直接分类的任务

## 评分

| 维度 | 评分 |
|------|------|
| 新颖性 | ⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 价值 | ⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Phoenix: A Motion-based Self-Reflection Framework for Fine-grained Robotic Action Correction](../../CVPR2025/robotics/phoenix_a_motion-based_self-reflection_framework_for_fine-grained_robotic_action.md)
- [\[AAAI 2026\] Cross Modal Fine-Grained Alignment via Granularity-Aware and Region-Uncertain Modeling](../../AAAI2026/robotics/cross_modal_fine-grained_alignment_via_granularity-aware_and_region-uncertain_mo.md)
- [\[CVPR 2025\] SortScrews: A Dataset and Baseline for Real-time Screw Classification](../../CVPR2025/robotics/sortscrews_a_dataset_and_baseline_for_real-time_screw_classification.md)
- [\[ECCV 2024\] LLM as Copilot for Coarse-Grained Vision-and-Language Navigation](../../ECCV2024/robotics/llm_as_copilot_for_coarse-grained_vision-and-language_navigation.md)
- [\[NeurIPS 2025\] PROFIT: A Specialized Optimizer for Deep Fine Tuning](../../NeurIPS2025/robotics/profit_a_specialized_optimizer_for_deep_fine_tuning.md)

</div>

<!-- RELATED:END -->
