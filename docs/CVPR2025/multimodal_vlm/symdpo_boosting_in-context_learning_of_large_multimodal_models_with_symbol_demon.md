---
title: >-
  [论文解读] SymDPO: Boosting In-Context Learning of Large Multimodal Models with Symbol Demonstration Direct Preference Optimization
description: >-
  [CVPR 2025][多模态VLM][上下文学习] SymDPO 发现LMM在多模态ICL中存在"视觉上下文忽视"问题（用空白图替换示例图不影响性能），提出将示例中的文本答案替换为无语义随机符号，迫使模型必须理解视觉内容才能正确匹配符号与答案，通过DPO训练在OpenFlamingo和IDEFICS上一致提升了多模态ICL效果。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "上下文学习"
  - "符号示例"
  - "DPO"
  - "多模态大模型"
  - "视觉上下文利用"
---

# SymDPO: Boosting In-Context Learning of Large Multimodal Models with Symbol Demonstration Direct Preference Optimization

**会议**: CVPR 2025  
**arXiv**: [2411.11909](https://arxiv.org/abs/2411.11909)  
**代码**: [https://github.com/APiaoG/SymDPO](https://github.com/APiaoG/SymDPO)  
**领域**: 对齐RLHF  
**关键词**: 上下文学习, 符号示例, DPO, 多模态大模型, 视觉上下文利用

## 一句话总结

SymDPO 发现LMM在多模态ICL中存在"视觉上下文忽视"问题（用空白图替换示例图不影响性能），提出将示例中的文本答案替换为无语义随机符号，迫使模型必须理解视觉内容才能正确匹配符号与答案，通过DPO训练在OpenFlamingo和IDEFICS上一致提升了多模态ICL效果。

## 研究背景与动机

**领域现状**：大语言模型展现出了强大的上下文学习能力，给几个示例就能解决新任务。研究者将ICL扩展到多模态大模型（LMM），期望模型能从图文示例中学习任务模式。

**现有痛点**：LMM在多模态ICL中存在严重的"视觉上下文忽视"（Visual Context Overlook）问题——将示例中的图片替换为空白占位符甚至完全删除，模型性能几乎不受影响。这意味着模型实际上只是在跟随文本模式做pattern matching，而非真正利用视觉信息。

**核心矛盾**：VQA任务大多可以仅通过文本模式匹配解答（如yes/no问题中选择出现频率最高的答案），导致模型"偷懒"不看图片。现有DPO方法针对通用任务优化，不特化于ICL场景，无法解决这一根本性的视觉-文本对齐缺失。

**本文目标** 如何强制LMM在ICL时真正利用视觉信息，而非仅依赖文本模式？

**切入角度**：一个巧妙的"反向设计"——如果模型可以通过文本模式匹配回避视觉理解，那就把文本答案替换为无语义的随机符号，让模型"无路可走"只能看图片。

**核心 idea**：将ICL示例中的答案替换为无意义符号（如"rhondda"代替"narrow"），迫使模型建立图像-符号映射来回答问题，而非依赖文本-文本的pattern matching。

## 方法详解

### 整体框架

SymDPO的流程分三步：(1) 从VQA数据集构建ICL格式数据 $D_1, D_2, \ldots, D_N, F$（N个示例+1个最终问题）；(2) 构建标准DPO正负样本对；(3) 核心创新——在DPO数据中将示例答案替换为无语义符号，形成SymDPO训练数据。推理时不使用符号，模型正常做ICL。

### 关键设计

1. **ICL数据构建与分组**:

    - 功能：构建结构化的多模态ICL训练数据
    - 核心思路：从GQA、VQAv2和ImageNet中收集图像-问题-答案三元组，将相似任务类型的问题分组（如二元yes/no问题、颜色属性问题、物体数量问题等）。每组包含N个示例和1个最终问答对，确保示例中至少包含2种不同答案，且至少有一个与最终答案匹配
    - 设计动机：保证示例之间有足够多样性，避免模型通过简单的多数投票解决问题

2. **符号替换策略（Symbol Demonstration）**:

    - 功能：消除文本模式匹配的可能性，强制视觉理解
    - 核心思路：将每个示例的答案 $A_i$ 替换为语义无关的随机符号 $S_i$（如"rhondda"、"odwyer"），得到 $\dot{D}_i = \{I_i, Q_i, S_i\}$。关键约束：某个示例 $D_k$ 的符号 $S_k$ 与最终答案对应的类别匹配，DPO的正样本为 $S_k$，负样本为不匹配的符号 $S_j$。论文维护5种不同的符号配置比例，包括标准替换、擦除示例问题的变体等
    - 设计动机：符号无语义信息，模型必须真正理解示例图片的视觉内容才能将正确的符号与视觉答案对应起来，由此建立真正的视觉-语义绑定

3. **SymDPO训练目标**:

    - 功能：通过偏好学习让模型学会利用视觉上下文
    - 核心思路：标准DPO目标 $\mathcal{L}_S = -\mathbb{E} \log \sigma(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})$，关键是训练数据使用符号化示例，迫使模型在偏好学习中强化视觉-符号的对应关系
    - 设计动机：直接用符号数据做SFT（SymTune）效果差甚至有害，因为自回归训练可能学到错误的生成模式。DPO通过对比学习提供更稳健的反馈信号

### 损失函数 / 训练策略

标准DPO损失，数据集87.2万样本中选取1万样本训练，使用GPT-4V进行质量增强。8×A100训练约1小时。lr=5e-6，线性退火。推理时无需符号，正常ICL即可。

## 实验关键数据

### 主实验

OpenFlamingo-9B结果：

| 设置 | 方法 | COCO (CIDEr) | Flickr-30K | VQAv2 | OK-VQA | TextVQA |
|------|------|-------------|------------|-------|--------|---------|
| 4-shot | Base | 89.0 | 65.8 | 51.0 | 40.1 | 26.1 |
| 4-shot | General DPO | 89.6 | 66.0 | 51.2 | 40.5 | 26.2 |
| 4-shot | **SymDPO** | **93.8** | **69.4** | 51.1 | **41.0** | 26.3 |
| 8-shot | Base | 96.3 | 62.9 | 54.8 | 41.1 | 27.3 |
| 8-shot | **SymDPO** | **102.5** | **67.3** | 55.0 | **42.3** | 27.7 |
| 16-shot | Base | 98.8 | 62.8 | 56.1 | 42.7 | 27.6 |
| 16-shot | **SymDPO** | **104.3** | **64.9** | 56.4 | **44.5** | 28.2 |

IDEFICS-9B结果：

| 设置 | 方法 | COCO (CIDEr) | Flickr-30K | OK-VQA |
|------|------|-------------|------------|--------|
| 8-shot | Base | 97.0 | 61.9 | 47.7 |
| 8-shot | **SymDPO** | **103.8** | **66.1** | **49.5** |
| 16-shot | Base | 99.7 | 64.5 | 48.4 |
| 16-shot | **SymDPO** | **107.9** | **69.3** | **50.6** |

### 消融实验

| 方法 | COCO 4-shot | Flickr 4-shot | OK-VQA 4-shot |
|------|------------|---------------|---------------|
| SymTune (SFT) | +7.8 | -5.2 | +0.3 |
| General DPO | +0.6 | +0.2 | +0.4 |
| **SymDPO** | **+4.7** | **+2.1** | **+1.0** |

SymDPO + RICES（检索式示例选择）叠加效果：

| 配置 | COCO 4-shot |
|------|------------|
| Base | 82.7 |
| + RICES | 90.5 (+7.8) |
| + SymDPO | 87.4 (+4.7) |
| **+ SymDPO & RICES** | **93.5 (+10.8)** |

### 关键发现

- **视觉上下文忽视是真实存在的**：用空白图/无图替换示例图片后，基线模型性能几乎不变；而SymDPO训练后的模型在空白图下性能显著下降，证明模型确实学会了利用视觉信息
- **SymDPO vs SymTune**：直接用符号数据SFT会在captioning上产生严重退化（Flickr -5.2），而DPO的偏好学习方式对模型知识的影响更加稳健
- **与示例选择策略互补**：SymDPO与RICES的提升是叠加的（4.7+7.8→10.8），说明两者解决的是不同维度的问题
- **符号数据比例**：70-100%符号数据比例效果最佳，说明符号训练是提升ICL的主要驱动力

## 亮点与洞察

- **"视觉上下文忽视"的发现本身就很有价值**：简单的blank image实验就能揭示LMM的根本缺陷，为后续研究提供了清晰的诊断工具
- **符号替换的思路极其简洁优雅**：不需要复杂的模型改动或额外模块，仅通过训练数据的巧妙设计就解决了根本问题。核心洞察是——如果模型能通过捷径解决问题，就堵住那条捷径
- **泛化到多个模型架构**：OpenFlamingo和IDEFICS架构不同但都有效，说明视觉上下文忽视是通用问题而非特定架构的缺陷

## 局限与展望

- 仅测试了OpenFlamingo和IDEFICS两种架构，未验证LLaVA、GPT-4V等更先进的模型
- TextVQA上提升很小（0.2-0.6），表明某些视觉文本任务对视觉-符号对齐的需求不大
- 符号的选择看起来是随意的（"rhondda"、"odwyer"），没有系统研究符号属性对效果的影响
- 需要GPT-4V做数据质量增强，限制了方法的可复现性和规模化

## 相关工作与启发

- **vs General DPO**: 通用DPO在ICL任务上几乎无提升（COCO +0.6），说明标准偏好数据无法解决视觉忽视问题
- **vs MIA-DPO**: 另一种多模态DPO变体，提升也很有限（+0.2-1.8），不如SymDPO的针对性设计
- **vs Symbol Tuning (NeurIPS 2023)**: 灵感来源，但直接SFT效果差；SymDPO将符号思路与DPO结合，在偏好学习框架下更好地保护了模型能力

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 符号替换的想法极其简洁而深刻，"堵住捷径"的思路具有广泛启发意义
- 实验充分度: ⭐⭐⭐⭐ 多模型+多benchmark+ablation充分，但架构覆盖可更广
- 写作质量: ⭐⭐⭐⭐ 问题motivation展示清晰（blank image实验），整体逻辑流畅
- 价值: ⭐⭐⭐⭐ 揭示了LMM ICL的根本缺陷并提供了优雅解决方案，对多模态ICL研究有重要推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Debiasing Multimodal Large Language Models via Noise-Aware Preference Optimization](debiasing_multimodal_large_language_models_via_noise-aware_preference_optimizati.md)
- [\[CVPR 2025\] Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment](task_preference_optimization_improving_multimodal_large_language_models_with_vis.md)
- [\[ICML 2026\] TUR-DPO: Topology- and Uncertainty-Aware Direct Preference Optimization](../../ICML2026/multimodal_vlm/tur-dpo_topology-_and_uncertainty-aware_direct_preference_optimization.md)
- [\[CVPR 2025\] HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks](hificl_high-fidelity_in-context_learning_for_multimodal_tasks.md)
- [\[CVPR 2025\] Mastering Negation: Boosting Grounding Models via Grouped Opposition-Based Learning](mastering_negation_boosting_grounding_models_via_grouped_opposition-based_learni.md)

</div>

<!-- RELATED:END -->
