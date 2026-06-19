---
title: >-
  [论文解读] LLaVA-Critic: Learning to Evaluate Multimodal Models
description: >-
  [CVPR 2025][多模态VLM][多模态评估器] LLaVA-Critic 是首个开源的通用多模态评估模型，通过在精心构建的113k评估指令数据上训练，使开源LMM具备了接近GPT-4o水平的Pointwise评分和Pairwise排序能力，并可作为奖励模型为迭代DPO提供有效的偏好信号，超越基于人类反馈训练的LLaVA-RLHF奖励模型。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "多模态评估器"
  - "LMM-as-a-Judge"
  - "偏好学习"
  - "评估指令数据"
  - "奖励信号"
---

# LLaVA-Critic: Learning to Evaluate Multimodal Models

**会议**: CVPR 2025  
**arXiv**: [2410.02712](https://arxiv.org/abs/2410.02712)  
**代码**: [https://github.com/LLaVA-VL/LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)  
**领域**: 多模态VLM  
**关键词**: 多模态评估器, LMM-as-a-Judge, 偏好学习, 评估指令数据, 奖励信号

## 一句话总结

LLaVA-Critic 是首个开源的通用多模态评估模型，通过在精心构建的113k评估指令数据上训练，使开源LMM具备了接近GPT-4o水平的Pointwise评分和Pairwise排序能力，并可作为奖励模型为迭代DPO提供有效的偏好信号，超越基于人类反馈训练的LLaVA-RLHF奖励模型。

## 研究背景与动机

随着LMM进入后训练时代，"学习评估"的能力变得至关重要：(1) 大量评估Benchmark依赖GPT-4V/4o作为Judge，成本高且不可定制；(2) 偏好学习（DPO/RLHF）需要可靠的奖励信号，但收集人类反馈昂贵且难以规模化；(3) 推理时搜索（如Best-of-N）需要评估器来选择最优回复。

核心矛盾在于：现有开源LMM虽然在各种视觉任务上取得长足进步，但在"判断回复质量"这一判别性能力上几乎未被训练过。直接用LLaVA-OneVision做评估时，它倾向于给出固定分数（如WildVision上总说"Tie"，MMHal上总打"6分"），无法提供有效的区分度。

LLaVA-Critic的切入角度：将"评估"视为一种可训练的指令遵循能力，通过构建高质量的评估指令数据来教模型"如何做好Judge"。核心idea：一个好的评估器不仅要给出分数，还需要提供有理有据的判断理由。

## 方法详解

### 整体框架

LLaVA-Critic的构建分为两步：(1) 数据收集——为Pointwise评分和Pairwise排序两种设置构建113k评估指令数据；(2) 模型训练——在LLaVA-OneVision预训练检查点上微调1个epoch。训练完成后的模型可应用于两个场景：作为评估器替代GPT-4o（Scenario 1），以及作为奖励模型提供偏好信号（Scenario 2）。

### 关键设计

1. **Pointwise评估数据构建**:
    - 功能：训练模型根据特定评估标准对单个回复进行打分并给出理由
    - 核心思路：从8个多模态指令微调数据集中选取指令（覆盖通用对话、复杂推理、OCR、医学、机器人等领域），收集12个off-the-shelf LMM的回复（来自VLFeedback），并用GPT-4o生成高质量参考答案。关键创新是构建了一个来自7个主流评估Benchmark的**评估提示池**（evaluation prompt pool），包含LLaVA-Bench、LLaVA-Wilder、MMVet、MMHal-Bench等的评估标准。通过GPT-4o作为Judge对每个(指令, 回复, 评估标准)组合输出分数和理由
    - 设计动机：不同Benchmark的评估标准差异很大（视觉聊天vs详细描述vs幻觉检测），模型需要学习理解和遵循多样化的评估提示。最终产出18,915个图像-问题对和72,782个Pointwise样本

2. **Pairwise排序数据构建**:
    - 功能：训练模型比较两个回复并判断偏好关系
    - 核心思路：从VLFeedback、LLaVA-RLHF、RLHF-V三个数据集收集已有偏好关系的回复对。VLFeedback中按GPT-4V三维度评分差距>0.6筛选20k对，另加5k个Tie样本确保多样性；RLHF和RLHF-V分别提供9.4k和5.7k人工标注偏好对。设计30个多样化评估提示模板，每对随机分配模板。用GPT-4o生成判断理由
    - 设计动机：Pairwise评估在实际中极为常见（Arena排名、A/B测试），且需要处理平局情况。30个模板确保模型不会过拟合特定评估格式。总计40.1k Pairwise样本

3. **迭代DPO偏好学习 (Scenario 2)**:
    - 功能：利用LLaVA-Critic作为奖励模型为偏好学习提供信号
    - 核心思路：对每个问题-图像对，用策略模型随机生成 $K=5$ 个候选回复。构造所有 $K\times(K-1)$ 个有序对，LLaVA-Critic对每对产生相对评分 $a_{ij}$。汇总得到每个回复的奖励分 $r_i = \sum_{k \neq i} a_{ki} - \sum_{l \neq i} a_{il}$，选取最高分作为 $y^+$、最低分作为 $y^-$ 进行DPO训练。迭代 $M=3$ 轮
    - 设计动机：通过所有有序对打分并对称聚合，有效缓解LLaVA-Critic可能存在的位置偏差（先出现的回复可能被偏好）。这种"循环赛"式的评分比单次比较更稳健

### 损失函数 / 训练策略

- LLaVA-Critic训练：标准交叉熵损失，同时在分数和理由上计算loss。学习率 $2\times10^{-6}$，batch size 32，1个epoch
- 偏好学习：标准DPO损失，温度0.7，top-p 0.9用于候选回复采样
- 完整数据集113k（72.8k Pointwise + 40.1k Pairwise）；精简版53k（42k + 11k）

## 实验关键数据

### 主实验：Pointwise评分（Pearson-r与GPT-4o的相关性）

| 评估器 | ImageDC | MMVet | WildVision | LLaVA-B | LLaVA-W | L-Wilder | MMHal | Avg |
|--------|---------|-------|-----------|---------|---------|----------|-------|-----|
| LLaVA-OV-7B | 0.056 | 0.349 | 0.251 | 0.335 | 0.533 | 0.592 | 0.433 | 0.364 |
| Qwen2-VL-7B | 0.199 | 0.463 | 0.096 | 0.208 | 0.476 | 0.694 | 0.329 | 0.352 |
| LLaVA-Critic-7B | **0.735** | **0.733** | **0.616** | **0.510** | **0.843** | **0.940** | **0.748** | **0.732** |
| LLaVA-OV-72B | 0.718 | 0.680 | 0.446 | 0.436 | 0.716 | 0.824 | 0.620 | 0.634 |
| LLaVA-Critic-72B | **0.802** | **0.723** | **0.705** | **0.524** | **0.782** | **0.951** | **0.790** | **0.754** |

### 主实验：Pairwise排序（WildVision Arena对齐人类偏好）

| 评估器 | Acc(含Tie)↑ | Acc(不含Tie)↑ | Kendall's τ↑ |
|--------|-----------|-------------|------------|
| GPT-4o | 0.617 | 0.734 | 0.819 |
| GPT-4V | 0.620 | 0.733 | 0.787 |
| LLaVA-OV-7B | 0.531 | 0.640 | 0.715 |
| LLaVA-Critic-7B | **0.596** | **0.722** | **0.763** |
| LLaVA-Critic-72B | **0.605** | **0.736** | **0.779** |

### 偏好学习效果

| 基础模型 | 奖励来源 | LLaVA-W | L-Wilder | WV-B | Live-B | V-DC | MMHal |
|---------|---------|---------|----------|------|--------|------|-------|
| OV-7B | 无(基线) | 90.7 | 67.8 | 54.0 | 77.1 | 3.75 | 3.19 |
| OV-7B | LLaVA-RLHF | 97.5 | 70.3 | 64.1 | 83.1 | 3.84 | 4.01 |
| OV-7B | Critic-7B | **100.3** | **71.6** | **67.3** | 84.5 | **3.87** | 3.91 |
| OV-72B | LLaVA-RLHF | 103.2 | 75.2 | 65.2 | 86.2 | 3.85 | 3.67 |
| OV-72B | Critic-72B | **104.4** | **75.9** | **70.0** | **88.5** | **3.86** | **3.77** |

### 消融实验

| 配置 | 平均Pearson-r | 说明 |
|------|-------------|------|
| LLaVA-Critic-7B (v0.5, 53k数据) | 0.712 | 较少数据和领域 |
| LLaVA-Critic-7B (113k数据) | 0.732 | 数据scaling有效 |
| LLaVA-Critic-72B | 0.754 | 模型scaling有效 |

### 关键发现

- **7B Critic接近72B水平**：LLaVA-Critic-7B (0.732) 与 72B (0.754) 的Pointwise评分差距极小，而远超Qwen2-VL-7B (0.352) 和 LLaMA3.2-V-11B (0.359)，说明评估能力可以通过少量高质量数据高效习得
- **LLaVA-OV原生评估能力极弱**：未经Critic训练的LLaVA-OV-7B平均Pearson-r仅0.364，会给出千篇一律的固定分数，无法区分回复质量差异
- **Critic奖励优于人类反馈奖励**：在偏好学习中，LLaVA-Critic-7B在5/6个Benchmark上超越LLaVA-RLHF（基于人类反馈训练的奖励模型），且仅用9.4k提示即可
- **Best-of-N推理搜索有效**：在已DPO训练的模型上，用Critic-7B做Best-of-5选择可额外获得+1.7 (LLaVA-W) 和 +3.2 (L-Wilder) 的提升
- **跨模态泛化**：仅用图像数据训练偏好对齐，在视频详细描述(Video-DC)任务上也获得提升

## 亮点与洞察

- **首个开源通用多模态Judge**：填补了开源社区在LMM评估器上的空白，GPT-4o评估一次迭代DPO约$690，LLaVA-Critic完全免费
- **评估提示池的设计哲学**：不是训练一个"万能判断标准"，而是让模型学习"理解和遵循不同的评估标准"，这使得Critic可以适应任意用户定义的评估维度
- **对称聚合消除位置偏差**：通过所有有序对打分并取 $r_i = \sum a_{ki} - \sum a_{il}$ 的设计，巧妙地消除了pairwise评估中常见的位置偏差问题
- **分数+理由的双重训练**：不仅训练模型给分数，还训练给理由，使评估过程透明且可验证

## 局限与展望

- 训练数据依赖GPT-4o生成分数和理由，存在蒸馏天花板
- Pointwise评估中不同Benchmark的分数标准不统一（1-10 vs 1-5），跨Benchmark泛化需要更统一的设计
- 113k数据量相对较小，在涉及专业领域（科学、医学）的评估上可能不够充分
- 未探索用LLaVA-Critic自身产生的评估数据来迭代训练更强的Critic（self-improving评估器）

## 相关工作与启发

- **vs Prometheus-Vision**: 首个VLM评估器但仅支持用户定义的评分标准，不是通用评估器。LLaVA-Critic覆盖7种评估场景，通用性远强
- **vs RLAIF-V (2405.17220)**: RLAIF-V用分治策略计算原子声明级奖励，LLaVA-Critic训练专门的评估器提供回复级奖励。在LLaVA-v1.5-7B的偏好学习对比中，LLaVA-Critic以9.4k提示达到与RLAIF-V 33.8k提示相当的效果，效率更高
- **vs CriticGPT**: CriticGPT专注于代码评估，LLaVA-Critic是多模态通用版本，可评估视觉聊天、详细描述、幻觉检测等多种任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个开源通用多模态评估器的定位有价值，但核心方法（用GPT-4o生成评估数据做微调）并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ In-domain/Out-of-domain评估、Pointwise/Pairwise设置、偏好学习对比、推理搜索、数据/模型scaling消融，极其全面
- 写作质量: ⭐⭐⭐⭐⭐ 两大场景（Judge/Preference）的组织结构清晰，数据构建流程描述详尽
- 价值: ⭐⭐⭐⭐⭐ 开源免费的GPT-4o替代评估器+偏好信号源，对LMM开发者有极高实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] LLaVA-FA: Learning Fourier Approximation for Compressing Large Multimodal Models](../../ICLR2026/multimodal_vlm/llava-fa_learning_fourier_approximation_for_compressing_large_multimodal_models.md)
- [\[CVPR 2026\] PhyCritic: Multimodal Critic Models for Physical AI](../../CVPR2026/multimodal_vlm/phycritic_multimodal_critic_models_for_physical_ai.md)
- [\[ICCV 2025\] LLaVA-KD: A Framework of Distilling Multimodal Large Language Models](../../ICCV2025/multimodal_vlm/llava-kd_a_framework_of_distilling_multimodal_large_language_models.md)
- [\[ACL 2025\] Can Vision-Language Models Evaluate Handwritten Math?](../../ACL2025/multimodal_vlm/can_vision-language_models_evaluate_handwritten_math.md)
- [\[CVPR 2025\] Debiasing Multimodal Large Language Models via Noise-Aware Preference Optimization](debiasing_multimodal_large_language_models_via_noise-aware_preference_optimizati.md)

</div>

<!-- RELATED:END -->
