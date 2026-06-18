---
title: >-
  [论文解读] Boosting MLLM Reasoning with Text-Debiased Hint-GRPO
description: >-
  [ICCV 2025][多模态VLM][MLLM推理] 揭示GRPO在MLLM推理中的两大问题——低数据利用率（难题上所有输出均错误导致梯度无效）和文本偏差（模型忽视图像仅依赖文本推理），提出Hint-GRPO（自适应提供推理提示）和文本偏差校准（测试时增强图像条件）两套方案，在3个基座MLLM上的11个数据集上显著提升推理能力。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "MLLM推理"
  - "GRPO强化学习"
  - "数据利用率"
  - "文本偏差"
  - "提示引导"
---

# Boosting MLLM Reasoning with Text-Debiased Hint-GRPO

**会议**: ICCV 2025  
**arXiv**: [2503.23905](https://arxiv.org/abs/2503.23905)  
**代码**: [github.com/hqhQAQ/Hint-GRPO](https://github.com/hqhQAQ/Hint-GRPO)  
**领域**: 多模态学习 / MLLM推理  
**关键词**: MLLM推理, GRPO强化学习, 数据利用率, 文本偏差, 提示引导

## 一句话总结

揭示GRPO在MLLM推理中的两大问题——低数据利用率（难题上所有输出均错误导致梯度无效）和文本偏差（模型忽视图像仅依赖文本推理），提出Hint-GRPO（自适应提供推理提示）和文本偏差校准（测试时增强图像条件）两套方案，在3个基座MLLM上的11个数据集上显著提升推理能力。

## 研究背景与动机

MLLM推理方法分为两大类：
- **PRM（过程奖励方法）**：监督中间推理步骤，但DeepSeek-R1指出PRM难以准确评估步骤质量，存在严重的reward hacking问题
- **ORM（结果奖励方法）**：仅监督最终结果（如GRPO），简单比较即可准确评估，泛化性更强

GRPO在LLM推理中表现出色，但在MLLM上存在两个关键问题：

**问题1 - 低数据利用率**：GRPO对每个问题采样G个输出，当所有输出都不正确时（$r_i = 0$ for all $i$），归一化优势 $\hat{A}_{i,t} = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})} = 0$，梯度中仅剩KL正则项。实测Qwen2-VL-7B在数学推理上有效样本比例仅40-50%。

**问题2 - 文本偏差**：随着GRPO训练进行，MLLM在去除图像条件后的准确率也在提升——说明模型学会了仅从文本推理而忽视图像。原因是许多query文本已经能充分描述问题，但当文本不足以描述时会导致错误。

## 方法详解

### 整体框架

Hint-GRPO扩展了标准GRPO：对于难题自适应提供推理提示以提高数据利用率，配合测试时的文本偏差校准来增强图像条件利用。整体框架如下：标准GRPO的无效样本 → 注入不同程度hint的多组输出 → 选择最合适难度的组训练 → 测试时通过CFG风格校准增强图像感知。

### 关键设计

1. **Hint-GRPO 提示引导训练**:

    - **数据构建**：基于LLaVA-CoT数据集，用GPT-4o将长推理文本拆分为多个结构化步骤；将选择题转为填空题避免随机猜对
    - **提示注入方式**：
        - $\mathcal{I}_{query}$（提示注入query）：将提示追加到query文本 → 训测不一致，效果差
        - $\mathcal{I}_{answer}$（提示注入answer）：保持query不变，将提示作为模型输出的开头，让模型补全剩余推理步骤 → 训测一致，效果显著更好
    - **自适应提示策略**：
        - 定义提示比例 $\alpha \in [0,1]$，从L个正确推理步骤中取前 $L \cdot \alpha$ 步作为提示
        - 将GRPO的单组输出扩展为M组（M=3），每组使用不同提示比例 $\alpha_i = \frac{i-1}{M}$
        - 从低提示到高提示依次检查，选择第一个有正确输出的组用于训练
        - 既避免低利用率，又避免过多提示阻碍推理学习
        - 相比原始GRPO仅增加20.5%训练时间

2. **文本偏差校准（测试时）**:

    - 受图像生成中Classifier-Free Guidance（CFG）启发
    - 分别计算有图像条件和无图像条件下的token logits：$\hat{\pi}_\theta(o_t | q_{img})$ 和 $\hat{\pi}_\theta(o_t)$
    - 校准公式：
    $\hat{\pi}_\theta^{calibrated}(o_t | q_{img}) = \hat{\pi}_\theta(o_t | q_{img}) + \gamma \cdot (\hat{\pi}_\theta(o_t | q_{img}) - \hat{\pi}_\theta(o_t))$
    - $\gamma=0.8$ 控制图像条件强度，使校准后的logits远离无图像版本而靠近有图像版本

3. **数据利用率度量**:

    - 定义有效样本比例 $S_{valid} = \frac{1}{B}\sum_{k=1}^B \mathbb{1}\{\text{std}(\mathbf{r}(z_k)) \neq 0\}$
    - Hint-GRPO将 $S_{valid}$ 从40-50%显著提升至更高水平

### 损失函数 / 训练策略

- 基于GRPO标准目标函数，增加KL正则项约束新模型不偏离太远
- AdamW优化器，学习率5e-5，8 GPU训练2 epochs
- DeepSpeed ZeRO-3加速，vLLM加速生成（1 GPU生成 + 7 GPU训练）
- 系统提示格式化输出为 `<think>...</think> <answer>...</answer>`

## 实验关键数据

### 主实验（几何推理 Geo170K Accuracy）

| 方法 | Geo170K | MathVista | MMStar | MathVerse | Math-Vision | 平均 |
|------|---------|-----------|--------|-----------|-------------|------|
| Qwen2-VL-7B 原始 | 30.63 | 44.50 | 40.52 | 27.92 | 10.89 | 30.40 |
| SFT | 37.53 | 41.66 | 37.07 | 14.47 | 2.86 | 25.50 |
| Mulberry (PRM) | 33.55 | 52.17 | 42.24 | 17.68 | 6.06 | 32.08 |
| Open-R1-Multimodal | 35.68 | 45.55 | 40.52 | 28.78 | 11.43 | 31.56 |
| R1-V | 38.72 | 47.26 | 41.38 | 28.12 | 12.51 | 33.19 |
| GRPO | 38.46 | 48.82 | 42.24 | 30.10 | 12.02 | 33.92 |
| Hint-GRPO | 45.62 | 52.77 | 43.97 | 31.68 | 14.38 | 37.60 |
| **Debiased Hint-GRPO** | **46.68** | **54.19** | **45.69** | **32.18** | **14.99** | **38.55** |

### 消融实验

| 配置/超参数 | Geo170K Acc |
|------------|-------------|
| GRPO + $\mathcal{D}_{original}$（含选择题） | 35.81 |
| GRPO + $\mathcal{D}_{new}$（填空题） | 38.46 |
| Hint-GRPO + $\mathcal{I}_{query}$ | 41.64 |
| **Hint-GRPO + $\mathcal{I}_{answer}$** | **45.62** |
| 固定α=0.25 | 较好但次优 |
| 固定α=0.75 | 反而有害 |
| 随机α | 好于固定 |
| **自适应α（M=3）** | **最优** |
| $\gamma=0$（无校准） | 45.62 |
| $\gamma=0.8$（校准） | **46.68** |
| $\gamma=1.6$（过度校准） | 44.69 |

### 通用多模态推理（Llama-3.2-11B-Vision）

| 方法 | MMStar | MMBench | MMVet | MathVista | AI2D | Hallusion | 平均 |
|------|--------|---------|-------|-----------|------|-----------|------|
| 原始 | 49.8 | 65.8 | 57.6 | 48.6 | 77.3 | 40.3 | 56.6 |
| LLaVA-o1 | 57.6 | 75.0 | 60.3 | 54.8 | 85.7 | 47.8 | 63.5 |
| **Ours** | **60.7** | **75.8** | **64.2** | **56.8** | **86.6** | **50.7** | **65.8** |

### 关键发现

- SFT在几何推理上反而降低了OOD性能，说明SFT只是机械记忆而非学会推理模式
- 将选择题转为填空题后GRPO准确率从35.81提升到38.46，消除了随机猜对的捷径
- 提示注入到answer比注入到query效果大幅领先（45.62 vs 41.64），原因是后者训测不一致
- $\gamma=0.8$ 是最优校准强度，过高($\gamma=1.6$)反而过度矫正导致性能下降
- 自适应M从1增到3持续提升，M=4时饱和，考虑效率选择3

## 亮点与洞察

- 低数据利用率问题的形式化分析非常清楚：通过梯度分解证明全零优势使样本完全无效
- Hint-GRPO的设计哲学巧妙——将"提供提示"理解为"部分推理作为前缀"，保持了训测一致性
- 文本偏差的发现深刻：GRPO训练中无图像准确率提升是一个值得警惕的信号
- CFG思想从图像生成迁移到MLLM推理的文本去偏差，跨领域借鉴

## 局限与展望

- Hint-GRPO依赖高质量推理步骤数据（LLaVA-CoT + GPT-4o拆分），获取成本不低
- 通用多模态推理的提升不如几何推理显著，需要更鲁棒的准确率估计方法（如IoU等宽松匹配）
- 文本偏差校准需要双倍前向传播，增加推理延迟
- M组采样增加了生成开销

## 相关工作与启发

- Hint的自适应策略可借鉴课程学习（curriculum learning）的思想进一步优化
- 文本偏差问题可能在其他多模态任务（如视觉问答）中也普遍存在
- GRPO的低数据利用率问题可能通过更好的采样策略（如重要性采样）来缓解

## 评分
- 新颖性: ⭐⭐⭐⭐ 两个问题的发现和解决方案都有创意，但核心仍是GRPO的工程改进
- 实验充分度: ⭐⭐⭐⭐⭐ 3个基座模型、11个数据集、详细消融，非常充分
- 写作质量: ⭐⭐⭐⭐ 问题分析深入透彻，梯度推导清晰
- 价值: ⭐⭐⭐⭐⭐ 直接提升MLLM推理GRPO训练效果，方法简洁实用，易于复现和推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] G2D: Boosting Multimodal Learning with Gradient-Guided Distillation](g2d_boosting_multimodal_learning_with_gradient-guided_distillation.md)
- [\[ICCV 2025\] Information Density Principle for MLLM Benchmarks](information_density_principle_for_mllm_benchmarks.md)
- [\[ICCV 2025\] BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models](basic_boosting_visual_alignment_with_intrinsic_refined_embeddings_in_multimodal_.md)
- [\[AAAI 2026\] AbductiveMLLM: Boosting Visual Abductive Reasoning Within MLLMs](../../AAAI2026/multimodal_vlm/abductivemllm_boosting_visual_abductive_reasoning_within_mll.md)
- [\[CVPR 2026\] Boosting Reasoning in Large Multimodal Models via Activation Replay](../../CVPR2026/multimodal_vlm/boosting_reasoning_in_large_multimodal_models_via_activation_replay.md)

</div>

<!-- RELATED:END -->
