---
title: >-
  [论文解读] Social Debiasing for Fair Multi-modal LLMs
description: >-
  [ICCV 2025][因果推理][社会偏见] 本文构建了包含 18 种社会概念的大规模反事实数据集 CMSC，并提出反刻板印象去偏策略 ASD（含偏差感知数据重采样 + Social Fairness Loss），在四种 MLLM 架构上有效降低了社会偏见，同时几乎不损害通用多模态能力。 领域现状：MLLM（如 LLaVA…
tags:
  - "ICCV 2025"
  - "因果推理"
  - "社会偏见"
  - "多模态大模型"
  - "反刻板印象"
  - "公平性"
  - "数据去偏"
---

# Social Debiasing for Fair Multi-modal LLMs

**会议**: ICCV 2025  
**arXiv**: [2408.06569](https://arxiv.org/abs/2408.06569)  
**代码**: 有（项目页面）  
**领域**: 因果推理 / 公平性  
**关键词**: 社会偏见、多模态大模型、反刻板印象、公平性、数据去偏

## 一句话总结

本文构建了包含 18 种社会概念的大规模反事实数据集 CMSC，并提出反刻板印象去偏策略 ASD（含偏差感知数据重采样 + Social Fairness Loss），在四种 MLLM 架构上有效降低了社会偏见，同时几乎不损害通用多模态能力。

## 研究背景与动机

**领域现状**：MLLM（如 LLaVA、Qwen-VL、Bunny）在通用视觉语言理解上取得了显著进展，广泛应用于各种下游任务。然而，这些模型从训练数据中继承了深层的社会偏见——例如模型可能强烈倾向于将"护士"与"女性"关联、将"科学家"与"白人男性"关联。

**现有痛点**：（1）现有去偏数据集要么规模小（VisoGender 仅 690 张），要么只覆盖单一社会概念——occupation（SocialCounterfactuals 有 171K 图但只有职业概念），无法全面减少偏见；（2）从方法论角度，直接在平衡数据集上微调（naive FT）效果有限，因为它对所有样本"一视同仁"，无法有针对性地纠正那些被模型严重忽视的群体。

**核心矛盾**：用"中性水"（平衡数据集 + 等权训练）来"中和酸性水"（社会偏见）是不够的——应该用"碱性水"（反刻板印象数据 + 差异化训练权重）。

**本文目标**：（1）构建覆盖多社会概念的大规模去偏数据集；（2）设计一种利用偏见反面来纠正偏见的训练策略。

**切入角度**：社会偏见可以用 Skew 指标量化为每个"社会属性-社会概念"对的偏差值。如果一个组合（如 Female-Nurse）的 Skew>0（被过度预测），就应该在训练中降低它的权重；反之（如 Male-Nurse）Skew<0，就应该增大权重。

**核心 idea**：构建 18 种社会概念 × 3 种社会属性类型的 CMSC 数据集，配合 ASD 策略——在数据采样层面增加被忽视样本的出现频率，在损失函数层面用 $e^{-\text{Skew}}$ 因子重新缩放每个样本的贡献。

## 方法详解

### 整体框架

整体流程分两大部分：（1）CMSC 数据集构建——定义 18 种社会概念（人格、责任、教育三类）、3 种社会属性（性别、种族、年龄），用 SDXL + Prompt-to-Prompt 控制生成 60K 高质量反事实图像；（2）ASD 训练策略——在微调 MLLM 时，先用 Skew 评估模型当前偏见，然后通过数据重采样和损失缩放动态调整训练过程。

### 关键设计

1. **CMSC 数据集：多概念反事实图像生成**:

    - 功能：解决现有去偏数据集概念覆盖不足的问题
    - 核心思路：定义 18 种社会概念，分为人格（compassionate/belligerent/authority/pleasant/unpleasant）、责任（tool/weapon/career/family/chef/earning money）和教育（middle school/high school/university/good student/bad student/science/arts）三组。为每个概念设计精心的 prompt 模板，使用交叉生成策略：先固定种族生成 4 组不同性别×年龄的基础图像，再用 Prompt-to-Prompt 控制生成不同种族的变体确保视觉一致性。经过人工过滤后得到 60K 高质量图像
    - 设计动机：单一职业概念无法覆盖现实中丰富多样的刻板印象。CMSC 的 18 概念覆盖了从"暴力倾向"到"教育水平"等多个社会刻板印象维度，使模型能学到更全面的去偏知识

2. **数据重采样（Dataset Resampling）**:

    - 功能：在训练数据层面增加被模型忽视群体的曝光率
    - 核心思路：对每个样本 $\mathcal{P}_i$，计算其 $\text{Skew}(\mathcal{P}_i)$（所有社会属性中绝对值最大的 Skew）。如果 $\text{Skew} > 0$（被过度关注的组合），以概率 $\text{Skew}/(\text{Skew}+\tau_1)$ 丢弃该样本；如果 $\text{Skew} \leq 0$（被忽视的组合），直接保留并通过累积 Skew 机制触发过采样——当 $\text{AcmSkew}$ 超过阈值 $\tau_2$ 时额外复制样本。这样每个 epoch 的训练数据都动态调整
    - 设计动机：平衡数据集给所有样本相同的出现概率，但模型对不同组合的偏见程度不同。直觉上，"男护士"出现得越多越好（因为模型倾向忽略它），"女护士"可以少出现一些

3. **Social Fairness Loss（SFLoss）**:

    - 功能：在损失函数层面差异化对待不同偏见程度的样本
    - 核心思路：将标准自回归损失 $\mathcal{L}$ 乘以权重因子 $e^{-\text{Skew}(\mathcal{P}_i)}$，得到 $\mathcal{E}_{fair} = \frac{1}{N}\sum_{i=1}^{N} e^{-\text{Skew}(\mathcal{P}_i)} \mathcal{L}(\mathbf{y}_i, \mathbf{x}_i^{\text{ins}}, \mathbf{x}_i^{\text{img}}; \theta)$。当 $\text{Skew} > 0$ 时权重 < 1（降权被过度预测的组合）；当 $\text{Skew} < 0$ 时权重 > 1（增权被忽视的组合）
    - 设计动机：灵感来自类不平衡研究中的重加权策略（如 focal loss），但这里不是根据类别频率而是根据偏见程度来调整权重。$e^{-\text{Skew}}$ 的指数形式使调整幅度随偏见程度非线性增加

### 损失函数 / 训练策略

基础损失为 MLLM 标准自回归交叉熵 $\mathcal{L} = -\sum_t \log P(y_t | y_{<t}, x^{\text{ins}}, x^{\text{img}}; \theta)$。ASD 在此基础上：（1）每个 epoch 前执行一次数据重采样；（2）训练时使用 SFLoss 缩放。超参数 $\tau_1 = \tau_2 = 1.0$，学习率因模型而异（LLaVA 用 5e-7，Qwen-VL 用 1e-6，Bunny 用 1e-7）。

## 实验关键数据

### 主实验

在 SocialCounterfactuals 上微调，跨数据集评估（3 个偏见数据集 + 3 个通用基准）：

| 模型 | SCounter MinS@C / MaxS@C | FairFace MinS@C / MaxS@C | CMSC MinS@C / MaxS@C | VQAv2 | MMBench |
|------|------|------|------|------|------|
| LLaVA-7B | -2.06 / 0.40 | -2.88 / 0.65 | -1.62 / 1.48 | 78.50 | 58.21 |
| LLaVA-7B+FT | -0.27 / 0.40 | -1.78 / 0.54 | -1.60 / 0.81 | 78.12 | 58.12 |
| **LLaVA-7B+ASD** | **-0.17 / 0.37** | **-0.86 / 0.49** | **-1.50 / 0.73** | 78.18 | 58.36 |
| Qwen-VL-7B | -0.61 / 0.60 | -1.63 / 0.85 | -1.51 / 1.10 | 79.37 | 61.39 |
| **Qwen-VL-7B+ASD** | **-0.26 / 0.43** | **-0.92 / 0.42** | **-0.71 / 0.85** | 79.37 | 60.88 |
| Bunny-8B+ASD | -0.30 / 0.55 | -1.07 / 0.46 | -1.03 / 0.95 | 82.41 | 65.20 |

### 消融实验

| 配置 | SCounter MinS@C | FairFace MinS@C | CMSC MinS@C |
|------|------|------|------|
| LLaVA-7B (原始) | -2.06 | -2.88 | -1.62 |
| +FT (直接微调) | -0.27 | -1.78 | -1.60 |
| +Resample (仅重采样) | -0.19 | -0.98 | -1.55 |
| +SFLoss (仅损失缩放) | -0.18 | -0.95 | -1.53 |
| **+ASD (完整方法)** | **-0.17** | **-0.86** | **-1.50** |

### 关键发现

- **ASD 全面优于 naive FT**：在所有模型和所有数据集上，ASD 的 MinSkew@C 和 MaxSkew@C 均更接近 0（更公平）
- **CMSC 数据集本身的贡献**：仅用 CMSC 做 FT 就比用 SocialCounterfactuals 做 FT 效果更好（Figure 3），多概念覆盖确实有帮助
- **模型规模与偏见不正相关**：LLaVA-13B 的偏见比 LLaVA-7B 更严重，可能是更大模型更好地学到了训练数据中的偏见
- **通用能力几乎不受影响**：ASD 在 VQAv2、TextVQA、MMBench 上的性能变化 < 0.5%
- **重采样和 SFLoss 各有贡献且可叠加**：两者单独使用都优于 FT，组合使用效果最佳
- **概念子集微调有跨概念泛化**：在 personality 子集上微调也能部分降低 responsibility 子集的偏见

## 亮点与洞察

- **"碱性水中和酸性水"的类比非常直观**：不是简单地提供平衡数据（中性水），而是提供反刻板印象数据（碱性水）。这个直觉通过 Skew 加权机制得到了优雅的数学实现
- **Prompt-to-Prompt 保证视觉一致性**：在生成不同种族的反事实图像时，只改变种族相关的 token 的 cross-attention map，其余保持不变。这确保了不同种族版本的图像只在种族特征上不同，避免引入其他混淆变量
- **实用且通用**：ASD 方法与 MLLM 架构无关，可以直接应用于任何基于自回归训练的 MLLM

## 局限与展望

- CMSC 使用 SDXL 生成图像，合成图像与真实图像之间有分布差异（但 FID 比较显示 CMSC 更接近真实数据）
- 18 种社会概念虽然比单一职业概念丰富很多，但仍无法覆盖所有类型的社会刻板印象
- 去偏效果依赖于 Skew 的准确计算，需要在每个 epoch 前重新评估模型偏见
- 学习率需要针对不同模型仔细调节，过高的学习率在提高公平性的同时会损害通用能力
- 未来可以探索：实时自适应的 Skew 更新、更多样的社会属性（如残疾、宗教）、与 RLHF 安全对齐的结合

## 相关工作与启发

- **vs SocialCounterfactuals (CVPR 2024)**：同为反事实数据去偏，但只覆盖 occupation 一个概念且未提出差异化训练策略。CMSC 覆盖 18 概念，ASD 提供了更有效的训练方法
- **vs POPE (EMNLP 2023)**：POPE 是训练无关的去偏方法，通过修改推理过程减少幻觉/偏见，但在实验中去偏效果不如 FT/ASD
- **vs 对比学习去偏（FairerCLIP 等）**：这些方法针对 CLIP-style 模型设计，不适用于自回归 MLLM。ASD 是首个专门针对自回归 MLLM 的去偏策略
- **vs focal loss**：SFLoss 的设计灵感类似 focal loss，但不是按类别频率而是按偏见程度加权，更适合公平性场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 反刻板印象加权的思路清晰且有效，但核心技术（重采样+重加权）并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 四种 MLLM 架构、三种偏见数据集 + 三种通用基准、完整消融
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，方法描述完整，实验分析详尽
- 价值: ⭐⭐⭐⭐ 对 MLLM 公平性研究有实际推动，数据集和方法都可重用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] MPF: Aligning and Debiasing Language Models post Deployment via Multi Perspective Fusion](../../ICML2025/causal_inference/mpf_aligning_and_debiasing_language_models_post_deployment_via_multi_perspective.md)
- [\[NeurIPS 2025\] A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning](../../NeurIPS2025/causal_inference/a_principle_of_targeted_intervention_for_multi-agent_reinforcement_learning.md)
- [\[ICLR 2026\] On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study](../../ICLR2026/causal_inference/on_the_eligibility_of_llms_for_counterfactual_reasoning_a_decompositional_study.md)
- [\[CVPR 2025\] Joint Scheduling of Causal Prompts and Tasks for Multi-Task Learning](../../CVPR2025/causal_inference/joint_scheduling_of_causal_prompts_and_tasks_for_multi-task_learning.md)
- [\[ICLR 2026\] SelfReflect: Can LLMs Communicate Their Internal Answer Distribution?](../../ICLR2026/causal_inference/selfreflect_can_llms_communicate_their_internal_answer_distribution.md)

</div>

<!-- RELATED:END -->
