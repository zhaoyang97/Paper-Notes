---
title: >-
  [论文解读] ACT as Human: Multimodal Large Language Model Data Annotation with Critical Thinking
description: >-
  [NeurIPS 2025][多模态VLM][data annotation] 提出ACT（Annotation with Critical Thinking）数据流水线，MLLM批量标注全部数据后由另一个MLLM作为批评者估计每条标注的错误概率，仅将高可疑样本交给人类审核，配合理论推导的ACT损失函数，在6个跨模态数据集上节省70-90%人工成本且下游性能差距<2%。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "data annotation"
  - "critical thinking"
  - "MLLM"
  - "error estimation"
  - "human-in-the-loop"
---

# ACT as Human: Multimodal Large Language Model Data Annotation with Critical Thinking

**会议**: NeurIPS 2025  
**arXiv**: [2511.09833](https://arxiv.org/abs/2511.09833)  
**代码**: 无  
**领域**: 数据标注 / MLLM应用  
**关键词**: data annotation, critical thinking, MLLM, error estimation, human-in-the-loop

## 一句话总结
提出ACT（Annotation with Critical Thinking）数据流水线，MLLM批量标注全部数据后由另一个MLLM作为批评者估计每条标注的错误概率，仅将高可疑样本交给人类审核，配合理论推导的ACT损失函数，在6个跨模态数据集上节省70-90%人工成本且下游性能差距<2%。

## 研究背景与动机
**领域现状**：监督学习依赖高质量标注数据，但人工标注昂贵且难以规模化。LLM/MLLM自动标注虽然廉价，但标注质量距人工仍有明显差距。

**现有痛点**：(1) 纯MLLM标注的准确率比人工标注低5-20%，直接用于下游训练性能明显下降；(2) 现有方法如CDI需要额外训练XGBoost检测器且泛化性差；(3) 部分方法限于白盒模型，无法利用GPT-4o等强大黑盒模型；(4) 现有active M-estimation使用的归一化采样规则在低预算下效果崩塌。

**核心矛盾**：如何在有限人工预算下，最大化利用MLLM的标注能力同时保证接近人工标注的数据质量？

**切入角度**：让MLLM同时承担标注者和批评者两个角色——先标注再自我/交叉批评，将人工精力精准分配到最可疑的样本上。

## 方法详解

### 整体框架
ACT是一个三阶段training-free流水线：(1) **标注阶段**：MLLM $f^{(m)}$ 对所有$N$条数据生成标签 $\hat{y}_i^{(m)}$；(2) **错误估计阶段**：另一个MLLM $g$ 作为批评者估计每条标注的错误概率 $\hat{\epsilon}_i = g(\mathbf{x}_i, \hat{y}_i^{(m)})$；(3) **校正阶段**：基于错误概率的预算感知采样 $\delta_i(B) \sim \mathbb{B}(\pi_B(\hat{\epsilon}_i))$ 选出样本交给人类审核，约束$\sum \delta_i(B) \leq B$。下游训练使用专门设计的ACT损失函数。

### 关键设计
1. **MLLM批评策略体系（Criticizer Strategies）**:
    - 功能：设计黑盒/白盒两类共7种批评策略，让MLLM估计标注的错误概率
    - 核心思路：黑盒策略包括Naïve直接估计、CoT推理后估计、多选题分级(MC)、Devil's Advocate（先审视标注者CoT再评判）；白盒策略利用logit概率 $\hat{\epsilon} = \mathbb{P}(\text{"yes"}) / (\mathbb{P}(\text{"yes"}) + \mathbb{P}(\text{"no"}))$ 或CoT困惑度(PPL)间接度量错误。实验发现CoT策略在批评中ABS提升最高达22.46%，交叉批评（用不同模型标注和批评）通常优于自我批评
    - 设计动机：不同任务和模型组合适合不同策略，系统探索为实际部署提供选择依据；训练无关设计使流水线可直接使用任何MLLM

2. **预算感知采样规则（Budget-Aware Sampling）**:
    - 功能：在有限人工预算$B$下决定哪些样本交给人类审核
    - 核心思路：提出三种采样规则——归一化 $\pi_B(\hat{\epsilon}_i) = B \cdot \hat{\epsilon}_i / \sum \hat{\epsilon}_i$、指数加权 $\pi_B(\hat{\epsilon}_i) = 1/(1 + e^{-\beta(\hat{\epsilon}_i - \alpha)})$、阈值化 $\pi_B(\hat{\epsilon}_i) = \mathbf{1}(\hat{\epsilon}_i \geq \tau)$。通过Theorem 5.2证明：ACT损失与真实损失的参数差距上界取决于$q$（被选样本转换后的错误概率下界），指数加权和阈值化将$q$推向1，而归一化在低预算时$q \to 0$导致崩塌
    - 设计动机：归一化采样（prior work使用）在人工预算受限时损失函数高度不稳定——Cars数据集上与全监督差距76.34%，而指数加权/阈值化仅1.69%

3. **ACT损失函数（Modified Loss for Downstream Training）**:
    - 功能：设计理论有保证的损失函数，使ACT数据训练的模型性能逼近全人工标注数据训练的模型
    - 核心思路：$\mathcal{L}_\theta^{(ACT)} = \frac{1}{N}\sum_{i=1}^{N}\left(\ell_{\theta,i}^{(m)} + (\ell_{\theta,i} - \ell_{\theta,i}^{(m)}) \frac{\delta_i(B)}{\pi_B(\hat{\epsilon}_i)}\right)$，其中$\ell_{\theta,i}^{(m)}$为机器标注损失，$\ell_{\theta,i}$为真实标签损失（用人工标注估计）。Proposition 5.1证明ACT损失是真实损失的无偏估计，方差在两种情况下最小化：完美标注器或精准批评者
    - 设计动机：直接混合人工+机器标注数据会引入标签噪声；仅用人工标注数据则浪费大量已标注样本。ACT损失通过重要性加权实现无偏估计，指数加权/阈值化确保权重不会爆炸

### 损失函数 / 训练策略
ACT损失基于active M-estimation改进。核心是用采样概率$\pi_B(\hat{\epsilon}_i)$做重要性加权校正——被选中审核的样本用人工标签计算真实损失$\ell_{\theta,i}$，未被选中的用机器标注损失$\ell_{\theta,i}^{(m)}$。推荐使用阈值化规则（仅需设阈值$\tau$，比指数加权的双超参$\alpha, \beta$更简单）。下游任务使用标准交叉熵损失加power-tuning超参。

## 实验关键数据

### 主实验：下游任务测试精度(%)

| 训练数据-损失 | CIFAR-10 | Fashion | Cars | Emotion | Irony | VQA-RAD |
|---|---|---|---|---|---|---|
| 纯人工标注-CE | 88.66±0.97 | 93.01±0.63 | 87.88±0.36 | 81.82±0.57 | 70.18±3.23 | 67.81±1.47 |
| 纯机器标注-CE | 81.55±1.93 | 82.86±0.84 | 83.68±0.17 | 78.96±2.40 | 60.71±5.43 | 61.03±2.05 |
| ACT-归一化损失 | 64.70±5.46 | 69.27±7.25 | 11.54±0.96 | 79.87±0.88 | 65.66±2.00 | 62.55±3.01 |
| **ACT-指数加权损失** | **87.73±0.36** | **89.73±0.35** | **86.19±0.14** | **81.44±0.51** | **68.49±3.20** | **67.73±1.33** |
| ACT-阈值化损失 | 87.95±0.35 | 89.16±0.89 | 86.00±0.26 | 81.41±0.64 | 68.21±1.94 | 67.02±1.32 |
| 人机性能差距 | 0.71% | 3.28% | 1.69% | 0.38% | 1.69% | 0.08% |
| 人工预算占比 | 11.52% | 21.81% | 9.56% | 17.98% | 33.79% | 30.15% |

### 消融实验：批评策略ABS(%)对比（GPT-4o标注+CoT）

| 批评者模型 | Naïve | CoT | MC | Devil |
|---|---|---|---|---|
| GPT-4o（自批评） | 41.2 | 53.8 | 48.6 | 50.1 |
| Gemini-1.5-Pro | 45.3 | **56.2** | 51.4 | 52.7 |
| Claude 3.5 Sonnet | 43.7 | 54.9 | 52.1 | 55.3 |
| InternVL 2.5 | 38.5 | 44.1 | 40.3 | 42.8 |

### 关键发现
- 7条核心洞察：GPT-4o是最佳通用标注器；CoT对批评比标注更有帮助（ABS提升22.46%）；交叉批评优于自批评；黑盒模型做批评者更强；标注能力与批评能力正相关
- 归一化采样在Cars上彻底崩塌（11.54%），指数加权/阈值化稳健（86%+）
- 白盒策略（logit/PPL）在2/6数据集上优于黑盒，但不一致

## 亮点与洞察
- "标注-批评-校正"三阶段流水线设计优雅且完全training-free，可即插即用任何MLLM
- 7条系统性洞察为实际部署提供了actionable的最佳实践指南
- ACT损失函数具有理论保证（无偏估计+方差控制），且指数加权/阈值化显著优于prior work的归一化规则
- 跨NLP/CV/VQA三个领域、6个数据集、6种MLLM的系统性探索，实验设计极为充分
- "标注能力与批评能力正相关"的发现简化了模型选择——用top-1做标注器、top-2做批评者

## 局限与展望
- 仅在分类任务上验证，文本摘要、开放式QA等生成任务未覆盖
- 批评者准确率受MLLM能力上限约束，5-15%假阳性率限制极限效果
- 预算设定基于标注器准确率（"理想预算"），实际预算分配策略未深入讨论
- 中文、小语种等非英语场景效果未验证

## 相关工作与启发
- **vs CDI**：CDI需训练XGBoost检测器且用归一化采样（低预算崩塌），ACT完全training-free且阈值化采样稳健
- **vs LLM-as-a-Judge**：ACT的批评者设计和LLM自评估密切相关，交叉批评优于自批评呼应了self-evaluation bias的文献
- **vs 主动学习**：传统主动学习需在标注循环中重训模型，ACT流水线无需任何训练
- **启发**：预算感知采样范式可推广到任何人机协作场景；"标注者能力≈批评者能力"的正相关性简化了pipeline配置

## 评分
- 新颖性: ⭐⭐⭐⭐ 批评者+预算采样+ACT损失的组合设计实用且新颖，但核心思想（LLM互评）并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 6数据集、6种MLLM、7种批评策略、3种采样规则、完整消融，极为系统
- 写作质量: ⭐⭐⭐⭐ 7条洞察总结清晰，理论分析与实验结合紧密
- 价值: ⭐⭐⭐⭐⭐ 对降低AI数据标注成本有直接实用价值，指导性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Face-Human-Bench: A Comprehensive Benchmark of Face and Human Understanding for Multi-modal Assistants](face-human-bench_a_comprehensive_benchmark_of_face_and_human_understanding_for_m.md)
- [\[NeurIPS 2025\] The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models](the_illusion_of_progress_a_critical_look_at_testtime_adaptat.md)
- [\[NeurIPS 2025\] See&Trek: Training-Free Spatial Prompting for Multimodal Large Language Model](seetrek_training-free_spatial_prompting_for_multimodal_large_language_model.md)
- [\[NeurIPS 2025\] Nautilus: A Large Multimodal Model for Underwater Scene Understanding](nautilus_a_large_multimodal_model_for_underwater_scene_understanding.md)
- [\[NeurIPS 2025\] NaViL: Rethinking Scaling Properties of Native Multimodal Large Language Models under Data Constraints](navil_rethinking_scaling_properties_of_native_multimodal_large_language_models_u.md)

</div>

<!-- RELATED:END -->
