---
title: >-
  [论文解读] Hallucination Detox: Sensitivity Dropout (SenD) for Large Language Model Training
description: >-
  [幻觉检测][hallucination] 提出Sensitivity Dropout (SenD)训练协议，通过识别并确定性丢弃训练过程中波动最大的嵌入索引（Sensitive Embedding Indices），减少LLM训练中幻觉的振荡行为，同时提出高效EigenScore近似方法(EES)实现2倍加速。
tags:
  - "幻觉检测"
  - "hallucination"
  - "dropout"
  - "training dynamics"
  - "EigenScore"
  - "sensitive embedding indices"
---

# Hallucination Detox: Sensitivity Dropout (SenD) for Large Language Model Training

**会议/期刊**: ACL 2025  
**arXiv**: [2410.15460](https://arxiv.org/abs/2410.15460)  
**代码**: [GitHub](https://github.com/EMZEDI/SEND)  
**领域**: 幻觉检测  
**关键词**: hallucination, dropout, training dynamics, EigenScore, sensitive embedding indices  

## 一句话总结

提出Sensitivity Dropout (SenD)训练协议，通过识别并确定性丢弃训练过程中波动最大的嵌入索引（Sensitive Embedding Indices），减少LLM训练中幻觉的振荡行为，同时提出高效EigenScore近似方法(EES)实现2倍加速。

## 研究背景与动机

- **问题定义**：LLM在训练过程中存在幻觉的振荡行为（oscillatory behavior），即模型在不同训练检查点时幻觉表现反复波动，难以通过训练损失收敛来判断最优停止点。
- **现有不足**：已有幻觉研究主要聚焦于推理阶段的检测和缓解（如RLHF、RAG），忽视了训练过程本身对幻觉产生的影响。
- **关键观察**：即使训练损失收敛，幻觉指标（SelfCheckGPT、HaluEval等）仍持续振荡；模型规模增大也无法有效解决振荡问题。
- **本文方案**：从训练内部动态出发，识别导致幻觉的敏感嵌入索引(SEI)，通过确定性dropout在训练中减少幻觉方差。

## 方法详解

### 整体框架

SenD训练协议包含三个核心组件：
1. **敏感嵌入索引(SEI)检测**：通过分析倒数第二层嵌入在检查点间的变化，识别波动最大的嵌入维度
2. **确定性Dropout**：在后续训练中丢弃这些高波动维度，迫使模型通过稳定维度学习
3. **高效EigenScore(EES)**：作为训练停止标准，利用Chebyshev多项式和随机迹估计加速EigenScore计算

### 关键设计

- **句子嵌入向量提取**：将倒数第二层激活矩阵 $\mathbb{R}^{n,m}$ 转换为嵌入向量 $e_k = \frac{1}{2}((\frac{1}{m}\sum_{i=1}^{m}H_{N-1}^i) + H_{N-1}^m)$
- **净变化公式**：$\Delta e_i^t = |e_i^t - e_i^{t-1}|$，衡量相邻检查点间嵌入索引的变化量
- **SEI定义**：选取最后C个检查点中变异性最高的top-K%嵌入索引，$V_i = Var(e_i)\sum_{t=T-C+1}^{T}\Delta e_i^t$
- **训练循环**：每3个检查点重新计算SEI并丢弃top-20%，直到损失和EES同时收敛

### 损失函数/停止标准

- 标准语言模型损失 + EES停止标准
- EES通过Chebyshev多项式和谱密度(DOS)近似EigenScore：$\text{EES} = \frac{1}{K}\sum_{m=0}^{M}d_m c_m$
- 时间复杂度从 $O(N^3)$ 降低到 $O(N^2)$，在大矩阵上实现约2倍加速

## 实验

### 主实验结果

| 指标 | SenD | Normal Training |
|------|:----:|:----:|
| FactScore | **0.44** | 0.39 |
| FactScore + RAG | **0.50** | 0.40 |
| HaluEval Accuracy | 0.74 | 0.74 |
| HaluEval Correctness | 0.98 | 0.98 |
| HaluEval Exact Match | 0.75 | 0.75 |

> Llama 3.1 8B在HELM数据集上的评估。

### 消融实验/下游任务影响

| 指标 | 数据集 | SenD | Normal |
|------|--------|:----:|:------:|
| HellaSwag | HELM | 0.73 | 0.74 |
| MMLU | HELM | 0.67 | 0.65 |
| Token Entropy | HELM | **0.79** | 0.95 |
| HellaSwag | CodeSearchNet | **0.69** | 0.40 |
| Token Entropy | CodeSearchNet | **0.21** | 0.33 |

> SenD不降低下游任务性能，同时降低token分布熵（提升置信度最高达17%）。

### 关键发现

- 幻觉振荡行为在70M到12B所有模型规模中持续存在，模型规模增大无法解决
- SEI dropout相比随机dropout，可显著降低EigenScore（尤其对幻觉输出效果更明显）
- SenD在Wikipedia、Medical、Legal、Coding四个领域均有效减少幻觉方差
- SenD与RAG可叠加使用：SenD+RAG的FactScore（0.50）优于单独RAG（0.40）
- 额外训练开销仅约11%（61分钟vs 55分钟/epoch）

## 亮点

- 首个从训练动态角度解决LLM幻觉的方法，填补了推理阶段缓解和训练阶段优化之间的空白
- 提出的SEI概念直觉清晰：高波动的嵌入维度对应不确定性高的知识表示
- EES作为EigenScore的高效近似，具有独立的实用价值
- SenD是正交于RAG等推理时方法的互补方案，可叠加使用

## 局限性

- 仅验证了continual training场景，未在预训练阶段测试
- 受算力限制，最大模型为Llama 3.1 8B，缺乏更大规模模型的验证
- SEI的K%阈值和检查点窗口C需要调参，不同数据集可能需要不同设置
- EES与原始EigenScore的量纲不同（$[0,\infty)$ vs $[-1,1]$），需注意解释

## 相关工作

- **幻觉检测**：EigenScore (Chen et al., 2024)、SelfCheckGPT (Manakul et al., 2023)、Semantic Entropy (Kossen et al., 2024)
- **正则化**：Random Dropout (Srivastava et al., 2014)、Adaptive Dropout (Ba & Frey, 2013)、确定性dropout (Santra et al., 2020)
- **训练动态**：Li et al. (2024) 首次观察到LLM训练中的幻觉振荡行为
- **RLHF**：Yu et al. (2024) 使用强化学习增强模型可靠性

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Copy-Paste to Mitigate Large Language Model Hallucinations](../../ICLR2026/hallucination/copy-paste_to_mitigate_large_language_model_hallucinations.md)
- [\[CVPR 2026\] CausalLens: Sensitivity-Guided Multi-Head Causal Intervention for Hallucination Mitigation in Large Vision-Language Models](../../CVPR2026/hallucination/causallens_sensitivity-guided_multi-head_causal_intervention_for_hallucination_m.md)
- [\[ACL 2025\] ReefKnot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models](reefknot_a_comprehensive_benchmark_for_relation_hallucination_evaluation_analysi.md)
- [\[CVPR 2026\] VES-RFT: Rewarding Visual Evidence Sensitivity to Mitigate Hallucinations in Large Vision-Language Models](../../CVPR2026/hallucination/ves-rft_rewarding_visual_evidence_sensitivity_to_mitigate_hallucinations_in_larg.md)
- [\[ACL 2025\] Beyond Facts: Evaluating Intent Hallucination in Large Language Models](intent_hallucination_eval.md)

</div>

<!-- RELATED:END -->
