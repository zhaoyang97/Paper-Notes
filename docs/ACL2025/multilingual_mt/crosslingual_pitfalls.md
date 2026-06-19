---
title: >-
  [论文解读] Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models
description: >-
  [ACL 2025][多语言/翻译][跨语言弱点] 提出一种基于 beam search 和 LLM 模拟的自动化方法，高效生成双语问题对以暴露多语言 LLM 在目标语言上的跨语言性能缺陷，构建了覆盖 16 种语言的 6000+ 样本数据集，揭示即使 GPT-4o 也有超 30% 的跨语言准确率下降。
tags:
  - "ACL 2025"
  - "多语言/翻译"
  - "跨语言弱点"
  - "多语言LLM"
  - "beam search"
  - "双语问题对"
  - "语言亲缘性"
---

# Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models

**会议**: ACL 2025  
**arXiv**: [2505.18673](https://arxiv.org/abs/2505.18673)  
**代码**: [GitHub](https://github.com/xzx34/Cross-Lingual-Pitfalls)  
**领域**: Multilingual / LLM Evaluation  
**关键词**: 跨语言弱点, 多语言LLM, beam search, 双语问题对, 语言亲缘性  

## 一句话总结

提出一种基于 beam search 和 LLM 模拟的自动化方法，高效生成双语问题对以暴露多语言 LLM 在目标语言上的跨语言性能缺陷，构建了覆盖 16 种语言的 6000+ 样本数据集，揭示即使 GPT-4o 也有超 30% 的跨语言准确率下降。

## 研究背景与动机

- **问题定义**: 跨语言弱点 (Cross-Lingual Weakness) 定义为：对于同一语义的问题，模型在英语上回答正确，但在至少一种目标语言上回答错误。这反映了 LLM 的跨语言能力不一致性。
- **现有方法局限**: 现有多语言评估基准大多是静态的翻译数据集，无法精准定位模型在某种语言上的薄弱环节。直接翻译英语题目往往找不到性能差异（因为模型对简单问题的多语言表现差别不大）。
- **核心动机**: 如果能系统性地"搜索"出那些模型英语答对但其他语言答错的问题，就能精确诊断模型的跨语言缺陷，为针对性改进提供依据。
- **核心挑战**: 如何生成既保持语义等价、又能最大化英语/目标语言准确率差距的双语问题对？直接穷举搜索成本过高。

## 方法详解

### 整体框架

整个流程分为四步：(1) 从高质量英语数据集采样问题并翻译为目标语言，形成双语对；(2) 对英语问题进行迭代扰动 (perturbation)，增加认知复杂度；(3) 用多个 LLM 模拟评估，计算 simulation score 衡量扰动效果；(4) 用 beam search 策略迭代优化，筛选出最能暴露跨语言弱点的问题对。

### 关键设计

1. **扰动函数 (Perturbation)**: 给定英语问题 $q^E$ 和一个错误选项 $\alpha^E$，使用代理 LLM 生成语义无关但上下文合理的扰动 $\delta q^E = \varphi(q^E, \alpha^E)$，拼接到原问题中增加认知负担。同时用翻译模块对扰动进行等价翻译，保证双语语义一致性。
2. **LLM 模拟评分 (Simulation Score)**: 用 K 个 LLM 分别回答英语和目标语言版本的扰动问题，计算准确率差异分数 $V(q^{E'}, q^{T'}) = (\bar{\beta}^{E'})^\gamma - \bar{\beta}^{T'}$。$\gamma > 1$ 的指数放大确保只选择英语准确率高、目标语言准确率低的样本。
3. **Beam Search 优化策略**: 包含三个机制——(a) 包含阈值 (Inclusion Threshold)：分数超过 $\theta_{inc}$ 的样本直接加入候选列表；(b) 早停机制：当有样本分数超过 $\theta_{pot}$ 时扩展搜索深度，否则限制在初始深度；(c) 冗余控制：对同一原始问题衍生的候选数量设置上限 $r$，确保多样性。

### 损失函数 / 优化目标

$$\min_{\delta q^E} \mathbb{E}[\mathbb{I}(\mathcal{F}(q^{T'}) = a_\star^T)] \quad \text{s.t.} \quad \mathbb{E}[\mathbb{I}(\mathcal{F}(q^{E'}) = a_\star^E)] \geq 1-\epsilon, \quad \mathbb{S}(q^E, q^{E'}) \geq \theta$$

即最小化目标语言准确率，同时保持英语准确率不低于 $1-\epsilon$ 且语义相似度不低于阈值 $\theta$。

## 实验

### 主实验：跨语言弱点识别

对 10 个 LLM 在 6600 个双语对 (16 种语言) 上的评估：

| 模型 | 英语准确率 | 中文准确率 | 准确率下降 |
|------|-----------|-----------|----------|
| Gemma-2-9B | ~100% | ~35% | >60% |
| LLaMA-3.1-8B | ~100% | ~30% | >70% |
| Qwen2.5-7B | ~100% | ~45% | >55% |
| GPT-4o-mini | ~100% | ~55% | >45% |
| GPT-4o | ~100% | ~70% | ~30% |
| Claude-3.5-sonnet | ~100% | ~65% | >30% |

即使最强的 GPT-4o 在中文上也有约 30% 的准确率下降；大多数模型在目标语言上平均准确率下降超 50%。

### 消融实验：搜索策略对比

| 方法 | 中文转换率 | 日语转换率 | 法语转换率 | 德语转换率 |
|------|----------|----------|----------|----------|
| NP (无扰动) | 0.000 | 0.000 | 0.000 | 0.000 |
| DP (直接扰动) | 0.036 | 0.071 | 0.018 | 0.027 |
| **Beam Search (本文)** | **0.431** | **0.594** | **0.132** | **0.323** |

Beam search 的转换率远超无扰动和直接扰动基线，验证了搜索策略的有效性。

### 关键发现

- **语言亲缘性影响共享弱点**: 亚洲语系 (中日韩) 之间共享相似的跨语言弱点；欧洲语系 (法德西) 之间同样如此。跨语系的弱点分享较少。
- **相似语言的微调迁移更强**: 对法语数据微调后，德语/意大利语的提升远大于中文/日语。相反对中文微调后，日韩提升更大。
- **生成成本极低**: 对大多数语言，找到一个暴露弱点的双语对平均成本低于 $0.05。但与英语结构越近的语言 (法语、西班牙语) 成本越高。
- **Relative Affinity Score (RAS)** 指标清晰刻画了语言间的亲缘关系与共享弱点模式。

## 亮点

- 首次提出系统性的"搜索+扰动"方法来自动发现多语言 LLM 的跨语言弱点，远超简单翻译测评。
- 覆盖 16 种语言的大规模评估，揭示了 GPT-4o、Claude-3.5 等 SOTA 模型普遍存在的跨语言性能缺陷。
- 发现了语言亲缘性与跨语言弱点间的定量关联，并提出 RAS 指标度量语言相似性。
- 方法成本极低（每条扰动问题约 $0.05），具有高度实用性。

## 局限性

- 当前仅使用选择题 (MCQ) 格式评估，未涵盖生成式问答、翻译等更复杂的任务形式。
- 扰动策略依赖代理 LLM 的质量（使用 GPT-4o-mini），扰动的多样性和质量可能受限。
- 仅测试了有限的 LLM 集合作为模拟器，更换模拟器后弱点分布可能变化。
- 部分跨语言弱点可能源于翻译质量（Google Translate）而非模型本身的缺陷。

## 相关工作

- **多语言 LLM 评估**: MEGA (Ahuja et al., 2023)、XTREME (Hu et al., 2020) 等提供多语言基准，但属于静态评估，无法针对性探测弱点。
- **对抗性评估**: AdvGLUE (Wang et al., 2021) 等针对英语的对抗鲁棒性评测；本文将对抗搜索扩展到跨语言场景。
- **跨语言迁移学习**: mBERT (Pires et al., 2019)、XLM-R (Conneau et al., 2020) 等研究语言间的零样本迁移；本文从评估角度反向分析迁移不足之处。

## 评分

| 维度 | 分数 (1-10) |
|------|-----------|
| 创新性 | 8 |
| 实验充分性 | 9 |
| 论文清晰度 | 8 |
| 实用性 | 8 |
| **总分** | **8.3** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Bridging the Language Gaps in Large Language Models with Inference-Time Cross-Lingual Intervention](bridging_the_language_gaps_in_large_language_models_with_inference-time_cross-li.md)
- [\[ACL 2025\] Cross-Lingual Optimization for Language Transfer in Large Language Models](cross-lingual_optimization_for_language_transfer_in_large_language_models.md)
- [\[ACL 2025\] Lost in Multilinguality: Dissecting Cross-lingual Factual Inconsistency in Transformer Language Models](lost_in_multilinguality_dissecting_cross-lingual_factual_inconsistency_in_transf.md)
- [\[ACL 2025\] MaXIFE: Multilingual and Cross-lingual Instruction Following Evaluation](maxife_multilingual_and_cross-lingual_instruction_following_evaluation.md)
- [\[ACL 2025\] Cross-Lingual Generalization and Compression: From Language-Specific to Shared Neurons](cross_lingual_neurons_compression.md)

</div>

<!-- RELATED:END -->
