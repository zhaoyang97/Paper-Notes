---
title: >-
  [论文解读] Enhancing Medical Dialogue Generation through Knowledge Refinement and Dynamic Prompt Adjustment
description: >-
  [ACL 2025][医疗NLP][medical dialogue system] 提出 MedRef，一种融合知识精炼机制和动态 Prompt 调整策略的医学对话系统，通过隐变量过滤无关知识图谱三元组、实体-行为联合预测、以及三元组过滤器和示例选择器动态构建系统 Prompt，在 MedDG 和 KaMed 两个基准上取得 SOTA 性能。
tags:
  - "ACL 2025"
  - "医疗NLP"
  - "medical dialogue system"
  - "knowledge refining"
  - "提示学习"
  - "entity prediction"
  - "knowledge graph"
---

# Enhancing Medical Dialogue Generation through Knowledge Refinement and Dynamic Prompt Adjustment

**会议**: ACL 2025  
**arXiv**: [2506.10877](https://arxiv.org/abs/2506.10877)  
**代码**: [GitHub](https://github.com/simon-p-j-r/MedReF)  
**领域**: 医疗NLP  
**关键词**: medical dialogue system, knowledge refining, dynamic prompt, entity prediction, knowledge graph

## 一句话总结

提出 MedRef，一种融合知识精炼机制和动态 Prompt 调整策略的医学对话系统，通过隐变量过滤无关知识图谱三元组、实体-行为联合预测、以及三元组过滤器和示例选择器动态构建系统 Prompt，在 MedDG 和 KaMed 两个基准上取得 SOTA 性能。

## 研究背景与动机

**领域现状**: 医学对话系统（MDS）旨在支持医患多轮上下文感知对话，需要能追踪患者不断变化的健康状态，并使用医学领域知识进行准确回应。现有方法常从医学知识图谱（MedKG）检索相关实体来增强响应生成。

**现有痛点**: 
   - 检索增强生成（RAG）方法常引入不相关的知识，反而降低响应质量
   - 大语言模型虽提升了流畅度，但对 Prompt 结构和内容高度敏感
   - 现有 Prompt 设计缺乏根据实时患者信息动态调整的能力

**核心矛盾**: 从知识图谱检索的实体存在大量噪声，且同一 Prompt 模板难以适配不同患者的多样化就诊情况。

**本文目标**: (1) 精炼检索到的知识以获得更准确的响应指导；(2) 动态调整系统 Prompt 以适配特定患者条件。

**切入角度**: 引入隐变量建模进行知识精炼（过滤无关三元组），结合实体-行为联合预测，然后通过 Triplet Filter 和 Demo Selector 动态构建多组件 Prompt。

**核心 idea**: 用 VAE 风格的隐变量精炼知识图谱检索结果，再通过动态调整 Prompt 的知识三元组和示例对话来提升医学对话的实体准确性和生成质量。

## 方法详解

### 整体框架

MedRef 包含三个阶段：(1) 编码对话历史并从 MedKG 检索相关实体子图；(2) 知识精炼 + 实体-行为联合预测；(3) 动态 Prompt 调整 + 大语言模型响应生成。

### 关键设计

1. **输入表示**: 使用 MedBERT 作为编码器骨干网络，编码患者话语和医生回复得到上下文表示 $e_{\bar{c}_t}$。从 MedKG 检索历史实体的一跳子图 $G_{\bar{x}_t}^0$，通过 GAT（图注意力网络）编码得到子图表示 $e_{\bar{x}_t}^{G_0}$。同时编码对话行为表示 $e_{\bar{a}_t}$。

2. **知识精炼机制（KRM）**: 引入隐变量 $z_t$ 建模先验分布 $p_\theta(z_t|\bar{c}_t, G_{\bar{x}_t}^0) = \mathcal{N}(\mu_\theta, \Sigma_\theta)$ 和后验分布 $q_\phi(z_t|\bar{c}_t, G_{\bar{x}_t}^0, x_t) = \mathcal{N}(\mu_\phi, \Sigma_\phi)$（后验利用了真实目标实体 $x_t$ 的信息）。采样 $z_t$ 后通过解码器并与原始实体嵌入残差连接：$e_{\bar{x}_t}^G = f_{dec}(z_t) + e_{\bar{x}_t}^{G_0}$，从而过滤掉无关知识，保留与当前对话上下文相关的实体信息。

3. **实体-行为联合预测**: 利用交叉注意力模块 $f_{ca}$ 建模上下文、精炼实体、历史行为之间的交互，经过 GRU 融合后通过线性层 + sigmoid 预测目标实体 $\hat{x}_t = \sigma(W_x \tilde{e}_{\bar{x}_t}^G + b_x)$ 和目标行为 $\hat{a}_t = \sigma(W_a \tilde{e}_{\bar{a}_t} + b_a)$。实体和行为的高对应关系（如"症状"对应"症状询问"，"疾病"对应"诊断"）使得联合预测非常合理。

4. **Triplet Filter**: 将检索到的一跳子图转化为三元组集合，统计实体出现频率并设定阈值 $\tau$（从 1 开始递增）进行迭代过滤：$Tri_{\bar{x}_t}^\tau = \{(e_{head}, r, e_{tail}) | \min(\#e_{head}, \#e_{tail}) \geq \tau\}$，直到三元组数量不超过预定义最大值 $M=25$。

5. **Demo Selector**: 三步对齐选择最相关示例对话：(a) 实体对齐——按首轮话语实体匹配训练集对话；(b) 相似度对齐——编码后用余弦相似度选最近对话；(c) 跨度对齐——用滑动窗口 $\xi=2$ 提取焦点片段作为最终示例。

6. **动态 Prompt 结构**: $\mathcal{P} = [\mathcal{I}; \mathcal{H}; \mathcal{K}; \mathcal{E}]$，包含任务指令、历史详情（对话上下文+实体+行为）、证据详情（预测实体/行为+过滤后三元组）、相关示例。

### 损失函数/训练策略

两阶段训练：

第一阶段（实体-行为预测）：$\mathcal{L} = \lambda_x \mathcal{L}_x + \lambda_a \mathcal{L}_a + \lambda_{kl} \mathcal{L}_{kl}$，其中 $\lambda_x=1, \lambda_a=0.05, \lambda_{kl}=0.05$。$\mathcal{L}_x$ 和 $\mathcal{L}_a$ 为 BCE 损失，$\mathcal{L}_{kl}$ 为先验/后验的 KL 散度。

第二阶段（响应生成）：固定预测模块，用 LoRA (rank=8, $\alpha$=32) 微调 ChatGLM3-6B，最大化响应似然 $\mathcal{L}_{gen} = -\sum_t \log \sum_k p_{gen}(r_{t_k}|r_{t_{<k}}, \mathcal{P})$。

## 实验关键数据

### 主实验

| 方法 | MedDG B-1 | B-4 | E-F1 | R-1 | KaMed B-1 | B-4 | E-F1 | R-1 |
|------|-----------|-----|------|-----|-----------|-----|------|-----|
| DFMed | 41.74 | 22.48 | 21.54 | 28.90 | 39.59 | 20.30 | 21.33 | 27.67 |
| GPT-4o | 42.19 | 23.32 | 13.15 | 13.99 | 41.88 | 23.34 | 13.86 | 13.94 |
| ChatGLM3-6B | 33.16 | 17.97 | 17.43 | 29.27 | 32.03 | 16.68 | 20.56 | 28.02 |
| **MedRef** | **43.51** | **23.04** | **22.70** | **30.07** | **40.47** | **21.28** | **21.96** | **28.14** |

MedRef 在 MedDG 上全面领先：比 GPT-4o 在 ROUGE-1 上高 **16.08%**，Entity-F1 高 **11.05%**。

### 消融实验

| 变体 | MedDG B-1 | E-F1 | R-1 | KaMed B-1 | E-F1 | R-1 |
|------|-----------|------|-----|-----------|------|-----|
| MedRef (完整) | 43.51 | 22.70 | 30.07 | 40.47 | 21.96 | 28.14 |
| w/o KRM | 42.58 | 21.94 | 29.88 | 40.29 | 21.51 | 27.95 |
| w/o Demo | 41.80 | 21.84 | 29.69 | 39.07 | 20.09 | 27.35 |
| w/o Kg | 41.76 | 21.58 | 29.86 | 39.82 | 20.55 | 28.09 |
| E-A&Cxt only | 41.63 | 21.30 | 28.68 | 39.30 | 20.81 | 26.72 |
| Cxt only | 33.16 | 17.43 | 29.27 | 32.03 | 20.56 | 28.02 |

### 人工评估

| 方法 | 流畅度(FLU) | 知识准确性(KC) | 总体质量(OQ) |
|------|------------|---------------|-------------|
| Ground-truth | 3.70 | 3.75 | 3.95 |
| DFMed | 3.42 | 3.57 | 3.65 |
| E-A&Cxt only | 2.91 | 3.05 | 3.14 |
| **MedRef** | **3.55** | **3.68** | **3.79** |

### 关键发现

- **知识精炼机制（KRM）是最重要的组件**——移除后在所有指标上下降最大
- **盲目增加知识三元组数量反而有害**：Weak Kg（不过滤直接随机取三元组）性能低于完整 MedRef
- **随机选择示例对话（Weak Demo）同样有害**，说明 Demo Selector 的对齐过程至关重要
- GPT-4o 虽然 BLEU 分数高但 Entity-F1 很低（13.15 vs 22.70），生成冗长的 QA 风格回复导致实体准确性差
- MedRef 的人工评估分数最接近 Ground-truth

## 亮点与洞察

- **VAE 风格的知识精炼**是核心创新，用后验分布引导先验学会过滤无关检索知识，比简单的注意力过滤更有原则性
- **多组件动态 Prompt** 设计思路值得借鉴：将不同类型的信息（指令、历史、证据、示例）模块化，每个模块可独立优化
- **Triplet Filter 的频率迭代过滤**简单有效——高频实体往往更核心，通过递增阈值自然筛选
- **Demo Selector 的三步对齐**（实体→相似度→跨度）既保证了语义相关性，又控制了 Prompt 长度

## 局限与展望

- MedRef 在 KaMed 上的 BLEU 分数略低于 HuatuoGPT-II 和 GPT-4o，作者解释为 KaMed 跨 100+ 科室太复杂，但也说明方法在超大知识范围时的局限
- 知识图谱依赖 CMeKG（中文医学知识图谱），英文场景迁移需要替换知识源
- 两阶段训练流程较复杂，预测模块和生成模块分开训练可能不是最优的端到端方案
- Demo Selector 需要预先组织好训练集对话的索引，在线推理时的检索效率未讨论
- 未评估在开放域医学问题上的泛化能力

## 相关工作与启发

- **DFMed** (Xu et al., 2023) 是最主要的比较对象，MedRef 在其基础上增加了知识精炼和动态 Prompt
- **VRBot** (Li et al., 2021) 建模了患者状态和医生行为，MedRef 进一步精炼了知识图谱检索
- **MedPIR** (Zhao et al., 2022) 召回关键信息作为前缀的思路与 MedRef 的 Prompt 设计有共通之处
- 启发：在知识图谱增强生成中，"检索后精炼"比"直接检索"更重要

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 知识精炼的 VAE 建模和三步 Demo 选择都有较好的原创性
- **实验充分度**: ⭐⭐⭐⭐ — 两个数据集、完整消融、人工评估、案例分析
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，公式推导完整，案例分析有说服力
- **价值**: ⭐⭐⭐⭐ — 对医学对话系统有实际意义，代码已开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Radar: Enhancing Radiology Report Generation with Supplementary Knowledge Injection](radar_radiology_report_gen.md)
- [\[ACL 2025\] Pattern Recognition or Medical Knowledge? The Problem with Multiple-Choice Questions in Medicine](pattern_recognition_or_medical_knowledge_the_problem_with_multiple-choice_questi.md)
- [\[ACL 2025\] Are LLMs Effective Psychological Assessors? Leveraging Adaptive RAG for Interpretable Mental Health Screening through Psychometric Practice](are_llms_effective_psychological_assessors_leveraging_adaptive_rag_for_interpret.md)
- [\[ACL 2025\] Towards Omni-RAG: Comprehensive Retrieval-Augmented Generation for Large Language Models in Medical Applications](omni_rag_medical.md)
- [\[ACL 2025\] MedBioRAG: Semantic Search and Retrieval-Augmented Generation with Large Language Models for Medical and Biological QA](medbiorag_semantic_search_and_retrieval-augmented_generation_with_large_language.md)

</div>

<!-- RELATED:END -->
