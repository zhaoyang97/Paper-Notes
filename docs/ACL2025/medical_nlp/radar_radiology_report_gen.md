---
title: >-
  [论文解读] Radar: Enhancing Radiology Report Generation with Supplementary Knowledge Injection
description: >-
  [ACL 2025][医疗NLP][radiology report generation] 提出 Radar 框架，通过区分 LLM 已掌握的可信内部知识和需要外部补充的知识，系统性地融合两种知识源以生成更准确的放射学报告。 研究领域现状：大语言模型（LLMs）在放射学报告生成任务中展现了卓越的文本生成能力…
tags:
  - "ACL 2025"
  - "医疗NLP"
  - "radiology report generation"
  - "knowledge injection"
  - "supplementary knowledge"
  - "LLM"
  - "chest X-ray"
---

# Radar: Enhancing Radiology Report Generation with Supplementary Knowledge Injection

**会议**: ACL 2025  
**arXiv**: [2505.14318](https://arxiv.org/abs/2505.14318)  
**代码**: [https://github.com/wjhou/Radar](https://github.com/wjhou/Radar)  
**领域**: 医学NLP  
**关键词**: radiology report generation, knowledge injection, supplementary knowledge, LLM, chest X-ray  

## 一句话总结

提出 Radar 框架，通过区分 LLM 已掌握的可信内部知识和需要外部补充的知识，系统性地融合两种知识源以生成更准确的放射学报告。

## 研究背景与动机

**研究领域现状：** 大语言模型（LLMs）在放射学报告生成任务中展现了卓越的文本生成能力。许多工作尝试通过检索领域特定知识来增强模型性能，但这些方法往往忽略了 LLM 内部已经编码的知识。

**现有方法的局限性：**（1）已有的知识增强方法常常检索到 LLM 已经掌握的冗余信息；（2）LLM 内部学到的知识并非总是可靠的，经常产生幻觉（如错误识别疾病）；（3）缺乏有效机制来区分模型的可信知识和不可信知识。

**核心动机：** 以图 1 的例子说明：LLM 正确识别了 Cardiomegaly（无需额外知识），生成的 Pleural Effusion 与专家模型一致（可信），但 Edema 存在不确定性（需要补充知识）。因此需要平衡利用 LLM 的内部知识和外部检索知识。

## 方法详解

### 整体框架

Radar 包含两个阶段：**Stage I: Preliminary Findings Generation（初步发现生成）** 和 **Stage II: Supplementary Findings Augmentation（补充发现增强）**。

### 关键设计

1. **内部知识可信度评估（Stage I）：** 先让 MLLM 生成初始报告，同时用一个独立的专家分类模型（图像编码器 + 文本编码器 + MLP）对影像进行观察分类。取初始报告的观察结果 $O_R$ 与专家模型的结果 $O_I$ 的交集 $O_\checkmark = O_I \cap O_R$ 作为高置信度的内部知识（Preliminary Findings）。

2. **补充知识检索与提取（Stage II）：** 利用专家模型的 14 类观察概率分布，通过 KL 散度计算样本相似度，检索 Top-K 相似报告。关键在于仅提取补充性知识：过滤掉与 Preliminary Findings 重叠的观察，只保留 $O_\delta = \mathcal{O} - O_\checkmark$ 对应的句子。

3. **观察识别增强生成（Observation Identification）：** 将 PF 和 SF 整合到临床上下文中，训练时要求模型先输出观察标签再生成报告文本，帮助模型在生成前先总结高层信息。

### 损失函数

- **专家模型训练：** 使用带 log-scale re-weighting 的二元交叉熵损失处理类别不平衡：$\alpha_i = \log(1 + |\mathcal{D}_{train}| / w_i)$
- **报告生成模型：** 标准的负对数似然损失：$\mathcal{L} = -\sum_{t=1}^{T} \log p(y_t)$

## 实验

### 主实验（MIMIC-CXR 数据集）

| 模型 | B-1 | B-4 | R-L | RG-F1 | 14Ma-F1 | 5Mi-F1 |
|------|-----|-----|-----|-------|---------|--------|
| R2GenGPT | 0.411 | 0.134 | 0.297 | - | 0.389 | - |
| LLaVA-Med | 0.354 | 0.149 | 0.276 | 0.191 | 0.269 | 0.439 |
| Med-PaLM | 0.323 | 0.115 | 0.275 | 0.267 | 0.398 | 0.579 |
| MAIRA-2 | 0.465 | 0.234 | 0.384 | **0.346** | 0.416 | 0.591 |
| Libra | **0.513** | 0.245 | 0.367 | 0.329 | 0.404 | 0.601 |
| **Radar (Ours)** | 0.509 | **0.262** | **0.397** | **0.346** | **0.460** | **0.627** |

Radar 在 B-4、ROUGE-L、14-class Macro-F1 和 5-class Micro-F1 上均取得最优或并列最优。

### 消融实验

| 消融变体 | 说明 | 效果 |
|----------|------|------|
| 去除 Preliminary Findings | 不区分可信/不可信内部知识 | 临床指标显著下降 |
| 去除 Supplementary Findings | 不使用外部检索知识 | 覆盖不全面的观察 |
| 去除 Observation Identification | 不预测观察标签 | 生成质量下降 |
| 使用全部检索知识（不过滤） | 引入冗余信息 | 性能不如过滤后 |

### 关键发现

- Radar 在三个基准数据集（MIMIC-CXR、CheXpert-Plus、IU X-ray）上均超越 SOTA
- 补充知识过滤（只保留非重叠观察）比使用全部检索知识更有效，验证了去冗余的必要性
- 专家模型引入临床上下文（Indication 等）后分类性能优于仅用图像的方案

## 亮点

- 创新性地区分 LLM 的可信内部知识和需要补充的外部知识，避免冗余检索
- 通过专家模型与 LLM 输出的交集来识别高置信度知识，设计巧妙
- Observation Identification 机制让模型先"思考"再"写报告"，提升生成质量
- 方法具有良好的通用性，可扩展到其他知识增强的医学 NLP 任务

## 局限性

- 依赖 CheXpert 的 14 类标签体系，无法覆盖所有放射学发现
- 专家模型的分类精度直接影响知识过滤质量
- 仅在胸部 X 光数据上验证，未扩展到 CT、MRI 等其他模态
- 检索知识库来源于训练集，可能存在分布偏差
- 两阶段推理引入额外计算开销（需要生成初步报告 + 检索 + 再生成）

## 相关工作

- **放射学报告生成：** Chen et al. 2020/2021 开创性工作；R2GenGPT、LLaVA-Med、Med-PaLM 等基于 LLM 的方法
- **知识增强生成：** Yang et al. 2021 的检索增强方法；Li et al. 2023 的领域知识注入
- **医学多模态模型：** MAIRA-1/2, CheXagent, LLaVA-Rad 等专用医学 MLLM
- **幻觉缓解：** Huang et al. 2025 关于 LLM 幻觉的研究

## 评分

| 维度 | 分数 (1-10) |
|------|------------|
| 创新性 | 7 |
| 技术深度 | 7 |
| 实验充分性 | 8 |
| 写作质量 | 7 |
| 实用价值 | 8 |
| 总分 | 7.4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Automated Structured Radiology Report Generation](automated_structured_radiology_report_generation.md)
- [\[ACL 2025\] Online Iterative Self-Alignment for Radiology Report Generation](oisa_radiology_report_gen.md)
- [\[ACL 2025\] Enhancing Medical Dialogue Generation through Knowledge Refinement and Dynamic Prompt Adjustment](enhancing_medical_dialogue_generation_through_knowledge_refinement_and_dynamic_p.md)
- [\[ACL 2025\] CSTRL: Context-Driven Sequential Transfer Learning for Abstractive Radiology Report Summarization](cstrl_context-driven_sequential_transfer_learning_for_abstractive_radiology_repo.md)
- [\[ACL 2026\] MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation](../../ACL2026/medical_nlp/march_multi-agent_radiology_clinical_hierarchy_for_ct_report_generation.md)

</div>

<!-- RELATED:END -->
