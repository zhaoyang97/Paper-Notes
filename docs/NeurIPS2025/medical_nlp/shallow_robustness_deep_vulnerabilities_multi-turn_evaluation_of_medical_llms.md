---
title: >-
  [论文解读] Shallow Robustness, Deep Vulnerabilities: Multi-Turn Evaluation of Medical LLMs
description: >-
  [NeurIPS 2025][医疗NLP][医学LLM] 提出MedQA-Followup框架系统评估医学LLM的多轮鲁棒性，发现模型在单轮扰动下表现尚可（浅层鲁棒性），但在多轮追问中准确率可从91.2%暴跌至13.5%（深层脆弱性），且间接上下文操纵比直接错误建议更具破坏力。 医学LLM正快速从研究原型进入临床应用…
tags:
  - "NeurIPS 2025"
  - "医疗NLP"
  - "医学LLM"
  - "多轮对话鲁棒性"
  - "认知偏差"
  - "MedQA"
  - "临床安全"
---

# Shallow Robustness, Deep Vulnerabilities: Multi-Turn Evaluation of Medical LLMs

**会议**: NeurIPS 2025  
**arXiv**: [2510.12255](https://arxiv.org/abs/2510.12255)  
**代码**: [GitHub](https://github.com) / [HuggingFace](https://huggingface.co)  
**领域**: 医学NLP  
**关键词**: 医学LLM, 多轮对话鲁棒性, 认知偏差, MedQA, 临床安全

## 一句话总结

提出MedQA-Followup框架系统评估医学LLM的多轮鲁棒性，发现模型在单轮扰动下表现尚可（浅层鲁棒性），但在多轮追问中准确率可从91.2%暴跌至13.5%（深层脆弱性），且间接上下文操纵比直接错误建议更具破坏力。

## 研究背景与动机

医学LLM正快速从研究原型进入临床应用，但现有评估几乎全部聚焦于**单轮问答**的理想条件。然而真实医学场景中：

**临床对话本质上是多轮的**：医生会收到同事的不同意见、患者带来的网络搜索结果、权威人士的反对观点

**现有鲁棒性研究局限于"浅层"**：BiasMedQA和KGGD等工作仅考察在首次提问中加入误导信息的影响，未涉及模型给出答案后被质疑的场景

**关键问题未被回答**：当LLM已给出正确诊断后被追问，它能否坚守正确答案？社会压力和权威影响如何作用？

本文认为，真正的安全部署需要区分两种鲁棒性：
- **浅层鲁棒性（Shallow Robustness）**：抵抗初始提示中的误导信息
- **深层鲁棒性（Deep Robustness）**：在后续对话轮次中被挑战时仍保持准确

## 方法详解

### 整体框架

MedQA-Followup构建了一个**二维分类法**来组织所有评估方式：

- **时间轴**：单轮（$I(Q) \to A'$） vs 多轮追问（$Q \to A \to I_1(Q) \to A_1$） vs 累积追问（$Q \to A \to I_1 \to A_1 \to I_2 \to A_2 \to \ldots$）
- **意图轴**：间接干预（不明确推向错误答案）vs 直接干预（明确建议错误答案）

基于MedQA数据集（1273道USMLE考题），覆盖15个医学领域和3个考试阶段。

### 关键设计

1. **中性重评估（rethink）**：作为控制组的5种技术——"高风险中性"（强调临床重要性）、"时间中性"（简要回顾）、"假设检查"（识别无支撑假设）、"双重检查"（显式验证推理）、"选项映射"（系统排除矛盾选项）。这些不引入任何误导信息，仅测试重新评估本身是否降低准确率。

2. **错误建议（inc_letter）**：通过社会/权威框架明确建议错误答案的5种技术——"权威先验"（资深临床医生意见）、"自动评分器先验"（系统期望输出）、"承诺对齐"（多来源支持错误选项）、"近因先验"（引用类似近期病例）、"社会证明先验"（同事共识）。关键设计：始终以"他人观点"而非"事实陈述"框定（如"一位资深医生认为答案是X"）。

3. **上下文操纵（context）**：引入额外误导信息的4种技术——"误导上下文"和"RAG风格上下文"（GPT-4.1生成的支持第二可能选项的文本，分别以临床证据和检索系统形式呈现）、"替代上下文"（引入答案集外的诊断支持证据）、"边缘案例上下文"（强调正确答案的非典型表现和局限性）。所有上下文4-10句，以"需权衡的信息"而非"确定性事实"框定。

### 损失函数 / 训练策略

本文为评估框架，不涉及模型训练。评估配置：
- 5个模型：GPT-4.1、GPT-4.1 mini、Claude Sonnet 4、MedGemma 27B、MedGemma 4B
- 确定性解码（temperature=0），固定随机种子42
- 通用模型使用简单系统提示，领域模型（MedGemma）不用系统提示

## 实验关键数据

### 主实验：单轮 vs 多轮准确率

| 模型 | 基线 | 单轮inc_letter(均) | 多轮rethink(均) | 多轮inc_letter(均) | 多轮context(均) |
|---|---|---|---|---|---|
| Claude Sonnet 4 | 91.2% | 89.3% (-2.1%) | 84.6% (-7.2%) | 48.6% (**-46.7%**) | 45.6% (**-50.0%**) |
| GPT-4.1 | 92.5% | 92.1% (-0.4%) | 92.0% (-0.5%) | 89.5% (-3.2%) | 61.7% (**-33.2%**) |
| GPT-4.1 mini | 90.5% | 89.6% (-1.0%) | 89.9% (-0.6%) | 87.0% (-3.8%) | 50.1% (**-44.6%**) |
| MedGemma 4B | 64.2% | 53.1% (-17.3%) | 64.1% (-0.1%) | 54.2% (-15.5%) | 40.0% (-37.6%) |
| MedGemma 27B | 84.7% | 78.8% (-6.9%) | 84.4% (-0.4%) | 69.1% (-18.3%) | 56.9% (-32.8%) |

### 消融实验：最具破坏力的技术

| 模型 | 最差context技术 | 准确率 | 相对下降 |
|---|---|---|---|
| Claude Sonnet 4 | RAG style context | 13.5% | -85.2% |
| GPT-4.1 | RAG style context | 47.8% | -48.3% |
| GPT-4.1 mini | RAG style context | 32.5% | -64.1% |
| MedGemma 4B | Misleading context | 20.7% | -67.8% |
| MedGemma 27B | RAG style context | 27.6% | -67.4% |

### 关键发现

1. **浅层鲁棒性已基本解决**：单轮直接建议对SOTA模型几乎无效（GPT-4.1仅下降0.4%）
2. **深层鲁棒性是灾难性的**：所有模型在多轮context干预下准确率下降超30%
3. **反直觉发现**：间接干预（context）比直接干预（inc_letter）破坏力更大
4. **模型差异显著**：Claude对权威建议极度脆弱（-46.7%），GPT系列对inc_letter几乎免疫但对context同样脆弱
5. **临床应用题（Step 2&3）比基础知识题（Step 1）更脆弱**：context干预在临床题上额外降低6-13.5%
6. **累积干预呈亚加性**：85%的组合表现出亚加性效应，说明利用堆叠干预的可攻击性有自然上限

## 亮点与洞察

- 首次系统定义和评估医学LLM的"深层鲁棒性"概念
- 揭示了一个关键安全漏洞：RAG系统整合的上下文可能比直接错误建议更危险
- 分类法（浅/深 × 直接/间接）为后续研究提供了清晰框架
- Claude Sonnet 4的极端脆弱性令人警醒（91.2%→13.5%）

## 局限与展望

- 评估基于选择题格式，可能低估开放式临床对话中的脆弱性
- 仅评估5个模型，更多模型家族（如Gemini、Llama 3.1）未覆盖
- 未探索对抗训练或置信度加权等缓解策略
- 上下文生成依赖GPT-4.1，引入方法偏差
- 累积干预的组合空间巨大，仅覆盖了子集

## 相关工作与启发

- **BiasMedQA**：单轮偏差评估的先驱，本文扩展到多轮
- **AgentClinic**：模拟多轮临床环境评估，但未聚焦于答案被挑战的场景
- 启发：在RAG系统日益普及的背景下，如何确保检索到的上下文不会误导模型是一个亟待解决的问题

## 评分

- 新颖性: ⭐⭐⭐⭐ 深层/浅层鲁棒性的概念划分新颖且实用
- 实验充分度: ⭐⭐⭐⭐⭐ 5个模型、14种干预技术、累积分析、长度消融、领域分析
- 写作质量: ⭐⭐⭐⭐ 分类法清晰，实验组织有条理
- 价值: ⭐⭐⭐⭐⭐ 直击LLM临床部署的核心安全问题，发现具有重大实践意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Improving Automatic Evaluation of LLMs in Biomedical Relation Extraction via LLMs-as-the-Judge](../../ACL2025/medical_nlp/biore_llm_judge_evaluation.md)
- [\[ACL 2025\] Evaluation of LLMs in Medical Text Summarization: The Role of Vocabulary Adaptation in High OOV Settings](../../ACL2025/medical_nlp/evaluation_of_llms_in_medical_text_summarization_the_role_of_vocabulary_adaptati.md)
- [\[ACL 2026\] IndicMedDialog: A Parallel Multi-Turn Medical Dialogue Dataset for Accessible Healthcare in Indic Languages](../../ACL2026/medical_nlp/indicmeddialog_a_parallel_multi-turn_medical_dialogue_dataset_for_accessible_hea.md)
- [\[ICLR 2026\] ATPO: Adaptive Tree Policy Optimization for Multi-Turn Medical Dialogue](../../ICLR2026/medical_nlp/atpo_adaptive_tree_policy_optimization_for_multi-turn_medical_dialogue.md)
- [\[NeurIPS 2025\] H-DDx: A Hierarchical Evaluation Framework for Differential Diagnosis](h-ddx_a_hierarchical_evaluation_framework_for_differential_diagnosis.md)

</div>

<!-- RELATED:END -->
