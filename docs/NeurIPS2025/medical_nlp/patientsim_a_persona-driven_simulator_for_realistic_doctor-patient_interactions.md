---
title: >-
  [论文解读] PatientSim: A Persona-Driven Simulator for Realistic Doctor-Patient Interactions
description: >-
  [NeurIPS 2025 Spotlight][医疗NLP][患者模拟器] 提出PatientSim——基于真实MIMIC临床数据和四维人格轴（性格、语言能力、病史记忆水平、认知混乱程度）的LLM患者模拟器，生成37种独特人格组合，在8个LLM上评估事实准确性和人格一致性，由4名临床专家验证平均质量得分3.89/4。
tags:
  - "NeurIPS 2025 Spotlight"
  - "医疗NLP"
  - "患者模拟器"
  - "LLM角色扮演"
  - "医患对话"
  - "人格建模"
  - "临床教育"
---

# PatientSim: A Persona-Driven Simulator for Realistic Doctor-Patient Interactions

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2505.17818](https://arxiv.org/abs/2505.17818)  
**作者**: Daeun Kyung, Hyunseung Chung, Seongsu Bae, Jiho Kim (KAIST), Jae Ho Sohn (UCSF), Taerim Kim (Samsung Medical Center), Soo Kyung Kim (Ewha Womans University), Edward Choi (KAIST)  
**代码**: [GitHub](https://github.com/dek924/PatientSim)  
**领域**: 医学图像  
**关键词**: 患者模拟器, LLM角色扮演, 医患对话, 人格建模, 临床教育  

## 一句话总结

提出PatientSim——基于真实MIMIC临床数据和四维人格轴（性格、语言能力、病史记忆水平、认知混乱程度）的LLM患者模拟器，生成37种独特人格组合，在8个LLM上评估事实准确性和人格一致性，由4名临床专家验证平均质量得分3.89/4。

## 研究背景与动机

### 问题背景
LLM在MedQA等医学问答基准上已超越人类专家，但这类基准均为单轮设定，患者数据直接提供。真实临床中，医生需要通过多轮上下文感知的对话主动收集患者信息。因此，评估医生LLM需要逼真的患者交互系统。传统标准化病人（Standardized Patients）依赖真人演员，成本高、可复制性差、扩展困难。

### 已有工作的不足
- 现有LLM患者模拟器主要关注症状信息的准确传达，忽略了真实患者行为的多样性
- 部分研究尝试用Big Five人格或职业关键词增加真实感，但未同时**实现和评估**多维度临床相关人格
- 心理咨询模拟器注重情感建模，但不适用于急诊等一般诊断场景
- 面对患者档案未定义的信息，现有方法通常拒绝回答或假设结果正常，限制了对话自然度

### 核心动机
构建一个同时兼顾临床事实准确性和多样化患者人格的开源模拟器，为医生LLM评估和医学教育提供可复现、可扩展、隐私合规的测试平台。

## 方法详解

### 问题范围与约束
聚焦急诊科（ED）首次就诊的单次会话场景：
- **仅病史采集阶段**：不涉及体检或实验室检查结果，因约80%诊断仅靠病史采集即可完成
- **单次会话**：不模拟纵向治疗效果或疾病进展，避免生成误导性临床结论
- **鉴别诊断**：基于口头信息进行鉴别诊断

### 患者档案构建
基于MIMIC-IV、MIMIC-IV-ED和MIMIC-IV-Note三个真实数据集：
- 从结构化表格提取准确数据，从临床笔记捕捉生活方式、当前症状等深层信息
- 每个患者档案包含**24个条目**，覆盖人口统计、社会/医学史、急诊就诊详情
- 共构建**170份档案**，覆盖5种高发疾病：心肌梗死、肺炎、尿路感染、肠梗阻、脑梗死
- 疾病选择标准：高临床流行率、可仅通过病史采集区分的症状、MIMIC-ED中数据充足

### 四维人格定义
1. **性格（Personality）**：6种——急躁型、过度焦虑型、不信任型、过度乐观型、话多型、中性型（基线）。基于文献综述和医学专家指导设计
2. **语言能力（Language Proficiency）**：3级——A（基础）、B（中级）、C（高级），基于CEFR框架整合
3. **病史回忆水平（Medical History Recall）**：2级——高回忆、低回忆
4. **认知混乱程度（Cognitive Confusion）**：2级——高混乱、正常。高混乱患者限定为中性性格+中级语言+高回忆，避免维度重叠

组合产生 6x3x2 = 36 种常规人格 + 1种高混乱人格 = **37种独特人格**。

### 提示词设计
- **PatientSim提示词**：包含档案信息、四维人格轴、通用行为准则，经LLM评估、作者定性分析和两轮医学专家反馈迭代优化
- **医生LLM提示词**：基于医学教科书和专家建议设计，确保包含所有常规必要问题

### 评估框架
设计三个研究问题：
- **RQ1**（人格忠实度）：LLM能否自然反映37种人格组合？自动+人工评估，5项标准4分制
- **RQ2**（事实准确性）：句子级+对话级双层评估，包括NLI蕴含/矛盾判断、信息覆盖率（ICov）和信息一致性（ICon）
- **RQ3**（临床合理性）：评估档案未定义信息的回答是否临床合理，采用4分制评分

## 实验关键数据

### 实验1：人格忠实度评估（RQ1）

使用Gemini-2.5-Flash作为评估器，在37种人格组合上评估8个LLM的人格反映能力。

| 模型 | 性格 | 语言 | 回忆 | 混乱 | 真实感 | 平均 |
|------|------|------|------|------|--------|------|
| Gemini-2.5-Flash | 3.94 | 3.54 | 3.64 | 3.38 | 3.37 | 3.57 |
| GPT-4o mini | 3.58 | 3.55 | 3.78 | 3.88 | 3.26 | 3.61 |
| DeepSeek-R1-Distill-Llama-70B | 3.87 | 3.58 | 3.42 | 2.50 | 3.19 | 3.31 |
| Qwen2.5-72B | 3.30 | 3.68 | 3.63 | 3.50 | 3.22 | 3.46 |
| **Llama-3.3-70B** | **3.92** | 3.40 | **3.78** | **4.00** | 3.28 | **3.68** |
| Llama-3.1-70B | 3.65 | 3.51 | 3.62 | 4.00 | 3.23 | 3.60 |
| Llama-3.1-8B | 3.53 | 3.29 | 3.70 | 4.00 | 3.20 | 3.54 |
| Qwen2.5-7B | 3.23 | 3.49 | 3.31 | 3.50 | 3.16 | 3.34 |

Llama系列在情感表达（性格、混乱）方面表现突出；通用基准性能与模拟忠实度不直接对应。

### 实验2：事实准确性与临床合理性评估（RQ2 & RQ3）

句子级评估覆盖信息句中的支持/非支持分类、NLI蕴含率和临床合理性评分。

| 模型 | 信息句比例 | 支持句比例 | 非支持句比例 | 蕴含率 | 矛盾率 | 合理性 |
|------|-----------|-----------|-------------|--------|--------|--------|
| Gemini-2.5-Flash | 97.2% | 76.3% | 31.6% | 97.8% | 2.2% | 3.953 |
| GPT-4o mini | 95.7% | 72.1% | 42.8% | 96.8% | 3.2% | 3.929 |
| **Llama-3.3-70B** | 95.8% | **79.6%** | 38.7% | **98.1%** | **1.9%** | **3.963** |
| Llama-3.1-70B | 94.8% | 81.3% | 40.7% | 96.8% | 3.2% | 3.955 |
| Llama-3.1-8B | 94.4% | 77.1% | 48.8% | 94.4% | 5.6% | 3.897 |
| Qwen2.5-72B | 97.5% | 68.3% | 46.8% | 95.4% | 4.6% | 3.928 |
| Qwen2.5-7B | 98.7% | 70.3% | 45.3% | 93.9% | 6.1% | 3.862 |

大模型（>=70B）在事实准确性和合理性上一致优于小模型（<=8B），Llama-3.3-70B在蕴含率和合理性上均为最优开源模型。

### 人工评估
4名临床专家对基于Llama-3.3-70B的PatientSim进行评估：
- 6项评估标准的**平均质量得分3.89/4**
- "该聊天机器人在临床教育中有用"维度得分**3.75/4**
- 非支持句合理性评分：4位临床专家分别为3.955、3.923、3.985、3.781
- 评估者间一致性（Gwet's AC1）：最高0.968，最低0.853，整体高度一致

## 亮点

- **多维人格建模**：首次在通用医患模拟器中系统定义并实现4轴37种人格组合，超越了简单的Big Five关键词描述
- **双层事实评估框架**：句子级NLI + 对话级覆盖率/一致性，提供了全面的患者模拟器评估方法论
- **临床专家高度认可**：4名临床专家平均3.89/4的质量评分，且评估者间一致性极高（AC1>0.85），验证了框架的临床可靠性
- **开放生成策略**：允许对档案未定义信息进行合理推理回答（而非简单拒答），提升了对话自然度，合理性评分3.91/4
- **开源可复现**：完整代码和基于Llama-3.3-70B的开源方案，降低使用门槛

## 局限与展望

- **数据源单一**：仅基于MIMIC数据库，可能限制发现的泛化性
- **文本模态限制**：无法模拟非语言表达（面部表情、肢体动作），人格表现不够完整
- **人工评估规模有限**：仅4名临床专家，评估结果的泛化性受限
- **疾病覆盖窄**：仅5种急诊常见疾病，未覆盖更广泛的临床场景
- **单次会话约束**：不支持纵向多次就诊模拟，无法评估治疗效果跟踪能力
- **语言限制**：目前仅支持英语对话，未涉及多语言场景

## 与相关工作的对比

- **Agent Hospital (Li et al., 2025)**：模拟整个医院工作流（患者、护士、医生），侧重最终任务准确率，本文专注患者模拟的真实感和多样性
- **MedIQ (Li et al., 2024a)**：评估医生LLM的交互问诊能力，但未关注患者模拟器本身的人格反映质量
- **Du et al. (2024)**：通过agent co-evolution模拟标准化病人，本文更注重系统化的人格定义和多维评估
- **心理咨询模拟 (Qiu et al., Wang et al.)**：强调情绪和主观反应建模，但不适用于急诊诊断场景
- **Fan et al. (2025)**：AI Hospital多agent基准，侧重基准评测而非模拟器真实感

## 评分

- 新颖性: ⭐⭐⭐⭐ — 四维人格建模和37种组合的系统化设计具有创新性，但LLM角色扮演的思路本身并不新颖
- 实验充分度: ⭐⭐⭐⭐ — 8个LLM对比+4名临床专家验证+句子级/对话级双层评估，较为全面；但仅5种疾病、170份档案规模有限
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，问题定义精确，评估方法论详尽
- 价值: ⭐⭐⭐⭐ — 开源可复现，对医学AI教育训练有实际应用价值；临床专家高度认可增强了说服力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Follow-up Question Generation for Enhanced Patient-Provider Conversations](../../ACL2025/medical_nlp/follow-up_question_generation_for_enhanced_patient-provider_conversations.md)
- [\[NeurIPS 2025\] Demo: Guide-RAG: Evidence-Driven Corpus Curation for Retrieval-Augmented Generation in Long COVID](demo_guide-rag_evidence-driven_corpus_curation_for_retrieval-augmented_generatio.md)
- [\[ACL 2025\] Query-driven Document-level Scientific Evidence Extraction from Biomedical Studies](../../ACL2025/medical_nlp/urca_biomedical_evidence_extraction.md)
- [\[ICML 2026\] MedCase-Structured: A Text-to-FHIR Dataset for Benchmarking Diagnostic Reasoning in Clinically Realistic EHR Settings](../../ICML2026/medical_nlp/medcase-structured_a_text-to-fhir_dataset_for_benchmarking_diagnostic_reasoning_.md)
- [\[ACL 2025\] CSTRL: Context-Driven Sequential Transfer Learning for Abstractive Radiology Report Summarization](../../ACL2025/medical_nlp/cstrl_context-driven_sequential_transfer_learning_for_abstractive_radiology_repo.md)

</div>

<!-- RELATED:END -->
