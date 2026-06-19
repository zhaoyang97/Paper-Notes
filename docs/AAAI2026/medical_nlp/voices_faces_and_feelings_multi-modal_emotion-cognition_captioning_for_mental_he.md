---
title: >-
  [论文解读] Voices, Faces, and Feelings: Multi-modal Emotion-Cognition Captioning for Mental Health Understanding
description: >-
  [AAAI 2026][医疗NLP][情感认知描述] 提出情感-认知协同多模态描述（ECMC）任务和框架，通过双流BridgeNet从视频、音频、文本中提取情感和认知特征，利用LLaMA生成自然语言描述，为心理健康评估提供可解释的情感-认知画像，显著提升辅助诊断的准确性和可解释性。 心理健康问题日益严峻——全球超过3亿人受抑…
tags:
  - "AAAI 2026"
  - "医疗NLP"
  - "情感认知描述"
  - "多模态"
  - "心理健康"
  - "大语言模型"
  - "抑郁检测"
---

# Voices, Faces, and Feelings: Multi-modal Emotion-Cognition Captioning for Mental Health Understanding

**会议**: AAAI 2026  
**arXiv**: [2603.01816](https://arxiv.org/abs/2603.01816)  
**代码**: [github](https://github.com/zhouzyhfut/ECMC)  
**领域**: 医学NLP  
**关键词**: 情感认知描述, 多模态, 心理健康, 大语言模型, 抑郁检测

## 一句话总结
提出情感-认知协同多模态描述（ECMC）任务和框架，通过双流BridgeNet从视频、音频、文本中提取情感和认知特征，利用LLaMA生成自然语言描述，为心理健康评估提供可解释的情感-认知画像，显著提升辅助诊断的准确性和可解释性。

## 研究背景与动机

心理健康问题日益严峻——全球超过3亿人受抑郁症影响，WHO预测到2030年未治疗的精神障碍将占总疾病负担的13%。现有辅助诊断方法面临三大核心挑战：

**挑战一：分类范式缺乏可解释性。** 大多数方法将多模态数据分类为某种精神障碍类别（如"抑郁"/"焦虑"），但不揭示**哪些线索**与心理状况相关。临床医生无法从一个分类标签中获得有价值的诊断依据。

**挑战二：LLM方法依赖文本语义。** 虽然LLM在自然语言理解上很强，但应用于心理健康分析时，它们主要检测症状相关词汇，无法捕捉面部表情、声音语调等非语言情感和认知信号——而这些在临床观察中至关重要。

**挑战三：情感-认知模式未被充分利用。** 神经科学研究表明，精神障碍不仅体现在体验到的症状上，更反映在情感和认知处理的动态机制中。例如，抑郁通常伴随持久的情感停滞和认知功能抑制，而焦虑障碍则表现为快速的情感和认知波动。但现有方法很少从多模态视角综合审视这些模式。

**本文核心idea**：将心理健康分析从**分类任务**转变为**描述生成任务**——不是简单输出一个标签，而是生成自然语言描述来刻画患者的情感状态和认知障碍，从而为临床诊断提供可解释的证据。这就是ECMC（Emotion-Cognition cooperative Multi-modal Captioning）任务。

## 方法详解

### 整体框架

ECMC采用编码器-解码器架构（如图2所示），包含三个核心组件：

1. **模态特定编码器**：分别提取视频、音频、文本的初始表征
2. **双流BridgeNet**：基于Q-Former的情感和认知特征提取与融合模块
3. **LLaMA解码器**：将对齐的情感-认知特征转换为自然语言描述

系统的工作流程为：**多模态输入 → 初始特征 → 双流BridgeNet压缩融合 → E-embedding + C-embedding → LLaMA生成描述 → 汇总为用户画像 → 辅助诊断**。

### 关键设计

#### 1. **模态特定编码器**

对于给定的话语样本 $\bm{x}_i = \{\bm{X}_v, \bm{X}_a, \bm{X}_t\}$，使用三个预训练模型提取初始表征：

- **视频**：VideoMAE提取面部表情和肢体语言特征
- **音频**：HuBERT提取语音音调、语速等声学特征  
- **文本**：BERT提取文本语义特征

这些编码器参数在训练中**冻结**，因为临床数据有限，从头训练编码器不可行。但预训练模型关注帧级表示，无法捕捉情感和认知相关的语义。

#### 2. **双流BridgeNet（核心创新）**

受BLIP-2启发，设计了基于Q-Former的双流BridgeNet来压缩和解耦情感与认知表征。

**情感BridgeNet**：

每个模态引入可学习的查询token $\bm{Q}_m$，通过自注意力建模查询间依赖，再通过交叉注意力从模态特征中提取信息：

$$\bm{Z}_m = \text{softmax}\left(\frac{\bm{Q}'_m \bm{W}^{\prime(c)}_q (\bm{H}'_m \bm{W}^{\prime(c)}_k)^\top}{\sqrt{d_k}}\right) \bm{H}'_m \bm{W}^{\prime(c)}_v$$

然后将三个模态的表征拼接、投影、归一化得到E-embedding $\bm{h}_e$。

**情感对比学习**：根据效价维度将表征分为负面/中性/正面三类，引入标签匹配的对比学习。损失函数同时优化类内紧凑性和类间可分离性：

$$\mathcal{L}_{\text{emo}} = -\frac{1}{N}\sum_{i=1}^{N}\frac{1}{|\{j:\bm{M}_{ij}=1\}|}\sum_{j:\bm{M}_{ij}=1}\log p_{ij} + \frac{1}{N}\sum_{i=1}^{N}\log(1+\sum_{j:\bm{M}_{ij}=0}\exp(\bm{S}_{ij}))$$

**认知BridgeNet**：

结构类似情感BridgeNet，但用于提取认知障碍表征。以MMSE临床认知量表为指导，关注四种认知障碍：**定向力障碍、注意力障碍、记忆力障碍、语言障碍**。

由于一个样本可能同时存在多种认知障碍（多标签），设计了基于**Jaccard相似度**的多标签对比学习：

$$\bm{W}_{ij} = \frac{|\bm{y}_{c,i} \cap \bm{y}_{c,j}|}{|\bm{y}_{c,i} \cup \bm{y}_{c,j}|}$$

标签重叠度更高的样本被拉得更近，实现了对多标签认知障碍的软对比。

#### 3. **LLaMA解码器**

将BOS token、E-embedding、C-embedding和Prompt拼接后输入LLaMA：

$$\bm{u} = \mathcal{F}_{llm}(\text{concat}(\text{<BOS>}, \bm{h}_e, \bm{h}_c, \text{Prompt}))$$

得到话语级描述后，再汇总生成用户画像 $\bm{p}$，最终辅助LLM进行心理障碍检测。

### 损失函数 / 训练策略

**两阶段训练**：

- **第一阶段**：联合训练情感和认知表征提取（$\mathcal{L}_1 = \mathcal{L}_{\text{emo}} + \mathcal{L}_{\text{cog}}$），冻结模态编码器，Q-Former用BERT预训练参数初始化。训练500个epoch，batch=64。
- **第二阶段**：微调BridgeNet使特征对齐LLM输入空间（$\mathcal{L}_2 = \text{CELoss}(\hat{\bm{u}}, \bm{u})$），冻结模态编码器和LLM参数，micro batch=8。

总参数约7.6B，其中可训练约605M。使用DeepSpeed ZeRO stage-2优化。

## 实验关键数据

### 主实验

**情感描述生成性能**：

| 方法 | 模态 | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | F_BERT |
|------|------|--------|--------|--------|---------|--------|
| InternVL-2.5-8B | VT | 8.35 | 0.96 | 10.06 | 13.98 | 6.48 |
| Sa2VA-8B | VT | 14.36 | 2.14 | 15.35 | 20.34 | 12.28 |
| Qwen2.5-Omni-7B | AVT | 12.74 | 1.42 | 13.31 | 16.99 | 8.93 |
| CPsyCoun | T | 17.44 | 2.33 | 15.07 | 18.90 | 9.30 |
| **Ours** | **AVT** | **34.76** | **8.28** | **29.47** | **24.91** | **27.13** |

**认知描述生成性能**：

| 方法 | 模态 | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | F_BERT |
|------|------|--------|--------|--------|---------|--------|
| Sa2VA-8B | VT | 15.82 | 1.61 | 13.15 | 22.59 | 18.78 |
| Qwen2.5-Omni-7B | AVT | 12.82 | 1.48 | 12.04 | 19.56 | 20.03 |
| **Ours** | **AVT** | **35.92** | **15.32** | **35.86** | **39.82** | **41.04** |

**辅助抑郁检测性能提升**（使用不同方法生成的情感-认知画像）：

| 画像来源 | 平均ACC提升 | 平均F1提升 |
|---------|-----------|-----------|
| InternVL-2.5-8B | +9.48% | +3.56% |
| Sa2VA-8B | +10.40% | +5.30% |
| EmoLLM | +6.42% | +5.39% |
| **Ours** | **+12.54%** | **+9.16%** |

### 消融实验

| 模态配置 | EmoCL | CogCL | F_BERT |
|---------|-------|-------|--------|
| Audio | ✓ | ✓ | 18.10 |
| Video | ✓ | ✓ | 16.30 |
| Audio+Text | ✓ | ✓ | 22.05 |
| Audio+Video+Text | ✗ | ✗ | 23.89 |
| Audio+Video+Text | ✓ | ✗ | 27.24 |
| Audio+Video+Text | ✗ | ✓ | 26.94 |
| **Audio+Video+Text** | **✓** | **✓** | **34.09** |

### 关键发现

1. **多模态融合至关重要**：音频贡献最大，三模态组合效果最佳（34.09 vs 单模态最好18.10）
2. **双流结构不可或缺**：去掉任一对比学习分支都会导致显著性能下降
3. **情感-认知画像显著提升辅助诊断**：抑郁检测ACC提升12.54%、F1提升9.16%
4. **低质量描述可能反而降低检测性能**：冗长但信息量低的文本增加了LLM提取相关指标的难度
5. **抑郁患者和焦虑患者的情感-认知模式存在显著差异**：抑郁与更高频率的认知障碍相关

## 亮点与洞察

1. **任务定义有创新性**：将心理健康分析从分类转为描述生成，开辟了新的研究范式
2. **双流BridgeNet设计精巧**：将情感和认知解耦提取，分别使用三分类对比和多标签Jaccard对比，符合领域特性
3. **临床指导明确**：以MMSE量表为指导设计认知障碍类型，具有实际临床意义
4. **实验评估维度丰富**：同时进行客观指标（BLEU/ROUGE等）和主观评估（心理学专家评分），增强说服力
5. **端到端可用**：从多模态输入到辅助诊断的完整流程，具有实际应用前景

## 局限与展望

1. 情感和认知描述的标注依赖LLM自动生成+人工修正，可能引入偏差
2. 仅在MMDA单一数据集上评估，泛化性有待验证
3. 认知准确度（CAcc）在人工评估中得分较低（3.9），说明多模态认知建模仍有提升空间
4. 模型总参数7.6B，部署成本较高，可探索更轻量级方案
5. 数据集中正常/异常样本比例可能不平衡，影响模型学习

## 相关工作与启发

- **BLIP-2** (Li et al., 2023)：BridgeNet的设计灵感来源，Q-Former架构的成功应用
- **SECap** (Xu et al., 2024)：语音情感描述任务，本文扩展到多模态和认知维度
- **Emotion-LLaMA** (Cheng et al., 2024)：用于生成初始情感标注，是数据管线的关键组件
- 启发：心理健康AI不应止步于"正常/异常"的二分类，而应提供人类可理解的分析报告；多模态融合中音频信号的重要性可能被低估

## 评分
- 新颖性: ⭐⭐⭐⭐⭐（首次提出ECMC任务，将心理健康分析从分类转为描述生成）
- 实验充分度: ⭐⭐⭐⭐（客观+主观评估全面，但仅单一数据集）
- 写作质量: ⭐⭐⭐⭐（结构清晰，动机阐述充分）
- 价值: ⭐⭐⭐⭐⭐（对心理健康AI领域有重要推动作用，具有实际临床意义）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling](learning_cell-aware_hierarchical_multi-modal_representations.md)
- [\[ACL 2026\] Responsible Evaluation of AI for Mental Health](../../ACL2026/medical_nlp/responsible_evaluation_of_ai_for_mental_health.md)
- [\[ACL 2026\] MHGraphBench: Knowledge Graph-Grounded Benchmarking of Mental Health Knowledge in Large Language Models](../../ACL2026/medical_nlp/mhgraphbench_knowledge_graph-grounded_benchmarking_of_mental_health_knowledge_in.md)
- [\[ACL 2026\] Measuring What Matters!! Assessing Therapeutic Principles in Mental-Health Conversation](../../ACL2026/medical_nlp/measuring_what_matters_assessing_therapeutic_principles_in_mental-health_convers.md)
- [\[ACL 2026\] MHSafeEval: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models](../../ACL2026/medical_nlp/mhsafeeval_role-aware_interaction-level_evaluation_of_mental_health_safety_in_la.md)

</div>

<!-- RELATED:END -->
