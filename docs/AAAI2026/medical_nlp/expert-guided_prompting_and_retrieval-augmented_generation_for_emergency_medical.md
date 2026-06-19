---
title: >-
  [论文解读] Expert-Guided Prompting and Retrieval-Augmented Generation for Emergency Medical Service Question Answering
description: >-
  [AAAI2026][医疗NLP][RAG] 构建首个EMS急救领域多选QA数据集EMSQA（24.3K题、10个临床主题、4个认证等级），提出Expert-CoT和ExpertRAG框架将领域专业属性注入LLM推理与检索，比标准RAG最高提升4.59%准确率。 领域现状：LLM在通用医学QA（MedQA、MedMCQA）上…
tags:
  - "AAAI2026"
  - "医疗NLP"
  - "RAG"
  - "chain-of-thought"
  - "EMS"
  - "domain expertise"
  - "MCQA"
---

# Expert-Guided Prompting and Retrieval-Augmented Generation for Emergency Medical Service Question Answering

**会议**: AAAI2026  
**arXiv**: [2511.10900](https://arxiv.org/abs/2511.10900)  
**代码**: [EMSQA](https://uva-dsa.github.io/EMSQA)  
**领域**: 医学NLP  
**关键词**: RAG, chain-of-thought, EMS, domain expertise, MCQA

## 一句话总结

构建首个EMS急救领域多选QA数据集EMSQA（24.3K题、10个临床主题、4个认证等级），提出Expert-CoT和ExpertRAG框架将领域专业属性注入LLM推理与检索，比标准RAG最高提升4.59%准确率。

## 研究背景与动机

**领域现状**：LLM在通用医学QA（MedQA、MedMCQA）上已表现出色，CoT推理和RAG检索是两种主流性能提升手段。

**现有痛点**：现有方法将推理和检索视为无差别的通用过程——模型看到问题后直接推理或检索，不区分问题涉及的具体临床领域（如创伤、气道管理、药理学）和认证等级（EMR/EMT/AEMT/Paramedic）。然而，真实的医疗专业人员在解题时，总是先判断问题属于哪个学科领域，再从对应的专业知识体系出发进行推理。

**核心矛盾**：EMS急救领域缺乏公开的QA数据集和结构化知识库，且现有CoT/RAG方法没有机制来利用问题级别的专业属性（subject area + certification level）指导推理和检索。

**本文解决方案**：构建EMSQA数据集和配套知识库，训练轻量Filter分类器推断问题的专业属性，然后通过Expert-CoT将属性注入提示模板引导推理、通过ExpertRAG按属性过滤知识库实现精准检索。核心idea是将"领域专家的认知流程"形式化为可注入LLM的显式信号。

## 方法详解

### 整体框架

三阶段pipeline：(1) 数据与知识构建——EMSQA数据集（24.3K MCQA）+ 分主题知识库（40K文档、2M tokens）+ 400万条NEMSIS患者记录；(2) Filter分类器——用LoRA微调LLM推断每道题的subject area和certification level；(3) 推理增强——Expert-CoT将属性注入提示模板、ExpertRAG按属性过滤检索范围，两者可组合使用。

### 关键设计

1. **Filter分类器（专业属性推断）**:
    - 功能：从输入问题和选项中自动推断subject area（10类，multi-label）和certification level（4类，single-label）
    - 核心思路：在LLM末尾追加`<classify>` token，提取最后一层hidden state $h_i$，通过两个分类head $W_{sub}$ 和 $W_{lvl}$ 分别输出预测。联合损失 $\mathcal{L} = w_{sub} \cdot \text{BCE}(p_i^{sub}, y_i^{sub}) + w_{lvl} \cdot \text{CE}(p_i^{lvl}, y_i^{lvl})$，用DWA动态调整两个任务的权重。推理时subject area用阈值0.5二值化，certification level取argmax
    - 设计动机：需要一个轻量模块在推理前快速判断问题的专业属性，LoRA fine-tuning仅需少量参数（rank=8），且multi-task训练让两个属性互相增强

2. **Expert-CoT（专业引导提示）**:
    - 功能：将Filter预测的subject area $\hat{s}_i$ 和certification level $\hat{l}_i$ 嵌入CoT提示模板，引导LLM从特定领域视角推理
    - 核心思路：标准CoT鼓励"step-by-step"但不指定起点，Expert-CoT显式提供领域起点。最终答案 $\hat{A}_i = f^{\text{CoT-Expert}}(q_i, \mathcal{O}_i, \hat{l}_i, \hat{s}_i)$，模板中包含"You are an expert in {subject area} at {certification level}"等引导语
    - 设计动机：模拟真实医疗专业人员的思维流程——先定位学科领域，再从对应知识体系出发推理，而非无差别地通用推理

3. **ExpertRAG（专业引导检索）**:
    - 功能：利用Filter预测的subject area过滤知识库和患者记录，实现领域对齐的检索增强生成
    - 核心思路：三种策略——Global（全库检索，baseline）、FTR（Filter-then-Retrieve，先按 $\hat{s}_i$ 过滤KB/PR再检索top-M/N）、RTF（Retrieve-then-Filter，先检索10倍候选再过滤）。最终 $\hat{A}_i = f^{\text{RAG}}(q_i, \mathcal{O}_i, \mathcal{R}(q_i, \hat{s}_i), \hat{l}_i, \hat{s}_i)$，使用MedCPT作为检索器，KB取top-32、PR取top-8
    - 设计动机：通用RAG检索全库会引入大量无关文档稀释相关性，按subject area分区检索相当于给检索器加上了领域先验，显著提高文档相关率

### 损失函数 / 训练策略

Filter用AdamW优化器（weight decay 0.01），LoRA参数 $r=8, \alpha=16$，dropout 0.05，序列长度128。DWA温度 $T=2$ 动态调整多任务权重。LLM推理阶段无额外训练，仅通过prompt工程和检索增强。在NVIDIA H200 GPU上运行。

## 实验关键数据

### 主实验

| 模型 | 方法 | Public Acc | Public F1 | Private Acc | Private F1 |
|------|------|-----------|-----------|------------|------------|
| Qwen3-32B | 0-shot | 83.55 | 83.55 | 85.11 | 85.89 |
| Qwen3-32B | CoT | 84.96 | 84.97 | 88.78 | 90.13 |
| Qwen3-32B | Expert-CoT (Filter) | 85.57 | 85.60 | 89.50 | 91.20 |
| OpenAI-o3 | 0-shot | 92.39 | 92.39 | — | — |
| Qwen3-4B | Global RAG + CoT | 78.12 | 79.17 | 75.46 | 76.87 |
| Qwen3-4B | ExpertRAG-GT RTF + Expert-CoT | **82.24** | **82.26** | **80.51** | **81.16** |
| Qwen3-4B | ExpertRAG-Filter RTF + Expert-CoT | 80.95 | 80.96 | 79.47 | 80.22 |

### 消融实验

| 配置 | Public Acc | Private Acc | 说明 |
|------|-----------|------------|------|
| Qwen3-4B 0-shot（无RAG） | 70.99 | 69.88 | 纯LLM基线 |
| + CoT | 72.35 | 70.58 | +1.36 |
| + Global RAG + CoT | 78.12 | 75.46 | RAG提升显著 |
| + Global RAG + Expert-CoT | 79.59 | 76.75 | Expert-CoT额外+1.47 |
| + ExpertRAG-GT FTR + Expert-CoT | 81.62 | 80.40 | 分区检索+3.50 vs Global |
| + ExpertRAG-GT RTF + Expert-CoT | 82.24 | 80.51 | RTF略优于FTR |

### 关键发现

- Expert-CoT比vanilla CoT稳定提升1-2%，在弱模型（OpenBioLLM）上提升更大（+2.05%），在强模型（Qwen3-32B）上提升约0.6%
- ExpertRAG + Expert-CoT组合比标准RAG最高提升4.59%准确率，两个增强手段有叠加效果
- Filter分类器subject area miF约80%、certification level miF约66%，即使存在分类误差仍能带来显著提升
- 使用Ground-truth expertise比Filter预测高约1.3%，说明Filter仍有改进空间
- Qwen3-32B + Expert-CoT通过了全部4个等级的NREMT认证模拟考试

## 亮点与洞察

- 将"领域专家的认知流程"形式化为可计算的属性注入机制，思路简洁但系统性好，且具有跨领域迁移潜力（法律、金融等专业领域可复用相同框架）。
- EMSQA是首个覆盖多认证等级、配有结构化KB和真实患者记录的EMS QA数据集，数据集本身具有长期基准价值。
- RTF策略（先大范围检索再按属性过滤）略优于FTR策略（先过滤再检索），暗示检索与过滤的顺序对效果有微妙影响——先检索保留了跨领域相关文档的可能性。

## 局限与展望

- Filter分类器准确率有限（subject area miF≈80%），误分类会传播到后续模块，可探索不确定性感知的软过滤
- 仅用Qwen3-4B做RAG实验，未在更大模型上验证ExpertRAG效果是否仍显著
- 数据集部分来自收费网站仅开放公开部分，限制了可复现性
- 知识库缺少certification level标注，无法实现更精细的认证等级对齐检索
- 仅评估多选QA场景，未扩展到开放式问答或临床决策支持

## 相关工作与启发

- **vs MedRAG**：MedRAG使用通用医学语料做hybrid sparse-dense检索（74.31%），ExpertRAG通过领域对齐KB高约8%，证明分区检索的优势
- **vs i-MedRAG**：i-MedRAG通过迭代query改写提升（77.96%），与ExpertRAG的subject area过滤路径正交，两者可组合
- **vs Self-BioRAG**：Self-BioRAG在EMSQA上仅55.71%，说明通用生物领域RAG不适用于EMS这种细分专业领域

## 评分

- 新颖性: ⭐⭐⭐ 核心idea直觉（把expertise注入CoT/RAG），但系统构建完整
- 实验充分度: ⭐⭐⭐⭐ 多模型、多RAG baseline对比，含认证考试实测
- 写作质量: ⭐⭐⭐⭐ 图表清晰，数据集构建过程详尽
- 价值: ⭐⭐⭐⭐ 数据集有持续价值，expertise注入框架可推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] HeteroRAG: A Heterogeneous Retrieval-Augmented Generation Framework for Medical Vision Language Tasks](../../ACL2026/medical_nlp/heterorag_a_heterogeneous_retrieval-augmented_generation_framework_for_medical_v.md)
- [\[ACL 2025\] Towards Omni-RAG: Comprehensive Retrieval-Augmented Generation for Large Language Models in Medical Applications](../../ACL2025/medical_nlp/omni_rag_medical.md)
- [\[ACL 2026\] SEMA-RAG: A Self-Evolving Multi-Agent Retrieval-Augmented Generation Framework for Medical Reasoning](../../ACL2026/medical_nlp/sema-rag_a_self-evolving_multi-agent_retrieval-augmented_generation_framework_fo.md)
- [\[ACL 2025\] MedBioRAG: Semantic Search and Retrieval-Augmented Generation with Large Language Models for Medical and Biological QA](../../ACL2025/medical_nlp/medbiorag_semantic_search_and_retrieval-augmented_generation_with_large_language.md)
- [\[ICLR 2026\] MedAraBench: Large-scale Arabic Medical Question Answering Dataset and Benchmark](../../ICLR2026/medical_nlp/medarabench_large-scale_arabic_medical_question_answering_dataset_and_benchmark.md)

</div>

<!-- RELATED:END -->
