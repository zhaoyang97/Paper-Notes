---
title: >-
  [论文解读] Towards Effective and Efficient Continual Pre-training of Large Language Models
description: >-
  [ACL 2025][预训练][持续预训练] 系统性地研究了对Llama-3 (8B)进行持续预训练的数据策略，通过主题级数据混合、困惑度课程学习和高质量合成科学QA数据三大策略，仅用100B token就显著增强了中文能力（C-Eval +8.81）和科学推理能力（MATH +12.00），同时有效保持了原始英语能力。
tags:
  - "ACL 2025"
  - "预训练"
  - "持续预训练"
  - "合成数据"
  - "灾难性遗忘"
  - "数据课程学习"
  - "双语适配"
---

# Towards Effective and Efficient Continual Pre-training of Large Language Models

**会议**: ACL 2025  
**arXiv**: [2407.18743](https://arxiv.org/abs/2407.18743)  
**代码**: [https://github.com/RUC-GSAI/Llama-3-SynE](https://github.com/RUC-GSAI/Llama-3-SynE)  
**领域**: LLM预训练  
**关键词**: 持续预训练, 合成数据, 灾难性遗忘, 数据课程学习, 双语适配

## 一句话总结

系统性地研究了对Llama-3 (8B)进行持续预训练的数据策略，通过主题级数据混合、困惑度课程学习和高质量合成科学QA数据三大策略，仅用100B token就显著增强了中文能力（C-Eval +8.81）和科学推理能力（MATH +12.00），同时有效保持了原始英语能力。

## 研究背景与动机

**领域现状**：大语言模型通过海量预训练获得了强大的通用能力，但在特定场景下仍存在不足。例如Llama-3主要在英语数据上预训练，中文任务表现不佳；作为通用模型，也可能缺乏多学科科学知识。持续预训练（CPT）是增强特定能力的主流方法。

**现有痛点**：CPT中的灾难性遗忘是核心技术挑战——新能力提升的同时，原始能力可能大幅下降。虽然CPT已被广泛使用，但关于数据选择、混合和课程安排的关键训练细节讨论不够充分，特别是如何在有限训练预算下同时发展新能力和维护旧能力。

**核心矛盾**：发展新能力需要大量领域数据，但过多的领域数据会覆盖原有的知识分布，导致遗忘。如何在"学新"与"保旧"之间取得精确平衡？

**本文目标** 在有限计算预算（~100B tokens）下，系统探索持续预训练的数据配方——如何高效增强Llama-3的中文和科学推理能力，同时最大限度保留英语通用能力。

**切入角度**：将CPT过程拆分为两个阶段：双语适配阶段（增强中文）和合成数据增强阶段（增强科学推理），分别设计针对性的数据策略，并利用小模型TinyLlama做代理实验来降低探索成本。

**核心 idea**：通过主题级动态数据混合+困惑度课程学习+大规模合成多学科科学QA数据，实现高效而平衡的持续预训练。

## 方法详解

### 整体框架

训练分两个阶段：(1) 双语适配阶段：在92.5B token上进行中英文混合训练，中英比例2:8，采用主题级数据混合和困惑度课程策略；(2) 合成增强阶段：在7.5B token上融入合成的多学科科学QA数据和代码QA数据，数据比例中:英:合成 = 1:7:2。总训练量约100B token。

### 关键设计

1. **主题级数据混合（Topic-based Data Mixture）**:

    - 功能：在比"数据集级别"更细粒度的"主题级别"上动态调整数据配比
    - 核心思路：基于MMLU/CMMLU定义11个英文和11个中文主题（如数学物理、计算机科学、法律政策等），训练TinyBERT分类器自动标注web pages的主题标签。训练过程中定期在验证集上监测各主题的PPL变化 $\Delta p_i = p_i^{(t)} - p_i^{(t-1)}$，归一化后计算调整系数 $f_i = 1 + \alpha \cdot \delta_{p_i} \cdot w_i$，动态更新各主题的采样比例 $r_i^{(t)} = \frac{r_i^{(t-1)} \cdot f_i}{\sum_j r_j^{(t-1)} \cdot f_j}$
    - 设计动机：数据集级别的混合过于粗粒度，同一数据集中不同主题的知识获取速度可能不同。主题级监控可以更精细地发现哪些主题在退化、哪些已充分学习

2. **困惑度课程学习（PPL-based Data Curriculum）**:

    - 功能：按从易到难的顺序组织中文训练数据
    - 核心思路：用模型自身的PPL分数衡量训练数据的难度等级，将中文数据按PPL从低到高排序，逐步增加训练难度
    - 设计动机：由于Llama-3几乎没有中文预训练数据，直接用复杂中文材料训练会产生较大的分布冲突。从"简单"中文数据开始，提供平滑的过渡，有助于缓解英语性能的灾难性遗忘

3. **多学科科学QA数据合成（Scientific QA Synthesis）**:

    - 功能：生成高质量科学QA对以增强多学科推理能力
    - 核心思路：覆盖9个科学学科（数学、物理、化学、生物、天文学、地球科学、医学、计算机科学、通识教育），从Dolma和C4的科学领域网页中提取内容片段，用Mistral-7B-Instruct生成对应的QA对。QA对以文本形式拼接后加入CPT语料。同样采用ICL方法基于LeetCode生成代码QA数据以保持编程能力
    - 设计动机：真实科学QA数据稀缺，合成数据能更好地提取web页面中的关键知识、减少无关信息干扰，且QA格式与下游任务更接近

### 损失函数 / 训练策略

使用标准的语言建模next-token prediction损失。优化器为AdamW（$\beta_1=0.9$, $\beta_2=0.95$），学习率调度采用WSD（Warmup-Stable-Decay），warmup阶段用10B token从 $1\times10^{-7}$ 线性增到 $1\times10^{-5}$，之后保持稳定。BFloat16混精度训练，梯度裁剪1.0，最大上下文长度8192 token。使用Flash Attention和DeepSpeed ZeRO Stage 2优化效率。

## 实验关键数据

### 主实验

| 模型 | MMLU | C-Eval | CMMLU | MATH | GSM8K | HumanEval | MBPP |
|--------|------|------|------|------|------|------|------|
| Llama-3-8B | 66.60 | 49.43 | 51.03 | 16.20 | 54.40 | 36.59 | 47.00 |
| Llama-3-SynE (本文) | 65.19 | **58.24** | **57.34** | **28.20** | **60.80** | **42.07** | 45.60 |
| MAmmoTH2-8B | 64.89 | 46.56 | 45.90 | 34.10 | 61.70 | 17.68 | 38.80 |
| Llama-3-Chinese-8B | 64.10 | 50.14 | 51.20 | 3.60 | 0.80 | 9.76 | 14.80 |

### 代理实验关键发现（TinyLlama）

| 实验 | 关键发现 |
|------|---------|
| 合成数据有效性 | 1B合成数据+4B正常数据 > 5B纯正常数据（科学基准平均+2.5%） |
| 合成数据质量 | 30%以下的错误率对性能影响很小，>50%才显著下降 |
| 合成数据比例 | 20%是最佳混合比例，过高（40%）反而下降 |
| 数据课程 | 从易到难（LH）优于从难到易（HL）和随机采样 |
| 按学科分离 | 刻意按学科分离训练反而损害性能 |

### 关键发现

- C-Eval提升8.81分、CMMLU提升6.31分，验证了双语适配策略的有效性
- MATH提升12.00分、SciEval提升4.13分，证明合成科学QA数据的价值
- MMLU仅下降1.41分，说明遗忘被有效控制
- GaoKao生物子任务提升25.71分（43.81→69.52），中文科学推理提升最为显著
- 即使合成数据存在30%的错误率，对模型性能的影响也很小，说明LLM对合成数据噪声有较好的容忍度
- 编程能力（HumanEval +5.48）也有提升，得益于合成代码QA数据的保护

## 亮点与洞察

- **代理模型策略非常实用**：先在TinyLlama上系统探索数据策略的各个维度（有效性、质量、比例、课程），再将最佳配方应用到Llama-3上，大幅降低了实验成本。这种方法论可以迁移到任何大模型CPT场景
- **主题级动态数据混合**比传统的数据集级混合更精细，通过PPL监控实现自动化调整，是一种可落地的工程实践
- 合成数据不需要完美——30%错误率下性能几乎不受影响，这为大规模低成本合成数据的使用提供了信心
- 课程学习中"刻意按学科分开"反而有害，说明跨学科知识的交叉学习比隔离学习更有效

## 局限与展望

- 英语MMLU仍有轻微下降（-1.41），完全消除遗忘仍具挑战
- 合成数据由Mistral-7B生成，质量上限受限于生成模型。用更强模型生成或引入人工质量控制可能带来进一步提升
- 主题分类器基于简单的TinyBERT，分类精度有限，可能影响数据混合的精确性
- 100B token的训练仍需要显著的计算资源，未探索更极端的低预算情况

## 相关工作与启发

- **vs Llama-3-Chinese-8B**: 该模型同样基于Llama-3做中文适配，但严重损害了数学和代码能力（MATH 3.60, HumanEval 9.76），说明缺乏合成数据保护会导致灾难性遗忘
- **vs MAmmoTH2-8B**: 在科学英文基准上稍弱，但在中文和代码上大幅领先，且MAmmoTH2在中文基准上反而退化（C-Eval 46.56 vs Llama-3的49.43），验证了多能力平衡的重要性
- **vs Galactica-6.7B**: 科学专用模型在通用任务上表现极差（MMLU 37.13），而本文方法实现了通用+科学的平衡

## 评分

- 新颖性: ⭐⭐⭐ 各个技术组件（课程学习、数据混合、合成数据）单独看不算新，但系统性整合和细致的探索值得肯定
- 实验充分度: ⭐⭐⭐⭐⭐ 代理实验+主实验的体系极为详尽，合成数据质量/比例/课程的控制变量实验很有价值
- 写作质量: ⭐⭐⭐⭐ 技术报告风格，条理清晰，实验描述详尽，但创新叙事偏弱
- 价值: ⭐⭐⭐⭐ 提供了可复现的CPT最佳实践指南，对实际应用有直接参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] AsyncLM: Efficient and Adaptive Async Pre-training of Language Models](asynclm_efficient_and_adaptive_async_pre-training_of_language_models.md)
- [\[ACL 2025\] Improving Continual Pre-training Through Seamless Data Packing](improving_continual_pre-training_through_seamless_data_packing.md)
- [\[ACL 2025\] Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation](emergent_abilities_continued_pt.md)
- [\[ACL 2025\] Velocitune: A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training](velocitune_a_velocity-based_dynamic_domain_reweighting_method_for_continual_pre-.md)
- [\[ACL 2025\] How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training](how_do_llms_acquire_new_knowledge_a_knowledge_circuits_perspective_on_continual_.md)

</div>

<!-- RELATED:END -->
