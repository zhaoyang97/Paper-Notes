---
title: >-
  [论文解读] Automated Structured Radiology Report Generation
description: >-
  [ACL 2025][医疗NLP][放射学报告生成] 提出结构化放射学报告生成（SRRG）新任务，利用LLM将自由文本报告重构为标准化格式，同时引入55标签的SRR-BERT疾病分类模型和F1-SRR-BERT评估指标，解决传统报告生成中风格多样导致的生成与评估困难。 自动化胸部X光（CXR）报告生成是一项重要的医学NLG任…
tags:
  - "ACL 2025"
  - "医疗NLP"
  - "放射学报告生成"
  - "结构化报告"
  - "疾病分类"
  - "胸部X光"
  - "评估指标"
---

# Automated Structured Radiology Report Generation

**会议**: ACL 2025  
**arXiv**: [2505.24223](https://arxiv.org/abs/2505.24223)  
**代码**: [huggingface.co/StanfordAIMI](https://huggingface.co/StanfordAIMI)  
**领域**: 医学NLP  
**关键词**: 放射学报告生成, 结构化报告, 疾病分类, 胸部X光, 评估指标

## 一句话总结

提出结构化放射学报告生成（SRRG）新任务，利用LLM将自由文本报告重构为标准化格式，同时引入55标签的SRR-BERT疾病分类模型和F1-SRR-BERT评估指标，解决传统报告生成中风格多样导致的生成与评估困难。

## 研究背景与动机

自动化胸部X光（CXR）报告生成是一项重要的医学NLG任务，能够减轻放射科医生的工作负担。目前主要的两个数据集MIMIC-CXR和CheXpert Plus均由自由文本报告组成，报告风格高度可变且缺乏结构化，这带来了两方面的挑战：

**生成困难**：自由文本报告的多样性使模型难以产生一致、临床有意义的报告

**评估困难**：现有评估指标（BLEU、ROUGE等NLG指标和F1-RadGraph等临床指标）难以准确捕捉放射学解读的细微差异，因为同一发现可能有多种不同的表述方式

与此同时，临床上也一直有呼吁使用更一致、结构化的放射学报告。这一现实需求和技术困境共同促使作者提出SRRG任务——将自由文本报告重构为标准化格式，并配套更精确的评估方法。

## 方法详解

### 整体框架

SRRG的工作包含三个核心贡献：(1) 定义结构化报告规范并利用LLM创建大规模结构化报告数据集；(2) 训练SRR-BERT细粒度疾病分类模型；(3) 提出F1-SRR-BERT评估指标。整体形成了从数据、模型到评估的完整体系。

### 关键设计

1. **结构化报告规范（Desiderata）**: 定义了严格的报告格式标准：

    - 报告由Exam Type、History、Technique、Comparison、Findings、Impression六个部分组成
    - Findings部分按预定义解剖学标题组织：Lungs and Airways、Pleura、Cardiovascular、Hila and Mediastinum、Tubes/Catheters/Support Devices、Musculoskeletal and Chest Wall、Abdominal、Other
    - Impression部分按临床重要性从高到低编号列出关键发现
    - 严格排除历史比较、可识别信息（日期、姓名、机构等），仅保留患者性别和年龄

2. **数据集构建**: 

    - 利用GPT-4 Turbo将MIMIC-CXR和CheXpert Plus的自由文本报告重构为结构化格式
    - SRRG-Findings包含184,542条（训练集181,874）
    - SRRG-Impression包含409,927条（训练集405,972）
    - 由5位执业放射科医生对464份报告进行人工审阅验证
    - 两个数据集的映射分别是：X光→Findings 和 X光→Impression

3. **SRR-BERT疾病分类模型（55标签）**:

    - 在CheXbert的14标签基础上扩展到55个疾病标签，覆盖更精细的肺部、胸膜、心脏、纵隔、肌骨及腹部发现
    - 每个发现映射到0个、1个或多个疾病标签
    - 每个疾病赋予三种状态：Present（存在）、Absent（不存在）、Uncertain（不确定）
    - 数据标注采用三模型投票：GPT-4 Turbo、GPT-4 Turbo 1106 Preview和GPT-4o分别标注，取至少两个模型一致的结果
    - 基于CXR-BERT微调，共标注1,506,158条有效语句

4. **F1-SRR-BERT评估指标**: 

    - 利用SRR-BERT对生成报告和参考报告分别进行疾病预测，计算F1分数
    - 提供两个粒度：leaves级（55标签最细粒度）和upper级（25个更粗的类别）
    - 支持aligned（按顺序对齐评估）和unaligned（按集合方式评估）两种模式
    - aligned模式可评估模型是否按临床重要性排序

### 损失函数 / 训练策略

SRR-BERT使用CXR-BERT作为预训练骨干，在StructUtterances数据集上进行弱监督微调。标注数据包含1,506,158条语句和1,782,983个标签。训练分为四种配置：leaves、upper、leaves with statuses、upper with statuses，分别训练独立模型。

## 实验关键数据

### 主实验

**疾病分类性能：**

| 模型配置 | Micro F1 | Macro F1 | Weighted F1 |
|---------|----------|----------|-------------|
| SRR-BERT (Leaves) | 0.84 | 0.55 | 0.82 |
| SRR-BERT (Upper) | 0.84 | 0.65 | 0.83 |
| SRR-BERT (Leaves+Statuses) | 0.80 | 0.28 | 0.77 |
| SRR-BERT (Upper+Statuses) | 0.80 | 0.38 | 0.78 |

**与CheXbert对比（映射到14类）：**

| 输入类型 | CheXbert F1 | SRR-BERT F1 | 说明 |
|---------|-------------|-------------|------|
| 结构化语句 (Leaves映射) | 0.65 | 0.84 | SRR-BERT +19% |
| 结构化语句 (Upper映射) | 0.50 | 0.86 | SRR-BERT +36% |
| 完整报告 (Upper映射) | 0.56 | 0.70 | SRR-BERT仍优 |

**报告生成模型基准（SRRG-Impression unaligned, Test）：**

| 模型 | BLEU | ROUGE-L | F1-RadGraph | F1-SRR-BERT |
|------|------|---------|-------------|-------------|
| CheXpert-Plus | 14.84 | 28.01 | 22.14 | 46.48 |
| MAIRA-2 | 8.12 | 27.82 | 20.37 | 50.36 |
| CheXagent | 6.95 | 27.18 | 19.70 | 50.63 |
| RaDialog | 3.32 | 21.59 | 12.32 | 39.22 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Unaligned评估 | BLEU 14.84 | 不考虑顺序的宽松评估 |
| Aligned评估 | BLEU 3.78 | 考虑顺序后大幅下降约11分 |
| Findings任务 | BLEU ~3.5 | 比Impression任务更具挑战 |
| Category预测 | F1 ~77% | 解剖分区预测较准确 |

### 关键发现

- Findings生成比Impression生成更具挑战性：传统指标分数显著更低
- Aligned评估比Unaligned评估更严格：CheXpert-Plus在SRRG-Impression上BLEU从14.84降至3.78
- SRR-BERT在所有对比设置中均显著优于CheXbert，验证了55标签细粒度分类的有效性
- 即使使用非结构化完整报告作为输入，SRR-BERT仍能保持较好性能
- 各模型的Category预测准确率约75-78%，说明解剖结构的正确归类是可实现的
- CheXagent在Recall上表现突出，而CheXpert-Plus在传统指标上领先

## 亮点与洞察

- **任务定义有创意**：将非结构化报告→结构化报告的转换定义为新任务，既符合临床需求又便于自动评估
- **评估指标设计精巧**：F1-SRR-BERT结合了层级疾病分类体系和对齐/非对齐两种评估模式，弥补了传统NLG指标在医学领域的不足
- **数据集规模大**：基于MIMIC-CXR和CheXpert Plus构建，总计近60万条结构化报告
- **临床验证充分**：5位执业放射科医生参与审阅，增强了结果的临床可信度
- **55标签覆盖全面**：从14标签到55标签的扩展大幅提升了疾病分类的细粒度

## 局限与展望

- 结构化重写依赖GPT-4，可能引入LLM特有的幻觉或信息丢失
- 标签空间存在部分模糊区域（如"Air space opacity"类别的F1仅0.62）
- Macro F1分数较低（leaves仅0.55），表明稀有标签的分类仍需改进
- 仅关注胸部X光，未扩展到其他影像类型（CT、MRI等）
- 未探索端到端结构化报告生成模型的训练
- 结构化报告的临床实用性需要更大规模的前瞻性临床验证

## 相关工作与启发

- CheXbert是经典的14标签疾病分类模型，本文将其扩展到55标签
- F1-RadGraph（Delbrouck et al., 2022）基于知识图谱评估报告质量，本文的F1-SRR-BERT提供了互补的细粒度评估
- GREEN（Ostmeier et al., 2024）和RadFact（Bannur et al., 2024）关注临床事实性评估
- MAIRA-2（Bannur et al., 2024）是当前领先的报告生成模型之一
- 结构化报告的思路可推广到其他医学报告生成任务（病理、超声等）

## 评分

- 新颖性: ⭐⭐⭐⭐ 结构化报告生成是有价值的新任务定义，SRR-BERT和F1-SRR-BERT设计精良
- 实验充分度: ⭐⭐⭐⭐ 多模型基准测试、详细的分类对比、人工审阅验证，但缺乏端到端训练实验
- 写作质量: ⭐⭐⭐⭐ 论文结构完整，数据集统计详尽，但部分表格较密集
- 价值: ⭐⭐⭐⭐ 为放射学报告生成提供了标准化框架和更好的评估工具，具有实际临床意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Online Iterative Self-Alignment for Radiology Report Generation](oisa_radiology_report_gen.md)
- [\[ACL 2025\] Radar: Enhancing Radiology Report Generation with Supplementary Knowledge Injection](radar_radiology_report_gen.md)
- [\[ACL 2025\] The Impact of Auxiliary Patient Data on Automated Chest X-Ray Report Generation and How to Incorporate It](auxiliary_patient_data_xray.md)
- [\[ACL 2025\] CSTRL: Context-Driven Sequential Transfer Learning for Abstractive Radiology Report Summarization](cstrl_context-driven_sequential_transfer_learning_for_abstractive_radiology_repo.md)
- [\[ACL 2026\] MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation](../../ACL2026/medical_nlp/march_multi-agent_radiology_clinical_hierarchy_for_ct_report_generation.md)

</div>

<!-- RELATED:END -->
