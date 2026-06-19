---
title: >-
  [论文解读] A Modular Approach for Clinical SLMs Driven by Synthetic Data with Pre-Instruction Tuning, Model Merging, and Clinical-Tasks Alignment
description: >-
  [ACL 2025][医疗NLP][临床小语言模型] 本文提出一种将小型语言模型（SLM）高效适配为临床领域模型的模块化框架，包含领域专家预指令微调（在医学语料上训练多个专家模型）、模型合并（将多个专家合并为统一的 MediPhi）、以及基于 250 万条合成指令（MediFlow）的临床任务对齐，最终 3.8B 参数的 MediPhi 在多项临床任务上超越 GPT-4。
tags:
  - "ACL 2025"
  - "医疗NLP"
  - "临床小语言模型"
  - "合成数据"
  - "预指令微调"
  - "模型合并"
  - "临床任务对齐"
---

# A Modular Approach for Clinical SLMs Driven by Synthetic Data with Pre-Instruction Tuning, Model Merging, and Clinical-Tasks Alignment

**会议**: ACL 2025  
**arXiv**: [2505.10717](https://arxiv.org/abs/2505.10717)  
**代码**: 无  
**领域**: 医学NLP  
**关键词**: 临床小语言模型, 合成数据, 预指令微调, 模型合并, 临床任务对齐

## 一句话总结

本文提出一种将小型语言模型（SLM）高效适配为临床领域模型的模块化框架，包含领域专家预指令微调（在医学语料上训练多个专家模型）、模型合并（将多个专家合并为统一的 MediPhi）、以及基于 250 万条合成指令（MediFlow）的临床任务对齐，最终 3.8B 参数的 MediPhi 在多项临床任务上超越 GPT-4。

## 研究背景与动机

**领域现状**：GPT-4 等大型语言模型在临床场景中展现出强大的能力，但其高计算成本和延迟限制了在医疗机构中的实际部署。小型语言模型（SLMs，如 3-4B 参数级别）在成本和延迟方面更具优势，但其有限的模型容量使得领域适配更加困难。医疗领域的模型适配面临两大特殊挑战：一是医学知识的专业性和多样性（从放射报告到 ICD 编码，从临床指南到药物信息），二是临床数据的极度稀缺和高度敏感性。

**现有痛点**：现有的医疗 LLM 适配方法通常采用"一刀切"策略——在一个混合的医学语料库上进行统一微调。这种方法对于 SLM 来说效果不佳，因为 SLM 的容量有限，难以在一次训练中同时掌握所有类型的医学知识。此外，高质量的临床标注数据极难获取，现有的临床 NLP 基准测试也覆盖面有限。

**核心矛盾**：SLM 的"小"与临床任务的"广"之间的矛盾——如何让一个仅 3.8B 参数的模型同时胜任命名实体识别、放射报告生成、ICD 编码、临床问答等十几种不同性质的任务。

**本文目标**：（1）设计一个将 SLM 适配为临床模型的系统框架；（2）构建覆盖面广的临床评估基准；（3）构建大规模高质量的合成临床指令数据集。

**切入角度**：借鉴"专家模型 + 合并"的模块化思路——先针对不同医学子领域分别训练多个专家 SLM，然后通过模型合并技术将它们统一为一个模型，最后用合成数据在任务层面对齐，实现"分而治之、合而为一"。

**核心 idea**：通过预指令微调分别训练领域专家模型、通过模型合并统一知识、通过合成数据对齐任务，三阶段流水线将 3.8B 的 Phi-3.5-mini 适配为超越 GPT-4 的临床模型 MediPhi。

## 方法详解

### 整体框架

MediPhi 的构建分为三个阶段：（1）预指令微调阶段——在 PMC（PubMed Central）、医学指南、MedWiki 等不同类型的医学语料上分别对 Phi-3.5-mini 进行继续预训练，得到多个领域专家模型；（2）模型合并阶段——使用模型合并技术（如 TIES-Merging 或 DARE）将多个专家模型的参数合并为一个统一的 MediPhi 基础模型；（3）临床任务对齐阶段——利用 MediFlow 合成数据集通过 SFT 和 DPO 对 MediPhi 进行指令微调和偏好对齐。

### 关键设计

1. **预指令微调（Pre-Instruction Tuning）**:

    - 功能：在通用 SLM 的基础上注入不同类型的医学领域知识，得到多个互补的专家模型
    - 核心思路：选取多个代表性的医学文本语料库——PMC 学术论文（覆盖最新研究知识）、医学指南和教科书（覆盖标准化临床知识）、MedWiki（覆盖通俗化的医学概念解释）等。对每个语料库，独立地在 Phi-3.5-mini 上进行继续预训练（next token prediction）。这一步不使用指令格式的数据，因此称为"预指令"微调。每个专家模型在其对应领域内有更深入的知识，但可能在其他领域表现下降
    - 设计动机：SLM 的容量限制意味着在混合语料上训练会出现"灾难性干扰"——不同领域的知识相互冲突。分别训练专家模型可以让每个模型充分吸收单一类型的知识，避免容量瓶颈

2. **模型合并（Model Merging）**:

    - 功能：将多个专家模型的参数合并为一个统一模型，保留各专家的优势
    - 核心思路：采用参数空间的合并策略，如 TIES-Merging（先识别各专家相对于基模型的关键参数变化，再合并这些变化）或 DARE（随机 drop 部分参数差异以减少冲突）。合并后的 MediPhi 在不需要额外训练的情况下，就能在所有专家对应的领域上保持接近各专家的性能。实验显示，合并后的模型在 CLUE+ 基准上的表现不低于各专家在其强项领域的表现
    - 设计动机：推理时只需加载一个模型而非多个专家，大幅降低部署复杂度。模型合并是一种"免费午餐"——不增加训练或推理成本，就能获得多领域能力

3. **MediFlow 合成数据集与临床任务对齐**:

    - 功能：提供大规模、高质量、覆盖广的临床指令数据用于最终的任务对齐
    - 核心思路：构建了 MediFlow 数据集，包含 250 万条合成指令，涵盖 14 种医学 NLP 任务（命名实体识别、关系抽取、文本分类、摘要生成、问答、ICD 编码等）和 98 种细粒度文档类型（放射报告、出院小结、病历记录等），并支持 JSON 格式输出。数据生成流程：从真实临床文本出发，利用 GPT-4 生成高质量的指令-输出对，经过质量过滤和去重。对齐阶段使用 SFT 在 MediFlow 上微调 MediPhi，再使用 DPO 进一步对齐偏好。CLUE+ 基准将原始 CLUE 基准扩展了一倍，覆盖更多临床任务和场景
    - 设计动机：真实临床标注数据极难获取，大规模合成数据是打破数据瓶颈的关键。14 种任务和 98 种文档类型的覆盖面确保了模型在真实临床场景中的泛化能力

### 损失函数 / 训练策略

预指令微调使用标准的下一个 token 预测损失。SFT 阶段使用标准的交叉熵损失在指令-输出对上训练。DPO 阶段使用直接偏好优化损失来进一步对齐模型输出与期望偏好。三阶段串行执行，每个阶段的学习率和训练轮数独立调整。

## 实验关键数据

### 主实验

| 任务类别 | 指标 | MediPhi-SFT | Phi-3.5-mini | GPT-4-0125 | 相对基线提升 |
|----------|------|-------------|--------------|------------|------------|
| 医学实体识别 | F1 | +64.3% | baseline | - | vs Phi-3.5-mini |
| 放射报告任务 | BLEU/ROUGE | +49.5% | baseline | - | vs Phi-3.5-mini |
| ICD-10 编码 | 准确率 | +44.0% | baseline | GPT-4-14% | 超越 GPT-4 14% |
| CLUE+ 综合 | 平均分 | +18.9% | MediPhi-merge | - | SFT+DPO vs 合并后 |
| 临床问答 | 准确率 | 73.2% | 58.1% | 71.5% | MediPhi > GPT-4 |

### 消融实验

| 配置 | CLUE+ 平均分 | 说明 |
|------|-------------|------|
| 完整 MediPhi（SFT+DPO） | 最优 | 三阶段完整流水线 |
| 仅合并（无 SFT/DPO） | 中等 | 已有不错的领域知识 |
| 单一专家（PMC） | 低于合并 | 仅擅长学术知识 |
| 单一专家（指南） | 低于合并 | 仅擅长临床标准知识 |
| 直接 SFT（无预指令微调） | 低于完整 | 缺少领域基础知识 |
| SFT 无 DPO | 低于完整 | DPO 进一步提升对齐 |
| Phi-3.5-mini 原始 | 最低 | 通用模型无医学适配 |

### 关键发现

- 预指令微调阶段各专家模型在其对应领域有显著提升（医学实体 +64.3%、放射报告 +49.5%），说明领域语料的继续预训练对 SLM 非常有效
- 模型合并成功保留了各专家的优势，合并后的 MediPhi 在跨领域评估中不低于最优专家，证明了"分而治之 + 合并"策略的可行性
- 在 ICD-10 编码任务上超越 GPT-4-0125 达 14%，证明了针对性适配的 3.8B SLM 在特定临床任务上可以打败通用大模型
- MediFlow 的 DPO 对齐在 SFT 基础上平均再提升 18.9%，说明偏好对齐对临床 NLP 的输出质量至关重要

## 亮点与洞察

- "预指令微调 → 模型合并 → 任务对齐"的三阶段流水线设计优雅且通用。这一范式不仅适用于医学领域，任何领域适配场景都可以借鉴——先分领域注入知识，再合并统一，最后任务对齐
- MediFlow 250 万条合成数据集覆盖 14 种任务和 98 种文档类型，是本文最重要的资源贡献之一。这种系统化的合成数据构建方法可以推广到其他数据稀缺的领域
- 模型合并作为"免费午餐"在医学 SLM 中的成功应用令人鼓舞——不需要额外训练成本就能统一多个专家的能力

## 局限与展望

- 预指令微调阶段需要为每种语料类型单独训练一个完整模型，扩展到更多子领域时训练成本线性增长
- MediFlow 合成数据的质量受限于生成它的教师模型（GPT-4），可能引入偏差
- 当前仅在 Phi-3.5-mini（3.8B）上验证，扩展到更大或更小的模型时效果是否一致未知
- CLUE+ 虽然扩展了覆盖面，但仍以英文为主，多语言临床场景未涉及
- 真实医院环境中的部署验证（如处理噪声电子病历、与 HIS 系统集成等）尚未进行

## 相关工作与启发

- **vs Med-PaLM**: Med-PaLM 使用更大的模型（540B），计算成本远高于 MediPhi（3.8B）。MediPhi 证明了通过精心的领域适配，SLM 可以达到接近甚至超越大模型的领域性能
- **vs BioMistral/OpenBioLLM**: 这些医学 LLM 通常在混合医学语料上统一微调，MediPhi 的模块化专家训练 + 合并策略更适合 SLM 的容量限制
- **vs PMC-LLaMA**: PMC-LLaMA 仅在 PMC 文献上训练，覆盖面有限；MediPhi 通过多源语料专家和 MediFlow 对齐实现了更全面的临床能力

## 评分

- 新颖性: ⭐⭐⭐⭐ 模块化框架设计系统性强，"预指令微调+合并+对齐"的三阶段范式有创新
- 实验充分度: ⭐⭐⭐⭐ 在扩展的 CLUE+ 基准上全面评估，覆盖多种临床任务
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，数据集构建过程详细
- 价值: ⭐⭐⭐⭐⭐ 面向实际部署需求的临床 SLM 方案，MediFlow 数据集和 CLUE+ 基准都是重要资源贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] RedactX: An LLM-Powered Framework for Automatic Clinical Data De-Identification](redactor_an_llm-powered_framework_for_automatic_clinical_data_de-identification.md)
- [\[ACL 2026\] CURA: Clinical Uncertainty Risk Alignment for Language Model-Based Risk Prediction](../../ACL2026/medical_nlp/cura_clinical_uncertainty_risk_alignment_for_language_model-based_risk_predictio.md)
- [\[ACL 2025\] Aligning AI Research with the Needs of Clinical Coding Workflows: Eight Recommendations Based on US Data Analysis and Critical Review](clinical_coding_eight_recommendations.md)
- [\[ACL 2026\] PrinciplismQA: A Philosophy-Grounded Approach to Assessing LLM-Human Clinical Medical Ethics Alignment](../../ACL2026/medical_nlp/principlismqa_a_philosophy-grounded_approach_to_assessing_llm-human_clinical_med.md)
- [\[ACL 2025\] ReflecTool: Towards Reflection-Aware Tool-Augmented Clinical Agents](reflectool_clinical_agent.md)

</div>

<!-- RELATED:END -->
