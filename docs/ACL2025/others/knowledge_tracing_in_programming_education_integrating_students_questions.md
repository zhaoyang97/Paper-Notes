---
title: >-
  [论文解读] Knowledge Tracing in Programming Education Integrating Students' Questions
description: >-
  [ACL 2025][知识追踪] 本文提出 SQKT（Students' Question-based Knowledge Tracing）模型，首次将学生提问和自动提取的技能信息整合到知识追踪中，用于预测编程教育中学生对后续编程题的完成情况，域内实验 AUC 提升高达 33.1%。 知识追踪（Knowledge Traci…
tags:
  - "ACL 2025"
  - "知识追踪"
  - "编程教育"
  - "学生提问"
  - "技能提取"
  - "自适应学习"
---

# Knowledge Tracing in Programming Education Integrating Students' Questions

**会议**: ACL 2025  
**arXiv**: [2502.10408](https://arxiv.org/abs/2502.10408)  
**代码**: 无（论文发表后公开）  
**领域**: 其他（教育AI/知识追踪）  
**关键词**: 知识追踪, 编程教育, 学生提问, 技能提取, 自适应学习

## 一句话总结

本文提出 SQKT（Students' Question-based Knowledge Tracing）模型，首次将学生提问和自动提取的技能信息整合到知识追踪中，用于预测编程教育中学生对后续编程题的完成情况，域内实验 AUC 提升高达 33.1%。

## 研究背景与动机

知识追踪（Knowledge Tracing, KT）旨在监测学生的知识状态并预测其未来表现。在编程教育领域，KT 面临独特挑战：

**代码提交的复杂性**：编程任务复杂，同一问题有多种正确解法，需要从非结构化、嘈杂的源码中评估学生能力

**传统 Q-matrix 方法的局限**：手动标注每道题所需技能（knowledge components）既耗时又难以捕捉学生实际使用的全部技能

**学生提问的信息价值被忽视**：学生的提问直接反映了困惑区域和理解深度，比代码提交更能清晰揭示学习状态

随着在线学习平台（如 Moodle、Canvas）的普及，学生提问和师生互动数据变得日益丰富。然而，现有 KT 模型完全忽略了这一宝贵信号。特别是在编程教育中，学生的提问往往能揭示代码提交中难以捕捉的概念理解和推理过程。

## 方法详解

### 整体框架

SQKT 模型接收以下输入序列来预测学生能否正确解决下一道编程题：

- **历史部分**：学生尝试过的每道题的描述、代码提交、学生提问、提取的技能信息
- **目标部分**：下一道题的描述和所需技能

模型架构包含四个核心组件：
1. 多特征输入嵌入层
2. 技能提取系统
3. 融合层
4. 多头自注意力预测层

### 关键设计

**学生提问嵌入（组件 A）**：
- 使用 CodeT5 模型编码学生提问，因其同时理解自然语言和代码语法
- 学生提问包含两类信息：(a) 自然语言问题——澄清概念或策略，(b) 代码问题——针对具体代码行或错误
- 通过辅助任务（生成教育者回复）微调 CodeT5，使提问嵌入捕捉学生困惑要点和教育者会如何回应
- 若无提问则使用零向量

**自动技能提取系统（组件 B）**：
- 定义了 **36 个核心 Python 概念 + 19 种 Python 错误类型**的技能集合
- 包含错误类型的动机：错误揭示学生的理解和误解，与学习差距相关
- 使用 GPT-4o 生成基于规则的技能提取脚本：给 GPT-4o 约 20 个标注示例和预定义技能列表，生成可批量应用的提取脚本
- 选择规则方法而非在线 GPT 调用的原因：高精度和一致性
- 验证：100 个样本上 Precision 0.85, Recall 0.88, F1 0.86，人工标注一致性 Cohen's kappa 0.98

**技能双向应用**：
- 从学生提问中提取学生正在挣扎的技能
- 从参考解答中提取目标题目所需的技能
- 对齐两者以增强预测准确性

**代码嵌入（组件 C）**：使用 CodeBERT 编码学生代码提交

**题目嵌入（组件 D）**：使用 BERT-base 编码题目描述

**融合层（组件 H）**：
- 将所有嵌入投射到统一的 512 维空间
- 使用三元组损失鼓励同一提交的嵌入靠近，不同提交/概念的嵌入远离

### 损失函数 / 训练策略

**三重损失函数**：

$$L_{total} = L_{pred} + L_{question} + \lambda L_{triplet}$$

1. **预测损失 $L_{pred}$**：二元交叉熵，预测学生对目标题的成功/失败
2. **提问辅助损失 $L_{question}$**：负对数似然，微调 CodeT5 生成教育者回复
3. **三元组损失 $L_{triplet}$**：统一异构嵌入空间
    - Anchor：当前题目的代码嵌入
    - Positive：当前题目描述或学生提问嵌入
    - Negative：随机题目描述或提问嵌入

**训练配置**：
- 优化器：Adam，学习率 3e-5
- Batch size：16，Dropout：0.1
- 三元组损失权重 λ = 1.0
- 6 层自注意力层，max-pooling 后接分类头
- GPU：NVIDIA A100 80GB，域内训练约 1.5 小时，跨域训练约 3 小时

## 实验关键数据

### 主实验

**数据集**：韩国在线编程教育平台（2022.1-2024.4），4 门 Python 课程：
- Python Basic (PB)：48 题，160 名学生
- First Python (FP)：60 题，8141 名学生
- Algorithm (Algo)：32 题，77 名学生（数据稀缺）
- Python Introduction (PI)：227 题，1092 名学生

**域内实验结果（AUC%）**：

| 模型 | Python Introduction | First Python | Python Basic |
|------|-------------------|-------------|-------------|
| KTMFF | 70.2 | 69.4 | 78.0 |
| KTMFF+ | 72.6 | 71.7 | 80.7 |
| OKT | 60.3 | 65.8 | 65.0 |
| OKT+ | 66.7 | 66.7 | 78.4 |
| **SQKT** | **93.4** | **90.3** | **93.3** |

SQKT 相比最佳基线 (KTMFF+) AUC 绝对提升 12.6-20.8 个百分点。

### 消融实验

**组件消融（Python Introduction 课程）**：

| 配置 | AUC (%) | ACC (%) | F1 (%) |
|------|---------|---------|--------|
| SQKT 完整 | 93.4 | 89.2 | 88.4 |
| - 提问（全 1 向量） | 91.3 | 86.3 | 89.9 |
| - 提问（仅技能） | 90.9 | 86.2 | 88.7 |
| - 技能（仅提问） | 89.7 | 81.3 | 83.1 |
| - 提问和技能 | 85.4 | 80.7 | 82.7 |

关键发现：
- 使用全 1 向量替代实际提问内容会降低性能 → 提问的具体内容很重要
- 仅技能或仅提问都不如两者结合 → 两种信号有协同效应
- 移除两者后 AUC 降 8 个百分点 → 提问和技能贡献显著

**辅助损失影响**：
- 移除提问损失：3 门课程 AUC 分别下降 0.7/0.5/1.4
- 移除三元组损失：下降 1.0/2.9/1.8（First Python 影响最大）

### 关键发现

1. **学生提问是高质量的预测信号**：即便添加到基线模型中（KTMFF+ vs KTMFF, OKT+ vs OKT）也能带来一致提升
2. **跨域泛化能力强**：内容结构迁移场景中，使用提问数据的模型 AUC 绝对提升 45.3%
3. **解决数据稀缺问题**：在 Algorithm 课程（仅 300 测试样本）上，跨域模型比域内模型高 11.4% AUC
4. **提问内容比提问存在更重要**：全 1 向量（仅指示"有提问"）不如实际提问嵌入有效

**错误分析**：60 个错误预测中：
- 复杂性 (55.6%)：混合语言语法挑战解析
- 混淆 (40.7%)：代码错误与学生提问无关
- 歧义 (22.2%) 和不完整性 (29.6%)：需要更清晰的上下文

## 亮点与洞察

- **首次将学生提问整合到知识追踪**：填补了 KT 领域长期忽视的信息源
- **GPT 驱动的自动化技能提取**：用 GPT-4o 生成规则脚本替代人工标注，兼顾可解释性和可扩展性
- **辅助任务设计精巧**：通过预测教育者回复来增强提问嵌入的信息密度
- **跨域实验设计充分**：两种跨域场景（内容结构迁移 + 数据稀缺泛化）验证了模型的实用性

## 局限与展望

- **未对输入提问做预处理**：实际课堂中的提问通常很嘈杂
- **技能提取器基于规则**：虽然保证了可解释性，但可扩展性受限；混合方法可能更优
- **数据集仅限韩语 + Python**：未验证其他语言和编程语言的泛化性
- **仅考虑文本提问**：未处理截图、图片等多模态提问形式
- **数据规模有限**：Algorithm 课程仅 77 名学生、32 道题

## 相关工作与启发

- 继承了 DKT (Piech et al., 2015) 和 BKT (Corbett and Anderson, 1994) 的知识追踪传统
- 与 Code-DKT (Shi et al., 2022) 相比，不仅利用代码特征还引入了学生提问
- GPT 自动化技能标注为 Q-matrix 方法提供了可扩展替代
- 对自适应学习系统的启示：师生互动数据应被充分利用以个性化学习路径

## 评分

- **创新性**：★★★★☆（提问整合是新颖思路，但整体架构相对标准）
- **实验充分性**：★★★★☆（域内/跨域/消融全面，但数据规模偏小）
- **实用价值**：★★★★☆（对编程教育平台有直接应用价值）
- **写作质量**：★★★★☆（结构清晰，图表丰富）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Automated Knowledge Component Generation and Interpretable Knowledge Tracing in Coding Problems](../../ACL2026/others/automated_knowledge_component_generation_for_interpretable_knowledge_tracing_in_.md)
- [\[ACL 2025\] Towards Comprehensive Argument Analysis in Education: Dataset, Tasks, and Method](towards_comprehensive_argument_analysis_in_education_dataset_tasks_and_method.md)
- [\[ACL 2025\] Generating Plausible Distractors for Multiple-Choice Questions via Student Choice Prediction](distractor_gen_multiple_choice.md)
- [\[ACL 2025\] TROVE: A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Classification](trove_a_challenge_for_finegrained_text.md)
- [\[ACL 2025\] Adaptive Retrieval without Self-Knowledge? Bringing Uncertainty Back Home](adaptive_retrieval_without_self-knowledge_bringing_uncertainty_back_home.md)

</div>

<!-- RELATED:END -->
