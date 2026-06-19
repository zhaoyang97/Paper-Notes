---
title: >-
  [论文解读] Structure-aware Domain Knowledge Injection for Large Language Models
description: >-
  [ACL 2025][知识编辑][领域知识注入] 提出 StructTuning 方法，通过自动提取训练语料的知识分类结构，设计结构感知的持续预训练（SCPT）和结构感知的监督微调（SSFT）两阶段策略，仅用传统方法5%的数据量即可达到100%的领域知识注入效果。 领域现状：将通用LLM适配为领域专家是当前的热门方向…
tags:
  - "ACL 2025"
  - "知识编辑"
  - "领域知识注入"
  - "结构化学习"
  - "持续预训练"
  - "知识分类体系"
  - "数据高效"
---

# Structure-aware Domain Knowledge Injection for Large Language Models

**会议**: ACL 2025  
**arXiv**: [2407.16724](https://arxiv.org/abs/2407.16724)  
**代码**: [https://github.com/alibaba/struxgpt](https://github.com/alibaba/struxgpt)  
**领域**: 知识编辑  
**关键词**: 领域知识注入, 结构化学习, 持续预训练, 知识分类体系, 数据高效

## 一句话总结

提出 StructTuning 方法，通过自动提取训练语料的知识分类结构，设计结构感知的持续预训练（SCPT）和结构感知的监督微调（SSFT）两阶段策略，仅用传统方法5%的数据量即可达到100%的领域知识注入效果。

## 研究背景与动机

**领域现状**：将通用LLM适配为领域专家是当前的热门方向，主要有两条路线：RAG（检索增强生成）和领域知识注入（持续预训练+监督微调）。RAG在需要逻辑推理、且用户查询与知识库存在语义鸿沟时效果不佳。而传统的持续预训练（CPT）通常需要数十亿token的海量数据，如MMedLM使用25.5B token进行医学领域适配。

**现有痛点**：传统CPT将训练语料随机拼接后按固定长度分块，完全忽略了文本中固有的知识结构（如教科书的目录体系），导致LLM只能从"数据多样性"中浅层吸收知识，而非像人类学生那样通过"结构化学习"系统掌握领域知识。

**核心矛盾**：人类教育是结构化的——学生按照教材章节顺序学习、复习知识点、通过练习应用知识。但LLM的训练过程缺乏这种结构性：CPT阶段随机切块丢失了知识结构，SFT阶段的QA对也没有引导模型利用结构化知识来回答问题。

**本文目标** 如何让LLM像人类学生一样，通过结构化学习方式高效地注入领域知识，大幅减少所需训练数据量。

**切入角度**：模拟人类教育流程——先按"教材章节"学习，将知识点关联到知识体系中；再通过"练习题"学会调用结构化知识解决实际问题。

**核心 idea**：自动提取语料的知识分类树，在CPT中让模型在知识路径条件下生成内容，在SFT中显式引导模型沿知识路径推理回答问题。

## 方法详解

### 整体框架

StructTuning包含三个核心步骤：(1) 知识结构提取：用LLM自动从原始语料中提取分类树状结构；(2) SCPT：将每个训练chunk关联到知识树中的路径节点，在知识路径条件下训练语言模型，并定期让模型回忆完整知识树；(3) SSFT：根据知识树生成包含推理路径的QA训练数据，教模型如何沿着知识结构回答问题。

### 关键设计

1. **知识结构提取（Knowledge Structure Extraction）**:

    - 功能：从原始语料中自动恢复层级分类结构
    - 核心思路：先用spaCy将文本按段落切句并合并为固定长度的chunk，再用Llama3-70B为每个chunk生成摘要标题（作为"知识点"），最后将标题列表输入专门训练的7B模型来识别层级结构（以mindmap形式呈现）
    - 设计动机：整个流程无需人工标注，可扩展到大规模语料。实验证明7B专用模型已能提取足够精确的结构，更强大的GPT-3.5/Llama3-70B并不能带来显著提升

2. **结构感知持续预训练（SCPT）**:

    - 功能：让模型在知识结构条件下学习文本内容
    - 核心思路：将知识mindmap转为自然语言模板，prepend到每个训练chunk前面，模型需在知识路径 $s^k$ 的条件下预测chunk内容 $x^k$，即 $p(x^k|s^k) = \prod_i p(x_i^k | x_{<i}^k, s^k)$。知识路径部分不参与loss计算。使用20种GPT-4生成的多样化模板。训练中每轮遍历完所有知识点后，让模型回忆完整知识层级 $p(\bar{s}) = \prod_k p(s^k)$
    - 设计动机：条件化建模迫使模型将碎片化的文本内容与完整的知识体系关联起来，"复习"环节确保模型记住整体结构

3. **结构感知监督微调（SSFT）**:

    - 功能：教模型利用结构化知识来回答实际问题
    - 核心思路：用随机游走从知识树中采样1到l条知识路径。单路径：利用对应文本生成知识密集型QA对；多路径：基于多个知识点生成需要多跳推理的QA对。每个QA同时生成两个版本——原始版和带CoT的版本（在答案前prepend知识mindmap路径），用两种数据混合训练
    - 设计动机：显式引导模型在回答时先"定位"相关知识路径再推理，形成"检索→推理→回答"的结构化思维模式

### 损失函数 / 训练策略

SCPT使用标准语言建模损失但仅计算content部分（知识路径前缀不计算loss），学习率2e-5，训练3 epochs。SSFT在合成QA数据上训练1 epoch避免过拟合。模型基于Llama2-7B、Llama2-13B、Llama3-8B、InternLM2-7B等多种基座验证。

## 实验关键数据

### 主实验（MMedBench多选题准确率）

| 模型 | 方法 | 平均准确率 | 数据量 | 提升 |
|--------|------|------|------|------|
| Llama3-8B | 基线 | 62.79 | - | - |
| Llama3-8B | MMed (SOTA) | 67.75 | 25.5B tokens | +4.96 |
| Llama3-8B | StructTuning | 65.36 | 76M tokens (0.3%) | +2.57 |
| Llama3-8B | StructTuning | 67.74 | 1.2B tokens (5%) | +4.95 |

### 消融实验（Llama2-7B, MMedBench英语子集）

| 配置 | 英语准确率 | 平均准确率 | 说明 |
|------|---------|------|------|
| SFT only | 44.54 | 35.91 | 仅微调无预训练 |
| CPT + SFT | 46.27 | 35.49 | 传统知识注入 |
| SCPT + SFT | 46.50 | 35.13 | 结构化预训练+普通微调 |
| SCPT + SSFT | 49.96 | 35.78 | 完整StructTuning |
| SCPT + SSFT* (+ 8K QA) | 49.10 | 38.27 | 增加合成QA后跨语言迁移显著 |
| RAG | 38.12 | 33.95 | 检索增强反而更差 |

### 关键发现

- 仅用0.3%数据（76M tokens）就达到了SOTA方法（25.5B tokens）50%以上的效果；5%数据即可接近100%效果
- SSFT相比传统SFT的提升幅度大于SCPT相比CPT的提升幅度，说明教模型"如何使用知识"比"注入知识"更重要
- 结构感知训练出现了有趣的跨语言知识迁移现象：用英语教科书训练后，其他5种语言的准确率也显著提升
- RAG在此场景下效果最差（33.95%），因为教科书语言与实际诊断问题之间存在较大的语义鸿沟
- 方法在InternLM2-7B（+4.46%）、Llama2-7B（+8.78%）、Llama2-13B（+6.17%）上均有显著提升，架构和规模泛化性好

## 亮点与洞察

- **数据效率惊人**：用0.3%的数据实现50%的效果、5%实现100%，这说明传统CPT中绝大部分数据的训练是"无目的"的知识吸收，结构化组织可以大幅提升学习效率。这个洞察可迁移到任何需要领域适配的场景
- **SSFT中的知识路径引导**类似于显式的chain-of-thought推理，但不是让模型自由思考，而是沿着领域知识结构进行"结构化思考"。这种将领域知识图谱嵌入模型推理过程的思路很有启发性
- 拟合的scaling law曲线 $p_s \approx -1.11(\log r)^2 + 7.63 \log r + 133.0$ 成功预测了5%数据点的性能，提供了刻画结构化知识注入数据效率的定量工具

## 局限与展望

- 知识结构提取依赖LLM，对于非教科书类的混杂文本（如网络爬虫数据）提取质量可能下降
- 训练流程增加了额外的计算开销：需要提取分类结构、生成QA对、使用Llama3-70B推理
- SCPT中知识路径前缀会增加上下文长度，占用有限的context window
- 当前主要在医学和long-document QA上验证，尚未测试代码、数学等更多领域

## 相关工作与启发

- **vs AdaptLLM**: 该方法在每个CPT chunk后附加阅读理解QA，但仅chunk级别的增强无法帮助模型理解全局知识结构，MMedBench英语子集上仅提升0.52%（46.79% vs 46.27%）
- **vs RAFT**: 检索增强的微调方法引入过多不相关chunk，反而伤害了QA判断能力（43.60% vs 46.27%）
- **vs RAG**: 在教科书→诊断问题的场景中，RAG因为查询与知识库的语义鸿沟表现最差，而知识注入方法通过将知识内化到模型参数中更有优势

## 评分

- 新颖性: ⭐⭐⭐⭐ 将人类教育模式（结构化学习→练习→复习）系统地映射到LLM训练流程，思路清晰新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型架构多规模、消融全面、scalability分析深入、方法对比详尽
- 写作质量: ⭐⭐⭐⭐ 图表丰富，类比人类教育的叙事方式易于理解，但部分细节在附录中
- 价值: ⭐⭐⭐⭐⭐ 解决了领域适配中数据效率低下的核心问题，实际应用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Neuron-Level Sequential Editing for Large Language Models](neuron-level_sequential_editing_for_large_language_models.md)
- [\[ACL 2025\] Context-Robust Knowledge Editing for Language Models](context-robust_knowledge_editing_for_language_models.md)
- [\[ACL 2025\] Memorizing is Not Enough: Deep Knowledge Injection Through Reasoning](memorizing_is_not_enough_deep_knowledge_injection_through_reasoning.md)
- [\[NeurIPS 2025\] UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models](../../NeurIPS2025/knowledge_editing/uniedit_a_unified_knowledge_editing_benchmark_for_large_language_models.md)
- [\[ACL 2025\] A General Knowledge Injection Framework for ICD Coding](a_general_knowledge_injection_framework_for_icd_coding.md)

</div>

<!-- RELATED:END -->
