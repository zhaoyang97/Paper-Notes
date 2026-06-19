---
title: >-
  [论文解读] M2rc-Eval: Massively Multilingual Repository-level Code Completion Evaluation
description: >-
  [ACL 2025][多语言/翻译][代码补全] 提出覆盖18种编程语言的大规模多语言仓库级代码补全基准 M2rc-Eval，配合基于 AST 的桶级和语义级细粒度标注，并构建 M2rc-Instruct 指令语料以提升模型性能。 仓库级代码补全（Repository-level Code Completion）是软件工程中…
tags:
  - "ACL 2025"
  - "多语言/翻译"
  - "代码补全"
  - "仓库级代码"
  - "多语言评估"
  - "抽象语法树"
  - "细粒度标注"
---

# M2rc-Eval: Massively Multilingual Repository-level Code Completion Evaluation

**会议**: ACL 2025  
**arXiv**: [2410.21157](https://arxiv.org/abs/2410.21157)  
**代码**: [github.com/M2RC-Eval-Team/M2RC-Eval](https://github.com/M2RC-Eval-Team/M2RC-Eval)  
**领域**: 多语言翻译  
**关键词**: 代码补全, 仓库级代码, 多语言评估, 抽象语法树, 细粒度标注

## 一句话总结

提出覆盖18种编程语言的大规模多语言仓库级代码补全基准 M2rc-Eval，配合基于 AST 的桶级和语义级细粒度标注，并构建 M2rc-Instruct 指令语料以提升模型性能。

## 研究背景与动机

仓库级代码补全（Repository-level Code Completion）是软件工程中受到广泛关注的任务，与简单的文件内补全不同，需要理解跨文件的上下文依赖关系。现有基准存在以下不足：

**语言覆盖不足**：主流基准如 CrossCodeEval 仅覆盖4种语言（Python、Java、TypeScript、C#），RepoBench 仅2种，无法全面评估代码 LLM 的跨语言能力

**缺乏细粒度分析**：现有基准通常只报告所有语言的平均分数，忽略了不同补全场景下的细粒度能力差异

**难度分类粗糙**：大多方法仅考虑涉及的文件数量，忽视代码在项目中的结构和语义上下文

上述问题在评估代码 LLM 的多语言仓库级代码补全能力时造成了显著盲区，亟需一个覆盖更广、标注更细的评估体系。

## 方法详解

### 整体框架

M2rc-Eval 系统由三个核心组件构成：

1. **M2rc-Eval 基准**：覆盖18种编程语言（C、C#、C++、Go、HTML、Haskell、Java、JavaScript、Kotlin、Lua、Objective-C、PHP、Python、R、Ruby、Rust、Scala、TypeScript），每种语言100个验证样本和500个测试样本
2. **细粒度标注体系**：基于 AST（抽象语法树）提供桶级和语义级两类标注
3. **M2rc-Instruct 指令语料**：每种语言50,000个文件，用于微调提升补全性能

### 关键设计

**数据收集与质量控制**：
- 数据来源于 The Stack v2（许可证合规的 GitHub 仓库），保留星标 >5 且包含 10-50 个文件的仓库
- 从 431,353,244 个文件中构建整体数据池
- 补全光标位置通过解析 AST 后随机选择节点确定，而非随机选取字符串，保证标识符和语句的完整性

**质量过滤规则**：
- 补全光标位置不超过5行
- 若补全真值少于20字符，至少20%为字母字符
- M2rc-Eval 中的仓库不出现在 M2rc-Instruct 中
- 30%的补全真值不少于2行
- 过滤掉 DeepSeekCoder-1.3B 无需跨文件上下文即能精确预测的样本

**桶级标注（Bucket-level）**：
- 将 AST 的 N 层划分为 M=10 个桶
- 第 i 层属于第 $\lceil i/(N/M) \rceil$ 个桶
- 反映补全的深度难度：越深层（越高桶号）代表代码结构越复杂

**语义级标注（Semantic-level）**：
- 预定义11个主要语义类别：程序结构、声明与定义、控制流结构、表达式、数据类型、语句、修饰符与属性、注释与文档、预处理指令、标识符与作用域、特殊语言结构
- 不同语言有特定的子类别设计
- 基于 Tree-sitter 解析器的语法标签映射到语义标签

**跨文件上下文检索**：
- 遵循 CrossCodeEval 的方法，从同一仓库中提取 L 行连续代码段
- 基于 Jaccard 相似度排序
- 按相关性降序拼接到文件内上下文，直到达到4096 token 上限

### 损失函数 / 训练策略

M2rc-Instruct 的微调采用标准的代码补全目标——给定文件内上下文和跨文件上下文，预测补全真值。训练使用 fill-in-the-middle（FIM）范式，与 Code Llama、DeepSeek-Coder 等模型的预训练策略一致。

## 实验关键数据

### 主实验

在三个代码 LLM（Code Llama-7B、StarCoder-7B、DeepSeekCoder-6.7B）上评估，使用 EM（精确匹配）和 ES（编辑相似度）指标：

**Baseline（仅文件内上下文）→ +Retrieval → +Retrieval & Tuning 的18语言平均结果**：

| 模型 | Baseline EM | +Retrieval EM | +Tuning EM |
|------|-----------|-------------|----------|
| Code Llama-7B | 19.4% | ~21% | ~43% |
| StarCoder-7B | ~20% | ~23% | ~46% |
| DeepSeekCoder-6.7B | ~22% | ~26% | **~48%** |

关键观察：
- 跨文件检索增强带来约2-4%的 EM 提升
- M2rc-Instruct 微调带来约20-25%的 EM 提升，效果显著
- DeepSeekCoder-6.7B 在微调后表现最佳

**与现有基准比较**：

| 基准 | 语言数 | 细粒度标注 | 训练集 | 测试仓库数 |
|------|-------|----------|-------|----------|
| RepoBench | 2 | ✗ | ✓ | 1669 |
| CrossCodeEval | 4 | ✗ | ✗ | 1002 |
| R2C2-Bench | 4 | ✗ | ✓ | 1353 |
| **M2rc-Eval** | **18** | **✓** | **✓** | **5993** |

### 消融实验

**桶级难度分析**：
- 随着桶号增加（AST 深度增加），补全难度逐渐增加
- 浅层节点（桶1-3）的补全 EM 约50%+，深层节点（桶8-10）降至约30%
- 微调对深层节点的提升更为显著

**语义级分析**：
- 不同编程语言在相同语义类别上表现差异较大
- 声明与定义类的补全表现最好
- 控制流结构和特殊语言结构的补全相对困难
- 不同语言有各自的难点，如 Haskell 在程序结构上困难，Go 在控制流上相对容易

### 关键发现

1. M2rc-Instruct 微调在所有18种语言上均带来显著提升，平均 EM 提升超过20个百分点
2. 跨文件上下文检索对补全性能的提升有限（约2-4% EM），但结合微调后效果放大
3. 不同编程语言的跨文件依赖复杂度差异大，HTML 最低，Scala 和 C++ 最高
4. 桶级标注有效区分了不同难度等级的补全场景
5. 语义级标注揭示了模型在不同代码结构上的能力差异，为模型改进提供方向

## 亮点与洞察

1. **AST 驱动的补全位置选择**：相比随机选取字符串，基于 AST 节点选择保证了补全的语义完整性
2. **双层细粒度标注体系**：桶级（代码深度难度）和语义级（代码语义类别）提供了多维度的分析能力
3. **质量控制中的"反向过滤"**：去除小模型无需跨文件上下文即能预测的样本，确保测试集真正考察仓库级能力
4. **覆盖面广**：18种编程语言覆盖了主流和部分小众语言，填补了多语言代码补全评估的空白
5. **实用价值**：M2rc-Instruct 提供了可直接使用的多语言仓库级代码指令数据

## 局限与展望

1. 补全光标位置限制为不超过5行，可能低估长序列补全的难度
2. 仅使用了文本匹配指标（EM 和 ES），未考虑语义等价性（如功能等价但写法不同的代码）
3. Tree-sitter 的语法标签非常细粒度，到语义标签的映射可能存在信息丢失
4. 仅评估了7B 级别的模型，未涵盖更大规模或更新的代码 LLM
5. 跨文件检索策略较为简单（Jaccard 相似度），可探索更先进的检索方法

## 相关工作与启发

- **CrossCodeEval**：4语言仓库级补全基准，本文在语言覆盖和细粒度标注上大幅扩展
- **RepoBench**：提供了仓库级补全的检索增强范式，本文沿用并扩展
- **MultiPL-E / McEval**：多语言文件内代码生成基准，本文将多语言思路推广到仓库级
- **R2C2-Bench**：4语言基准，本文在规模和标注维度上全面超越
- 启发：细粒度的评估标注体系对于理解模型能力瓶颈至关重要

## 评分

- **创新性**：⭐⭐⭐ — 核心贡献在基准构建和标注设计，技术新颖性一般
- **实用性**：⭐⭐⭐⭐⭐ — 填补了多语言仓库级代码补全评估的重要空白
- **实验充分性**：⭐⭐⭐⭐ — 18种语言 × 3种模型 × 3种设置的全面评估
- **写作质量**：⭐⭐⭐ — 内容详实但表格过多，主线有时不够清晰

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding](code-switching_red-teaming_llm_evaluation_for_safety_and_multilingual_understand.md)
- [\[ACL 2025\] MiLiC-Eval: Benchmarking Multilingual LLMs for China's Minority Languages](milic-eval_benchmarking_multilingual_llms_for_chinas_minority_languages.md)
- [\[ACL 2025\] KnowCoder-X: Boosting Multilingual Information Extraction via Code](knowcoder-x_boosting_multilingual_information_extraction_via_code.md)
- [\[ACL 2025\] CruxEval-X: A Benchmark for Multilingual Code Reasoning, Understanding and Execution](cruxeval-x_a_benchmark_for_multilingual_code_reasoning_understanding_and_executi.md)
- [\[ACL 2025\] MaXIFE: Multilingual and Cross-lingual Instruction Following Evaluation](maxife_multilingual_and_cross-lingual_instruction_following_evaluation.md)

</div>

<!-- RELATED:END -->
