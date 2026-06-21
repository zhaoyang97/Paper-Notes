---
title: >-
  [论文解读] Multi-LCB: Extending LiveCodeBench to Multiple Programming Languages
description: >-
  [ICLR 2026][代码智能][LiveCodeBench] 把只评 Python 的 LiveCodeBench 扩展到 12 种编程语言——靠一条把 LeetCode 函数式任务统一改写成 STDIN/STDOUT 的转换流水线，在不丢任务、不破坏污染控制的前提下实现同题跨语言对比，揭示出当前 LLM 普遍存在的 "Python 过拟合" 与语言特异性数据污染。
tags:
  - "ICLR 2026"
  - "代码智能"
  - "LiveCodeBench"
  - "多语言代码生成"
  - "污染感知评测"
  - "Pass@1"
  - "STDIN/STDOUT"
  - "Python 过拟合"
---

# Multi-LCB: Extending LiveCodeBench to Multiple Programming Languages

**会议**: ICLR 2026  
**代码**: [https://github.com/Multi-LCB/Multi-LCB](https://github.com/Multi-LCB/Multi-LCB)  
**领域**: 代码智能 / 代码生成评测基准  
**关键词**: LiveCodeBench, 多语言代码生成, 污染感知评测, Pass@1, STDIN/STDOUT, Python 过拟合  

## 一句话总结
把只评 Python 的 LiveCodeBench 扩展到 12 种编程语言——靠一条把 LeetCode 函数式任务统一改写成 STDIN/STDOUT 的转换流水线，在不丢任务、不破坏污染控制的前提下实现同题跨语言对比，揭示出当前 LLM 普遍存在的 "Python 过拟合" 与语言特异性数据污染。

## 研究背景与动机
**领域现状**：LiveCodeBench（LCB）凭借持续抓取竞赛题、按发布日期过滤、支持污染感知评测，已成为评估 LLM 代码生成能力的事实标准；DeepMind、DeepSeek 等都拿它当主力指标。

**现有痛点**：LCB 只评 Python 一门语言。而真实软件工程里开发者要在 C++、Java、JavaScript、Rust 等多种语言间切换，每门语言有各自的语法、语义和惯用法。一个只会用 Python 解题的模型，未必能在系统编程要 C++、企业软件要 Java 时同样胜任。

**核心矛盾**：现有多语言基准要么是静态快照、早已饱和并存在污染（MBXP、MultiPL-E、HumanEval-XL 都靠为每门语言重写函数签名和单元测试，工作量大且对语法/运行时差异敏感），要么每门语言用不同任务集（McEval、BigCodeBench）导致无法直接做跨语言对比，要么不持续更新（xCodeEval）。**没有一个基准能同时做到：同题跨语言 + 污染可控 + 持续自动更新**。

**本文目标**：在完整继承 LCB 污染控制和评测协议的前提下，把每一道 LCB 任务无损复制到 12 种语言上，让模型在完全相同的问题上做跨语言能力对比，并随 LCB 演进自动更新。

**核心 idea**：**统一 I/O 协议而非翻译测试**——只保留题目的自然语言描述，把所有隐藏测试统一转成语言无关的 STDIN/STDOUT 格式，这样一套评测 harness 就能驱动任意目标语言，彻底绕开 "为每门语言重写单元测试" 的不可持续工程。

## 方法详解

### 整体框架
Multi-LCB 是一条把 LCB 数据集重新打包成多语言基准的转换流水线：从 HuggingFace 加载某个版本的 LCB 代码生成数据集（保留 LeetCode / AtCoder / Codeforces 三平台全部任务及发布日期元数据），对其中 LeetCode 的函数式任务做格式转换，统一成 STDIN/STDOUT，再把每道题包装成指定目标语言的 prompt 喂给 LLM，最后在隔离沙箱里编译/执行生成代码、用官方隐藏测试判分得到 Pass@1。

```mermaid
flowchart LR
    A[LCB 数据集<br/>HuggingFace] --> B{任务格式}
    B -->|STDIN/STDOUT<br/>AtCoder/Codeforces| D[统一评测池]
    B -->|Functional<br/>LeetCode| C[测试转换器<br/>prompt 改写 + 测试转换]
    C --> D
    D --> E[目标语言 prompt<br/>12 种语言]
    E --> F[LLM 生成代码]
    F --> G[沙箱编译/执行<br/>官方隐藏测试]
    G --> H[Pass@1]
```

### 关键设计

**1. 函数式任务的统一 STDIN/STDOUT 转换：把多语言适配的复杂度压到零**。LCB 有两种原生格式：AtCoder/Codeforces 的 STDIN/STDOUT（程序读标准输入、写标准输出），以及 LeetCode 的 Functional（实现指定函数、由评测系统调用）。后者直接扩展到多语言是个噩梦——每道 LeetCode 题的 Python starter code 与其测试 harness 紧耦合，要为每门目标语言生成等价的 starter code 和调用签名，就得为每门语言维护定制模板，既不可持续又极易出错。作者的破局点是设计一条自动转换流水线，把每道 Functional 任务重写为统一的 STDIN/STDOUT 格式，由两个组件构成：(1) **prompt 改写**——解析题面里的样例，重排成 STDIN/STDOUT 形式放进模型输入；(2) **测试转换**——把所有公开和隐藏测试从原格式转成统一形式。转换按 I/O 结构分三类处理：标量（整数/浮点/布尔/简单字符串）、一维数组（列表空格分隔）、二维数组（首行给行数，随后逐行空格分隔值）。这样一来同一套 harness 就能同时处理原生 STDIN/STDOUT 题和适配后的函数式题。关键的可行性论证是：LeetCode/AtCoder/Codeforces 的题本就由人类专家设计、刻意避开语言特异歧义（因为这些平台本来就支持多语言），所以转换后任务天然语言无关，无需任何针对单一语言的改写；作者人工抽检约 500 道题未发现任何语言相关特征引入不一致。

**2. 零样本 prompt 协议与 Pass@1 自动判分：完全沿用 LCB 评测精神**。代码生成沿用 LCB 原始零样本策略，prompt 三段式构造：系统消息要求模型扮演 "目标语言的专家程序员"（如 "You are an expert Python programmer..."），用户消息给出完整自然语言题面 + 明确的 STDIN/STDOUT 规范 + 输入输出样例，再加一个代码块占位符（其语言标签设为目标语言以保证正确解析）。模型只需输出读标准输入、写标准输出的完整源程序。判分上，一道题只有在成功编译/解释且通过全部隐藏官方测试、无运行时错误和超时的情况下才算正确，主指标是 Pass@1（首个生成解通过所有公开+隐藏测试的任务比例），跨 12 种语言用的是同一套任务，因此任务难度差异不会干扰多语言能力对比。

**3. 格式兼容带来的自动追踪与语言集选择：让基准 "活" 起来**。因为 Multi-LCB 完全兼容原始 LCB 格式，它能自动追踪 LCB 未来的更新——新题进 LCB，就自动进 Multi-LCB，无需人工干预，污染控制（按发布日期过滤、剔除多解或需显式构造数据结构的不适合严格 I/O 判分的题）也直接继承 LCB 官方的过滤结果。语言集选了 C++、C#、Python、Java、Rust、Go、TypeScript、JavaScript、Ruby、PHP、Kotlin、Scala 共 12 门，依据三条标准平衡：(1) 流行度（综合 GitHub、StackOverflow、RedMonk、TIOBE 排名），(2) 基础设施稳定（能用 Conda 等包管理器做可复现执行），(3) 范式多样（覆盖不同编译策略、类型系统、内存管理模型）。每门语言在隔离沙箱容器里执行，绑定各自编译器/解释器（GCC 13、Rust 1.79、OpenJDK 21、.NET 8、CPython 3.11、Node.js 20），强制 6 秒墙钟/测试、4GB 内存、无外网，保证确定性、安全和语言无关的执行环境。

## 实验关键数据
评测 24 个公开 LLM（7B–685B，覆盖通用与代码专精、指令微调与推理增强），主结果用 Dataset v6（2025-02 至 2025-05，限定 2025-02-01 之后发布的题以保证 post-cutoff 评测）。推理用 vLLM/SGLang，温度 0.2、top-p 0.95，Pass@1 取 10 次平均。

### 主实验表格（Pass@1 % · 部分代表模型 · *=推理模式）

| 模型 | Python | C++ | Java | Go | Rust | Scala | 12 语言 Avg |
|------|--------|-----|------|-----|------|-------|-------------|
| GPT-OSS-120B* (Medium) | 71.1 | **72.3** | 70.4 | **69.9** | 70.5 | 54.1 | **67.8** |
| Qwen3-235B-A22B-Thk-2507* | **74.0** | 75.8 | **73.9** | 56.7 | 47.7 | 57.6 | 64.0 |
| DeepSeek-R1-0528* | 66.3 | 68.0 | 67.8 | 55.0 | 63.1 | 62.3 | 63.1 |
| Qwen3-30B-A3B-Thk-2507* | 64.0 | 65.7 | 62.4 | 44.1 | 51.7 | 43.6 | 53.2 |
| OpenRsn-Nmt-32B* | 64.4 | 44.2 | 40.8 | 11.5 | 2.8 | 6.0 | 22.7 |
| OpenCodeRsn-Nmt-1.1-32B* | 56.0 | 37.3 | 33.1 | 9.9 | 1.1 | 7.0 | 19.8 |

### 复现保真度（Multi-LCB 的 Python 子集 vs 官方 LCB 排行榜）

| 模型 | ORIG | OUR | Δ |
|------|------|-----|-----|
| Qwen3-235B-A22B-Thk-2507 | 74.1 | 74.0 | −0.1 |
| DeepSeek-R1-0528 | 68.7 | 66.3 | −2.4 |
| Qwen3-235B-A22B-Ins-2507 | 51.8 | 43.8 | −8.0 |
| **平均绝对偏差** | — | — | **≈3%** |

### 关键发现
- **Python 不是非 Python 语言的可靠代理**：GPT-OSS-120B* 在 Python 上输给 Qwen3-235B-Thk，却在 Go/JS/TS/Rust/Ruby/Kotlin 上反超；强 Python 能力≠强跨语言能力，必须直接在目标语言上评测。
- **Python 过拟合普遍存在**：Python vs 跨语言平均的散点几乎全在 y=x 对角线上方；OpenRsn-Nmt-32B*、OpenCodeRsn-Nmt-1.1-32B* 这类无显式多语言训练的模型 Python 超 60% 但其他语言普遍低于 30%，缺口超 60%。
- **难度梯度清晰**：Python 平均 Pass@1 最高（0.482），Java/C++ 紧随（≈0.44），中间梯队 0.33–0.39，Scala 垫底（<0.29）——反映编译复杂度、所有权语义、生态资源规模等结构性差异。
- **语言特异性污染**：按月看 top-10 模型 Pass@1，老题（cutoff 前）系统性偏高，跨过 cutoff 时出现阶梯式下跌并维持低位，说明老窗口的高分来自预训练曝光而非真实泛化；污染程度因语言而异。
- **远未饱和**：仅 GPT-OSS-120B*、Qwen3-235B-Thk、DeepSeek-R1 三个模型构成第一梯队，多数模型 12 语言平均低于 40%。

## 亮点与洞察
- **以 "统一协议" 取代 "逐语言翻译"**，是整个工作的工程杠杆：把 N 门语言 × M 道题的适配复杂度从 "每语言定制模板" 压成 "一次性 I/O 转换 + 共享 harness"，这也是它能保持随 LCB 自动更新的根因。
- **同题跨语言**这一设计让 "Python 过拟合" 第一次被干净地量化——因为任务集完全相同，语言间的分差只能归因于语言本身而非题目难度，这是 MBXP/MultiPL-E 等做不到的。
- 把污染分析做到**逐月、逐语言**粒度，揭示了 "污染不是均匀的，而是跟着预训练语料分布走" 这一更细的结论，对解读排行榜分数很有警示意义。

## 局限与展望
- **语言覆盖有限**：12 门语言不含 Swift、Haskell、R 等，选择依据 2025 流行度排名，未覆盖专用/新兴语言，也未区分语言的方言和版本。
- **任务域单一**：尽管语言跨越系统编程/Web/数据科学，但题目本身仍根植于竞赛编程，算法解题与工业软件工程只是间接相关，不能完全代表真实开发任务。
- **转换可行性依赖平台假设**：统一 STDIN/STDOUT 转换之所以成立，是建立在 "竞赛平台题目天然语言无关" 之上，对结构更复杂或带语言特异语义的任务（已被官方过滤）不适用。
- **改进方向**：作者明确指出提升非 Python 语言的训练覆盖有望缩小跨语言差距，这为后续 "语言无关代码模型" 训练提供了明确的评测靶子。

## 相关工作与启发
- **单语言基准**（HumanEval、MBPP、APPS、CodeContests、CodeXGLUE）：静态快照、无日期过滤、Python 中心、已饱和。
- **多语言基准**（MBXP、MultiPL-E、HumanEval-XL、xCodeEval、McEval、BigCodeBench、Ag-LiveCodeBench-X）：或靠逐语言重写单元测试，或每语言任务集不同，或不持续更新；Multi-LCB 用 STDIN/STDOUT 统一 + 同题 + 自动追踪三点同时破解。
- **污染感知评测**（LiveCodeBench、EvoCodeBench）：LCB 的日期过滤是 Multi-LCB 的地基，EvoCodeBench 思路相近但停止维护且仍限 Python。
- **启发**：当一个评测维度（语言）需要横向扩展时，"统一被评对象的接口协议" 往往比 "为每个实例做定制适配" 更可持续——这条工程哲学可迁移到多模态、多任务等其他评测扩展场景。

## 评分
- **新颖性**: ⭐⭐⭐（方法不是全新发明，是 LCB 的多语言扩展；但 "统一 STDIN/STDOUT + 同题跨语言 + 自动追踪" 的组合定位精准，填补了真实空白）
- **实验充分度**: ⭐⭐⭐⭐（24 个模型 × 12 语言 × 10 次平均，含复现保真度对照和逐月污染分析，证据链完整）
- **写作质量**: ⭐⭐⭐⭐（动机—设计—发现层层递进，图表清晰，结论可操作）
- **价值**: ⭐⭐⭐⭐（暴露了 LLM 代码能力评估的关键盲区 "Python 过拟合"，并提供了一个能持续自更新的多语言靶子，对代码模型训练和评测都有直接指导意义）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Gradient-Based Program Synthesis with Neurally Interpreted Languages](gradient-based_program_synthesis_with_neurally_interpreted_languages.md)
- [\[ICLR 2026\] VisCoder2: Building Multi-Language Visualization Coding Agents](viscoder2_building_multi-language_visualization_coding_agents.md)
- [\[AAAI 2026\] Extracting Events Like Code: A Multi-Agent Programming Framework for Zero-Shot Event Extraction](../../AAAI2026/code_intelligence/extracting_events_like_code_a_multi-agent_programming_framework_for_zero-shot_ev.md)
- [\[ICLR 2026\] The Matthew Effect of AI Programming Assistants: A Hidden Bias in Software Evolution](the_matthew_effect_of_ai_programming_assistants_a_hidden_bias_in_software_evolut.md)
- [\[ICLR 2026\] CARD: Towards Conditional Design of Multi-agent Topological Structures](card_towards_conditional_design_of_multi-agent_topological_structures.md)

</div>

<!-- RELATED:END -->
