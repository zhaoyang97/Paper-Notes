---
title: >-
  [论文解读] EffiCoder: Enhancing Code Generation in Large Language Models through Efficiency-Aware Fine-tuning
description: >-
  [ICML2025][代码智能][代码生成] EffiCoder 通过构建“正确且高效”的指令微调数据集 EffiInstruct，让代码大模型在提升 pass@1 的同时显著降低执行时间和总内存开销，证明“效率可以通过数据配方学习出来”。 为什么“只看正确率”不够 当前代码大模型大多围绕正确率优化，例如 pass@k、单元…
tags:
  - "ICML2025"
  - "代码智能"
  - "代码生成"
  - "效率感知微调"
  - "指令微调"
  - "执行时间"
  - "内存开销"
---

# EffiCoder: Enhancing Code Generation in Large Language Models through Efficiency-Aware Fine-tuning

**会议**: ICML2025  
**arXiv**: [2410.10209](https://arxiv.org/abs/2410.10209)  
**代码**: [GitHub - EffiCoder](https://github.com/huangd1999/EffiCoder)  
**领域**: 代码智能  
**关键词**: 代码生成, 效率感知微调, 指令微调, 执行时间, 内存开销

## 一句话总结
EffiCoder 通过构建“正确且高效”的指令微调数据集 EffiInstruct，让代码大模型在提升 pass@1 的同时显著降低执行时间和总内存开销，证明“效率可以通过数据配方学习出来”。

## 研究背景与动机

### 为什么“只看正确率”不够
当前代码大模型大多围绕正确率优化，例如 pass@k、单元测试通过率等。
但在真实工程里，代码还必须满足效率约束：
- 运行更快（execution time 更低）
- 占用更少内存（memory usage 更低）
- 在资源受限设备上可部署

论文指出，主流模型即便能生成正确代码，也常常效率不如人工 canonical solution。
这会带来更高算力与能耗成本，也限制模型在边缘场景中的可用性。

### 作者的关键观察
作者先做了一个前置实验：
- 用不同“效率水平”的训练数据微调同一模型
- 测试生成代码的效率指标

结果显示训练数据效率与模型输出效率高度相关：
- ET 相关系数: 0.972
- MU 相关系数: 0.950
- TMU 相关系数: 0.986

这说明“训练集里有什么风格，模型就会学到什么风格”。
因此，问题转化为：如何构造一个大规模、可执行验证、跨语言的“高效代码指令集”。

## 方法详解

### 总体框架
EffiCoder 的核心不是改模型结构，而是改训练数据配方：
1. 汇集多源开源代码指令数据
2. 对每个任务让多个 LLM 生成候选解
3. 本地执行候选解，测量运行时间和内存
4. 选取“正确且更高效”的解作为监督标签
5. 形成 EffiInstruct，对目标代码 LLM 做 SFT

### Step 1: 候选任务池构建
论文从 9 个公开数据源汇总任务（如 SelfCodeAlign、CodeFeed、APPS 等），
对原始约 79 万候选样本做清洗与过滤，最终得到约 6.57 万任务。

这些任务覆盖 5 种语言：Python、C++、Java、Rust、Go，保证了跨语言泛化能力。

### Step 2: 多模型候选解生成
对于每个任务，不只依赖单一模型，而是让多个 LLM 生成多组候选代码。
这样做的目的：
- 增加解空间多样性
- 提高获得高效实现的概率
- 避免单模型风格偏置

### Step 3: 本地可执行效率评测
作者直接在本地执行候选代码，记录三类指标：
- ET: Execution Time
- MU: Max Memory Usage
- TMU: Total Memory Usage（文中也给了标准化指标）

然后选择执行时间和内存表现更优的候选解作为最终 supervision。

### Step 4: 形成 EffiInstruct 并微调
最终得到 EffiInstruct：
- 既包含任务描述
- 也包含经过效率筛选后的高质量目标代码
- 还有部分可用于分析的元数据

训练设置上，论文采用标准监督微调流程（LlamaFactory 实现）：
- max length: 2048
- batch size: 128
- lr: 5e-6
- scheduler: cosine
- warmup ratio: 0.03
- epoch: 4

### 方法本质
EffiCoder 的思想可概括为：
- 不是告诉模型“怎么优化算法”
- 而是通过数据分布，把“效率偏好”蒸馏进模型

这种方法对不同基座模型、不同语言都较友好，工程迁移成本低。

## 实验关键数据

### 数据集规模与语言分布（EffiInstruct）

| 语言 | 任务数 |
|------|--------|
| Python | 33,489 |
| Java | 14,726 |
| C++ | 11,547 |
| Rust | 4,270 |
| Go | 1,678 |
| 总计 | 65,710 |

这说明该数据集不是单语种小样本，而是可支撑中大模型微调的跨语言资源。

### 主实验（EffiBench）

| 模型 | Pass@1 (原始) | Pass@1 (+EffiInstruct) | ET 改善 |
|------|---------------|------------------------|---------|
| Qwen2.5-Coder-7B-Instruct | 44.8 | 57.7 | 0.31s -> 0.16s（-48.4%） |
| Qwen2.5-Coder-14B | 57.5 | 63.6 | 0.36s -> 0.15s（-58.3%） |
| Qwen2.5-Coder-7B | 50.1 | 57.3 | 0.26s -> 0.17s（-34.6%） |
| DeepSeek-Coder-6.7B-Instruct | 44.4 | 51.7 | 0.34s -> 0.22s（-35.3%） |
| CodeLlama-7B | 15.0 | 17.6 | 0.24s -> 0.20s（-16.7%） |

可以看到两点：
- 正确率上升（pass@1 普遍提升）
- 效率指标下降（ET/TMU 明显降低）

这正是论文想证明的“正确性与效率可同时优化”。

### 额外对比（文中给出的代表性结论）
- 在 HumanEvalPlus 上，Qwen2.5-Coder-7B-Instruct 的 pass@1 从 76.2 提升到 78.0。
- 与 PIE 对比时，CodeLlama-7B 在 HumanEvalPlus 的 pass@1 可提升到 31.1，且执行时间降幅也更明显（文中给出 EffiCoder 相比 PIE 更大幅度降低时延）。

### 关键发现
1. 训练数据效率与生成效率的相关性非常强，验证了方法动机。
2. 小模型到大模型都能受益，说明方法具备规模可迁移性。
3. 某些模型在 MU 上提升不明显，但在 ET/TMU 和 pass@1 上依然稳定受益。
4. 数据工程（构造高效监督）可以带来接近“算法改进”的收益。

## 亮点与洞察

1. 选题非常实用。
	代码生成从“能跑”走向“跑得好”，符合产业真实需求。

2. 方法简单但有效。
	不改网络结构、不引入复杂 RL，仅靠数据构建和 SFT 就能明显提升。

3. 可扩展性好。
	框架天然支持更多语言、更多基座模型、更多候选生成器。

4. 评测设计闭环。
	候选解通过本地执行统一评测，避免只靠启发式打分。

5. 对“数据即算法”提供了强证据。
	对代码任务而言，高质量目标分布本身就是很强的优化信号。

## 局限与展望

1. 评测成本较高。
	大规模本地执行、跨语言环境配置、沙箱安全都会增加数据构建成本。

2. 效率目标的定义仍偏单维。
	目前主要关注 ET/MU/TMU，尚未系统纳入可读性、鲁棒性、并发性能等工程指标。

3. 候选空间依赖生成模型能力。
	如果候选模型都给不出优质实现，上限会受限。

4. 对任务难度分层分析还可加强。
	不同算法题类型（DP/图论/字符串）收益可能不同，值得做更细粒度报告。

5. 安全与副作用尚需进一步评估。
	“高效代码”有时可能引入更激进实现，需额外验证边界输入与异常处理行为。

## 相关工作与启发

- 与传统指令微调数据构造（Self-Instruct、Evol-Instruct、OSS-Instruct）相比，
  EffiCoder 不是只提升“任务覆盖和正确率”，而是显式引入“效率偏好”。

- 与只做推理期优化的方法不同，EffiCoder把效率能力前移到参数中，
  减少推理阶段额外开销。

- 对后续工作的启发：
  1. 可以把“效率”扩展到“能耗/碳排”目标。
  2. 可以做多目标数据选择（正确率、时延、内存、可读性联合打分）。
  3. 可以结合测试生成与形式化验证，减少“高效但脆弱”的解。

## 评分
- 新颖性: ⭐⭐⭐⭐☆（4.0/5）
- 实验充分度: ⭐⭐⭐⭐☆（4.5/5）
- 写作质量: ⭐⭐⭐⭐☆（4.0/5）
- 价值: ⭐⭐⭐⭐⭐（5.0/5）

综合评价：这篇工作把“效率感知数据构造”系统化，并在多个主流代码模型上给出稳定收益，是非常有工程转化价值的一类论文。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](../../ACL2025/code_intelligence/dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)
- [\[ACL 2025\] GiFT: Gibbs Fine-Tuning for Code Generation](../../ACL2025/code_intelligence/gift_gibbs_fine_tuning_code_gen.md)
- [\[ICML 2025\] SparseLoRA: Accelerating LLM Fine-Tuning with Contextual Sparsity](sparselora_accelerating_llm_fine-tuning_with_contextual_sparsity.md)
- [\[ACL 2025\] CodeIF: Benchmarking the Instruction-Following Capabilities of Large Language Models for Code Generation](../../ACL2025/code_intelligence/codeif_benchmarking_the_instruction-following_capabilities_of_large_language_mod.md)
- [\[ACL 2025\] Personality-Guided Code Generation Using Large Language Models](../../ACL2025/code_intelligence/personality_guided_code_gen.md)

</div>

<!-- RELATED:END -->
