---
title: >-
  [论文解读] The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs
description: >-
  [ICLR 2026][Reasoning][长期执行] 揭示短任务基准给出"收益递减"的假象——单步准确率的微小提升在长任务中指数级放大；发现 LLM 的"自我条件化效应"（自身错误增加后续出错概率），thinking 模型可修复此效应；GPT-5 thinking 可执行超过 2100 步长任务。
tags:
  - "ICLR 2026"
  - "Reasoning"
  - "长期执行"
  - "自我条件化"
  - "思维链"
  - "规模缩放"
  - "diminishing returns"
---

# The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs

**会议**: ICLR 2026  
**arXiv**: [2509.09677](https://arxiv.org/abs/2509.09677)  
**代码**: 有  
**领域**: LLM推理  
**关键词**: 长期执行, 自我条件化, 思维链, 规模缩放, diminishing returns  

## 一句话总结
揭示短任务基准给出"收益递减"的假象——单步准确率的微小提升在长任务中指数级放大；发现 LLM 的"自我条件化效应"（自身错误增加后续出错概率），thinking 模型可修复此效应；GPT-5 thinking 可执行超过 2100 步长任务。

## 研究背景与动机

**领域现状**：LLM 在复杂推理基准上持续进步，但当简单任务变长时会失败。这被解读为"推理能力的根本缺陷"或"thinking 只是幻觉"。

**现有痛点**：长任务失败的原因被混淆——是规划失败还是执行失败？现有分析没有隔离执行能力。

**核心矛盾**：短任务基准上的边际改善看起来在递减，但这可能低估了实际能力增长——单步准确率从 95% 到 99% 的提升，在 100 步任务中意味着成功率从 0.6% 到 36.6%。

**本文目标**：(a) 量化 LLM 的长期执行能力；(b) 识别执行失败的根本原因；(c) 分析规模和推理时间计算对长期执行的影响。

**切入角度**：显式提供知识和计划，隔离"执行"能力本身进行测量。

**核心 idea**：LLM 长任务失败主要是执行错误而非推理不足，且存在"自我条件化"——上下文中自身的错误会增加后续出错概率。

## 方法详解

### 整体框架

这是一篇纯评估研究，不训练任何模型，核心是一套受控的"长期执行（long-horizon execution）"测量协议。常规推理基准把规划（planning，决定查什么、按什么顺序）、知识（knowledge）和执行（execution，照计划一步步做）三件事混在一起测，模型在长任务上栽了跟头时没人说得清是哪一环出了问题。本文把前两者从任务里剥掉——直接把所需知识和完整计划喂给模型，只留下"严格照计划逐步执行"这一个变量，再逐步加长任务、看逐步准确率和整体完成率随长度怎么衰减。整篇围绕三件事展开：先把执行能力从规划和知识里单独隔离出来测、再用反事实实验挖出让准确率越走越低的"自我条件化"失效模式、最后用一个公式说清为什么短任务基准上的"收益递减"只是假象。

### 关键设计

**1. 执行能力隔离：把"会不会规划"和"能不能执行到底"分开**

常规推理基准把规划、知识、执行三件事混在一起测，模型在长任务上栽了跟头，没人说得清是哪一环出了问题。本文用一个"键值字典"抽象把前两者钉死：知识是一本固定的"五字母英文单词 → 整数（取值 $[-99, 99]$）"字典放进上下文，计划是每轮直接告诉模型要查哪几个键，模型只需做最机械的"取值再累加（retrieve-then-compose）"——按键查出整数、加进一个 running sum $S_t = S_{t-1} + \sum_{i=1}^{K} \mathcal{D}[k_{t,i}]$。这样规划和知识都不再是变量，剩下唯一要考的就是"照计划一步步做"的执行能力；任务长度由两个轴相乘控制——轮数 $T$ 和每轮复杂度 $K$（一次给几个键）。在这种设置下，模型在长任务上的失败就能干净地归因于执行错误的累积，而不是推理或知识的缺失——这也是本文反驳"长任务失败 = 推理是幻觉"的实验基础。

**2. 自我条件化效应检测：让模型看见自己的错误，看它会不会越错越多**

作者发现一个反直觉现象：人类做重复任务往往越做越熟，但 LLM 在上下文里看到自己先前的错误后，反而更容易接着犯错。要把这个效应从"长上下文本身变难"里剥出来，作者做了个反事实实验——人为篡改模型的对话历史，按一个指定错误率往前面几轮注入假的错误输出，再固定测第 100 轮这一步的准确率。如果历史被"治愈"成 0% 错误率后准确率仍下降，那是长上下文的锅；如果错误率越高、后续准确率越低，就坐实了自我条件化。结果两者都有，且是因果性的：错误率从 0% 升到 5% 后续准确率就明显下滑，升到 20% 出现"错误雪崩"式急跌。关键是这个效应不随模型变大而缓解（连 200B+ 的前沿模型也中招），只有 thinking 模型不受影响——它需要靠推理过程主动"无视"错误历史才能压住，作者推测是 RL 训练让模型偏向"把当前这步做对"而非"续写上下文"。

**3. Horizon Length 度量：用一个公式说清"收益递减"为什么是假象**

要量化单步能力和任务长度之间的关系，作者在"单步准确率独立恒定、且一旦出错即失败"两条简化假设下，定义任务视野长度 $H_s(p) = \lceil \frac{\ln(s)}{\ln(p)} \rceil$，其中 $p$ 是单步准确率、$s$ 是目标整体成功率、$H$ 是在该成功率下还能撑住的最大步数。因为 $H(p) \propto 1/\ln(p)$ 是一条双曲线，越靠近 $p=1$ 曲线越陡：单步准确率从 95% 提到 99%，在 100 步任务上的成功率会从 0.6% 跳到 36.6%。换句话说，短任务基准上看起来"边际收益在递减"的那点提升，一旦放到长任务里就被指数级放大——所谓 diminishing returns，只是用错了尺子才产生的幻觉。

## 实验关键数据

### 主实验（Frontier Thinking 模型单轮最大执行步数）

| 模型 | 最大执行步数 | 说明 |
|------|------------|------|
| GPT-5 (Horizon) thinking | **>2100** | 远超所有竞争者 |
| Claude-4 Sonnet thinking | 432 | 第二名 |
| DeepSeek-R1 (thinking) | >100 | thinking 显著帮助 |
| DeepSeek-V3 (no thinking) | <4 | 无 thinking 几乎无法执行 |

### 自我条件化效应

| 历史错误率 | 后续步骤准确率变化 | 说明 |
|-----------|------------------|------|
| 0% | 基线 | 正常准确率 |
| 5% | 显著下降 | 自我条件化开始 |
| 20% | 急剧下降 | 错误雪崩 |

### 关键发现
- **单步准确率递减是假象**：高准确率区间的微小提升在长任务中指数放大
- **自我条件化效应**：模型不像人类越练越好，反而看到自己的错误后更容易犯错。Thinking 模型不受此效应影响
- **规模缩放在执行上有巨大收益**：即使小模型单步准确率接近完美，大模型在长任务上仍显著更好
- **Thinking 从根本上改善执行**：DeepSeek-V3 无法执行 4 步，R1 可执行 100+ 步
- GPT-5 thinking 的 2100+ 步执行能力标志着 LLM 在长期执行上的质变

## 亮点与洞察
- **"diminishing returns 是幻觉"的论证**极具启发性：$H(p) \propto 1/\ln(p)$ 的双曲增长意味着从 99% 到 99.5% 的提升价值远超从 90% 到 95%。这从根本上改变了对缩放投资回报的认知。
- **自我条件化效应**是全新发现，且不能通过简单的规模缩放解决——只有 thinking 能修复。这对 agent 系统设计有重大启示：需要在执行中清理或隔离错误历史。
- 将长任务失败归因于"执行"而非"推理"的视角非常重要，有助于正确引导研究方向。

## 局限与展望
- 实验任务相对简单（受控环境），真实世界任务的执行失败可能有更多因素
- 未分析自我条件化的机制（是注意力模式、训练数据分布还是其他原因？）
- 仅评估闭源模型的 thinking 能力，无法分析 thinking 如何修复自我条件化

## 相关工作与启发
- **vs Shojaee et al. (Illusion of Thinking)**: 他们声称 thinking 模型在长任务上失败说明推理是幻觉，本文反驳：这是执行失败而非推理失败
- **vs Mirzadeh et al.**: 他们认为 LLM 不能真正推理，本文区分了推理和执行
- 对 agent 系统的启示：需要设计能管理执行历史、避免自我条件化的 agent 框架

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 自我条件化效应和diminishing returns幻觉的发现极具影响力
- 实验充分度: ⭐⭐⭐⭐ 控制实验设计严谨，frontier模型评估全面
- 写作质量: ⭐⭐⭐⭐⭐ 论证逻辑清晰，数学分析简洁有力
- 价值: ⭐⭐⭐⭐⭐ 对LLM缩放投资决策和agent系统设计有根本性影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks](../../ACL2026/llm_reasoning/sppo_sequence-level_ppo_for_long-horizon_reasoning_tasks.md)
- [\[ICML 2026\] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](../../ICML2026/llm_reasoning/toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)
- [\[ICLR 2026\] R-HORIZON: How Far Can Your Large Reasoning Model Really Go in Breadth and Depth?](r-horizon_how_far_can_your_large_reasoning_model_really_go_in_breadth_and_depth.md)
- [\[ICLR 2026\] Long Chain-of-Thought Reasoning Across Languages](long_chain-of-thought_reasoning_across_languages.md)
- [\[ICLR 2026\] When More Is Less: Understanding Chain-of-Thought Length in LLMs](when_more_is_less_understanding_chain-of-thought_length_in_llms.md)

</div>

<!-- RELATED:END -->
