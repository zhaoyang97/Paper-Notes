---
title: >-
  [论文解读] ALE-Bench: A Benchmark for Long-Horizon Objective-Driven Algorithm Engineering
description: >-
  [NeurIPS 2025 (Datasets & Benchmarks Track)][图像生成][算法工程基准] 提出ALE-Bench，首个面向分数制算法工程竞赛（AtCoder Heuristic Contest）的AI基准，收集40道NP-hard优化赛题并提供交互式Agent评估框架，发现最强模型o3-high在one-shot设置下仅达人类平均水平，且AI在跨问题一致性和长时间迭代改进上与人类专家差距显著。
tags:
  - "NeurIPS 2025 (Datasets & Benchmarks Track)"
  - "图像生成"
  - "算法工程基准"
  - "长时间跨度推理"
  - "分数制编程竞赛"
  - "迭代优化"
  - "LLM Agent"
---

# ALE-Bench: A Benchmark for Long-Horizon Objective-Driven Algorithm Engineering

**会议**: NeurIPS 2025 (Datasets & Benchmarks Track)  
**arXiv**: [2506.09050](https://arxiv.org/abs/2506.09050)  
**代码**: [GitHub](https://github.com/SakanaAI/ALE-Bench)  
**领域**: 图像生成  
**关键词**: 算法工程基准, 长时间跨度推理, 分数制编程竞赛, 迭代优化, LLM Agent

## 一句话总结

提出ALE-Bench，首个面向分数制算法工程竞赛（AtCoder Heuristic Contest）的AI基准，收集40道NP-hard优化赛题并提供交互式Agent评估框架，发现最强模型o3-high在one-shot设置下仅达人类平均水平，且AI在跨问题一致性和长时间迭代改进上与人类专家差距显著。

## 研究背景与动机

**领域现状**：编程基准（HumanEval、CodeContests、LiveCodeBench）聚焦于短时间、pass/fail的精确解题任务，LLM在这些基准上已接近顶级人类水平，面临饱和。

**现有痛点**：现实世界充满NP-hard优化问题（物流路径规划、工厂排产调度、电网平衡），这些问题没有精确解，需要人类专家花费数天甚至数周通过模拟退火、beam search等启发式方法迭代改进。现有编程基准无法评估AI在此类"长时间跨度、分数驱动"任务上的能力——它们只测单次提交，不测迭代improvement。

**核心矛盾**：AtCoder Heuristic Contest（AHC）是全球最大的分数制算法竞赛之一，每场约1000人参与，选手花数周迭代优化。但AI在这种长时间推理+反复试错的范式下表现如何，缺乏系统化的评估平台。

**本文切入角度**：将AHC的40道赛题标准化为AI可参与的benchmark，提供Session交互接口模拟真实比赛体验（读题→测试运行→可视化→提交），设计ALE-Agent作为强基线。核心idea是将"长时间迭代改进能力"作为AI的新评估维度。

## 方法详解

### 整体框架

ALE-Bench由三部分组成：(1) 数据集——40道AHC赛题（含问题描述、Rust评分器、可视化工具、人类排行榜），覆盖路径规划、排产、拼图、多agent控制、贝叶斯推断等多种类型；(2) 评估框架——Python库提供Session接口，AI通过读题→测试运行（获得分数反馈）→可视化→最终提交的循环迭代，Docker沙箱确保执行环境一致；(3) ALE-Agent——专为算法工程设计的Agent原型。

### 关键设计

1. **Session交互系统**:
    - 功能：为AI提供模拟真实AHC比赛的完整交互环境，支持四种操作（查看题目、测试运行、可视化、最终提交）
    - 核心思路：Session对象创建时启动计时器，AI在时间限制内可反复执行测试运行获取反馈分数。代码在Docker沙箱中执行，支持C++、Python、Rust三种语言。可视化工具提供静态图像和交互式Web两种模式。最终提交触发私有评测并计算Performance指标（类Elo rating，0-3500分）
    - 设计动机：忠实复现人类参赛者的工作流，确保AI与人类在完全相同的条件下比较。使用Amazon EC2 C6i实例标准化CPU环境，消除硬件差异对分数的影响

2. **ALE-Agent（算法工程专用Agent）**:
    - 功能：在ALE-Bench框架上构建的强baseline Agent，通过领域知识注入和多样性搜索两个技术显著提升LLM的算法工程能力
    - 核心思路：**方法1——领域知识注入**：将模拟退火、beam search等优化算法的专家知识直接嵌入prompt，包括搜索空间设计、评价函数构造、邻域生成和加速技巧。**方法2——多样性导向搜索**：基于best-first search的树搜索，每次从最优节点扩展 $k=30$ 个子节点并行生成候选解，beam-search式保留多条有潜力的搜索路径避免过早收敛，同时分摊API延迟
    - 设计动机：算法工程领域有明确的"标准技术"（模拟退火、贪心等），选择正确的高层策略至关重要。但即使策略正确，实现细节和超参调优也极大影响结果，因此需要多样性搜索来探索不同实现变体

3. **评估指标体系**:
    - 功能：提供细粒度（单题Performance）和聚合（average performance、rating）两层评估指标，支持AI与人类的公平比较
    - 核心思路：Performance基于Elo-rating方法从排名计算，范围0-3500，跨问题可比。聚合指标推荐使用average performance（rating会被个别高分题拉高，高估AI能力）。提供lite版（10题）和full版（40题）两种规模
    - 设计动机：分数制任务无法用pass/fail评判，需要连续的Performance指标。rating系统为人类设计（鼓励冒险策略），对AI评估存在偏差——单次特别高分会大幅膨胀rating

### 损失函数 / 训练策略

作为评估基准，无训练过程。ALE-Agent使用迭代prompt策略：每轮包含历史摘要、当前最优代码、改进方向提示。测试运行反馈（分数+错误信息）直接拼入下一轮prompt。

## 实验关键数据

### 主实验

| 模型/设置 | Avg Perf (short) | Avg Perf (long) | Avg Perf (overall) | Rating | Top% |
|-----------|-----------------|-----------------|-------------------|--------|------|
| o3-high (one-shot) | 1116 | 946 | 1044 | 1456 | 43.2% |
| Claude 3.7 Sonnet (one-shot) | 851 | 810 | 833 | 1197 | 63.2% |
| GPT-4.1 mini (迭代) | 1293 | 1114 | 1217 | 1636 | 30.5% |
| o4-mini-high (迭代) | 1677 | 1307 | 1520 | 2104 | 11.8% |
| 人类平均 | — | — | 1260 | 1414 | — |

### 消融实验

| 配置 | Avg Perf (lite) | 说明 |
|------|----------------|------|
| Self-Refine基线 | 1264 | 简单迭代改进 |
| + 领域知识(M1) | 略提升 | 高层策略选择改善 |
| + 多样性搜索(M2) | 1879 | 大幅提升，探索多条路径 |
| ALE-Agent (M1+M2) | **1879** | 对应rating 2222，top 8.6% |
| 150次独立one-shot最高分 | < 迭代 | 证明基准测量迭代推理而非并行探索 |

### 关键发现

- 迭代设置显著优于one-shot：所有模型在迭代设置下average performance提升400+点
- 推理模型（o3-high等）明显优于非推理模型，但即使最强的o3-high one-shot也仅达人类平均水平
- C++代码质量普遍优于Python和Rust：C++在average performance上高出Python约44、Rust约57
- AI在模拟退火类问题上较强，但在需要问题特定洞察的规划类问题上与人类差距最大
- ALE-Agent实际参加AHC046获得第154名（performance 1915），验证了基准的真实性和Agent的实战能力

## 亮点与洞察

- 填补了AI评估的重要空白：现有基准测"解题能力"，ALE-Bench测"迭代改进能力"——这是人类专家在现实工作中最核心的能力之一。分数无上限意味着即使AI超越人类最高分也仍有区分度。
- ALE-Agent的多样性搜索策略（beam=30并行生成）是一种有趣的inference-time scaling方法，同时起到了amortize API延迟的实用作用。
- 揭示了当前AI的核心弱点：不是"单次最佳表现"而是"跨问题的一致性"——AI在某些问题上能达到很高分但在其他问题上表现平庸，而人类专家的分数分布更均匀。

## 局限与展望

- 数据集规模有限（40题），但每题允许长时间迭代使得方差较小
- AI主要使用文本反馈，未充分利用可视化工具——人类选手大量依赖可视化调试，多模态Agent可能进一步缩小差距
- Rating指标会高估AI能力（一道高分题可大幅拉高），论文推荐使用average performance
- 未来方向：生成合成题目扩展规模、训练RL策略进行长时间推理、结合多模态理解可视化输出

## 相关工作与启发

- **vs SWE-Bench**：SWE-Bench测代码理解和修复（pass/fail），ALE-Bench测算法设计和迭代优化（分数制），两者正交互补
- **vs MLE-Bench**：MLE-Bench来自Kaggle ML竞赛（需GPU、侧重数据科学），ALE-Bench侧重算法工程（仅CPU、成本更低）
- **vs FunSearch**：FunSearch用LLM改进人类模板代码的局部片段，ALE-Bench要求AI从头设计和迭代改进完整解决方案

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个分数制长时间算法工程基准，评估维度独特
- 实验充分度: ⭐⭐⭐⭐⭐ 22个模型×3种语言×3种设置，分析深入
- 写作质量: ⭐⭐⭐⭐ 结构清晰，附录偏长但信息丰富
- 价值: ⭐⭐⭐⭐⭐ AI能力评估的重要新维度，可持续更新

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] LongT2IBench: A Benchmark for Evaluating Long Text-to-Image Generation with Graph-structured Annotations](../../AAAI2026/image_generation/longt2ibench_a_benchmark_for_evaluating_long_text-to-image_generation_with_graph.md)
- [\[ICML 2025\] Quantum Algorithms for Finite-horizon Markov Decision Processes](../../ICML2025/image_generation/quantum_algorithms_for_finite-horizon_markov_decision_processes.md)
- [\[NeurIPS 2025\] CORAL: Disentangling Latent Representations in Long-Tailed Diffusion](coral_disentangling_latent_representations_in_longtailed_dif.md)
- [\[CVPR 2026\] One Algorithm to Align Them All](../../CVPR2026/image_generation/one_algorithm_to_align_them_all.md)
- [\[ICCV 2025\] ALE: Attribute-Leakage-free Editing for Text-based Image Editing](../../ICCV2025/image_generation/ale_attribute_leakage_free_editing.md)

</div>

<!-- RELATED:END -->
