---
title: >-
  [论文解读] Scaling Up, Speeding Up: A Benchmark of Speculative Decoding for Efficient LLM Test-Time Scaling
description: >-
  [ICLR 2026][LLM效率][推测解码] 这篇论文构建了首个专门评测「推测解码（speculative decoding）加速 LLM 测试时扩展（test-time scaling）」的基准，在 BoN 与多轮思考两种范式下统一协议对比了 9 种推测解码方法，核心发现是：测试时扩展产生的推理轨迹高度重复，连最简单的 N-gram 类方法（尤其 SAM）都能逼近甚至超过需要训练的 EAGLE-3，而把二者杂交的混合方法能拿到全场最高加速。
tags:
  - "ICLR 2026"
  - "LLM效率"
  - "推测解码"
  - "测试时扩展"
  - "N-gram"
  - "Best-of-N"
  - "多轮思考"
---

# Scaling Up, Speeding Up: A Benchmark of Speculative Decoding for Efficient LLM Test-Time Scaling

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=DjOmnwX4wJ](https://openreview.net/forum?id=DjOmnwX4wJ)  
**代码**: https://github.com/sunshy-1/SpecTTS-Bench (有)  
**领域**: LLM效率 / 推测解码 / 测试时扩展  
**关键词**: 推测解码, 测试时扩展, N-gram, Best-of-N, 多轮思考

## 一句话总结
这篇论文构建了首个专门评测「推测解码（speculative decoding）加速 LLM 测试时扩展（test-time scaling）」的基准，在 BoN 与多轮思考两种范式下统一协议对比了 9 种推测解码方法，核心发现是：测试时扩展产生的推理轨迹高度重复，连最简单的 N-gram 类方法（尤其 SAM）都能逼近甚至超过需要训练的 EAGLE-3，而把二者杂交的混合方法能拿到全场最高加速。

## 研究背景与动机

**领域现状**：测试时扩展已经成为提升 LLM 推理能力的主流手段——不改模型参数，靠在推理时多花算力让模型「想得更久」。代表范式有两类：Best-of-N（BoN，采样 N 条轨迹再用 verifier 选最优）和多轮思考（multi-round thinking，把上一轮答案塞回 prompt 让模型反复自我修正）。

**现有痛点**：这种「想得更久」是用算力换性能——要生成多条完整回答或多轮推理链，延迟开销巨大，在实时交互场景里几乎不可用。推测解码（用一个快的 draft 机制生成候选 token、再让大的 target 模型一次性并行验证，无损加速）本是缓解延迟的利器，但它在测试时扩展这种「高度结构化、富含重复」的场景里到底好不好用，没人系统研究过。

**核心矛盾**：测试时扩展的推理轨迹有一个被忽视的特性——**冗余**。同一道题反复采样/反复重答时，模型会大量复读相同的逻辑短语、模板代码、过渡句（论文用 intra-turn / inter-turn redundancy 两类例子展示：同一轮里重复计算 $8^3=512$，跨轮里几乎逐字重写开头的 `<think>`）。这种重复天然适合基于检索/缓存的 N-gram 类推测解码。但现有推测解码基准只在通用任务上测，无法回答：**一个简单自适应的 N-gram 机制，能不能在这种富含重复的场景里打赢复杂的预训练 draft 模型？** 各类方法在灵活性、训练成本、实时适配性上的 trade-off 又如何？

**本文目标**：建立第一个系统对比不同推测解码方法加速测试时扩展的基准，量化「灵活性 vs 训练成本 vs 实时适配」的取舍，并验证「N-gram 模式特别适合加速测试时扩展」这一假设。

**切入角度**：把推测解码切成三大家族（模型类 / 训练类 / N-gram 类）外加一个混合方法，统一在 BoN 与多轮思考两种范式、4 个推理数据集、两种采样温度下做 head-to-head 对比，用统一指标 MAT 和 Walltime Speedup 量化。

**核心 idea**：与其设计新方法，不如建一个公平的实验场，逼问「重复性场景下到底哪类推测解码最划算」——答案出人意料地指向了最简单的 N-gram。

## 方法详解

### 整体框架
这是一篇**基准/实证研究**论文，不提出新模型，而是搭一个统一评测框架，把「要加速的测试时扩展范式」「评测数据集」「被对比的推测解码算法」三块拼起来，在同一套协议下跑全。

整个基准沿三个轴展开：**轴一（加速对象）**是两种测试时扩展范式——BoN（生成 N 条候选 + verifier 选优）和多轮思考（M 轮迭代自我修正）；**轴二（被测方法）**是覆盖几乎所有 SOTA 的 9 种推测解码，分成模型类（SpS）、训练类（EAGLE-3）、N-gram 类（PLD/REST/Lookahead/PIA/SAM/Recycling）三家族，外加一个杂交的 SAM[EAGLE-3]；**轴三（评测条件）**是 4 个推理数据集（AIME24/25、MATH500、GPQA 各取 30 题共 120 题）× 2 个目标模型（DeepSeek-R1-Distill-Llama-8B、Qwen3-8B）× 2 个温度（$T=0$ 贪心、$T=0.6$ 采样）。所有方法都汇报两个指标：**MAT（Mean Accepted Tokens，每个推测步平均被接受的 token 数，衡量 draft 质量）**和 **Speed（Walltime Speedup Ratio，相对原始自回归解码的真实墙钟加速比，衡量端到端收益）**——区分这两个指标正是本文很多结论的关键。

### 关键设计

**1. 双范式 + 冗余建模：把测试时扩展的「重复」摆上台面**

基准的加速对象不是泛泛的长文本生成，而是刻意选了两个会制造大量重复的范式。BoN 让模型对同一问题独立采样 $N$ 条轨迹（本文 $N=4$），这些轨迹起手、套路、模板高度相似；多轮思考则把上一轮答案拼回原问题 `{原问题} + 上一轮答案 + Please re-answer`（本文 $M=2$ 轮），后一轮往往逐字复读前一轮的开头与推理骨架。论文用 intra-turn（一轮内复读同一段计算）和 inter-turn（跨轮重写几乎相同的 `<think>` 开头）两类冗余把这件事说清楚。这一步的意义在于：它把「为什么 N-gram 类方法在这里会赢」的前提条件给坐实了——正是这种结构化重复，让「缓存并复用最近生成过的 token 序列」变成高命中率的策略。

**2. 三家族 + 混合的分类对比：用 MAT 与 Speed 拆开「接受率」和「真加速」**

被测方法按 draft 怎么来分三类：**模型类 SpS** 用一个同源小模型（如 Qwen3-0.6B）当 drafter，输出分布对齐好、MAT 最高；**训练类 EAGLE-3** 在 target 模型中间层挂可训练的自回归头来出 draft；**N-gram 类**则完全不靠神经网络，用 Trie / 后缀自动机（suffix automaton）/ top-K logits 缓存等数据结构，从生成历史里检索重复 token 序列当 draft。论文特意把 MAT 和 Speed 拆开看：SpS 的 MAT 最高（草稿质量好），但 0.6B drafter 相对 8B target 仍然太重，跑一遍 draft 省下的时间被 draft 自己的开销吃掉，**Speed 反而 <1×（不到 1，实际变慢）**。这条对比直接证伪了「draft 接受率高就一定快」的直觉，凸显在紧延迟约束下「draft 必须够轻」才是真加速的前提。

**3. N-gram 内部再分「token 级」与「概率级」：定位最划算的那一类**

N-gram 家族内部，论文进一步区分两条技术路线。**token N-gram**（SAM、PLD、PIA、Lookahead、REST）在 token 层面找重复后缀，其中 SAM 用后缀自动机做后缀匹配、复杂度低、线性验证，效率最高，贪心设定下甚至比 EAGLE-3 还快（DSL-8B 上 overall speedup 高出 38%，$1.93×\to2.66×$）；但它们对温度敏感——温度升高输出更发散、重复变少，SAM 在 QW3-8B 上 overall speedup 从 $2.28×$ 掉到 $1.78×$（约 −22%）。**概率 N-gram**（Recycling）则缓存每个 token 的 top-K 后继 token 概率，因为保留了近似的 next-token 分布，即使温度变化也能和 target 采样结果大量重叠，**对温度几乎不敏感**（$T:0\to0.6$ 仅降 <5%）；代价是它用树形验证、单步投机多达 81 个 token（约为 SAM 的两倍），计算开销大，于是 **MAT 是 N-gram 里最高、但 Speed 没能成比例转化**。这一拆解给出实用结论：要稳定选概率 N-gram，要极致速度选 token N-gram（SAM）。

**4. SAM[EAGLE-3] 混合：把「语义对齐」和「重复捕捉」缝在一起**

混合方法把训练类的语义建模能力和 N-gram 的重复复用能力动态切换：当 SAM 找到的匹配后缀太短、不值得投机时，系统回退到 EAGLE-3。结果是 SAM[EAGLE-3] 在所有场景拿到**全场最高 overall speedup**——贪心 + 多轮思考下，在 QW3-8B 上比 EAGLE-3、SAM 分别再快 20% 和 53%。但混合也继承了 SAM 的温度敏感缺陷：$T=0.6$ 时它对 EAGLE-3 的增益明显缩水（因为此时 SAM 那一半准确率不行，整体性能基本退化到由 EAGLE-3 兜底）。论文同时指出当前混合策略仍是启发式的（短后缀回退），没充分榨干 draft 模型的语义丰富度和 N-gram 的可复用性，留有改进空间。

## 实验关键数据

实验在 DeepSeek-R1-Distill-Llama-8B（DSL-8B）和 Qwen3-8B（QW3-8B）两个推理模型上，跑多轮思考（2 轮）与 BoN（4 条轨迹），float16、batch=1，指标为 MAT 与 Walltime Speedup。

### 主实验（多轮思考，Overall 加速比）

| 方法 | 类别 | DSL-8B (T=0) Speed | DSL-8B (T=0) MAT | QW3-8B (T=0) Speed | QW3-8B (T=0) MAT |
|------|------|------|------|------|------|
| EAGLE-3 | 训练类 | 1.93× | 2.35 | 2.91× | 4.38 |
| SAM | token N-gram | **2.66×** | 2.93 | 2.28× | 2.37 |
| Recycling | 概率 N-gram | 2.10× | 2.99 | 2.15× | 3.01 |
| PLD | token N-gram | 1.84× | 2.33 | 1.74× | 2.05 |
| SpS | 模型类 | — | — | 0.87× | **7.07** |
| **SAM[EAGLE-3]** | 混合 | **3.97×** | 4.72 | **3.49×** | 4.76 |

关键看点：SpS 的 MAT 高达 7.07（草稿质量最好），但 Speed 仅 0.87×（反而比不加速还慢）；SAM 用最简单的后缀自动机在 DSL-8B 上 2.66× 反超 EAGLE-3 的 1.93×；混合方法 SAM[EAGLE-3] 全面最高。

### 温度敏感性 / 跨轮分析

| 现象 | T=0 | T=0.6 | 变化 | 说明 |
|------|------|------|------|------|
| SAM（QW3-8B, 多轮）overall speed | 2.28× | 1.78× | −22% | token N-gram 对温度敏感 |
| Recycling（QW3-8B）overall speed | 2.15× | 2.06× | <5% | 概率 N-gram 几乎不敏感 |
| EAGLE-3（QW3-8B）avg speed | — | — | −6% | 训练类对温度鲁棒 |
| SAM 跨轮（图 2a，T=0）turn1→turn2 | 2.13× | 2.83× | +33% | token N-gram 渐进加速 |
| PIA 跨轮（图 2a，T=0）turn1→turn2 | 1.45× | 2.10× | +45% | 复用前轮计算，越跑越快 |

### 关键发现
- **MAT 高 ≠ 真加速**：SpS 和 Recycling 都靠高 MAT 但被计算开销拖累，端到端 Speed 上不去；衡量推测解码必须看墙钟加速而非接受率。
- **简单 N-gram 出奇制胜**：SAM 用 $O$ 高效后缀匹配 + 推理轨迹的天然重复，贪心下逼近甚至超过需要训练的 EAGLE-3，且零训练成本。
- **温度是 token N-gram 的命门**：温度升高→输出发散→重复减少→token 级复用命中率下降；而概率 N-gram 因保留 top-K 分布而稳。
- **跨轮渐进加速是 token N-gram 独有红利**：SAM/PIA 能复用前几轮算过的中间结果，轮数越多加速越大；其他方法各轮独立、加速比基本恒定。
- **混合是当前最优解但仍启发式**：SAM[EAGLE-3] 缝合语义对齐与重复捕捉拿到 SOTA，但短后缀回退策略粗糙，没真正榨干两边潜力。

## 亮点与洞察
- **把「冗余」当成一等公民**：论文的洞察是测试时扩展不只是「更长的生成」，而是「富含重复的生成」，这个视角把 N-gram 类从「老土的基线」翻盘成「场景特化的赢家」。
- **MAT vs Speed 的双指标拆解极有价值**：它戳破了「draft 接受率越高越好」的常见误区，指出在 8B-target/0.6B-draft 这种小尺寸差下，draft 自身开销才是瓶颈——这条经验可直接迁移到任何推测解码选型。
- **渐进加速的发现可复用**：「复用前轮中间结果让加速随轮数累积」这个机制，对一切多轮/迭代式推理（agent、self-refine、ReAct）都有直接启发——轮越多，缓存型推测解码越赚。
- **混合方向的明确指路**：明确指出当前 hybrid 是启发式回退、有改进空间，为后续「学习式切换 SAM/EAGLE」类工作铺好了动机。

## 局限与展望
- **规模有限**：每个数据集只取 30 题（共 120 题），目标模型只测了两个 8B 模型，更大模型（draft/target 尺寸差更大时 SpS 是否翻身）未覆盖。
- **混合策略粗糙**：SAM[EAGLE-3] 用「后缀太短就回退」这种启发式切换，没学习何时该信 N-gram、何时该信 draft 模型，温度高时直接退化为 EAGLE-3。
- **温度敏感无解法**：论文诊断出 token N-gram 的温度敏感问题，但只是观测，没给出缓解方案（如自适应温度下的 N-gram 复用）。
- **改进思路**：可探索「学习式混合」——用轻量控制器根据当前上下文的重复度动态在 SAM/EAGLE-3 间分配，或把概率 N-gram 的温度鲁棒性嫁接到 token N-gram 的低开销上。

## 相关工作与启发
- **vs 通用推测解码基准（Xia et al., 2024）**：他们在通用任务上评测推测解码，本文专攻测试时扩展这一「高重复」场景，复用了其评测 pipeline 以保证公平，但把结论重写了——通用任务里不起眼的 N-gram，在重复场景里成了主角。
- **vs 单一推测解码方法（EAGLE-3 / SAM / Recycling 各自论文）**：那些工作各自宣称 SOTA 但在不同设定下不可比，本文用统一协议把它们拉到同一张表上，揭示了「MAT 冠军（SpS/Recycling）≠ 加速冠军（SAM/混合）」的反直觉结论。
- **vs 测试时扩展方法（BoN、多轮思考）**：那些工作关注「如何更准」，本文正交地关注「如何更快」，把推测解码当成给测试时扩展提速的插件层。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个测试时扩展场景下的推测解码基准，视角（冗余特化）新颖，但方法本身是评测而非新算法
- 实验充分度: ⭐⭐⭐⭐ 9 方法 × 2 范式 × 2 模型 × 2 温度 × 4 数据集，覆盖全面；但每集仅 30 题、模型规模单一
- 写作质量: ⭐⭐⭐⭐ 用 6 条 Takeaway 把结论结构化，MAT/Speed 拆解清晰
- 价值: ⭐⭐⭐⭐⭐ 给推测解码加速推理模型提供了直接可用的选型指南，并指明混合方法的改进方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws](fast_catch-up_late_switching_optimal_batch_size_scheduling_via_functional_scalin.md)
- [\[ICLR 2026\] Test-Time Training Done Right](test-time_training_done_right.md)
- [\[ICLR 2026\] Scaling Attention via Feature Sparsity](scaling_attention_via_feature_sparsity.md)
- [\[ICLR 2026\] Scaling Laws Meet Model Architecture: Toward Inference-Efficient LLMs](scaling_laws_meet_model_architecture_toward_inference-efficient_llms.md)
- [\[ICLR 2026\] xLSTM Scaling Laws: Competitive Performance with Linear Time-Complexity](xlstm_scaling_laws_competitive_performance_with_linear_time-complexity.md)

</div>

<!-- RELATED:END -->
