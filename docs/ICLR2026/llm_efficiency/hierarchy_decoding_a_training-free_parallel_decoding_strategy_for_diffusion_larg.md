---
title: >-
  [论文解读] Hierarchy Decoding: A Training-free Parallel Decoding Strategy for Diffusion Large Language Models
description: >-
  [ICLR 2026][LLM效率][扩散模型] 针对扩散大语言模型（dLLM）并行解码"一步多 token 就掉点"的痛点，本文提出训练无关的 **Hierarchy-dLLM**：用分治思想把连续掩码区递归切成稀疏的小子区并行解码，让未解码 token 保持稀疏分布以抑制分布漂移，在保持甚至略升精度的同时把解码速度最高提升 17×、比 Fast-dLLM 快约 1.5×。
tags:
  - "ICLR 2026"
  - "LLM效率"
  - "扩散模型"
  - "parallel decoding"
  - "divide-and-conquer"
  - "training-free"
  - "inference acceleration"
---

# Hierarchy Decoding: A Training-free Parallel Decoding Strategy for Diffusion Large Language Models

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ZsIQUjQtdW](https://openreview.net/forum?id=ZsIQUjQtdW)  
**代码**: [https://github.com/inclusionAI/dInfer](https://github.com/inclusionAI/dInfer)  
**领域**: LLM 推理加速 / 扩散语言模型并行解码  
**关键词**: diffusion LLM, parallel decoding, divide-and-conquer, training-free, inference acceleration  

## 一句话总结
针对扩散大语言模型（dLLM）并行解码"一步多 token 就掉点"的痛点，本文提出训练无关的 **Hierarchy-dLLM**：用分治思想把连续掩码区递归切成稀疏的小子区并行解码，让未解码 token 保持稀疏分布以抑制分布漂移，在保持甚至略升精度的同时把解码速度最高提升 17×、比 Fast-dLLM 快约 1.5×。

## 研究背景与动机
- **领域现状**：自回归（AR）LLM 逐 token 生成，串行限制了效率；扩散语言模型（dLLM，主流为 Masked Diffusion Model）用双向注意力迭代去噪，天然支持一步刷新多个 token 的并行解码，被视为打破 AR 串行瓶颈的潜力路线。
- **现有痛点**：开源 dLLM（LLaDA、Dream）由于双向注意力使单步计算远慢于同规模 AR 模型，必须靠并行解码的吞吐增益来补偿；但它们默认退回贪心解码，每步只出一个 token，效率反而不如 AR。盲目放大并行又会触发"并行解码诅咒"——同一步预测的 token 本应满足条件独立假设，强行同时生成会产生"pens, pens, and pens"式语义崩坏。
- **核心矛盾**：并行解码需要多 token 的**联合分布** $p(x^1,\dots,x^k\mid X_t)$，但模型单步采样只给出各位置的**边缘分布** $p(x^i\mid X_t)$；如何用边缘近似联合，是并行解码的中心难题。已有方法多聚焦"置信度阈值决定解谁"，却忽略了**未解码位置的空间分布**对解码的影响。
- **本文目标**：在不重训模型的前提下，设计一个能在精度几乎不掉的同时大幅提速的并行解码策略。
- **核心 idea**：作者用 KL 近似做预实验发现 —— 当未解码 token **稀疏散布**时，一步解码的分布几乎贴合逐步解码；当未解码 token 形成**连续片段**时，分布漂移随片段变长急剧增大。**【核心假设】** 若能用解码策略主动把未解码区结构性地维持成稀疏分布，就能既加速又保精度。由此提出**【分治解码】** Hierarchy-dLLM：把连续掩码区递归切分成相互独立的稀疏小子区并行解。

## 方法详解

### 整体框架
Hierarchy-dLLM 把每个生成块（block）的解码看成一个递归的分治过程：**初始化**一个连续掩码子区 → 在每个子区内按置信度**并行解码**至少一个 token → 用已解码 token 作"锚点"把残余连续掩码区**再切分**成更小的独立子区 → 在每个子区内迭代直到无掩码。已解码的锚点天然把未解码 token 打散成稀疏布局，从而抑制分布漂移；同时多个子区可并行处理，理想情况下达到 $O(\log n)$ 级加速。

```mermaid
flowchart LR
    A["初始化<br/>连续掩码子区"] --> B["子区内并行解码<br/>(置信度分层规则)"]
    B --> C["以已解码 token 为锚点<br/>切分残余掩码区"]
    C --> D{"还有掩码?"}
    D -- 是 --> B
    D -- 否 --> E["输出该 block"]
```

### 关键设计

**1. 分治解码结构（Divide-and-Conquer）：把连续掩码切成稀疏子区。** 这是全文的骨架，直接落实"稀疏布局更稳"的预实验结论。每个 block 起始时是一段长度为 $l$ 的连续掩码，作为初始子区；解码后，本步生成的 token 充当锚点，把每段残余连续掩码切成更小的独立子区进入下一轮。由于子区彼此独立、可并行，且解码总是制造分散的锚点，未解码 token 始终趋向稀疏分布——这正好命中预实验中"稀疏掩码 KL 漂移小"的甜区。这种递归把"一次性并行解一大片高耦合 token"这个困难问题，转化为一连串"在小而稀疏的子区里解少量近独立 token"的易解子问题，理想下带来对数级步数压缩。

**2. 子区内分层置信度解码：高阈严选 + 低阈兜底 + 全局保底。** 子区内的核心难题是"一步尽量多解、又别引入会传播的错误"。先定义位置置信度 $c_i=\max_{v\in V}p_\theta(x^i_s=v\mid X_t)$，再用三级递进规则：(i) **高阈值** $\tau_{high}$——只要 $c_i\ge\tau_{high}$ 就提交该 token，证据强时允许子区内多位置同时并行解，保证语义稳定；(ii) **低阈值** $\tau_{low}$——若某子区没有任何位置过高阈（词表分布平坦），则放宽到只解该子区内最自信的一个 $i^*=\arg\max_{i\in A}c_i$ 且需 $c_{i^*}\ge\tau_{low}$，避免子区原地踏步；(iii) **全局兜底**——若连低阈都无人达标，则强制解全序列里最自信的那个位置，保证每步至少出一个 token、绝不停滞。这套"先严后松再保底"的 best-effort 流程，在效率、可靠性、连贯性之间动态权衡。

**3. 置信度重掩码（Remasking）：纠正被后文证伪的早期预测。** 随着解码推进，早期提交的 token 可能与逐渐成形的上下文冲突，表现为置信度回落。每轮再切分子区**之前**，对所有已解码 token 做一次检查：凡 $c_i<\tau_{remask}$ 的就重新置回 $[\text{MASK}]$ 等待后续重解。这一步把不可逆的逐 token 提交变成可回滚，防止错误沿迭代累积、维护全局一致性，是分治结构能在"激进多解"下仍保住精度的关键安全阀。

整体上，方法**完全训练无关**——只改推理期解码调度，不动模型权重，因此可即插即用到任意开源 MDM 类 dLLM 上。

## 实验关键数据

### 主实验
在 LLaDA-Instruct-8B / LLaDA-1.5-8B / Dream-7B 三个开源模型、五个基准（GSM8K、MATH500、HumanEval、MBPP、IF-Eval）上，对比 Vanilla、Fast-dLLM、WINO。效率用 TPF（每次前向解出的 token 数）和 TPS（每秒吞吐）衡量。

| 任务 (LLaDA-1.5-8B) | 方法 | Score↑ | TPF↑ | TPS↑ |
|---|---|---|---|---|
| GSM8K | Vanilla | 83.17 | 0.65 | 1.99 |
| | Fast-dLLM | 83.32 | 3.10 (4.77×) | 9.54 (4.79×) |
| | **Hierarchy-dLLM** | **83.70** | **4.25 (6.54×)** | **14.83 (7.45×)** |
| MATH500 | Vanilla | 39.80 | 0.84 | 7.96 |
| | **Hierarchy-dLLM** | **41.60** | **3.99 (4.75×)** | **42.25 (5.31×)** |
| HumanEval | Vanilla | 43.29 | 0.93 | 8.56 |
| | **Hierarchy-dLLM** | **45.12** | **4.20 (4.52×)** | **44.18 (5.16×)** |
| MBPP | Vanilla | 40.20 | 0.16 | 0.80 |
| | **Hierarchy-dLLM** | 40.40 | **2.29 (14.31×)** | **12.70 (15.88×)** |

- 在 LLaDA-Instruct-8B 的 MBPP 上 TPS 达 **17.23×** 加速；数学推理任务上不仅提速且精度还涨约 1 点，说明分治解码能缓解长推理链的误差累积。

### 消融实验

| 维度 | 结论 |
|---|---|
| 块长 16/32/64 | 两法精度都稳定，Hierarchy-dLLM 的 TPF/TPS 显著更高，且块越大优势越明显 |
| 生成长度 | 长序列因双向注意力使每步算量增大、TPS 下降，但 Hierarchy-dLLM 的减速远小于 vanilla，加速比随生成长度上升 |

### 关键发现
- **稀疏 vs 连续掩码**：预实验显示连续掩码的近似 KL 漂移随未解 token 数显著快增，稀疏掩码则始终低位——验证了"稀疏布局让双向注意力用上左右锚点上下文"是方法生效的根因。
- **跨架构**：Dream-7B（源自 AR base）也能获得相当加速，但精度保持弱于 LLaDA，作者归因于 AR 出身的模型对并行解码的内在支持更弱。

## 亮点与洞察
- **新视角**：把并行解码从"解谁（置信度）"的一维问题，扩展到"未解码 token 的空间布局"这一被忽视的维度，并用 KL 近似实验给出了清晰证据，是本文最有价值的洞察。
- **训练无关、即插即用**：只改推理调度不动权重，工程落地成本极低，可直接套在现有开源 dLLM 上。
- **分治的优雅性**：用"已解码 token 作锚点切分残余掩码"这一简单机制，同时实现了"制造稀疏布局"和"产生可并行子区"两个目标，一举两得。

## 局限与展望
- **依赖多个阈值**：$\tau_{high},\tau_{low},\tau_{remask}$ 需网格搜索调参，跨任务/模型的最优值可能不同，缺乏自适应机制。
- **AR 出身模型受限**：Dream-7B 上精度保持明显弱于 LLaDA，方法对模型本身并行能力有依赖。
- **$O(\log n)$ 仅理想界**：实际加速受置信度分布平坦、兜底强制解码等影响，难以恒达对数级。
- **评测规模**：主要在 7-8B 量级与五个标准基准上验证，更大模型、开放式长文本生成上的表现待考。

## 相关工作与启发
- **Fast-dLLM**（Wu et al. 2025）：本文最直接的对比基线，主打置信度阈值并行解码，但忽略空间布局；Hierarchy-dLLM 在其之上再快约 1.5×。
- **WINO**（Hong et al. 2025）：多阶段解码，更复杂但在不少任务上掉点更多。
- **SADD / 半自回归扩散解码**：把序列切块从左到右生成、块内并行，是 LLaDA/MMaDA 采用的范式；本文的分治可视为在块内进一步引入递归稀疏化。
- **启发**：对任何"用边缘近似联合"的并行生成（投机解码、并行 token 预测）而言，"主动维持待预测位置的稀疏/低耦合布局"都可能是一条通用的提质思路。

## 评分
- **新颖性**: ⭐⭐⭐⭐ —— "空间布局影响并行解码"的视角新颖，分治 + 稀疏化的组合是首个 position-based dLLM 解码框架。
- **实验充分度**: ⭐⭐⭐⭐ —— 三模型五基准、TPF/TPS 双指标、块长/生成长度消融较完整；但模型规模与生成场景仍偏窄。
- **写作质量**: ⭐⭐⭐⭐ —— 从预实验观察到方法动机的逻辑链清晰，公式与图示到位。
- **价值**: ⭐⭐⭐⭐ —— 训练无关、即插即用、最高 17× 加速且不掉点，对 dLLM 落地有直接工程价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Learning to Parallel: Accelerating Diffusion Large Language Models via Learnable Parallel Decoding](learning_to_parallel_accelerating_diffusion_large_language_models_via_learnable_.md)
- [\[ICLR 2026\] ReFusion: A Diffusion Large Language Model with Parallel Autoregressive Decoding](refusion_a_diffusion_large_language_model_with_parallel_autoregressive_decoding.md)
- [\[ICLR 2026\] dParallel: Learnable Parallel Decoding for dLLMs](dparallel_learnable_parallel_decoding_for_dllms.md)
- [\[ICLR 2026\] Diffusion Language Models Know the Answer Before Decoding](diffusion_language_model_knows_the_answer_before_it_decodes.md)
- [\[ICLR 2026\] ParaRNN: Unlocking Parallel Training of Nonlinear RNNs for Large Language Models](pararnn_unlocking_parallel_training_of_nonlinear_rnns_for_large_language_models.md)

</div>

<!-- RELATED:END -->
