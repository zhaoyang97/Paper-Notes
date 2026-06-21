---
title: >-
  [论文解读] Learning to Parallel: Accelerating Diffusion Large Language Models via Learnable Parallel Decoding
description: >-
  [ICLR 2026][LLM效率][扩散模型] 针对扩散语言模型（dLLM）并行解码依赖固定启发式（如置信度阈值）、对不同输入不自适应的痛点，本文用一个极轻量（2 层 MLP、约 2 千参数、6 分钟训练）的可学习 filter 去逼近"一旦预测正确就立即定稿"的 oracle 策略，再配合 End-of-Text 早停，在 LLaDA-8B 上实现最高 22.58× 加速且几乎不掉点，叠加 KV-Cache 后达 57.51×。
tags:
  - "ICLR 2026"
  - "LLM效率"
  - "扩散模型"
  - "并行解码"
  - "可学习 filter"
  - "推理加速"
  - "LLaDA"
  - "KV-Cache"
---

# Learning to Parallel: Accelerating Diffusion Large Language Models via Learnable Parallel Decoding

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=bFJ8Sdr224](https://openreview.net/forum?id=bFJ8Sdr224)  
**代码**: 已开源（项目页 + GitHub，论文 abstract 中给出）  
**领域**: LLM 推理加速 / 扩散语言模型  
**关键词**: diffusion LLM, 并行解码, 可学习 filter, 推理加速, LLaDA, KV-Cache  

## 一句话总结
针对扩散语言模型（dLLM）并行解码依赖固定启发式（如置信度阈值）、对不同输入不自适应的痛点，本文用一个极轻量（2 层 MLP、约 2 千参数、6 分钟训练）的可学习 filter 去逼近"一旦预测正确就立即定稿"的 oracle 策略，再配合 End-of-Text 早停，在 LLaDA-8B 上实现最高 22.58× 加速且几乎不掉点，叠加 KV-Cache 后达 57.51×。

## 研究背景与动机
- **领域现状**：自回归 LLM 逐 token 解码需要 $O(n)$ 个串行步，吞吐受限。扩散语言模型（LLaDA、Dream、DiffuLLaMA 等）通过迭代去噪一次性刷新整段序列，理论上可并行出多个 token，并普遍采用"分块从左到右"的半自回归解码来兼顾质量与并行度。
- **现有痛点**：要真正吃到并行红利，关键在于"每一步该定稿哪些 token、哪些该重新 mask"。现有方法（confidence 阈值、Fast-dLLM、SlowFast-Sampling、Prophet 等）都是**静态、与输入无关的启发式**——一刀切的规则在不同任务/不同样本上无法自适应，导致速度-质量权衡次优。
- **核心矛盾**：作者实测发现 dLLM 存在严重的**冗余重复解码**——大量 token 早就被预测正确，却因保守的 remask 策略被反复 mask、反复重算。在 GSM8K 上多数 token 在首次预测正确后还要被解码 10 次以上；按 EGP oracle 统计每个 block 中位数只需 2 步，而 vanilla 要 32 步。此外当生成长度拉到 1024 时，约 **89.59%** 的算力浪费在 `[EoT]` 之后的 padding token 反复解码上。
- **本文目标**：把"一刀切静态规则"换成"逐样本、逐 token 自适应"的并行解码策略，在不动主模型参数、不掉精度的前提下榨干并行潜力。
- **核心 idea**：**[Oracle 逼近]** 先定义一个用 ground-truth 的理想策略 EGP（预测对就立刻定稿），证明它有 15-20× 加速潜力；再训练一个**只看置信度信号的轻量 filter** 在推理时逼近这个 oracle；**[早停]** 额外用 EoTP 在 `[EoT]` 出现时砍掉后续 padding。

## 方法详解

### 整体框架
Learn2PD 把"该不该定稿某个 token"建模为一个二分类问题：主扩散模型每步给出各位置的预测 token 和置信度 $c_i$，把这些置信度喂给一个冻结的轻量 filter $f_\theta$，由它输出每个位置"无需 remask"的 logit；超过阈值 $\tau$ 的位置立即定稿、退出 mask 集合。主模型参数全程冻结，只在后训练阶段花极少算力训练 $f_\theta$。再叠加 EoTP 早停模块处理长生成场景。

```mermaid
flowchart LR
    A[掩码序列 X] --> B[扩散主模型 M<br/>冻结]
    B -->|预测 token + 置信度 conf| C[Filter 模型 fθ<br/>2层 MLP]
    C -->|logit > τ?| D{定稿判断}
    D -->|是| E[定稿该 token<br/>移出 mask 集]
    D -->|否| F[保持 [MASK]<br/>下一步重算]
    E --> G[EoTP: 检测到 EoT<br/>丢弃后续位置]
    G --> B
    F --> B
```

### 关键设计
**1. EGP oracle：把"冗余解码"量化成可逼近的上界。** 作者先建一个理想策略 Extremely Greedy Parallel：在第 $k$ 步，当且仅当模型预测 $M(x_k)_i = y_i$（$y_i$ 为参考答案）时才解掩该位置，永不 remask 已正确的 token。它需要 ground truth 因而推理时不可用，但意义在于给出了"并行能快到什么程度"的天花板——实测 15-20× 加速且不掉质量，每 block 中位数仅 2 步 vs vanilla 的 32 步。这把一个模糊的"加速空间"变成了一个明确的、可被监督学习模仿的目标。

**2. 可学习 filter $f_\theta$：用置信度模式逼近 oracle。** 关键观察是扩散模型的置信度有可预测的波动规律——置信度本身就携带"模型对该预测是否真正接受"的信息，足以判断某 token 是否已收敛。于是把 EGP 的定稿决策蒸馏为二分类，用 BCE 损失训练 filter：
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{m}\sum_{i=1}^{m}\Big[y_i\log\sigma(z_i) + (1-y_i)\log(1-\sigma(z_i))\Big]$$
其中 $y_i\in\{0,1\}$ 是 EGP 给出的标签（1=可定稿，0=需 remask），$z_i=f_\theta(\text{conf})$ 是 filter logit，经 $\sigma$ 后与阈值 $\tau$ 比较离散化。训练分两阶段：先按 EGP 策略跑一遍收集每步的置信度与定稿标签（4×A6000 约 3 小时），再用这批数据在一张 T4 上训 filter（仅 6 分钟）。令人意外的是，**最简单的两层 MLP（block size 32 时仅 2,112 个可训练参数）就已足够**——block 级置信度模式信息量充分，无需复杂结构或特征工程；推理时 filter 冻结、无梯度更新，额外开销可忽略。

**3. End-of-Text Prediction（EoTP）：砍掉 padding 的算力黑洞。** 当生成长度设为 1024 而真实答案远短时，多出来的位置全被 `[EoT]` 填充，且模型会反复解码这些 padding——实测占总算力的 89.59%。EoTP 的做法很直接：每一步一旦某个 block 内解出 `[EoT]` token，就**丢弃其后所有位置**，用缩短后的序列作为下一步输入，从而在去噪过程中动态压缩有效长度。它与 Learn2PD 正交，主要在长生成场景带来额外大幅加速（22.58× 中相当一部分来自此模块）。

## 实验关键数据

### 主实验（LLaDA-8B-Instruct，TPS=tokens/sec，Score=任务精度）

| 任务 | 方法 | Gen Len | TPS | 加速 | Score |
|------|------|---------|-----|------|-------|
| GSM8K (5-shot) | LLaDA baseline | 1024 | 0.54 | 1.00× | 77.60 |
|  | + Learn2PD | 1024 | 6.63 | 12.21× | 77.26 |
|  | + Learn2PD + EoTP | 1024 | 12.26 | **22.58×** | 79.83 |
| Math (4-shot) | + Learn2PD + EoTP | 1024 | 12.27 | 7.22× | 34.60 |
| HumanEval (0-shot) | + Learn2PD + EoTP | 1024 | 6.63 | 12.55× | 35.98 |
| MBPP (3-shot) | + Learn2PD + EoTP | 1024 | 9.89 | 17.16× | 11.02 |

长度 256 时通常 3-5× 加速，长度 1024 时 6-22× 加速，精度普遍落在基线 ±1-2 点内，部分任务（GSM8K）甚至略升。

### KV-Cache 兼容 + 消融

| 配置 | TPS | 加速 | Score |
|------|-----|------|-------|
| Learn2PD & EoTP | 12.26 | 22.58× | 79.83 |
| + Dual Cache | 31.23 | **57.51×** | 74.00 |
| + Prefix Cache | 14.79 | 27.23× | 77.71 |

| Filter 深度 | TPS | 加速 | Score |
|-------------|-----|------|-------|
| 单层 | 8.77 | 2.57× | 78.62 |
| **两层** | 14.07 | 4.13× | 78.62 |
| 四层 | 11.41 | 3.35× | 78.85 |

### 关键发现
- **方法与 KV-Cache 完全正交**，叠加 Dual Cache 把加速推到 57.51×（精度略降到 74.00），Prefix Cache 则在几乎不掉点下到 27.23×。
- **两层 MLP 是甜点**：单层表征不足、四层精度微升但速度反降，2 层在效率/质量间最优。
- **生成长度越长收益越大**：128→1024 加速从 3.36× 升到 22.58×，因为长序列里 EoTP 能砍掉的 padding 冗余更多。

## 亮点与洞察
- **把"加速空间"先证明再逼近**：EGP oracle 这一步很漂亮——先用 ground truth 量化出 15-20× 的天花板，证明冗余确实存在且巨大，再用可学习 filter 去蒸馏，逻辑闭环、说服力强。
- **极致轻量的后训练**：只训 2 千参数、6 分钟 T4 时间、主模型完全冻结，几乎零成本即插即用，工程友好度极高。
- **正交叠加**：Learn2PD（解决 block 内冗余 remask）+ EoTP（解决长序列 padding）+ KV-Cache（解决步间重复算）三条独立路径相乘，复合加速到两位数甚至 57×。

## 局限与展望
- **只在 LLaDA-8B 单一主模型上验证**：对 Dream、DiffuLLaMA 等其它 dLLM 的泛化性未充分展示。
- **filter 监督依赖"参考答案"**：标签来自 LLaDA 自身标准解码产生的 reference answer，本质是蒸馏自身行为，filter 上限受主模型质量约束；若主模型本身错了，filter 学到的是"自信地定稿一个错 token"。
- **激进定稿的质量风险**：叠加 Dual Cache 后精度从 79.83 掉到 74.00，说明在追求极端速度时质量并非完全无损，τ 与缓存策略的组合需谨慎调。
- **训练数据来自 FLAN 66 类共 2640 样本**：filter 的跨域鲁棒性（尤其面对训练分布外任务）还需更多检验。

## 相关工作与启发
- **对比静态加速方法**：Fast-dLLM（置信度阈值 + 近似 KV-Cache）、SlowFast-Sampling（两阶段采样器）、Prophet（top-2 logit gap 早提交）、dllm-Cache/FreeCache（训练free缓存）——它们都是固定规则，本文最大区别是把"定稿决策"做成**可学习、输入自适应**的。
- **启发**：将一个不可用的 oracle（依赖 ground truth）显式定义出来、量化其上界、再用最廉价的监督模型逼近，是一种通用且高性价比的加速范式，可迁移到投机解码、early-exit 等"该不该停/该不该信"的决策场景。置信度信号作为唯一输入特征竟足够，提示扩散模型的内部不确定性结构有很强的可读性，值得进一步挖掘。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — "EGP oracle 量化 + 轻量 filter 逼近"的组合在 dLLM 并行解码里是首个可学习策略，思路清晰且立得住。
- **实验充分度**: ⭐⭐⭐ — 四个 benchmark + KV-Cache 兼容 + 深度/长度消融较完整，但只用了 LLaDA 单一主模型，跨 dLLM 泛化欠缺。
- **写作质量**: ⭐⭐⭐⭐ — 从"发现冗余→定义 oracle→证明潜力→蒸馏逼近"的叙事链条非常顺，图表支撑充分。
- **价值**: ⭐⭐⭐⭐ — 几乎零成本、即插即用、与现有加速正交，对 dLLM 实际部署有直接落地价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] dParallel: Learnable Parallel Decoding for dLLMs](dparallel_learnable_parallel_decoding_for_dllms.md)
- [\[ACL 2026\] CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credit](../../ACL2026/llm_efficiency/creditdecoding_accelerating_parallel_decoding_in_diffusion_large_language_models.md)
- [\[ICLR 2026\] Hierarchy Decoding: A Training-free Parallel Decoding Strategy for Diffusion Large Language Models](hierarchy_decoding_a_training-free_parallel_decoding_strategy_for_diffusion_larg.md)
- [\[ICLR 2026\] ReFusion: A Diffusion Large Language Model with Parallel Autoregressive Decoding](refusion_a_diffusion_large_language_model_with_parallel_autoregressive_decoding.md)
- [\[ICLR 2026\] Parallel Sampling from Masked Diffusion Models via Conditional Independence Testing](parallel_sampling_from_masked_diffusion_models_via_conditional_independence_test.md)

</div>

<!-- RELATED:END -->
