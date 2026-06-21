---
title: >-
  [论文解读] EntropyLong: Effective Long-Context Training via Predictive Uncertainty
description: >-
  [ICLR 2026][LLM效率][长上下文训练] EntropyLong 用模型自身的预测熵定位"信息缺口"，检索远端上下文并**实测它能否降低该位置的熵**，只保留真正带来信息增益的依赖来拼接 128K 训练样本，从而构造出"被验证过的"长程依赖，在 RULER 和 LongBench-v2 上显著超越启发式数据构造方法。
tags:
  - "ICLR 2026"
  - "LLM效率"
  - "长上下文训练"
  - "预测熵"
  - "信息增益"
  - "数据构造"
  - "模型在环验证"
---

# EntropyLong: Effective Long-Context Training via Predictive Uncertainty

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=SFXX5Pjl5K](https://openreview.net/forum?id=SFXX5Pjl5K)  
**代码**: 待确认（论文声明将开源数据集）  
**领域**: LLM效率  
**关键词**: 长上下文训练、预测熵、信息增益、数据构造、模型在环验证

## 一句话总结
EntropyLong 用模型自身的预测熵定位"信息缺口"，检索远端上下文并**实测它能否降低该位置的熵**，只保留真正带来信息增益的依赖来拼接 128K 训练样本，从而构造出"被验证过的"长程依赖，在 RULER 和 LongBench-v2 上显著超越启发式数据构造方法。

## 研究背景与动机
**领域现状**：架构创新（Longformer、Big Bird、RoPE 外推）已把理论上下文窗口推到百万 token，但真正瓶颈是**训练数据**——缺少能让模型有效利用长上下文的样本。主流做法是把短文档直接拼接，或用启发式合成（Quest 按主题相关检索、NExtLong 插入干扰文档）。

**现有痛点**：所有启发式方法都**预设**了什么是"好的长上下文样本"（如语义相似、有干扰项），却从未从模型视角**直接验证**这种依赖是否真有用。最近的 RE³SYN 用小代理模型的困惑度重排候选，但这是文档级、代理标准，与目标模型的参数知识错位且粒度太粗。

**核心矛盾**：拼接得到的样本可能只有"伪依赖"（spurious correlation）——模型注意力跨度变长了，但并没有学到跨远程整合信息的能力。

**本文目标**：用模型自己的预测不确定性作为信号，构造**可证明带来信息增益**的长程依赖数据。

**核心 idea**：高熵位置 = 信息缺口；若引入远端相关上下文能**实测降低**该位置的预测熵，就说明形成了一条真正需要整合远程信息的长程依赖。把"信息缺口 → 验证熵降 → 有效依赖"建成一个闭环，灵感来自主动学习与不确定性引导的数据筛选。

## 方法详解

### 整体框架
EntropyLong 是一条四阶段、模型在环（model-in-the-loop）的数据构造流水线：先用基模型对文档逐位置算预测熵、自适应阈值挑出高熵锚点；为每个锚点检索语义相关上下文；再把候选上下文前置回去**实测熵是否下降超过阈值**，只留下通过验证的；最后把验证过的上下文随机打乱后与原文档拼成 128K 训练序列。

```mermaid
flowchart LR
    A[根文档 D] --> B[Step1 自适应阈值<br/>选高熵位置 τH=μ+ασ]
    B --> C[Step2 信息论检索<br/>用锚点邻域 query 检索 top-K]
    C --> D[Step3 熵降验证<br/>前置候选, ΔI>ε 才保留]
    D --> E[Step4 策略拼接<br/>Shuffle 验证上下文 + D]
    E --> F[128K 训练样本<br/>含已验证长程依赖]
```

### 关键设计

**1. 自适应阈值的高熵位置选择：让文档自己定义"哪里不确定"** 给定文档 $D=\{x_1,\dots,x_n\}$，用基模型 $M_\theta$ 算每个位置的香农熵 $H_\theta(x_t|x_{<t})=-\sum_{v\in V}P_\theta(v|x_{<t})\log P_\theta(v|x_{<t})$。不用全局固定阈值，而是按**每篇文档自身**的熵分布定阈值 $\tau_H=\mu_H+\alpha\sigma_H$（实际取 $\alpha=2.0$），超过 $\tau_H$ 的位置即"高不确定位置"。这样能自适应不同文档的难度，避免简单文档全军覆没、复杂文档全部入选。

**2. 信息论上下文检索：用锚点邻域当 query** 对每个高熵位置 $t_i$，取其前后各 $w$ 个词（$w=16$）作为查询 $q_i=x_{t_i-w:t_i+w}$，用预训练 sentence transformer 编码，按余弦相似度从超过 1B 文档的大语料中检索 top-K 候选。注意这一步只是**召回候选**，相似度高不等于有用，真正的取舍交给下一步。

**3. 熵降验证（方法基石）：实测信息增益才算数** 把候选上下文 $C_j$ 前置到原文档前，重新计算高熵位置的熵 $H'_\theta$，定义**上下文信息增益**为相对熵降：$\Delta I_{t_i}(C_j,D)=\frac{H_\theta(x_{t_i}|x^D_{<t_i})-H'_\theta(x_{t_i}|x^{[C_j;D]}_{<t_i+|C_j|})}{H_\theta(x_{t_i}|x^D_{<t_i})}$。只有当 $\Delta I_{t_i}>\epsilon$（取 $\epsilon=0.4$）才保留该上下文。这一步把"语义相似"换成"实测有用"，过滤掉伪依赖和冗余，是全方法的核心——也是消融里贡献最大的一环。最终数据集平均信息增益 $\bar{\Delta I}=0.68$。

**4. 策略拼接：随机打乱优于保序** 对一篇文档收集到的多个验证上下文 $\{C_1,\dots,C_m\}$，构造样本 $S=[C_{\pi(1)};C_{\pi(2)};\dots;C_{\pi(m)};D]$。比较两种排列：Sequence（按原 token 顺序）与 Shuffle（随机置换 $\pi$）。实验显示随机打乱略优，因为它能防止模型学到"依赖固定位置"的捷径，强迫真正去远程定位信息。

## 实验关键数据
基模型 Meta-Llama-3-8B，仅把 RoPE base 改为 200,000,000 扩到 128K；用 FineWeb-Edu + Cosmopedia，采样 100K 文档为源、全语料（>1B 文档）为检索库，生成 4B token 的 128K 序列；训练 1000 步、global batch 4M token。基线 Quest / NExtLong 用相同配置和 4B token。

### 主实验

| 数据集 | 指标 | 本文 | SOTA | Δ |
|--------|------|------|------|---|
| RULER (avg 8K–128K) | 平均分 | 87.37 | 85.22 (NExtLong) | +2.15 |
| RULER @128K | 分数 | 81.26 | 77.99 (NExtLong) | +3.27 |
| RULER @128K vs Quest | 分数 | 81.26 | 60.11 (Quest) | +21.15 |
| LongBench-v2 (SFT 后) | Overall | 27.60 | 24.10 (NExtLong) | +3.50 |
| LongBench-v2 Long 任务 | 分数 | 31.50 | 23.10 (NExtLong) | +8.40 |

### 消融实验

| 配置 | 指标 (RULER avg) | 说明 |
|------|------|------|
| 完整方法 | 87.37 | 含熵降验证 |
| NoVerify（仅语义相似） | 85.82 | 去掉验证步，−1.55，128K 处 −1.79 |
| α=1.5（913 tokens） | 82.49 | 位置太多噪声大 |
| α=2.0（292 tokens） | 87.37 | 最优平衡 |
| α=2.5（83 tokens） | 85.52 | 位置太少数据不足 |
| ε=0.2（62 依赖） | 85.45 | 验证太松收进弱依赖 |
| ε=0.4（46 依赖） | 87.37 | 最优平衡 |
| ε=0.6 / 0.8（29/13 依赖） | 86.14 / 86.47 | 太严依赖太少 |

### 关键发现
- 熵降验证不可或缺：去掉后全程下降，且**上下文越长掉得越多**（128K −1.79），说明验证过滤掉的伪依赖在长程任务上危害最大。
- 阈值存在明确最优点：$\alpha=2.0$、$\epsilon=0.4$ 同时验证了"质量 vs 数量"的权衡假设——既不能太松收噪声，也不能太严缺训练信号。
- 优势随上下文长度递增：EntropyLong 相对 NExtLong 的领先在更长上下文上更明显，印证信息论构造确实增强了远程整合能力。
- 窗口 $w$ 呈非单调：$w=8$ 在短上下文更好但长上下文退化，$w=16$ 全程最均衡。
- 随机打乱优于保序拼接：打乱破坏了"上下文按固定顺序出现"的捷径，迫使模型真正学习远程定位，而非记忆位置模式。
- 数据量并非越多越好：α=1.5 虽选出 913 个高熵 token（远多于 α=2.0 的 292 个），但因噪声位置混入反而把平均分从 87.37 拉低到 82.49，凸显"验证过的少量优质依赖" > "未验证的大量依赖"。

## 亮点与洞察
- **把"好数据"的定义权交还给模型**：从人定启发式（相似/干扰）转向模型实测的信息增益，是一种自监督、可验证的数据构造范式。
- **熵降作为"长程依赖是否成立"的可操作判据**：第一次把"信息缺口→验证熵降→有效依赖"形式化成闭环，每条依赖都有 $\bar{\Delta I}=0.68$ 的可量化证据。
- **位置级、模型对齐**：相比 RE³SYN 的文档级代理困惑度，本文是 token 级且用目标模型自身，避免知识错位，且无需排列搜索即可扩展。

## 局限与展望
- 验证步需要对每个候选上下文做一次前向推理，构造成本随检索 top-K 和高熵位置数线性增长，数据流水线开销较大。
- 实验仅在 Llama-3-8B 单一规模验证，更大模型 / 不同架构下熵阈值是否同样最优未知。
- 高熵不一定都是"需要远程上下文"——也可能是固有歧义或噪声 token，验证步虽能部分过滤，但 query 取邻域仍受局部信息限制。
- $\alpha$、$\epsilon$、$w$ 都需调参，最优值可能随目标上下文长度变化（窗口非单调已显现），缺乏自动选择机制。

## 相关工作与启发
- **vs Quest（连贯驱动）**：Quest 假设主题相关就能形成有用依赖，本文指出语义相似≠有用，用实测熵降替代相似度，128K 处领先 21 分。
- **vs NExtLong（区分驱动）**：NExtLong 靠插入干扰文档逼模型区分，本文不预设"难度来源"而直接验证信息增益，长程任务领先更明显。
- **vs RE³SYN（验证驱动）**：同样讲"验证"，但 RE³SYN 用小代理模型文档级困惑度重排，本文用目标模型 token 级熵降，模型对齐且无需排列搜索。
- **启发**：预测熵/不确定性长期用于主动学习选标注样本，本文把它**建设性**地用于无监督预训练数据构造，为"模型自主精炼自己训练数据"的自动化管线提供了样板。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把预测熵从"诊断信号"转为"建设性数据构造判据"，model-in-the-loop 验证长程依赖的思路清晰且原创。
- 实验充分度: ⭐⭐⭐⭐ RULER+LongBench-v2 双基准、对 α/ε/w/拼接策略系统消融，验证步必要性论证扎实；但仅单一基模型规模。
- 写作质量: ⭐⭐⭐⭐ 理论动机（前提→原则→假设）与方法四阶段对应清晰，公式与流程图配套，可读性好。
- 价值: ⭐⭐⭐⭐ 长上下文数据构造的实用范式，承诺开源 128K 数据集，对社区有直接价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] NExtLong: Toward Effective Long-Context Training without Long Documents](../../ICML2025/llm_efficiency/nextlong_toward_effective_long-context_training_without_long_documents.md)
- [\[ICLR 2026\] AutoSP: Unlocking Long-Context LLM Training Via Compiler-Based Sequence Parallelism](autosp_unlocking_long-context_llm_training_via_compiler-based_sequence_paralleli.md)
- [\[ICLR 2026\] Let's (not) just put things in Context: Test-time Training for Long-context LLMs](lets_not_just_put_things_in_context_test-time_training_for_long-context_llms.md)
- [\[ICLR 2026\] Wide-In, Narrow-Out: Revokable Decoding for Efficient and Effective DLLMs](wide-in_narrow-out_revokable_decoding_for_efficient_and_effective_dllms.md)
- [\[ICLR 2026\] Long-Context Attention Benchmark: From Kernel Efficiency to Distributed Context Parallelism](long-context_attention_benchmark_from_kernel_efficiency_to_distributed_context_p.md)

</div>

<!-- RELATED:END -->
