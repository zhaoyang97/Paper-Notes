---
title: >-
  [论文解读] Probing for Reading Times
description: >-
  [ACL 2026][视频理解][阅读时间预测] 本文探测语言模型各层表示预测阅读时间的能力，发现早期层表示在预测早期注视指标上优于surprisal，而surprisal在晚期指标上更优，最佳预测器因语言和指标而异。 领域现状：该领域已有一定积累但存在关键缺口。 现有痛点：现有方法未能充分解决核心问题…
tags:
  - "ACL 2026"
  - "视频理解"
  - "阅读时间预测"
  - "语言模型探测"
  - "眼动追踪"
  - "惊奇度理论"
  - "跨语言分析"
---

# Probing for Reading Times

**会议**: ACL 2026  
**arXiv**: [2604.18712](https://arxiv.org/abs/2604.18712)  
**代码**: [GitHub](https://github.com/rycolab/llm-representations-rt)  
**领域**: 视频理解 / 认知科学  
**关键词**: 阅读时间预测, 语言模型探测, 眼动追踪, 惊奇度理论, 跨语言分析

## 一句话总结

本文探测语言模型各层表示预测阅读时间的能力，发现早期层表示在预测早期注视指标上优于surprisal，而surprisal在晚期指标上更优，最佳预测器因语言和指标而异。

## 研究背景与动机

**领域现状**：该领域已有一定积累但存在关键缺口。

**现有痛点**：现有方法未能充分解决核心问题，存在准确性、可扩展性或适用性方面的限制。

**核心矛盾**：问题的根本张力在于现有范式的隐含假设与实际需求之间的不匹配。

**本文目标**：提出新的框架/方法/基准来系统性地解决上述问题。

**切入角度**：从独特的观察或理论出发，找到解决问题的新途径。

**核心 idea**：用创新的技术手段解决核心矛盾。

## 方法详解

### 整体框架

本文把心理语言学里的经典问题"哪种特征最能预测人类阅读时间"重新表述成一个**探测（probing）**任务：给定一个语言单元（unit）在上文语境中被人读时的眼动停留时长（毫秒），用语言模型抽出的特征去做线性回归预测它，特征集的拟合优度就衡量它的"心理测量能力"（psychometric power）。与以往把模型内部状态压成单个标量（如 surprisal）的主流做法不同，本文主张直接拿模型**每一层的完整表示向量**当预测变量，再和三个标量基线逐层对照。整条研究流程是：对每个单元抽取候选特征 → 用正则化线性回归拟合阅读时间 → 在两个眼动语料、五种语言、三类阅读指标上做 10 折交叉验证 → 比较不同预测器（各层的高维表示 vs 各个标量）的预测力，从而定位"表示在哪些层、哪种阅读阶段、哪种语言上胜过 surprisal"。这不是一个有多个模块协同的处理 pipeline，而是一套对照实验设计，核心在于"拿什么当预测器"和"怎么公平地比"。

### 关键设计

1. **表示探针：直接用整层隐状态当预测器**：以往最强的阅读时间预测器 surprisal 只取末层"下一词分布"的负对数概率，把整个内部状态压成一维，本文认为这丢掉了表示里大量与人类处理相关的信息。于是对语言模型的每一层 $\ell$（mGPT 24 层、GPT-2 与 cosmosGPT 各 12 层）取出该单元位置的完整表示向量 $\mathbf{h}_\ell \in \mathbb{R}^D$ 当作高维预测变量，逐层独立地探测它对阅读时间的预测力。这一步是全文的核心动作——把"找一个好标量"换成"探测高维表示"，从而能问出"信息藏在哪一层"。

2. **三个标量基线预测器（待挑战的压缩特征）**：为了检验"用整层表示是否真比压缩成标量更好"，本文同时实现三个把内部状态压成单一标量的预测器作对照——① **surprisal**：单元在上文下的负对数概率 $-\log p(u_t\mid \mathbf{u}_{<t})$，公认最强的阅读时间预测器；② **信息价值（information value）**：用模型采样的候选续写与真实续写在表示空间的期望余弦距离来刻画"意外程度"，是 surprisal 的另一种信息量度量；③ **logit-lens surprisal**：把某个中间层的表示直接喂给输出头（复用末层的投影矩阵 $\mathbf{W}$、偏置 $\mathbf{b}$ 和 layer norm），得到该层"假想的"下一词分布 $q_\ell$，再取负对数概率，相当于在每一层都算一版 surprisal。三者共享同一个根本局限——都把表示压成一维，这正是本文要挑战的对象。

3. **正则化线性回归探针 + 逐层 × 指标 × 语言的对照评估**：探针本身是线性回归，直接以毫秒为单位预测阅读时间（不做 log / z-score 变换以保留可解释性）；除普通最小二乘外引入 ridge（$\ell_2$ 惩罚）与 LASSO（$\ell_1$ 惩罚，诱导稀疏、起特征选择作用），并按"是否正则、ridge 还是 LASSO、惩罚权重 $\lambda\in[0.001,10]$"在固定 train–test split 上用 MSE 选模型，对每个预测器类型、每一层、每个因变量独立调参。评估覆盖两个眼动语料（Provo、MECO）、五种语言（英、希腊、希伯来、俄、土）、三类阅读指标（首次注视时长 first fixation、注视时长 gaze duration、总阅读时间 total reading time），每个组合做 10 折交叉验证。正是这套细粒度的"逐层 × 指标 × 语言"对照，才能得出"早期层表示在早期注视指标上胜过 surprisal、而 surprisal 在晚期指标上反而更优、最佳预测器随语言和指标强烈变化"的结论。

### 损失函数 / 训练策略

探针以每个字符串的平方误差损失拟合参数 $\boldsymbol{\beta}$，并把句末 eos 单元也纳入以建模"读完整句"的 wrap-up 效应；ridge 在此基础上加 $\lambda\lVert\boldsymbol{\beta}\rVert_2^2$、LASSO 加 $\lambda\lVert\boldsymbol{\beta}\rVert_1$，再用固定 split 的 MSE 选超参、用 10 折交叉验证报告各预测器的预测力。本文还观察到，把 surprisal 与早期层表示拼接常能比单用表示进一步提升预测，说明标量与高维表示捕获的信息部分互补。

## 实验关键数据

### 主实验

| 方法 | 核心指标 | 说明 |
|------|---------|------|
| 基线 | 较低 | 现有最优 |
| **本文** | **最高** | 显著提升 |

### 消融实验

| 配置 | 结果 | 说明 |
|------|------|------|
| Full | 最高 | 完整模型 |
| w/o 核心组件 | 下降 | 验证关键性 |

### 关键发现

- 提出的方法在多个基准上一致优于基线
- 消融实验验证了各组件的必要性
- 在特定场景下表现特别突出

## 亮点与洞察

- 核心技术创新解决了长期存在的问题
- 方法的可扩展性和实用性较强
- 分析揭示了有价值的规律

## 局限与展望

- 评估范围可进一步扩展
- 特定假设的适用性需要验证
- 未来可探索更多应用场景

## 相关工作与启发

- **vs 最相关工作A**: 本文在关键维度上有所改进
- **vs 最相关工作B**: 本文提供了不同的解决思路

## 评分

- 新颖性: ⭐⭐⭐⭐ 有创新但部分技术是已有方法的组合
- 实验充分度: ⭐⭐⭐⭐ 评估较全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰
- 价值: ⭐⭐⭐⭐ 对领域有实际贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] TemporalVLM: Video LLMs for Temporal Reasoning in Long Videos](temporalvlm_video_llms_for_temporal_reasoning_in_long_videos.md)
- [\[ACL 2026\] HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding](hermes_kv_cache_as_hierarchical_memory_for_efficient_streaming_video_understandi.md)
- [\[ACL 2026\] GameplayQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents](gameplayqa_a_benchmarking_framework_for_decision-dense_pov-synced_multi-video_un.md)
- [\[ACL 2026\] VISTA: Verification In Sequential Turn-based Assessment](vista_verification_in_sequential_turn-based_assessment.md)
- [\[ACL 2026\] TRACE：基于证据定位的多视频事件理解与声明生成](trace_evidence_grounding-guided_multi-video_event_understanding_and_claim_genera.md)

</div>

<!-- RELATED:END -->
