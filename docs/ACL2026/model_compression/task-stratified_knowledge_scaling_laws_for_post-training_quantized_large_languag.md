---
title: >-
  [论文解读] Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs
description: >-
  [ACL 2026 Findings][模型压缩][后训练量化] 本文建立了首个面向后训练量化（PTQ）的任务分层知识缩放定律，将 LLM 能力分为记忆/应用/推理三层，统一建模模型大小、位宽、组大小和校准集大小四个因素，在 293 种 PTQ 配置上验证，揭示推理对精度敏感、应用随规模提升、记忆对校准敏感的差异化规律。
tags:
  - "ACL 2026 Findings"
  - "模型压缩"
  - "后训练量化"
  - "缩放定律"
  - "知识分层"
  - "记忆应用推理"
  - "细粒度量化因素"
---

# Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs

**会议**: ACL 2026 Findings  
**arXiv**: [2508.18609](https://arxiv.org/abs/2508.18609)  
**代码**: 无  
**领域**: 模型压缩 / 量化  
**关键词**: 后训练量化, 缩放定律, 知识分层, 记忆应用推理, 细粒度量化因素

## 一句话总结

本文建立了首个面向后训练量化（PTQ）的任务分层知识缩放定律，将 LLM 能力分为记忆/应用/推理三层，统一建模模型大小、位宽、组大小和校准集大小四个因素，在 293 种 PTQ 配置上验证，揭示推理对精度敏感、应用随规模提升、记忆对校准敏感的差异化规律。

## 研究背景与动机

**领域现状**：PTQ 已成为 LLM 压缩的主流策略（~70% 量化相关研究聚焦 PTQ）。现有缩放定律（如 Chinchilla）主要描述全精度模型的行为，少数量化缩放定律仅考虑模型大小和位宽。

**现有痛点**：(1) 忽略了组大小和校准集大小等细粒度 PTQ 参数的系统性影响；(2) 将所有任务的性能混在一起，无法捕捉量化对记忆、应用和推理能力的差异化影响。

**核心矛盾**：现有缩放定律无法指导"在低位量化下如何配置组大小和校准集大小以保持特定能力"这类实际问题。

**本文目标**：建立统一的四因素幂律框架，为三层知识能力分别拟合缩放定律。

**切入角度**：基于 Bloom's Taxonomy 将 LLM 能力分为记忆（精确事实回忆）、应用（灵活知识运用）和推理（多步逻辑），用 14 个基准测试覆盖三层。

**核心 idea**：$-\ln(\text{Acc}_{\text{adj}}) = A \cdot N^{\alpha} \cdot (\log_2 B)^{\beta} \cdot (\log_2 C_b)^{\gamma} \cdot G^{\delta}$，其中指数 $\alpha, \beta, \gamma, \delta$ 是任务层级特定的，量化了不同能力对各因素的敏感度。

## 方法详解

### 整体框架

对 Qwen3 系列（0.6B-14B）+ Llama-3 系列做系统性 PTQ 配置扫描（位宽 3/4/8、组大小 32/64/128/1024、校准集大小 8/32/128/1024），共 293 种配置。用 GPTQ 作为统一量化方法，在 14 个基准上评估后用 OLS 回归拟合对数变换的幂律。

### 关键设计

**1. 四因素统一幂律框架：把模型大小、位宽、校准集、组大小放进同一个可解释的公式**

现有量化缩放定律只看模型大小 $N$ 和位宽 $B$，把组大小、校准集这些"细粒度旋钮"当噪声忽略掉，于是没法回答"低位时该怎么配组大小和校准集"这类落地问题。本文把四个因素一次性塞进 $-\ln(\text{Acc}_{\text{adj}}) = A \cdot N^{\alpha} \cdot (\log_2 B)^{\beta} \cdot (\log_2 C_b)^{\gamma} \cdot G^{\delta}$：对位宽 $B$ 和校准集 $C_b$ 取对数，是因为多一个 bit、多一倍校准数据带来的边际收益是递减的，对数恰好刻画这种饱和；再用 $-\ln(\cdot)$ 把有界的 accuracy 拉到无界的"损失"空间，让幂律拟合所需的单调凸性成立。

拟合前先做基线调整 $\text{Acc}_{\text{adj}} = \frac{\text{Acc} - \text{Acc}_{\text{random}}}{1 - \text{Acc}_{\text{random}}}$，把不同任务随机猜测的基线差异抹掉——否则二分类任务 50% 起步、四选一 25% 起步会污染跨任务的指数比较。最后对整条对数变换后的方程做 OLS 回归，拟合出的指数 $\alpha,\beta,\gamma,\delta$ 可以直接读作弹性系数：每个因素相对变化 1% 时性能损失的相对变化，敏感度一目了然。

**2. 任务分层知识体系：把"能力"拆成记忆/应用/推理三层各自拟合，而不是混成一个平均分**

只拟合聚合性能会掩盖最关键的差异：量化到某个位宽时，推理可能已经崩溃，但应用类任务还看着挺好，平均分却显示"问题不大"。本文借 Bloom's Taxonomy 把 14 个基准分成三层——L1 记忆（TriviaQA/NQ/LAMA 等精确事实回忆）、L2 应用（MMLU/Hellaswag 等灵活知识运用）、L3 推理（GSM8K/ARC-C 等多步逻辑），对每一层单独拟合一套 $\alpha,\beta,\gamma,\delta$。这样得到的不是一条曲线，而是三层各自对模型大小、位宽、校准集、组大小的敏感度画像，才能回答"低位量化下哪种能力先垮"。

**3. 低位场景下细粒度因素的关键作用：证明组大小和校准集在 2-3 bit 不是可选项而是防崩溃的必需品**

实践者做低位量化时往往沿用默认组大小和校准集，以为这些参数无关紧要——本文的消融正是要戳破这一点。只用 $f(N,B)$ 两因素拟合时 $R^2 = 0.91$，加入组大小 $G$ 后跳到 0.95，组大小独立解释了约 4% 的额外方差；而这 4% 并非均匀分布，恰好集中在低位区域。换句话说，高位时组大小确实近似冗余，但到了 3-bit、2-bit，默认配置可能直接把某类能力推下悬崖，这些细粒度因素从"调优余地"变成了"防崩溃的安全垫"。

### 损失函数 / 训练策略

不涉及训练。GPTQ 用 Hessian 矩阵逐层最小化量化重建误差。

## 实验关键数据

### 主实验

**各层知识能力的缩放指数对比**

| 能力层 | α(N) | β(B) | γ(Cb) | δ(G) | Adj R² |
|--------|------|------|-------|------|--------|
| 通用 | -0.359 | -1.067 | -0.032 | 0.073 | 0.9475 |
| L1 记忆 | -0.315 | -0.964 | **-0.040** | 0.064 | 0.9350 |
| L2 应用 | **-0.400** | -1.100 | -0.030 | 0.075 | 0.9500 |
| L3 推理 | -0.320 | **-1.200** | -0.025 | **0.085** | 0.9300 |

### 关键发现

- 推理精度瓶颈：$\beta_{\text{KR}} = -1.200$（最大绝对值），说明推理对位宽最敏感
- 应用规模响应：$\alpha_{\text{KA}} = -0.400$（最大绝对值），说明应用能力随模型规模显著提升
- 记忆校准敏感：$\gamma_{\text{KM}} = -0.040$（最大绝对值），说明精确事实回忆对校准数据量最敏感
- 四因素模型比二因素基线（N,B）提升 3.5% 的 Adj R²，且 Qwen3-32B 上的外推验证成功

## 亮点与洞察

- 任务分层缩放定律的思路非常有实用价值——指导实践者在资源约束下做出知情的量化配置决策
- 细粒度因素在低位场景的关键性是重要发现——默认配置在 3-bit 下可能导致能力崩溃
- 跨架构（Qwen3→Llama-3）的一致性证明了定律的普适性

## 局限与展望

- 仅使用 GPTQ 一种量化方法
- 2-bit 数据因性能崩溃被排除在拟合外
- 未考虑 QAT 或混合精度场景
- 未覆盖生成任务的评估

## 相关工作与启发

- **vs Chinchilla Laws**: 针对全精度模型，本文扩展到量化场景并增加组大小/校准集因素
- **vs QiD Laws**: 仅建模聚合退化，本文分层建模三种知识能力

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个任务分层PTQ缩放定律，四因素统一框架
- 实验充分度: ⭐⭐⭐⭐⭐ 293种配置+14基准+跨架构验证+外推测试
- 写作质量: ⭐⭐⭐⭐ 公式推导清晰，图表有说服力
- 价值: ⭐⭐⭐⭐⭐ 对LLM量化实践有直接指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] LLMs as Noisy Channels: A Shannon Perspective on Model Capacity and Scaling Laws](../../ICML2026/model_compression/llms_as_noisy_channels_a_shannon_perspective_on_model_capacity_and_scaling_laws.md)
- [\[ACL 2026\] TELL-TALE: Task Efficient LLMs with Task Aware Layer Elimination](tell-tale_task_efficient_llms_with_task_aware_layer_elimination.md)
- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](../../ICML2026/model_compression/model_merging_scaling_laws_in_large_language_models.md)
- [\[ACL 2026\] WISCA: A Lightweight Model Transition Method to Improve LLM Training via Weight Scaling](wisca_a_lightweight_model_transition_method_to_improve_llm_training_via_weight_s.md)
- [\[AAAI 2026\] Stratified Knowledge-Density Super-Network for Scalable Vision Transformers](../../AAAI2026/model_compression/stratified_knowledge-density_super-network_for_scalable_vision_transformers.md)

</div>

<!-- RELATED:END -->
