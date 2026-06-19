---
title: >-
  [论文解读] Searching Latent Program Spaces
description: >-
  [NeurIPS 2025 Spotlight][代码智能][Latent Program Network] 提出 Latent Program Network（LPN），通过编码器将输入-输出示例映射为潜在程序表示，在测试时通过梯度搜索潜在空间来适应新任务，在 ARC-AGI 基准上显著优于 in-context learning 和 test-time training 方法。
tags:
  - "NeurIPS 2025 Spotlight"
  - "代码智能"
  - "Latent Program Network"
  - "程序合成"
  - "测试时搜索"
  - "ARC-AGI"
  - "变分推理"
---

# Searching Latent Program Spaces

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2411.08706](https://arxiv.org/abs/2411.08706)  
**代码**: 有（推测开源，基于 re-ARC 数据集训练）  
**领域**: 程序合成 / 少样本学习 / 测试时自适应  
**关键词**: Latent Program Network, 程序合成, 测试时搜索, ARC-AGI, 变分推理

## 一句话总结

提出 Latent Program Network（LPN），通过编码器将输入-输出示例映射为潜在程序表示，在测试时通过梯度搜索潜在空间来适应新任务，在 ARC-AGI 基准上显著优于 in-context learning 和 test-time training 方法。

## 研究背景与动机

通用智能的核心是**技能获取效率**——用最少经验学习新任务并泛化到训练分布之外。当前方法面临两难：

**归纳式方法（Program Synthesis）**: 在领域专用语言（DSL）中搜索程序，泛化能力强但搜索空间爆炸，且需要人工设计 DSL，不可扩展。

**转导式方法（In-Context Learning）**: 神经网络直接从示例预测输出，可扩展但缺乏结构化测试时适应能力，且无法保证与给定示例的**一致性**（consistency）。

Test-Time Training（TTT）通过在测试时微调参数解决一致性问题，但在完整参数空间搜索计算代价高且容易过拟合。

LPN 的核心洞察：**在紧凑的潜在空间中搜索程序比在参数空间中微调更高效、更不容易过拟合**。

## 方法详解

### 整体框架

LPN 由三个核心组件构成：
1. **编码器** $q_\phi(z|x,y)$: 将输入-输出对映射到潜在程序分布
2. **潜在空间优化** $z' = f(p_\theta, z, x, y)$: 在潜在空间中搜索更好的程序表示
3. **解码器** $p_\theta(y|x,z')$: 根据潜在程序和新输入生成输出

推理流程为：编码器生成直觉性初始猜测（System 1）→ 潜在空间优化搜索更好的假设（System 2）→ 解码器执行程序。

### 关键设计

1. **概率编码器与均值聚合**: 编码器将每个 I/O 对独立编码为高斯分布 $q_\phi(z|x,y) = \mathcal{N}(\mu, \Sigma)$。对同一任务的多个 I/O 对，取潜在变量的均值聚合 $z = \frac{1}{n-1}\sum_{j \neq i} z_j$，这保证了对示例顺序的置换不变性，避免了 Transformer 自注意力的二次复杂度。设计动机是：给定一个 I/O 对，有多个可能的程序能解释它，概率建模自然处理这一多义性。

2. **梯度上升搜索（Gradient Ascent in Latent Space）**: 测试时从编码器初始化 $z'_0$ 出发，迭代优化：
    $z'_k = z'_{k-1} + \alpha \cdot \nabla_z \sum_{i=1}^{n} \log p_\theta(y_i | x_i, z)\Big|_{z=z'_{k-1}}$
   搜索目标是最大化解码器对所有给定 I/O 对的对数似然。相比 Test-Time Training 在完整参数空间微调，LPN 只在低维潜在空间（如 256 维）搜索，效率高且不易过拟合。选择最佳似然的 $z'$ 而非最后一步的。

3. **训练时感知测试时搜索**: 训练时也执行少量梯度搜索步（如 1 步），使潜在空间专门为测试时搜索优化。实验表明，训练时用 Grad 1 使测试时 100 步搜索达到 99.5% 准确率，而训练时 Grad 0 仅 67.5%。使用 stop-gradient 通过潜在更新来减少计算开销且效果更好。

### 损失函数 / 训练策略

训练使用变分目标，包含重建损失和 KL 散度：

$$\mathcal{L}_{\text{total}}(\phi, \theta) = \mathcal{L}_{\text{rec}}(\phi, \theta) + \beta \mathcal{L}_{\text{KL}}(\phi)$$

- 重建损失：$\mathcal{L}_{\text{rec}} = \sum_{i=1}^{n} -\log p_\theta(y_i | x_i, z'_i)$（交叉熵）
- KL 损失：$\mathcal{L}_{\text{KL}} = \sum_{i=1}^{n} D_{\text{KL}}(q_\phi(z|x_i,y_i) \| \mathcal{N}(0,I))$

关键训练技巧：重建第 $i$ 个输出时，用其他 $n-1$ 个 I/O 对的编码来推断程序，避免编码器直接压缩输出（shortcut）。

## 实验关键数据

### 主实验（Pattern 任务，Exact Match %）

| 训练方式 \ 推理步数 | Grad 0 | Grad 1 | Grad 5 | Grad 20 | Grad 100 |
|-------------------|--------|--------|--------|---------|----------|
| Grad 0 | 3.2 | 3.6 | 18.8 | 52.5 | 67.5 |
| **Grad 1** | **8.6** | **44.6** | **85.4** | **98.4** | **99.5** |
| Grad 5 | 0.0 | 0.4 | 31.9 | 88.5 | 98.1 |
| In-Context | 77.7 | — | — | — | — |

### ARC-AGI 2024 结果

| FLOPs | LPN (ID) | TTT (ID) | LPN (OOD) | TTT (OOD) |
|-------|----------|----------|-----------|-----------|
| 2E+11 | 68.75 | 45.75 | 7.75 | 5.85 |
| 2E+12 | 75.95 | 51.75 | 10.25 | 7.35 |
| 2E+13 | 80.00 | 58.50 | **15.25** | 13.50 |
| 2E+14 | 76.25 | 58.75 | 15.50 | 16.00 |
| 2E+15 | 78.50 | 57.00 | 15.50 | 15.25 |

### 消融实验（OOD Pattern 任务）

| 方法 | Grad 0 | Grad 10 | Grad 100 | 说明 |
|------|--------|---------|----------|------|
| In-Context | 0.0 | — | — | 无测试时适应 |
| TTT | 0.0 | 1.8 | 0.3 | 过拟合 |
| LPN Grad 0 | 0.3 | 18.8 | 41.1 | 基础版 |
| **LPN Grad 1** | **0.0** | **59.9** | **88.0** | **最优** |

### 关键发现

- **测试时搜索带来质变**：LPN 在 OOD 任务上，开启梯度搜索后性能翻倍（7.75→15.25% on ARC-AGI）
- **训练时包含梯度步至关重要**：Grad 1 训练使潜在空间被优化为支持高效搜索（99.5% vs 67.5%）
- **TTT 在小模型上容易过拟合**：100 步 TTT 的 OOD 性能从 1.8% 降至 0.3%，而 LPN 持续提升
- **编码器初始化不可或缺**：从先验 $p(z)$ 而非编码器初始化搜索，性能大幅下降
- **规格大小泛化**：LPN 平滑泛化到比训练时更多的 I/O 对，而 In-Context Learning 过拟合到训练时的规格大小
- **比预训练 LLM 方法更高效**：178M 参数的 LPN 超越了 CodeT5（220M）和 text-davinci（175B）在 ARC-AGI 上的表现

## 亮点与洞察

- **在潜在空间搜索程序是核心创新**——结合了归纳方法的适应性和转导方法的可扩展性
- System 1 / System 2 的类比非常直观：编码器是快速直觉，潜在优化是深思熟虑
- 均值聚合的置换不变性避免了注意力的二次复杂度，使方法可扩展到更多示例
- 训练时感知测试时搜索（train-with-search-awareness）是使方法奏效的关键归纳偏置

## 局限与展望

- 训练数据分布有限——仅在 re-ARC 上训练，限制了潜在空间的表达能力
- 梯度搜索是一阶方法，可能陷入局部最优；可探索进化策略（CMA-ES）等全局搜索
- 程序表示为连续向量，牺牲了可解释性
- 在高计算预算下，TTT 可能追上 LPN（ARC-AGI OOD 2E+14 FLOPs 时两者持平）
- 组合泛化能力有限——虽然有成功案例但不够鲁棒

## 相关工作与启发

- LEAPS 在 Karel 程序空间中进行潜在搜索，LPN 将其推广到更一般的任务
- Neural Processes (Garnelo et al.) 提供了核心架构灵感——引入瓶颈学习隐式表示
- 半摊销变分推理（Semi-Amortized VI）提供了理论视角：编码器的摊销推理 + 潜在优化弥合摊销差距
- LEO (Rusu et al.) 在潜在空间做 MAML，但需要超网络生成参数；LPN 利用 Transformer 的 in-context 能力直接条件化

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 潜在程序空间搜索的思路原创且优雅
- 实验充分度: ⭐⭐⭐⭐ — Pattern/String/ARC-AGI 多任务验证，丰富消融
- 写作质量: ⭐⭐⭐⭐ — 清晰的动机阐述和系统的实验分析
- 价值: ⭐⭐⭐⭐ — 为测试时自适应提供了新的范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Program Synthesis via Test-Time Transduction](program_synthesis_via_test-time_transduction.md)
- [\[NeurIPS 2025\] Once Upon an Input: Reasoning via Per-Instance Program Synthesis](once_upon_an_input_reasoning_via_per-instance_program_synthesis.md)
- [\[ACL 2025\] Program Synthesis Benchmark for Visual Programming in XLogoOnline Environment](../../ACL2025/code_intelligence/program_synthesis_benchmark_for_visual_programming_in_xlogoonline_environment.md)
- [\[ICCV 2025\] TikZero: Zero-Shot Text-Guided Graphics Program Synthesis](../../ICCV2025/code_intelligence/tikzero_zero-shot_text-guided_graphics_program_synthesis.md)
- [\[ICML 2026\] HE-SNR: Uncovering Latent Logic via Entropy for Guiding Mid-Training on SWE-bench](../../ICML2026/code_intelligence/he-snr_uncovering_latent_logic_via_entropy_for_guiding_mid-training_on_swe-bench.md)

</div>

<!-- RELATED:END -->
