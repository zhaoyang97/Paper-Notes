---
title: >-
  [论文解读] Decomposition of Small Transformer Models
description: >-
  [NeurIPS 2025 (Workshop: Mechanistic Interpretability)][时间序列][参数空间分解] 将 Stochastic Parameter Decomposition (SPD) 扩展到 Transformer，设计适用于序列数据的因果重要性函数和新损失函数，在玩具 induction head 上恢复期望两步电路，在 GPT-2-small 上定位到"高尔夫""篮球"等可解释概念对应的 rank-1 参数子空间。
tags:
  - "NeurIPS 2025 (Workshop: Mechanistic Interpretability)"
  - "时间序列"
  - "参数空间分解"
  - "Stochastic Parameter Decomposition"
  - "Induction Head"
  - "GPT-2"
  - "因果重要性"
---

# Decomposition of Small Transformer Models

**会议**: NeurIPS 2025 (Workshop: Mechanistic Interpretability)  
**arXiv**: [2511.08854](https://arxiv.org/abs/2511.08854)  
**代码**: 无（基于 SPD 开源框架扩展）  
**领域**: 时间序列  
**关键词**: 参数空间分解, Stochastic Parameter Decomposition, Induction Head, GPT-2, 因果重要性

## 一句话总结
将 Stochastic Parameter Decomposition (SPD) 扩展到 Transformer，设计适用于序列数据的因果重要性函数和新损失函数，在玩具 induction head 上恢复期望两步电路，在 GPT-2-small 上定位到"高尔夫""篮球"等可解释概念对应的 rank-1 参数子空间。

## 研究背景与动机

**领域现状**：机械化可解释性两波浪潮——第一波理解单神经元受制于多义性；第二波转向激活空间，SAE 发现大量可解释概念。但 SAE 存在特征吸收/分裂问题。

**现有痛点**：激活空间方法只回答"给定输入什么被激活"，无法将模型本身分解为少量可复用机制。参数空间方法理论上更根本——梯度下降直接将机制写入权重。

**核心矛盾**：SPD 此前仅在玩具模型验证，无法处理序列数据（Transformer），玩具到"真实模型"的鸿沟未跨越。

**本文目标** 将 SPD 扩展到 Transformer，验证参数空间分解能否恢复已知电路和发现可解释子组件。

**切入角度**：SPD 将权重分解为稀疏 rank-1 矩阵 $W_c^l = \vec{U_c^l} \otimes \vec{V_c^l}$，学习因果重要性函数。针对序列位置依赖性设计新因果重要性计算。

**核心 idea**：引入位置感知注意力因果重要性函数和部分重建损失，使 SPD 能分解 Transformer 并提取可解释参数空间机制。

## 方法详解

### 整体框架
SPD 将 $W^l$ 分解为 $C$ 个 rank-1 子组件。组装权重 $W'^l = \sum_{c} \alpha \cdot W_c^l$，$\alpha \in [0,1]$ 由因果重要性 $g_c^l(x)$ 控制。目标：忠实性（子组件求和恢复原权重）+ 最小性（尽量少激活）。

### 关键设计

1. **序列感知因果重要性函数**:

    - 功能：为序列中不同位置分配不同因果重要性
    - 核心思路：在 $\gamma$-MLP 前加最小注意力网络（1 head, 1 layer），用学习的相对位置编码做交叉位置注意力：$g_{c,n}^l = \sigma_H(\gamma_c^l(\bar{x}_n))$，$\bar{x}_n = (\text{softmax}(\frac{q_n K^\top + r_n}{\sqrt{d_k}})V) \oplus x_n$
    - 设计动机：原始 SPD 各位置独立计算重要性，但序列模型中同一 token 不同位置重要性不同（"bank" 在 "river bank" vs "bank manager" 中不同）。OV 电路中相同 value 可能被不等 attend

2. **部分重建损失**:

    - 功能：防止分解模型在未用组件中"作弊"
    - 核心思路：$\mathcal{L}_{\text{partial}} = D_{KL}(f(x|W^1,...,W^{l\in\mathcal{M}}(x,g^l(x)),...,W^L), f(x|W))$，随机只替换部分层权重为分解版
    - 设计动机：小样本分解大模型时，未用组件可能被改写为快捷方式。部分重建迫使每层分解独立可替换

3. **忠实性与最小性损失**:

    - 功能：核心训练目标
    - 核心思路：忠实性 $\mathcal{L}_{faith} = \frac{1}{N}\sum_{l}\sum_{i,j}(W_{i,j}^l - \sum_c U_{i,c}^l V_{c,j}^l)^2$；最小性 $\mathcal{L}_{min} = \sum_l\sum_c |g_c^l(x)|^p$；随机重建用 $\alpha \sim \mathcal{U}(g_c^l(x), 1)$ 保证重要性为 0 的子组件仍有梯度信号
    - 设计动机：随机采样是双重作用——给"关闭"的子组件梯度通路，同时通过尽可能少组件重建原模型输出来设定因果重要性下界

## 实验关键数据

### Induction Head 分解

| 组件 | 唯一子组件数 | 关键位置激活 |
|------|------------|------------|
| $Q_0$ | 1 | $m$ 位置 (1.0) |
| $K_0$ | 1 | $s_1$ 位置 (1.0) |
| $V_0$ | 1 | $s_1$ 位置 (1.0) |
| $Q_1$ | 1 | $s_2$ 位置 (1.0) |
| $K_1$ | 1 | $m$ 位置 (1.0) |
| $V_1$ | 11 | $m$ 位置 (5.053) |

$\mathcal{L}_{faithful} = 3 \times 10^{-9}$，$\mathcal{L}_{recon} = 1 \times 10^{-4}$

### GPT-2-small 分解

| 指标 | 值 |
|------|-----|
| 总活跃子组件 | 96（原模型 99% 缩减） |
| "obe"+"Bryant" 抑制 | basketball 概率显著下降 |
| "Woods" 抑制 | golf 概率显著下降 |
| 反向保留 | "golf 最著名运动员" 仍正确回答 Tiger Woods |

### 关键发现
- Induction head 恢复期望两步电路：Layer 0 让 $m$ attend $s_1$（学习"跟在 $s$ 后"），Layer 1 让 $s_2$ attend $m$
- $V_1$ 需 11 个子组件：$m$ 身份在 128 token 中需高于 rank-1 信息
- GPT-2 "Kobe Bryant -> basketball" 知识早在 Layer 0 MLP 写入残差流，与 Meng et al. 因果追踪互补
- 知识存储不对称：抑制 "athlete->sport" 不影响 "sport->athlete"

## 亮点与洞察
- **参数空间因果手柄**：SPD 的 rank-1 方向是精准的——抑制特定方向选择性降低目标概率不影响其他样本，比激活空间方法更精准
- **部分重建损失**解决"小样本分解大模型"难题，迫使分解反映原模型，可迁移到稀疏化/分解场景

## 局限与展望
- 仅小模型验证（2 层玩具 + GPT-2-small），LLaMA/Mistral 等大模型可扩展性未知
- 因果重要性参数化引入额外计算和内存开销（注意力网络 per subcomponent）
- GPT-2 实验仅 2 个样本，缺乏系统的定量评估和对比基准
- 非线性交互（GELU、LayerNorm、残差连接）的影响未充分分析
- 未与 SAE、activation patching 等现有方法在相同任务上做头对头对比
- 分解粒度（子组件数 C）需手动选择，自动确定最优 C 是开放问题

## 相关工作与启发
- **vs SAE**: SAE 在激活空间工作发现可解释特征但有吸收/分裂问题；SPD 在参数空间分解为 rank-1 机制，互补视角
- **vs ROME**: ROME 因果追踪定位中间层 MLP 为编辑点；SPD 发现信息早在 Layer 0 MLP 就存在，编辑点不等于存储点
- **vs APD**: APD 用 batch top-k 硬编码稀疏性；SPD 用学习的因果重要性更灵活
- **vs L3D**: L3D 通过梯度重建学习稀疏活动参数方向，允许更高秩（Tucker 分解）；SPD 保持 rank-1 约束更可解释

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将参数空间分解扩展到 Transformer，序列因果重要性和部分重建损失有意义
- 实验充分度: ⭐⭐⭐ Workshop 规模限制，实验主要定性
- 写作质量: ⭐⭐⭐⭐ 动机方法清晰，定位准确
- 价值: ⭐⭐⭐⭐ 参数空间可解释性在 Transformer 上的重要一步

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Transformer Embeddings for Fast Microlensing Inference](transformer_embeddings_for_fast_microlensing_inference.md)
- [\[ACL 2025\] Revisiting LLMs as Zero-Shot Time-Series Forecasters: Small Noise Can Break Large Models](../../ACL2025/time_series/revisiting_llms_as_zero-shot_time_series_forecasters_small_noise_can_break_large.md)
- [\[ICLR 2026\] Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data](../../ICLR2026/time_series/relational_transformer_toward_zero-shot_foundation_models_for_relational_data.md)
- [\[ICML 2026\] Incremental Transformer Neural Processes](../../ICML2026/time_series/incremental_transformer_neural_processes.md)
- [\[ICLR 2026\] TimeSliver: Symbolic-Linear Decomposition for Explainable Time Series Classification](../../ICLR2026/time_series/timesliver_symbolic-linear_decomposition_for_explainable_time_series_classificat.md)

</div>

<!-- RELATED:END -->
