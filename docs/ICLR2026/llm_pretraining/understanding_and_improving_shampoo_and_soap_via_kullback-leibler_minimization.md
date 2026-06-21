---
title: >-
  [论文解读] Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization
description: >-
  [预训练] 从 KL 散度最小化角度重新解释 Shampoo 和 SOAP 的结构化二阶矩估计，揭示其固有局限，并提出 KL-Shampoo 和 KL-SOAP 两种实用方案，在无需 Adam grafting 的情况下匹配或超越原始方法。 核心问题 Shampoo 及其高效变体 SOAP 使用 Kronecker 结构的二…
tags:
  - "预训练"
---

# Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization

## 论文信息
- **会议**: ICLR 2026
- **arXiv**: [2509.03378](https://arxiv.org/abs/2509.03378)
- **代码**: [https://github.com/yorkerlin/KL-Methods](https://github.com/yorkerlin/KL-Methods)
- **领域**: LLM预训练
- **关键词**: Shampoo, SOAP, KL 散度, Kronecker 结构, 二阶优化, 协方差估计

## 一句话总结
从 KL 散度最小化角度重新解释 Shampoo 和 SOAP 的结构化二阶矩估计，揭示其固有局限，并提出 KL-Shampoo 和 KL-SOAP 两种实用方案，在无需 Adam grafting 的情况下匹配或超越原始方法。

## 研究背景与动机

### 核心问题
Shampoo 及其高效变体 SOAP 使用 Kronecker 结构的二阶矩估计进行预条件优化。然而：
1. Shampoo 通常需要与 Adam 的 step-size grafting 才能达到竞争力
2. SOAP 通过在 Shampoo 特征基下运行 Adam 缓解了这个问题，但引入额外内存开销
3. 先前分析主要基于 Frobenius 范数，忽略了 SPD（对称正定）约束

### 为什么选 KL 散度？
1. KL 散度天然尊重 SPD 约束（Frobenius 范数不尊重）
2. 在拟牛顿方法（BFGS、DFP）中，KL 提供统一解释框架
3. SPD 矩阵的各项不具等价角色，Frobenius 范数等价对待所有项
4. KL 可自然扩展到张量值情况

## 方法详解

### 整体框架

把 Shampoo 那对 Kronecker 因子 $\boldsymbol{S}_a,\boldsymbol{S}_b$ 看成一个矩阵高斯的协方差，于是"估计预条件器"就被改写成"用 KL 散度去逼近真实的梯度二阶矩 $\mathbb{E}[\boldsymbol{g}\boldsymbol{g}^\top]$"。这一视角先解释了原始 Shampoo 为何只是单侧近似、因而离不开 Adam grafting；再顺势导出两因子联合最小化的最优不动点条件，把它变成可在线运行的 EMA 更新，并用 QR 分解替代特征分解把成本压回 SOAP 量级。这条线最终得到无需 grafting 的 KL-Shampoo，同时把 Shampoo、SOAP、Adafactor 都收进同一张"散度 + 预条件结构 + 估计方案"的表里，使它的内存优势变得一目了然。

> 这是一篇优化器分析与改进的论文，方法主体是一串矩阵高斯/KL 推导（重解释 → 联合最优条件 → EMA → QR），并非多模块数据流水线，flowchart 无法有意义地表达矩阵运算，因此不画框架图；下面四个关键设计按推导顺序展开。

### 关键设计

**1. Shampoo 的 KL 重解释：暴露单侧近似的缺口**

论文首先证明（Claim 1）：当指数 $p=1/2$ 时，Shampoo 对单个因子的估计规则恰好是 KL 最小化 $\min_{\boldsymbol{S}_a}\text{KL}(\mathbb{E}[\boldsymbol{g}\boldsymbol{g}^\top],\boldsymbol{S})$ 在 $\boldsymbol{S}=(1/d_b\,\boldsymbol{S}_a)\otimes\boldsymbol{I}_b$ 下的最优解，给出 $\boldsymbol{S}_a^*=\mathbb{E}[\boldsymbol{G}\boldsymbol{G}^\top]$。之所以选 KL 而非以往的 Frobenius 范数，是因为 KL 天然尊重协方差的 SPD（对称正定）约束、且对各项区别对待，而 Frobenius 把所有元素等价看待、会破坏这种结构。问题在于 Shampoo 把两个因子各自独立地这样估计，等于把另一侧固定成单位阵；这种单侧处理无法真正求解两因子联合的 KL 问题，留下的缺口正是它必须靠 Adam step-size grafting 才能稳定收敛的根源。把这套"散度 + 预条件结构 + 估计方案"的拆解并列起来，多个优化器就落进同一张表，KL-Shampoo 与 Adafactor、Frobenius 版 Shampoo 的取舍一目了然：

| 方法 | 散度 | 预条件结构 | 估计方案 |
|------|------|-----------|---------|
| KL-Shampoo | KL | 稠密 Kronecker | 最大似然 |
| Adafactor | von Neumann | 对角 Kronecker | 矩阵矩匹配 |
| F-Shampoo | Frobenius | 稠密 Kronecker | SVD-based |

**2. KL-Shampoo 的联合最优条件：一步到位的矩阵高斯白化**

与其各自为政，不如让两个因子彼此感知。Claim 2 给出联合最小化 $\min_{\boldsymbol{S}_a,\boldsymbol{S}_b}\text{KL}(\mathbb{E}[\boldsymbol{g}\boldsymbol{g}^\top],\boldsymbol{S})$ 的最优解满足一对相互耦合的不动点方程

$$\boldsymbol{S}_a^* = \frac{1}{d_b}\mathbb{E}[\boldsymbol{G}(\boldsymbol{S}_b^*)^{-1}\boldsymbol{G}^\top], \quad \boldsymbol{S}_b^* = \frac{1}{d_a}\mathbb{E}[\boldsymbol{G}^\top(\boldsymbol{S}_a^*)^{-1}\boldsymbol{G}]$$

每个因子都在"已被对侧白化过"的梯度上做估计，因此真正利用了 Kronecker 两侧的耦合信息。这组解还有清晰的统计身份：它正是零均值矩阵高斯的最大似然估计，等价于对梯度做矩阵高斯白化。也正因为最优特征基下梯度已被 Kronecker 完全对角化，再叠一层 Adam 式的对角修正（即 KL-SOAP 的做法）就显得多余——这也预示了后文 KL-Shampoo 反而胜过 KL-SOAP 的结果。

**3. EMA 更新：把不动点变成可在线运行的随机步**

上述不动点无法精确求期望，于是用指数滑动平均在线逼近：

$$\boldsymbol{S}_a \leftarrow (1-\beta_2)\boldsymbol{S}_a + \frac{\beta_2}{d_b}\boldsymbol{G}\boldsymbol{S}_b^{-1}\boldsymbol{G}^\top$$

$\boldsymbol{S}_b$ 对称更新。关键的一点（Claim 3）是这条 EMA 并非随手拼的启发式，而被证明是 KL 目标的一个随机近端梯度步，因此继承了向最优白化解收敛的理论保证。代价是更新里多了一次 $\boldsymbol{G}\boldsymbol{S}_b^{-1}\boldsymbol{G}^\top$ 形式的矩阵乘法，用对侧逆来耦合两个因子。

**4. QR 分解 + EMA 特征值：把成本压回 SOAP 量级**

直接对 $\boldsymbol{S}_a,\boldsymbol{S}_b$ 反复做特征分解太贵，论文改用 QR 分解来维护特征基，使每步迭代的运行时间回落到与 SOAP 相当。但 QR 给出的特征基是"过时"的，直接用瞬时估计的特征值会严重退化，所以对特征值本身也做 EMA 校正：

$$\begin{pmatrix}\boldsymbol{\lambda}_a \\ \boldsymbol{\lambda}_b\end{pmatrix} \leftarrow (1-\beta_2)\begin{pmatrix}\boldsymbol{\lambda}_a \\ \boldsymbol{\lambda}_b\end{pmatrix} + \beta_2\begin{pmatrix}\text{diag}(\boldsymbol{Q}_a^\top \Delta_a \boldsymbol{Q}_a) \\ \text{diag}(\boldsymbol{Q}_b^\top \Delta_b \boldsymbol{Q}_b)\end{pmatrix}$$

即把新信息 $\Delta$ 投影到当前特征基 $\boldsymbol{Q}$ 上、只滑动更新对角的特征值。这个 EMA 特征值方案是 KL-Shampoo 实际能跑赢 SOAP 的关键工程环节；它也直接支持 3D 及以上的张量权重，无需先 reshape 成矩阵。最实在的收益落在内存上：Shampoo 要额外存 $d_a d_b$ 的 Adam 二阶矩做 grafting，SOAP 同样要 $d_a d_b$ 来在特征基里跑 Adam，而 KL-Shampoo 因为不动点本身已给出可用的预条件器，这块额外开销直接归零。

| 方法 | Kronecker | 特征基 | 特征值 | Adam 2nd | 额外开销 |
|------|-----------|--------|--------|----------|---------|
| Shampoo | $d_a^2+d_b^2$ | $d_a^2+d_b^2$ | $d_a+d_b$ | **$d_a d_b$ (grafting)** | 有 |
| SOAP | $d_a^2+d_b^2$ | $d_a^2+d_b^2$ | N/A | **$d_a d_b$ (eigenbasis)** | 有 |
| **KL-Shampoo** | $d_a^2+d_b^2$ | $d_a^2+d_b^2$ | $d_a+d_b$ | **无** | **无** |

## 实验

### 语言模型预训练

使用 150 次随机搜索的公平对比：

| 模型 | KL-Shampoo | SOAP | Shampoo+grafting | Shampoo (无 grafting) |
|------|-----------|------|-----------------|---------------------|
| NanoGPT (123M) | **最低 loss** | 次优 | 第三 | 较差 |
| NanoRWKV7 (162M) | **最低 loss** | 次优 | 中等 | 完全失败 |
| Llama (134M) | **最低 loss** | 次优 | - | - |
| NanoMoE (227M, 3D tensors) | **最低 loss** | 次优 | - | - |

### 关键发现

1. **KL-Shampoo 一致优于 SOAP**：在所有 4 个模型上——出乎意料
2. **KL-Shampoo 无需 grafting**：Shampoo (p=1/2) 不用 grafting 在 RWKV7 的 150 次运行中全部失败
3. **KL-Shampoo 优于 KL-SOAP**：核心原因——在最优特征基下梯度已被 Kronecker 对角化，额外的 Adam 修正是多余的
4. **EMA 特征值方案至关重要**：瞬时估计在使用过时特征基时严重退化
5. **VN-Shampoo (trace scaling) + EMA 方案也能超越 SOAP**

## 亮点

1. **深刻的理论洞察**：KL 视角统一了 Shampoo、SOAP、Adafactor 的解释
2. **实用改进**：消除 Adam 依赖，减少内存，保持 SOAP 级运行时间
3. **KL-Shampoo > KL-SOAP 的解释**：在最优特征基下，矩阵高斯白化已满足，无需进一步对角修正
4. **张量自然扩展**：KL 框架直接支持 3D+ 权重，无需 reshape

## 局限性

1. KL-Shampoo 的 EMA 方案引入了额外的矩阵乘法（$\boldsymbol{G}\boldsymbol{S}_b^{-1}\boldsymbol{G}^\top$）
2. 理论分析假设零均值高斯，实际梯度分布可能不满足
3. 实验主要在 100-200M 规模模型上验证，未测试数十亿参数模型
4. QR 分解在 PyTorch 中不支持半精度，需要精度转换

## 相关工作
- **Shampoo**: Gupta et al. (2018) — 原始 Kronecker 预条件器
- **SOAP**: Vyas et al. (2025a) — 在 Shampoo 特征基上运行 Adam
- **拟牛顿方法**: BFGS/DFP — KL 散度的经典应用
- **二阶优化**: K-FAC, EKFAC — Fisher 信息矩阵近似

## 评分
- **创新性**: ⭐⭐⭐⭐⭐ — KL 视角提供了深刻且统一的新理解
- **实验充分性**: ⭐⭐⭐⭐ — 150 次随机搜索的公平对比很有说服力
- **写作质量**: ⭐⭐⭐⭐ — 数学严谨，但篇幅较长
- **实用性**: ⭐⭐⭐⭐⭐ — 实际改进显著，内存减少且性能更好

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling](../../ACL2025/llm_pretraining/fr_spec_speculative_sampling.md)
- [\[ICCV 2025\] FlowMo: Flow to the Mode — Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization](../../ICCV2025/llm_pretraining/flow_to_the_mode_mode-seeking_diffusion_autoencoders_for_state-of-the-art_image_.md)
- [\[ICLR 2026\] StochasTok: Improving Fine-Grained Subword Understanding in LLMs](stochastok_improving_fine-grained_subword_understanding_in_llms.md)
- [\[ICLR 2026\] TNT: Improving Chunkwise Training for Test-Time Memorization](tnt_improving_chunkwise_training_for_test-time_memorization.md)
- [\[ICLR 2026\] Understanding the Emergence of Seemingly Useless Features in Next-Token Predictors](understanding_the_emergence_of_seemingly_useless_features_in_next-token_predicto.md)

</div>

<!-- RELATED:END -->
