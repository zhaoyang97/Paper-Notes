---
title: >-
  [论文解读] Noise Stability of Transformer Models
description: >-
  [ICLR 2026][可解释性][noise stability] 提出噪声稳定性（noise stability）替代平均敏感度（average sensitivity）作为衡量 Transformer 简单性偏差的更优指标，并基于此设计正则化方法，在合成任务和语言建模上分别加速训练约 35% 和 75%。
tags:
  - "ICLR 2026"
  - "可解释性"
  - "noise stability"
  - "simplicity bias"
  - "Transformer"
  - "grokking"
  - "Fourier analysis"
  - "regularization"
  - "Boolean function analysis"
---

# Noise Stability of Transformer Models

**会议**: ICLR 2026  
**arXiv**: [2602.08287](https://arxiv.org/abs/2602.08287)  
**代码**: 未公开  
**领域**: 可解释性  
**关键词**: noise stability, simplicity bias, Transformer, grokking, Fourier analysis, regularization, Boolean function analysis

## 一句话总结

提出噪声稳定性（noise stability）替代平均敏感度（average sensitivity）作为衡量 Transformer 简单性偏差的更优指标，并基于此设计正则化方法，在合成任务和语言建模上分别加速训练约 35% 和 75%。

## 研究背景与动机

深度学习中的简单性偏差（simplicity bias）是理解模型泛化、可解释性和鲁棒性的核心概念。神经网络倾向于收敛到能解释训练数据的最简函数。量化这种"简单性"的传统度量源自布尔函数分析中的**平均敏感度（average sensitivity）**，即模型输出对单个 token 扰动的期望变化。

先前工作表明 Transformer 学到的函数比 LSTM 的敏感度更低（Bhattamishra et al., 2022），且 Transformer 难以学习高敏感度函数如 Parity（Hahn 2020）。Vasudeva et al.（2024）将平均敏感度与 grokking 现象联系起来。

然而，作者指出平均敏感度存在两个关键缺陷：

**理论缺陷**：布尔域上的定义难以自然推广到实值域，基于超网格的扩展方法笨拙且采样不切实际

**实证缺陷**：未能解释 GPT-2、Gemma、RoBERTa 等现代 LLM 中观察到的"junta-like"输入依赖现象——输出仅依赖于极小子集的输入 token（实验中 256 个 token 仅 5-10 个有显著影响），而 Friedgut 定理的上界预测高达 1024 个，差距极大

## 方法详解

### 整体框架

这篇论文要解决一个度量问题：用什么指标刻画 Transformer 的"简单性偏差"才既严谨又能解释现象。传统的平均敏感度（average sensitivity）逐个翻转 token、只在布尔域成立，既难推广到实值域，也解释不了大模型里"输出只依赖极少数 token"的 junta-like 现象。本文的做法是换一个度量——**噪声稳定性（noise stability）**：不逐个扰动坐标，而是给全部输入坐标同时注入一份关联噪声，看输出还剩多少相关性。整篇方法是一条从度量到应用的链条——先给出噪声稳定性的形式化定义并把它和函数频谱挂钩；再证一条引理，说明"高稳定性"必然推出"低频/junta 式简单性"；接着把这条度量逐层穿过 ReLU MLP 与注意力层、分析它在多层堆叠下如何衰减；最后把"鼓励高稳定性"写成一个可微正则项，反过来加速 Transformer 的训练与 grokking。

### 关键设计

**1. 噪声稳定性的形式化定义：用关联噪声一次性替代逐 token 扰动**

平均敏感度的麻烦在于它逐个翻转 token、又只在布尔域上自然成立。噪声稳定性换了个思路：不再单独扰动每个坐标，而是给所有输入坐标同时加上**一份关联的高斯噪声**，看函数输出还剩多少相关性。形式上，对高斯测度 $\gamma$ 下的函数 $f \in L^2(\gamma)$，先构造关联对 $(X,Y)$，再取两者输出的内积期望：

$$\text{Stab}_\rho(f) := \mathbb{E}_{(X,Y)}[f(X) f(Y)]$$

其中 $Y = \rho X + Z\sqrt{1-\rho^2}$，$Z \sim \gamma$ 独立于 $X$，相关系数 $\rho \in (0,1)$ 控制噪声强度——$\rho$ 越接近 1 噪声越弱。这个定义天然活在实值域上（靠 Ornstein-Uhlenbeck 半群），不需要平均敏感度那套笨拙的超网格采样。更关键的是，它通过 Hermite-Fourier 系数和频谱直接挂钩——展开后每一阶系数被 $\rho^{|\alpha|}$ 指数加权，阶数 $|\alpha|$ 越高压得越狠：

$$\text{Stab}_\rho(f) = \sum_{\alpha \in \mathbb{N}^d} \rho^{|\alpha|} \tilde{f}(\alpha)^2$$

于是"输出对关联噪声稳定"就等价于"能量集中在低阶 Fourier 系数上"，把一个鲁棒性概念和频谱结构绑在了一起。

**2. 谱集中引理（Lemma 1）：把"稳定"翻译成"低频主导"**

光有定义还不够，得把"稳定"严格地证成"简单"。这条引理给出桥梁：只要噪声稳定性接近函数的总能量，Fourier 质量就必然堆在低阶系数上。具体地，若 $\text{Stab}_\rho(f) \geq (1-\delta)\|f\|_2^2$，则 $f$ 是 $(\varepsilon, T)$-谱集中的——即截断阶数 $T$ 以外的频谱尾部质量不超过 $\varepsilon$，且

$$T \geq \log_{1/\rho}\left(1 - \frac{\delta}{\varepsilon}\right)$$

稳定性越高（$\delta$ 越小），同样的尾部预算 $\varepsilon$ 下能压住的阶数越高、尾部被压得越狠。这正是后面在 GPT-2、RoBERTa 等模型上算"度数 ≥15 的 Fourier 尾部质量上界"的理论依据，也是噪声稳定性能给出比平均敏感度更紧上界的根源。

**3. 逐层稳定性传播分析：穿过 ReLU MLP 与注意力，看它在深网里如何衰减**

定义和引理是针对"一个函数"的；要把结论落到真实 Transformer，得知道稳定性**逐层怎么传**。论文先逐组件给出闭式传播率，再递推到多层。对 $\rho$-关联的高斯输入 $(X,Y)$，单层 ReLU（Theorem 5.1）的输出内积有闭式解

$$\mathbb{E}[\text{ReLU}(X) \cdot \text{ReLU}(Y)] = \frac{1}{2\pi}\left(\sqrt{1-\rho^2} + \rho(\pi - \arccos\rho)\right),$$

二阶 Taylor 展开约为 $\frac{1}{2\pi} + \frac{1}{4}\rho + \frac{1}{4\pi}\rho^2$，主项随 $\rho$ 近似线性——一层非线性不会把关联噪声抹平，只按一个可控比例往下传。注意力层（Theorem 5.2/5.3）则取决于 query-key 矩阵 $W = W_Q W_K^T$ 的结构：**恒等 $W=I_d$** 下高维极限里注意力矩阵收敛到 $I_n$，稳定性与 $\rho$ 保持线性、代价仅 $o(1)$；**低秩 $W=UU^T$** 经 Johnson-Lindenstrauss 变换归约回恒等情形，结论一致；而**非结构化 $W \sim \mathcal{N}(0,I)$**（随机初始化的最坏情形）下注意力矩阵趋向随机排列矩阵，稳定性退化为

$$\rho \cdot s(\rho) \cdot \|(W_V)_{:,j}\|_2^2,$$

其中 $s(\rho):=\mathbb{P}(k=k')$ 是两路噪声仍选中同一输入 token、即注意力模式被保持的概率。结构化注意力几乎无损地传稳定性，随机注意力才会多吃一道衰减。

把单层结论递推到多层时，两类组件表现不同。纯 ReLU FFN 的递推 $\rho_L = \frac{1}{2\pi}(\sqrt{1-\rho_{L-1}^2} + \rho_{L-1}(\pi - \arccos\rho_{L-1}))$ 收敛到非零固定点 $\frac{2}{3\pi} \approx 0.212$，是**弱衰减**——信号被压到一个有限下界而不会被层数吃光。但带注意力的多层 Transformer 没有这么好的性质：同样的递推**不再**给出弱衰减，论文观察到当 $\|(W_V)_{:,j}\|_2 = \gamma < 1$ 时稳定性会一路衰减到零，注意力图改变分布的程度足以破坏 FFN 那种固定点行为。因此多层情形论文不靠单一固定点，而改用**协方差区间传播（covariance interval propagation）**逐层维护稳定性的上下界。

### 损失函数 / 训练策略

把上面"高稳定性 = 简单 = 易泛化"的结论反过来用，就得到一个鼓励稳定性的正则项（Definition 6.1，方向参数 $S=1$ 表示鼓励、$S=0$ 表示抑制）：

$$R_{M,S,\rho}(X) = (-1)^S \cdot \sum_{i=1}^C M(X)_i \cdot M(Y)_i$$

它把噪声稳定性的内积定义直接搬到模型输出分布 $M(\cdot)$ 上：构造扰动序列 $Y$ 时，每个坐标 $Y_i$ 以概率 $\frac{1+\rho}{2}$ 保持为 $X_i$、否则从 $\text{uniform}([U])$ 重采样（这是关联噪声在离散 token 上的实现）。最终训练目标为

$$\ell_{\text{reg}}(M,X) = \ell(M,X) + \gamma \cdot R_{M,S,\rho}(X),$$

$\gamma$ 控制正则强度。正则项可微、且依赖模型在训练数据上的输出（而非仅参数），每次迭代只多一次前向传播，计算开销极低，却能稳定地催化 grokking、加速训练。

## 实验关键数据

### 主实验

**谱集中上界对比（n=256, 度数 ≥15 的 Fourier 尾部质量）**：

| 模型 | 平均敏感度上界 | 噪声稳定性上界 |
|------|---------------|---------------|
| GPT-2 | 0.003 | **0.0005** |
| BERT | 0.04 | **0.02** |
| RoBERTa | 0.19 | **0.02** |
| Gemma | 0.043 | **0.0157** |

噪声稳定性在所有模型上都给出更紧的 Fourier 尾部质量估计（6× 到 9.5× 的改进）。

**Grokking 加速效果**：

| 任务 | 超参数 (γ, ρ) | 无正则化收敛步数 | 有正则化收敛步数 | 加速比 |
|------|--------------|-----------------|-----------------|--------|
| 模加法 (K=113) | (0.75, 0.25) | ~4500 | ~3300 | **36%** |
| 噪声 k-sparse parity | (0.05, 0.05) | 基线 | 加速 | **~35%** |
| WikiText-2 NTP | - | 基线 | 加速 | **~75%** |

### 消融实验

- **LLM 的 junta-like 特性**：在 256 token 输入上，GPT-2/RoBERTa/Gemma 仅 5-10 个 token 具有显著几何影响力，远少于 Friedgut 定理预测的上界 1024 个
- **位置偏差**：首尾 token 一致地具有最高影响力，与 KV Cache 压缩文献中"attention sinks"的观察一致
- **训练动态监控**：在 noisy sparse parity 任务中，Transformer 的噪声稳定性在训练过程中自然下降以匹配目标函数，稳定性变化是泛化的先行指标
- **WikiText-2 语言建模**：正则化模型的噪声稳定性保持高位，而未正则化模型变得越来越不稳定

### 关键发现

1. 噪声稳定性比平均敏感度能更精确地刻画 Transformer 的谱集中（所有模型均给出更紧上界）
2. ReLU MLP 层对稳定性产生弱衰减（收敛到固定点 $2/(3\pi)$），而非完全消除信号
3. 注意力层在恒等/低秩 $W$ 下保持稳定性（线性关系），在非结构化 $W$ 下引入额外衰减因子 $s(\rho)$
4. 噪声稳定性正则化是 grokking 的催化剂，在多种任务上一致地加速训练

## 亮点与洞察

1. **统一理论框架**：通过 Ornstein-Uhlenbeck 半群将布尔域分析自然推广到实值域，保留了与函数频谱的严格联系，比几何影响力更具分析力
2. **跨领域桥接**：建立了信号传播（C-maps/Q-maps）与简单性偏差/可解释性之间的新连接——噪声稳定性可视为相关性映射的更简洁类比
3. **实用正则化**：仅需一次额外前向传播的低成本正则化方法，75% 的 NTP 训练加速极具实用价值
4. **LLM 内部结构洞察**：量化了 GPT-2 等模型的 junta-like 依赖（仅 5-10 个 token 具有显著影响），为 KV cache 压缩、token 剪枝提供理论支撑
5. **训练监控新指标**：噪声稳定性的变化可作为 grokking 的先行信号

## 局限性

1. 理论分析中省略了残差连接、层归一化、注意力掩码等实际 Transformer 组件
2. 语言建模实验仅在小规模 WikiText-2 上进行，缺乏亿级参数 LLM 上的验证
3. 多层 Transformer 的稳定性区间传播的实际紧致度尚未充分验证
4. 正则化超参数 $(\gamma, \rho)$ 需要针对不同任务调优（模加法用 (0.75,0.25)，parity 用 (0.05,0.05)）
5. 未探讨噪声稳定性与对抗鲁棒性之间的定量关系

## 相关工作与启发

与 Vasudeva et al.（2024）使用平均敏感度追踪 grokking 不同，本文的噪声稳定性提供了更强的谱集中保证。与 Hua et al.（2023）的 Transformer 微调噪声稳定性方法在动机（简单性偏差 vs 微调稳定性）、应用范围和关联噪声定义上都有根本区别。最直接的启发来自 Li & Mossel（2025）的层级函数噪声敏感度分析。

**启发**：噪声稳定性可以作为训练监控指标——稳定性下降往往预示 grokking 即将发生，为自适应训练策略提供新思路。此外，junta-like 依赖的量化分析为提示词工程中"哪些 token 真正重要"提供了理论依据。

## 评分

- 新颖性: ⭐⭐⭐⭐ (将信号传播与简单性偏差统一的视角很新颖，理论分析完善)
- 实验充分度: ⭐⭐⭐ (理论扎实但实验规模偏小，缺乏大模型验证)
- 写作质量: ⭐⭐⭐⭐ (理论推导清晰，行文流畅，图表直观)
- 价值: ⭐⭐⭐⭐ (为理解 Transformer 内部机制提供了新工具，正则化方法有实用潜力)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Latent Concept Disentanglement in Transformer-based Language Models](latent_concept_disentanglement_in_transformer-based_language_models.md)
- [\[ICLR 2026\] How Stable is the Next Token? A Geometric View of LLM Prediction Stability](how_stable_is_the_next_token_a_geometric_view_of_llm_prediction_stability.md)
- [\[ICML 2026\] Certified Circuits: Stability Guarantees for Mechanistic Circuits](../../ICML2026/interpretability/certified_circuits_stability_guarantees_for_mechanistic_circuits.md)
- [\[NeurIPS 2025\] AdaptGrad: Adaptive Sampling to Reduce Noise](../../NeurIPS2025/interpretability/adaptgrad_adaptive_sampling_to_reduce_noise.md)
- [\[ICLR 2026\] Hedonic Neurons: A Mechanistic Mapping of Latent Coalitions in Transformer MLPs](hedonic_neurons_a_mechanistic_mapping_of_latent_coalitions_in_transformer_mlps.md)

</div>

<!-- RELATED:END -->
