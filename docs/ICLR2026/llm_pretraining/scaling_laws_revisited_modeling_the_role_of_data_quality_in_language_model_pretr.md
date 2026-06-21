---
title: >-
  [论文解读] Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining
description: >-
  [ICLR 2026][预训练][数据质量] 本文在经典 Chinchilla 缩放定律里塞进一个无量纲的数据质量参数 $Q \in (0,1]$，得到 $L(N,D,Q)=A/N^\alpha + B/(D^\beta Q^\gamma) + E$，并在机器翻译和因果语言建模上系统注入噪声做受控实验，证明损失随数据质量可预测地下降、且高质量数据能换取更小的模型与更少的算力。
tags:
  - "ICLR 2026"
  - "预训练"
  - "数据质量"
  - "缩放定律"
  - "有效样本量"
  - "Chinchilla"
---

# Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=x54wwB6QvL](https://openreview.net/forum?id=x54wwB6QvL)  
**领域**: LLM 预训练 / Scaling Law  
**关键词**: 数据质量, 缩放定律, 有效样本量, Chinchilla, 预训练

## 一句话总结
本文在经典 Chinchilla 缩放定律里塞进一个无量纲的数据质量参数 $Q \in (0,1]$，得到 $L(N,D,Q)=A/N^\alpha + B/(D^\beta Q^\gamma) + E$，并在机器翻译和因果语言建模上系统注入噪声做受控实验，证明损失随数据质量可预测地下降、且高质量数据能换取更小的模型与更少的算力。

## 研究背景与动机

**领域现状**：以 Kaplan 和 Chinchilla 为代表的缩放定律已经把"损失随模型规模 $N$ 和数据量 $D$ 怎么变"刻画得很精确，成为指导大规模训练算力分配的标尺。但这些定律几乎都假设训练数据"质量固定"，只在 $N$、$D$ 两个轴上建模。

**现有痛点**：人人都知道"干净的数据训出更好的模型"，但这只是直觉，没有被写进定量的缩放定律里。实践中大家用过滤、去重、去噪等手段改善数据，观测到"过滤数据带来的收益堪比增加算力"，却没有一个公式能预测"质量提升 X 能换来多少损失下降 / 能省多少参数量"。在医学、科学、商业等专业领域，语料天然量少且质量参差，同样的算力预算下结果可能天差地别，缺一个能量化质量的框架就无法做"该花力气清洗数据还是该堆模型"的权衡。

**核心矛盾**：数据质量是一个多维、模糊的概念（信息系统领域常拆成准确性、完整性、一致性、时效性、唯一性、有效性等多个维度），直接把这一堆维度塞进缩放定律既不可解也无法估计；但完全忽略质量，定律又和现实脱节。需要一个足够简单、能平滑退化、可估计的质量抽象。

**本文目标**：(1) 给数据质量一个单一标量定义；(2) 把它以理论自洽的方式嵌入 Chinchilla 形式；(3) 用受控实验验证这条定律的预测力，并给出"质量换规模"的实操指导。

**切入角度**：作者借用"有效样本量"（effective sample size）和信息论视角——劣质数据本质上是降低了数据集里"可用信息"的比例，相当于把 $D$ 个样本缩水成 $D \cdot g(Q)$ 个有效样本。这个视角既有 PAC 学习、Fisher 信息、信道容量等经典结论背书，又能自然导出一个乘性的 $Q^\gamma$ 修正项。

**核心 idea**：用一个无量纲标量 $Q\in(0,1]$ 表征语料的可用信息（$Q=1$ 为完全干净、越小越脏），把它作为有效样本量的乘子写进缩放定律的数据项，得到 $B/(D^\beta Q^\gamma)$。

## 方法详解

### 整体框架
本文不是提出一个模型，而是提出一条**带数据质量的缩放定律**并给出它的理论推导与估计方法。整体逻辑分三步：先给数据质量 $Q$ 一个可操作的标量定义（两种估计器）；再从"有效样本量 + 信息论"两条独立路径论证为什么 $Q$ 应该以 $Q^\gamma$ 的乘性形式进入缩放定律；最后用注入合成噪声的受控实验把 $B,\beta,\gamma,E$ 拟合出来、验证预测力。

最终的质量感知缩放定律为：

$$L(N, D, Q) = \frac{A}{N^\alpha} + \frac{B}{D^\beta Q^\gamma} + E$$

其中 $N$ 为参数量、$D$ 为训练 token 数、$Q$ 为数据质量、$E$ 为不可约最小损失。当 $Q=1$（最高质量）时退化为标准 Chinchilla 定律 $L=A/N^\alpha + B/D^\beta + E$；$Q$ 越小损失越大，反过来 $Q$ 越大、达到同样损失所需的 $D$ 就越少，这正是"用质量换数据量/算力"的数学表达。

### 关键设计

**1. 数据质量标量 Q 的两种估计器：把模糊的"质量"压成一个可平滑退化的数**

痛点是质量本身多维、难以直接放进定律。作者要求 $Q$ 只需满足"随污染平滑退化、能代理可用信息比例"即可，并给出两个互补的估计器。其一是**数据污染率（corruption rate）**：若数据集污染率为 $CR$（假设 $0\le CR<1$），则 $Q(\omega)=1-CR$——10% 的 token 被污染就对应 $Q=0.9$，可用标准抽样方法估计。其二是更一般的**数据缺陷度（deficiency）** $\Delta(\omega)$，要求满足正性、连续性和可加性（两个独立子集并起来缺陷度相加），再定义 $Q(\omega)=\exp(-\Delta(\omega))$。缺陷度可进一步分解为噪声、覆盖度/多样性、重复数据、合成数据四项：

$$\Delta(\omega) = \mu_1 E + \mu_2 \frac{1}{F} + \mu_3 G + \mu_4 H$$

这个分解的妙处在于：给各项取特定形式就能**复现文献里多种质量缩放定律**（如 Chen et al. 的聚类密度、Chang et al. 的压缩多样性 + 合成度、Goyal et al. 的重复 epoch 边际递减），说明本文的 $Q$ 是一个统一容器而非又一个孤立指标。

**2. 有效样本量因子化：用 Fisher 信息 / PAC 噪声理论证明 Q 该以 Q^γ 进入**

这是定律的理论地基。核心假设（Assumption 1）是存在单调链接函数 $g:[0,1]\to\mathbb{R}^+$ 且 $g(1)=1$，使得给定参数量下的损失满足

$$L_N(D,Q) \approx \frac{B}{D_{\text{eff}}^\beta} = \frac{B}{(D\cdot g(Q))^\beta}$$

即"$D$ 个脏样本 = $D\cdot g(Q)$ 个干净样本"。作者用两条经典结论锚定 $g(Q)\approx Q^\gamma$ 的指数：在加性高斯噪声回归里（Lemma 1），每个观测贡献的 Fisher 信息正比于 $1/\sigma^2$，总信息 $I_D\propto D/\sigma^2$，定义 $\Delta=\ln(\sigma^2/\sigma_0^2)$、$Q=e^{-\Delta}=\sigma_0^2/\sigma^2$，于是 $D_{\text{eff}}=D\cdot Q^\gamma$ 在 $\gamma=1$ 时恰好还原"$D\cdot$ 信噪比"的经典缩放；在对称标签噪声里（Lemma 2，翻转率 $\eta$），校正损失方差膨胀 $(1-2\eta)^{-2}$ 倍，有效样本量缩为 $D\cdot(1-2\eta)^2=D\cdot(2Q-1)^2$，在高质量区间 $Q\in(1/2,1]$ 局部可用 $Q^\gamma$ 近似、指数 $\gamma\approx 2$。两条路径都落到 $D_{\text{eff}}=D\cdot g(Q)$，把 $Q^\gamma$ 这一形式从理论上坐实。

**3. 信息论视角的独立佐证：互信息的乘性衰减导出同一条定律**

为了让定律不依赖单一假设，作者再给一条信息论推导（Proposition 1）。把脏 token $\tilde{X}$ 看作干净 token $X$ 经过一个由质量 $Q$ 参数化的无记忆信道 $C_Q$ 的输出，假设污染让可用互信息乘性衰减 $I(\tilde{X};Z)=\rho(Q)\,I(X;Z)$，其中 $\rho(1)=1$、$\rho(Q)$ 单调且在 $Q\to 1$ 附近 $\rho(Q)\approx cQ^\gamma$。再结合信息论泛化界 $L_D\propto 1/(D\cdot I(\tilde{X};Z))^\beta$，重新整理常数后同样得到 $L(N,D,Q)\approx A/N^\alpha + B/(D^\beta Q^\gamma)+E$。对二元对称信道，$\rho(Q)$ 正比于容量 $1-H_2(\eta)$、$Q=1-\eta$，实践区间内 $\gamma\approx 2$；高斯噪声回归则 $\gamma\approx 1$。两个不同视角（有效样本量、互信息）殊途同归到同一条定律，是本文论证强度的关键。

### 损失函数 / 训练策略
实验里优化的是上下文平均的交叉熵损失，并在留出测试集上汇报同一损失以衡量定律对样本外数据的预测力。定律参数 $B,\beta,\gamma,E$ 用 Hoffmann et al.（Chinchilla）的参数化拟合流程估计，并行用最小二乘和 Huber 两种回归（Huber 对异常点更鲁棒）。

## 实验关键数据

实验在两类任务上训练 decoder-only 模型：神经机器翻译（NMT，英德，Paracrawl v8，~133M 参数的 8 层 GPT-Neo）和因果语言建模（CLM，C4-en，8 层 Llama-3 结构）。每个任务取 3 种数据量 × 7 个质量等级，各跑 3 个随机种子，共 63 次运行。质量通过给一部分样本注入合成噪声来控制：NMT 把选中样本里 50% 的非特殊 token 替换为 pad，CLM 把 50% 的 token 随机换成词表里的合法 token；噪声样本占比 $\eta=\{0,10,20,25,30,40,50\}\%$，对应 $Q=\{1.0,0.9,0.8,0.75,0.7,0.6,0.5\}$。数据量上 NMT 为 0.5M/1M/2M 句对，CLM 为 0.1B/1B/10B token，且用嵌套子集采样保证噪声与样本单调性。

### 主实验：拟合出的质量感知缩放定律参数

| 任务 | 拟合方法 | $B$ | $\beta$ | $\gamma$ | $E$ |
|------|---------|------|---------|----------|------|
| NMT | 最小二乘 | 166.57 | 0.263 | 0.185 | 0.147 |
| NMT | Huber | 139.60 | 0.250 | 0.173 | 0.067 |
| CLM | 最小二乘 | 1428.23 | 0.395 | 0.389 | 3.440 |
| CLM | Huber | 1441.51 | 0.396 | 0.401 | 3.439 |

损失随数据量 $D$ 和质量 $Q$ 升高而可预测地下降。最值得注意的是估计出的质量指数 $\hat\gamma$ **显著小于 1**（NMT $\approx 0.173$，CLM $\approx 0.401$，Huber），意味着有效数据量随质量**亚线性**衰减——模型对中等程度的污染比 PAC 学习 / 信道容量理论（通常预测 $\gamma\ge 1$）所暗示的更鲁棒。

### 样本外泛化验证（CLM）

| 设置 | 方法 | $B$ | $\beta$ | $\gamma$ | $E$ |
|------|------|------|---------|----------|------|
| CLM（in-dist） | 最小二乘 | 1428.23 | 0.395 | 0.389 | 3.440 |
| CLM（unseen） | 最小二乘 | 1589.07 | 0.397 | 0.332 | 4.552 |
| CLM（in-dist） | Huber | 1441.51 | 0.396 | 0.401 | 3.439 |
| CLM（unseen） | Huber | 1427.30 | 0.391 | 0.337 | 4.540 |

用训练好的模型在未见数据上评估损失，拟合出的 $\beta,\gamma$ 与分布内高度接近，说明定律具备样本外预测力（scaling generalization）。

### 关键发现
- **亚线性衰减 ($\gamma<1$) 是核心结论**：自然语言里有冗余，即使部分被污染的样本仍携带句法、对齐、共现等有用上下文，所以损失随 $Q$ 下降而增长得比线性更慢——要看到损失明显上升，$Q$ 得掉很多。这给"数据没那么干净也能用"提供了定量支撑。
- **任务越"自回归"对噪声越敏感**：CLM 的 $\gamma\approx 0.40$ 明显大于 NMT 的 $\gamma\approx 0.17$。作者解释：CLM 的 token 交换破坏了局部依赖、增加熵、会误导分布建模；而 NMT 即便 pad 掉半数 token，跨序列的对齐和上下文仍能泄露足够信息。$\gamma$ 因此可当作"鲁棒性指数"——越小越抗污染。
- **质量效应可被干净隔离**：定义 $\Delta L(Q)=L(N,D,Q)-L(N,D,1)\approx \hat B D^{-\hat\beta}(Q^{-\hat\gamma}-1)$，把 $\Delta L(Q)$ 对 $Q^{-\hat\gamma}-1$ 作图近似为过原点的直线，且在不同数据量（0.1B/1B/10B、0.5M/1M/2M）下稳定，说明 $A/N^\alpha+E$ 这两项确实不随 $Q$ 变化，$\gamma$ 是任务内在参数而非优化或采样的伪影。
- **噪声代理有效性**：合成噪声随注入比例增大，嵌入相似度严格单调下降，说明合成噪声是语义退化的合理代理；学习率扫描下 $\gamma$ 也保持稳定。

## 亮点与洞察
- **把"质量"压成一个标量再嵌定律**：最巧妙之处是没去纠缠质量的多维分类，而是要求 $Q$ 只满足"平滑退化 + 代理可用信息"，用污染率或 $\exp(-\Delta)$ 两种估计器落地，既可操作又能向后兼容 Chinchilla（$Q=1$ 即退化）。
- **双视角导出同一定律**：有效样本量（Fisher / PAC）和信息论（信道互信息）两条独立路径都收敛到 $B/(D^\beta Q^\gamma)$，比单纯经验拟合更可信，也把 $\gamma$ 的理论取值（高斯噪声 $\approx 1$、对称标签噪声 $\approx 2$）和实测值对照起来。
- **$\gamma$ 作为"鲁棒性指数"可迁移**：把质量指数解读为任务-模型对的抗污染程度，这个视角能直接迁移到任何要评估"数据清洗值不值"的场景——先小规模测出 $\gamma$，再用 $\Delta L(Q)\approx \hat B D^{-\hat\beta}(Q^{-\hat\gamma}-1)$ 预算质量提升的收益。
- **缺陷度分解统一了文献**：$\Delta=\mu_1 E+\mu_2/F+\mu_3 G+\mu_4 H$ 一式收编了多篇质量缩放工作，说明本文是更上位的框架而非又一个并列指标。

## 局限与展望
- **质量靠合成噪声注入，而非真实脏数据**：实验通过人为 pad / token 交换控制 $Q$，虽然验证了嵌入相似度单调下降，但真实网络语料的污染形态（事实错误、领域漂移、低质量改写）和均匀注入的 iid 噪声差别很大，$\gamma$ 在真实清洗场景下是否同样稳定仍待验证。
- **模型规模偏小、单 epoch**：NMT ~133M、CLM 8 层，数据量上限 10B token，远小于真实大模型预训练，$A/N^\alpha$ 项基本没被充分扫；定律里"质量换模型规模"的结论是从小规模外推的。
- **$Q$ 的实际估计仍是开放问题**：定律好用的前提是能在真实语料上估出 $Q$。污染率需要知道哪些样本"坏"，缺陷度分解里的 $\mu_i$ 和各项度量在真实数据上怎么标定，本文只给了附录指引，落地仍有距离。
- **两个噪声模型都很激进（50% token 破坏）**：噪声样本内部的破坏比例固定为 50%，没有扫不同破坏强度，$\gamma$ 对"单样本内污染深度"的依赖未知。

## 相关工作与启发
- **vs Chinchilla (Hoffmann et al. 2022)**：Chinchilla 给出 $L=A/N^\alpha+B/D^\beta+E$ 但假设数据质量固定；本文加一个 $Q^\gamma$ 把数据项扩成 $B/(D^\beta Q^\gamma)$，$Q=1$ 时完全退化为 Chinchilla，是严格的超集。
- **vs Bansal et al. 2022（NMT 噪声缩放）**：他们发现架构和中等噪声主要平移缩放曲线、不改指数；本文则把噪声/质量显式参数化为 $Q$ 并给出它独立的指数 $\gamma$，从"现象观察"升级为"可预测定律"。
- **vs Goyal et al. 2024（数据过滤非算力无关）**：他们指出高质量数据的价值依赖算力预算；本文用 $D_{\text{eff}}=D\cdot Q^\gamma$ 把这种"质量-数据量可互换"关系写成等损失轮廓线，并能复现他们的重复 epoch 边际递减项。
- **vs Chen et al. 2025 / Chang et al. 2024**：这两篇分别用聚类密度、压缩多样性 + 合成度刻画质量；本文的缺陷度分解 $\Delta=\mu_1E+\mu_2/F+\mu_3G+\mu_4H$ 给特定取值即可复现它们，定位为统一框架。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把数据质量首次以理论自洽的标量形式正式写进 Chinchilla 定律，且双视角导出。
- 实验充分度: ⭐⭐⭐ 受控实验干净、有样本外验证，但规模小、靠合成噪声、未用真实脏数据。
- 写作质量: ⭐⭐⭐⭐ 推导清晰、定律到估计器到实验链条完整，表格自洽。
- 价值: ⭐⭐⭐⭐ 给"清洗数据 vs 堆模型"的权衡提供了可计算的指导，对专业领域小模型尤其有用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Pretraining Scaling Laws for Generative Evaluations of Language Models](pretraining_scaling_laws_for_generative_evaluations_of_language_models.md)
- [\[ICML 2026\] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](../../ICML2026/llm_pretraining/infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)
- [\[ICML 2026\] Explaining Data Mixing Scaling Laws](../../ICML2026/llm_pretraining/explaining_data_mixing_scaling_laws.md)
- [\[ICLR 2026\] Learned Meta-Tokens for Language Modeling](learned_meta-tokens_for_language_modeling.md)
- [\[ICLR 2026\] Reformulation for Pretraining Data Augmentation](reformulation_for_pretraining_data_augmentation.md)

</div>

<!-- RELATED:END -->
