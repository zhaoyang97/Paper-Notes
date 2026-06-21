---
title: >-
  [论文解读] Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers
description: >-
  [ICLR 2026][可解释性][稀疏自编码器] 这篇论文给当下火热的稀疏自编码器（SAE）做了一次"理智检查"：把 SAE 同时套到训练好的 Transformer 和**随机初始化**的 Transformer 上，发现常用的自动可解释性分数（auto-interp AUROC）和重建指标在两者之间几乎区分不开，说明高可解释性分数本身**不能**证明 SAE 抓到了模型真正学到的计算特征。
tags:
  - "ICLR 2026"
  - "可解释性"
  - "稀疏自编码器"
  - "自动可解释性"
  - "随机基线"
  - "叠加表示"
  - "评测指标"
---

# Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=USyGD0eUod](https://openreview.net/forum?id=USyGD0eUod)  
**代码**: 待确认  
**领域**: 可解释性 / 机制可解释性 / 稀疏自编码器  
**关键词**: 稀疏自编码器, 自动可解释性, 随机基线, 叠加表示, 评测指标

## 一句话总结
这篇论文给当下火热的稀疏自编码器（SAE）做了一次"理智检查"：把 SAE 同时套到训练好的 Transformer 和**随机初始化**的 Transformer 上，发现常用的自动可解释性分数（auto-interp AUROC）和重建指标在两者之间几乎区分不开，说明高可解释性分数本身**不能**证明 SAE 抓到了模型真正学到的计算特征。

## 研究背景与动机

**领域现状**：机制可解释性（mechanistic interpretability）近两年的主流工具是稀疏自编码器。它在 Transformer 某一层激活上训练一个隐层维度远大于输入的自编码器，靠稀疏约束逼出几万个"latent"，希望每个 latent 对应一个人类可读的"概念/特征"，从而把模型内部的叠加表示（superposition）拆解开。为了比较不同 SAE 的好坏，社区发展出一套**自动可解释性**评测：用一个大模型给某个 latent 的激活模式生成自然语言解释，再用这个解释去预测 latent 在新文本上是否激活，用 AUROC 衡量解释的保真度。

**现有痛点**：这套评测有一个很少被追问的前提——高分到底意味着什么？大家默认"auto-interp 分数高 = SAE 找到了模型学到的有意义特征"。但没人系统验证过：这个分数会不会只是反映了**数据本身或架构归纳偏置**带来的简单统计结构，而跟"训练"这件事根本无关？

**核心矛盾**：一个可信的可解释性方法，其评测指标必须能把"通过训练学到的特征"和"数据/架构自带的伪影"区分开。检验这一点的经典手段是和一个**强 null model** 对比——比如把权重换成随机初始化的网络（这正是 Adebayo 等人 2020 年给显著图做 sanity check 的思路）。可解释性研究反复强调要解释"模型学到了什么"，却几乎从不拿随机权重模型做对照。

**本文目标**：把 saliency-map 领域的 sanity check 搬到 SAE 上，回答两个子问题——(1) 常用 SAE 指标能否区分训练 vs 随机 Transformer？(2) 如果不能，是什么机制让随机网络的激活也显得"稀疏可解释"？

**切入角度**：作者直接在多个尺寸的 Pythia 模型上，对训练版和若干随机化变体分别训练 SAE，把全套指标摆在一起看是否重合。这个角度的好处是几乎零假设——只要随机变体的分数贴着训练版，就足以证伪"高分=学到特征"。

**核心 idea**：用随机初始化 Transformer 作为 null model 给 SAE 评测做对照，揭示"自动可解释性分数无法区分训练与随机网络"这一现象，进而主张把随机基线和"抽象度"度量纳入常规评测。

## 方法详解

这是一篇批判/分析向论文，没有提出新模型，"方法"是一套对照实验设计 + 一个解释现象的玩具模型。核心是：构造一组从"完全训练"到"完全随机"的 Transformer 变体，在每个变体上独立训练 SAE，再用同一套指标横向比较；当指标区分不开时，再用玩具模型去解释"为什么随机网络也会产生看似可解释的稀疏激活"。

### 整体框架

实验对象是 Pythia 家族（70M–6.9B 参数），数据为 RedPajama。对每个底层 Transformer 的某一层残差流激活，训练一个 TopK 稀疏自编码器（扩张因子 $R=64$，稀疏度 $k=32$，100M token）。关键是构造了五种底层 Transformer 变体，从"真训练"一路退化到"纯随机"，让 SAE 分别去解释，再看指标谱是否能把它们排开：

- **Trained**：正常训练好的模型。
- **Re-randomized incl. embeddings**：把所有权重（含词嵌入）重采样为高斯噪声，但**均值方差对齐**到原训练权重矩阵的统计量。
- **Re-randomized excl. embeddings**：同上，但保留训练好的嵌入/解嵌入矩阵，只随机化其余权重。
- **Step-0**：Pythia 提供的初始化时刻 checkpoint（训练前的原始随机权重）。
- **Control**：用训练好的模型，但推理时把每个 token 的输入嵌入替换成 i.i.d. 标准高斯噪声——同一个 token 每次都没有固定嵌入向量。这是"应该掉到随机水平"的负对照，预期 auto-interp 表现接近瞎猜。

评测指标包括：fuzzing / detection 两种自动可解释性 AUROC、解释方差 $R^2$、重建余弦相似度、L1 范数、CE loss 恢复分数，以及作者自定义的 token 分布熵。叙事主线很简单：除了 Control，其余四个变体（含纯随机的 Step-0）在大多数指标上彼此重合、都贴近 Trained，只有 Control 掉到随机水平——这正是"指标区分不开训练与随机"的实证。

### 关键设计

**1. 随机化方案：保范数的高斯重采样，而非简单清零**

要让"随机 null model"有说服力，关键是随机权重不能制造平凡差异。作者发现训练模型和其初始化状态（Step-0）的参数范数可能差很多，而范数尺度会影响激活在残差流里的增长。于是 Re-randomized 变体在重采样高斯时**逐矩阵对齐原训练权重的均值和方差**，刻意保住参数范数。结果是：保范数的两个随机变体（蓝、橙线）反而比初始化时刻的 Step-0（绿线）更贴近 Trained——这说明很多"看起来像学到了"的指标，其实只对参数尺度敏感，对"是否真的训练过"并不敏感。这个对比本身就是一处反直觉的证据。

**2. 五段式变体阶梯 + 嵌入解耦：定位"信号到底来自哪里"**

光有"训练 vs 随机"二元对照还不够，因为信号可能来自嵌入、来自架构、或来自数据。作者把变体拆成阶梯：Re-randomized 的 incl./excl. embeddings 两版用来隔离"嵌入是否携带可解释结构"；Step-0 代表纯初始化；而 Control 是最狠的负对照——它**保留全部训练权重**，只在推理时给每个 token 灌入随机嵌入，破坏 token 与表示的一致映射。Control 之所以重要，是它把"瞎猜下限"标定了出来（fuzzing AUROC≈0.50）：既然连训练好的模型在随机嵌入下都只能瞎猜，那随机权重变体却能拿到≈0.87 的 AUROC，就更说明这些分数衡量的不是"学到的计算"，而是被保留下来的输入/架构结构。

**3. Token 分布熵：一个能看出"抽象度"的旁路指标**

作者最有价值的建设性贡献是指出：标准指标漏掉了特征的"抽象度"（abstractness）。很多 SAE latent 其实只在单个或极少数 token ID 上激活——这种 latent 当然容易被解释，但它平凡。于是作者用 latent 激活在 token ID 上分布的**熵**来量化这件事：把某 latent 在其最大激活样本集上、按 token 聚合的总激活看作一个分布，

$$H = -\sum_{i} p_i \log p_i,$$

熵越大说明激活越"散"、越不绑死在某个具体 token 上，特征越抽象。关键发现是：Trained 变体的熵**随层数升高**（越深的层特征越抽象，越不像 token 嵌入）；随机变体的熵则**始终偏低**（latent 死死绑在一两个 token 上，不会随层加深变复杂）；Control 因为嵌入是逐次高斯采样，熵恒高。所以同样高的 auto-interp 分数下，token 分布熵能把"会随训练变抽象"的特征和"永远停在 token 级"的特征分开——这正是聚合指标看不见的差异，作者把它当作"应该补上随机基线 + 抽象度度量"的概念验证。

**4. 叠加表示的玩具模型：解释"为什么随机网络也显得稀疏可解释"**

指标区分不开之后，作者用玩具模型去回答机制问题，提出两个假设：(1) 输入数据本就带叠加结构，随机网络只是**保留**它；(2) 随机网络甚至会**放大/引入**叠加。第一点用线性代数即可说明：若稀疏特征 $z$ 经投影矩阵 $D$ 压成稠密输入 $x \sim \mathcal{N}(Dz,\Sigma)$，那么乘任意权重 $W$ 后 $x'=Wx$ 仍服从同构的叠加生成模型 $x' \sim \mathcal{N}(WDz, W\Sigma W^{T})$——矩阵乘法不破坏叠加。第二点靠实验：把叠加输入和同均值方差的高斯对照分别过一个随机初始化的两层 MLP，比较"解释方差 vs 稀疏度"的 Pareto 前沿；结果随机 MLP 的输出在给定解释方差下普遍更稀疏，且这种稀疏对输入分布不太敏感，暗示随机网络确实在"稀疏化"输入。作者还在 GloVe 词向量和 Pythia 嵌入上重复，发现真实语言嵌入的叠加程度其实没玩具数据那么强，但过一遍随机 MLP 后同样被"稀疏化"。

### 一个完整示例：fuzzing 评测怎么被"骗"过

以 Pythia-6.9b 为例走一遍：先在某层残差流上训出 SAE，随机抽 100 个 latent，用 Llama-3.1-70B 给每个 latent 写解释；fuzzing 评分时把正/负例 token（激活/不激活）用特殊符号标出，让大模型按解释判断哪些标注是对的，算 AUROC。Trained 拿到约 0.79，而 Re-randomized（含/不含嵌入）和 Step-0 全在 0.87–0.88——随机模型反而更高；唯有把训练模型喂随机嵌入的 Control 掉到 0.50（瞎猜）。一个只看这张 ROC 图的人会得出"这些 SAE 都抓到了可解释特征"，但其中三条线背后是没经过任何语言训练的随机权重——评测就这样被"骗"了过去。

## 实验关键数据

### 主实验
Pythia-6.9b 的 fuzzing ROC（每个 SAE 抽 100 latent）显示训练与随机变体几乎重叠，只有 Control 接近随机线：

| 变体 | Fuzzing AUROC | 说明 |
|------|---------------|------|
| Trained | 0.79 | 正常训练模型 |
| Re-randomized excl. emb | 0.87 | 随机权重、保留训练嵌入 |
| Re-randomized incl. emb | 0.87 | 全部权重随机（保范数） |
| Step-0 | 0.88 | 初始化时刻原始随机权重 |
| Control | 0.50 | 训练权重 + 随机嵌入（负对照，瞎猜） |

### 跨指标 / 跨尺度分析
在 70M–6.9B 全谱上，除 Control 外的所有变体在解释方差、余弦相似度、AUROC（fuzz/detection）上趋势高度一致；只有 token 分布熵把训练与随机分开：

| 指标 | Trained | 随机变体 | Control | 现象 |
|------|---------|----------|---------|------|
| 解释方差 $R^2$ / 余弦相似度 | 高 | 接近 Trained | 明显偏低 | Control 最难重建（高斯熵最大、最无结构） |
| AUROC（随模型增大） | 上升 | 上升 | ≈随机 | 大模型 latent 更具体、分类更易 |
| CE loss 恢复分数 | 有意义 | 无意义 | — | 仅 Trained 变体的 CE 分数可解释 |
| Token 分布熵 | 随层升高 | 始终偏低 | 恒高 | 仅此指标能体现"抽象度"差异 |

### 关键发现
- **指标区分不开**：保范数的随机权重模型在 auto-interp 和重建指标上贴着训练模型，AUROC 甚至更高，证明高聚合分数不等于抓到学到的计算。
- **抽象度才是分水岭**：训练模型的特征随层加深变抽象（熵升），随机模型的特征永远停在 token 级（熵低）；聚合指标看不到这一点。
- **稀疏可能来自架构而非训练**：随机 MLP 会保留甚至放大输入的叠加/稀疏结构，所以"激活看起来稀疏可解释"未必源于学习。
- **AUROC 随模型增大而升**（除 Control），作者推测大 SAE 里每个 latent 只需解释更少输入、特征更具体，使分类任务更易。

## 亮点与洞察
- **把 saliency-map 的 sanity check 移植到 SAE**：用随机权重 null model 检验可解释性方法，是个朴素但杀伤力极大的实验范式——只要随机基线贴着训练版，整套评测的解释力就被证伪了，几乎不需要额外假设。
- **保范数随机化是点睛之笔**：如果只是简单随机或清零，很容易被反驳"差异来自尺度"。逐矩阵对齐均值方差后随机变体反而更像训练版，反而强化了结论，说明很多指标只对参数尺度敏感。
- **token 分布熵给出了建设性出口**：论文不只是"拆台"，还提供了一个廉价、可立即采用的旁路指标来捕捉"抽象度"，并指明方向——常规评测应配随机基线 + 抽象度度量。
- **可迁移的方法论**：任何号称"发现了模型内部结构"的可解释性工具，都可以照此构造保统计量的随机 null model 做对照；这套思路能直接迁移到 probing、circuit 发现、steering 向量等评测上。

## 局限与展望
- **作者承认**：只在 Pythia 家族 + RedPajama 上验证，无法穷举所有数据/架构；解释模型固定用了 EleutherAI 框架默认的 Llama-3.1-70B，换解释器可能影响聚合行为。
- **作者澄清边界**：论文**不主张** SAE 在训练模型上学不到东西，只主张"聚合 auto-interp 指标不足以证明学到了有意义特征"——这个 caveat 很重要，避免被误读成"SAE 没用"。
- **机制未定论**：玩具模型只证明随机网络"既可能保留也可能放大"叠加，但到底哪种主导真实 Transformer 的现象，留给未来工作。
- **token 分布熵只是 proof-of-concept**：作者明说它不是"抽象度"的直接度量，只是个能体现差异的代理，离一个稳健的"计算显著性"指标还很远。

## 相关工作与启发
- **vs Bricken et al. (2023)**：他们在单层 Transformer 上发现 auto-interp 分数能有效区分训练与随机；本文在更大模型上复现了"小模型（如 Pythia-70m）随机基线分数较低"，但指出**模型变大后差距迅速收窄**（Pythia-6.9b 几乎重合），把结论从"单层"推广到了规模谱。
- **vs Karvonen et al. (2024c)（棋类随机 Transformer）**：他们发现 SAE 在预训练棋类模型上明显优于随机权重；本文指出语言与棋类数据性质不同——语言的稀疏结构与概念语义"对齐"，而棋类不一定，所以两者在随机基线上的表现不可直接类比。
- **vs Zhong & Andreas (2024)（只训嵌入）**：他们展示只训练嵌入就能让随机 Transformer 涌现算法能力；本文设置正相反——连嵌入也随机化（incl. embeddings 变体），或冻结预训练嵌入但不训练（excl. embeddings），刻意把"训练信号"剥离干净。
- **vs Adebayo et al. (2020)（显著图 sanity check）**：本文是其思想在 SAE 评测上的直接延伸——任何可解释性指标都该先过"随机权重 null model"这一关。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把成熟的 sanity-check 范式精准移植到当下最热的 SAE 评测，问题切中要害，虽非全新机制但视角锋利。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 70M–6.9B 全谱、五种变体、多指标 + 玩具模型佐证，唯模型族与解释器较单一。
- 写作质量: ⭐⭐⭐⭐ 论证链条清晰、caveat 诚实，玩具模型部分稍偏 appendix 化。
- 价值: ⭐⭐⭐⭐⭐ 直接动摇了机制可解释性社区对 auto-interp 分数的默认信任，给出可立即采用的随机基线建议，影响面广。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees](formal_mechanistic_interpretability_automated_circuit_discovery_with_provable_gu.md)
- [\[ICLR 2026\] Sparse Autoencoders Trained on the Same Data Learn Different Features](sparse_autoencoders_trained_on_the_same_data_learn_different_features.md)
- [\[ICLR 2026\] How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Understanding](how_do_transformers_learn_to_associate_tokens_gradient_leading_terms_bring_mecha.md)
- [\[ICLR 2026\] Learning Pseudorandom Numbers with Transformers: Permuted Congruential Generators, Curricula, and Interpretability](learning_pseudorandom_numbers_with_transformers_permuted_congruential_generators.md)
- [\[ACL 2025\] Enhancing Automated Interpretability with Output-Centric Feature Descriptions](../../ACL2025/interpretability/output_centric_interpretability.md)

</div>

<!-- RELATED:END -->
