---
title: >-
  [论文解读] Polynomial, trigonometric, and tropical activations
description: >-
  [ICLR 2026][预训练][激活函数] 系统探索基于正交基（Hermite多项式、Fourier三角基）和热带化（tropicalization）的可学习激活函数族，通过方差保持初始化解决多项式激活的梯度爆炸/消失问题，在GPT-2和ConvNeXt上成功替代GELU实现有效训练。 激活函数是深度神经网络的核心组件…
tags:
  - "ICLR 2026"
  - "预训练"
  - "激活函数"
  - "Hermite多项式"
  - "Fourier三角基"
  - "Tropical多项式"
  - "方差保持初始化"
---

# Polynomial, trigonometric, and tropical activations

**会议**: ICLR 2026  
**arXiv**: [2502.01247](https://arxiv.org/abs/2502.01247)  
**代码**: [K-H-Ismail/torchortho](https://github.com/K-H-Ismail/torchortho)  
**领域**: LLM预训练  
**关键词**: 激活函数, Hermite多项式, Fourier三角基, Tropical多项式, 方差保持初始化

## 一句话总结
系统探索基于正交基（Hermite多项式、Fourier三角基）和热带化（tropicalization）的可学习激活函数族，通过方差保持初始化解决多项式激活的梯度爆炸/消失问题，在GPT-2和ConvNeXt上成功替代GELU实现有效训练。

## 研究背景与动机

激活函数是深度神经网络的核心组件，它引入了非线性使网络能够逼近复杂函数。自ReLU、GELU、SwiGLU等激活函数提出以来，绝大多数现代深度学习模型都采用固定形式的激活函数。然而，一个自然的问题是：**哪些函数可以用作深度神经网络的激活函数？** 能否使用更具表达力的函数族（如多项式、三角函数）作为可学习的激活函数？

**现有痛点**：多项式激活函数虽然理论上具有强大的逼近能力，但在实践中面临严重的梯度爆炸和激活值爆炸问题。即使是低阶多项式，在深度网络中也会因层层复合而导致数值不稳定。这一问题长期限制了多项式类激活函数的实际应用。

**核心矛盾**：多项式/三角函数族具有丰富的数学结构和强大的表达能力，但直接在深度网络中使用会导致训练不稳定；而标准激活函数（ReLU、GELU）虽然稳定，但函数形式固定、缺乏可学习的自适应能力。

**切入角度**：本文基于正交基的数学性质，设计了方差保持的初始化方案，使得基于正交基的可学习激活函数能够在深度网络中稳定训练，无需额外的截断（clamping）机制。

**核心idea**：通过选择合适的正交基（Hermite多项式基、Fourier三角基）和设计方差保持初始化，可以将可学习的多项式和三角函数用作深度网络的激活函数，同时保持训练稳定性。

## 方法详解

### 整体框架
本文要回答一个老问题——"到底什么函数能当深度网络的激活？"——并把激活函数本身变成可学习对象：每个标量激活 $F(x)$ 写成一组**正交基**函数的可学习线性组合，系数随网络端到端训练。难点在于多项式、三角这类基直接堆进深网会爆方差：激活值与梯度逐层复合后指数级膨胀或衰减，几层就数值溢出，这正是多项式激活长期被弃用的根源。作者的破局点是一套**方差保持初始化**——借正交基让激活的二阶矩有闭式解，从而解析地把每层"前向增益"与"后向增益"都钉到相等，方差不再随深度漂移。在这套初始化的支撑下，作者落地了三族激活（Hermite 多项式、Fourier 三角、Tropical 分段线性），并用 Hermite 插值让它们能无缝顶替预训练模型里的 GELU，于是无需 clamping、梯度裁剪或额外归一化就能从头训练 GPT-2、ConvNeXt。

> 说明：方法是"一套初始化技术 + 几族并列的激活函数"，没有多阶段串行的数据流管线，框架图（flowchart）无从表达逐层方差/二阶矩的代数推导，故按惯例跳过框架图；下文整体框架、关键设计、术语已逐一对齐。

### 关键设计

**1. 方差保持初始化：用正交基的闭式二阶矩稳住深层方差**

这是全文的地基，前三族激活能否实用全押在它上面。沿用 He 初始化的思路，一层 MLP 要稳定，需让激活前后的方差守恒，即"前向增益" $\Gamma=\mathrm{Var}[x]\cdot \mathbb{E}[F(x)^2]^{-1}$ 与"后向增益" $\Gamma'=\mathrm{Var}[x]\cdot \mathbb{E}[F'(x)^2]^{-1}$ 都为 1。先前 Yang & Wang 用有理函数做可学习激活时卡在这里：有理分式的二阶矩没有闭式表达，只能靠拟合 ReLU 来初始化。作者的关键观察是——**换成正交基，二阶矩积分就有了简单闭式解**。在标准正态输入假设下，Hermite 多项式两两正交（$\int \mathrm{He}_m \mathrm{He}_n\, e^{-x^2/2}\mathrm{d}x = \sqrt{2\pi}\,n!\,\delta_{nm}$），于是各阶贡献互不串扰、能直接解析地求出令前后向增益相等的系数初始化。正是这一约束让 12 层 Transformer 从头稳定收敛，消融里去掉它训练直接崩溃，印证它是必要条件而非锦上添花。

**2. Hermite 与 Fourier：两族互补的正交基可学习激活**

有了方差保持初始化这把钥匙，作者落地两族正交基激活。Hermite 激活把激活展成概率学家 Hermite 多项式的加权和

$$F(x) = \sum_{k=0}^{n} \frac{a_k}{k!}\,\mathrm{He}_k(x),$$

其中 $\mathrm{He}_k$ 是第 $k$ 阶 Hermite 多项式、$a_k$ 为可学习系数；按定理给出的初始化（$a_k=1\ (k\ge1)$、$a_0=\sqrt{1-1/n!}$）即可让前后向增益相等、方差钉在 1 附近，从源头消掉普通多项式 $x^k$ 的爆炸隐患。多项式擅长局部逼近却难表达周期性，作者再补一族 Fourier 三角基 $F(x) = a_0 + \sum_{k=1}^{n}\frac{a_k\cos(kx)+b_k\sin(kx)}{k!}$ 与之互补：它在均匀分布输入下正交，天然适合刻画数据里的振荡/周期结构，并复用同一套方差保持初始化稳住训练。选正交多项式还有一项理论回报——全网用多项式激活时整个网络可视作一个多元多项式映射，把网络的表达能力放进代数几何框架来分析。

**3. Tropical 激活：把 ReLU 推广成可学的分段线性函数**

前两族给的是光滑曲线，作者再补一族分段线性的。做法是对普通多项式做"热带化"：在 tropical 代数里加法换成 $\max$、乘法换成普通加法，一个 tropical 多项式于是退化为若干仿射函数的逐点最大值，也就是一条分段线性曲线——这恰好把 ReLU（最简单的 tropical 多项式 $\max(0,x)$）推广为可学习的多段折线。作者还引入 tropical **有理**函数（两个 tropical 多项式相减），把表达力扩展到非凸形状，并指出它可解释为某可学习函数的离散凸共轭，与前两族的光滑曲线形成对照。

**4. Hermite 插值迁移：让可学习激活无缝接入预训练模型**

从头训练之外，作者想把可学习激活用到海量已有预训练权重上。但直接拿随机初始化的激活替换 GELU 会扰动已学好的表示。解法是用 Hermite 插值**同时匹配 GELU 的函数值与一阶导数**，使新激活在初始化时几乎与原 GELU 重合，微调便从一个近乎等价的起点出发、收敛更快更稳。这把可学习激活的适用范围从"从头训练"扩展到了"在预训练模型上微调"。

### 损失函数 / 训练策略
训练目标与标准范式完全一致，激活系数只是多出来的一组可学习参数一并端到端优化：GPT-2 在 OpenWebText 上做 next-token prediction、用交叉熵损失；ConvNeXt 在 ImageNet-1K 上做标准分类训练。无需为可学习激活引入额外正则或特殊调度。

## 实验关键数据

### 主实验

| 模型/任务 | 指标 | GELU基线 | Hermite | Fourier | Tropical | 说明 |
|-----------|------|----------|---------|---------|----------|------|
| GPT-2 / OpenWebText | Perplexity | 基线值 | 降低 | 降低 | 可比 | 可学习激活改善语言建模 |
| ConvNeXt-T / ImageNet | Top-1 Acc | 基线值 | 提升 | 提升 | 可比 | 视觉任务同样有效 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无方差保持初始化 | 训练崩溃 | 证明初始化是必要条件 |
| 不同多项式阶数 | 性能vs稳定性 | 阶数越高越有效但需更谨慎初始化 |
| Hermite插值微调 vs 从头训练 | 微调效率 | 插值初始化显著加速收敛 |

### 关键发现
- 通过方差保持初始化，多项式和三角激活函数可以成功训练GPT-2（12层Transformer）和ConvNeXt等深度模型
- 可学习激活在语言建模（perplexity）和图像分类（accuracy）上均匹配甚至超越固定的GELU激活
- Tropical多项式提供了ReLU到更复杂分段线性函数的自然推广
- Hermite插值使得将可学习激活引入预训练模型变得可行
- 方差保持初始化是成功的关键——没有它，多项式激活在几层之后就会数值溢出

## 亮点与洞察
- **理论优美**：将正交基理论与神经网络激活函数设计完美结合，数学推导严谨
- **实用性好**：提供了torchortho库，可直接替换PyTorch中的标准激活函数
- **覆盖全面**：同时探索了多项式（Hermite）、三角（Fourier）和tropical三个函数族，给出了统一的分析框架
- **迁移学习友好**：Hermite插值方法使得可学习激活可以无缝接入已有的预训练模型
- **理解深刻**：揭示了多项式激活网络作为多元多项式映射的代数结构，打开了利用代数几何分析神经网络的大门
- **效率潜力**：可学习激活可能通过自适应调整激活形式来提升大规模模型的训练效率

## 局限与展望
- 实验规模相对有限：仅在GPT-2（124M）和ConvNeXt-T上验证，未扩展到更大规模模型（如GPT-3、LLaMA等）
- 可学习激活引入了额外的参数（系数），在超大规模模型中可能增加内存开销
- Tropical激活虽然理论有趣，但实际性能提升相比Hermite和Fourier不够显著
- 未充分探索可学习激活在训练过程中的演化动态——激活形状如何变化？不同层是否学到不同形式？
- 方差保持初始化依赖于输入为高斯分布的假设，实际网络中间层的激活值分布可能偏离此假设
- 缺乏与KAN（Kolmogorov-Arnold Network）等同期的可学习激活方法的直接对比

## 相关工作与启发
- **KAN（Kolmogorov-Arnold Networks）**：另一种基于可学习激活函数的网络设计，使用B样条基函数
- **SwiGLU、GeGLU**：门控线性单元类激活，在LLM中广泛使用
- **Maxout Networks**：最早探索分段线性激活的工作，Tropical激活可看作其推广
- **Mish、Swish**：自动搜索得到的激活函数，与本文的可学习方法形成对比
- 启发：可学习激活函数可能在需要特定函数逼近性质的场景（如科学计算、物理信息网络）中特别有价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ （系统性探索全新的激活函数族）
- 实验充分度: ⭐⭐⭐ （规模有限但验证充分）
- 写作质量: ⭐⭐⭐⭐ （数学推导清晰）
- 价值: ⭐⭐⭐⭐ （为激活函数设计开辟新方向，有开源代码支持）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Identifying and Evaluating Inactive Heads in Pretrained LLMs](identifying_and_evaluating_inactive_heads_in_pretrained_llms.md)
- [\[ICLR 2026\] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)
- [\[ICLR 2026\] Pre-training Limited Memory Language Models with Internal and External Knowledge](pre-training_limited_memory_language_models_with_internal_and_external_knowledge.md)
- [\[ICLR 2026\] StochasTok: Improving Fine-Grained Subword Understanding in LLMs](stochastok_improving_fine-grained_subword_understanding_in_llms.md)
- [\[ICLR 2026\] Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining](scaling_laws_revisited_modeling_the_role_of_data_quality_in_language_model_pretr.md)

</div>

<!-- RELATED:END -->
