---
title: >-
  [论文解读] One Protein Is All You Need
description: >-
  [ICLR 2026][计算生物][蛋白质语言模型] 本文提出 ProteinTTT，把测试时训练搬到蛋白质语言模型上——给定一条待测蛋白序列，在推理前用掩码语言建模目标在这一条序列上对骨干网络做几十步自监督微调，使模型对该序列的困惑度下降、表示变好，从而在不改动任何下游任务头的前提下提升结构、适应度、功能三类预测，并在 ProteinGym 上刷新 SOTA。
tags:
  - "ICLR 2026"
  - "计算生物"
  - "蛋白质语言模型"
  - "测试时训练"
  - "自监督定制"
  - "结构预测"
  - "困惑度最小化"
---

# One Protein Is All You Need

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=5zAde2jch7](https://openreview.net/forum?id=5zAde2jch7)  
**代码**: https://github.com/anton-bushuiev/ProteinTTT  
**领域**: 计算生物学 / 蛋白质语言模型 / 测试时训练  
**关键词**: 蛋白质语言模型, 测试时训练, 自监督定制, 结构预测, 困惑度最小化

## 一句话总结
本文提出 ProteinTTT，把测试时训练搬到蛋白质语言模型上——给定一条待测蛋白序列，在推理前用掩码语言建模目标在这一条序列上对骨干网络做几十步自监督微调，使模型对该序列的困惑度下降、表示变好，从而在不改动任何下游任务头的前提下提升结构、适应度、功能三类预测，并在 ProteinGym 上刷新 SOTA。

## 研究背景与动机
**领域现状**：蛋白质机器学习的主流范式是先在海量序列上自监督预训练一个大骨干（如 ESM2、ESM3、SaProt），再接一个任务头（ESMFold 结构头、适应度打分头、功能分类器）做下游预测。这类模型为「在整个数据集上取得最好的平均性能」而优化。

**现有痛点**：生物学家关心的往往不是平均性能，而是某一条具体蛋白——某个代谢病相关蛋白、某个致癌信号蛋白、某条变异频繁的病毒蛋白。这些目标常常数据稀缺、与训练分布偏移，通用模型在它们身上恰恰预测得很差。论文开篇的例子很直白：CASP14 靶标 T1074，ESMFold 因为底层 ESM2 对这条序列「很困惑」（困惑度高达 13.0），结构预测 TM-score 只有 0.63。

**核心矛盾**：「在所有蛋白上都做好」和「在某一条蛋白上做到极致」之间存在张力——为了泛化而牺牲了对单个目标的拟合精度。而实验生物学需要的是后者，因为湿实验代价高昂，必须靠准确的单蛋白预测来指导。

**本文目标**：让模型能针对单条目标蛋白「就地定制」，在推理时临时把骨干调到更懂这条序列，且不假设有任何额外数据（没有同源序列、没有标注、没有验证集）。

**切入角度**：作者抓住一个简单而有力的前提——如果语言模型对一条序列不那么「困惑」（perplexity 更低），说明它更理解这条序列的独特模式，那它生成的表示就更适合下游预测。而掩码语言建模（MLM）正好是蛋白预训练的统治性范式，可以直接拿来在单条序列上最小化困惑度。

**核心 idea**：用测试时训练（TTT）的思路，在推理前用与预训练相同的掩码自监督目标，仅在一条目标序列上微调骨干，把「降低困惑度」当作免费的自监督信号，换取更好的下游表示。

## 方法详解

### 整体框架
ProteinTTT 建立在蛋白质机器学习普遍采用的 **Y 型架构** 上：一个共享骨干特征提取器 $f$ 作用在蛋白 token $x$ 上，上面挂两条分支——自监督预训练头 $g$（掩码语言建模头，把表示映回氨基酸类型）和下游微调头 $h$（结构头 / 适应度头 / 功能分类器）。常规流程里，预训练完成后 $h \circ f$ 就被冻结，测试时不再改动。

ProteinTTT 在这个固定流程里插入一步「定制」：拿到一条待测序列 $x$ 后，先不急着预测，而是借用闲置的自监督分支 $g \circ f$，在 $x$ 这一条序列上做 $T$ 步掩码微调，把骨干参数从预训练的 $\theta_0$ 调到针对该序列的 $\theta_x$，下游头 $h$ 始终冻结：

$$\text{ProteinTTT}: (h \circ f(\cdot; \theta_0),\, x) \mapsto h \circ f(\cdot; \theta_x)$$

定制完成后再用 $h \circ f(\cdot; \theta_x)$ 出最终预测。关键是定制只改 $f$ 的表示、不动 $h$ 的参数（例如 ESMFold 那 14 亿参数的结构头一个都不碰），靠「更好的表示」来抬升下游精度。处理完一条蛋白后参数重置回 $\theta_0$，再处理下一条。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["输入：单条目标序列 x"] --> B["单序列掩码定制<br/>在 x 上最小化 MLM 困惑度"]
    B --> C["稳定化训练<br/>LoRA + SGD + 梯度累积"]
    C --> D["置信度选步<br/>T 步中按 c 选最优 θx"]
    D -->|骨干换成 f(·;θx)，下游头 h 冻结| E["输出：结构 / 适应度 / 功能预测"]
```

### 关键设计

**1. 单序列掩码定制：把预训练目标直接当测试时自监督信号**

痛点是单蛋白没有任何额外数据可学，怎么微调？作者的答案是：序列本身就是监督信号。定制目标就是在目标序列 $x$ 上最小化掩码语言建模损失

$$L(x; \theta) = \mathbb{E}_{M \sim p_{\text{mask}}(M)}\left[\sum_{i \in M} -\log p(x_i \mid x_{\backslash M}; \theta)\right]$$

即随机遮住一部分位置 $M$，让模型从剩下的上下文 $x_{\backslash M}$ 重建被遮的真实（野生型）氨基酸 $x_i$，其中 $\log p(x_i\mid x_{\backslash M};\theta) = g(f(x_{\backslash M};\theta))_i$。这一步等价于在该序列上最小化困惑度——模型越能补全这条序列，就越「理解」它的独特模式，表示也就越好。为了不引入分布偏移，定制严格复刻预训练时的所有掩码与预处理习惯：相同的掩码比例（如 15%）或掩码数分布（如 beta 分布）、把 10% 被遮 token 换成随机 token、另 10% 保持原样、对长序列裁剪成 1024-token 片段等。正因为掩码建模在蛋白预训练里无处不在，这个定制目标几乎可以即插即用到任何 Y 型模型上；作者还在附录里演示了它同样能套到自回归模型和离散扩散模型（如 DPLM2）。

**2. 置信度引导的选步：无验证集时如何决定「调到第几步」**

定制只有一条蛋白，没法像常规训练那样在验证集上 early stopping，过度微调反而可能损害表示。作者的处理是：先固定跑 $T$ 步，得到一串参数快照 $\Theta = \{\theta_0, \theta_1, \dots, \theta_T\}$，再用一个置信度函数 $c$ 把最优的一步挑出来：

$$\theta_x = \arg\max_{\theta \in \Theta}\, c\big(h(f(x;\theta))\big)$$

对结构预测，$c$ 天然就是 pLDDT（模型自带的置信度），图 3 展示了 7EBL 链 B 随定制步数 TM-score 从 0.29 一路爬到 0.92、困惑度同步下降，第 7 步 pLDDT 最高正好被选中。若某个模型没有可用的置信度（如 DPLM2 没有训练好的 pLDDT 头），则退化为直接取末步 $\theta_x = \theta_T$。论文指出，用 pLDDT 当 $c$ 让 ProteinTTT 对超参数相当鲁棒，$T$ 可固定（如 $T=30$），只需调学习率和 batch size。

**3. 稳定化的大模型定制：让几十步微调既稳又省**

直接对 30 亿参数的 ESM2 做全参数微调既贵又容易训崩。作者用三件套把定制做轻做稳：用 **LoRA** 低秩适配，只更新少量参数即可适配大骨干；用 **梯度累积** 在显存受限下放大有效 batch；并刻意用 **SGD 而非 Adam**（跟随 Gandelsman 等的 TTT 经验），因为 SGD 在这种单样本、少步数的定制里更稳定、更可预测，不会因自适应学习率在一条样本上剧烈震荡。这三点合起来让 ProteinTTT 能挂到 ESM2-3B、ESM3 这类大模型上，且保持 ESMFold 量级的推理效率，比 AlphaFold2 快一个数量级。

### 损失函数 / 训练策略
定制目标即上文的掩码语言建模损失 $L(x;\theta)$，仅在单条目标序列（或其 MSA，见附录 C）上优化；优化器为 SGD + LoRA + 梯度累积，固定步数 $T$（如 30），按置信度函数 $c$（结构任务用 pLDDT）选出最优快照作为 $\theta_x$。下游头 $h$ 全程冻结，每处理完一条蛋白把骨干参数重置回 $\theta_0$。

## 实验关键数据

### 主实验
结构预测在 CAMEO 上、聚焦 ESMFold 低置信度的 18 条难靶标，5 个随机种子取均值。ProteinTTT 在四个不同骨干上都稳定超过各自的强基线（ESMFold 的掩码预测 MP、ESM3 的思维链 CoT）。

| 方法 | TM-score ↑ | LDDT ↑ |
|------|-----------|--------|
| ESM3 | 0.3480 | 0.3723 |
| ESM3 + CoT | 0.3677 | 0.3835 |
| **ESM3 + ProteinTTT** | **0.3954** | **0.4214** |
| ESMFold | 0.4649 | 0.5194 |
| ESMFold + MP | 0.4862 | 0.5375 |
| **ESMFold + ProteinTTT** | **0.5047** | **0.5478** |
| HelixFold-Single | 0.4709 | 0.4758 |
| **HelixFold-Single + ProteinTTT** | **0.4839** | **0.4840** |

适应度预测在 ProteinGym 零样本设置（186 条蛋白、250 万突变）上，ProteinTTT 对所有模型都提升 Spearman，且 **ProSST + ProteinTTT 刷新 SOTA**（配对 t 检验 $p<0.05$ 显著）。

| 模型 | Avg. Spearman ↑（基线 → +ProteinTTT） |
|------|----------------------------------------|
| ESM2 (35M) | 0.3211 → 0.3407 |
| ProGen2-small (151M) | 0.3255 → 0.3591 |
| SaProt (650M) | 0.4569 → 0.4583 |
| ProSST (K=2048) | 0.5068 → **0.5087**（新 SOTA） |

### 消融实验
论文以「不同骨干 / 不同规模 / 不同基线」的横向对照充当消融，核心结论可整理如下：

| 配置 | 关键现象 | 说明 |
|------|---------|------|
| ProteinTTT vs MP / CoT | TM-score 全面更高 | 定制优于「多次掩码采样」与「思维链迭代解码」 |
| DPLM2（无 pLDDT 头） | 仍有提升（取末步 $\theta_T$） | 即便无置信度函数、骨干是离散扩散预训练，掩码定制依旧有效 |
| 小模型 vs 大模型 | 小模型（35M）提升更大 | 大模型在 benchmark 上接近饱和，提升空间被压缩 |
| 低 MSA 深度蛋白 | 提升最明显 | 单序列定制对「同源序列稀少」的蛋白增益最大 |

### 关键发现
- ProteinTTT 的增益主要来自「困惑度下降 ↔ 表示变好 ↔ 下游变准」这条链路：图 3 中 TM-score 随困惑度下降单调上升，是方法假设的直接证据。
- 增益与数据稀缺程度强相关：MSA 越浅、训练里相似序列越少的蛋白，单序列定制帮助越大——正好命中「通用模型最弱、生物学家最需要」的区间。
- 难靶标上「修好的远多于修坏的」：ESMFold/DPLM2/HelixFold/ESM3 分别显著改善 7/4/5/6 条结构，仅轻微破坏 2/1/1/1 条，净收益为正。

## 亮点与洞察
- 把 CV 里的测试时训练范式干净地迁到蛋白质——「困惑度即免费监督」这个观察让方法不需要任何额外数据，却能针对单条序列做精修，思路简洁但切中生物场景痛点。
- 「不动下游头、只调骨干表示」是巧妙的解耦：ESMFold 14 亿参数的结构头原封不动，仅靠更好的输入表示就抬升结构预测，意味着任何 Y 型模型都能几行代码接入。
- 用 pLDDT 当置信度选步，把「无验证集怎么 early stop」这个 TTT 老问题在结构任务上自然化解，同时换来对超参数的鲁棒性。
- 方法对预训练范式不挑食：双向掩码、自回归、离散扩散都能用同一个掩码定制目标，迁移性可推广到更多蛋白基础模型乃至其他「Y 型 + 自监督」的科学建模任务。

## 局限与展望
- 选步依赖置信度函数 $c$：结构任务有 pLDDT 可用，但适应度/功能任务往往没有可靠的置信度，只能退回取末步 $\theta_T$，可能错过更优快照。
- 单条蛋白要跑 $T$ 步微调，相比一次前向推理有额外开销；虽仍比 AlphaFold2 快一个数量级，但大规模数据库（如 35 万条 BFVD）批量定制成本不可忽视。
- 在已接近饱和的大模型 / 高 MSA 深度蛋白上增益很小（部分指标甚至略降），方法的价值高度集中在「数据稀缺、分布偏移」的难样本上。
- 自定义指标如困惑度、pLDDT、TM-score 的相互关系是经验观察，理论保证仍较弱（作者在附录 A 给出与困惑度最小化的联系作为直觉解释）。

## 相关工作与启发
- **vs 测试时训练（TTT, Sun et al. / Gandelsman et al.）**：经典 TTT 为缓解 CV 的分布偏移而生，本文把它落到蛋白 PLM，并指出即使没有显式分布偏移、纯粹为「定制到单样本」也能涨点，拓宽了 TTT 的适用动机。
- **vs 蛋白特异/家族特异微调（Notin et al. / Samusevich et al.）**：这类方法需要收集目标相关的额外数据（同源序列、家族数据），而 ProteinTTT 只用目标序列本身、不假设任何外部数据，适用面更广。
- **vs ESMFold 掩码预测 / ESM3 思维链**：MP 靠对一条序列多次随机掩码采样多个结构、CoT 靠迭代解码，二者都不更新模型；ProteinTTT 真正把骨干调到更懂这条序列，主实验上稳定优于二者。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个面向蛋白机器学习的「单蛋白测试时定制」方法，把困惑度最小化当免费监督，角度新且自洽。
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖结构/适应度/功能三类任务、多骨干多规模、ProteinGym 刷 SOTA，外加抗体环与病毒库两个真实案例。
- 写作质量: ⭐⭐⭐⭐⭐ Y 型架构、定制目标、选步策略层层递进，图 1/3 的困惑度-精度对照很有说服力。
- 价值: ⭐⭐⭐⭐⭐ 几行代码即插即用、直击「单蛋白数据稀缺」的真实生物需求，对湿实验指导价值高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Extending Sequence Length is Not All You Need: Effective Integration of Multimodal Signals for Gene Expression Prediction](extending_sequence_length_is_not_all_you_need_effective_integration_of_multimoda.md)
- [\[NeurIPS 2025\] Is Sequence Information All You Need for Bayesian Optimization of Antibodies?](../../NeurIPS2025/computational_biology/is_sequence_information_all_you_need_for_bayesian_optimization_of_antibodies.md)
- [\[ICLR 2026\] SimpleFold: Folding Proteins is Simpler Than You Think](simplefold_folding_proteins_is_simpler_than_you_think.md)
- [\[ICLR 2026\] Towards Understanding the Shape of Representations in Protein Language Models](towards_understanding_the_shape_of_representations_in_protein_language_models.md)
- [\[ICLR 2026\] ProTDyn: A Foundation Protein Language Model for Thermodynamics and Dynamics Generation](protdyn_a_foundation_protein_language_model_for_thermodynamics_and_dynamics_gene.md)

</div>

<!-- RELATED:END -->
