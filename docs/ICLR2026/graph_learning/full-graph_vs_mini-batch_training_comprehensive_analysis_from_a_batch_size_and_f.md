---
title: >-
  [论文解读] Full-Graph vs. Mini-Batch Training: Comprehensive Analysis from a Batch Size and Fan-Out Size Perspective
description: >-
  [ICLR 2026][图学习][全图训练] 这篇论文把 GNN 的全图训练（full-graph）看作 batch size 与 fan-out size 都取到最大的 mini-batch 训练特例，从这两个超参数的视角同时做收敛、泛化、计算效率的理论与实证分析，得出一个反直觉结论：全图训练并不总是优于精心调参的小 batch mini-batch 训练。
tags:
  - "ICLR 2026"
  - "图学习"
  - "全图训练"
  - "mini-batch 训练"
  - "batch size"
  - "fan-out size"
  - "泛化界"
---

# Full-Graph vs. Mini-Batch Training: Comprehensive Analysis from a Batch Size and Fan-Out Size Perspective

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ZSfgsh43vT](https://openreview.net/forum?id=ZSfgsh43vT)  
**代码**: https://github.com/LIUMENGFAN-gif/GNN_fullgraph_minibatch_training (有)  
**领域**: 图学习 / GNN 训练分析  
**关键词**: 全图训练, mini-batch 训练, batch size, fan-out size, 泛化界

## 一句话总结
这篇论文把 GNN 的全图训练（full-graph）看作 batch size 与 fan-out size 都取到最大的 mini-batch 训练特例，从这两个超参数的视角同时做收敛、泛化、计算效率的理论与实证分析，得出一个反直觉结论：全图训练并不总是优于精心调参的小 batch mini-batch 训练。

## 研究背景与动机
**领域现状**：GNN 训练有两条主流路线。全图训练一次性处理整张图，每个节点跨多层 message-passing 聚合全部邻居信息；mini-batch 训练则把图切成子图、对采样的局部邻域迭代训练。两者的系统设计需求截然不同——全图训练要高效的跨设备聚合通信，mini-batch 训练要优化 CPU-GPU 数据加载。选哪条路线直接决定了系统怎么搭，因此弄清两者优劣很关键。

**现有痛点**：要系统比较这两条路线，`batch size`（一个 batch 里采样多少节点）和 `fan-out size`（每个节点每跳采样多少邻居）是天然的分析视角，因为全图训练正好是 batch size 取到 $n_{\text{train}}$、fan-out 取到最大度 $d_{\max}$ 的极端 mini-batch。但现有研究要么只孤立看其中一个超参数，要么只看收敛、只看精度、或只看系统效率的单一侧面，给不出两条路线之间整体权衡的洞察。已有的经验性比较（如 Bajaj 等）又高度依赖硬件和运行环境，结论难以推广。

**核心矛盾**：现有 GNN 理论分析往往依赖过强的简化假设——无限宽度（把每个神经元的梯度噪声平均掉）或线性模型 + 凸损失（消掉局部极小），这些假设恰好抹掉了 batch size 和 fan-out size 对训练动态的真实影响，因此无法回答它们到底怎么作用。

**本文目标**：在 transductive 节点分类任务上，系统刻画 batch size $b$ 与 fan-out size $\beta$ 如何影响 GNN 的优化动态、泛化能力和计算效率，并据此判断全图训练相对小 batch 的真实优劣。

**切入角度**：作者用带 ReLU 的单层有限宽 GNN 作分析载体——它保留了有限宽度和非线性带来的现象（这正是以往简化掉的部分），又简单到能精确刻画两个超参数对训练动态的作用；并把全图训练统一进 mini-batch 框架，使二者可在同一套理论里直接对比。

**核心 idea**：用 batch size 和 fan-out size 这一对超参数当"统一标尺"，证明它们对收敛和泛化的影响是**非各向同性（non-isotropic）**的——batch size 主导收敛、fan-out size 主导泛化——从而在资源受限时给出该调哪个参数的可操作指南。

## 方法详解

### 整体框架
论文的"方法"不是一个训练系统，而是一套**统一的分析框架**：把全图训练改写成 mini-batch 训练在 $b=n_{\text{train}}$、$\beta=d_{\max}$ 时的极限，于是"全图 vs mini-batch"之争被翻译成"两个超参数怎么取值"的问题。在这套框架下，作者沿三个维度分别建立分析工具——优化动态（收敛）、泛化、计算效率，每个维度都先做理论推导、再用实证验证，最后汇总成调参建议。

整张图里被反复用到的对象是归一化邻接矩阵 $\tilde{A} = (D^{\text{in}}+I)^{-\frac12}(A+I)(D^{\text{out}}+I)^{-\frac12}$ 的行 $\tilde{a}_i$，单层 GNN 输出为 $z_i = \sigma(\tilde{a}_{\text{train},i} X W^\top)$，其中 $\tilde{a}_{\text{train},i}X$ 就是节点 $i$ 的聚合嵌入。batch size 改变的是被纳入的训练节点集合（$\tilde{A}$ 的行数），fan-out size 改变的是每行里有多少非零邻居项。两个超参数因此从两个不同方向逼近全图的 $\tilde{A}^{\text{full}}$，这正是后面所有分析的物理直觉来源。这是纯机制分析（核心是矩阵/梯度推导），没有清晰的多阶段 pipeline，故不配框架图。

### 关键设计

**1. 解耦聚合与非线性激活：让带 ReLU 的有限宽 GNN 可被精确分析**

这一步针对的是"现有理论必须假设无限宽或线性"的痛点。难点在于 ReLU 把聚合后的节点特征当输入，使损失和梯度表达式解析上不可处理。作者的做法是把聚合从激活里"剥离"出来：要么通过重写平方损失项把聚合从 ReLU 中提取出来，要么用一个逐位置的 0/1 指示矩阵改写 ReLU，让它能直接乘到聚合后的特征上。这样一来，graph 结构（由 $b,\beta$ 决定）在损失和梯度里的作用就被孤立出来，可以单独追踪，而不必再依赖会"平均掉梯度噪声"的无限宽假设。这是整套理论能延伸到非正则图 + 非线性激活、贴近真实实践的前提。

**2. 收敛分析与 Obs.1：batch size 比 fan-out size 更主导优化动态**

在上述设置下，作者给出 mini-batch 训练在 MSE（Theorem 1）和 CE（Theorem 2）下到达 $\epsilon$ 误差所需迭代数 $T$ 的上界，例如 MSE 下 $T = O(n_{\text{train}} h^2 b^{5/2}\beta^{-1/2}\epsilon^{-1}\log(h^2\epsilon^{-1}))$，且当 $\beta\to d_{\max}$、$b\to n_{\text{train}}$ 时上界正好退化为全图训练的结果。关键观察分两层：Remark 3.1 指出**增大 batch size 在 MSE 下需要更多迭代、在 CE 下却更少**（方向相反，这点不同于 DNN），而**增大 fan-out size 在两种损失下都一致地减少迭代**；Remark 3.2 进一步用斜率 $|\partial T/\partial \beta|$ 刻画 fan-out 影响的强弱——MSE 下为 $O(\beta^{-3/2}b^{5/2})$、CE 下为 $O(\beta^{-7/2}b^{-1})$，说明 $\beta$ 增大到一定程度后边际收益递减。由此得到 **Obs.1：收敛对 batch size 比对 fan-out size 更敏感**——因为 $b$ 会让两种损失出现相反的收敛趋势，而 $\beta$ 始终单向。注意这无法用 DNN 那套"大 batch 降梯度方差所以更快收敛"来解释，作者把 message-passing 对损失/梯度的影响纳入后才给出自洽解释。**可操作含义**：内存受限时若关心收敛，调 fan-out size 更可靠，且取中等值（如 15 附近）能在收敛速度与计算开销间取得平衡。

**3. 用 Wasserstein 距离做泛化分析与 Obs.2：fan-out size 比 batch size 更主导泛化**

收敛之外，作者在 PAC-Bayesian 框架下给出 mini-batch + MSE 的泛化界（Theorem 3）：期望测试风险与经验训练风险之差 $= O\!\left(\frac{1}{n_{\text{train}}} + \Delta(\beta,b)\right)$。核心是用 **Wasserstein 距离** $\Delta(\beta,b)$ 量化训练图与测试图的结构差异——它对 non-i.i.d. 的图数据、尤其是几何变化的度量很合适。其距离函数里关键项是 $\delta^{\text{full-mini}}_i = \lVert \tilde{a}^{\text{full}}_{\text{train},i} - \tilde{a}^{\text{mini}}_{\text{train},i}\rVert_F^2$，刻画每个训练节点上全图聚合与 mini-batch 聚合的结构差异。Remark 4.1 表明增大 $b$ 或 $\beta$ 都会缩小 $\Delta(\beta,b)$、从而改善泛化，但二者机制不同：增大 $\beta$ 会把"原本没采到但真实存在的边"重新纳入，让 $\delta^{\text{full-mini}}_i$ 里的零项变非零，可能引入轻微的非单调波动；而增大 $b$ 只是把更多训练节点纳入求和，不会新增边。因此得到 **Obs.2：泛化对 fan-out size 比对 batch size 更敏感**，因为 $\beta$ 直接控制每个训练节点的感受野、在 $\Delta(\beta,b)$ 里作用更复杂。**可操作含义**：内存受限时若关心泛化，调 batch size 更稳定（波动更小）。这条 Obs.2 与 Obs.1 恰好"对称互补"，共同构成 non-isotropic 的核心论点。

**4. iteration-to-accuracy：一个硬件无关的收敛评测指标**

最后一块针对实证评测的痛点：常用的 time-to-accuracy（达到目标验证精度所需时间）高度依赖硬件，把"每次迭代的模型性能提升"和"每秒处理节点数的计算效率"纠缠在一起，导致同一方法在不同硬件上结论翻转。作者补充 **iteration-to-accuracy**（达到目标精度所需迭代数）作为硬件无关指标，只刻画训练过程中的性能提升本身。实证上它在不同硬件环境间的最大波动仅 41.28%，而 time-to-accuracy 高达 2787.05%。这让从业者能用已知的 iteration-to-accuracy 趋势先缩小 batch/fan-out 的取值范围，再用少量短跑考虑硬件相关的实际运行时间，从而降低调参开销。

### 损失函数 / 训练策略
全图训练用梯度下降（GD）更新 $W^{\text{full}}_{t+1} = W^{\text{full}}_t - \eta_t \nabla \hat{L}_{\text{train}}$；mini-batch 训练用 SGD，随机梯度 $\hat{G}_t = \frac1b\sum_{i\in\text{采样节点}}\nabla l(W^{\text{mini}}_t, \tilde{a}^{\text{mini}}_{\text{train},i})$。损失同时覆盖 Cross-Entropy（CE）与 Mean Squared Error（MSE）两种，这一区分至关重要——因为 batch size 在两种损失下对收敛的影响方向相反。

## 实验关键数据

### 主实验
在 reddit、ogbn-arxiv、ogbn-products、ogbn-papers100M 四个真实数据集与 GCN / GraphSAGE / GAT 三个模型上验证。下表为多层 GraphSAGE（去掉 dropout）在图相关超参数网格调优后，全图 vs mini-batch 的最佳测试精度对比：

| 数据集 | 全图训练 | mini-batch 训练 | 差距 |
|--------|---------|----------------|------|
| Reddit | 96.13 | **96.32** | mini-batch 反超 +0.19 |
| Ogbn-arxiv | 70.96 | **71.16** | mini-batch 反超 +0.20 |
| Ogbn-products | 77.92 | **78.80** | mini-batch 反超 +0.88 |
| Ogbn-papers100M | **59.54** | 58.52 | 全图领先 1.02 |

mini-batch 的最佳精度与全图训练的差距始终在 1.74% 以内，且在三个数据集上还反超——直接支撑"全图训练并不一致优于调好参的 mini-batch"这一核心结论。

### 消融 / 分析实验
| 分析维度 | 关键现象 | 说明 |
|---------|---------|------|
| iteration-to-acc vs time-to-acc | 跨硬件波动 41.28% vs 2787.05% | iteration-to-acc 才是硬件无关的可靠指标 |
| batch size 对收敛 | MSE 与 CE 下趋势相反 | 验证 Obs.1：收敛对 batch size 更敏感 |
| fan-out size 对泛化 | 引起更频繁、更剧烈的测试精度波动 | 验证 Obs.2：泛化对 fan-out size 更敏感 |
| 大 fan-out / 大 batch（CE） | fan-out > 15 或 batch > 半数训练节点时精度退化 | fan-out 导致的退化比 batch 更严重 |

### 关键发现
- **非各向同性是主线**：batch size 主导收敛、fan-out size 主导泛化，两者在收敛/泛化/效率三个维度上的作用各不相同，必须分开调。
- **CE 下的性能退化**：大 batch 让模型收敛到 sharp minima（尖锐极小），大 fan-out 因聚合过多邻居导致过拟合，两者都伤泛化，且 fan-out 的伤害更重；MSE 因梯度在预测边界附近更弱、倾向 flatter minima，退化不明显。
- **计算效率**：吞吐量随 batch size 增大而提升（固定计算被摊到更多数据），随 fan-out size 增大而下降（message-passing 计算需求上升）；整体上 mini-batch 的计算效率优于全图训练。
- **调参建议**：平均度小于 50 的数据集，建议 batch size 保持在训练节点数的一半以下、fan-out size 控制在 15 以内，以兼顾性能与效率、避免泛化退化。

## 亮点与洞察
- **把"路线之争"降维成"超参数取值"**：将全图训练视为 mini-batch 的 $b,\beta$ 极限，是个非常干净的统一视角，让原本依赖硬件的经验比较变成可证明的理论问题——这个 framing 本身就值得借鉴到其它"两条路线哪个好"的系统问题上。
- **non-isotropic 的实用价值**：batch size 管收敛、fan-out size 管泛化这条结论可以直接落到内存受限的调参决策上（关心收敛调 fan-out、关心泛化调 batch），是少有的"理论能直接指导旋钮怎么拧"的结果。
- **iteration-to-accuracy 这个指标可迁移**：任何想做硬件无关、可跨环境推广的训练效率评测，都可以借用"迭代数指标 + 短跑校准硬件"的思路，把模型性能提升与硬件吞吐解耦。
- **解耦聚合与 ReLU 的技巧**：用 0/1 指示矩阵改写 ReLU、把图结构从非线性里孤立出来，是让"有限宽 + 非线性"GNN 可分析的关键手法，对后续 GNN 训练动态理论有参考意义。

## 局限与展望
- **理论主要建立在单层 GNN 上**：虽然附录讨论了多层推广、且多层 GraphSAGE 实验也大体吻合，但深层 GNN 优化动态更复杂，实验里已观察到跨 batch/fan-out 的轻微波动，严格的多层理论仍待补全。
- **任务范围有限**：分析聚焦 transductive 节点分类，link prediction 等任务只在附录展望中提及，结论能否迁移尚未验证。
- **激活与损失的覆盖**：理论主要围绕 ReLU + MSE/CE，其它激活函数下的行为列为未来工作。
- **泛化界是上界形式**：$O(1/n_{\text{train}} + \Delta(\beta,b))$ 给的是趋势性指导，$\Delta(\beta,b)$ 随 $\beta$ 变化还存在小幅非单调波动，实际调参时这些波动可能让"调哪个更稳"的判断没那么干净。

## 相关工作与启发
- **vs Bajaj 等（唯一已有的全图 vs mini-batch 比较）**：他们做纯经验评测整体性能，但不研究 batch size / fan-out size 这些关键超参数对性能和效率的影响，因此漏掉了调参带来的权衡空间；本文用统一的超参数视角 + 理论分析补上了这块。
- **vs Yuan 等 / Hu 等（关注超参数但受限）**：Yuan 等缺理论支撑、只考虑远小于全图规模的有限 batch/fan-out 值、且忽略两者交互；Hu 等只靠梯度方差解释 batch size、不考虑 fan-out size，导致解释与自身实证观察冲突。本文同时建模两个超参数及其交互，并指出 DNN 那套梯度方差解释不能直接套到 GNN 上。
- **vs DNN 的 batch size 理论**：由于 GNN 的 message-passing 过程，DNN 上"大 batch 降方差 → 更快收敛"的结论无法直接迁移（本文 Obs.1 即一个反例：MSE 下大 batch 反而更慢）。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把全图训练统一进 mini-batch 框架、用 Wasserstein 距离做 GNN 泛化分析，视角和工具都新。
- 实验充分度: ⭐⭐⭐⭐ 覆盖四数据集三模型、CE/MSE 双损失、多硬件环境，但多为趋势验证图，主表只有一张精度对比。
- 写作质量: ⭐⭐⭐⭐ 理论—观察—含义的结构清晰，但定理与符号较密，对非理论读者门槛偏高。
- 价值: ⭐⭐⭐⭐⭐ "全图不总是更好"+ non-isotropic 调参指南是对 GNN 训练实践很实在的指导。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Out-of-Distribution Graph Models Merging](out-of-distribution_graph_models_merging.md)
- [\[ICLR 2026\] DHG-Bench: A Comprehensive Benchmark for Deep Hypergraph Learning](dhg-bench_a_comprehensive_benchmark_for_deep_hypergraph_learning.md)
- [\[ICML 2025\] Does Graph Prompt Work? A Data Operation Perspective with Theoretical Analysis](../../ICML2025/graph_learning/does_graph_prompt_work_a_data_operation_perspective_with_theoretical_analysis.md)
- [\[ICLR 2026\] Adaptive Mixture of Disentangled Experts for Dynamic Graph Out-of-Distribution Generalization](adaptive_mixture_of_disentangled_experts_for_dynamic_graph_out-of-distribution_g.md)
- [\[ICML 2025\] LLM Enhancers for GNNs: An Analysis from the Perspective of Causal Mechanism Identification](../../ICML2025/graph_learning/llm_enhancers_for_gnns_an_analysis_from_the_perspective_of_causal_mechanism_iden.md)

</div>

<!-- RELATED:END -->
