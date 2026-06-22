---
title: >-
  [论文解读] The Coverage Principle: How Pre-Training Enables Post-Training
description: >-
  [ICLR2026][学习理论][coverage profile] 这篇论文从理论上回答了"预训练到底给后训练（RL / 测试时扩展）留下了什么"——答案不是交叉熵，而是一个叫 **coverage profile（覆盖度剖面）** 的量；作者证明 next-token prediction 会**隐式地**优化覆盖度，而且覆盖度比交叉熵泛化得更快、不受序列长度拖累，从而解释了"为什么交叉熵更低的模型反而 Best-of-N 更差"这一反常现象。
tags:
  - "ICLR2026"
  - "学习理论"
  - "预训练"
  - "测试时扩展"
  - "coverage profile"
  - "next-token prediction"
  - "最大似然"
  - "Best-of-N"
  - "泛化分析"
---

# The Coverage Principle: How Pre-Training Enables Post-Training

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=AUXvYQlQLZ](https://openreview.net/forum?id=AUXvYQlQLZ)  
**代码**: 随论文 supplementary material 提供（含 Figure 1/2 复现脚本）  
**领域**: 学习理论 / 预训练 / 测试时扩展  
**关键词**: coverage profile、next-token prediction、最大似然、Best-of-N、泛化分析

## 一句话总结
这篇论文从理论上回答了"预训练到底给后训练（RL / 测试时扩展）留下了什么"——答案不是交叉熵，而是一个叫 **coverage profile（覆盖度剖面）** 的量；作者证明 next-token prediction 会**隐式地**优化覆盖度，而且覆盖度比交叉熵泛化得更快、不受序列长度拖累，从而解释了"为什么交叉熵更低的模型反而 Best-of-N 更差"这一反常现象。

## 研究背景与动机

**领域现状**：现代语言模型走的是"大规模预训练（next-token prediction + 交叉熵）→ 针对性后训练（通常是可验证奖励的 RL，或 Best-of-N 这类测试时扩展）"两段式管线。业界默认"预训练投入越大、交叉熵越低，后训练就能炼出越强的模型"，于是用交叉熵 / 困惑度作为衡量预训练好坏的核心指标。

**现有痛点**：这个默认假设在经验上**会翻车**。已有多项工作观察到，从一个交叉熵更低的 next-token predictor 出发做后训练，下游表现**不一定**更好，有时甚至更差（论文 Figure 1 直接画出交叉熵 / KL 与 Pass@N 反相关的曲线）。也就是说，交叉熵这个我们花了无数算力去压低的量，**并不能可靠预测下游成功**。

**核心矛盾**：交叉熵（等价地，序列级 KL 散度）衡量的是模型在**整个分布**上的平均对数似然，它会被"缺失质量"（missing mass，即模型给某些罕见高质量回答分配了过低甚至零概率）拖出巨大的、甚至无穷的代价，而这个代价又随序列长度 $H$ 线性增长。但下游的 Best-of-N / RL 真正需要的，只是"模型对高质量回答有没有留下**足够采样到的概率**"——这是一个尾部 / 阈值性质，和平均似然根本不是一回事。

**本文目标**：精确刻画 next-token prediction 损失与下游表现之间的关系，找出"比交叉熵更能预测下游成功"的指标，并解释 next-token prediction 在什么机制下、什么条件下会产出一个下游可用的模型。

**切入角度**：作者提出用 **coverage（覆盖度）** 这把尺子来重新审视预训练——它直接量化"模型在高质量回答上压了多少概率质量"，而这恰好是 Best-of-N 成功的充要条件。

**核心 idea**：用**覆盖度剖面**代替交叉熵来连接预训练与后训练，并证明一个叫 **coverage principle** 的现象：最大似然 / next-token prediction 会隐式地把模型推向高覆盖度，而且覆盖度的泛化速度快于交叉熵、绕开了序列长度等问题相关参数的拖累。

## 方法详解

### 整体框架

这是一篇**理论**论文，"方法"就是一整条环环相扣的论证链，目标是把"预训练 → 下游成功"这件事建立在覆盖度而非交叉熵之上。整条链可以这样鸟瞰：

1. **建立度量**：先定义覆盖度剖面 $\text{Cov}_N$，并证明它对 Best-of-N 是**充要**的——好覆盖度 ⇔ 后训练能成功（Section 2）。
2. **否定旧度量**：证明交叉熵 / 序列级 KL 虽然能给出一个"换算到覆盖度"的缩放律（Proposition 3.1），但这个换算在有限样本下会因序列长度 $H$ 而退化成空洞预测（Proposition 3.2），所以交叉熵不是对的尺子（Section 3）。
3. **正面主结果**：证明 coverage principle——next-token prediction（最大似然）会隐式优化覆盖度，且覆盖度**泛化更快**、不依赖 $H$（Theorem 4.1 / 4.2，Section 4）。
4. **落到优化器**：把分析从"理想最大似然解"换成更真实的单遍 SGD，指出朴素 SGD 的覆盖度会被 $H$ 拖累，而**梯度归一化**能把这个依赖消掉（Section 5）。
5. **给出干预手段**：提出三类有可证收益的算法干预——测试时解码、checkpoint 选择锦标赛（Section 6）。

整条链的因果关系是清晰的：度量（覆盖度）→ 为什么旧度量不行（交叉熵的 $H$ 依赖）→ 为什么新度量天然被预训练优化（coverage principle）→ 真实优化器下怎么补救（梯度归一化）→ 实操还能怎么加码（干预）。下面的关键设计就按这条流向逐个展开，每个对应论证链上的一环。

### 关键设计

**1. 覆盖度剖面：把"能不能采样到好答案"量化成一个尾部 CDF**

针对的痛点是：交叉熵 / KL 取的是对数密度比的**均值**，会被缺失质量炸成无穷，无法预测下游。作者改用一个尾部量。对模型 $\hat\pi$ 相对参考分布 $\pi$（取数据分布 $\pi_D$）定义覆盖度剖面：

$$\text{Cov}_N(\pi \,\|\, \hat\pi) := \Pr_{x\sim\mu,\,y\sim\pi(\cdot|x)}\!\left[\frac{\pi(y\mid x)}{\hat\pi(y\mid x)} \ge N\right],$$

其中 $N$ 就是 Best-of-N 的采样次数。直觉上 $\text{Cov}_N$ 小，意味着"密度比 $\pi/\hat\pi$ 超过 $N$ 的坏样本很少"，即 $\hat\pi$ 在 $\pi$ 看重的回答上没有被压得太低，那么抽 $\tilde\Theta(N)$ 个样本就大概率能撞上好回答。作者证明（Propositions F.6 / F.7）：对任意下游任务策略 $\pi_T$，Best-of-N 的次优性 $\asymp \text{Cov}_N(\pi_T \,\|\, \hat\pi)$，即**好覆盖度是 Best-of-N 成功的充要条件**；再借助覆盖度的传递性，只要数据分布 $\pi_D$ 对任务有覆盖，就把问题归约成"研究 next-token prediction 何时让 $\text{Cov}_N(\pi_D\,\|\,\hat\pi)$ 好"。形式上，覆盖度剖面是对数密度比 $\log\frac{\pi(y|x)}{\hat\pi(y|x)}$ 的**整条 CDF**，而 KL 只是它的**均值**——这正是它比交叉熵信息更丰富、又不被尾部炸穿的原因。

**2. Coverage principle：next-token prediction 隐式优化覆盖度，且泛化快于交叉熵**

这是全文的核心定理（Theorem 4.1）。先看为什么交叉熵不行：作者证明序列级 KL 即使在最简单的自回归线性模型里也会随 $H$ 线性增长（Proposition 3.2，$D_{KL}\gtrsim H/n$），于是套用 KL→覆盖度的缩放律 $\text{Cov}_N \le \frac{D_{KL}}{\log(N/e)}$（Proposition 3.1）会预测出"测试时算力须随序列长度指数爆炸"的空洞结论。但实验里（Figure 2）覆盖度在不同 $H$ 下几乎不变、Best-of-N 照样成功。

作者用类似 Mendelson "small-ball" 反集中的技巧解释了这一点：利用对数损失的独特结构，证明最大似然估计满足

$$\text{Cov}_N(\hat\pi) \lesssim \underbrace{\frac{1}{\log N}\inf_{\varepsilon>0}\!\left(\frac{\log\mathcal{N}_\infty(\Pi,\varepsilon)}{n}+\varepsilon\right)}_{\text{fine-grained}} + \underbrace{\frac{\log\mathcal{N}_\infty(\Pi, c\log N)+\log(1/\delta)}{n}}_{\text{coarse-grained}}.$$

关键在两点：① **fine-grained 项不含 $H$、也不含密度比 $\log W_{\max}$**，只取覆盖数在小尺度 $\varepsilon$ 上的值，这正是 KL 界做不到的；② 这一项还被 $1/\log N$ 缩放，意味着**越往尾部走（$N$ 越大）覆盖度收敛越快**——这是对数损失带来的一种全新隐式偏置。coarse-grained 项则刻画"缺失质量"，它只在极大尺度 $\alpha\approx\log N$ 上评估覆盖数，复杂度随 $N$ 增大而消失。对过参数化自回归线性模型，作者进一步用"固有方差" $\sigma_\star^2$（可理解为**有效序列长度**，只数那些真正不确定的 token）替换 $H$，得到 $\mathbb{E}[\text{Cov}_N(\hat\pi)]\lesssim\sqrt{\sigma_\star^2/(n\log N)}+B^2/n$（Theorem 4.2），彻底摆脱对名义序列长度的依赖。这就是 coverage principle：**预训练在背地里优化的其实是覆盖度，而且优化得比交叉熵更快**。

**3. 梯度归一化：让真实的单遍 SGD 也拿到与长度无关的覆盖度**

前面的主结果针对"理想最大似然解"，但真实预训练用的是单遍（compute-optimal）SGD。作者证明朴素的序列级 SGD 虽然能优化覆盖度，但会吃 $H$ 的亏：$\mathbb{E}[\frac1T\sum_t\text{Cov}_N(\pi_{\theta_t})]\lesssim\frac{1}{\log N}(\sqrt{\sigma_\star^2/T}+B^2H/T)$，第二项的 $H$ 依赖还被下界证明是紧的（Proposition 5.1）。根因是 prompt 之间的异质性：有些 prompt 梯度尺度随 $H$ 增长、逼着学习率取小，另一些 prompt 又需要大学习率才不至于慢收敛。

解法是一个简单干预——**梯度归一化**：用 mini-batch 梯度 $\hat g$ 做归一化更新 $\theta_{t+1}\leftarrow\text{Proj}_\Theta(\theta_t+\eta\cdot\frac{\hat g}{\lambda+\|\hat g\|})$。作者证明（Theorem 5.1）它能拿到**与序列长度无关**的覆盖度界 $\sqrt{\sigma_\star^2/(T\log N)}+B^2/T+B/(K\log N)$，定性上追平 Theorem 4.2。值得注意的是 minibatch 不可省——它是抵消归一化引入偏置的必要条件；而这个归一化更新与 SignSGD / Adam 同源（Bernstein & Newhouse 2024），暗示 Adam 类优化器之所以好用，可能也是因为它们在隐式优化覆盖度。

**4. 测试时解码与锦标赛选 checkpoint：两类即插即用的覆盖度干预**

最后作者跳出标准 next-token prediction，给出两个有可证收益的干预，作为"用覆盖度指导算法设计"的概念验证。**测试时训练式解码**（Theorem 6.1）：在 token 级 SGD 基础上，解码时每采一个 token 就沿其对数似然做一步梯度上升、采完整段再重置参数，这种"improper"的采样能绕过 Proposition 5.1 给 proper 方法的 $H$ 下界，把覆盖度领头项再改进一个 $1/\sqrt{\log N}$ 因子。**覆盖度锦标赛选 checkpoint**（Theorem 6.2）：用经验覆盖度 $\widehat{\text{Cov}}_N(\pi'\|\pi)=\frac1n|\{i:\pi'(y_i|x_i)/\pi(y_i|x_i)\ge N\}|$，选 $\hat\pi=\arg\min_\pi\max_{\pi'}\widehat{\text{Cov}}_N(\pi'\|\pi)$，即挑出"对任何对手都最难被比下去"的模型。它的好处是**去掉了 $\pi_D\in\Pi$ 的可实现性假设**，只要类里存在好覆盖度模型就能选出来；用它替代"按交叉熵选 checkpoint"的标准做法，能为后续 RL / 测试时扩展挑到下游更强的起点（正是 Figure 1 里 ♢ 标记的选择）。

### 损失函数 / 训练策略
本文不引入新损失，分析的是标准最大似然目标 $\hat L_n(\pi)=\sum_{i=1}^n\log\pi(y_i\mid x_i)$（自回归展开即 next-token prediction）。"训练策略"层面的贡献是两类更新规则：① 梯度归一化更新 $\theta_{t+1}\leftarrow\text{Proj}_\Theta(\theta_t+\eta\,\hat g/(\lambda+\|\hat g\|))$；② token 级 SGD 配合测试时训练解码。全篇基于可实现性假设（Assumption 2.1，$\pi_D\in\Pi$）与有界参数 / 特征假设（Assumption 2.2）。

## 实验关键数据

> 本文是理论论文，实验为**图省合一的验证性实验**（图任务上的图推理 graph reasoning task），用来佐证"覆盖度比 KL 更能预测下游 Pass@N"以及"覆盖度不依赖序列长度"两个核心论断，而非刷 benchmark。

### 主实验：覆盖度 vs KL 谁更能预测 Pass@N（Figure 1）

| 观察对象 | KL / 交叉熵的表现 | 覆盖度剖面的表现 |
|--------|------------------|------------------|
| 训练过程中的变化 | 随训练**单调下降** | 可能**随训练退化**（与下游同步） |
| 与 Pass@N 相关性（小 $N$） | 与覆盖度相当 | 与覆盖度相当 |
| 与 Pass@N 相关性（大 $N$） | 明显变差，甚至**反相关** | 仍是**更好的预测子** |
| 按它选 checkpoint | 选出的模型 Pass@N 偏弱（红点） | 锦标赛选出的模型 Pass@N 更优（♢） |

核心结论：交叉熵单调变好，但 Pass@N 不跟着变好；覆盖度才与 Pass@N 同涨同跌，尤其在大 $N$ 的尾部区。

### 分析实验：覆盖度对序列长度的依赖（Figure 2）

| 量 | 随序列长度 $H\in\{8,16,24\}$ 的行为 | 含义 |
|----|-----------------------------------|------|
| 序列级 KL（收敛后） | 随 $H$ **线性增长** | 印证 Proposition 3.2 的 $H/n$ 下界 |
| 覆盖度 $\text{Cov}_{N=16}$（收敛后） | 几乎**与 $H$ 无关** | 印证 Theorem 4.1/4.2 的无 $H$ 依赖 |
| KL / Cov 之比 | 随 $H$ 显著放大 | 说明 Proposition 3.1 的 KL→Cov 换算过于保守 |

### 关键发现
- **交叉熵会"反指标"**：在图推理任务上交叉熵 / KL 可以与 Best-of-N 表现反相关，直接证伪"交叉熵越低下游越强"的默认假设。
- **缺失质量是元凶**：KL 被那些"学习器再好也覆盖不到的罕见回答"贡献了 $\log W_{\max}$（最坏可达 $H$）量级的代价，而覆盖度作为尾部 CDF 对此免疫；Bernoulli 模型的极端例子里 KL 期望为 $+\infty$、覆盖度却 $\lesssim\log(1/\delta)/n$。
- **尾部越深收敛越快**：覆盖度的 fine-grained 项带 $1/\log N$ 缩放，$N$ 越大泛化越快，这是对数损失独有的隐式偏置。
- **优化器层面可补救**：朴素 SGD 的覆盖度被 $H$ 拖累且下界紧，但梯度归一化（与 Adam / SignSGD 同源）能把 $H$ 依赖消掉。

## 亮点与洞察
- **换尺子而非换模型**：论文最"啊哈"的地方是指出我们一直用错了衡量预训练的指标——不是交叉熵不够低，而是交叉熵根本测不到下游真正需要的东西（尾部覆盖）。这把"为什么更低的 loss 没换来更强的下游"从悬案变成了可证现象。
- **覆盖度 = 对数密度比的 CDF，KL = 它的均值**：这个一句话的刻画极具迁移价值——任何"平均量被尾部炸穿"的场景都可考虑换成阈值 / CDF 量。
- **"固有方差 $\sigma_\star^2$ = 有效序列长度"**：把名义长度 $H$ 替换成"真正不确定的 token 数"，呼应了语言模型里"大多数 token 近乎确定、只有少数高熵"的经验观察，是一个可能被更广泛复用的实例相关复杂度概念。
- **理论解释了 Adam 为何好用**：梯度归一化能改善覆盖度、且与 Adam / SignSGD 同源，给"自适应优化器为何在 LLM 训练里更稳"提供了一个覆盖度视角的解释。
- **可直接落地的干预**：用覆盖度锦标赛而非交叉熵来选 RL 起点的 checkpoint，是个不改训练、只改"挑模型准则"的低成本提升。

## 局限与展望
- **可实现性假设较强**：核心定理依赖 $\pi_D\in\Pi$（Assumption 2.1）以及有界特征 / 凸参数空间，真实 Transformer 预训练并不满足；附录虽讨论了误设定情形，但主结论的直接适用性受限。
- **模型类受限**：SGD / 梯度归一化 / 测试时解码的细致分析都落在**自回归线性模型**（冻结特征图）上，离真实非线性 Transformer 还有距离。
- **覆盖度不可观测**：覆盖度和 KL 一样都不是可直接测量的量（交叉熵只是 KL 的可估上界），论文里的缩放律是理论预测而非可即时落地的工程指标，实操中如何高效估计覆盖度仍待解决。
- **实验为合成 / 受控任务**：验证用的是图推理任务，未在真实大模型预训练 → RL 全链路上做端到端验证。
- **改进方向**：把覆盖度分析推广到非线性模型与现代泛化理论（benign overfitting 等）的结合；设计可在大模型上廉价估计的覆盖度代理指标，使锦标赛选 checkpoint 等干预真正进入工程管线。

## 相关工作与启发
- **vs 交叉熵 / 缩放律范式（Kaplan、Hoffmann 等）**：他们以交叉熵 / 困惑度为预训练好坏的核心度量，本文证明这一度量在有限样本下会因序列长度退化成空洞预测，主张用覆盖度替代——更贴近下游 Best-of-N / RL 的真实需求。
- **vs "更低 loss ≠ 更强下游"的经验观察（Liu 2022、Zeng 2025、Chen 2025 等）**：那些工作经验性地发现了断裂现象，本文给出**理论机制**（缺失质量 + 覆盖度比交叉熵泛化更快）来解释它。
- **vs Best-of-N / 测试时扩展分析（Yue 2025、Wu 2025）**：他们指出 BoN 表现能预示 RL 后训练表现，本文进一步证明 BoN 成功的充要条件就是覆盖度，把测试时扩展接到了预训练度量上。
- **vs 测试时训练 / 动态评估（Krause 2019、Sun 2024、Akyürek 2025）**：本文借用其解码时更新参数的思想，但给出了**可证的覆盖度收益**（绕过 proper 方法的 $H$ 下界），赋予 TTT 一个新的理论动机。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用覆盖度重新定义"预训练为后训练留下什么"，并证明 coverage principle，是对最大似然泛化的全新细粒度理解。
- 实验充分度: ⭐⭐⭐ 理论为主，仅在受控图推理任务上做验证性实验，缺真实大模型端到端验证（对理论论文属合理范畴）。
- 写作质量: ⭐⭐⭐⭐⭐ 论证链条清晰，从度量→否定旧度量→主结果→优化器→干预层层递进，直觉与定理交替。
- 价值: ⭐⭐⭐⭐⭐ 把一个困扰业界的经验反常现象上升为可证理论，并给出选 checkpoint / 优化器 / 解码三类可落地干预，影响面大。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] From Predictors to Samplers via the Training Trajectory](from_predictors_to_samplers_via_the_training_trajectory.md)
- [\[ICLR 2026\] Strong Correlations Induce Cause Only Predictions in Transformer Training](strong_correlations_induce_cause_only_predictions_in_transformer_training.md)
- [\[ICLR 2026\] Training-Free Determination of Network Width via Neural Tangent Kernel](training-free_determination_of_network_width_via_neural_tangent_kernel.md)
- [\[ICLR 2026\] Tractability via Low Dimensionality: The Parameterized Complexity of Training Quantized Neural Networks](tractability_via_low_dimensionality_the_parameterized_complexity_of_training_qua.md)
- [\[ICLR 2026\] On the Convergence of Two-Layer Kolmogorov-Arnold Networks with First-Layer Training](on_the_convergence_of_two-layer_kolmogorov-arnold_networks_with_first-layer_trai.md)

</div>

<!-- RELATED:END -->
