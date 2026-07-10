---
title: >-
  [论文解读] Geometric Gradient Rectification for Safe Open-Set Semi-Supervised Learning
description: >-
  [ECCV 2026][开放集半监督学习] 本文提出 GGR（Geometric Gradient Rectification），一种即插即用的梯度空间校正框架：以有监督梯度为锚点，将与之冲突的无监督辅助梯度投影到安全半空间内，使辅助更新在所选参数块上与有监督方向保持一阶非对抗，从而在不依赖精确 OOD 检测的前提下同时提升闭集泛化与开集鲁棒性。
tags:
  - "ECCV 2026"
  - "开放集半监督学习"
  - "梯度校正"
  - "优化几何"
  - "伪标签噪声"
  - "非对抗更新"
---

# Geometric Gradient Rectification for Safe Open-Set Semi-Supervised Learning

**会议**: ECCV 2026  
**arXiv**: [2606.26973](https://arxiv.org/abs/2606.26973)  
**代码**: [https://github.com/JiaheChen2002/GGR](https://github.com/JiaheChen2002/GGR) (有)  
**领域**: 自监督 / 表示学习  
**关键词**: 开放集半监督学习、梯度校正、优化几何、伪标签噪声、非对抗更新

## 一句话总结
本文提出 GGR（Geometric Gradient Rectification），一种即插即用的梯度空间校正框架：以有监督梯度为锚点，将与之冲突的无监督辅助梯度投影到安全半空间内，使辅助更新在所选参数块上与有监督方向保持一阶非对抗，从而在不依赖精确 OOD 检测的前提下同时提升闭集泛化与开集鲁棒性。

## 研究背景与动机
开放集半监督学习（OSSL）的现实场景是：标注数据仅覆盖已知类别（ID），而无标注池中混有分布外（OOD）样本。现有方法分两大范式——过滤派（如 OpenMatch、ProSub）通过阈值/熵/角度检测剔除可疑样本以避免 OOD 污染，但过度过滤会误杀难分 ID 样本，造成特征饥饿；利用派（如 IOMatch、MTCF）将 OOD 归入统一未知类以最大化数据利用率，但伪标签错误时会把难分 ID 推向错误目标，生成的辅助梯度与有监督学习方向直接对抗（$\langle g_s, g_u \rangle < 0$）。

核心矛盾在于：**样本层面的 ID/OOD 分离本质上不可靠**——OOD 和难分 ID 在低标签 regime 下难以区分，而错误的分离决策在梯度层面表现为破坏性干扰。本文做了一个关键的视角转换：不再纠结"该不该用这个样本"，而是直接检查"这个样本产生的梯度会不会伤害有监督学习"。从梯度几何来看（原文 Figure 1 右图），OOD 梯度大多与有监督方向近似正交（随机噪声），而误分类难分 ID 产生的梯度才是真正的方向性对抗——它们与有监督方向高度负相关。核心 insight：**OSSL 的鲁棒性不应依赖脆弱的样本筛选，而可以在梯度空间直接做几何控制——只纠正冲突分量，保留正交的有用信号。**

## 方法详解

### 整体框架
GGR 是一个纯梯度空间的后处理框架，不改动任何前向传播或损失函数定义。给定 OSSL 基方法（如 FixMatch、IOMatch、DAC），在每次迭代中先正常计算有监督损失 $\mathcal{L}_s$ 和无监督辅助损失 $\mathcal{L}_u$，得到两个梯度 $g_s$ 和 $g_u$，然后对 $g_u$ 做几何校正得到 $\tilde{g}_u$，最后用校正后的组合梯度 $g_s + \lambda_u \tilde{g}_u$ 更新参数。整个流程可以概括为三步：计算双梯度、检测冲突并投影、组合更新。由于是纯梯度后处理，GGR 不对基方法的伪标签机制、置信度阈值、数据增强策略做任何假设，可作为插件接入任意 OSSL 方法。

### 关键设计

**1. 梯度冲突诊断与一阶非对抗原则：从样本筛选到梯度控制**

作者首先形式化了 OSSL 中梯度冲突的来源。在标准复合目标 $\min_\theta \mathcal{L}_s + \lambda_u \mathcal{L}_u$ 下，当无标注样本的伪标签错误时，$\mathcal{L}_u$ 对该样本产生的梯度 $g_u$ 可能与 $g_s$ 的内积为负，即 $\langle g_s, g_u \rangle < 0$。这种破坏性干扰会削弱 ID 判别特征——比简单的方差噪声更致命，因为它在方向上直接对抗有监督信号。由此提出**一阶非对抗要求**：校正后的辅助方向 $\tilde{g}_u$ 必须满足 $\langle g_s, \tilde{g}_u \rangle \geq 0$，即辅助更新在当前步不得对抗有监督下降方向。这定义了一个以 $g_s$ 为法向量的闭半空间 $\mathcal{H}_{\text{safe}}(g_s) = \{d \in \mathbb{R}^D \mid \langle d, g_s \rangle \geq 0\}$，所有校正操作都在这个几何约束下进行。这个原则的非对称性是关键——有监督梯度是标签验证的锚点，不可修改；只纠正辅助梯度，这区别于 PCGrad 等对称梯度手术方法。

**2. 向量级校正器 (VLR)：半空间投影的闭式解**

VLR 是最基础的校正器，直接用当前 mini-batch 的 $g_s$ 作为锚点。将 $g_u$ 欧氏投影到 $\mathcal{H}_{\text{safe}}(g_s)$ 上，得到闭式解：

$$\tilde{g}_u = \begin{cases} g_u - \frac{\langle g_u, g_s\rangle}{\|g_s\|_2^2} g_s, & \text{if } \langle g_u, g_s\rangle < 0 \text{ and } \|g_s\|_2 > 0 \\ g_u, & \text{otherwise} \end{cases}$$

当检测到冲突（内积为负）时，该公式精确移除 $g_u$ 中指向 $-g_s$ 的分量，同时保留正交分量 $g_u - \frac{\langle g_u, g_s\rangle}{\|g_s\|_2^2} g_s$。这个正交分量可能仍携带有用的表示学习信号（如对 ID 类内方差的学习）。VLR 满足两个优雅的数学性质：(1) 对齐恒等式 $\langle g_s, \tilde{g}_u \rangle = \max(0, \langle g_s, g_u \rangle) \geq 0$，即校正后的内积非负；(2) 最小干预原则——在所有满足约束的方向中，VLR 对 $g_u$ 的修改量（欧氏距离）最小。VLR 的优点是零超参、计算开销仅 $O(D)$（一次内积加一次向量减法），是默认推荐配置。

**3. 子空间感知校正 (OSR 与 CSR)：用历史梯度平滑锚点噪声**

VLR 的问题是 mini-batch 的 $g_s$ 在低标签 regime 下噪声很大，单步锚点可能不稳定。为此，作者维护一个由近期有监督梯度构成的低维正交基 $U \in \mathbb{R}^{D \times k}$（通过 Gram-Schmidt 增量更新 + 定期 QR 正交化），用子空间 $\mathcal{S} = \text{span}(U)$ 替代瞬时 $g_s$ 作为锚点，提出两种变体：

- **正交子空间校正 (OSR)**：将 $g_u$ 投影到 $\mathcal{S}$ 的正交补空间，$\tilde{g}_u^{\text{orth}} = (I - UU^\top) g_u$。这是更保守的选择——完全移除 $g_u$ 在 $\mathcal{S}$ 内的分量，确保 $\langle g_s, \tilde{g}_u^{\text{orth}} \rangle = 0$（当 $g_s \in \mathcal{S}$ 时）。实验显示 OSR 在开集 BA 上更稳定，对各种维度 $d$ 不敏感。

- **锥子空间校正 (CSR)**：将 $g_u$ 投影到由 $U$ 各列定义的凸锥 $\mathcal{C}(U) = \{d \mid \langle d, u_i \rangle \geq 0, \forall i\}$ 上，$\tilde{g}_u^{\text{signed}} = g_u - U \min(U^\top g_u, 0)$。这是更宽松的选择——只裁剪与各锚向量方向冲突的负分量，保留正分量和正交分量。CSR 可以取得更高的闭集精度峰值（如 CIFAR-100 上 $d=10$ 时 58.47%），但对基的质量更敏感。

两种子空间变体体现了"保守性-利用率"权衡：OSR 更安全但可能丢弃有用信号，CSR 更充分利用但风险更高。维度 $d$ 的选择也很关键——太小锚点不足，太大会过度约束、压制有益的辅助信号。VLR 仍是效率最优的默认方案，子空间变体适合标签极少时追求更稳定表现。

### 一个完整示例：难分 ID 样本的梯度冲突与校正

考虑 CIFAR-10 的 6/4 开集划分，标签量 5/类。一个属于"猫"类（ID）的难分样本，由于图像模糊且与"狗"类特征接近，基方法 FixMatch 的伪标签机制将其误标为"狗"。在标准 Softmax 交叉熵下：

- $g_s$（来自正确标注的猫/狗/...样本的有监督梯度）主要沿增强猫-狗判别边界的方向
- $g_u$（来自该误标样本的辅助梯度）推动模型将该样本从猫移向狗——恰好与 $g_s$ 形成负内积，$\langle g_s, g_u \rangle \approx -0.3\|g_s\|\|g_u\|$

VLR 检测到冲突后，计算校正量 $\frac{\langle g_u, g_s\rangle}{\|g_s\|_2^2} g_s$（约为 $g_u$ 在 $-g_s$ 方向上的投影长度），从 $g_u$ 中减去该分量。校正后的 $\tilde{g}_u$ 与 $g_s$ 内积归零——不再对抗有监督方向，但 $g_u$ 中与 $g_s$ 正交的部分（如对该样本纹理、背景等通用视觉特征的学习）被完整保留。最终参数更新 $g_s + \lambda_u \tilde{g}_u$ 既推进了有监督目标，又没有丢失无标注池中的表示学习价值。

### 损失函数 / 训练策略
GGR 不修改基方法的损失函数定义，只在梯度聚合阶段介入。参数更新公式为 $\theta_{t+1} \leftarrow \theta_t - \eta(g_s + \lambda_u \tilde{g}_u)$，其中 $\tilde{g}_u$ 由所选校正器（VLR / OSR / CSR）产生。校正范围 $\mathcal{P}$ 可灵活指定为 backbone、task-specific head 或全部参数；默认只对 backbone 做校正，head 用原始梯度更新，以隔离表示层面的冲突缓解。所有基方法的超参（学习率、$\lambda_u$、置信度阈值等）保持不变，确保性能增益可归因于梯度校正本身。VLR 是同质函数，$\lambda_u$ 在整流前后作用等价——可以先乘权重再投影，也可以先投影再乘权重。

## 实验关键数据

### 主实验
在 CIFAR-10/100（多种 seen/unseen 划分 + 标签预算组合）和 ImageNet-30（20/10 split，1% 与 5% 标签比）上，将 GGR（默认 VLR）作为插件接入 OpenMatch、IOMatch、DAC 三个代表性 OSSL 基线。

**CIFAR-10/100 闭集精度（代表性设置摘录）**：

| 数据集 | 划分 | 标签/类 | IOMatch | +GGR | DAC | +GGR |
|--------|------|---------|---------|------|-----|------|
| CIFAR-10 | 6/4 | 5 | 89.87 | **91.87** | 86.69 | **87.63** |
| CIFAR-10 | 6/4 | 25 | 93.61 | **93.64** | 93.02 | **93.25** |
| CIFAR-100 | 50/50 | 5 | 59.49 | **60.61** | 56.94 | **57.19** |
| CIFAR-100 | 20/80 | 5 | 56.48 | **56.75** | 53.30 | **53.45** |

**ImageNet-30 闭集/开集精度**：

| 评估 | 标签比 | OpenMatch | +GGR | IOMatch | +GGR | DAC | +GGR |
|------|--------|-----------|------|---------|------|-----|------|
| 闭集 | 1% | 58.55 | **59.15** | 85.18 | **85.90** | 86.40 | **87.40** |
| 闭集 | 5% | 86.72 | **87.43** | 90.32 | **90.82** | 93.25 | **93.33** |
| 开集 BA | 1% | 14.75 | **15.56** | 74.88 | **75.46** | 77.78 | **78.40** |
| 开集 BA | 5% | 61.51 | **68.15** | 81.03 | **82.14** | 87.81 | **88.74** |

趋势清晰：GGR 在低标签 regime 下提升最大（标签越少伪标签噪声越严重，梯度冲突越频繁），且对过滤派（OpenMatch）和利用派（IOMatch、DAC）均有效，验证了梯度级控制是独立于样本筛选策略的互补手段。

### 消融实验

**校正范围消融（CIFAR10-6-30，VLR）**：

| 方法 | 校正范围 | 闭集精度 | 开集 BA |
|------|----------|----------|---------|
| DAC | — | 86.69 | 72.02 |
| +GGR | backbone | 87.63 | 73.72 |
| +GGR | head | **90.38** | **75.24** |
| +GGR | both | 88.47 | 75.06 |
| IOMatch | — | 89.87 | 72.50 |
| +GGR | backbone | **91.87** | **74.13** |
| +GGR | head | 90.34 | 72.61 |
| +GGR | both | 89.56 | 71.99 |

关键发现：最优校正范围因基方法和任务难度而异。DAC 有多辅助头时，对 head 做校正收益最大；IOMatch 则 backbone 校正最稳定。校正 backbone 或全模型是稳健的默认选择。

**子空间维度 $d$ 的影响（CIFAR-100 50/50）**：OSR 在整个 $d \in [1, 50]$ 范围内开集 BA 表现稳定，CSR 在 $d=10$ 附近取得闭集精度峰值但方差更大。中等 $d$（如 10）在"保守-利用"之间取得最佳平衡。

**效率分析（基于 DAC，单步时间/显存）**：VLR 相比基线仅增加约 11% 时间和约 300MB 显存；OSR ($d=10$) 开销约 1%，CSR ($d=10$) 约 11%。子空间维度增大到 50 后开销线性增长但精度不再同比例提升，推荐默认使用 VLR 或 $d=10$ 子空间变体。

### 关键发现
- 梯度冲突诊断显示，基线的原始冲突率非零且持续存在，GGR 将施加更新后的残余冲突率降至接近零，累积冲突遗憾几乎平坦——直接验证了一阶非对抗原则的实际效果。
- GGR 在标签最稀缺的设置（CIFAR-10 每类 5 标签、ImageNet-30 的 1% 标签）下提升最显著，因为此时伪标签错误率最高，梯度冲突最频繁。这与"Idea 正确则效果最显著"的期望一致。
- 与通用优化控制（PCGrad 对称投影、梯度裁剪、冲突丢弃）的对比实验中，GGR 在所有设置下全面胜出——PCGrad 因对称修改有监督锚点而不稳定，裁剪/丢弃则丢失了有用信号，只有 GGR 的"非对称锚定+仅移除冲突分量"的设计同时保留了监督锚点和辅助信号中的有价值部分。

## 亮点与洞察
- **视角转变是最核心的贡献**：从"样本该不该用"的离散决策切换到"梯度有没有害"的连续校正，绕过了 OSSL 中 ID/OOD 不可分的根本困难。这个思路可以推广到任何"一个目标可能干扰另一个目标"的多任务/多损失场景。
- **半空间投影的最小干预原则**：VLR 只移除精确冲突的分量、保留一切正交信号，数学上是在满足安全约束下对原始梯度改动最小的方案。这种"只删除坏的、不碰好的"的设计哲学比粗暴的梯度裁剪或丢弃高明得多，且推导出闭式解，实现极简。
- **子空间锚点的保守-利用权衡**：OSR（全移除子空间内分量）和 CSR（只裁剪负分量）的设计形成了一个自然的"安全性-利用率"谱系，用户可根据任务的噪声水平选择合适的保守程度。这个设计空间分析本身对其他梯度手术方法也有参考价值。

## 局限与展望
- 理论保证是一阶局部的——仅保证在当前迭代步辅助更新不反对有监督方向，不保证全局最优或复合目标单调下降。当 mini-batch 有监督梯度本身噪声极大或信息量为零时，VLR 锚点可能保守或不稳定（子空间变体部分缓解此问题）。
- GGR 是现有 OSSL 方法的补充而非替代——更强的 OOD 检测/不确定性估计仍然可以进一步提升校正后辅助信号的有效性。GGR 解决的是"冲突梯度"问题，但不解决"伪标签本身就错了导致辅助目标无意义"的问题。
- 校正范围 choice（backbone/head/both）目前需要手动选择，最优范围因基方法和任务而异。一个自适应的范围选择策略（如根据各层冲突频率动态决定）是自然的下一步。
- 作者未在更大规模数据集（如全量 ImageNet-1K）或非分类任务（如 OSSL 目标检测/分割）上验证，方法的泛化边界尚不明确。

## 相关工作与启发
- **vs PCGrad**：PCGrad 对冲突的两个梯度各退一步，在 OSSL 场景下会削弱有监督锚点。GGR 的关键区别是非对称锚定——有监督梯度不可修改，只纠正辅助梯度。附录中证明了当两梯度近乎反向时，PCGrad 的有监督进度趋近于零，而 GGR 保持 $\|g_s\|_2^2$。
- **vs OpenMatch/IOMatch/DAC**：这些方法的共同假设是"可以通过更好的检测/分配策略来区分 ID 和 OOD"。GGR 的立场是"不用区分也可以安全利用所有样本"——这是一种更根本的解决方案，不与任何检测改进矛盾，可以叠加使用。
- **vs 梯度裁剪 (GradClip)**：裁剪均匀缩放所有分量，好坏一起削弱；GGR 只移除冲突方向的分量，保留正交信号。实验证实裁剪效果最差。
- **vs ConfDrop（冲突丢弃）**：检测到 $\langle g_s, g_u \rangle < 0$ 就丢弃整个 $g_u$，等价于对该步降级为纯有监督学习——浪费了无标注数据和正交分量中的有用信号。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从样本选择到梯度控制的视角转换是 OSSL 领域的原创贡献，半空间投影的非对称锚定设计简洁而有效
- 实验充分度: ⭐⭐⭐⭐ 覆盖 CIFAR-10/100 和 ImageNet-30、多种 split 和标签预算、三类基方法、通用对比控制、效率分析，但缺少更大规模数据集的验证
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰（Figure 1 梯度几何分析极具说服力），数学推导严谨（附录给出完整证明），理论-实验对照紧密（冲突诊断、累积遗憾等 metric 设计精巧）
- 价值: ⭐⭐⭐⭐⭐ 即插即用、零超参（VLR）、不改前向传播，实用性强；梯度空间几何控制的思路可迁移到多任务学习、持续学习、鲁棒优化等更广泛的场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](../../AAAI2026/others/sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)
- [\[ECCV 2024\] Bidirectional Uncertainty-Based Active Learning for Open-Set Annotation](../../ECCV2024/others/bidirectional_uncertainty-based_active_learning_for_open-set_annotation.md)
- [\[ICML 2026\] Target-Agnostic Calibration under Distribution Shift with Frequency-Aware Gradient Rectification](../../ICML2026/others/target-agnostic_calibration_under_distribution_shift_with_frequency-aware_gradie.md)
- [\[NeurIPS 2025\] Semi-Supervised Regression with Heteroscedastic Pseudo-Labels](../../NeurIPS2025/others/semi-supervised_regression_with_heteroscedastic_pseudo-labels.md)
- [\[ECCV 2026\] Articulat3D: Reconstructing Articulated Digital Twins From Monocular Videos with Geometric and Motion Constraints](articulat3d_reconstructing_articulated_digital_twins_from_monocular_videos_with_.md)

</div>

<!-- RELATED:END -->
