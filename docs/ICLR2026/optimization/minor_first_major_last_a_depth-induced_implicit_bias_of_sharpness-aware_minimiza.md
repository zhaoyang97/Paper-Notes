---
title: >-
  [论文解读] Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization
description: >-
  [ICLR 2026][优化/理论][Sharpness-Aware Minimization] 深入分析了 SAM 在线性对角网络上训练时的隐式偏差，揭示深度从 $L=1$ 到 $L=2$ 引发的质变：$\ell_\infty$-SAM 的极限方向对初始化高度敏感，$\ell_2$-SAM 则展现出"先弱后强"的**序列特征放大**（sequential feature amplification）现象，指出仅关注 $t\to\infty$ 极限的分析不足以揭示 SAM 的完整动态行为。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "Sharpness-Aware Minimization"
  - "隐式偏差"
  - "线性对角网络"
  - "特征放大"
  - "深度诱导"
---

# Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization

**会议**: ICLR 2026  
**arXiv**: [2603.08290](https://arxiv.org/abs/2603.08290)  
**代码**: 无  
**领域**: 优化理论  
**关键词**: Sharpness-Aware Minimization, 隐式偏差, 线性对角网络, 特征放大, 深度诱导

## 一句话总结

深入分析了 SAM 在线性对角网络上训练时的隐式偏差，揭示深度从 $L=1$ 到 $L=2$ 引发的质变：$\ell_\infty$-SAM 的极限方向对初始化高度敏感，$\ell_2$-SAM 则展现出"先弱后强"的**序列特征放大**（sequential feature amplification）现象，指出仅关注 $t\to\infty$ 极限的分析不足以揭示 SAM 的完整动态行为。

## 研究背景与动机

### 问题背景
Sharpness-Aware Minimization (SAM) 通过最小化邻域内最坏情况损失来寻找平坦最小值，在实践中显著提升泛化能力。先前理论工作主要分析 SAM 在有限极小值设定（如平方损失）下的隐式偏差，而在损失下确界位于无穷处（如逻辑损失）的情形尚未被充分理解。

### 动机
作者考察 SAM 在 $L$ 层线性对角网络上训练线性可分二分类数据（逻辑损失）时的隐式偏差。令人惊讶的发现是：

- **深度 $L=1$（线性模型）**：$\ell_\infty$-SAM 和 $\ell_2$-SAM 都收敛到 $\ell_2$ 最大间隔分类器，与梯度下降（GD）一致
- **深度 $L=2$**：行为发生质变——即使在单样本数据集 $\{(\boldsymbol\mu, +1)\}$，$\boldsymbol\mu=(1,2)$ 上，SAM 的轨迹就可能偏离 GD 的 $\ell_1$ 最大间隔方向

这一观察揭示了仅增加一层就能从根本上改变 SAM 的隐式偏差。

## 方法详解

### 整体框架

论文不提出新算法，而是在 $L$ 层线性对角网络 $f(\mathbf{x}) = \langle \boldsymbol\beta(\boldsymbol\theta), \mathbf{x}\rangle$（其中等效预测器 $\boldsymbol\beta(\boldsymbol\theta) = \bigodot_{\ell=1}^L \mathbf{w}^{(\ell)}$ 是各层权重的逐元素乘积）上，用逻辑损失 $\ell(u)=\log(1+\exp(-u))$ 训练线性可分数据，刻画 $\ell_\infty$-SAM 与 $\ell_2$-SAM 的连续时间隐式偏差。分析以单样本数据集 $\mathcal{D}_{\boldsymbol\mu}=\{(\boldsymbol\mu,+1)\}$ 为最小可解模型，借助 SAM 流的 ODE 与一个简化分析的重标度流，按深度 $L$ 与扰动范数类型分情形给出极限方向与有限时间轨迹。

### 关键设计

**1. 深度 1 的退化结论：确立"SAM 从哪一层开始变特殊"的对照基线**

要论证"深度诱导"，首先得排除单层就有的差异。定理 3.1 证明：对几乎所有线性可分数据集、任意扰动半径 $\rho$ 与任意初始化，$\ell_\infty$-SAM 流的方向都收敛到 $\ell_2$ 最大间隔方向，和梯度下降完全一致；单样本设定下 $\ell_\infty$-SAM 的轨迹甚至与 GD 逐点重合。也就是说，在线性模型（$L=1$）上 SAM 与 GD 的隐式偏差根本无法区分。这把后续所有反直觉现象的来源锁定到深度本身，而非 SAM 的扰动形式——深度才是诱因。

**2. 深度 $L\ge 2$ 的 $\ell_\infty$-SAM：扰动半径化身坐标级门控阈值**

多一层之后，$\ell_\infty$-SAM 为什么能选出次要特征？在单样本数据集上，定理 3.2 给出每个坐标 $\beta_j(t)$ 的演化完全由初始值 $\alpha_j$ 相对 $\rho$ 的大小决定：$\alpha_j<\rho$ 时被压制（$\beta_j(t)\to 0$ 对偶数 $L$、$\to\rho^L$ 对奇数 $L$），$\alpha_j=\rho$ 时停在常数 $\beta_j(t)=\rho^L$，$\alpha_j>\rho$ 时发散并在 $L=2$ 时指数增长。由于各坐标彼此解耦，极限方向只由唯一的胜出坐标 $j^*=\arg\max_{j:\alpha_j>\rho}\mu_j(\alpha_j-\rho)^{L-2}$ 决定（推论 3.5）。这意味着仅靠调初始化，$\ell_\infty$-SAM 就能收敛到**任意一个标准基向量方向**，包括特征权重 $\mu_j$ 很小的次要方向——与始终锁定主要特征的 GD 形成鲜明反差。关键在于 $\rho$ 在此扮演了逐坐标的开关：初始幅度过不了 $\rho$ 这道门的坐标被关掉，过了的才参与竞争。

**3. 深度 2 的 $\ell_2$-SAM：极限与 GD 相同，有限时间却"序列特征放大"**

这是全文最核心、也最反直觉的发现。定理 4.2 先说明 $\ell_2$-SAM 流的极限方向仍是 $\ell_1$ 最大间隔解，与 GD 无异——只看 $t\to\infty$ 会误判成"SAM 毫无区别"。但有限时间动态完全是另一回事：沿时间维度（定理 4.4）预测器 $\boldsymbol\beta(t)$ 早期**依赖次要坐标**，随训练推进才**逐渐转向主要坐标**；沿初始化维度（定理 4.5）增大初始化规模也触发同样的"次要→主要"转变。根源是 $\ell_2$-SAM 的扰动里带了梯度归一化因子：归一化把梯度中较小的坐标相对放大，使对应 $\beta_j$ 在早期获得更高增长率，于是弱特征先被放大；随着训练进行，特征权重 $\mu_j$ 更大的主要坐标凭量级优势后来居上，整个交接是渐进而非瞬时的。这正是标题所说的 "minor first, major last"，也是"极限分析不足以刻画 SAM"的具体反例。

**4. 重标度流：约去公共速度项，让坐标级轨迹可闭式求解**

上面两条定理之所以能精确刻画轨迹，靠的是一个技术工具。单样本数据集上的 SAM 流里始终含有正的损失导数项 $-\ell'(\langle\boldsymbol\beta(\hat{\boldsymbol\theta}(t)),\boldsymbol\mu\rangle)>0$，它对所有坐标是公共的标量，只影响演化快慢、不影响方向。把它约去就得到重标度流，等价于对时间做一次重参数化：空间轨迹一字不差地保留（因此极限方向、有限时间内坐标间的相对次序这些结论都不受影响），但坐标级演化方程变得可闭式分析。它是定理 3.2 与定理 4.4 能给出精确轨迹、而不仅是定性描述的前提。

## 实验关键数据

### 合成实验
- 2D 单样本数据集 $\boldsymbol\mu=(1,2)$，清晰展示了 GD vs $\ell_\infty$-SAM vs $\ell_2$-SAM 的轨迹差异
- 多样本数据集验证了理论预测在实际设定中的适用性
- 深度 $L=3$ 的实验验证了更深网络中 $\ell_\infty$-SAM 对初始化更加敏感

### 实际网络实验（MNIST + CNN）

| 方法 | Grad-CAM 观察 | 说明 |
|------|-------------|------|
| GD | 聚焦于主要数字像素 | 传统行为 |
| $\ell_2$-SAM | 强调背景/弱像素区域 | 与"先次要后主要"理论一致 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| $L=1$, 任意 $\rho$ | 与GD一致 | SAM 不改变偏差 |
| $L=2$, $\ell_\infty$, $\alpha_j < \rho$ | $\beta_j \to 0$ | 坐标被压制 |
| $L=2$, $\ell_\infty$, $\alpha_j > \rho$ | 指数增长 | 可选择次要特征 |
| $L=2$, $\ell_2$ | 先弱后强 | 序列特征放大 |

### 关键发现
- $\ell_\infty$-SAM 在 $L\geq 2$ 时对初始化极度敏感，扰动半径 $\rho$ 充当坐标级别的"门控"阈值
- $\ell_2$-SAM 的有限时间行为与无穷时间极限存在本质差异——极限分析不够
- 离散 SAM 更新的行为与连续时间 SAM 流高度一致，验证了理论的实用性

## 亮点与洞察

1. **"minor first, major last"现象**：揭示了 $\ell_2$-SAM 一个反直觉的有限时间行为——优化器先关注弱特征再关注强特征，这是由梯度归一化导致的
2. **有限时间 vs 无穷时间**：给出了一个明确的例子，说明仅看 $t\to\infty$ 的隐式偏差分析可能遗漏关键动态信息，呼吁更多有限时间分析
3. **深度的质变效应**：仅增加一层（$L=1\to L=2$）就能彻底改变 SAM 的行为，揭示了深度与优化算法之间的深层交互
4. **精确的轨迹刻画**：$\ell_\infty$-SAM 的坐标独立演化性质允许精确的轨迹表征，这是一个优美的理论简化

## 局限与展望

- 理论分析限于**线性对角网络**这一简化模型，与实际深度非线性网络有差距
- 多样本数据集的分析面临额外技术困难，目前仅限于实验验证
- $\ell_2$-SAM 的极限方向定理（定理 4.2）依赖方向收敛假设
- 序列特征放大对实际深度学习中的泛化有何影响仍不清楚
- 未涉及 SAM 变体（如 ASAM、GSAM）的分析

## 相关工作与启发

- **Soudry et al. (2018)**：GD 在线性模型上收敛到 $\ell_2$ 最大间隔方向的经典结果
- **Gunasekar et al. (2018)**：GD 在线性对角网络上偏向 $\ell_1$ 稀疏解
- **Pesme & Flammarion (2023)**：GD 的 saddle-to-saddle 动态——类似的阶段性学习，但机制不同
- **Foret et al. (2020)**：SAM 算法的原始论文
- 启发：有限时间分析对于理解优化器行为至关重要，未来应更多关注训练过程中的动态变化而非仅关注收敛结果

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — "sequential feature amplification"是一个全新且反直觉的发现
- 实验充分度: ⭐⭐⭐⭐ — 合成实验与MNIST/CNN验证，但缺乏大规模实验
- 写作质量: ⭐⭐⭐⭐⭐ — 理论推导严谨，图示清晰，有限时间 vs 无穷时间的对比引人深思
- 价值: ⭐⭐⭐⭐ — 对 SAM 的理论理解有重要推进，但实际应用指导有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Hyperbolic Aware Minimization: Implicit Bias for Sparsity](hyperbolic_aware_minimization_implicit_bias_for_sparsity.md)
- [\[ICLR 2026\] MASAM: Multimodal Adaptive Sharpness-Aware Minimization for Heterogeneous Data Fusion](masam_multimodal_adaptive_sharpness-aware_minimization_for_heterogeneous_data_fu.md)
- [\[ICLR 2026\] Towards Understanding the Calibration Benefits of Sharpness-Aware Minimization](towards_understanding_the_calibration_benefits_of_sharpness-aware_minimization.md)
- [\[ICLR 2026\] Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models](bi-lora_efficient_sharpness-aware_minimization_for_fine-tuning_large-scale_model.md)
- [\[ICLR 2026\] Implicit Bias of Per-sample Adam on Separable Data: Departure from the Full-batch Regime](implicit_bias_of_per-sample_adam_on_separable_data_departure_from_the_full-batch.md)

</div>

<!-- RELATED:END -->
