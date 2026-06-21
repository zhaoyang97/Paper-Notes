---
title: >-
  [论文解读] Reducing Class-Wise Performance Disparity via Margin Regularization
description: >-
  [3D视觉] 提出 MR2（Margin Regularization for performance disparity Reduction），通过在 logit 和表征空间动态调整类别相关的 margin，基于理论推导的泛化界减少类间性能差异，同时提升整体准确率。 - 深度网络即使在类别平衡数据上训练…
tags:
  - "3D视觉"
---

# Reducing Class-Wise Performance Disparity via Margin Regularization

## 元信息
- **会议**: ICLR 2026
- **arXiv**: [2602.00205](https://arxiv.org/abs/2602.00205)
- **代码**: [https://github.com/BeierZhu/MR2](https://github.com/BeierZhu/MR2)
- **领域**: LLM预训练
- **关键词**: class-wise disparity, margin regularization, generalization bound, Rademacher complexity, representation learning

## 一句话总结
提出 MR2（Margin Regularization for performance disparity Reduction），通过在 logit 和表征空间动态调整类别相关的 margin，基于理论推导的泛化界减少类间性能差异，同时提升整体准确率。

## 研究背景与动机
- 深度网络即使在类别平衡数据上训练，也存在严重的类间准确率差异。例如 ResNet-50 在 ImageNet 上最好类 100%、最差类仅 16%。
- 先前工作发现"难"类（准确率低）具有更大的特征变异性（图 1b），但解决方案主要是经验性的（数据增强、表征学习），缺乏理论基础。
- 现有 margin 方法（LDAM、Logit Adjustment 等）为不平衡分类设计，在类别均衡时退化为标准交叉熵，无法解决性能差异问题。

## 方法详解

### 整体框架
MR2 的出发点是一个类敏感的泛化界：它把每个类的泛化风险拆成"经验风险"加上一个与类内特征变异性成正比、与该类 margin 平方成反比的复杂度项。这意味着只要给特征更分散的"难"类分配更大的 margin、同时压缩它们的类内变异，就能在不牺牲"易"类的前提下抹平类间差异。围绕这个思路，方法在 logit 空间和表征空间各放一个正则项，联合优化一个标准交叉熵改造而来的目标。

### 关键设计

**1. 类敏感泛化界：把"难类"和"margin"挂上钩的理论地基**

整个方法的出发点是命题 1 给出的一条类敏感泛化上界：

$$\mathcal{R}(f) \leq \frac{1}{\ln 2} \hat{\mathcal{R}}_{\mathcal{D}}^{\bm{\gamma},\mathsf{ce}}(f) + \frac{4\sqrt{2}\Lambda K}{\sqrt{N}} \sqrt{\sum_{k=1}^K \frac{\|\hat{\bm{\mu}}_k\|_2^2 + \|\hat{\mathbf{s}}_k\|_2^2}{\gamma_k^2}} + \mathcal{O}(1/\sqrt{N})$$

它把泛化风险拆成"经验风险"加一个复杂度项，而复杂度项里每个类贡献 $(\|\hat{\bm{\mu}}_k\|_2^2 + \|\hat{\mathbf{s}}_k\|_2^2)/\gamma_k^2$：分子是第 $k$ 类的特征能量与变异性，分母是它的 margin $\gamma_k$。这条界恰好对应图 1(b) 的经验观察——准确率低的"难"类特征更分散（分子更大），泛化自然更差。要压低整体界只有两条路：减小分子，或按分子比例放大 $\gamma_k$。后面两个正则项就是分别走这两条路，且因为优化的是同一条全局上界，提升难类时不会反噬易类，没有去偏方法常见的零和权衡。

**2. Logit margin 损失：按特征分散度给每个类分配 margin 预算**

第一条路是放大难类的 margin。MR2 把标准交叉熵对所有类一视同仁的做法，改成给每个类配一个专属温度 $\gamma_y$ 当 margin：

$$\ell_{\bm{\gamma},\mathsf{ce}}(f, \mathbf{x}, y) = -\mathbf{1}_y^\top \ln[\text{softmax}(\mathbf{z} / \gamma_y)]$$

这个 $\gamma_y$ 不是手调超参，而是在"平均预算固定（$\bar{\gamma}=\bar{c}$）"约束下最小化上界复杂度项、用拉格朗日求解直接得到的闭式解（推论 1）：

$$\gamma_y = \frac{\bar{c} \cdot K (\|\hat{\bm{\mu}}_y\|_2^2 + \|\hat{\mathbf{s}}_y\|_2^2)^{1/3}}{\sum_{k=1}^K (\|\hat{\bm{\mu}}_k\|_2^2 + \|\hat{\mathbf{s}}_k\|_2^2)^{1/3}}$$

其中 $\hat{\bm{\mu}}_k$ 是第 $k$ 类特征均值、$\|\hat{\mathbf{s}}_k\|_2^2$ 是其均方偏差，二者训练中用 EMA 在线维护，$\bar{c}$ 控制整体 margin 强度；当所有 $\gamma_k=1$ 时它就退化回标准交叉熵。$1/3$ 次方分配律让特征越分散的难类拿到越大的 $\gamma_y$，等于把更多泛化预算倾斜给它们，从源头缩小类间差距，而不是像重加权那样靠牺牲易类来补难类。

**3. 表征 margin 损失：直接压缩难类的类内变异，减少需要分配的预算**

第二条路是减小分子。光放大 margin 治标不治本，难类特征本身分散才是病根。这一项以全局平均均方偏差对应的 $2\bar{s}$ 为 margin（$\bar{s}=\frac{1}{K}\sum_k\|\hat{\mathbf{s}}_k\|_2^2$），对同类样本对施加

$$\ell_{\bar{s}}(f, \mathbf{x}, y) = \ln\Big[1 + \sum_{\mathbf{x}^+ \in \mathcal{D}_y \setminus \{\mathbf{x}\}} \exp(\|\phi(\mathbf{x}) - \phi(\mathbf{x}^+)\|_2^2 - 2\bar{s})\Big]$$

当某类的类内距离超过这个 margin 时受罚，从而把它拉紧，等价于最小化类内均方偏差。它和 logit 项互补：logit 项分配预算、表征项直接降低需要分配的预算量，两路都在压同一条界。两项以系数 $\lambda$ 加权得到最终训练目标

$$\min_{f} \frac{1}{N} \sum_{\mathbf{x},y \in \mathcal{D}} \big[\ell_{\bm{\gamma},\mathsf{ce}}(f, \mathbf{x}, y) + \lambda \cdot \ell_{\bar{s}}(f, \mathbf{x}, y)\big]$$

## 实验关键数据

### 主实验：CIFAR-100 & ImageNet

| 方法 | 整体准确率 | Easy | Medium | Hard |
|------|----------|------|--------|------|
| ERM (标准训练) | 70.9 | 84.5 | 71.0 | 56.7 |
| LfF | 69.1 | 83.6 (-0.9) | 70.1 (-0.9) | 53.7 (-3.0) |
| JTT | 70.6 | 84.3 (-0.2) | 70.8 (-0.2) | 56.2 (-0.5) |
| DRO | ~70.0 | 降低 | ~71.0 | ~56.0 |
| **MR2 (Ours)** | **71.8** | **85.0 (+0.5)** | **72.0 (+1.0)** | **58.5 (+1.8)** |

> MR2 显著提升"难"类性能（+1.8），同时"易"类也有提升（+0.5），无需权衡。

### 消融实验：预训练骨干 + 微调方式

| 骨干 + 方式 | ERM | MR2 | Hard 提升 |
|-----------|-----|-----|---------|
| MAE (end-to-end) | 基线 | +提升 | 显著 |
| MoCov2 (linear probe) | 基线 | +提升 | 显著 |
| CLIP (linear probe) | 基线 | +提升 | 显著 |
| ResNet-50 (from scratch) | 70.9 | **71.8** | +1.8 |
| ViT-B/16 (from scratch) | 基线 | +提升 | 显著 |

> MR2 在所有预训练方法（MAE/MoCov2/CLIP）和训练范式（端到端/线性探针）上均适用。

### 关键发现
1. 现有去偏方法（LfF、JTT、DRO）在改善"难"类时通常牺牲"易"类——MR2 没有此权衡
2. Logit margin 和表征 margin 互补：前者分配更大泛化预算给"难"类，后者减少类内变异
3. 理论推导的 $\gamma_k$ 在实践中与通过扫描选择的最优值高度一致
4. 即使在 L2 归一化的 CLIP 特征上，使用 $L_p (p \neq 2)$ 范数仍可恢复类敏感 margin

## 亮点与洞察
- **理论驱动的方法**：从泛化界出发推导 margin 设计，而非经验性调参
- **无权衡改进**：同时提升难类和易类，这在公平性/去偏方法中极为罕见
- **广泛适用性**：跨 7 个数据集、CNN/ViT 架构、多种预训练范式均有一致提升
- **不与长尾方法冲突**：在类别平衡场景下仍有意义，填补了均衡数据中性能差异的理论空白

## 局限性
- EMA 维护类统计量增加少量计算开销
- 表征 margin loss 需要同类样本配对，对极少样本的类可能不够稳定
- 理论分析假设分类器权重范数均匀有界（$\Lambda$），可能在某些模型中不完全成立
- 超参 $\bar{c}$ 和 $\lambda$ 仍需调优

## 相关工作
- **长尾分类 margin**: LDAM (Cao et al., 2019), Logit Adjustment (Menon et al., 2021), Balanced Softmax (Ren et al., 2020)
- **性能差异研究**: Cui et al. (2024) 发现差异源于表征而非分类器偏差
- **Neural Collapse**: Papyan et al. (2020) 的理想化假设在大数据集上不成立
- **对比学习**: SupCon (Khosla et al., 2020) 不含 margin 约束

## 评分
- 新颖性: ⭐⭐⭐⭐ — 类别平衡数据下的 margin 正则化，理论推导与经验洞察统一
- 理论深度: ⭐⭐⭐⭐⭐ — 完整的 Rademacher 复杂度分析和泛化界
- 实验充分性: ⭐⭐⭐⭐⭐ — 7 数据集、多架构、多预训练、详细消融
- 实用价值: ⭐⭐⭐⭐ — 即插即用，开源实现，适用于各种分类模型

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Universal Beta Splatting](universal_beta_splatting.md)
- [\[ICLR 2026\] Topology-Preserved Auto-regressive Mesh Generation in the Manner of Weaving Silk](topology-preserved_auto-regressive_mesh_generation_in_the_manner_of_weaving_silk.md)
- [\[ICLR 2026\] UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images](ufo-4d_unposed_feedforward_4d_reconstruction_from_two_images.md)
- [\[CVPR 2026\] TeHOR: Text-Guided 3D Human and Object Reconstruction with Textures](../../CVPR2026/3d_vision/tehor_text-guided_3d_human_and_object_reconstruction_with_textures.md)
- [\[CVPR 2026\] No Calibration, No Depth, No Problem: Cross-Sensor View Synthesis with 3D Consistency](../../CVPR2026/3d_vision/no_calibration_no_depth_no_problem_cross-sensor_view_synthesis_with_3d_consisten.md)

</div>

<!-- RELATED:END -->
