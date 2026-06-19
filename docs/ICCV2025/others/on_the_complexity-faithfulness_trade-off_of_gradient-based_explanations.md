---
title: >-
  [论文解读] On the Complexity-Faithfulness Trade-off of Gradient-Based Explanations
description: >-
  [ICCV 2025][可解释性] 提出统一的频谱框架来系统性分析和量化梯度解释的平滑性（复杂度）与忠实度之间的权衡，引入期望频率（EF）度量网络对高频信息的依赖程度，并通过将 ReLU 与高斯函数卷积来控制解释复杂度，同时定义"解释间隙"来量化替代模型导致的忠实度损失。 问题定义 梯度解释方法（如 VanillaGrad）…
tags:
  - "ICCV 2025"
  - "可解释性"
  - "梯度解释"
  - "频谱分析"
  - "忠实度-复杂度权衡"
  - "ReLU网络"
---

# On the Complexity-Faithfulness Trade-off of Gradient-Based Explanations

**会议**: ICCV 2025  
**arXiv**: [2508.10490](https://arxiv.org/abs/2508.10490)  
**代码**: [github.com/Amir-Mehrpanah/On-the-Complexity-Faithfulness-Trade-off-of-Gradient-Based-Explanations-ICCV25](https://github.com/Amir-Mehrpanah/On-the-Complexity-Faithfulness-Trade-off-of-Gradient-Based-Explanations-ICCV25/)  
**领域**: 其他  
**关键词**: 可解释性, 梯度解释, 频谱分析, 忠实度-复杂度权衡, ReLU网络

## 一句话总结

提出统一的频谱框架来系统性分析和量化梯度解释的平滑性（复杂度）与忠实度之间的权衡，引入期望频率（EF）度量网络对高频信息的依赖程度，并通过将 ReLU 与高斯函数卷积来控制解释复杂度，同时定义"解释间隙"来量化替代模型导致的忠实度损失。

## 研究背景与动机

### 问题定义

梯度解释方法（如 VanillaGrad）是计算机视觉中常用的可解释性工具，但存在两个核心矛盾：

**解释复杂度**：ReLU 网络具有尖锐的过渡特性，有时依赖单个像素进行预测，导致梯度解释呈现"噪声化"的外观，难以被人类理解

**解释忠实度**：为降低复杂度，后处理方法（如 GradCAM、SmoothGrad）通过创建代理模型来平滑解释，但代价是偏离原始模型的真实行为

### 已有方法的不足

**度量割裂**：现有指标如熵方法、像素移除分数，各自只衡量复杂度或忠实度中的一个维度，无法统一分析两者的权衡

**外在因素干扰**：像素移除分数受基线选择和移除顺序等外在因素影响，阻碍了对权衡的原理性分析

**代理模型隐式性**：现有解释方法通过试错来设计平滑策略，产生的代理模型通常是隐式且不可访问的，难以直接衡量解释间隙

**缺乏架构级理解**：现有工作未建立网络架构（特别是激活函数选择）与解释复杂度之间的形式化联系

### 核心动机

**关键洞察**：VanillaGrad 解释中的"噪声"并非真正的噪声，而是来自网络架构的结构性属性——特别是 ReLU 引入的尖锐过渡。如果能在频域中建立网络功率谱尾部与梯度空间功率谱尾部之间的联系，就能同时理解和控制解释的复杂度，并量化后处理方法引入的忠实度损失。

## 方法详解

### 整体框架

方法基于频谱视角构建统一分析框架，包含三个核心部分：
1. 通过空间功率谱尾部（TSPS）度量解释复杂度
2. 建立网络功率谱尾部（TPS）与梯度 TSPS 的形式化联系
3. 在傅里叶域中定义解释间隙来量化忠实度

### 关键设计

#### 1. **期望频率（Expected Frequency, EF）**：度量解释复杂度

- **功能**：通过空间功率谱的加权积分来量化解释中高频成分的占比
- **核心思路**：定义期望频率为：

$$\operatorname{EF}(e_f(x)) \coloneq \int \omega \operatorname{S}_{e_f(x)}(\omega) \, d\omega$$

其中 $\operatorname{S}$ 是解释方法 $e_f$ 的空间功率谱。EF 值越低，解释在空间域中越平滑、越简单。对图像数据使用频域径向平均得到一维功率谱。

- **设计动机**：解释越"噪声化"，空间功率谱的尾部就越重。EF 提供了一个简洁且有效的统计量来捕捉尾部行为，同时受模型和解释方法的共同影响。

#### 2. **网络 TPS 与梯度 TSPS 的形式化联系**

- **功能**：建立网络功率谱尾部与输入梯度空间功率谱尾部之间的理论关系
- **核心思路**：

**定理 1（非形式化）**：在具有高输入特征相关性的数据域（如图像数据）中，给定训练好的神经网络 $f(x)$，$f(x)$ 的功率谱的尾部行为与 $\nabla f(x)$ 的空间功率谱的尾部行为成正比。

关键推论：通过 **引理 1**，将 ReLU 与高斯函数卷积得到平滑参数化（SP）：

$$\xi = \phi * g_\beta$$

其中 $\beta$ 是高斯精度参数，$\beta \to \infty$ 时恢复标准 ReLU。实际实现使用 SoftPlus 作为高效近似：

$$\text{SoftPlus}(x;\beta) = \frac{1}{\beta} \ln(1 + e^{\beta x}) \approx \text{ReLU} * g_\beta(x)$$

- **设计动机**：功率谱尾部越重意味着网络对高频信息依赖越强，导致梯度解释越复杂。通过控制激活函数的平滑度就能直接控制解释复杂度，同时保持零解释间隙。

#### 3. **解释间隙（Explanation Gap）**：量化忠实度

- **功能**：衡量后处理解释方法在引入代理模型后偏离原始模型的程度
- **核心思路**：定义解释间隙为原始模型与代理模型梯度差的 $L^2$ 范数：

$$\mathcal{G}(f, \tilde{f}) = \int_{x \in \mathcal{X}} \|\nabla f(x) - \nabla \tilde{f}(x)\|_2^2 \, dx$$

利用 Parseval 定理将其转换到傅里叶域：

$$\mathcal{G}(f, \tilde{f}) \approx \int_{\omega \in \mathcal{F}_{\text{high}}} \omega^2 \|\hat{f}(\omega) - \hat{\tilde{f}}(\omega)\|^2 \, d\omega$$

由于代理模型抑制高频成分，间隙主要由高频部分决定。最终使用 EF 变化量作为代理度量：

$$\Delta \operatorname{EF}(e_f) \coloneq |\operatorname{EF}(\nabla f) - \operatorname{EF}(e_f)|$$

- **设计动机**：后处理方法本质上是低通滤波器，抑制高频以换取视觉平滑性。解释间隙量化了这种"工程化"程度，VanillaGrad 的间隙为零（因为不创建代理模型），而 GradCAM 等方法间隙最大。

### 损失函数 / 训练策略

本文不涉及新的损失函数设计。核心技术贡献在于：
- 使用 SoftPlus($\beta$) 替换 ReLU 训练网络，通过调节 $\beta$ 控制解释复杂度
- 设置验证精度上限进行早停，确保不同平滑参数下训练预算可比
- 使用逆变换方法对每个像素进行梯度幅值排名归一化

## 实验关键数据

### 主实验

**Imagenette-CNN 上不同解释方法的 EF 与解释间隙**：

| 方法 | ReLU: EF + ΔEF | SP(β=0.9): EF + ΔEF |
|------|----------------|---------------------|
| VanillaGrad | .390 + Δ.000 | .202 + Δ.000 |
| SmoothGrad | .286 + Δ.104 | .196 + Δ.005 |
| IntGrad | .396 + Δ.007 | .205 + Δ.003 |
| GuidedBP | .300 + Δ.090 | .202 + Δ.000 |
| DeepLift | .394 + Δ.005 | .204 + Δ.002 |
| GradCAM | .293 + Δ.097 | .177 + Δ.025 |

**ImageNet 上的 EF 与解释间隙（×10⁴）**：

| 方法 | ResNet50: EF + ΔEF | ViT-B16: EF + ΔEF |
|------|--------------------|--------------------|
| VanillaGrad | .263 + Δ.000 | .222 + Δ.000 |
| SmoothGrad | .247 + Δ.017 | .221 + Δ.001 |
| GradCAM | .133 + Δ.130 | .181 + Δ.041 |

### 消融实验

| 配置 | 观察结果 | 说明 |
|------|---------|------|
| β 增大（→ReLU） | EF 单调增大 | 证实 ReLU 导致更重的功率谱尾部 |
| SP(β=0.9) + VG | EF=.202, ΔEF=0 | 可在零间隙下获得低复杂度 |
| ReLU + SmoothGrad | EF=.286, ΔEF=.104 | 后处理降低复杂度但引入大间隙 |
| ViT + GELU | 低 EF 和低变异性 | ViT 架构比激活函数更重要 |
| 网络深度变化 | 频谱衰减率几乎不变 | 深度对解释复杂度影响有限 |
| 学习率变化 | 曲线形状有变但尾部行为不变 | 学习率影响细节但不影响整体趋势 |

### 关键发现

1. **ReLU 是解释复杂度的根源**：$\beta$ 增大时 EF 单调增加，证实 ReLU 引入的尖锐过渡是梯度解释"噪声化"的根本原因
2. **GradCAM 的间隙最大**：在所有架构上 GradCAM 始终引入最大的解释间隙，这是需要特别关注的忠实度风险
3. **SP 提供了更好的权衡**：通过 SP(β=0.9) 可以在零解释间隙下将 EF 从 .390 降至 .202，而 SmoothGrad 只能降至 .286 且间隙为 .104
4. **ViT 天然更平滑**：ViT 使用 GELU 激活且注意力机制提供全局感受野，其后处理方法变异性更低

## 亮点与洞察

1. **统一框架**：首次在统一的频谱框架下同时量化解释复杂度和忠实度，两个指标使用一致定义
2. **根因分析与控制**：不仅揭示 ReLU 是解释复杂度的根源，还提供了通过平滑参数化控制复杂度的实用方案
3. **解释间隙概念**：形式化定义了"解释间隙"，为评估后处理方法的隐性忠实度风险提供了工具
4. **无超参数度量**：EF 和 ΔEF 不依赖基线选择、移除顺序等外在因素，具有更清晰的直觉
5. **跨架构验证**：在 CNN、ResNet、ViT 等不同架构上验证了理论预测的一致性

## 局限与展望

1. **理论依赖核方法**：与核方法的联系在深度网络中可能失效，特别是在深度维度上核视角可能不直觉
2. **仅关注空间频率**：频谱分析只在空间域进行，可能忽略其他维度的信息
3. **精度-复杂度权衡**：平滑参数化 ReLU 可能牺牲分类精度，尽管论文设置了验证精度上限来缓解
4. **仅适用于分类器**：分析框架基于标量分类器 $f: \mathbb{R}^n \to \mathbb{R}$，对其他任务的推广未讨论
5. **理论适用范围**：定理 1 要求输入特征具有高相关性，在低相关性数据上可能不成立

## 相关工作与启发

- 与 SmoothGrad、GradCAM 等方法的区别：本文不是新的解释方法，而是分析框架——用于评估现有方法的复杂度-忠实度权衡
- 与 B-cos 网络等固有可解释性方法的关系：SP 可看作另一种"先天"可解释性策略，直接降低网络的高频依赖
- 频谱视角的独特性：不同于以往用优化来强加频谱属性的方法，本文仅用频谱进行测量和分析

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 频谱视角下的统一分析框架很有创意，EF 和解释间隙的定义简洁有力
- **实验充分度**: ⭐⭐⭐⭐ — 跨数据集、跨架构验证充分，消融实验覆盖深度、学习率等多种因素
- **写作质量**: ⭐⭐⭐⭐⭐ — 数学推导严谨，叙述清晰，理论与实验紧密结合
- **价值**: ⭐⭐⭐⭐ — 为可解释性社区提供了重要的分析工具，但实际应用场景有待进一步拓展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Forest vs Tree: The (N, K) Trade-off in Reproducible ML Evaluation](../../AAAI2026/others/forest_vs_tree_the_n_k_trade-off_in_reproducible_ml_evaluation.md)
- [\[NeurIPS 2025\] The Computational Complexity of Counting Linear Regions in ReLU Neural Networks](../../NeurIPS2025/others/the_computational_complexity_of_counting_linear_regions_in_relu_neural_networks.md)
- [\[CVPR 2026\] More Than Meets the Eye: A Unified Image Fusion Framework via Semantic-Pixel Entropy Trade-off for Zero-Shot Generalization](../../CVPR2026/others/more_than_meets_the_eye_a_unified_image_fusion_framework_via_semantic-pixel_entr.md)
- [\[NeurIPS 2025\] The Cost of Robustness: Tighter Bounds on Parameter Complexity for Robust Memorization in ReLU Nets](../../NeurIPS2025/others/the_cost_of_robustness_tighter_bounds_on_parameter_complexity_for_robust_memoriz.md)
- [\[ICML 2026\] Guaranteed Optimal Compositional Explanations for Neurons](../../ICML2026/others/guaranteed_optimal_compositional_explanations_for_neurons.md)

</div>

<!-- RELATED:END -->
