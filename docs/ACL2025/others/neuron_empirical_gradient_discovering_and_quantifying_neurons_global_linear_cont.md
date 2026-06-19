---
title: >-
  [论文解读] Neuron Empirical Gradient: Discovering and Quantifying Neurons' Global Linear Controllability
description: >-
  [ACL 2025][神经元可解释性] 揭示了预训练语言模型 FF 层神经元激活值与模型输出之间存在全局线性关系，提出了神经元经验梯度（NEG）来量化这种线性关系，并设计了高效估算方法 NeurGrad，最终通过技能神经元探测实验证明 NEG 能有效表征多种语言技能。 预训练语言模型（PLM）中 Feed-Forward (…
tags:
  - "ACL 2025"
  - "神经元可解释性"
  - "神经元梯度"
  - "知识归因"
  - "技能神经元"
  - "语言模型"
---

# Neuron Empirical Gradient: Discovering and Quantifying Neurons' Global Linear Controllability

**会议**: ACL 2025  
**arXiv**: [2412.18053](https://arxiv.org/abs/2412.18053)  
**代码**: [有 (GitHub)](https://github.com/xzhao-tkl/NEG)  
**领域**: 其他  
**关键词**: 神经元可解释性, 神经元梯度, 知识归因, 技能神经元, 语言模型

## 一句话总结

揭示了预训练语言模型 FF 层神经元激活值与模型输出之间存在全局线性关系，提出了神经元经验梯度（NEG）来量化这种线性关系，并设计了高效估算方法 NeurGrad，最终通过技能神经元探测实验证明 NEG 能有效表征多种语言技能。

## 研究背景与动机

预训练语言模型（PLM）中 Feed-Forward (FF) 层的神经元能够编码知识，这已被多项研究证实。但现有研究存在两个主要问题：

**只能排序，无法量化**：现有方法（如知识神经元发现方法）主要是按重要性对神经元排名，但无法量化神经元激活值变化与模型输出变化之间的精确关系。这限制了知识编辑等应用场景——如果你不知道修改一个神经元会对输出产生多大影响，就无法精确控制模型行为。

**计算成本高昂**：现有方法需要反复修改激活值进行推理、或进行大量张量运算，导致无法对所有神经元进行大规模分析，尤其是在 Llama2-70B 这种大模型上。

作者提出了一个自然的问题：**神经元激活值的变化与模型输出变化之间到底是什么关系？** 如果这种关系是可量化的，就能打开精确控制 PLM 输出的大门。

## 方法详解

### 整体框架

论文分三步推进：

1. 通过神经元干预实验发现**激活偏移与输出偏移之间的全局线性关系**
2. 提出 **NeurGrad** 高效估算神经元经验梯度
3. 通过**技能神经元探测**验证 NEG 能表征多种语言技能

### 关键设计

#### 1. **神经元线性关系的发现（NEG）**

核心做法：在 [-10, 10] 范围内以 0.2 为步长修改特定神经元的激活值，观察目标 token 概率的变化。在 7 个 PLM 上（包括 BERTbase、BERTlarge 和多个 Llama/Qwen 模型）进行实验。

关键发现：
- 在 ±2 范围内，激活偏移与输出偏移的 Pearson 相关系数 $r$ 普遍超过 0.95
- 超过 90% 的神经元表现出线性行为
- 正极性和负极性的神经元数量大致相等（约各 50%）

基于此，定义 **NEG**（神经元经验梯度）为激活偏移-输出偏移线性回归的斜率。

#### 2. **NeurGrad：高效 NEG 估算**

直接计算 NEG 需要对每个神经元进行约 100 次推理（不同偏移值），计算成本高昂。作者发现：
- 计算梯度（CG，通过反向传播获得）的**绝对值**与 NEG 高度相关（$r = 0.961$），但**符号**不可靠
- 神经元激活值的符号可以修正 CG 的方向

由此提出 NeurGrad：

$$\bar{G_E} = CG \times \text{sign}(A)$$

其中 $CG$ 是计算图梯度，$A$ 是神经元激活值。运行时间仅为 IG（积分梯度）的 1/120。

#### 3. **多神经元控制**

实验验证了 NEG 在多神经元同时干预时是否仍然有效：
- 同时干预 $2^{12}$ 个神经元时，预测偏移与实际偏移的相关性仍 ≥ 0.7
- 但随着干预神经元数量增加或偏移量增大，线性度逐渐下降

提出**局部线性近似假说**解释这一现象：类似于一阶泰勒展开，小范围内的局部可微性保证了线性，但范围扩大后非线性效应增强。

#### 4. **MCEval8K 基准与技能神经元探测**

构建了 MCEval8K 基准，覆盖 6 大类 22 个语言理解任务（语言学、内容分类、NLI、事实性、自省、多语言），每个任务上限 8K 样例。

设计三种探测器：
- **Polar-Probe**：基于极性的多数投票分类器
- **Magn-Probe**：基于 NEG 幅值的多数投票分类器
- **Tree-Probe**：使用随机森林建模神经元间依赖关系

### 损失函数 / 训练策略

NEG 的计算使用零截距线性回归拟合；NeurGrad 仅需一次前向传播和一次反向传播，无需额外训练。技能神经元探测器中 Tree-Probe 使用 scikit-learn 的随机森林默认设置（100棵树，无深度限制）。

## 实验关键数据

### NeurGrad 与基线方法的 NEG 估算对比

| 方法 | 相关性 r (BERT-base) | 相关性 r (Llama2-7B) | MAE (BERT-base) | 运行时间 |
|------|---------------------|---------------------|------------------|---------|
| CG | -0.891 | 0.302 | 6.1e-03 | 0.149s |
| IG | 0.736 | 0.538 | 3.0e-03 | 19.349s |
| LPI | - | 0.647 | - | 6.086s |
| **NeurGrad** | **0.9998** | **0.814** | **2.6e-05** | **0.161s** |

### 技能神经元探测（Llama2-7B）

| 任务 | LM-Prob | Act（激活） | Magn（梯度） | Tree-Probe |
|------|---------|------------|------------|------------|
| NER | 0.361 | 0.453 | 0.498 | 0.740 |
| Agnews | 0.588 | 0.849 | 0.702 | 0.872 |
| PAWS | 0.524 | 0.825 | 0.815 | 0.888 |
| CSQA | 0.610 | 0.613 | 0.639 | 0.773 |
| HaluEval | 0.520 | 0.788 | 0.783 | 0.818 |
| mLAMA | 0.608 | 0.622 | 0.637 | 0.724 |

### 关键发现

1. **NEG 在 BERT 上几乎完美估算**：NeurGrad 在 BERTbase 上的相关性达到 0.9998，MAE 仅 2.6e-05
2. **梯度 vs 激活各有优势**：NEG 在知识密集型任务（mLAMA, CSQA）上优于激活方法，可能因为复杂知识在预训练中未充分学习
3. **Tree-Probe 大幅超越多数投票探测器**：说明神经元之间的依赖关系对表征语言技能至关重要
4. **技能神经元高度高效**：大多数任务仅需 256 个神经元即可达到最优精度
5. **技能神经元具有鲁棒性和可替代性**：对不同 prompt 模板鲁棒，且使用不同神经元子集都能达到较好性能
6. **不同任务的神经元依赖模式不同**：PAWS 偏好深层树，CSQA 偏好多棵树，HaluEval 需要平衡

## 亮点与洞察

- **从定性到定量的飞跃**：此前对 FF 层神经元的理解多为"哪些神经元重要"，本文首次给出了"神经元重要多少（精确的梯度值）"的定量回答
- **极简而有效的方法**：NeurGrad 公式只有一行（$CG \times \text{sign}(A)$），但效果远超复杂的积分梯度和因果追踪方法
- **90%+ 神经元都是线性的**：这个发现如果能被进一步利用（如精确知识编辑），可能对模型可解释性和可控性产生深远影响
- **正/负极性神经元各占一半**：这意味着简单地增加或减少激活值是不够的，必须考虑极性方向

## 局限与展望

1. 当前分析局限于单 token 事实性 prompt，未扩展到多 token 生成场景
2. 如何利用 NEG 进行实际的语言技能级别输出调整（如知识编辑、偏见消除）尚未探索
3. 线性关系在大偏移范围下减弱，限制了实际神经元修改的幅度
4. MCEval8K 基准虽然覆盖面广，但每个任务内的难度分布可能不均匀
5. 与 Sparse Autoencoders 等线路的可解释性方法的对比缺失

## 相关工作与启发

- Knowledge Neurons (Dai et al., 2022) 开创了 FF 层神经元知识归因的研究方向
- ROME/MEMIT (Meng et al., 2022) 提供了基于因果追踪的知识编辑方法
- Skill Neurons (Wang et al., 2022) 首次提出用激活值进行技能神经元探测
- 本文将这些方向统一到一个 NEG 框架下，提供了更精确的量化工具

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次揭示并量化了神经元-输出的全局线性关系，NeurGrad 方法极简且高效
- **实验充分度**: ⭐⭐⭐⭐⭐ — 7 个 PLM、22 个任务、多种探测器，覆盖全面；MCEval8K 基准有长期使用价值
- **写作质量**: ⭐⭐⭐⭐ — 逻辑清晰，层层递进，但公式符号较多需要仔细阅读
- **价值**: ⭐⭐⭐⭐⭐ — 对模型可解释性、知识编辑和偏见消除等方向有重要的基础性贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Quantifying Lexical Semantic Shift via Unbalanced Optimal Transport](quantifying_lexical_semantic_shift_via_unbalanced_optimal_transport.md)
- [\[ICML 2025\] Probably Approximately Global Robustness Certification](../../ICML2025/others/probably_approximately_global_robustness_certification.md)
- [\[ICML 2026\] Guaranteed Optimal Compositional Explanations for Neurons](../../ICML2026/others/guaranteed_optimal_compositional_explanations_for_neurons.md)
- [\[NeurIPS 2025\] Statistical Inference for Gradient Boosting Regression](../../NeurIPS2025/others/statistical_inference_for_gradient_boosting_regression.md)
- [\[ICML 2025\] Gradient Aligned Regression via Pairwise Losses](../../ICML2025/others/gradient_aligned_regression_via_pairwise_losses.md)

</div>

<!-- RELATED:END -->
