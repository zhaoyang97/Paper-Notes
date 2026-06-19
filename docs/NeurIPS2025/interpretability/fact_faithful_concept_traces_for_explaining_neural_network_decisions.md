---
title: >-
  [论文解读] FaCT: Faithful Concept Traces for Explaining Neural Network Decisions
description: >-
  [NeurIPS 2025][可解释性][概念解释] 提出 FaCT，一种结合 B-cos 变换和稀疏自编码器 (SAE) 的内在可解释模型，能够**忠实地**将模型预测分解为概念贡献（Logit = $\sum$ 概念贡献），并将每个概念忠实地可视化到输入像素级别（概念激活 = $\sum$ 像素贡献），同时提出基于 DINOv2 的 C²-score 用于评估概念一致性。
tags:
  - "NeurIPS 2025"
  - "可解释性"
  - "概念解释"
  - "忠实归因"
  - "B-cos 网络"
  - "稀疏自编码器"
  - "可解释模型"
---

# FaCT: Faithful Concept Traces for Explaining Neural Network Decisions

**会议**: NeurIPS 2025  
**arXiv**: [2510.25512](https://arxiv.org/abs/2510.25512)  
**代码**: [https://github.com/m-parchami/FaCT](https://github.com/m-parchami/FaCT)  
**领域**: 可解释性 / 概念解释  
**关键词**: 概念解释, 忠实归因, B-cos 网络, 稀疏自编码器, 可解释模型

## 一句话总结
提出 FaCT，一种结合 B-cos 变换和稀疏自编码器 (SAE) 的内在可解释模型，能够**忠实地**将模型预测分解为概念贡献（Logit = $\sum$ 概念贡献），并将每个概念忠实地可视化到输入像素级别（概念激活 = $\sum$ 像素贡献），同时提出基于 DINOv2 的 C²-score 用于评估概念一致性。

## 研究背景与动机

**领域现状**：深度网络在各种任务上表现优异，但理解其内部工作机制仍然困难。归因方法（如 Grad-CAM）只能显示输入中哪些像素重要，无法解释模型使用了哪些**高层概念**。

**现有概念解释方法**：
   - **Part-Prototype 网络**和**概念瓶颈模型 (CBM)**：尝试创建内在可解释模型，但特征提取器本身仍不可解释，概念的 grounding 可能不忠实于模型。
   - **CRAFT 等 post-hoc 方法**：通过 NMF 分解模型激活为概念，但概念并非直接用于预测，需要依赖近似方法估计概念重要性和可视化，不一定忠实。

**现有方法的限制性假设**：
   - 概念是类别特定的（class-specific），无法看到跨类共享的概念
   - 假设概念只对应小的图像 patch 或物体部件
   - 概念来自预定义集合

**核心矛盾**：现有方法在概念提取和归因环节引入近似，导致解释不忠实（unfaithful）。而且评估概念一致性的指标依赖人工标注的部件 mask，覆盖不全且假设每个概念都对应标注部件。

## 核心问题

如何设计一个模型，使其提供的概念解释**在设计上就忠实于模型决策**——即概念对输出的贡献是精确可计算的（加和等于 logit），概念在输入空间的可视化也是精确的（加和等于概念激活值），而非依赖近似？

## 方法详解

### 整体架构

FaCT 由两个核心组件构成：

1. **B-cos 变换层**：替代常规 ReLU 层，实现动态线性（dynamic-linear）变换
2. **无偏置稀疏自编码器 (Bias-free SAE)**：在中间层提取稀疏概念表示

#### B-cos 变换

常规 ReLU 层：$f^{\text{Standard}}(x) = \text{ReLU}(\mathbf{W}x + \mathbf{b})$

B-cos 变换去掉偏置，使用行归一化权重 $\hat{\mathbf{W}}$ 和余弦非线性：

$$f^{\text{B-cos}}(x; B) = (\hat{\mathbf{W}} x) \odot |c(\hat{\mathbf{W}}; x)|^{B-1} = \tilde{\mathbf{W}}(x) x$$

关键性质：一系列 B-cos 变换可以化简为输入的**动态线性变换**：

$$f_{1 \to n}^{\text{B-cos}}(x) = \tilde{\mathbf{W}}_{1 \to n}(x) \cdot x$$

这意味着对于任意输入 $x$，模型可以产生一个忠实地再现 logit 的解释 $\tilde{\mathbf{W}}_{1 \to n}(x)$。

#### 无偏置稀疏自编码器

在中间层 $l$ 处，特征 $F = f_{1 \to l}(I)$，SAE 将其编码为稀疏概念激活张量：

$$\mathbf{U} = \text{Encoder}(F) = \text{ReLU}(\text{conv}(\mathbf{W}, F))$$

$$\breve{F} = \text{conv}(\mathbf{V}, \mathbf{U})$$

模型使用重建特征 $\breve{F}$ 计算最终 logit：

$$L^{\text{FaCT}} = f_{l \to n}(\breve{F})$$

**关键设计**：SAE 不含偏置，使得编码过程也是动态线性的，保证了从概念到输入的忠实归因。

### 忠实概念贡献（Logit 分解）

由于 $f_{l \to n}$ 由 B-cos 层组成，logit 可以精确分解为每个概念的贡献：

$$L_c^{\text{FaCT}} = \sum_{k}^{K} \text{Contribution}_k^c$$

其中 $\text{Contribution}_k^c = \sum_{i,j}^{H,W} \tilde{\mathbf{W}}(\mathbf{U})_{i,j,k} \cdot \mathbf{U}_{i,j,k}$。

这是**精确等式**而非近似：所有概念贡献之和等于 logit 值。这与 CRAFT/VCC 等方法形成鲜明对比——后者概念不直接参与计算 logit，需要 post-hoc 的近似重要性度量。

### 忠实输入级可视化

同理，每个概念的激活值可以精确表示为输入像素的动态线性组合：

$$\text{Concept Activation}_k = \sum_{i,j,c}^{H_0,W_0,3} [\tilde{\mathbf{W}}(I) \cdot I]_{i,j,c}$$

每个概念都可以在输入空间获得**像素级精确的可视化**，而非近似的裁剪或上采样热力图。

### C²-score：概念一致性评估指标

现有评估方法依赖人工标注的部件 mask（如 PartImageNet），但存在三个问题：(1) 只覆盖少数类别，(2) 不支持跨类共享的概念，(3) 标注粒度与模型学到的概念不匹配。

FaCT 提出 C²-score：
1. 使用 DINOv2 + LoftUp 提取每张图的高分辨率特征
2. 对每个概念 $k$ 和图像 $I$，用概念归因加权 DINOv2 特征得到概念嵌入 $\mathcal{E}^k(I)$
3. 计算加权余弦相似度衡量一致性：

$$\text{Consistency}^k = \sum_{(I,J) \in \mathcal{D}^2, I \neq J} S^{k,I} S^{k,J} \cos(\mathcal{E}^k(I), \mathcal{E}^k(J))$$

4. 减去随机基线消除偏差：$\text{C}^2\text{-score} = \frac{1}{K}\sum_K \text{Consistency}^k - \text{Consistency}^{rand}$

C²-score 的优势：类无关（class-agnostic）、无需人工标注、考虑归因空间分布、同时支持共享和类特定概念集。

## 实验关键数据

### 实验设置
- 数据集：ImageNet
- 架构：B-cos ResNet-50、B-cos DenseNet-121、B-cos ViT c-S
- SAE 配置：TopK ∈ {8, 16, 32}，总概念数 $K$ ∈ {8192, 16384}
- 在多个层（早期/中期/晚期）训练 SAE

### 性能保持
- ImageNet 精度下降 < 3%，同时概念一致性大幅提升
- DenseNet Block 3/4 的 C²-score 从 0.11 提升到 0.39

### 概念一致性（C²-score 比较）

| 方法 | C²-score |
|------|----------|
| B-cos channels | 0.09 |
| CRP | - (低于 FaCT) |
| CRAFT | - (低于 FaCT) |
| **FaCT** | **0.37** |

FaCT 的概念一致性显著优于所有基线方法。

### 概念删除实验
- 按贡献度从高到低删除概念，FaCT 的 Eq. 9（忠实贡献）导致的 logit 和准确率下降远比 Saliency、Sobol 等 post-hoc 方法陡峭
- 特别是在早期层 Block 2/4，删除少量概念即导致准确率急剧下降，验证了忠实贡献的有效性

### 用户研究（38 名参与者）
- FaCT 概念可解释性评分远高于 B-cos 通道基线（早期和晚期层均如此）
- 输入级可视化显著提升可解释性，尤其对早期层概念平均提升约 0.5/5 分
- C²-score 与用户评分的 Spearman 相关性：全部 38 名参与者均为正相关，33/38 为中等以上相关（> 0.4）

## 亮点

1. **设计上忠实（Faithful by design）**：概念贡献精确加和等于 logit，概念可视化精确加和等于激活值——不是近似，是数学上的等式
2. **跨类共享概念**：概念在所有类别间共享（如"轮子"概念出现在校车和自行车类别中），提供了统一的概念基础，有助于理解误分类
3. **跨层概念层次**：可在不同层提取概念，形成从底层纹理到高层语义的层次结构
4. **概念多样性**：不假设固定空间大小，概念从小局部（头盔）到大范围（木纹理）都有覆盖
5. **C²-score 评估指标**：利用基础模型的通用特征评估概念一致性，避免了对人工标注的依赖
6. **误分类分析**：共享概念基础使得可以分析误分类原因——如篮球被误分类为排球时，可以看到"球"、"球衣"等共同概念的贡献

## 局限与展望

1. **依赖 B-cos 架构**：需要使用 B-cos 变换替代标准层，不能直接应用于任意现有模型
2. **精度下降**：虽然 < 3%，但在某些场景下可能不可接受
3. **概念数量大**：$K$ 为 8192 或 16384，浏览和理解所有概念对用户来说有负担
4. **SAE 训练的不稳定性**：存在"死亡"概念（从不激活）和"始终活跃"概念（> 60% 数据上激活）的问题
5. **概念没有文本标签**：虽然可以用 CLIP-Dissect 辅助命名，但不是方法本身的一部分
6. **仅在 ImageNet 上充分评估**：附录有 CUB 结果，但缺少更多领域（医疗、遥感等）的验证

## 与相关工作的对比

| 方法 | 概念忠实性 | 输入可视化 | 共享概念 | 评估方式 |
|------|-----------|-----------|---------|---------|
| CRAFT | 近似（NMF） | 近似（上采样） | ❌ 类特定 | 标注 IoU |
| CRP | 近似 | 近似 | ✅ | 标注 IoU |
| Part-Prototype | 不忠实 | Patch 相似度 | ❌ | 标注 IoU |
| CBM | 不忠实 | 无 | ❌ | 预定义集合 |
| **FaCT** | **精确等式** | **像素级精确** | **✅ 共享** | **C²-score** |

## 启发与关联

1. **SAE + 可解释架构的组合范式**：将 SAE（最初用于理解 LLM 的特征）与 B-cos 可解释架构结合，是一个值得推广的研究方向——未来可能扩展到视频、3D 等模态
2. **忠实性 vs. 近似性的权衡**：本文清晰地展示了"设计上忠实"和"post-hoc 近似"之间的差距，对可解释 AI 领域有方法论意义
3. **基础模型作为评估工具**：用 DINOv2 特征替代人工标注来评估概念一致性，这一思路可以推广到其他评估任务

## 评分
- 新颖性: ⭐⭐⭐⭐ (B-cos + SAE 的组合新颖，C²-score 是有价值的贡献)
- 实验充分度: ⭐⭐⭐⭐⭐ (多架构、多层、用户研究、消融实验、概念删除、误分类分析)
- 写作质量: ⭐⭐⭐⭐⭐ (结构清晰，数学推导严谨，可视化优秀)
- 价值: ⭐⭐⭐⭐ (对可解释 AI 领域有实质推进，但 B-cos 依赖限制了通用性)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] FastCAV: Efficient Computation of Concept Activation Vectors for Explaining Deep Neural Networks](../../ICML2025/interpretability/fastcav_efficient_computation_of_concept_activation_vectors_for_explaining_deep_.md)
- [\[ICLR 2026\] Provably Explaining Neural Additive Models](../../ICLR2026/interpretability/provably_explaining_neural_additive_models.md)
- [\[ICLR 2026\] Causal Interpretation of Neural Network Computations with Contribution Decomposition](../../ICLR2026/interpretability/causal_interpretation_of_neural_network_computations_with_contribution_decomposi.md)
- [\[NeurIPS 2025\] Towards Interpretability Without Sacrifice: Faithful Dense Layer Decomposition with Mixture of Decoders](towards_interpretability_without_sacrifice_faithful_dense_layer_decomposition_wi.md)
- [\[NeurIPS 2025\] Explaining Similarity in Vision-Language Encoders with Weighted Banzhaf Interactions](explaining_similarity_in_vision-language_encoders_with_weighted_banzhaf_interact.md)

</div>

<!-- RELATED:END -->
