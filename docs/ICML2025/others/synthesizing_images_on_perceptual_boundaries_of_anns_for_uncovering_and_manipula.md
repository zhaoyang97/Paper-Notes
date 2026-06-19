---
title: >-
  [论文解读] Synthesizing Images on Perceptual Boundaries of ANNs for Uncovering and Manipulating Human Perceptual Variability
description: >-
  [ICML 2025][感知变异性] 提出 BAM（Boundary Alignment & Manipulation）框架，通过在 ANN 感知决策边界上采样生成图像刺激，系统性地揭示、预测和操控人类个体间的感知差异。 人类在认知任务中表现出显著的决策变异性，即面对相同的物理刺激，不同个体可能产生截然不同的感知体验。例如…
tags:
  - "ICML 2025"
  - "感知变异性"
  - "决策边界采样"
  - "扩散模型引导"
  - "人类-AI对齐"
  - "个体差异建模"
---

# Synthesizing Images on Perceptual Boundaries of ANNs for Uncovering and Manipulating Human Perceptual Variability

**会议**: ICML 2025  
**arXiv**: [2505.03641](https://arxiv.org/abs/2505.03641)  
**代码**: 无  
**领域**: 认知科学 / 人工智能  
**关键词**: 感知变异性, 决策边界采样, 扩散模型引导, 人类-AI对齐, 个体差异建模

## 一句话总结

提出 BAM（Boundary Alignment & Manipulation）框架，通过在 ANN 感知决策边界上采样生成图像刺激，系统性地揭示、预测和操控人类个体间的感知差异。

## 研究背景与动机

人类在认知任务中表现出显著的决策变异性，即面对相同的物理刺激，不同个体可能产生截然不同的感知体验。例如，同一张手写数字图片，有人看成"3"，有人看成"5"。这种**感知变异性（perceptual variability）**在复杂认知任务（如审美判断、道德判断）中已被广泛记录，但在简单视觉决策任务（如手写数字分类）中的个体差异却鲜少被深入研究。

现有研究表明，ANN 的潜在表征与人类心理表征之间存在强相关性。受此启发，作者提出核心假设：**ANN 的感知决策边界与人类个体间的感知变异性相关**——在这些边界上采样生成的图像能够在人类被试中引发分歧性的感知体验。

此前的相关工作存在以下不足：

- **争议刺激**（Golan et al., 2020）仅关注模型间的分类差异，未扩展到人类感知
- **对抗扰动**（Veerabadran et al., 2023; Gaziv et al., 2024）生成的图像自然度不足，难以有效影响人类认知
- **模型元图**（Feather et al., 2023）揭示了模型与人类的不一致性，但未探索个体差异

## 方法详解

### 整体框架

BAM 框架由三个互联步骤组成：

1. **Labeling（标注）**：在 ANN 感知边界上采样生成图像，通过大规模人类行为实验收集标注，构建高感知变异性数据集 variMNIST
2. **Aligning（对齐）**：利用人类行为数据微调 ANN 模型，在群体和个体两个层面建立感知变异性的计算模型
3. **Manipulating（操控）**：使用个体化模型作为对抗生成器，合成争议刺激以放大特定被试对之间的感知差异

### 关键设计

#### 1. 感知边界采样算法

采用分类器引导的扩散模型（Classifier-guided Diffusion Model）在 ANN 决策边界上生成图像。具体使用两种引导策略：

**不确定性引导（Uncertainty Guidance）**：使单个分类器对生成图像的预测概率在两个类别间保持均等，使图像落在决策边界附近。

$$\mathcal{L} = H(p_1(y|x), q_1(y))$$

其中 $H$ 为交叉熵，目标分布 $q_1(y)$ 确保两个类别（如"3"和"7"）的概率相等。

**争议引导（Controversial Guidance）**：使两个分类器对同一图像产生相反的高置信度预测。

$$\mathcal{L} = H(p_1(y|x), q_1(y)) + H(p_2(y|x), q_2(y))$$

目标分布让分类器 1 高置信度预测类别 A，分类器 2 高置信度预测类别 B，从而生成处于两个模型决策边界交叉区域的图像。

#### 2. 扩散先验增强自然度

ANN 决策边界区域通常包含高噪声图像。本文利用 DiT（Diffusion Transformer）作为图像生成核心，通过 DDPM 采样引入 MNIST 数据分布先验：

$$x_{t-1} = DDPM^{-}(x_t) - \gamma \nabla_{x_t} \ell(f_\phi(x_t), y)$$

其中 $\gamma$ 控制引导强度。DiT 配置为：patch size 2×2，隐藏维度 128，4 层 Transformer，8 头注意力。扩散先验有效解决了此前方法生成图像不自然的问题。

#### 3. 数字判断代理模型（Digit Judgment Surrogate）

为过滤不合格图像，训练代理模型预测图像是否像数字。该模型基于 SmallVGG 架构，以人类判断频率为回归目标训练。代理模型预测分数与人类评分的 Spearman 秩相关系数达 0.8035，验证了其有效性。代理引导损失：

$$\mathcal{L}_{surr} = \mathcal{L} + \max((1 - f_{surr}(x))^2, 0.5)$$

#### 4. 模型对齐与个性化微调

- **群体微调（GroupNet）**：在 MNIST + variMNIST（1:1 比例）混合数据上微调
- **个体微调（IndivNet）**：在 variMNIST-i + variMNIST + MNIST（2:1:1 比例）上从 GroupNet 继续微调
- 使用 AdamW 优化器（lr=1e-3），CrossEntropyLoss，batch size 128，16 epochs

#### 5. 操控实验范式

选择 18 位被试（6 组 × 3 人），组内两两配对形成 18 对。两轮实验：

- **第一轮**：每对完成 ~500 trials，收集行为数据后微调个体模型
- **第二轮**：用两个个体化模型通过争议引导生成 ~180 个刺激，验证操控效果

### 损失函数 / 训练策略

引导生成超参设置：引导强度 γ=0.1，重采样步数 5，推理步数 50。对于带 MSE 约束的引导，像素级约束强度 α=50。

过滤策略：不确定性采样要求 top-2 概率值 >0.4 且代理分数 >0.5；争议采样要求两个分类器输出与引导方向一致、最高概率 >0.9 且代理分数 >0.5。

## 实验关键数据

### 主实验

数据集 variMNIST：19,943 张图像，246 名被试，116,715 次试验。

| 模型 | MNIST 准确率 | variMNIST 准确率 | variMNIST-i 准确率 |
|------|------------|----------------|-------------------|
| BaseNet | ~98% | ~60% | ~55% |
| GroupNet | ~98% | ~80% (+20%) | ~75% |
| IndivNet | ~98% | ~80% | ~80% (+5%) |

五种分类器的 MNIST 基线准确率：

| 分类器 | 架构 | MNIST 准确率 |
|--------|------|-------------|
| ViT | Vision Transformer | 97.2% |
| VGG | Small VGG | 98.2% |
| CORNet | CORnet-Z | 98.9% |
| MLP | Multi-Layer Perceptron | 98.3% |
| LRM | Logistic Regression | 92.7% |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| VGG 模型-人类熵相关（微调前） | Spearman ρ=0.08 | 基线模型几乎无法捕捉人类变异性 |
| VGG 模型-人类熵相关（微调后） | Spearman ρ=0.74 | 微调显著提升对齐度 |
| 个体微调提升 | 241/246 被试提升 | 仅 5 人略有下降 |
| 争议引导：CORNet/VGG | 成功率 ~0.6 | 最优分类器组合 |
| 争议引导：LRM | 成功率 ~0.2 | 最弱分类器 |
| 操控实验成功率（variMNIST → IndivNet） | +3% (p<0.001) | 个性化模型提升引导成功率 |
| 操控实验定向比（variMNIST → IndivNet） | +12% (p<0.001) | 个性化模型显著提升引导方向性 |

### 关键发现

1. **ANN 变异性能引发人类变异性**：超过一半生成图像的人类选择熵显著大于零；引导成功率 + 偏向率平均接近 80%
2. **个体微调在高难度样本上更有效**：IndivNet 相对 GroupNet 的提升主要集中在高熵（高难度）图像上
3. **被试聚类分析**发现 8 个感知簇，同簇被试的模型预测更准确，证实了高层感知差异的存在
4. **最有效引导目标对**为 (1,7)、(1,2)、(4,9)，成功率 >0.35；最难引导对为 (1,8)、(2,9)、(7,8)，成功率 <0.03
5. **ImageNet 验证**：在自然图像（Restricted ImageNet 9 类）上复现了一致结论，群体微调提升 ~4%，个体微调再提升 ~2%

## 亮点与洞察

- **桥接计算模型与人类个体差异研究**：首次系统性地将 ANN 决策边界采样与人类感知变异性联系起来，开创了"从模型间争议到人类间争议"的新范式
- **数据驱动的个性化感知模型**：仅需约 200 个试次即可建立有效的个体化感知模型，成本极低
- **扩散先验的巧妙使用**：解决了决策边界区域图像噪声严重的长期问题，使生成图像在保持边界特性的同时足够自然
- **大规模人类实验验证**：246 名被试、116,715 次试验的规模在同类研究中属顶尖水平

## 局限与展望

1. **文化偏差**：训练数据来自特定群体，可能无法充分捕捉文化差异导致的感知变异。未来需招募不同文化背景的被试
2. **任务范围有限**：仅聚焦物体识别任务，未涉及相似性判断、情感识别、视觉注意等更复杂的视觉现象
3. **AI-人类对齐差距依然显著**：微调后的模型在预测感知变异性上仍远不及标准分类任务的表现
4. **个体实验试次受限**：行为实验中每位被试的试次有限，限制了个体模型的精度上限
5. **未来方向**：结合最优实验设计（Bayesian Optimal Experimental Design），用个体化模型生成最大化信息量的刺激，形成"实验→微调→生成→实验"闭环

## 相关工作与启发

- **争议刺激**（Golan et al., 2020, 2023）：将模型间争议扩展到人类间争议
- **对抗扰动影响人类**（Veerabadran et al., 2023; Gaziv et al., 2024）：验证了 ANN 扰动可影响人类感知
- **DreamSim**（Fu et al., 2023）：利用合成数据对齐感知度量
- **扩散模型反事实解释**（Jeanneret et al., 2023; Wei et al., 2024b）：为自然图像生成提供技术基础
- **启发**：该框架可扩展至更广泛的 AI-人类对齐场景，如个性化推荐系统中的用户偏好建模、医学图像诊断中的专家差异分析

## 评分

| 维度 | 评分 | 说明 |
|------|------|------|
| 创新性 | ⭐⭐⭐⭐⭐ | 首次系统性研究 ANN 决策边界与人类感知变异性的关系 |
| 方法严谨性 | ⭐⭐⭐⭐⭐ | 大规模人类实验 + 完整的"采样→对齐→操控"闭环验证 |
| 实验充分性 | ⭐⭐⭐⭐⭐ | 246 被试，116K trials，涵盖 5 种分类器和两个数据域 |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，但部分细节需参考附录 |
| 影响力 | ⭐⭐⭐⭐ | 为 AI-人类对齐和个性化感知研究提供新工具 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Manipulating Feature Visualizations with Gradient Slingshots](../../NeurIPS2025/others/manipulating_feature_visualizations_with_gradient_slingshots.md)
- [\[AAAI 2026\] On the Variability of Concept Activation Vectors](../../AAAI2026/others/on_the_variability_of_concept_activation_vectors.md)
- [\[ACL 2025\] From Real to Synthetic: Synthesizing Millions of Diversified and Complicated User Instructions with Attributed Grounding](../../ACL2025/others/from_real_to_synthetic_synthesizing_millions_of_diversified_and_complicated_user.md)
- [\[CVPR 2025\] Locally Orderless Images for Optimization in Differentiable Rendering](../../CVPR2025/others/locally_orderless_images_for_optimization_in_differentiable_rendering.md)
- [\[ACL 2025\] A Little Human Data Goes A Long Way](../../ACL2025/others/a_little_human_data_goes_a_long_way.md)

</div>

<!-- RELATED:END -->
