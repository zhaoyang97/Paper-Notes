---
title: >-
  [论文解读] MPA: Multimodal Prototype Augmentation for Few-Shot Learning
description: >-
  [AAAI 2026][医学图像][小样本学习] 本文提出 MPA 框架，通过 LLM 生成多变体语义描述增强原型的语义信息（LMSE）、层次化多视角数据增强丰富视觉特征（HMA）、以及自适应不确定类吸收器建模类间不确定性（AUCA），在 4 个单域和 6 个跨域小样本学习基准上显著超越现有方法，5-way 1-shot 下单域和跨域分别比次优方法高出 12.29% 和 24.56%。
tags:
  - "AAAI 2026"
  - "医学图像"
  - "小样本学习"
  - "多模态原型"
  - "LLM语义增强"
  - "数据增强"
  - "CLIP"
---

# MPA: Multimodal Prototype Augmentation for Few-Shot Learning

**会议**: AAAI 2026  
**arXiv**: [2602.10143](https://arxiv.org/abs/2602.10143)  
**代码**: [GitHub](https://github.com/ww36user/MPA)  
**领域**: 小样本学习 / 多模态学习  
**关键词**: 小样本学习, 多模态原型, LLM语义增强, 数据增强, CLIP

## 一句话总结

本文提出 MPA 框架，通过 LLM 生成多变体语义描述增强原型的语义信息（LMSE）、层次化多视角数据增强丰富视觉特征（HMA）、以及自适应不确定类吸收器建模类间不确定性（AUCA），在 4 个单域和 6 个跨域小样本学习基准上显著超越现有方法，5-way 1-shot 下单域和跨域分别比次优方法高出 12.29% 和 24.56%。

## 研究背景与动机

小样本学习（FSL）旨在仅用少量标注样本识别新类别，基于原型的度量方法因简洁高效而广受关注。这类方法通常从支持集图像中计算原型，然后根据查询样本与原型的距离分类。

然而，现有方法存在两个关键局限：（1）**仅依赖视觉模态**——在有限的支持集样本中，视觉信息不足以捕捉类别的完整表征，尤其是视觉外观极其相似但属于不同类别的情况（如不同鸟类物种颜色、形态相似，但喙和翅膀条纹不同）；（2）**缺乏多视角表示**——单一视角的图像无法充分表达目标的多维特征。

本文的核心切入点：将多模态信息（语义文本 + 多视角视觉）和不确定性建模引入原型构建过程。利用大语言模型生成丰富的类别语义描述来补充视觉信息，通过多层次数据增强获取多视角特征来增强表示多样性，并通过构造不确定类来吸收难以分类的边界样本。

## 方法详解

### 整体框架

MPA 框架基于 CLIP 模型，包含三个核心模块：LMSE 利用 LLM 生成多变体语义特征并通过 CLIP 文本编码器嵌入；HMA 通过自然变换和几何视角增强支持集图像的视觉多样性；AUCA 通过插值和高斯采样构造自适应不确定类来吸收不确定样本。最终使用逻辑回归分类器在增强后的特征上进行分类。

### 关键设计

1. **LLM-based Multi-Variant Semantic Enhancement (LMSE)**:

    - 功能：利用大语言模型为每个类别生成多变体语义描述，丰富支持集的语义信息。
    - 核心思路：给定类别名称 $c$，利用 LLM（如 GPT-4.0）生成外观描述及其 4 个改写变体，得到语义集合 $\{t_m\}_{m=1}^{M}$。通过 CLIP 文本编码器 $h(\cdot)$ 将语义变体投影到嵌入空间：$F_t = h(\mathcal{G}_\text{LLM}(c))$。同时用 CLIP 图像编码器提取支持集图像特征 $F_i = f(\{I_n\}_{n=1}^{N})$。
    - 设计动机：语义描述聚焦于关键属性（不受嘈杂背景干扰），LLM 生成的多变体引入了上下文多样性和潜在知识，能显著增强泛化能力。不同于固定或手写类名，LLM 变体提供了更丰富的语义空间覆盖。

2. **Hierarchical Multi-View Augmentation (HMA)**:

    - 功能：通过层次化的数据增强策略生成多视角特征，增强支持集的视觉多样性。
    - 核心思路：对每个支持集图像 $I$ 应用两层增强——(a) 自然视角增强 $I_a = \{\tau_n(I) | \tau_n \in \mathcal{T}\}$，包括中心裁剪（120、170、200 像素）、旋转（45°、90°、180°、270°、315°）和颜色抖动（亮度/对比度/饱和度 0.5，色调 0.2）；(b) 几何视角增强，通过水平翻转生成互补视角。所有增强图像通过 CLIP 图像编码器提取特征 $F_a \in \mathbb{R}^{M \times d}$。
    - 设计动机：模拟真实场景中观察距离、相机角度和光照条件的自然变化，在不改变标签的前提下增加特征多样性，减轻对少量支持样本的过度依赖。

3. **Adaptive Uncertain Class Absorber (AUCA)**:

    - 功能：动态构造不确定类，吸收位于类别决策边界附近的不确定样本，减轻类间干扰。
    - 核心思路：首先通过类间原型插值生成混合特征 $D_i = [\alpha, 1-\alpha] \cdot [F_j; F_k]$（$\alpha \in [0.2, 0.8]$），同时从标准正态分布采样 $D_n \sim \mathcal{N}(0,1)$。计算所有类对原型的余弦相似度矩阵 $\mathbf{S}$，归一化后求平均差异 $\lambda = 1 - \frac{2}{\binom{C}{2}} \sum_{j<k} S'_{j,k}$。最终不确定类数据由两者按 $\lambda$ 概率混合：$\mathbb{E}[D_u] = (1-\lambda) \cdot D_n + \lambda \cdot D_i$。
    - 设计动机：$\lambda$ 自适应反映数据特性——跨域场景中特征更聚集、$\lambda$ 更小（更多随机性）；单域场景中特征更可分、$\lambda$ 更大（更多插值）。不确定类为分类器提供了负类信息，建立更鲁棒的决策边界。

### 损失函数 / 训练策略

- 特征提取器使用预训练 CLIP（ViT-L/14），冻结不训练
- LLM 语义生成为离线预处理步骤
- 最终分类使用逻辑回归分类器，在增强后的多模态特征上训练
- 每个 epoch 随机采样 100 个 episode，每个 episode 包含 5 类的支持集和查询集

## 实验关键数据

### 主实验

**5-way 1-shot 单域数据集：**

| 方法 | miniImageNet | tieredImageNet | CIFAR-FS | FC100 | 平均 |
|------|-------------|----------------|----------|-------|------|
| SPM (AAAI'24) | 93.70 | 88.79 | 82.40 | 68.35 | 83.31 |
| MLVLM (AAAI'25) | 98.24 | 98.06 | 95.02 | - | - |
| **MPA (本文)** | **98.87** | **98.57** | **97.47** | **87.47** | **95.60** |

**5-way 1-shot 跨域数据集：**

| 方法 | CUB | Cars | Places | Plantae | EuroSAT | CropDisease | 平均 |
|------|-----|------|--------|---------|---------|-------------|------|
| SVasP (AAAI'25) | 85.56 | 40.51 | 75.93 | 56.25 | 75.51 | 83.98 | 69.62 |
| SPM (AAAI'24) | 84.39 | 41.71 | 72.35 | 53.85 | 74.97 | 84.43 | 68.62 |
| **MPA (本文)** | **98.95** | **98.51** | **93.55** | **91.73** | **87.05** | **95.28** | **94.18** |

MPA 在跨域 1-shot 设定下平均提升高达 24.56%，在 Cars 数据集上的提升尤为惊人（从 41.71% 到 98.51%）。

### 消融实验

| LMSE | HMA | AUCA | EuroSAT | Places | CIFAR-FS |
|------|-----|------|---------|--------|----------|
| ✗ | ✗ | ✗ | 76.41 | 87.24 | 93.69 |
| ✓ | ✗ | ✗ | 83.03 | 93.43 | 95.36 |
| ✗ | ✓ | ✗ | 79.44 | 84.71 | 94.17 |
| ✓ | ✓ | ✗ | 85.69 | 92.64 | 96.32 |
| ✓ | ✓ | ✓ | **87.05** | **93.55** | **97.47** |

### 关键发现

- **LMSE 贡献最大**：单独使用即可在 EuroSAT 上提升 6.62%（76.41→83.03），在 Places 上提升 6.19%
- **三模块互补性强**：每个模块逐步叠加都带来一致提升，完整框架在所有数据集上最优
- **骨干无关性**：在 ViT-L/14、ViT-B/32、ViT-B/16 和 ResNet101 四种 CLIP 骨干上均显著优于基线
- **LLM 选择鲁棒**：GPT-4.0 虽然最优，但 GPT-3.5、DeepSeek、Claude-4、Gemini-2.5 等 10 种 LLM 之间差异很小（tieredImageNet 上 97.79-98.57%）
- **跨域性能惊人**：特别是 Cars 数据集 5-shot 下达到 99.63%，比次优方法高出 33.16%

## 亮点与洞察

- **多模态原型是小样本学习的有效范式**：通过 LLM 语义和多视角视觉的联合增强，原型质量大幅提升，尤其在细粒度任务中优势显著
- **AUCA 的自适应机制设计巧妙**：$\lambda$ 基于类间相似度自动调节不确定类的组成，无需手动设定超参数
- **性能提升幅度极大**：尤其在跨域和细粒度场景下的提升（Cars 上 40% → 99%）颇为惊人
- **方法简洁实用**：基于 CLIP 冻结特征 + 逻辑回归分类器，无需复杂训练流程

## 局限与展望

- 强烈依赖 CLIP 预训练质量和 LLM 生成能力，在 CLIP 覆盖不好的特殊领域（如医学影像的专业术语）效果可能受限
- 逻辑回归分类器可能不是最优选择，更复杂的分类头或许能进一步提升
- 数据增强策略（裁剪尺寸、旋转角度等）为手动设定，自适应增强策略可能更好
- 性能提升的相当部分可能来自 CLIP 本身的强大表示能力，消融实验中基线（无任何模块，仅 CLIP 特征 + 逻辑回归）在部分数据集上已经很高
- LLM 语义生成是离线步骤，对于全新类别需要额外调用 LLM，增加了部署复杂度

## 相关工作与启发

本文工作处于 FSL 与多模态学习的交叉点。LMSE 的思想与 SPM（利用语义信息作为提示）和 SEVPro（语义增强视觉原型）相关，但更进一步地利用 LLM 生成多变体描述。AUCA 的不确定类思想新颖，为小样本场景下的决策边界建模提供了新视角。整体框架表明：在少样本场景下，"用更多模态和视角补充有限数据"比"开发更复杂的元学习算法"可能更有效。值得探索将类似思路应用到其他低资源学习任务中。

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](../../CVPR2026/medical_imaging/interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)
- [\[CVPR 2026\] Reclaiming Lost Text Layers for Source-Free Cross-Domain Few-Shot Learning](../../CVPR2026/medical_imaging/reclaiming_lost_text_layers_for_source-free_cross-domain_few-shot_learning.md)
- [\[CVPR 2026\] Mind the Discriminability Trap in Source-Free Cross-domain Few-shot Learning](../../CVPR2026/medical_imaging/mind_the_discriminability_trap_in_source-free_cross-domain_few-shot_learning.md)
- [\[NeurIPS 2025\] Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling](../../NeurIPS2025/medical_imaging/few-shot_learning_from_gigapixel_images_via_hierarchical_vision-language_alignme.md)
- [\[CVPR 2026\] Universal-to-Specific: Dynamic Knowledge-Guided Multiple Instance Learning for Few-Shot Whole Slide Image Classification](../../CVPR2026/medical_imaging/universal-to-specific_dynamic_knowledge-guided_multiple_instance_learning_for_fe.md)

</div>

<!-- RELATED:END -->
