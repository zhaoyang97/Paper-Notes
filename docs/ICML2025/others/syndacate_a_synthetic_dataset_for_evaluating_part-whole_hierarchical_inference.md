---
title: >-
  [论文解读] SynDaCaTE: A Synthetic Dataset for Evaluating Part-Whole Hierarchical Inference
description: >-
  [ICML 2025 (MOSS Workshop)][胶囊网络] 提出SynDaCaTE合成数据集和Mereological Inference框架，将部分-整体层次推断分解为Image-to-Parts和Parts-to-Wholes两个可独立评估的子任务，通过精心设计的控制实验证明CapsNet的瓶颈在于从图像提取部件而非从部件推断整体，同时发现置换等变的SetTransformer在部件到整体推断中显著优于所有基线（超过10倍精度优势）。
tags:
  - "ICML 2025 (MOSS Workshop)"
  - "胶囊网络"
  - "部分-整体层次"
  - "合成数据集"
  - "Transformer"
  - "归纳偏置"
---

# SynDaCaTE: A Synthetic Dataset for Evaluating Part-Whole Hierarchical Inference

**会议**: ICML 2025 (MOSS Workshop)  
**arXiv**: [2506.17558](https://arxiv.org/abs/2506.17558)  
**代码**: [GitHub](https://github.com/jakelevi1996/syndacate-public)  
**领域**: 计算机视觉 / 归纳偏置  
**关键词**: 胶囊网络, 部分-整体层次, 合成数据集, SetTransformer, 归纳偏置

## 一句话总结
提出SynDaCaTE合成数据集和Mereological Inference框架，将部分-整体层次推断分解为Image-to-Parts和Parts-to-Wholes两个可独立评估的子任务，通过精心设计的控制实验证明CapsNet的瓶颈在于从图像提取部件而非从部件推断整体，同时发现置换等变的SetTransformer在部件到整体推断中显著优于所有基线（超过10倍精度优势）。

## 研究背景与动机

**领域现状**：部分-整体层次（part-whole hierarchy）是人类视觉系统的核心能力，胶囊网络（CapsNet）声称能够学习这种层次结构。然而自Hinton提出以来，CapsNet已逐渐被CNN和Vision Transformer取代，其承诺的"层次推断"能力从未被严格验证。

**现有痛点**：现有视觉数据集缺乏ground-truth的部件信息——我们知道一张图片包含什么物体，但不知道物体由哪些部件组成、部件的精确位姿是什么。没有这种标注，就无法判断模型到底是学到了部分-整体推断，还是只是用了某种捷径完成分类任务。

**核心矛盾**：CapsNet在标准分类任务上表现不如现代CNN/Transformer，但我们不清楚它到底是在哪个环节失败了——是从图像提取部件（Image-to-Parts）的能力不足，还是从部件组装整体（Parts-to-Wholes）的能力不足？这两个子任务被耦合在端到端训练中，无法分离评估。

**本文目标** （1）定义一个清晰的框架来形式化"部分-整体推断"的含义；（2）构建一个带有ground-truth部件信息的合成数据集来解耦两个子任务；（3）精确定位CapsNet的瓶颈，并为未来归纳偏置设计提供方向。

**切入角度**：作者从认知科学中的mereology（整体学）出发，将视觉推断严格分解为两步：先推断部件集合，再从部件推断整体，并通过合成数据提供每一步的ground-truth。

**核心 idea**：用带有完整部件标注的合成数据集将层次推断解耦为两个独立可评估的子任务，从而精确诊断模型能力。

## 方法详解

### 整体框架
定义Mereological Inference为两步推断过程：（1）Image-to-Parts：从图像 $I \in \mathbb{R}^{C \times H \times W}$ 推断部件集合 $\mathcal{P}$（每个部件有类别标签+姿态向量）；（2）Parts-to-Wholes：从部件集合 $\mathcal{P}$ 推断整体集合 $\mathcal{W}$。通过SynDaCaTE数据集，可以分别评估模型在这两个子任务上的表现。

### 关键设计

1. **SynDaCaTE数据集**:

    - 功能：提供一个包含完整部件ground-truth信息的合成视觉数据集
    - 核心思路：数据集包含3类对象（线段、字符、单词）共21种类型，每个对象有类别标签和连续姿态向量（位置/大小/旋转/亮度等）。图像按层次生成——先采样顶层对象（如单词），再递归生成子部件（字符→线段），最后渲染成图像。通过控制生成参数，可以定义多种任务：ImToClass（图像→分类）、ImToParts（图像→部件集合）、PartsToChars（部件→字符）、PartsToClass（部件→分类）等
    - 设计动机：自然图像的部件标注昂贵且模糊，合成数据可以精确控制部件信息，实现对两个子任务的独立评估。虽然数据集很简单，但正是这种简单性使得实验结论清晰可靠

2. **PreTrainedPartsToClass任务设计**:

    - 功能：通过预训练CNN提取的部件表征来替代原始图像输入，测试模型在"已有部件信息"时的分类能力
    - 核心思路：先在ImToParts任务上训练一个CNN学习从图像提取部件，然后用训练好的CNN的最后一层特征作为新的输入，在此基础上训练CapsNet和CNN做分类。如果CapsNet在使用部件表征后性能与CNN持平，则说明CapsNet的瓶颈在Image-to-Parts而非Parts-to-Wholes
    - 设计动机：这是定位CapsNet瓶颈的关键控制实验——通过提供预提取的部件信息，绕过Image-to-Parts阶段，直接考察Parts-to-Wholes能力

3. **置换等变模型对部件-整体推断的评估**:

    - 功能：在PartsToChars任务上对比SetTransformer、DeepSetToSet、逐元素MLP和扁平化MLP
    - 核心思路：部件集合是无序的，因此置换等变/不变的模型应该具有更好的归纳偏置。SetTransformer通过自注意力机制处理集合输入，在深度≥2时MSE比其他基线低超过一个数量级。增加宽度（参数量×4）对浅层SetTransformer几乎无帮助，说明≥2层自注意力提供了某种本质上不可替代的计算能力
    - 设计动机：大多数视觉模型直接在像素级操作，忽略了"从部件推断整体"这一子任务的最优归纳偏置。通过在纯Parts-to-Wholes任务上比较不同架构，为未来设计更好的视觉模型提供指导

### 损失函数 / 训练策略
ImToClass和PartsToClass使用交叉熵损失；ImToParts使用Chamfer MSE损失（集合预测常用）；PartsToChars使用在输出集合上平均的MSE损失。所有模型用Adam优化器，ImToClass训练5k步，ImToParts训练100个epoch，PartsToChars训练100个epoch。

## 实验关键数据

### 主实验：CapsNet瓶颈定位

| 任务 | 输入类型 | CNN精度 | CapsNet精度 | 结论 |
|------|---------|---------|------------|------|
| ImToClass | 原始图像 | ~95%（100样本） | ~75%（100样本） | CNN显著优于CapsNet |
| ImToClass | 原始图像 | ~99%（60k样本） | ~97%（60k样本） | 大数据下差距缩小 |
| PreTrainedPartsToClass | 部件表征 | ~97%（100样本） | ~97%（100样本） | **给定部件信息后两者持平** |
| PartsToClass | Ground-truth部件 | - | - | SetTransformer远优于其他 |

### 消融实验：Parts-to-Wholes模型对比（PartsToChars任务）

| 模型 | 深度=1 MSE | 深度=2 MSE | 深度=4 MSE | 特点 |
|------|-----------|-----------|-----------|------|
| SetTransformer | ~0.1 | ~0.005 | **~0.002** | ≥2层后突变性提升 |
| SetTransformer (2×宽) | ~0.08 | ~0.004 | ~0.002 | 增加宽度几乎无帮助 |
| DeepSetToSet | ~0.15 | ~0.08 | ~0.06 | 提升缓慢 |
| Element-wise MLP | ~0.2 | ~0.15 | ~0.12 | 无法利用集合间关系 |
| Flattened MLP | ~0.25 | ~0.2 | ~0.15 | 最差，缺乏置换不变性 |

### 关键发现
- **CapsNet的瓶颈精确定位于Image-to-Parts阶段**：一旦提供预提取的部件信息，CapsNet与CNN的分类精度几乎相同，证明CapsNet在从部件推断整体方面并不比CNN更强
- **SetTransformer在深度≥2时性能突变**：MSE从~0.1骤降至~0.005（提升20倍），且增加宽度无效，暗示≥2层自注意力提供了某种不可替代的计算结构（可能与"Induction Heads"现象相关）
- **部件信息是高效分类的有力表征**：即使部件表征有噪声，使用它们做分类也比从原始图像做分类更高效，支持part-whole层次作为有用归纳偏置的假设

## 亮点与洞察
- **精准的实验设计**：通过PreTrainedPartsToClass这个巧妙的"桥接"任务，干净利落地将CapsNet的失败归因于Image-to-Parts阶段，这是此前CapsNet文献中从未有过的清晰定位。
- **SetTransformer的深度敏感性**：≥2层自注意力是Parts-to-Wholes推断所必需的，与Transformer circuits文献中的Induction Heads发现遥相呼应，暗示部件组装需要某种二阶推理能力。
- **简洁但深刻的数据集设计**：SynDaCaTE虽然只用了线段/字符/单词三层简单对象，但正是这种极简设计使得结论极其清晰——复杂数据集可能引入过多干扰变量。

## 局限与展望
- 合成数据过于简单（2D线段和字符），与自然图像/3D场景差距很大，结论能否迁移存疑
- 仅评估了Sabour et al. (2017)的原始CapsNet，未测试后续改进版本（如Matrix Capsules、Efficient-CapsNet）
- Parts-to-Wholes任务假设已知ground-truth部件，但实际中部件提取本身就是核心挑战
- 作为Workshop paper篇幅有限，缺少对MereoFormer等后续方向的完整实验验证

## 相关工作与启发
- **vs CapsNet系列（Sabour 2017, Hinton 2018）**：这些工作声称CapsNet能学习部分-整体层次，但从未提供证据。本文首次通过控制实验证明CapsNet在最基础的Image-to-Parts上就失败了
- **vs Slot Attention（Locatello 2020）**：Slot Attention也关注物体分解，但侧重无监督的object-centric表征，未明确评估part-whole层次
- **vs Vision Transformer**：本文发现的"自注意力在集合推断中的优势"为理解ViT的成功提供了新视角——ViT可能部分受益于自注意力对local patch间part-whole关系的建模

## 评分
- 新颖性: ⭐⭐⭐⭐ 框架定义清晰，首次将层次推断解耦评估，但数据集本身较简单
- 实验充分度: ⭐⭐⭐ Workshop paper篇幅下消融尚可，但缺少更多CapsNet变体和自然图像实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，论证逻辑严密，结论简洁有力
- 价值: ⭐⭐⭐⭐ 对理解CapsNet的失败原因和未来归纳偏置设计有重要指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding](../../ACL2025/others/kodcode_a_diverse_challenging_and_verifiable_synthetic_dataset_for_coding.md)
- [\[CVPR 2025\] BenDFM: A taxonomy and synthetic CAD dataset for manufacturability assessment in sheet metal bending](../../CVPR2025/others/bendfm_a_taxonomy_and_synthetic_cad_dataset_for_manufacturability_assessment_in_.md)
- [\[ICML 2025\] Provably Improving Generalization of Few-Shot Models with Synthetic Data](provably_improving_generalization_of_few-shot_models_with_synthetic_data.md)
- [\[ICML 2025\] Improving Generalization with Flat Hilbert Bayesian Inference](improving_generalization_with_flat_hilbert_bayesian_inference.md)
- [\[ACL 2025\] Evaluating the Evaluation of Diversity in Commonsense Generation](../../ACL2025/others/evaluating_the_evaluation_of_diversity_in_commonsense_generation.md)

</div>

<!-- RELATED:END -->
