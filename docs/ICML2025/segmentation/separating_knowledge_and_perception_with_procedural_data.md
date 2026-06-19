---
title: >-
  [论文解读] Separating Knowledge and Perception with Procedural Data
description: >-
  [ICML2025][语义分割][procedural data] 仅用程序化生成数据（非真实图像）训练视觉表征模型，再通过 visual memory（KNN 检索数据库）注入真实世界知识，在分类和分割任务上逼近真实数据训练的性能，同时实现对所有真实数据的完全可控（隐私保护、高效遗忘）。 现代视觉模型通过梯度下降将图像"消…
tags:
  - "ICML2025"
  - "语义分割"
  - "procedural data"
  - "visual memory"
  - "KNN classification"
  - "data privacy"
  - "差分隐私"
  - "自监督学习"
---

# Separating Knowledge and Perception with Procedural Data

**会议**: ICML2025  
**arXiv**: [2508.11697](https://arxiv.org/abs/2508.11697)  
**代码**: 待确认  
**领域**: 图像分割  
**关键词**: procedural data, visual memory, KNN classification, 语义分割, data privacy, 差分隐私, 自监督学习  
**作者**: Adrián Rodríguez-Muñoz, Manel Baradad, Phillip Isola, Antonio Torralba (MIT)

## 一句话总结

仅用程序化生成数据（非真实图像）训练视觉表征模型，再通过 visual memory（KNN 检索数据库）注入真实世界知识，在分类和分割任务上逼近真实数据训练的性能，同时实现对所有真实数据的完全可控（隐私保护、高效遗忘）。

## 研究背景与动机

现代视觉模型通过梯度下降将图像"消化"进权重中，带来三大问题：

**隐私与偏见**：权重以黑盒方式存储知识，难以追踪或删除特定数据（如人脸、敏感医疗图像）；

**数据遗忘困难**：法律要求删除某些数据时，需要重新训练整个模型，代价极高；

**知识编辑不灵活**：添加/删除/更新知识需 fine-tune 或重训。

先前工作提出了 **visual memory** 的思路：用 KNN 检索替代参数化分类器，使知识以数据库形式存储，便于增删。但问题在于——特征提取器本身仍在真实数据上训练，知识与感知并未真正分离。

本文的关键洞察：**用程序化数据（procedural data）训练特征提取器**，使模型完全不接触真实图像。程序化数据是通过简单代码（OpenGL shader）生成的非真实图像，隐私风险极低。这样所有真实数据都仅存在于 visual memory 中，实现完全的知识-感知分离。

## 方法详解

### 整体框架

系统分为三个阶段：
1. **训练阶段**：用程序化数据 + DINO 自监督目标训练 ViT-S 特征提取器；
2. **知识注入**：构建真实图像嵌入的 visual memory 数据库（无需额外训练）；
3. **推理阶段**：对查询图像提取特征，在 memory 中做 KNN 检索，输出多数标签。

### 核心创新：Shaders KML / Shaders KML Mixup

先前最佳程序化数据集为 Shaders Mixup（Baradad et al., 2022），其在像素空间用常量 mask 做 Mixup 缓解 short-cut 问题。本文提出两项改进：

**Shaders KML（K-Means Leaves）**：
1. 采样三个 shader 图像 $s_1, s_2, s_3$；
2. 对 $s_1$ 在 RGB 空间做 KMeans 聚类，提取数据驱动的 mixing mask $m$；
3. 用 $m$ 混合 $s_2$ 和 $s_3$，得到最终样本。

关键区别在于 mask 从数据本身提取而非固定常量，大幅提升了数据集多样性——此前研究已表明多样性是程序化数据性能的最大驱动因素。

**Shaders KML Mixup**：在 Shaders KML 基础上再叠加标准 Mixup，进一步抑制 short-cut solution，取得新的 SOTA。

### 训练细节

- 骨干网络：ViT-S（Vision Transformer Small）
- 训练目标：DINO 的 local-to-global similarity（学习局部与全局视图的一致表征）
- 在真实数据上，DINO 目标教模型学习同一物体不同部分的相似表征；在程序化数据上，则学习抽象形状/纹理的部分相似性

### 差分隐私分析

定义 $\epsilon$-差分隐私：对于算法 $\mathcal{A}$，若对仅在样本 $x$ 上不同的数据集 $D_1, D_2$，满足：

$$\Pr[\mathcal{A}(D_1) \in S] \leq e^{\epsilon} \Pr[\mathcal{A}(D_2) \in S]$$

对于确定性算法（如 KNN），简化为：有/无样本 $x$ 时所有测试集预测相同。程序化嵌入 + visual memory 的架构下，只需比较有/无某真实图像时 KNN 预测是否变化，无需重训模型。

## 实验关键数据

### 视觉相似性（NIGHTS 数据集）

最佳程序化模型 Shaders KML 与人类判断对齐度达 **82.4%**，仅比 Places 模型低 **0.9%**。PSNR/SSIM 等白盒指标接近随机。

### KNN 分类

| 数据类型 | 数据集 | Flowers102 | CUB200 | Food101 | ImageNet-1K |
|---------|---------|-----------|--------|---------|-------------|
| 真实 | Places | 59.51 | 19.09 | 47.78 | 47.30 |
| 程序化 | S. KML Mixup | **75.20** | **27.08** | **48.70** | 37.88 |
| 白盒 | Random init. | 11.18 | 1.93 | 5.32 | 1.84 |

**关键发现**：在细粒度分类上，程序化模型反超 Places 模型 **+15%（Flowers）、+8%（CUB）、+1%（Food）**。原因是 Places 模型的语义容量被场景知识占用，而程序化模型学到的是领域无关的视觉技能。ImageNet-1K 上差距约 10%。

### 零样本语义分割（COCO）

| 数据类型 | 数据集 | $R^2$ |
|---------|---------|-------|
| 真实 | ImageNet | 63.7 |
| 真实 | Places | 62.1 |
| 程序化 | S. KML | **55.9** |
| 程序化 | S. KML Mixup | 53.7 |
| 白盒 | Random init. | 36.7 |

最佳程序化模型 $R^2$ 在 COCO 上与真实数据模型差距约 **10%**，远高于随机初始化。

### 医疗数据（MedMNIST）

程序化模型在 10 个 MedMNIST 数据集中的 **7 个**上匹配或超越了原论文中标准训练 ResNet 的最佳结果。这对医疗隐私场景极具价值。

### 模型规模扩展

ViT 从 S 扩展到更大规模时，程序化模型不会过拟合——更大容量带来更高性能，说明泛化良好。

### 隐私分析

在 ImageNet 上仅 **<0.6%** 的训练样本是非隐私的（删除后会改变至少一个测试预测），且准确率与非隐私样本比例呈线性关系。

## 亮点与洞察

1. **反直觉的强结果**：完全不看真实图像的模型，在细粒度分类上竟然超过了在真实场景数据上训练的模型，说明领域无关的视觉感知技能本身极具价值。
2. **优雅的隐私解决方案**：将所有真实数据限制在可增删的数据库中，隐私保护、数据遗忘变为 O(1) 操作。
3. **Gestalt 分析**：发现无论真实还是程序化训练的视觉模型都不具备格式塔感知能力，揭示了当前视觉模型与人类感知的本质差距。
4. **实用的存储/计算/精度权衡分析**：KNN 方法训练成本仅为线性分类器的 1/64，内存嵌入存储整个 ImageNet 仅需 ~2GiB。

## 局限与展望

1. **部分-整体问题**：程序化模型从未见过真实物体，无法将同一物体的视觉不同部分（如自行车的轮毂与辐条）关联为整体，导致 KNN 语义分割中出现 spurious matching。这是性能差距的核心原因。
2. **仅验证了 ViT-S**：虽然展示了规模扩展不过拟合的趋势，但缺少 ViT-B/L 的完整实验。
3. **推理延迟**：KNN 推理在大规模 memory 下比参数化分类器慢（原始实现约 2 倍），虽然 faiss 等可优化至 <0.03ms/query，但工程复杂度增加。
4. **分割任务局限**：zero-shot PCA 分割尚可，但 KNN 语义分割因"过度局部"的表征而表现较差，未提出解决方案。
5. **程序化数据的天花板**：当前最佳程序化数据仍来自 OpenGL shader，生成过程的多样性上限不清晰。

## 相关工作与启发

- **Visual Memory**（Geirhos et al., 2024; Nakata et al., 2022）：KNN 检索替代分类器，本文在此基础上引入程序化嵌入实现完全分离
- **Procedural Data**（Baradad et al., 2021/2022; Kataoka et al., 2020）：从噪声/分形/shader 学表征，本文推进到分割任务并提出新的 KML 数据增强
- **差分隐私 SGD**（Abadi et al., 2016）：传统方法在训练中加噪声，本文通过架构设计绕过了这一需求

## 评分

- 新颖性: ⭐⭐⭐⭐ — 程序化数据 + visual memory 的组合思路清晰优雅，KML mask 生成也有技术贡献
- 实验充分度: ⭐⭐⭐⭐ — 覆盖相似性/分类/分割/医疗/隐私/Gestalt 等多维度，定量定性分析充分
- 写作质量: ⭐⭐⭐⭐⭐ — 结构清晰，分析深入，trade-off 讨论务实
- 价值: ⭐⭐⭐⭐ — 对隐私敏感场景（医疗、人脸）有实际意义，但部分-整体问题限制了通用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Exploring CLIP's Dense Knowledge for Weakly Supervised Semantic Segmentation](../../CVPR2025/segmentation/exploring_clips_dense_knowledge_for_weakly_supervised_semantic_segmentation.md)
- [\[CVPR 2025\] DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception](../../CVPR2025/segmentation/declip_decoupled_learning_for_open-vocabulary_dense_perception.md)
- [\[ICML 2025\] Using Multiple Input Modalities Can Improve Data-Efficiency and O.O.D. Generalization for ML with Satellite Imagery](using_multiple_input_modalities_can_improve_data-efficiency_and_ood_generalizati.md)
- [\[ICCV 2025\] Advancing Visual Large Language Model for Multi-granular Versatile Perception](../../ICCV2025/segmentation/advancing_visual_large_language_model_for_multi-granular_versatile_perception.md)
- [\[CVPR 2026\] GKD: Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](../../CVPR2026/segmentation/gkd_generalizable_knowledge_distillation_vfm.md)

</div>

<!-- RELATED:END -->
