# UNIStainNet: Foundation-Model-Guided Virtual Staining of H&E to IHC

**会议**: arXiv 2026  
**arXiv**: [2603.12716](https://arxiv.org/abs/2603.12716)  
**作者**: Jillur Rahman Saurav, Thuong Le Hoai Pham, Pritam Mukherjee, Paul Yi, Brent A. Orr
**代码**: [https://github.com/facevoid/UNIStainNet](https://github.com/facevoid/UNIStainNet)  
**领域**: 医学图像  
**关键词**: unistainnet, foundation-model-guided, virtual, staining, h&e  

## 一句话总结
苏木精和伊红 (H&E) 图像的虚拟免疫组织化学 (IHC) 染色可以直接从常规切片中提供初步的分子洞察，从而加快诊断速度，从而减少组织有限时重复切片的需要。
## 背景与动机
Virtual immunohistochemistry (IHC) staining from hematoxylin and eosin (H&E) images can accelerate diagnostics by providing preliminary molecular insight directly from routine sections, reducing the need for repeat sectioning when tissue is limited.. Existing methods improve realism through contrastive objectives, prototype matching, or domain alignment, yet the generator itself receives no direct guidance from pathology foundation models.

## 核心问题
苏木精和伊红 (H&E) 图像的虚拟免疫组织化学 (IHC) 染色可以直接从常规切片中提供初步的分子洞察，从而加快诊断速度，从而减少组织有限时重复切片的需要。
## 方法详解

### 整体框架
- We present UNIStainNet, a SPADE-UNet conditioned on dense spatial tokens from a frozen pathology foundation model (UNI), providing tissue-level semantic guidance for stain translation.

### 关键设计
待深读后补充。

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 现有方法通过对比目标、原型匹配或域对齐来提高真实性，但生成器本身没有收到来自病理学基础模型的直接指导。
- 在 MIST 上，UNIStainNet 通过单个统一模型实现了所有四种染色剂（HER2、Ki67、ER、PR）的最先进的分布指标，其中先前的方法通常训练单独的每个染色剂模型。
- 在 BCI 上，它也实现了最佳的分布指标。
## 局限性 / 可改进方向
- 待深读后补充

## 与相关工作的对比
待深读后补充。

## 与我的研究方向的关联
- 待分析

## 评分
- 新颖性: ⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐
- 对我的价值: ⭐⭐⭐⭐
