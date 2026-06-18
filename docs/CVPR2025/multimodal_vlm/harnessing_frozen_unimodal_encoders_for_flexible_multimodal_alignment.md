---
title: >-
  [论文解读] Harnessing Frozen Unimodal Encoders for Flexible Multimodal Alignment
description: >-
  [CVPR 2025][多语言/翻译][冻结编码器] 提出一种新的视觉-语言对齐框架：冻结预训练好的单模态视觉编码器（DINOv2）和语言编码器（All-Roberta-Large），仅训练轻量MLP投影层实现多模态对齐，以20倍数据缩减和65倍计算缩减达到了CLIP级别甚至超越的性能。 领域现状： CLIP等对比多模态模型…
tags:
  - "CVPR 2025"
  - "多语言/翻译"
  - "冻结编码器"
  - "投影层对齐"
  - "CKA"
  - "概念平衡数据集"
  - "多模态学习"
---

# Harnessing Frozen Unimodal Encoders for Flexible Multimodal Alignment

**会议**: CVPR 2025  
**arXiv**: [2409.19425](https://arxiv.org/abs/2409.19425)  
**代码**: [GitHub](https://github.com/mayug/freeze-align)  
**领域**: 多语言翻译  
**关键词**: 冻结编码器、投影层对齐、CKA、概念平衡数据集、多模态学习

## 一句话总结

提出一种新的视觉-语言对齐框架：冻结预训练好的单模态视觉编码器（DINOv2）和语言编码器（All-Roberta-Large），仅训练轻量MLP投影层实现多模态对齐，以20倍数据缩减和65倍计算缩减达到了CLIP级别甚至超越的性能。

## 研究背景与动机

**领域现状**：
CLIP等对比多模态模型通过在4亿图文对上从头训练视觉和文本编码器来实现模态对齐，已成为视觉-语言应用的标准骨干模型，展现出强大的零样本能力。

**现有痛点**：
1. CLIP式训练需要巨大的计算资源（约2万GPU小时A100）和数据（4亿图文对）
2. CLIP的视觉编码器因全局训练目标，在像素级任务（如分割）上表现不如单模态视觉模型（如DINOv2）
3. CLIP的文本编码器仅支持英语、最多77 token，在多语言和长文本场景下能力受限
4. 现有高效方法如LiT仍需大规模计算，LiLT等小规模方法性能不足

**核心矛盾**：
训练强大的多模态模型需要巨大计算和数据成本，但已经存在许多领域专精的单模态编码器（如DINOv2的视觉定位能力、多语言文本编码器），如何低成本地复用这些能力？

**本文目标**
如何以极低的成本（仅训练投影层）将已有的强大单模态编码器对齐，获得不亚于CLIP的多模态模型。

**切入角度**：
基于最近的发现——训练良好的单模态视觉和语言编码器之间存在高度的语义相似性（CKA分数高），这暗示着简单的投影变换就可能实现对齐。

**核心 idea**：
选择CKA语义相似度最高的编码器对，构建概念丰富的数据集，训练简单投影层即可实现CLIP级别的多模态对齐。

## 方法详解

### 整体框架

框架包含三个步骤：(1) 编码器对选择——使用CKA指标在COCO验证集上评估不同视觉-语言编码器对的语义相似度，选出CKA最高的对（DINOv2-Large + All-Roberta-Large-v1, CKA=0.69）；(2) 数据集构建——通过概念平衡采样从LAION-400M等数据源构建概念密集的20M训练集；(3) 投影层训练——仅训练11.5M参数的投影层，使用InfoNCE对比损失。

### 关键设计

1. **基于CKA的编码器对选择**:
    - 功能：高效筛选最适合对齐的视觉-语言编码器组合
    - 核心思路：在5000对COCO图文上计算不同编码器对的CKA分数，发现CKA与对齐后的下游检索性能呈强正相关（Figure 1, Figure 4），因此CKA可作为高效的编码器选择指标
    - 设计动机：避免逐对训练所有编码器组合的高昂成本，CKA提供了一个无需训练的先验

2. **概念平衡数据集构建**:
    - 功能：构建支持高效投影层训练的高质量数据集
    - 核心思路：
        - 从ImageNet等数据集收集约3000个唯一概念
        - 为每个概念构建few-shot图像原型（使用CLIP ViT-Large编码）
        - 从LAION-400M中按概念平衡采样，每概念2000样本，优先采稀有概念
        - 结合CC3M、CC12M、SBU等高语义对齐数据集，组成20M MIX-CLASS-Collected
    - 设计动机：高概念覆盖确保密集覆盖单模态空间各区域（利于分类），高语义对齐（利于检索），两者缺一不可

3. **轻量Token投影器架构**:
    - 功能：用最少参数实现模态对齐
    - 核心思路：
        - 视觉端：对local token和CLS token分别使用Token Projector（线性+非线性分支的残差结构），local token共享权重，CLS token单独权重
        - 文本端：Token Projector处理token，再用2层MLP作为全局投影
        - 所有适配后的local token平均后加到适配后的CLS token形成最终全局embedding
    - 设计动机：DINOv2的DINO目标作用于CLS token、iBOT目标作用于patch token，需要分别处理；文本编码器的嵌入空间与视觉距离较远，需要额外的全局投影

### 损失函数 / 训练策略

- 标准InfoNCE对比损失
- 仅训练投影层（11.5M参数），冻结DINOv2-Large（300M）和ARL文本编码器（355M）
- 8×A100 GPU，约50小时训练（相比CLIP的21845 GPU小时减少65倍）
- 20M训练数据（相比CLIP的400M减少20倍）

## 实验关键数据

### 主实验

**零样本分类迁移**

| 模型 | 数据量 | ImageNet | ImageNetv2 | Caltech | Pets | Cars | 平均 |
|------|--------|----------|------------|---------|------|------|------|
| OpenAI CLIP ViT-L | 400M | 75.3 | 69.8 | 92.6 | 93.5 | 77.3 | - |
| LAION CLIP ViT-L | 400M | 72.7 | 65.4 | 92.5 | 91.5 | 89.6 | - |
| DINOv2-ARL (Ours) | **20M** | **76.3** | **69.2** | **92.8** | 92.1 | 73.9 | - |

**图文检索（Flickr30K / COCO）**

| 模型 | Flickr I2T | Flickr T2I | COCO I2T | COCO T2I |
|------|-----------|-----------|---------|---------|
| OpenAI CLIP ViT-L | 85.2 | 64.9 | 56.3 | 36.5 |
| LAION CLIP ViT-L | 87.6 | 70.2 | 59.7 | 43.0 |
| DINOv2-ARL (Ours) | **87.5** | **74.1** | **60.1** | **45.1** |

### 消融实验

**投影器架构消融（ImageNet零样本准确率）**

| 视觉局部 | 视觉CLS | 文本局部 | 文本全局 | ImageNet |
|----------|---------|---------|---------|----------|
| token | identity | identity | identity | 68.84 |
| token | identity | token | mlp | 72.15 |
| identity | token | token | mlp | 75.53 |
| token | token | token | mlp | **76.12** |

**数据集消融**

| 数据源 | 数据量 | ImageNet | Flickr I2T | Flickr T2I |
|--------|--------|----------|-----------|-----------|
| LAION-CLASS-Collected | 6M | 76.12 | 52.70 | 42.48 |
| CC3M+CC12M+SBU | 14M | 54.17 | 85.30 | 72.44 |
| Both | 20M | 75.04 | 81.32 | 71.38 |
| Both + longer | 20M | **76.30** | **87.54** | **74.17** |

### 关键发现

1. **投影层对齐可以匹配CLIP**：仅训练1%的参数（11.5M/670M），ImageNet零样本准确率76.3%，超越OpenAI CLIP（75.3%）和LAION CLIP（72.7%）
2. **CKA是有效的编码器选择指标**：CKA与最终检索性能呈明显正相关，DINOv2+ARL的CKA=0.69是非CLIP文本编码器中最高的
3. **概念覆盖和语义对齐缺一不可**：仅高覆盖数据分类好但检索差，仅高对齐数据检索好但分类差，混合后两者兼得
4. **单模态能力在对齐后保留**：
    - DINOv2的定位能力保留：零样本分割Pascal VOC IoU 31.37%（CLIP仅23.46%）
    - MpNet的多语言能力保留：仅用英文训练就在多语言检索上超越专门训练的多语言模型
    - ARL的长文本能力保留：超过77 token后检索性能持续提升至200-300 token

## 亮点与洞察

1. **范式革新**：证明了CLIP级别的多模态对齐不一定需要从头训练编码器，冻结已有强单模态编码器+训练投影层是一条可行且高效的路线
2. **计算民主化**：65倍计算减少和20倍数据减少使得多模态模型开发对学术界更加可及
3. **灵活组合的前景**：可以根据需求灵活选择不同的单模态编码器——多语言文本编码器→多语言VLM，长上下文编码器→长文本VLM，3D编码器→3D-语言模型
4. **CKA作为配对指标**：提供了一个简单有效的先验来评估哪些编码器对容易对齐，避免盲目尝试
5. **单模态特征的优越性**：DINOv2这类纯视觉训练的编码器在定位等视觉核心任务上优于CLIP的视觉编码器，通过冻结保留了这一优势

## 局限与展望

1. 框架假设已有强大的单模态编码器可用，如果没有预训练好的编码器则无法使用
2. CKA选择尚未考虑不同任务的需求差异（CKA最高不一定在所有下游任务上最优）
3. 投影层的容量有限，可能无法弥补CKA较低的编码器对之间的差距
4. 仅在20M数据上训练，更大规模能否进一步提升尚未探索
5. 未与LLM集成（如LLaVA范式），作为视觉特征提取器的效果有待验证

## 相关工作与启发

- **CLIP/ALIGN**: 从头训练对比多模态模型的代表，是本文的主要对比对象
- **LiT（Locked Image Tuning）**: 冻结视觉编码器仅训练文本编码器，计算仍然较大
- **DINOv2**: 无标签自监督视觉编码器，兼具强大的全局和局部特征
- **Platonic Representation Hypothesis**: 训练良好的不同模态模型趋向于收敛到共享的语义结构
- 启发：模态之间的语义鸿沟可能比我们想象的小，简单的投影变换就足以桥接

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Modular Sentence Encoders: Separating Language Specialization from Cross-Lingual Alignment](../../ACL2025/multilingual_mt/modular_sentence_encoders.md)
- [\[ACL 2025\] COSMMIC: Comment-Sensitive Multimodal Multilingual Indian Corpus](../../ACL2025/multilingual_mt/cosmmic_commentsensitive_multimodal_multilingual_indian_corpus.md)
- [\[ACL 2025\] mOSCAR: A Large-scale Multilingual and Multimodal Document-level Corpus](../../ACL2025/multilingual_mt/moscar_a_large-scale_multilingual_and_multimodal_document-level_corpus.md)
- [\[ACL 2025\] Implicit Cross-Lingual Rewarding for Efficient Multilingual Preference Alignment](../../ACL2025/multilingual_mt/implicit_cross-lingual_rewarding_for_efficient_multilingual_preference_alignment.md)
- [\[ACL 2025\] Multi-perspective Alignment for Increasing Naturalness in Neural Machine Translation](../../ACL2025/multilingual_mt/multi-perspective_alignment_for_increasing_naturalness_in_neural_machine_transla.md)

</div>

<!-- RELATED:END -->
