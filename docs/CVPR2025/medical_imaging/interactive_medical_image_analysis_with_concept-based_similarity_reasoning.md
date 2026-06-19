---
title: >-
  [论文解读] Interactive Medical Image Analysis with Concept-based Similarity Reasoning
description: >-
  [CVPR 2025][医学图像][可解释医学影像] 本文提出 CSR（Concept-based Similarity Reasoning）网络，通过学习概念原型在图像局部区域的相似性来进行分类推理，同时支持医生在训练和测试时从空间级和概念级两个维度进行交互式干预，在三个医学数据集上以高达 4.5% 的 F1 提升超越了现有可解释方法。
tags:
  - "CVPR 2025"
  - "医学图像"
  - "可解释医学影像"
  - "概念瓶颈模型"
  - "原型学习"
  - "空间交互"
  - "相似性推理"
---

# Interactive Medical Image Analysis with Concept-based Similarity Reasoning

**会议**: CVPR 2025  
**arXiv**: [2503.06873](https://arxiv.org/abs/2503.06873)  
**代码**: [https://github.com/tadeephuy/InteractCSR](https://github.com/tadeephuy/InteractCSR)  
**领域**: 医学图像 / 可解释AI  
**关键词**: 可解释医学影像, 概念瓶颈模型, 原型学习, 空间交互, 相似性推理

## 一句话总结
本文提出 CSR（Concept-based Similarity Reasoning）网络，通过学习概念原型在图像局部区域的相似性来进行分类推理，同时支持医生在训练和测试时从空间级和概念级两个维度进行交互式干预，在三个医学数据集上以高达 4.5% 的 F1 提升超越了现有可解释方法。

## 研究背景与动机

1. **领域现状**：可解释医学图像分析主要有两条路线——概念瓶颈模型（CBM）预测可解释概念再用概念做分类，原型方法（ProtoPNet 等）学习训练图像的 patch 原型并用相似性分数分类。
2. **现有痛点**：CBM 仅提供图像级概念解释，无法定位概念激活的具体区域；原型方法提供了 patch 级解释，但需要事后分析来关联原型与语义概念——在医学影像中视觉差异微妙，这种事后分析尤其困难。
3. **核心矛盾**：概念可解释性（知道是什么概念）和空间定位性（知道在哪里）目前是分离的——没有一种方法能同时提供"patch 级 + 概念可解释"的解释。
4. **本文目标** (a) 如何实现 patch 级的概念内在可解释性？(b) 如何让医生能在空间维度直接与模型交互？(c) 如何在不牺牲解释性的前提下保持诊断精度？
5. **切入角度**：受放射科医生使用图谱参考的启发——医生诊断时会将疑似区域与已知典型案例做比较。
6. **核心 idea**：学习概念原型，通过计算原型与输入图像各 patch 的余弦相似度生成 2D 相似性图作为解释，支持医生通过画框（空间交互）和拒绝概念（概念交互）来校正模型判断。

## 方法详解

### 整体框架
CSR 包含三个组件：(1) Concept 模型——提取概念特征并生成概念激活图；(2) 特征投影器 P——通过对比学习增强概念特征的紧凑性和泛化性，学习概念原型；(3) 任务头 H——从概念相似性分数向量预测目标类别。推理时，对每个概念原型计算其与输入图像各 patch 的余弦相似性图，取最大值作为该概念的相似性分数，汇聚所有分数构成向量输入分类头。

### 关键设计

1. **概念原型学习 + 多原型对比损失**:

    - 功能：学习紧凑、泛化的概念原型，使其能在新图像上准确定位对应概念
    - 核心思路：首先训练 Concept 模型做多标签分类，生成概念激活图 $\text{cam}^k$；然后通过空间 softmax 加权求和获取局部概念向量 $v^k = \sum_{H,W} \text{softmax}_{h,w}(\text{cam}^k) \cdot \mathbf{f}$；再用投影器 P 映射到紧凑空间 $v' = P(v)$，通过多原型对比损失学习每个概念的 M 个原型 $\{p^{k_m}\}$——将概念向量拉向最近的同概念原型，推远其他概念原型。概率赋值使用 $q_m = \text{softmax}_m(\gamma \langle p^{\tilde{k}_m}, v'^{\tilde{k}}_i \rangle)$，总相似度为加权和
    - 设计动机：直接使用局部概念向量 $v^k$ 在新图像上泛化差（如图4所示，起搏器概念的 $v$ 无法在新图像上定位）；对比学习+多原型策略解决了概念特征的跨样本泛化问题

2. **空间级交互机制 (Spatial-level Interaction)**:

    - 功能：允许医生通过画正/负边框引导模型关注或忽略特定区域
    - 核心思路：医生画正框 $\text{bb}^+$ 和负框 $\text{bb}^-$ 生成重要性图 $A(h,w)$：正框内为 1，负框内为 0，其余为 $\alpha \in [0,1)$。将重要性图与所有概念相似性图逐元素相乘 $[\hat{S}] = A \odot [S]$，重新计算相似性分数和预测。由于取最大值操作，放大重要区域会增加其被选中的概率，置零伪相关区域则消除其影响
    - 设计动机：深度学习模型常捕获"Clever-Hans"式的伪相关，医生可以直接告诉模型"看哪里/不看哪里"而无需指定"看什么"，交互方式自然直观

3. **概念级交互 + 训练时交互**:

    - 功能：允许医生拒绝错误概念和清除低质量原型
    - 核心思路：**测试时概念交互**——医生审查概念相似性分数后，如果发现某概念不存在，可拒绝该概念，模型将对应的所有 $s^{k_m}$ 置零重新预测。**训练时交互**——医生检查概念原型图谱 $\{\mathcal{I}(p^{k_m})\}$，剔除通过伪相关学得的低质量原型，从源头消除 Clever-Hans 效应
    - 设计动机：CBM 的概念交互已被证明能显著提升性能，本文将其扩展到更灵活的多层次交互，结合空间交互和概念交互实现全面的 doctor-in-the-loop

### 损失函数 / 训练策略
分阶段训练：(1) 用 BCE 训练 Concept 模型做多标签概念分类；(2) 用多原型对比损失 $\ell_{\text{con-m}}$ 训练投影器 P 和概念原型，含 margin $\delta$ 扩展决策边界；(3) 用 CE 训练任务头 H 从相似性分数预测目标。

## 实验关键数据

### 主实验

| 数据集 | 方法 | F1 ↑ | 原型数 ↓ | 解释数 ↓ |
|--------|------|------|---------|---------|
| TBX11K | CBM (joint) | 88.6 | - | 14 |
| TBX11K | ProtoPNet | 94.1 | 3000 | 3000 |
| TBX11K | PIP-Net | 94.0 | 768 | 158 |
| TBX11K | **CSR** | **94.4** | 1400 | **14** |
| VinDr-CXR | CBM (joint) | 50.1 | - | 14 |
| VinDr-CXR | PIP-Net | 45.1 | 768 | 9 |
| VinDr-CXR | **CSR** | **54.6** | 1400 | **14** |
| ISIC | PIP-Net | 69.9 | 768 | 90 |
| ISIC | **CSR** | **71.5** | 400 | **4** |

### 消融实验 — Pointing Game 定位准确率

| 方法 | PG Hit Rate ↑ |
|------|---------------|
| ProtoPNet | 8.8% |
| ProtoTree | 7.8% |
| PIP-Net | 19.5% |
| CBM | 55.1% |
| CSR | 60.9% |
| **CSR (refined)** | **79.5%** |

### 关键发现
- CSR 在三个数据集上均超越所有可解释基线，在 VinDr-CXR 上相比 CBM 提升 4.5% F1
- 解释数量极为精简：CSR 每次预测仅需 14 个（概念数）解释，而 ProtoPNet 需要 3000 个
- Pointing Game 准确率达 60.9%（训练后精炼达 79.5%），远超 ProtoPNet 的 8.8% 和 PIP-Net 的 19.5%，证明概念定位的准确性
- 训练时交互（医生精炼原型图谱）虽然略降 F1（94.4→94.0），但将定位准确率从 60.9% 提升到 79.5%
- 投影器 P 的对比学习显著改善了概念特征空间的紧凑性和跨样本泛化性

## 亮点与洞察
- **概念原型=可解释的局部比较器**这个设计非常巧妙：不需要事后分析来解释原型含义，每个原型天然绑定到一个语义概念，同时提供 patch 级定位。这种"设计即可解释"的思路比事后解释更可靠。
- **空间交互的设计**极其实用：医生无需专业 ML 知识，只需在图像上"画框"告诉模型看哪/不看哪，模型自动调整预测。交互方式符合放射科医生的工作习惯。
- **概念交互和空间交互的组合使用**能处理更复杂的情况：当正框区域被错误关联到不存在的概念时，可以同时拒绝该概念并保留正框，引导模型关注正确概念。

## 局限与展望
- 概念需要预定义且需要概念级标注，标注成本较高
- 当前仅使用视觉特征，未探索多模态（如CLIP、LLM）概念定义
- 多原型数量 M 的选择缺乏自动化策略
- 空间交互中的 $\alpha$ 参数需要手动设定，不同概念可能需要不同的 $\alpha$
- 可探索将 CSR 的概念原型与 foundation model 结合，减少对概念标注的依赖

## 相关工作与启发
- **vs CBM**: CBM 仅提供图像级概念解释，CSR 提供 patch 级定位解释，且在 VinDr-CXR 上 F1 绝对提升 4.5%
- **vs ProtoPNet/PIP-Net**: 原型方法需要事后分析关联语义，CSR 的概念原型天然具有语义可解释性；ProtoPNet 的 Pointing Game 仅 8.8% vs CSR 的 60.9%
- **vs PHCBM**: PHCBM 用 CLIP 迁移概念，但医学影像与自然图像差异大，直接迁移效果有限；CSR 从数据学习概念表示更适合医学领域

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 概念瓶颈+原型学习+空间交互的融合非常新颖，填补了可解释医学 AI 中的重要空白
- 实验充分度: ⭐⭐⭐⭐ 三个数据集对比充分，Pointing Game 评估定位准确性，但缺少更大规模数据集验证
- 写作质量: ⭐⭐⭐⭐⭐ 图1的交互示例非常直观，方法阐述逻辑清晰
- 价值: ⭐⭐⭐⭐⭐ 在临床 AI 的可信度和可用性方面有重要实践意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline](interactive_medical_image_segmentation_a_benchmark_dataset_and_baseline.md)
- [\[CVPR 2025\] Multi-modal Vision Pre-training for Medical Image Analysis (BrainMVP)](multi-modal_vision_pre-training_for_medical_image_analysis.md)
- [\[CVPR 2025\] Boltzmann Attention Sampling for Image Analysis with Small Objects](boltzmann_attention_sampling_for_image_analysis_with_small_objects.md)
- [\[NeurIPS 2025\] DermaCon-IN: A Multi-concept Annotated Dermatological Image Dataset of Indian Skin Disorders](../../NeurIPS2025/medical_imaging/dermacon-in_a_multi-concept_annotated_dermatological_image_dataset_of_indian_ski.md)
- [\[ICCV 2025\] SIC: Similarity-Based Interpretable Image Classification with Neural Networks](../../ICCV2025/medical_imaging/sic_similarity-based_interpretable_image_classification_with_neural_networks.md)

</div>

<!-- RELATED:END -->
