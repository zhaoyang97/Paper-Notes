---
title: >-
  [论文解读] SimMLM: A Simple Framework for Multi-modal Learning with Missing Modality
description: >-
  [ICCV 2025][医学图像][多模态] 提出 SimMLM，一个简洁高效的多模态缺失学习框架，由动态模态专家混合架构（DMoME）和 More vs. Fewer（MoFe）排序损失组成，在脑肿瘤分割和多模态分类任务上以更少参数和计算量全面超越 SOTA，同时提供模态重要性可解释性。 多模态学习虽然能利用互补信息提升性…
tags:
  - "ICCV 2025"
  - "医学图像"
  - "多模态"
  - "missing modality"
  - "mixture of experts"
  - "ranking loss"
  - "图像分割"
---

# SimMLM: A Simple Framework for Multi-modal Learning with Missing Modality

**会议**: ICCV 2025  
**arXiv**: [2507.19264](https://arxiv.org/abs/2507.19264)  
**代码**: [https://github.com/LezJ/SimMLM](https://github.com/LezJ/SimMLM)  
**领域**: 医学图像  
**关键词**: multi-modal learning, missing modality, mixture of experts, ranking loss, brain tumor segmentation

## 一句话总结

提出 SimMLM，一个简洁高效的多模态缺失学习框架，由动态模态专家混合架构（DMoME）和 More vs. Fewer（MoFe）排序损失组成，在脑肿瘤分割和多模态分类任务上以更少参数和计算量全面超越 SOTA，同时提供模态重要性可解释性。

## 研究背景与动机

多模态学习虽然能利用互补信息提升性能，但实际部署中经常遇到**测试时模态缺失**问题（硬件故障、环境限制、数据采集约束等）。现有解决方案存在以下不足：

**数据填补方法**（用生成模型合成缺失模态）：合成数据质量难以保证，存在幻觉、对抗脆弱性等问题，且计算开销大

**复杂架构方法**（mmFormer、ShaSpec 等）：需要特殊的网络设计来对齐/共享跨模态特征，参数量大、灵活性差，且共享编码器要求所有模态维度一致

**现有 MoE 方法**（MoMKE）：每个模态需通过所有专家，计算复杂度随模态数量平方增长

SimMLM 的核心理念是**简单即有效**：每个模态只需对应一个专家网络，通过动态门控网络自适应调整贡献权重，再用 MoFe 排序损失确保"模态越多性能越好"的直觉性质。

## 方法详解

### 整体框架

SimMLM 由两个组件组成：
1. DMoME 架构：模态特定专家网络 + 门控网络，处理完整/部分模态输入
2. MoFe 排序损失：在训练时随机丢弃模态，约束"更多模态→更好性能"的排序关系

### 关键设计

1. **动态模态专家混合（DMoME）**：

    - 每个模态 $m$ 对应一个专家网络 $E^m(\mathbf{x}^m; \theta_m)$，产生任务输出 $\mathbf{o}^m$（如分类 logits）
    - 门控网络 $G(\cdot; \phi)$ 以所有可用模态为输入，产生门控值 $\{g^m\}_{m=1}^M$，经 softmax 得到权重 $w^m$
    - 缺失模态的输入置零，对应 $g_m = -\infty$（softmax 后权重为 0），自然排除缺失模态的贡献
    - 最终输出为加权组合：$\mathbf{o} = \sum_{m=1}^M w^m E^m(\mathbf{x}^m; \theta_m)$
    - **关键优势**：每个模态只过自己的专家网络（vs. MoMKE 的每个过所有专家），FLOPS 线性而非二次增长；对专家网络的架构无约束，可异构

2. **More vs. Fewer (MoFe) 排序损失**：

    - 核心直觉：有更多模态输入时，模型性能应不低于较少模态的情况
    - 每次训练迭代，从完整模态 $\mathbf{x}^{full}$ 中采样两组：$\mathbf{x}^+ \supseteq \mathbf{x}^-$
    - MoFe 损失定义为：
    $\ell_{\text{MoFe}}(\mathbf{o}^+, \mathbf{o}^-, \mathbf{y}) = \max(0, \ell_{\text{task}}(\mathbf{o}^+, \mathbf{y}) - \ell_{\text{task}}(\mathbf{o}^-, \mathbf{y}))$
    - 即当更多模态的 task loss 反而高于更少模态时产生梯度，鼓励网络消除这种"反直觉"现象
    - **创新点**：在 loss level 而非 confidence level 做排序，直接正则化损失景观的几何结构，且通用于分类、分割、回归等任意任务

3. **两阶段训练**：

    - **Stage 1（独立学习）**：各模态专家独立训练，避免某个模态主导学习过程
    - **Stage 2（协作学习）**：联合训练专家和门控网络，使用 MoFe 损失，总损失为：
    $\ell_{\text{total}} = \ell_{\text{task}}(\mathbf{o}^+, \mathbf{y}) + \ell_{\text{task}}(\mathbf{o}^-, \mathbf{y}) + \lambda \ell_{\text{MoFe}}$
    - Stage 1 可并行训练，新增模态时可重用已训练专家

### 损失函数 / 训练策略

- MoFe 系数 $\lambda = 0.1$（对不同值均鲁棒）
- BraTS 2018：nnUNet 作为专家网络（4 个模态各一个），门控网络使用轻量 CNN + Linear，Adam (lr=0.01)，task loss = Dice + BCE
- UPMC Food-101：Inception V3（图像）+ BERT（文本）作为专家，MLP 门控，Adam (lr=0.0001)，CE loss
- avMNIST：LeNet-5（图像和音频），Adam (lr=0.001)，CE loss
- 所有实验在单 A100 GPU 上完成

## 实验关键数据

### 主实验 (表格)

BraTS 2018 官方评估集（15种缺失/完整模态配置的平均 Dice score）：

| 方法 | ET ↑ | TC ↑ | WT ↑ | #Params | FLOPS |
|------|------|------|------|---------|-------|
| mmFormer | 59.85 | 72.97 | 82.94 | 106M | 748G |
| ShaSpec | 61.58 | 77.45 | 85.92 | 187.7M | 713G |
| MoMKE | 65.56 | 78.58 | 86.69 | 7.8M | 490G |
| DMoME (w/o MoFe) | 66.05 | 79.14 | 86.71 | 7.8M | 123G |
| **SimMLM (DMoME + MoFe)** | **67.16** | **80.20** | **87.67** | **7.8M** | **123G** |

分类任务结果：

| 数据集 | 模态设置 | ShaSpec | MoMKE | **SimMLM** |
|--------|---------|---------|-------|------------|
| UPMC Food-101 | Image only | 69.22 | 70.46 | **72.20** |
| | Text only | 86.55 | 86.59 | **87.20** |
| | Image + Text | 92.73 | 92.71 | **94.99** |
| avMNIST | Image only | 91.90 | 92.61 | **92.69** |
| | Audio only | 89.28 | 91.16 | **91.61** |
| | Image + Audio | 98.71 | 98.69 | **99.27** |

### 消融实验 (表格)

校准误差对比（BraTS 2018 验证集，15 种配置平均）：

| 方法 | ET ECE↓ | TC ECE↓ | WT ECE↓ | ET SCE↓ | TC SCE↓ | WT SCE↓ |
|------|---------|---------|---------|---------|---------|---------|
| MoMKE | 3.46 | 4.10 | 4.04 | 14.82 | 11.35 | 7.69 |
| DMoME | 3.34 | 3.86 | 3.98 | 13.40 | 11.01 | 7.11 |
| DMoME + $\mathcal{L}_{Conf}$ | 3.17 | 3.86 | 3.61 | 14.80 | 11.58 | 6.81 |
| **DMoME + $\mathcal{L}_{MoFe}$** | **3.15** | **3.75** | **3.55** | **13.21** | **10.83** | **5.65** |

反直觉率（Counterintuitive Rate, CR）对比：

| 方法 | ET CR↓ | TC CR↓ | WT CR↓ |
|------|--------|--------|--------|
| MoMKE | 14.60 | 32.47 | 9.86 |
| DMoME (w/o MoFe) | 10.47 | 30.94 | 3.87 |
| **SimMLM** | **7.20** | **28.19** | **3.75** |

单模态 T1ce（仅一个模态可用）时的提升：

| Tumor | ShaSpec | MoMKE | **SimMLM** |
|-------|---------|-------|------------|
| ET | 73.29 | 72.30 | **78.58** |
| TC | 78.65 | 78.71 | **83.45** |
| WT | 73.82 | 74.00 | **80.02** |

### 关键发现

- SimMLM 以 7.8M 参数和 123G FLOPS 超越了 187.7M/713G 的 ShaSpec 和 106M/748G 的 mmFormer，效率提升一个数量级
- MoFe 损失使 ET 的反直觉率从 14.60% 降至 7.20%（50.68% 降幅），有效维护了"更多模态→更好性能"的单调性
- 门控权重分析显示 DMoME 能自动识别关键模态：ET 分割优先 T1ce，WT 分割优先 FLAIR 和 T2，与临床知识一致
- 当 T1ce 缺失时，DMoME（+MoFe）自动将权重转移到 T1（T1ce 的未增强版），符合临床实践
- MoFe 损失系数 $\lambda$ 在 [0.01, 0.5] 范围内均优于不使用，对超参不敏感
- UPMC Food-101 上的全模态准确率提升最大（92.71→94.99），说明 SimMLM 在噪声环境下的优势更明显

## 亮点与洞察

- **极致的简洁与高效**：不需要特殊架构、不需要共享编码器、不需要生成缺失数据，仅通过"专家+门控+排序损失"实现 SOTA，体现了 Occam's Razor
- **MoFe 排序损失** 的设计理念优雅：直接在 task loss level 施加约束而非 confidence level，使其天然适配任意任务（分割/分类/回归），且起到了正则化损失景观的作用，附带改善了模型校准
- **可解释性** 是一个重要的实际价值：门控权重直接告诉临床医生每个模态的相对重要性，在部分模态缺失时尤为关键
- DMoME 的设计使新增模态只需训练新专家 + 重新协作训练，而非从头训练，具有工程友好性

## 局限与展望

- 两阶段训练虽然有合理性，但增加了训练复杂度
- 门控网络依赖轻量 CNN/MLP，对于高度异构的模态（如结构化临床数据 + 影像）的建模能力待验证
- MoFe 损失假设所有模态都是正向贡献的（"更多模态→更好性能"），但对于噪声模态或冗余模态，这一假设可能不成立
- 仅测试了 2-4 个模态的场景，对更大规模的模态数量（如 10+）的可扩展性未验证
- 训练时缺失模态的模拟策略（随机丢弃）是否最优未深入探讨

## 相关工作与启发

- MoMKE 是最直接的前驱工作，SimMLM 通过专属专家（vs. MoMKE 的共享专家）大幅降低了计算量
- ShaSpec 的模态共享/特定特征解耦思路复杂但灵活性差，DMoME 在输出空间而非特征空间做融合更通用
- MoFe 损失的排序学习思想与 RLHF 中的偏好排序有相似之处
- 对临床 AI 部署的重要启示：在资源有限环境中，耐受部分模态缺失 + 提供可解释的模态权重 = 临床可信赖的 AI

## 评分

- **新颖性**: ⭐⭐⭐⭐ MoFe 排序损失是简洁而深刻的创新，DMoME 虽然简单但非常有效
- **实验充分度**: ⭐⭐⭐⭐⭐ 三个数据集、15种模态配置、效率分析、校准分析、可解释性分析、反直觉率等极其全面
- **写作质量**: ⭐⭐⭐⭐ 动机清晰，方法描述简洁，分析深入
- **价值**: ⭐⭐⭐⭐⭐ 提供了一个通用、高效、可解释的多模态缺失处理方案，具有很高的实际部署价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] CLoE: Expert Consistency Learning for Missing Modality Segmentation](../../CVPR2025/medical_imaging/cloe_expert_consistency_learning_for_missing_modality_segmentation.md)
- [\[CVPR 2025\] Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](../../CVPR2025/medical_imaging/federated_modality-specific_encoders_and_partially_personalized_fusion_decoder_f.md)
- [\[CVPR 2026\] MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality](../../CVPR2026/medical_imaging/must_modality-specific_representation-aware_transformer_for_diffusion-enhanced_s.md)
- [\[CVPR 2025\] Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](../../CVPR2025/medical_imaging/human_knowledge_integrated_multi-modal_learning_for_single_source_domain_general.md)
- [\[CVPR 2025\] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](../../CVPR2025/medical_imaging/semitooth_a_generalizable_semi-supervised_framework_for_multi-source_tooth_segme.md)

</div>

<!-- RELATED:END -->
