---
title: >-
  [论文解读] EchoONE: Segmenting Multiple Echocardiography Planes in One Model
description: >-
  [CVPR 2025][医学图像][超声心动图分割] 本文提出 EchoONE，首次用一个统一模型解决超声心动图多切面分割（MPS）问题，通过先验可组合掩码学习（PC-Mask）模块生成语义感知的稠密 prompt，并设计局部特征融合与适配（LFFA）模块将 CNN 局部特征注入 SAM 解码器，在 6 个切面上持续达到 SOTA 性能。
tags:
  - "CVPR 2025"
  - "医学图像"
  - "超声心动图分割"
  - "多切面分割"
  - "SAM适配"
  - "先验可组合掩码"
  - "统一模型"
---

# EchoONE: Segmenting Multiple Echocardiography Planes in One Model

**会议**: CVPR 2025  
**arXiv**: [2412.02993](https://arxiv.org/abs/2412.02993)  
**代码**: [https://github.com/a2502503/EchoONE](https://github.com/a2502503/EchoONE)  
**领域**: 医学图像  
**关键词**: 超声心动图分割、多切面分割、SAM适配、先验可组合掩码、统一模型

## 一句话总结

本文提出 EchoONE，首次用一个统一模型解决超声心动图多切面分割（MPS）问题，通过先验可组合掩码学习（PC-Mask）模块生成语义感知的稠密 prompt，并设计局部特征融合与适配（LFFA）模块将 CNN 局部特征注入 SAM 解码器，在 6 个切面上持续达到 SOTA 性能。

## 研究背景与动机

**领域现状**：超声心动图检查需要从多个切面（如心尖二腔、三腔、四腔、胸骨旁短轴等）观察心脏结构，进行全面评估。机器学习分割模型通常需要为每个切面单独训练，因为不同切面的心脏结构差异巨大（长轴 vs 短轴、腔室数量不同、解剖标志各异），导致开发和部署复杂度很高。

**现有痛点**：现有多切面方案分为两类：(1) 多分支架构——每个分支处理一个切面，再通过跨视角注意力/一致性约束融合，本质上仍是分而治之，无法泛化到新切面；(2) 混合训练——将多个切面数据合并训练单一模型，但性能通常明显下降（如 U-Net 多切面联合训练 Dice 从 ~89 掉到 ~86）。SAM 作为通用分割基础模型在自然图像上表现出色，但直接应用于超声图像由于低对比度、高噪声、组织边界模糊等特点导致性能极差（Dice 仅 ~25%）。

**核心矛盾**：多切面超声图像的结构差异太大（长轴看到的是纵向切面，短轴是横向切面，腔室形态完全不同），而 SAM 的标准 prompt 机制不具备区分这些结构差异的语义感知能力。如何让一个模型"知道"当前输入是哪种切面并自适应调整分割行为？

**本文目标** (1) 构建一个统一模型处理多切面超声分割，无需知道输入的切面信息；(2) 设计语义感知的稠密 prompt 生成机制；(3) 有效将 SAM 适配到超声心动图的多切面场景。

**切入角度**：作者利用先验结构知识——不同切面的图像在特征空间中有聚类规律，可以预计算每个聚类的平均掩码作为结构先验。对新输入图像，通过计算其与各聚类原型的相似度加权组合先验掩码，无需显式的切面标签即可生成语义感知的稠密 prompt。

**核心 idea**：用聚类先验掩码的相似度加权组合生成语义感知的稠密 prompt，让 SAM 自适应分割不同超声切面。

## 方法详解

### 整体框架

EchoONE 由三部分组成：(1) SAM 架构主体（ViT-B 图像编码器 + 掩码解码器）；(2) PC-Mask 模块负责生成语义感知的稠密 prompt，输入到 SAM 的掩码编码器；(3) CNN 分支通过 LFFA 模块与 SAM 解码器进行局部特征交互，辅助 SAM 适配。统一掩码表示将所有数据集的标注重映射为 4 类：背景(0)、左心室(1)、左心室腔(2)、心肌(3)。

### 关键设计

1. **先验可组合掩码学习（PC-Mask）**:

    - 功能：在不知道输入切面信息的前提下，自动生成高质量的语义感知稠密 prompt
    - 核心思路：首先在预训练 ResNet34 的潜在空间中，将所有训练图像聚类为 $K$ 个组，每个组有一个特征原型 $u_i$ 和对应的平均掩码 $m_i$。对新输入图像 $I_j$，计算其与各原型的余弦相似度 $w(i,j) = \text{cossim}(E_{Lat}(I_j), u_i)$，然后用相似度加权拼接各组的平均掩码 $PE_j = \text{concat}([w(i,j) \times m_i])$，形成多通道先验嵌入。最后用一个轻量 U-Net 将先验嵌入细化为最终的稠密 prompt $PCM_j = UNet_\theta(PE_j)$。
    - 设计动机：关键在于 PC-Mask 的语义来自聚类先验而非输入图像本身，这使得它提供的是与传统分割网络不同的结构引导信息。且不依赖切面标签，天然适合多切面统一模型。

2. **局部特征融合与适配模块（LFFA）**:

    - 功能：将 CNN 提取的局部特征注入 SAM 的解码器，弥补 ViT 对局部细节的不足
    - 核心思路：设计一个 CNN 分支（Residual blocks + 跨分支注意力），其每层输出的局部特征与 SAM 解码器中对应 Transformer block 的 key 通过拼接+1×1 卷积融合，得到融合特征 $f_{F,l} = \text{conv}_{1\times1}(\text{concat}(f_{CNN,l}, f_{DM-K,l}))$，然后送入下一个 Transformer block 作为新的 image embedding。在原始 SAM 的 2 个解码器 Transformer block 基础上额外增加 3 个可学习 block 来容纳 4 层 CNN 特征的注入。
    - 设计动机：SAM 的 ViT 在自然图像上预训练，缺乏超声图像中低对比度结构的局部理解能力。CNN 分支提供互补的局部细节特征，且通过 skip 连接直接注入解码器比仅调整编码器更有效，同时加速了收敛。

3. **统一掩码表示**:

    - 功能：统一来自不同数据源、不同标注协议的训练数据
    - 核心思路：将所有数据集的标注重映射为统一的 4 类语义（背景、LV、LV cavity、MYO）。对仅有心肌标注的数据，通过检测解剖标志点并填充来生成 LV cavity 掩码。
    - 设计动机：多源数据集使用不同标注协议（有的标左心室有的标心肌有的标左心房），必须统一才能联合训练。

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_{SEG} + 0.5 \cdot \mathcal{L}_{PCM}$。$\mathcal{L}_{SEG}$ 监督最终分割输出（0.8 Dice + 0.2 BCE），$\mathcal{L}_{PCM}$ 监督 PC-Mask 模块的掩码学习（同 Dice + BCE 组合）。使用 Adam 优化器（lr=1e-4），训练 100 epochs，图像 resize 到 256×256，单卡 A6000 训练。

## 实验关键数据

### 主实验（内部评估，按切面平均）

| 方法 | 2CH mDice | 4CH mDice | PSAX mDice | 整体均值 |
|------|-----------|-----------|------------|---------|
| U-Net | 86.40 | 86.62 | 82.32 | ~85.3 |
| SAMUS | 87.51 | 88.37 | 85.92 | ~87.3 |
| SAM (zero-shot) | 26.24 | 27.22 | 25.10 | ~26.2 |
| MedSAM | 81.76 | 84.14 | 76.99 | ~81.0 |
| **EchoONE** | **89.67** | **90.28** | **88.26** | **~89.4** |

EchoONE 在所有切面上均取得最高 Dice，且在 PSAX 短轴切面上优势特别明显（88.26 vs SAMUS 85.92），可能归功于 PC-Mask 的结构先验对短轴心脏切面的有效引导。

### 消融/外部验证

| 数据集 | EchoONE mDice | SAMUS mDice | U-Net mDice |
|--------|---------------|-------------|-------------|
| 中心 A (内部) | 89.67 | 87.51 | 86.75 |
| 中心 B (内部) | 87.27 | 85.54 | 77.35 |
| HMC_QU (外部) | 73.94 | 72.38 | 67.47 |
| EchoNet (外部) | 87.62 | 85.77 | 83.10 |

在两个完全未参与训练的外部数据集上，EchoONE 仍取得最佳性能，展现出良好泛化能力。即使 HMC_QU 包含低质量图像和噪声标注，EchoONE 仍优于所有竞争方法。

### 关键发现
- SAM 直接用于超声图像完全失败（Dice ~25%），说明自然图像与医学超声存在巨大分布差异
- 传统方法中 SAMUS（专门为超声设计的 SAM 适配方法）表现最好，但 EchoONE 在所有切面上仍显著优于它
- PC-Mask 模块贡献了主要性能增益，尤其在 PSAX 切面上效果最突出
- LFFA 模块不仅提升了分割精度，还加速了模型收敛

## 亮点与洞察
- **PC-Mask 的"先验组合"思路**非常巧妙：不直接从图像生成 prompt，而是从预计算的聚类先验中加权组合，这使得 prompt 包含了跨数据集的结构知识，具有更强的泛化性
- **统一掩码表示**解决了多源超声数据标注不一致的实际问题，这个工程化贡献对超声社区很有价值
- **一个模型处理 6 个切面**是该领域首次，相比传统方法需要 6 个独立模型，部署复杂度大幅降低

## 局限与展望
- PC-Mask 的聚类数 $K$ 是预设超参，文中未充分讨论其选择对性能的影响
- 当前仅在心脏超声上验证，是否可推广到其他多切面/多平面医学影像（如 MRI、CT）需要进一步探索
- 统一掩码表示将 LA 的标注简化处理，可能丢失了部分左心房的结构信息
- ViT-B 作为图像编码器在推理时的计算开销较大，移动端部署可能需要更轻量的backbone
- 外部泛化在 HMC_QU 上 Dice 仅 73.94%，与内部评估有差距，低质量图像场景仍有提升空间

## 相关工作与启发
- **vs SAMUS**: SAMUS 同样使用 CNN 侧分支适配 SAM 到超声，但缺少语义感知的稠密 prompt 机制，在多切面场景下效果不如 EchoONE
- **vs MedSAM**: MedSAM 使用 bounding box prompt 进行医学图像分割，但对超声多切面的结构差异缺乏适应能力
- **vs TransFusion 等多分支方法**: 这些方法每个切面用独立分支，无法泛化到新切面，EchoONE 的统一设计在实用性上明显优于它们

## 评分
- 新颖性: ⭐⭐⭐⭐ PC-Mask 的聚类先验组合设计新颖，但整体框架基于 SAM 适配的研究较多
- 实验充分度: ⭐⭐⭐⭐⭐ 7 个数据集（5 内部 + 2 外部）、6 个切面的全面验证极为充分
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述规范，图示信息量大
- 价值: ⭐⭐⭐⭐ 首次解决超声多切面统一分割问题，对临床部署有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] EchoWorld: Learning Motion-Aware World Models for Echocardiography Probe Guidance](echoworld_learning_motion-aware_world_models_for_echocardiography_probe_guidance.md)
- [\[ICCV 2025\] PVChat: Personalized Video Chat with One-Shot Learning](../../ICCV2025/medical_imaging/pvchat_personalized_video_chat_with_one-shot_learning.md)
- [\[CVPR 2025\] MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification](mil-pf_multiple_instance_learning_on_precomputed_features_for_mammography_classi.md)
- [\[ICML 2025\] Do Multiple Instance Learning Models Transfer?](../../ICML2025/medical_imaging/do_multiple_instance_learning_models_transfer.md)
- [\[ICML 2025\] Mastering Multiple-Expert Routing: Realizable H-Consistency and Strong Guarantees](../../ICML2025/medical_imaging/mastering_multiple-expert_routing_realizable_h-consistency_and_strong_guarantees.md)

</div>

<!-- RELATED:END -->
