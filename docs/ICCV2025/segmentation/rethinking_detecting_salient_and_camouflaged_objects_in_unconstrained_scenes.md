---
title: >-
  [论文解读] Rethinking Detecting Salient and Camouflaged Objects in Unconstrained Scenes
description: >-
  [ICCV 2025][语义分割][显著性检测] 构建首个无约束显著性和伪装目标检测数据集USC12K（覆盖四种场景类型），提出基于SAM的USCNet网络，通过属性关系建模（ARM）模块显式建模显著和伪装目标的关系，并设计新指标CSCS衡量混淆程度，在所有场景中达到SOTA。 显著性目标检测（SOD）和伪装目标检测（COD…
tags:
  - "ICCV 2025"
  - "语义分割"
  - "显著性检测"
  - "伪装目标检测"
  - "无约束场景"
  - "属性关系建模"
  - "SAM"
---

# Rethinking Detecting Salient and Camouflaged Objects in Unconstrained Scenes

**会议**: ICCV 2025  
**arXiv**: [2412.10943](https://arxiv.org/abs/2412.10943)  
**代码**: [GitHub](https://github.com/)  
**领域**: 图像分割  
**关键词**: 显著性检测, 伪装目标检测, 无约束场景, 属性关系建模, SAM

## 一句话总结
构建首个无约束显著性和伪装目标检测数据集USC12K（覆盖四种场景类型），提出基于SAM的USCNet网络，通过属性关系建模（ARM）模块显式建模显著和伪装目标的关系，并设计新指标CSCS衡量混淆程度，在所有场景中达到SOTA。

## 研究背景与动机

显著性目标检测（SOD）和伪装目标检测（COD）是计算机视觉中两个相关但对立的任务——SOD检测最引人注目的物体，COD检测与环境融为一体的物体。两者在医学异常检测、自动驾驶障碍识别、军事侦察等领域有重要应用。

**核心矛盾：现有模型无法区分显著和伪装目标**。论文发现了一个反直觉现象：SOD模型在COD数据集上依然能产生较高的检测分数（如ICON在COD10K上$F_\beta^\omega=0.6384$），COD模型在SOD数据集上也是如此（如SINet-V2在DUTS上达0.7412）。也就是说，SOD模型将伪装目标误判为显著，COD模型将显著目标误判为伪装。

**两个根本原因**：

**数据集标注范式缺陷**：现有SOD/COD数据集施加了互斥约束——假设场景中仅包含显著或伪装目标之一。这与真实世界不符——场景可能同时包含两类目标、仅包含一类、或两者都不包含。这种受限标注导致COD数据集中的显著物体被当作背景，SOD数据集中的伪装物体也被当作背景，产生标注冲突。

**模型缺乏显式属性关系建模**：现有SOD/COD模型独立学习两个任务，即使是统一模型（如VSCode、UJSC）也仅通过对比学习等方式间接建立联系，缺乏对样本内部显著-伪装关系的显式建模。

**核心idea**：(1) 构建覆盖四种场景的USC12K数据集消除数据限制；(2) 设计ARM模块从样本间（Inter-SPQ）和样本内（Intra-SPQ）两个维度显式建模属性关系。

## 方法详解

### 整体框架
USCNet基于SAM构建，包含三个主要组件：(1) 带有Adapter层的SAM图像编码器；(2) 属性关系建模（ARM）模块，生成显著性、伪装性、背景三种属性提示；(3) 冻结的SAM掩码解码器，基于属性提示预测三类掩码。

### 关键设计

1. **USC12K数据集（四场景无约束数据集）**:

    - 功能：构建首个不限制显著/伪装目标存在的数据集
    - 核心设计：12000张图像覆盖四种场景——
        - **Scene A**：仅有显著目标（3000张，来自DUTS和HKU-IS）
        - **Scene B**：仅有伪装目标（3000张，来自COD10K和CAMO）
        - **Scene C**：同时含有显著和伪装目标（3000张，含2617张网络采集）
        - **Scene D**：既无显著也无伪装目标的背景场景（3000张）
    - 标注：覆盖9个超类、179个子类，使用SAM粗标注+人工精炼。训练集8400张，测试集3600张。
    - 设计动机：只有覆盖所有逻辑可能的场景，模型才能真正学会区分显著和伪装目标

2. **属性关系建模（ARM）模块**:

    - 功能：显式建模显著-伪装-背景三种属性的关系
    - 核心思路：设计两种互补的提示查询机制——
        - **Inter-SPQ（样本间提示查询）**：一组可学习的查询嵌入 $Q_{S_r}, Q_{C_r}, Q_{B_r} \in \mathbb{R}^{N \times C}$，在推理时固定不变。捕获跨样本的通用判别特征（如大小、位置、颜色、纹理的统计规律）。
        - **Intra-SPQ（样本内提示查询）**：从编码器特征 $F$ 动态生成，随样本变化。通过注意力头 $\Phi_{AH}$ 生成注意力图（由GT监督），提取样本特定的属性特征：
       $$[Q_{S_a}, Q_{C_a}, Q_{B_a}] = \text{Linear}(\sigma(\Phi_{AH}(F)) \otimes F)$$
        - 两者求和后通过自注意力 + Query-to-Image交叉注意力 + MLP生成最终属性提示 $P$：
       $$P = \text{MLP}(\text{Q2I}(\text{SA}(\text{Intra-SPQ} + \text{Inter-SPQ}), F))$$
    - 设计动机：Inter-SPQ学习全局通用规律（如"显著目标通常颜色鲜明"），Intra-SPQ关注单样本内的具体关系（如"这张图中花朵是显著的而蝴蝶是伪装的"），两者互补

3. **冻结SAM掩码解码器**:

    - 功能：基于三种属性提示预测三类掩码 $M_S, M_C, M_B$
    - 核心思路：$[M_S, M_C, M_B] = \text{MaskDe}([P_S, P_C, P_B], F)$，最后用softmax生成最终预测
    - 设计动机：利用SAM的强大分割能力，仅需通过提示即可引导输出不同属性的掩码

4. **CSCS指标（伪装-显著混淆分数）**:

    - 功能：量化模型混淆显著和伪装目标的程度
    - 公式：$\text{CSCS} = \frac{1}{2}(\frac{\mathcal{P}_{CS}}{\mathcal{P}_{BS}+\mathcal{P}_{SS}+\mathcal{P}_{CS}} + \frac{\mathcal{P}_{SC}}{\mathcal{P}_{BC}+\mathcal{P}_{SC}+\mathcal{P}_{CC}})$
    - 其中 $\mathcal{P}_{CS}$ 表示伪装被预测为显著的区域比例，$\mathcal{P}_{SC}$ 表示显著被预测为伪装的区域比例。CSCS越低越好。
    - 设计动机：现有指标（如加权F-measure）只评估前景-背景分离，无法衡量模型对两类目标的混淆程度

### 损失函数 / 训练策略
- 总损失：$\mathcal{L}_{Total} = \lambda_p \mathcal{L}_{pred.} + \lambda_a \mathcal{L}_{att.}$
- 预测损失和注意力损失均使用Focal Loss：$\mathcal{L}_{focal} = -\frac{1}{N}\sum_{i=1}^{N}\alpha_{t_i}(1-p_{t_i})^\gamma \log(p_{t_i})$
- 三类权重比：背景:显著:伪装 = 1:4:6（按像素数比例设定），$\gamma=2$
- $\lambda_p=1, \lambda_a=0.5$
- 使用SAM2 hiera-large版本，AdamW优化器，lr=0.0001，batch=24，最大90 epoch

## 实验关键数据

### 主实验
USC12K四场景综合评估（与21个相关方法对比）：

| 方法 | 类别 | IoU_S↑ | IoU_C↑ | mIoU↑ | mAcc↑ | CSCS↓ |
|------|------|--------|--------|-------|-------|-------|
| PGNet (SOD) | SOD | 74.69 | 57.31 | 71.82 | 80.76 | 7.71 |
| CamoFormer (COD) | COD | 75.88 | 66.19 | 74.81 | 84.17 | 7.57 |
| VSCode (统一) | Unified | 76.04 | 60.31 | 74.17 | 84.01 | 8.17 |
| SAM2-Adapter | Adapter | 78.75 | 70.28 | 74.98 | 84.74 | 9.12 |
| **USCNet (Ours)** | Unified | **79.70** | **74.99** | **78.03** | **87.92** | **7.49** |

### 消融实验
ARM模块各组件的贡献（Overall Scenes）：

| Encoder | Decoder | Intra-S | Inter-S | Q2I | I2Q | mIoU | CSCS↓ |
|---------|---------|---------|---------|-----|-----|------|-------|
| Frozen | Tuning | ✗ | ✗ | ✗ | ✗ | 68.78 | 11.58 |
| Tuning | Tuning | ✗ | ✗ | ✗ | ✗ | 74.98 | 9.12 |
| Tuning | Frozen | ✗ | ✔ | ✔ | ✔ | 75.31 | 9.07 |
| Adapter | Frozen | ✔ | ✔ | ✔ | ✔ | **78.03** | **7.49** |

误检率改善（USC12K训练后 vs 训练前）：

| 模型/测试 | 训练前$F_\beta^\omega$ | 训练后$F_\beta^\omega$ | 说明 |
|-----------|----------------------|----------------------|------|
| ICON→COD10K | 0.6384 | 0.0146 | SOD误检COD大幅降低 |
| SINet-V2→DUTS | 0.7412 | 0.0708 | COD误检SOD大幅降低 |

### 关键发现
- Scene C（两类目标共存）是最具挑战性的场景，所有模型表现均低于单属性场景
- USCNet在参数量仅4.04M（最小）的情况下取得最佳性能
- 用USC12K训练后，误检分数从0.6-0.8降至0.01-0.07，几乎消除了混淆
- USCNet在传统SOD/COD数据集（DUTS、HKU-IS、NC4K、COD10K）上也展现了泛化能力

## 亮点与洞察
- **问题洞察深刻**：发现并系统验证了SOD/COD模型互相"误检"的现象，揭示了根本原因在于数据标注范式而非模型能力
- **数据集设计哲学正确**：从"有约束"走向"无约束"，四种场景覆盖了显著/伪装目标的所有逻辑组合
- **ARM模块设计精妙**：Inter-SPQ捕获通用属性差异，Intra-SPQ关注样本特定关系，两者互补
- **CSCS指标填补空白**：现有指标无法衡量显著-伪装混淆，这个新指标让评估更有针对性
- 参数效率极高——仅4.04M可训练参数

## 局限与展望
- USC12K的Scene C数据中2617张来自网络采集，标注质量可能不如专业数据集
- 当前模型将场景划分为严格的三类（显著/伪装/背景），未考虑"半伪装"等中间状态
- 固定的类别权重比(1:4:6)可能不适用于所有数据分布
- 仅限于2D静态图像，未扩展到视频场景
- 数据集规模相对不大（12K），可能限制模型在更复杂场景中的泛化能力

## 相关工作与启发
- 无约束检测的概念可推广到其他对立任务：如"完整物体 vs 遮挡物体"检测
- ARM模块的Inter/Intra-SPQ设计理念可用于其他多属性分类/分割任务
- SAM作为通用分割基座+任务特定提示的范式在特定领域仍有很大潜力
- CSCS指标的设计思路（衡量类别间混淆而非前景-背景分离）值得学习

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 问题定义新颖（无约束SOD+COD），数据集、模型、指标均为首创
- 实验充分度: ⭐⭐⭐⭐⭐ 21个方法对比，四场景评估，丰富的消融和泛化实验
- 写作质量: ⭐⭐⭐⭐ 问题阐述清晰有说服力，但部分内容较密集
- 价值: ⭐⭐⭐⭐ 推动SOD/COD统一化发展，数据集和基准有长期影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Beyond Single Images: Retrieval Self-Augmented Unsupervised Camouflaged Object Detection](beyond_single_images_retrieval_self-augmented_unsupervised_camouflaged_object_de.md)
- [\[CVPR 2025\] Rethinking Query-Based Transformer for Continual Image Segmentation](../../CVPR2025/segmentation/rethinking_query-based_transformer_for_continual_image_segmentation.md)
- [\[CVPR 2025\] SAM2-LOVE: Segment Anything Model 2 in Language-Aided Audio-Visual Scenes](../../CVPR2025/segmentation/sam2-love_segment_anything_model_2_in_language-aided_audio-visual_scenes.md)
- [\[CVPR 2025\] Paint by Inpaint: Learning to Add Image Objects by Removing Them First](../../CVPR2025/segmentation/paint_by_inpaint_learning_to_add_image_objects_by_removing_them_first.md)
- [\[NeurIPS 2025\] FineRS: Fine-grained Reasoning and Segmentation of Small Objects with Reinforcement Learning](../../NeurIPS2025/segmentation/finers_fine-grained_reasoning_and_segmentation_of_small_objects_with_reinforceme.md)

</div>

<!-- RELATED:END -->
