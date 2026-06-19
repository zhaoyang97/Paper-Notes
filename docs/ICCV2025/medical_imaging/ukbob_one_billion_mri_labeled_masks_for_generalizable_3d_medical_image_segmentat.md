---
title: >-
  [论文解读] UKBOB: One Billion MRI Labeled Masks for Generalizable 3D Medical Image Segmentation
description: >-
  [ICCV 2025][医学图像][医学图像分割] 本文构建了UKBOB——迄今最大的标注医学影像分割数据集（51,761个MRI 3D样本，72类器官，13.7亿2D分割mask），提出Specialized Organ Label Filter (SOLF)清洗自动标注和Entropy Test-Time Adaptation (ETTA)处理带噪标签的域迁移，训练的Swin-BOB基础模型在BRATS和BTCV基准上达到SOTA。
tags:
  - "ICCV 2025"
  - "医学图像"
  - "医学图像分割"
  - "UK Biobank"
  - "大规模数据集"
  - "自动标注质量控制"
  - "测试时适应"
---

# UKBOB: One Billion MRI Labeled Masks for Generalizable 3D Medical Image Segmentation

**会议**: ICCV 2025  
**arXiv**: [2504.06908](https://arxiv.org/abs/2504.06908)  
**代码**: [https://emmanuelleb985.github.io/ukbob](https://emmanuelleb985.github.io/ukbob)  
**领域**: 医学图像分割 / 基础模型 / 数据集  
**关键词**: 医学图像分割, UK Biobank, 大规模数据集, 自动标注质量控制, 测试时适应

## 一句话总结

本文构建了UKBOB——迄今最大的标注医学影像分割数据集（51,761个MRI 3D样本，72类器官，13.7亿2D分割mask），提出Specialized Organ Label Filter (SOLF)清洗自动标注和Entropy Test-Time Adaptation (ETTA)处理带噪标签的域迁移，训练的Swin-BOB基础模型在BRATS和BTCV基准上达到SOTA。

## 研究背景与动机

**领域现状**：大规模标注数据（如ImageNet、LAION）是计算机视觉基础模型成功的基石。但医学影像领域因隐私法规、标注成本高昂和后勤复杂性，缺乏大规模标注数据集。现有数据集如BRATS（1,470样本/3类）、BTCV（50样本/12类）都规模有限。

**现有痛点**：(1) 现有医学数据集要么缺乏多样性，要么规模太小，无法训练泛化性强的基础模型；(2) 自动标注虽能解决规模问题但引入噪声标签；(3) 训练-测试域偏移（不同设备、协议、模态）导致性能退化。

**核心矛盾**：医学影像迫切需要大规模标注数据来训练基础模型，但人工标注17.9M张图像不可行，自动标注又必然引入噪声。

**本文目标** (a) 构建最大规模的医学影像分割标注数据集；(b) 设计标签质量控制机制；(c) 处理带噪标签训练的模型在测试时的域偏移问题。

**切入角度**：利用UK Biobank的51,761个全身MRI扫描作为数据源，配合TotalVibeSegmentator自动标注，然后设计基于器官几何特性的统计过滤器清洗标签。

**核心 idea**：用自动标注+统计过滤构建超大规模医学影像数据集，再通过熵驱动的测试时适应处理残留噪声，训练出高泛化性的3D医学分割基础模型。

## 方法详解

### 整体框架

Pipeline：UK Biobank MRI数据 → TotalVibeSegmentator自动标注72类器官 → SOLF统计过滤清洗噪声标签 → 训练Swin-UNetr得到Swin-BOB基础模型 → 下游微调（BTCV、BRATS等）→ ETTA测试时适应提升鲁棒性。

### 关键设计

1. **UKBOB数据集构建**:

    - 功能：从UK Biobank的51,761个颈到膝全身MRI扫描（4序列：fat-only、water-only、in-phase、out-of-phase）中生成72类器官的分割标注
    - 核心思路：使用TotalVibeSegmentator自动标注，生成17.9M 2D图像和13.7亿2D分割mask
    - 规模比较：比之前最大的TotalSegmentator（1,204样本/104类/580万mask）大约237倍mask数量
    - 手动验证集UKBOB-manual：300个MRI，11个腹部器官，3000张2D图像，作为自动标注质量验证

2. **Specialized Organ Label Filter (SOLF)**:

    - 功能：基于器官几何特性的统计过滤器，去除自动标注中的错误标签
    - 核心思路：对每个器官类别c，计算三个特征：
        - **归一化体积**：$v_c = V_c / V_{\text{body}}$（器官体积相对全身体积）
        - **球度(Sphericity)**：$\Phi_c = \pi^{1/3}(6V_c)^{2/3}/A_c$（形状接近球体的程度）
        - **离心率(Eccentricity)**：$E_c = \sqrt{1-\lambda_{\min}/\lambda_{\max}}$（形状的细长程度）
    - 过滤规则：对每个特征排除极端 $\epsilon$ 百分位。当三个特征中至少两个超出正常范围时，标记为不准确标签
    - 设计动机：人体器官形态有进化确定的几何规律，一个真正异常的患者不太可能同时在两个以上独立几何特征上偏离正常范围。因此多特征联合异常更可能是标注错误
    - vs IQR过滤：SOLF利用了患者全身统计信息（归一化体积）和器官特定几何特征，比简单的单一统计量IQR过滤更精准

3. **Entropy Test-Time Adaptation (ETTA)**:

    - 功能：测试时优化batch normalization参数以提升分割鲁棒性
    - 核心思路：给定测试样本 $\mathbf{x}$，计算预测概率的熵损失：
    $\mathcal{L}_{\text{ent}} = -\frac{1}{N}\sum_{i=1}^N \sum_{c=1}^C p_{i,c} \log p_{i,c}$
    - 仅更新BN参数 $\theta_{\text{BN}}$，冻结其他参数：$\theta_{\text{BN}}^* = \arg\min_{\theta_{\text{BN}}} \mathcal{L}_{\text{ent}}(f_{\theta_{\text{fixed}}, \theta_{\text{BN}}}(\mathbf{x}))$
    - 设计动机：当训练数据含噪声标签时，标准TTA方法可能不够鲁棒。ETTA通过鼓励低熵（高置信度）预测来自适应测试样本，同时只更新少量BN参数保持高效
    - 架构无关：可用于任何包含BN层的分割网络

4. **Swin-BOB基础模型**:

    - 功能：在过滤后的UKBOB上预训练Swin-UNetr，作为3D医学分割基础模型
    - 训练配置：96³裁剪，batch size 8, AdamW, 3000 epochs, 2×A6000 GPU
    - 损失函数：Binary Cross-Entropy + Dice Similarity Coefficient
    - 下游微调：保持相同配置，减少warm-up到50 epochs，共500 epochs

### 损失函数 / 训练策略

预训练和微调均使用Binary Cross-Entropy + DSC。ETTA测试时使用熵损失仅更新BN参数。

## 实验关键数据

### 主实验

**BTCV腹部CT分割（12类，行业标准基准）**：

| 模型 | Dice Score |
|------|-----------|
| UNetr | 0.856 |
| Swin-UNetr | 0.869 |
| nnUNet | 0.802 |
| MedSegDiff | 0.879 |
| **Swin-BOB (Ours)** | **0.892** |
| MedSegDiff-V2 (10-fold ens.) | 0.895 |
| **Swin-BOB (10-fold ens.)** | **0.897** |

**BRATS脑肿瘤MRI分割（3类）**：

| 模型 | Dice Score | Hausdorff Distance |
|------|-----------|-------------------|
| SegResNet | 0.890 | 8.650 |
| Swin-UNetr | 0.886 | 9.016 |
| **Swin-BOB (Ours)** | **0.894** | **8.650** |

### 消融实验

**SOLF过滤效果（零样本泛化到BTCV）**：

| 过滤配置 | BTCV平均Dice | AMOS平均Dice |
|---------|-------------|-------------|
| 无过滤 | 0.856 | 0.818 |
| + 体积过滤 | 0.875 | 0.832 |
| + **完整SOLF** | **0.882** | **0.840** |

**ETTA效果（不同模型×不同数据集）**：

| 配置 | BTCV Dice | AMOS Dice | BRATS Dice |
|------|-----------|-----------|-----------|
| Swin-BOB | 0.883 | 0.847 | 0.882 |
| Swin-BOB + TTA [基线] | 0.883 | 0.857 | 0.887 |
| **Swin-BOB + ETTA** | **0.892** | **0.864** | **0.894** |

**SOLF过滤阈值ε消融**：

| ε值 | BTCV Dice |
|-----|-----------|
| 0 (无过滤) | 0.792 |
| 1 | 0.884 |
| **2** | **0.892** |
| 3 | 0.766 |
| 4-5 | 0.745 |

### 关键发现

- **SOLF过滤显著提升标签质量**：零样本BTCV Dice从0.856提升到0.882（+2.6%），远优于简单IQR过滤
- **ε=2是最优过滤阈值**：过松（ε=0）保留太多噪声，过严（ε≥3）过滤掉了正常数据
- **ETTA一致提升所有模型**：在3个网络×3个数据集的9个组合中均优于标准TTA
- **数据规模律**：UKBOB从10%扩大到100%，下游BTCV和BRATS的Dice Score一致增长，验证了大规模预训练的价值
- **跨模态泛化**：在MRI上预训练的Swin-BOB零样本迁移到CT数据（BTCV）仍表现出色，说明学到的表示具有跨模态泛化性
- **手动标注验证**：未过滤UKBOB标注与手动标注的Dice为0.873（腹部）和0.811（脊柱），SOLF过滤后提升到0.891和0.867

## 亮点与洞察

- **规模震撼**：13.7亿2D mask，比此前最大的TotalSegmentator大237倍。首次在医学影像领域实现了"十亿级"标注数据集，有望催生医学影像的"ImageNet时刻"。
- **SOLF过滤设计精巧**：利用进化赋予器官的几何规律（体积、球度、离心率）做质量控制。多特征联合判断增强了鲁棒性——一个特征异常可能是真实病理，两个以上同时异常大概率是标注错误。
- **ETTA的架构无关性**：只操作BN层参数，可即插即用到任何包含BN的分割网络，实用性强。
- **零样本MRI→CT泛化**：证明大规模MRI预训练学到的解剖结构知识可跨模态迁移，打破了MRI/CT模型需分别训练的传统观念。

## 局限与展望

- 数据仅限颈到膝MRI扫描，不包含头颅（除BRATS外的脑部任务）和四肢
- 自动标注即使经过SOLF过滤，仍可能存在残留噪声，尤其对小器官（如十二指肠）
- ETTA需要在测试时做参数优化，增加了推理延迟
- 目前仅基于Swin-UNetr架构，未探索更现代的架构（如nnU-Net V2、Universal Model等）
- UK Biobank的参与者以英国白人为主，数据集可能存在人群偏差
- 72类器官中部分小器官的分割精度可能不高（论文未给出每类的详细结果）

## 相关工作与启发

- **vs TotalSegmentator [2023]**: 104类CT器官分割数据集，但仅1,204样本。UKBOB虽然类别少（72类），但样本数大43倍，且跨模态为MRI。
- **vs AbdomenAtlas [2024]**: 20,460个CT腹部扫描/25类。UKBOB规模更大、覆盖全身、且为MRI模态。
- **vs TotalVibeSegmentator [2025]**: UKBOB的自动标注工具来源，但该工具仅在85+16个样本上训练。UKBOB通过大规模标注+质量过滤，将其转化为可实用训练基础模型的数据集。
- **vs TENT [2020]**: 标准的测试时熵最小化方法。ETTA在此基础上针对带噪声标签预训练的场景做了优化，在6/9个实验配置中优于标准TTA。
- UKBOB的构建范式（大规模自动标注+统计过滤+人工验证子集）可迁移到其他医学影像模态（CT、X-ray、超声等）。

## 评分

- 新颖性: ⭐⭐⭐⭐ 数据集规模上的突破性贡献，SOLF过滤和ETTA方法本身技术创新有限但设计合理
- 实验充分度: ⭐⭐⭐⭐⭐ 标签验证+零样本泛化+多基准评测+消融+数据规模律+特征可视化，非常充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据集构建-质量控制-模型训练-评测的流程完整
- 价值: ⭐⭐⭐⭐⭐ 医学影像领域的里程碑级数据集+基础模型，对社区有深远影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MRGen: Segmentation Data Engine for Underrepresented MRI Modalities](mrgen_segmentation_data_engine_for_underrepresented_mri_modalities.md)
- [\[CVPR 2026\] Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision](../../CVPR2026/medical_imaging/learning_generalizable_3d_medical_image_representations_from_mask-guided_self-su.md)
- [\[ICCV 2025\] PVChat: Personalized Video Chat with One-Shot Learning](pvchat_personalized_video_chat_with_one-shot_learning.md)
- [\[CVPR 2026\] MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation](../../CVPR2026/medical_imaging/medclipseg_probabilistic_vision-language_adaptation_for_data-efficient_and_gener.md)
- [\[CVPR 2025\] Revisiting MAE Pre-Training for 3D Medical Image Segmentation](../../CVPR2025/medical_imaging/revisiting_mae_pre-training_for_3d_medical_image_segmentation.md)

</div>

<!-- RELATED:END -->
