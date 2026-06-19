---
title: >-
  [论文解读] Brain Netflix: Scaling Data to Reconstruct Videos from Brain Signals
description: >-
  [ECCV 2024][医学图像][脑信号视频重建] 本文提出了一种从功能磁共振成像（fMRI）信号重建视频的新方法，通过多数据集多被试训练和三阶段pipeline，利用预训练的文本到视频和视频到视频模型，实现了跨数据集和跨被试的SOTA视频重建能力。 领域现状：脑信号到刺激物重建（brain-to-stimuli reco…
tags:
  - "ECCV 2024"
  - "医学图像"
  - "脑信号视频重建"
  - "fMRI"
  - "多被试学习"
  - "扩散模型"
  - "神经解码"
---

# Brain Netflix: Scaling Data to Reconstruct Videos from Brain Signals

**会议**: ECCV 2024  
**代码**: 无  
**领域**: 其他  
**关键词**: 脑信号视频重建, fMRI, 多被试学习, 扩散模型, 神经解码

## 一句话总结

本文提出了一种从功能磁共振成像（fMRI）信号重建视频的新方法，通过多数据集多被试训练和三阶段pipeline，利用预训练的文本到视频和视频到视频模型，实现了跨数据集和跨被试的SOTA视频重建能力。

## 研究背景与动机

**领域现状**：脑信号到刺激物重建（brain-to-stimuli reconstruction）是神经科学和计算机视觉交叉领域的热门方向。近年来已有不少工作成功地从fMRI信号重建出与被试观看内容相似的静态图像。但视频重建是一个更具挑战性的任务，因为需要同时捕捉空间视觉特征和时间动态信息。

**现有痛点**：（1）现有方法通常是被试特定的（subject-specific），需要为每个新被试重新训练模型，泛化性差；（2）大多数方法只在单个数据集上评估，跨数据集的性能未知；（3）fMRI数据采集昂贵且耗时，单个被试的数据量有限，制约了模型能力上限；（4）从脑信号到视频的映射需要同时回归大量潜在向量和条件向量，回归精度不足。

**核心矛盾**：fMRI数据的稀缺性与视频重建任务所需的大数据之间存在根本矛盾。单被试数据不足以学习到通用的神经-视觉映射，而不同被试和不同数据集之间的fMRI信号存在显著差异（不同扫描仪、不同分辨率、不同脑区划分）。

**本文目标** （1）如何有效聚合多个数据集、多个被试的fMRI数据来扩展训练规模；（2）如何设计一个通用的pipeline来处理不同来源的fMRI数据；（3）如何准确回归预训练视频生成模型所需的关键向量。

**切入角度**：作者认为数据量是当前瓶颈的关键因素，提出通过多数据集多被试联合训练来扩大训练数据规模。同时设计了一个三阶段pipeline，将脑信号解码分解为语义对齐、向量回归和视频生成三个子问题。

**核心 idea**：通过多数据集多被试训练扩大数据规模，用三阶段pipeline（语义对齐→向量回归→视频生成）将fMRI信号准确转化为预训练视频模型的输入向量来重建视频。

## 方法详解

### 整体框架

整体pipeline分为三个阶段。输入是被试观看视频时记录的fMRI信号（体素激活值），输出是重建的视频片段（2-3秒）。

第一阶段：fMRI信号对齐。将不同被试、不同数据集的fMRI信号映射到统一的语义嵌入空间（如CLIP空间），消除个体差异和数据集差异。

第二阶段：向量回归。从对齐后的语义嵌入回归预训练视频生成模型所需的关键潜在向量和条件向量（包括文本嵌入、图像嵌入等）。

第三阶段：视频生成。将回归得到的向量输入预训练的text-to-video和video-to-video模型，生成与原始刺激匹配的重建视频。

### 关键设计

1. **多数据集多被试对齐策略**:

    - 功能：将来自不同数据集、不同被试的fMRI信号投影到统一的语义空间
    - 核心思路：为每个被试学习一个线性投影层，将其fMRI体素空间映射到共享的CLIP语义空间。训练时使用对比学习，拉近同一视频刺激的fMRI嵌入和视频CLIP嵌入，推开不匹配的对。通过共享后续网络层，不同被试的数据可以相互增强
    - 设计动机：不同被试的大脑结构和功能区划分不同，不同数据集使用的扫描参数也不同，不能直接混合使用。per-subject的线性投影层保留了个体特异性，而共享的语义空间则实现了数据聚合

2. **关键向量回归网络**:

    - 功能：从语义嵌入准确回归视频生成模型的条件和潜在向量
    - 核心思路：设计专门的回归网络来估计text-to-video模型所需的文本条件向量、图像条件向量等关键输入。回归网络使用残差MLP结构，对不同类型的向量使用不同的回归头。训练时使用MSE损失加上对比损失来约束回归精度
    - 设计动机：视频生成模型的输入空间维度高且敏感，微小的回归误差可能导致生成视频出现显著差异。通过专门的回归网络和多种损失约束，提升回归精度

3. **三阶段解耦训练**:

    - 功能：将复杂的fMRI到视频映射分解为可控的子任务，提升训练稳定性
    - 核心思路：三个阶段依次训练——先训练fMRI对齐模块，再训练向量回归网络，最后使用预训练的视频生成模型进行推理。每个阶段有清晰的目标和损失函数，避免端到端训练的不稳定性
    - 设计动机：端到端从fMRI到视频的映射过于复杂，中间跨越了太多语义层次。分阶段训练可以确保每一步的映射都是准确的

### 损失函数 / 训练策略

- 第一阶段使用对比学习损失（CLIP-style contrastive loss）进行语义对齐
- 第二阶段使用MSE回归损失和感知损失来约束向量回归精度
- 在多被试训练中，梯度在不同被试之间累积，有效扩大了每次更新的数据量
- 使用预训练的视频生成模型（text-to-video和video-to-video），推理时不训练

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 多数据集评估 | SSIM | SOTA | - | 显著提升 |
| 多数据集评估 | 语义一致性 | SOTA | - | 显著提升 |
| 跨被试评估 | 视觉质量 | 良好 | 退化严重 | 显著提升 |

在多个fMRI数据集上进行了定性和定量评估，包括人工众包评估。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|----------|------|
| 单被试 vs 多被试 | SSIM提升 | 多被试训练显著优于单被试 |
| 单数据集 vs 多数据集 | 语义一致性提升 | 多数据集扩展有效 |
| 不同对齐策略 | 重建质量 | CLIP对齐优于其他策略 |
| 增加被试数量 | 性能曲线 | 更多被试持续带来提升 |

### 关键发现

- 多数据集多被试训练的提升是显著的，数据扩展策略有效
- 三阶段pipeline比直接端到端方法更稳定可靠
- 人工众包评估验证了重建视频与原始刺激的语义一致性
- 作者观察到随着更多被试数据的加入，性能持续改善，暗示了零样本重建的可能性

## 亮点与洞察

- **Data Scaling思路**：将NLP/CV领域"more data, better performance"的理念引入脑信号解码，通过巧妙的对齐策略实现多源数据聚合
- **三阶段解耦**：将极端困难的fMRI→视频映射分解为三个可控步骤，每步都有成熟的技术支撑
- **跨被试泛化**：向零样本脑解码迈出了重要一步，展示了利用已有被试数据帮助新被试的可能性
- **完整评估体系**：包含了定量指标、定性可视化和众包人工评估

## 局限与展望

- fMRI的时间分辨率限制（TR约2秒），无法捕捉快速视觉变化的细节
- 重建视频的时长较短（2-3秒），难以处理长视频
- 依赖于预训练视频生成模型的质量，生成模型的幻觉问题可能导致不准确的细节
- 零样本重建（unseen subject）的性能仍有较大提升空间
- 隐私和伦理问题：脑信号解码技术的发展需要审慎对待

## 相关工作与启发

- **脑信号图像重建**：Mind-Reader、Brain-Diffuser等为静态图像重建奠定基础
- **视频生成模型**：Stable Video Diffusion等提供了强大的条件视频生成能力
- **启发**：数据扩展策略在其他数据稀缺的领域（如医学影像）也可能有效

## 评分

- 新颖性: ⭐⭐⭐⭐ 多数据集多被试训练扩展思路新颖，三阶段pipeline设计合理
- 实验充分度: ⭐⭐⭐⭐ 多数据集评估+消融+众包人工评估
- 写作质量: ⭐⭐⭐⭐ 清晰的pipeline描述和充分的分析
- 价值: ⭐⭐⭐⭐ 推进了脑信号视频重建的SOTA，为数据扩展策略提供了有效范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] CardiacNet: Learning to Reconstruct Abnormalities for Cardiac Disease Assessment from Echocardiogram Videos](cardiacnet_learning_to_reconstruct_abnormalities_for_cardiac_disease_assessment_.md)
- [\[ECCV 2024\] UMBRAE: Unified Multimodal Brain Decoding](umbrae_unified_multimodal_brain_decoding.md)
- [\[ECCV 2024\] Brain-ID: Learning Contrast-agnostic Anatomical Representations for Brain Imaging](brain-id_learning_contrast-agnostic_anatomical_representations_for_brain_imaging.md)
- [\[AAAI 2026\] MindCross: Fast New Subject Adaptation with Limited Data for Cross-subject Video Reconstruction from Brain Signals](../../AAAI2026/medical_imaging/mindcross_fast_new_subject_adaptation_with_limited_data_for_cross-subject_video_.md)
- [\[NeurIPS 2025\] BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals](../../NeurIPS2025/medical_imaging/brainomni_a_brain_foundation_model_for_unified_eeg_and_meg_signals.md)

</div>

<!-- RELATED:END -->
