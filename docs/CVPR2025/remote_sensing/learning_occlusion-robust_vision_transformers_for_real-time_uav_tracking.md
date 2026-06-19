---
title: >-
  [论文解读] Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking
description: >-
  [CVPR 2025][遥感][无人机跟踪] 提出 ORTrack 框架，通过基于空间 Cox 过程的随机遮罩来学习遮挡鲁棒的 ViT 特征表征（训练时加遮罩约束、推理时零开销），并设计自适应特征蒸馏方法将大模型压缩为轻量级学生模型 ORTrack-D，在多个无人机跟踪基准上实现 SOTA 精度与实时速度的最佳平衡。
tags:
  - "CVPR 2025"
  - "遥感"
  - "无人机跟踪"
  - "遮挡鲁棒"
  - "Transformer"
  - "知识蒸馏"
  - "空间Cox过程"
---

# Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking

**会议**: CVPR 2025  
**arXiv**: [2504.09228](https://arxiv.org/abs/2504.09228)  
**代码**: [https://github.com/wuyou3474/ORTrack](https://github.com/wuyou3474/ORTrack)  
**领域**: 视频理解 / 目标跟踪  
**关键词**: 无人机跟踪, 遮挡鲁棒, Vision Transformer, 知识蒸馏, 空间Cox过程

## 一句话总结

提出 ORTrack 框架，通过基于空间 Cox 过程的随机遮罩来学习遮挡鲁棒的 ViT 特征表征（训练时加遮罩约束、推理时零开销），并设计自适应特征蒸馏方法将大模型压缩为轻量级学生模型 ORTrack-D，在多个无人机跟踪基准上实现 SOTA 精度与实时速度的最佳平衡。

## 研究背景与动机

**领域现状**：无人机跟踪近年从 DCF（判别相关滤波器）方法转向基于 ViT 的单流架构（如 OSTrack、MixFormer），因其结构简洁且精度高而成为主流。Aba-ViTrack 等轻量化方法已实现了实时 UAV 跟踪。

**现有痛点**：无人机场景中遮挡问题极为频繁——建筑物、树木等障碍物经常遮挡目标，而现有单流 ViT 跟踪器缺乏专门的遮挡处理策略，遮挡发生时跟踪精度显著下降。此外，Aba-ViTrack 使用可变 token 数量的策略导致非结构化访存操作，实际推理时间开销大。

**核心矛盾**：需要在保持实时速度的同时提升 ViT 的遮挡鲁棒性——加入复杂的遮挡处理模块会牺牲效率，但不处理遮挡又影响精度。

**本文目标** (1) 如何让 ViT 在不增加推理开销的前提下学到遮挡鲁棒的特征表征；(2) 如何将性能强的大模型压缩为部署友好的轻量模型。

**切入角度**：如果在训练时对模板图像进行随机遮罩、同时约束遮罩前后的特征表征保持一致，那模型就能自然学到对遮挡不敏感的特征。关键创新是用空间 Cox 过程（而非均匀随机）建模遮罩分布，使遮罩更集中在目标中心区域，更真实地模拟实际遮挡。

**核心 idea**：训练时用空间 Cox 过程驱动的随机遮罩模拟遮挡并最小化遮罩前后特征差异，推理时无额外开销即获得遮挡鲁棒性。

## 方法详解

### 整体框架

ORTrack 包含两阶段串行训练：第一阶段训练教师模型（加入遮挡鲁棒表征学习），第二阶段用自适应知识蒸馏将教师压缩为学生模型。整个框架基于单流 ViT 跟踪架构，输入模板 $Z$ 和搜索图 $X$ 拼接后送入 ViT，输出 token 经预测头得到目标框。

### 关键设计

1. **基于空间 Cox 过程的遮挡鲁棒表征学习 (ORR)**:

    - 功能：在训练时模拟遮挡，使 ViT 学到对遮挡不变的特征
    - 核心思路：对模板图像 $Z$ 应用随机遮罩 $\mathfrak{m}(Z)$ 得到 $Z'$，将 $(Z, X)$ 和 $(Z', X)$ 分别输入 ViT，最小化两者输出中模板对应 token 的 MSE。关键在于遮罩分布：不用 MAE 的均匀遮罩，而是用空间 Cox 过程生成非均匀遮罩。Cox 过程是"双重随机泊松过程"——先生成随机强度函数 $\lambda(x,y) = \Gamma e^{-(x^2+y^2)} / \int e^{-(x^2+y^2)}$，这是一个钟形函数，使得遮罩更集中在模板的中心区域（即目标所在区域）。随机变量 $\Gamma$ 引入遮罩比例的随机性，使得训练中看到更多样的遮挡模式。
    - 设计动机：均匀遮罩在目标模板上不合理——模板通常包含背景，均匀遮罩可能大部分遮的是背景而非目标。Cox 过程的钟形强度函数让目标区域更大概率被遮罩，更好模拟实际遮挡。**推理时不需要遮罩操作，零额外开销。**

2. **自适应特征知识蒸馏 (AFKD)**:

    - 功能：将教师模型压缩为更小的学生模型同时保持性能
    - 核心思路：学生模型与教师结构相同但 ViT 层数更少。蒸馏损失为教师-学生输出 token 的 MSE，但乘以一个自适应权重 $\varpi = \alpha + \beta(\mathcal{L}_{iou} - \overline{\mathcal{L}_{iou}})$，其中 $\mathcal{L}_{iou}$ 是学生预测与 GT 的 GIoU 损失。当任务困难（损失高于平均）时加大蒸馏力度让学生多学教师，当任务简单（损失低于平均）时减弱蒸馏以保留学生自身的泛化能力。
    - 设计动机：传统蒸馏对所有样本一视同仁，但过度模仿教师可能让学生继承教师的虚假相关性。自适应策略让学生在"已经会做的事上保持自主"、在"不会做的事上向教师学习"。

3. **预测头与损失函数**:

    - 功能：输出目标定位框
    - 核心思路：搜索图对应的输出 token 重塑为 2D 特征图，经 4 层 Conv-BN-ReLU 预测分类分数、位置偏移和框大小。训练损失：加权 Focal Loss（分类）+ L1 + GIoU（回归）+ MSE（ORR约束）。
    - 设计动机：沿用 OSTrack 的标准预测头，简洁高效。

### 损失函数 / 训练策略

教师阶段：$\mathcal{L}_T = \mathcal{L}_{pred} + \gamma \mathcal{L}_{orr}$，其中 $\gamma = 2 \times 10^{-4}$。学生阶段：冻结教师权重，$\mathcal{L}_S = \mathcal{L}_{pred} + \mathcal{L}_{afkd}$。训练 300 epoch，AdamW 优化器，初始学习率 $4 \times 10^{-5}$。

## 实验关键数据

### 主实验

| 基准 | 方法 | Prec. | Succ. | GPU FPS |
|------|------|-------|-------|---------|
| UAVDT | ORTrack-DeiT | **83.4** | **60.1** | 226 |
| UAVDT | Aba-ViTrack | 83.4 | 59.9 | 182 |
| VisDrone2018 | ORTrack-DeiT | **88.6** | **66.8** | 206 |
| VisDrone2018 | AQATrack (CVPR24) | 87.2 | 66.9 | 53 |
| DTB70 | ORTrack-DeiT | **86.2** | **66.4** | 226 |
| UAV123 | ORTrack-DeiT | 84.3 | 66.4 | 226 |

ORTrack-D-DeiT（蒸馏版）在4个基准平均 Prec. 83.7%、Succ. 63.7%，速度达 292 FPS（GPU）/ 65 FPS（CPU）。

### 消融实验

| 配置 | UAVDT Prec. | UAVDT Succ. | 说明 |
|------|-------------|-------------|------|
| Baseline (无 ORR) | 80.8 | 57.4 | 基线 ViT-tiny 跟踪器 |
| + 均匀遮罩 ORR | 82.1 | 58.5 | MAE 式均匀遮罩有帮助但有限 |
| + Cox 过程遮罩 ORR | **83.4** | **60.1** | Cox 遮罩显著优于均匀遮罩 |
| w/o AFKD（常规蒸馏） | 81.5 | 58.9 | 非自适应蒸馏效果一般 |
| + AFKD | **82.5** | **59.7** | 自适应蒸馏优于固定权重蒸馏 |

### 关键发现

- Cox 过程遮罩比 MAE 均匀遮罩提升约 1.3% Prec.，验证了中心聚集遮罩更好模拟真实遮挡的假设
- ORR 仅在训练时增加计算，推理完全零开销，非常适合资源受限的 UAV 部署
- 蒸馏版 ORTrack-D-DeiT 只损失约 1% Prec. 但速度提升 30%（226→292 FPS），实用性极强
- 在 VisDrone2018 上，ORTrack-DeiT 以 206 FPS 速度超越了所有深度跟踪器（如 AQATrack 53 FPS），速度快 4 倍但精度相当

## 亮点与洞察

- **空间 Cox 过程模拟遮挡是非常巧妙的建模**：不同于简单的均匀遮罩或固定模式遮罩，Cox 过程同时引入了空间非均匀性（中心更可能被遮）和遮罩比例随机性（$\Gamma$ 的泊松分布），更好地模拟现实遮挡的多样性。这个 idea 可以迁移到其他需要遮挡鲁棒性的任务（行人重识别、姿态估计等）。
- **"训练时加约束、推理时零开销"的范式**：ORR 只是一个训练时的辅助损失，不改变网络结构，可以即插即用到任何 ViT 跟踪器中，实用性很强。
- **自适应蒸馏的"难易区分"思路**：GIoU 偏差作为任务难度的代理指标，简单直观且有效。避免了在简单样本上过度模仿教师导致的过拟合。

## 局限与展望

- Cox 过程假设遮挡集中在模板中心，但实际遮挡可能是任意方向的（如从侧面遮挡），更通用的遮挡模拟可能需要条件性的遮罩分布
- 使用极小的 $\gamma = 2 \times 10^{-4}$ 权重系数来平衡 ORR 损失，暗示该损失的量级和方向可能与跟踪损失不完全兼容，更深入的梯度分析值得探索
- 蒸馏只用最后一层特征的 MSE，多层特征蒸馏或关系蒸馏可能带来进一步提升
- 未在 LaSOT 等大规模通用跟踪基准上测试，UAV 场景之外的泛化性需验证

## 相关工作与启发

- **vs Aba-ViTrack**: Aba-ViTrack 用自适应 token 计算提升效率但带来非结构化访存；ORTrack 保持固定 token 数并用蒸馏提速，更硬件友好。两者在精度上相当但 ORTrack-D 速度更快。
- **vs DropMAE**: DropMAE 也用遮罩增强跟踪鲁棒性，但用的是 MAE 框架进行重建；ORTrack 不做重建，只约束特征一致性，更轻量。
- **vs SGDViT**: SGDViT 从视觉表征角度增强跟踪但速度慢（111 FPS）；ORTrack 在精度和速度上全面超越。

## 评分

- 新颖性: ⭐⭐⭐⭐ 空间 Cox 过程用于遮挡模拟有理论新意，自适应蒸馏也不错
- 实验充分度: ⭐⭐⭐⭐⭐ 4 个 UAV 基准 + 通用跟踪对比 + 详细消融
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨但 Cox 过程部分对非数学背景读者偏硬核
- 价值: ⭐⭐⭐⭐ 实用性强，"训练约束+推理零开销"范式可广泛迁移

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] ChA-MAEViT: Unifying Channel-Aware Masked Autoencoders and Multi-Channel Vision Transformers for Improved Cross-Channel Learning](../../NeurIPS2025/remote_sensing/chamaevit_unifying_channelaware_masked_autoencoders_and_mult.md)
- [\[CVPR 2025\] Hierarchical Dual-Change Collaborative Learning for UAV Scene Change Captioning](hierarchical_dual-change_collaborative_learning_for_uav_scene_change_captioning.md)
- [\[NeurIPS 2025\] OrbitZoo: Real Orbital Systems Challenges for Reinforcement Learning](../../NeurIPS2025/remote_sensing/orbitzoo_real_orbital_systems_challenges_for_reinforcement_learning.md)
- [\[ICML 2025\] ExPLoRA: Parameter-Efficient Extended Pre-Training to Adapt Vision Transformers under Domain Shifts](../../ICML2025/remote_sensing/explora_parameter-efficient_extended_pre-training_to_adapt_vision_transformers_u.md)
- [\[ICML 2025\] Resampling Augmentation for Time Series Contrastive Learning: Application to Remote Sensing](../../ICML2025/remote_sensing/resampling_augmentation_for_time_series_contrastive_learning_application_to_remo.md)

</div>

<!-- RELATED:END -->
