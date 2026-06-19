---
title: >-
  [论文解读] Self-Supervised ControlNet with Spatio-Temporal Mamba for Real-World Video Super-Resolution
description: >-
  [CVPR 2025][图像生成][视频超分辨率] 提出 SCST 框架，将时空连续 Mamba（STCM）用于全局 3D 注意力建模，并结合基于 MoCo 的自监督 ControlNet 提取退化无关特征，配合三阶段混合训练策略，在真实世界视频超分辨率基准上取得了 SOTA 的感知质量。 领域现状：视频超分辨率（VSR）旨…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "视频超分辨率"
  - "Mamba"
  - "自监督学习"
  - "ControlNet"
  - "对比学习"
---

# Self-Supervised ControlNet with Spatio-Temporal Mamba for Real-World Video Super-Resolution

**会议**: CVPR 2025  
**arXiv**: [2506.01037](https://arxiv.org/abs/2506.01037)  
**代码**: [https://ssj9596.github.io/scst-project/](https://ssj9596.github.io/scst-project/)  
**领域**: 扩散模型 / 视频超分辨率  
**关键词**: 视频超分辨率, Mamba, 自监督学习, ControlNet, 对比学习

## 一句话总结

提出 SCST 框架，将时空连续 Mamba（STCM）用于全局 3D 注意力建模，并结合基于 MoCo 的自监督 ControlNet 提取退化无关特征，配合三阶段混合训练策略，在真实世界视频超分辨率基准上取得了 SOTA 的感知质量。

## 研究背景与动机

**领域现状**：视频超分辨率（VSR）旨在利用低分辨率帧序列之间的时间互补信息重建高分辨率视频。基于扩散模型的方法（如 StableSR、MGLD）已展现出强大的生成先验能力，但存在两个核心挑战：(1) 扩散采样的随机性导致帧间时间不一致；(2) 真实世界复杂退化（模糊、噪声、压缩等叠加）使得条件注入不稳定。

**现有痛点**：现有方法在解决时间一致性方面采用 3D 卷积或时间注意力（如 Upscale-A-Video），但感受野受限。使用光流对齐（如 MGLD）可以增强时间相干性，但在空间恢复方面存在不足。更关键的是，真实世界 LR 视频中的未知复杂退化导致 ControlNet 难以提取干净特征，直接训练容易产生伪影。

**核心矛盾**：同时需要解决两个耦合问题——建模帧间时空依赖（时间一致性）和抵抗未知退化（空间质量），而两者相互制约，增加了学习复杂度。

**本文目标**：设计一个既能保证帧间时空一致又能抵抗复杂退化的真实世界 VSR 框架。

**切入角度**：(1) Mamba 的线性复杂度全局注意力适合视频的 3D 时空建模；(2) 对比学习可以提取退化无关的特征表征，避免 LR 中的退化信息干扰条件注入。

**核心 idea**：将 Mamba 扩展到 3D 时空连续扫描实现帧内+帧间的全局注意力，并用 MoCo 式的对比学习训练 ControlNet 使其学会从 LR 中提取与 HR 对齐的干净特征。

## 方法详解

SCST 在预训练 Stable Diffusion 2.1 的基础上，引入两个核心模块：时空连续 Mamba（STCM）负责全局 3D 注意力，自监督 ControlNet（MoCoCtrl）负责从退化的 LR 中提取干净条件特征。两个模块通过精心设计的三阶段训练策略逐步引入。

### 整体框架

输入为 T 帧的低分辨率视频序列 $x^l \in \mathbb{R}^{T \times H \times W \times 3}$，输出为对应的高分辨率序列 $x^h \in \mathbb{R}^{T \times sH \times sW \times 3}$。框架基于 LDM，ControlNet 编码器 $E$ 提取 LR 的多尺度特征注入去噪 U-Net $D$ 中。U-Net 中嵌入 STCM 模块实现 3D 注意力，ControlNet 通过 MoCo 对比学习优化以抵抗退化。

### 关键设计

1. **时空连续 Mamba（STCM）**:

    - 功能：以线性复杂度实现视频帧内和帧间的全局 3D 注意力
    - 核心思路：核心组件是 3D-Mamba Block，使用 3D 深度卷积捕获时空依赖，然后通过 6 种扫描路径（3 种模式 × 2 方向，原始+翻转）处理特征序列。关键创新是**时空连续扫描策略**：与传统 3D 扫描（在帧间重置扫描状态）不同，该策略保持像素在帧内（橙色路径，逐像素顺序扫描）和帧间（青色路径，跟踪同一空间位置跨帧）的连续性。每个 patch 通过压缩的隐状态获取沿扫描路径的上下文知识，SSM 以线性复杂度处理。
    - 设计动机：全 3D 注意力（如某些视频生成方法）性能优秀但计算成本极高。解耦的空间+时间注意力计算量小但感受野受限。STCM 通过 Mamba 的线性复杂度实现全局 3D 注意力，在效率和效果之间取得平衡。连续扫描相比传统重置扫描能更好地保持像素级的时空一致性。

2. **自监督 ControlNet（MoCoCtrl）**:

    - 功能：从含退化的 LR 视频中提取与 HR 对齐的干净特征
    - 核心思路：采用类 MoCo 架构，使用查询编码器 $E_q$ 处理 LR 帧，动量编码器 $E_k$ 处理 HR 帧，$E_k$ 的权重是 $E_q$ 的指数移动平均（EMA）。通过投影头生成 $P \times P$ 的 patch 级特征，在 patch 级别计算对比损失 $\mathcal{L}_q = \frac{1}{P^2}\sum_p -\log \frac{\exp(q^p \cdot k_+^p / \tau)}{\exp(q^p \cdot k_+^p / \tau) + \sum_Q \exp(q^p \cdot Q / \tau)}$。负样本来自包含 HR 和 LR 编码的内存队列。
    - 设计动机：直接用退化的 LR 作为条件训练 ControlNet 会导致优化不稳定和伪影。通过对比学习，LR 编码器被迫学会产生与 HR 编码器输出对齐的特征，从而过滤掉退化噪声。传统 MoCo 用全局池化特征做分类，这里适配到 patch 级以捕获超分所需的空间细节。

3. **三阶段混合训练策略**:

    - 功能：逐步引入各模块，稳定训练过程
    - 核心思路：Stage 1——仅训练 ControlNet，使用 HR/LR 混合视频（HR 可视为零退化的 LR），混合比从 1.0 渐降到 0.3，让 ControlNet 先学会重建再适应退化。引入重建/SR 标签区分两种任务。Stage 2——引入 MoCoCtrl，HR/LR 比固定 1:1，对比学习充分利用 Stage 1 学到的重建先验。Stage 3——引入 STCM，冻结 ControlNet，仅用 LR 训练时间模块。
    - 设计动机：直接端到端训练所有模块会导致不稳定。先让 ControlNet 学会提取干净特征（Stage 1+2），再让时间模块在稳定的条件特征基础上学习时空一致性（Stage 3），解耦了两个难题。

### 损失函数 / 训练策略

主要损失为去噪损失 $\mathbb{E}_{t,x^h}\|D(x_t^h, t, E(x^l)) - \epsilon_t\|^2$ 加上 MoCoCtrl 的 patch 级对比损失。原始 SD 2.1 U-Net 权重冻结，仅训练新增层。8×A100 训练，Stage 1 约 12h，Stage 2 约 30h，Stage 3 约 30h。推理时序列长度 8 帧，20 步采样。

## 实验关键数据

### 主实验

| 数据集 | 指标 | RealBasicVSR | MGLD | Upscale-A-Video | **SCST** |
|--------|------|-------------|------|-----------------|----------|
| REDS4 | LPIPS↓ | 0.2545 | 0.2660 | 0.3639 | **0.2518** |
| REDS4 | DISTS↓ | 0.1196 | 0.1171 | 0.1840 | **0.1094** |
| UDM10 | LPIPS↓ | 0.2812 | 0.2551 | 0.2799 | **0.2156** |
| VideoLQ | CLIP-IQA↑ | 0.3881 | 0.3462 | 0.2818 | **0.4859** |
| VideoLQ | MUSIQ↑ | 55.61 | 50.94 | 43.34 | **59.20** |
| VideoLQ | NIQE↓ | 3.698 | 3.727 | 4.876 | **3.566** |

### 消融实验

| 模型 | MoCoCtrl | STCM | PSNR↑ | LPIPS↓ | DISTS↓ |
|------|----------|------|-------|--------|--------|
| (a) 基线 | ✗ | ✗ | 21.22 | 0.2824 | 0.1596 |
| (b) + MoCoCtrl | ✓ | ✗ | - | 提升 | 提升 |
| (c) + STCM | ✗ | ✓ | - | 提升 | 提升 |
| (d) 完整 | ✓ | ✓ | 24.31 | **0.2525** | **0.1344** |

### 关键发现

- SCST 在所有合成测试集上的感知指标（LPIPS、DISTS）均为最佳，在真实世界 VideoLQ 上的所有无参考指标也全部最优
- MoCoCtrl 和 STCM 各自都有独立贡献，组合使用效果叠加
- 在真实世界数据上的优势尤为突出——CLIP-IQA 领先第二名约 0.1（0.4859 vs 0.3881）
- 时空连续扫描优于传统的 flatten 重置扫描（消融验证）
- 定性对比显示 SCST 是唯一能清晰描绘鹰眼细节和轮胎纹理的方法

## 亮点与洞察

- 首次将 Mamba 引入视频超分辨率任务，时空连续扫描策略是有说服力的设计——保持帧内空间连续性和帧间像素追踪连续性
- MoCoCtrl 的 patch 级对比学习巧妙地复用了 MoCo 框架但适配到像素级恢复任务
- 三阶段训练策略设计合理——先学干净特征，再学退化鲁棒性，最后学时间一致性
- 在真实世界退化场景下的大幅领先验证了退化无关特征提取的重要性

## 局限与展望

- 推理时需要将视频分段处理（受内存限制），段间可能存在不一致
- 三阶段训练总计超过 72 小时（8×A100），训练成本较高
- 基于 SD 2.1 ，使用更新的基础模型（如 SDXL）可能带来进一步提升
- 仅评估了 4× 超分，对于其他尺度因子（如 2×、8×）的适用性未验证
- 时空连续扫描的 6 条路径可能有冗余，更经济的路径设计值得探索

## 相关工作与启发

- **Upscale-A-Video**：使用 3D 卷积+时间注意力，但感受野受限
- **MGLD**：使用光流对齐，时间一致性好但空间恢复不足
- **StableSR**：图像级 SR 直接用于视频导致时间不一致
- 启发：对比学习在底层视觉恢复任务中的应用（非高层语义），特征对齐思路可推广到去雨、去雾等其他恢复任务

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4 | 首个将Mamba引入VSR，MoCoCtrl设计巧妙 |
| 实验充分度 | 4 | 多数据集全面评估，含消融和定性对比 |
| 写作质量 | 4 | 框架清晰，图示丰富 |
| 实用价值 | 4 | 真实世界VSR场景下大幅领先 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution](../../NeurIPS2025/image_generation/dove_efficient_one-step_diffusion_model_for_real-world_video_super-resolution.md)
- [\[CVPR 2026\] STCDiT: Spatio-Temporally Consistent Diffusion Transformer for High-Quality Video Super-Resolution](../../CVPR2026/image_generation/stcdit_spatio-temporally_consistent_diffusion_transformer_for_high-quality_video.md)
- [\[CVPR 2025\] ORIDa: Object-Centric Real-World Image Composition Dataset](orida_object-centric_real-world_image_composition_dataset.md)
- [\[CVPR 2025\] Arbitrary-Steps Image Super-Resolution via Diffusion Inversion](arbitrary-steps_image_super-resolution_via_diffusion_inversion.md)
- [\[CVPR 2025\] UniReal: Universal Image Generation and Editing via Learning Real-world Dynamics](unireal_universal_image_generation_and_editing_via_learning_real-world_dynamics.md)

</div>

<!-- RELATED:END -->
