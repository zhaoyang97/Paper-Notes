---
title: >-
  [论文解读] Learning Temporally Consistent Video Depth from Video Diffusion Priors
description: >-
  [CVPR 2025][视频生成][视频深度估计] 提出 ChronoDepth——基于 Stable Video Diffusion (SVD) 的视频深度估计方法，通过在训练时为每帧独立采样噪声水平并在推理时使用无噪声前序帧作为上下文（Consistent Context-Aware Strategy），在保持空间精度的同时实现了 SOTA 的时序一致性，MFC 指标平均排名第一。
tags:
  - "CVPR 2025"
  - "视频生成"
  - "视频深度估计"
  - "时序一致性"
  - "视频扩散模型"
  - "上下文感知推理"
  - "SVD微调"
---

# Learning Temporally Consistent Video Depth from Video Diffusion Priors

**会议**: CVPR 2025  
**arXiv**: [2406.01493](https://arxiv.org/abs/2406.01493)  
**代码**: [https://xdimlab.github.io/ChronoDepth](https://xdimlab.github.io/ChronoDepth)  
**领域**: 视频生成  
**关键词**: 视频深度估计, 时序一致性, 视频扩散模型, 上下文感知推理, SVD微调

## 一句话总结
提出 ChronoDepth——基于 Stable Video Diffusion (SVD) 的视频深度估计方法，通过在训练时为每帧独立采样噪声水平并在推理时使用无噪声前序帧作为上下文（Consistent Context-Aware Strategy），在保持空间精度的同时实现了 SOTA 的时序一致性，MFC 指标平均排名第一。

## 研究背景与动机

1. **领域现状**：单帧深度估计近年取得巨大进步（Marigold、DepthAnything），但受限于逐帧独立预测的 IID 假设，生成的深度视频存在闪烁和时序不一致。判别式视频深度方法（如 NVDS、RNN-based）要么依赖精确相机位姿，要么时序一致性仍不理想。
2. **现有痛点**：
    - 视频扩散模型天然具有时序建模能力，但直接将单帧扩散深度模型扩展到视频并不 trivial。
    - 处理任意长视频需要分 clip 推理，关键挑战是如何在 clip 之间传递上下文信息。
    - 已有的"replacement trick"（给前序帧加噪声后拼入当前 clip）在数学上不严谨——每步采样时加的噪声不同导致上下文信息不一致。
3. **核心矛盾**：如何在 clip 之间共享一致的上下文信息，使长视频深度估计在时序维度上真正连贯？
4. **本文目标** 设计能保证跨 clip 一致上下文的训练和推理策略。
5. **切入角度**：受 Diffusion Forcing 启发，训练时为每帧独立采样噪声水平，使模型学会在不同噪声水平帧共存时去噪→推理时前序帧设为近零噪声（已预测结果），后续帧设为正常噪声水平。
6. **核心 idea**：独立帧级噪声训练 + 无噪声前序帧上下文推理 = 跨 clip 一致的视频深度估计。

## 方法详解

### 整体框架
基于 SVD（Stable Video Diffusion）的 image-to-video 变体。RGB 视频通过 VAE 编码器映射到潜在空间，与深度潜在拼接后送入 UNet 去噪。训练分两阶段：先训练空间层（单帧深度），再冻结空间层只训练时序层（多帧 clip）。推理时用滑窗策略处理任意长度视频，前 W 帧用已预测深度初始化（不加噪声），后 F-W 帧从高斯噪声开始去噪。

### 关键设计

1. **帧级独立噪声采样训练（Per-Frame Independent Noise Levels）**:

    - 功能：使模型学会处理同一 clip 中不同帧处于不同噪声水平的情况，为推理时的上下文注入策略打基础。
    - 核心思路：标准 video diffusion 为整个 clip 采样同一个噪声水平 $\sigma_t$。ChronoDepth 改为每帧独立采样：$\boldsymbol{\sigma}_t = [\sigma_1, \sigma_2, ..., \sigma_F]$，其中 $\log \sigma_i \sim \mathcal{N}(P_{mean}, P_{std}^2)$ 独立采样。预处理函数 $c_{skip}, c_{out}, c_{in}, c_{noise}$ 也相应地逐帧应用。训练损失保持 DSM 形式不变。
    - 设计动机：这是实现 consistent context-aware 推理的关键前提——只有模型学会了"有些帧几乎是干净的（已预测结果），有些帧充满噪声（待预测）"这种混合情况，推理时才能正确利用前序帧的上下文。

2. **一致上下文感知推理（Consistent Context-Aware Inference）**:

    - 功能：在 clip 之间传递一致的深度信息，消除跨 clip 的闪烁和尺度跳变。
    - 核心思路：使用滑窗策略处理长视频，每个 clip 有 F 帧，相邻 clip 重叠 W 帧。第一个 clip 用标准去噪（所有帧从纯噪声开始）。后续 clip 中，前 W 帧直接用前一 clip 已预测的深度潜在（不加噪声），后 F-W 帧从纯噪声开始。去噪器的噪声条件为 $\boldsymbol{\sigma}_t = [\underbrace{\sigma_\epsilon, ..., \sigma_\epsilon}_W, \underbrace{\sigma_t, ..., \sigma_t}_{F-W}]$，其中 $\sigma_\epsilon$ 是极小的噪声值。
    - 设计动机：相比 replacement trick（给前序帧加大噪声），本方法保持了上下文在每个采样步的一致性——前序帧始终是同一个干净预测，不会因为每步加的噪声不同而引入不一致。$\sigma_\epsilon$ 设为极小但非零值，因为前序预测不是 GT，需要保留少量不确定性以缓解长程累积误差。

3. **顺序空间-时序微调策略（Sequential Spatial-Temporal Fine-Tuning）**:

    - 功能：分阶段微调空间层和时序层，避免联合训练中空间精度和时序一致性的冲突。
    - 核心思路：第一阶段——冻结时序层，用单帧深度数据集（Hypersim，39K 样本）训练空间层 20K 步。第二阶段——冻结空间层，用多帧视频数据集（TartanAir + Virtual KITTI2 + MVS-Synth，共 938 视频序列）训练时序层 18K 步，clip 长度随机采样 $F \in [1, F_{max}]$（$F_{max}=5$）。贯穿两阶段都使用单帧数据。
    - 设计动机：联合训练容易让时序层"偷懒"依赖空间层，或者时序目标干扰空间精度。分阶段训练让空间层先达到最佳空间精度，再让时序层专注于学习时序一致性。随机 clip 长度作为数据增强，增强模型对不同运动速度的鲁棒性。

### 损失函数 / 训练策略
- 使用 EDM 框架的 DSM 损失：$\mathcal{L} = \mathbb{E}[\lambda(\boldsymbol{\sigma}_t) \|\hat{\mathbf{z}}_0 - \mathbf{z}_0\|_2^2]$，权重函数 $\lambda(\sigma) = (1+\sigma^2)\sigma^{-2}$。
- RGB 条件通过通道拼接引入（禁用 SVD 原有的 cross-attention 条件）。
- 深度图三通道复制模拟 RGB 以复用 VAE，解码后取三通道均值。
- Adam 优化器，学习率 3e-5，8 × A100-80GB，总训练约 1.5 天。

## 实验关键数据

### 主实验

**零样本视频深度估计（MFC=Multi-Frame Consistency，越低越好）：**

| 方法 | KITTI-360 AbsRel/MFC | ScanNet++ AbsRel/MFC | Sintel AbsRel/MFC | MFC 平均排名 |
|------|---------------------|---------------------|-------------------|-------------|
| Marigold | 0.213 / 0.776 | 0.192 / 0.109 | 0.573 / 1.112 | 4.00 |
| DepthAnything V2 | 0.207 / 0.807 | 0.170 / 0.103 | 0.387 / 1.504 | 5.00 |
| DepthCrafter | 0.293 / 0.655 | 0.199 / 0.094 | 0.374 / 1.270 | 3.00 |
| **ChronoDepth** | 0.215 / **0.407** | 0.176 / **0.092** | 0.493 / **0.728** | **1.00** |

### 消融实验

**推理策略对比（时序一致性 MFC↓）：**

| 推理策略 | MFC | 说明 |
|---------|-----|------|
| Naive sliding window | 高 | 无上下文传递 |
| Replacement trick | 中 | 上下文不一致 |
| **Consistent context-aware** | **最低** | 一致上下文 |

**训练策略对比：**

| 配置 | AbsRel | MFC | 说明 |
|------|--------|-----|------|
| Joint spatial+temporal | 较高 | 较高 | 联合训练相互干扰 |
| **Sequential (spatial→temporal)** | **较低** | **最低** | 分阶段最优 |
| 固定 clip 长度 | 中等 | 中等 | 缺乏鲁棒性 |
| **随机 clip 长度** | **较低** | **较低** | 数据增强效果 |

### 关键发现
- **MFC 指标全面 SOTA**：ChronoDepth 在三个 benchmark 上的 MFC 均为最低，平均排名第 1，大幅领先 DepthAnything V2（排名 5.00）和 Marigold（排名 4.00）。
- **空间精度略有代价**：AbsRel 排名 4.00（vs DepthAnything V2 的 1.67），说明为了时序一致性牺牲了部分空间精度。但这个 tradeoff 对视频应用更合理。
- **Replacement trick 的问题被定量验证**：与 DepthCrafter（使用 replacement trick）相比，ChronoDepth 在 KITTI-360 上 MFC 从 0.655 降到 0.407，验证了一致上下文的重要性。
- **顺序训练是关键**：先训练空间层再训练时序层的策略比联合训练在时序一致性上好很多，且不损失空间精度。
- **$\sigma_\epsilon$ 不能为零**：极小但非零的噪声水平比零噪声效果更好，因为它承认了前序预测的不确定性。

## 亮点与洞察
- **Consistent context-aware 策略的优雅设计**：通过训练时的"每帧独立噪声"这一小改动，自然地支持了推理时"干净帧+噪声帧共存"的推理模式，不需要修改网络架构。这个 trick 可迁移到任何需要自回归处理的视频扩散模型任务。
- **Replacement trick 的问题被真正理解和解决**：之前的工作（包括 DepthCrafter）只是启发式地使用 replacement trick 而未认识到其数学上的不严谨性。本文明确指出问题根源（每步噪声不同导致条件不一致）并给出简洁的解决方案。
- **顺序训练策略的发现**：联合训练空间+时序不如分阶段训练这一发现，对其他视频生成/理解任务也有参考意义——让模型先学空间再学时序可能是更好的课程学习策略。

## 局限与展望
- **空间精度不是最优**：AbsRel 排名第 4，落后于 DepthAnything V2 和 DepthAnything。可能可以通过更大的单帧训练数据集或更强的 foundation model 改善。
- **推理速度受扩散步数限制**：需要多步去噪，比判别式方法慢。可以考虑引入 flow matching 或一致性蒸馏加速。
- **$F_{max}=5$ 的训练限制**：最长训练 clip 只有 5 帧，推理时用 10 帧会有 train-test gap。更长的训练 clip（需要更多显存）可能进一步提升。
- **依赖 SVD 的 VAE**：深度图三通道复制是 workaround，不如为深度定制的 VAE 高效。
- **累积误差**：虽然 $\sigma_\epsilon$ 缓解了长程误差，但对超长视频（数千帧）的累积误差仍缺乏分析。

## 相关工作与启发
- **vs DepthCrafter [Hu et al.]**: 都是基于视频扩散模型，但 DepthCrafter 用 replacement trick 传递上下文，导致跨 clip 不一致。ChronoDepth 的一致上下文策略从根本上解决了这个问题。
- **vs Marigold [Ke et al.]**: Marigold 是单帧扩散深度估计的先驱，ChronoDepth 可看作其视频版本的进阶——不仅继承了扩散模型的空间精度，还通过视频扩散先验获得了时序一致性。
- **vs Diffusion Forcing [Chen et al.]**: Diffusion Forcing 提出了类似的"每帧独立噪声"概念但用于生成任务且基于 RNN。ChronoDepth 将其应用于确定性视频深度估计任务，基于 attention 架构，不需要额外隐状态。

## 评分
- 新颖性: ⭐⭐⭐⭐ 独立帧噪声+一致上下文推理的思路新颖，但核心概念受 Diffusion Forcing 启发
- 实验充分度: ⭐⭐⭐⭐⭐ 三个 benchmark、多种推理策略对比、训练策略对比、y-t slice 可视化，非常充分
- 写作质量: ⭐⭐⭐⭐⭐ 三种推理策略的对比解释清晰，Algorithm 1-3 的伪代码让方法一目了然
- 价值: ⭐⭐⭐⭐ 对视频深度估计的时序一致性有显著贡献，一致上下文策略可迁移到其他视频扩散任务

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] NormalCrafter: Learning Temporally Consistent Normals from Video Diffusion Priors](../../ICCV2025/video_generation/normalcrafter_learning_temporally_consistent_normals_from_video_diffusion_priors.md)
- [\[CVPR 2025\] VideoGuide: Improving Video Diffusion Models without Training Through a Teacher's Guide](videoguide_improving_video_diffusion_models_without_training_through_a_teachers_.md)
- [\[CVPR 2025\] StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text](streamingt2v_consistent_dynamic_and_extendable_long_video_generation_from_text.md)
- [\[CVPR 2025\] Optical-Flow Guided Prompt Optimization for Coherent Video Generation](optical-flow_guided_prompt_optimization_for_coherent_video_generation.md)
- [\[CVPR 2025\] Tracktention: Leveraging Point Tracking to Attend Videos Faster and Better](tracktention_leveraging_point_tracking_to_attend_videos_faster_and_better.md)

</div>

<!-- RELATED:END -->
