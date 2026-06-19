---
title: >-
  [论文解读] VEGGIE: Instructional Editing and Reasoning Video Concepts with Grounded Generation
description: >-
  [ICCV 2025][语义分割][视频编辑] VEGGIE 提出了一个端到端统一框架，将 MLLM 与视频扩散模型连接，仅用扩散损失就能在单一模型中同时完成指令式视频编辑、概念定位和推理分割等 8 种任务。 视频编辑方法在近年取得长足进步，但距"简单、通用的视频概念编辑器"目标仍有三个核心挑战： 非端到端：大多数方法需要中…
tags:
  - "ICCV 2025"
  - "语义分割"
  - "视频编辑"
  - "指令编辑"
  - "视频概念定位"
  - "推理分割"
  - "多任务统一模型"
---

# VEGGIE: Instructional Editing and Reasoning Video Concepts with Grounded Generation

**会议**: ICCV 2025  
**arXiv**: [2503.14350](https://arxiv.org/abs/2503.14350)  
**代码**: [https://veggie-gen.github.io/](https://veggie-gen.github.io/) (项目页)  
**领域**: 语义分割  
**关键词**: 视频编辑, 指令编辑, 视频概念定位, 推理分割, 多任务统一模型

## 一句话总结

VEGGIE 提出了一个端到端统一框架，将 MLLM 与视频扩散模型连接，仅用扩散损失就能在单一模型中同时完成指令式视频编辑、概念定位和推理分割等 8 种任务。

## 研究背景与动机

视频编辑方法在近年取得长足进步，但距"简单、通用的视频概念编辑器"目标仍有三个核心挑战：

**非端到端**：大多数方法需要中间步骤——布局/掩码/人工标注或模型生成的描述作为引导（如 VPLM 需要 mask 输入、tool-use 方法需要多步 pipeline），这增加了用户负担，破坏了无缝编辑体验

**训练目标复杂**：现有连接 MLLM 和 VidDM 的 pipeline 需要多种训练目标（语言损失、掩码损失等），增加了优化难度和超参数调优成本

**任务覆盖不全**：现有模型只擅长部分编辑任务。例如 LGVI 擅长去除但不支持风格化，VidToMe 擅长全局编辑但本地编辑差，TokenFlow 不支持添加/删除物体

这些挑战的根源有二：(1) 缺乏覆盖广泛技能的高质量多任务视频编辑训练数据；(2) 模型缺乏两个关键能力——**多模态推理**（从指令推断编辑意图）和**语言定位**（精确定位要编辑的区域）。

VEGGIE 的核心 idea：将视频定位和编辑统一为**像素空间的端到端生成式任务**。不需要额外的检测/分割模块，不需要中间文本 token 作为条件，仅用扩散损失就能同时学习编辑、定位和推理。关键创新是用**连续可学习的 grounded task query embeddings** 替代离散文本 token 作为 MLLM 到 VidDM 的桥梁，实现梯度端到端传播。

## 方法详解

### 整体框架

VEGGIE 由四个组件组成：

1. **MLLM**：接受视频帧序列 $V = [f_1, ..., f_n]$ 和用户指令 $I$，生成逐帧的 grounded task tokens $C = [c_1, ..., c_n]$
2. **可学习的 Grounded Task Queries**：连续嵌入向量，每帧一组，作为 MLLM 的输入和输出并行处理
3. **Alignment Network**：单层 MLP，将 MLLM 输出映射到扩散模型的条件空间
4. **Video Diffusion Model**：从指令式图像编辑模型初始化，接受原始视频（拼接到噪声）和 task tokens（送入 cross-attention）生成编辑后视频

关键区别于先前方法：task query 是连续的、可微的，梯度可以从 VidDM 反传到 MLLM，实现真正的端到端训练。而 VPLM 等方法使用离散文本 token，切断了梯度流。

### 关键设计

1. **Curriculum Learning（课程学习）两阶段训练**：

    - 功能：先在图像级别对齐 MLLM 和 VidDM，再在视频级别端到端微调
    - Stage 1（图像-语言空间对齐）：MLLM 冻结，更新 alignment network、task queries 和扩散 UNet。使用 340 万图像编辑数据训练，让扩散模型学会理解 MLLM 生成的 task guidance
    - Stage 2（视频时序增强）：全部组件解冻（包括 MLLM），2D UNet 通过 temporal attention layers 膨胀为 3D。使用 13.6 万视频编辑数据端到端微调
    - 设计动机：直接在视频数据上端到端训练会导致模型崩溃（MLLM 和 VidDM 表示空间不对齐）。先用大量图像数据预对齐是必要的

2. **统一的任务表示（Unified Task Formulation）**：

    - 功能：将编辑和定位/分割统一为 video-to-video 生成任务
    - 核心思路：
        - 视频编辑：输出是编辑后的视频帧
        - 视频概念定位：输出是高亮目标物体的视频帧（颜色填充）
        - 推理分割：输出是分割掩码可视化的视频帧
        - 所有任务共享同一个像素空间输出格式，只需扩散损失
    - 设计动机：避免为不同任务设计不同的 head 和损失函数，极大简化训练流程

3. **数据合成 Pipeline（VEG-Edit）**：

    - 功能：将高质量图像编辑数据提升为视频编辑数据
    - 核心流程：
        - 输入：原始图像 $I$、编辑后图像 $\bar{I}$、编辑指令
        - Step 1：MLLM 生成图像描述和动画 prompt
        - Step 2：Image-to-Video 模型将 $I$ 动画化为视频 $V$
        - Step 3：First-frame-conditioned 视频编辑模型利用 $\bar{I}$ 生成编辑后视频 $\bar{V}$
        - Step 4：自动视频质量评估过滤低质量样本
    - 设计动机：高质量标注的视频编辑数据极其稀缺，但图像编辑数据丰富（如 MagicBrush、Seed-Data-Edit 共 340 万）。通过 I2V 生成+质量过滤可以高效扩展视频训练数据

### 损失函数 / 训练策略

- 全程仅使用**扩散损失**，不引入语言损失或掩码损失
- Classifier-Free Guidance：测试时对 task tokens 和输入视频两路条件分别做 CFG：
    - $\tilde{e_\theta}(z_t, c_T, c_V) = e_\theta(z_t, \varnothing, \varnothing) + g_T \cdot (e_\theta(z_t, c_V, c_T) - e_\theta(z_t, c_V, \varnothing)) + g_V \cdot (e_\theta(z_t, c_V, \varnothing) - e_\theta(z_t, \varnothing, \varnothing))$
- 数据：Stage 1 共 340 万图像对（Seed-Data-Edit 300 万 + 分割/推理数据），Stage 2 共 13.6 万视频对

## 实验关键数据

### 主实验

VEGGIE vs 基线方法在 VEG-Bench 上 8 种编辑技能的表现（部分关键技能）：

| 能力 | 指标 | VidToMe | TokenFlow | InsV2V | LGVI | **VEGGIE** |
|------|------|---------|-----------|--------|------|-----------|
| 概念添加 | MLLM-Judge↑ | 5.00 | 5.80 | 5.69 | 2.73 | **7.44** |
| 概念添加 | Detection↑ | 47.98 | 49.53 | 48.01 | 14.42 | **57.96** |
| 概念去除 | MLLM-Judge↑ | 2.60 | 3.73 | 2.78 | 6.59 | 5.07 |
| 概念去除 | Detection↑ | 34.31 | 55.16 | 25.64 | **78.40** | 70.22 |
| 物体替换 | MLLM-Judge↑ | 5.00 | 6.53 | 6.60 | 2.06 | **6.63** |
| 视频定位 | mIoU↑ | 0.00 | 0.00 | 0.00 | 0.00 | **47.30** |
| 推理分割 | mIoU↑ | 0.00 | 0.00 | 0.00 | 0.00 | **32.80** |

VEGGIE 是唯一能同时完成所有 8 种任务的方法。其他基线在定位和推理分割任务上完全失败（mIoU=0）。

### 消融实验

| 配置 | Add Judge↑ | Remove Detect↑ | Grounding mIoU↑ | 说明 |
|------|-----------|----------------|-----------------|------|
| w/o 视频数据 (Stage 2) | 6.80 | 65.10 | 40.20 | 仅图像预训练 |
| w/o 图像数据 (Stage 1) | 5.20 | 55.30 | 35.10 | 直接在视频上训练 |
| w/o 定位数据 | 7.10 | 68.50 | 0.00 | 编辑能力存在但无法定位 |
| w/o 编辑数据 | 3.20 | 30.20 | 45.80 | 可定位但编辑崩塌 |
| **VEGGIE (full)** | **7.44** | **70.22** | **47.30** | 多任务互相促进 |

### 关键发现

- **多任务互促效应**：定位数据帮助编辑模型更精确地识别编辑区域，编辑数据帮助定位模型理解物体语义。去掉任一类数据都会导致相关和无关任务的性能下降
- VEGGIE 在添加/替换/风格化等创造性编辑任务上显著优于指令式基线 InsV2V，同时在去除任务上接近专注去除的 LGVI
- **零样本涌现能力**：VEGGIE 展现出多模态指令跟随能力（从参考图添加物体/迁移风格）和少样本上下文编辑能力（通过示例对学习编辑模式），这些能力未被显式训练

## 亮点与洞察

1. **"仅用扩散损失"统一多任务**是一个优雅的设计——避免了多损失函数的权重调优问题，证明了像素空间生成可以天然承载分割、定位等理解任务
2. **连续 task query 替代离散文本 token** 是实现端到端的关键设计选择。这解决了 MLLM→VidDM pipeline 中梯度断裂的老问题
3. 课程学习的必要性被实验清晰验证：跳过 Stage 1 直接在视频上训练会导致模型崩溃
4. 数据合成 pipeline（图像→视频提升）为视频编辑领域的数据稀缺问题提供了可扩展的解决方案
5. VEG-Bench 首次提供了覆盖 8 种编辑技能的统一评估基准

## 局限与展望

- 基于 SD1.5 的 UNet 限制了视频质量和分辨率上限，未来可迁移到 SD3/FLUX 等更强的基础模型
- 去除任务上仍落后于专注去除的 LGVI（70.22 vs 78.40 Detection），说明统一模型在细分任务上可能存在天花板
- 所有基线的时序平滑度(Smoothness)普遍很高(>94)，说明该指标区分度有限
- 推理分割的 mIoU(32.8) 相比图像领域方法仍有较大差距，视频推理分割仍是开放问题

## 相关工作与启发

- InstructPix2Pix 的指令式编辑范式被扩展到视频领域
- LISA 的推理分割思路被统一到生成框架中，不再需要专门的分割 head
- 数据合成策略（高质量图像数据 → I2V → 视频数据）对视频领域的数据瓶颈具有通用参考价值
- 对于 MLLM 与生成模型结合的其他任务（如 3D 生成、音频编辑），"连续 query + 课程学习"的框架可能同样适用

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Online Reasoning Video Segmentation with Just-in-Time Digital Twins](online_reasoning_video_segmentation_with_just-in-time_digital_twins.md)
- [\[ICCV 2025\] Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation](towards_omnimodal_expressions_and_reasoning_in_referring_audio-visual_segmentati.md)
- [\[ICLR 2026\] RegionReasoner: Region-Grounded Multi-Round Visual Reasoning](../../ICLR2026/segmentation/regionreasoner_region-grounded_multi-round_visual_reasoning.md)
- [\[ICLR 2026\] VINCIE: Unlocking In-context Image Editing from Video](../../ICLR2026/segmentation/vincie_unlocking_in-context_image_editing_from_video.md)
- [\[ICCV 2025\] Latent Expression Generation for Referring Image Segmentation and Grounding](latent_expression_generation_for_referring_image_segmentation_and_grounding.md)

</div>

<!-- RELATED:END -->
