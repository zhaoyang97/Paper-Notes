---
title: >-
  [论文解读] SVFR: A Unified Framework for Generalized Video Face Restoration
description: >-
  [CVPR 2025][图像生成][视频人脸修复] 本文提出 SVFR，一个基于 Stable Video Diffusion 的统一视频人脸修复框架，将盲人脸修复（BFR）、着色和修复三个任务纳入同一模型中联合训练，通过任务嵌入、统一隐空间正则化和面部先验学习等设计，在多个视频人脸修复任务上取得 SOTA 效果。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "视频人脸修复"
  - "多任务学习"
  - "扩散模型"
  - "时序一致性"
  - "统一框架"
---

# SVFR: A Unified Framework for Generalized Video Face Restoration

**会议**: CVPR 2025  
**arXiv**: [2501.01235](https://arxiv.org/abs/2501.01235)  
**代码**: [https://github.com/wangzhiyaoo/SVFR](https://github.com/wangzhiyaoo/SVFR)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 视频人脸修复, 多任务学习, 扩散模型, 时序一致性, 统一框架

## 一句话总结
本文提出 SVFR，一个基于 Stable Video Diffusion 的统一视频人脸修复框架，将盲人脸修复（BFR）、着色和修复三个任务纳入同一模型中联合训练，通过任务嵌入、统一隐空间正则化和面部先验学习等设计，在多个视频人脸修复任务上取得 SOTA 效果。

## 研究背景与动机

**领域现状**：人脸修复（Face Restoration）是图像/视频处理中的重要领域，旨在从退化输入中恢复高质量人像。图像级别的人脸修复已有大量工作（如 CodeFormer、GPEN、GFPGAN），但视频级别的人脸修复仍然相对欠探索。

**现有痛点**：视频人脸修复面临三大核心挑战：(1) 时序维度带来的额外复杂性——需要在多帧间保持一致性，同时处理运动伪影、遮挡和光照变化；(2) 高质量视频训练数据稀缺，难以训练鲁棒模型；(3) 现有方法（如 BasicVSR++、KEEP）架构容量有限，生成质量低，存在严重的时域抖动和不连续问题。

**核心矛盾**：传统人脸修复方法通常针对单一任务（如超分辨率）单独训练，忽略了 BFR、着色、修复之间共享的先验知识。实际上，低质量视频的颜色退化与着色任务密切相关，遮挡区域修复与修复任务目标一致，这些任务之间存在天然的互补性。

**本文目标**：设计一个统一框架，同时处理视频 BFR、着色和修复三个任务，利用任务间的共享表示来增强监督信号，提高整体修复质量和时序稳定性。

**切入角度**：作者首先通过先导实验（Pilot Study）验证了多任务迁移学习的有效性：用 GPEN 分别在有/无预训练先验的情况下训练三个任务，结果表明任务间的先验知识确实能互相提升 FID 指标。

**核心 idea**：基于 Stable Video Diffusion 构建统一的多任务视频人脸修复框架（GVFR），通过任务嵌入区分不同任务、统一隐空间正则化对齐跨任务特征、面部先验学习注入人脸结构信息，三者协同实现高质量且时序稳定的视频修复。

## 方法详解

### 整体框架
SVFR 基于预训练的 Stable Video Diffusion（SVD）构建。输入为退化源视频 $\mathbf{V}_d$（低质量视频/灰度视频/带遮罩的视频），输出为修复后的视频 $\mathbf{V}_r$。源视频经 VAE 编码后与噪声拼接，送入扩散 U-Net 进行去噪。整体 pipeline 包含三个核心模块：(1) 统一人脸修复框架（任务嵌入 + 统一隐空间正则化）；(2) 面部先验学习；(3) 自参考精炼策略。

### 关键设计

1. **任务嵌入与统一隐空间正则化（Unified Face Restoration Framework）**:

    - 功能：使模型能够区分不同任务（BFR/着色/修复），并在统一的特征空间中对齐来自不同任务的中间特征。
    - 核心思路：任务嵌入将每个任务表示为二值向量 $\gamma = [t_1, t_2, t_3]$（如 $[0,1,1]$ 表示着色+修复任务激活），通过嵌入层映射后加到 U-Net 的时间嵌入上。统一隐空间正则化（ULR）通过对比学习损失约束 U-Net 中间层特征：同一视频的不同退化形式作为正样本对，不同视频作为负样本对，使模型学习任务无关的共享特征表示。ULR 损失为 $\mathcal{L}_{ULR} = -\log \frac{\exp(\hat{x}_i \cdot \hat{x}_i^+ / \tau)}{\sum_{j=1}^N \exp(\hat{x}_i \cdot \hat{x}_j^- / \tau)}$。
    - 设计动机：不同任务的退化源视频携带不同的先验信息（BFR 保留结构但缺纹理，修复任务在未遮挡区域保留完整信息），简单拼接无法将它们编码到一致的隐空间。任务嵌入提供显式任务标识，避免模型混淆；ULR 确保跨任务的特征一致性，促进知识迁移。

2. **面部先验学习（Facial Prior Learning）**:

    - 功能：将人脸结构先验注入预训练的 SVD 中，引导模型生成结构一致的面部细节。
    - 核心思路：从 U-Net 中间块提取特征 $x_d$，通过一个由平均池化层和五层 MLP 组成的 landmark 预测器 $P_{lm}$ 预测 68 个面部关键点。使用混合损失函数 $\mathcal{L}_{prior}$：小偏差时用对数函数 $w \ln(1 + |x|/\epsilon)$ 精确对齐，大偏差时用线性函数 $|x| - C$ 保持鲁棒性。Ground truth landmark 由预训练检测模型从 GT 帧中提取。
    - 设计动机：SVD 原始训练目标（噪声预测）不包含人脸结构约束，直接微调容易生成结构不一致的面部（如变形的眼睛、嘴巴）。辅助 landmark 预测目标迫使模型从噪声隐变量中学习人脸结构先验。

3. **自参考精炼策略（Self-referred Refinement）**:

    - 功能：在长视频推理时维持跨片段的时序一致性和风格连贯性。
    - 核心思路：训练阶段随机提供参考帧 $I_{ref}$，其 VAE 编码注入 U-Net 初始噪声，身份特征通过映射网络注入交叉注意力层，50% 的概率 dropout 参考帧以增强泛化。推理阶段先无参考帧生成第一个片段，然后从中选取一帧作为后续片段的参考，确保长序列中的风格和结构连续性。
    - 设计动机：视频人脸修复常需处理长视频，分段生成时段间容易出现颜色漂移、身份不一致等问题。自参考机制通过前段生成结果引导后段，实现全局一致。

### 损失函数 / 训练策略
总损失为三项加权组合：$\mathcal{L} = \mathcal{L}_{noise} + \lambda_1 \mathcal{L}_{ULR} + \lambda_2 \mathcal{L}_{prior}$，其中 $\lambda_1 = 0.01$、$\lambda_2 = 0.1$。训练数据源自 VoxCeleb2、CelebV-Text 和 VFHQ，经 ARNIQA 评分筛选出 20,000 个高质量视频片段。

## 实验关键数据

### 主实验
在 VFHQ-test 数据集上与 SOTA 方法对比，SVFR 作为统一模型同时处理三个任务，其他方法需为每个任务单独训练。

| 方法 | 任务 | PSNR↑ | SSIM↑ | LPIPS↓ | IDS↑ | FVD↓ |
|------|------|-------|-------|--------|------|------|
| GPEN | BFR | 26.237 | 0.795 | 0.320 | 0.786 | 412.81 |
| CodeFormer | BFR | 26.528 | 0.762 | 0.361 | 0.784 | 379.53 |
| PGTFormer | BFR | 28.996 | 0.843 | 0.248 | 0.845 | 154.86 |
| KEEP | BFR | 27.335 | 0.813 | 0.259 | 0.790 | 399.24 |
| **SVFR** | **BFR** | **29.563** | **0.862** | **0.223** | **0.902** | **89.32** |
| **SVFR** | **着色** | **23.079** | **0.896** | **0.272** | **0.980** | **204.26** |
| **SVFR** | **修复** | **29.119** | **0.904** | **0.153** | **0.888** | **88.35** |

### 消融实验

| 配置 | PSNR (BFR)↑ | FVD (BFR)↓ | PSNR (着色)↑ | FVD (修复)↓ |
|------|------------|-----------|------------|-----------|
| 单任务训练 | 28.323 | 167.31 | 22.233 | 106.52 |
| 多任务训练 | 28.936 | 98.78 | 22.921 | 101.15 |
| +ULR | 29.296 | 90.35 | 22.987 | 93.62 |
| +ULR+FPL (完整) | **29.563** | **89.32** | **23.079** | **88.35** |

### 关键发现
- 多任务联合训练相比单任务训练在所有指标上均有提升，特别是 FVD 从 167.31 降至 98.78（BFR），验证了任务间知识共享的有效性
- ULR 在多任务基础上进一步提升性能，尤其在 LPIPS 和 FVD 上改善明显，表明特征空间对齐对生成质量至关重要
- 面部先验学习对 BFR 和修复任务贡献最大（PSNR 提升约 0.3-0.8），因为这两个任务更依赖人脸结构准确性
- 自参考精炼策略在长视频上效果显著，有效消除了颜色漂移和身份不一致的问题

## 亮点与洞察
- **多任务互助的验证路径**：通过先导实验（Pilot Study）简洁有力地证明了 BFR、着色、修复三个任务之间确实存在可迁移的共享先验，为后续统一框架设计提供了坚实的实验基础——这种"先验证再设计"的研究范式值得借鉴
- **统一隐空间正则化的对比学习设计**：不同于简单的特征拼接或共享权重，ULR 用对比学习显式约束同一视频不同退化形式的特征相似性，既保留了任务特异性又促进了跨任务一致性
- **自参考精炼策略**：训练时随机 dropout 参考帧的设计非常巧妙，使模型同时具备无参考和有参考生成的能力，推理时自然过渡

## 局限与展望
- 训练数据仅 20,000 个视频片段，在极端退化（如极低分辨率+遮挡+灰度同时存在）场景下的泛化能力有待验证
- 方法依赖预训练的人脸关键点检测器提供 GT landmark，当退化非常严重时 GT 提取可能不准确
- 推理速度受限于扩散模型的多步去噪过程，难以实时应用
- 当前仅支持三种退化任务，可以考虑扩展到更多视频人脸处理任务（如去模糊、去雾等）

## 相关工作与启发
- **vs KEEP**: KEEP 是专门针对视频 BFR 的方法，采用独立训练策略。SVFR 通过多任务学习获得更好的 FVD（89.32 vs 399.24），证明了联合训练对时序稳定性的巨大提升
- **vs CodeFormer**: CodeFormer 是图像级方法逐帧处理，缺乏时序建模。SVFR 基于 SVD 天然具备时序先验，在 VIDD 和 FVD 上优势明显
- **vs PGTFormer**: PGTFormer 在 BFR 上效果不错但不支持着色和修复任务，SVFR 一个模型覆盖三个任务且 BFR 性能更优

## 评分
- 新颖性: ⭐⭐⭐⭐ 多任务统一视频人脸修复的想法有新意，但各个技术组件（对比学习、landmark 监督）相对成熟
- 实验充分度: ⭐⭐⭐⭐ 三个任务的对比和消融实验完整，时域稳定性可视化分析到位
- 写作质量: ⭐⭐⭐⭐ 先导实验-方法设计的逻辑清晰，结构合理
- 价值: ⭐⭐⭐⭐ 建立了视频人脸修复的新范式，开源代码将推动该方向发展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] OSDFace: One-Step Diffusion Model for Face Restoration](osdface_one-step_diffusion_model_for_face_restoration.md)
- [\[ICCV 2025\] MoFRR: Mixture of Diffusion Models for Face Retouching Restoration](../../ICCV2025/image_generation/mofrr_mixture_of_diffusion_models_for_face_retouching_restoration.md)
- [\[CVPR 2025\] OFER: Occluded Face Expression Reconstruction](ofer_occluded_face_expression_reconstruction.md)
- [\[ICCV 2025\] Unlocking the Potential of Diffusion Priors in Blind Face Restoration](../../ICCV2025/image_generation/unlocking_the_potential_of_diffusion_priors_in_blind_face_restoration.md)
- [\[CVPR 2025\] Pursuing Temporal-Consistent Video Virtual Try-On via Dynamic Pose Interaction](pursuing_temporal-consistent_video_virtual_try-on_via_dynamic_pose_interaction.md)

</div>

<!-- RELATED:END -->
