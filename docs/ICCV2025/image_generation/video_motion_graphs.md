---
title: >-
  [论文解读] Video Motion Graphs
description: >-
  [ICCV 2025][图像生成][视频运动图] Video Motion Graphs 提出了一个基于检索+生成的通用人体运动视频系统，通过将参考视频构建为运动图结构并进行条件化路径搜索获取关键帧，再利用 HMInterp（一个双分支扩散帧插值模型，结合运动扩散模型的骨骼引导和渐进式条件训练）来无缝连接不连续帧，在多种条件（音乐、语音、动作标签）下生成高质量人体运动视频，显著优于生成式和检索式基线。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "视频运动图"
  - "视频帧插值"
  - "运动扩散模型"
  - "检索式视频生成"
  - "人体运动视频"
---

# Video Motion Graphs

**会议**: ICCV 2025  
**arXiv**: [2503.20218](https://arxiv.org/abs/2503.20218)  
**代码**: 无（Adobe Research）  
**领域**: 扩散模型 / 视频生成  
**关键词**: 视频运动图, 视频帧插值, 运动扩散模型, 检索式视频生成, 人体运动视频

## 一句话总结

Video Motion Graphs 提出了一个基于检索+生成的通用人体运动视频系统，通过将参考视频构建为运动图结构并进行条件化路径搜索获取关键帧，再利用 HMInterp（一个双分支扩散帧插值模型，结合运动扩散模型的骨骼引导和渐进式条件训练）来无缝连接不连续帧，在多种条件（音乐、语音、动作标签）下生成高质量人体运动视频，显著优于生成式和检索式基线。

## 研究背景与动机

**领域现状**：人体运动视频生成主要分为两条路线——(1) 生成式方法从条件输入直接合成所有像素，灵活但易产生肢体扭曲等伪影；(2) 检索式方法利用参考视频的真实帧保证质量，但需要帧插值模型平滑过渡

**现有痛点**：
   - 生成式方法即使使用 DiT 架构（如 SVD），仍难以避免手部和面部的结构错误
   - 现有检索式方法（GVR、TANGO）仅为对话手势设计，使用线性混合运动引导，无法处理舞蹈等复杂动态运动
   - 线性插值仅能合理近似 78% 的温和手势动作，但对复杂舞蹈动作仅 17% 可行

**核心矛盾**：检索式方法在视频质量上有天然优势（直接使用真实帧），但其帧插值模块成为瓶颈——现有方案无法处理非线性的大幅度运动

**本文目标**：设计一个通用的人体运动视频生成系统，支持多种条件输入（音乐、语音、动作标签），同时保证视频的纹理质量和运动轨迹准确性

**切入角度**：用 Motion Diffusion Model 替代线性混合来生成运动引导，结合渐进式条件训练策略解决身份一致性问题

**核心 idea**：双分支帧插值——运动扩散模型保证骨骼运动轨迹正确，扩散式 VFI 保证视频纹理质量，两者通过渐进式条件训练融合

## 方法详解

### 整体框架

Video Motion Graphs 是一个四阶段系统：
1. **图初始化**：将参考视频表示为有向图，节点为帧，边表示可平滑过渡的帧对（基于 3D 姿态距离阈值）
2. **路径搜索**：给定条件信号（音乐/语音/标签），通过动态规划或 Beam Search 找到最优帧路径
3. **帧插值**：使用 HMInterp 平滑不连续帧边界（生成 12 帧 = 0.5s@24fps）
4. **背景重组**（可选）：对动态背景视频进行前景分离和背景生成

### 关键设计

1. **运动扩散模型 (MDM)**:

    - 功能：在起始帧和结束帧之间生成插值的 2D 关节位置序列
    - 核心思路：基于 Transformer 的去噪网络，但改造为 UNet 风格（加入 skip connection 和特征拼接），从浅到深层融合特征，生成更精确的非线性运动插值轨迹
    - 设计动机：相比线性混合，MDM 能处理鼓击、舞蹈等高动态动作的非线性轨迹；相比原始 MDM 使用的 8 层 vanilla Transformer，UNet 结构保留更多运动细节

2. **视频帧插值 (VFI) Backbone**:

    - 功能：以起始/结束帧和 MDM 生成的姿态为条件，生成插值视频帧
    - 核心思路：基于 AnimateDiff 的 UNet T2V 模型，加入 ReferenceNet 注入分层外观特征、Seed Image Guider 注入起始/结束帧的 VAE latent、Pose Guider 注入 MDM 生成的 2D 姿态。将 CLIP 文本编码器替换为图像编码器
    - 改进的 Reference Decoder：基于 ToonCrafter 的 Reference Decoder（用 SVD 的 temporal decoder 初始化），在 VAE 解码时注入低层 latent 特征保留面部细节。创新地使用重复填充的参考帧代替零填充，PSNR 提升超过 1.0

3. **渐进式条件训练 (Condition Progressive Training)**:

    - 功能：分阶段训练 VFI 模块，先学身份条件再学姿态条件
    - 核心思路：阶段1—Seed Pre-Training 仅用图像条件训练 100k steps，确保插值帧忠实于参考外观；阶段2—Few-Step Pose Finetuning 同时使用图像+姿态条件训练 8k steps
    - 设计动机：直接联合训练图像和姿态条件会导致生成帧与真实帧的外观不一致。实验发现交换训练顺序（先姿态后图像）、或延长姿态微调步数都会损害身份保持能力

### 损失函数 / 训练策略

- VFI 模块使用 v-prediction 训练
- MDM 使用 $x_0$-prediction 训练
- Reference Decoder 使用 MSE + 感知损失训练
- MDM 和 Reference Decoder 先分别训练再冻结

## 实验关键数据

### 主实验

**人体运动视频生成质量对比（Tab.1）**:

| 方法 | PSNR↑ | LPIPS↓ | MOVIE↓ | FVD↓ |
|------|-------|--------|--------|------|
| AnimateAnyone | 35.55 | 0.044 | 54.68 | 1.369 |
| MagicPose | 35.64 | 0.048 | 51.97 | 1.277 |
| UniAnimate | 36.75 | 0.042 | 49.89 | 1.090 |
| MimicMotion | 36.30 | 0.047 | 46.84 | 1.078 |
| **Ours (f=32)** | **42.91** | **0.009** | **37.31** | **0.180** |
| Ours (f=64) | 42.75 | 0.010 | 37.53 | 0.213 |
| Ours (f=216) | 39.75 | 0.029 | 39.89 | 0.799 |

即使在最困难的 f=216 设置下，仍优于所有 pose2video 方法。

**用户研究胜率（Tab.2）**:

| 对比维度 | vs Dance | vs Gesture | vs Action |
|---------|----------|-----------|-----------|
| 纹理质量 | 82.10% | 78.38% | 69.12% |
| 跨模态对齐 | 88.39% | 47.63% | 45.21% |
| 整体偏好 | 84.99% | 70.24% | 61.05% |

### 消融实验

**HMInterp 模块消融（Tab.5）**:

| 配置 | PSNR↑ | LPIPS↓ | MOVIE↓ | FVD↓ |
|------|-------|--------|--------|------|
| **HMInterp (s=1)** | **39.53** | **0.034** | **39.18** | **1.210** |
| w/o 运动引导 | 39.17 | 0.048 | 41.34 | 1.391 |
| 线性运动引导 | 39.16 | 0.042 | 41.06 | 1.297 |
| w/o Reference Decoder | 37.21 | 0.039 | 49.67 | 1.283 |
| 零填充 Reference Decoder | 38.13 | 0.034 | 40.11 | 1.221 |

**渐进式条件训练消融（Tab.6）**:

| Stage 1 | Stage 2 | PSNR↑ | LPIPS↓ | FVD↓ |
|---------|---------|-------|--------|------|
| P (pose-to-video) | - | 35.55 | 0.044 | 1.369 |
| P+SI 同时 | - | 36.62 | 0.041 | 1.325 |
| **SI** | **P+SI (8k)** | **37.21** | **0.039** | **1.283** |
| SI | P+SI (30k) | 36.87 | 0.041 | 1.307 |

### 关键发现

- **运动引导是关键**：去掉 MDM 运动引导后 LPIPS 从 0.034 恶化到 0.048，说明模型容易从错误区域获取特征。MDM 引导的非线性轨迹远优于线性混合
- **Reference Decoder 的改进**：重复填充参考帧比零填充在低分辨率（256×256）下显著改善面部和背景细节
- **渐进式训练的必要性**：过早引入姿态条件会破坏外观一致性。先学"像谁"再学"怎么动"是关键
- **参考视频长度影响**：100s 的参考数据库即可超越当前生成模型，1000s 数据库时运动多样性接近真实视频

## 亮点与洞察

- **检索+生成的混合范式** — 只在少数帧上使用生成模型，大部分帧来自真实视频，这种策略在当前生成模型质量仍不完美时极具实用价值。可迁移到其他"质量优先"的视频编辑场景
- **渐进式条件训练** — 解决了多条件联合训练中身份一致性问题的通用策略，思路（先学强条件再学弱条件）可直接迁移到其他 pose-to-video 或 audio-to-video 任务
- **MDM 的 UNet 化改造** — 在扩散 Transformer 中加入 skip connection 是一个简单但有效的改进，适用于需要细节保留的生成任务

## 局限与展望

- 对动态背景的处理依赖额外的背景分离/生成模块，非端到端
- 手势和动作标签任务中跨模态对齐偏好仅约 45-47%，说明在非舞蹈场景的条件对齐仍有提升空间
- 系统工程复杂度高，四阶段 pipeline 不够优雅
- 目标条件信号的搜索算法为启发式规则，可能不是最优

## 相关工作与启发

- **vs GVR/TANGO**: 它们只能处理对话手势的线性运动插值，限于静态背景。Video Motion Graphs 通过 MDM + 渐进训练扩展到通用人体运动
- **vs DanceAnyBeat**: 生成式舞蹈方法在用户研究中纹理质量远低于 Video Motion Graphs（82.10% 偏好），凸显检索式方法的质量优势
- **vs DCInterp/ACInterp**: 使用线性运动引导的扩散帧插值方法，在动态场景下因运动轨迹错误导致帧质量下降

## 评分

- 新颖性: ⭐⭐⭐⭐ 运动图+扩散帧插值的组合是新颖的工程系统，渐进训练策略有洞察
- 实验充分度: ⭐⭐⭐⭐⭐ 包含客观指标、82人用户研究、多任务评估、详细消融
- 写作质量: ⭐⭐⭐⭐ 系统描述清晰，但整体 pipeline 较复杂
- 价值: ⭐⭐⭐⭐ 实用性强，支持实时生成和关键帧编辑，有工业应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video](bitrate-controlled_diffusion_for_disentangling_motion_and_content_in_video.md)
- [\[ICCV 2025\] REDUCIO! Generating 1K Video within 16 Seconds using Extremely Compressed Motion Latents](reducio_generating_1k_video_within_16_seconds_using_extremely_compressed_motion_.md)
- [\[CVPR 2025\] EDEN: Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation](../../CVPR2025/image_generation/eden_enhanced_diffusion_for_high-quality_large-motion_video_frame_interpolation.md)
- [\[ICCV 2025\] InfiniDreamer: Arbitrarily Long Human Motion Generation via Segment Score Distillation](infinidreamer_arbitrarily_long_human_motion_generation_via_segment_score_distill.md)
- [\[CVPR 2025\] GraphGPT-o: Synergistic Multimodal Comprehension and Generation on Graphs](../../CVPR2025/image_generation/graphgpt-o_synergistic_multimodal_comprehension_and_generation_on_graphs.md)

</div>

<!-- RELATED:END -->
