---
title: >-
  [论文解读] SteerX: Creating Any Camera-Free 3D and 4D Scenes with Geometric Steering
description: >-
  [ICCV 2025][视频生成][3D场景生成] SteerX 提出了一种零样本推理时引导方法，通过将场景重建融入视频生成过程中，利用无需相机参数的前馈重建模型设计几何奖励函数，引导生成分布朝向更好的几何一致性，实现了高质量的无相机条件 3D/4D 场景生成。 领域现状：3D/4D 场景生成是计算机视觉的热门方向…
tags:
  - "ICCV 2025"
  - "视频生成"
  - "3D场景生成"
  - "4D场景生成"
  - "几何引导"
  - "推理时引导"
---

# SteerX: Creating Any Camera-Free 3D and 4D Scenes with Geometric Steering

**会议**: ICCV 2025  
**arXiv**: [2503.12024](https://arxiv.org/abs/2503.12024)  
**代码**: [https://github.com/byeongjun-park/SteerX](https://github.com/byeongjun-park/SteerX)  
**领域**: 视频生成  
**关键词**: 3D场景生成, 4D场景生成, 几何引导, 推理时引导, 视频生成

## 一句话总结

SteerX 提出了一种零样本推理时引导方法，通过将场景重建融入视频生成过程中，利用无需相机参数的前馈重建模型设计几何奖励函数，引导生成分布朝向更好的几何一致性，实现了高质量的无相机条件 3D/4D 场景生成。

## 研究背景与动机

**领域现状**：3D/4D 场景生成是计算机视觉的热门方向，目前主流做法是先用视频生成模型生成多视角视频，再用重建模型从视频中恢复 3D/4D 场景。这两个阶段——视频生成和场景重建——通常被独立优化。

**现有痛点**：现有方法在视频生成和场景重建两个阶段分别提升几何一致性，但一个阶段的细微不对齐很难在另一个阶段被修复。例如，视频生成模型可能产生微小的多视角不一致，而重建模型无法纠正这种累积误差；反之，重建阶段的优化也无法反馈到生成阶段。这种"各自为战"的策略导致最终 3D/4D 场景的几何质量受限。

**核心矛盾**：视频生成和场景重建被视为两个独立的优化问题，缺乏一个统一的框架来同时约束两者的几何对齐。

**本文目标**：设计一种方法能够在视频生成的推理过程中直接引入场景重建的几何约束，使生成的视频天然具有更好的几何一致性。

**切入角度**：作者观察到，如果能在视频生成过程中实时评估当前帧序列的几何质量（通过快速前馈重建），就可以将这个评估信号作为奖励来引导采样过程。这类思路在 LLM 领域已有先例（如 RLHF），但在 3D 生成中尚未被充分探索。

**核心 idea**：利用无需相机参数的前馈 3D 重建模型（如 MEt3R）设计几何奖励函数，通过 Feynman-Kac 引导框架在推理时将数据分布"倾斜"向几何对齐更好的样本，从而统一生成与重建。

## 方法详解

### 整体框架

SteerX 的输入是文本描述或单张图像，输出是几何一致的多视角视频（可进一步重建为 3D/4D 场景）。整个流程基于扩散模型的去噪过程：在每个去噪步骤中，SteerX 生成多个候选样本（粒子），用几何奖励函数评估每个粒子的几何质量，然后通过重采样保留高奖励的粒子继续去噪。这是一种零样本方法，不需要额外训练，适用于任意视频生成模型。

### 关键设计

1. **GS-MEt3R 几何奖励（静态 3D 场景）**:

    - 功能：评估生成视频帧之间的 3D 几何一致性
    - 核心思路：将中间生成的视频帧送入 MEt3R（无需相机参数的前馈 3D 高斯重建模型），得到 3D 高斯表示，然后将 3D 高斯渲染回各视角图像。比较原始生成帧与渲染帧之间的 DINO 特征相似度作为奖励。具体来说，先用 DINO 提取原始帧和渲染帧的特征图，再对特征图进行余弦相似度计算，上采样后取平均值作为最终奖励分数。
    - 设计动机：如果生成的多视角视频几何一致性好，那么从它们重建出的 3D 高斯再渲染回来的图像应该与原始帧高度相似。DINO 特征对语义信息敏感，比像素级比较更鲁棒。

2. **Dyn-MEt3R 几何奖励（动态 4D 场景）**:

    - 功能：评估动态场景中背景区域的几何一致性
    - 核心思路：将视频帧分为两半，用 MEt3R 从前半部分帧重建 3D 点云，提取背景区域的 DINO 特征并反投影到 3D 空间，然后将这些 3D 特征投影到后半部分帧上，计算投影特征与后半部分帧原始特征之间的相似度作为奖励。
    - 设计动机：动态场景中前景物体在运动，直接做全局一致性检查不合理。但背景应该在不同帧之间保持一致，因此通过检查背景特征的跨帧一致性来衡量几何质量。

3. **Feynman-Kac 几何引导**:

    - 功能：在扩散模型推理时利用奖励信号引导采样
    - 核心思路：基于 Feynman-Kac 粒子引导框架，在去噪过程中维护多个粒子（候选样本）。每隔若干步计算一次几何奖励，根据奖励值对粒子进行加权重采样——高奖励的粒子被复制，低奖励的被淘汰。通过反复重采样，数据分布逐渐向几何一致性高的方向倾斜。
    - 设计动机：这种方法不需要修改生成模型的权重，完全在推理时操作，因此是零样本的、即插即用的。相比梯度引导方法，粒子引导避免了梯度计算的高成本和不稳定性。

### 损失函数 / 训练策略

SteerX 是一种推理时方法，不涉及额外训练。核心的"损失"是几何奖励函数：$R = \frac{1}{N}\sum_{i=1}^{N} \text{cos\_sim}(\phi(I_i), \phi(\hat{I}_i))$，其中 $\phi$ 是 DINO 特征提取器，$I_i$ 和 $\hat{I}_i$ 分别是原始帧和渲染帧。

## 实验关键数据

### 主实验

SteerX 在 3D 和 4D 场景生成上均取得了显著提升，支持多种视频生成骨干网络。

| 方法 | CLIP-Score ↑ | 3D一致性 ↑ | 用户偏好 ↑ | 骨干网络 |
|------|-------------|-----------|-----------|---------|
| Wan2.1 (基线) | 0.287 | 0.71 | 23% | Wan2.1 |
| SteerX + Wan2.1 | **0.301** | **0.82** | **77%** | Wan2.1 |
| CogVideoX (基线) | 0.275 | 0.68 | 19% | CogVideoX |
| SteerX + CogVideoX | **0.293** | **0.79** | **81%** | CogVideoX |

### 消融实验

| 配置 | 几何一致性 ↑ | 视觉质量 ↑ | 说明 |
|------|------------|-----------|------|
| Full SteerX | 0.82 | 0.94 | 完整模型（GS-MEt3R + FK引导） |
| w/o 几何奖励 | 0.71 | 0.93 | 去掉奖励后退化为原始生成 |
| w/o 重采样 | 0.74 | 0.92 | 只计算奖励但不做粒子重采样 |
| 像素级奖励替代DINO | 0.76 | 0.88 | 用MSE替代DINO特征相似度 |
| 粒子数=2 | 0.77 | 0.93 | 减少粒子数量 |
| 粒子数=8 | 0.83 | 0.94 | 增加粒子但收益递减 |

### 关键发现

- 几何奖励函数是核心贡献，去掉后几何一致性大幅下降（0.82→0.71），说明原始视频生成模型确实存在显著的几何不一致问题
- DINO 特征比像素级 MSE 更适合作为奖励信号，因为 DINO 对小的外观变化更鲁棒而对几何错误更敏感
- 粒子数量 4 是性价比最优的选择，继续增加粒子改善有限但计算成本线性增长
- SteerX 对不同视频生成骨干（Wan2.1、CogVideoX 等）均有效，体现了框架的通用性

## 亮点与洞察

- **推理时引导范式**：将几何奖励引入扩散模型的采样过程，是一种非常优雅的设计。无需重训练模型，即可让任意视频生成器产生几何一致的输出。这种"即插即用"思路可以迁移到其他需要满足特定约束的生成任务中。
- **利用前馈重建作为评估器**：MEt3R 等无需相机参数的前馈重建模型速度快，作为奖励函数的核心组件非常合适。这种"用重建质量评估生成质量"的思路避免了需要 GT 数据。
- **3D 与 4D 统一框架**：通过设计两种不同的奖励函数（GS-MEt3R 对应静态场景，Dyn-MEt3R 对应动态场景），实现了 3D 和 4D 场景生成的统一处理。

## 局限与展望

- 推理成本随粒子数量线性增加，4 个粒子意味着生成时间约为原来的 4 倍
- 几何奖励依赖于 MEt3R 的重建质量，如果重建模型本身在某些场景上失效，奖励信号也不可靠
- 目前仅在较短的视频（几十帧）上验证，长视频场景下粒子引导的效果尚不清楚
- 背景一致性奖励（Dyn-MEt3R）假设背景是静态的，对于动态背景场景可能不适用

## 相关工作与启发

- **vs DreamFusion/Score Jacobian Chaining**: 这些方法用 SDS loss 从 2D 扩散模型蒸馏 3D 内容，但需要逐场景优化。SteerX 是零样本的，速度快得多。
- **vs ViewCrafter**: ViewCrafter 通过 fine-tune 视频模型来理解 3D，但失去了通用性。SteerX 不修改模型权重，保持了通用性。
- **vs Feynman-Kac Steering (SVDD)**: SteerX 的引导框架基于此工作，但创新在于设计了适用于 3D/4D 场景的几何奖励函数。

## 评分

- 新颖性: ⭐⭐⭐⭐ 将推理时引导引入 3D/4D 生成是新颖的，但 Feynman-Kac 框架本身已有先例
- 实验充分度: ⭐⭐⭐⭐ 覆盖了 3D 和 4D 场景、多种骨干网络、消融实验充分
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐ 即插即用的框架设计实用性强，但推理成本倍增限制了实际应用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] ReDirector: Creating Any-Length Video Retakes with Rotary Camera Encoding](../../CVPR2026/video_generation/redirector_creating_any-length_video_retakes_with_rotary_camera_encoding.md)
- [\[ICCV 2025\] Free-Form Motion Control: Controlling the 6D Poses of Camera and Objects in Video Generation](free-form_motion_control_controlling_the_6d_poses_of_camera_and_objects_in_video.md)
- [\[CVPR 2025\] VideoScene: Distilling Video Diffusion Model to Generate 3D Scenes in One Step](../../CVPR2025/video_generation/videoscene_distilling_video_diffusion_model_to_generate_3d_scenes_in_one_step.md)
- [\[ICCV 2025\] DACoN: DINO for Anime Paint Bucket Colorization with Any Number of Reference Images](dacon_dino_for_anime_paint_bucket_colorization_with_any_number_of_reference_imag.md)
- [\[ICCV 2025\] ReCamMaster: Camera-Controlled Generative Rendering from A Single Video](recammaster_camera-controlled_generative_rendering_from_a_single_video.md)

</div>

<!-- RELATED:END -->
