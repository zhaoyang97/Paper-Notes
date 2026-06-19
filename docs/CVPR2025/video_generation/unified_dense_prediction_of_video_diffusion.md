---
title: >-
  [论文解读] Unified Dense Prediction of Video Diffusion
description: >-
  [CVPR 2025][视频生成][密集预测] 提出 UDPDiff，首次在视频扩散模型中实现 RGB 视频生成与实体分割、深度估计的联合生成，通过 Pixelplanes 统一表示和可学习任务嵌入提升视频质量和一致性。 视频生成已取得显著进展，但现有模型仍面临帧间一致性问题（主体外观变化、背景不稳定、运动不自然等）…
tags:
  - "CVPR 2025"
  - "视频生成"
  - "密集预测"
  - "实体分割"
  - "深度估计"
  - "统一表示"
---

# Unified Dense Prediction of Video Diffusion

**会议**: CVPR 2025  
**arXiv**: [2503.09344](https://arxiv.org/abs/2503.09344)  
**代码**: 无  
**领域**: 视频生成  
**关键词**: 视频生成, 密集预测, 实体分割, 深度估计, 统一表示

## 一句话总结

提出 UDPDiff，首次在视频扩散模型中实现 RGB 视频生成与实体分割、深度估计的联合生成，通过 Pixelplanes 统一表示和可学习任务嵌入提升视频质量和一致性。

## 研究背景与动机

视频生成已取得显著进展，但现有模型仍面临帧间一致性问题（主体外观变化、背景不稳定、运动不自然等）。现有改进多聚焦于网络结构设计（如 3D VAE、MM-DiT），但缺乏显式的语义和几何推理信号。

REPA 等工作表明，与自监督方法对齐表示可加速扩散训练，但这些表示仍是隐式的。密集预测信号（分割提供物体形状和运动约束，深度提供空间位置感知）可作为显式训练信号。

**核心挑战**: (1) 不存在同时包含视频、分割和深度标注的大规模数据集；(2) 如何设计统一表示和架构，在不增加计算成本的前提下联合生成视频与多种密集预测结果。

此前图像级别的 UniGS 使用基于位置的 colormap 表示分割，但该方法无法处理视频中运动实体的颜色歧义问题。

## 方法详解

### 整体框架

UDPDiff 基于 CogVideoX 5B 构建。将视频潜在码 $z_t^v$ 与密集预测潜在码 $z_t^c$ 在通道维度拼接（共 32 通道）输入 Transformer 进行去噪。通过可学习任务嵌入 $e_\theta^d(d)$ 加到时间步嵌入上区分不同任务。输入输出通道翻倍，密集预测结果使用同一 3D VAE 编解码，推理时间几乎不增加。同时构建了 Panda-Dense 大规模数据集（约 300K 样本）。

### 关键设计1: Pixelplanes 统一表示

**功能**: 将实体分割和深度图编码为 RGB 图像，与视频共享同一 VAE。

**核心思路**: 对于实体分割，为每个实体随机采样 RGB 颜色 $M_c = (r_n, g_n, b_n)$，保证不同实体颜色不重复。对于深度图，使用光谱风格的值投影 $D_c = \Upsilon(D)$ 将单通道深度映射到 RGB 空间。两种任务统一为 RGB 格式后可直接使用 3D VAE 编解码。

**设计动机**: UniGS 的位置感知 colormap 使用固定颜色网格，基于实体质心坐标分配颜色。问题：(1) 固定网格在密集场景中不同实体被分配相同颜色；(2) 视频中实体运动导致质心变化，后续帧出现颜色歧义。随机颜色分配消除位置依赖，彻底避免运动歧义。

### 关键设计2: 可学习任务嵌入

**功能**: 在单一多任务模型中显式区分分割和深度估计任务。

**核心思路**: 定义任务嵌入层 $e_\theta^d$，接收任务 ID $d$ 作为输入，输出加到时间步嵌入 $e_\theta^t(t)$ 上：$t_d = e_\theta^d(d) + e_\theta^t(t)$。训练损失为标准扩散去噪损失 $\mathcal{L}_{\text{train}} = \frac{1}{2}\|f_\theta(z_t, t_d, c_t) - \epsilon\|^2$。推理时输入不同任务 ID 即可切换分割/深度生成。

**设计动机**: 仅用文本提示区分任务是隐式条件，容易产生语义歧义。可学习的任务嵌入提供显式的任务信号，让模型更准确地理解当前应执行的任务类型。

### 关键设计3: Panda-Dense 数据集构建

**功能**: 提供大规模视频+分割+深度标注训练数据。

**核心思路**: 从 Panda-70M 采样约 300K 视频子集。分割标注流程：(1) 用 EntitySeg CropFormer 对首帧做实体分割；(2) 用 SAM2 将分割结果传播到全视频。深度标注使用 DepthCrafter 生成一致的视频深度图。使用 13B Video-LLaVA 重新生成详细文本描述。

**设计动机**: 现有数据集不同时包含视频、分割和深度。EntitySeg 确保分割粒度一致性（避免 SAM 点网格初始化导致的过细/过粗问题），DepthCrafter 保证帧间深度一致性（逐帧深度估计会产生抖动）。

### 损失函数

标准扩散去噪 MSE 损失：$\mathcal{L}_{\text{train}} = \frac{1}{2}\|f_\theta(z_t, t_d, c_t) - \epsilon\|^2$。多任务训练时按任务 ID 切换，联合优化任务嵌入和生成模型参数。

## 实验关键数据

### 主实验结果 (多任务模型 vs CogVideoX 5B)

| 模型 | SC↑ | BC↑ | MS↑ | FVD↓ |
|------|-----|-----|-----|------|
| CogVideoX 5B | 94.57 | 95.80 | 97.67 | 343.92 |
| UDPDiff (seg) | 95.21 | 95.69 | 98.24 | 316.76 |
| UDPDiff (depth) | **97.07** | **96.89** | **99.23** | **302.55** |

*SC=主体一致性, BC=背景一致性, MS=运动平滑度*

### 消融实验

| 方法 | SC↑ | BC↑ | MS↑ |
|------|-----|-----|-----|
| Location-aware colormap (UniGS) | 81.26 | 79.33 | 88.79 |
| **Pixelplanes** | **94.98** | **95.92** | **98.62** |

| 任务区分方式 | SC↑ | BC↑ | MS↑ | FVD↓ |
|-------------|-----|-----|-----|------|
| Text prompt | 95.17 | 95.78 | 98.67 | 321.43 |
| **Task embedding** | **97.07** | **96.89** | **99.23** | **302.55** |

### 关键发现

1. **密集预测显著提升一致性**: 多任务 UDPDiff (depth) 在所有指标上全面超越 CogVideoX，FVD 降低 41.37（相对降低 12%）。
2. **Pixelplanes 远优于 UniGS colormap**: SC 从 81.26 提升至 94.98（+13.72），证明随机颜色方案消除位置歧义的有效性。
3. **任务嵌入优于文本提示**: FVD 从 321.43 降至 302.55，显式任务条件更有效。
4. **几乎零推理开销**: 单任务模型 205.75s vs 原始 CogVideoX 204.46s，增加不到 1%。
5. **多任务优于单任务**: 联合训练的多任务模型优于单独训练分割/深度模型，分割和深度提供互补信号。

## 亮点与洞察

- **统一范式新颖**: 首次将视频级生成和密集预测统一为同一扩散过程，密集预测作为"免费"副产品输出。
- **互利关系**: 密集预测不仅是输出，更是训练信号——帮助视频生成模型学习更好的场景理解。
- **实用价值**: 一次推理同时获得视频、分割和深度，对下游视频编辑任务极有价值。

## 局限与展望

- **数据规模有限**: 仅 300K 样本训练，深度估计精度（$\delta_1=0.4176$）与专用模型 Depth Anything V2（$\delta_1=0.5808$）有差距。
- **3D VAE 限制**: 分割和深度以 RGB colormap 形式编解码，VAE 的压缩损失可能影响精确度。
- **仅两种任务**: 未探索更多密集预测任务（如光流、法线估计）的联合训练。
- 未来可扩展数据规模、增加更多密集预测任务、探索密集预测结果作为可控条件进行编辑。

## 相关工作与启发

- **UniGS**: 图像级 colormap 表示的开创者，本文将其扩展到视频级并解决了运动歧义问题。
- **Marigold/SemFlow**: 扩散模型做密集预测的代表工作，但仅限于单任务单图。
- **启发**: "生成即理解"的范式——通过联合训练密集预测来提升生成质量，可推广到 3D 生成等领域。

## 评分

⭐⭐⭐⭐ — 首次在视频扩散中实现生成与密集预测联合训练，Pixelplanes 设计简洁有效。多任务提升生成质量的实验结论有说服力。数据集构建和实验全面。深度估计精度与专用模型存在差距是主要不足。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] OCK: Unsupervised Dynamic Video Prediction with Object-Centric Kinematics](../../ICCV2025/video_generation/ock_unsupervised_dynamic_video_prediction_with_object-centric_kinematics.md)
- [\[ICLR 2026\] Lumos-1: On Autoregressive Video Generation with Discrete Diffusion from a Unified Model Perspective](../../ICLR2026/video_generation/lumos-1_on_autoregressive_video_generation_with_discrete_diffusion_from_a_unifie.md)
- [\[ICCV 2025\] FuXi-RTM: A Physics-Guided Prediction Framework with Radiative Transfer Modeling](../../ICCV2025/video_generation/fuxi-rtm_a_physics-guided_prediction_framework_with_radiative_transfer_modeling.md)
- [\[CVPR 2025\] World-Consistent Video Diffusion with Explicit 3D Modeling](world-consistent_video_diffusion_with_explicit_3d_modeling.md)
- [\[ICCV 2025\] WorldScore: A Unified Evaluation Benchmark for World Generation](../../ICCV2025/video_generation/worldscore_a_unified_evaluation_benchmark_for_world_generation.md)

</div>

<!-- RELATED:END -->
