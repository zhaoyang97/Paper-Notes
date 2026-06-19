---
title: >-
  [论文解读] DynaRend: Learning 3D Dynamics via Masked Future Rendering for Robotic Manipulation
description: >-
  [NeurIPS 2025][3D视觉][表征学习] 提出 DynaRend，通过掩码重建和未来预测两个互补目标，利用可微体渲染在 triplane 表征上联合学习 3D 几何、语义和动态信息，预训练后可高效迁移到下游机器人操控任务。 学习可泛化的机器人操控策略面临真实训练数据匮乏的核心瓶颈。现有自监督预训练方法存在三类局限…
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "表征学习"
  - "机器人操控"
  - "可微渲染"
  - "Triplane"
  - "动态预测"
  - "预训练"
---

# DynaRend: Learning 3D Dynamics via Masked Future Rendering for Robotic Manipulation

**会议**: NeurIPS 2025  
**arXiv**: [2510.24261](https://arxiv.org/abs/2510.24261)  
**代码**: 未开源  
**领域**: 3D视觉  
**关键词**: 表征学习, 机器人操控, 可微渲染, Triplane, 动态预测, 预训练  

## 一句话总结

提出 DynaRend，通过掩码重建和未来预测两个互补目标，利用可微体渲染在 triplane 表征上联合学习 3D 几何、语义和动态信息，预训练后可高效迁移到下游机器人操控任务。

## 背景与动机

学习可泛化的机器人操控策略面临真实训练数据匮乏的核心瓶颈。现有自监督预训练方法存在三类局限：

1. **2D 视觉预训练**（如 MAE、对比学习）：只能捕获静态语义，缺乏 3D 几何理解和动态建模
2. **视频预测模型**（如 VPP、VidMan）：利用大规模视频学习 2D 动态，但缺乏显式 3D 空间感知
3. **显式 3D 动态**（如 ManiGaussian）：用动态高斯学习空间和动态信息，但表示复杂度高，难以与下游策略集成，且需要大量标定新视角监督

核心思路：用**可微体渲染**作为桥梁，在紧凑的 triplane 表征上同时学习几何（通过 RGB+深度）、语义（通过视觉基础模型蒸馏）和动态（通过未来帧预测），三者统一于一个预训练框架。

## 核心问题

如何设计一个表征学习框架，在不依赖大量标注和密集视角的前提下，联合学习 3D 感知的几何、语义和动态表征，并高效迁移到机器人操控？

## 方法详解

### 3D 场景 Triplane 表示

从多视角 RGB-D 图像重建点云，经 MLP 编码后通过轴对齐最大池化投影到三个正交平面：

$$\mathbf{f}_{xy} \in \mathbb{R}^{H \times W \times C}, \quad \mathbf{f}_{xz} \in \mathbb{R}^{H \times D \times C}, \quad \mathbf{f}_{yz} \in \mathbb{R}^{W \times D \times C}$$

Triplane 分辨率为 $16 \times 16 \times 16$，兼顾效率和表达力。

### 掩码未来预测

随机掩码部分 triplane 特征，替换为可学习嵌入。结合 CLIP 编码的语言指令，依次通过：

1. **重建网络** $\mathcal{E}_{\text{recon}}$：恢复当前场景完整 3D 表征 $\mathcal{V}_{\text{now}}$
2. **预测网络** $\mathcal{E}_{\text{pred}}$：预测最近未来关键帧的 3D 表征 $\mathcal{V}_{\text{future}}$

两个网络均采用 4 层 Transformer（含 SwiGLU、QK Norm、RoPE）。

### 可微体渲染监督

对当前和未来 triplane 分别独立渲染。沿射线采样 $N$ 个点，在三个平面上双线性插值查询特征 $\mathbf{v}_i$，经 MLP 头解码为密度 $\sigma$、RGB $\mathbf{c}$ 和语义特征 $\mathbf{s}$：

$$\hat{\mathbf{C}}(\mathbf{r}) = \sum_{i=1}^N w_i \mathbf{c}(\mathbf{v}_i, \mathbf{d}), \quad \hat{\mathbf{D}}(\mathbf{r}) = \sum_{i=1}^N w_i t_i$$

其中 $w_i = T_i(1 - \exp(\sigma(\mathbf{v}_i)\delta_i))$，$T_i = \exp(-\sum_{j=1}^{i-1} \sigma(\mathbf{v}_j)\delta_j)$。

语义特征用 RADIOv2.5 提取的特征监督，深度用 SiLog 损失优化。

### 视角增强

利用预训练生成模型 See3D 从已有视角合成新视角 RGB-D 对作为额外监督，解决真实场景中相机视角有限的问题。

### 预训练与微调损失

预训练损失：$\mathcal{L}_{\text{pretrain}} = \lambda_{\text{recon}} \mathcal{L}_{\text{recon}} + \lambda_{\text{pred}} \mathcal{L}_{\text{pred}}$

微调时，triplane 特征经卷积+上采样生成动作热力图，用 cross-entropy 监督平移、旋转和夹爪状态：

$$\mathcal{L}_{\text{finetune}} = \lambda_{\text{trans}} \text{CE}(\mathbf{a}_{\text{trans}}, \hat{\mathbf{a}}_{\text{trans}}) + \lambda_{\text{rot}} \text{CE}(\mathbf{a}_{\text{rot}}, \hat{\mathbf{a}}_{\text{rot}}) + \lambda_{\text{gripper}} \text{CE}(\mathbf{a}_{\text{gripper}}, \hat{\mathbf{a}}_{\text{gripper}})$$

## 实验关键数据

### RLBench 18 任务

| 方法 | 平均成功率↑ | 平均排名↓ | 推理速度 |
|------|-----------|----------|---------|
| PerAct | 49.4 | 5.1 | 4.9 FPS |
| RVT | 62.9 | 4.3 | 11.6 FPS |
| 3D-MVP | 67.5 | 3.2 | 11.6 FPS |
| 3D Diffuser Actor | 81.3 | 2.2 | 1.4 FPS |
| RVT-2 | 81.4 | 2.2 | 20.6 FPS |
| **DynaRend** | **83.2** | **1.5** | **19.6 FPS** |

相比 RVT 基线，成功率提升 32.3%（62.9→83.2）。

### Colosseum 泛化性测试

DynaRend 在 12 种环境扰动（颜色、纹理、大小、光照变化）下均展现出优于基线的鲁棒性。

### 真实世界实验

在 5 个真实任务上验证了有效性和实用性。

### 关键消融

- 去掉预测目标（仅重建）：成功率显著下降，证明动态建模的重要性
- 去掉掩码策略：成功率下降，掩码重建迫使模型学习更完整的 3D 表征
- 视角增强对真实世界部署至关重要

## 亮点

1. **三合一预训练**：几何、语义、动态统一于体渲染监督框架，设计优雅
2. **轻量高效**：triplane 表示紧凑，19.6 FPS 推理速度接近最快的 RVT-2
3. **视角合成增强**：用生成模型合成新视角，解决真实场景标定视角不足的痛点
4. 预训练仅需 ~60k 步，微调 ~30k 步，训练开销可控

## 局限与展望

- triplane 分辨率固定 $16^3$，对精细操控可能不足
- 预训练需要多视角 RGB-D 输入，单臂相机设置下需额外深度传感器
- See3D 视角增强引入的伪标签质量对最终性能的影响需要更多分析
- 仅评估了关键帧操控范式，未涉及连续控制

## 与相关工作的对比

- **vs 3D-MVP**: 3D-MVP 用 MAE 做 3D 重建预训练但缺乏动态建模，DynaRend 增加未来预测分支
- **vs VPP/VidMan**: 视频扩散模型只建模 2D 动态，DynaRend 在 3D 空间中建模动态
- **vs ManiGaussian**: 显式 3D 高斯表示复杂且需密集新视角监督，DynaRend 用紧凑 triplane 表示且支持视角合成增强
- **vs GNFactor/SPA**: 只关注静态 3D 一致性预训练，不含动态预测

## 启发与关联

渲染即监督的思路很有启发——不需要显式 3D 标注，通过渲染的 RGB/深度/语义与 2D 观测对比即可学习 3D 表征。Triplane 作为机器人操控的 3D 表示在效率和表达力上取得良好平衡，值得进一步探索。

## 评分

- ⭐ 新颖性: 8/10 — 首个在 triplane 上联合掩码重建+未来预测+体渲染的机器人预训练框架
- ⭐ 实验充分度: 9/10 — RLBench 18+71 任务、Colosseum 泛化、真实世界，消融详尽
- ⭐ 写作质量: 8/10 — 动机清晰，框架图直观，对比全面
- ⭐ 价值: 8/10 — 对 3D 表征学习+机器人操控的交叉领域有推动意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] EnerVerse: Envisioning Embodied Future Space for Robotics Manipulation](enerverse_envisioning_embodied_future_space_for_robotics_manipulation.md)
- [\[NeurIPS 2025\] UMAMI: Unifying Masked Autoregressive Models and Deterministic Rendering for View Synthesis](umami_unifying_masked_autoregressive_models_and_deterministic_rendering_for_view.md)
- [\[NeurIPS 2025\] MPMAvatar: Learning 3D Gaussian Avatars with Accurate and Robust Physics-Based Dynamics](mpmavatar_learning_3d_gaussian_avatars_with_accurate_and_robust_physics-based_dy.md)
- [\[ICCV 2025\] RoboTron-Mani: All-in-One Multimodal Large Model for Robotic Manipulation](../../ICCV2025/3d_vision/robotron-mani_all-in-one_multimodal_large_model_for_robotic_manipulation.md)
- [\[CVPR 2026\] PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation](../../CVPR2026/3d_vision/pointworld_scaling_3d_world_models_for_in-the-wild_robotic_manipulation.md)

</div>

<!-- RELATED:END -->
