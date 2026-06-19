---
title: >-
  [论文解读] SMGDiff: Soccer Motion Generation using Diffusion Probabilistic Models
description: >-
  [ICCV 2025][图像生成][motion generation] 提出 SMGDiff，一个两阶段扩散模型框架，能够根据用户控制信号实时生成高质量、多样化的足球运动动画，同时通过接触引导模块优化球-脚交互细节。 足球是全球最受欢迎的运动，在游戏和 VR/AR 领域有广泛应用需求。然而，生成逼真的足球运动动画面临以下核…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "motion generation"
  - "扩散模型"
  - "soccer animation"
  - "human-object interaction"
  - "character control"
---

# SMGDiff: Soccer Motion Generation using Diffusion Probabilistic Models

**会议**: ICCV 2025  
**arXiv**: [2411.16216](https://arxiv.org/abs/2411.16216)  
**代码**: [https://github.com/SMGDiff/SMGDiff](https://github.com/SMGDiff/SMGDiff)  
**领域**: 图像生成  
**关键词**: motion generation, diffusion model, soccer animation, human-object interaction, character control

## 一句话总结

提出 SMGDiff，一个两阶段扩散模型框架，能够根据用户控制信号实时生成高质量、多样化的足球运动动画，同时通过接触引导模块优化球-脚交互细节。

## 研究背景与动机

足球是全球最受欢迎的运动，在游戏和 VR/AR 领域有广泛应用需求。然而，生成逼真的足球运动动画面临以下核心挑战：

**复杂的人-球交互**：足球运动涉及球员与足球之间精细的物理接触，尤其是球-脚之间的接触精度要求极高

**实时性要求**：游戏和交互应用需要实时推理，现有扩散模型通常计算量大，难以满足

**动作多样性**：足球运动包含盘带、花式、射门等众多技能类别，需要覆盖宽泛的动作谱

**数据集缺乏**：缺少公开的大规模足球动作捕捉数据集

现有方法的局限：

- **商业游戏**（如 EA SPORTS FC 系列）依赖庞大的预录制动作库进行 motion matching，容易产生视觉瑕疵
- **强化学习/物理模拟方法**仅限于特定技能（如盘带、射门、颠球），无法覆盖全谱足球动作
- **通用运动扩散模型**（如 CAMDM）只关注人体运动风格切换，不处理人-物交互
- **人-物交互生成方法**需要耗时的后优化，不适合实时交互场景

## 方法详解

SMGDiff 采用**两阶段框架**：第一阶段生成轨迹，第二阶段基于轨迹条件生成足球动作。

### 运动表示

足球运动状态 $x^i = \{h, b, c\}$ 由三部分组成：

- **人体状态** $h \in \mathbb{R}^{3+24 \times 6}$：包括根节点位置和 24 个 SMPL 关节的 6-DOF 旋转
- **球状态** $b \in \mathbb{R}^{7}$：包括相对球位置、全局球速度和球控制权重
- **二值接触标签** $c = \{c_g, c_b\}$：脚-地面接触和脚-球接触

球控制权重 $w_b = 1 - \|b_p^{xy} - h_p^{xy}\| / r$ 用于将全局球位置转换为相对位置，当球远离人体根部超过半径 $r=2m$ 时权重趋近 0，有效分离了球在受控和不受控状态下的表示。

### 第一阶段：轨迹生成模型 (TGM)

**目标**：将用户的粗粒度控制信号（方向、速度、技能类别）转化为精细的角色全局轨迹。

**网络结构**：基于 Transformer Encoder 的轻量级单步扩散模型。输入包括：

- 足球技能标签 $\mathbf{S}$（6 类：dribble, trick, shoot, stand, celebrate, off-the-ball move）
- 目标轨迹点 $\mathbf{G}$（由键盘方向键和按压强度计算）
- 过去轨迹 $\mathbf{T}^{\mathcal{P}}$
- 高斯噪声 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$（增强多样性）

推理时类似单步 DDPM 逆过程，单次前向传播即完成轨迹生成，确保实时性。

**轨迹混合**：采用 CAMDM 的 HFTE（Heuristic Future Trajectory Extension）策略，当用户输入新控制信号时，将新生成的轨迹与前一帧的结果混合，防止角色方向和速度突变。

### 第二阶段：足球运动扩散模型

**网络结构**：基于 Transformer 的自回归扩散模型，以轨迹为条件生成足球动作序列。

**条件输入** $\mathbf{C} = \{\mathbf{S}, \mathbf{X}^{\mathcal{P}}, \mathbf{T}^{\mathcal{F}}\}$：

- 技能标签 $\mathbf{S}$
- 过去动作 $\mathbf{X}^{\mathcal{P}}$（10 帧历史）
- 未来轨迹 $\mathbf{T}^{\mathcal{F}}$（45 帧）

**训练损失**由四项组成：

$$\mathcal{L} = \mathcal{L}_{\text{simple}} + \lambda_{\text{pos}} \mathcal{L}_{\text{pos}} + \lambda_{\text{vel}} \mathcal{L}_{\text{vel}} + \lambda_{\text{foot}} \mathcal{L}_{\text{foot}}$$

- $\mathcal{L}_{\text{simple}}$：直接预测 $\mathbf{X}_0^{\mathcal{F}}$ 的重建损失
- $\mathcal{L}_{\text{pos}}$：前向运动学得到的关节位置损失
- $\mathcal{L}_{\text{vel}}$：速度一致性损失
- $\mathcal{L}_{\text{foot}}$：脚-地面接触约束损失（接触时脚不滑动）

### 接触引导模块 (CGM)

在推理阶段的扩散过程中引入接触引导，通过一个专用损失函数优化球-脚接触。

**接触检测**：当球加速度超过阈值 $\tau_a = 2 \text{m/s}^2$ 时，认为存在脚-球接触（仅有地面摩擦时加速度较小且恒定）：

$$\hat{c}_b = \mathbb{I}(\|b_a\| > \tau_a)$$

**接触关节选择**：计算球与各脚关节的距离，优先选择悬空的脚（通过惩罚权重 $w_d=2$ 增大着地脚的距离）：

$$d = \min_{j \in \text{foot joints}} ((f_p^j - b_p) \cdot (1 + (w_d - 1) \cdot c_g^j))$$

**接触损失**：

$$L = \sum_{i=1}^{F} d^i \cdot \frac{\mathbb{I}(d^i > \tau_d) \cdot \hat{c}_b^i}{\mathbb{I}(d^i > \tau_d) + \delta}$$

仅在球-脚距离超过阈值 $\tau_d = 0.1m$ 且检测到接触时激活，引导脚靠近球。

**梯度引导**：采用 DSG 方法的自适应步长策略，将梯度方向与无条件采样方向加权混合（引导率 $w_r = 0.5$），在增强接触精度的同时保持运动多样性。

**部署策略**：接触引导仅在扩散模型最后 2 个去噪步施加（共 8 步），原因是早期噪声较高时引导效果差且可能导致不自然结果。

### 实现细节

- 帧率 30Hz，过去帧数 P=10，未来帧数 F=45
- 扩散去噪步数 8（训练和推理一致）
- 运行环境：Unity（用户交互和可视化）+ Python（模型推理），通过 TCP 通信
- 硬件：Intel i7-10700K + NVIDIA RTX 3080 Ti

## 数据集：Soccer-X

作者构建了 Soccer-X 数据集，这是首个面向数据驱动足球运动生成的大规模数据集：

| 属性 | 数值 |
|------|------|
| 捕捉系统 | 16 台 OptiTrack Prime x13 相机 |
| 捕捉区域 | 6m × 7.5m × 2.5m |
| 原始帧率 | 240 fps（后降采样至 30 fps） |
| 球员数量 | 30 名 |
| 总帧数 | ~1.08M |
| 总时长 | >10 小时 |
| 人体格式 | SMPL |
| 动作类别 | 6 类 |

六类动作：Dribble（含不同速度/脚/方向）、Stand、Off-the-ball Move、Trick（5 种花式过人）、Shoot（室内捕捉后在 Unity 中物理模拟完整弹道）、Celebrate。

## 实验结果

### 定量比较

与三种实时控制器方法对比（LMP、MANN-DP、CM），使用测试集轨迹作为条件输入：

| 方法 | FID↓ | 脚滑距离↓ | 加速度↓ | 多样性↑ | 轨迹误差↓ | 朝向误差↓ | 技能准确率↑ |
|------|------|-----------|---------|---------|-----------|-----------|-------------|
| LMP | 0.354 | 1.068 | 1.607 | 0.398 | 4.116 | 6.493 | 73.3% |
| MANN-DP | 0.359 | 1.351 | 1.565 | 0.475 | 4.069 | 5.299 | 69.1% |
| CM | 0.249 | 1.650 | 1.175 | 0.352 | 3.103 | 5.066 | 52.9% |
| **SMGDiff** | **0.181** | **0.854** | 1.200 | **0.618** | **2.413** | **4.939** | **93.3%** |

SMGDiff 在几乎所有指标上都显著优于基线方法，尤其是技能准确率达到 93.3%（比最强基线高 20 个百分点），FID 最低说明生成的动作分布最接近真实数据。

### 消融实验

| 变体 | FID↓ | 脚滑距离↓ | 加速度↓ | 多样性↑ |
|------|------|-----------|---------|---------|
| w/o TGM | 0.365 | 1.003 | 1.197 | 2.433 |
| w/o CGM | 0.370 | 1.005 | 1.196 | 2.691 |
| Full | **0.358** | 1.005 | 1.201 | **2.693** |

- **TGM 的作用**：轨迹生成模型将直线轨迹替换为多样化轨迹，显著降低 FID 并提升多样性
- **CGM 的作用**：接触引导虽略微增加脚滑和加速度，但明显降低 FID，说明生成的人-球交互更逼真

### 运行时分析

| 去噪步数 | 2 | 4 | **8** | 16 | 32 |
|----------|---|---|-------|----|----|
| 推理时间 | 3ms | 6ms | **12ms** | 25ms | 52ms |
| FID | 0.395 | 0.390 | **0.370** | 0.366 | 0.338 |

8 步去噪在推理速度和生成质量之间取得最优平衡，仅需 12ms 即可满足实时要求。

## 局限性

1. **缺乏物理约束**：训练和推理中未加入真实物理模拟，可能产生物理不合理的动作
2. **交互局限**：仅考虑球-脚交互，忽略头球、胸部控球等其他身体部位的球交互
3. **单人限制**：不涉及多人交互场景（如传球、对抗），未来可结合物理模拟生成多人足球动作

## 个人思考

- **两阶段解耦设计**是本文的核心亮点：将「粗粒度控制→精细轨迹」和「轨迹→全身动作」分离，既降低了单个模型的学习难度，又使轨迹可以作为中间表示灵活替换或调整
- **接触引导的轻量化策略**值得借鉴：仅在最后 2 步施加引导（而非全程），在保持实时性的同时有效改善接触质量，这种「选择性引导」思路可推广到其他需要物理约束的扩散生成任务
- **球控制权重**的设计简洁有效，通过距离衰减将球的全局/相对位置表示统一，避免了远处球对动作生成的干扰
- Soccer-X 数据集的构建和开源对社区有重要价值，但 30fps 室内捕捉与真实足球场景仍有差距
- 未来最有前景的方向是结合 Isaac Gym 等物理引擎做后优化，以及扩展到多人交互场景

## 亮点与洞察

## 局限与展望

## 相关工作与启发

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Elucidating the SNR-t Bias of Diffusion Probabilistic Models](../../CVPR2026/image_generation/dcw_snr_t_bias_diffusion.md)
- [\[ICCV 2025\] Diffusion-based 3D Hand Motion Recovery with Intuitive Physics](diffusion-based_3d_hand_motion_recovery_with_intuitive_physics.md)
- [\[ICCV 2025\] Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video](bitrate-controlled_diffusion_for_disentangling_motion_and_content_in_video.md)
- [\[ICCV 2025\] Less is More: Improving Motion Diffusion Models with Sparse Keyframes](less_is_more_improving_motion_diffusion_models_with_sparse_keyframes.md)
- [\[CVPR 2025\] SyncSDE: A Probabilistic Framework for Diffusion Synchronization](../../CVPR2025/image_generation/syncsde_a_probabilistic_framework_for_diffusion_synchronization.md)

</div>

<!-- RELATED:END -->
