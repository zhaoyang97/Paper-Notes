---
title: >-
  [论文解读] Toward Material-Agnostic System Identification from Videos
description: >-
  [ICCV2025][system identification] 提出 MASIV，首个无需预定义材质先验的视觉系统辨识框架：采用可学习的神经本构模型替代手工设计的弹性/塑性方程，通过重建连续体粒子轨迹提供时间密集的几何约束，从多视角视频中推断物体的内在动力学特性。 视觉系统辨识 系统辨识旨在从视觉观测中恢复物体的几何形状…
tags:
  - "ICCV2025"
  - "system identification"
  - "neural constitutive model"
  - "material point method"
  - "3D Gaussian splatting"
  - "differentiable simulation"
---

# Toward Material-Agnostic System Identification from Videos

**会议**: ICCV2025  
**arXiv**: [2508.01112](https://arxiv.org/abs/2508.01112)  
**代码**: [Skaldak/MASIV](https://github.com/Skaldak/MASIV)  
**领域**: 物理仿真 / 系统辨识 / 可微渲染  
**关键词**: system identification, neural constitutive model, material point method, 3D Gaussian splatting, differentiable simulation

## 一句话总结

提出 MASIV，首个无需预定义材质先验的视觉系统辨识框架：采用可学习的神经本构模型替代手工设计的弹性/塑性方程，通过重建连续体粒子轨迹提供时间密集的几何约束，从多视角视频中推断物体的内在动力学特性。

## 研究背景与动机

### 视觉系统辨识

系统辨识旨在从视觉观测中恢复物体的几何形状和控制其运动的物理定律。典型方法将可微渲染（NeRF、3DGS）与可微仿真器（MPM 等）端到端集成，通过优化参数化物理模型来拟合观测。

### 现有方法的材质依赖问题

现有方法（PAC-NeRF、Spring-Gaus、NeuMA、GIC）均依赖**材质特定的本构定律**：
- 需要预先知道材质类型（弹性体、塑料、沙子、流体等）
- 使用手工设计的弹性/塑性模型（如 neo-Hookean 弹性 + 恒等塑性返回函数）
- 仅估计少量物理参数（杨氏模量、粘度、摩擦角等）

问题：
1. 限制了方法在未知材质场景中的适应性
2. 需要为每种材质类型选择对应的本构模型
3. 无法泛化到野外场景中材质属性未知的情况

### 核心挑战

直接将神经本构模型（NCLaw）应用于视觉系统辨识面临重大挑战：NCLaw 假设可获得完整的粒子状态信息（位置、速度、形变梯度、仿射动量），而视觉观测无法提供如此完整的信息。仅依赖逐帧像素监督提供的约束不足，导致优化不稳定和物理不合理行为。

## 方法详解

### 总体流程（三阶段）

**Phase I - 几何重建**：从多视角视频重建动态高斯和稠密粒子轨迹
**Phase II - 材质无关系统辨识**：利用视觉观测和重建的运动线索学习神经本构模型
**Phase III - 可泛化数字孪生**：获得可模拟新交互（新速度/力）的数字孪生

### Phase I：动态高斯重建

维护一组规范空间的高斯核，用变形网络在时间上扭曲：

- 基函数网络：将时间步映射到 B 个基，产生位置和尺度的变形基
- 系数网络：从规范高斯中心坐标和时间步估计每个基的权重
- 优化目标：L1 + SSIM 损失 + 尺度正则化

### Phase I（续）：连续体轨迹估计

关键创新之一。将高斯粒子转换为固体连续体，并估计时间密集的粒子轨迹：

1. 按 GIC 方法填充内部体积，形成均匀密度的连续体粒子
2. 微调变形网络以适应连续体粒子（包括内部粒子），用 Chamfer 距离损失优化
3. 利用运动基函数的时间位置编码实现时间插值
4. 在每个仿真时间步（N = T/tau >> T，远多于视频帧数）推断粒子位置，作为后续优化的伪真值

### Phase II：材质无关系统辨识

#### 神经本构模型

借鉴 NCLaw，用两个 MLP 参数化弹性本构定律和塑性本构定律：

- 弹性本构：从弹性形变梯度计算第一 Piola-Kirchhoff 应力
- 塑性本构：对试验弹性形变梯度施加塑性约束
- 物理先验：帧不变性（旋转不变输入表示）+ 未变形状态平衡（消除偏置项）
- 初始化：使用 NCLaw 的预训练模型作为稳定初始化

#### MPM 状态转移

每个时间步：弹性本构计算应力 -> 时间积分更新粒子状态 -> 塑性本构修正形变梯度

#### 优化目标

最小化几何损失 + 轮廓损失：

- **几何损失（轨迹监督）**：仿真粒子位置与变形网络预测轨迹之间的 L1 损失，在所有仿真时间步上计算（时间密集）
- **轮廓损失**：渲染掩码与物体轮廓之间的 L1 损失

### 为何需要时间密集的几何约束

- 稀疏的逐帧监督（Surface 或 Continuum 级别）对复杂神经本构模型约束不足
- 仿真步之间的行为未被约束，可能导致过拟合和物理不合理变形
- 轨迹级别的密集监督在所有仿真步上施加约束，实现最低且最稳定的误差

## 实验结果

### 实验设置

- **数据集**：PAC-NeRF（合成，含弹性体/塑料/沙子/牛顿流体/非牛顿流体）、Spring-Gaus（合成+真实，弹性体）
- **基线**：PAC-NeRF、Spring-Gaus、NeuMA、GIC（均需材质先验）
- **指标**：Chamfer Distance (CD)、PSNR、SSIM
- **硬件**：单卡 NVIDIA A100

### 可观测状态仿真

PAC-NeRF 数据集（CD 指标）：

| 方法 | Newtonian | Non-Newtonian | Elasticity | Plasticine | Sand | 平均 |
|------|-----------|---------------|------------|------------|------|------|
| PAC-NeRF | 0.277 | 0.236 | 0.238 | 0.429 | 0.212 | 0.278 |
| GIC | 0.243 | 0.195 | 0.178 | 0.196 | 0.250 | 0.212 |
| **MASIV** | **0.233** | **0.198** | **0.192** | **0.201** | **0.229** | **0.210** |

Spring-Gaus 合成数据集（CD 指标）：

| 方法 | 平均 CD |
|------|--------|
| PAC-NeRF | 2.11 |
| Spring-Gaus | 0.85 |
| GIC | 0.17 |
| **MASIV** | **0.13** |

MASIV 在不使用任何材质先验的情况下，几何精度超过所有需要材质先验的方法。

### Spring-Gaus 真实数据集

| 方法 | 平均 PSNR | 平均 SSIM |
|------|----------|----------|
| Spring-Gaus | 32.90 | 0.994 |
| GIC | 38.11 | 0.996 |
| **MASIV** | **41.12** | **0.997** |

### 未来状态预测

在 Spring-Gaus 合成数据集上，MASIV 在 CD 上优于除 GIC 外的所有方法。GIC 略优是因为其预知的本构模型提供了类别级正则化，而 MASIV 在数据稀缺时缺乏此正则化。更多数据时 MASIV 的泛化能力增强。

### 消融实验：几何约束的影响

| 几何约束类型 | 特点 | 效果 |
|-------------|------|------|
| 无几何约束 | 仅用轮廓监督 | CD 误差最高且不稳定 |
| Surface | 逐帧表面对齐 | 适度降低误差但不稳定 |
| Continuum | 逐帧连续体对齐 | 进一步降低但某些材质不稳定 |
| **Trajectory** | 每仿真步轨迹对齐 | **最低且最稳定的误差** |

定性分析：稀疏监督可能导致物理不合理变形（如奶油在仿真步之间异常膨胀），轨迹监督通过时间密集约束有效缓解。

## 优势与局限

### 优势

- 首个视觉系统辨识中不需要材质先验的方法
- 时间密集的轨迹约束是关键创新，有效稳定了神经本构模型的优化
- 在可观测状态仿真上实现 SOTA，跨越多种材质类型
- 产生的数字孪生可泛化到新的初始条件

### 局限

- 仍需 NCLaw 预训练模型作为初始化
- 未来状态预测在数据稀缺时略逊于使用材质先验的 GIC
- 轨迹估计依赖变形网络的插值质量
- 单一场景的优化仍需较长时间

## 个人思考

1. "材质无关"是非常有价值的方向——真实世界中材质通常是未知的，手工选择本构模型不可扩展
2. 稠密轨迹约束的思路巧妙：不仅给仿真提供了更多监督信号，还通过运动基函数的时间插值实现了"从稀疏到稠密"的过渡
3. 方法仍依赖预训练的神经本构模型初始化，从 NCLaw 预训练中隐含了物理先验，并非完全无先验
4. 未来可探索多场景联合训练以学习更通用的神经本构模型，而非单场景优化

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] A Measure of the System Dependence of Automated Metrics](../../ACL2025/others/a_measure_of_the_system_dependence_of_automated_metrics.md)
- [\[CVPR 2025\] Scene-Agnostic Pose Regression for Visual Localization](../../CVPR2025/others/scene-agnostic_pose_regression_for_visual_localization.md)
- [\[ICCV 2025\] Switch-a-View: View Selection Learned from Unlabeled In-the-wild Videos](switch-a-view_view_selection_learned_from_unlabeled_in-the-wild_videos.md)
- [\[NeurIPS 2025\] Recurrent Self-Attention Dynamics: An Energy-Agnostic Perspective from Jacobians](../../NeurIPS2025/others/recurrent_self-attention_dynamics_an_energy-agnostic_perspective_from_jacobians.md)
- [\[ACL 2025\] GeNRe: A French Gender-Neutral Rewriting System Using Collective Nouns](../../ACL2025/others/genre_a_french_gender-neutral_rewriting_system_using_collective_nouns.md)

</div>

<!-- RELATED:END -->
