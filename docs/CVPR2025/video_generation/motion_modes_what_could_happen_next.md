---
title: >-
  [论文解读] Motion Modes: What Could Happen Next?
description: >-
  [CVPR 2025][视频生成][运动预测] 提出 Motion Modes，一种免训练方法，通过设计四种引导能量函数探索预训练图像到视频生成器的潜在分布，从单张图像中发现物体的多种合理且多样的运动模式，同时将物体运动与相机运动解耦。 从单张静态图像预测物体的多种可能运动是一个开放性挑战。现有方法的局限：(1) 视频生成模…
tags:
  - "CVPR 2025"
  - "视频生成"
  - "运动预测"
  - "免训练方法"
  - "扩散模型引导"
  - "物体运动多样性"
  - "图像到视频"
---

# Motion Modes: What Could Happen Next?

**会议**: CVPR 2025  
**arXiv**: [2412.00148](https://arxiv.org/abs/2412.00148)  
**代码**: [https://motionmodes.github.io](https://motionmodes.github.io)  
**领域**: 视频生成  
**关键词**: 运动预测, 免训练方法, 扩散模型引导, 物体运动多样性, 图像到视频

## 一句话总结

提出 Motion Modes，一种免训练方法，通过设计四种引导能量函数探索预训练图像到视频生成器的潜在分布，从单张图像中发现物体的多种合理且多样的运动模式，同时将物体运动与相机运动解耦。

## 研究背景与动机

从单张静态图像预测物体的多种可能运动是一个开放性挑战。现有方法的局限：(1) 视频生成模型通常将物体运动与相机运动、场景变化耦合在一起；(2) 基于运动箭头输入的方法（如 Motion-I2V）虽能预测特定运动，但依赖合成训练数据和预定义运动，难以处理复杂场景（如海浪翻涌）；(3) 更关键的是，这些方法需要用户 **给定** 运动指令，而非 **自动发现** 多种可能运动。

核心洞察：预训练的图像到视频生成器已在大量数据上编码了丰富的运动分布。问题是能否探索这种潜在分布来发现物体的多种运动？直接随机采样效率低且会混入大量相机运动。Motion Modes 通过精心设计的引导能量函数来高效探索这一分布。

## 方法详解

### 整体框架

给定输入图像 $\mathbf{y}$ 和物体掩码 $\mathbf{m}$，目标是发现物体的多种可能运动集合 $\mathcal{X} = \{\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \ldots\}$。运动表示为时间依赖的二维向量场 $\mathbf{x} \in \mathbb{R}^{F \times H \times W \times 2}$。基于 Motion-I2V 作为骨干模型——该模型分离生成运动和外观，天然解耦运动与其他场景变化。通过迭代采样 + 停止准则构建运动集合。

### 关键设计

1. **四种引导能量函数**: 在推理阶段的去噪过程中注入四种能量引导，无需训练或微调：
    - **静态相机引导 $E_c$**: 惩罚物体掩码外区域的平均运动幅度，鼓励相机静止
    - **物体运动引导 $E_o$**: 鼓励掩码内外运动幅度差异，确保物体有动作
    - **多样性引导 $E_d$**: 受 particle guidance 启发，对已生成运动集合 $\mathcal{X}$ 中每个运动施加排斥力，鼓励新运动在方向（$w_{\text{angle}}=0.75$）和幅度（$w_{\text{mag}}=0.25$）上与已有运动不同
    - **平滑性引导 $E_s$**: 正则化连续帧间运动变化，避免抖动

   总能量 $E = \lambda_d E_d + \lambda_c E_c + \lambda_o E_o + \lambda_s E_s$，在去噪时通过梯度下降修改噪声轨迹：$\mathbf{x}_t' = \mathbf{x}_t - \nabla_{\mathbf{x}_t} E(x_\theta^0(\mathbf{x}_t; t, \mathbf{y}), \mathbf{m}, \mathcal{X})$

2. **迭代采样与停止准则**: 逐个采样运动并加入集合 $\mathcal{X}$，最多采样 6 个运动。当最终去噪运动的引导能量超过阈值 $\rho=5.0$ 时丢弃并重采样，连续丢弃两次则停止——这自动适应不同场景允许的运动数量（如抽屉只能开/关，但旗帜可以多种方式飘动）。

3. **距离度量设计**: 运动向量间的距离综合考虑方向和幅度：$d(\mathbf{a}, \mathbf{b}) = w_{\text{mag}}(|\|\mathbf{a}\| - \|\mathbf{b}\||) + w_{\text{angle}}(1 - \frac{\mathbf{a}^\top \mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|})$。多样性引导侧重方向差异，平滑性引导侧重幅度平稳。

### 损失函数 / 训练策略

Motion Modes 是完全 **免训练** 的方法——不修改预训练模型任何参数，仅在推理时通过能量引导改变去噪轨迹。

引导能量权重设置：$\lambda_d=3.0$, $\lambda_c=0.2$, $\lambda_o=0.025$, $\lambda_s=0.1$。使用 $\phi(a) = \text{softplus}((a+e)^{-1} - \tau)$ 作为激活函数（软反函数），$\tau$ 对物体运动引导设为 40，对多样性引导设为 1。

## 实验关键数据

### 主实验（定量指标）

| 方法 | 多样性 $\bar{E}_d$ ↓ | 聚焦度 $\bar{E}_f$ ↓ | 相机静止 $\bar{E}_c$ ↓ | 物体运动 $\bar{E}_o$ ↓ |
|------|---------------------|---------------------|---------------------|---------------------|
| Prompt Generation (GPT-4o) | 1.28 | 1.71 | 1.11 | 2.31 |
| ControlNet | 1.75 | 1.14 | 0.07 | 2.22 |
| Random Arrows | 1.77 | 1.17 | 0.07 | 2.27 |
| Random Noise | 1.27 | 2.20 | 1.36 | 3.05 |
| FPS Noise | 1.21 | 1.98 | 1.23 | 2.74 |
| **Motion Modes (ours)** | **1.04** | **0.07** | **0.09** | **0.05** |

### 消融实验

| 配置 | $\bar{E}_d$ ↓ | $\bar{E}_f$ ↓ | $\bar{E}_c$ ↓ | $\bar{E}_o$ ↓ | $\bar{E}$ ↓ |
|------|--------------|--------------|--------------|--------------|-------------|
| 去除 $E_c$ | 1.02 | 0.64 | 1.29 | 0.00 | 0.83 |
| 去除 $E_o$ | 1.03 | 0.91 | 0.06 | 1.75 | 0.97 |
| 去除 $E_d$ | 1.36 | 0.08 | 0.13 | 0.04 | 0.72 |
| FPS 替代 $E_d$ | 1.49 | 0.10 | 0.11 | 0.08 | 0.79 |
| ControlNet 替代 $E_c$+$E_o$ | 0.96 | 0.80 | 0.15 | 1.45 | 0.88 |
| **Motion Modes (完整)** | **1.04** | **0.07** | **0.09** | **0.05** | **0.55** |

### 关键发现

1. **用户研究 I (32人, 320组对比)**：Motion Modes 在合理性、多样性、符合预期三个维度上全面超越所有基线，其中 Prompt Generation 基线在多样性上最接近但聚焦度差
2. **用户研究 II (12人)**：96% 的生成运动被认为合理，92% 命中用户预期，19% 提供了超出预期但仍合理的新运动——说明方法既准确又有启发性
3. ControlNet 和 Random Arrows 虽有好的聚焦度指标，但实际是因为物体运动也被抑制（静态场景）
4. FPS 噪声采样比随机采样好但远不如引导能量，证明简单的噪声空间策略不足以保证运动多样性
5. 各引导能量缺一不可，完整组合才能在多样性和聚焦度间取得最优平衡

## 亮点与洞察

- **免训练** 是最大亮点——直接利用预训练模型的隐含运动先验，环境适应性强，不受训练数据限制
- 引导能量设计巧妙：相机静止 + 物体运动 = 解耦；多样性排斥力 = 探索分布；平滑性 = 运动质量
- 迭代采样 + 停止准则可自适应场景复杂度，避免为简单场景强行生成不合理运动
- 运动补全应用展示了实用价值：用户粗糙箭头可自动匹配最近的详细运动，避免拖拽编辑中的歧义

## 局限与展望

- 继承预训练模型的数据偏差——无法产生训练数据中不存在的运动类型
- 连续运动空间只能离散采样（如笔记本电脑可向任意方向移动，但只能采样有限方向）
- 前向传递次数等于运动数量，计算成本随运动数线性增长
- 仅处理 2D 运动场，未扩展到 3D 运动
- 不支持带相机运动的场景（如运动跟拍）

## 相关工作与启发

- particle guidance [6] 的排斥能量思想被迁移到视频生成的运动多样性中，但巧妙绕过了内存限制（迭代而非并行采样）
- Motion-I2V 的运动/外观分离架构是该方法成立的基础，说明解耦表征对下游操控很重要
- 引导能量的思路可扩展到其他生成任务中的可控多样化采样
- 运动补全应用为 drag-based 图像编辑提供了新范式：粗糙输入 → 详细合理运动

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个免训练方法发现物体多样运动，引导能量设计精巧
- 实验充分度: ⭐⭐⭐⭐ 有定量指标+两个用户研究+消融，但图像数量较少(28张)
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机清晰，公式推导完整，可视化效果好
- 价值: ⭐⭐⭐⭐ 开创了物体运动模式发现这一新方向，应用前景广泛

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Motion Prompting: Controlling Video Generation with Motion Trajectories](motion_prompting_controlling_video_generation_with_motion_trajectories.md)
- [\[CVPR 2025\] ConMo: Controllable Motion Disentanglement and Recomposition for Zero-Shot Motion Transfer](conmo_controllable_motion_disentanglement_and_recomposition_for_zero-shot_motion.md)
- [\[CVPR 2026\] Video-as-Answer: Predict and Generate Next Video Event with Joint-GRPO](../../CVPR2026/video_generation/video-as-answer_predict_and_generate_next_video_event_with_joint-grpo.md)
- [\[CVPR 2025\] DiTFlow: Video Motion Transfer with Diffusion Transformers](video_motion_transfer_with_diffusion_transformers.md)
- [\[CVPR 2026\] What Are You Doing? A Closer Look at Controllable Human Video Generation](../../CVPR2026/video_generation/what_are_you_doing_a_closer_look_at_controllable_human_video_generation.md)

</div>

<!-- RELATED:END -->
