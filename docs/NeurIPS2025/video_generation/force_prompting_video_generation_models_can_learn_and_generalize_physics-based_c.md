---
title: >-
  [论文解读] Force Prompting: Video Generation Models Can Learn and Generalize Physics-based Control Signals
description: >-
  [NeurIPS 2025][视频生成][提示学习] 提出Force Prompting，将物理力（局部点力和全局风力）作为视频生成模型的控制信号，仅用~15K合成训练视频（Blender旗帜和滚球）和单日4xA100训练，即可在多样真实场景图像上展现跨物体/材质/几何的惊人泛化，包括初步的质量理解能力。
tags:
  - "NeurIPS 2025"
  - "视频生成"
  - "提示学习"
  - "video generation"
  - "physics control"
  - "sim2real"
  - "CogVideoX"
---

# Force Prompting: Video Generation Models Can Learn and Generalize Physics-based Control Signals

**会议**: NeurIPS 2025  
**arXiv**: [2505.19386](https://arxiv.org/abs/2505.19386)  
**代码**: [force-prompting.github.io](https://force-prompting.github.io/)（数据集+代码+模型权重全开源）  
**领域**: 视频生成 / 物理可控生成 / 世界模型  
**关键词**: force prompting, video generation, physics control, sim2real, CogVideoX

## 一句话总结
提出Force Prompting，将物理力（局部点力和全局风力）作为视频生成模型的控制信号，仅用~15K合成训练视频（Blender旗帜和滚球）和单日4xA100训练，即可在多样真实场景图像上展现跨物体/材质/几何的惊人泛化，包括初步的质量理解能力。

## 研究背景与动机

**领域现状**：视频生成模型（Sora/CogVideoX/Wan2.1）视觉质量和运动真实性进步巨大，但主要依赖文本和图像输入，缺乏物理交互精确控制。可控方向主要是相机控制和轨迹控制。

**现有痛点**：(a) 轨迹控制需预先指定每帧像素位置，无法处理全局现象（风/流体）；(b) 轨迹与力是不同物理量——相同力对不同质量物体产生不同位移；(c) 物理模拟器方法需要3D几何或推理时运行模拟器。

**核心矛盾**：获取高质量力-视频配对训练数据极其困难。

**本文目标** 用极少量合成物理模拟数据教会预训练视频生成模型理解力控制信号。

**切入角度**：假设SOTA视频模型已编码了关于视觉动力学的强先验，合成数据只需"激发"而非"教会"模型。

**核心 idea**：视频生成模型可以从极少量合成数据学习以物理力为条件的生成，并展现跨物体/材质/几何的惊人泛化。

## 方法详解

### 整体框架
输入三元组 $(\tau, \phi, \pi)$：文本、初始帧、物理控制信号。输出49帧@8fps视频。在CogVideoX-5B-I2V上添加6层ControlNet注入力信号，基座冻结。

### 关键设计

1. **合成训练数据**:

    - 全局风力(15K)：Blender旗帜飘动，随机化旗帜数(1~64)、颜色(100种)、HDRI背景(50种)、风向、风速
    - 局部点力(23K)：12K Blender滚球（足球vs保龄球质量比1:4）+ 11K PhysDreamer康乃馨被戳
    - 干扰球的存在对力的空间定位至关重要

2. **力编码策略**:

    - 全局风力：3通道全图统一值（力大小映射[-1,1]、cos/sin角度）
    - 局部点力：高斯blob从指定位置沿力方向移动，位移与力大小成正比。blob位置远离受力像素——与轨迹控制本质不同

3. **架构与训练**:

    - CogVideoX-5B-I2V（冻结）+ 6层ControlNet（前6个transformer层克隆）
    - 4xA100, batch 8, 5000步(~1天), AdamW lr=1e-5

## 实验关键数据

### 局部力模型 vs 基线（2AFC人类偏好%，>50%表示偏好本方法）

| 基线 | 力遵循 | 物理真实 | 视觉质量 |
|------|--------|---------|---------|
| Text-only zero-shot | 72/67/73 | 50/48/48 | 48/52/49 |
| Text-only fine-tuned | 79/62/74 | 53/52/55 | 52/58/54 |
| Motion Prompting | **91/89/86** | **93/76/76** | 100/99/98 |

### vs PhysDreamer（6种植物均值）

| 指标 | Force Prompting偏好率 |
|------|---------------------|
| 运动真实性 | 48.3%（接近持平） |
| 视觉质量 | 36.7%（PhysDreamer更好） |
| **力遵循度** | **58.3%（本方法更好）** |

### 关键发现
- 仅用旗帜训练的风力模型泛化到烟雾、雪花、气球等完全不同材质
- 仅用滚球+康乃馨训练的点力模型泛化到各种植物、热气球、秋千等
- **质量理解**：足球在相同力下移动距离始终大于保龄球，位移与力大小呈线性关系
- **多力零样本**：推理时添加多个高斯blob可控制多个物体，无需重训
- 训练数据设计比数量更重要：去掉干扰球导致无法定位力；单一背景导致前景/背景混淆
- 训练时使用"wind/breeze/blow"文本关键词对泛化至关重要

## 亮点与洞察
- **极小数据+极低算力的泛化奇迹**：15K合成视频、单日训练，从旗帜泛化到烟雾/雪花/气球——暗示预训练视频模型已编码丰富直觉物理知识，合成数据只是"钥匙"
- **力不等于轨迹**：高斯blob位置远离受力像素（花的振荡），与轨迹控制有本质差异。对世界模型设计有重要启发
- **训练数据消融的工程洞察**：干扰物体对力定位至关重要；视觉多样性对材质泛化至关重要；文本关键词帮助激发模型内在物理理解

## 局限与展望
- 视觉质量略逊于PhysDreamer（使用3D几何+物理模拟器）
- 力的数值在训练场景间未标定，只有相对大小无绝对物理单位
- 质量理解限于训练时的足球vs保龄球质量比
- 全局和局部力模型分开训练，合并时局部力控制略弱
- 未探索更多力类型（重力、摩擦力、弹力）
- ControlNet仅克隆前6层受限于GPU memory，更多层可能提升视觉质量
- 目前仅适用于CogVideoX架构，迁移到其他视频模型需重新训练ControlNet

### 损失函数 / 训练策略
- 使用CogVideoX-5B-I2V原生的扩散训练损失,ControlNet引入zero convolution初始化
- AdamW优化器，学习率1e-5，cosine with restarts schedule，250步warmup
- bf16混合精度+tf32加速，batch size 8（2步梯度累积），seed=42
- 每500步保存checkpoint，总共5000步完成训练

## 相关工作与启发
- **vs Motion Prompting**: 用时空稀疏轨迹条件化，Force Prompting在力遵循和物理真实性上大幅优于它。3帧轨迹无法表达力的物理语义——轨迹不等于力
- **vs PhysDreamer**: 需每个场景的3D Gaussian表示+物理模拟器，视觉质量更好但泛化性差——只能处理训练过的具体场景。Force Prompting不需要3D信息即可泛化
- **vs PhysGen/PhysMotion**: 推理时依赖物理模拟器运行，限制了可建模的动力学类型。Force Prompting让视频模型本身充当"模拟器"
- **vs PhysCtrl**: 需学习3D点云轨迹模型再传入视频生成器，流程更复杂。Force Prompting端到端更简洁
- **vs text-only baselines**: 即使fine-tuned的文本条件模型也无法可靠传达力的方向和大小，证明力控制需要专门的条件化机制
- 该方法为embodied AI提供了一条新路径：智能体可通过force prompt预测物理交互结果，用于任务规划
- Sim2Real泛化的成功暗示预训练视频模型存在待发掘的"物理潜能"，合成数据在多种物理属性上可能都起到激活作用
- 局部力+全局力的框架可扩展到其他物理量（温度变化、电场等）作为控制信号

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 物理力作为视频生成控制信号是全新范式，质量理解发现令人兴奋
- 实验充分度: ⭐⭐⭐⭐ 人类评估设计合理，数据消融深入，但定量评估主要依赖主观偏好
- 写作质量: ⭐⭐⭐⭐⭐ 写作清晰，可视化出色，项目页面极好
- 价值: ⭐⭐⭐⭐⭐ 对世界模型和可控视频生成领域的方向性贡献，开源全套资源
<!-- NeurIPS 2025 | video_understanding -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] PhysCtrl: Generative Physics for Controllable and Physics-Grounded Video Generation](physctrl_generative_physics_for_controllable_and_physicsgrou.md)
- [\[CVPR 2025\] Motion Prompting: Controlling Video Generation with Motion Trajectories](../../CVPR2025/video_generation/motion_prompting_controlling_video_generation_with_motion_trajectories.md)
- [\[CVPR 2025\] Can Text-to-Video Generation Help Video-Language Alignment?](../../CVPR2025/video_generation/can_text-to-video_generation_help_video-language_alignment.md)
- [\[CVPR 2026\] Inference-time Physics Alignment of Video Generative Models with Latent World Models](../../CVPR2026/video_generation/inference-time_physics_alignment_of_video_generative_models_with_latent_world_mo.md)
- [\[CVPR 2025\] PhyT2V: LLM-Guided Iterative Self-Refinement for Physics-Grounded Text-to-Video Generation](../../CVPR2025/video_generation/phyt2v_llm-guided_iterative_self-refinement_for_physics-grounded_text-to-video_g.md)

</div>

<!-- RELATED:END -->
