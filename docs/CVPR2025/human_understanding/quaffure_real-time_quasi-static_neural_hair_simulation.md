---
title: >-
  [论文解读] Quaffure: Real-Time Quasi-Static Neural Hair Simulation
description: >-
  [CVPR 2025][人体理解][头发仿真] Quaffure 提出首个基于物理自监督的实时准静态头发仿真方法，通过将头发形变分解为刚性姿态变换和学习到的修正，使用改进的 Cosserat 弹性能量作为自监督损失训练 CNN 解码器，在消费级硬件上仅需几毫秒即可为不同发型、体型和姿态预测物理合理的头发悬垂效果。
tags:
  - "CVPR 2025"
  - "人体理解"
  - "头发仿真"
  - "准静态"
  - "自监督学习"
  - "Cosserat杆模型"
  - "实时推理"
---

# Quaffure: Real-Time Quasi-Static Neural Hair Simulation

**会议**: CVPR 2025  
**arXiv**: [2412.10061](https://arxiv.org/abs/2412.10061)  
**代码**: 无  
**领域**: 人体理解  
**关键词**: 头发仿真, 准静态, 自监督学习, Cosserat杆模型, 实时推理

## 一句话总结

Quaffure 提出首个基于物理自监督的实时准静态头发仿真方法，通过将头发形变分解为刚性姿态变换和学习到的修正，使用改进的 Cosserat 弹性能量作为自监督损失训练 CNN 解码器，在消费级硬件上仅需几毫秒即可为不同发型、体型和姿态预测物理合理的头发悬垂效果。

## 研究背景与动机

**领域现状**：真实的头发运动是高质量数字人的关键组成部分。物理仿真能产生高质量结果但计算量大（GPU 加速的 XPBD 方法一次准静态仿真仍需 63 秒），数据驱动方法（GroomGen）需要预先生成大量仿真训练数据。

**现有痛点**：(1) 物理仿真不适合实时应用；(2) 数据驱动方法面临数据生成瓶颈——没有公开头发仿真数据集；(3) GroomGen 只能处理单一发型和简单变化（重力方向），无法泛化到复杂姿态和体型；(4) 全 Cosserat 杆模型在 NN 训练中收敛极慢。

**核心矛盾**：实时性能和物理真实性之间的矛盾——快速的方法不够真实，真实的方法太慢。

**本文目标**：不需要仿真训练数据、3ms 内预测头发准静态悬垂效果、能泛化到不同发型/体型/姿态的神经网络方法。

**切入角度**：将头发形变分解为可精确计算的刚性姿态变换和需要学习的物理修正。仅对后者训练神经网络，且使用物理能量作为自监督损失。

**核心 idea**：用改进的 Cosserat 弹性势能（仅优化位置偏移）作为自监督损失训练 groom deformation decoder，完全避免训练数据生成。

## 方法详解

### 整体框架

输入为发型隐空间编码（16维）、体型参数（10维）和骨骼姿态（81维），输出自然悬垂的头发。管线：(1) 发型自编码器将不同发型编码为隐向量；(2) Groom Deformation Decoder 预测形变场。最终：头发 = 刚性变换 + 学习修正。所有几何用 64x64 的 2D 纹理图表示（头发丝根部映射到头皮 UV）。

### 关键设计

1. **头发形变分解策略（Posed Hair + Learned Corrector）**:

    - 功能：将复杂头发仿真分解为简单刚性变换和可学习修正
    - 核心思路：每根丝附着在头皮三角面片上，随头部姿态变化做刚性变换得到 posed hair。Deformation decoder 预测残差，最终 hair = posed + deformation
    - 设计动机：将大的刚性运动解析解决，NN 只学习较小的物理形变，降低学习难度

2. **改进的 Cosserat 弹性势能**:

    - 功能：作为自监督损失指导网络预测物理合理的形变
    - 核心思路：完整 Cosserat 需同时优化位置和朝向四元数，训练极慢。简化版只优化位置偏移，用刚性变换后的边方向替代四元数。额外添加 Hookean 拉伸势能独立控制拉伸，因为简化 Cosserat 同时惩罚拉伸和剪切导致过于刚性
    - 设计动机：完整 Cosserat 训练不可行；质点弹簧无法保持卷发的卷曲度和体积

3. **基于 SPH 的自碰撞处理**:

    - 功能：防止头发丝之间的互穿透
    - 核心思路：用 SPH 密度估计检测局部密度异常。当密度超过参考值时三次惩罚将顶点推开。body collision 类似，用带符号距离的三次惩罚维持最小距离
    - 设计动机：SPH 将碰撞检测转化为场查询，可高效 GPU 并行，三次惩罚确保梯度平滑

### 损失函数 / 训练策略

总损失 5 项：弹性势能（拉伸 + 修正 Cosserat）、重力势能、身体碰撞、自碰撞、姿态正则化。姿态正则化鼓励连续帧形变平滑。全程自监督，无需仿真数据。2D 卷积网络架构，PyTorch 实现，RTX A6000 训练。

## 实验关键数据

### 主实验

| 方法 | 推理时间 | 身体穿模率 | 长度保持 | 朝向保持 |
|------|---------|-----------|----------|----------|
| Adam (优化) | 179.38s | 0.22% | 103.53 | 76.15 |
| L-BFGS | 281.18s | 0.22% | 89.53 | 70.22 |
| XPBD (GPU) | 63.26s | 0.01% | 57.96 | 18.10 |
| GroomGen | 0.00249s | 0.39% | 1319.74 | 1281.96 |
| **Quaffure** | 0.00286s | **0.26%** | **175.42** | **286.13** |

### 消融实验

| 配置 | 效果 |
|------|------|
| 改进 Cosserat | 卷发保持卷曲度，完整模型 |
| Mass-spring 替代 | 卷发变直体积塌陷，无法保持复杂发型 |
| 高刚度 | 发型更保形，艺术控制 |
| 低刚度 | 更多重力影响，艺术控制 |

### 关键发现

- 比 GroomGen 长度保持好 7.5 倍、朝向保持好 4.5 倍，穿模更少
- 推理时间固定 ~3ms，与复杂度无关
- 可同时预测 1000 个发型仅需 0.3 秒，线性扩展
- 改进 Cosserat 比质点弹簧在保持卷发形状上有质的提升
- 时间稳定性优秀，头发平滑滑过肩膀无抖动

## 亮点与洞察

- **完全自监督的物理仿真学习**：首个将物理自监督应用于头发，不需要仿真数据/工具/专家知识。思路可迁移到布料、绳索等柔性物体
- **改进 Cosserat 的实用价值**：找到只需位置偏移不需朝向优化的近似公式，训练速度快数量级。这个简化有普遍工程价值
- **形变分解的设计哲学**：将问题分为"能精确计算的部分"和"需要学习的部分"，让 NN 只学习残差

## 局限与展望

- 只处理准静态，不能模拟动态运动（跑步时飘动）
- 所有发型使用同一组物理参数，无法区分不同发质
- 只在内部数据集评估，缺乏公开标准基准
- 对极端体型变化泛化有限

## 相关工作与启发

- **vs GroomGen**: 监督训练需仿真数据，只能处理单一发型。Quaffure 自监督，多发型，量化指标大幅领先
- **vs XPBD**: 质量最高但需 63 秒。Quaffure 以轻微质量代价换取 22000 倍速度提升
- **vs DrapNet**: 布料自监督的头发领域扩展，头发有更高自由度和更复杂自碰撞

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个头发物理自监督，改进 Cosserat 具有原创性
- 实验充分度: ⭐⭐⭐⭐ 消融充分，多基线对比完善，缺少公开基准
- 写作质量: ⭐⭐⭐⭐ 物理公式清晰，图示丰富
- 价值: ⭐⭐⭐⭐⭐ 3ms 推理可直接部署在游戏和 VR 中

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] VI3NR: Variance Informed Initialization for Implicit Neural Representations](vi3nr_variance_informed_initialization_for_implicit_neural_representations.md)
- [\[CVPR 2025\] Remote Photoplethysmography in Real-World and Extreme Lighting Scenarios](remote_photoplethysmography_in_real-world_and_extreme_lighting_scenarios.md)
- [\[CVPR 2026\] Avatar Forcing: Real-Time Interactive Head Avatar Generation for Natural Conversation](../../CVPR2026/human_understanding/avatar_forcing_real-time_interactive_head_avatar_generation_for_natural_conversa.md)
- [\[CVPR 2026\] ReMoGen: Real-time Human Interaction-to-Reaction Generation via Modular Learning from Diverse Data](../../CVPR2026/human_understanding/remogen_real-time_human_interaction-to-reaction_generation_via_modular_learning_.md)
- [\[ICML 2026\] DiscoForcing: A Unified Framework for Real-Time Audio-Driven Character Control with Diffusion Forcing](../../ICML2026/human_understanding/discoforcing_a_unified_framework_for_real-time_audio-driven_character_control_wi.md)

</div>

<!-- RELATED:END -->
