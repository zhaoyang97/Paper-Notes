---
title: >-
  [论文解读] VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents
description: >-
  [NeurIPS 2025][多模态VLM][VLM Agent] 提出VAGEN框架，通过将VLM智能体的推理过程结构化为StateEstimation和TransitionModeling来构建内部世界模型，结合WorldModeling Reward和Bi-Level GAE实现高效的多轮RL训练，使3B模型（0.82）超越GPT-5（0.75）和Gemini 2.5 Pro（0.67）。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "VLM Agent"
  - "世界模型"
  - "强化学习"
  - "POMDP"
  - "多轮交互"
---

# VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents

**会议**: NeurIPS 2025  
**arXiv**: [2510.16907](https://arxiv.org/abs/2510.16907)  
**代码**: [http://mll.lab.northwestern.edu/VAGEN](http://mll.lab.northwestern.edu/VAGEN)  
**领域**: 多模态VLM  
**关键词**: VLM Agent, 世界模型, 强化学习, POMDP, 多轮交互

## 一句话总结

提出VAGEN框架，通过将VLM智能体的推理过程结构化为StateEstimation和TransitionModeling来构建内部世界模型，结合WorldModeling Reward和Bi-Level GAE实现高效的多轮RL训练，使3B模型（0.82）超越GPT-5（0.75）和Gemini 2.5 Pro（0.67）。

## 研究背景与动机

多轮智能体任务要求准确解读和跟踪动态环境，当智能体通过视觉而非文本感知世界时挑战倍增。VLM智能体任务本质上是**部分可观测马尔可夫决策过程（POMDP）**：智能体接收的视觉观测 $o_t$ 只是真实状态 $s_t$ 的部分视图，需要先估计世界真实状态才能有效行动。

**核心问题**：能否教会VLM智能体通过显式的视觉状态推理来构建内部世界模型？

现有工作的不足：
- 当前VLM智能体在多轮任务中缺乏显式的内部世界建模来增强视觉状态推理
- 现有RL框架（如VLM-R1）主要基于单轮优化，无法捕捉演化的交互上下文
- 标准GAE方法在多轮场景中存在稀疏奖励信号长距离传播不稳定的问题
- 没有现成的VLM能很好地解决多轮视觉智能体任务（最强的GPT-5也只有0.75）

## 方法详解

### 整体框架

将多轮VLM智能体任务建模为POMDP $(\mathcal{S}, \mathcal{O}, \mathcal{A}, P, R, \Omega, \gamma)$。在每个轮次 $t$，智能体生成包含推理token和可执行动作的输出 $a_t = \langle z_t, a_t^e \rangle$。关键创新在于将推理token $z_t$ 结构化为世界模型的两个核心组件。

### 关键设计

1. **五种推理策略的系统比较**：通过控制RL训练中的格式奖励，研究了从隐式到显式的五种推理策略：

    - **NoThink**：仅生成可执行动作，$z_t = \emptyset$
    - **FreeThink**：自由形式自然语言推理
    - **StateEstimation**：显式描述当前状态信念 $\hat{s}_t$，学习 $\hat{s}_t \to s_t$
    - **TransitionModeling**：显式预测下一状态 $\hat{s}_{t+1}$，学习 $\hat{s}_{t+1} \to s_{t+1}$
    - **WorldModeling**：同时包含状态估计和转移建模，完整世界模型

   结论：WorldModeling（0.76）> FreeThink（0.67）>> NoThink（0.28），显式视觉状态推理至关重要。

2. **视觉状态表示选择**：探索了三种内部信念表示方式：

    - **自然语言**：通用任务中最优，利用预训练语义知识（Sokoban: 0.61, FrozenLake: 0.71）
    - **结构化格式**：高精度操控任务中最优，提供精确坐标（PrimitiveSkill avg: 0.94）
    - **符号表示**：最差，模型难以将抽象符号与原始视觉输入对接
   
   核心洞察：**表示选择是任务依赖的**，非通用解。

3. **WorldModeling Reward**：利用LLM-as-a-Judge框架评估智能体的状态描述和预测与真实状态的匹配度：
    $r_t^{reason} = \beta_s \cdot \mathbb{I}_{\text{StateEstimation}}(\hat{s}_t, s_t) + \beta_w \cdot \mathbb{I}_{\text{TransitionModeling}}(\hat{s}_{t+1}, s_{t+1})$
   提供逐轮密集奖励信号，弥补任务奖励的稀疏性。

4. **Bi-Level GAE**：分两层计算优势估计，解决多轮场景中的信用分配问题：

    - **Turn-level**：在轮次间用 $\gamma_{\text{turn}}$ 计算轮次级TD误差和优势
    $\delta_t^{turn} = r_t + \gamma_{turn} V_\phi(\bar{\tau}_{\leq a_{t+1}}) - V_\phi(\bar{\tau}_{\leq a_t})$
    - **Token-level**：在各轮内部用 $\gamma_{\text{token}}$ 传播轮次优势到token级
    - 关键链接：用已计算的轮次优势 $A_t^{turn}$ 初始化该轮最后一个token的优势，层次化传递

### 损失函数 / 训练策略

使用PPO目标，带观测token掩码确保不从观测token学习（关键发现：不掩码会导致训练失败）。组合奖励为 $r_t = r_t^{reason} + r_t^{format} + R(s_t, a_t)$。训练使用Qwen2.5-VL-3B作为骨干，$\beta_s = \beta_w = 0.5$。

## 实验关键数据

### 主实验

| 模型/方法 | Sokoban | FrozenLake | Navigation | PrimitiveSkill | SVG | Overall |
|-----------|---------|------------|------------|---------------|-----|---------|
| Qwen2.5-VL-3B (原始) | 0.14 | 0.22 | 0.24 | 0.00 | 0.54 | **0.21** |
| GPT-5 | 0.70 | 0.75 | 0.78 | 0.66 | 0.85 | **0.75** |
| Claude 4.5 Sonnet | 0.31 | 0.67 | 0.67 | 0.53 | 0.88 | **0.64** |
| Gemini 2.5 Pro | 0.58 | 0.63 | 0.63 | 0.50 | 0.86 | **0.67** |
| VAGEN-Base (WorldModeling) | 0.61 | 0.78 | 0.79 | 0.91 | 0.78 | **0.76** |
| **VAGEN-Full** | **0.79** | **0.80** | **0.81** | **0.97** | **0.79** | **0.82** |

### 消融实验

| 配置 | Sokoban | Navigation | PrimitiveSkill | 说明 |
|------|---------|------------|---------------|------|
| NoThink | 0.57/0.09 | 0.00 | 0.00 | 无推理直接崩溃 |
| FreeThink | 0.57/0.68 | 0.67 | 0.66 | 隐式推理有效但不充分 |
| StateEstimation | 0.56/0.68 | 0.74 | 0.00 | 导航好，操控差 |
| TransitionModeling | 0.41/0.76 | 0.62 | 0.82 | 操控好，导航稍差 |
| WorldModeling | 0.61/0.71 | 0.79 | 0.91 | 最均衡 |
| VAGEN-Full | **0.79/0.74** | **0.81** | **0.97** | 奖励+信用分配全面提升 |

**推理策略 vs RL基线**：Vanilla-PPO (0.26) << GRPO w/ Mask (0.54) < Turn-PPO (0.55) << VAGEN-Base (0.76) < VAGEN-Full (0.82)

### 关键发现

- 3B模型通过VAGEN训练后（0.82）超越所有专有推理模型（GPT-5: 0.75，o3: 0.73）
- StateEstimation擅长导航（理解当前观测），TransitionModeling擅长操控（预测未来状态），WorldModeling最均衡
- 观测token掩码是必须的——不掩码导致Vanilla-PPO完全失败
- Bi-Level GAE和WorldModeling Reward各自贡献不一致的增益，但两者结合最稳定

## 亮点与洞察

- 将VLM智能体建模为POMDP并显式构建世界模型是一个优雅的理论框架
- 五种推理策略的系统对比提供了关于"VLM智能体应该思考什么"的深入洞察
- 小模型超越大模型的结果非常有说服力，展示了正确的训练范式比模型规模更重要
- Bi-Level GAE解决了多轮RL的关键痛点（稀疏奖励的信用分配），具有通用价值

## 局限与展望

- LLM-as-a-Judge的奖励信号可能不够精确，存在奖励黑客风险（论文中观察到响应收敛和过优化现象）
- 需要环境提供ground-truth状态信息来计算WorldModeling Reward，限制了应用范围
- SVG任务没有世界动态，只能用Bi-Level GAE，WorldModeling Reward不适用
- 未在开放世界环境中验证（所有任务都是结构化的）

## 相关工作与启发

- **vs VLM-R1**: VLM-R1是单轮RL，在多轮智能体任务上没有优势；VAGEN优化完整交互轨迹
- **vs DeepSeek-R1**: 借鉴了R1的格式奖励思路，但扩展到多轮场景并增加了世界模型推理
- **vs GiGPO/AReaL**: 并发工作关注长视野信用分配和异步扩展，但未探索显式世界模型推理

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ POMDP视角+世界模型推理+Bi-Level GAE组合独特，五种策略的系统研究有开创性
- 实验充分度: ⭐⭐⭐⭐⭐ 5种任务、5种推理策略、3种表示、多种RL基线、专有模型对比，非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，研究问题层层递进，表格信息量大
- 价值: ⭐⭐⭐⭐⭐ 3B超越GPT-5的结果令人印象深刻，为VLM智能体训练提供了系统性指南

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Video-R1: Reinforcing Video Reasoning in MLLMs](video-r1_reinforcing_video_reasoning_in_mllms.md)
- [\[NeurIPS 2025\] SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards](spatialthinker_reinforcing_3d_reasoning_in_multimodal_llms_via_spatial_rewards.md)
- [\[ICML 2025\] Towards Efficient Online Tuning of VLM Agents via Counterfactual Soft Reinforcement Learning](../../ICML2025/multimodal_vlm/towards_efficient_online_tuning_of_vlm_agents_via_counterfactual_soft_reinforcem.md)
- [\[NeurIPS 2025\] Recognition through Reasoning: Reinforcing Image Geo-localization with Large Vision-Language Models](recognition_through_reasoning_reinforcing_image_geo-localization_with_large_visi.md)
- [\[CVPR 2026\] STAR-R1: Multi-View Spatial TrAnsformation Reasoning by Reinforcing Multimodal LLMs](../../CVPR2026/multimodal_vlm/star-r1_multi-view_spatial_transformation_reasoning_by_reinforcing_multimodal_ll.md)

</div>

<!-- RELATED:END -->
