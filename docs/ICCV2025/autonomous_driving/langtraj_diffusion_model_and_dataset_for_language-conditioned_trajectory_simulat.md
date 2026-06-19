---
title: >-
  [论文解读] LangTraj: Diffusion Model and Dataset for Language-Conditioned Trajectory Simulation
description: >-
  [ICCV 2025][自动驾驶][traffic simulation] 提出 LangTraj，首个在训练阶段直接以自然语言为条件的扩散模型轨迹仿真器，并构建了包含 150K 人工标注交互行为的 InterDrive 数据集，支持语言可控的多智能体交互仿真和安全关键场景生成。 交通仿真是自动驾驶安全测试的核心环节…
tags:
  - "ICCV 2025"
  - "自动驾驶"
  - "traffic simulation"
  - "language conditioning"
  - "扩散模型"
  - "trajectory generation"
---

# LangTraj: Diffusion Model and Dataset for Language-Conditioned Trajectory Simulation

**会议**: ICCV 2025  
**arXiv**: [2504.11521](https://arxiv.org/abs/2504.11521)  
**代码**: [项目主页](https://langtraj.github.io/)  
**领域**: 自动驾驶  
**关键词**: traffic simulation, language conditioning, diffusion model, trajectory generation, autonomous driving

## 一句话总结

提出 LangTraj，首个在训练阶段直接以自然语言为条件的扩散模型轨迹仿真器，并构建了包含 150K 人工标注交互行为的 InterDrive 数据集，支持语言可控的多智能体交互仿真和安全关键场景生成。

## 研究背景与动机

交通仿真是自动驾驶安全测试的核心环节。现有方法存在以下问题：

**手工设计场景不可扩展**：传统结构化测试依赖人工设计失效场景，无法覆盖长尾分布

**扩散模型可控性受限**：现有扩散轨迹生成方法（CTG、SAFE-SIM 等）依赖后训练的领域特定引导函数（guidance function），需要专家知识设计且推理变慢

**语言条件的缺失**：语言是最直觉的控制接口，但尚无方法在训练阶段直接学习语言-轨迹的映射关系

**交互行为数据稀缺**：现有数据集（如 ProSim-Instruct）以单智能体行为为主，缺乏多智能体交互的精细标注

核心洞察：**在训练时直接引入语言条件，使模型从数据分布中学习语义，而非在推理时用启发式引导函数逼近。语言条件与引导函数正交，可以组合使用。**

## 方法详解

### 整体框架

LangTraj 包含三大组件：Scene Encoder、Language Encoder、Denoiser，以扩散模型联合建模所有智能体的未来轨迹。额外提出闭环训练策略提升闭环仿真稳定性。

### 关键设计

1. **Scene Encoder（场景编码器）**: 采用 query-centric + GNN 方法，在每个场景元素的局部坐标系中提取特征（与全局坐标无关），通过注意力机制编码相对时空信息，输出每个智能体的嵌入 $\mathbf{z}_{enc}^i = E_{enc}(I, \mathbf{S}_{t-T_{hist}:t})$。设计动机：局部坐标系保证跨智能体的对称编码。

2. **Language Encoder（语言编码器）**: 对输入句子进行智能体角色重写（"target agent"、"other agent1" 等），通过 DistilBERT 提取句子嵌入 $\mathbf{e}_{lang}$，再与场景嵌入融合：

    $\mathbf{z}_{lang}^i = E_L(\mathbf{e}_{lang}, \mathbf{z}_{enc}^i)$

   使用 LoRA 端到端微调，平衡效率和效果。设计动机：角色重写确保语言条件能精确对应到特定智能体。

3. **Denoiser（去噪器）**: 堆叠 Transformer 块，包含三种注意力：

    - 智能体间 query-centric 注意力：捕获多智能体交互
    - Agent-Context 交叉注意力：对齐行为与时空上下文
    - Text-Cross 注意力：注入语言条件

4. **闭环训练策略**: 核心创新之一。传统扩散模型以开环方式训练，导致闭环推理时分布偏移和误差累积。本文提出将模型自身生成的样本融入训练：

    - 每步对 GT 轨迹加噪 → 一步去噪生成 $M$ 个候选 → 选择距 GT 最近的候选执行
    - 在全局坐标系中计算执行轨迹与 GT 的 L2 损失
    - 采用 teacher forcing：部分智能体仍跟随 GT，稳定训练

5. **双模式可控生成**:

    - **Classifier-free guidance**：插值条件/无条件预测，$\hat{\tau}_0 = (1+w) \cdot g(\mathbf{e}_{lang}, \mathbf{z}_{enc}) - w \cdot g(\emptyset, \mathbf{z}_{enc})$
    - **Classifier-based guidance**：通过目标函数 $J(\tau)$ 的梯度修改去噪步均值，支持碰撞引导等

### 损失函数 / 训练策略

开环预训练 + 闭环微调。闭环训练中损失为执行轨迹与 GT 在全局坐标系的 L2 距离。降低去噪步数 $K=100 \to 5$ 不影响性能，大幅提升效率。

## 实验关键数据

### 主实验（WOSAC 测试集）

| 方法 | Meta ↑ | Kinematic ↑ | Interactive ↑ | Map ↑ |
|------|--------|------------|--------------|-------|
| UniMM | 0.769 | 0.491 | 0.811 | 0.874 |
| VBD | 0.720 | 0.417 | 0.814 | 0.776 |
| **LangTraj** | **0.719** | **0.426** | **0.795** | **0.789** |
| ProSim | 0.718 | 0.401 | 0.778 | 0.822 |
| SceneDiffuser | 0.703 | 0.430 | 0.776 | 0.768 |

- 在扩散模型中表现**最优**，与 VBD 和 SceneDiffuser 竞争力相当

### 语言可控性评估

| 文本条件方式 | Meta ↑ | Map ↑ | mADE ↓ |
|-------------|--------|-------|--------|
| 无条件 | 0.72 | 0.80 | 2.65 |
| **直接条件 (本文)** | **0.72** | **0.79** | **2.29** |
| LLM-Based Guidance (CTG++) | 0.70 | 0.80 | 2.70 |

- 语言直接条件使 mADE 降低 **13.6%**，而 LLM-based guidance 甚至不如无条件 baseline

### 安全关键场景生成

| 设置 | 碰撞率 ↑ | Kinematic ↑ | Map ↑ |
|------|---------|------------|-------|
| 无文本 + 无引导 | 0.04 | 0.42 | 0.81 |
| 无文本 + 碰撞引导 | 0.41 | 0.39 | 0.70 |
| **直接条件 + 碰撞引导** | **0.43** | **0.41** | **0.74** |
| LLM Guidance + 碰撞引导 | 0.33 | 0.37 | 0.72 |

### 消融实验

| 设置 | Meta ↑ | Map ↑ |
|------|--------|-------|
| 开环 (K=100) | 0.68 | 0.73 |
| 开环 (K=5) | 0.68 | 0.72 |
| 闭环 | 0.69 | 0.75 |
| **闭环 + Teacher Forcing** | **0.70** | **0.79** |

### 关键发现

- 语言直接条件远优于 LLM-based guidance，尤其在交互行为描述上
- 闭环训练 + teacher forcing 显著改善 map adherence（0.72 → 0.79），防止模型漂移
- 去噪步数从 100 降到 5 不影响真实度，5 步足够
- 语言条件与碰撞引导正交且互补，组合使用可达 43% 碰撞率 + 更好的 map 指标

## 亮点与洞察

- **训练时直接语言条件 vs 推理时引导函数**：本文用实验有力证明了直接条件的优越性，这对整个可控生成领域有启发意义
- **InterDrive 数据集**：150K 人工标注的交互行为（merging/yielding/passing 等 6 大类），填补了交互级语言标注的空白
- **闭环训练**扩散模型的方案简洁有效，通过单步去噪 + 选择最优候选避免了双重循环的计算开销

## 局限与展望

- 数据集以 Waymo 为主，其他域（如中国道路场景）的泛化性待验证
- DistilBERT 作为语言编码器能力有限，换用更强 LLM 可能进一步提升
- 闭环训练需要在收敛速度和 teacher forcing 比例间仔细平衡
- 未来可探索多轮对话式场景编辑

## 相关工作与启发

- 与 ProSim 的对比表明扩散模型比自回归方法更适合推理时引导和分布外采样
- 闭环训练策略可迁移到其他基于扩散的序列生成任务（如机器人操作规划）
- 语言-轨迹对齐与 text-to-motion 领域方法有潜在交叉

## 评分

- 新颖性：⭐⭐⭐⭐ — 首个训练时语言条件 + 闭环训练扩散轨迹模型
- 技术深度：⭐⭐⭐⭐ — 闭环训练策略设计巧妙，多种可控生成方式融合
- 实验充分度：⭐⭐⭐⭐⭐ — WOSAC 基准 + 语言对齐 + 安全关键 + 消融，非常全面
- 实用价值：⭐⭐⭐⭐ — 支持灵活 AV 测试，但需大规模语言标注数据支撑

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Epona: Autoregressive Diffusion World Model for Autonomous Driving](epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)
- [\[ICCV 2025\] Generative Active Learning for Long-tail Trajectory Prediction via Controllable Diffusion Model](generative_active_learning_for_long-tail_trajectory_prediction_via_controllable_.md)
- [\[ICCV 2025\] DONUT: A Decoder-Only Model for Trajectory Prediction](donut_a_decoder-only_model_for_trajectory_prediction.md)
- [\[CVPR 2026\] SparseWorld-TC: Trajectory-Conditioned Sparse Occupancy World Model](../../CVPR2026/autonomous_driving/sparseworld_tc_trajectory_conditioned_sparse_occupancy_world_model.md)
- [\[ICCV 2025\] LightsOut: Diffusion-based Outpainting for Enhanced Lens Flare Removal](lightsout_diffusion-based_outpainting_for_enhanced_lens_flare_removal.md)

</div>

<!-- RELATED:END -->
