---
title: >-
  [论文解读] UniAct: Universal Actions for Enhanced Embodied Foundation Models
description: >-
  [CVPR 2025][机器人][具身智能] UniAct提出在通用动作空间（Universal Action Space）中构建具身基础模型，通过向量量化codebook编码跨具身平台共享的原子行为，0.5B参数模型性能超越14倍大的SOTA模型，并支持快速适配新机器人。 开发通用具身基础模型面临的核心挑战是动作异质性（a…
tags:
  - "CVPR 2025"
  - "机器人"
  - "具身智能"
  - "通用动作空间"
  - "跨具身迁移"
  - "视觉语言动作模型"
  - "向量量化"
---

# UniAct: Universal Actions for Enhanced Embodied Foundation Models

**会议**: CVPR 2025  
**arXiv**: [2501.10105](https://arxiv.org/abs/2501.10105)  
**代码**: [项目页面](https://2toinf.github.io/UniAct/)  
**领域**: 机器人  
**关键词**: 具身智能, 通用动作空间, 跨具身迁移, 视觉语言动作模型, 向量量化

## 一句话总结

UniAct提出在通用动作空间（Universal Action Space）中构建具身基础模型，通过向量量化codebook编码跨具身平台共享的原子行为，0.5B参数模型性能超越14倍大的SOTA模型，并支持快速适配新机器人。

## 研究背景与动机

开发通用具身基础模型面临的核心挑战是**动作异质性（action heterogeneity）**：

- **具身差异**：不同自由度的机器人（机械臂、四足、汽车）拥有完全不同的动作空间
- **控制接口差异**：即使同一机器人，末端执行器位置控制和速度控制具有根本不同的物理含义
- **行为多模态性**：不同操作者在同一平台上收集的数据也存在多模态性

现有方案的不足：
- **粗暴统一**：RT-X、Octo、OpenVLA强行将不同动作空间视为等价，导致相似编码可能代表完全不同的物理含义
- **朴素聚合**：CrossFormer、RDT聚合所有动作空间但未挖掘跨平台的共性
- **潜在动作**：LAPA等通过视频帧变化推断潜在动作，但会捕获与控制无关的干扰信息（如新物体出现）

关键洞察是：尽管不同机器人的控制信号差异巨大，面对正前方目标时它们应该执行相似的"向前移动"行为。这种抽象的原子行为可以跨具身共享。

## 方法详解

### 整体框架

UniAct基于预训练的VLM（LLaVA-OneVision-0.5B）构建，包含三个核心组件：(1) 共享VLM作为通用动作提取器；(2) 向量量化codebook $\mathcal{U} \in \mathbb{R}^{256 \times 128}$作为通用动作空间；(3) 轻量级异构解码头将通用动作翻译为具体控制信号。

### 关键设计

**设计一：通用动作空间 — 离散向量量化Codebook**

- **功能**：将跨具身平台的异构动作蒸馏为共享的原子行为表示
- **核心思路**：使用$N=256$个$D=128$维向量组成codebook $\mathcal{U} = (u_1, u_2, \ldots, u_N)$，每个code编码一个通用的原子行为。所有机器人被迫使用同一codebook，形成关键的信息瓶颈，驱动模型发现和利用跨平台的共享原始行为
- **设计动机**：离散表示在复杂推理、规划和预测学习中展现了强大能力（如LLM的成功）。限制为离散空间迫使模型压缩信息，提取真正跨平台共享的行为本质

**设计二：通用动作提取器 — 基于VLM的任务导向提取**

- **功能**：根据观察$o$和任务目标$g$推断最相关的通用动作$u^* = \arg\max_{u \in \mathcal{U}} p(u|o,g)$
- **核心思路**：微调预训练VLM输出codebook上的概率分布，通过Gumbel-Softmax实现可微分的动作选择：$u^* = \sum_{i=1}^n w_i u_i$，其中权重$w_i$通过Gumbel-Softmax计算。训练过程中逐步降低温度$\tau$
- **设计动机**：与仅通过视频帧变化推断潜在动作不同，本方法以任务进展为导向提取通用动作，避免捕获与控制无关的观测变化。利用VLM的视觉-语言推理能力和预训练知识提高样本效率

**设计三：异构解码头 — 轻量级具身特定翻译**

- **功能**：将高度抽象的通用动作翻译为各具身平台可执行的精确控制信号
- **核心思路**：为每种具身类型设计简单的MLP解码头$h_k$，输入通用动作$u^*$和视觉特征$o$，输出具身特定的控制命令$\hat{a}^{(k)} = h_k(u^*, o)$。新机器人适配只需添加新的解码头
- **设计动机**：保持解码头轻量确保主要学习集中在通用动作空间，最大化跨具身泛化能力。通用行为已被捕获，解码器只需添加具身细节

### 损失函数

总训练目标为所有领域的行为克隆损失之和：$\min_{\mathcal{U},\theta} \sum_{k=1}^K \mathbb{E}_{a_i \in \tau_i, \tau_i \in \mathcal{D}_k} \mathcal{L}_k(\hat{a}^{(k)}, a_i^{(k)})$，其中$\mathcal{L}_k$可根据动作类型定制（离散动作用交叉熵，连续动作用MSE/Huber/扩散损失）。Codebook和提取器全局更新，解码头按领域更新。

## 实验关键数据

### 主实验：真实世界WidowX机器人（19个任务，190次rollout）

| 模型 | 参数量 | Visual | Motion | Physical | Semantic | Language |
|------|--------|--------|--------|----------|----------|----------|
| Octo | 0.1B | 低 | 低 | 低 | 低 | 低 |
| CrossFormer | 0.1B | 低 | 低 | 低 | 低 | 低 |
| OpenVLA | 7B | 中 | 中 | 中 | **高** | **高** |
| **UniAct** | **0.5B** | **高** | **高** | **高** | 高 | 高 |

UniAct-0.5B在视觉、运动、物理泛化任务上超越14倍大的OpenVLA-7B。

### 消融/适配实验：新机器人AIRBOT快速适配

| 预训练模型 | Sweep Plate | Fold Towel | Cup on Plate | Transport Pen |
|-----------|-------------|------------|--------------|---------------|
| LLaVa-OV-0.5B | 7.5% | 20% | 2.5% | 15% |
| **UniAct-0.5B** | **45%** | **62.5%** | **50%** | **65%** |

UniAct仅用0.8%参数（4M/500M）微调即可适配新机器人，远低于OpenVLA的1.4%和Octo的2%。

### 关键发现

- 通过手动检查256个通用动作，至少40%在不同机器人间展现完全一致的语义行为
- 同一任务在不同机器人上的通用动作利用分布相似（低JS散度），不同任务则不同
- 可以直接通过选择通用动作ID手动控制机器人执行复杂任务，无需了解正/逆运动学

## 亮点与洞察

1. **信息瓶颈的精妙设计**：离散codebook强制不同具身共享同一抽象空间，自然地驱动模型发现跨平台共性
2. **任务导向 vs 观测导向**：通用动作基于任务进展而非视频帧差异提取，避免了无关视觉变化的干扰
3. **极致的高效性**：0.5B模型碾压7B模型，证明了"正确的表示空间比模型规模更重要"

## 局限与展望

- 当前仅使用0.5B参数和单臂机器人评估，受资源限制
- 未来将扩展至更大模型和更多具身类型（双臂机器人、自动驾驶）
- 通用动作提取器可作为动作tokenizer，为未来大规模具身基础模型的规划提供支持

## 相关工作与启发

- **OpenVLA/RT-X/Octo**：直接在异构动作空间上训练，未解决动作语义冲突
- **VQ-BeT/QueST**：单具身场景下的离散动作编码，处理人类演示多模态性
- **LAPA/IGOR**：通过视频帧变化推断潜在动作，缺乏控制因果关系
- 启发：跨域/跨模态学习的关键在于找到正确的共享表示空间，离散瓶颈是实现这一目标的有效工具

## 评分

⭐⭐⭐⭐⭐ — 工作极具开创性，通用动作空间的概念清晰优雅，实验设计全面（真实机器人+仿真+新机器人适配），0.5B碾压7B的结果令人印象深刻。对具身AI领域的表示学习具有重要启示意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Self-Improving Embodied Foundation Models](../../NeurIPS2025/robotics/self-improving_embodied_foundation_models.md)
- [\[ICML 2025\] FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making](../../ICML2025/robotics/founder_grounding_foundation_models_in_world_models_for_open-ended_embodied_deci.md)
- [\[CVPR 2025\] DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness](dexgrasp_anything_towards_universal_robotic_dexterous_grasping_with_physics_awar.md)
- [\[CVPR 2025\] Lift3D Foundation Policy: Lifting 2D Large-Scale Pretrained Models for Robust 3D Robotic Manipulation](lift3d_policy_lifting_2d_foundation_models_for_robust_3d_robotic_manipulation.md)
- [\[NeurIPS 2025\] Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics](../../NeurIPS2025/robotics/robot-r1_reinforcement_learning_for_enhanced_embodied_reasoning_in_robotics.md)

</div>

<!-- RELATED:END -->
