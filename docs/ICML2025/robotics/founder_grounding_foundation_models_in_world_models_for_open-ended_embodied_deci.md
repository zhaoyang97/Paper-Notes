---
title: >-
  [论文解读] FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making
description: >-
  [ICML 2025][机器人][foundation model] 提出 FOUNDER 框架，通过学习映射函数将 Foundation Model (FM) 的多模态任务表示对齐到 World Model (WM) 的状态空间，结合时间距离预测器生成奖励信号，实现无需环境奖励的开放式多任务具身决策。
tags:
  - "ICML 2025"
  - "机器人"
  - "foundation model"
  - "world model"
  - "Goal-Conditioned RL"
  - "Embodied Decision Making"
  - "Temporal Distance"
---

# FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making

**会议**: ICML 2025  
**arXiv**: [2507.12496](https://arxiv.org/abs/2507.12496)  
**代码**: [项目主页](https://sites.google.com/view/founder-rl)  
**领域**: 机器人  
**关键词**: foundation model, world model, Goal-Conditioned RL, Embodied Decision Making, Temporal Distance

## 一句话总结

提出 FOUNDER 框架，通过学习映射函数将 Foundation Model (FM) 的多模态任务表示对齐到 World Model (WM) 的状态空间，结合时间距离预测器生成奖励信号，实现无需环境奖励的开放式多任务具身决策。

## 研究背景与动机

**领域现状**：Foundation Model (FM) 拥有强大的视觉-语言泛化能力，World Model (WM) 则擅长对环境动态进行建模并通过想象生成轨迹以提升样本效率。两者在任务泛化上提供互补优势——FM 提供高层语义知识，WM 提供低层动态建模。

**现有痛点**：WM 依赖人工设计的任务奖励函数来完成特定任务，这限制了其在文本或视频等开放式任务指令下的适用性。FM 虽然能理解多模态任务指令，但其通用表示与目标 RL 环境不对齐，无法直接用于 WM 中的策略学习。

**核心矛盾**：现有方法如 GenRL 采用逐步视觉对齐（step-by-step visual alignment）来连接 FM 与 WM，但这种方式本质上是"风格迁移"——仅对齐视觉外观而非深层语义。当任务涉及复杂观测、固有静态属性或跨域视角差异时，容易产生误导性的任务解读和奖励信号，导致 reward hacking（如智能体只是在灯开关附近挥手而非真正开灯）。

**切入角度**：将 FM 表示映射为 WM 中的目标状态（goal state），将任务求解转化为 Goal-Conditioned RL (GCRL)，并用时间距离作为奖励信号来捕获深层任务语义，而非依赖视觉相似度。

**核心 idea**：学一个 FM→WM 的映射函数来"推断"多模态任务提示对应的物理状态，再用时间距离预测器生成信息丰富的奖励，在 WM 中通过想象进行目标导向的策略学习。

## 方法详解

### 整体框架

FOUNDER 分为两个阶段：

1. **预训练阶段**：(a) 用离线数据训练 DreamerV3 风格的 World Model；(b) 学习状态间的时间距离预测器；(c) 学习从 VLM 表示到 WM 状态的映射函数。
2. **行为学习阶段**：给定文本/视频任务提示 → VLM 编码 → 映射到 WM 目标状态 → 在 WM 中用时间距离作为奖励进行 GCRL 策略学习。

三个预训练组件协同支撑行为学习，且均为任务无关的，可复用于多种下游任务。

### 关键设计

1. **World Model 学习**：采用 DreamerV3 的 RSSM 架构，在离线观测-动作数据上优化 ELBO。状态 $z_t = (h_t, s_t)$ 包含确定性和随机性部分，保留完整的历史信息。与 GenRL 不同，FOUNDER 保持了原始 DreamerV3 结构（编码器和解码器均条件化于 $h_t$），使 WM 状态包含更丰富的环境信息。不含奖励模型（因数据无奖励标注）。

2. **映射函数学习（FM→WM Grounding）**：核心思路是将 VLM 对短视频的嵌入 $e_t$ 视为环境中某个物理状态 $z_t$ 的外部观测，学习推断函数 $Q_\psi(z|e)$。具体做法：

    - 用离线轨迹构建配对数据：$e_t = \text{VLM}(o_{t-k:t})$ 和 $z_t \sim \text{WM}(\cdot|o_{\leq t}, a_{<t})$
    - 优化两个目标：(a) KL 约束使映射状态分布对齐 WM 编码分布；(b) 自编码器重建损失确保映射状态保留完整的 VLM 语义信息
    - 这等价于变分推断，但用 WM 状态对齐替代了随机先验的 KL 约束
    - 推理时，任务提示经 VLM 编码后直接映射为目标状态：$z_g \sim Q_\psi(\cdot|e_g)$

3. **时间距离预测器**：学习模型 $D_\theta$ 预测两个 WM 状态间的时间步距离。从同一轨迹采样正样本对 $(z_t, z_{t+c})$ 预测归一化距离 $c/T$；从不同轨迹采样负样本对预测最大距离 1。核心公式：$\min_{D_\theta} \text{MSE}(D_\theta(z_t, z_{t+c}), c/T)$。时间距离比余弦相似度更鲁棒，因其提取了环境动态信息，能捕获超越视觉细节的深层任务语义。

### 损失函数 / 训练策略

- **WM 训练**：标准 ELBO 目标（重建 + KL 散度），500K 梯度步
- **映射函数**：$\mathcal{L}_{map} = \mathbb{D}_{KL}[Q_\psi(\cdot|e_t) \| \text{WM}(\cdot|o_{\leq t}, a_{<t})] + \mathbb{E}[-\ln P_\psi(e_t|\hat{z}_t)]$
- **时间距离**：MSE 损失（正负样本对）
- **策略学习**：奖励 $r_D(z_t, z_g) = -D_\theta(z_t, z_g)$，DreamerV3 风格 Actor-Critic，行为学习最多 50K 步
- VLM 使用 InternVideo2，所有方法共享同一 VLM

## 实验关键数据

### 主实验（DMC + Kitchen 多任务文本指令）

| 任务 | GenRL | WM-CLIP | FOUNDER w/o TempD | FOUNDER | 提升 |
|------|-------|---------|-------------------|---------|------|
| Cheetah Flip | -0.04 | -0.11 | -0.26 | **0.97** | 从失败到成功 |
| Cheetah Run | 0.68 | 0.51 | 0.21 | **0.81** | +19% |
| Kitchen Light | 0.00 | 0.35 | 1.00 | **0.97** | 从0到近满分 |
| Kitchen Burner | 0.35 | 0.10 | 1.00 | **0.60** | +71% vs GenRL |
| Overall (19 tasks) | 0.60 | 0.57 | 0.52 | **0.81** | +35% |

FOUNDER 在 19 个任务中有 14 个排名最高，Overall 从 0.60 提升至 0.81。

### 消融实验

| 配置 | Overall (19 tasks) | 说明 |
|------|-------------------|------|
| FOUNDER (完整) | **0.81** | 映射 + 时间距离 + GCRL |
| FOUNDER w/o TempD | 0.52 | 用余弦相似度替代时间距离，严重退化 |
| GenRL-TempD | 0.59 | 仅给 GenRL 加时间距离，效果不佳 |
| GenRL | 0.60 | 逐步视觉对齐 |
| WM-CLIP | 0.57 | 在 VLM 空间对齐 |

**关键发现**：时间距离单独加到 GenRL 上无效（GenRL-TempD 0.59 vs GenRL 0.60），只有在 FOUNDER 的 GCRL 框架内才发挥作用。FOUNDER w/o TempD 大幅退化到 0.52，说明余弦相似度易导致 reward hacking。

### 奖励一致性评估

| 方法 | Corr↑ | Regret↓ | Precision↑ | F1↑ |
|------|-------|---------|-----------|-----|
| GenRL | 0.12 | 0.37 | 0.47 | 0.44 |
| FOUNDER | **0.54** | **0.07** | **1.00** | **0.59** |

FOUNDER 学到的伪奖励与真实奖励一致性最高，Precision=1.0 意味着不会误判低奖励为高奖励，有效避免 reward hacking。

### 关键发现

- **跨域迁移**：在 12 个跨体态（cross-embodiment）任务中 FOUNDER 赢了 11 个，是唯一在 Cheetah 域成功完成以 Walker/Stickman 视频为提示的方法
- **Minecraft**：在 5 个任务中 3 个显著超越 GenRL，且匹配或超越使用大规模互联网数据预训练的 MineCLIP oracle
- **GenRL 的本质局限**：GenRL 在 Kitchen 静态任务上完全失败（Light=0.00），因其缺乏时间感知，仅进行视觉级别匹配

## 亮点与洞察

- **优雅的问题转化**：将"理解多模态任务提示"转化为"推断环境模拟器中的物理状态"，这个视角比逐步对齐更深刻
- **时间距离 vs 余弦相似度**：揭示了在 WM 状态空间中余弦相似度的脆弱性——容易被静态的"视觉模仿"所欺骗，而时间距离编码了动态信息
- **精度优于召回**：在策略学习的奖励设计中，避免 False Positive（误判低奖励为高奖励）比识别所有高奖励行为更重要，类似代价敏感学习

## 局限与展望

- 依赖离线数据质量——如果提示中的元素未在数据集中出现，映射可能失败
- 目前实验主要聚焦短时域任务，长时域任务需要结合 FM 的推理能力进行任务分解
- 映射函数仅在目标环境的交互数据上训练，虽然 VLM 的泛化能力可扩展到其他域，但对分布外场景的鲁棒性需要更多验证
- 可探索真实世界视频融入 WM 学习以扩展能力边界

## 相关工作与启发

- **GenRL** (Mazzaglia et al., 2024)：最相关的工作，用 seq2seq 方式建模 FM→WM 的映射，但本质是视觉级别的风格迁移
- **LEXA** / **Director**：在 WM 中进行 GCRL 的先驱工作，但用于探索或子任务分解而非直接行为学习
- **MineDojo** (Fan et al., 2022)：提供了 Minecraft 开放任务基准，本文在其上验证了 FOUNDER 的有效性
- 启发：FM 和 WM 的融合是具身智能的重要方向，关键在于如何在合适的表示空间中建立桥梁

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models](hi_robot_open-ended_instruction_following_with_hierarchical_vision-language-acti.md)
- [\[ICML 2025\] SENSEI: Semantic Exploration Guided by Foundation Models to Learn Versatile World Models](sensei_semantic_exploration_guided_by_foundation_models_to_learn_versatile_world.md)
- [\[NeurIPS 2025\] Self-Improving Embodied Foundation Models](../../NeurIPS2025/robotics/self-improving_embodied_foundation_models.md)
- [\[CVPR 2025\] Decision SpikeFormer: Spike-Driven Transformer for Decision Making](../../CVPR2025/robotics/decision_spikeformer_spike-driven_transformer_for_decision_making.md)
- [\[CVPR 2025\] UniAct: Universal Actions for Enhanced Embodied Foundation Models](../../CVPR2025/robotics/universal_actions_for_enhanced_embodied_foundation_models.md)

</div>

<!-- RELATED:END -->
