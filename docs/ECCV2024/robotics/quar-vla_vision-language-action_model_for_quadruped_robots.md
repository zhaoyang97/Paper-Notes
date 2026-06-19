---
title: >-
  [论文解读] QUAR-VLA: Vision-Language-Action Model for Quadruped Robots
description: >-
  [ECCV 2024][机器人][quadruped robot] 首次提出四足机器人视觉-语言-动作（QUAR-VLA）范式，构建 259K episode 的多任务数据集 QUARD 和基于预训练多模态大模型的 QUART 模型，实现感知、导航、全身操作等多任务统一控制。 领域现状：四足机器人学习通常将视觉感知（QUAR…
tags:
  - "ECCV 2024"
  - "机器人"
  - "quadruped robot"
  - "视觉语言"
  - "imitation learning"
  - "sim-to-real"
  - "multi-task"
---

# QUAR-VLA: Vision-Language-Action Model for Quadruped Robots

**会议**: ECCV 2024  
**arXiv**: [2312.14457](https://arxiv.org/abs/2312.14457)  
**代码**: 未公开  
**领域**: 多模态VLM  
**关键词**: quadruped robot, vision-language-action, imitation learning, sim-to-real, multi-task

## 一句话总结
首次提出四足机器人视觉-语言-动作（QUAR-VLA）范式，构建 259K episode 的多任务数据集 QUARD 和基于预训练多模态大模型的 QUART 模型，实现感知、导航、全身操作等多任务统一控制。

## 研究背景与动机
**领域现状**：四足机器人学习通常将视觉感知（QUAR-VA）和语言交互（QUAR-LA）分开处理。VA 方法利用第一/第三人称图像引导动作但只用粗粒度目标图像指令；LA 方法利用语言执行细粒度任务但缺乏视觉感知无法自主导航。

**现有痛点**：
   - VA 方法依赖单一粗粒度图像指令，难以处理组合任务（如"先...然后..."）
   - LA 方法缺少视觉模态，无法感知环境障碍物
   - 缺乏大规模四足机器人多任务数据集

**核心矛盾**：四足机器人需要同时理解视觉场景和语言指令来自主决策，但现有范式将两者割裂；同时四足的灵活步态和全身控制使得动作空间设计更复杂。

**切入角度**：定义 12 维高层级命令动作空间（速度+步态+姿态+终止信号），将连续动作离散化为 256 个 bin，利用预训练 VLM 直接输出动作 token。

**核心 idea**：将预训练多模态大模型微调为四足机器人的统一策略，输入图像+语言指令，输出离散化动作 token。

## 方法详解

### 整体框架
QUAR-VLA 系统由三部分组成：(1) QUARD 大规模多任务数据集（仿真 256K + 真实 3K episodes）；(2) QUART 模型基于预训练 8B VLM，接收第一人称图像和语言指令，输出 12 维离散化动作；(3) 低层级命令跟踪控制器将高层级命令转化为关节动作。

### 关键设计

#### 1. **动作空间设计 (Action Space Design)**
    - 功能：定义适合四足机器人的高层级命令空间，平衡灵活性和计算效率
    - 核心思路：
      - 12 维动作向量：$[v_x, v_y, \omega_z, \theta_1, \theta_2, \theta_3, f, h_z, \phi, s_y, h_z^f, t]$
      - 含义：$v_x, v_y$ 为 x/y 轴速度，$\omega_z$ 为 z 轴角速度，$\theta_{1,2,3}$ 为步态模式，$f$ 为频率，$h_z$ 为机器人高度，$\phi$ 为俯仰角，$s_y$ 为脚宽，$h_z^f$ 为抬脚高度，$t$ 为终止信号
      - 连续维度用 256 个均匀区间离散化
    - 设计动机：避免过于简单的 2D 导航速度输出（缺乏全身控制），也避免直接输出关节电机控制（频率要求太高）。高层级命令由预训练的低层级策略执行，解耦了决策和执行。

#### 2. **QUARD 数据集 (QUAdruped Robot Dataset)**
    - 功能：构建首个包含视觉、语言指令和机器人命令的大规模四足机器人数据集
    - 核心思路：
      - 7 种任务分 3 个难度等级：Easy（识别字母 10K）、Medium（导航到物体 sim 72K + real 3K）、Hard（穿隧道 48K / 避障导航 63K / 爬杆 1K / 卸载物体 52K）
      - 仿真数据在 Isaac Gym 中并行采集，使用 A*/D* 路径规划 + PD 控制器转化为目标速度
      - 真实数据在实验室用遥控操作 WR-2 四足机器人采集
      - 多样性设计：多种颜色（绿/红/蓝/黄）、室内外物体（书架/烤箱/垃圾桶等）、三速度等级、多种步态
    - 一致性约束：仿真与真实的起始/目标位置范围一致（目标 x∈[2.7,3.3]m, y∈[0.9,1.1]m）

#### 3. **QUART 模型 (QUAdruped Robotic Transformer)**
    - 功能：将预训练视觉-语言模型微调为四足机器人的统一控制策略
    - 核心思路：
      - 基于 8B 预训练 VLM（fuyu-8b），将整数 token（0-255）与动作 bin 关联，这是一种 symbol tuning
      - 输入：单张第一人称 RGB 图像 $s$ + 语言指令 $w$
      - Tokenizer 转换为 token 序列：$\tau(t|s,w)$
      - Decoder-only Transformer 输出离散动作 token：$p(a_d|t)$
      - 策略：$\text{QUART}(a_d|s,w) = p(a_d|t)\tau(t|s,w)$
      - Action Detokenize：$a_c = \text{Detokenize}(a_d)$ 将离散 token 转回连续动作值
    - 训练：标准 categorical cross-entropy + causal masking，lr=2e-5, batch=256, 100K steps
    - 推理速度：2Hz，满足高层级命令控制的频率要求
    - 设计动机：利用 VLM 预训练的视觉-语言对齐能力和世界知识，通过 symbol tuning 直接输出动作，不需要额外的 action head

### 损失函数 / 训练策略
- 损失：标准 categorical cross-entropy（行为克隆损失），对 12 维动作 token 做 next-token prediction
- Sim-to-Real 联合训练：大量仿真数据 + 少量真实数据混合训练，仿真数据提供多样性，真实数据保证适用性

## 实验关键数据

### 主实验：多任务成功率

| 方法 | Distinguish | Go to | Go avoid | Go through | Crawl | Unload |
|------|-----------|-------|---------|-----------|-------|--------|
| CLIP | 0.44 | 0.43 | 0.45 | 0.19 | 0 | 0 |
| R3M | 0.58 | 0 | 0 | 0 | 0 | 0 |
| VC-1 | 0.46 | 0.43 | 0.45 | 0.31 | 0 | 0 |
| **QUART** | **0.66** | **0.60** | **0.53** | **0.41** | **0.32** | **0.12** |

QUART 在所有任务上均超越 baseline，尤其在高难度任务（Crawl/Unload）上是唯一能成功的方法。

### 泛化能力

| 方法 | Unseen Object | Unseen Verbal |
|------|-------------|---------------|
| CLIP | 0.11 | 0.14 |
| R3M | 0 | 0 |
| VC-1 | 0.29 | 0.19 |
| **QUART** | **0.35** | **0.33** |

### Sim-to-Real 扩展实验

| 仿真数据 : 真实数据 | 成功率 |
|-------------------|--------|
| 0K : 3K | 3/20 |
| 25.6K : 3K | 7/20 |
| **256K : 3K** | **13/20** |

### 关键发现
- **R3M 缺乏语言对齐**：虽然在简单感知任务（辨别字母）上有一定能力，但缺乏语言语义对齐导致其他任务全部失败。
- **VLM baseline 的局限**：CLIP 和 VC-1 能完成基础导航，但涉及复杂机械运动（爬杆、卸载）时完全失败，说明 VLM 能理解世界抽象概念但无法直接转化为物理任务执行。
- **QUART 的优势来源**：Decoder-only VLA 架构允许隐式学习不同动作维度间的依赖关系，而单层 MLP policy head 做不到。
- **未见指令的泛化**：QUART 借助大模型的语言能力，能理解训练集中未出现的语义变体（如"navigate to target" vs "go to object"），甚至能理解组合指令（"first...then..."）和空间关系指令。
- **仿真数据有效扩展**：仿真数据从 0K 增加到 256K 使真实场景成功率从 15% 提升到 65%。

## 亮点与洞察
- **QUAR-VLA 范式定义清晰**：明确了视觉-动作（VA）、语言-动作（LA）和视觉-语言-动作（VLA）三种范式的区别和各自局限，VLA 的提出有实际意义且逻辑自洽。
- **动作空间设计的工程洞察**：12 维高层级命令既包含速度控制又包含步态/姿态参数，2Hz 推理频率配合低层级控制器，是一个在灵活性和可行性之间优雅的工程折中。类比自动驾驶中规划层和控制层的解耦。
- **仿真规模扩展曲线**：清晰展示了仿真数据量与真实部署成功率的正相关关系，为 sim-to-real 提供了实证参考。

## 局限与展望
- 仿真环境视觉保真度不足，且仅在平坦地形实验，未考虑复杂地形
- 真实数据仅 3K episodes 且在实验室环境采集，Sim2Real gap 仍然显著
- Unload 任务成功率仅 12%，高难度全身操作任务仍然很困难
- 推理速度 2Hz 对于一些需要快速反应的场景可能不够
- 数据集中的语言模板较单调（预定义格式），缺乏自然语言的多样性

## 相关工作与启发
- **vs RT-2**：RT-2 将 VLM 用于机械臂操作，QUART 将类似思路迁移到四足机器人，但四足的运动学更复杂，需要更丰富的动作空间
- **vs Tang et al. (QUAR-LA)**：首个四足语言控制工作，但缺乏视觉感知导致机器人无法自主导航，QUAR-VLA 融合了视觉和语言
- **vs VC-1/R3M**：这些视觉表示模型在简单感知任务上有效，但缺乏端到端动作生成能力，QUART 的 VLA 架构显著更强

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次定义四足VLA范式 + 首个大规模数据集
- 实验充分度: ⭐⭐⭐⭐ 多任务/泛化/sim2real全面评估，但缺少与更多VLA方法的对比
- 写作质量: ⭐⭐⭐ 结构清晰但部分表述重复
- 价值: ⭐⭐⭐⭐ 为四足机器人智能化开辟了VLA方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Bridging Embodiment Gaps: Deploying Vision-Language-Action Models on Soft Robots](../../NeurIPS2025/robotics/bridging_embodiment_gaps_deploying_vision-language-action_models_on_soft_robots.md)
- [\[CVPR 2026\] ACoT-VLA: Action Chain-of-Thought for Vision-Language-Action Models](../../CVPR2026/robotics/acot-vla_action_chain-of-thought_for_vision-language-action_models.md)
- [\[ICML 2026\] From Abstraction to Instantiation: Learning Behavioral Representation for Vision-Language-Action Model](../../ICML2026/robotics/from_abstraction_to_instantiation_learning_behavioral_representation_for_vision-.md)
- [\[ICLR 2026\] AutoFly: Vision-Language-Action Model for UAV Autonomous Navigation in the Wild](../../ICLR2026/robotics/autofly_vision-language-action_model_for_uav_autonomous_navigation_in_the_wild.md)
- [\[ICML 2026\] Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model](../../ICML2026/robotics/dual-stream_diffusion_for_world-model_augmented_vision-language-action_model.md)

</div>

<!-- RELATED:END -->
