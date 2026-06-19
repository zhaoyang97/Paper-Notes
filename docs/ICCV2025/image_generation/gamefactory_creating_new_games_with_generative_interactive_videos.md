---
title: >-
  [论文解读] GameFactory: Creating New Games with Generative Interactive Videos
description: >-
  [ICCV 2025][图像生成][游戏生成] 提出 GameFactory，通过在预训练视频扩散模型上**解耦游戏风格与动作控制**的多阶段训练策略，实现了从小规模 Minecraft 数据学到的动作控制能力**泛化到开放域任意场景**的交互式游戏视频生成——这是首个提供完整技术论文且验证复杂动作空间（7键+鼠标）的场景泛化方法。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "游戏生成"
  - "视频扩散模型"
  - "动作控制"
  - "场景泛化"
  - "自回归视频生成"
  - "Minecraft"
  - "世界模型"
---

# GameFactory: Creating New Games with Generative Interactive Videos

**会议**: ICCV 2025  
**代码**: [https://yujiwen.github.io/gamefactory/](https://yujiwen.github.io/gamefactory/) (项目页面)  
**领域**: 图像生成  
**关键词**: 游戏生成, 视频扩散模型, 动作控制, 场景泛化, 自回归视频生成, Minecraft, 世界模型

## 一句话总结

提出 GameFactory，通过在预训练视频扩散模型上**解耦游戏风格与动作控制**的多阶段训练策略，实现了从小规模 Minecraft 数据学到的动作控制能力**泛化到开放域任意场景**的交互式游戏视频生成——这是首个提供完整技术论文且验证复杂动作空间（7键+鼠标）的场景泛化方法。

## 研究背景与动机

视频生成模型正在成为生成式游戏引擎的有力候选者，但当前研究有三个根本性局限：

**绑定特定游戏**：DIAMOND（Atari/CS:GO）、GameNGen（DOOM）、Oasis（Minecraft）都只能在训练过的固定游戏中生成内容

**缺乏场景泛化**：无法创造超越现有游戏的新内容——这限制了其作为"创造新游戏"的能力

**收集大规模动作标注数据不现实**：对开放域视频进行动作标注成本极高

**核心洞察**：互联网上有海量的开放域视频，预训练视频生成模型已经蕴含了丰富的场景生成先验。如果能把从小规模游戏数据中学到的动作控制能力**迁移**到任意场景，就能创造全新游戏。

**关键挑战**：直接在游戏数据上微调预训练模型会导致**域坍缩**——模型输出会继承 Minecraft 的像素风格，丧失开放域生成能力。风格和动作控制是**纠缠**的。

## 方法详解

### GF-Minecraft 数据集

针对训练数据的三个关键需求设计：

1. **无人类偏见的动作分布**：将键盘和鼠标输入分解为原子动作，确保均匀分布。对比 VPT 数据集：W 键占 50.11% vs 本文的 13.56%（均匀），S 键仅 0.32% vs 13.56%
2. **多样场景**：跨不同场景、天气、时间段捕获 70 小时游戏视频
3. **文本标注**：使用 MiniCPM 为分段视频生成文本描述

### 动作控制模块

将动作控制模块注入视频扩散 Transformer 的每个 block：

**鼠标控制（连续信号）**：使用拼接方式——将分组后的鼠标动作 $\mathbf{M}_{repeat} \in \mathbb{R}^{(n+1) \times l \times rwd_1}$ 与特征 $\mathbf{F}$ 沿通道维拼接，再通过 MLP + 时间自注意力

**键盘控制（离散信号）**：使用交叉注意力——学习键盘动作嵌入后，将分组动作嵌入 $\mathbf{K}_{group}$ 作为 key/value，特征 $\mathbf{F}$ 作为 query

**关键设计——滑动窗口分组**：由于时间压缩比 $r=4$，动作数量（$rn$）与特征数量（$n+1$）不匹配。引入窗口大小 $w$ 的滑动窗口分组来对齐，同时捕获动作的延迟效应（如跳跃命令影响多帧）

### 自回归长视频生成

核心思想来自 Diffusion Forcing：允许不同帧使用不同的噪声级别。

**训练**：$N+1$ 帧中随机选 $k+1$ 帧作为无噪声条件帧，仅对剩余 $N-k$ 帧计算噪声预测损失

**推理**：每次取最近 $k+1$ 帧作为条件生成 $N-k$ 新帧，重复以实现无限长视频

**训练损失**：
$$\mathcal{L}_{\mathbf{a}}(\phi) = \mathbb{E}[||\boldsymbol{\epsilon}_\phi(\mathbf{Z}_t, \mathbf{p}, \mathbf{A}, t) - \boldsymbol{\epsilon}||_2^2]$$

关键发现：仅在预测帧上计算损失（而非所有帧）性能更好（Flow: 85.45 vs 148.73）

### 风格-动作解耦训练（核心贡献）

**多阶段训练策略**：

- **Phase #0**：预训练视频生成模型（开放域数据）
- **Phase #1**：用 LoRA（rank=128，lr=1e-4）微调以适配 Minecraft 风格——LoRA 捕获游戏特定视觉风格
- **Phase #2**：冻结预训练参数和 LoRA，仅训练动作控制模块（lr=1e-5）——因 Phase #1 已通过 LoRA 处理了风格适配，此时的训练损失主要由动作控制主导
- **Phase #3**：推理时**移除 LoRA 权重**，仅保留动作控制模块→ 实现动作控制在开放域的泛化

**解耦原理**：Phase #1 让 LoRA 学游戏风格，Phase #2 让控制模块学动作控制。由于两者由不同参数承载，移除 LoRA 后动作控制能力得以保留而风格约束被去除。

## 实验关键数据

### 主实验：动作控制机制消融

| 键盘控制 | 鼠标控制 | Only-Key Cam↓ | Only-Key Flow↓ | Mouse-Small Cam↓ | Mouse-Large Flow↓ |
|---|---|---|---|---|---|
| Cross-Attn | Cross-Attn | 0.0527 | 8.67 | 0.0798 | 325.18 |
| Concat | Concat | 0.0853 | 22.37 | 0.0756 | 258.93 |
| **Cross-Attn** | **Concat** | **0.0439** | **7.79** | **0.0685** | **249.54** |

最优组合：**键盘用交叉注意力，鼠标用拼接**。离散信号适合基于相似度的交叉注意力；连续信号适合保留幅度信息的拼接。

### 场景泛化对比

| 策略 | 域 | Cam↓ | Flow↓ | Dom↑ | CLIP↑ | FID↓ | FVD↓ |
|---|---|---|---|---|---|---|---|
| Multi-Phase | In-domain | 0.0839 | 43.48 | — | — | — | — |
| **Multi-Phase** | **Open-domain** | **0.0997** | **54.13** | **0.7565** | **0.3181** | **121.18** | **1256.94** |
| One-Phase | Open-domain | 0.1134 | 76.02 | 0.7345 | 0.3111 | 167.79 | 1323.58 |

多阶段策略在所有指标上显著优于单阶段，特别是：
- 动作跟随能力（Flow: 54.13 vs 76.02）
- 域一致性（Dom: 0.7565 vs 0.7345）
- 生成质量（FID: 121.18 vs 167.79）

### 数据集对比（GF-Minecraft vs VPT）

| 数据集 | Cam↓ | Flow↓ | CLIP↑ | FID↓ | FVD↓ |
|---|---|---|---|---|---|
| VPT (人类偏见) | 0.1324 | 107.67 | 0.3174 | 156.69 | 1233.15 |
| **GF-Minecraft (无偏)** | **0.0839** | **43.48** | **0.3135** | **125.85** | **1047.59** |

无偏数据集在动作跟随方面有巨大优势（Flow: 43.48 vs 107.67），验证了去除人类行为偏见的重要性。

### 长视频生成消融

| 损失范围 | Cam↓ | Flow↓ | CLIP↑ | FID↓ | FVD↓ |
|---|---|---|---|---|---|
| 所有帧 | 0.1547 | 148.73 | 0.2965 | 176.07 | 1592.43 |
| **仅预测帧** | **0.0924** | **85.45** | **0.3190** | **136.95** | **1154.45** |

### 关键发现

1. 模型成功生成超过 **300 帧**的长视频，覆盖前进/后退/左/右移动、跳跃、加速/减速 + 鼠标移动等复杂动作空间
2. 从 Minecraft 学到的动作控制可以泛化到沙漠、森林、城市、室内等完全不同的场景
3. 单阶段训练会导致"风格泄漏"——生成的开放域视频带有明显的 Minecraft 视觉伪影

## 亮点与洞察

1. **风格-动作解耦的巧妙设计**：用 LoRA 捕获风格、用独立模块捕获动作、推理时移除 LoRA——思路简单但极其有效
2. **无偏数据集的重要性**：VPT 数据中 W 键占 50%、S 键仅 0.32% 的极端偏斜导致模型无法执行"后退"动作——数据偏见直接传递为能力缺陷
3. **从"模拟已有游戏"到"创造新游戏"的范式跳跃**：这是第一个严肃尝试将游戏动作控制从特定域泛化到开放域的工作
4. **自回归设计的工程价值**：每步生成多帧（而非逐帧），大幅减少长视频生成时间

## 局限性

1. **基础模型不公开**：使用的 11B 内部 text-to-video 模型无法复现
2. **分辨率有限**：360×640，距离游戏画质标准仍有差距
3. **缺乏统一基准对比**：由于各方法使用不同游戏源、分辨率和控制粒度，无法进行公平定量对比
4. **未评估物理真实性**：生成的视频是否遵循合理物理规律（如碰撞、重力）未验证
5. **延迟/帧率未报告**：实时游戏需要 30+ FPS，但推理效率未讨论
6. **复杂交互缺失**：不支持物体操作、库存管理等游戏核心功能

## 相关工作与启发

- **Genie 2**：通过大规模动作标注数据实现控制泛化，走的是数据规模路线；GameFactory 走预训练先验 + 小数据迁移路线，两者互补
- **Diffusion Forcing**：GameFactory 的自回归生成借鉴了此工作的不同帧不同噪声级别思想
- **LoRA 的新用途**：通常 LoRA 用于适配，这里 LoRA 用于"隔离"——让其吸收特定风格后推理时移除，思路上可推广到其他域迁移场景
- **对未来游戏产业的影响**：如果场景泛化和物理真实性进一步提升，生成式游戏引擎可能成为全新的游戏形态——玩家可以即时创造和探索任何想象中的游戏世界

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性**: ⭐⭐⭐⭐⭐ — 风格-动作解耦的多阶段训练策略是核心亮点，场景泛化方向具有前瞻性
- **实验完整性**: ⭐⭐⭐ — 消融充分但缺乏与同类方法的定量对比，基础模型不公开
- **实用性**: ⭐⭐⭐ — 概念验证阶段，距实际游戏应用仍有差距
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，方法描述详细

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Market Games for Generative Models: Equilibria, Welfare, and Strategic Entry](../../ICLR2026/image_generation/market_games_for_generative_models_equilibria_welfare_and_strategic_entry.md)
- [\[ICCV 2025\] SDMatte: Grafting Diffusion Models for Interactive Matting](sdmatte_grafting_diffusion_models_for_interactive_matting.md)
- [\[ICCV 2025\] StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation](streamdiffusion_a_pipeline-level_solution_for_real-time_interactive_generation.md)
- [\[ICCV 2025\] EC-Flow: Enabling Versatile Robotic Manipulation from Action-Unlabeled Videos via Equivariant Flow Matching](ec-flow_enabling_versatile_robotic_manipulation_from_action-unlabeled_videos_via.md)
- [\[ICCV 2025\] MotionDiff: Training-Free Zero-Shot Interactive Motion Editing via Flow-Assisted Multi-View Diffusion](motiondiff_training-free_zero-shot_interactive_motion_editing_via_flow-assisted_.md)

</div>

<!-- RELATED:END -->
