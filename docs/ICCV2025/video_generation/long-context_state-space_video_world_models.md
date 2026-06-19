---
title: >-
  [论文解读] Long-Context State-Space Video World Models
description: >-
  [ICCV 2025][视频生成][世界模型] 本文提出将状态空间模型（SSM/Mamba）引入视频世界模型，通过 block-wise SSM 扫描方案在空间一致性和时序记忆之间权衡，配合局部帧注意力，实现了线性训练复杂度、常数推理开销下的长期空间记忆保持，在 Memory Maze 和 Minecraft 数据集上大幅超越有限上下文的 Transformer 基线。
tags:
  - "ICCV 2025"
  - "视频生成"
  - "世界模型"
  - "SSM"
  - "Mamba"
  - "长期记忆"
  - "视频扩散模型"
  - "自回归生成"
---

# Long-Context State-Space Video World Models

**会议**: ICCV 2025  
**arXiv**: [2505.20171](https://arxiv.org/abs/2505.20171)  
**代码**: 待确认  
**领域**: 视频世界模型 / 状态空间模型  
**关键词**: 世界模型, SSM, Mamba, 长期记忆, 视频扩散模型, 自回归生成

## 一句话总结
本文提出将状态空间模型（SSM/Mamba）引入视频世界模型，通过 block-wise SSM 扫描方案在空间一致性和时序记忆之间权衡，配合局部帧注意力，实现了线性训练复杂度、常数推理开销下的长期空间记忆保持，在 Memory Maze 和 Minecraft 数据集上大幅超越有限上下文的 Transformer 基线。

## 研究背景与动机

**领域现状**：视频扩散模型作为世界模型已展现出潜力，可以通过自回归帧预测 + 动作条件来交互式模拟环境。最新方法（OpenSora、CogVideoX、GameGen-X）使用 Transformer + 滑动窗口推理进行无限长度视频生成。

**现有痛点**：
   - **注意力机制的记忆限制**：现有视频世界模型的注意力窗口极为有限（通常仅几秒），玩家在游戏中简单地左右看一下，环境就可能完全改变
   - **计算代价与记忆的矛盾**：完整因果注意力训练复杂度随上下文长度二次增长，推理时每帧成本线性增长；滑动窗口推理虽然降低了复杂度，但彻底牺牲了长期记忆

**核心矛盾**：要实现持久一致的世界模拟，需要模型能"记住"之前见过的环境；但 Transformer 架构要么内存开销过大，要么因滑动窗口丢失长期信息。需要一种既高效又能保持长期记忆的架构。

**本文目标** 设计一种视频世界模型架构，在保持常数推理时间和内存的同时，具备长期空间记忆能力。

**切入角度**：SSM（Mamba）天然是因果序列模型，具有固定大小的隐状态，推理时无需 KV-cache 线性增长。其压缩式记忆虽然不如全注意力精确，但恰好满足世界模型"记住大局、局部精细"的需求。

**核心 idea**：Block-wise SSM 扫描方案 + 局部帧注意力混合架构，用 SSM 做长期时序记忆、用注意力做短期空间精细化，训练线性、推理常数、记忆持久。

## 方法详解

### 整体框架
模型基于 diffusion forcing 训练策略（每帧独立噪声水平），支持自回归推理。每个网络层由两部分组成：①Block-wise SSM 扫描 → ②帧局部注意力。输入为动作-帧序列对，输出为下一帧的去噪预测。推理时仅需维护固定长度的 KV-cache（$k$ 帧）+ SSM 隐状态，内存恒定。

### 关键设计

1. **Block-wise SSM Scan（分块 SSM 扫描）**

    - 功能：在 SSM 扫描中平衡空间一致性与时序记忆
    - 核心思路：将空间维度划分为 $(b_h, b_w)$ 大小的块，每个块独立进行时序 SSM 扫描。不同层使用不同的块大小——小块时序邻近 token 距离近（增强时序记忆），大块空间交互多（增强空间一致性）
    - 设计动机：标准的 spatial-major 扫描中时序相邻 token 之间隔着 $H \times W$ 个空间 token，导致 SSM 的有限隐状态无法有效保持时序信息。通过分块，相邻帧的对应位置 token 距离缩短为 $b_h \times b_w$。同时分块也增加了每层的 SSM 有效隐状态维度（每个块分配独立状态）

2. **Frame Local Attention（帧局部注意力）**

    - 功能：弥补 SSM 在精确局部信息检索上的不足
    - 核心思路：每个 SSM 层后接一个块因果注意力层，每个 token 只能看到当前帧和前 $k$ 帧的所有 token。注意力掩码 $M_{i,j} = 1$ iff $j \in [i-k, i]$（帧索引），推理时仅维护 $k$ 帧的 KV-cache
    - 设计动机：SSM 在 associative recall 任务上表现不佳（如精确回忆某个 token）。局部注意力提供帧内双向处理 + 短期跨帧精确对齐，与 SSM 的长期压缩记忆互补

3. **Long-Context Training（长上下文训练策略）**

    - 功能：鼓励模型学会利用远距离上下文帧
    - 核心思路：混合使用标准 diffusion forcing（所有帧独立加噪）和改进方案——保持随机长度的前缀帧完全干净（$t_i = 0$），仅对后续帧加噪且仅计算后续帧的损失
    - 设计动机：标准训练下，模型总是倾向于利用最近的帧（信息最丰富），不会主动去看远距离的上下文。通过给近帧加大噪声同时给远距离帧保持干净，迫使模型学会利用远距离的清晰信息

### 推理策略
自回归逐帧生成，每层仅需：① $k$ 帧的固定长度 KV-cache + ② 每个 block 的 SSM 隐状态。内存和每帧推理时间完全恒定，不随生成长度增长。

## 实验关键数据

### 主实验 - Memory Maze 空间检索任务（400 帧生成 | 400 帧上下文）

| 模型 | SSIM ↑ | LPIPS ↓ | PSNR ↑ |
|------|--------|---------|--------|
| Causal (192帧上下文) | 0.829 | 0.147 | 26.4 |
| Mamba2 | 0.747 | 0.313 | 20.4 |
| Mamba2 + Frame Local Attn | 0.735 | 0.336 | 19.3 |
| **Ours** | **0.898** | **0.069** | **30.8** |
| Causal (全上下文, 参考) | 0.914 | 0.057 | 32.6 |

### 消融实验 - Memory Maze 空间推理任务（200 帧生成）

| 配置 | SSIM ↑ | LPIPS ↓ | PSNR ↑ |
|------|--------|---------|--------|
| w/o block-wise scan | 0.845 | 0.113 | 27.5 |
| w/ block size 1 | 0.766 | 0.198 | 23.1 |
| w/o 长上下文训练 | 0.809 | 0.143 | 25.3 |
| **Full model** | **0.855** | **0.099** | **28.2** |

### Minecraft 空间推理任务

| 模型 | SSIM ↑ | LPIPS ↓ | PSNR ↑ |
|------|--------|---------|--------|
| DFoT (SOTA) | 0.450 | 0.281 | 17.1 |
| Causal (25帧) | 0.417 | 0.350 | 15.8 |
| **Ours** | **0.454** | **0.259** | **17.8** |

### 关键发现
- **本文方法接近全上下文 Transformer 的记忆能力**（PSNR 30.8 vs 32.6），但训练复杂度从二次降为线性，推理复杂度从线性降为常数
- Mamba2 直接替换注意力效果很差（PSNR 仅 20.4），说明 naive 替换不可行，block-wise 方案至关重要
- Block size 太小（=1）虽然时序记忆好但空间一致性差，block size 太大时序记忆差——需要分层混合使用不同 block size
- 长上下文训练策略贡献显著（PSNR 从 25.3 提升到 28.2），证明标准 diffusion forcing 无法自动学会利用远距离上下文
- 随距离增大，因果 Transformer 的检索 PSNR 急剧下降（超出训练长度后），本文方法保持稳定

## 亮点与洞察
- **第一个将 SSM 用于其天然优势场景（因果时序建模）的视频生成工作**。之前的视频 SSM 工作都用双向扫描替代注意力做非因果任务，没有发挥 SSM 的核心价值。本文从世界模型出发，SSM 的因果性、固定状态、线性复杂度全部得到利用
- **Block-wise scan 方案**是一种灵活的空间-时序权衡机制。小块强化时序、大块强化空间，分层混合实现两全。这个思路可以迁移到其他空间-时序联合建模任务
- **长上下文训练策略**简单但有效——通过给近帧加噪、让远帧保持干净来打破模型对近距离信息的依赖。这个 trick 可以广泛使用

## 局限与展望
- 尚未实现交互式帧率（实时推理），未来可通过时间步蒸馏加速
- 无法有效处理超出训练上下文长度的记忆，有待借鉴 Mamba 长度外推技术
- 实验限于低分辨率合成视频（Memory Maze、Minecraft），高分辨率真实视频有待验证
- SSM 的压缩记忆本质上会丢信息，在需要精确像素级回忆的场景可能不足
- 可考虑引入显式记忆模块（如 memory bank）与 SSM 配合

## 相关工作与启发
- **vs DFoT (Diffusion Forcing Transformer)**：双向 Transformer + diffusion forcing 的 SOTA，但受训练上下文长度限制（25帧），二次训练复杂度。本文在 Minecraft 上超越 DFoT
- **vs 因果 Transformer + 滑动窗口**：滑动窗口推理实现常数速度但彻底丢失长期记忆。本文保持常数推理+长期记忆
- **vs GameGen-X / Genie**：这些开放世界生成方法也用滑动窗口，明确承认缺乏长期一致性，属于本文要解决的核心问题
- **vs Mamba 视频生成（DiS、ZigMa）**：它们用双向 Mamba 替换注意力做非因果生成，未利用 SSM 的因果优势

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将 SSM 用于其天然优势场景的视频世界模型，block-wise scan 设计巧妙
- 实验充分度: ⭐⭐⭐⭐ Memory Maze + Minecraft 双数据集，完善的消融和效率分析
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，动机充分，Tab.1 的复杂度对比一目了然
- 价值: ⭐⭐⭐⭐⭐ 解决了视频世界模型中长期记忆这一关键瓶颈，为可交互式持久世界模拟奠定架构基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Long Context Tuning for Video Generation](long_context_tuning_for_video_generation.md)
- [\[CVPR 2025\] Out of Sight, Out of Mind? Evaluating State Evolution in Video World Models](../../CVPR2025/video_generation/out_of_sight_out_of_mind_evaluating_state_evolution_in_video_world_models.md)
- [\[ICCV 2025\] Disentangled World Models: Learning to Transfer Semantic Knowledge from Distracting Videos for Reinforcement Learning](disentangled_world_models_learning_to_transfer_semantic_knowledge_from_distracti.md)
- [\[CVPR 2025\] Navigation World Models](../../CVPR2025/video_generation/navigation_world_models.md)
- [\[CVPR 2025\] World2Act: Latent Action Post-Training via Skill-Compositional World Models](../../CVPR2025/video_generation/world2act_latent_action_post-training_via_skill-compositional_world_models.md)

</div>

<!-- RELATED:END -->
