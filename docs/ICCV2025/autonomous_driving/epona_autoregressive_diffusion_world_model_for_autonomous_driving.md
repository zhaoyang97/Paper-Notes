---
title: >-
  [论文解读] Epona: Autoregressive Diffusion World Model for Autonomous Driving
description: >-
  [ICCV 2025][自动驾驶][世界模型] 提出 Epona，一种自回归扩散世界模型，通过解耦时空建模和异步多模态生成，实现高分辨率长时程驾驶视频生成与实时轨迹规划的统一框架。 现有驾驶世界模型主要分两类：1) 扩散式：方法（如 Vista）通过联合分布建模固定长度视频帧，视觉质量好但无法灵活生成变长序列…
tags:
  - "ICCV 2025"
  - "自动驾驶"
  - "世界模型"
  - "扩散模型"
  - "trajectory planning"
  - "视频生成"
---

# Epona: Autoregressive Diffusion World Model for Autonomous Driving

**会议**: ICCV 2025  
**arXiv**: [2506.24113](https://arxiv.org/abs/2506.24113)  
**代码**: [https://github.com/Kevin-thu/Epona/](https://github.com/Kevin-thu/Epona/)  
**领域**: 自动驾驶  
**关键词**: 世界模型, autoregressive diffusion, trajectory planning, 视频生成, 自动驾驶

## 一句话总结

提出 Epona，一种自回归扩散世界模型，通过解耦时空建模和异步多模态生成，实现高分辨率长时程驾驶视频生成与实时轨迹规划的统一框架。

## 研究背景与动机

现有驾驶世界模型主要分两类：1) **扩散式**方法（如 Vista）通过联合分布建模固定长度视频帧，视觉质量好但无法灵活生成变长序列，也无法集成轨迹规划；2) **GPT 式**自回归方法（如 GAIA-1）通过 next-token prediction 支持变长生成，但量化和分词过程严重降低视觉质量和规划精度。两种范式的缺陷互补——扩散模型缺乏时间分解能力，自回归 Transformer 牺牲了连续视觉精度。因此需要一个统一框架来调和这两种方法的优势。

## 方法详解

### 整体框架

Epona 将世界建模重新定义为时间域上的逐步未来预测过程。给定历史驾驶观测和轨迹，模型同时预测：1) 未来轨迹规划的策略分布 π；2) 下一帧相机观测的条件分布 p。整体框架由三个核心组件构成：多模态时空 Transformer (MST)、轨迹规划 DiT (TrajDiT) 和下一帧预测 DiT (VisDiT)。模型总参数量 2.5B。

### 关键设计

1. **多模态时空 Transformer (MST, 1.3B 参数)**：编码历史上下文 {O_t, a_t} 为紧凑的潜在表示。采用交错的多模态空间注意力层和因果时间注意力层。先将 visual latent patches Z ∈ R^{B×T×L×C} 和动作序列 a ∈ R^{B×T×3} 投影到嵌入空间，拼接后通过因果时间层（causal mask）和多模态空间层交替处理。最终取最后一帧的嵌入 F ∈ R^{B×(L+3)×D} 作为紧凑历史表示。这种设计显著降低了全序列注意力的内存消耗，并自然支持变长历史上下文。

2. **轨迹规划扩散 Transformer (TrajDiT, 50M 参数)**：使用 Dual-Single-Stream 架构的微型扩散 Transformer 预测未来 3 秒轨迹。双流阶段中，历史潜在表示 F 和轨迹数据独立处理，仅通过注意力操作关联；单流阶段拼接后通过后续 Transformer 块融合信息。训练时对目标轨迹 ā ∈ R^{B×N×3} 加噪并用 Rectified Flow 损失优化：L_traj = E[||v_traj(ā_(t), t) - (ā - ε)||²]。

3. **下一帧预测扩散 Transformer (VisDiT, 1.2B 参数)**：架构类似 TrajDiT，额外增加了动作控制 a_{T→T+1} 的调制分支。同样使用 Flow Matching 目标：L_vis = E[||v_vis(Z_{T+1(t)}, t) - (Z_{T+1} - ε)||²]。推理时根据 F 和动作（来自 TrajDiT 预测或用户提供）对噪声去噪得到下一帧潜变量，再用 DCAE 解码器生成图像。

### 损失函数 / 训练策略

- **总损失**：L = L_traj + L_vis，端到端联合训练
- **Chain-of-Forward 训练策略**：为缓解自回归漂移问题（训练用 GT、推理用自身预测的域差距），每隔 10 步执行一次多步前向传播——利用模型预测的速度 v_Θ 在一步内估计去噪潜变量 x̂_(0) = x_(t) + t·v_Θ(x_(t), t)，然后用估计结果作为下一步的条件。每次执行 3 次前向传播，模拟推理噪声以增强鲁棒性
- **时间感知 DCAE 解码器**：在 DCAE 解码器前引入时空自注意力层增强帧间一致性，解决逐帧解码的闪烁问题，编码器冻结只微调解码器
- **训练设置**：48 张 A100 GPU 训练约两周，600K 迭代，batch size 96，AdamW 优化器，lr=1e-4，weight decay=5e-2。图像分辨率 512×1024

## 实验关键数据

### 主实验

| 方法 | FID ↓ | FVD ↓ | 最大时长/帧数 |
|------|-------|-------|-------------|
| DriveGAN | 73.4 | 502.3 | N/A |
| DriveDreamer | 52.6 | 452.0 | 4s / 48 |
| Drive-WM | 15.8 | 122.7 | 8s / 16 |
| Vista | 6.9 | 89.4 | 15s / 150 |
| DrivingWorld | 7.4 | 90.9 | 40s / 400 |
| **Epona** | **7.5** | **82.8** | **120s / 600** |

NAVSIM 规划性能：

| 方法 | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|------|------|-------|-------|---------|------|--------|
| UniAD | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| DRAMA | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 |
| **Epona** | **97.9** | **95.1** | **93.8** | **99.9** | **80.4** | **86.2** |

### 消融实验

| 设置 | NC ↑ | DAC ↑ | PDMS ↑ |
|------|------|-------|--------|
| w/o 联合训练（仅轨迹） | 94.5 | 89.7 | 78.1 |
| 完整 Epona | 97.9 | 95.1 | 86.2 |

Chain-of-Forward 训练效果：无该策略时视觉质量在 10-20 秒后快速退化；有该策略时可保持分钟级高质量生成。

时间感知 DCAE 解码器：

| 方法 | FVD10 ↓ | FVD25 ↓ | FVD40 ↓ |
|------|---------|---------|---------|
| w/o 时间模块 | 52.95 | 76.46 | 100.11 |
| 完整模型 | 50.77 | 61.46 | 74.88 |

### 关键发现

- 共享潜变量联合训练视频和轨迹显著提升规划性能（PDMS 从 78.1 → 86.2）
- Chain-of-Forward 策略在长时程生成中效果随序列增长愈发显著
- 条件帧从 2 帧增加到 10 帧，FVD40 从 103.70 降至 74.88
- 模型仅通过自监督未来预测就能隐式学习交通规则（如红灯停车）

## 亮点与洞察

- **范式创新**：首次将自回归和扩散模型在时空维度上解耦并统一，既保留扩散模型的视觉质量，又获得自回归模型的时间灵活性
- **实时规划**：通过模块化设计，仅用 MST + TrajDiT 即可实现 20Hz 实时轨迹规划
- **极长生成**：120 秒/600 帧的生成长度大幅超越同期方法（Vista 仅 15s）
- Chain-of-Forward 是一种通用的自回归漂移缓解策略，可推广到其他领域

## 局限与展望

- FID 指标（7.5）略高于 Vista（6.9），单帧质量仍有提升空间
- 仅使用前视单相机，未扩展到多视角全景生成
- 训练成本较高（48 张 A100 训练两周），部署门槛高
- 未评估极端天气和罕见场景下的鲁棒性

## 相关工作与启发

- 与 DrivingWorld 等 GPT 式方法相比，Epona 在连续空间而非离散 token 空间进行自回归生成，保留了视觉细节
- Diffusion Forcing 和 FIFO-Diffusion 也探索了自回归+扩散的结合，但 Epona 重新定义了架构为两阶段端到端框架
- 模块化设计（MST/TrajDiT/VisDiT 可独立使用）为灵活部署提供了可能

## 评分

- **新颖性**: ⭐⭐⭐⭐ 自回归扩散统一框架的思路新颖且实用
- **实验充分度**: ⭐⭐⭐⭐ 视频生成和轨迹规划两个维度均有充分评估和消融
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，公式推导完整，图表丰富
- **价值**: ⭐⭐⭐⭐ 对自动驾驶世界模型的发展有显著推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] LangTraj: Diffusion Model and Dataset for Language-Conditioned Trajectory Simulation](langtraj_diffusion_model_and_dataset_for_language-conditioned_trajectory_simulat.md)
- [\[CVPR 2025\] DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving](../../CVPR2025/autonomous_driving/diffusiondrive_truncated_diffusion_model_for_end-to-end_autonomous_driving.md)
- [\[ICCV 2025\] DriveX: Omni Scene Modeling for Learning Generalizable World Knowledge in Autonomous Driving](drivex_omni_scene_modeling_for_learning_generalizable_world_knowledge_in_autonom.md)
- [\[ICCV 2025\] World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model](world4drive_end-to-end_autonomous_driving_via_intention-aware_physical_latent_wo.md)
- [\[ICLR 2026\] Astra: General Interactive World Model with Autoregressive Denoising](../../ICLR2026/autonomous_driving/astra_general_interactive_world_model_with_autoregressive_denoising.md)

</div>

<!-- RELATED:END -->
