---
title: >-
  [论文解读] Hermes: A Unified Self-Driving World Model for Simultaneous 3D Scene Understanding and Generation
description: >-
  [ICCV 2025][自动驾驶][驾驶世界模型] 提出 Hermes，第一个统一 3D 场景理解（VQA/描述）和未来场景生成（点云预测）的驾驶世界模型，通过 BEV 表征和 world queries 将 LLM 的世界知识注入未来场景生成，3s 点云生成误差降低 32.4%，场景理解 CIDEr 提升 8.0%。
tags:
  - "ICCV 2025"
  - "自动驾驶"
  - "驾驶世界模型"
  - "3D 场景理解"
  - "点云生成"
  - "BEV"
  - "大语言模型"
---

# Hermes: A Unified Self-Driving World Model for Simultaneous 3D Scene Understanding and Generation

**会议**: ICCV 2025  
**arXiv**: [2501.14729](https://arxiv.org/abs/2501.14729)  
**代码**: [https://github.com/LMD0311/HERMES](https://github.com/LMD0311/HERMES)  
**领域**: 自动驾驶 / 世界模型  
**关键词**: 驾驶世界模型, 3D 场景理解, 点云生成, BEV, 大语言模型

## 一句话总结

提出 Hermes，第一个统一 3D 场景理解（VQA/描述）和未来场景生成（点云预测）的驾驶世界模型，通过 BEV 表征和 world queries 将 LLM 的世界知识注入未来场景生成，3s 点云生成误差降低 32.4%，场景理解 CIDEr 提升 8.0%。

## 研究背景与动机

现有驾驶世界模型（DWM）分为两个阵营：
- **场景生成型**（如 ViDAR、OccWorld）：擅长预测未来场景演化（2D 视频/3D 点云/占用网格），但无法理解和描述当前场景
- **场景理解型**（如 OmniDrive、DriveGPT4）：利用 VLM 进行场景描述和 VQA，但缺乏未来预测能力

两种能力对自动驾驶决策都至关重要，但此前没有框架能同时实现。统一两个任务面临两大挑战：

**多视角的大空间性**：自动驾驶需处理 6 个环视相机图像，直接转为 token 会超出 LLM 长度限制,且无法捕获视角间交互

**理解与生成的融合**：简单共享特征但分别建模两个任务，无法利用理解和生成之间的潜在互动

## 方法详解

### 整体框架

Hermes 流程：多视角图像 $I_t$ → BEV Tokenizer → 扁平化 BEV $\mathcal{F}_t$ → LLM（理解 + world queries）→ 编码 BEV $\mathcal{B}_t$ + 增强后的 world queries → Current-to-Future Link → 未来 BEV → 共享 Render → 点云序列

### 关键设计

1. **BEV-based World Tokenizer**：将 6 视角图像通过 CLIP 图像编码器和 BEVFormer v2 转为 BEV 特征 $\mathcal{F}^{bev}_t \in \mathbb{R}^{w \times h \times c}$（$w=h=200, c=256$），再经 4× 下采样压缩为 $50 \times 50$ token 序列送入 LLM。BEV 的两大优势：(1) 将多视角压缩到统一潜在空间，解决 token 长度问题；(2) 保持几何空间关系，捕获多视角物体间的交互。

2. **World Queries 机制**：初始化 $\Delta t$ 组 world queries（每组 $n=4$ 个），通过 BEV 特征的 max pooling 获得初始值，叠加 ego-motion 嵌入和帧位置嵌入后拼接到 LLM 输入序列末尾。LLM 的因果注意力机制使 world queries 能访问理解过程中产生的世界知识（场景描述、VQA 推理），从而将语义/知识信息传递到生成任务。经 LLM 处理后的 world queries 通过 Current-to-Future Link（3 层交叉注意力块）与编码 BEV $\mathcal{B}_t$ 交互，生成未来 BEV 特征。

3. **BEV-to-Point Render**：将 BEV 特征上采样并重塑为 3D 体素表征 $\mathcal{F}^{vol}_t \in \mathbb{R}^{w \times h \times z \times c'}$（$z=32$），基于隐式 SDF 场进行可微体渲染：对每条 LiDAR 射线采样点，通过三线性插值提取局部特征，用浅层 MLP 预测 SDF 值，最终加权积分得到渲染深度。当前帧点云预测作为辅助任务正则化 BEV 表征。

### 损失函数 / 训练策略

- **场景理解损失**：Next Token Prediction (NTP)，$\mathcal{L}_N = -\sum_i \log P(\mathcal{T}_i | \mathcal{F}_t, \mathcal{T}_{1:i-1}; \Theta)$
- **点云生成损失**：L1 深度监督，$\mathcal{L}_D = \sum_{i=0}^{\Delta t} \lambda_i \frac{1}{N_i} \sum_k |d(\mathbf{r}_k) - \tilde{d}(\mathbf{r}_k)|$
- 帧权重 $\lambda_i = 1 + 0.5i$（远帧权重更大）
- 总损失 $\mathcal{L} = \mathcal{L}_N + 10 \mathcal{L}_D$
- 三阶段训练：(1) Tokenizer + Render 预训练 (2) BEV-Text 对齐+LoRA 微调 (3) 理解与生成统一训练
- LLM 基于 InternVL2-2B（1.8B 参数），BEV backbone 用 OpenCLIP ConNeXt-L

## 实验关键数据

### 主实验

| 方法 | 类型 | 生成 0s↓ | 1s↓ | 2s↓ | 3s↓ | METEOR↑ | ROUGE↑ | CIDEr↑ |
|------|------|---------|-----|-----|-----|---------|--------|--------|
| 4D-Occ | 仅生成 | - | 1.13 | 1.53 | 2.11 | - | - | - |
| ViDAR | 仅生成 | - | 1.12 | 1.38 | 1.73 | - | - | - |
| GPT-4o | 仅理解 | - | - | - | - | - | 0.223 | 0.244 |
| OmniDrive | 仅理解 | - | - | - | - | 0.380 | 0.326 | 0.686 |
| Separated | 统一 | 0.60 | 0.84 | 1.08 | 1.37 | 0.384 | 0.327 | 0.745 |
| **Hermes** | **统一** | **0.59** | **0.78** | **0.95** | **1.17** | **0.384** | **0.327** | **0.741** |

Hermes 在 3s 点云生成上 Chamfer Distance 为 1.17，比 ViDAR 降低 32.4%（1.73→1.17）。场景理解 CIDEr 达 0.741，比 OmniDrive 提升 8.0%。值得注意的是 ViDAR 使用 3s 历史帧，Hermes 仅用当前帧。

### 消融实验

| 设置 | 生成 3s↓ | METEOR↑ | ROUGE↑ | CIDEr↑ |
|------|---------|---------|--------|--------|
| 仅理解 | - | 0.379 | 0.323 | 0.728 |
| 仅生成 | 1.687 | - | - | - |
| 分离统一 | 1.875 | 0.377 | 0.321 | 0.722 |
| 统一（Hermes） | 1.718 | 0.377 | 0.321 | 0.720 |
| 无 world queries | ~1.30 (est.) | - | ~0.745 | - |
| 有 world queries | ~1.17 | - | ~0.741 | - |
| BEV 25×25 | 1.698 | 0.367 | 0.311 | 0.671 |
| BEV 50×50 | 1.718 | 0.377 | 0.321 | 0.720 |
| LLM 0.8B | 1.809 | 0.372 | 0.318 | 0.703 |
| LLM 1.8B | 1.718 | 0.377 | 0.321 | 0.720 |
| LLM 3.8B | 1.701 | 0.381 | 0.325 | 0.730 |

### 关键发现

1. **统一优于分离**：Hermes 在生成上显著优于分离统一方案（3s: 1.718 vs 1.875），跨任务知识迁移有效
2. **World queries 至关重要**：引入 world queries 使 3s 生成 Chamfer Distance 降低约 10%
3. **Max pooling 最优**：world queries 初始化中 max pooling 优于 attention pooling 和 avg pooling
4. **BEV 尺寸影响显著**：50×50 比 25×25 在 CIDEr 上提升 7.3%，点云 0s 生成提升 10%
5. **LLM 缩放有效**：从 0.8B 到 3.8B，理解和生成性能持续提升

## 亮点与洞察

- 第一个同时完成 3D 场景理解和生成的统一驾驶世界模型，开创了新的研究方向
- World queries 设计巧妙：利用 LLM 的因果注意力实现从理解到生成的知识传递，无需额外模块
- BEV 表征作为桥梁解决了多视角 token 爆炸和几何保持两个问题
- 当前帧点云预测作为辅助任务免费正则化 BEV 编码，不增加推理开销

## 局限与展望

- 未探索感知任务（检测、分割等）在统一框架中的集成
- 未支持未来图像生成（仅支持点云），融合图像生成是重要扩展方向
- 仅使用 1.8B LLM，scaling law 显示更大模型仍有提升空间
- 仅在 nuScenes 数据集上验证，未测试其他大规模驾驶数据集
- World queries 数量增多时性能反而下降（n=8/16 不如 n=4），优化难度问题待解决

## 相关工作与启发

- **ViDAR**：自监督从图像预测未来点云，本文生成部分的主要对标
- **OmniDrive**：基于 Q-Former 的驾驶场景 VQA，本文理解部分的主要对标
- **InternVL2**：提供 LLM 骨干，其因果注意力是 world queries 发挥作用的关键
- **BEVFormer v2**：提供 BEV 表征，是统一多视角信息的基础

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次统一驾驶场景理解与3D生成，world queries 设计新颖
- **实验充分度**: ⭐⭐⭐⭐ 消融全面（world queries/BEV大小/LLM规模/任务交互），但仅限 nuScenes
- **写作质量**: ⭐⭐⭐⭐ 动机清晰，图示直观，方法阐述通顺
- **价值**: ⭐⭐⭐⭐⭐ 开创统一驾驶世界模型范式，对后续研究有重要指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] GaussianDWM: 3D Gaussian Driving World Model for Unified Scene Understanding and Multi-Modal Generation](../../CVPR2026/autonomous_driving/gaussiandwm_3d_gaussian_driving_world_model_for_unified_scene_understanding_and_.md)
- [\[ICCV 2025\] Controllable 3D Outdoor Scene Generation via Scene Graphs](controllable_3d_outdoor_scene_generation_via_scene_graphs.md)
- [\[ICCV 2025\] AdaDrive: Self-Adaptive Slow-Fast System for Language-Grounded Autonomous Driving](adadrive_self-adaptive_slow-fast_system_for_language-grounded_autonomous_driving.md)
- [\[ICCV 2025\] EmbodiedOcc: Embodied 3D Occupancy Prediction for Vision-based Online Scene Understanding](embodiedocc_embodied_3d_occupancy_prediction_for_vision-based_online_scene_under.md)
- [\[ICCV 2025\] Epona: Autoregressive Diffusion World Model for Autonomous Driving](epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)

</div>

<!-- RELATED:END -->
