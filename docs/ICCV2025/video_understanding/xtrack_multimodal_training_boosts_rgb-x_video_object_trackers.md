---
title: >-
  [论文解读] XTrack: Multimodal Training Boosts RGB-X Video Object Trackers
description: >-
  [ICCV 2025][视频理解][多模态跟踪] 提出 XTrack，通过 Mixture of Modal Experts (MeME) 框架和软路由分类器，实现 RGB-D/T/E 跨模态知识共享，使推理时仅用单模态即可受益于多模态训练知识，平均精度提升 3%。 多模态感知（深度/热红外/事件相机）可弥补 RGB 跟踪在…
tags:
  - "ICCV 2025"
  - "视频理解"
  - "多模态跟踪"
  - "混合专家"
  - "跨模态知识迁移"
  - "视频目标跟踪"
  - "RGB-X"
---

# XTrack: Multimodal Training Boosts RGB-X Video Object Trackers

**会议**: ICCV 2025  
**arXiv**: [2405.17773](https://arxiv.org/abs/2405.17773)  
**代码**: [公开](https://arxiv.org/abs/2405.17773)  
**领域**: 视频理解  
**关键词**: 多模态跟踪, 混合专家, 跨模态知识迁移, 视频目标跟踪, RGB-X

## 一句话总结

提出 XTrack，通过 Mixture of Modal Experts (MeME) 框架和软路由分类器，实现 RGB-D/T/E 跨模态知识共享，使推理时仅用单模态即可受益于多模态训练知识，平均精度提升 3%。

## 研究背景与动机

多模态感知（深度/热红外/事件相机）可弥补 RGB 跟踪在极端场景下的不足，但面临关键限制：

**数据稀缺**：不存在同时包含所有模态的综合数据集，通常只有成对的 RGB-X 数据

**刚性分支设计**：现有统一模型（如 ViPT、UnTrack）在推理时根据输入模态激活预定的分支，模态间无交互

**跨模态知识浪费**：严格的模态隔离阻止了跨模态知识迁移——例如 RGB-Depth 序列中的快速运动低光场景，模型应该学到超越单模态边界的知识

核心洞察：**不同模态中的相似样本有更多可共享的知识**。当一个"弱"分类器无法准确区分样本来自哪个模态时，说明该样本处于跨模态知识共享的最佳位置（域差距最小）。这种"混淆"不是缺点，而是知识迁移的信号。

## 方法详解

### 整体框架

XTrack 在冻结的 RGB 基础跟踪器（OSTrack/SeqTrack）上，在每个注意力块和 FFN 之后插入 MeME 模块。MeME 双向处理 RGB 和 X 模态 token，增强特征建模：
$$T_{rgb}^{attn} = T_{rgb}^l + Attn(T_{rgb}^l) + MeME(T_{rgb}^l, T_x^l)$$
$$T_{rgb}^{l+1} = T_{rgb}^{attn} + FFN(T_{rgb}^{attn}) + MeME(T_{rgb}^{attn}, T_x^{attn})$$

### 关键设计

1. **软路由器与分类损失**：路由函数 $y = \sum_{i \in top\_k} p_i(T_x) \epsilon_i(T_x)$ 不仅有传统的专家负载均衡损失 $\mathcal{L}_{balance} = \mathcal{L}_{Imp} + \mathcal{L}_{Load}$，还引入模态分类损失 $\mathcal{L}_{cls}$。分类损失使每个专家保持一定的模态专业性，同时软（非刚性）分类允许跨模态样本访问其他模态的专家——分类概率约 80% 时达到最佳平衡。

2. **模态特定专家 + 共享专家**：

    - **模态特定专家**：每个模态分配 $k$ 个专家（实验表明 $k=2$ 最优），在低维空间 $k \ll c$ 中进行特征分解和重构
    - **边缘门控共享专家**：共享专家引入 Laplacian 滤波器初始化的 EdgeMix 门控模块，将高频边缘信息作为跨模态的天然共性先验：$Out = (\sigma(EdgeMix(XW_1)) \cdot XW_2)W_3 + m_{s_k}$

3. **模态提示（Modal Prompting）**：将 MeME 输出的低维模态矩阵 $M_k$ 作为门控信号调制 RGB token：$Out = ((X_i W_5 \cdot \sigma(X_m W_6))W_7 + I_k)W_8$，使 RGB 特征变得模态感知。

### 损失函数 / 训练策略

- 跟踪损失：继承 RGB 基础跟踪器的 IoU + L1 损失
- MoE 损失：$\mathcal{L}_{moe} = \mathcal{L}_{cls} + \lambda \cdot \mathcal{L}_{balance}$
- 仅训练 MeME 参数，冻结基础跟踪器
- 训练设置：batch 32，学习率 4e-4，90 epochs，78 epoch 后降 10 倍
- 训练数据：DepthTrack（RGB-D）+ LasHeR（RGB-T）+ VisEvent（RGB-E），每次仅一对 RGB-X 可用

## 实验关键数据

### 主实验

RGB-Depth 跟踪：

| 方法 | DepthTrack F-score | VOT-RGBD22 EAO | VOT-RGBD22 Acc. | VOT-RGBD22 Rob. |
|------|--------------------|-----------------|-----------------|-----------------|
| ViPT | 59.4 | 72.1 | 81.5 | 87.1 |
| UnTrack | 61.0 | 72.1 | 82.0 | 86.9 |
| SDSTrack | 61.4 | 72.8 | 81.2 | 88.3 |
| **XTrack-B** | **61.5** | **74.0** | **82.1** | **88.8** |
| **XTrack-L** | **64.8** | **74.0** | **82.8** | **88.9** |

RGB-Thermal 跟踪（LasHeR/RGBT234）：

| 方法 | LasHeR Pr | LasHeR Sr | RGBT234 MPR | RGBT234 MSR |
|------|----------|----------|-------------|-------------|
| ViPT | 65.1 | 52.5 | 83.5 | 61.7 |
| SDSTrack | 66.5 | 53.1 | 84.8 | 62.5 |
| OneTracker | 67.2 | 53.8 | 85.7 | 64.2 |
| **XTrack-B** | **69.1** | **55.7** | **87.4** | **64.9** |
| **XTrack-L** | **73.1** | **58.7** | **87.8** | **65.4** |

RGB-Event 跟踪（VisEvent）：

| 方法 | Pr | Sr |
|------|-----|-----|
| ViPT | 75.8 | 59.2 |
| OneTracker | 76.7 | 60.8 |
| SDSTrack | 76.7 | 59.7 |
| **XTrack-B** | **77.5** | **60.9** |
| **XTrack-L** | **80.5** | **63.3** |

### 消融实验

关键组件分析：

| 共享专家 | 模态专家 | DepthTrack F-score | LasHeR Pr | VisEvent Pr |
|---------|---------|-------------------|----------|-------------|
| ✓ | - | 59.1 | 67.8 | 76.5 |
| - | ✓ | 57.6 | 68.0 | 76.2 |
| **✓** | **✓** | **61.5** | **69.1** | **77.5** |

多模态联合训练收益：

| 训练模态 | VisEvent Pr | DepthTrack F-score | LasHeR Pr |
|---------|------------|-------------------|----------|
| 基线（无MeME） | 69.5 | 52.9 | 51.5 |
| 仅 Event | 76.3 | 45.5 | 58.1 |
| Event + Depth | 76.6 | 60.8 | 58.1 |
| **E + D + T** | **77.5** | **61.5** | **69.1** |

### 关键发现

- VOT-RGBD22 存在域差距，XTrack 反而超 SOTA 更大幅度，说明多模态训练增强了域泛化能力
- 仅用 Event 训练时，在 Thermal 上有不错的零样本泛化（58.1 Pr），因为事件相机和热相机都处理光照变化
- 软分类概率 80% 时效果最佳；刚性分离和随机分配都表现更差
- 每个模态 2 个专家最优；1 个表示受限，3 个产生内部冲突

## 亮点与洞察

- "混淆即机遇"的核心思想新颖：分类器的失败不是缺陷，而是跨模态知识共享的信号
- 首次系统性地实现 RGB-X 视频目标跟踪中的跨模态知识迁移
- EdgeMix 的 Laplacian 先验为共享专家提供了合理的归纳偏置
- 实验设计严谨：渐进式添加训练模态，清晰展示每个模态的贡献

## 局限与展望

- 训练仍依赖成对的 RGB-X 数据，无法利用无配对的单模态数据
- 低维投影虽降低计算量，但可能损失部分信息
- 未在更多传感器模态（如 LiDAR、SAR）上验证
- 推理时仍需选择模态专家，未实现真正的模态无关推理

## 相关工作与启发

- 软路由的 MoE 设计思路（分类损失 + 均衡损失的平衡）对其他多模态任务有通用价值
- "模态混淆"作为知识迁移信号的理论分析可推广到多模态融合的其他领域
- Laplacian 初始化的边缘共享先验是一个值得在其他视觉任务中探索的初始化策略

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ⭐⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 总体 | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] What You Have is What You Track: Adaptive and Robust Multimodal Tracking](what_you_have_is_what_you_track_adaptive_and_robust_multimodal_tracking.md)
- [\[ICCV 2025\] An Empirical Study of Autoregressive Pre-training from Videos](an_empirical_study_of_autoregressive_pre-training_from_videos.md)
- [\[CVPR 2026\] DarkAct: A RGB-Thermal Dataset and Fusion Framework for Multimodal Low-Light Action Recognition](../../CVPR2026/video_understanding/darkact_a_rgb-thermal_dataset_and_fusion_framework_for_multimodal_low-light_acti.md)
- [\[CVPR 2026\] M4-SAM: Multi-Modal Mixture-of-Experts with Memory-Augmented SAM for RGB-D Video Salient Object Detection](../../CVPR2026/video_understanding/m4-sam_multi-modal_mixture-of-experts_with_memory-augmented_sam_for_rgb-d_video_.md)
- [\[ICCV 2025\] MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation](memfof_high-resolution_training_for_memory-efficient_multi-frame_optical_flow_es.md)

</div>

<!-- RELATED:END -->
