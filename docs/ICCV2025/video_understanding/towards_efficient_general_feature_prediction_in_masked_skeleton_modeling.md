---
title: >-
  [论文解读] Towards Efficient General Feature Prediction in Masked Skeleton Modeling
description: >-
  [ICCV 2025][视频理解][掩码骨架建模] 提出 GFP（General Feature Prediction）框架，将掩码骨架建模的重建目标从低层关节坐标提升为多层次高层语义特征预测，配合轻量级目标生成网络和信息最大化约束，实现 6.2 倍训练加速的同时达到 SOTA 性能。 - 掩码骨架建模（masked ske…
tags:
  - "ICCV 2025"
  - "视频理解"
  - "掩码骨架建模"
  - "高层语义预测"
  - "目标生成网络"
  - "自监督学习"
  - "动作识别"
---

# Towards Efficient General Feature Prediction in Masked Skeleton Modeling

**会议**: ICCV 2025  
**arXiv**: [2509.03609](https://arxiv.org/abs/2509.03609)  
**代码**: 无  
**领域**: 骨架动作识别 / 自监督学习  
**关键词**: 掩码骨架建模, 高层语义预测, 目标生成网络, 自监督学习, 动作识别

## 一句话总结

提出 GFP（General Feature Prediction）框架，将掩码骨架建模的重建目标从低层关节坐标提升为多层次高层语义特征预测，配合轻量级目标生成网络和信息最大化约束，实现 6.2 倍训练加速的同时达到 SOTA 性能。

## 研究背景与动机

- 掩码骨架建模（masked skeleton modeling）沿用 MAE 范式，随机遮蔽关节后重建缺失坐标
- 现有方法存在两个关键问题：
  1. **解码器计算量大**：90% 遮蔽率 + Transformer 解码器导致解码序列极长（750个低层目标），训练缓慢
  2. **缺乏语义引导**：低层坐标重建缺乏高层时空语义监督，与下游任务存在语义鸿沟
- S-JEPA 虽然使用模型生成特征作为目标，但其 patch 级预测需要庞大的 EMA 编码器（28.32G FLOPs），收敛极慢（需1200 epoch）
- 核心思路：用更少的高层语义目标（251个）替代大量低层目标（750个），同时提升特征质量和训练效率

## 方法详解

### 整体框架

GFP 建立编码器-解码器架构与目标生成网络（TGN）之间的双向学习范式。编码器处理可见关节特征，解码器渐进预测多层次高层特征（短期运动模式→全局动作语义），TGN 通过一致性学习提供在线监督。方差-协方差正则化防止坍塌。

### 关键设计

1. **高层特征预测（High-Level Feature Prediction）**: 替代传统的逐 patch 坐标重建，设计层级化预测目标：

    - 将学习目标从 $\mathcal{L}_p = \frac{1}{N}\|f(E_N) - X_e\|^2_F$（低层）提升为 $\mathcal{L}_p = \frac{1}{M}\|f(E_N) - g(X)\|^2_F$（高层）
    - 解码器输入经时序平均池化（核大小 $t_1, t_2, ...$）渐进降低时序分辨率，通过级联 Transformer 解码器构建金字塔结构
    - 四级层次特征：$t_1=5$（5帧局部运动）、$t_2=10$（10帧中期动态）、$t_3=30$（30帧长周期）、全局语义
    - 最终目标数从 750 降至 **251**，解码器 FLOPs 从 17.70G 降至 **1.57G**

2. **目标生成网络（Target Generation Network, TGN）**: 轻量级多 MLP 结构，为各层次提供在线监督

    - 每个层次使用独立 MLP：局部语义提取器用 3 层 MLP（512 hidden），全局语义提取器用 3 层 MLP（2048 hidden）
    - TGN 输入为运动特征（帧间差分）$X_e = X_e[1:] - X_e[:-1]$，避免与编码器共享相同输入导致的偏向
    - 计算量仅 0.64G FLOPs（对比 S-JEPA 的 28.32G）
    - 双向学习：解码器预测 TGN 生成的目标，TGN 同步适应解码器的特征表示

3. **信息最大化约束（Information Maximization Constraint）**: 防止双向学习坍塌为平凡解

    - **方差正则化**：确保每个特征维度的 batch 方差超过阈值 $\gamma=1$
    $\mathcal{L}_{var} = \frac{1}{C_t}\sum_{i=1}^{C_t}\max(0, \gamma - \sqrt{\text{Var}(Z_{t_g}[:,i])})$
    - **协方差正则化**：驱动特征维度间的协方差趋近零，消除冗余
    $\mathcal{L}_{cov} = \frac{1}{C_t}\sum_{i \neq j}[\text{Cov}(Z_{t_g})]^2_{i,j}$
    - 灵感来自 VICReg 的信息最大化表示学习

### 损失函数 / 训练策略

- 总损失：$\mathcal{L}_{total} = \lambda \mathcal{L}_{pred} + \mathcal{L}_{reg}$
- 正则化损失：$\mathcal{L}_{reg} = \sum_{j \in \mathcal{J}}(\alpha \mathcal{L}_{cov}(Z_{t_j}) + \beta \mathcal{L}_{var}(Z_{t_j}))$
- 超参数：$\lambda=5, \alpha=5, \beta=1$
- 预训练 400 epochs，AdamW（$\beta_1=0.9, \beta_2=0.95$, weight decay 0.05）
- 学习率：20 warmup epochs 后余弦退火（1e-3 → 5e-4）
- 编码器 8 层，每层 8 头 256 维注意力 + 1024 FFN
- 运动感知遮蔽策略，segment length $l=4$

## 实验关键数据

### 主实验 (表格)

**NTU-60 骨架动作识别（MAE 方法对比，单卡 RTX 4090）：**

| 方法 | 目标类型 | Enc FLOPs | Dec FLOPs | TGN FLOPs | 训练时间 | 加速比 | x-sub | x-view |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SkeletonMAE | Joint | 1.97G | 17.70G | - | 20h27m | 1× | 74.8 | 77.7 |
| MAMP | Motion | 1.97G | 17.70G | - | 20h27m | 1× | 84.9 | 89.1 |
| S-JEPA | Patch-level | 1.97G | 17.70G | 28.32G | 90h57m | 0.2× | 85.3 | 89.8 |
| **GFP** | **Hierarchical** | **1.97G** | **1.57G** | **0.64G** | **3h14m** | **6.2×** | **85.9** | **92.0** |

**NTU-120 + PKU-MMD II：**

| 方法 | NTU-120 x-sub | NTU-120 x-setup | PKU-II x-sub |
|:---:|:---:|:---:|:---:|
| MAMP | 78.6 | 79.1 | 53.8 |
| S-JEPA | 79.6 | 79.9 | 53.5 |
| **GFP** | **79.1** | **80.3** | **56.2** |

### 消融实验 (表格)

**TGN 输入消融（NTU-60）：**

| TGN 输入 | x-sub | x-view |
|:---:|:---:|:---:|
| 关节坐标 | 85.0 | 90.9 |
| 遮蔽关节 | 84.2 | 90.3 |
| **运动特征（帧差分）** | **85.9** | **92.0** |

**半监督学习（NTU-60，1%/10% 标签）：**

| 方法 | x-sub 1% | x-sub 10% | x-view 1% | x-view 10% |
|:---:|:---:|:---:|:---:|:---:|
| HaLP | 46.6 | 72.6 | 48.7 | 77.1 |
| USDRL | 57.3 | 80.2 | 60.7 | 84.0 |
| MAMP | 66.0 | 88.0 | 68.7 | 91.5 |
| S-JEPA | 67.5 | 88.4 | 69.1 | 91.4 |
| **GFP** | **71.8** | **88.7** | **72.9** | **92.1** |

### 关键发现

- **训练效率提升显著**：GFP 仅需 3h14m 完成预训练，比 SkeletonMAE 快 **6.2 倍**，比 S-JEPA 快 **28 倍**
- 高层语义目标 > 低层重建目标：移除 TGN 后直接预测低层目标（如展平向量），性能显著下降
- **层次化特征互补**：逐步移除全局和局部目标均导致性能退化（Figure 3）
- 运动特征（帧差分）作为 TGN 输入最优，遮蔽输入反而有害
- 动作检索任务提升尤为突出：x-view 上比 MAMP 高 17.1%（87.1 vs 70.0），证明高层语义目标对全局表示学习的价值
- TGN 架构鲁棒性强：2/3/4 层 MLP 变体性能差异微小

## 亮点与洞察

- 精准的问题诊断：指出低层重建的计算冗余+语义不足是当前瓶颈，解决方案直接有效
- 层级化目标设计（5帧→10帧→30帧→全局）巧妙平衡了局部运动细节和全局动作理解
- VICReg 信息最大化约束的引入优雅地解决了协同训练中的坍塌问题
- 轻量级 TGN（0.64G FLOPs，3层 MLP）的效率远超 S-JEPA 的 EMA 编码器（28.32G）

## 局限与展望

- 层级粒度（5/10/30帧）为手动设定，可探索自适应确定方式
- 仅在骨架数据上验证，未扩展到 RGB 视频的掩码建模
- 协方差正则化在大 batch size 下的计算开销需关注
- 缺乏与对比学习+掩码建模联合训练方法的对比

## 相关工作与启发

- 该工作证明了在掩码建模中"**目标质量比数量更重要**"的原则
- 从低层重建到高层语义预测的范式转换可推广到其他模态的掩码建模
- TGN 的轻量化在线目标生成方案可替代 EMA 更新的大编码器

## 评分

- **新颖性**: ⭐⭐⭐⭐ 高层语义预测替代低层重建的思路清晰有效，但核心组件（金字塔解码、VICReg）均借鉴已有工作
- **实验充分度**: ⭐⭐⭐⭐⭐ 三个数据集、识别+检索+半监督多任务、详尽消融（目标类型/TGN输入/架构/投影器层数）
- **写作质量**: ⭐⭐⭐⭐ 问题动机阐述充分，效率对比直观（表1包含FLOPs和训练时间）
- **价值**: ⭐⭐⭐⭐⭐ 6.2倍加速+SOTA性能的组合极具实用价值，为骨架自监督学习设立了新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] General Compression Framework for Efficient Transformer Object Tracking](general_compression_framework_for_efficient_transformer_object_tracking.md)
- [\[ECCV 2024\] Data Collection-Free Masked Video Modeling](../../ECCV2024/video_understanding/data_collection-free_masked_video_modeling.md)
- [\[CVPR 2026\] Self-Paced and Self-Corrective Masked Prediction for Movie Trailer Generation](../../CVPR2026/video_understanding/self-paced_and_self-corrective_masked_prediction_for_movie_trailer_generation.md)
- [\[CVPR 2025\] Bootstrap Your Own Views: Masked Ego-Exo Modeling for Fine-Grained View-Invariant Video Representations](../../CVPR2025/video_understanding/bootstrap_your_own_views_masked_ego-exo_modeling_for_fine-grained_view-invariant.md)
- [\[ICCV 2025\] BlinkTrack: Feature Tracking over 80 FPS via Events and Images](blinktrack_feature_tracking_over_80_fps_via_events_and_images.md)

</div>

<!-- RELATED:END -->
