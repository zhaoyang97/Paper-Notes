---
title: >-
  [论文解读] PressTrack-HMR: Pressure-Based Top-Down Multi-Person Global Human Mesh Recovery
description: >-
  [AAAI 2026][3D视觉][人体网格恢复] 提出 PressTrack-HMR，首个仅基于压力信号实现多人全局人体网格恢复的自上而下流水线，通过创新的 UoE 相似度度量实现压力足迹跟踪（93.6% MOTA），并构建了首个多人交互压力数据集 MIP。 多人全局人体网格恢复（HMR）对于理解人群动态和交互至关重要…
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "人体网格恢复"
  - "压力传感"
  - "多人跟踪"
  - "隐私保护"
  - "触觉交互"
---

# PressTrack-HMR: Pressure-Based Top-Down Multi-Person Global Human Mesh Recovery

**会议**: AAAI 2026  
**arXiv**: [2511.09147](https://arxiv.org/abs/2511.09147)  
**代码**: [github.com/Jiayue-Yuan/PressTrack-HMR](https://github.com/Jiayue-Yuan/PressTrack-HMR)  
**领域**: 3D视觉  
**关键词**: 人体网格恢复, 压力传感, 多人跟踪, 隐私保护, 触觉交互

## 一句话总结

提出 PressTrack-HMR，首个仅基于压力信号实现多人全局人体网格恢复的自上而下流水线，通过创新的 UoE 相似度度量实现压力足迹跟踪（93.6% MOTA），并构建了首个多人交互压力数据集 MIP。

## 研究背景与动机

多人全局人体网格恢复（HMR）对于理解人群动态和交互至关重要。然而，**现有基于RGB的 HMR 方法在实际场景中面临三大瓶颈**：

**遮挡问题**：拥挤环境中行人之间不可避免的相互遮挡限制了单视角信息的完整性，部署多相机成本高且复杂。

**光照问题**：光照不足会降低视觉数据质量和可用性。

**隐私问题**：日益增长的隐私保护需求使得基于相机的方案在家庭、医院等敏感环境中越来越不受欢迎。

作者观察到：**人与地面的触觉交互提供了丰富的压力信息**，这种信息天然避免上述问题。压力数据对遮挡和光照鲁棒，且天然具有隐私保护优势。

然而，将压力感知从单人扩展到多人场景面临**两个核心挑战**：

**帧内个体压力信号分离**：当多人同时在触觉垫上行走时，不同个体的压力信号交织混合（如Figure 1a），需要区分不同个体的信号。

**帧间个体压力信号关联**：需要在连续时间点识别属于同一个人的压力信号，以获取时序连续的单人数据用于更准确的 HMR。

此外，作者分析了压力足迹的独特动力学特性（区别于视觉跟踪目标）：
- **尺寸突变**：单脚/双脚着地交替导致检测框大小突然变化，降低 IoU 可靠性。
- **跳跃式不连续运动**：检测框的运动呈跳跃式，使卡尔曼滤波等平滑运动预测失效。

这些特性使得现有视觉跟踪方法（ByteTrack、BoT-SORT）无法直接应用。

## 方法详解

### 整体框架

PressTrack-HMR 采用**自上而下**的流水线，包含两个主要模块：
1. **PressTrack 模块**：基于跟踪-检测策略，从原始压力数据中提取每个个体的时序压力信号
2. **HMR 模块**：从时序单人压力图重建全局人体网格

### 关键设计

#### 1. **帧内足迹检测（Intra-frame Footprint Detection）**

**功能**：从原始压力数据中检测每个个体的足迹边界框。

微调预训练的 YOLOv11 模型用于压力足迹检测。标签自动生成流程（Figure 4）：
- 用 OpenCV 阈值法提取初始离散压力区域
- 从RGB数据的3D关节坐标提取足部关节点（左右脚趾和踝关节），投影到压力垫2D坐标系
- 将每个压力区域分配给最近足部关节点对应的个体：

$$\text{ID}(r_j) = \arg\min_{i \in \{1,\ldots,N\}} \min_{f \in F_i} \|c_j - f\|_2$$

- 合并同一ID的所有区域为最小外接矩形作为训练标签

**设计动机**：利用RGB数据的关节信息自动生成训练标签，避免手动标注的巨大代价。

#### 2. **帧间足迹关联（Inter-frame Footprint Association）—— UoE 度量**

**功能**：跨帧关联同一个人的压力足迹。这是本文最核心的技术创新。

提出 Union over Enclosure (UoE) 替代传统 IoU 作为相似度度量：

$$\text{UoE} = \frac{|A \cup B|}{|C|}$$

其中 $A$、$B$ 是当前帧和前一帧的检测框，$C$ 是包围 $A$ 和 $B$ 的最小外接矩形，$|\cdot|$ 表示面积。

**为什么不用 IoU？** 压力足迹存在尺寸突变（单脚→双脚切换时检测框大小剧变），导致 IoU 不可靠。UoE 基于一个关键观察：**不同个体的压力足迹不会同时重叠**，因此用封闭区域面积归一化比交集面积归一化更稳定。

**为什么不用运动预测？** 压力足迹的运动是跳跃式不连续的（脚步交替），卡尔曼滤波等平滑运动预测模型不适用。因此直接在相邻帧间计算 UoE 代价矩阵，用匈牙利算法求最优匹配。

轨迹管理策略：
- 低置信度未匹配检测丢弃，高置信度初始化为新轨迹（新人进入）
- 未匹配轨迹暂时标记为"丢失"并保留若干帧（处理短暂双脚跳跃等情况）
- 长期未匹配则认为已离开场景

#### 3. **人体网格恢复模块**

**功能**：从时序单人压力图恢复 SMPL 参数。

架构包含三部分（Figure 5）：

- **图像编码器**：ResNet 编码单人压力图（128×128），与检测框中心坐标 $c_{\text{bbox}}$ 和垫子四角坐标 $c_{\text{mat}}$（空间先验）拼接后降维
- **时序编码器**：2层 Transformer encoder，利用位置编码捕获时序依赖
- **SMPL 回归器**：N-to-1映射（取均值特征），迭代误差反馈（IEF）策略回归 SMPL 参数 $(\boldsymbol{\theta}, \boldsymbol{\beta}, T)$

### 损失函数 / 训练策略

$$\mathcal{L} = \mathcal{L}_{\text{SMPL}} + \mathcal{L}_{3D}$$

$$\mathcal{L}_{\text{SMPL}} = \lambda_\beta |\hat{\beta} - \beta| + \lambda_\theta |\hat{\theta} - \theta| + \lambda_T |\hat{T} - T|$$

$$\mathcal{L}_{3D} = \lambda_{3D} \|J_{3D} - \hat{J}_{3D}\|_2$$

采用两种数据集划分策略训练：
- **Unseen Sequence Split**：80%/20% 按序列划分，测试对已知个体未见序列的泛化
- **Unseen Subject Split**：2名志愿者数据仅用于测试，评估对全新个体的泛化

## 实验关键数据

### 主实验

#### 足迹跟踪性能

| 方法 | MOTA↑ | MOTP↑ | FN↓ | FP↓ | ID sw↓ | IDF1↑ |
|------|-------|-------|-----|-----|--------|-------|
| ByteTrack | 66.1% | 79.1% | 35207 | 22769 | 14453 | 4.8% |
| BoT-SORT | 82.5% | 87.7% | 17119 | 9020 | 10494 | 6.2% |
| **PressTrack** | **93.6%** | **94.8%** | **7317** | **6764** | **437** | **63.1%** |

PressTrack 的 ID switching 仅 437 次（86条轨迹、平均每条2660帧，约每523帧发生一次），远优于基线。

#### 端到端 HMR 性能

| 方法 | 数据模式 | MPJPE↓ | PA-MPJPE↓ | PVE↓ | WA-MPJPE100↓ | GMPJPE↓ |
|------|---------|--------|-----------|------|-------------|---------|
| GT Dets. | Unseen Seq | 81.8 | 46.1 | 115.7 | 90.9 | 99.4 |
| GT Dets. | Unseen Subj | 92.2 | 43.1 | 132.9 | 100.8 | 112.8 |
| Tracked Dets. | Unseen Seq | 89.2 | 48.8 | 134.4 | 112.6 | 118.3 |
| Tracked Dets. | Unseen Subj | 96.8 | 44.3 | 145.3 | 115.0 | 125.0 |

使用跟踪检测框vs真值框仅增加约7.4mm MPJPE，说明跟踪误差对 HMR 影响可控。对未见个体泛化良好（仅增加7.6mm）。

### 消融实验

| 序列长度 | GT Dets. MPJPE | Tracked Dets. MPJPE | 说明 |
|---------|---------------|---------------------|------|
| 1 | 94.2 | 96.2 | 仅单帧 |
| 4 | 88.7 | 92.3 | 时序帮助 |
| 8 | 86.6 | 90.7 | 继续改善 |
| 12 | 83.8 | 89.9 | |
| 16 | **81.8** | **89.2** | 最优长度 |
| 32 | 82.1 | 89.7 | 过长反而退化 |

最优序列长度为16帧：更短的序列时序感知不足，过长的序列因时间关联减弱和跟踪误差累积而退化。

### 关键发现

1. **UoE 显著优于视觉跟踪方法**：相比 ByteTrack 和 BoT-SORT，ID switching 降低了 96.9%/95.8%。
2. **时序信息关键**：序列长度从1增加到16时 MPJPE 降低12.4mm。
3. **跟踪误差可控**：端到端流水线中跟踪引入的额外误差有限（7-8mm MPJPE）。
4. **泛化能力强**：对未见个体的性能损失很小（7.6mm MPJPE）。

## 亮点与洞察

1. **填补重要空白**：首个仅从压力信号实现多人全局 HMR 的工作，开辟了隐私保护运动分析的新范式。
2. **问题分析深入**：对压力足迹动力学特性的分析（尺寸突变、跳跃运动）直接指导了 UoE 度量的设计，而非盲目套用视觉跟踪方法。
3. **数据集贡献**：MIP 数据集填补了多人交互压力数据的空白，分辨率（1cm×1cm）优于所有现有数据集。
4. **端到端可行性验证**：从原始压力数据到多人3D网格和全局轨迹，完整流水线的可行性得到充分验证。

## 局限与展望

1. **GT检测框与跟踪检测框的差距**：端到端性能与隔离 HMR 模块之间存在明显差距，可分析不同类型跟踪误差（ID switching vs 定位抖动）对 HMR 的影响并针对性优化。
2. **压力垫覆盖范围有限**：1.2×2.4m² 的尺寸限制了应用场景。
3. **依赖多视角RGB进行标注**：数据集标注依赖EasyMocap处理多视角视频，采集成本较高。
4. **未探索更复杂的交互场景**：如大规模人群（>3人）、接触式交互（如握手、拥抱）。

## 相关工作与启发

- **与视觉MOT的对比**：文中详细分析了压力足迹跟踪与视觉跟踪的本质差异，UoE 的设计体现了"因地制宜"的方法论。
- **压力感知的广泛应用前景**：除了步行场景，压力感知在卧床监护（PID-HMR）、运动分析（VP-MoCap）等领域也有应用。
- **隐私保护计算的趋势**：此工作代表了"隐私优先"传感方案的重要进展。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首创性工作，多人压力HMR、UoE度量、MIP数据集均为首次提出
- **实验充分度**: ⭐⭐⭐⭐ — 跟踪和HMR均有详细评估，有消融和泛化实验
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，动力学分析到位，流水线描述完整
- **实用价值**: ⭐⭐⭐⭐ — 隐私保护场景下多人运动分析的实用方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Multi-HMR: Multi-Person Whole-Body Human Mesh Recovery in a Single Shot](../../ECCV2024/3d_vision/multi-hmr_multi-person_whole-body_human_mesh_recovery_in_a_single_shot.md)
- [\[ECCV 2024\] Global-to-Pixel Regression for Human Mesh Recovery](../../ECCV2024/3d_vision/global-to-pixel_regression_for_human_mesh_recovery.md)
- [\[CVPR 2026\] MetricHMSR: Metric Human Mesh and Scene Recovery from Monocular Images](../../CVPR2026/3d_vision/metrichmsr_metric_human_mesh_and_scene_recovery_from_monocular_images.md)
- [\[CVPR 2026\] OnlineHMR: Video-based Online World-Grounded Human Mesh Recovery](../../CVPR2026/3d_vision/onlinehmr_video-based_online_world-grounded_human_mesh_recovery.md)
- [\[CVPR 2025\] PromptHMR: Promptable Human Mesh Recovery](../../CVPR2025/3d_vision/prompthmr_promptable_human_mesh_recovery.md)

</div>

<!-- RELATED:END -->
