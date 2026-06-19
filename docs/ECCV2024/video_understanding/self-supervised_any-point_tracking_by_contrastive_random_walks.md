---
title: >-
  [论文解读] Self-Supervised Any-Point Tracking by Contrastive Random Walks
description: >-
  [ECCV 2024][视频理解][Tracking Any Point] 提出 GMRW（Global Matching Random Walk），将全局匹配 Transformer 架构与对比随机游走自监督目标结合，首次在无标注的情况下实现了强劲的"任意点跟踪"（TAP）性能，并设计 label warping 数据增强来避免 Transformer 的捷径解。
tags:
  - "ECCV 2024"
  - "视频理解"
  - "Tracking Any Point"
  - "自监督学习"
  - "对比随机游走"
  - "Transformer"
  - "循环一致性"
---

# Self-Supervised Any-Point Tracking by Contrastive Random Walks

**会议**: ECCV 2024  
**arXiv**: [2409.16288](https://arxiv.org/abs/2409.16288)  
**代码**: [https://ayshrv.com/gmrw](https://ayshrv.com/gmrw)  
**领域**: 视频理解（点跟踪 / 自监督学习）  
**关键词**: Tracking Any Point, 自监督学习, 对比随机游走, 全局匹配Transformer, 循环一致性

## 一句话总结

提出 GMRW（Global Matching Random Walk），将全局匹配 Transformer 架构与对比随机游走自监督目标结合，首次在无标注的情况下实现了强劲的"任意点跟踪"（TAP）性能，并设计 label warping 数据增强来避免 Transformer 的捷径解。

## 研究背景与动机

**Tracking Any Point (TAP)** 是近年兴起的视频理解任务：给定视频中任意一个物理点的查询位置，预测该点在所有帧中的轨迹及可见性。不同于光流（仅相邻帧局部运动）或语义跟踪（可能匹配到同类但不同物理点），TAP 要求精确跟踪同一个物理点。

现有方法的局限：
- **有监督方法**（TAP-Net、PIPs、TAPIR、CoTracker）依赖合成数据的 GT 轨迹训练，标注数据有限
- **自监督光流方法**（RAFT 无标注版、ARFlow）只能预测短程运动，长时间跟踪靠链式传播累积误差
- **扩散特征方法**（DIFT）适合语义对应但非物理点跟踪
- **早期对比随机游走**（CRW）在粗粒度 patch 上操作（7×7），空间精度极低

**核心动机**：利用全局匹配 Transformer（GMFlow 架构）的"全对比较"能力来定义随机游走的转移矩阵，在高分辨率（64×64 grid）上实现精确的自监督点跟踪。

## 方法详解

### 整体框架

1. CNN 提取每帧特征，添加位置编码
2. 成对帧通过 6 层自注意力-交叉注意力-FFN 的全局匹配 Transformer，输出相关特征 $F_t$, $F_{t+1}$
3. 计算转移矩阵 $A_t^{t+1} = \text{softmax}(F_t F_{t+1}^\top / \tau)$
4. 用对比随机游走损失（循环一致性）进行自监督训练
5. 推理时，取转移矩阵的期望坐标作为轨迹预测

### 关键设计

1. **全局匹配转移矩阵**：

    - 采用 GMFlow 的 Transformer 架构进行帧间全局匹配
    - 与早期 CRW（7×7 粗粒度 patch）相比，GMRW 在 64×64 分辨率上操作，空间精度大幅提升
    - "全对比较"机制允许模型同时考虑大量匹配假设，产生更丰富的对比学习信号
    - 转移矩阵 $A_t^{t+1}$ 直接用做随机游走的概率转移矩阵，无需额外的从粗到细匹配

2. **Label Warping 解决捷径问题**：

    - 循环一致性训练的经典捷径：模型忽略视觉内容，仅基于位置编码匹配（因为位置不变，回到原位是平凡解）
    - Tang et al. 提出对前向/后向循环使用不同裁剪增强，但对 Transformer 无效——Transformer 有足够的全局自注意力层"撤销"空间变换
    - **GMRW 的创新**：不对特征做 warp，而是 warp 标签
    - 对回文序列 $[T^f(I_1), T^f(I_2), T^b(I_1)]$ 使用不同空间变换 $T^f$, $T^b$
    - 损失变为 $\mathcal{L}_{crw} = \mathcal{L}_{CE}(A_s, T_f^b(I))$，其中 $T_f^b(I)$ 是变换后的单位矩阵
    - 模型必须在不同空间变换之间找到正确对应，无法通过位置编码"作弊"

3. **采样步幅策略**：

    - 通过上采样原始图像来实现不同步幅（stride $s \in \{1, 2, 4\}$）的特征采样
    - $s=1$ 时达到最高空间精度（但计算量大），$s=4$ 时更快但更粗
    - 训练时用 $s=2$，评估时可灵活切换

### 损失函数 / 训练策略

对比随机游走损失（CRW）+ 可选平滑损失：

$$\mathcal{L}_{total} = \mathcal{L}_{crw} + \lambda_s \mathcal{L}_{smooth}$$

- **CRW 损失**：最大化随机游走在回文序列中回到起点的概率
  $$\mathcal{L}_{crw} = \mathcal{L}_{CE}(A_t^{t+1} A_{t+1}^t, I)$$
- **平滑损失**：边缘感知的二阶导数正则化，促进运动场空间平滑
- 训练数据：Kubric 合成数据集（38,325 个视频），**不使用任何标注**
- 也在 Kinetics-400 上训练了"in the wild"版本

## 实验关键数据

### 主实验

TAPVid 基准测试（Kubric 和 DAVIS，自监督方法对比）：

| 方法 | 类型 | Kubric AJ↑ | Kubric δ_avg↑ | DAVIS AJ↑ | DAVIS δ_avg↑ |
|------|------|-----------|---------------|----------|--------------|
| CRW-C | 自监督 | 31.4 | 48.1 | 7.7 | 13.5 |
| CRW-D | 自监督 | 35.8 | 52.4 | 23.6 | 38.0 |
| DIFT-D | 自监督 | 41.6 | 59.8 | 29.7 | 48.2 |
| FlowWalk-C | 自监督 | 49.4 | 66.7 | 35.2 | 51.4 |
| ARFlow-C | 自监督 | 52.3 | 68.1 | 35.0 | 51.8 |
| **GMRW-C** | **自监督** | **54.2** | **72.4** | **41.8** | **60.9** |
| TAP-Net | 有监督 | 65.4 | 77.7 | 38.4 | 53.1 |
| TAPIR | 有监督 | 84.7 | 92.1 | 61.3 | 73.6 |

GMRW 在自监督方法中全面领先，Kubric AJ 超过 FlowWalk 4.8 点，DAVIS AJ 超过 6.6 点。在 DAVIS 上甚至超越有监督的 TAP-Net（41.8 vs 38.4 AJ）。

### 消融实验

模型变体分析（Kubric + DAVIS 上的 AJ/δ_avg/OA）：

| 配置 | Kubric AJ↑ | Kubric δ_avg↑ | DAVIS AJ↑ | DAVIS δ_avg↑ | 说明 |
|------|-----------|---------------|----------|--------------|------|
| 有监督 Oracle | 63.7 | 83.2 | 39.1 | 59.6 | 上界 |
| 基础 CRW | 25.4 | 39.3 | 10.4 | 19.2 | 无 label warping |
| + Label Warping | 45.3 | 62.2 | 32.1 | 48.9 | **+19.9 AJ，核心贡献** |
| + Smoothness | 49.0 | 66.7 | 33.0 | 50.5 | +3.7 |
| + Train stride s=2 | 47.7 | 65.6 | 34.5 | 52.1 | 略有不同 |
| Eval stride s=4 | 37.8 | 53.8 | 23.5 | 38.4 | 粗步幅性能下降 |
| Eval stride s=1 | 54.2 | 72.4 | 41.8 | 60.9 | 最细步幅最优 |
| Kinetics 训练 | 47.5 | 65.0 | 34.6 | 52.6 | 真实视频亦可训练 |

### 关键发现

- **Label Warping 是最关键贡献**：仅此一项从 AJ 25.4 → 45.3（+19.9），没有它 Transformer 直接学到捷径解
- **评估步幅对性能影响极大**：$s=1$ vs $s=4$ 在 Kubric 上 AJ 差 16.4，精细匹配确实重要
- **在 Kubric 上训练可泛化到 DAVIS 真实视频**：虽然训练数据是合成的，但特征对应能力迁移良好
- **在 Kinetics 上训练效果相近**（AJ 34.6 vs Kubric 训练的 34.5 on DAVIS），说明方法不依赖合成数据
- **自监督 vs 有监督差距仍在**：与 TAPIR（84.7 Kubric AJ）相比差距明显，但考虑到完全无标注，结果已非常可观
- **可见性预测通过循环一致性阈值实现**：若前向-后向循环的位移误差超过 $\tau_{cyc}=3$ 像素，则标记为被遮挡

## 亮点与洞察

- **简洁高效的自监督方案**：整个方法概念简单——全局匹配 + 随机游走 + label warping，无需复杂的多阶段训练或粗到细匹配
- **Label warping 极具启发性**：以往对 CNN 有效的数据增强策略对 Transformer 失效，因为 Transformer 的全局感受野可以轻松"反转"空间变换。warp 标签而非特征是一个优雅的解决方案
- **全局匹配的双重优势**：(1) 高空间精度（64×64 vs 早期 7×7），(2) 每次迭代考虑大量路径，提供更丰富的梯度信号
- 可见性检测无需额外模块，直接复用循环一致性检查

## 局限与展望

- 与有监督 SOTA（TAPIR/CoTracker）差距仍然显著，特别是在遮挡处理和长时跟踪上
- 仅使用单尺度匹配（GMFlow 原始设计支持双尺度），加入多尺度可能进一步提升
- 评估步幅 $s=1$ 的计算开销大：64×64 分辨率的全对匹配 = 4096×4096 的注意力矩阵
- 未探索在大规模无标注视频上训练的潜力（如用 YouTube 数据）
- 循环一致性天然无法处理"消失再出现"的遮挡模式

## 相关工作与启发

- CRW（对比随机游走）是本文的理论基础，GMRW 将其从粗粒度 CNN 升级到高精度全局 Transformer
- GMFlow 的全局匹配架构被创造性地用于自监督——原始设计是有监督光流估计
- Label warping 可能对其他使用 Transformer + 循环一致性的任务（如自监督深度估计、3D 对应学习）也有启发
- 证明了"自监督可以逼近有监督基线"的趋势在点跟踪领域也成立

## 评分

- 新颖性: ⭐⭐⭐⭐ （全局匹配+CRW 的组合新颖，label warping 巧妙解决 Transformer 特有问题）
- 实验充分度: ⭐⭐⭐⭐ （四个 TAPVid 基准 + 多种自监督/有监督基线对比 + 完整消融）
- 写作质量: ⭐⭐⭐⭐ （方法描述清晰，但论文较短）
- 价值: ⭐⭐⭐⭐ （自监督点跟踪方向有潜力，但距实用还有距离）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] DINO-Tracker: Taming DINO for Self-Supervised Point Tracking in a Single Video](dino-tracker_taming_dino_for_self-supervised_point_tracking_in_a_single_video.md)
- [\[ECCV 2024\] Local All-Pair Correspondence for Point Tracking](local_all-pair_correspondence_for_point_tracking.md)
- [\[CVPR 2025\] ETAP: Event-based Tracking of Any Point](../../CVPR2025/video_understanding/etap_event-based_tracking_of_any_point.md)
- [\[CVPR 2026\] MV-TAP: Tracking Any Point in Multi-View Videos](../../CVPR2026/video_understanding/mv-tap_tracking_any_point_in_multi-view_videos.md)
- [\[ECCV 2024\] TimeCraft: Navigate Weakly-Supervised Temporal Grounded Video Question Answering via Bi-directional Reasoning](timecraft_navigate_weakly-supervised_temporal_grounded_video_question_answering_.md)

</div>

<!-- RELATED:END -->
