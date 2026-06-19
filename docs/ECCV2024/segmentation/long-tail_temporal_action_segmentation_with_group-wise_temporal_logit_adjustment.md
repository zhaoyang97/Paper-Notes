---
title: >-
  [论文解读] Long-Tail Temporal Action Segmentation with Group-wise Temporal Logit Adjustment
description: >-
  [ECCV 2024][语义分割][时序动作分割] 首次系统性地解决时序动作分割中的长尾问题，提出 Group-wise Temporal Logit Adjustment (G-TLA) 框架，利用活动标签进行分组分类并结合动作时序先验进行 logit 调整，在大幅提升尾部类别性能的同时不损失头部类别。
tags:
  - "ECCV 2024"
  - "语义分割"
  - "时序动作分割"
  - "长尾分布"
  - "Logit 调整"
  - "程序化活动视频"
  - "过分割"
---

# Long-Tail Temporal Action Segmentation with Group-wise Temporal Logit Adjustment

**会议**: ECCV 2024  
**arXiv**: [2408.09919](https://arxiv.org/abs/2408.09919)  
**代码**: [pangzhan27/GTLA](https://github.com/pangzhan27/GTLA)  
**领域**: 时序动作分割  
**关键词**: 时序动作分割, 长尾分布, Logit 调整, 程序化活动视频, 过分割

## 一句话总结

首次系统性地解决时序动作分割中的长尾问题，提出 Group-wise Temporal Logit Adjustment (G-TLA) 框架，利用活动标签进行分组分类并结合动作时序先验进行 logit 调整，在大幅提升尾部类别性能的同时不损失头部类别。

## 研究背景与动机

时序动作分割将程序化活动视频逐帧分类为不同动作。该任务存在严重的长尾分布问题，且有双重来源：

**段级不均衡**：某些动作是可选的（如泡茶时加糖是可选的），出现频率差异悬殊（Breakfast 数据集不均衡比高达 639:1）

**帧级不均衡**：不同动作持续时间差异大，如"倒水"远比"加茶包"占更多帧

现有 SOTA（ASFormer、DiffAct）完全忽略长尾问题，在部分尾部类别上准确率为零。而将图像分类的长尾方法直接搬用又面临独特挑战：
- 动作间存在**时序依赖**，违反了传统方法的类别独立假设
- 简单 Logit Adjustment (LA) 会引入活动无关的假阳性（如泡茶时预测出"搅拌咖啡"）和违反时序逻辑的假阳性（如"搅茶"之后出现"加茶包"）
- 需要同时平衡帧级和段级指标，简单方法往往顾此失彼导致过分割

## 方法详解

### 整体框架

G-TLA 框架包含两个核心组件，均作用于基础分割模型（如 MSTCN、ASFormer）的分类层：
1. **分组分类 (Group-wise Classification)**：根据活动标签将序列分组，每组使用独立分类器
2. **时序 Logit 调整 (Temporal Logit Adjustment)**：在组内利用动作顺序先验约束 logit 调整的时间范围

### 关键设计

1. **分组分类策略**：

    - 根据活动标签将视频序列划分为互斥组 $\mathbf{G}$，如"泡茶"为 $G_1$、"泡咖啡"为 $G_2$
    - 每组引入辅助类"others"表示不属于该组的动作
    - 共享动作（如"加糖"同时出现在泡茶和泡咖啡中）在不同组中视为不同类别
    - 最后特征层 $z_t$ 同时送入 $n$ 个组分类器：$s_{c,t}^{(i)}(X) = \sum_j z_t[j] \cdot W_{j,c}^{(i)} + b^{(i)}$
    - 损失函数分为目标组动作分类和非目标组"others"分类两项
    - 设计动机：消除活动不兼容类别的干扰（$p(c|a)=0$ 的类不会被 LA 错误提升），减少语义相似动作的混淆

2. **时序 Logit 调整**：

    - 标准 LA 对所有帧统一调整：$s_{c,t}^{(k)}(X) + \tau \log p(c|G_k)$
    - G-TLA 引入时序因子 $\mathcal{T}_{c,t}^{(k)}(X)$：$s_{c,t}^{(k)}(X) + \tau \mathcal{T}_{c,t}^{(k)}(X) \log p(c|G_k)$
    - **时序界限**：对每个动作 $c$ 计算其前驱集 $S_{bf}[c]$ 和后继集 $S_{af}[c]$，确定可出现的时间窗口 $[t_1(c,X), t_2(c,X)]$
    - 窗口内正常调整（$\mathcal{T}=1$），窗口外保持一致调整防止违反时序先验
    - 窗口外的调整因子：$\mathcal{T}_{c,t}^{(k)} = \frac{\log p(y_t|G_k)}{\log p(c|G_k)}$，确保不破坏真实标签与候选类之间的决策边界
    - 设计动机：防止 LA 在时序不合理的位置引入假阳性（如搅茶应在倒水之后，不应出现在之前）

3. **推理策略**：

    - 活动标签未知时，选择"others"预测概率最低的组作为活动预测：$\hat{k} = \arg\min_i \frac{1}{T}\sum_t \hat{p}(o_t^{(i)})$
    - 推理时不使用时序 logit 调整，直接 argmax 预测概率
    - 无活动标签时可通过 KL 散度聚类替代

### 损失函数 / 训练策略

- 总损失：$\mathcal{L} = \mathcal{L}_{GTLA} + \lambda \mathcal{L}_{sm}$
- 分组损失：$\mathcal{L}_{GTLA} = \alpha_k \frac{1}{T}\sum_t -\log \tilde{p}(y_t^{(k)}) + \eta \sum_{i \neq k}^n \frac{1}{T}\sum_t -\log \hat{p}(o_t^{(i)})$
- $\eta$ 控制目标组与非目标组损失平衡，$\alpha_k$ 用于组间样本数均衡
- $\tau$ 控制均衡误差与偏斜误差的权衡
- 平滑损失 $\mathcal{L}_{sm}$ 鼓励帧间预测平滑过渡，阈值 $\delta=4$
- 使用预提取的 I3D 特征，与原始 backbone 训练协议一致

## 实验关键数据

### 主实验

YouTube Instructional Videos（ASFormer backbone，增量报告）：

| 方法 | 类型 | Frame Acc Hmean | Seg F1@25 Hmean | 全局 Acc |
|------|------|----------------|-----------------|---------|
| ASFormer 基线 | — | 26.0 | 28.4 | 69.8 |
| + CB (重加权) | reweight | +2.8 | +0.8 | -0.2 |
| + LA (logit adj.) | logit adj. | +5.3 | +2.1 | -1.9 |
| + BAGS (集成) | ensemble | +3.3 | +1.6 | -0.5 |
| **+ G-TLA (ours)** | **logit adj.** | **+7.5** | **+4.6** | **+0.1** |

Breakfast（MSTCN backbone）：

| 方法 | Frame Acc Hmean | Seg F1@25 Hmean | 全局 Acc |
|------|----------------|-----------------|---------|
| MSTCN 基线 | 47.7 | 44.8 | 67.7 |
| + LA | +2.1 | +0.9 | -0.1 |
| + Seesaw | +2.4 | +0.5 | +0.9 |
| **+ G-TLA (ours)** | **+5.0** | **+6.7** | **+2.6** |

MSTCN 上 G-TLA 在 Breakfast 的 Seg F1 提升 6.7 点，全局 Acc 也提升 2.6 点，多数竞争方法牺牲全局指标。

### 消融实验

Breakfast 上逐步添加组件（MSTCN backbone）：

| GP | LA | TF | Frame Hmean | Seg F1 Hmean | 说明 |
|----|----|----|------------|-------------|------|
| ✗ | ✗ | ✗ | 47.7 | 44.8 | 基线 |
| ✗ | ✓ | ✗ | 49.8 | 45.7 | 朴素 LA，头部下降 |
| ✓ | ✗ | ✗ | 50.9 | 51.3 | 分组分类显著减少过分割 (+5.5% F1) |
| ✓ | ✓ | ✗ | 51.3 | 51.2 | 组内 LA |
| ✓ | ✓ | ✓ | **52.7** | **51.5** | 时序因子进一步减少假阳性 |

超参数敏感性：$\eta=0.5$ 和 $\tau=0.5$ 时最优，变化范围 0.1-0.7 内性能较稳定。

### 关键发现

- 分组分类是最关键的组件：MSTCN 上单独 GP 即带来 F1 +5.5 提升
- 传统长尾方法（CB、Focal、LA 等）在时序分割上表现不佳，原因是忽略了动作间的依赖关系
- 许多方法提升帧准确率但牺牲段级 F1（过分割问题），G-TLA 能同时改善两者
- ASFormer/DiffAct 等 SOTA 在近 10% 的类别上准确率为零，G-TLA 有效缓解
- 在平衡数据集（50Salads、GTEA）上也有稳定提升，适用范围广

## 亮点与洞察

- **问题定义贡献大于方法贡献**：首次系统揭示时序分割的长尾问题，并提出对应的评估指标（per-class harmonic mean）
- 分组分类巧妙规避了条件概率 $p(c|a)=0$ 带来的数值问题（$\log 0$ 问题）
- 时序因子的设计非常精巧：在时间窗口内正常调整，窗口外通过比值保持一致性，既不过度抑制也不引入假阳性
- 即插即用：G-TLA 只修改分类层，可直接搭载到 MSTCN、ASFormer、DiffAct 等任意 backbone

## 局限与展望

- 需要活动标签（或聚类结果）来确定分组，增加了先验信息需求
- 时序约束 $S_{bf}[c]$ 和 $S_{af}[c]$ 从训练数据统计得到，对于训练集中未出现的新顺序可能失效
- Assembly101 等超大规模数据集上尾部提升仍然有限（4.7→9.2 Frame Acc，绝对值仍低）
- 未探索将分组信息编码到特征学习中（目前仅作用于分类层）

## 相关工作与启发

- 与 Logit Adjustment [Menon et al.] 的关系：G-TLA 是其在时序结构化预测任务上的非trivial 扩展，核心创新在于条件化先验和时序因子
- 与 BAGS 集成方法的关系：分组分类有类似分而治之的思路，但 G-TLA 更加系统化且引入了时序约束
- 启发：程序化活动视频中的结构化先验（活动-动作层次、时序顺序）是被严重低估的信息源

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次解决时序分割长尾问题，分组+时序 logit 调整的组合设计合理且新颖
- **实验充分度**: ⭐⭐⭐⭐⭐ — 5 个数据集，3 种 backbone，7 种长尾基线对比，详细消融和超参分析
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，泡茶的 running example 贯穿全文非常直观
- **价值**: ⭐⭐⭐⭐ — 填补重要研究空白，方法即插即用实用性强，新评估指标有推广价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Skeleton Motion Words for Unsupervised Skeleton-Based Temporal Action Segmentation](../../ICCV2025/segmentation/skeleton_motion_words_for_unsupervised_skeleton-based_temporal_action_segmentati.md)
- [\[ECCV 2024\] VP-SAM: Taming Segment Anything Model for Video Polyp Segmentation via Disentanglement and Spatio-Temporal Side Network](vp-sam_taming_segment_anything_model_for_video_polyp_segmentation_via_disentangl.md)
- [\[ECCV 2024\] Segmentation-Guided Layer-Wise Image Vectorization with Gradient Fills](segmentation-guided_layer-wise_image_vectorization_with_gradient_fills.md)
- [\[ECCV 2024\] DreamLIP: Language-Image Pre-training with Long Captions](dreamlip_language-image_pre-training_with_long_captions.md)
- [\[ICCV 2025\] Temporal Rate Reduction Clustering for Human Motion Segmentation](../../ICCV2025/segmentation/temporal_rate_reduction_clustering_for_human_motion_segmentation.md)

</div>

<!-- RELATED:END -->
