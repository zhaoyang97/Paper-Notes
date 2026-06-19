---
title: >-
  [论文解读] SPAMming Labels: Efficient Annotations for the Trackers of Tomorrow
description: >-
  [ECCV 2024][视频理解][多目标跟踪] 提出 SPAM 视频标注引擎，将合成数据预训练、伪标签自训练和基于图层级的主动学习相结合，仅需 3-20% 的人工标注量即可产生接近 GT 质量的多目标跟踪标注。 多目标跟踪（MOT）是视频理解中的核心任务，但高质量的轨迹标注极为昂贵： - 标注成本高：每帧需要检测、定位（b…
tags:
  - "ECCV 2024"
  - "视频理解"
  - "多目标跟踪"
  - "视频标注引擎"
  - "伪标签"
  - "主动学习"
  - "图神经网络"
---

# SPAMming Labels: Efficient Annotations for the Trackers of Tomorrow

**会议**: ECCV 2024  
**arXiv**: [2404.11426](https://arxiv.org/abs/2404.11426)  
**代码**: [https://research.nvidia.com/labs/dvl/projects/spam](https://research.nvidia.com/labs/dvl/projects/spam)  
**领域**: 视频理解（多目标跟踪 / 高效标注）  
**关键词**: 多目标跟踪, 视频标注引擎, 伪标签, 主动学习, 图神经网络

## 一句话总结

提出 SPAM 视频标注引擎，将合成数据预训练、伪标签自训练和基于图层级的主动学习相结合，仅需 3-20% 的人工标注量即可产生接近 GT 质量的多目标跟踪标注。

## 研究背景与动机

多目标跟踪（MOT）是视频理解中的核心任务，但高质量的轨迹标注极为昂贵：
- **标注成本高**：每帧需要检测、定位（bounding box 两次点击）和跨帧身份关联（一次点击），时间密度大
- **现有数据集小**：MOT17 仅 14 个视频序列，MOT20 仅 8 个，远不及图像领域的大规模数据集
- **现有高效标注方案不足**：
    - 大多数方法忽略视频的稠密时序依赖（如仅选关键帧标注）
    - 或仅限单目标场景
    - 没有同时处理检测和关联标注的统一方案

**核心洞察**：
1. 大多数跟踪场景中的关联是"容易的"——预训练模型可以以零成本生成高质量伪标签
2. 轨迹标注具有时空依赖性——标注一条轨迹会级联影响邻近轨迹，应以轨迹为中心（而非帧为中心）进行标注

## 方法详解

### 整体框架

SPAM = **S**ynthetic pre-training + **P**seudo-labeling + **A**ctive learning + graph-based **M**odel

流水线：
1. 在合成数据（MOTSynth）上预训练检测器、ReID 网络和 GNN 层级
2. 用预训练模型在目标真实数据集上生成伪标签，然后自训练微调
3. 用更新后模型的伪标签标注大部分数据；对不确定的困难样本使用主动学习选择人工标注
4. 输出最终的高质量标注供下游跟踪器训练

### 关键设计

1. **图层级模型（Hierarchical GNN + GNN_node）**：

    - 基于 SUSHI 的层级图神经网络：将视频分为子序列构建子图，逐层合并短轨迹为长轨迹
    - 节点 = 检测候选，边 = 关联假设
    - **创新：新增 GNN_node 层用于检测过滤**：
        - 使用低置信度阈值的检测器获取过完整的候选集（高召回率 → 多假阳性）
        - GNN_node 利用时空一致性在图上分类节点为有效/无效检测
        - 实验证明只加低置信框不加 GNN_node 性能暴跌（MOTA 从 64.4 降到 60.6），加 GNN_node 后提升到 65.4

2. **合成预训练 + 域差距分析**：

    - 深入分析三大跟踪组件（检测、关联、ReID）的合成-真实域差距
    - 结论：**检测受域差距影响最大**（9.9 HOTA 点差距），ReID 几乎不受影响，关联影响中等（2.1 HOTA）
    - 因此标注重点应放在检测和关联上，ReID 直接用合成数据训练的模型即可

3. **基于不确定性的主动学习（图层级标注）**：

    - 对每个节点 $v$，计算不确定性：$\text{uncert}(v) = \max_{u \in N_v} H(\hat{y}_{(v,u)})$
    - $H$ 为二值交叉熵不确定度
    - 高不确定性节点交给人工标注，其余用模型伪标签
    - **层级标注**：将标注预算 $B$ 分配到各层级 $B_1, ..., B_L$
    - 深层节点代表整条轨迹，标注一次可解决多个检测的身份关联 → 预算使用更高效
    - 标注操作类型：(i) 接受/拒绝检测（1 次点击），(ii) 修正框（2 次点击），(iii) 跨帧关联（1 次点击）

### 损失函数 / 训练策略

- GNN 模型端到端训练，边分类 + 节点分类
- 合成数据预训练 → 伪标签自训练（无人工标注成本） → 主动学习标注困难样本
- 伪标签自训练带来 4-6 HOTA 点提升（零人工成本）

## 实验关键数据

### 主实验

SPAM 作为跟踪器的测试集结果（与 SOTA 跟踪器对比）：

| 方法 | MOT17 HOTA↑ | MOT17 IDF1↑ | MOT20 HOTA↑ | DanceTrack HOTA↑ |
|------|------------|-------------|------------|-----------------|
| ByteTrack | 62.8 | 77.1 | 60.4 | 47.7 |
| GHOST | 62.8 | 77.1 | 61.2 | 56.7 |
| SUSHI | 66.5 | 83.1 | 64.3 | 63.3 |
| **SPAM** | **67.5** | **84.6** | **65.8** | **64.0** |

SPAM 标签训练下游跟踪器 vs GT 标签（MOT17 验证集）：

| 跟踪器 | 标签来源 | 标注量 | HOTA↑ | MOTA↑ |
|--------|---------|--------|-------|-------|
| ByteTrack | GT | 100% | 52.6 | 60.4 |
| ByteTrack | SPAM | 3.3% | 52.5 | 61.8 |
| GHOST | GT | 100% | 49.5 | 58.0 |
| GHOST | SPAM | 3.3% | 51.3 | 61.9 |

**仅用 3.3% 人工标注量即达到甚至超过 GT 训练水平！**

### 消融实验

| 配置 | HOTA↑ | MOTA↑ | IDF1↑ | 说明 |
|------|-------|-------|-------|------|
| 仅高置信框（无 GNN_node） | 59.9 | 64.4 | 74.7 | 基线 |
| 加低置信框（无 GNN_node） | 58.5 | 60.6 | 71.4 | 假阳性增多，性能下降 |
| 加低置信框 + GNN_node | 60.4 | 65.4 | 75.1 | GNN_node有效过滤假阳性 |

伪标签自训练效果（SPAM 模型本身，无人工标注）：

| 数据集 | 无伪标签 HOTA | 有伪标签 HOTA | 提升 |
|--------|-------------|-------------|------|
| MOT17 | 60.0 | 63.8 | +3.8 |
| MOT20 | 52.2 | 58.7 | +6.5 |
| DanceTrack | 41.8 | 48.1 | +6.3 |

### 关键发现

- **合成预训练足以覆盖大部分简单场景**：ReID 在合成数据上训练即可，检测和关联才需要真实数据微调
- **伪标签自训练效果惊人**：无需任何人工标注，仅靠合成预训练模型生成伪标签再自训练，就能提升 4-6 HOTA
- **图层级主动学习显著优于帧级标注**：对比实验显示，在节点级做不确定性采样远优于图像级采样
- **层级标注更高效**：深层节点代表长轨迹，一次标注解决多处不确定性

## 亮点与洞察

- **SPAM 理念极具实用价值**：3% 标注量 ≈ 100% 效果，对大规模跟踪数据集的构建意义重大
- **统一检测+关联标注的图框架**：GNN_node + 边分类在一个统一的图结构中同时处理两类标注问题
- **域差距分析提供了标注优先级指导**：检测 > 关联 > ReID，这个结论对跟踪领域的数据采集有直接指导意义
- 自训练 loop（合成预训练 → 伪标签 → 重训练）形成了无需人工标注的强力基线

## 局限与展望

- 标注器本身不生成新检测——如果检测器漏检，只能通过低置信阈值弥补，无法完全恢复
- 对极端密集场景（如 MOT20）GNN_node 的假阳性过滤可能不够充分
- 未探索标注后的再训练迭代——多轮自训练是否能继续提升？
- 当前仅验证了 ByteTrack 和 GHOST 两个下游跟踪器，更多跟踪器的验证会更有说服力

## 相关工作与启发

- SUSHI 是本文 GNN 层级架构的直接前身，SPAM 在其基础上增加了 GNN_node 和标注引擎
- 与图像领域的高效标注方法（如 DINO 自训练）思路相通，但首次系统化地应用于视频跟踪领域
- 对未来视频理解数据的构建有启发：从"标注所有帧"转向"选择性标注困难实例"

## 评分

- 新颖性: ⭐⭐⭐⭐ （系统集成方案新颖，各单点技术是已有的巧妙组合）
- 实验充分度: ⭐⭐⭐⭐⭐ （MOT17/20/DanceTrack 三数据集 + 完整消融 + 域差距分析 + 下游验证）
- 写作质量: ⭐⭐⭐⭐ （系统描述清晰，实验组织合理）
- 价值: ⭐⭐⭐⭐⭐ （对跟踪数据集的扩展有直接实用价值，3% 标注量的结论非常有吸引力）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] OneTrack: Demystifying the Conflict Between Detection and Tracking in End-to-End 3D Trackers](onetrack_demystifying_the_conflict_between_detection_and_tracking_in_end-to-end_.md)
- [\[ECCV 2024\] VideoMamba: State Space Model for Efficient Video Understanding](videomamba_state_space_model_for_efficient_video_understanding.md)
- [\[ICCV 2025\] XTrack: Multimodal Training Boosts RGB-X Video Object Trackers](../../ICCV2025/video_understanding/xtrack_multimodal_training_boosts_rgb-x_video_object_trackers.md)
- [\[ECCV 2024\] SAFNet: Selective Alignment Fusion Network for Efficient HDR Imaging](safnet_selective_alignment_fusion_network_for_efficient_hdr_imaging.md)
- [\[ECCV 2024\] SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow](sea-raft_simple_efficient_accurate_raft_for_optical_flow.md)

</div>

<!-- RELATED:END -->
