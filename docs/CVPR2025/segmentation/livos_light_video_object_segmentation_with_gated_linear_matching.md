---
title: >-
  [论文解读] LiVOS: Light Video Object Segmentation with Gated Linear Matching
description: >-
  [CVPR 2025][语义分割][图像分割] 提出 LiVOS——首个使用门控线性注意力替代 softmax 注意力进行内存匹配的轻量 VOS 网络，将时空注意力矩阵压缩为恒定大小的 2D 状态矩阵，实现任意长视频的恒定内存占用，并在 32G 消费级 GPU 上支持 4096p 推理。 领域现状： 半监督 VOS 主要由时…
tags:
  - "CVPR 2025"
  - "语义分割"
  - "图像分割"
  - "注意力机制"
  - "gated linear matching"
  - "memory network"
  - "4096p inference"
---

# LiVOS: Light Video Object Segmentation with Gated Linear Matching

**会议**: CVPR 2025  
**arXiv**: [2411.02818](https://arxiv.org/abs/2411.02818)  
**代码**: [uncbiag/LiVOS](https://github.com/uncbiag/LiVOS)  
**领域**: 图像分割  
**关键词**: video object segmentation, linear attention, gated linear matching, memory network, 4096p inference

## 一句话总结

提出 LiVOS——首个使用门控线性注意力替代 softmax 注意力进行内存匹配的轻量 VOS 网络，将时空注意力矩阵压缩为恒定大小的 2D 状态矩阵，实现任意长视频的恒定内存占用，并在 32G 消费级 GPU 上支持 4096p 推理。

## 研究背景与动机

**领域现状**: 半监督 VOS 主要由时空记忆（STM）网络驱动，通过 softmax 注意力在 query 帧与所有 memory 帧间进行像素级匹配。代表方法有 XMem、Cutie 等。

**现有痛点**: Softmax 匹配需要存储 $\mathcal{O}(HW \times THW)$ 的注意力矩阵，空间复杂度关于视频长度线性增长、关于分辨率二次方增长。随着视频变长或分辨率提高，计算过慢或内存溢出。

**核心矛盾**: 固定大小的 memory bank 在遮挡、快速运动时会失败；降低分辨率则丢失细粒度掩码细节；两者都是因为 softmax 匹配的固有限制。

**本文切入角度**: 识别 softmax 匹配为核心瓶颈，从根本上替换匹配机制而非打补丁。

**核心 idea**: 将 softmax 注意力改写为线性注意力的递推形式，注意力矩阵退化为恒定大小的 2D 状态 $\mathbf{S}_t \in \mathbb{R}^{C_k \times C_v}$，并引入数据相关的门控矩阵增强选择性。

## 方法详解

### 整体框架

1. 图像编码器（ResNet-50）提取 query 帧的 key
2. 掩码编码器（ResNet-18）提取 memory 帧的 value
3. **门控线性匹配（核心）**: 用恒定大小的状态矩阵递推更新替代全量 softmax 匹配
4. 结合 sensory memory（低级信息）和 object memory（高级语义）增强 readout
5. 轻量掩码解码器输出分割结果

### 关键设计

**1. 线性匹配——从 softmax 到递推状态**
- **功能**: 将 softmax 匹配 $\mathbf{V}_{t+1} = \text{Softmax}(\mathbf{K}_{t+1}\mathbf{K}_{1:t}^T)\mathbf{V}_{1:t}$ 改写为核函数近似 $\phi(\mathbf{K}_{t+1})\mathbf{S}_t$。
- **核心思路**: 利用矩阵乘法结合律，将 $\sum_i \phi(\mathbf{K}_{t+1})\phi(\mathbf{K}_i)^T\mathbf{V}_i$ 重新分组为 $\phi(\mathbf{K}_{t+1}) \cdot \sum_i \phi(\mathbf{K}_i)^T\mathbf{V}_i$。定义状态 $\mathbf{S}_t = \mathbf{S}_{t-1} + \phi(\mathbf{K}_i)^T\mathbf{V}_i$，$\mathbf{S}_t \in \mathbb{R}^{C_k \times C_v}$ 为恒定大小。核函数 $\phi$ 使用行级 softmax。
- **设计动机**: 状态 $\mathbf{S}_t$ 是无关时空的 2D 矩阵，大小仅取决于特征维度（$64 \times 256$），与视频长度和分辨率无关。

**2. 门控线性匹配（Gated Linear Matching）**
- **功能**: 在状态更新中引入数据相关的遗忘门 $\mathbf{G}_t$，选择性保留或丢弃历史信息。
- **核心思路**: $\mathbf{S}_t = \mathbf{G}_t \odot \mathbf{S}_{t-1} + \phi(\mathbf{K}_i)^T\mathbf{V}_i$。门 $\mathbf{G}_t = \alpha_t \mathbf{1}^T$ 通过低秩参数化实现，$\alpha_t \in (0,1)^{C_k}$ 由深度卷积 + 空间求和 + Sigmoid 从图像编码器特征提取。
- **设计动机**: 纯线性匹配无选择机制，在长序列中性能退化；门控提供了类似 GRU/LSTM 的遗忘能力，在场景变化、遮挡等场景中能主动丢弃过时信息。

**3. 外部记忆融合**
- **功能**: 复用 Cutie 的 sensory memory（元素加法融合低级时序信息）和 object memory（交叉注意力融合高级对象语义）。
- **核心思路**: 线性匹配输出的 readout 依次与 sensory memory 和 object transformer 交互，补充恒定状态压缩丢失的信息。
- **设计动机**: 恒定状态压缩了时空信息，外部记忆提供互补的高频和语义信息。

### 损失函数 / 训练策略

- 交叉熵 + soft dice loss 等权结合
- AdamW 优化器，初始学习率 $10^{-4}$，batch size 16，权重衰减 0.001
- 每 batch 8 帧，裁剪到 480×480，125K 迭代训练
- 图像编码器学习率乘 0.1 降低过拟合，梯度裁剪 $\tau=3$
- 点采样监督（12544 个点），遵循 Cutie 的训练策略

## 实验关键数据

### 主实验

| 方法 | STM? | MOSE J&F↑ | DAVIS-17 val J&F↑ | DAVIS-17 test J&F↑ | YouTube-VOS 𝒢↑ |
|---|---|---|---|---|---|
| RDE | ✗ | 46.8 | 84.2 | 77.4 | 81.9 |
| Cutie-small† (1帧) | ✗ | 49.3 | 76.4 | 71.6 | 79.0 |
| **LiVOS（本文）** | ✗ | **64.8** | **85.1** | - | - |
| Cutie-small | ✓ | 62.2 | 87.2 | 84.1 | 86.2 |
| Cutie-base | ✓ | 64.0 | 88.8 | 84.2 | 86.1 |
| XMem | ✓ | 56.3 | 86.2 | 81.0 | 85.5 |

### 效率对比

| 指标 | LiVOS vs STM 方法 |
|---|---|
| GPU 内存节约 | **53%** |
| 长视频内存增长 | 恒定（vs softmax 线性增长） |
| 分辨率内存增长 | 线性（vs softmax 二次方） |
| 最大可推理分辨率 | **4096p**（32G GPU） |
| CPU 延迟随帧数 | 恒定（vs softmax 线性增长） |

### 关键发现

1. **LiVOS 超越所有非 STM 方法并缩小与 STM 方法的差距**：MOSE 64.8 vs Cutie-small 62.2，DAVIS 85.1 vs 87.2。
2. **在长视频和高分辨率场景中匹配 STM 方法性能**，同时节约 53% GPU 内存。
3. **4096p 推理成为可能**：STM 方法因 softmax 注意力在高分辨率下内存溢出，而 LiVOS 的恒定状态使其可在消费级 GPU 上处理。
4. **门控机制显著提升长序列性能**：在场景变化、遮挡等挑战场景中，门控状态能有效遗忘过时信息。

## 亮点与洞察

- 将 softmax→线性注意力的改造从文本/图像分类扩展到 VOS 这一视频内存密集型任务，具有示范意义
- 恒定大小状态矩阵的 insight 优雅：$C_k \times C_v = 64 \times 256$ 即可压缩任意长视频的全部时空信息
- 门控线性匹配的低秩参数化设计简洁高效
- 为长时间高分辨率视频基础模型的发展铺平了道路

## 局限与展望

- 恒定状态存在信息压缩损失，在短视频标准精度上仍有差距
- 门控参数化采用最简单的低秩形式，可探索更丰富的参数化
- 未针对高分辨率视频优化训练（仅 480p 训练），4096p 是测试时泛化
- 多对象场景中为每个对象维护独立状态，对象数量过多时仍有开销
- 未探索与 Flash Attention 等硬件优化的组合

## 相关工作与启发

- Cutie 通过 object transformer 增强了 XMem 的 readout 质量，LiVOS 在此基础上替换了核心匹配机制
- 门控线性注意力（GLA）在语言建模中已证明有效，本文将其拓展到视觉内存匹配
- 启发：恒定状态压缩思想可应用于其他需要时序内存的视频任务（如视频问答、视频生成）

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次将线性注意力应用于 VOS 内存匹配，门控机制设计合理
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖 MOSE/DAVIS/YouTube-VOS/LVOS，含效率和高分辨率实验
- **写作质量**: ⭐⭐⭐⭐ 从 softmax→线性→门控线性的推导过程清晰流畅
- **价值**: ⭐⭐⭐⭐⭐ 解决了 VOS 领域的核心可扩展性瓶颈，开启高分辨率长视频处理新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] O-MaMa: Learning Object Mask Matching between Egocentric and Exocentric Views](../../ICCV2025/segmentation/o-mama_learning_object_mask_matching_between_egocentric_and_exocentric_views.md)
- [\[CVPR 2025\] M3-VOS: Multi-Phase, Multi-Transition, and Multi-Scenery Video Object Segmentation](m3-vos_multi-phase_multi-transition_and_multi-scenery_video_object_segmentation.md)
- [\[CVPR 2026\] ELVIS: Enhance Low-Light for Video Instance Segmentation in the Dark](../../CVPR2026/segmentation/elvis_enhance_low-light_for_video_instance_segmentation_in_the_dark.md)
- [\[ICCV 2025\] MOVE: Motion-Guided Few-Shot Video Object Segmentation](../../ICCV2025/segmentation/move_motion-guided_few-shot_video_object_segmentation.md)
- [\[ECCV 2024\] ActionVOS: Actions as Prompts for Video Object Segmentation](../../ECCV2024/segmentation/actionvos_actions_as_prompts_for_video_object_segmentation.md)

</div>

<!-- RELATED:END -->
