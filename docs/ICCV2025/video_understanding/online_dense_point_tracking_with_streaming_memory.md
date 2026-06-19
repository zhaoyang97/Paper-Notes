---
title: >-
  [论文解读] Online Dense Point Tracking with Streaming Memory
description: >-
  [ICCV 2025][视频理解][稠密点跟踪] 提出 SPOT 框架，通过定制的记忆读取模块、感知记忆（sensory memory）和可见性引导的 splatting 实现在线稠密长程点跟踪，以 10× 更少参数和 2× 更快速度达到 CVO 基准上的 SOTA，在多个稀疏跟踪基准上也超越或媲美离线方法。
tags:
  - "ICCV 2025"
  - "视频理解"
  - "稠密点跟踪"
  - "流式记忆"
  - "光流"
  - "在线处理"
  - "可见性估计"
---

# Online Dense Point Tracking with Streaming Memory

**会议**: ICCV 2025  
**arXiv**: [2503.06471](https://arxiv.org/abs/2503.06471)  
**代码**: [项目页面](https://dqiaole.github.io/SPOT/)  
**领域**: 视频理解  
**关键词**: 稠密点跟踪, 流式记忆, 光流, 在线处理, 可见性估计

## 一句话总结

提出 SPOT 框架，通过定制的记忆读取模块、感知记忆（sensory memory）和可见性引导的 splatting 实现在线稠密长程点跟踪，以 10× 更少参数和 2× 更快速度达到 CVO 基准上的 SOTA，在多个稀疏跟踪基准上也超越或媲美离线方法。

## 研究背景与动机

稠密点跟踪要求持续跟踪初始帧中每个点在整段视频中的位置，即使面对遮挡也不能丢失。这本质上等价于长程光流估计。现有方法面临三个关键问题：

**光流方法的外观漂移**：传统方法用相邻帧训练的光流模型直接回归长程光流，但不考虑时间一致性，容易发生外观漂移。链式方法（chaining）有固有局限：前向累积处理遮挡差，后向累积导致处理时间线性增长。

**滑动窗口方法的低效性**：CoTracker、SpatialTracker 等最新点跟踪方法依赖滑动窗口做间接信息传播——从第一帧到当前帧需要通过窗口逐步传递信息，这既慢又对长程跟踪不够有效。且需要离线处理（依赖未来帧提高精度）。

**稠密跟踪的计算瓶颈**：Online TAPIR 虽然支持在线跟踪，但逐像素跟踪耗时巨大。DOT 结合光流和稀疏跟踪，但继承了两者的缺点且参数量大。

**核心问题**：能否仅基于过去的观测，同时实现高精度和高效率的稠密点跟踪？

**核心思路**：将长程信息传播这个困难任务拆解为两个简单子步骤——① 通过 splatting 将第一帧特征传送到近期帧位置（利用已预测的精确长程光流），② 通过注意力机制从近期帧特征中检索相关信息增强当前帧（近期帧间外观漂移小，相似度可靠）。

## 方法详解

### 整体框架

SPOT 的流程：
1. 提取当前帧 4× 下采样特征
2. 通过记忆读取模块用记忆库增强当前帧特征
3. 用标准光流解码器（RAFT）预测长程光流和可见性掩码
4. 用可见性引导的 splatting 将第一帧特征传送到当前帧坐标，更新记忆库

### 关键设计

1. **记忆读取的特征增强**

   记忆库存储 key-value 对，key 为近期帧的查询特征，value 为经 splatting 传送的第一帧特征。当前帧特征作为 query 通过注意力机制检索记忆：

    $\mathbf{M}_t = \text{Softmax}(\frac{1}{\sqrt{D_k}} \times q \times k^T) \times v$

   **关键设计——融合层**：直接使用 readout 特征 $\mathbf{M}_t$ 做光流回归会失败，因为 splatting 产生的 value 存在大量**空洞伪影**（disocclusion 导致）。为此引入一个简单的融合层——仅一个卷积——用原始特征"修补"空洞：

    $\mathbf{E}_t = \mathbf{F}_t + \text{Conv}(\mathbf{F}_t \oplus \mathbf{M}_t)$

   其中 $\oplus$ 为拼接操作。这个残差设计确保即使记忆特征有伪影，原始特征仍可保底。

2. **光流解码与感知记忆（Sensory Memory）**

   使用 RAFT 解码器架构，计算增强特征 $\mathbf{E}_t$ 与参考特征 $\mathbf{F}_1$ 之间的 4D 相关体积。GRU 单元迭代更新光流和可见性：

    $\Delta f_{1\to t}^i, \Delta v_{1\to t}^i, h_t^i = \text{GRU}(h_t^{i-1}, f_c, f_m^i, s_{t-1})$

   **感知记忆 $s_{t-1}$** 捕获短期运动动态，通过额外的 GRU 更新：

    $s_t = \text{GRU}_{sensory}(s_{t-1}, f_m^N)$

   设计动机：光流解码器的相关体积只捕获静态空间相似性，短期运动趋势需要额外建模。感知记忆使模型能感知"物体正在往哪个方向移动"，辅助长程光流预测。

3. **可见性引导的 Splatting**

   记忆读取依赖特征相似性，而相似性仅在近期几帧内可靠（因外观漂移）。因此需要**不断将第一帧的判别性特征传送到最新帧的坐标**。

   使用前向 splatting（forward warping）实现高效的长程信息传播：

    $F_t^{\Sigma}[(x_t, y_t)] = \sum_{(x_1, y_1)} b(\Delta) \cdot \mathbf{F}_1[(x_1, y_1)]$

   **可见性引导**：遮挡区域的 splatting 会产生不一致伪影，用预测的可见性掩码进行加权归一化：

    $F_t^{1\to t} = \frac{\sum^{\to}(\mathbf{v}_{1\to t} \cdot \mathbf{F}_1, \mathbf{f}_{1\to t})}{\sum^{\to}(\mathbf{v}_{1\to t}, \mathbf{f}_{1\to t})}$

   记忆库维护两个 FIFO 队列（长度 $L=3$），缓存近期帧的 splatting 结果作为 value，当前帧的查询特征作为 key。

### 损失函数 / 训练策略

- 光流预测：L1 损失
- 可见性预测：二元交叉熵损失

**Warm-Start 策略**：利用前一帧信息初始化当前帧的估计。GRU 隐藏状态 $h_t^0 = h_{t-1}^N$，光流通过一步外推初始化：$\mathbf{f}_{1\to t}^0 = \mathbf{f}_{1\to t-1}^0 + 2 \times (\mathbf{f}_{1\to t-1}^N - \mathbf{f}_{1\to t-1}^0)$。

训练分两阶段：
- Kubric-CVO 上预训练 500K 步（384×384）
- Kubric-MOVi-F 上微调 100K 步（24帧，384×384）

推理时 GRU 迭代 $N=16$ 次。

## 实验关键数据

### 主实验

CVO 长程光流（EPE↓，越低越好）：

| 方法 | 模式 | Clean | Final | Extended |
|------|------|-------|-------|----------|
| RAFT | 在线 | 2.82 | 2.88 | 28.6 |
| MFT | 在线 | 2.91 | 3.16 | 21.4 |
| DOT† | 在线 | 1.92 | 1.98 | 12.1 |
| DOT | 离线 | 1.34 | 1.37 | 5.12 |
| CoTracker2 | 离线 | 1.50 | 1.47 | 5.45 |
| **SPOT** | **在线** | **1.11** | **1.23** | **4.77** |

SPOT 在所有集合上 EPE 最低，Extended 集相比在线 DOT† 降低 60.5%（12.08→4.77），**甚至超越所有离线方法**。

TAP-Vid 稀疏点跟踪（AJ↑）：

| 方法 | 模式 | DAVIS(First) | RGB-S.(First) | Kinetics(First) |
|------|------|--------------|---------------|-----------------|
| Online TAPIR | 在线 | 56.2 | 65.9 | 49.6 |
| DOT† | 在线 | 53.3 | 61.3 | 45.3 |
| CoTracker2 | 离线 | 60.8 | 60.5 | 48.4 |
| **SPOT** | **在线** | **61.5** | **73.3** | **50.2** |

### 消融实验

模块消融（CVO Extended, 10帧训练）：

| 配置 | EPE↓ (all/vis/occ) | OA↑ | 说明 |
|------|---------------------|-----|------|
| Full | 6.42/3.86/9.98 | 88.5 | 完整模型 |
| - Feature fusion | NaN | NaN | 空洞伪影导致训练崩溃 |
| - Memory bank | 38.98/28.34/57.15 | 78.8 | 退化为普通光流模型 |
| - Sensory memory | 8.64/4.55/14.60 | 88.2 | 短期动态建模重要 |
| - Query projector | 6.48/3.88/9.82 | 88.3 | 轻微影响 |

Splatting 类型对比：

| Splatting | EPE↓ | 说明 |
|-----------|------|------|
| **Linear** | **6.42** | 最优 |
| Softmax | 7.04 | 次优 |
| Summation | 7.17 | - |
| Average | 7.34 | 最差 |

Warm-Start 消融：

| 配置 | EPE↓ | 说明 |
|------|------|------|
| Full (warm-start) | 6.42 | - |
| - Hidden state | 7.94 | 隐藏状态继承提升 23.7% |
| - Flow init | 6.49 | 光流初始化略有帮助 |

训练视频长度：7帧→10帧→24帧，EPE 从 7.14→6.42→4.77，长训练视频持续改善。

### 关键发现

- **记忆驱动的设计是核心**：去除记忆库后 EPE 从 6.42 飙升到 38.98，说明记忆是长程跟踪的关键。
- **融合层不可或缺**：没有融合层模型直接 NaN——splatting 的空洞伪影问题严重。
- **参数极其高效**：仅 8.7M 参数，比 Online TAPIR (29.3M)、DOT (56.5M) 小 3-6 倍。
- **速度领先**：512×512 视频上 12.4 FPS（H100），比所有稀疏跟踪方法都快，且快于 DOT。
- **因果处理**：完全基于过去帧，不依赖未来帧，适合实时在线部署。

## 亮点与洞察

- **拆解长程传播为两步**："精确 splatting 到近期 + 近期到当前的注意力"比"直接传播到当前"高效且准确得多。
- **统一架构**：使用标准 RAFT 解码器，光流和点跟踪共享同一架构，未来光流架构的改进可直接受益。
- **感知记忆的巧妙设计**：用额外 GRU 建模短期运动趋势，而非将所有帧放入窗口处理，大幅降低计算开销。
- **可见性掩码的双重作用**：既用于评估跟踪质量，又用于 splatting 的遮挡处理。

## 局限与展望

- 记忆库长度固定为 $L=3$，更长或自适应的记忆可能进一步提升长视频性能。
- 仅支持"第一帧跟踪所有点"，不支持中间帧查询。
- 未利用自监督真实视频微调（如 BootsTAP），引入后性能可能进一步提升。
- 依赖 RAFT 架构，更先进的光流解码器（如 SKFlow）可能带来额外增益。

## 相关工作与启发

- MemFlow 的短期记忆启发了本文，但 MemFlow 仅建模相邻帧运动，本文扩展到长程。
- VOS（视频目标分割）的记忆机制对像素级跟踪有参考价值，但 VOS 关注物体级别。
- Splatting 的可见性引导思路可推广到其他前向 warping 场景。

## 评分

- 新颖性: ⭐⭐⭐⭐ 流式记忆设计新颖，两步传播思路精巧
- 实验充分度: ⭐⭐⭐⭐⭐ CVO+TAP-Vid+RoboTAP 全覆盖，详细消融
- 写作质量: ⭐⭐⭐⭐ 问题清晰，对比全面
- 价值: ⭐⭐⭐⭐⭐ 10× 小参数量+2× 快速度+SOTA 精度，工程和学术价值兼具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] AllTracker: Efficient Dense Point Tracking at High Resolution](alltracker_efficient_dense_point_tracking_at_high_resolution.md)
- [\[ICCV 2025\] VideoLLaMB: Long Streaming Video Understanding with Recurrent Memory Bridges](videollamb_long_streaming_video_understanding_with_recurrent_memory_bridges.md)
- [\[ICCV 2025\] Hierarchical Event Memory for Accurate and Low-latency Online Video Temporal Grounding](hierarchical_event_memory_for_accurate_and_low-latency_online_video_temporal_gro.md)
- [\[ICCV 2025\] ResidualViT for Efficient Temporally Dense Video Encoding](residualvit_for_efficient_temporally_dense_video_encoding.md)
- [\[NeurIPS 2025\] LiveStar: Live Streaming Assistant for Real-World Online Video Understanding](../../NeurIPS2025/video_understanding/livestar_live_streaming_assistant_for_real-world_online_video_understanding.md)

</div>

<!-- RELATED:END -->
