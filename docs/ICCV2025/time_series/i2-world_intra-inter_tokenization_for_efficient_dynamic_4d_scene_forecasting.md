---
title: >-
  [论文解读] I²-World: Intra-Inter Tokenization for Efficient Dynamic 4D Scene Forecasting
description: >-
  [ICCV 2025][时间序列][4D occupancy forecasting] 提出 I²-World，通过将 3D 场景 tokenization 解耦为帧内（intra-scene）多尺度残差量化和帧间（inter-scene）时序量化两个互补过程，在保持 3D tokenizer 高压缩率的同时获得 4D tokenizer 的时序建模能力，实现高效且高质量的 4D occupancy 预测。
tags:
  - "ICCV 2025"
  - "时间序列"
  - "4D occupancy forecasting"
  - "scene tokenization"
  - "世界模型"
  - "自回归生成"
  - "自动驾驶"
---

# I²-World: Intra-Inter Tokenization for Efficient Dynamic 4D Scene Forecasting

**会议**: ICCV 2025  
**arXiv**: [2507.09144](https://arxiv.org/abs/2507.09144)  
**代码**: [GitHub](https://github.com/lzzzzzm/II-World)  
**领域**: 时间序列  
**关键词**: 4D occupancy forecasting, scene tokenization, 世界模型, 自回归生成, 自动驾驶

## 一句话总结

提出 I²-World，通过将 3D 场景 tokenization 解耦为帧内（intra-scene）多尺度残差量化和帧间（inter-scene）时序量化两个互补过程，在保持 3D tokenizer 高压缩率的同时获得 4D tokenizer 的时序建模能力，实现高效且高质量的 4D occupancy 预测。

## 研究背景与动机

Occupancy-based world model 在自动驾驶中用于预测未来 3D 场景演变，对应对 corner case 至关重要。现有方法面临一个核心矛盾：

**3D 场景 tokenizer**（如 OccWorld、OccLLaMA）：将单帧场景压缩为紧凑 token，重建精度高，但无法建模时序动态，预测能力受限

**4D 场景 tokenizer**（如 OccSora、DOME）：处理时空 token 序列，动态建模能力强，但 token 维度暴增，计算开销极大，难以满足实时性需求

此外，现有方法大多采用 GPT 风格的纯解码器自回归架构或扩散模型，资源消耗高。本文的核心问题是：**如何在紧凑的 token 表示中同时编码空间细节和时序动态？**

## 方法详解

### 整体框架

I²-World 由两个核心组件构成，采用两阶段训练：

- **阶段一**：训练 I²-Scene Tokenizer（VAE pipeline），学习场景压缩与重建
- **阶段二**：冻结 tokenizer，训练 I²-Former 学习动态转换

输入当前 occupancy $O_t \in \mathbb{R}^{H \times W \times Z}$ 和 $G$ 帧历史，tokenizer 输出紧凑连续 token $\hat{B}_t \in \mathbb{R}^{h \times w \times C}$（$h, w \ll H, W$），I²-Former 自回归预测未来 $K$ 帧 token 并解码为 occupancy。

### 关键设计

1. **Intra-Scene Tokenizer（帧内多尺度残差量化）**: 受 RQ-VAE 和 VAR 启发，将编码器输出特征 $B_t$ 迭代量化为 $S$ 个多尺度 token map $\{b_t^s\}_{s=1}^S$。从低分辨率到高分辨率逐级量化，每级与共享 codebook $\mathcal{C} \in \mathbb{R}^{N \times C}$ 匹配后计算残差，下一级在残差上继续量化。核心公式：

    $b_t^s = f_{intp}(B_t, h_s, w_s), \quad \hat{b}_t^{s_{i,j}} = \mathcal{Q}(b_t^{s_{i,j}}, \mathcal{C})$
    $B_t = B_t - f_{intp}(\hat{b}_t^s, h, w), \quad \hat{B}_t = \hat{B}_t + \phi_s(\hat{b}_t^s)$

   其中 $\phi_s$ 是可学习卷积层，用于缓解分辨率缩放的信息损失。设计动机：3D occupancy 天然支持多尺度表示，从粗到细逐级补充空间细节。

2. **Inter-Scene Tokenizer（帧间时序量化）**: 维护历史特征队列 $\{B_{t-g}\}_{g=1}^G$，先用 ego-pose 变换矩阵 $T_{t-g}^t$ 对齐到当前帧坐标系。然后将 Intra-Scene 的残差与对齐后的历史特征相加，进行 $G$ 次时序量化：

    $b_t^{S+g} = B_t + B_{t-g}', \quad \hat{B}_t = \hat{B}_t + \psi_g(\hat{b}_t^{S+g})$

   关键点：与 Intra-Scene 共享同一 codebook $\mathcal{C}$，仅增加 $G+S$ 个轻量卷积层即可同时编码空间和时序信息。

3. **I²-Former（编码器-解码器架构）**: 区别于 GPT 风格纯解码器，采用 Intra-Encoder + Inter-Decoder：

    - **Intra-Encoder**：用 Spatial Self-Attention (SSA) + 多头交叉注意力融合 plan embedding 和场景 token，3 层多尺度设计（每层下采样 2×），输出通过 FPN 聚合。关键输出：回归**变换矩阵** $T_{t+k}^{t+k+1} \in \mathbb{R}^{4 \times 4}$，编码帧间时空变换
    - **Inter-Decoder**：以变换矩阵为条件信号，维护历史 token 队列，用 SSA + 轻量 MLP 时序融合（拼接通道维度 + 单层 MLP）自回归生成下一帧 token

   设计动机：变换矩阵提供比轨迹更精细的空间约束，支持可控生成。

### 损失函数 / 训练策略

**Tokenizer 阶段**：
$$\mathcal{L}_{token} = \mathcal{L}_{focal}(O_t, \hat{O}_t) + \mathcal{L}_{lov}(O_t, \hat{O}_t) + \mathcal{L}_{vq}$$

其中 $\mathcal{L}_{vq}$ 包含 codebook 对齐项和承诺损失（$\beta=1$），仅对 Intra-Scene 部分监督以保证训练稳定性。

**生成阶段**：
$$\mathcal{L}_{gen} = \sum_{k=1}^K w_k \mathcal{L}_{mse}(\hat{B}_{t+k}', \hat{B}_{t+k})$$

变换矩阵分解为平移（L2 loss）和旋转（四元数 cosine loss）分别监督。

## 实验关键数据

### 主实验（Occ3D-nuScenes）

| 方法 | 输入 | mIoU Avg (%) | IoU Avg (%) | FPS |
|------|------|------------|------------|-----|
| OccWorld-O | Occ | 17.14 | 26.63 | 18.0 |
| DOME | Occ | 27.10 | 36.36 | 6.54 |
| UniScene | Occ | 31.76 | 34.84 | - |
| **I²-World-O** | **Occ** | **39.73** | **49.80** | **37.04** |
| DOME-STC | Camera | 14.53 | 23.33 | 2.75 |
| **I²-World-STC** | **Camera** | **18.97** | **28.77** | **4.21** |

- GT occupancy 输入下 mIoU 提升 **25.1%**（39.73 vs 31.76），IoU 提升 **36.9%**
- 训练仅需 **2.9 GB** 显存，推理 **37 FPS**（RTX 4090），远超现有方法

### 消融实验

| 组件 | mIoU (%) | IoU (%) | 说明 |
|------|---------|---------|------|
| Baseline (单尺度) | 66.52 | 61.07 | 无 Inter-Scene |
| + Inter-Scene (无对齐) | 70.37 | 62.18 | 时序建模 +5.7% mIoU |
| + 对齐 (Intra 单尺度) | 77.12 | 64.20 | 对齐后 +15.9% mIoU |
| + 多尺度 Intra + 对齐 | **81.22** | **68.30** | 最优重建质量 |
| 仅 Trans 条件 | 28.74 | 36.44 | 平移信息贡献最大 |
| Trans + Rot + Encoder + MS | **39.73** | **49.80** | 各组件互补 |

### 关键发现

- 帧间对齐（ego-pose alignment）是时序建模的关键，带来 15.9% mIoU 提升
- 平移条件是生成的主要驱动力（+67.8% mIoU vs baseline），旋转单独贡献较小但与平移互补
- 零样本迁移 Occ3D-Waymo 上表现出色，10Hz 下 mIoU 43.73 vs copy-paste 的 28.34

## 亮点与洞察

- **解耦设计精妙**：intra 捕获空间细节、inter 编码时序动态，共享 codebook 保证时空一致性
- **极致高效**：2.9 GB 显存 + 37 FPS，使基于 occupancy 的 world model 首次具备实时部署潜力
- **可控生成**：通过变换矩阵直接操控场景演化（米/弧度精度），支持 corner case 模拟

## 局限与展望

- 变换矩阵在训练集未覆盖的模式（如倒车）上会产生不真实的生成结果
- 连续 token（非离散 VQ token）是否限制了与 LLM 等离散生成范式的集成
- 可探索更长时序（>3s）的预测能力和累积误差缓解

## 相关工作与启发

- **RQ-VAE / VAR** 的多尺度残差量化思想值得在更多 3D 任务中应用
- 变换矩阵作为条件信号优于传统轨迹指导，可启发其他场景生成任务
- 与 iVideoGPT 的双编解码器设计类似，都是通过解耦冗余来提升效率

## 评分

- 新颖性：⭐⭐⭐⭐ — 帧内/帧间解耦 tokenization 设计新颖
- 技术深度：⭐⭐⭐⭐ — 多尺度残差量化 + 时序量化 + 变换矩阵条件生成
- 实验充分度：⭐⭐⭐⭐ — 主实验/消融/泛化/可视化齐全
- 实用价值：⭐⭐⭐⭐⭐ — 极高效率，首个实时 4D occupancy world model

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Universal Spectral Tokenization via Self-Supervised Panchromatic Representation Learning](../../NeurIPS2025/time_series/universal_spectral_tokenization_via_self-supervised_panchromatic_representation_.md)
- [\[ICML 2025\] TQNet: Temporal Query Network for Efficient Multivariate Time Series Forecasting](../../ICML2025/time_series/temporal_query_network_for_efficient_multivariate_time_series_forecasting.md)
- [\[NeurIPS 2025\] Diffusion Transformers as Open-World Spatiotemporal Foundation Models](../../NeurIPS2025/time_series/diffusion_transformers_as_open-world_spatiotemporal_foundation_models.md)
- [\[ICLR 2026\] Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework](../../ICLR2026/time_series/towards_robust_real-world_multivariate_time_series_forecasting_a_unified_framewo.md)
- [\[ICML 2026\] Dynamic-TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series](../../ICML2026/time_series/dynamic_tmoe_a_drift-aware_dynamic_mixture_of_experts_framework_for_non-stationa.md)

</div>

<!-- RELATED:END -->
