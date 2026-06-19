---
title: >-
  [论文解读] Uncovering Zero-Shot Generalization Gaps in Time-Series Foundation Models Using Real-World Videos
description: >-
  [AAAI 2026][视频理解][时间序列基础模型] 提出从真实视频中通过光流提取时间序列数据的管线，构建了 REAL-V-TSFM 数据集（6130 条序列），揭示了当前时间序列基础模型（Chronos、TimesFM 等）在面对真实物理动态时的零样本泛化能力不足。 问题定义 时间序列基础模型（TSFMs）旨在像 NLP…
tags:
  - "AAAI 2026"
  - "视频理解"
  - "时间序列基础模型"
  - "零样本泛化"
  - "光流"
  - "视频数据"
  - "benchmark"
---

# Uncovering Zero-Shot Generalization Gaps in Time-Series Foundation Models Using Real-World Videos

**会议**: AAAI 2026  
**arXiv**: [2509.26347](https://arxiv.org/abs/2509.26347)  
**代码**: [github.com/DobricLilujun/benchmarking_nature_tsfm](https://github.com/DobricLilujun/benchmarking_nature_tsfm)  
**领域**: 视频理解  
**关键词**: 时间序列基础模型, 零样本泛化, 光流, 视频数据, benchmark

## 一句话总结

提出从真实视频中通过光流提取时间序列数据的管线，构建了 REAL-V-TSFM 数据集（6130 条序列），揭示了当前时间序列基础模型（Chronos、TimesFM 等）在面对真实物理动态时的零样本泛化能力不足。

## 研究背景与动机

### 问题定义

时间序列基础模型（TSFMs）旨在像 NLP 中的 BERT/GPT 一样，通过大规模预训练学习通用时序模式，实现跨域零样本预测。然而，与 NLP 社区经过海量用户和研究者验证的泛化性不同，TSFMs 的泛化能力由于**数据集多样性有限**和**用户基数较小**而远未得到充分验证。

### 已有方法的不足

**训练数据过度依赖合成增强**：例如 Chronos 使用 KernelSynth 和 TSMixup 生成合成训练数据，这些合成数据能否覆盖真实世界的时序动态存疑

**评估数据集多样性不足**：现有 benchmark 多来自金融、能源、交通等传统领域，分布单一。M4 数据集仅 5% 的序列是平稳的，多样性有限

**传感器数据和股价数据已被广泛研究**：缺乏来自全新来源的时间序列数据来测试模型的真正泛化能力

### 核心动机

视频是现代世界中最丰富的时间序列数据来源之一，但几乎未被用于构建时间序列 benchmark。视频中蕴含丰富的物理时序动态——人的摆动、动物的运动、物体的轨迹等。通过光流方法从视频中提取像素轨迹，可以获得反映真实物理运动规律的时间序列，为评估 TSFMs 的泛化能力提供全新视角。

**核心研究问题**：当前的 TSFMs 到底有多通用？它们能否预测从日常真实事件中提取的数据？

## 方法详解

### 整体框架

提出一种从视频中提取时间序列的完整管线，包含 6 个步骤：

1. **视频选择**：从 LaSOT 数据集选取长序列视频（保证有明确主体）
2. **帧提取**：逐帧提取图像
3. **前景检测**：使用 MOG2（混合高斯模型）分离前景/背景
4. **角点检测**：使用 Shi-Tomari 算法在前景主体上检测角点
5. **光流跟踪 + 一致性检查**：使用金字塔 Lucas-Kanade 光流跟踪角点，前向-后向一致性检查过滤不可靠轨迹
6. **后处理**：线性插值标准化长度，保留 5 条最低相关性的轨迹，x/y 坐标分别作为独立时间序列

### 关键设计

#### 1. **前向-后向一致性检查**：过滤不可靠的光流跟踪

**功能**：对每个跟踪点，先前向跟踪再后向跟踪，检查是否回到原始位置。

**核心公式**：

$$e_{fb}(\mathbf{p}_0) = \|\mathbf{p}_0 - (f_{backward} \circ f_{forward})(\mathbf{p}_0)\|_2$$

如果 $e_{fb}(\mathbf{p}_0) < \epsilon$，则跟踪有效。

**设计动机**：光流跟踪存在大量错误（目标丢失、跨帧误识别），前向-后向一致性检查是一种简单有效的质量控制手段。实验中设置了较宽松的阈值（$FB\_ERR\_THRESH=50.0$）来保留更多跟踪点并延长时间序列长度。

#### 2. **多样性保持策略**：选择最低相关性的轨迹

**功能**：计算同一视频中所有轨迹的互相关，保留 5 条相关性最低的。

**设计动机**：主体通常映射到 1-2 个角点，背景角点（由相机运动引入）与主体角点的运动模式不同。保留相关性最低的轨迹可以最大化信息多样性，同时抑制噪声。

#### 3. **数据集 REAL-V-TSFM 的特性**

- **规模**：6,130 条时间序列，609 个不同物体
- **长度**：平均 2,043 个时间步，变异系数 0.516
- **类别多样性**：涵盖飞机、船、猫等多种物体
- **平稳性**：44% 的序列是平稳的（vs M4 仅 5%）
- **信息熵**：3.88 bits（vs M4 的 4.17 bits）
- **PCA 分布**：与 M4 相比分布更均匀，说明覆盖了更多样的时序模式

### 评估设置

- **窗口化**：统一将序列切分为 500 时间步的窗口（450 步作为上下文，50 步作为预测）
- **滑动窗口**：长序列以 500 步步长滑动切分
- **短序列**：线性插值至 500 步

**评估指标**：MAPE、sMAPE、Agg. Relative WQL（加权分位数损失）、Agg. Relative MASE

## 实验关键数据

### 主实验

| 模型 | 数据集 | MAPE↓ | sMAPE↓ | Agg. Rel. WQL↓ | Agg. Rel. MASE↓ |
|------|--------|-------|--------|---------------|-----------------|
| chronos-bolt-base | REAL-V-TSFM | 7.32±17.06 | 6.57±9.93 | 0.93±0.90 | 0.67±0.63 |
| chronos-bolt-base | M4-Weekly | 5.72±3.83 | 5.70±3.83 | 0.79±0.87 | 0.50±0.49 |
| chronos-bolt-base | M4-Daily | 4.93±3.82 | 5.03±3.22 | 1.00±0.85 | 0.63±0.59 |
| chronos-t5-large | REAL-V-TSFM | 9.32±18.46 | 8.40±9.69 | **5.45±31.23** | **5.58±34.82** |
| chronos-t5-large | M4-Daily | 7.11±7.02 | 7.18±4.65 | 1.56±1.26 | 0.98±0.88 |
| timesfm-2.0-500m | REAL-V-TSFM | 6.97±16.63 | 6.24±9.17 | 0.91±1.02 | 0.64±0.65 |
| timesfm-2.0-500m | M4-Daily | **1.9±1.08** | **2.03±2.55** | **0.39±0.22** | **0.23±0.25** |
| LinearRegression | REAL-V-TSFM | 15.52±28.44 | 14.28±20.21 | 1.00 | 1.00 |

### 消融实验（模型规模效应）

| 模型 | REAL-V-TSFM MAPE↓ | REAL-V-TSFM WQL↓ | 参数量 |
|------|-------------------|------------------|--------|
| chronos-bolt-tiny | 7.43±17.40 | 0.92±0.88 | ~7M |
| chronos-bolt-mini | 7.37±17.22 | 0.92±0.90 | ~21M |
| chronos-bolt-small | 7.50±18.80 | 0.92±0.88 | ~48M |
| chronos-bolt-base | 7.32±17.06 | 0.93±0.90 | ~205M |
| chronos-t5-tiny | 10.40±20.13 | 5.40±30.04 | ~8M |
| chronos-t5-large | 9.32±18.46 | 5.45±31.23 | ~709M |

### 关键发现

1. **REAL-V-TSFM 确实更具挑战性**：几乎所有模型在该数据集上的性能都排名最差或倒数第二
2. **chronos-t5-large 的分布捕捉能力极差**：Agg. Relative WQL 在 REAL-V-TSFM 上高达 5.45，而在其他数据集上仅约 1.0，说明模型完全无法捕捉反映真实物理运动的预测分布
3. **timesfm-2.0 泛化性更好**：decoder-only 架构在该数据集上表现相对更稳定，但在 M4 上的巨大优势在 REAL-V-TSFM 上无法复现
4. **Scaling law 在 TSFMs 中不明显**：从 7M 到 709M 参数，MAPE 的改善微乎其微，甚至 tiny 模型有时不逊于 large
5. **bolt 版本显著优于 t5 版本**：架构改进比参数量的提升更有效

## 亮点与洞察

1. **数据获取范式的创新**：从视频中提取时间序列是一个简单但有启发性的想法。视频是互联网上最丰富的数据来源之一，这个管线有潜力大规模扩展时序数据的多样性
2. **揭示了 TSFMs 的关键弱点**：合成数据增强（KernelSynth、TSMixup）生成的数据分布无法覆盖真实物理动态，这对整个 TSFM 社区是一个重要警示
3. **Scaling law 的质疑**：在 TSFMs 中，扩大模型规模并不能显著提升泛化性能，这与 LLM 社区的主流认知形成对比
4. **数据集分布分析**：通过 PCA 投影对比 REAL-V-TSFM 和 M4 的分布差异，直观展示了现有 benchmark 的多样性不足

## 局限与展望

1. **数据集规模偏小**：6,130 条序列相比常用 benchmark（M4 有 10万+序列）规模有限
2. **仅使用 Lucas-Kanade 光流**：更先进的光流方法（如 RAFT）可能生成更高质量的轨迹
3. **只评估了零样本预测**：未探索少样本微调场景（论文已提及）
4. **光流提取的噪声不可避免**：前向-后向检查减少了错误但不能完全消除
5. **任务范围有限**：仅评估预测任务，未涉及异常检测、分类等其他时序任务
6. **"3D视觉"分类存疑**：该论文更贴近时间序列分析/基础模型评估领域

## 相关工作与启发

- Chronos 是当前最有影响力的 TSFM，其在 42 个数据集上的评估看似全面，但本文的实验表明其泛化能力仍有显著缺口
- GIFT-EVAL benchmark 提供了标准化的评估框架，本文的评估基于此
- LaSOT 数据集作为视频源提供了丰富的长序列目标跟踪视频

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 从视频提取时序数据评估 TSFM 是新颖的角度，但技术贡献（光流提取管线）相对简单
- **实验充分度**: ⭐⭐⭐ — 对比了 3 个主要模型 + 1 个基线，但基础模型种类和评估任务有限
- **写作质量**: ⭐⭐⭐⭐ — 问题动机清晰，分析有深度
- **价值**: ⭐⭐⭐⭐ — 为 TSFM 社区提供了重要的泛化盲区警示，数据集和管线开源有推广价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection](../../CVPR2026/video_understanding/no_need_for_real_anomaly_mllm_empowered_zero-shot_video_anomaly_detection.md)
- [\[AAAI 2026\] UVLM: Benchmarking Video Language Model for Underwater World Understanding](uvlm_benchmarking_video_language_model_for_underwater_world_understanding.md)
- [\[CVPR 2026\] UniVBench: Towards Unified Evaluation for Video Foundation Models](../../CVPR2026/video_understanding/univbench_towards_unified_evaluation_for_video_foundation_models.md)
- [\[CVPR 2026\] Real-World Point Tracking with Verifier-Guided Pseudo-Labeling](../../CVPR2026/video_understanding/realworld_point_tracking_with_verifierguided_pseud.md)
- [\[ICML 2026\] ProAct-VL: A Proactive VideoLLM for Real-Time AI Companions](../../ICML2026/video_understanding/proact-vl_a_proactive_videollm_for_real-time_ai_companions.md)

</div>

<!-- RELATED:END -->
