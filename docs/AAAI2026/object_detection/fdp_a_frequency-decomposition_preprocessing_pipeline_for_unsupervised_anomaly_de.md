---
title: >-
  [论文解读] FDP: A Frequency-Decomposition Preprocessing Pipeline for Unsupervised Anomaly Detection in Brain MRI
description: >-
  [AAAI 2026][目标检测][无监督异常检测] 首次系统分析脑 MRI 异常的频域特征，发现病变主要集中在低频分量中，据此提出**频率分解预处理（FDP）**框架，通过可学习先验上下文库重建低频信号来抑制病变同时保留解剖结构，作为即插即用模块可一致提升多种 UAD 基线的检测性能（LDM 上 DICE 提升 17.63%）。
tags:
  - "AAAI 2026"
  - "目标检测"
  - "无监督异常检测"
  - "脑MRI"
  - "频域分析"
  - "频率分解"
  - "扩散模型"
---

# FDP: A Frequency-Decomposition Preprocessing Pipeline for Unsupervised Anomaly Detection in Brain MRI

**会议**: AAAI 2026  
**arXiv**: [2511.12899](https://arxiv.org/abs/2511.12899)  
**代码**: [github](https://github.com/ls1rius/MRI_FDP)  
**领域**: 医学图像  
**关键词**: 无监督异常检测, 脑MRI, 频域分析, 频率分解, 扩散模型

## 一句话总结

首次系统分析脑 MRI 异常的频域特征，发现病变主要集中在低频分量中，据此提出**频率分解预处理（FDP）**框架，通过可学习先验上下文库重建低频信号来抑制病变同时保留解剖结构，作为即插即用模块可一致提升多种 UAD 基线的检测性能（LDM 上 DICE 提升 17.63%）。

## 研究背景与动机

### 脑 MRI 无监督异常检测的困境

由于脑部解剖多样性高、标注数据稀缺，监督式异常检测面临瓶颈。现有无监督异常检测（UAD）方法的标准范式是：
1. 在健康 MRI 上训练生成模型（VAE/DDPM/LDM 等）学习正常解剖的表示
2. 推理时输入可能含异常的扫描，模型尝试重建为"正常"版本
3. 通过原始与重建图的逐像素残差检测异常

关键问题在于：训练时使用的**人工合成噪声**（如随机掩码、高斯噪声）缺乏真实临床病变的**生物物理保真度和形态复杂性**，限制了模型对真实病变的泛化。

### 频域视角的启发

MRI 本身就是在频域（k-space）中采集的——每个空间频率分量编码了不同的结构信息。然而现有 UAD 方法几乎完全在空间域操作，忽略了频域分析的诊断潜力。本文首次从频域视角系统分析了 MRI 异常的特征。

### 两个关键发现

**病变主要集中在低频**：对病变 MRI 应用不同阈值的高通滤波后计算与 ground truth 的 DICE，发现当阈值 $m$ 达到 0.2 时 DICE 即降至 < 0.1，说明病变信号主要由低频分量携带

**健康扫描的低频具有一致性**：正常 MRI 在低频区域（$m \leq 0.1$）的实部信号模式高度一致，而病变扫描在同一频段显示出明显更大的离散度；两组在高频区域（$m > 0.1$）则具有可比的信号变异性

## 方法详解

### 整体框架

FDP 由两个主要模块组成，作为生成模型的前置预处理管道：

**训练阶段**：健康 MRI → FFT 变换到频域 → 高通滤波分离为高频 $f_h$ 和低频 $f_l$ → FRM 用先验上下文库重建低频 → 合并后 IFFT 回空间域 → 送入生成模型训练；同时高频图像 $I_h$ 作为辅助结构先验（HFSup）

**推理阶段**：含病变的 MRI → 同样的频率分解 → FRM 重建低频（此时低频中的病变信号被先验上下文替代） → 合并高频后送入生成模型 → 产生更干净的"健康"重建

### 关键设计

#### 1. **频率分解（Frequency Decomposition）**

使用理想高通滤波器将 MRI 图像的频域表示分为高频和低频：

$$H_{hp}(u,v) = \begin{cases} 0 & \text{if } D(u,v) \leq \mathfrak{D}_0 \\ 1 & \text{if } D(u,v) > \mathfrak{D}_0 \end{cases}$$

其中 $D(u,v) = \sqrt{(u-H/2)^2 + (v-W/2)^2}$，$\mathfrak{D}_0 = \min(m*H, m*W)$，$m$ 为高通滤波阈值。

低频 $f_l$ 包含全局解剖结构（也包含病变信号），高频 $f_h$ 包含细节、边缘和纹理（几乎不含病变信息）。

设计动机：既然病变主要在低频，那么只需重建低频就能在消除病变的同时保留解剖细节。

#### 2. **频率重建模块（FRM）**

核心思想：利用从健康 MRI 训练集学到的可学习先验上下文库 $\boldsymbol{P} = [p_1, p_2, \dots, p_k]$ 来重建低频信号。

- 先验上下文初始化：使用 k-means++ 对训练集低频信号聚类
- 重建过程使用注意力检索机制：

$$\hat{f_l} = \text{ATTN}(f_l, P, P)$$
$$\hat{f} = \text{MERGE}(\hat{f_l}, f_h)$$
$$\hat{I} = \text{IDFT}(\hat{f})$$

- 使用 L1 损失监督低频重建：$L1(\hat{f_l}, f_l)$

设计动机：通过 PCA、t-SNE 和最大似然估计验证，健康 MRI 的低频信号具有低方差且近似位于低维流形上，因此可以用先验上下文库的线性组合来良好近似。推理时，含病变的低频会被"拉回"到健康分布上。

#### 3. **高频补充（HFSup）**

将高频信号变换回空间域得到 $I_h$，作为辅助结构先验输入生成模型，增强解剖结构保留和边缘锐化。

设计动机：单独使用高频信息不足以完成 MRI 重建（缺少全局结构），但作为补充特征可以增强基线模型的结构保持能力。

### 损失函数 / 训练策略

- FRM 使用 L1 损失训练低频重建
- 生成模型（如 LDM、VAE）保持原有训练策略不变
- Adam 优化器，学习率 2e-5，batch size 32，800 epochs
- 4 张 NVIDIA V100 GPU
- 先验上下文数量默认 128，$m$ 默认 0.10

## 实验关键数据

### 主实验

在 BraTS20 数据集（T2 加权）上的结果：

| 模型 | DICE | AUPRC | AUROC |
|------|------|-------|-------|
| VAE | 34.90 | 29.95 | 94.46 |
| **FDP + VAE** | **46.32 (+11.42)** | **41.32 (+11.37)** | 92.16 |
| LDM | 35.02 | 30.75 | 91.62 |
| **FDP + LDM** | **52.66 (+17.63)** | **51.67 (+20.92)** | 93.12 |
| AnoDDPM | 36.19 | 32.01 | 91.37 |
| **FDP + AnoDDPM** | **48.24 (+12.05)** | **47.56 (+15.55)** | 92.97 |
| DAE (coarse) | 56.87 | 43.23 | 95.71 |
| **FDP + DAE (coarse)** | **63.03 (+6.16)** | **61.41 (+18.18)** | 93.95 |
| pDDPM | 46.15 | 45.67 | 92.01 |
| **FDP + pDDPM** | **54.09 (+8.04)** | **51.03 (+5.36)** | 93.72 |

跨数据集泛化验证（LDM 基线）：

| 数据集 | LDM DICE | LDM+FDP DICE | 提升 |
|--------|---------|-------------|------|
| BraTS21 | 29.53 | 45.06 | +15.53 |
| MSLUB | 8.35 | 13.06 | +4.71 |
| MSSEG-2 | 20.15 | 34.63 | +14.48 |

### 消融实验

| FRM | HFSup | DICE | AUPRC | AUROC |
|-----|-------|------|-------|-------|
| ✗ | ✗ | 35.02 | 30.75 | 91.62 |
| ✗ | ✓ | 42.18 | 40.55 | 92.25 |
| ✓ | ✗ | 50.00 | 45.86 | 92.97 |
| ✓ | ✓ | **52.66** | **51.67** | **93.12** |

$m_{\text{FRM}}$ 敏感性分析：

| $m_{\text{FRM}}$ | DICE | AUPRC | AUROC |
|-----------------|------|-------|-------|
| 0.01 | 39.45 | 37.13 | 90.54 |
| 0.05 | 50.00 | 44.86 | 92.97 |
| **0.10** | **52.66** | **51.67** | **93.12** |
| 0.15 | 45.24 | 43.56 | 91.78 |
| 0.20 | 43.89 | 41.87 | 93.08 |

### 关键发现

1. **FRM 是核心贡献**：单独使用 FRM 即获得 +15.0 DICE 提升（35.02 → 50.00），HFSup 再补充 +2.66
2. **一致性提升**：所有 7 个基线方法集成 FDP 后 DICE 至少提升 6.16%，AUPRC 至少提升 5.36%
3. **$m=0.10$ 是最优阈值**：太小（0.01）无法有效去除病变，太大（>0.15）导致低频过于分散难以用先验重建
4. **AUROC 改善有限**：个别方法甚至轻微下降，作者归因于预测区域略小于 GT（精度高、召回略低）

## 亮点与洞察

1. **首次系统性频域分析**：用实验验证了"MRI 病变主要在低频"这一直觉，并量化了这一关系
2. **即插即用设计**：FDP 是预处理模块，不需要修改下游生成模型结构，直接作为"频域去病变"前处理
3. **先验上下文库的巧妙设计**：利用健康低频的低维流形性质，通过注意力检索实现低频重建
4. **频域与空间域的互补利用**：低频重建 + 高频保留的策略既消除病变又保持结构

## 局限与展望

1. **假设病变主要在低频**：某些细微纹理型异常（如脱髓鞘早期变化）可能部分出现在中频段
2. **理想高通滤波器的阶梯效应**：可考虑使用高斯或 Butterworth 滤波器以减少振铃效应
3. **仅验证 T2 加权**：虽然也在 FLAIR/T1 数据集上做了泛化测试，但主要结论基于 T2
4. **先验上下文数量固定**：128 个先验上下文是否适用于更大规模或不同协议的数据集有待验证
5. **缺乏对小病灶的分析**：残差图中预测区域偏小可能影响小病灶的检出率

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次从频域视角切入 MRI UAD 问题，观察新颖且方法直觉
- 实验充分度: ⭐⭐⭐⭐⭐ — 7 个基线、4 个数据集、详细消融和超参分析
- 写作质量: ⭐⭐⭐⭐ — 分析清晰，图表丰富
- 价值: ⭐⭐⭐⭐ — 提供了一个通用的预处理提升方案，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] RcAE: Recursive Reconstruction Framework for Unsupervised Industrial Anomaly Detection](rcae_recursive_reconstruction_framework_for_unsupervised_industrial_anomaly_dete.md)
- [\[AAAI 2026\] Towards Multiple Missing Values-Resistant Unsupervised Graph Anomaly Detection](towards_multiple_missing_values-resistant_unsupervised_graph_anomaly_detection.md)
- [\[AAAI 2026\] AquaSentinel: Next-Generation AI System Integrating Sensor Networks for Urban Underground Water Pipeline Anomaly Detection via Collaborative MoE-LLM Agent Architecture](aquasentinel_next-generation_ai_system_integrating_sensor_ne.md)
- [\[CVPR 2026\] GPFlow: Gaussian Prototype Probability Flow for Unsupervised Multi-Modal Anomaly Detection](../../CVPR2026/object_detection/gpflow_gaussian_prototype_probability_flow_for_unsupervised_multi-modal_anomaly_.md)
- [\[CVPR 2025\] TailedCore: Few-Shot Sampling for Unsupervised Long-Tail Noisy Anomaly Detection](../../CVPR2025/object_detection/tailedcore_few-shot_sampling_for_unsupervised_long-tail_noisy_anomaly_detection.md)

</div>

<!-- RELATED:END -->
