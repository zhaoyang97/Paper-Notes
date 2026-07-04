---
title: >-
  [论文解读] scSplit: Bringing Severity Cognizance to Image Decomposition in Fluorescence Microscopy
description: >-
  [NEURIPS2025][图像恢复][fluorescence microscopy] 提出 scSplit，通过引入混合比例感知的归一化模块（SCIN）和回归网络（Reg），使基于 InDI 的迭代图像分解方法能够感知荧光显微镜图像中两种结构叠加的严重程度，在5个公开数据集上统一解决图像分离和渗透去除两个任务。
tags:
  - "NEURIPS2025"
  - "图像恢复"
  - "fluorescence microscopy"
  - "image decomposition"
  - "severity cognizance"
  - "mixing ratio"
  - "bleedthrough removal"
---

# scSplit: Bringing Severity Cognizance to Image Decomposition in Fluorescence Microscopy

**会议**: NEURIPS2025  
**arXiv**: [2503.22983](https://arxiv.org/abs/2503.22983)  
**代码**: [juglab/scSplit](https://github.com/juglab/scSplit/)  
**领域**: 图像复原  
**关键词**: fluorescence microscopy, image decomposition, severity cognizance, mixing ratio, bleedthrough removal  

## 一句话总结
提出 scSplit，通过引入混合比例感知的归一化模块（SCIN）和回归网络（Reg），使基于 InDI 的迭代图像分解方法能够感知荧光显微镜图像中两种结构叠加的严重程度，在5个公开数据集上统一解决图像分离和渗透去除两个任务。

## 研究背景与动机
- **领域背景**: 荧光显微镜是生命科学的关键成像技术，利用不同荧光标记物在不同通道分别成像不同细胞结构，但受限于可同时成像的结构数量上限
- **现有方案与不足**: 计算多路复用技术（如 μSplit、denoiSplit、MicroSplit）可以将多种结构叠加到单通道再用深度学习分解，但它们假设输入是两种结构的简单平均（t=0.5），忽视了叠加强度比（mixing ratio）的变化
- **关键痛点**: 实际显微镜采集中，样本属性、标记密度、显微镜配置不同导致叠加比例大幅变化。当测试图像的叠加特性与训练时不同时，现有方法性能显著下降
- **渗透问题**: Bleedthrough（渗透）是一个相关问题——成像某结构的专用通道中，其他结构因光学滤波不精确而"渗入"，本质上也是不同混合比例的叠加分解问题
- **为何难以统一**: 只有有效解决叠加比例变化的挑战，才能用单一网络同时处理图像分离（t≈0.5）和渗透去除（t接近0或1）
- **方法选型依据**: InDI 将退化建模为干净图像与退化图像的线性混合，其归纳偏置与荧光显微镜中的线性叠加天然吻合，是理想的基础框架

## 方法详解

### 整体框架
给定两种结构 c₀∈C₀ 和 c₁∈C₁，叠加输入定义为 cₜ = (1−t)c₀ + tc₁，其中 t∈[0,1] 为混合比例。scSplit 包含三个关键组件：(1) 严重程度感知归一化模块 SCIN，(2) 两个生成网络 Gen₀ 和 Gen₁ 分别预测两个通道，(3) 回归网络 Reg 预测混合比例 t。训练时随机采样 t，推理时由 Reg 估计 t 并通过聚合进一步优化。

### 模块一：严重程度感知输入归一化（SCIN）
- **功能**: 对不同混合比例 t 的叠加输入 cₜ 进行 t-specific 的均值-方差归一化，确保所有 t 下 E[μ(t)]=0, E[σ²(t)]=1
- **核心思路**: 将区间[0,1]划分为 n=100 个等距子区间，对每个子区间预计算叠加 patch 的均值和标准差，存储为查找表 D[i]=(μᵢ, σᵢ)。训练时按 t 所在区间从查找表中取归一化参数；推理时，因为保证了所有 t 下 E[μ(t)]=0 和 E[σ²(t)]=1，直接用测试图像集的均值和标准差归一化即可
- **设计动机**: 理论推导表明当 C₀ 和 C₁ 分别标准化后，叠加图像 cₜ 的期望方差 E[σ²(t)] = t² + (1−t)² + 2t(1−t)Cov(p₀,p₁) 是 t 的函数。如果不做 t-specific 归一化，不同 t 的输入对网络呈现不同的统计特性，推理时在未知 t 的情况下无法正确归一化，导致分布不匹配

### 模块二：回归网络 Reg
- **功能**: 给定归一化后的叠加输入 xₜ，预测混合比例 t 的估计值
- **核心思路**: Reg 与生成网络共享同一 SCIN 归一化模块，确保输入统计特性一致。推理时利用领域知识——同一显微镜会话中所有图像共享相同的激光功率和荧光团类型，因此混合比例相同——对会话中所有图像的 t 估计值取算术平均（聚合），提高预测精度
- **设计动机**: 准确的 t 估计需要正确的归一化，而归一化又依赖 t，形成循环依赖。SCIN 打破了这个循环。聚合操作利用了荧光显微镜的先验知识，单张图像的 t 估计可能有噪声，但对批次图像取平均可显著降低方差

### 模块三：生成网络 Gen₀ 和 Gen₁
- **功能**: 分别预测通道 c₀ 和 c₁ 的归一化版本。输入为归一化叠加图像 xₜ = cₜᴺᵒʳᵐ + tεn（n~N(0,I), ε=0.01）加上混合比例信息
- **核心思路**: 继承 InDI 的训练方法，将混合比例 t 作为额外条件输入。对通道 i 的预测为 ĉᵢᴺᵒʳᵐ = Genᵢ(xₜ, tδᵢ + (1−t)δ₁₋ᵢ)，其中分离通道 c₀ 的严重程度为 t，分离 c₁ 的严重程度为 1−t
- **设计动机**: InDI 框架天然支持线性混合退化建模，通过将 t 作为条件输入，网络能够根据不同叠加严重程度调整分解策略

### 损失函数与训练策略
- 训练时从分布 p(t) = 1/(1+a)·U[0,1] + a/(1+a)·δ₀.₅（a=1）采样混合比例 t，相比 InDI 给端点更多权重，这里给 t=0.5 更多权重，因为图像分离任务的输入通常包含两种结构
- 推理时使用单步推理（one-step inference）以最小化失真
- 高斯噪声扰动 ε=0.01 用于正则化

## 实验关键数据

### 表1: 5个数据集上的定量评估（PSNR，按叠加严重程度分三组）

| 方法 | Hagen-Dom | Hagen-Bal | Hagen-Weak | HTLIF24-Dom | HTLIF24-Bal | HTLIF24-Weak |
|------|-----------|-----------|------------|-------------|-------------|--------------|
| U-Net | 31.8 | 28.2 | 22.0 | 45.9 | 44.6 | 36.0 |
| μSplit_D | 33.1 | 32.4 | 23.4 | 45.9 | 44.9 | 36.4 |
| InDI | 33.1 | 32.1 | 24.2 | 45.2 | 43.9 | 37.6 |
| scSplit₀.₅ | 34.1 | 33.7 | 25.0 | 45.9 | 45.1 | 37.4 |
| **scSplit** | **40.9** | **33.9** | **29.3** | **51.8** | **45.5** | **39.9** |

### 表2: BioSR 与 HTT24 数据集上的补充结果（PSNR）

| 方法 | BioSR-Dom | BioSR-Bal | BioSR-Weak | HTT24-Dom | HTT24-Bal | HTT24-Weak |
|------|-----------|-----------|------------|-----------|-----------|------------|
| MicroSplit | 38.5 | 34.3 | 26.6 | 36.9 | 36.6 | 30.3 |
| InDI | 35.9 | 33.4 | 26.3 | 37.6 | 36.5 | 30.5 |
| scSplit₀.₅ | 37.3 | 35.0 | 26.4 | 38.1 | 38.6 | 31.4 |
| **scSplit** | **40.1** | **35.3** | **28.7** | **44.5** | **39.1** | **34.7** |

### 关键发现
- **Dominant 区间优势巨大**: scSplit 在 Dominant 区间（w=0.7~0.9，渗透去除场景）较最佳基线 PSNR 提升 2~8 dB，表明严重程度感知对渗透去除至关重要
- **SCIN 贡献显著**: scSplit₀.₅ vs InDI（仅区别在归一化）平均 PSNR 提升 1.2 dB
- **聚合有效**: scSplit vs scSplit₋ₐgg 在所有数据集 PSNR 指标上一致优于后者
- **下游任务验证**: 在 BioSR 分割任务中，scSplit 的 DICE 不相似度最低（如w=0.9时 Ch1: 0.035 vs InDI 0.044）
- **分布偏移鲁棒**: 当训练-测试混合比例差异大时（Table 3），scSplit 以 35.8 dB 大幅超越 U-Net 30.1 dB
- **增强的 μSplit 仍不敌 scSplit**: 给 μSplit_D 添加混合比例增强后性能提升，但 scSplit 仍平均高出 2.4 dB PSNR

## 亮点与洞察
1. **问题定义精准**: 作者清晰识别了叠加严重程度变化这一被忽视的关键问题，并证明它是统一图像分离和渗透去除的核心障碍
2. **归一化模块理论支撑充分**: 从数学上推导了 cₜ 的期望方差与 t 的依赖关系，SCIN 的设计有严格的理论动机而非经验性做法
3. **领域知识的优雅利用**: 聚合模块巧妙利用了"同一采集会话所有图像共享混合比例"的显微镜先验，提升 t 估计精度
4. **全面的消融实验**: 通过 scSplit₀.₅、scSplit₋ₐgg 等变体，清晰量化了每个组件的贡献

## 局限性
- 仅处理两种结构的叠加分解，未扩展到 k>2 的情况
- 线性叠加假设在某些成像条件下可能不完全成立
- 单步推理虽减少失真但可能限制对复杂叠加的处理能力
- PaviaATN 数据集上 Weak 区间的改进相对有限（24.3 vs 21.8 InDI），暗示某些数据特性下方法效果受限
- 回归网络对 t 的估计仍有误差，Figure 3 显示当假设 w 偏差较大时性能明显下降

## 相关工作与启发
- **InDI** (Delbracio & Milanfar, 2023): 将退化建模为线性混合的迭代恢复方法，scSplit 继承其归纳偏置并显著扩展
- **μSplit** (Ashesh et al.): GPU 高效的图像分离元架构，但不感知叠加严重程度
- **MicroSplit**: 量化了叠加变异性的影响但未提出解决方案，scSplit 填补了这一空白
- **denoiSplit**: 结合无监督去噪和有监督图像分离，但同样忽视了混合比例的变化
- **启发**: 对于任何涉及"退化程度未知"的图像恢复问题（如噪声水平、模糊程度），引入退化感知归一化和退化级别预测可能是有效策略
- **更广视角**: 图像分离可视为特殊的图像翻译任务，与反射去除、去雾、去雨等相关但在线性叠加和ground truth可用性方面有本质不同

## 评分
- 新颖性: ⭐⭐⭐⭐ (SCIN 归一化和严重程度感知的组合是针对荧光显微镜的创新贡献)
- 实验充分度: ⭐⭐⭐⭐⭐ (5个数据集、3类场景、多消融实验、下游分割任务验证)
- 写作质量: ⭐⭐⭐⭐ (理论推导清晰，问题动机阐述充分)
- 价值: ⭐⭐⭐⭐ (对荧光显微镜领域有实际应用价值，统一了两个重要任务)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] A Flag Decomposition for Hierarchical Datasets](../../CVPR2025/image_restoration/a_flag_decomposition_for_hierarchical_datasets.md)
- [\[ECCV 2024\] DenoiSplit: A Method for Joint Microscopy Image Splitting and Unsupervised Denoising](../../ECCV2024/image_restoration/denoisplit_a_method_for_joint_microscopy_image_splitting_and_unsupervised_denois.md)
- [\[CVPR 2026\] ReasonX: MLLM-Guided Intrinsic Image Decomposition](../../CVPR2026/image_restoration/reasonx_mllm-guided_intrinsic_image_decomposition.md)
- [\[NeurIPS 2025\] Learning Cocoercive Conservative Denoisers via Helmholtz Decomposition for Poisson Inverse Problems](learning_cocoercive_conservative_denoisers_via_helmholtz_decomposition_for_poiss.md)
- [\[ICCV 2025\] Lightweight and Fast Real-time Image Enhancement via Decomposition of the Spatial-aware Lookup Tables](../../ICCV2025/image_restoration/lightweight_and_fast_real-time_image_enhancement_via_decomposition_of_the_spatia.md)

</div>

<!-- RELATED:END -->
