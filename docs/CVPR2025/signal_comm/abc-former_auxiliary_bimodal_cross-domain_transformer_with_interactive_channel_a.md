---
title: >-
  [论文解读] ABC-Former: Auxiliary Bimodal Cross-domain Transformer with Interactive Channel Attention
description: >-
  [CVPR 2025][信号/通信][白平衡] 提出 ABC-Former，通过引入 CIELab 色彩空间和 RGB 直方图作为辅助双模态信息，利用跨域 Transformer 和交互通道注意力（ICA）模块实现全局色彩知识的跨模态迁移，在 sRGB 白平衡矫正任务上取得 SOTA 效果；同时扩展为 ABC-FormerM 处理混合光照场景。
tags:
  - "CVPR 2025"
  - "信号/通信"
  - "白平衡"
  - "双模态"
  - "Transformer"
  - "交互通道注意力"
  - "CIELab直方图"
---

# ABC-Former: Auxiliary Bimodal Cross-domain Transformer with Interactive Channel Attention

**会议**: CVPR 2025  
**代码**: [https://github.com/ytpeng-aimlab/ABC-Former](https://github.com/ytpeng-aimlab/ABC-Former)  
**领域**: 图像处理 / 白平衡  
**关键词**: 白平衡, 双模态, 跨域Transformer, 交互通道注意力, CIELab直方图

## 一句话总结
提出 ABC-Former，通过引入 CIELab 色彩空间和 RGB 直方图作为辅助双模态信息，利用跨域 Transformer 和交互通道注意力（ICA）模块实现全局色彩知识的跨模态迁移，在 sRGB 白平衡矫正任务上取得 SOTA 效果；同时扩展为 ABC-FormerM 处理混合光照场景。

## 研究背景与动机

**领域现状**：白平衡（White Balance, WB）矫正是相机图像信号处理（ISP）流水线中的核心环节，目标是消除不准确色温导致的图像色偏，使图像呈现自然中性的颜色。现有白平衡方法主要分两类：(1) 全局色彩调整方法，在相机 ISP 前对 RAW 图像施加全局校正矩阵；(2) 端到端深度学习方法，直接从 sRGB 图像学习白平衡映射。

**现有痛点**：全局调整方法只考虑整体色温，在混合光照场景（如室内灯光 + 窗外日光）中产生局部色偏；端到端模型虽能学习像素级调整，但不显式利用全局色彩分布的先验信息（如 CIELab 直方图反映的色温趋势），导致矫正结果在色彩一致性上不够理想。两类方法各有偏废，缺乏有效的全局-局部信息融合机制。

**核心矛盾**：白平衡矫正需要同时理解全局色温偏移（整张图片整体偏暖/偏冷）和局部色彩变化（不同区域光源不同），但现有架构难以在特征层面有效整合这两种不同尺度的信息。

**本文目标**：设计一个能够充分利用全局色彩统计信息（CIELab + RGB 直方图）和局部像素信息（sRGB 图像）的白平衡矫正网络。

**切入角度**：将白平衡矫正视为跨域信息融合问题——全局色彩直方图是一个域，局部像素特征是另一个域。用 Transformer 的跨注意力机制实现两个域之间的信息交互。

**核心 idea**：引入 CIELab 和 RGB 直方图作为辅助双模态（Auxiliary Bimodal）信息，用跨域 Transformer 在两个模态之间进行特征交互，并通过交互通道注意力动态融合多模态特征。

## 方法详解

### 整体框架
ABC-Former 的输入包含三部分：(1) sRGB 图像，经编码器提取空间特征；(2) CIELab 直方图，反映全局色彩分布；(3) RGB 直方图，提供三通道的色彩统计。三部分分别编码后，通过跨域 Transformer 模块进行特征交互，再由交互通道注意力进行自适应融合，最终解码输出矫正后的图像。

### 关键设计

1. **辅助双模态编码（Auxiliary Bimodal Encoding）**：

    - 功能：从色彩统计中提取全局色温/偏色信息
    - 核心思路：将输入 sRGB 图像转换到 CIELab 色彩空间（L 亮度、a 红绿轴、b 蓝黄轴），计算 a/b 通道的直方图作为色温分布的紧凑表示；同时计算 RGB 三通道直方图。两种直方图分别通过轻量 MLP 编码为向量特征 $\mathbf{h}_{lab} \in \mathbb{R}^{D}$ 和 $\mathbf{h}_{rgb} \in \mathbb{R}^{D}$
    - 设计动机：CIELab 空间的 a/b 通道直接反映色温偏移方向和程度，比 RGB 空间更适合白平衡任务。RGB 直方图补充三通道各自的色偏信息。两种模态提供互补的全局色彩先验

2. **跨域 Transformer（Cross-domain Transformer）**：

    - 功能：在图像空间特征与直方图全局特征之间进行信息交互
    - 核心思路：采用标准的跨注意力架构——以图像空间特征序列为 Query，以直方图特征为 Key/Value，计算注意力权重实现全局色彩信息向每个空间位置的定向注入。反向也进行一次（直方图特征查询图像特征），让全局表示也能感知局部分布。多层堆叠实现渐进式特征对齐
    - 设计动机：全局直方图特征和局部像素特征处于不同的"域"，直接拼接会导致信息泯灭。跨注意力让每个像素位置选择性地从全局统计中提取与自身最相关的色彩调整信号

3. **交互通道注意力（Interactive Channel Attention, ICA）**：

    - 功能：在通道维度自适应融合跨域 Transformer 输出的多模态特征
    - 核心思路：对融合后的多模态特征在通道维度计算注意力权重，动态决定每个通道应更多保留直方图模态的信息还是图像模态的信息。通过全局平均池化 → FC → Sigmoid 生成通道权重 $\alpha \in \mathbb{R}^{C}$，然后加权融合 $\mathbf{F}_{out} = \alpha \odot \mathbf{F}_{img} + (1-\alpha) \odot \mathbf{F}_{hist}$
    - 设计动机：不同通道对应不同层次的色彩特征——低层通道可能更需要全局色温指导，高层通道更依赖局部纹理。ICA 让网络自主学习最优的通道级融合策略

### 损失函数 / 训练策略
训练采用像素级 $\mathcal{L}_1$ 损失和感知损失（Perceptual Loss）的组合，在 Rendered WB 数据集（Set1）上进行。同时提供 ABC-FormerM 变体处理混合光照任务，在 Mixed-Illumination 数据集上训练，使用 Two-stage Distortion-based (TDS) 和 TDS with full correction (TDSFC) 策略。

## 实验关键数据

### 主实验（RenderedWB 数据集）

| 方法 | MSE ↓ | MAE ↓ | $\Delta E_{ab}$ ↓ | PSNR (dB) ↑ |
|------|-------|-------|---------------------|-------------|
| Deep WB (CVPR 2020) | 147.3 | 8.21 | 3.42 | 28.65 |
| Mixed-Ill WB (WACV 2022) | 132.8 | 7.56 | 3.15 | 29.34 |
| AWB-Transformer | 118.4 | 6.89 | 2.87 | 30.12 |
| **ABC-Former (Ours)** | **98.6** | **5.74** | **2.31** | **31.48** |

### 消融实验

| 配置 | MSE ↓ | $\Delta E_{ab}$ ↓ | 说明 |
|------|-------|--------------------|------|
| Full model | 98.6 | 2.31 | 完整 ABC-Former |
| w/o CIELab 直方图 | 112.5 | 2.68 | 仅用 RGB 直方图 |
| w/o RGB 直方图 | 108.3 | 2.54 | 仅用 CIELab 直方图 |
| w/o Cross-domain Transformer | 126.7 | 2.95 | 简单拼接替代跨注意力 |
| w/o ICA | 105.4 | 2.48 | 均匀融合替代自适应通道注意力 |

### 关键发现
- 跨域 Transformer 贡献最大（去掉后 MSE 上升 28.5%），证实跨域信息交互的必要性
- CIELab 和 RGB 双模态直方图提供互补信息，各自去掉后均有明显退化，同时去掉退化更严重
- ICA 带来约 7% 的 MSE 改善，说明自适应通道融合优于均匀融合
- 在混合光照场景（ABC-FormerM 变体），性能提升更显著，因为不同区域光源不同时更需要全局-局部协同
- 已被引用 3 次（截至 2026 年 4 月），包括 WACV 2026 的后续工作

## 亮点与洞察
- **双模态辅助信息设计直观有效**：CIELab 色彩空间天然与人类色觉感知对齐，将其直方图作为全局色温指示器非常自然。与 RGB 直方图互补融合进一步增强鲁棒性
- **跨域 Transformer 的信息桥梁作用**：将全局统计与局部像素视为两个"域"，用跨注意力建立桥梁，这一设计范式可推广到其他需要全局-局部信息融合的任务（如色调映射、曝光校正）
- **扩展到混合光照的通用性**：同一框架稍加修改即可处理更复杂的多光源场景，体现了方法的灵活性

## 局限与展望
- 直方图编码是全局的，在极端局部色偏场景（如小区域强色灯光）中可能信息不足
- 对非常规色温（如极冷/极暖的艺术风格图像）的泛化性有待验证
- 可以探索与相机 RAW 域白平衡方法的结合——先在 RAW 域粗调，再在 sRGB 域用 ABC-Former 精调
- ICA 目前是通道级的，扩展为空间-通道联合注意力可能进一步提升混合光照场景的效果

## 相关工作与启发
- **vs Deep WB (CVPR 2020)**: Deep WB 是首个深度学习白平衡方法，在 ISP 前操作 RAW 图像。ABC-Former 在 sRGB 域工作，利用双模态直方图补偿了 sRGB 空间信息的损失
- **vs Mixed-Ill WB (WACV 2022)**: Mixed-Ill 方法通过空间分割处理混合光照，但分割边界容易产生色彩不连续。ABC-Former 的跨域融合更平滑
- **vs AWB-Transformer**: AWB-Transformer 仅使用图像特征做白平衡，缺乏全局色彩先验。ABC-Former 的辅助直方图提供了关键的全局信息补充

## 评分
- 新颖性: ⭐⭐⭐⭐ 双模态直方图辅助 + 跨域 Transformer 的组合较新颖
- 实验充分度: ⭐⭐⭐⭐ 多数据集验证，消融充分，有混合光照扩展
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，动机有说服力
- 价值: ⭐⭐⭐⭐ 白平衡矫正的实用改进，架构设计可迁移
---
title: >-
  [论文解读] ABC-Former: Auxiliary Bimodal Cross-domain Transformer with Interactive Channel Attention
description: >-
  [CVPR 2025][白平衡][跨域Transformer] 提出辅助双模态跨域Transformer和交互通道注意力用于sRGB图像白平衡矫正
tags:
  - CVPR 2025
  - 白平衡
  - 跨域Transformer
  - 通道注意力
  - 色温矫正
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Breaking the Low-Rank Dilemma of Linear Attention](breaking_the_low-rank_dilemma_of_linear_attention.md)
- [\[ICCV 2025\] Rectifying Magnitude Neglect in Linear Attention](../../ICCV2025/signal_comm/rectifying_magnitude_neglect_in_linear_attention.md)
- [\[ICLR 2026\] Efficient Message-Passing Transformer for Error Correcting Codes](../../ICLR2026/signal_comm/efficient_message-passing_transformer_for_error_correcting_codes.md)
- [\[AAAI 2026\] Balancing Multimodal Domain Generalization via Gradient Modulation and Projection](../../AAAI2026/signal_comm/balancing_multimodal_domain_generalization_via_gradient_modulation_and_projectio.md)
- [\[ICML 2025\] Fourier Position Embedding: Enhancing Attention's Periodic Extension for Length Generalization](../../ICML2025/signal_comm/fourier_position_embedding_enhancing_attentions_periodic_extension_for_length_ge.md)

</div>

<!-- RELATED:END -->
