---
title: >-
  [论文解读] VisionTS: Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters
description: >-
  [ICML2025][时间序列][时间序列预测] 将时间序列重构为图像，利用 ImageNet 预训练的 MAE（Masked Autoencoder）在零样本：设置下进行时序预测，无需任何时序数据训练即可匹敌甚至超越专门的时序基础模型。 时间序列预测（TSF）的基础模型目前有两条路线： 文本路线：复用 LLM（如 GPT4…
tags:
  - "ICML2025"
  - "时间序列"
  - "时间序列预测"
  - "视觉基础模型"
  - "Masked Autoencoder"
  - "零样本预测"
  - "跨模态迁移"
---

# VisionTS: Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters

**会议**: ICML2025  
**arXiv**: [2408.17253](https://arxiv.org/abs/2408.17253)  
**代码**: [Keytoyze/VisionTS](https://github.com/Keytoyze/VisionTS)  
**领域**: 时序预测 / 跨模态迁移  
**关键词**: 时间序列预测, 视觉基础模型, Masked Autoencoder, 零样本预测, 跨模态迁移

## 一句话总结

将时间序列重构为图像，利用 ImageNet 预训练的 MAE（Masked Autoencoder）在**零样本**设置下进行时序预测，无需任何时序数据训练即可匹敌甚至超越专门的时序基础模型。

## 研究背景与动机

时间序列预测（TSF）的基础模型目前有两条路线：

**文本路线**：复用 LLM（如 GPT4TS、TimeLLM），但语言与时序之间的模态差距大，迁移效果受质疑。

**时序路线**：从头构建大规模时序数据集训练（如 Moirai、TimesFM），但时序数据在长度、频率、领域等方面高度异构，数据集构建困难。

本文提出**第三条路线**：利用预训练视觉模型做时序预测。核心观察是图像与时序共享多个关键特性：

- **连续性**：两者均为连续信号，不同于离散文本
- **同源性**：都是对真实物理系统的观测
- **信息密度**：都具有大量冗余，不同于高语义密度的语言
- **特征相似性**：ImageNet 图像的像素行天然包含趋势、季节性、平稳性等时序特征

## 方法详解

VisionTS 将时序预测重新表述为 MAE 的图像重建任务，核心流程为：**分段 → 归一化 → 渲染 → 对齐 → 重建 → 预测**。

### 1. 分段（Segmentation）

给定单变量输入 $X \in \mathbb{R}^L$，按周期 $P$ 分段为 $\lfloor L/P \rfloor$ 个长度为 $P$ 的子序列，堆叠成二维矩阵：

$$\boldsymbol{I}_{\text{raw}} \in \mathbb{R}^{P \times \lfloor L/P \rfloor}$$

周期 $P$ 可通过 FFT 或采样频率的先验知识确定。当序列无明显周期时直接设 $P=1$。

### 2. 归一化（Normalization）

对 $\boldsymbol{I}_{\text{raw}}$ 做实例归一化，标准差缩放至超参 $r$（默认 0.4）：

$$\boldsymbol{I}_{\text{norm}} = r \cdot \frac{\boldsymbol{I}_{\text{raw}} - \text{Mean}(\boldsymbol{I}_{\text{raw}})}{\text{Std}(\boldsymbol{I}_{\text{raw}})}$$

$r < 1$ 的设计是因为 MAE 预训练时像素值范围有限，压缩幅值可防止越界。

### 3. 渲染（Rendering）

将归一化矩阵渲染为灰度图像 $\boldsymbol{I}_{\text{grey}} \in \mathbb{R}^{P \times \lfloor L/P \rfloor \times 3}$，三通道取相同值。实验表明额外学习颜色变换并无显著收益。

### 4. 对齐（Alignment）

MAE 预训练图像尺寸为 $224 \times 224$，划分为 $N \times N$ 个 patch（每个 $S \times S$）。设可见 patch 列数为 $n$，掩码列数为 $N - n$：

$$n = \left\lfloor c \cdot N \cdot \frac{L}{L + H} \right\rfloor$$

其中 $c \in [0,1]$（默认 0.4）控制可见区域宽度。缩小可见区比例可使掩码率更接近 MAE 预训练时的 75%。通过双线性插值将灰度图调整到 $(N \cdot S, n \cdot S)$。

### 5. 重建与预测

MAE 重建完整图像后，反向操作：调整尺寸回原始分段维度 → 三通道取均值 → 反归一化 → 展平 → 提取预测窗口。

### 多变量处理

采用通道独立（Channel Independence）策略，每个变量单独预测，不建模变量间交互。

## 实验关键数据

### 长期预测（8 个数据集，平均 {96,192,336,720} 步）

| 方法 | 预训练数据 | 设置 | Avg MSE | Avg MAE |
|------|-----------|------|---------|---------|
| **VisionTS** | **图像** | **零样本** | **0.309** | **0.345** |
| Moirai-S | 时序 | 零样本 | 0.327 | 0.357 |
| Moirai-B | 时序 | 零样本 | 0.310 | 0.344 |
| Moirai-L | 时序 | 零样本 | 0.329 | 0.350 |
| TimeLLM | 文本 | 少样本 | 0.336 | 0.368 |
| GPT4TS | 文本 | 少样本 | 0.360 | 0.378 |
| PatchTST | 无 | 少样本 | 0.378 | 0.389 |

**关键发现**：VisionTS 零样本在 8 个数据集中 7 次取得最佳 MSE，总体平均 MSE 优于所有 TS-based 和 Text-based 基础模型。

### Monash 基准（29 个数据集）

VisionTS 在 Monash 基准的聚合指标上优于大多数零样本基础模型，覆盖多种频率和领域。

### GIFT-Eval 排行榜（23 个数据集）

VisionTS 在 GIFT-Eval 排行榜上同样取得与专用时序基础模型可比的零样本性能。

### 微调（1 epoch）

仅微调 1 个 epoch，VisionTS 即可在大多数长期预测 benchmark 上达到 SOTA。

## 亮点与洞察

1. **跨模态免费午餐**：纯视觉模型无需任何时序数据适配即可做出高质量预测，揭示了图像与时序的深层相似性
2. **极简设计**：整个方法不引入任何可训练参数（零样本模式），全靠 MAE 预训练权重
3. **Prompt 范式创新**：将 TSF 重构为 MAE 的掩码图像重建，类似 NLP 中的 prompt tuning
4. **表征可视化**：通过 MAE 编码器的 t-SNE 可视化发现，时序数据与 ImageNet 图像存在表征重叠，且图像可充当不同时序领域之间的"桥梁"
5. **超大规模评测**：覆盖 60+ 数据集（8 长期 + 29 Monash + 23 GIFT-Eval），是当时 TSF 基础模型中最大规模的评测

## 局限与展望

1. **多变量限制**：仅采用通道独立策略，未建模变量间交互，受限于图像通道数（仅 3 通道）
2. **超参敏感性**：归一化尺度 $r$ 和可见区比例 $c$ 对性能有明显影响，当前依赖经验设定
3. **周期性假设**：分段依赖周期 $P$ 的估计质量，对非周期序列可能不够鲁棒
4. **分辨率限制**：MAE 固定 224×224 输入尺寸，对超长序列或高分辨率需求存在信息压缩损失
5. **灰度限制**：仅使用灰度渲染，未充分利用 MAE 对彩色图像的建模能力

## 相关工作与启发

- **TimesNet**（Wu et al., 2023）：同样将 1D 时序变换为 2D 结构，但需从头训练
- **Moirai**（Woo et al., 2024）：TS-based 基础模型，VisionTS 的主要零样本对比对象
- **MAE**（He et al., 2022）：VisionTS 的骨干网络，利用其图像重建能力
- **SparseTSF**（Lin et al., 2024）：同样利用周期性分段的稀疏预测方法
- **启发**：视觉预训练模型可能在更多非视觉任务上具有迁移价值，值得探索图像→音频、图像→传感器等跨模态方向

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次系统性证明纯视觉模型可做零样本时序预测，研究视角极具创新
- 实验充分度: ⭐⭐⭐⭐⭐ — 60+ 数据集全面评测，含消融实验和跨模态表征分析
- 写作质量: ⭐⭐⭐⭐ — 动机阐述清晰，图表丰富，部分数学符号较密
- 价值: ⭐⭐⭐⭐⭐ — 打开了"视觉→时序"跨模态基础模型的新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Revisiting LLMs as Zero-Shot Time-Series Forecasters: Small Noise Can Break Large Models](../../ACL2025/time_series/revisiting_llms_as_zero-shot_time_series_forecasters_small_noise_can_break_large.md)
- [\[ICML 2025\] IMTS is Worth Time × Channel Patches: Visual Masked Autoencoders for Irregular Multivariate Time Series Prediction](imts_is_worth_time_times_channel_patches_visual_masked_autoencoders_for_irregula.md)
- [\[NeurIPS 2025\] Rotary Masked Autoencoders are Versatile Learners](../../NeurIPS2025/time_series/rotary_masked_autoencoders_are_versatile_learners.md)
- [\[NeurIPS 2025\] TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning](../../NeurIPS2025/time_series/tirex_zero-shot_forecasting_across_long_and_short_horizons_with_enhanced_in-cont.md)
- [\[ICLR 2026\] Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data](../../ICLR2026/time_series/relational_transformer_toward_zero-shot_foundation_models_for_relational_data.md)

</div>

<!-- RELATED:END -->
