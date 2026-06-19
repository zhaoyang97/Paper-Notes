---
title: >-
  [论文解读] Restoring Images in Adverse Weather Conditions via Histogram Transformer
description: >-
  [ECCV 2024][图像恢复][adverse weather removal] 提出 Histoformer，一种基于直方图自注意力机制的高效 Transformer，通过将空间特征按像素强度排序分箱（bin），在箱内和箱间执行自注意力，实现动态范围的空间注意力以高效处理天气退化像素，配合动态范围卷积和 Pearson 相关性损失，在去雪/去雨雾/去雨滴三大任务上统一建模并达到 SOTA。
tags:
  - "ECCV 2024"
  - "图像恢复"
  - "adverse weather removal"
  - "注意力机制"
  - "dynamic-range convolution"
  - "Pearson correlation loss"
---

# Restoring Images in Adverse Weather Conditions via Histogram Transformer

**会议**: ECCV 2024  
**arXiv**: [2407.10172](https://arxiv.org/abs/2407.10172)  
**代码**: [GitHub（已开源）](https://github.com/sunshangquan/Histoformer)  
**领域**: 图像恢复 / 恶劣天气去除  
**关键词**: image restoration, adverse weather removal, histogram self-attention, dynamic-range convolution, Pearson correlation loss

## 一句话总结

提出 Histoformer，一种基于直方图自注意力机制的高效 Transformer，通过将空间特征按像素强度排序分箱（bin），在箱内和箱间执行自注意力，实现动态范围的空间注意力以高效处理天气退化像素，配合动态范围卷积和 Pearson 相关性损失，在去雪/去雨雾/去雨滴三大任务上统一建模并达到 SOTA。

## 研究背景与动机

**领域现状**：图像恢复在雨、雾、雪等恶劣天气下已被广泛研究，从早期 CNN 方法到近年 Transformer 方法取得显著进展。All-in-One 天气去除（单一模型处理多种天气退化）成为重要方向。

**现有痛点**：Transformer 方法为降低计算开销，要么将自注意力限制在固定空间窗口内（如 Swin Transformer），要么仅在通道维度做注意力（如 Restormer），牺牲了对长距离空间特征的捕获能力。

**核心矛盾**：天气退化（雨滴、雪花、雾气）的像素空间分布是动态的、全局分散的，但具有相似的强度模式——固定窗口注意力无法将分散但相似的退化像素关联起来。

**本文目标**：设计一种高效的注意力机制，能自适应地关注空间上分散但强度相似的天气退化像素。

**切入角度**：观察到天气退化通常导致相似的遮挡和亮度模式，因此可按像素强度排序分箱，将相似退化的像素聚到一起做注意力。

**核心 idea**：将空间特征按强度排序后分入直方图的 bin 中，在 bin 内（fine-grained）和 bin 间（global）分别做自注意力，实现动态范围的高效空间注意力。

## 方法详解

### 整体框架

Histoformer 采用 U-Net 形式的 encoder-decoder 架构。输入低质量图像 $I^{lq} \in \mathbb{R}^{3 \times H \times W}$ 经 $3 \times 3$ 卷积做 patch embedding，经过多阶段的 **Histogram Transformer Block (HTB)** 提取特征。encoder 和 decoder 之间通过 skip-connection 连接，阶段之间用 pixel-unshuffle/shuffle 做降采样/升采样。encoder 阶段还有 crude skip-connection（avg pooling + pointwise conv + depthwise conv）从输入补充原始特征，使 encoder 更专注学习天气退化残差。

每个 HTB 包含两个核心模块：
- **Dynamic-range Histogram Self-Attention (DHSA)**
- **Dual-scale Gated Feed-Forward (DGFF)**

### 关键设计

#### 1. Dynamic-range Histogram Self-Attention (DHSA)

DHSA 是本文最核心的创新，包含动态范围卷积和双路径直方图自注意力两部分。

**动态范围卷积（Dynamic-range Convolution）**：在常规卷积前对特征做排序重组，使卷积能在动态范围上操作。将输入特征 $F \in \mathbb{R}^{C \times H \times W}$ 沿通道分为 $F_1, F_2$，对 $F_1$ 先水平排序再垂直排序，然后与 $F_2$ 拼接后做 $1 \times 1$ 点卷积 + $3 \times 3$ 深度可分离卷积：

$$F_1 = \text{Sort}_v(\text{Sort}_h(F_1))$$
$$F = \text{Conv}^d_{3 \times 3}(\text{Conv}_{1 \times 1}(\text{Concat}(F_1, F_2)))$$

**设计动机**：排序后，高/低强度像素分别集中在矩阵对角，天气退化像素（高强度相似）自然聚集，使小卷积核也能捕获动态范围的特征关联。

**直方图自注意力（Histogram Self-Attention）**：对动态范围卷积的输出分离出 Value $V$ 和两对 Query-Key $(Q_1, K_1), (Q_2, K_2)$。按 $V$ 的像素强度排序并重排 Q-K，然后定义两种 reshape 方式：

- **Bin-wise Histogram Reshaping (BHR)**：设定 $B$ 个 bin，每个 bin 含 $HW/B$ 个像素。每个 bin 覆盖较宽强度范围→全局特征聚合。
- **Frequency-wise Histogram Reshaping (FHR)**：每个 bin 含 $B$ 个频率，共 $HW/B$ 个 bin。每个 bin 只含少量强度相近的像素→精细特征提取。

两路分别做自注意力后逐元素相乘融合：

$$A_B = \text{softmax}\left(\frac{\mathbf{R}_B(Q_1) \cdot \mathbf{R}_B(K_1)^\top}{\sqrt{k}}\right) \mathbf{R}_B(V)$$

$$A_F = \text{softmax}\left(\frac{\mathbf{R}_F(Q_2) \cdot \mathbf{R}_F(K_2)^\top}{\sqrt{k}}\right) \mathbf{R}_F(V)$$

$$A = A_B \odot A_F$$

最终排序回原始空间位置，通过 $1 \times 1$ 卷积输出。

#### 2. Dual-scale Gated Feed-Forward (DGFF)

替代标准 FFN，引入双尺度双范围的深度卷积路径。输入先通过 $1 \times 1$ 卷积扩展通道（因子 $r=2.667$），经 pixel-shuffle 后分为两支：
- 分支 1：$5 \times 5$ 深度卷积（多尺度）
- 分支 2：$3 \times 3$ 膨胀深度卷积（多范围）

分支 2 经 Mish 激活后作为门控图与分支 1 逐元素相乘，再经 pixel-unshuffle + $1 \times 1$ 卷积输出：

$$F_{l+1} = \text{Conv}_{1 \times 1}(\text{Unshuffle}(\text{Mish}(F_{l,2}) \odot F_{l,1}))$$

**设计动机**：天气退化的多尺度特性需要不同感受野的卷积协同工作，门控机制自适应选择有效特征。

#### 3. Pearson 相关性损失

观察到像素级 L1 损失忽略了输出与 GT 之间的整体线性关联。引入 Pearson 相关系数作为辅助损失：

$$\rho(I^{hq}, I^{gt}) = \frac{\sum_{i=1}^{3HW}(I^{hq}_i - \bar{I}^{hq})(I^{gt}_i - \bar{I}^{gt})}{3HW \cdot \sigma(I^{hq}) \cdot \sigma(I^{gt})}$$

$$\mathcal{L}_{cor} = \frac{1}{2}(1 - \rho(I^{hq}, I^{gt}))$$

总损失：$\mathcal{L} = \mathcal{L}_{rec} + \alpha \mathcal{L}_{cor}$，默认 $\alpha = 1$。

**设计动机**：天气退化打乱了图像内部像素的强度排序关系，Pearson 损失强制恢复像素遵循与 GT 相同的排序，补充 L1 损失未覆盖的 patch 级结构信息。

### 损失函数 / 训练策略

- **训练数据**：Snow100K（9,000 张）+ Raindrop（1,069 张）+ Outdoor-Rain（9,000 张），统一训练
- **优化器**：AdamW，初始 lr $3 \times 10^{-4}$，cosine annealing 到 $1 \times 10^{-6}$
- **训练**：300K 迭代，渐进式学习（batch size 8，patch size 128 起步）
- **网络配置**：每阶段 block 数 {4,4,6,8}，通道数 C=36，注意力头数 {1,2,4,8}
- **数据增强**：随机水平/垂直翻转
- **硬件**：NVIDIA V100

## 实验关键数据

### 主实验

| 任务 | 数据集 | 指标 | Histoformer | Restormer | TransWeather | WeatherDiff64 | AWRCP |
|------|--------|------|-------------|-----------|-------------|-------------|-------|
| 去雪 | Snow100K-S | PSNR | **37.41** | 36.01 | 32.51 | 35.83 | 36.92 |
| 去雪 | Snow100K-L | PSNR | **32.16** | 30.36 | 29.31 | 30.09 | 31.92 |
| 去雨雾 | Outdoor-Rain | PSNR | **32.08** | 30.03 | 28.83 | 29.64 | 31.39 |
| 去雨滴 | RainDrop | PSNR | **33.06** | 32.18 | 30.17 | 30.71 | 31.93 |

注：Restormer、NAFNet 等在每个任务上单独训练，Histoformer 用统一模型处理所有任务仍超越它们。

### 消融实验

**自注意力类型对比**（Outdoor-Rain）：

| 自注意力 | PSNR | SSIM |
|----------|------|------|
| MDTA (Restormer) | 30.94 | 0.9278 |
| TKSA (稀疏注意力) | 31.12 | 0.9295 |
| w/o BHR | 31.05 | 0.9301 |
| w/o FHR | 31.79 | 0.9364 |
| **DHSA (完整)** | **32.08** | **0.9389** |

**前馈网络对比**：

| FFN 类型 | PSNR | SSIM |
|----------|------|------|
| Vanilla FFN | 31.32 | 0.9313 |
| GDFN (Restormer) | 31.42 | 0.9347 |
| MSFN | 31.78 | 0.9367 |
| **DGFF** | **32.08** | **0.9389** |

**相关性损失权重**：

| $\alpha$ | PSNR | SSIM |
|----------|------|------|
| 0 (无 $\mathcal{L}_{cor}$) | 31.77 | 0.9358 |
| 0.1 | 32.01 | 0.9369 |
| **1** | **32.08** | **0.9389** |
| 5 | 32.03 | 0.9392 |
| 10 | 31.96 | 0.9375 |

### 关键发现

1. **DHSA 贡献最大**：完整 DHSA 比 TKSA 提升 0.96 dB PSNR，BHR 和 FHR 缺一不可，但 FHR 贡献更大（去掉 BHR 降 1.03 dB，去掉 FHR 降 0.29 dB）。
2. **动态范围卷积有效但提升温和**：排序后卷积比普通卷积提升 0.14 dB，排序顺序影响不大。
3. **DGFF 比所有对比 FFN 都好**：比 MSFN 提升 0.3 dB，说明 pixel-shuffle + 膨胀卷积的组合有效。
4. **Pearson 损失一致有效**：添加后提升 0.31 dB，权重在 0.1-5 范围内均有效，对超参不敏感。
5. **C×B 增大持续提升性能**但 44 时已 OOM，36 为最佳可用配置。

## 亮点与洞察

- **直方图分箱做注意力**是一个非常巧妙的设计：天气退化的"相似像素分散分布"特点被排序-分箱操作优雅地解决，既保持线性复杂度又获得长距离空间注意力。
- **BHR + FHR 双路径**在全局-精细之间平衡：BHR 负责跨强度范围的全局聚合，FHR 负责同强度内的细粒度处理，两者互补。
- **Pearson 相关性损失**是一个通用的 insight：像素级损失忽略排序关系，相关性损失可以推广到任何图像恢复任务。
- **统一模型超越专用模型**：Histoformer 用一个模型在所有天气任务上超越了 Restormer 等专门训练的方法，说明直方图注意力对不同天气退化模式有强泛化性。

## 局限与展望

1. 排序操作的计算开销未详细分析，在超高分辨率图像上可能成为瓶颈。
2. 仅验证了雨/雪/雾三种天气，未涉及沙尘、冰冻等极端条件。
3. 通道数 C=36 受限于 OOM，说明模型容量可能未充分释放。
4. 数据集相对较小（~19k 训练），在更大数据集上的表现有待验证。
5. 直方图分箱数 B 为固定值，自适应 bin 数可能进一步提升效果。

## 相关工作与启发

- **Restormer** [Zamir et al.] 是最强对比方法，采用通道维度自注意力。Histoformer 证明了空间维度注意力（通过分箱策略）比通道注意力更有效。
- **TransWeather** [Valanarasu et al.] 是 All-in-One 天气去除的先驱，但性能远逊 Histoformer。
- **WeatherDiff** [Özdenizci & Bhatt] 引入扩散模型做天气去除，但定量和视觉效果均不及 Histoformer。
- 启发：基于内容排序的注意力机制可能在其他退化模式（如噪声、模糊）中同样有效，值得进一步探索。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 直方图自注意力是全新的注意力范式，思路新颖且动机自然
- 实验充分度: ⭐⭐⭐⭐ 三大任务全面对比 + 详细消融，但缺乏复杂度分析
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，公式完整，图示直观
- 价值: ⭐⭐⭐⭐⭐ 统一模型 SOTA + 通用性强的注意力机制，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Teaching Tailored to Talent: Adverse Weather Restoration via Prompt Pool and Depth-Anything Constraint](teaching_tailored_to_talent_adverse_weather_restoration_via_prompt_pool_and_dept.md)
- [\[ICCV 2025\] Robust Adverse Weather Removal via Spectral-based Spatial Grouping (SSGformer)](../../ICCV2025/image_restoration/robust_adverse_weather_removal_via_spectral-based_spatial_grouping.md)
- [\[ECCV 2024\] Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration](seeing_the_unseen_a_frequency_prompt_guided_transformer_for_image_restoration.md)
- [\[NeurIPS 2025\] MoDEM: A Morton-Order Degradation Estimation Mechanism for Adverse Weather Image Restoration](../../NeurIPS2025/image_restoration/modem_a_morton-order_degradation_estimation_mechanism_for_adverse_weather_image_.md)
- [\[ECCV 2024\] A New Dataset and Framework for Real-World Blurred Images Super-Resolution](a_new_dataset_and_framework_for_real-world_blurred_images_super-resolution.md)

</div>

<!-- RELATED:END -->
