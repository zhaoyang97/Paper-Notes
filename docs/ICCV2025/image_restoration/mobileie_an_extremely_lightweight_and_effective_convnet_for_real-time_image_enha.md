---
title: >-
  [论文解读] MobileIE: An Extremely Lightweight and Effective ConvNet for Real-Time Image Enhancement on Mobile Devices
description: >-
  [ICCV 2025][图像恢复][移动端图像增强] 提出 MobileIE，一个仅有约 4K 参数的极致轻量 CNN 框架，通过多分支重参数化卷积（MBRConv）、特征自变换（FST）模块、分层双路径注意力（HDPA）以及增量权重优化（IWO）策略，首次在移动设备上实现超过 1100 FPS 的实时图像增强，同时在低光增强、水下增强和 ISP 三个任务上取得最优的速度-性能平衡。
tags:
  - "ICCV 2025"
  - "图像恢复"
  - "移动端图像增强"
  - "重参数化"
  - "轻量级CNN"
  - "实时推理"
  - "注意力机制"
---

# MobileIE: An Extremely Lightweight and Effective ConvNet for Real-Time Image Enhancement on Mobile Devices

**会议**: ICCV 2025  
**arXiv**: [2507.01838](https://arxiv.org/abs/2507.01838)  
**代码**: [https://github.com/AVC2-UESTC/MobileIE.git](https://github.com/AVC2-UESTC/MobileIE.git)  
**领域**: 图像增强 / 图像恢复  
**关键词**: 移动端图像增强, 重参数化, 轻量级CNN, 实时推理, 注意力机制

## 一句话总结

提出 MobileIE，一个仅有约 4K 参数的极致轻量 CNN 框架，通过多分支重参数化卷积（MBRConv）、特征自变换（FST）模块、分层双路径注意力（HDPA）以及增量权重优化（IWO）策略，首次在移动设备上实现超过 1100 FPS 的实时图像增强，同时在低光增强、水下增强和 ISP 三个任务上取得最优的速度-性能平衡。

## 研究背景与动机

深度学习驱动的图像增强（IE）模型在质量上取得了显著进展，但**部署到资源受限的移动设备**仍面临严峻挑战：

- **Transformer/Diffusion 方法**：自注意力和迭代扩散的计算开销巨大，完全不适合移动端
- **现有轻量模型**：虽然降低了 FLOPs，但往往牺牲增强质量，且通常针对特定退化类型设计
- **高分辨率需求**：用户对高分辨率图像的需求不断增加，进一步加剧了移动端的计算负担

核心理念：**移动端 IE 应在速度和性能之间取得平衡**，使用通用架构和硬件友好的算子。训练和推理阶段应解耦——训练时使用复杂的多分支结构学习特征，推理时重参数化为简单的单卷积。

## 方法详解

### 整体框架

MobileIE 架构极其精简：
浅层特征提取（MBRConv5×5 + PReLU）→ 深层特征提取（2×MBRConv3×3 + FST）→ 注意力（HDPA）→ 输出精修（MBRConv3×3）

推理时所有 MBRConv 重参数化为标准卷积，模型仅约 4K 参数。

### 关键设计

1. **多分支重参数化卷积（MBRConv）**：

    - 训练时包含多个不同大小卷积核的并行分支，捕获多尺度特征
    - 每个分支附带**并行 Batch Norm**——虽然 BN 对 IE 任务效果有限，但它增强了非线性且可以与卷积合并
    - 分支输出拼接后通过 Conv 1×1 压缩映射到目标维度
    - 推理时所有分支重参数化为一个标准卷积，零额外开销
    - **关键创新**：与 RepVGG 等方法不同，MBRConv 的并行 BN 同时保留了平滑和原始特征，提高了跨数据分布的鲁棒性

2. **增量权重优化策略（IWO）**：

    - 解决紧凑网络在训练后期**性能停滞**的问题
    - $W_{final} = \text{Frozen}(W_{pre}) + W_{learn}$
    - $W_{pre}$：前期训练的最优权重（冻结），提供稳定的初始特征表示
    - $W_{learn}$：动态更新的新权重，负责细化任务特定细节
    - 效果：增强卷积核骨架特征（中心行/列），减少通道冗余（KL 散度显著增加），打破训练收敛瓶颈

3. **特征自变换模块（FST）**：

    - $\text{FST}(x) = Scale \cdot (x * x) + bias$
    - 通过二次交互捕获高阶非线性特征关系，弥补线性卷积的表达能力不足
    - 可学习的 Scale 和 bias 自适应调整特征动态范围
    - 在频域分析中，平方运算比 ReLU 对高频信息更敏感，能更好地保留边缘和细节
    - 计算开销极低，适合轻量模型

4. **分层双路径注意力（HDPA）**：

    - 全局路径：Adaptive AvgPool → MBRConv1×1 → Sigmoid → 通道注意力权重 $A_g$
    - 局部路径：全局加权后 → MaxPool → MBRConv1×1 → Sigmoid → 局部注意力权重 $A_l$
    - 最终输出：$\hat{F} = (A_g * A_l) * F$
    - 双路径设计在反向传播中支持互相优化，层级式捕获全局上下文和局部细节

### 损失函数 / 训练策略

**Local Variance Weighted (LVW) Loss**：
- 计算每像素预测误差 $\Delta_{m,n} = \|O_{m,n} - L_{m,n}\|_1$
- 基于空间维度计算局部均值 $\mu$ 和方差 $\sigma^2$
- 权重 $W_\Delta = \text{Tanh}(\frac{|\Delta_{m,n} - \mu_{m,n}|}{\sigma_{m,n} + \epsilon})$
- 最终损失 $\mathcal{L}_{LVW} = \frac{1}{HW}\sum(W_\Delta \cdot \Delta_{m,n})$
- 优势：避免 L2 对极端像素的过度敏感，同时比 L1 更好地处理异常值，动态调整每个像素的贡献

训练设置：Adam 优化器，余弦退火学习率（初始 0.001），每 50 epoch 重置，2000 epochs，10 epoch warm-up。

## 实验关键数据

### 主实验 (表格)

**低光增强（LOLv1 + LOLv2-Real）**

| 方法 | 参数量 | GPU延迟(ms) | SoC延迟(ms) | LOLv1 PSNR | LOLv2 PSNR |
|------|--------|------------|------------|------------|------------|
| IAT | 86.9K | 6.204 | 202.33 | 23.38 | 25.46 |
| SYELLE | 5.3K | 0.944 | 7.73 | 21.03 | 21.26 |
| Zero-DCE++ | 10.6K | 1.974 | 57.91 | 14.68 | 17.23 |
| **MobileIE** | **4.0K** | **0.895** | **6.72** | **23.62** | **25.08** |

**水下图像增强（UIEB）**

| 方法 | 参数量 | SoC延迟(ms) | PSNR | SSIM |
|------|--------|------------|------|------|
| FiveA+ | 9.0K | 423.43 | 22.51 | 0.902 |
| Boths | 6.4K | 58.04 | 22.23 | 0.904 |
| **MobileIE** | **4.0K** | **8.94** | **22.81** | **0.906** |

**ISP（ZRR）**

| 方法 | 参数量 | SoC延迟(ms) | PSNR | SSIM |
|------|--------|------------|------|------|
| SYEISP | 5.6K | 16.47 | 20.84 | 0.728 |
| NAFNet | 7.8K | 78.62 | 21.12 | 0.736 |
| **MobileIE** | **4.1K** | **14.40** | **21.43** | **0.731** |

### 消融实验 (表格)

**重参数化和损失函数消融（UIEB）**

| 设置 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| 仅推理网络（无重参数化） | 21.48 | 0.887 | 0.192 |
| L1 loss | 22.20 | 0.902 | 0.168 |
| L2 loss | 21.74 | 0.894 | 0.175 |
| Charbonnier loss | 22.31 | 0.905 | 0.162 |
| **LVW loss** | **22.57** | **0.906** | **0.160** |
| RepVGG | 22.69 | 0.821 | 0.202 |
| ECBSR | 23.96 | 0.816 | 0.204 |
| MBRConv (无BN) | 23.27 | 0.821 | 0.232 |
| MBRConv (无IWO) | 24.02 | 0.823 | 0.199 |
| **MBRConv + IWO (完整)** | **24.35** | **0.829** | **0.189** |

**注意力机制对比（LOLv1）**

| 注意力方法 | 参数量 | PSNR | SSIM |
|-----------|--------|------|------|
| SE-Net | 4.2K | 22.27 | 0.804 |
| CBAM | 4.3K | 22.38 | 0.796 |
| ECA-Net | 3.9K | 21.79 | 0.799 |
| **HDPA(ours)** | **4.0K** | **23.62** | **0.812** |

### 关键发现

- MobileIE 是首个在移动端实现**超过 1100 FPS**（600×400 分辨率）的 IE 模型
- 仅 4K 参数即可达到与参数量大 20 倍模型相当甚至更好的增强质量
- IWO 策略能有效打破训练后期收敛停滞——训练损失在应用 IWO 后继续下降
- IWO 增强了卷积核的中心行/列骨架特征，并显著减少了通道间的冗余（KL 散度可视化验证）
- FST 的平方变换在频域中比 ReLU 更强的高频响应，更好地保留图像细节
- LVW 损失在所有测试的损失函数中（L1, L2, Smooth L1, Charbonnier, Robust Loss）表现最优

## 亮点与洞察

- **极致简约哲学**：整个模型架构仅需几个卷积和一些轻量模块，证明了"少即是多"在移动端 IE 中的有效性
- **训练-推理解耦**的巧妙设计：训练时的多分支结构提供丰富的特征学习能力，推理时重参数化为单一卷积确保极致效率
- **IWO 策略**可泛化到其他紧凑模型训练场景，本质上是一种知识蒸馏思想（从自身先前状态蒸馏）
- **三个不同 IE 任务**（低光、水下、ISP）使用同一架构，验证了方法的通用性

## 局限与展望

- PSNR 虽然领先轻量方法，但与大型方法（DDNet 等）仍有差距
- 未在去噪、去雾等其他低级视觉任务上验证
- 4K 参数的极致压缩对更复杂的退化场景（如混合退化）可能力不从心
- IWO 策略的最优冻结时机目前是固定的（1000 epochs），自适应策略可能更优
- 当前仅在 Snapdragon 8 Gen 3 上测试了 SoC 延迟，更多移动平台的适配需验证

## 相关工作与启发

- **RepVGG / ACNet / DBB**：重参数化的先驱工作，MBRConv 在此基础上增加了并行 BN 和 IWO
- **StarNet**：星形操作启发了 FST 的二次交互设计
- **VanillaNet**：极简主义设计理念与 MobileIE 一脉相承
- **SYELLE**（ICCV'23）：同一赛道的轻量 IE 方法，是 MobileIE 最直接的对比对象
- **NTIRE 挑战赛**：高效 IE 赛道的多个方案使用了重参数化结构

## 评分

- **新颖性**: ⭐⭐⭐⭐ — IWO 策略和 FST 模块设计新颖且有效，4K 参数达到 1100 FPS 令人印象深刻
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三个 IE 任务全面评估，消融涵盖每个模块、损失函数、注意力机制，还有可视化分析和频域分析
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，每个模块都有直觉解释和理论分析，可视化丰富
- **价值**: ⭐⭐⭐⭐⭐ — 对移动端 AI 部署有直接的实用价值，4K 参数的极致效率为移动端开发者提供了新选项

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Lightweight and Fast Real-time Image Enhancement via Decomposition of the Spatial-aware Lookup Tables](lightweight_and_fast_real-time_image_enhancement_via_decomposition_of_the_spatia.md)
- [\[ICCV 2025\] Learning Pixel-adaptive Multi-layer Perceptrons for Real-time Image Enhancement](learning_pixel-adaptive_multi-layer_perceptrons_for_real-time_image_enhancement.md)
- [\[ICCV 2025\] CWNet: Causal Wavelet Network for Low-Light Image Enhancement](cwnet_causal_wavelet_network_for_low-light_image_enhancement.md)
- [\[ICCV 2025\] Low-Light Image Enhancement using Event-Based Illumination Estimation (RetinEV)](low-light_image_enhancement_using_event-based_illumination_estimation.md)
- [\[ICCV 2025\] Devil is in the Uniformity: Exploring Diverse Learners within Transformer for Image Restoration](devil_is_in_the_uniformity_exploring_diverse_learners_within_transformer_for_ima.md)

</div>

<!-- RELATED:END -->
