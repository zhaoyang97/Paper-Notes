---
title: >-
  [论文解读] Lightweight Optimal-Transport Harmonization on Edge Devices
description: >-
  [AAAI 2026 Oral][模型压缩][颜色协调] 提出 MKL-Harmonizer，利用经典最优传输理论中的 Monge-Kantorovich 线性映射（MKL），训练一个轻量级编码器预测 12 维颜色变换参数，实现边缘设备上的实时图像颜色协调，在 AR 场景的感知质量-速度综合指标上达到最优。
tags:
  - "AAAI 2026 Oral"
  - "模型压缩"
  - "颜色协调"
  - "最优传输"
  - "边缘设备"
  - "增强现实"
  - "轻量化推理"
---

# Lightweight Optimal-Transport Harmonization on Edge Devices

**会议**: AAAI 2026 Oral  
**arXiv**: [2511.12785](https://arxiv.org/abs/2511.12785)  
**代码**: [github](https://github.com/maria-larchenko/mkl-harmonizer)  
**领域**: 模型压缩  
**关键词**: 颜色协调, 最优传输, 边缘设备, 增强现实, 轻量化推理

## 一句话总结
提出 MKL-Harmonizer，利用经典最优传输理论中的 Monge-Kantorovich 线性映射（MKL），训练一个轻量级编码器预测 12 维颜色变换参数，实现边缘设备上的实时图像颜色协调，在 AR 场景的感知质量-速度综合指标上达到最优。

## 研究背景与动机

### 领域现状
图像颜色协调（Image Harmonization）旨在调整合成图像中前景物体的颜色，使其与背景在视觉感知上保持一致。现有方法主要基于深度学习的密集预测模型（如 DoveNet、RainNet），虽然效果好，但计算和内存需求大，主要局限于 256×256 低分辨率输入。

### 现有痛点

**计算资源要求高**：密集预测模型（encoder-decoder 架构）无法在移动 GPU、XR 头显等边缘设备上实时运行

**AR 场景缺乏支持**：ARKit、ARCore 等主流 AR 平台仅依赖光照估计（方向光、环境图、球谐光照），缺少先进的颜色协调

**曝光偏差问题**：标准训练数据集（如 iHarmony4）中的 mask 包含边界处的背景像素泄漏，模型会过度依赖这些信息。而 AR 场景中渲染引擎提供像素精确的 mask，导致训练-推理不匹配

**AR 评测数据稀缺**：不存在带像素精确 mask 的真实 AR 合成图像数据集

### 核心矛盾
颜色协调对 AR 真实感至关重要，但现有方法在延迟和计算量上无法满足实时 AR 要求。

### 切入角度与核心 idea
颜色协调本质上是一个颜色分布映射问题。当源和目标颜色分布近似为多元高斯分布时，最优传输映射有闭式线性解——Monge-Kantorovich 线性（MKL）滤波器，仅需 12 个参数（3×3 矩阵 A + 3 维偏移 S）。作者提出训练一个紧凑的编码器来预测这 12 个参数，从而实现极其轻量且快速的推理。

## 方法详解

### 整体框架
输入 4 通道（RGB + mask）的合成图像 → EfficientNet-B0 编码器 → 输出 12 维向量 [A, S] → 构造 MKL 滤波器 → 对前景像素应用仿射颜色变换 → 输出协调后图像。

### 关键设计

1. **最优传输理论基础**:

    - 将颜色协调建模为从源分布 π₀ 到目标分布 π₁ 的传输映射问题
    - 当两个分布近似高斯时，最优传输映射为闭式线性解：$T^*_{im}(x) = \mu_1 + A(x - \mu_0)$
    - 其中 $A = \Sigma_0^{-1/2}(\Sigma_0^{1/2}\Sigma_1\Sigma_0^{1/2})^{1/2}\Sigma_0^{-1/2}$
    - 设计动机：理想 MKL 滤波器在 iHarmony4 上 MSE 仅约 7.0，证明线性滤波器对大多数协调任务已足够

2. **Ground-Truth 滤波器生成**:

    - 对 iHarmony4 中每张图像计算精确的 MKL 变换参数作为监督信号
    - 将问题从预测协调后图像简化为预测 12 个滤波器参数
    - 直接预测 [A, S] 而非 [μ₁, Σ₁]，实验表明对预测误差更鲁棒

3. **理论误差分析**:

    - 给出了线性 MKL 映射近似非线性真实映射的误差上界
    - $\mathcal{E} \leq 2\mathcal{E}_{clip} + 2\mathcal{E}_{lin}$
    - $\mathcal{E}_{lin} \leq 2B^2 + 2(\|A\|_{op} + L)^2 \cdot \text{tr}(\Sigma_0)$
    - 当真实映射平滑（小 Lipschitz 常数）且颜色分布不集中在色域边界时，线性近似有效
    - 深色物体分布集中在色域角落，可能导致结果不佳

4. **ARCore 评测数据集**:

    - 修改 ARCore 示例应用，构建数据采集工具
    - 收集 327 对合成图-mask 对，涵盖室内外、不同时间和天气
    - 所有 mask 直接从渲染引擎获得，像素精确
    - 这是首个此类开源 AR 合成图像数据集

### 损失函数 / 训练策略

采用混合损失函数：

$$L_{total} = L_{labels} + \alpha \cdot L_{content}$$

- **标签损失**：$L_{labels} = \|\text{Model}(im) - [A, S]\|_1$，使用 L1 而非 L2 损失
    - L2 会迫使预测为所有可能 MKL 解的算术均值（可能对应无效滤波器）
    - L1 允许收敛到更锋利的解
- **内容损失**：$L_{content} = \|M * X_0 - M * (X_0 \cdot A' + S')\|_1$，逐像素 L1 损失
    - 如果仅用内容损失，模型会学到接近恒等变换的滤波器（模式崩塌）
    - 内容损失权重 α=10

编码器使用 EfficientNet-B0，输入 256×256，4 通道（RGB+mask），训练 210 个 epoch，Adam 优化器，学习率分段衰减，batch size 64。

## 实验关键数据

### 主实验

| 方法 | MSE↓ | PSNR↑ | fMSE↓ |
|------|------|-------|-------|
| Ideal Linear OT | 7.6 | 43.6 | 45.9 |
| PCT-Net | 29.1 | 38.0 | 201 |
| Harmonizer | 40.1 | 36.6 | 258 |
| **Ours (L1)** | 65.0 | 34.1 | 438 |
| INR | 67.2 | 35.3 | 392 |
| Unharmonized | 182 | 31.0 | 984 |

| 方法 | 256×256 | 512×512 | 1024×2048 | 4096×4096 |
|------|---------|---------|-----------|-----------|
| **Ours** | **175.01** | **166.76** | **137.21** | **40.85** |
| DoveNet | 123.39 | – | – | – |
| PCT-Net | 104.57 | 98.65 | 63.74 | 11.84 |
| Harmonizer | 95.01 | 89.82 | 47.63 | 7.45 |
| INR | 6.35 | 3.22 | 0.81 | 0.12 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Ours L1 损失 | MSE=65.0 | L1 损失整体更优 |
| Ours L2 损失 | MSE=66.3 | L2 略差，但差异不大 |
| 仅内容损失 | 模式崩塌 | 模型学到恒等变换 |
| 预测 [μ₁, Σ₁] | 更差 | 对预测误差不够鲁棒 |
| 预测 [A, S] | 更好 | 直接预测滤波器参数更稳定 |

### 关键发现
1. **感知质量**：用户研究（20 名参与者，642 次评分）表明 MKL-Harmonizer 在真实 AR 数据上的感知质量与领先基线持平
2. **速度-质量权衡**：本方法同时具备最高感知得分和最快推理速度
3. **边缘设备部署**：在 Google Pixel 4a/7 上实现 12-15 fps，通过零拷贝优化可能达到 24-30 fps
4. **曝光偏差**：INR 在 MSE 指标上优于 PCT-Net，但人类评估中感知质量反而更差，说明 MSE 在 AR 场景不可靠
5. **高分辨率优势**：密集预测模型在高分辨率下产生退化（条纹、JPEG 伪影），而滤波器方法无此问题

## 亮点与洞察
1. 将经典最优传输理论与深度学习巧妙结合，理论扎实且实用性强
2. 首次提出并分析了图像协调中的"曝光偏差"问题，指出标准评估指标的不足
3. 仅预测 12 个参数，模型极其轻量，开辟了滤波器参数预测的新思路
4. 提供了完整的理论误差分析，证明线性映射的有效条件
5. 构建了首个带像素精确 mask 的 AR 合成图像数据集

## 局限与展望
1. 在 iHarmony4 标准指标上不如 SOTA（MSE 较高），但作者认为这是曝光偏差导致的指标失真
2. 深色物体处理效果差（颜色分布集中于色域边界）
3. 不适合视频协调（帧间预测可能跳变），目前仅用指数移动平均缓解
4. iHarmony4 存在高频伪影，作者清洗了数据集但未成为标准做法
5. 仅在 EfficientNet-B0 上验证，可探索更高效的架构

## 相关工作与启发
- **PCT-Net**：预测像素级仿射变换参数 → 本文简化为全局 12 参数
- **INR-Harmonization**：隐式神经表示 → 计算量大，不适合实时
- **Harmonizer**：回归亮度/对比度滤波器系数 → 本文基于最优传输理论更系统化
- 最优传输在颜色迁移中的经典方法（Pitié 2007）历久弥新

## 评分
- 新颖性: ⭐⭐⭐⭐ (最优传输+编码器预测的组合虽新颖，但整体框架复杂度不高)
- 实验充分度: ⭐⭐⭐⭐ (iHarmony4+ARCore+用户研究+边缘部署，但标准指标偏弱)
- 写作质量: ⭐⭐⭐⭐⭐ (理论分析完整，动机清晰，数据集贡献有价值)
- 价值: ⭐⭐⭐⭐ (实际应用价值高，AR 场景需求明确)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Optimizing Distributional Geometry Alignment with Optimal Transport for Generative Dataset Distillation](../../NeurIPS2025/model_compression/optimizing_distributional_geometry_alignment_with_optimal_transport_for_generati.md)
- [\[ICLR 2026\] Cross-Domain Lossy Compression via Rate- and Classification-Constrained Optimal Transport](../../ICLR2026/model_compression/cross_domain_lossy_compression_optimal_transport.md)
- [\[CVPR 2026\] MEMO: Human-like Crisp Edge Detection Using Masked Edge Prediction](../../CVPR2026/model_compression/memo_human-like_crisp_edge_detection_using_masked_edge_prediction.md)
- [\[CVPR 2026\] Critical Patch-Aware Sparse Prompting with Decoupled Training for Continual Learning on the Edge](../../CVPR2026/model_compression/critical_patch-aware_sparse_prompting_with_decoupled_training_for_continual_lear.md)
- [\[AAAI 2026\] Satisficing and Optimal Generalised Planning via Goal Regression (Extended Version)](satisficing_and_optimal_generalised_planning_via_goal_regression_extended_versio.md)

</div>

<!-- RELATED:END -->
