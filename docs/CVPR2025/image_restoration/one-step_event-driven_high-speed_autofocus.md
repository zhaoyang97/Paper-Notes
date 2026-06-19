---
title: >-
  [论文解读] One-Step Event-Driven High-Speed Autofocus
description: >-
  [CVPR 2025][图像恢复][事件相机] 提出Event Laplacian Product (ELP)对焦检测函数，结合事件数据与灰度拉普拉斯信息，将对焦搜索重新定义为检测任务，首次实现事件驱动的一步自动对焦，对焦时间减少2/3，对焦误差降低22-24倍。 高速自动对焦在极端场景（低光、运动模糊）中仍是重大挑战…
tags:
  - "CVPR 2025"
  - "图像恢复"
  - "事件相机"
  - "自动对焦"
  - "一步对焦"
  - "拉普拉斯算子"
  - "高速对焦"
---

# One-Step Event-Driven High-Speed Autofocus

**会议**: CVPR 2025  
**arXiv**: [2503.01214](https://arxiv.org/abs/2503.01214)  
**代码**: 将公开  
**领域**: Image Restoration / Computational Photography  
**关键词**: 事件相机, 自动对焦, 一步对焦, 拉普拉斯算子, 高速对焦

## 一句话总结

提出Event Laplacian Product (ELP)对焦检测函数，结合事件数据与灰度拉普拉斯信息，将对焦搜索重新定义为检测任务，首次实现事件驱动的一步自动对焦，对焦时间减少2/3，对焦误差降低22-24倍。

## 研究背景与动机

高速自动对焦在极端场景（低光、运动模糊）中仍是重大挑战。传统对比度AF需要围绕焦点反复采样导致"对焦搜索"(focus hunting)，PDAF虽能一步对焦但受限于双像素设计复杂性和低光性能。

现有事件驱动AF方法（EGS、PBF）虽利用了事件相机的高时间分辨率（1μs级）和高动态范围优势，但仍需捕获完整的对焦堆栈（从失焦→对焦→失焦），然后搜索焦点位置并驱动电机返回——本质上仍需一次完整的"对焦搜索"。

核心洞察：如果能在对焦过程中**实时检测**到焦点位置并立即停止电机，就无需捕获完整堆栈，实现真正的"一步对焦"。

## 方法详解

### 整体框架

ELP方法基于对焦过程中图像空间二阶导数与时间一阶导数之间的内在联系。系统监测ELP值的符号：正值表示正在趋近焦点，负值表示正在远离焦点，符号突变表示已到达焦点位置。对于仅有事件的相机，先通过EvTemMap获取灰度图像，再结合事件流计算ELP。

### 关键设计1：Event Laplacian Product (ELP)对焦检测函数

- **功能**: 实时检测焦点位置，判断是否正在趋近或远离焦点
- **核心思路**: 定义$\text{ELP}(t) = -\sum(\nabla^2 I(t) \cdot E(t))$，其中$\nabla^2 I(t)$是灰度图像的拉普拉斯算子，$E(t)$是事件帧。理论推导表明$S(t) = -\int \frac{\partial G}{\partial t} \cdot \frac{\partial^2 G}{\partial x^2} dx = -\alpha[\int(F * \frac{\partial^2 h}{\partial x^2})^2 dx]$，其符号完全由$\alpha$（高斯模糊方差的变化率）决定
- **设计动机**: 传统"峰值型"对焦评估函数需要搜索最大值，而ELP的"符号突变"特性使其成为检测函数——只需检测一次从正到负的跳变即可确定焦点，无需遍历整个堆栈

### 关键设计2：ELP自适应滤波器

- **功能**: 抑制ELP值的局部波动同时保留焦点处的陡峭跳变
- **核心思路**: 计算过去$W$个ELP值的均值$\overline{ELP}$，若当前值偏差小于阈值$ELP_{\text{thd}}$则进行滑动平均平滑（因子$S$），否则保留原值。此条件判断确保在远离焦点时平滑波动，在接近焦点时保留突变的陡峭性
- **设计动机**: 短时间间隔$\Delta t$（<1ms）虽提高灵敏度但引入噪声波动。简单低通滤波会模糊符号突变点。自适应策略兼顾了抗噪和灵敏度

### 关键设计3：纯事件一步AF管线

- **功能**: 为仅有事件输出的相机（如Prophesee EVK4）实现完整的一步AF
- **核心思路**: 两阶段管线——(1) 光圈打开阶段：使用EvTemMap将事件时间戳映射为HDR灰度图像（20ms）；(2) 对焦阶段：用户选择ROI，电机移动中连续计算ELP值，检测符号突变停止（~100ms）
- **设计动机**: 纯事件相机无法直接获取灰度图像，EvTemMap提供了一种利用主动透射率调制获取灰度信息的方法，使ELP可扩展到纯事件相机

### 损失函数

ELP是基于物理原理推导的检测函数，无需训练。

## 实验关键数据

### 合成数据集MAE对比（μm）

| 方法 | 静态 | 中等运动 | 剧烈运动 |
|------|------|---------|---------|
| EGS | 33.31 | 29.33 | 21.78 |
| PBF | 4.93 | 3.99 | 2.69 |
| ELP (1FPS) | 2.00 | 3.66 | 3.66 |
| ELP (20FPS) | 2.00 | 2.40 | 2.97 |
| **ELP (50FPS)** | **2.00** | **1.51** | **2.26** |

### 真实数据集对焦误差对比

| 数据集 | EGS误差 | PBF误差 | ELP误差 | ELP降低倍数 |
|--------|---------|---------|---------|-----------|
| DAVIS346 | 大 | 中 | 最小 | 24× |
| EVK4 | 大 | 中 | 最小 | 22× |

### 关键发现

- ELP一步AF将对焦时间减少2/3（无需返回行程）
- EGS在28.6%的合成场景中无法提供焦点位置，68.3%的情况误差超过一个焦深
- ELP即使只用单帧灰度图像（1FPS）仍能在静态场景中达到最优
- 自适应滤波器有效抑制了剧烈抖动引起的ELP波动

## 亮点与洞察

1. **从优化问题到检测问题的范式转变**: 将对焦从搜索峰值的优化问题转化为检测符号突变的检测问题，是核心理论贡献
2. **事件+灰度的互补性**: 事件提供高时间分辨率的亮度变化信息，灰度提供空间结构信息（拉普拉斯），两者乘积恰好产生焦点处的判别信号
3. **物理原理驱动设计**: 基于对焦光学的严格理论推导，而非数据驱动

## 局限与展望

- 需要灰度图像提供拉普拉斯信息，纯事件方案需额外的EvTemMap阶段
- 理论推导假设场景变化远慢于对焦过程，极端动态场景可能受限
- 当前仅在单一焦平面ROI上操作，扩展到全画面多区域对焦值得探索

## 相关工作与启发

- ELP的时空导数关系可能启发其他需要实时检测光学状态变化的应用
- 事件相机在计算摄影中的应用潜力巨大，本文展示了纯物理方法的优势

## 评分

⭐⭐⭐⭐⭐ — 理论严谨，物理直觉清晰，"检测vs搜索"的范式转变是真正的技术突破。对焦误差降低22-24倍的实验结果令人印象深刻。事件相机+传统光学的结合展示了计算摄影的优美一面。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Vision-Language Gradient Descent-driven All-in-One Deep Unfolding Networks](vision-language_gradient_descent-driven_all-in-one_deep_unfolding_networks.md)
- [\[CVPR 2026\] FiDeSR: High-Fidelity and Detail-Preserving One-Step Diffusion Super-Resolution](../../CVPR2026/image_restoration/fidesr_high-fidelity_and_detail-preserving_one-step_diffusion_super-resolution.md)
- [\[CVPR 2025\] INFP: Audio-Driven Interactive Head Generation in Dyadic Conversations](infp_audio-driven_interactive_head_generation_in_dyadic_conversations.md)
- [\[CVPR 2025\] Degradation-Aware Feature Perturbation for All-in-One Image Restoration](degradation-aware_feature_perturbation_for_all-in-one_image_restoration.md)
- [\[CVPR 2025\] Visual-Instructed Degradation Diffusion for All-in-One Image Restoration](visual-instructed_degradation_diffusion_for_all-in-one_image_restoration.md)

</div>

<!-- RELATED:END -->
