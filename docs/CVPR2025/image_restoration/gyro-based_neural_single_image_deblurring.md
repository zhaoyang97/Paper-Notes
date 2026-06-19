---
title: >-
  [论文解读] Gyro-based Neural Single Image Deblurring
description: >-
  [CVPR 2025][图像恢复][gyro sensor] 提出 GyroDeblurNet，通过新颖的相机运动场嵌入表示复杂手抖、陀螺仪细化模块利用图像模糊信息校正陀螺仪误差、陀螺仪去模糊模块用校正后的运动信息去除模糊，配合课程学习策略，在合成和真实数据集上大幅超越现有方法。 领域现状： 单图去模糊由于严重的不适定性仍极…
tags:
  - "CVPR 2025"
  - "图像恢复"
  - "gyro sensor"
  - "image deblurring"
  - "camera motion field"
  - "curriculum learning"
  - "deformable convolution"
---

# Gyro-based Neural Single Image Deblurring

**会议**: CVPR 2025  
**arXiv**: [2404.00916](https://arxiv.org/abs/2404.00916)  
**代码**: 待确认  
**领域**: 图像复原  
**关键词**: gyro sensor, image deblurring, camera motion field, curriculum learning, deformable convolution

## 一句话总结

提出 GyroDeblurNet，通过新颖的相机运动场嵌入表示复杂手抖、陀螺仪细化模块利用图像模糊信息校正陀螺仪误差、陀螺仪去模糊模块用校正后的运动信息去除模糊，配合课程学习策略，在合成和真实数据集上大幅超越现有方法。

## 研究背景与动机

**领域现状**: 单图去模糊由于严重的不适定性仍极具挑战。最近的 DNN 方法在大模糊上仍然失败。手机内置陀螺仪提供了宝贵的相机运动信息，可帮助缓解不适定性。

**现有痛点**:
- **陀螺仪数据不准确**: 真实陀螺仪信号包含噪声、旋转中心偏差、缺少平移运动信息，导致陀螺仪编码的运动与图像实际模糊不一致（gyro error）
- **运动表示过于简单**: 现有方法（DeepGyro、EggNet、INformer）用每像素 1-2 个向量或少量 homography 表示相机运动，无法捕捉时间复杂的手抖模式
- **训练数据不真实**: 现有数据集的陀螺仪数据来自随机采样或 Visual-Inertial 数据集（持续运动），不反映拍照时的手抖特征

**核心矛盾**: 如何在陀螺仪数据包含大误差的情况下，仍能有效利用其提供的运动信息来辅助去模糊？

**本文切入角度**: 不要求陀螺仪数据准确，而是设计网络架构主动处理误差——先用图像模糊信息细化陀螺仪数据，再用细化后的数据指导去模糊。

## 方法详解

### 整体框架

GyroDeblurNet 包含两个模块：
1. **图像去模糊模块**: U-Net 架构（NAFBlock 为基本单元），bottleneck 处使用 Gyro Deblurring Block
2. **陀螺仪模块**: 卷积层嵌入运动场 → 多个 Gyro Refinement Block（利用图像特征细化）→ 步进卷积降分辨率

输入：模糊图像 $B$ + 相机运动场 $\mathcal{V}$，输出：残差 $R$，$D = B + R$

### 关键设计

**1. 相机运动场（Camera Motion Field）嵌入**
- **功能**: 将任意长度的陀螺仪数据序列 $G = \{g_0, ..., g_{T-1}\}$ 转化为固定大小的张量 $\mathcal{V} \in \mathbb{R}^{W/s \times H/s \times 2M}$
- **核心思路**:
    - 三次样条插值将 $T$ 个陀螺仪采样重采样为 $M+1$ 个
    - 积分获得 $M+1$ 个相机朝向 → 计算 $M+1$ 个 homography $H_m = KR(\theta_m)K^{-1}$
    - 对每像素计算连续时刻间的位移向量，堆叠得到 $M$ 个 2D 向量（$2M$ 通道）
    - 利用相机抖动的空间平滑性，以 $s=2$ 降采样
- **设计动机**: $M=8$ 个向量（比现有 1-2 个多得多）可捕捉时序复杂的手抖，同时固定通道大小兼容 CNN；空间降采样减少内存

**2. Gyro Refinement Block（陀螺仪细化模块）**
- **功能**: 利用图像编码器的特征来全局校正陀螺仪特征中的误差
- **核心思路**:
    - 输入陀螺仪特征编码了多个带扰动的运动候选
    - 将图像特征与陀螺仪特征拼接 → 全局平均池化 + 1×1 卷积 → 通道权重
    - 用通道权重选择与图像模糊一致的运动候选通道
- **设计动机**: 陀螺仪误差是全局性的（旋转中心偏差、噪声），需要全局信息来校正；图像的模糊模式提供互补线索

**3. Gyro Deblurring Block（陀螺仪去模糊模块）**
- **功能**: 利用细化后的陀螺仪特征执行空间自适应去模糊
- **核心思路**: 两个子模块：
    - **Sub-block 1**: 拼接图像特征和陀螺仪特征 → 预测可变形卷积的 offset → 对图像特征执行空间自适应去卷积
    - **Sub-block 2**: 通过空间注意力（卷积+Sigmoid）进一步细化去模糊结果
- **设计动机**: 可变形卷积的 offset 同时由图像和陀螺仪信息驱动，即使陀螺仪有残留误差，图像信息也可以补偿

### 损失函数 / 训练策略

**课程学习策略（Curriculum Learning）**:
- 初始用无误差陀螺仪数据训练，逐步增加误差
- 混合运动场: $\mathcal{V}_\alpha = (1-\alpha)\mathcal{V}_{clean} + \alpha \mathcal{V}_{noisy}$
- $\alpha$ 从 0 渐增到 1
- 噪声运动场: 随机扰动旋转中心 + 添加从真实静止手机测量的高斯噪声

损失函数: PSNR Loss (即 Charbonnier Loss)

训练配置: 256×256 patch, batch 16, Adam, 300 epochs, cosine annealing lr

## 实验关键数据

### 主实验（GyroBlur-Synth + GyroBlur-Real-S）

| 方法 | 类别 | PSNR↑ | SSIM↑ | NIQE↓ | TOPIQ↑ |
|---|---|---|---|---|---|
| NAFNet | 单图 | 25.06 | 0.709 | 5.27 | 0.409 |
| FFTformer | 单图 | 26.01 | 0.748 | 4.98 | 0.434 |
| Stripformer | 单图 | 25.93 | 0.740 | 4.71 | 0.456 |
| DeepGyro | Gyro | 23.78 | 0.665 | 5.64 | 0.381 |
| EggNet | Gyro | 25.49 | 0.727 | 5.18 | 0.413 |
| INformer | Gyro | 25.11 | 0.710 | 5.29 | 0.408 |
| Nan et al. | Non-blind | 22.22 | 0.531 | 5.81 | 0.348 |
| **Ours** | **Gyro** | **27.28** | **0.780** | **4.47** | **0.548** |

PSNR 超越最佳单图方法 1.27 dB，超越最佳陀螺仪方法 1.79 dB。

### 消融实验

| 配置 | PSNR | SSIM | 说明 |
|---|---|---|---|
| (a) No gyro data | 24.90 | 0.700 | 无陀螺仪信息 |
| (b) Train w/ error-free | 24.94 | 0.711 | 用无误差数据训练，真实数据失效 |
| (c) No refinement | 25.47 | 0.713 | 有陀螺仪但不细化 |
| (d) Refine w/o image feat | 26.17 | 0.747 | 陀螺仪细化不用图像特征 |
| (e) Deform conv only gyro | 26.32 | 0.754 | 可变形卷积仅用陀螺仪特征 |
| (f) Full w/o curriculum | 26.94 | 0.767 | 完整模型无课程学习 |
| (g) **Full model** | **27.28** | **0.780** | 完整模型 |

### 关键发现

1. **用无误差数据训练几乎无效**: (b) 仅比 (a) 高 0.04 dB，说明模型学会忽略误差太大的陀螺仪数据
2. **图像特征指导细化至关重要**: (d) vs (c) 提升 0.7 dB，(f) vs (e) 提升 0.6 dB
3. **课程学习额外提升 0.34 dB**: 帮助模型逐步学会处理误差
4. **$M=8$ 是最佳时序分辨率**: $M=2$ → 25.71, $M=8$ → 27.28, $M=16$ → 27.32（收益递减）
5. **跨设备泛化**: 在 Huawei P30 Pro 设备上（训练用 Samsung Galaxy S22）仍优于其他方法

## 亮点与洞察

- "不要求传感器准确，而是让网络学会处理误差"的设计哲学优雅且实用
- 相机运动场嵌入是一个通用的陀螺仪数据表示方案，$M$ 参数灵活可调
- GyroBlur-Synth 数据集构建方案可扩展——仅需几分钟陀螺仪录制+简单标定
- 课程学习策略专为带噪辅助信号学习设计，具有通用借鉴价值

## 局限与展望

- $M$ 作为超参数固定，长曝光可能需要更大 $M$
- 未利用加速度计数据，可能提供额外运动信息
- 图像去模糊模块架构（NAFBlock U-Net）相对简单
- 仅验证手机陀螺仪，未测试其他 IMU 设备
- 运动物体的去模糊仅通过误差鲁棒性间接处理，未显式建模

## 相关工作与启发

- DeepGyro 首次将陀螺仪数据用于 DNN 去模糊，但假设准确数据
- EggNet 用可变形卷积适配陀螺仪数据但仅 1-2 个向量
- RSBlur 的模糊合成管线被本文采用来生成逼真合成数据
- 启发：传感器辅助的低级视觉任务中，传感器误差处理比传感器本身更重要

## 评分

⭐⭐⭐⭐ — 问题定义清晰（处理陀螺仪误差），技术方案有针对性（细化-去模糊-课程学习三环联动），实验设置严谨（合成+真实+跨设备），PSNR 提升幅度显著（+1.3 dB over best single-image）。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Gyro-based Deep Video Deblurring](../../CVPR2026/image_restoration/gyro-based_deep_video_deblurring.md)
- [\[CVPR 2025\] Progressive Focused Transformer for Single Image Super-Resolution](progressive_focused_transformer_for_single_image_super-resolution.md)
- [\[CVPR 2025\] Reversible Decoupling Network for Single Image Reflection Removal](reversible_decoupling_network_for_single_image_reflection_removal.md)
- [\[CVPR 2025\] Efficient Visual State Space Model for Image Deblurring](efficient_visual_state_space_model_for_image_deblurring.md)
- [\[CVPR 2025\] Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging](proximal_algorithm_unrolling_flexible_and_efficient_reconstruction_networks_for_.md)

</div>

<!-- RELATED:END -->
