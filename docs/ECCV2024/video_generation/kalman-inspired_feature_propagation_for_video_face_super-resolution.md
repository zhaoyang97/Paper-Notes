---
title: >-
  [论文解读] Kalman-Inspired Feature Propagation for Video Face Super-Resolution
description: >-
  [ECCV 2024][视频生成][视频人脸超分辨率] 本文提出 KEEP 框架，借鉴卡尔曼滤波原理在隐空间中递归融合前帧先验与当前帧观测，实现视频人脸超分辨率中面部细节的高保真恢复与时序一致性，在 VFHQ 数据集上 PSNR 超过此前最优方法 0.8 dB。 人脸图像超分辨率（FSR）近年来取得了显著进展…
tags:
  - "ECCV 2024"
  - "视频生成"
  - "视频人脸超分辨率"
  - "卡尔曼滤波"
  - "时序一致性"
  - "CodeFormer"
  - "特征传播"
---

# Kalman-Inspired Feature Propagation for Video Face Super-Resolution

**会议**: ECCV 2024  
**arXiv**: [2408.05205](https://arxiv.org/abs/2408.05205)  
**代码**: [https://github.com/jnjaby/KEEP](https://github.com/jnjaby/KEEP)  
**领域**: 视频生成  
**关键词**: 视频人脸超分辨率, 卡尔曼滤波, 时序一致性, CodeFormer, 特征传播

## 一句话总结
本文提出 KEEP 框架，借鉴卡尔曼滤波原理在隐空间中递归融合前帧先验与当前帧观测，实现视频人脸超分辨率中面部细节的高保真恢复与时序一致性，在 VFHQ 数据集上 PSNR 超过此前最优方法 0.8 dB。

## 研究背景与动机
人脸图像超分辨率（FSR）近年来取得了显著进展，各种先验（几何先验、参考先验、生成先验、码本先验）被成功应用。然而视频人脸超分辨率（VFSR）仍然相对欠探索。现有方案面临两难困境：一类方法是将通用视频超分网络（如 BasicVSR）应用于人脸数据，但它们不针对人脸设计，在严重退化时无法重建精细面部细节；另一类方法逐帧应用人脸图像 SR 模型（如 CodeFormer），虽然单帧质量好，但帧间存在严重的时序不一致——因为 FSR 本身是一个病态问题，同一退化图像可对应多种高分辨率解释。核心矛盾在于如何在保持面部生成质量的同时维护视频的时序连贯性。本文的切入点是：已恢复的帧可以作为"参考"来引导和约束当前帧的恢复，这种利用历史信息递归更新的思想恰好与卡尔曼滤波原理高度契合。核心 idea：在 CodeFormer 的隐空间中引入卡尔曼滤波框架，通过预测-更新机制递归传播稳定的面部先验。

## 方法详解

### 整体框架
KEEP 基于 CodeFormer 构建，整体 pipeline 包含四个模块：LQ 编码器 $\mathcal{E}_L$、带码本查找的解码器 $\mathcal{D}_Q$、卡尔曼滤波网络（KFN）以及跨帧注意力（CFA）。在每个时间步，LQ 编码器将当前退化帧编码为观测状态 $\tilde{z}_t$；状态动态系统利用前帧的后验估计 $\hat{z}_{t-1}^+$ 通过光流 warp 和 HQ 编码器得到先验预测 $\hat{z}_t^-$；卡尔曼滤波网络融合两者得到更准确的后验估计 $\hat{z}_t^+$；最终由解码器生成高质量输出帧。

### 关键设计
1. **状态空间建模 (State Space Model)**:
    - 功能：将 VFSR 问题形式化为隐空间中的状态估计问题
    - 核心思路：将卡尔曼滤波的线性状态空间模型推广为非线性形式，通过生成模型 $g_\theta$ 将隐状态 $z_t$ 映射到高质量帧 $y_t = g_\theta(z_t)$，用 LQ 编码器作为观测估计器 $\tilde{z}_t = \mathcal{E}_L(x_t)$，用光流 warp + HQ 编码器实现状态转移 $\hat{z}_t^- = \mathcal{E}_H(\omega(\mathcal{D}_Q(\hat{z}_{t-1}^+), \Phi_{t-1 \to t}))$
    - 设计动机：在低维隐空间建模比直接在像素空间操作更高效，且隐空间的表示更关注感知显著的变化；卡尔曼滤波的预测-更新两步机制天然适合利用时序信息递归修正噪声估计

2. **卡尔曼增益网络 (Kalman Gain Network, KGN)**:
    - 功能：自适应地融合先验预测 $\hat{z}_t^-$ 和当前观测 $\tilde{z}_t$
    - 核心思路：后验状态通过线性插值更新 $\hat{z}_t^+ = \mathcal{K}_t \hat{z}_t^- + (1 - \mathcal{K}_t) \tilde{z}_t$，其中卡尔曼增益 $\mathcal{K}_t$ 由 KGN 直接从数据分布学习，而不是显式维护协方差矩阵。KGN 包含不确定性网络（使用时空注意力估计预测的不确定性）和增益网络（计算每个码本 token 的增益值），并引入首帧 $\tilde{z}_1$ 作为锚点
    - 设计动机：高维信号的协方差估计不可行（KalmanNet 的结论），且协方差仅用于计算增益，因此直接学习增益更实际；锚点机制有助于防止长期漂移

3. **跨帧注意力 (Cross-Frame Attention, CFA)**:
    - 功能：在解码器中促进局部时序一致性
    - 核心思路：在解码器的小尺度特征（$16 \times 16$ 和 $32 \times 32$）上使用交叉注意力，当前帧特征 $v_t$ 作为 Query，前帧特征 $v_{t-1}$ 作为 Key 和 Value，搜索匹配前帧中的相似 patch 并融合
    - 设计动机：KFN 在隐码层面保证全局风格一致，但局部纹理细节（如头发）仍需在解码器特征层面进行传播；选择小尺度特征是为了避免引入模糊

### 损失函数 / 训练策略
三阶段训练策略：Stage I 训练 800k 迭代（lr=$2 \times 10^{-4}$），Stage II 训练 400k 迭代（lr=$2 \times 10^{-4}$），Stage III 训练 50k 迭代（lr=$1 \times 10^{-4}$）。损失函数包含像素级 L1 损失（$\lambda_1 = 10^{-2}$）、VGG 感知损失（$\lambda_{VGG} = 1$）和 GAN 对抗损失（$\lambda_{GAN} = 10^{-2}$）。使用 GMFlow 进行光流估计。人脸对齐阶段对关键点采用高斯低通滤波以抑制对齐步骤引入的时序不一致。

## 实验关键数据

### 主实验
在 VFHQ-mild 测试集上的定量对比：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | IDS↑ | AKD↓ | σ_IDS↓ | σ_AKD↓ |
|------|-------|-------|--------|------|------|--------|--------|
| GPEN | 25.52 | 0.752 | 0.299 | 0.714 | 11.47 | 4.74 | 3.51 |
| GFPGAN | 26.29 | 0.780 | 0.248 | 0.744 | 10.55 | 4.57 | 3.65 |
| CodeFormer | 24.66 | 0.745 | 0.274 | 0.627 | 11.50 | 6.37 | 3.69 |
| BasicVSR++ | 27.20 | 0.806 | 0.196 | 0.764 | 11.31 | 5.25 | 4.64 |
| **KEEP (Ours)** | **27.99** | **0.827** | **0.162** | **0.796** | **8.82** | **3.69** | **3.25** |

KEEP 在所有指标上均取得最优，PSNR 超出次优方法 BasicVSR++ 约 0.8 dB，时序一致性指标（σ_IDS、σ_AKD、AKD）大幅领先。

### 消融实验

| 配置 | LPIPS↓ | IDS↑ | AKD↓ | 说明 |
|------|--------|------|------|------|
| w/o CFA | 0.1621 | 0.7970 | 8.90 | 移除跨帧注意力，局部纹理一致性下降 |
| w/o KFN | 0.1721 | 0.7773 | 9.20 | 移除卡尔曼滤波网络，时序一致性显著下降 |
| Full model | 0.1619 | 0.7960 | 8.82 | 完整模型最优 |
| PWC-Net flow | 0.1623 | 0.7957 | 8.78 | 换用 PWC-Net 光流，性能差异极小 |
| GMFlow (Ours) | 0.1619 | 0.7960 | 8.82 | 光流精度对最终性能影响不大 |

### 关键发现
- KFN 是保证全局风格和身份一致性的关键，CFA 进一步提升局部纹理的时序连贯性
- 光流估计的精度对最终性能影响不大，因为隐码被下采样了 32 倍，小的空间偏差在此尺度下可忽略
- 在严重退化场景下优势尤其明显：单帧模型性能急剧下降，而 KEEP 能利用帧间互补信息保持稳健
- 对非正面人脸更鲁棒：稳定的先验估计使模型在侧脸等挑战性视角下仍能产出合理结果
- 身份相似度跨帧波动显著低于 CodeFormer，避免了身份突变

## 亮点与洞察
- 将经典信号处理中的卡尔曼滤波思想引入生成式人脸恢复，建立了优雅的理论框架，将"如何融合历史信息与当前观测"以概率估计的视角形式化
- 方法通用性强：虽然以 CodeFormer 为 case study，但底层的卡尔曼滤波框架可以推广到任何基于编码器-码本-解码器的人脸恢复模型
- 通过锚点机制（首帧 $\tilde{z}_1$）有效缓解了递归传播中的长期漂移问题
- 人脸对齐时使用高斯低通滤波关键点这一工程细节虽然简单但很有效

## 局限与展望
- 目前仅支持单向（前向）递归传播，无法利用未来帧信息，双向传播可能进一步提升质量
- 依赖预训练的 CodeFormer 作为骨干，受限于其码本的表达能力
- 光流估计在大运动或遮挡场景下可能不准确，虽然论文指出影响不大但极端场景仍存疑
- 三阶段训练流程相对复杂，端到端训练可能更高效
- 实验仅在 VFHQ 数据集上验证，缺乏在更多真实场景视频上的系统评测

## 相关工作与启发
- **BasicVSR/BasicVSR++**: 利用双向隐状态传播的通用视频超分方法，但不含人脸特定先验
- **CodeFormer**: 基于码本先验的单帧人脸恢复，本文在其基础上引入时序传播
- **KalmanNet**: 提出直接从数据学习卡尔曼增益的思想，启发了本文的 KGN 设计
- **Tune-A-Video**: 跨帧注意力机制被本文采用以增强局部时序一致性
- 启发：将经典控制/信号处理理论与深度生成模型结合是一个有前景的方向

## 评分
- 新颖性: ⭐⭐⭐⭐ 卡尔曼滤波与生成式人脸恢复的结合思路新颖，但各组件并非全新
- 实验充分度: ⭐⭐⭐⭐ 多级退化测试、多维度消融、定性分析详尽，但仅在一个数据集上验证
- 写作质量: ⭐⭐⭐⭐⭐ 公式推导清晰，从卡尔曼滤波到具体实现的逻辑链完整且优雅
- 价值: ⭐⭐⭐⭐ 为视频人脸恢复提供了一个有理论支撑的通用框架，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] RealViformer: Investigating Attention for Real-World Video Super-Resolution](realviformer_investigating_attention_for_real-world_video_super-resolution.md)
- [\[CVPR 2025\] VideoGigaGAN: Towards Detail-rich Video Super-Resolution](../../CVPR2025/video_generation/videogigagan_towards_detail-rich_video_super-resolution.md)
- [\[CVPR 2025\] PatchVSR: Breaking Video Diffusion Resolution Limits with Patch-Wise Video Super-Resolution](../../CVPR2025/video_generation/patchvsr_breaking_video_diffusion_resolution_limits_with_patch-wise_video_super-.md)
- [\[CVPR 2026\] VSRELL: A Simple Baseline for Video Super-Resolution and Enhancement in Low-Light Environment](../../CVPR2026/video_generation/vsrell_a_simple_baseline_for_video_super-resolution_and_enhancement_in_low-light.md)
- [\[ICCV 2025\] VSRM: A Robust Mamba-Based Framework for Video Super-Resolution](../../ICCV2025/video_generation/vsrm_a_robust_mamba-based_framework_for_video_super-resolution.md)

</div>

<!-- RELATED:END -->
