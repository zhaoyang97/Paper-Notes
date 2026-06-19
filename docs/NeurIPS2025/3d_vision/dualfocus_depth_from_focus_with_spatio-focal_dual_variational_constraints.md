---
title: >-
  [论文解读] DualFocus: Depth from Focus with Spatio-Focal Dual Variational Constraints
description: >-
  [NeurIPS 2025][3D视觉][Depth from Focus] 提出 DualFocus，通过空间变分约束（利用焦距相关梯度模式区分深度边缘与纹理伪影）和焦距变分约束（强制单峰单调的对焦概率分布）双重约束，实现从焦距堆栈中鲁棒精确的深度估计。 Depth-from-Focus (DFF) 利用焦距堆栈（不同焦距…
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "Depth from Focus"
  - "变分约束"
  - "焦距堆栈"
  - "深度估计"
  - "空间-焦距双约束"
---

# DualFocus: Depth from Focus with Spatio-Focal Dual Variational Constraints

**会议**: NeurIPS 2025  
**arXiv**: [2509.21992](https://arxiv.org/abs/2509.21992)  
**代码**: 未开源  
**领域**: 3D视觉  
**关键词**: Depth from Focus, 变分约束, 焦距堆栈, 深度估计, 空间-焦距双约束  

## 一句话总结

提出 DualFocus，通过空间变分约束（利用焦距相关梯度模式区分深度边缘与纹理伪影）和焦距变分约束（强制单峰单调的对焦概率分布）双重约束，实现从焦距堆栈中鲁棒精确的深度估计。

## 背景与动机

Depth-from-Focus (DFF) 利用焦距堆栈（不同焦距拍摄的图像序列）中的对焦线索进行深度估计，具有无需特殊硬件、无尺度歧义等优势。然而现有学习方法存在两个关键问题：

1. **纹理-深度边缘混淆**：直接从图像特征回归深度时，强纹理梯度容易被误判为深度不连续，尤其在重复纹理区域
2. **焦距维度建模不足**：现有方法通常独立处理各焦平面或仅做有限正则化，未充分利用对焦概率沿焦距轴的物理连续性（应呈单峰分布）

核心 insight：焦距堆栈中同一场景点在不同焦平面上会呈现不同的梯度模式——对焦区域梯度一致且强，失焦区域梯度弥散或噪声化。利用这种跨焦平面的梯度差异可以间接推断锐度，区分真正的深度边缘。

## 核心问题

如何在 DFF 中同时利用空间域和焦距域的物理先验，提升复杂场景（细纹理、深度突变）下的深度估计精度和鲁棒性？

## 方法详解

### 焦距体积建模

给定 $N$ 张不同焦距图像，提取特征后沿焦距维度堆叠形成 4D 焦距体积 $V \in \mathbb{R}^{H \times W \times C_1 \times N}$。计算焦距维度差分并拼接，得到增强体积：

$$V_n^* = \begin{cases} [V_n, V_{n+1} - V_n], & n = 1, \ldots, N-1 \\ [V_n, V_n - V_{n-1}], & n = N \end{cases}$$

### 空间变分约束

网络预测每个焦平面的多通道梯度特征 $\Gamma_n \in \mathbb{R}^{2HW \times C_2}$，编码 x/y 方向的深度变化线索。为确保全局可积性（避免无法对应真实表面的噪声梯度），通过最小二乘投影到可积梯度场：

$$z_n^{*(c)} = \arg\min_z \|Pz - \Gamma_n^{(c)}\|_2^2 = (P^\top P)^{-1} P^\top \Gamma_n^{(c)}$$

其中 $P$ 为固定有限差分算子。重建的隐式表面 $z_n^*$ 在对焦平面产生一致的几何结构，在失焦平面产生噪声表面——这种差异自然编码了几何线索的可靠性。

仅在对焦区域监督 $z_n^*$，定义逐像素逐平面的锐度权重：

$$q_n(\mathbf{x}) = \frac{\exp(-|f_n - D^*(\mathbf{x})|)}{\sum_{m=1}^N \exp(-|f_m - D^*(\mathbf{x})|)}$$

空间变分损失：

$$L_{\text{sv}} = \sum_{\mathbf{x},n} q_n(\mathbf{x}) \|\nabla D^*(\mathbf{x}) - \theta_{\text{grad}}(z_n^*)(\mathbf{x})\|_1$$

### 焦距变分约束

对焦概率 $p_n(\mathbf{x})$ 应在正确深度处达峰值并向两侧单调递减。定义双向软单调性损失：

$$L_{\text{fv}} = \sum_{\mathbf{x}} \left( \sum_{i=1}^{k(\mathbf{x})-1} (\max(0, p_i - p_{i+1}))^2 + \sum_{i=k(\mathbf{x})}^{N-1} (\max(0, p_{i+1} - p_i))^2 \right)$$

其中 $k(\mathbf{x}) = \arg\max_n p_n(\mathbf{x})$。峰值前惩罚递减，峰值后惩罚递增。

### 深度融合与总损失

重建表面特征与焦距体积拼接，经 3D 卷积解码得到对焦概率图 $p \in \mathbb{R}^{H' \times W' \times N}$，加权求和焦距值得到深度：

$$\hat{D}(\mathbf{x}) = \sum_{n=1}^N p_n(\mathbf{x}) f_n$$

总损失 $L = L_{\text{depth}} + \lambda_{\text{sv}} L_{\text{sv}} + \lambda_{\text{fv}} L_{\text{fv}}$，其中 $L_{\text{depth}}$ 为 smooth L1 损失。

## 实验关键数据

### NYU Depth v2（合成焦距堆栈）

| 方法 | 类型 | RMSE ↓ | AbsRel ↓ | δ₁ ↑ |
|------|------|--------|----------|------|
| Depth Anything | SIDE | 0.206 | 0.056 | 0.984 |
| HybridDepth | DFF | 0.128 | 0.026 | 0.995 |
| DFV | DFF | 0.094 | 0.020 | 0.998 |
| **DualFocus** | **DFF** | **0.075** | **0.013** | **0.999** |

相比 DFV：RMSE 降低 20.2%，AbsRel 降低 35.0%。

### FoD500

| 方法 | MSE ↓ | RMSE ↓ | Bump ↓ |
|------|-------|--------|--------|
| DFV | 0.020 | 0.129 | 1.43 |
| **DualFocus** | **0.015** | **0.112** | **1.31** |

### DDFF 12-Scene

| 方法 | MSE ↓ | RMSE ↓ | δ₁ ↑ |
|------|-------|--------|------|
| HybridDepth | 5.1×10⁻⁴ | 0.0200 | 0.789 |
| **DualFocus** | **4.7×10⁻⁴** | **0.0194** | **0.800** |

### 零样本迁移（ARKitScenes）

| 方法 | 类型 | RMSE ↓ | AbsRel ↓ | 参数量 |
|------|------|--------|----------|--------|
| Depth Anything | SIDE | 0.53 | 0.32 | 336M |
| HybridDepth | DFF | 0.29 | 0.42 | 67M |
| **DualFocus** | **DFF** | **0.28** | **0.40** | **27M** |

### 消融实验

去掉两个约束后 RMSE 从 0.075 升至 0.094；空间约束贡献大于焦距约束，因为它直接编码了每个焦平面的表面梯度信息。

## 亮点

1. 将梯度场投影到可积空间的做法优雅——不仅正则化梯度，还自然编码了对焦/失焦的区别
2. 焦距变分约束利用物理先验（对焦概率单峰性），约束形式简洁且有效
3. 仅 27M 参数，远小于 SIDE 模型（336M），零样本迁移性能更好
4. 在四个数据集上全面 SOTA

## 局限与展望

- NYU 数据集使用合成焦距堆栈，与真实焦点扫描存在 domain gap
- 对焦堆栈需要 N 帧输入，实时应用受限
- 对极端纹理缺失区域（如纯白墙面）的表现未详细分析
- 未与最新的 Depth Anything V2 等大模型对比

## 与相关工作的对比

- **vs DFV**: DFV 仅捕获焦距维度一阶导数，DualFocus 同时建模空间梯度变化和焦距概率分布
- **vs HybridDepth**: HybridDepth 依赖预训练相对深度模型，DualFocus 纯端到端，参数更少
- **vs VA-DepthNet**: VA-DepthNet 在单图上用变分约束，DualFocus 将其扩展到焦距堆栈，利用多焦平面的梯度差异

## 启发与关联

变分约束思路可推广到其他多视角/多条件深度估计任务。将可积性约束作为归纳偏置融入网络训练是一种值得借鉴的物理先验注入方式。

## 评分

- ⭐ 新颖性: 8/10 — 空间+焦距双变分约束设计新颖，可积性投影在 DFF 中首次使用
- ⭐ 实验充分度: 8/10 — 四个数据集+零样本迁移+消融，但缺少真实焦距堆栈大规模验证
- ⭐ 写作质量: 8/10 — 公式推导清晰，动机阐述完整
- ⭐ 价值: 7/10 — DFF 领域强工作，但应用场景相对窄

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Simulating Dual-Pixel Images From Ray Tracing For Depth Estimation](../../ICCV2025/3d_vision/simulating_dual-pixel_images_from_ray_tracing_for_depth_estimation.md)
- [\[ICML 2026\] FSI2P: A Hierarchical Focus–Sweep Registration Network with Dynamically Allocated Depth](../../ICML2026/3d_vision/fs-i2pa_hierarchical_focus-sweep_registration_network_with_dynamically_allocated.md)
- [\[NeurIPS 2025\] 3D Visual Illusion Depth Estimation](3d_visual_illusion_depth_estimation.md)
- [\[ECCV 2024\] IDOL: Unified Dual-Modal Latent Diffusion for Human-Centric Joint Video-Depth Generation](../../ECCV2024/3d_vision/idol_unified_dual-modal_latent_diffusion_for_human-centric_joint_video-depth_gen.md)
- [\[CVPR 2026\] Variational Graph-based Normal Integration](../../CVPR2026/3d_vision/variational_graph-based_normal_integration.md)

</div>

<!-- RELATED:END -->
