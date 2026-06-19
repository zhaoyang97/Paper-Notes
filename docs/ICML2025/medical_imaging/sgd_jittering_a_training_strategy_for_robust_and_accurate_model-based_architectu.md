---
title: >-
  [论文解读] SGD Jittering: A Training Strategy for Robust and Accurate Model-Based Architectures
description: >-
  [ICML2025][医学图像][SGD jittering] 提出 SGD jittering 训练策略，在模型迭代重建过程中逐步注入零均值高斯噪声，理论证明其同时提升模型鲁棒性和泛化精度，且无需对抗训练的高计算开销。 逆问题（Inverse Problems）旨在从退化/不完整的观测 $\boldsymbol{y} =…
tags:
  - "ICML2025"
  - "医学图像"
  - "SGD jittering"
  - "model-based architecture"
  - "loop unrolling"
  - "鲁棒性"
  - "泛化"
  - "逆问题"
  - "MRI reconstruction"
---

# SGD Jittering: A Training Strategy for Robust and Accurate Model-Based Architectures

**会议**: ICML2025  
**arXiv**: [2410.14667](https://arxiv.org/abs/2410.14667)  
**代码**: 待确认  
**领域**: 模型鲁棒性 / 逆问题  
**关键词**: SGD jittering, model-based architecture, loop unrolling, 鲁棒性, 泛化, 逆问题, MRI reconstruction

## 一句话总结

提出 SGD jittering 训练策略，在模型迭代重建过程中逐步注入零均值高斯噪声，理论证明其同时提升模型鲁棒性和泛化精度，且无需对抗训练的高计算开销。

## 研究背景与动机

逆问题（Inverse Problems）旨在从退化/不完整的观测 $\boldsymbol{y} = \boldsymbol{A}\boldsymbol{x} + \boldsymbol{z}$ 中恢复原始信号 $\boldsymbol{x}$。模型基础架构（MBAs）如 loop unrolling 将经典优化迭代展开为可训练网络，比黑盒方法更具可解释性且重建质量更高。

然而，现有研究集中于重建精度，对鲁棒性和泛化能力关注不足：

- **鲁棒性**：模型在面对测量噪声或对抗攻击时的稳定性
- **泛化性**：模型在分布外（OOD）数据上的表现
- **两者的权衡**：对抗训练（AT）提升鲁棒性但牺牲精度，输入 jittering 也存在类似问题

作者观察到：AT 和输入 jittering 都在观测 $\boldsymbol{y}$ 上注入噪声，导致学习的逆映射偏离真实逆过程。例如在无噪声情况下，AT 学习 $H_\theta(\boldsymbol{y}+\boldsymbol{e}) \approx \boldsymbol{x}$，但真实解为 $\boldsymbol{A}^{-1}(\boldsymbol{y}+\boldsymbol{e}) = \boldsymbol{x} + \boldsymbol{A}^{-1}\boldsymbol{e}$，AT 忽略后者会引入泛化误差。

## 方法详解

### 核心思想：SGD Jittering

在 MBA 的迭代重建过程中，不对输入 $\boldsymbol{y}$ 注入噪声，而是在每一步梯度更新时加入独立的零均值高斯噪声 $\boldsymbol{w}_k$：

$$\boldsymbol{x}_{k+1} = \boldsymbol{x}_k - \eta \left( \boldsymbol{A}^\top(\boldsymbol{A}\boldsymbol{x}_k - \boldsymbol{y}) + f_\theta(\boldsymbol{x}_k) + \boldsymbol{w}_k \right)$$

其中 $\boldsymbol{w}_k \sim \mathcal{N}(0, \sigma_{w_k}^2/n \cdot \boldsymbol{I})$，每次迭代独立采样。推理时移除噪声，模型按标准方式运行。

### 训练损失

SGD jittering 最小化如下风险：

$$J^{SGD}_{\sigma_{w_k}}(\theta) = \mathbb{E}_{\bar{\boldsymbol{w}}, (\boldsymbol{x},\boldsymbol{y})\sim\mathcal{D}} \|\boldsymbol{x} - \hat{\boldsymbol{x}}\|_2^2$$

关键区别在于下层优化目标与标准 MSE 训练完全一致，噪声仅影响求解路径而非目标函数本身。

### 隐式正则化

SGD jittering 的训练损失隐含正则化项：

$$\text{regularization} = \mathbb{E}\left\|\sum_{i=0}^{K-1} \eta(1-\eta)^{K-1-i} \left(f_\theta(\boldsymbol{x}_i') - f_\theta(\boldsymbol{x}_i^{sgd})\right)\right\|_2^2$$

该正则化惩罚了噪声引起的中间重建偏差，促使网络 $f_\theta$ 在输入空间中具有更平坦的 Hessian，从而提升稳定性。

### 理论保证

- **泛化**（Theorem 7.4）：$\mathcal{G}(\theta^{sgd}) \leq \mathcal{G}(\theta^{mse})$，SGD jittering 的泛化风险不超过标准 MSE 训练
- **鲁棒性**（Theorem 7.5）：$R_e(\theta^{sgd}) \leq R_e(\theta^{mse})$，SGD jittering 对平均攻击更鲁棒
- **收敛性**（Corollary 6.2）：MBA-SGD 以 $O(1/\sqrt{K})$ 速率收敛到驻点

### 与其他方法的对比

| 方法 | 噪声位置 | 推理时噪声 | 泛化 | 鲁棒性 | 训练速度 |
|------|----------|-----------|------|--------|---------|
| MSE 训练 | 无 | 无 | 基线 | 基线 | 快 |
| 对抗训练（AT） | 输入 $\boldsymbol{y}$ | 无 | 差 | 最优（worst-case） | 慢 |
| 输入 Jittering | 输入 $\boldsymbol{y}$ | 无 | 差 | 中等 | 快 |
| **SGD Jittering** | **迭代梯度** | **无** | **最优** | **中上** | **快** |

## 实验关键数据

### 地震反卷积（Seismic Deconvolution）

| 方法 | ID 数据 (PSNR/SSIM) | 对抗攻击 (PSNR/SSIM) | OOD 数据 (PSNR/SSIM) |
|------|---------------------|---------------------|---------------------|
| MSE 训练 | 34.64 / 0.921 | 28.85 / 0.829 | 34.57 / 0.918 |
| AT | 33.50 / 0.903 | **30.27 / 0.849** | 33.45 / 0.902 |
| 输入 Jittering | 32.92 / 0.882 | 30.10 / 0.832 | 32.91 / 0.882 |
| **SGD Jittering** | **35.10 / 0.928** | 29.89 / 0.842 | **34.93 / 0.927** |

### 加速 MRI 重建（4× 加速）

| 方法 | fastMRI (PSNR/SSIM) | 对抗攻击 (PSNR/SSIM) | 肿瘤 OOD (PSNR/SSIM) |
|------|---------------------|---------------------|---------------------|
| MSE 训练 | 28.21 / 0.603 | 25.68 / 0.382 | 29.92 / 0.779 |
| AT | 27.68 / 0.564 | **27.17 / 0.549** | 27.74 / 0.597 |
| 输入 Jittering | 28.18 / 0.595 | 25.05 / 0.420 | 29.97 / 0.740 |
| **SGD Jittering** | **28.22 / 0.607** | 26.77 / 0.552 | **30.36 / 0.788** |

**关键发现**：SGD jittering 在 ID 和 OOD 数据上均取得最佳性能，对抗攻击下仅次于 AT 但远优于 MSE 和输入 jittering。AT 在 OOD 肿瘤数据上 PSNR 比 SGD jittering 低 2.62 dB，说明 AT 严重牺牲泛化精度。

## 亮点与洞察

1. **简单而有效**：仅在训练时迭代中加噪声，推理零开销，几行代码即可实现
2. **理论完备**：首次为逆问题中的泛化精度给出形式化定义和理论分析，填补了隐层输入平坦性→泛化的理论空白
3. **噪声注入位置的深刻洞察**：输入级噪声（AT/input jittering）改变了优化目标导致泛化损失，而迭代级噪声保持目标不变仅影响求解路径
4. **医学影像价值**：在 MRI 肿瘤 OOD 数据上的显著优势对临床安全性至关重要
5. **SPGD 扩展**：进一步提出随机近端梯度下降（SPGD）变体，适用于近端算法

## 局限与展望

1. **理论限于去噪问题**：Theorem 7.4 和 7.5 的证明假设前向模型为恒等映射（去噪），对一般逆问题的理论仍需补充
2. **仅分析平均攻击**：理论保证仅覆盖 average-case robustness，对 worst-case 攻击的理论分析缺失
3. **噪声方差选择**：虽然实验展示了噪声水平的影响（Section 8.2），但缺乏自动选择策略
4. **实验规模有限**：MRI 实验仅使用单线圈 4× 加速，未验证多线圈或更高加速率场景
5. **未与最新方法对比**：如扩散模型等新范式未纳入比较

## 相关工作与启发

- **Krainovic et al. (2023)**：输入 jittering 用于黑盒网络，鲁棒性可比 AT 但精度显著下降
- **Lim et al. (2021)**：层级噪声注入提升分类器泛化，但未解释为何隐层输入的平坦性能改善回归
- **Foret et al. (2021)**：SAM 通过 sharpness-aware 优化提升泛化，但关注参数空间而非迭代空间
- **启发**：这种"在求解路径而非目标上注入噪声"的思路可推广到其他迭代式架构（如扩散模型的反向过程）

## 评分

- 新颖性: ⭐⭐⭐⭐ — 噪声注入位置的新视角简单而深刻
- 实验充分度: ⭐⭐⭐⭐ — 三个任务+多维度评估，但MRI实验规模偏小
- 写作质量: ⭐⭐⭐⭐ — 理论与实验结合清晰，符号体系一致
- 价值: ⭐⭐⭐⭐ — 对安全关键型逆问题（医学影像）有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DiN: Diffusion Model for Robust Medical VQA with Semantic Noisy Labels](../../CVPR2025/medical_imaging/din_diffusion_model_for_robust_medical_vqa_with_semantic_noisy_labels.md)
- [\[CVPR 2025\] Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation](../../CVPR2025/medical_imaging/multi-resolution_pathology-language_pre-training_model_with_text-guided_visual_r.md)
- [\[ICLR 2026\] Cross-Timestep: 3D Diffusion Model with Trans-temporal Memory LSTM and Adaptive Priori Decoding Strategy for Medical Segmentation](../../ICLR2026/medical_imaging/cross-timestep_3d_diffusion_model_with_trans-temporal_memory_lstm_and_adaptive_p.md)
- [\[ICML 2025\] Certification for Differentially Private Prediction in Gradient-Based Training](certification_for_differentially_private_prediction_in_gradient-based_training.md)
- [\[ICLR 2026\] SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding](../../ICLR2026/medical_imaging/seed_towards_more_accurate_semantic_evaluation_for_visual_brain_decoding.md)

</div>

<!-- RELATED:END -->
