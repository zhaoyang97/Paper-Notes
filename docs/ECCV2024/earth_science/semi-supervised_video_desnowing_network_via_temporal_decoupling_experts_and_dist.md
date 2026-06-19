---
title: >-
  [论文解读] Semi-supervised Video Desnowing Network via Temporal Decoupling Experts and Distribution-Driven Contrastive Regularization
description: >-
  [ECCV2024][地球科学][video desnowing] 提出首个半监督视频去雪框架 SemiVDN，通过物理先验引导的时序解耦专家模块和分布驱动的对比正则化，利用无标签真实雪景视频缩小合成-真实域差距，在合成与真实数据集上均超越现有方法。 雪是户外视频中常见的恶劣天气退化因素，雪粒和雪条严重损害视频帧的可见性…
tags:
  - "ECCV2024"
  - "地球科学"
  - "video desnowing"
  - "半监督学习"
  - "mixture of experts"
  - "对比学习"
  - "atmospheric scattering model"
---

# Semi-supervised Video Desnowing Network via Temporal Decoupling Experts and Distribution-Driven Contrastive Regularization

**会议**: ECCV2024  
**arXiv**: [2410.07901](https://arxiv.org/abs/2410.07901)  
**代码**: [TonyHongtaoWu/SemiVDN](https://github.com/TonyHongtaoWu/SemiVDN)  
**领域**: 地球科学  
**关键词**: video desnowing, semi-supervised learning, mixture of experts, contrastive learning, atmospheric scattering model

## 一句话总结

提出首个半监督视频去雪框架 SemiVDN，通过物理先验引导的时序解耦专家模块和分布驱动的对比正则化，利用无标签真实雪景视频缩小合成-真实域差距，在合成与真实数据集上均超越现有方法。

## 背景与动机

雪是户外视频中常见的恶劣天气退化因素，雪粒和雪条严重损害视频帧的可见性，进而影响自动驾驶等下游任务。现有深度学习去雪方法在合成基准上表现良好，但由于合成数据与真实数据之间存在显著的分布偏移（雪的形状、运动轨迹差异大），在真实场景中效果严重退化。同时获取配对的真实雪景数据极其困难——天气条件多变、物体和相机位置难以对齐，使得完全监督方案不可行。

本文的核心动机是：引入无标签真实雪景视频，以半监督方式训练，从而提升模型在真实场景下的泛化能力。

## 核心问题

1. **域差距问题**：合成雪与真实雪在形态、运动模式上分布差异大，导致在合成数据上训练的模型泛化能力差
2. **缺少配对真实数据**：真实场景中几乎无法获取成对的雪景/干净视频
3. **物理分量解耦不充分**：先前方法（如 SVDNet）虽利用大气散射模型，但对雪层、透射率、大气光的解耦不够显式和准确

## 方法详解

### 整体框架

SemiVDN 基于 Mean-Teacher 半监督架构，包含学生网络和教师网络。学生网络由编码器（ConvNeXt backbone）、Prior-guided Temporal Decoupling Experts 模块和解码器组成。教师网络通过指数移动平均（EMA，衰减系数 η=0.99）更新权重。

训练时使用合成有标签数据计算监督损失，同时使用真实无标签数据计算无监督损失。

### 大气散射模型

基于经典退化公式：$I_{snow}(x) = J(x)T(x) + A(x)(1-T(x)) + S(x)$，其中 $J$ 为干净视频，$T$ 为透射率图，$A$ 为大气光，$S$ 为雪图。网络在特征空间中显式分解这些物理分量。

### Prior-guided Temporal Decoupling Experts (PTDE)

这是方法的核心模块，替换 Transformer block 中的 MLP 层：

- **Physics Transformer Block**：编码器提取帧特征后进行 overlapped patch embedding 得到 token 序列，送入两层 Transformer——第一层使用 fusion feed-forward network 增强特征融合，第二层将 MLP 替换为 Temporal Decoupling Experts
- **Temporal Decomposition Router**：计算时序自适应权重 $Q_{ij}$，通过 softmax 在时序维度 $N_f \cdot m$ 上归一化，聚合连续帧之间的互补信息
- **专家网络**：设置三个专家分别对应雪层（Snow Expert）、透射率（Transmission Expert）和大气光（Atmospheric Light Expert），每个专家处理经 router 加权后的时序自适应 token
- **连续可微路由**：与稀疏 MoE 的离散路由不同，本方法使用连续可微的 softmax 操作，避免了 token dropping 和专家不平衡问题
- **Prior-guided Recovery Module**：利用大气散射公式在特征空间执行物理反推：$F'_B = (F'_I - F'_S - (1-F'_T)F'_A) / (F'_T + \beta)$

### 半监督训练策略

**监督损失**（用于有标签合成数据）：

$$\mathcal{L}_{sup} = \mathcal{L}_{pixel} + 0.03 \cdot \mathcal{L}_{perceptual} + 10 \cdot \mathcal{L}_{Frequency}$$

包含 Charbonnier 像素损失、VGG-16 感知损失和 Focal Frequency Loss。

**无监督损失**（用于无标签真实数据）：

$$\mathcal{L}_{un} = 2 \cdot \mathcal{L}'_{pixel} + 0.1 \cdot \mathcal{L}_{cl} + 0.1 \cdot \mathcal{L}_{DCP} + 0.5 \cdot \mathcal{L}_{TV}$$

包含师生一致性损失、感知对比损失、暗通道先验损失和全变分损失。

总损失使用高斯预热函数动态调节监督/无监督损失权重。

### Distribution-driven Contrastive Regularization (DCR)

为缩小合成与真实雪的分布差距，设计了基于分布的对比学习策略：

1. **物理分量分离**：从学生和教师网络分别获取背景特征和雪层特征
2. **GMM 建模真实雪分布**：用高斯混合模型（K=3 个组件）拟合真实雪层特征的分布
3. **Ultra-positive 样本选择**：通过计算 KL 散度，从合成雪层中选出与真实雪分布最接近的样本作为"超正样本"
4. **对比正则化**：正样本用真实背景+超正合成雪构建，负样本用合成背景+增强后的真实雪构建，锚点为教师网络的真实背景+学生网络的真实雪，从而让网络聚焦于恢复与雪无关的背景细节

### 真实数据集 Realsnow85

收集了 85 段真实雪景视频（涵盖城市、公园、乡村、自然场景，多种降雪强度和光照条件），其中 60 段用于训练，25 段用于测试。

## 实验关键数据

在 RVSD 合成测试集上：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| SVDNet (ICCV2023) | 25.06 | 0.9210 | 0.0842 |
| Restormer (CVPR2022) | 24.34 | 0.8929 | 0.1164 |
| Snowformer (2022) | 24.01 | 0.8939 | 0.1219 |
| **SemiVDN** | **25.68** | **0.9254** | **0.0785** |

在真实数据集上（无参考指标）：SemiVDN 取得 NIMA 4.259、MUSIQ 51.57，均为最优。

消融实验表明各组件贡献：

| 配置 | PSNR | SSIM | NIMA |
|------|------|------|------|
| M1（基线） | 24.41 | 0.9116 | 4.165 |
| M2（+TDE） | 25.16 | 0.9217 | 4.212 |
| M3（+TDE+SST） | 25.29 | 0.9237 | 4.239 |
| SemiVDN（+TDE+SST+DCR） | 25.68 | 0.9254 | 4.259 |

时序解耦专家模块贡献最大（+0.75 dB），半监督训练和分布对比正则化各带来进一步提升。

## 亮点

- **首个半监督视频去雪框架**，填补了该任务中半监督学习的空白
- **物理先验驱动的 MoE 设计**独具巧思：将传统 MoE 的离散稀疏路由改为连续可微操作，并让专家与物理分量（S、A、T）一一对应，隐式由大气散射公式引导训练
- **GMM + KL 散度选择 ultra-positive 样本**的对比策略新颖，有效桥接合成-真实域差距
- 收集并开源了 Realsnow85 真实雪景视频数据集，为后续研究提供了宝贵资源
- 在 15 个 SOTA 方法上全面超越，且兼顾效率与性能的 trade-off

## 局限与展望

- 真实数据集规模较小（仅 85 段视频），可能不足以覆盖所有雪景场景
- 帧数固定为 3 帧，对于长时序依赖的场景可能不够充分
- GMM 组件数固定为 3，未探讨自动选择最优组件数的策略
- 仅考虑雪这一种退化，未探索多种恶劣天气联合去除的场景
- 无参考指标（NIMA/MUSIQ）的提升幅度相对有限，真实场景的评估仍缺乏更可靠的度量标准
- 大气散射模型的简化假设在极端场景（如暴风雪、强遮挡）下可能不成立

## 与相关工作的对比

- **vs SVDNet (ICCV2023)**：同样利用大气散射模型，但 SVDNet 的物理分量解耦依赖普通卷积层，容易混入背景噪声；SemiVDN 用专家网络显式解耦且引入时序路由，PSNR 提高 0.62 dB
- **vs 单图去雪方法**（JSTASR、HDCW-Net、Snowformer）：缺少时序信息利用，性能全面落后
- **vs 半监督图像恢复方法**（AECR-Net、S2VD）：这些方法未针对视频雪退化的物理特性设计，在合成和真实数据上均不如 SemiVDN
- **vs 通用视频恢复方法**（BasicVSR++、IconVSR）：虽有强大的时序建模能力，但缺乏针对雪退化的先验，效果显著不如专用方法

## 启发与关联

- 将物理先验融入 MoE 架构的思路可推广到其他基于物理模型的图像恢复任务（去雾、去雨），每个专家对应一个物理分量
- 半监督策略 + 域适应对比学习的组合模式适用于所有缺乏配对真实数据的低级视觉任务
- GMM 建模退化分布 + KL 散度选择最优正样本的方法可作为通用的域自适应对比学习策略
- Temporal Decomposition Router 的连续可微设计对其他需要时序聚合的视频处理任务有参考价值

## 评分
- 新颖性: 8/10 — 首次在视频去雪中引入半监督学习，物理先验驱动 MoE 和分布驱动对比正则化设计新颖
- 实验充分度: 8/10 — 对比 15 个方法，消融完整，提供合成和真实数据评估，并收集新数据集
- 写作质量: 7/10 — 结构清晰，公式推导完整，但部分符号较多，可读性可进一步提升
- 价值: 7/10 — 半监督去雪是有意义的研究方向，但应用面相对较窄

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] MdaIF: Robust One-Stop Multi-Degradation-Aware Image Fusion with Language-Driven Semantics](../../AAAI2026/earth_science/mdaif_robust_one-stop_multi-degradation-aware_image_fusion_with_language-driven_.md)
- [\[NeurIPS 2025\] Reasoning With a Star: A Heliophysics Dataset and Benchmark for Agentic Scientific Reasoning](../../NeurIPS2025/earth_science/reasoning_with_a_star_a_heliophysics_dataset_and_benchmark_for_agentic_scientifi.md)
- [\[NeurIPS 2025\] ControlFusion: A Controllable Image Fusion Framework with Language-Vision Degradation Prompts](../../NeurIPS2025/earth_science/controlfusion_a_controllable_image_fusion_framework_with_language-vision_degrada.md)
- [\[NeurIPS 2025\] A Probabilistic U-Net Approach to Downscaling Climate Simulations](../../NeurIPS2025/earth_science/a_probabilistic_unet_approach_to_downscaling_climate_simulat.md)
- [\[NeurIPS 2025\] Power Ensemble Aggregation for Improved Extreme Event AI Prediction](../../NeurIPS2025/earth_science/power_ensemble_aggregation_for_improved_extreme_event_ai_prediction.md)

</div>

<!-- RELATED:END -->
