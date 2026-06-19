---
title: >-
  [论文解读] Semi-Supervised High Dynamic Range Image Reconstructing via Bi-Level Uncertain Area Masking
description: >-
  [AAAI 2026][HDR重建] 提出半监督 HDR 重建框架，通过不确定性估计分支：评估伪 HDR 标签质量，在 patch 和像素两个层面掩码不可靠区域，仅用 6.7% HDR 真值即可达到与全监督 SOTA 可比的性能。 1. 领域现状：从不同曝光的 LDR 图像集重建无鬼影的 HDR 图像是计算摄影的重要任务…
tags:
  - "AAAI 2026"
  - "HDR重建"
  - "半监督学习"
  - "不确定性估计"
  - "伪标签"
  - "双层掩码"
---

# Semi-Supervised High Dynamic Range Image Reconstructing via Bi-Level Uncertain Area Masking

**会议**: AAAI 2026  
**arXiv**: [2511.12939](https://arxiv.org/abs/2511.12939)  
**代码**: [https://github.com/JW20211/SmartHDR](https://github.com/JW20211/SmartHDR)  
**领域**: 计算摄影 / 半监督学习  
**关键词**: HDR重建, 半监督学习, 不确定性估计, 伪标签, 双层掩码

## 一句话总结

提出半监督 HDR 重建框架，通过**不确定性估计分支**评估伪 HDR 标签质量，在 patch 和像素两个层面掩码不可靠区域，仅用 6.7% HDR 真值即可达到与全监督 SOTA 可比的性能。

## 研究背景与动机

1. **领域现状**：从不同曝光的 LDR 图像集重建无鬼影的 HDR 图像是计算摄影的重要任务。学习型方法（如 GFHDR、SAFNet）取得显著进展但需配对的 LDR-HDR 数据。
2. **现有痛点**：高质量 HDR 真值需昂贵的专业设备或严格控制场景运动获取，难以大规模收集。现有标注高效方法如 FSHDR 合成 LDR 存在域差距，SMAE 的自适应伪标签选择基于参考图像相似度而忽略了饱和区域，且教师-学生差距不够大。
3. **核心矛盾**：伪标签不可避免地包含鬼影或噪声伪影，学生如果学习这些错误会导致**确认偏差**（confirmation bias），但又不能过度丢弃伪标签导致训练数据不足。
4. **本文目标**：如何在有限 HDR 真值下训练高质量 HDR 重建模型？如何有效筛选伪HDR标签中可靠的区域？
5. **切入角度**：引入不确定性估计分支，将预测视为高斯分布、真值视为 Dirac delta 函数，通过 KL 散度学习每个像素的不确定性得分，据此在两个粒度上过滤不可靠区域。
6. **核心 idea**：用可学习的不确定性图来评估伪 HDR 标签的逐像素可靠度，在 patch 和 pixel 双层面掩码不可信区域。

## 方法详解

### 整体框架

采用教师-学生伪标签范式。学生通过梯度下降更新，教师通过 EMA（$\alpha=0.999$）更新。教师生成伪 HDR 标签及对应的不确定性图。基于不确定性图，在 patch 级和 pixel 级掩码不可靠区域后，学生仅学习可信区域。训练分两阶段：30 epoch 预热（仅用有标签数据）+ 170 epoch 半监督。

### 关键设计

1. **不确定性估计分支（Judge Network）**

    - 功能：为预测的 HDR 图像的每个像素生成可靠性评分（不确定性图）。
    - 核心思路：在 GFHDR 的空间注意力模块输出特征 $F_{att}$ 上接三层 3×3 卷积 + skip connection + sigmoid，预测不确定性图 $\sigma$。损失函数基于高斯假设的 KL 散度：$\mathcal{L}^k = \frac{1}{n_h} \sum_i \frac{(\bar{h}_i - h_i)^2}{2\sigma_i^2} + \frac{1}{2}\log(\sigma_i^2)$。当预测值 $\bar{h}$ 不准确时，网络学会预测更大的 $\sigma$ 使损失降低，从而自动标识低质量区域。
    - 设计动机：HDR 重建是密集回归任务，没有类似分类中的置信度得分来评估伪标签质量。将预测建模为高斯分布并学习方差作为不确定性是自然且有效的解决方案。

2. **双层不确定区域掩码**

    - 功能：在 patch 和 pixel 两个粒度上过滤伪标签中的不可靠区域，确保学生仅学习可信信号。
    - 核心思路：**Patch 级**：计算每个 64×64 patch 的不确定性均值并全局归一化得到 $S_{pa}$，通过阈值 $\tau_{pa}=0.4$ 生成二值掩码 $M_{pa}$，丢弃整个高不确定性 patch。**Pixel 级**：对保留 patch 内的每个像素，归一化单通道不确定性图，通过阈值 $\tau_{pi}=0.4$ 生成像素级掩码 $M_{pi}$，精确过滤残余不可靠像素。
    - 设计动机：仅靠 patch 级掩码可能遗漏局部小区域伪影；仅靠 pixel 级掩码则计算效率低且可能保留包含大面积伪影的 patch。双层结合兼顾效率和精度。

3. **数据增强策略**

    - 功能：增大教师-学生间的差距，提供更好的学习信号。
    - 核心思路：受 FixMatch 启发，对无标签数据用强增强（RGB 随机混洗 + 水平翻转 + 90° 旋转），有标签数据用弱增强（垂直翻转 + 90° 旋转）。强增强无标签数据的伪标签由弱增强的教师生成。
    - 设计动机：教师-学生差距不够大（如 SMAE）会导致学生学不到新知识。FixMatch 已证明适当差距的一致性学习+伪标签组合能成功。

### 损失函数 / 训练策略

$\mathcal{L} = \mathcal{L}_s^r + \lambda_v \mathcal{L}_s^v + \mathcal{L}_s^k + \lambda_u(\mathcal{L}_u^r + \lambda_v \mathcal{L}_u^v + \mathcal{L}_u^k)$，其中 $\mathcal{L}^r$ 为 tone-mapped L1 重建损失，$\mathcal{L}^v$ 为 VGG 感知损失，$\mathcal{L}^k$ 为不确定性损失。$\lambda_u=1$。色调映射使用 $\mu$-law（$\mu=5000$）。Adam 优化器，学习率 $2 \times 10^{-4}$，训练 200 epochs，仅用 $N^l=5$ 个样本有标签。

## 实验关键数据

### 主实验

在 Kalantari 和 Hu 数据集上的 PSNR 结果（半监督设定 vs 全监督）：

| 方法 | 标注 | Kalantari PSNR-μ | Kalantari PSNR-l | Hu PSNR-μ | Hu PSNR-l |
|------|------|-----------------|-----------------|----------|----------|
| SAFNet | 100% GT | 44.66 | 43.18 | - | - |
| GFHDR | 100% GT | 44.32 | 42.18 | - | - |
| FSHDR | 6.7% GT | 41.94 | 40.80 | 43.98 | 47.13 |
| SMAE | 6.7% GT | 41.61 | 41.54 | 44.24 | 47.41 |
| **Ours** | **6.7% GT** | **44.04** | **41.67** | **45.10** | **47.93** |

### 消融实验

| 配置 | PSNR-μ | 变化 | 说明 |
|------|--------|------|------|
| Full model | 44.04 | - | 完整模型 |
| w/o patch 掩码 | 降低 | -显著 | patch 级过滤重要 |
| w/o pixel 掩码 | 降低 | -中等 | pixel 级精细过滤有用 |
| w/o 不确定性 | 降低 | -最大 | 无过滤导致确认偏差 |
| w/o 强增强 | 降低 | -中等 | 教师-学生差距不足 |
| 仅用标签数据 | ~42 | -较大 | 验证半监督有效性 |

### 关键发现

- 仅用 6.7% HDR 真值（5 个场景），PSNR-μ 达 44.04，接近全监督 SOTA 的 44.66（SAFNet），远超前半监督方法 SMAE（41.61）。
- 不确定性驱动的掩码是性能提升的关键——移除后性能大幅下降。
- Patch 级掩码贡献大于 pixel 级，因为伪影通常是块状分布。
- EMA 教师比直接复制权重的教师稳定得多，$\alpha=0.999$ 效果最优。
- 双层阈值 $\tau_{pa}=\tau_{pi}=0.4$ 在精度和数据利用率间取得最佳平衡。

## 亮点与洞察

- **不确定性估计 + 伪标签过滤的优雅结合**：将回归任务的预测建模为高斯分布，学习方差作为不确定性，这套方案可直接迁移到其他密集回归半监督任务（深度估计、光流等）。
- **形成正反馈循环**：更好的学生 → 更好的 EMA 教师 → 更可信的伪标签 → 更准确的不确定性估计 → 可以解锁之前不确定的区域。
- **极端标注效率**：仅 5 个样本即可达到近全监督水平，对实际 HDR 数据获取有重大意义。

## 局限与展望

- 仅测试了 6.7% 标注比例（5/75），未系统评估不同标注量下的性能曲线。
- 不确定性阈值 $\tau_{pa}, \tau_{pi}$ 是固定的，自适应阈值策略可能更好。
- 基于 GFHDR 骨干，未验证在更新的骨干（如 SAFNet、扩散模型）上的效果。
- 伪标签每 epoch 更新一次，更频繁的更新可能提升质量但增加计算成本。

## 相关工作与启发

- **vs FSHDR**：FSHDR 合成 LDR 存在域差距导致伪影；本文直接用 EMA 教师生成伪 HDR 避免了域差距问题。
- **vs SMAE**：SMAE 的伪标签选择基于与参考图像的相似度（忽略饱和区域），本文的不确定性估计更通用且自适应。
- **vs FixMatch/Mean Teacher**：将半监督分类的成功范式有效迁移到 HDR 密集回归任务。

## 评分

- 新颖性: ⭐⭐⭐⭐ 不确定性驱动的双层掩码在 HDR 半监督中是首次
- 实验充分度: ⭐⭐⭐⭐ 两个标准数据集验证充分，但标注比例消融不足
- 写作质量: ⭐⭐⭐⭐ 动机清晰，pipeline 图示直观
- 价值: ⭐⭐⭐⭐ 极大降低 HDR 数据标注需求，有实际部署价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)
- [\[NeurIPS 2025\] Semi-Supervised Regression with Heteroscedastic Pseudo-Labels](../../NeurIPS2025/others/semi-supervised_regression_with_heteroscedastic_pseudo-labels.md)
- [\[AAAI 2026\] I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks](i2e_real-time_image-to-event_conversion_for_high-performance_spiking_neural_netw.md)
- [\[AAAI 2026\] Area-Optimal Control Strategies for Heterogeneous Multi-Agent Pursuit](area-optimal_control_strategies_for_heterogeneous_multi-agen.md)
- [\[ECCV 2024\] Rebalancing Using Estimated Class Distribution for Imbalanced Semi-Supervised Learning under Class Distribution Mismatch](../../ECCV2024/others/rebalancing_using_estimated_class_distribution_for_imbalanced_semi-supervised_le.md)

</div>

<!-- RELATED:END -->
