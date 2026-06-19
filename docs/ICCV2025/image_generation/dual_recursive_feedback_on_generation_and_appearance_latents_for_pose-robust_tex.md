---
title: >-
  [论文解读] Dual Recursive Feedback on Generation and Appearance Latents for Pose-Robust Text-to-Image Diffusion
description: >-
  [ICCV 2025][图像生成][T2I 扩散模型] 提出 **Dual Recursive Feedback (DRF)**，一种无需训练的双递归反馈系统，通过**外观反馈**和**生成反馈**递归精修中间隐变量，解决可控 T2I 扩散模型在跨类别（class-invariant）场景下结构/外观分离不彻底的问题，实现细粒度的姿态迁移和外观融合。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "T2I 扩散模型"
  - "可控生成"
  - "姿态迁移"
  - "外观保真"
  - "Score Distillation"
  - "training-free"
---

# Dual Recursive Feedback on Generation and Appearance Latents for Pose-Robust Text-to-Image Diffusion

**会议**: ICCV 2025  
**代码**: [GitHub](https://github.com/jwonkm/DRF)  
**领域**: 图像生成  
**关键词**: T2I 扩散模型, 可控生成, 姿态迁移, 外观保真, Score Distillation, training-free  
**作者**: Jiwon Kim, Pureum Kim, SeonHwa Kim 等 (Korea University, Sookmyung Women's University)

## 一句话总结

提出 **Dual Recursive Feedback (DRF)**，一种无需训练的双递归反馈系统，通过**外观反馈**和**生成反馈**递归精修中间隐变量，解决可控 T2I 扩散模型在跨类别（class-invariant）场景下结构/外观分离不彻底的问题，实现细粒度的姿态迁移和外观融合。

## 研究背景与动机

可控 T2I 扩散模型的核心目标是根据用户意图生成同时满足**结构控制**（姿态、边缘、深度）和**外观迁移**的图像。现有方法各有局限：

**ControlNet**：需要针对不同空间因素分别微调多个模型，增加训练开销

**FreeControl**：无需额外训练，但依赖梯度优化最小化隐空间损失，计算成本高

**Ctrl-X**：效率较高，通过前馈特征替换保持结构、通过自注意力层对齐外观统计量，但在**跨类别**设定下性能退化严重

关键挑战：当结构图像和外观图像属于不同类别（如将人体运动迁移到老虎形态），简单地分离注意力图中的外观和结构信息远远不够——会出现"外观泄漏"（appearance leakage）和结构保真度下降的问题。

## 方法详解

### 整体框架

DRF 构建在 Ctrl-X 的基础上。首先将结构图像 $\mathbf{I}^s$ 和外观图像 $\mathbf{I}^a$ 注入预训练 T2I 模型获得初始生成隐变量：

$$\mathbf{z}_{t-1}^g = \text{Ctrl-X}(\mathbf{z}_t^g | t, y, f_t^s, A_t^s, h_t^a)$$

然后在扩散生成过程中，DRF 通过双重反馈递归精修隐变量。

### 外观反馈 (Appearance Feedback)

解决"外观泄漏"问题——确保外观信息在生成过程中不会丢失。

关键设计：借鉴 IDS 中固定点概念，但避免其过度约束导致的编辑能力受限。为解决大时间步下后验均值 $\mathbf{z}_{0|t}^a$ 与 $\mathbf{z}_0^a$ 差异过大导致的过度引导问题，提出**修正随机隐变量**：

$$\tilde{\mathbf{z}}_t^a = \sqrt{\frac{\alpha_t}{\alpha_{t-1}}} \mathbf{z}_0^a + \sqrt{1 - \frac{\alpha_t}{\alpha_{t-1}}} \epsilon$$

外观反馈损失：
$$\mathcal{L}_{\text{app}} = d(\mathbf{z}_{0|t}^a, \mathbf{z}_0^a)$$

### 生成反馈 (Generation Feedback)

解决仅有外观反馈时的外观过拟合问题——确保生成结果与用户意图对齐。

将前一次迭代的生成结果 $\mathbf{z}_{\text{prev}}^g$ 作为另一个固定点，使当前更新方向与之对齐：

$$\mathcal{L}_{\text{gen}} = d(\mathbf{z}_{0|t}^g, \mathbf{z}_{\text{prev}}^g)$$

### 双递归反馈整合

采用指数加权方案渐进式调整反馈权重：

$$w_{\text{iter}}^{(i)} = \sqrt{\frac{\exp(k \cdot \frac{i}{N-1}) - 1}{\exp(k) - 1}}$$

设计逻辑：迭代初期侧重外观反馈确保身份信息被正确反映，随着迭代推进逐步加大生成反馈权重以更好匹配用户意图。

最终 DRF 损失：
$$\mathcal{L}_{\text{DRF}}^{(i)} = d(\mathbf{z}_{0|t}^a, \mathbf{z}_0^a) + \rho \cdot w_{\text{iter}}^{(i)} \cdot d(\mathbf{z}_{0|t}^g, \mathbf{z}_{\text{prev}}^g)$$

通过更新注入噪声 $\epsilon$ 最小化 DRF 损失：
$$\epsilon \leftarrow \epsilon - \lambda \nabla_\epsilon \mathcal{L}_{\text{DRF}}^{(i)}$$

DRF 应用于中间 20 步（跳过前 5 步），在每步内执行 $N$ 次递归迭代。

## 实验关键数据

### 主实验：定量比较

| 方法 | Mesh Self-Sim↓ | Mesh CLIP↑ | Mesh DINO-I↑ | Pose Self-Sim↓ | Pose CLIP↑ | Pose DINO-I↑ | Successive Rate |
|------|---------------|------------|-------------|---------------|------------|-------------|----------------|
| T2I-Adapter+IP-Adapter | 0.2374 | 0.3062 | 0.6627 | 0.2949 | 0.2865 | 0.5304 | 0.9718 |
| ControlNet+IP-Adapter | 0.2024 | 0.3320 | 0.6068 | 0.3035 | 0.2904 | 0.6857 | 0.8873 |
| FreeControl | 0.1503 | 0.3270 | 0.7288 | 0.2839 | 0.2880 | 0.6162 | 0.9152 |
| Ctrl-X | 0.1542 | 0.3464 | 0.7139 | 0.2332 | 0.3429 | 0.7378 | 0.9577 |
| **DRF (Ours)** | **0.1532** | **0.3492** | **0.7342** | **0.2294** | **0.3503** | **0.7391** | **0.9859** |

DRF 在 Successive Rate 上达到最高的 **0.9859**，说明生成图像最忠实地融合了结构和外观。

### 用户研究

| 方法 | Text Preference↑ | Structure Preference↑ | Appearance Preference↑ |
|------|-----------------|----------------------|----------------------|
| T2I-Adapter+IP-Adapter | 7.53% | 8.09% | 11.43% |
| ControlNet+IP-Adapter | 16.02% | 12.90% | 14.14% |
| FreeControl | 12.93% | 14.83% | 17.63% |
| Ctrl-X | 17.37% | 9.44% | 13.75% |
| **DRF (Ours)** | **35.52%** | **43.73%** | **33.72%** |

DRF 在三个维度全面领先，结构保持偏好率高达 **43.73%**。

### 调度器无关性验证

| 调度器 | Steps | CLIP↑ | Self-Sim↓ | DINO-I↑ | Time(s) |
|--------|-------|-------|-----------|---------|---------|
| DPM-Solver++ + DRF | 10 | 0.3256 | 0.1605 | 0.826 | 15.56 |
| DDIM + DRF | 30 | 0.3492 | 0.1204 | 0.891 | 35.74 |
| DPM-Solver++ + DRF | 10 vs DDIM 40 | 相当 | 相当 | 相当 | **3× 更快** |

DRF 与快速求解器（DPM-Solver++）结合可将延迟减少约 3×，同时保持感知质量。

### 跨主干网络验证

| 主干 | CLIP↑ | Self-Sim↓ | DINO-I↑ | GPU Memory(GiB) |
|------|-------|-----------|---------|-----------------|
| SD 1.5 + DRF | 0.3108 | 0.1820 | 0.6176 | 5.87 |
| SD 2.0 + DRF | 0.2934 | 0.2166 | 0.453 | 6.28 |
| **SDXL + DRF** | **0.3331** | **0.1586** | **0.6957** | 18.98 |

DRF 可移植到 SD 1.5 和 SD 2.0，保持保真度的同时大幅降低 GPU 内存消耗。

## 亮点与洞察

1. **双反馈机制的互补设计**：外观反馈确保身份保持，生成反馈防止外观过拟合——两者缺一不可（仅有外观反馈导致过度强调外观图像，忽略结构）
2. **修正随机隐变量的技巧**：通过 $\alpha_t/\alpha_{t-1}$ 缩放避免大时间步下的过度引导，是对 IDS 的有效改进
3. **指数加权的迭代策略**：外观优先→生成逐增的权重调度符合扩散生成的直觉
4. **跨类别姿态迁移**：成功实现人体运动→老虎/企鹅等极端跨类别融合
5. **即插即用特性**：可集成到 ControlNet+IP-Adapter 等现有模型中进一步提升效果
6. **调度器无关**：兼容 DDIM、DPM-Solver++、UniPC 等多种调度器

## 局限性

1. **计算开销**：双递归反馈不可避免地增加推理时间（N=3 迭代约 57 秒 vs Ctrl-X 的 15 秒）
2. **细节保持**：对训练数据中未涵盖的普通人面部等细粒度细节保持能力有限
3. **依赖基础模型能力**：DRF 增强的上界受限于底层 T2I 模型的表达能力
4. **仅 512×512 分辨率**：未在更高分辨率上验证

## 相关工作与启发

- **与 Ctrl-X 的关系**：DRF 以 Ctrl-X 为基础骨架，通过递归反馈解决其在 class-invariant 设定下的失败案例
- **与 IDS/SDS 的联系**：外观反馈借鉴 IDS 的固定点正则化，但通过修正随机隐变量避免了过度约束
- **对可控生成的启示**：无训练方法通过 score guidance 可以实现精细化的结构-外观解耦，为高效可控生成提供了新思路

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 双递归反馈的设计有创新性，外观/生成反馈的互补机制论证充分
- **实验**: ⭐⭐⭐⭐ — 定量指标+用户研究+消融+跨调度器/跨主干验证，评估全面
- **写作**: ⭐⭐⭐⭐ — 流程图清晰，公式推导完整，定性结果直观有力
- **价值**: ⭐⭐⭐⭐ — Training-free + plug-and-play 特性使其有较高实际应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Rethink Sparse Signals for Pose-guided Text-to-Image Generation](rethink_sparse_signals_for_pose-guided_text-to-image_generation.md)
- [\[ICCV 2025\] DPoser-X: Diffusion Model as Robust 3D Whole-Body Human Pose Prior](dposer-x_diffusion_model_as_robust_3d_whole-body_human_pose_prior.md)
- [\[ICML 2025\] GaussMarker: Robust Dual-Domain Watermark for Diffusion Models](../../ICML2025/image_generation/gaussmarker_robust_dual-domain_watermark_for_diffusion_models.md)
- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](../../CVPR2025/image_generation/dual_diffusion_for_unified_image_generation_and_understanding.md)
- [\[ICCV 2025\] DiffuMatch: Category-Agnostic Spectral Diffusion Priors for Robust Non-rigid Shape Matching](diffumatch_category-agnostic_spectral_diffusion_priors_for_robust_non-rigid_shap.md)

</div>

<!-- RELATED:END -->
