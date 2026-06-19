---
title: >-
  [论文解读] Bridging the Perception Gap in Image Super-Resolution Evaluation
description: >-
  [CVPR 2026][图像恢复][超分辨率评估] 通过大规模用户研究揭示现有 SR 评估指标（PSNR、SSIM、LPIPS 等）与人类感知严重不一致，分析其内在缺陷后提出极简但有效的 RQI（Relative Quality Index）框架，通过学习图像对之间的相对质量差异实现更可靠的 SR 评估，且可作为损失函数指导 SR 训练。
tags:
  - "CVPR 2026"
  - "图像恢复"
  - "超分辨率评估"
  - "图像质量指标"
  - "感知差距"
  - "相对质量指数"
  - "用户研究"
---

# Bridging the Perception Gap in Image Super-Resolution Evaluation

**会议**: CVPR 2026  
**arXiv**: [2503.13074](https://arxiv.org/abs/2503.13074)  
**代码**: [项目页面](https://color.cvc.uab.cat/rqi/)  
**领域**: 图像超分辨率 / 图像质量评估  
**关键词**: 超分辨率评估, 图像质量指标, 感知差距, 相对质量指数, 用户研究

## 一句话总结
通过大规模用户研究揭示现有 SR 评估指标（PSNR、SSIM、LPIPS 等）与人类感知严重不一致，分析其内在缺陷后提出极简但有效的 RQI（Relative Quality Index）框架，通过学习图像对之间的相对质量差异实现更可靠的 SR 评估，且可作为损失函数指导 SR 训练。

## 研究背景与动机
**领域现状**: SR 技术快速发展（RealESRGAN → SwinIR → StableSR → SeeSR），模型输出质量越来越高，但评估指标长期未变。

**现有痛点**: 研究者对评估指标日益不信任——高指标分数的模型不一定产生更好的视觉效果。大量工作不得不做用户研究或堆叠多个指标来验证。

**核心矛盾**: SR 模型进化快但评估标准停滞，指标与人类感知之间存在三类固有挑战：
   - (a) **失真类 FR 指标**（PSNR、SSIM）偏好平滑平均解，与感知偏好相反
   - (b) **感知类 FR 指标**（LPIPS、DISTS）在 GT 质量不佳时失效
   - (c) **无参考指标**（NIQE、CLIP-IQA）无法评估保真度
   - (d) 高质量 SR 输出之间差异微妙，现有指标无法区分

**本文要解决**: 设计一个能同时应对上述四个挑战的 SR 评估框架。

**切入角度**: 用相对质量差异替代绝对质量分数——允许任意图像（包括有退化的）作为参考，学习目标-参考之间的质量落差。

**核心idea**: 既然 GT 可能不完美、SR 输出可能超越 GT，那就不要假设参考是完美的，而是学习**相对**质量关系。

## 方法详解

### 整体框架

RQI 的核心转念是：既然 GT 可能不完美、SR 输出甚至可能超越 GT，那就别再假设参考图像是完美的、也别去预测「绝对质量分数」，而是学习两张图之间的相对质量落差。训练时，从 IQA 数据集里构造稠密的图像对 $\{I_i, I_j\}$，以它们的 MOS 差值 $q_i - q_j$ 为标签，训练一个 FR-IQA 模型去预测这个差值；评估时，把 SR 输出 $I_{HR}$ 和 GT $I_{GT}$ 一起喂进去，输出 $s = f_{RQI}(I_{HR}, I_{GT})$，正值就表示 SR 质量已经优于 GT。

### 关键设计

**1. 相对质量框架的三个特性：用一套设计同时回应三个评估难题**

现有指标的麻烦在于：失真类 FR 指标偏好平滑解、感知类 FR 指标在 GT 不佳时失效、而高质量 SR 输出之间的差异又太微妙难以区分。RQI 用三个相互咬合的特性来对应——非对称性 $f_{RQI}(I_i, I_j) = -f_{RQI}(I_j, I_i)$，交换输入即取反，区别于传统 FR 指标的对称性，对应保真度评估；相对差异，不预测绝对分而只学两图的感知落差，允许参考本身有退化，对应不完美 GT 的鲁棒性；稠密配对比较，传统方法只构造「参考 vs 退化」的 $\{I_0, I_i\}$ 对，RQI 则构造任意「退化 vs 退化」的 $\{I_i, I_j\}$ 对，既大幅扩充训练样本又天然包含微妙的质量差异，对应细粒度区分。

**2. Huber loss 回归相对差值：在微妙差异上稳住训练**

学相对落差要在很小的质量差上给出稳定梯度。RQI 用 Huber loss 回归 MOS 差值，当残差 $|\hat{y}_{ij} - (q_i - q_j)| \le \delta$ 时取 $\tfrac{1}{2}(\hat{y}_{ij} - (q_i - q_j))^2$、否则取 $\delta(|\hat{y}_{ij} - (q_i-q_j)| - \tfrac{1}{2}\delta)$，并把标签归一化到 $[-1, 1]$、去掉最后回归层的激活以支持负值输出。Huber loss 对小差异提供平滑梯度、对大偏差又不过分敏感，因此在微妙质量差异上训练更稳定。

**3. 通用框架而非新指标：提升现有 FR-IQA 的范式**

RQI 不打算再造一个具体指标，而是改造现有指标的训练范式：它可以套在任意 FR-IQA 模型上（AHIQ、MANIQA、TOPIQ），在任意 IQA 数据集上训练（Kadid-10K、PieAPP、PIPAL），无需收集 SR 专用数据就能零样本迁移到 SR 评估。正因为只改了「训练数据怎么配对、目标怎么定义」而不碰架构，它能普适地把多种现成指标的人类一致性顶上去。

### 损失函数 / 训练策略

用 Huber loss 回归相对差值，$\delta$ 为平滑阈值；按 8:2 划分训练/验证且场景不重叠，取验证集最优模型做零样本迁移评估。

## 实验关键数据

### 主实验（与人类感知一致性，SRCC 指标）

| 指标 | DIV2K | RealSR | DRealSR | Set5&14 |
|------|-------|--------|---------|---------|
| SSIM | -0.348 | -0.220 | -0.354 | -0.321 |
| PSNR | -0.079 | -0.116 | -0.355 | -0.204 |
| LPIPS | 0.415 | 0.008 | -0.141 | 0.282 |
| CLIP-IQA | 0.593 | 0.377 | 0.268 | 0.642 |
| AFINE | 0.581 | 0.449 | 0.484 | 0.578 |
| DeQA-Score | 0.613 | 0.452 | 0.437 | 0.699 |
| **RQI** | **0.744** | **0.504** | **0.529** | 0.664 |

### 消融实验（RQI 框架有效性）

| 训练集 / 模型 | 传统 FR | RQI | 提升 |
|--------------|--------|-----|------|
| PIPAL / MANIQA (DIV2K) | 0.624 | **0.744** | +0.120 |
| PIPAL / TOPIQ (DRealSR) | 0.042 | **0.357** | +0.315 |
| Kadid / AHIQ (Set5&14) | 0.292 | **0.426** | +0.134 |

### 关键发现
- PSNR 和 SSIM 在所有数据集上与人类感知**负相关**！这是对 SR 领域评估惯例的严重挑战
- LPIPS 在真实世界 SR 数据集（RealSR、DRealSR）上接近零相关
- NR 指标（NIQE、CLIP-IQA）整体优于 FR 指标，但无法评估保真度
- RQI 框架一致性地提升所有模型在所有数据集上的表现
- RQI 作为损失函数训练 SR 模型可同时提升感知质量和结构保真度

## 亮点与洞察
- 大规模用户研究（7 个 SR 模型 × 5 个基准 × 15 参与者/对比）提供了权威的人类偏好数据
- "PSNR/SSIM 与人类感知负相关"的发现对 SR 领域是当头棒喝
- RQI 框架的巧妙之处在于"简单到极致"——只改变训练数据构建方式和目标定义，不改架构
- 可作为损失函数的双重用途增加了实用价值

## 局限与展望
- 用户研究的参与者数量和多样性可能影响结论的普遍性
- 当前仅在 ×4 SR 任务上验证，其他放大比例、降质类型待测
- RQI 仍需 GT 图像作为参考，完全无参考的场景不适用
- MOS 差值作为线性近似可能在极端质量差异下不够准确

## 相关工作与启发
- AFINE 也考虑了不完美 GT 假设，但需要 SR 特定数据训练，RQI 无此限制
- DeQA-Score 等 LLM 指标性能好但计算成本高，RQI 用传统架构就能达到类似水平
- 启示：评估指标的范式创新（如何定义"好"）可能比模型创新更重要

## 评分
- 新颖性: ⭐⭐⭐⭐ RQI 相对质量框架思路简洁而深刻，但核心 idea 不复杂
- 实验充分度: ⭐⭐⭐⭐⭐ 大规模用户研究 + 系统性分析 + 多模型多数据集 + 作为损失函数
- 写作质量: ⭐⭐⭐⭐⭐ 问题分析透彻，三个 Goal 的抽象精准
- 价值: ⭐⭐⭐⭐⭐ 对 SR 评估领域有根本性推进，"PSNR/SSIM 负相关"将改变社区惯例

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Bridging the Distribution Gap to Harness Pretrained Diffusion Priors for Super-Resolution](../../ICLR2026/image_restoration/bridging_the_distribution_gap_to_harness_pretrained_diffusion_priors_for_super-r.md)
- [\[CVPR 2026\] Task-Aware Image Signal Processor for Advanced Visual Perception](task-aware_image_signal_processor_for_advanced_visual_perception.md)
- [\[CVPR 2026\] AceTone: Bridging Words and Colors for Conditional Image Grading](acetone_bridging_words_and_colors_for_conditional_image_grading.md)
- [\[CVPR 2026\] Disentangled Textual Priors for Diffusion-based Image Super-Resolution](disentangled_textual_priors_for_diffusion-based_image_super-resolution.md)
- [\[CVPR 2026\] DVAR: Dynamic Visual Autoregressive Modeling for Image Super-Resolution](dvar_dynamic_visual_autoregressive_modeling_for_image_super-resolution.md)

</div>

<!-- RELATED:END -->
