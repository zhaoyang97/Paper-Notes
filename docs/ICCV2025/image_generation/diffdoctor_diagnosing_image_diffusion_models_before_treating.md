---
title: >-
  [论文解读] DiffDoctor: Diagnosing Image Diffusion Models Before Treating
description: >-
  [ICCV 2025][图像生成][扩散模型] 提出 DiffDoctor，首个利用像素级反馈微调扩散模型的方法：先训练鲁棒的 artifact 检测器（1M+ 样本，类别平衡策略），再通过最小化合成图中每个像素的 artifact 置信度反向传播梯度到扩散模型，使其在未见 prompt 上也能显著减少 artifact 生成。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "扩散模型"
  - "Artifact Detection"
  - "像素级反馈"
  - "图像质量"
  - "模型微调"
---

# DiffDoctor: Diagnosing Image Diffusion Models Before Treating

**会议**: ICCV 2025  
**arXiv**: [2501.12382](https://arxiv.org/abs/2501.12382)  
**代码**: 无  
**领域**: 扩散模型  
**关键词**: 扩散模型, Artifact Detection, 像素级反馈, 图像质量, 模型微调

## 一句话总结

提出 DiffDoctor，首个利用像素级反馈微调扩散模型的方法：先训练鲁棒的 artifact 检测器（1M+ 样本，类别平衡策略），再通过最小化合成图中每个像素的 artifact 置信度反向传播梯度到扩散模型，使其在未见 prompt 上也能显著减少 artifact 生成。

## 研究背景与动机

**领域现状**：图像扩散模型（如 FLUX.1、SDXL、Kolors）虽然能生成多样图像，但仍会产生形状扭曲（如畸形手指/面部）、不合理内容（如多余肢体）和水印等 artifact。现有改进方法主要利用图像级分数或人类偏好反馈来优化模型。

**现有痛点**：
   - **反馈粒度粗**：ImageReward、DDPO、DiffDPO 等方法均使用全图级别的质量评分或成对比较，忽略了 artifact 在图像中是稀疏分布的——一张图可能只有局部区域有问题，全图级反馈无法精确指导修正。
   - **artifact 数据不平衡**：现有标注数据集（RichHF、PAL4VST）存在严重的类别不平衡——低质量模型生成的人像中手/脸几乎总是有 artifact，导致检测器学到"所有手都是 artifact"的捷径。
   - **像素级标注未被利用**：PAL4VST 和 RichHF 虽然提供了细粒度标注，但仅用于后处理（批量排序或 inpainting），未直接用于微调扩散模型。

**核心矛盾**：解决问题要先定位问题——在不知道 artifact 在哪的情况下用全图级分数微调模型，效率低且可能导致整体图像质量下降。

**核心idea**：Diagnose-then-Treat——先训练一个像素级 artifact 检测器准确定位问题区域，再将检测图的梯度信号直接反传给扩散模型进行像素级治疗。

## 方法详解

### 整体框架

DiffDoctor 是一个两阶段 pipeline：
- **Diagnosing（诊断）**：训练一个鲁棒的 artifact 检测器，输入合成图像，输出逐像素 artifact 置信度图（0-1）
- **Treating（治疗）**：冻结检测器，扩散模型生成图像后通过检测器前向传播得到 artifact 图，最小化 artifact 置信度并反向传播到扩散模型参数

### 关键设计

1. **类别平衡的 Artifact 检测器训练**：

    - 功能：解决现有数据中 artifact 标注的类别不平衡问题，训练可靠的像素级检测器
    - 核心思路：(1) 引入高质量真实照片作为负样本平衡分布；(2) 用 MLLM 对图像做类别标注，统计各类别的聚合 artifact 置信度，找到异常偏高/偏低的类别；(3) 用 LLM 为这些类别生成多样 prompt，用 SOTA 模型（生成质量更高的图像）合成更多平衡样本；(4) 从中筛选 2K 困难样本进行人工精标
    - 设计动机：针对性地修复检测器在"人手/人脸总是 artifact"等捷径上的误判

2. **伪标签规模化（Human-in-the-Loop）**：

    - 功能：利用半监督学习思路将标注数据扩展到 1M+
    - 核心思路：未被选为困难样本的图像，由当前检测器预测伪标签。设计动态增广策略——将最大 artifact 置信度低于阈值的图像缩小后 padding 回原尺寸，增大出现小复杂区域（更可能无 artifact）的概率，结合强增广
    - 设计动机：小区域更容易出错，缩小的无 artifact 图像更可能包含小而正确的复杂结构，帮助平衡分布
    - 模型骨架：SegFormer-b5，输出 logits 后 sigmoid 得到置信度图

3. **像素级治疗（Pixel-Aware Treating）**：

    - 功能：利用检测器的梯度信号直接优化扩散模型
    - 核心思路：扩散模型在带梯度追踪的情况下执行去噪，生成图像 $\pi_\theta(z_T)$ 后通过冻结的检测器得到 artifact 图 $C(\pi_\theta(z_T))$。像素级损失：$\mathcal{L}_{\text{pixel}} = \frac{1}{N_{\text{aggr}}}\sum_{i,j} M \circ C(\pi_\theta(z_T))[i,j]$，其中 $M$ 是阈值掩码（仅处理置信度 >0.1 的像素），梯度在最后 25% 的去噪步截断以节省显存。仅训练 LoRA 层（rank=16）
    - 设计动机：像素级惩罚比全图级更精准，避免全面压低导致的质量退化

4. **离线正则化（Offline Regularization）**：

    - 功能：防止模型崩塌（生成模糊图像）
    - 核心思路：加入标准扩散损失 $\mathcal{L}_{\text{offline}} = \|(z_T - z_0) - v_\theta(z_t, t)\|$ 作为 KL 正则，约束更新后模型不偏离真实图像分布。最终损失 $\mathcal{L} = \mathcal{L}_{\text{pixel}} + 0.25 \cdot \mathcal{L}_{\text{offline}}$
    - 设计动机：纯像素级治疗训练过久会导致模型崩塌（类似 reward hacking），正则化可以延缓崩塌

### 损失函数 / 训练策略

- **检测器训练**：MSE 损失 $\mathcal{L}_{\text{AD}} = \frac{1}{N}\sum_i \|\hat{C}_\theta(x_i) - C(x_i)\|_2^2$
- **扩散模型治疗**：$\mathcal{L} = \mathcal{L}_{\text{pixel}} + 0.25 \cdot \mathcal{L}_{\text{offline}}$
- 学习率 1e-4，主要在 FLUX.1 Schnell（4步推理）上实验，也适用于 SDXL 和 Kolors

## 实验关键数据

### 主实验

**Artifact 检测器对比**：

| 方法 | MSE (Ours)↓ | KL (FN)↓ | KL(1-) (FP)↓ | MSE (Real)↓ | KL(1-) (Real)↓ |
|------|------------|---------|-------------|------------|----------------|
| PAL4VST | 0.480 | 5.053 | 2.394 | 0.591 | 5.740 |
| RichHF* | 1.601 | 1.059 | 7.044 | 0.979 | 6.082 |
| +real photos | 1.167 | 1.111 | 4.803 | 0.029 | 1.558 |
| +hard cases | 0.504 | 0.981 | 2.983 | 0.003 | 0.458 |
| **+pseudo 1M** | **0.337** | 1.004 | **2.231** | **0.002** | **0.371** |

**扩散模型治疗效果**：

| 方法 | Artifact 频率↓ | ImageReward↑ | CLIP-T↑ |
|------|--------------|-------------|---------|
| FLUX.1 (原始) | 82.66% | 1.179 | 35.463 |
| FLUX.1 + HPSv2 | 80.67% | 1.022 | 35.037 |
| **FLUX.1 + DiffDoctor** | **22.00%** | **1.183** | **35.611** |
| SDXL (原始) | 55.33% | 0.974 | 36.211 |
| **SDXL + DiffDoctor** | **27.50%** | **1.008** | **36.217** |
| Kolors (原始) | 65.31% | 0.823 | 34.251 |
| **Kolors + DiffDoctor** | **29.33%** | 0.824 | **34.424** |

DiffDoctor 将 FLUX.1 的 artifact 频率从 82.66% 降至 22.00%（降幅 60%+），同时 ImageReward 和 CLIP-T 维持甚至略有提升。

### 消融实验

**像素选择策略**（使用最佳检测器治疗 FLUX.1）：

| 策略 | ImageReward | CLIP-T |
|------|-----------|-------|
| All pixels | 1.161 | 35.278 |
| Max pixel | 1.159 | 35.510 |
| **Threshold** | **1.183** | **35.611** |

- 阈值筛选优于全像素或单像素，更精细的像素级控制效果更好
- 使用朴素检测器（高误报率）治疗会导致严重崩塌，ImageReward 降至 -0.9

## 个人思考

- **亮点**：首次将像素级反馈用于微调扩散模型，artifact 频率降幅惊人；类别平衡策略系统性解决了检测器的捷径学习问题
- **局限**：治疗过程需完整前向去噪 + 检测器前向 + 反向传播，内存和计算开销大；长时间训练仍会崩塌，需要 early stopping
- **启发**：检测器的质量决定治疗效果，这个"诊断先于治疗"的范式可推广到其他生成模型的质量控制

## 亮点与洞察

## 局限与展望

## 相关工作与启发

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models](compass_enhancing_spatial_understanding_in_text-to-image_diffusion_models.md)
- [\[ICCV 2025\] Make Me Happier: Evoking Emotions Through Image Diffusion Models](make_me_happier_evoking_emotions_through_image_diffusion_models.md)
- [\[ICML 2026\] Diagnosing and Correcting Concept Omission in Multimodal Diffusion Transformers](../../ICML2026/image_generation/diagnosing_and_correcting_concept_omission_in_multimodal_diffusion_transformers.md)
- [\[ICCV 2025\] Golden Noise for Diffusion Models: A Learning Framework](golden_noise_for_diffusion_models_a_learning_framework.md)
- [\[NeurIPS 2025\] Understand Before You Generate: Self-Guided Training for Autoregressive Image Generation](../../NeurIPS2025/image_generation/understand_before_you_generate_self-guided_training_for_autoregressive_image_gen.md)

</div>

<!-- RELATED:END -->
