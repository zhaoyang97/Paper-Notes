---
title: >-
  [论文解读] IntLoRA: Integral Low-rank Adaptation of Quantized Diffusion Models
description: >-
  [ICML2025][图像生成][LoRA] 提出 IntLoRA，通过整数型低秩参数实现量化扩散模型的微调，合并权重后无需额外 PTQ 即可直接获得量化推理权重，兼顾训练与推理效率。 大规模文本到图像扩散模型（如 Stable Diffusion、SDXL、FLUX）在个性化生成任务中表现出色，但全参数微调在消费级 GPU…
tags:
  - "ICML2025"
  - "图像生成"
  - "LoRA"
  - "网络量化"
  - "扩散模型"
  - "低秩适配"
  - "整数运算"
  - "推理加速"
---

# IntLoRA: Integral Low-rank Adaptation of Quantized Diffusion Models

**会议**: ICML2025  
**arXiv**: [2410.21759](https://arxiv.org/abs/2410.21759)  
**代码**: [csguoh/IntLoRA](https://github.com/csguoh/IntLoRA)  
**领域**: 图像生成  
**关键词**: LoRA, 网络量化, 扩散模型, 低秩适配, 整数运算, 推理加速

## 一句话总结

提出 IntLoRA，通过整数型低秩参数实现量化扩散模型的微调，合并权重后无需额外 PTQ 即可直接获得量化推理权重，兼顾训练与推理效率。

## 研究背景与动机

大规模文本到图像扩散模型（如 Stable Diffusion、SDXL、FLUX）在个性化生成任务中表现出色，但全参数微调在消费级 GPU 上受限于显存。LoRA 和量化技术的结合（如 QLoRA）已可在量化权重上直接微调，降低了训练成本。

**核心问题**：现有方法在训练时使用 FP16 的低秩参数，合并时需将量化的预训练权重转回 FP16，部署时再做一次 PTQ。这个流程存在两大缺陷：

**流程冗余**：训练→反量化→合并→再量化，额外 PTQ 步骤增加了部署复杂度

**性能下降严重**：低比特（如 4-bit）下 PTQ 导致显著质量退化（DINO 从 0.48 降至 0.21）

根本原因在于预训练权重（INT）与适配权重（FP16）的**算术类型不一致**，导致合并后权重必须回到浮点域。

## 方法详解

IntLoRA 的核心思想：让低秩适配参数也工作在整数域，使合并后的权重天然为量化格式。包含三项关键技术：

### 1. 适配-量化分离（AQS）

原始 LoRA 将 $\mathbf{AB}$ 零初始化以保持微调起点与预训练一致，但全零分布对量化不友好（缩放因子 $s=0$ 导致除零）。AQS 引入辅助矩阵 $\mathbf{R}$ 将梯度与量化解耦：

$$\mathbf{W'} = \mathcal{Q}[\mathbf{W} - \text{sg}(\mathbf{R})] + \text{sg}(\mathbf{R}) + \mathbf{AB}$$

其中 $\text{sg}(\cdot)$ 为停止梯度操作。$\mathbf{AB}$ 仍零初始化保持原始 LoRA 梯度，而 $\text{sg}(\mathbf{R}) + \mathbf{AB}$ 提供非零分布便于量化。$\mathbf{R}$ 可通过分布统计和固定随机种子在线生成，无需额外存储。

### 2. 乘法低秩适配（MLA）

原始 LoRA 采用加法形式 $\mathbf{W} + \mathbf{AB}$，当两者独立量化后无法直接合并（需共享量化器，限制参数空间）。MLA 将加法重写为等价的乘法形式：

$$\mathbf{W'} = \underbrace{\left[s \cdot \mathbf{I} + \frac{1}{\mathbf{W}_{\text{round}} - z} \odot (\mathbf{R} + \mathbf{AB})\right]}_{\text{适配项（可训练）}} \odot \underbrace{(\mathbf{W}_{\text{round}} - z)}_{\text{预训练项（整数）}}$$

适配项与预训练项可使用**独立量化器**，消除了共享量化参数的约束。

### 3. 方差匹配控制（VMC）

辅助矩阵 $\mathbf{R}$ 的方差 $\sigma_\mathbf{R}$ 存在选择困境：过大则量化后无法重建原始 $\mathbf{W}$，过小则适配项分布不够集中于零附近。VMC 通过方差比对齐：

$$\mathbf{R}^* = r^\alpha \cdot \mathbf{R}, \quad r = \frac{\sigma_\mathbf{W}}{\sigma_\mathbf{R}}$$

标量 $\alpha$ 作为精细调节指数，在量化难度与信息保留之间取得平衡。

### 两种实现版本

**IntLoRA_MUL**（整数乘法）：对适配项施加均匀仿射量化，合并通过整数 Hadamard 乘积完成：

$$\mathbf{W'} = \bar{s} \cdot (\mathbf{U}_{\text{round}} - \bar{z}) \odot (\mathbf{W}_{\text{round}} - z)$$

**IntLoRA_SHIFT**（位移）：对适配项施加 $\log_2$ 量化，通过位移操作完成适配：

$$\mathbf{W'} = \text{sign}(\mathbf{V}) \odot [(\mathbf{W}_{\text{round}} - z) \gg \text{shift}]$$

训练时使用 STE（直通估计器）反向传播量化梯度。

## 实验关键数据

### 主体驱动生成（DreamBooth, SD v1.5）

| 方法 | 位宽 | DINO↑ | CLIP-I↑ | CLIP-T↑ |
|------|------|-------|---------|---------|
| LoRA (FP16) | W16A16 | 0.4828 | 0.6968 | 0.2954 |
| QLoRA | W8A8 | 0.4153 | 0.6661 | 0.2824 |
| IR-QLoRA | W8A8 | 0.4070 | 0.6630 | 0.2841 |
| **IntLoRA_MUL** | **W8A8** | **0.4498** | **0.6882** | **0.2858** |
| QLoRA | W4A8 | 0.2136 | 0.6134 | 0.2510 |
| QA-LoRA | W4A8 | 0.4127 | 0.6897 | 0.2700 |
| **IntLoRA_MUL** | **W4A8** | **0.4242** | **0.6913** | **0.2710** |

### 可控生成 FID↓（ControlNet）

| 方法 | 8-bit S2I | 8-bit L2F | 4-bit S2I | 4-bit L2F |
|------|-----------|-----------|-----------|-----------|
| LoRA (FP16) | 31.39 | 37.50 | 31.39 | 37.50 |
| QLoRA | 31.09 | 38.88 | 71.75 | 117.37 |
| IR-QLoRA | 31.81 | 36.30 | 35.83 | 39.63 |
| **IntLoRA_MUL** | **31.08** | **37.52** | **30.87** | **33.62** |

### 训练与推理效率（SD v1.5, RTX 3090）

| 方法 | 位宽 | 训练速度 | 模型大小 | 需要PTQ |
|------|------|----------|----------|---------|
| LoRA | W32A32 | 0.68s/img | 7700MB | ✔ |
| QLoRA | W8A8 | 0.85s/img | 1925MB | ✔ |
| IntLoRA_MUL | W8A8 | 0.87s/img | 1925MB | **✘** |
| QLoRA | W4A8 | 0.85s/img | 963MB | ✔ |

IntLoRA 训练速度与 QLoRA 持平，但推理时**省去 PTQ 步骤**，直接获得量化权重。

## 亮点与洞察

1. **消除推理 PTQ**：核心贡献在于将适配参数转为整数运算，合并后权重天然量化，端到端免 PTQ
2. **MLA 的数学等价性**：从加法到乘法的重构保持数学等价，却解耦了预训练与适配的量化器约束
3. **AQS 梯度-量化解耦**：巧妙利用 stop-gradient 分离"学习需要零初始化"与"量化需要非零分布"的矛盾
4. **VMC 理论分析扎实**：从方差-相关系数的 trade-off 出发推导最优辅助矩阵，有数学依据
5. **辅助矩阵零存储开销**：$\mathbf{R}$ 通过固定种子在线生成，不增加模型存储
6. **4-bit 场景优势显著**：QLoRA 在 W4A8 下 DINO 暴跌至 0.21，IntLoRA 维持 0.42，gap 巨大

## 局限与展望

1. **仅验证扩散模型**：未在 LLM 上验证，尽管技术上可迁移，泛化性待考察
2. **训练速度未加速**：训练阶段 STE 和辅助矩阵引入额外计算，速度并未优于 QLoRA
3. **$\alpha$ 超参搜索**：VMC 中的指数 $\alpha$ 需要针对不同任务搜索，自动化程度不足
4. **激活量化仍为PTQ**：论文聚焦权重量化，激活量化仍用传统方案，未一体化解决
5. **IntLoRA_SHIFT 表现弱于 IntLoRA_MUL**：$\log_2$ 量化精度受限，实际优势主要来自 MUL 版本

## 相关工作与启发

- **QLoRA / IR-QLoRA**：在量化权重上做LoRA但仍需PTQ，是本文直接改进目标
- **QA-LoRA**：通过分组量化共享参数，损失适配能力
- **EfficientDM**：LoRA用于扩散模型QAT，但训练开销大
- **启发**：整数域适配的思路可推广到其他 PEFT 方法（如 Adapter），也可探索混合精度策略

## 评分

- 新颖性: ⭐⭐⭐⭐ 加法→乘法重构+AQS解耦+VMC调控三件套设计精巧
- 实验充分度: ⭐⭐⭐⭐ 覆盖主体生成/可控生成/风格定制三任务，含消融分析
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰，图示直观，问题定义明确
- 价值: ⭐⭐⭐⭐ 解决量化LoRA落地的实际痛点，4-bit场景优势突出

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Flat-LoRA: Low-Rank Adaptation over a Flat Loss Landscape](flat-lora_low-rank_adaptation_over_a_flat_loss_landscape.md)
- [\[NeurIPS 2025\] GraLoRA: Granular Low-Rank Adaptation for Parameter-Efficient Fine-Tuning](../../NeurIPS2025/image_generation/gralora_granular_low-rank_adaptation_for_parameter-efficient_fine-tuning.md)
- [\[NeurIPS 2025\] StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold](../../NeurIPS2025/image_generation/stella_subspace_learning_in_low-rank_adaptation_using_stiefel_manifold.md)
- [\[ICCV 2025\] Transformed Low-rank Adaptation via Tensor Decomposition and Its Applications to Text-to-image Models](../../ICCV2025/image_generation/transformed_low-rank_adaptation_via_tensor_decomposition_and_its_applications_to.md)
- [\[CVPR 2026\] Low-Rank Residual Diffusion Models](../../CVPR2026/image_generation/low-rank_residual_diffusion_models.md)

</div>

<!-- RELATED:END -->
