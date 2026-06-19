---
title: >-
  [论文解读] Penalizing Boundary Activation for Object Completeness in Diffusion Models
description: >-
  [ICCV 2025][图像生成][物体完整性] 本文深入分析了扩散模型生成不完整物体的根本原因——训练中使用的 RandomCrop 数据增强，并提出一种训练免费的边界激活惩罚方法，通过在早期去噪步骤中利用交叉注意力和自注意力约束抑制物体在图像边缘生成，将 SDv2.1 的物体不完整率从 45.7% 降至 17.3%。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "物体完整性"
  - "RandomCrop"
  - "注意力约束"
  - "训练免费"
  - "边界惩罚"
---

# Penalizing Boundary Activation for Object Completeness in Diffusion Models

**会议**: ICCV 2025  
**arXiv**: [2509.16968](https://arxiv.org/abs/2509.16968)  
**代码**: [https://github.com/HaoyXu7/Object_Completeness](https://github.com/HaoyXu7/Object_Completeness)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 物体完整性, RandomCrop, 注意力约束, 训练免费, 边界惩罚

## 一句话总结

本文深入分析了扩散模型生成不完整物体的根本原因——训练中使用的 RandomCrop 数据增强，并提出一种训练免费的边界激活惩罚方法，通过在早期去噪步骤中利用交叉注意力和自注意力约束抑制物体在图像边缘生成，将 SDv2.1 的物体不完整率从 45.7% 降至 17.3%。

## 研究背景与动机

**普遍问题**：扩散模型生成的图像经常出现物体不完整（物体被边缘裁切），如"一辆红色汽车"只显示前半部分、"一个浴缸"缺失右半部分。这一问题在 DALL-E 2 中高达 47.3%，SDv2.1 为 45.7%，即使 SDXL 仍有 18%。

**被忽视的问题**：多数研究将此视为生成随机性的固有缺陷或简单的生成失败，缺乏深入的原因分析和解决方案。

**原因排查**：
   - **数据集因素**？手动检查发现训练集中物体不完整率仅 4%，远低于生成图像的 45.7%
   - **数据增强因素？✓**：使用 RandomCrop 微调后模型的不完整率随 epoch 单调上升；使用原始图像微调则单调下降。且这一趋势对见过/未见过的类别一致

**RandomCrop 不可或缺**：虽然是罪魁祸首，但 RandomCrop 对模型多样性和泛化至关重要，且重新训练成本高昂。需要训练免费的推理时解决方案。

## 方法详解

### 核心思想

在去噪早期步骤（决定粗结构的关键阶段）通过梯度优化潜在表示，抑制物体在图像边缘出现的概率。

### 关键设计一：交叉注意力约束

提取与 prompt 中物体类别对应的交叉注意力映射 $M^{cross}_x$，该映射反映物体在图像中的语义"印记"。

### 关键设计二：自注意力约束

交叉注意力主要编码语义信息，缺乏空间结构信息。因此引入自注意力映射：
- 对 $M^{cross}_x$ 进行高斯平滑
- 使用聚类算法选取 $K$ 个关键点 $p_1, \dots, p_K$
- 提取各关键点对应的自注意力映射并取平均 $M^{self}_{avg}$

### 驱散损失 (Dispelling Loss)

$$\mathcal{L} = \alpha \cdot A_{sur} - \beta \cdot A_{inter}$$

- $A_{sur}$：边缘区域的注意力激活值（需抑制）
- $A_{inter}$：随机选取的内部区域激活值（需增强）
- 效果：隐式引导主体向图像中心聚拢

### 潜在表示优化

$$z_t' = z_t - \alpha_t \cdot \nabla_{z_t}(\mathcal{L}^{cross} + \mathcal{L}^{self})$$

仅在早期时间步 $t > T_1$ 应用（此时决定粗结构），后续步骤正常去噪以保持图像质量。

## 实验

### 方法对比

| 方法 | 需LLM | HOIR↓ | LOIR↓ | Time(s)↓ | CLIP-IQA↑ | PickScore↑ | ImageReward↑ |
|------|-------|-------|-------|----------|-----------|------------|-------------|
| SD v2.1 | ✗ | 45.7% | 32.0% | 5.51 | 0.714 | 20.03 | 0.221 |
| GLIGEN | ✗ | 34.2% | 27.9% | 8.74 | 0.672 | 20.59 | 0.177 |
| LayoutGPT | ✓ | 30.3% | 31.5% | 14.36 | 0.631 | 21.94 | 0.175 |
| SLD | ✓ | 27.1% | 23.1% | 9.54 | 0.694 | 23.07 | 0.253 |
| **Ours** | **✗** | **17.3%** | **11.7%** | **5.75** | 0.703 | **23.41** | **0.327** |

### 各模型不完整率统计

| 模型 | HOIR |
|------|------|
| DALL-E 2 | 47.3% |
| SDv2 | 45.5% |
| SDv3 | 30.1% |
| SDXL | 18.0% |

### 关键发现

- **原因验证实验**是论文核心贡献：通过控制变量（有/无 RandomCrop 微调、类别可见/不可见）严格证明了 RandomCrop 是主因
- 方法将 SDv2.1 不完整率降至 17.3%（接近 SDXL 的 18%），但无需任何训练
- 计算开销极小：仅增加 0.24s（5.51→5.75s），远低于 LLM-based 方法
- 图像质量指标（PickScore、ImageReward）不降反升，说明完整性改善不牺牲质量
- 同时约束交叉注意力和自注意力比单独使用任一更有效

## 亮点与洞察

1. **根因分析深入**：不仅提出方法，更严格分析了问题根源，具有科学价值
2. **极简设计**：无需训练、无需 LLM、无需额外网络，仅需修改推理过程增加几个梯度步
3. **与 SDXL 正交**：SDXL 从训练端缓解问题，本方法从推理端补充，两者可叠加

## 局限性

- 惩罚边缘激活可能在某些图像中过度限制布局（例如需要跨边缘构图的场景）
- 超参数 $\alpha, \beta, T_1, K$ 需要调整
- 对多物体场景的效果未充分探讨

## 相关工作

- **受控生成**: Attend-and-Excite, P2P, MasaCtrl
- **布局控制**: GLIGEN, LayoutDiffusion, BoxDiff
- **种子选择**: SeedSelect, S2ST

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 根因分析加简洁解决方案
- 技术深度：⭐⭐⭐⭐ — 实验设计严谨
- 实验充分度：⭐⭐⭐⭐ — 多模型、多指标、原因验证
- 实用价值：⭐⭐⭐⭐⭐ — 训练免费、即插即用、代价极低

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Synthesizing Near-Boundary OOD Samples for Out-of-Distribution Detection](synthesizing_near-boundary_ood_samples_for_out-of-distribution_detection.md)
- [\[ICCV 2025\] Efficient Input-Level Backdoor Defense on Text-to-Image Synthesis via Neuron Activation Variation](efficient_input-level_backdoor_defense_on_text-to-image_synthesis_via_neuron_act.md)
- [\[ICCV 2025\] ScoreHOI: Physically Plausible Reconstruction of Human-Object Interaction via Score-Guided Diffusion](scorehoi_physically_plausible_reconstruction_of_human-object_interaction_via_sco.md)
- [\[NeurIPS 2025\] LinEAS: End-to-end Learning of Activation Steering with a Distributional Loss](../../NeurIPS2025/image_generation/lineas_end-to-end_learning_of_activation_steering_with_a_distributional_loss.md)
- [\[ICLR 2026\] ContextBench: Modifying Contexts for Targeted Latent Activation](../../ICLR2026/image_generation/contextbench_modifying_contexts_for_targeted_latent_activation.md)

</div>

<!-- RELATED:END -->
