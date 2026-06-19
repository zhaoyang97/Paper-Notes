---
title: >-
  [论文解读] STORM: Spatial Transport Optimization by Repositioning Attention Map for Training-Free Text-to-Image Synthesis
description: >-
  [CVPR 2025][图像生成][文本到图像生成] STORM 提出基于最优传输理论的空间传输优化方法（STO），在扩散模型去噪过程中动态调整物体的注意力图位置，无需任何训练即可实现精确的空间布局控制，有效解决了 T2I 模型中"物体位置错误"这一被忽视的关键问题。 领域现状：基于扩散的 T2I 模型在高质量图像生成方面取…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "文本到图像生成"
  - "空间对齐"
  - "最优传输"
  - "注意力图重定位"
  - "无训练方法"
---

# STORM: Spatial Transport Optimization by Repositioning Attention Map for Training-Free Text-to-Image Synthesis

**会议**: CVPR 2025  
**arXiv**: [2503.22168](https://arxiv.org/abs/2503.22168)  
**代码**: 无  
**领域**: 扩散模型 / 图像生成  
**关键词**: 文本到图像生成, 空间对齐, 最优传输, 注意力图重定位, 无训练方法

## 一句话总结
STORM 提出基于最优传输理论的空间传输优化方法（STO），在扩散模型去噪过程中动态调整物体的注意力图位置，无需任何训练即可实现精确的空间布局控制，有效解决了 T2I 模型中"物体位置错误"这一被忽视的关键问题。

## 研究背景与动机

**领域现状**：基于扩散的 T2I 模型在高质量图像生成方面取得了巨大成功。无训练方法因其低成本的适应性和泛化能力而受到广泛关注。现有方法主要集中解决两类问题：(1) "物体缺失"（missing objects）——提示词中的物体在生成图像中不出现；(2) "属性错配"（mismatched attributes）——物体的颜色、纹理等属性与文本不匹配。

**现有痛点**：另一个同样关键但被忽视的问题是"物体位置错误"（mislocated objects）——生成图像中物体的空间位置无法准确对应文本描述。例如"左边一只猫，右边一只狗"，模型可能将猫和狗的位置搞反或都放在同一侧。令人惊讶的是，即便是最先进的 T2I 模型，这种看似基础的空间控制功能仍然是一个挑战。

**核心矛盾**：文本形式天然难以施加明确的空间引导——自然语言中的空间描述是模糊的（"左边"、"上方"），而像素级的空间位置是精确的。现有方法缺乏有效的机制来弥合这一语义-空间鸿沟。

**本文目标**：(1) 在无训练框架下实现精确的空间布局控制；(2) 同时改善物体缺失和属性错配问题。

**切入角度**：作者注意到扩散模型的 cross-attention 图本质上编码了物体在图像中的空间分布信息。如果能精确控制这些注意力图的空间分布，就能控制物体的生成位置。

**核心 idea**：利用最优传输理论设计空间传输代价函数，在去噪早期阶段动态"搬运"注意力图，使其对齐到目标空间位置，后期阶段则专注于细节精化。

## 方法详解

### 整体框架
输入文本提示词和目标空间布局 → 扩散模型正常去噪过程中，在早期步骤提取 cross-attention map → 使用空间传输优化（STO）根据目标布局调整 attention map 的空间分布 → 将优化后的 attention map 注入去噪过程 → 后期步骤正常去噪精化细节 → 输出空间对齐的图像。

### 关键设计

1. **空间传输优化 (Spatial Transport Optimization, STO)**:

    - 功能：将物体的注意力图从当前位置"传输"到目标空间位置
    - 核心思路：将注意力图的重分布问题建模为最优传输问题。给定当前注意力图的空间分布和目标位置，STO 求解一个传输计划（transport plan），以最小的代价将注意力质量从当前位置搬运到目标位置。具体实现上，利用 Sinkhorn 算法近似求解最优传输，得到传输矩阵后更新注意力图的空间分布。这确保了注意力质量的重新分配是平滑且全局优化的
    - 设计动机：直接修改注意力图（如硬性截断或缩放）会破坏分布的连续性，导致生成伪影。最优传输提供了一个数学上优雅且实际效果平滑的空间重分布方案

2. **空间传输代价函数 (Spatial Transport Cost)**:

    - 功能：定义注意力质量在不同空间位置间传输的代价度量
    - 核心思路：ST 代价函数综合考虑两个因素：(a) 空间距离——物体注意力图中心与目标位置的欧氏距离；(b) 空间紧凑性——鼓励注意力分布集中在目标区域而非弥散在全图。代价函数设计为距离目标越远的位置传输代价越高，从而引导注意力质量高效地聚集到正确位置
    - 设计动机：仅用空间距离作为代价不够精细——还需要考虑注意力的集中程度。代价函数的设计增强了模型对空间位置的"理解"能力

3. **阶段感知的引导策略**:

    - 功能：在去噪过程的不同阶段应用不同强度的空间引导
    - 核心思路：分析发现空间布局主要在去噪早期阶段确定（大噪声 → 粗略结构），而后期阶段负责细节精化。因此 STORM 仅在早期步骤（如前 30%-50%）施加空间传输优化，后期步骤恢复正常去噪。这避免了过度干预导致的图像质量下降
    - 设计动机：如果在所有步骤都强制空间约束，后期精化阶段的微小位置调整会被覆盖，导致生成图像不自然

### 损失函数 / 训练策略
STORM 是完全无训练的方法，不引入额外的损失函数。空间传输优化以在线方式嵌入去噪采样过程，仅修改 cross-attention map 的空间分布。

## 实验关键数据

### 主实验

| 方法 | 空间对齐准确率 | 物体缺失率↓ | 属性匹配率 | FID |
|------|-------------|-----------|-----------|-----|
| Stable Diffusion | 较低 | 较高 | 中等 | 基准 |
| Attend-and-Excite | 中等 | 改善 | 改善 | 略高 |
| Layout Guidance | 较好 | 中等 | 中等 | 略高 |
| **STORM (Ours)** | **最优** | **最低** | **最优** | **可比** |

### 消融实验

| 配置 | 空间对齐 | 生成质量 | 说明 |
|------|---------|---------|------|
| Full STORM | 最优 | 高 | 完整方法 |
| w/o STO (无最优传输) | 大幅下降 | 高 | 证明 STO 是核心 |
| w/o ST Cost | 下降 | 中等 | 代价函数设计关键 |
| 全步骤引导 | 略优对齐 | 明显下降 | 过度干预伤质量 |
| 仅后期引导 | 下降明显 | 高 | 早期阶段决定布局 |

### 关键发现
- STORM 不仅解决了物体位置错误问题，还附带改善了物体缺失和属性错配，说明空间对齐与其他生成质量指标正相关
- 早期阶段（前 30%-50%）施加空间引导效果最佳，与理论分析一致
- 最优传输比简单的 attention 缩放或截断等启发式方法效果显著更好，验证了数学优化框架的必要性
- 方法在不同 T2I backbone（如 SD 1.5、SDXL）上都有效

## 亮点与洞察
- **最优传输的创新应用**：将注意力图的空间重分布建模为最优传输问题，是一个非常优雅的形式化。这种思路可以推广到其他需要空间控制的生成任务（如视频生成、3D 生成）
- **阶段感知引导的洞察**：发现"空间布局在早期确定、细节在后期精化"的去噪阶段性质，这一发现对所有基于扩散的引导方法都有指导意义
- **三合一的效果**：一个方法同时改善三类问题（位置错误+物体缺失+属性错配），说明空间对齐是比之前认识到的更基础的问题

## 局限与展望
- 需要用户指定目标空间布局（如 bounding box 或关键点），增加了使用门槛
- 最优传输的求解增加了推理时间开销（Sinkhorn 迭代），对实时生成场景可能不够高效
- 对于重叠或遮挡的多物体场景，注意力图的分离和传输可能变得困难
- 未来可以考虑从文本中自动解析空间关系来生成布局，减少人工指定的需求
- 将 STO 扩展到 3D 空间控制（如 depth-aware 布局）是有趣的方向

## 相关工作与启发
- **vs Attend-and-Excite**: A&E 通过最大化最小注意力值来解决物体缺失问题，但不直接处理空间位置。STORM 的 STO 明确优化空间分布，更精确
- **vs Layout-guidance / GLIGEN**: 这些方法通过额外训练适配空间控制，而 STORM 完全无训练，灵活性更强
- **vs BoxDiff**: BoxDiff 也做无训练的空间控制，但基于简单的 attention 限制，STORM 的最优传输框架更优雅且效果更好

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将最优传输引入注意力图重定位是全新的视角，形式化优美
- 实验充分度: ⭐⭐⭐⭐ 消融充分，但可增加更多 backbone 和更大规模的评估
- 写作质量: ⭐⭐⭐⭐ 方法阐述清晰，问题定义准确
- 价值: ⭐⭐⭐⭐ 解决了T2I生成中一个重要但被忽视的问题，方法优雅且实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Finite Difference Flow Optimization for RL Post-Training of Text-to-Image Models](finite_difference_flow_optimization_for_rl_post-training_of_text-to-image_models.md)
- [\[NeurIPS 2025\] Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](../../NeurIPS2025/image_generation/training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)
- [\[CVPR 2025\] Minority-Focused Text-to-Image Generation via Prompt Optimization](minority-focused_text-to-image_generation_via_prompt_optimization.md)
- [\[CVPR 2025\] Stable Flow: Vital Layers for Training-Free Image Editing](stable_flow_vital_layers_for_training-free_image_editing.md)
- [\[CVPR 2025\] Noise Diffusion for Enhancing Semantic Faithfulness in Text-to-Image Synthesis](noise_diffusion_for_enhancing_semantic_faithfulness_in_text-to-image_synthesis.md)

</div>

<!-- RELATED:END -->
