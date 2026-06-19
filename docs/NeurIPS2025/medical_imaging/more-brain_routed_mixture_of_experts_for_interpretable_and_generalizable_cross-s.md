---
title: >-
  [论文解读] MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding
description: >-
  [NeurIPS 2025][医学图像][fMRI视觉解码] 提出 MoRE-Brain，一种神经科学启发的 fMRI 视觉解码框架，采用层级混合专家（MoE）架构模拟大脑视觉通路的专门化处理，配合动态时间-空间双路由机制引导扩散模型生成图像，在保持高保真重建的同时实现了高效跨被试泛化和前所未有的机制可解释性。
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "fMRI视觉解码"
  - "混合专家模型"
  - "可解释性"
  - "跨被试泛化"
  - "扩散模型"
---

# MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding

**会议**: NeurIPS 2025  
**arXiv**: [2505.15946](https://arxiv.org/abs/2505.15946)  
**代码**: [GitHub](https://github.com/yuxiangwei0808/MoRE-Brain)  
**领域**: 3D视觉  
**关键词**: fMRI视觉解码, 混合专家模型, 可解释性, 跨被试泛化, 扩散模型

## 一句话总结

提出 MoRE-Brain，一种神经科学启发的 fMRI 视觉解码框架，采用层级混合专家（MoE）架构模拟大脑视觉通路的专门化处理，配合动态时间-空间双路由机制引导扩散模型生成图像，在保持高保真重建的同时实现了高效跨被试泛化和前所未有的机制可解释性。

## 研究背景与动机

从 fMRI 信号解码视觉体验是理解人类视觉感知和开发脑-机接口的重要途径。当前 SOTA 方法（如 MindEye2、MindBridge）虽然在重建保真度上取得了显著进展，但存在三个根本性问题：

**1. 缺乏可解释性**：现有方法将 fMRI 信号当作单体输入，缺少反映神经处理原理的架构设计。虽然一些工作将大脑粗略分为"高级视觉皮层"和"低级视觉皮层"，但这种简化忽略了视觉通路中多个层级处理阶段和多个专门化皮层区域的复杂性。

**2. 跨被试泛化困难**：由于个体间大脑变异性，模型通常需要被试特定的训练。现有方法（如共享空间映射、被试特定 token）需要额外训练且难以扩展新被试。

**3. 过度依赖生成先验**：领先方法可能主要依赖扩散模型学到的生成先验（而非 fMRI 信号），导致重建质量对 fMRI 信息量的变化不敏感。

核心 idea：参考神经科学中大脑功能专门化和层级处理的原理，设计层级 MoE 架构让不同专家处理来自功能相关体素群的 fMRI 信号。个体间差异主要来自功能网络的空间拓扑差异（而非基本计算差异），因此可以共享专家计算、仅适配被试特定的路由器。

## 方法详解

### 整体框架

MoRE-Brain 采用两阶段流程：

- **阶段一**：层级 MoE 将 fMRI 信号映射到冻结的 CLIP 空间（学习专门化解码）
- **阶段二**：通过动态时间-空间路由机制将多级专家嵌入集成到 SDXL 的去噪过程中（引导图像生成）

### 关键设计

#### 1. 层级 MoE fMRI 编码器

给定输入 fMRI 体素 $\mathcal{F} \in \mathbb{R}^v$，MoE 架构不预定义大脑分区，而是学习数据驱动的体素分配。

**路由计算**：在第 $l$ 层，路由器根据输入特征计算体素-专家亲和度分数：

$$A^{(l)} = W_r^{(l)} X^{(l)}$$

对专家维度做 softmax 得到概率 $P^{(l)}$，然后对每个专家进行 Top-K 选择：

$$(S_j^{(l)}, I_j^{(l)}) = \text{TopK}(P_{:,j}^{(l)}, k)$$

其中 $k = \lfloor v / e_l \times c_f \rfloor$，容量因子 $c_f = 1$ 实现不重叠选择。

**层级结构**：从第一层 $e_0 = 2$ 个专家开始，每层加倍，共 $L = 4$ 层，最终层有 $2^4 = 16$ 个专家。这个数字近似于视觉皮层图谱中识别的功能性 ROI 数量。每个专家是简单的 MLP，处理其分配到的体素子集。

**设计动机**：模拟大脑视觉通路从粗到细的层级处理——低层专家整合广泛视觉区域的信号，高层专家聚焦于更细粒度的特定皮层亚区，形成功能专门化的内部表示。

#### 2. 动态时间-空间路由机制

**时间路由器 $\mathcal{R}_T$**：根据当前扩散时间步 $t$ 决定使用哪个层级的专家嵌入：

$$P_T = \text{softmax}\left(\frac{(W_Q t_c) \cdot (W_K \phi)^T}{\sqrt{d_k}}\right)$$

其中 $\phi = [\phi_1, ..., \phi_L]$ 是可学习的层级嵌入。为鼓励粗到细处理，通过高斯引导分布做 KL 散度正则化：

$$\bar{P}_{T,l} = \frac{\exp(-(l - \mu_t)^2 / 2\sigma^2)}{\sum_{k=1}^L \exp(-(k - \mu_t)^2 / 2\sigma^2)}, \quad \mu_t = L \cdot \frac{t}{T}$$

**空间路由器 $\mathcal{R}_S$**：在选定层级的专家嵌入基础上，根据当前噪声潜变量 $z_t$ 做空间调制：

$$C = \text{softmax}\left(\frac{\hat{W}_Q z_t \cdot \hat{W}_K E_{sel}}{\sqrt{d_k}}\right) \cdot \hat{W}_V E_{sel}$$

结果 $C$ 作为 SDXL U-Net 交叉注意力层的条件。

**设计动机**：时间路由反映了视觉处理的粗到细动态——扩散早期（高噪声）对应全局场景布局（低层专家），后期（低噪声）对应精细细节（高层专家）。空间路由模拟大脑将不同专门化区域（形状、颜色、运动）的特征整合为空间一致感知的过程。

#### 3. 跨被试泛化策略

基于神经科学发现：个体间差异主要源于功能网络空间拓扑的差异，而非基本计算差异。因此：

- **共享**：所有专家网络（核心解码计算）
- **被试特定**：仅路由器权重（将个体脑拓扑映射到专家）

在被试 1 上训练完整模型后，冻结专家，仅微调路由器就能泛化到新被试。

### 损失函数 / 训练策略

- 阶段一：将专家输出与冻结的 CLIP 嵌入对齐
- 阶段二：标准 LDM 噪声预测损失 + 时间路由的 KL 散度正则化。SDXL U-Net 通过 LoRA 微调
- 训练数据：NSD 数据集（8 位被试观看 30000+ 自然图像的 fMRI 响应）

## 实验关键数据

### 主实验（瓶颈分析揭示真实 fMRI 利用）

| 方法 | 总参数(M) | 微调参数(%) | SSIM↑ | DreamSim↓ | InceptionV3↑ |
|------|----------|-----------|-------|-----------|-------------|
| MindEye2 | 729.3 | 100% | ~0.41 | ~0.53 | ~0.94 |
| MindBridge | 552.9 | 98.48% | ~0.42 | ~0.56 | ~0.90 |
| **MoRE-Brain** | **293.4** | **44.84%** | ~0.42 | ~0.51 | ~0.96 |

MoRE-Brain 用不到一半的参数达到了竞争性能。瓶颈分析显示 MoRE-Brain 对信息量限制最为敏感（性能降最多），证明其真正利用了 fMRI 信号；而 MindEye2 即使在极端瓶颈下 CLIP-Cos 仍高达 ~0.8，暗示其过度依赖生成先验。

### 消融实验

| 配置 | SSIM↑ | Alex(2)↑ | InceptionV3↑ | DreamSim↓ |
|------|-------|---------|-------------|-----------|
| 文本+图像 (TS) | **0.415** | **0.792** | **0.962** | **0.507** |
| 仅文本 | 0.410 | 0.748 | 0.926 | 0.560 |
| 仅文本 (TS) | 0.402 | 0.754 | 0.948 | 0.542 |
| 仅图像 | 0.422 | 0.742 | 0.893 | 0.605 |
| 仅图像 (TS) | 0.382 | 0.802 | 0.957 | 0.528 |
| 文本+图像 (仅S) | 0.375 | 0.762 | 0.943 | 0.546 |
| 文本+图像 (T,固定) | 0.403 | 0.765 | 0.940 | 0.533 |
| 文本+图像 (T,硬) | 0.397 | 0.650 | 0.867 | 0.626 |
| 文本+图像 (T,软) | 0.402 | 0.764 | 0.941 | 0.543 |

双条件化（文本+图像）+ 双路由器（TS）组合达到最佳综合性能。

### 关键发现

1. **真正利用 fMRI**：瓶颈分析显示 MoRE-Brain 是唯一在信息受限时性能显著下降的方法，证明其依赖真实神经信息而非仅靠生成先验
2. **层级专门化涌现**：低层专家处理广泛视觉区域，高层专家聚焦特定皮层亚区；层内不同专家偏好不同体素群
3. **语义专门化**：高层专家发展出对特定语义类别的偏好（如某专家偏好"户外"场景），而低层专家响应较为弥散
4. **跨被试高效泛化**：仅用 2.5% 数据（1 个 session）微调路由器即可在新被试上获得合理性能
5. **ICA 分析验证**：独立成分分析显示模型学到的特征与已知的视觉处理区域和高级联合区域吻合

## 亮点与洞察

- **神经科学与深度学习的深度融合**：不是简单地"受启发"，而是将层级处理、功能专门化、粗到细动态等神经科学原理系统性地体现在架构设计中
- **可解释性即设计目标**：通过路由权重可精确追踪不同建模大脑区域如何在不同时空维度上形塑重建图像的语义和空间属性
- **瓶颈分析方法论**：提出了一种区分"真实神经解码"和"过度依赖生成先验"的系统性方法，对整个 fMRI 解码领域都有方法论意义

## 局限与展望

- 目前仅在 NSD 数据集上验证，泛化到其他 fMRI 数据集未知
- 16 个最终级专家的数量是根据视觉皮层图谱经验选择的，更大的搜索空间可能更优
- 跨被试泛化虽然高效，但仍需要一定量的新被试数据进行路由器微调
- 时间路由器的粗到细假设可能过于刚性，部分视觉信息处理并非严格粗到细

## 相关工作与启发

- MindEye2 实现了跨被试共享模型但缺乏可解释性，MoRE-Brain 通过 MoE 同时解决了泛化和可解释性
- IP-Adapter 的图像条件注入方式被应用于 fMRI 嵌入到扩散模型的集成
- MoE 架构在 fMRI 解码中的应用是全新的，可能影响其他神经影像分析任务

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将 MoE 与神经科学原理深度融合，时间-空间双路由机制设计精巧
- 实验充分度: ⭐⭐⭐⭐ 瓶颈分析、消融、跨被试、可解释性分析全面，但仅一个数据集
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、方法严谨、可视化丰富，神经科学解读有深度
- 价值: ⭐⭐⭐⭐⭐ 在 fMRI 解码领域推动了可解释性和跨被试泛化的同步进展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Zebra: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding](zebra_towards_zero-shot_cross-subject_generalization_for_universal_brain_visual_.md)
- [\[ICLR 2026\] SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding](../../ICLR2026/medical_imaging/seed_towards_more_accurate_semantic_evaluation_for_visual_brain_decoding.md)
- [\[ICML 2025\] I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts](../../ICML2025/medical_imaging/i2moe_interpretable_multimodal_interaction-aware_mixture-of-experts.md)
- [\[ICLR 2026\] Towards Interpretable Visual Decoding with Attention to Brain Representations](../../ICLR2026/medical_imaging/towards_interpretable_visual_decoding_with_attention_to_brain_representations.md)
- [\[ICCV 2025\] Beyond Brain Decoding: Visual-Semantic Reconstructions to Mental Creation Extension Based on fMRI](../../ICCV2025/medical_imaging/beyond_brain_decoding_visualsemantic_reconstructions_to_ment.md)

</div>

<!-- RELATED:END -->
