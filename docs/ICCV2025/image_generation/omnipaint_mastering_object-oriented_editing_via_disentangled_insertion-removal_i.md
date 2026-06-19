---
title: >-
  [论文解读] OmniPaint: Mastering Object-Oriented Editing via Disentangled Insertion-Removal Inpainting
description: >-
  [ICCV2025][图像生成][object removal] 提出 OmniPaint 统一框架，将物体移除与插入重新定义为互逆互补的关联任务，基于 FLUX 扩散先验并引入 CycleFlow 无配对训练机制和 CFD 无参考评估指标，仅用 3K 真实配对样本即可实现高保真的物体编辑，尤其擅长处理阴影、反射等复杂物理效果。
tags:
  - "ICCV2025"
  - "图像生成"
  - "object removal"
  - "object insertion"
  - "图像修复"
  - "扩散模型"
  - "CycleFlow"
  - "FLUX"
  - "flow matching"
  - "CFD metric"
---

# OmniPaint: Mastering Object-Oriented Editing via Disentangled Insertion-Removal Inpainting

**会议**: ICCV2025  
**arXiv**: [2503.08677](https://arxiv.org/abs/2503.08677)  
**代码**: [GitHub](https://github.com/yeates/OmniPaint)  
**领域**: 图像生成  
**关键词**: object removal, object insertion, inpainting, diffusion model, CycleFlow, FLUX, flow matching, CFD metric

## 一句话总结

提出 OmniPaint 统一框架，将物体移除与插入重新定义为互逆互补的关联任务，基于 FLUX 扩散先验并引入 CycleFlow 无配对训练机制和 CFD 无参考评估指标，仅用 3K 真实配对样本即可实现高保真的物体编辑，尤其擅长处理阴影、反射等复杂物理效果。

## 研究背景与动机

### 问题定义

物体导向的图像编辑（object-oriented editing）包括两个核心子任务：**物体移除**（object removal）和**物体插入**（object insertion）。传统方法将二者视为完全独立的任务，采用不同的技术路线分别建模，但这带来了以下问题：

**数据依赖严重**：现有方法高度依赖大规模配对真实数据或合成数据来训练，尤其是物体插入需要精确的几何对齐和逼真的物理效果集成（阴影、反射、遮挡）

**物理效果处理困难**：移除物体时不仅要消除前景语义，还需同时清除其产生的阴影、反射等物理附属效果，现有方法常残留伪影

**幻觉物体问题**：基于大型扩散模型的 inpainting 方法容易在被掩膜区域"幻觉"出不存在的物体，且缺乏有效的评估指标来检测这种问题

**部署成本高**：为物体移除和插入分别部署不同模型参数，增加了系统复杂度和计算开销

### 现有方法的不足

**物体移除方面：**
- **文本驱动方法**（InstructPix2Pix 等）：受限于文本嵌入的语义理解能力，难以精确指定复杂场景中的目标物体
- **掩膜引导方法**（PowerPaint、CLIPAway 等）：虽然控制更精确，但难以同步处理物体及其物理效果
- **合成数据方法**（MagicEraser、SmartEraser、RORem）：合成数据无法充分复现真实世界中物体的光影交互

**物体插入方面：**
- **传统融合方法**：简单的图像混合和协调无法处理复杂的物理效果（如正确的阴影投射和光照一致性）
- **特征提取方法**（AnyDoor、IMPRINT 等）：依赖 CLIP 或 DINOv2 等额外特征提取器来保持身份一致性，增加架构复杂度
- **大规模数据方法**（ObjectMate 等）：需要百万级配对数据集，数据获取成本极高

### 核心动机

作者的关键洞察是：**物体移除和插入本质上是互逆过程**——移除一个物体后再将其重新插入，应该可以恢复原始图像。基于这一观察，作者提出利用这种互补关系进行联合建模，并通过 CycleFlow 机制让训练好的移除模型帮助插入模型利用大规模无配对数据，从根本上缓解配对数据的稀缺瓶颈。

## 方法详解

### 整体框架

OmniPaint 基于 FLUX-1.dev（采用 MM-DiT 架构的 flow matching 模型）构建，通过 LoRA 微调两组参数 $\theta$（插入）和 $\phi$（移除），并引入可学习的任务嵌入实现无提示自适应控制。

### 1. 掩膜图像条件化（Masked Image Conditioning）

给定输入图像 $\mathbf{I}$ 和二值掩膜 $\mathbf{M}$，模型在掩膜图像 $\mathbf{X} = \mathbf{I} \odot (1 - \mathbf{M})$ 上操作。利用 FLUX 自带的 VAE 编码器和 2×2 patchify 层将 $\mathbf{X}$ 映射为条件 token 序列 $\mathbf{z}_c^{\mathcal{X}}$。

### 2. 参考物体条件化（Reference Object Conditioning）

对于插入任务，额外引入参考物体图像 $\mathbf{O}$：
- 使用 CarveKit 去除参考物体背景，消除背景干扰
- 将处理后的物体图像缩放至与掩膜图像相同尺寸
- 经过同样的 VAE 编码和 patchify 得到 $\mathbf{z}_c^{\mathcal{O}}$
- 最终条件 token 通过拼接获得：$\mathbf{z}_c = [\mathbf{z}_c^{\mathcal{X}}; \mathbf{z}_c^{\mathcal{O}}]$

### 3. 无提示自适应控制（Prompt-Free Adaptive Control）

鉴于任务高度依赖图像条件，文本提示反而可能引入歧义。作者用可学习的任务嵌入向量替代文本嵌入：

$$\tau_{\text{removal}}, \tau_{\text{insertion}} \sim \mathcal{N}(0, I)$$

两个向量从空字符串嵌入初始化，分别对应移除和插入任务优化。推理时通过选择不同嵌入即可切换任务。

### 4. 三阶段渐进训练流水线

**阶段一：Inpainting 预训练**
- 使用随机掩膜在 LAION 数据集上训练基础 inpainting 能力
- 同时初始化 $\theta$ 和 $\phi$ 两组 LoRA 参数
- 目标：让模型学会基本的区域填补能力

**阶段二：配对数据热身（Paired Warmup）**
- 使用 3000 对真实配对样本，分别训练移除参数 $\phi$ 和插入参数 $\theta$
- 移除方向：$\mathbf{z}_1$ 采样自物体被物理移除后的图像
- 插入方向：$\mathbf{z}_1$ 采样自保留前景物体的原始图像
- 该阶段能有效处理反射和阴影等复杂移除场景，但 3000 样本不足以保证插入时的身份一致性

**阶段三：CycleFlow 无配对后训练**
- 利用 COCO-Stuff 和 HQSeg 等大规模分割数据集作为无配对数据
- 核心问题：这些数据集缺少物体效果（阴影/反射）标注，直接训练会导致插入效果类似"复制粘贴"
- 解决方案：利用训练好的移除参数 $\phi$ 作为预处理步骤，即使 NFE=1 也能有效移除物理效果

**CycleFlow 的具体机制：**

定义两个映射：
- $F$（移除方向）：$\mathbf{z}_1' \leftarrow \mathbf{z}_t - u_t^{\phi}(\mathbf{z}_t, \mathbf{z}_c^{\mathcal{X}}, \tau_{\text{removal}}) \cdot t$
- $G$（插入方向）：$\bar{\mathbf{z}}_1 \leftarrow \mathbf{z}_t' - u_t^{\theta}(\mathbf{z}_t', \mathbf{z}_c, \tau_{\text{insertion}}) \cdot t$

循环一致性约束：先移除物体再重新插入，应恢复原始 latent：

$$\mathbf{z}_1 \rightarrow \mathbf{z}_t \rightarrow F(\mathbf{z}_t) \rightarrow \mathbf{z}_t' \rightarrow G(\mathbf{z}_t') \approx \mathbf{z}_1$$

循环损失：$\mathcal{L}_{\text{cycle}}(\theta) = \mathbb{E}_{t, \mathbf{z}_t}[\|G_\theta(\lfloor F(\mathbf{z}_t) \rfloor) - \mathbf{z}_1\|^2]$

其中 $\lfloor \cdot \rfloor$ 是梯度截断算子，冻结移除参数 $\phi$。总损失为 $\mathcal{L}_{\text{warmup}} + \gamma \mathcal{L}_{\text{cycle}}$，$\gamma = 1.5$ 为最优平衡点。

### 5. CFD 评估指标（Context-Aware Feature Deviation）

为弥补现有评估指标（如 ReMOVE）无法有效检测幻觉物体的缺陷，作者提出无参考的 CFD 指标，包含两个分量：

**幻觉惩罚项（Hallucination Penalty）：**
- 用 SAM-ViT-H 对编辑结果分割，将掩膜分为嵌套掩膜（完全在移除区域内）和重叠掩膜（跨越边界）
- 使用 DINOv2 特征相似度判断嵌套区域是否为幻觉产物
- 通过面积加权汇总：$d_{\text{hallucination}} = \sum \omega_i \cdot (1 - \mathbf{f}(\Omega_{\mathcal{M}_i^n})^\top \mathbf{f}(\Omega_{\bar{\mathcal{M}}_i}))$

**上下文一致性项（Context Coherence）：**
- 计算被修复区域与其周围背景在 DINOv2 特征空间中的偏差
- $d_{\text{context}} = 1 - \mathbf{f}(\Omega_{\mathbf{M}})^\top \mathbf{f}(\Omega_{\mathbf{B} \setminus \mathbf{M}})$

**最终指标：** $\text{CFD} = d_{\text{context}} + d_{\text{hallucination}}$，越低越好。

## 实验关键数据

### 物体移除（300-sample 自建测试集）

| 指标 | LaMa | PowerPaint | FreeCompose | FLUX-Inpaint | **OmniPaint** |
|------|-------|------------|-------------|--------------|---------------|
| FID ↓ | 105.10 | 103.61 | 88.77 | 132.60 | **51.66** |
| CMMD ↓ | 0.3729 | 0.2182 | 0.1790 | 0.3257 | **0.0473** |
| CFD ↓ | 0.3531 | 0.4031 | 0.3743 | 0.4609 | **0.2619** |
| PSNR ↑ | 20.86 | 19.46 | 21.27 | 20.86 | **23.08** |
| LPIPS ↓ | 0.1353 | 0.1428 | 0.1182 | 0.1451 | **0.0738** |

### 物体移除（RORD 1000-sample 测试集）

OmniPaint 同样全面领先：FID 19.17（次优 PowerPaint 42.65），PSNR 23.23，CFD 0.3682。

### 物体插入（565-sample 测试集）

| 指标 | AnyDoor | IMPRINT | FreeCompose | **OmniPaint** |
|------|---------|---------|-------------|---------------|
| CLIP-I ↑ | 89.26 | 90.63 | 88.17 | **92.27** |
| DINOv2 ↑ | 76.96 | 76.89 | 76.01 | **84.37** |
| CUTE ↑ | 85.26 | 86.15 | 82.86 | **90.29** |
| DreamSim ↓ | 0.2208 | 0.1854 | 0.2134 | **0.1557** |
| MUSIQ ↑ | 69.28 | 68.72 | 66.67 | **70.59** |

### 超参数分析

- **NFE（推理步数）**：NFE=1 即可有效移除物体及其物理效果（虽有模糊），NFE=18 已获得干净的移除/高保真插入，NFE=28 收益边际递减但作为默认值
- **$\gamma$（循环损失权重）**：$\gamma=0$ 无法合成物理效果（类似复制粘贴），$\gamma=1.5$ 最优平衡，$\gamma=3.0$ 过度松弛导致不自然伪影

## 亮点与洞察

1. **移除-插入互逆建模**：将两个传统上独立的任务统一为互补过程，利用训练好的移除模型帮助插入模型突破配对数据瓶颈，思路优雅且实用
2. **CycleFlow 无配对训练**：仅需 3K 真实配对样本进行热身，随后借助 CycleFlow 在大规模无配对分割数据上后训练，大幅降低了数据获取成本
3. **Prompt-Free 设计**：用可学习任务嵌入替代文本提示，避免了文本描述带来的歧义，同时支持推理时灵活切换任务
4. **CFD 指标**：首个专门针对物体移除中幻觉检测的无参考评估指标，弥补了 ReMOVE 等现有指标无法区分高 ReMOVE 分数但存在明显幻觉的情况
5. **物理效果处理**：明确建模并处理反射、阴影、遮挡等复杂物理附属效果，而非仅仅聚焦于前景物体本身
6. **极少配对数据**：相比 ObjectMate 等需百万配对样本的方法，OmniPaint 仅用 3K 配对样本即可达到 SOTA

## 局限与展望

1. **单物体插入限制**：当前仅支持单参考物体的插入，多物体同时编辑场景未被涉及
2. **CFD 指标依赖 SAM**：幻觉检测依赖 SAM 的分割质量，对于极小物体或模糊边界较难处理
3. **固定推理步数**：NFE=28 对所有场景统一设置，简单场景可能浪费计算，复杂场景可能不足
4. **CycleFlow 仅用于插入**：作者提到 warmup 已满足移除需求而未对移除使用 CycleFlow，但更困难的移除场景可能受益
5. **背景处理依赖 CarveKit**：参考物体的背景去除质量直接影响插入效果，第三方工具的鲁棒性成为瓶颈
6. **掩膜标注需求**：仍需用户提供精确的掩码指定编辑区域，端到端的自动定位未被考虑

## 相关工作与启发

- **CycleGAN → CycleFlow**：将 CycleGAN 的循环一致性思想从 GAN 框架迁移至 flow matching 框架，在 latent 空间中实现无配对训练，为其他需要配对数据的任务提供了可借鉴的范式
- **Flow Matching 的编辑适应性**：FLUX 的 flow matching 框架天然适合建模互逆过程，velocity field 的单步估计即可粗略完成任务这一特性（NFE=1 移除）值得关注
- **任务嵌入 vs 文本提示**：在高度图像条件化的编辑任务中，可学习嵌入比文本提示更高效的发现，对其他条件生成任务有参考价值
- **评估指标创新**：CFD 结合分割模型+特征相似度来检测幻觉的思路具有通用性，可扩展至其他生成任务的幻觉检测

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] EffectErase: Joint Video Object Removal and Insertion for High-Quality Effect Erasing](../../CVPR2026/image_generation/effecterase_joint_video_object_removal_and_insertion_for_high-quality_effect_era.md)
- [\[CVPR 2025\] MTADiffusion: Mask Text Alignment Diffusion Model for Object Inpainting](../../CVPR2025/image_generation/mtadiffusion_mask_text_alignment_diffusion_model_for_object_inpainting.md)
- [\[ICCV 2025\] Inpaint4Drag: Repurposing Inpainting Models for Drag-Based Image Editing via Bidirectional Warping](inpaint4drag_repurposing_inpainting_models_for_drag-based_image_editing_via_bidi.md)
- [\[ICML 2026\] AdaEraser: Training-Free Object Removal via Adaptive Attention Suppression](../../ICML2026/image_generation/adaeraser_training-free_object_removal_via_adaptive_attention_suppression.md)
- [\[ICCV 2025\] Multi-turn Consistent Image Editing](multi-turn_consistent_image_editing.md)

</div>

<!-- RELATED:END -->
