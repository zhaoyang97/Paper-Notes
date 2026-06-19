---
title: >-
  [论文解读] NGP-RT: Fusing Multi-Level Hash Features with Lightweight Attention for Real-Time Novel View Synthesis
description: >-
  [ECCV 2024][3D视觉][NeRF] 提出NGP-RT，通过轻量注意力机制聚合多级显式哈希特征替代per-point MLP，并引入占用距离网格减少光线行进中的内存访问，在Mip-NeRF 360数据集上实现1080p 108fps的实时NeRF渲染。 领域现状： Instant-NGP通过多级哈希网格存储隐式特征…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "NeRF"
  - "实时渲染"
  - "哈希特征"
  - "注意力机制"
  - "占用距离网格"
---

# NGP-RT: Fusing Multi-Level Hash Features with Lightweight Attention for Real-Time Novel View Synthesis

**会议**: ECCV 2024  
**arXiv**: [2407.10482](https://arxiv.org/abs/2407.10482)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: NeRF, 实时渲染, 哈希特征, 注意力机制, 占用距离网格

## 一句话总结

提出NGP-RT，通过轻量注意力机制聚合多级显式哈希特征替代per-point MLP，并引入占用距离网格减少光线行进中的内存访问，在Mip-NeRF 360数据集上实现1080p 108fps的实时NeRF渲染。

## 研究背景与动机

**领域现状**: Instant-NGP通过多级哈希网格存储隐式特征+浅层MLP解码，实现了快速训练和高质量渲染。Deferred NeRF架构（SNeRG、MERF）通过存储显式颜色/密度并将MLP从逐点执行减少到逐射线执行，实现了实时渲染。

**现有痛点**: (a) Instant-NGP的渲染瓶颈在于**逐点MLP执行**——每个采样点都需要MLP将隐式特征解码为颜色和密度，限制了渲染速度（~10fps@1080p）；(b) SNeRG的低分辨率稀疏网格无法覆盖Instant-NGP高分辨率特征的细节；(c) MERF的简单求和聚合对高分辨率NGP特征的哈希碰撞不够灵活。

**核心矛盾**: 多级哈希特征表达能力强但依赖MLP聚合，去掉MLP会导致**哈希碰撞无法消歧**——多个不同3D位置映射到同一哈希表项，需要MLP隐式学习的掩码函数来区分。

**本文目标**: 在保留多级哈希特征强表达能力的同时，去除逐点MLP的计算负担，实现>100fps的实时NeRF渲染。

**切入角度**: 将MLP的隐式掩码功能显式化为**可学习的注意力参数**，用极轻量的通道加权和替代MLP，同时引入占用距离网格减少光线行进开销。

**核心 idea**: MLP在NGP中的核心作用是为哈希碰撞的不同位置分配不同重要性，这可以用空间可变的轻量注意力参数替代。

## 方法详解

### 整体框架

NGP-RT将多级哈希特征分为粗粒度（低分辨率）和细粒度（高分辨率）两部分。粗粒度特征$\tilde{\mathbf{f}}$由辅助NGP模型解码后烘焙到低分辨率稀疏体素网格$\tilde{\mathcal{F}}$中；细粒度特征$\hat{\mathbf{f}}$存储为$L$级显式哈希特征，通过轻量注意力机制聚合。最终特征$\mathbf{f} = \tilde{\mathbf{f}} + \hat{\mathbf{f}}$送入deferred NeRF体渲染管线：先对密度和漫反射颜色做体积累积，最后一个tiny MLP仅对每条射线执行一次。

### 关键设计

1. **轻量注意力机制**: 核心思想是为每个细粒度级别$l$学习两个空间可变的注意力参数$\omega^l$（密度）和$\beta^l$（颜色特征），通过通道加权和聚合多级特征：

$$\hat{\mathbf{f}} = \text{Att}(\hat{\mathbf{f}}^1, \ldots, \hat{\mathbf{f}}^L; \mathbf{a}) = \left[\sum_{l=1}^{L} \omega^l \cdot \hat{\sigma}^l, \quad \sum_{l=1}^{L} \beta^l \cdot \hat{\mathbf{c}}_d^l, \quad \sum_{l=1}^{L} \beta^l \cdot \hat{\mathbf{v}}_s^l\right]$$

其中$\mathbf{a} = [\omega^1, \beta^1, \ldots, \omega^L, \beta^L]$。设计动机：(a) 密度和颜色是不同模态，应独立加权；(b) 参数空间可变性模拟了MLP隐式掩码功能——哈希碰撞的不同位置通过不同的注意力权重来消歧。

2. **注意力参数的训练与推理分离**: 训练时用辅助NGP模型（隐式哈希特征+浅层MLP）解码出注意力参数$\mathbf{a}$和粗粒度特征$\tilde{\mathbf{f}}$，并在网格角点$L_C$分辨率处评估后三线性插值。训练完成后丢弃辅助NGP，将参数烘焙到稀疏网格$\mathcal{A}$中，推理时直接查表插值：

$$\tilde{\mathbf{f}} = \text{Interp}(\tilde{\mathcal{F}}, \mathbf{x}), \quad \mathbf{a} = \text{Interp}(\mathcal{A}, \mathbf{x})$$

3. **占用距离网格（Occupancy Distance Grid）**: 现有方法逐体素检查占用状态导致大量全局内存访问。NGP-RT预计算一个$256^3$的距离网格$\mathcal{G}$，存储每个位置到最近占用体素的距离（uint8整数，单位为体素大小），允许空旷区域直接跳过：

$$s_{\mathbf{p}} = \begin{cases} v \cdot \mathcal{G}_{\mathbf{p}}, & \text{if } \mathcal{G}_{\mathbf{p}} > 0 \\ s_{\mathcal{O}}, & \text{otherwise} \end{cases}$$

仅当位置未占用且等出占用级别分辨率小于$\mathcal{G}$分辨率时才查询$\mathcal{G}$，减少40%+的冗余行进点。

### 损失函数 / 训练策略

标准NeRF的多视角光度重建损失。训练分两阶段：(1) 端到端优化多级显式哈希特征、辅助NGP的隐式特征与MLP参数、以及deferred NeRF的tiny MLP；(2) 训练完成后烘焙粗粒度特征和注意力参数到网格，丢弃辅助NGP。

## 实验关键数据

### 主实验

**Mip-NeRF 360 全场景 (1080p)**:

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ |
|------|-------|-------|--------|------|
| Instant-NGP | 25.62 | 0.703 | 0.301 | 10.4 |
| MERF | 25.24 | 0.722 | 0.311 | 119 |
| BakedSDF | 24.51 | 0.697 | 0.309 | >60 |
| Gaussian-7K | 25.91 | 0.766 | 0.288 | 107 |
| **NGP-RT** | **25.64** | **0.737** | **0.299** | **108** |

NGP-RT以~10倍于Instant-NGP的速度达到相当质量。相比MERF在室内场景显著更好（29.25 vs 27.80 PSNR）。

### 消融实验

**特征聚合方式对比（L=4）**:

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ |
|------|-------|-------|--------|------|
| MLP | 26.22 | 0.764 | 0.268 | 14.7 |
| SUM | 25.51 | 0.719 | 0.315 | 66.9 |
| Shared-Att (Inv) | 25.69 | 0.738 | 0.295 | 65.9 |
| Separate-Att (V) | **26.05** | **0.753** | **0.280** | 61.8 |

Separate-Att(V)在质量上接近MLP但**速度快4倍以上**，证明轻量注意力是MLP的优秀替代。

**细粒度级别数$L$的影响**:

| L | PSNR↑ | FPS↑ |
|---|-------|------|
| 2 | 25.64 | 108 |
| 3 | 25.93 | 79.7 |
| 4 | 26.05 | 61.8 |

**占用距离网格效果**:

| 配置 | #行进点↓ | 时间(ms)↓ |
|------|----------|-----------|
| 无距离网格 | 85.1 | 9.98 |
| 有距离网格 | **46.7** | **9.26** |

行进点减少45%，加速7-10%。

### 关键发现

- 轻量注意力的核心价值在于**空间可变性**——空间不变的注意力（Inv）效果明显不如空间可变的（V），因为哈希碰撞本身是空间相关的
- 密度和颜色分离注意力（Separate-Att）优于共享注意力（Shared-Att），确认不同模态需要不同的重要性分配
- 可视化显示注意力有效将纹理细节分配到不同哈希级别，且碰撞位置的大部分被赋予小权重以避免梯度干扰
- L=4比L=3提升有限但速度代价明显,L=2是速度-质量最优权衡

## 亮点与洞察

- **MLP功能的精确分析**: 深入分析了MLP在NGP中的真正作用——不是简单的特征变换，而是为哈希碰撞提供隐式掩码，这一洞察指导了轻量替代方案的设计
- **减少90%以上MAC运算**: 轻量注意力相比浅层MLP减少>90%乘累加运算
- **全局内存访问优化**: 占用距离网格的思路简洁实用，仅16MB（$256^3$×uint8）额外存储换取7-10%加速
- **NeRF实时化的NeRF-native方案**: 证明了NeRF体渲染框架本身可以做到100+fps，不必转向3DGS

## 局限与展望

- 室外场景PSNR略低于MERF，高频细微结构建模能力仍有不足
- 质量仍低于Zip-NeRF等离线高质量方法（25.64 vs 28.54 PSNR）
- $256^3$占用距离网格对于超大场景可能需要更精细的多级设计
- 烘焙过程引入离散化误差，可能对极精细结构造成损失
- 与3DGS相比，在质量相当的前提下存储效率和灵活性仍有差距

## 相关工作与启发

- **Instant-NGP**: NGP-RT的直接基线，继承其多级哈希特征的强表达力
- **SNeRG**: 开创了deferred NeRF架构，将MLP减少到逐射线执行
- **MERF**: 改进SNeRG的存储方案（三平面+稀疏网格），NGP-RT进一步用多级哈希特征增强表达力
- **3DGS**: 代表了渲染效率的另一条路线（基于光栅化vs体渲染），本文证明NeRF路线也可达到实时

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 轻量注意力替代MLP解哈希碰撞的洞察新颖，占用距离网格思路简洁
- **实验充分度**: ⭐⭐⭐⭐ — 在标准Mip-NeRF 360全面评测，消融详尽
- **写作质量**: ⭐⭐⭐⭐ — 对MLP功能的分析清晰，与SNeRG/MERF的对比图直觉明了
- **价值**: ⭐⭐⭐⭐ — 推动了NeRF实时渲染的工程实用性，但3DGS的竞争压力较大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] A Compact Dynamic 3D Gaussian Representation for Real-Time Dynamic View Synthesis](a_compact_dynamic_3d_gaussian_representation_for_realtime_dy.md)
- [\[ECCV 2024\] MegaScenes: Scene-Level View Synthesis at Scale](megascenes_scene-level_view_synthesis_at_scale.md)
- [\[CVPR 2026\] LagerNVS: Latent Geometry for Fully Neural Real-time Novel View Synthesis](../../CVPR2026/3d_vision/lagernvs_latent_geometry_for_fully_neural_real-time_novel_view_synthesis.md)
- [\[ECCV 2024\] CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians](coherentgs_sparse_novel_view_synthesis_with_coherent_3d_gaussians.md)
- [\[ECCV 2024\] VersatileGaussian: Real-Time Neural Rendering for Versatile Tasks Using Gaussian Splatting](versatilegaussian_real-time_neural_rendering_for_versatile_tasks_using_gaussian_.md)

</div>

<!-- RELATED:END -->
