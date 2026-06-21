---
title: >-
  [论文解读] CroCoDiLight: Repurposing Cross-View Completion Encoders for Relighting
description: >-
  [ICLR 2026][图像生成][重光照] 本文揭示几何视觉预训练模型 CroCo 的隐空间里其实已经隐式编码了光照信息，于是用极小数据（比 CroCo 少两个数量级）把其 patch 隐表示解耦成"一个全局光照向量 + 逐 patch 本征向量"，从而免训练地撬动重光照、阴影去除、反照率估计等一系列光度任务。
tags:
  - "ICLR 2026"
  - "图像生成"
  - "重光照"
  - "跨视角补全 (CroCo)"
  - "光照解耦"
  - "阴影去除"
  - "反照率估计"
  - "自监督"
---

# CroCoDiLight: Repurposing Cross-View Completion Encoders for Relighting

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=GKvb3HCyNk](https://openreview.net/forum?id=GKvb3HCyNk)  
**代码**: [https://github.com/alistairfoggin/CroCoDiLight](https://github.com/alistairfoggin/CroCoDiLight)  
**领域**: 图像生成 / 重光照 / 本征图像分解  
**关键词**: 重光照, 跨视角补全 (CroCo), 光照解耦, 阴影去除, 反照率估计, 自监督  

## 一句话总结
本文揭示几何视觉预训练模型 CroCo 的隐空间里其实已经隐式编码了光照信息，于是用极小数据（比 CroCo 少两个数量级）把其 patch 隐表示解耦成"一个全局光照向量 + 逐 patch 本征向量"，从而免训练地撬动重光照、阴影去除、反照率估计等一系列光度任务。

## 研究背景与动机
**领域现状**：跨视角补全（Cross-view Completion, CroCo）是近年很火的 3D 几何视觉预训练范式——给定同一场景两个不同视角的图像，遮住其中一张的若干 patch，让模型借助另一张补全。它被证明对立体深度、光流、点云预测（DUSt3R / MASt3R）等纯几何下游任务非常有效。

**现有痛点**：而光度类任务（重光照、阴影去除、本征分解）长期受困于训练数据稀缺——像素对齐、仅光照变化的成对图像极难采集，合成数据又有 sim-to-real gap，导致这些任务往往要为每个 benchmark 单独训练、泛化性差。

**核心矛盾**：CroCo 的训练对里两张图常常光照不同，模型要补全被遮 patch，就**必须**隐式地"估计目标光照→把内容去光照→几何投影→再按目标光照重新打光"。换句话说，CroCo 是少有的在预训练目标里就被迫学会重光照的视觉模型，但这份光度知识被埋在 patch embedding 里没人用。

**本文目标**：验证"CroCo 隐空间已含光照理解"这一假设，并把它显式抽取出来变成可操作的光照隐空间。

**核心 idea**：**【提取而非从零学】** 既然光度知识已存在于冻结的 CroCo 编码器中，就只需用很小的数据训两个轻量 transformer 把隐表示**解耦**成光照与本征两部分，再**重组**回 CroCo 隐空间，便能在光照隐空间里做插值、迁移、阴影去除、反照率估计。

## 方法详解

### 整体框架
方法由四个组件串成一条"编码→解光照→换光照→重光照→解码"的管线：冻结的 CroCo v2 编码器 $E$ 把图像编成 patch 隐 $z$；解光照 transformer $I$ 把 $z$ 拆成一个全局光照隐 $s_0$ 和逐 patch 本征隐 $\{s_1..s_N\}$；重光照 transformer $R$ 再把（可替换的）光照隐与本征隐组合回 CroCo 隐空间 $z'$；最后一个单视角解码器 $D$ 把 $z'$ 高保真还原成 RGB。训练时用"同视角、不同光照"的对齐图像对来约束这一解耦。

```mermaid
flowchart LR
    X[输入图 X] --> E[冻结 CroCo 编码器 E]
    E -->|patch 隐 z| I[解光照 Transformer I]
    I -->|光照隐 s0| SW{换光照}
    I -->|本征隐 s1..sN| SW
    SW --> R[重光照 Transformer R]
    R -->|CroCo 隐 z'| D[单视角解码器 D]
    D --> Y[输出 RGB]
```

### 关键设计

**1. 单视角解码器 $D$：先把 CroCo 隐空间变成可看的图像。** 原始 CroCo 解码器是双目的，只监督被遮 patch 的重建，单视角解码画质很差。作者为此单独训一个 12 层、16 头自注意力 + DPT head 的解码器，用自编码目标——冻结 CroCo 编码器给出隐 $z$，让解码器还原原图 $X' = D(z)$。损失为感知损失与 MSE 的加权：$L_{img}=\lambda L_{LPIPS}+(1-\lambda)L_{MSE}$，取 $\lambda=0.5$。这一步不需要成对数据，可在任意图像集（用了 ImageNet）上训，等于先打通"隐空间↔RGB"的高保真通道。

**2. 解光照 / 重光照双 transformer：把光照塞进一个瓶颈向量。** 解光照时在 patch 隐前面追加一个可学习 query $z_0$（赋予一个不会与真实 patch 冲突的位置 $-1$ 的 RoPE 编码），送入 8 层自注意力的 $I$，输出 $\hat{s}=I(\hat{z})$，其中 $s_0$ 承载整图光照、其余 $s_i$ 是去掉光照影响的本征 patch。重光照 transformer $R$ 架构与 $I$ 完全相同，把光照隐与本征隐重新纠缠回 CroCo 隐 $\hat{z}'=R(\hat{s})$，只保留 patch 部分 $z'$ 解码。关键在于输入 $R$ 的光照隐 $s_0$ **不必来自原图**——换成另一张图的光照隐，就实现了重光照。

**3. 成对图像的双重自监督：本征一致 + 交叉光照。** 训练只用"同视角、像素对齐、光照不同"的图像对 $X_A, X_B$。一方面，两图的本征 patch 应当一致（几何材质不变），用 MSE 约束 $L_{intrinsic}=\frac{1}{N}\sum_i \|s_i^A - s_i^B\|^2$；另一方面，用 B 的光照隐去点亮 A 的本征隐，解码后应当还原出 B：$X'_{A,B}=D(R(\{s_0^B, s_1^A,...,s_N^A\}))$，对称地得到 $X'_{B,A}$，构成交叉光照损失 $L_{cross\text{-}light}=L_{img}(X_A, X'_{B,A})+L_{img}(X_B, X'_{A,B})$。两个损失一推一拉，迫使光照信息被瓶颈进 $s_0$、本征隐对光照不变。

**4. 光照隐空间里的下游变换 $S$ / $A$：把任务都化成"换一个光照隐"。** 既然光照与阴影都被编码进 $s_0$，下游任务就退化为学一个 $s_0\to s_0'$ 的映射。阴影去除模型 $S$（架构同 $R$、用 $R$ 权重初始化）把含阴影光照隐映成无阴影光照隐，以无阴影图编码出的光照隐为 MSE 监督；反照率估计模型 $A$ 同理把 $s_0$ 映成"反照率光照隐"。训练时除 $S$/$A$ 外全部冻结，再用 $R$ 重组、$D$ 解码。此外，由于光照隐工作在图像空间，对定相机时间序列（timelapse）可直接做光照插值、光照稳定化与时序上采样；高分辨率推理则用滑窗切成 $448\times448$ tile、逐 tile 配光照隐再混合。

## 实验关键数据

### 主实验：阴影去除（ISTD+ / SRD，无需 mask）

| 方法 | Mask | ISTD+ SSIM↑ | ISTD+ PSNR↑ | SRD SSIM↑ | SRD PSNR↑ |
|------|------|------|------|------|------|
| HomoFormer (有 mask) | Yes | 0.968 | 35.26 | 0.955 | 35.33 |
| OmniSR | No | 0.966 | 33.30 | 0.941 | 31.96 |
| StableShadowRemoval | No | 0.968 | 35.10 | 0.944 | 33.24 |
| **CroCoDiLight (S)** | No | 0.929 | 30.17 | 0.931 | 30.01 |
| **CroCoDiLight (oracle)** | No | 0.936 | 33.41 | 0.937 | 32.47 |

指标上未达 SOTA（作者指出主要被全图细微色偏拖累），但作为**联合训练、不为单一 benchmark 微调的通用模型**已具竞争力；oracle（直接用无阴影图的光照隐）证明更优的隐空间映射确实存在。

### 反照率估计（IIW WHDR，未在 IIW 上训练）

| 方法 | WHDR(%)↓ |
|------|------|
| Ordinal Shading (2023) | 24.9 |
| IntrinsicDiffusion (2024) | 17.9 |
| **CroCoDiLight** | **15.4** |
| **CroCoDiLight + 0.5** | **14.3** |

在"未训练于 IIW"的公平设定下取得 SOTA，反照率估计还只是副产物。

### 时序上采样（FloLPIPS↓）

| 方法 | Clock Shadows | Day-night | Indoor Shadows |
|------|------|------|------|
| Image Space | 0.309 | 0.041 | 0.950 |
| Latent Space (Ours) | **0.286** | 0.043 | **0.923** |

### 消融：CroCo 预训练到底有没有用

| CroCo | Pre-trained | IIW WHDR↓ | ISTD+ PSNR↑ | SRD PSNR↑ |
|------|------|------|------|------|
| ✗ (简单编码器) | — | 19.1% | 29.71 | 29.87 |
| ✓ | ✗ (随机权重) | 27.7% | 25.34 | 22.98 |
| ✓ | ✓ | **15.4%** | **30.17** | **30.01** |

### 关键发现
- 用**预训练**的 CroCo 编码器明显优于从零训的简单架构，而把 CroCo 架构随机初始化反而最差——说明增益来自 CroCo 学到的**光度知识**而非架构本身，直接坐实了核心假设。
- 仅用 57k 图像对（36k 真实 + 21k 合成）即可完成解耦，比 CroCo v2 的 7M 对少两个数量级。

## 亮点与洞察
- **视角新颖**：第一次指出几何预训练目标 CroCo 因训练对存在光照差异，被迫隐式学会"去光照—重光照"，把它从纯几何模型重新定位为光度模型，这是很漂亮的"模型已会、只需提取"叙事。
- **统一性强**：把重光照、阴影去除、反照率估计、时序插值统统归约为"在同一个光照隐空间里换/插一个向量"，一个通用模型撑起多任务，无需逐 benchmark 微调。
- **数据效率**：用两个数量级更少的数据撬动光度任务，为数据稀缺的本征/光照研究提供了一条"借力几何预训练"的低成本路径。

## 局限与展望
- **瓶颈太小损失锐利度**：每个 tile 只用一个光照隐，压缩导致硬阴影难以重建，是 fidelity 与解耦之间的取舍。
- **缺全局上下文**：高分辨率滑窗逐 tile 处理会在阴影去除时产生色偏，这正是指标落后 SOTA 的主因。
- **图像空间而非世界空间**：像素对齐训练让光照在图像空间被提取，对错位敏感，且目前缺乏对重光照的精细控制（无法从任意图或文本条件指定光照）。
- **展望**：多光照隐 / 空间变化光照图、与其他预训练编码器对比、训练光照隐 mapper 引导视频插值、可控重光照等。

## 相关工作与启发
- **重定位基础模型**思路一脉相承：ViT 隐式会分割、扩散生成器隐式会单目深度（Marigold）/本征分解，本文把这套"挖掘隐式能力"的范式搬到了 CroCo 与光度任务上。
- 与 UniRelight / Neural Gaffer / IC-Light / LumiNet 等**生成式重光照**不同，本文不追求生成新光照，而是解耦真实图像的光照以做插值与迁移，定义更清晰、更偏"理解"。
- 启发：任何在预训练目标中"被迫"解决某子问题的模型，其隐空间都可能藏着可低成本提取的能力；解耦+重组的轻量适配范式值得推广到其他"隐式已会"的场景。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — "CroCo 隐式学会重光照"是极具洞察力的发现，并用消融严谨证实，叙事与切入角度都很出彩。
- 实验充分度: ⭐⭐⭐⭐ — 覆盖阴影去除/反照率/时序插值三类任务并含关键消融；反照率达 SOTA，但阴影去除指标未领先，且对硬阴影/高分辨率色偏的分析多停留在定性。
- 写作质量: ⭐⭐⭐⭐ — 假设—验证逻辑清晰，图 1 把"CroCo 隐式重光照"讲得很直观，方法描述完整可复现。
- 价值: ⭐⭐⭐⭐ — 为数据稀缺的光度任务提供"借几何预训练"的通用低成本路线，代码开源，对重光照/本征分解社区有实际启发。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] MVCustom: Multi-View Customized Diffusion via Geometric Latent Rendering and Completion](mvcustom_multi-view_customized_diffusion_via_geometric_latent_rendering_and_comp.md)
- [\[ECCV 2024\] PanoFree: Tuning-Free Holistic Multi-view Image Generation with Cross-view Self-Guidance](../../ECCV2024/image_generation/panofree_tuning-free_holistic_multi-view_image_generation_with_cross-view_self-g.md)
- [\[ICLR 2026\] PI-Light: Physics-Inspired Diffusion for Full-Image Relighting](pi-light_physics-inspired_diffusion_for_full-image_relighting.md)
- [\[ICLR 2026\] Aligning Visual Foundation Encoders to Tokenizers for Diffusion Models](aligning_visual_foundation_encoders_to_tokenizers_for_diffusion_models.md)
- [\[ICLR 2026\] SSCP: Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning](flow-based_single-step_completion_for_efficient_and_expressive_policy_learning.md)

</div>

<!-- RELATED:END -->
