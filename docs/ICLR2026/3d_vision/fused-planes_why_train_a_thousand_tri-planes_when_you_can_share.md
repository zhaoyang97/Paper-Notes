---
title: >-
  [论文解读] Fused-Planes: Why Train a Thousand Tri-Planes When You Can Share?
description: >-
  [ICLR 2026][3D视觉][tri-plane] 提出 Fused-Planes，通过宏观-微观分解将 Tri-Plane 表示分为共享的类级基平面（macro）和对象特有的细节平面（micro），结合潜空间渲染，实现 7× 训练加速、3× 内存压缩，同时保持甚至超越独立 Tri-Plane 的重建质量。
tags:
  - "ICLR 2026"
  - "3D视觉"
  - "tri-plane"
  - "NeRF"
  - "shared representation"
  - "large-scale 3D"
  - "latent space"
---

# Fused-Planes: Why Train a Thousand Tri-Planes When You Can Share?

**会议**: ICLR 2026  
**arXiv**: [2410.23742](https://arxiv.org/abs/2410.23742)  
**代码**: [https://fused-planes.github.io](https://fused-planes.github.io)  
**领域**: 3D视觉 / 大规模3D重建  
**关键词**: tri-plane, NeRF, shared representation, large-scale 3D, latent space

## 一句话总结
提出 Fused-Planes，通过宏观-微观分解将 Tri-Plane 表示分为共享的类级基平面（macro）和对象特有的细节平面（micro），结合潜空间渲染，实现 7× 训练加速、3× 内存压缩，同时保持甚至超越独立 Tri-Plane 的重建质量。

## 研究背景与动机

**领域现状**：Tri-Planar NeRF 是强大的 3D 表示（与 2D 视觉模型兼容），但大规模场景重建需要为每个对象独立训练——千个对象 = 千次训练，计算成本极高。

**现有痛点**：(a) 独立训练忽视了同类对象间的结构相似性；(b) 已有共享表示方法（CodeNeRF）要么扩展性差（C3-NeRF 仅 20 场景），要么缺乏平面结构的优势。

**切入角度**：同类 3D 对象（如同类汽车）共享大量几何/纹理模式。将每个对象的 Tri-Plane 分解为"共享基底的加权组合 + 对象特有残差"，大幅减少重复计算。

**核心 idea**：$T_i = T_i^{mic} \oplus (W_i \cdot \mathcal{B})$——每个对象的 Tri-Plane 由少量共享基平面的加权和（macro）加上对象特有的微观特征（micro）组成。

## 方法详解

### 整体框架
Fused-Planes 要解决的是"千个对象就要训千次 Tri-Plane"的浪费问题，办法是把同类对象之间高度重复的几何/纹理模式抽出来共享，只为每个对象单独训练它真正独特的那部分。具体来说，全体对象共用一组 $M=50$ 个基平面 $\mathcal{B} = \{B_1, ..., B_{50}\}$，每个对象 $i$ 只持有一个小小的微观平面 $T_i^{mic}$ 和一个权重向量 $W_i$；推理时用 $W_i$ 把基平面线性组合出"宏观平面"，再与微观平面拼接成完整的 Fused-Plane。渲染不在 RGB 空间进行，而是先在一个低维潜空间出图，再由解码器恢复 RGB，从而把单对象训练从一小时压到十分钟以内。训练上分两段：先用少量对象把共享部件练熟，再冻结它们快速吞下剩余对象。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    IN["对象 i 的多视图图像"] --> DECOMP
    subgraph DECOMP["宏观-微观分解（设计 1）"]
        direction TB
        B["共享基平面 𝓑<br/>M=50（全局一份）"] -->|"权重 W_i 线性组合"| MAC["宏观平面 T_i^mac<br/>类级共性 22 维"]
        MIC["微观平面 T_i^mic<br/>私有细节 10 维"]
        MAC --> FUSE["拼接成 Fused-Plane<br/>32 维特征"]
        MIC --> FUSE
    end
    FUSE --> RENDER["潜空间渲染（设计 2）<br/>低维潜空间出图"]
    RENDER --> DEC["解码器<br/>潜表示→RGB"]
    DEC --> OUT["重建图像"]
    REGIME["两阶段训练（设计 3）<br/>Regime 1 练共享件 →<br/>Regime 2 冻结后吞新对象"] -.-> DECOMP
    REGIME -.-> RENDER

### 关键设计

**1. 宏观-微观分解：把每个对象的 Tri-Plane 拆成"共享基底 + 私有残差"，避免重复训练同类共性**

这一步直接针对"独立训练忽视同类结构相似性"的痛点。一个对象的特征被拆成两部分相加：宏观平面 $T_i^{mac} = \sum_k w_i^k B_k$ 是 50 个共享基平面的加权和，负责承载类级别的共性结构（22 维），微观平面 $T_i^{mic}$ 则只编码这个对象独有的细节（10 维），两者拼接得到 32 维特征。这样设计的直接好处是存储成本被压到极致——共享的基平面是全局一份，分摊到每个对象后，单对象只需存一个 480KB 的微观平面加一个 811B 的权重向量，而不是完整的 1.5MB Tri-Plane。共性越强、对象越多，这套分摊就越划算。

**2. 潜空间渲染：把渲染从 RGB 空间搬到低维潜空间，并和表示一起从零联合训练**

逐对象优化慢，很大一部分开销在高分辨率 RGB 体渲染上。这里引入一个基于 SD VAE 的图像自编码器，让 NeRF 直接在它压出的低维潜空间里渲染，分辨率大幅下降、训练随之提速。关键的一点是这个自编码器不能拿现成预训练权重直接用——预训练 VAE 的分布与 NeRF 渲染出的特征分布对不上——所以它必须与 Fused-Planes 从头联合训练，渲出的潜表示再经解码器还原成 RGB，质量才不掉。这也解释了为什么后面消融里"去掉潜空间、回到 RGB 空间"会让训练时间从 8.92 分钟反弹到 63.52 分钟。

**3. 两阶段训练策略：先用少量对象把共享件练熟，再冻结它们快速吞下剩余对象**

如果每来一个新对象都要顺带优化基平面和自编码器，共享的意义就打了折扣。于是训练分两段：Regime 1 只用前 500 个对象，联合优化全部组件（基平面、编码器、解码器），把这些全局共享的部件练到收敛；Regime 2 处理剩余对象时直接冻结已经收敛的编码器，只训练各自的微观平面和权重。共享件一旦固定，新对象的训练就退化成一个极轻量的拟合问题，这正是规模化时单对象成本能稳定维持在分钟级的原因。

### 损失函数 / 训练策略
训练目标由三项相加构成：

$$\mathcal{L} = \mathcal{L}^{latent} + \mathcal{L}^{RGB} + 0.1 \cdot \mathcal{L}^{ae}$$

其中 $\mathcal{L}^{latent}$ 监督潜空间里的渲染结果，$\mathcal{L}^{RGB}$ 约束解码回 RGB 后的图像保真，$\mathcal{L}^{ae}$（权重 0.1）则保证自编码器自身的重建能力不退化。三项配合，使得"在潜空间渲染、再解码回像素"这条链路端到端可靠。

## 实验关键数据

### 主实验

| 方法 | 训练(min/obj) | 存储(MB/obj) | ShapeNet PSNR | FPS |
|------|-------------|-------------|-------------|-----|
| Tri-Planes | 64.32 | 1.50 | 28.15 | 42.9 |
| K-Planes | 75.35 | 410.17 | 30.88 | 14.3 |
| **Fused-Planes** | **8.96** | **0.48** | **30.47** | **91.3** |
| **Fused-Planes-ULW** | **7.16** | **0.0008** | **29.02** | - |

Fused-Planes 比 Tri-Planes: 7.2× 快，3.2× 省存储，PSNR 高 2.32dB，渲染速度 2.1× 快。

### 消融实验

| 配置 | PSNR | 训练(min) | 存储(MB) |
|------|------|----------|----------|
| RGB空间（无潜空间）| 27.71 | 63.52 | 0.48 |
| 仅micro（无共享）| 27.64 | 12.84 | 1.50 |
| M=1 基平面 | 27.69 | 8.48 | 0.48 |
| M=50 基平面 | **28.64** | 8.92 | 0.48 |
| M=75 基平面 | 29.62 | 8.99 | 1348 总 |

### 关键发现
- **潜空间渲染是加速关键**：RGB→潜空间训练从 63.52 降至 8.92 分钟（7.1×加速），且质量不降
- **共享基平面有效**：M=50 是最优选择；更多基平面性能递减且增加内存
- **ULW 变体极端压缩**：完全不用 micro 平面，每个对象仅需 811B（权重向量），PSNR 仍达 29.02
- **多类训练可行**：跨 4 个 ShapeNet 类训练仅有轻微质量下降
- **规模化收益**：10000 对象时总内存仅 5GB（Tri-Planes 14.6GB，K-Planes 4TB）

## 亮点与洞察
- **微观-宏观分解**思想可迁移到其他 3D 表示——任何基于逐对象优化的方法都可以尝试提取共享基底
- **潜空间渲染与表示学习联合训练**是关键——预训练的 VAE 无法适应 NeRF 的特殊分布
- 在保持平面结构（2D 兼容）的前提下实现了接近 Instant-NGP 的训练速度，这对下游生成任务（如用平面做 diffusion）非常有价值

## 局限与展望
- 质量上限受限于 Tri-Plane 本身（30.47 vs TensoRF 36.74）——共享加速但不提升表示上限
- 需要预先定义基平面数量 M，不同类别的最优 M 可能不同
- 仅在合成数据（ShapeNet + Basel Faces）上验证，真实场景泛化未知
- 编码器冻结策略在类别分布变化大时可能失效

## 相关工作与启发
- **vs Tri-Planes**: 直接替代品——更快、更小、更好，保持平面兼容性
- **vs CodeNeRF**: CodeNeRF 用 latent code 共享，但没有平面结构；Fused-Planes 保持了平面的 2D 兼容性
- **vs Instant-NGP**: NGP 训练速度接近但存储 189MB/对象 vs 0.48MB/对象

## 评分
- 新颖性: ⭐⭐⭐⭐ 微观-宏观分解思路简洁有效，潜空间联合训练有洞察
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多基线、全面消融、规模化分析、渲染速度评估
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，实验详尽，表格丰富
- 价值: ⭐⭐⭐⭐ 大规模 3D 重建的实用加速方案，与下游生成任务兼容

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Compression of 3D Gaussian Splatting with Optimized Feature Planes and Standard Video Codecs](../../ICCV2025/3d_vision/compression_of_3d_gaussian_splatting_with_optimized_feature_planes_and_standard_.md)
- [\[ICLR 2026\] The Less You Depend, the More You Learn: Synthesizing Novel Views from Sparse, Unposed Images with Minimal 3D Knowledge](the_less_you_depend_the_more_you_learn_synthesizing_novel_views_from_sparse_unpo.md)
- [\[ICLR 2026\] A Scene is Worth a Thousand Features: Feed-Forward Camera Localization from a Collection of Image Features](a_scene_is_worth_a_thousand_features_feed-forward_camera_localization_from_a_col.md)
- [\[ICLR 2026\] YoNoSplat: You Only Need One Model for Feedforward 3D Gaussian Splatting](yonosplat_you_only_need_one_model_for_feedforward_3d_gaussian_splatting.md)
- [\[NeurIPS 2025\] You Can Trust Your Clustering Model: A Parameter-free Self-Boosting Plug-in for Deep Clustering](../../NeurIPS2025/3d_vision/you_can_trust_your_clustering_model_a_parameter-free_self-boosting_plug-in_for_d.md)

</div>

<!-- RELATED:END -->
