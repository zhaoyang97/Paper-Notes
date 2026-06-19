---
title: >-
  [论文解读] NormalCrafter: Learning Temporally Consistent Normals from Video Diffusion Priors
description: >-
  [ICCV 2025][视频生成][表面法线估计] NormalCrafter 基于视频扩散模型（SVD）提出视频法线估计方法，通过语义特征正则化（SFR）和两阶段训练策略，生成具有精细细节和时序一致性的法线序列，在视频基准上大幅超越现有单帧方法。 表面法线估计是 3D 重建、重光照、视频编辑等应用的基石…
tags:
  - "ICCV 2025"
  - "视频生成"
  - "表面法线估计"
  - "时序一致性"
  - "视频扩散模型"
  - "语义特征正则化"
  - "SVD"
---

# NormalCrafter: Learning Temporally Consistent Normals from Video Diffusion Priors

**会议**: ICCV 2025  
**arXiv**: [2504.11427](https://arxiv.org/abs/2504.11427)  
**代码**: [https://github.com/NormalCrafter](https://github.com/NormalCrafter)  
**领域**: 法线估计/视频理解  
**关键词**: 表面法线估计, 时序一致性, 视频扩散模型, 语义特征正则化, SVD

## 一句话总结

NormalCrafter 基于视频扩散模型（SVD）提出视频法线估计方法，通过语义特征正则化（SFR）和两阶段训练策略，生成具有精细细节和时序一致性的法线序列，在视频基准上大幅超越现有单帧方法。

## 研究背景与动机

表面法线估计是 3D 重建、重光照、视频编辑等应用的基石。虽然单帧法线估计已有长足进步（判别式方法如 DSINE、生成式方法如 Marigold），但将其应用于视频时面临严重的**时序不一致**（flickering）问题：

**判别式方法**（DSINE、Omnidata v2）受限于训练数据规模和质量，零样本泛化能力有限

**扩散式方法**（Marigold-E2E-FT、StableNormal）利用预训练扩散先验在单帧上取得 SOTA，但完全忽视时序信息，逐帧处理视频会产生闪烁

**简单加时序模块**并非最优解：在图像模型上增加时序层（如 BufferAnytime）依赖光流监督，而光流无法保证法线对应的正确性（忽略了相机运动和场景动态）

作者的核心洞察：视频扩散模型（如 SVD）本身已经学习了丰富的时空先验。然而直接将 SVD 用于法线估计会产生过度平滑的结果——因为 SVD 的中间特征缺乏足够的高层语义信息。通过将扩散特征与 DINO 语义特征对齐，可以引导模型关注场景的内在语义，从而产生精细且准确的法线预测。

## 方法详解

### 整体框架

NormalCrafter 基于 SVD（Stable Video Diffusion）构建。给定输入视频 c ∈ R^{F×W×H×3}，通过条件扩散过程生成法线序列 n ∈ R^{F×W×H×3}。

核心修改：将 SVD 的图像输入替换为逐帧拼接的噪声法线潜变量 z_t^n 和条件视频潜变量 z^c。VAE 解码器在法线数据上微调以提升重建质量。

### 关键设计

**1. 语义特征正则化（SFR）**

这是本文最重要的创新。观察发现：

- SVD 初始中间特征存在**语义模糊**——背景区域被过度模糊，丢失了细节几何信息
- DINO 编码器特征则与几何结构高度相关——能精确区分石头、植物等不同区域

SFR 的做法：从输入视频帧中提取 DINO 特征 h_dino，从扩散模型 U-Net 解码器的第二个上采样块提取中间特征 h_l，通过可学习 MLP h_φ 将 h_l 投影到 DINO 特征空间，用 patch-wise 余弦相似度进行正则化：

L_reg = -E[1/N · Σ cossim(h_dino^[n], h_φ(h_l^[n]))]

关键优势：SFR **仅在训练时引入开销**，推理时不需要 DINO 编码器，零额外推理成本。

**2. 两阶段训练策略**

- **阶段一（潜空间训练）**：训练整个 U-Net，损失 = L_DSM + L_reg。序列长度随机采样 [1,14] 帧，能够学习长程时序关系。训练 20,000 步。
- **阶段二（像素空间微调）**：仅微调空间层，解码潜变量到像素空间，使用 angular loss + L_reg。序列长度缩短为 [1,4] 帧以节省显存。训练 10,000 步。

这样设计的好处：阶段一在潜空间中以较长序列学习时序先验，阶段二在像素空间中提升空间精度，两者互补。由于阶段二只微调空间层，不会破坏阶段一学到的时序能力。

### 损失函数 / 训练策略

- **L_DSM**：去噪得分匹配损失，带噪声级别加权 λ(σ_t) = (1+σ_t²)σ_t^(-2)
- **L_reg**：DINO 语义特征对齐损失（余弦相似度）
- **L_angular**：像素空间角度损失，arccos(n* · n̂ / (||n*|| · ||n̂||))

训练数据：5 个合成数据集（Replica、3D Ken Burns、Hypersim、MatrixCity、Objaverse），涵盖室内外场景和物体序列。使用 AdamW 优化器 + 指数衰减 + 100 步 warmup，8 GPU，batch size 8，U-Net 训练约 1.5 天。

## 实验关键数据

### 主实验

**单帧+视频法线估计基准**（角度误差 mean↓/med↓，阈值比例 11.25°/22.5°/30° ↑）：

| 方法 | NYUv2 mean↓ | Scannet mean↓ | Sintel mean↓ | Sintel 22.5°↑ |
|------|:-:|:-:|:-:|:-:|
| DSINE | 16.4 | 15.5 | 34.9 | 41.5 |
| Marigold-E2E-FT | 16.2 | 14.1 | 33.5 | 43.0 |
| Lotus-D | 16.2 | 14.3 | 32.3 | 44.9 |
| **NormalCrafter** | **15.4** | **13.3** | **30.7** | **47.5** |

Sintel 上 NormalCrafter 的 mean 角度误差比第二名 Lotus-D 低 1.6°，22.5° 阈值比例高 2.6 个百分点。

### 消融实验

**各组件贡献**（在 Scannet + Sintel 视频基准上）：

| 设置 | Scannet mean↓ | Sintel mean↓ | 说明 |
|------|:-:|:-:|------|
| w/o SFR | 较高 | 较高 | 过度平滑 |
| w/o Stage 1 | 中 | 中 | 缺乏长程时序 |
| w/o Stage 2 | 中 | 中 | 缺乏空间精度 |
| w/o VAE-FT | 中 | 中 | 法线重建差 |
| **Full Model** | **13.3** | **30.7** | 最佳 |

SFR 贡献最大，移除后法线质量显著下降（过度平滑）。两阶段训练缺少任一阶段都会损害性能。

### 关键发现

1. **SFR 是关键**：DINO 语义对齐显著改善了法线细节，解决了 SVD 特征语义模糊导致的过度平滑
2. **视频基准优势巨大**：在 Sintel（大运动、动态物体）上提升最明显，体现了视频扩散先验的价值
3. **单帧也能用**：设置帧长为 1 即可做单帧估计，NYUv2 上 mean 误差降至 15.4°，超过所有单帧方法
4. **时序一致性**：y-t profile 可视化清晰显示 NormalCrafter 输出平滑，而 Marigold-E2E-FT 有明显锯齿

## 亮点与洞察

1. **SFR 思路精妙**：训练时用 DINO 对齐语义，推理时无任何额外开销——本质上是一种知识蒸馏
2. **两阶段训练策略平衡得当**：长序列潜空间→短序列像素空间的渐进式训练很实用
3. **问题分析到位**：通过 PCA 可视化清晰展示了 SVD 特征 vs DINO 特征的语义差异
4. **支持任意长视频**：通过滑动窗口推理，不受训练序列长度限制

## 局限与展望

1. **仅限合成数据训练**：所有训练数据来自合成环境，真实世界复杂场景（强反射、半透明材质）可能表现不佳
2. **推理速度**：基于扩散模型的多步去噪推理较慢，不适合实时应用
3. **VAE 重建瓶颈**：法线的高频细节可能受 VAE 编解码精度限制
4. **DINO 选择**：仅验证了 DINO 作为语义编码器，其他基础模型（CLIP、DINOv2、SAM）可能也有效
5. **缺乏真实世界 GT 评估**：主要基准（NYUv2、ScanNet、Sintel）的 GT 法线精度有限

## 相关工作与启发

- **Marigold / Marigold-E2E-FT**：单帧扩散法线估计的 SOTA，本文在此基础上引入视频先验
- **SVD (Stable Video Diffusion)**：作为视频扩散基础模型，提供时空先验
- **DSINE**：判别式法线估计 SOTA，利用射线方向和邻域法线关系
- **REPA**：SFR 的灵感来源，将扩散特征与外部表示对齐的训练策略
- **DepthCrafter**：同期的视频深度估计工作，类似的 SVD 适配思路
- **启发**：视频扩散模型蕴含丰富的几何先验，通过语义对齐可以有效激活这些先验

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Learning Temporally Consistent Video Depth from Video Diffusion Priors](../../CVPR2025/video_generation/learning_temporally_consistent_video_depth_from_video_diffusion_priors.md)
- [\[ICCV 2025\] Multi-identity Human Image Animation with Structural Video Diffusion](multi-identity_human_image_animation_with_structural_video_diffusion.md)
- [\[ICCV 2025\] Disentangled World Models: Learning to Transfer Semantic Knowledge from Distracting Videos for Reinforcement Learning](disentangled_world_models_learning_to_transfer_semantic_knowledge_from_distracti.md)
- [\[ICCV 2025\] Generating, Fast and Slow: Scalable Parallel Video Generation with Video Interface Networks](generating_fast_and_slow_scalable_parallel_video_generation_with_video_interface.md)
- [\[ICCV 2025\] Long-Context State-Space Video World Models](long-context_state-space_video_world_models.md)

</div>

<!-- RELATED:END -->
