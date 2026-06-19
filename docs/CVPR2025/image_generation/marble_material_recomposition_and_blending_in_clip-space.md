---
title: >-
  [论文解读] MARBLE: Material Recomposition and Blending in CLIP-Space
description: >-
  [CVPR 2025][图像生成][material editing] 仅在 CLIP 空间操作材质嵌入，通过定向注入 UNet 中的材质响应层实现材质迁移和混合，并通过轻量 MLP 预测属性编辑方向实现粗糙度/金属度/透明度/发光的参数化控制，无需微调扩散模型。 领域现状： 材质编辑是计算机图形学核心问题…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "material editing"
  - "CLIP-space"
  - "material blending"
  - "parametric control"
  - "扩散模型"
---

# MARBLE: Material Recomposition and Blending in CLIP-Space

**会议**: CVPR 2025  
**arXiv**: [2506.05313](https://arxiv.org/abs/2506.05313)  
**代码**: [项目页面](https://marblecontrol.github.io/)  
**领域**: 图像生成  
**关键词**: material editing, CLIP-space, material blending, parametric control, diffusion model

## 一句话总结

仅在 CLIP 空间操作材质嵌入，通过定向注入 UNet 中的材质响应层实现材质迁移和混合，并通过轻量 MLP 预测属性编辑方向实现粗糙度/金属度/透明度/发光的参数化控制，无需微调扩散模型。

## 研究背景与动机

**领域现状**: 材质编辑是计算机图形学核心问题，传统方法需要显式估计几何、光照和材质属性。近期扩散模型方法分为：ZeST（零样本材质迁移但粒度粗）和 Alchemist（微调 SD 实现细粒度控制但可能破坏先验）。

**现有痛点**:
1. **ZeST 粒度不足**: 仅支持高层材质迁移，无法进行粗糙度等细粒度属性控制。
2. **Alchemist 过拟合风险**: 在合成数据上微调整个扩散模型，可能破坏模型内嵌的物体先验。
3. **几何泄露**: ZeST 将 CLIP 特征注入 UNet 所有 block，导致非材质信息（如身份特征）也被传递，引起几何变形。

**核心矛盾**: 需要在保持基础扩散模型完好的前提下，实现既能粗粒度材质迁移又能细粒度属性调控的统一框架。

**本文切入角度**: 保持 SD 模型冻结，仅操作 CLIP 空间的材质嵌入——找到 UNet 中专门负责材质归因的层进行定向注入，并在 CLIP 空间学习属性编辑方向。

## 方法详解

### 整体框架

基于 ZeST 架构改进，使用 SDXL 修复模型 $\mathcal{S}$：
1. 输入：目标图像 $I$（含前景 mask $F_I$ 和深度图 $D_I$）+ 材质参考图 $I_m$
2. CLIP 编码 $I_m$ 得到材质嵌入 $z_m$
3. 通过 IP-Adapter 仅注入 UNet 的"材质层"（靠近瓶颈层的特定 block）
4. 生成结果：$I_{gen} = \mathcal{S}(I_{init}, F_I, D_I, f(z_m))$

### 关键设计

**1. 定向材质层注入 (Targeted Material Block Injection)**
- **功能**: 通过穷举可视化实验，找到 UNet 中专门负责材质归因的 attention block（靠近瓶颈层），仅向该层注入 CLIP 材质嵌入，而非注入所有层。
- **核心思路**: 受 InstantStyle 启发，逐层注入并观察输出，发现材质归因和风格归因由同一个 block 负责。仅注入该层时，物体几何和光照得到更好保留。
- **设计动机**: 全层注入导致 CLIP 特征中的非材质信息（如物体身份语义）泄露，引起几何变形。定向注入实现了材质信息与几何/光照的解耦。

**2. CLIP 空间材质混合 (Material Blending)**
- **功能**: 给定两个材质参考图的 CLIP 嵌入 $z_{m_1}$ 和 $z_{m_2}$，通过线性插值生成混合材质：$I_{gen} = \mathcal{S}(I_{init}, F_I, D_I, f(\alpha z_{m_1} + (1-\alpha) z_{m_2}))$。
- **核心思路**: 利用 CLIP 空间中可解释的方向性（类似 GAN latent space 的可解释性），插值嵌入可产生语义上连续的混合材质。三种配置均有效：(1) 不同物体不同材质，(2) 同材质不同属性值，(3) 同物体同材质不同属性。
- **设计动机**: CLIP 空间的线性结构天然支持语义插值，无需额外训练。

**3. 参数化属性控制 (Parametric Control via MLP)**
- **功能**: 对每个属性（粗糙度/金属度/透明度/发光），训练一个 2 层 MLP $p_\theta$，从材质参考图和编辑强度 $\delta$ 预测 CLIP 空间的编辑方向。推理时：$z_{m_{a+\delta}} = \text{CLIP}(I_m) + p_\theta(I_m, \delta)$。
- **核心思路**: 用 Blender 渲染 300 个合成物体（250 训练 / 50 验证），每个物体在不同属性值下渲染，计算 CLIP 特征差值并用 SVD 低秩近似（去噪），然后训练 MLP 预测这些低秩方向。训练目标：$\arg\min_\theta [\text{cossim}(s_{m_{a+\delta}}, p_\theta) + \text{MSE}(s_{m_{a+\delta}}, p_\theta)]$。
- **设计动机**: (1) 不微调扩散模型，保留原始先验；(2) SVD 去噪解决 CLIP 特征噪声问题；(3) 各属性 MLP 独立训练，可组合使用实现多属性同时编辑。

### 损失函数 / 训练策略

- **MLP 训练**: 余弦相似度损失 + MSE 损失联合优化
- **极少数据**: 仅需 250 个合成物体训练，低至 16 个物体仍可获得不错结果
- **SVD 低秩近似**: 对堆叠的编辑方向矩阵做 SVD，保留解释 67%-80% 方差的主分量
- **不微调 SD 模型**: 所有属性控制器独立训练，可任意组合

## 实验关键数据

### 主实验（参数化属性控制）

| 方法 | PSNR↑ | LPIPS↓ | CLIP↑ | DreamSim↓ |
|---|---|---|---|---|
| **粗糙度** | | | | |
| Concept Slider (Images) | 18.87 | 0.356 | 0.597 | 0.567 |
| **MARBLE** | **26.56** | **0.056** | **0.931** | **0.129** |
| **金属度** | | | | |
| Concept Slider (Images) | 19.45 | 0.317 | 0.655 | 0.479 |
| **MARBLE** | **26.82** | **0.053** | **0.928** | **0.121** |
| **透明度** | | | | |
| Concept Slider (Images) | 19.85 | 0.346 | 0.639 | 0.525 |
| **MARBLE** | **26.99** | **0.070** | **0.905** | **0.163** |
| **发光** | | | | |
| Concept Slider (Images) | 16.92 | 0.301 | 0.661 | 0.509 |
| **MARBLE** | **19.73** | **0.111** | **0.890** | **0.213** |

### 用户研究

87.5% 的参与者（16 人中 14 人）选择 MARBLE 的结果优于 Image Concept Slider。

### 数据效率消融

| 训练物体数 | PSNR↑ | DreamSim↓ |
|---|---|---|
| 8 | ~25 | ~0.18 |
| 16 | ~26 | ~0.15 |
| 250 (完整) | 26.99 | 0.163 |

仅 16 个物体即可获得接近完整数据集的性能。

### 关键发现

1. **定向注入 vs 全层注入**: 仅注入材质层显著改善几何保持——全层注入时物体可能出现几何变形（如玩具上长出"手"）。
2. **属性解耦**: 粗糙度和金属度等属性可独立控制且互不干扰，支持多属性网格编辑。
3. **风格泛化**: 由于不修改 SD 模型权重，学到的编辑方向可泛化到动漫、油画等不同风格。
4. **Baseline 失败模式**: InstructPix2Pix 常导致几何/颜色变化；Concept Slider 需要 DDIM 逆变换导致重建不准确，且无法捕捉透明度和发光概念。

## 亮点与洞察

- **最小化侵入设计**: 仅操作 CLIP 空间 + 定向注入一层，不修改任何预训练权重，最大程度保留了扩散模型的先验知识
- **多功能统一框架**: 材质迁移 + 混合 + 细粒度属性控制在同一框架下完成
- **高数据效率**: 16 个合成物体就能训练出可用的属性控制器，对实际部署非常友好
- **CLIP 空间的材质可解释性**: 揭示了 CLIP 嵌入蕴含可解编的材质属性方向，这一发现本身有独立价值

## 局限与展望

- 参数化控制有时会改变物体纹理花纹（如皮革背包的纹路随粗糙度变化）
- 对某些本身就满足属性条件的物体（如玻璃增加透明度），可能产生伪影
- 依赖 SDXL 的编解码过程，存在高频细节损失
- 仅支持 4 种属性（roughness/metallic/transparency/glow），未覆盖更多 PBR 参数

## 相关工作与启发

- **ZeST**: 零样本材质迁移框架，MARBLE 在其基础上做了定向注入改进
- **InstantStyle**: 发现 UNet 中风格归因层的方法论，MARBLE 证实材质归因和风格归因共享同一层
- **Concept Slider**: LoRA 适配器方法实现连续概念控制，但需要 DDIM 逆变换且对材质概念捕捉不佳
- **Alchemist**: 微调 SD 实现细粒度材质控制的先驱工作，但修改了 SD 权重

## 评分 ⭐⭐⭐⭐

**创新性**: ⭐⭐⭐⭐ CLIP 空间材质编辑的新范式，定向注入 + SVD 去噪的组合巧妙  
**实验充分度**: ⭐⭐⭐⭐ 定量+定性+用户研究+数据效率分析全面  
**写作质量**: ⭐⭐⭐⭐ 图示丰富直观，方法描述清晰  
**实用价值**: ⭐⭐⭐⭐⭐ 极低训练数据需求 + 即插即用设计，实际应用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Dynamic Motion Blending for Versatile Motion Editing (MotionReFit)](dynamic_motion_blending_for_versatile_motion_editing.md)
- [\[CVPR 2025\] Latent Space Imaging](latent_space_imaging.md)
- [\[CVPR 2025\] Visual Lexicon: Rich Image Features in Language Space](visual_lexicon_rich_image_features_in_language_space.md)
- [\[CVPR 2025\] LaTexBlend: Scaling Multi-concept Customized Generation with Latent Textual Blending](latexblend_scaling_multi-concept_customized_generation_with_latent_textual_blend.md)
- [\[CVPR 2025\] Probability Density Geodesics in Image Diffusion Latent Space](probability_density_geodesics_in_image_diffusion_latent_space.md)

</div>

<!-- RELATED:END -->
