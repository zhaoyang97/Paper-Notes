---
title: >-
  [论文解读] Textured 3D Regenerative Morphing with 3D Diffusion Prior
description: >-
  [ICCV 2025][3D视觉][3D Morphing] 提出基于3D扩散先验的再生式3D morphing方法，通过在初始噪声、模型参数和条件特征三个层级进行插值，结合Attention Fusion、Token Reordering和Low-Frequency Enhancement三种策略，首次实现了跨类别纹理3D物体的平滑、合理变形序列生成。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3D Morphing"
  - "扩散模型"
  - "纹理3D表示"
  - "注意力融合"
  - "频域增强"
---

# Textured 3D Regenerative Morphing with 3D Diffusion Prior

**会议**: ICCV 2025  
**arXiv**: [2502.14316](https://arxiv.org/abs/2502.14316)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 3D Morphing, 扩散模型, 纹理3D表示, 注意力融合, 频域增强

## 一句话总结

提出基于3D扩散先验的再生式3D morphing方法，通过在初始噪声、模型参数和条件特征三个层级进行插值，结合Attention Fusion、Token Reordering和Low-Frequency Enhancement三种策略，首次实现了跨类别纹理3D物体的平滑、合理变形序列生成。

## 研究背景与动机

3D morphing旨在生成两个3D物体之间平滑、合理的插值序列，在影视视觉特效等创意应用中至关重要。与图像morphing相比，3D morphing更具挑战性，因为它需要整体地插值3D物体（图像morphing可视为特定视角下的特例）。

**现有方法的局限性**：

**仅支持形状morphing**：以往方法主要依赖建立点对点对应关系和确定平滑变形轨迹，局限于无纹理的拓扑对齐数据集（如FAUST人体形状、Shrec'20四足动物），无法处理纹理

**劳动密集的预处理**：对新数据进行morphing需要繁琐的配准和匹配步骤

**有限的morphing能力**：受限于物体多样性不足和小规模数据集，导致模糊和不合理的插值

作者提出两个关键问题：(a) 显式的点对点对应关系是否真正必要？(b) 能否通过通用生成先验增强纹理3D morphing的泛化能力？

**核心思路**：利用3D扩散模型的隐式对应能力和生成能力来融合源和目标信息，再生插值的纹理3D表示，即"再生式morphing"（Regenerative Morphing）。

## 方法详解

### 整体框架

方法基于Gaussian Anything作为3D扩散先验，这是一个两阶段的原生3D扩散模型：第一阶段通过几何扩散模型 $\epsilon_G$ 生成结构化点云表示，第二阶段通过纹理扩散模型 $\epsilon_T$ 生成纹理特征。整体流程包含三步：基础插值（Basic Interpolation）、平滑性改进（Attention Fusion）、合理性改进（Token Reordering + Low-Frequency Enhancement）。

### 关键设计

1. **三层级基础插值（Basic Interpolation）**：

   在源和目标之间以权重 $(1-\alpha)$ 和 $\alpha$ 在三个层级进行插值：
    - **初始噪声插值**：通过扩散反演获取源和目标的输入噪声，使用球面线性插值（SLERP）生成中间噪声 $[\mathbf{z}_T^\alpha, \mathbf{z}_G^\alpha]$，以保持高斯噪声属性
    - **模型参数插值**：分别对源和目标进行LoRA微调，线性插值两组LoRA参数，得到morphing模型 $\epsilon_G^\alpha$ 和 $\epsilon_T^\alpha$
    - **条件特征插值**：通过CLIP编码器将源和目标的文本提示编码为 $\mathbf{c}^{src}$ 和 $\mathbf{c}^{tgt}$，线性插值得到 $\mathbf{c}^\alpha$
   
   但基础插值存在两个问题：突变（非线性多步去噪导致映射变异性）和伪影（条件空间与扩散空间的错位）。

2. **注意力融合（Attention Fusion）**：

   将源、目标和插值的噪声同时输入morphing模型，获取三组(Q, K, V)，然后通过融合注意力增强平滑性：
   
    $\text{Fused-Attn}(Q^\alpha, K^\alpha, V^\alpha) = \text{Attn}(Q^\alpha, [(1-\alpha)K^{src} + \alpha K^{tgt}, K^\alpha], [(1-\alpha)V^{src} + \alpha V^{tgt}, V^\alpha])$
   
   该策略结合了自注意力和交叉注意力融合，使用微调模型的统一注意力特征来增强平滑性。但过度的Attention Fusion会导致结构坍塌和表面质量问题。

3. **Token重排序（Token Reordering）**：

   核心动机：3D物体被token化为序列 $\{h_j\}_{j=1}^M$，仅依赖注意力机制的隐式对应可能导致语义不合理的连接（如将椅子腿与甜甜圈糖霜匹配）。通过分析发现3D扩散特征确实捕捉了语义对应关系。
   
   实现方式：在DiT block之间重排序源和目标token序列，使语义相似的token对齐到相同索引位置：
   
    $\text{minimize} \sum_{j=1}^{M} \|h_j^{src} - h_{\sigma(j)}^{tgt}\|$
   
   根据 $\alpha$ 值设定不同策略：$\alpha \in [0, 0.5)$ 时基于源重排目标；$\alpha \in [0.5, 1]$ 时基于目标重排源。

4. **低频增强（Low-Frequency Enhancement）**：

   频域分析揭示：在3D生成中，低频噪声控制整体布局，高频噪声控制表面细节。过度的注意力融合会放大高频分量，干扰低频分量，从而降低3D表面生成质量。

   实现方式：通过FFT将token变换到频域，增强低频信号后通过IFFT变换回来：
   
    $F'_{\omega < \omega_0}(h) = F_{\omega < \omega_0}(h) \odot scale$
    $h' = \text{IFFT}([F'_{\omega < \omega_0}(h), F_{\omega \geq \omega_0}(h)])$
   
   其中 $scale = 5$，$\omega_0 = 0.1\pi$。

### 损失函数 / 训练策略

- LoRA微调参数：rank=16, alpha=20，目标层=['to_k', 'to_q', 'to_v', 'qkv']，500步训练
- 使用250个去噪时间步
- $\alpha$ 从Beta分布采样10个插值点
- Attention Fusion范围：几何扩散 step 1-120~180，纹理扩散 step 1-5
- Token Reordering范围：step 80-200
- Low-Frequency Enhancement范围：step 200-230

## 实验关键数据

### 主实验

| 方法 | FID↓ | STP-GPT↑ | SEP-GPT↑ | PPL↓ | PDV↓ | STP-U↑ | SEP-U↑ |
|------|------|----------|----------|------|------|--------|--------|
| DiffMorpher | 218.07 | 0.23 | 0.13 | 5.23 | 0.0535 | 0.435 | 0.300 |
| AID | 115.72 | 0.67 | 0.70 | 4.68 | 0.0118 | 0.380 | 0.505 |
| MV-Adapter | 120.93 | 0.63 | 0.57 | 7.29 | 0.0152 | 0.225 | 0.350 |
| Luma | 95.49 | 0.83 | 0.77 | 7.37 | 0.0007 | 0.415 | 0.330 |
| MorphFlow | 147.70 | 0.87 | 0.90 | 3.10 | 0.0001 | 0.555 | 0.505 |
| **本文** | **6.36** | **1.00** | **1.00** | **3.02** | **0.0001** | **0.915** | **0.950** |

本文方法在所有指标上全面领先：FID降低至6.36（对比第二名95.49），GPT评估结构和语义合理性均达完美1.0分，用户研究中STP-U和SEP-U分别达到0.915和0.950。

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| Basic Interpolation | 基线 | 三层级插值提供基本融合 |
| + Attention Fusion (少量step) | 平滑性↑ | 改善过渡但过多导致坍塌 |
| + Token Reordering | 合理性↑ | 缓解结构坍塌但过远时间步仍有质量下降 |
| + Low-Frequency Enhancement | 平滑+合理平衡 | 频域增强保持表面质量 |

### 关键发现

1. 2D扩散方法（DiffMorpher, AID）存在模式坍塌问题且缺乏3D一致性
2. 多视角扩散受限于像素级对齐，大空间距离匹配时产生插值错误
3. 视频生成模型（Luma）控制性有限，结构一致性不佳
4. 3D扩散先验的再生式方法能自然避免断裂伪影，因其融合时考虑了整个潜在空间的分布

## 亮点与洞察

- **首创纹理3D再生式morphing**：突破了以往方法局限于形状morphing的限制，无需显式对应关系
- **多层级融合策略设计精巧**：从噪声、参数、条件三个层级进行插值，覆盖了扩散模型的不同控制维度
- **频域分析指导改进**：通过低频/高频信号分析理解质量退化的原因，有针对性地提出增强策略
- **跨类别morphing能力惊艳**：能在靴子和泰迪熊、南瓜和蘑菇等差异极大的物体之间生成语义合理的过渡

## 局限与展望

- 受限于底层3D生成模型的能力，复杂纹理3D物体的morphing仍具挑战
- 未来可结合更先进的3D生成模型（如Trellis）提升保真度和多样性
- 可扩展到4D内容的morphing（如动画序列之间的过渡）
- 时间步范围等超参数需要针对不同场景进行调整

## 相关工作与启发

- DiffMorpher/AID等2D morphing方法为注意力融合策略提供了灵感，但本文指出了其在3D场景下的不足
- DIFT等工作表明扩散特征可表示语义，本文将这一发现扩展到3D扩散空间
- 频域分析的思路可借鉴到其他3D生成任务中改善表面质量

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次将3D扩散先验用于纹理3D morphing，范式创新显著
- **实验充分度**: ⭐⭐⭐⭐ 定量指标全面（FID/GPT/用户研究），消融清晰，但缺少大规模定量评估
- **写作质量**: ⭐⭐⭐⭐ 动机清晰、分析深入，频域和语义分析有说服力
- **价值**: ⭐⭐⭐⭐ 为纹理3D morphing开辟新方向，但应用场景相对专业

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Jasmine: Harnessing Diffusion Prior for Self-Supervised Depth Estimation](../../NeurIPS2025/3d_vision/jasmine_harnessing_diffusion_prior_for_self-supervised_depth_estimation.md)
- [\[ICCV 2025\] HairCUP: Hair Compositional Universal Prior for 3D Gaussian Avatars](haircup_hair_compositional_universal_prior_for_3d_gaussian_avatars.md)
- [\[ICCV 2025\] Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework](bridging_diffusion_models_and_3d_representations_a_3d_consistent_super-resolutio.md)
- [\[ICCV 2025\] Boost 3D Reconstruction using Diffusion-based Monocular Camera Calibration](boost_3d_reconstruction_using_diffusion-based_monocular_camera_calibration.md)
- [\[ICCV 2025\] MoGA: 3D Generative Avatar Prior for Monocular Gaussian Avatar Reconstruction](moga_3d_generative_avatar_prior_for_monocular_gaussian_avatar_reconstruction.md)

</div>

<!-- RELATED:END -->
