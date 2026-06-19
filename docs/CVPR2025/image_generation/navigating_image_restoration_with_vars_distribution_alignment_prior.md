---
title: >-
  [论文解读] Navigating Image Restoration with VAR's Distribution Alignment Prior
description: >-
  [CVPR 2025][图像生成][图像恢复] 本文发现Visual AutoRegressive (VAR) 模型的next-scale预测具有天然的多尺度分布对齐能力——低尺度修复全局退化（如低光照、雾霾），高尺度修复局部退化（如噪声、雨滴），基于此构建VarFormer框架，通过Degradation-Aware Enhancement (DAE)自适应选择尺度先验、Adaptive Feature Transformation (AFT)融合先验与退化特征，在6类恢复任务上超越现有multi-task方法。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "图像恢复"
  - "VAR"
  - "生成先验"
  - "多尺度分布对齐"
  - "通用退化恢复"
  - "VarFormer"
---

# Navigating Image Restoration with VAR's Distribution Alignment Prior

**会议**: CVPR 2025  
**arXiv**: [2412.21063](https://arxiv.org/abs/2412.21063)  
**代码**: [https://github.com/siywang541/Varformer](https://github.com/siywang541/Varformer)  
**领域**: 图像生成/图像恢复  
**关键词**: 图像恢复, VAR, 生成先验, 多尺度分布对齐, 通用退化恢复, VarFormer

## 一句话总结

本文发现Visual AutoRegressive (VAR) 模型的next-scale预测具有天然的多尺度分布对齐能力——低尺度修复全局退化（如低光照、雾霾），高尺度修复局部退化（如噪声、雨滴），基于此构建VarFormer框架，通过Degradation-Aware Enhancement (DAE)自适应选择尺度先验、Adaptive Feature Transformation (AFT)融合先验与退化特征，在6类恢复任务上超越现有multi-task方法。

## 研究背景与动机

**领域现状**：图像恢复旨在从退化的低质量图像重建高质量图像。特定任务方法（如Restormer去模糊、SwinIR去噪）在单一退化上表现出色但缺乏通用性。通用方法（如AirNet、IDR、Prompt-IR）尝试单模型处理多种退化，但仅以退化图像作为特征来源，忽略了高质量图像的先验分布。基于扩散模型的生成先验方法（如DiffUIR）虽有效但推理速度慢。

**现有痛点**：(1) 纯判别式通用恢复模型缺乏干净图像的结构先验，导致纹理修复和结构重建能力有限；(2) 基于GAN/Diffusion的生成先验方法推理昂贵且训练不稳定；(3) 不同退化类型影响的频率尺度不同（雾霾影响全局色调，噪声影响局部纹理），现有方法缺乏多尺度自适应处理能力。

**核心矛盾**：不同退化类型需要不同尺度层面的修复力度——全局退化（低光照、雾霾）需要修正整体色调和对比度，局部退化（噪声、雨滴）需要精细纹理重建，一个恢复网络难以同时最优处理。

**本文目标** 如何利用VAR这种新型生成范式的内在多尺度先验来赋予通用恢复模型自适应的、多尺度的退化感知修复能力。

**切入角度**：作者发现VAR的next-scale预测过程天然地将退化图像和干净图像的表示对齐到公共空间——通过替换退化图像的不同尺度VAR特征为自回归预测特征，可以选择性地消除不同类型的退化。

**核心 idea**：利用VAR的多尺度分布对齐先验作为恢复引导——低尺度先验纠正全局退化，高尺度先验修复局部退化，通过DAE模块自适应加权不同尺度先验来处理任意退化类型。

## 方法详解

### 整体框架

VarFormer分两个训练阶段。**Stage 1**：在冻结VAR的基础上训练Adapter模块，获取增强的多尺度分布对齐嵌入$S_v$，用Feature Matching Loss拉近退化和干净图像的VAR表示。**Stage 2**：以U-Net风格的编码器-解码器为恢复骨架，在每层插入DAE模块（自适应选择VAR尺度先验的权重）和AFT模块（将加权先验融合到恢复特征中），通过Reconstruction Loss端到端训练恢复网络。

### 关键设计

1. **Degradation-Aware Enhancement (DAE)**：
    - 功能：为恢复网络的每一层自适应地选择最有效的VAR尺度先验组合
    - 核心思路：VAR编码器不同层的输出$F_{e_v}^i$天然知道每层应关注什么级别的信息。将$F_{e_v}^i$通过Swin-Transformer blocks（滤除图像内容干扰）和投影卷积得到K个尺度先验的权重$W=[w_i]_{i=1}^K$，加权求和VAR先验$\hat{S}_w^i = \mathcal{M}(\sum_{j=1}^K w_j \cdot S_v^j)$。再通过RSTB+Softmax产生区域自适应的融合权重$w_1^g, w_2^g$，最终$F_{g_{e/d}}^i = F_{e/d}^i \times w_1^g + \hat{S}_w^i \times w_2^g$
    - 设计动机：去雾需要低尺度全局先验而去噪需要高尺度局部先验，一刀切的固定融合策略无法最优——必须让网络自适应地选择"修复这层特征最需要哪个尺度的干净图像知识"

2. **Adaptive Feature Transformation (AFT)**：
    - 功能：缓解高质量先验与低质量退化特征融合时产生的结构扭曲和纹理失真
    - 核心思路：引入低维中间特征$M$（桥接特征）作为先验和退化特征的"调解者"。$Q$来自退化特征$F_{e/d}^i$，$K,V$来自增强特征$F_{g_{e/d}}^i$，$M$由两者拼接投影得到。注意力分两步：先算$A_{q,m} = \text{Softmax}(QM^T/\sqrt{d})$，再算$A_{m,k} = \text{Softmax}(MK^T/\sqrt{d})$，最终$F_{in}^{i+1} = A_{q,m} \cdot (A_{m,k} \cdot V) + F_{e/d}^i$
    - 设计动机：直接cross-attention融合质量差异大的特征会引入伪影。通过中间mediator让两个分布差异大的特征空间间接对齐，比直接query-key匹配更稳定

3. **Adapter for Domain Shift (Stage 1训练)**：
    - 功能：在冻结VAR参数的前提下弥合预训练分布与退化图像分布的差距
    - 核心思路：在VAR encoder之后插入包含self-attention blocks的Adapter模块，用Feature Matching Loss训练——结合交叉熵损失（对齐多尺度预测token）和L2损失（拉近Adapter输出与GT的VQVAE量化特征$F_{e_{gt}}^q$）
    - 设计动机：直接fine-tune VAR会破坏预训练知识，而VAR的编码器看到的退化图像与预训练时的干净图像有分布差异——轻量Adapter是两者之间的最佳平衡

### 损失函数

- **Stage 1**: Feature Matching Loss $\mathcal{L}_{fema} = \sum_{i=1}^K -s_i \log(\hat{s}_i) + \|F_a - sg(F_{e_{gt}}^q)\|_2^2$
- **Stage 2**: Reconstruction Loss $\mathcal{L}_{rec} = -\text{PSNR}(I_{gt}, I_{rec}) + \|\psi(I_{gt}) - \psi(I_{rec})\|_2^2$，其中$\psi$为预训练VGG19

## 实验关键数据

### 主实验表

**四类恢复任务对比（PSNR↑ / SSIM↑）**：

| 方法 | 去雨 | 去模糊 | 低光增强 | 去雾 |
|------|------|--------|----------|------|
| Restormer (task-specific) | 33.96/0.935 | 32.92/0.961 | 20.41/0.806 | 30.87/0.969 |
| AirNet (universal) | 25.44/0.743 | 27.14/0.832 | 18.49/0.767 | 25.48/0.944 |
| DiffUIR (universal) | 31.14/0.907 | 29.88/0.874 | 25.02/0.901 | 32.74/0.944 |
| **VarFormer** | **31.33/0.913** | **30.99/0.956** | **25.13/0.917** | **32.96/0.956** |

VarFormer在所有通用方法中取得最优，在去模糊和低光增强上尤其突出。

### 消融实验表

**关键组件消融（去雨任务PSNR）**：

| 配置 | PSNR↑ |
|------|-------|
| 基线（无VAR先验） | ~29.5 |
| + VAR先验（固定权重） | ~30.2 |
| + DAE（自适应权重） | ~30.8 |
| + AFT（特征变换） | ~31.1 |
| + Adaptive mix-up skip | **31.33** |

### 关键发现

- t-SNE可视化证实VAR的next-scale预测确实将各种退化图像和干净图像映射到更接近的分布——这不是人为设计而是VAR的固有属性
- 替换低尺度VAR特征可消除全局退化（雾霾、低光照），替换高尺度特征可消除局部退化（噪声、雨滴）——完美对应多分辨率分析的直觉
- VarFormer在**未见任务**（如真实去噪）上也展现出良好泛化，说明VAR先验提供了超越训练任务的通用"干净图像知识"
- 相比DiffUIR，VarFormer减少了训练计算成本——Stage 1仅训练轻量cross-attention和Adapter

## 亮点与洞察

1. **发现VAR的分布对齐能力**是本文最核心的贡献——这不是显式设计的功能，而是next-scale prediction范式的固有属性，非常有insight
2. 将生成先验从"直接生成干净图像"转化为"提取多尺度对齐嵌入来引导恢复"，**避免了逐像素生成的累积误差和速度瓶颈**
3. DAE的"退化类型→尺度权重"机制是一个简洁但有效的设计，让单模型自动适配不同退化而无需显式退化分类
4. 首次将VAR用于图像恢复任务，**开辟了VAR先验在低层视觉中的应用方向**

## 局限性

- VAR本身的VQVAE量化误差会限制恢复精度的上限
- 需要预训练VAR的forward pass来提取先验，增加了推理时的内存和计算开销
- 仅在256×256分辨率上验证，高分辨率（如4K修复）场景未探索
- Stage 1的重建预训练和Stage 2的恢复训练需要分步进行，流程较复杂
- 对严重退化（如极端过曝、大面积缺失）的重建能力有待验证

## 相关工作与启发

- **VAR** [Tian et al.]: 提出next-scale prediction范式替代next-token prediction，本文发现并利用了其内在的分布对齐性质
- **Restormer** [Zamir et al.]: Transformer-based的SOTA特定任务恢复方法，VarFormer在通用设置下接近其专用性能
- **DiffUIR** [Zheng et al.]: 用扩散先验做通用恢复，VarFormer比它更快且效果更好
- **IDR** [Zhang et al.]: 成分导向的通用恢复范式，本文与之互补——IDR分析任务关联，VarFormer提供生成先验
- **启发**：自回归生成模型的中间表示往往包含丰富的结构先验，值得在更多判别式任务中探索其"免费"的知识

## 评分

⭐⭐⭐⭐ — 对VAR分布对齐性质的发现非常有洞察力，VarFormer框架设计合理且实验扎实。首次将VAR引入图像恢复是一个有意义的开创。DAE和AFT模块设计简洁有效。代码开源。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Reversing Flow for Image Restoration](reversing_flow_for_image_restoration.md)
- [\[CVPR 2025\] InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](interedit_navigating_text-guided_multi-human_3d_motion_editing.md)
- [\[ICCV 2025\] DIIP: Diffusion Image Prior](../../ICCV2025/image_generation/diffusion_image_prior.md)
- [\[ICLR 2026\] Infinity and Beyond: Compositional Alignment in VAR and Diffusion T2I Models](../../ICLR2026/image_generation/infinity_and_beyond_compositional_alignment_in_var_and_diffusion_t2i_models.md)
- [\[CVPR 2025\] Dual Prompting Image Restoration with Diffusion Transformers (DPIR)](dual_prompting_image_restoration_with_diffusion_transformers.md)

</div>

<!-- RELATED:END -->
