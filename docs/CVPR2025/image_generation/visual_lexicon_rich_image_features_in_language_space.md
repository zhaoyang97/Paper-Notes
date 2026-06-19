---
title: >-
  [论文解读] Visual Lexicon: Rich Image Features in Language Space
description: >-
  [CVPR 2025][图像生成][视觉词典] ViLex 提出了一种将图像编码到文本词汇空间的视觉编码器，通过冻结的文生图扩散模型进行自监督训练，使得生成的图像 token 同时兼具高层语义和细粒度视觉细节，在图像重建和视觉理解任务上均超越了传统方法。 领域现状：计算机视觉中图像表征长期存在两条路线——以 CLIP/DIN…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "视觉词典"
  - "图像表征"
  - "扩散模型"
  - "视觉语言模型"
---

# Visual Lexicon: Rich Image Features in Language Space

**会议**: CVPR 2025  
**arXiv**: [2412.06774](https://arxiv.org/abs/2412.06774)  
**代码**: 无  
**领域**: 图像生成 / 视觉表征学习  
**关键词**: 视觉词典, 图像表征, 扩散模型, 图像生成, 视觉语言模型

## 一句话总结

ViLex 提出了一种将图像编码到文本词汇空间的视觉编码器，通过冻结的文生图扩散模型进行自监督训练，使得生成的图像 token 同时兼具高层语义和细粒度视觉细节，在图像重建和视觉理解任务上均超越了传统方法。

## 研究背景与动机

**领域现状**：计算机视觉中图像表征长期存在两条路线——以 CLIP/DINO 为代表的理解导向表征，捕获高层语义但丢失像素级细节；以 VAE/MAE 为代表的重建导向表征，保留视觉细节但语义信息薄弱。

**现有痛点**：这两类表征在各自擅长的任务之外表现不佳。CLIP 特征无法用于高保真图像重建，VAE 特征在下游理解任务（如线性评估）中表现欠佳。DeDiffusion 等方法尝试将图像逆向为离散文本 token，但文本 token 的表达能力有限，重建质量受限于自然语言能描述的范围。

**核心矛盾**：语义表征与重建表征之间存在根本性 trade-off——一个表征能否同时在图像生成和视觉理解两个方向上都做到最好？

**本文目标**：设计一种统一的图像表征，既能被用作文生图模型的"文本提示词"实现高保真图像重建，又能作为视觉编码器在VLM中提升理解能力。

**切入角度**：作者观察到扩散模型在去噪过程中天然编码了丰富的语义和视觉细节信息。与其直接用扩散模型提取特征，不如将其作为自编码器框架中的解码器，训练一个轻量级编码器来"蒸馏"扩散模型中的丰富视觉知识。

**核心 idea**：将图像映射到 T2I 扩散模型的文本词汇嵌入空间中，利用冻结的扩散模型作为解码器进行图像重建训练，从而让视觉表征继承扩散模型的语义-视觉丰富性。

## 方法详解

### 整体框架

ViLex 采用自编码器架构：编码器是一个 ViT 视觉编码器 + 注意力池化层，将输入图像转换为一组 ViLex token（位于文本词汇嵌入空间中）；解码器是一个冻结的预训练 T2I 扩散模型（Imagen）。训练时只更新编码器参数，通过扩散模型的图像重建损失来反向传播梯度。训练完成后，ViLex token 可以直接作为"文本提示词"送入冻结的文本编码器和扩散模型，无需真实文本即可重建出语义和视觉上高度相似的图像。

### 关键设计

1. **图像到文本空间的投影（Image-to-Text Projection）**:

    - 功能：将 ViT 输出的 patch-level 视觉特征转换为与 T2I 模型文本编码器兼容的词汇嵌入
    - 核心思路：使用多头交叉注意力层，包含 $n$ 个可学习 query，以 ViT 的 $k$ 个 patch token 作为 key 和 value，将视觉信息池化为 $n$ 个 ViLex 嵌入向量。这些嵌入被训练为隐式对齐到 BPE 词汇查找矩阵 $\mathcal{V}$ 的潜在空间中，确保与 T2I 扩散模型兼容
    - 设计动机：文本词汇空间具有组合性，ViLex token 可以像真实文本 token 一样独立使用或与自然语言拼接，实现多模态图像生成

2. **TailDrop 动态 token 压缩策略**:

    - 功能：在训练中随机丢弃末尾 $k$ 个 ViLex token，使早期 token 携带更丰富的语义信息
    - 核心思路：类似 SoundStream 中的可变码率策略，由于训练中前面的 token 更频繁地被单独用于图像生成，模型被迫让早期 token 编码尽可能多的语义信息。推理时可以动态调整 token 数量以平衡压缩率和细节
    - 设计动机：不同图像含有不同信息量，TailDrop 提供了灵活的 token 预算机制，实现从 1 个 token 的粗略语义到 75 个 token 的精细重建的连续过渡

3. **Text-Free Guidance (TFG) 无文本引导**:

    - 功能：在多模态图像生成中平衡 ViLex 视觉 token 和文本提示词的影响力
    - 核心思路：类似 Classifier-Free Guidance，TFG 组合了以视觉+文本条件的噪声预测和仅以 ViLex 条件的噪声预测：$\epsilon_{\text{tfg}} = \epsilon_\theta(x_t, v) + w_{\text{tfg}} \cdot (\epsilon_\theta(x_t, [v,c]) - \epsilon_\theta(x_t, v))$，通过引导尺度 $w_{\text{tfg}}$ 控制文本对生成结果的影响程度
    - 设计动机：TFG 使得 ViLex 能在不微调 T2I 模型、不修改模型架构的情况下，实现零样本无监督 DreamBooth 风格的多模态图像生成

### 损失函数 / 训练策略

训练损失为标准扩散模型去噪目标：$\mathcal{L}_{\text{denoise}} = \mathbb{E}_{x_0, \epsilon, t}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$。训练数据来自 WebLI 数据集，可仅使用图像或图像-文本对联合训练。使用 Adafactor 优化器，batch size 2048，训练 300K 步（约 2.5 天 / 64 TPUv5）。ViT 初始化自预训练 SigLIP，注意力池化层随机初始化，两者使用不同学习率（ViT: $1\times10^{-5}$，池化层: $3\times10^{-4}$）。

## 实验关键数据

### 主实验

| 方法 | Token 数 | FID ↓ | IS ↑ |
|------|---------|-------|------|
| Imagen (text→image) | - | 6.52 | 14.06 |
| DeDiffusion | 75 | 3.89 | 14.68 |
| ViLex | 1 | 3.65 | 15.33 |
| ViLex | 16 | 2.91 | 15.42 |
| ViLex | 75 | 2.07 | 15.88 |

人类评估中，ViLex 在布局/语义/风格一致性上对 DeDiffusion 的胜率分别为 98%/95%/98%，对 DALL·E 3 的胜率为 91%/76%/90%。

### 消融实验

| Backbone | FID ↓ | COCOcap | TextCaps | VQAv2-Val | SciQA | RC-val |
|----------|-------|---------|----------|-----------|-------|--------|
| Original SigLIP | 2.54 | 139.7 | 122.1 | 81.4 | 85.9 | 66.2 |
| ViLex SigLIP | 2.38 | 141.5 | 124.0 | 81.6 | 87.9 | 67.6 |
| ViLex (全模型含池化层) | **2.07** | **142.8** | **137.7** | - | - | - |

### 关键发现

- 即使只用 **1 个连续 token**，ViLex 的 FID (3.65) 已经超越了 DeDiffusion 使用 75 个离散 token (3.89) 的结果，说明连续嵌入的表达力远超离散文本 token
- ViLex 编码器替换原始 SigLIP 后，在 15 个 VLM 基准上一致性提升，包括图像/视频描述、VQA、指代分割等，证明重建与理解可以协同提升
- Token 数量从 1 增加到 75 时，重建效果从捕获高级语义（类别、数量、姿态）逐步过渡到精细视觉细节（颜色、纹理、物体形状），展现出优雅的信息分层特性

## 亮点与洞察

- **将扩散模型作为自编码器解码器**是最巧妙的设计：不直接从扩散模型提特征（如 ODISE、l-DAE），而是让扩散模型"教"编码器学习丰富表征，使编码器轻量化且可迁移到理解任务
- **连续 token 超越离散 token**的发现令人印象深刻：1 个 ViLex token 就打败了 75 个 DeDiffusion 文本 token，定量证明了自然语言的表达瓶颈
- **TailDrop 策略**可以直接迁移到任何需要可变长度 token 的场景，如视频 token 压缩或多模态 LLM 中的视觉 token 预算控制

## 局限与展望

- 当前仅在 64×64 分辨率下验证了图像重建，高分辨率下的效果和效率有待验证
- ViLex 依赖特定的 T2I 模型（Imagen），对其他扩散模型（如 SDXL、Flux）的泛化性未知
- 文本+视觉的零样本 DreamBooth 虽然免去了微调，但在身份保持精度上可能不如专门优化的 LoRA 方案
- 注意力池化层的 token 数上限受 CLIP 文本编码器 77 上下文长度限制，扩展到更长序列需要更换文本编码器

## 相关工作与启发

- **vs DeDiffusion**: DeDiffusion 将图像转为离散文本 token 再生成，受限于自然语言表达力；ViLex 绕过离散化，直接在连续嵌入空间中编码视觉信息，质量大幅领先
- **vs CLIP/SigLIP**: 这些方法只优化理解目标；ViLex 在 SigLIP 基础上额外引入扩散重建目标，同时提升了理解和生成能力
- **vs Textual Inversion/DreamBooth**: 这些方法需要对每个实例进行测试时微调；ViLex 是通用编码器，单次前向推理即可获得身份嵌入

## 评分

- 新颖性: ⭐⭐⭐⭐ 将扩散模型重新定位为自编码器解码器来学习视觉词典，切入角度新颖
- 实验充分度: ⭐⭐⭐⭐ 生成和理解两个方向均有定量+人类评估，但缺少高分辨率验证
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，图表精美，故事线完整
- 价值: ⭐⭐⭐⭐ 统一生成与理解的表征方向具有重要意义，TailDrop 等技巧可广泛复用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Do Visual Imaginations Improve Vision-and-Language Navigation Agents?](do_visual_imaginations_improve_vision-and-language_navigation_agents.md)
- [\[CVPR 2025\] CTRL-O: Language-Controllable Object-Centric Visual Representation Learning](ctrl-o_language-controllable_object-centric_visual_representation_learning.md)
- [\[CVPR 2025\] Probability Density Geodesics in Image Diffusion Latent Space](probability_density_geodesics_in_image_diffusion_latent_space.md)
- [\[CVPR 2025\] CleanDIFT: Diffusion Features without Noise](cleandift_diffusion_features_without_noise.md)
- [\[CVPR 2025\] Language-Guided Image Tokenization for Generation](language-guided_image_tokenization_for_generation.md)

</div>

<!-- RELATED:END -->
