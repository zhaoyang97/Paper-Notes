---
title: >-
  [论文解读] Exploring In-Image Machine Translation with Real-World Background
description: >-
  [ACL 2025][多语言/翻译][图像内机器翻译] 提出 DebackX 模型，通过将图像分离为背景和文字图像分别处理，首次解决了真实复杂背景下的图像内机器翻译 (IIMT) 任务，在翻译质量和视觉效果上均优于现有方法。 领域现状： 图像内机器翻译 (In-Image Machine Translation…
tags:
  - "ACL 2025"
  - "多语言/翻译"
  - "图像内机器翻译"
  - "背景分离"
  - "向量量化"
  - "视觉文本生成"
  - "多模态翻译"
---

# Exploring In-Image Machine Translation with Real-World Background

**会议**: ACL 2025  
**arXiv**: [2505.15282](https://arxiv.org/abs/2505.15282)  
**代码**: [GitHub](https://github.com/BITHLP/DebackX)  
**领域**: 多语言翻译  
**关键词**: 图像内机器翻译, 背景分离, 向量量化, 视觉文本生成, 多模态翻译  

## 一句话总结

提出 DebackX 模型，通过将图像分离为背景和文字图像分别处理，首次解决了真实复杂背景下的图像内机器翻译 (IIMT) 任务，在翻译质量和视觉效果上均优于现有方法。

## 研究背景与动机

**领域现状**: 图像内机器翻译 (In-Image Machine Translation, IIMT) 旨在将图像中的文字从一种语言翻译为另一种语言，输入输出均为图像模态。应用场景包括视频字幕翻译、图片翻译工具等。

**现有痛点**: 以往 IIMT 研究仅在极度简化的场景下进行：Tian et al. (2023) 使用黑字白底单行文本，Lan et al. (2024) 使用单色背景。这些设定与真实世界的复杂背景文字（如视频字幕叠加在自然图像上）差距巨大，无法直接应用于实际场景。

**核心矛盾**: 将 IIMT 推向真实场景面临两大挑战：一是复杂背景干扰翻译质量——传统 OCR-NMT-Render 级联框架存在错误传播问题；二是文本区域擦除破坏背景完整性，且无法保持字体风格一致性，导致视觉效果差。

**本文目标**: 在真实世界背景条件下，实现高翻译质量与高视觉效果兼顾的 IIMT。

**切入角度**: 将图像"解构"为背景和文字图像，对文字图像直接进行图像到图像的翻译（避免 OCR 级联错误），再将翻译后的文字图像与背景融合。

**核心 idea**: 背景-文字分离 + 文字图像直接翻译 + 融合重建，避免了 OCR 错误传播和背景破坏。

## 方法详解

### 整体框架

DebackX 包含三个组件：
1. **Text-Image Background Separation**: 将源图像分解为背景图像和源文字图像
2. **Image Translation**: 将源文字图像变换为目标文字图像
3. **Text-Image Background Fusion**: 将背景与目标文字图像融合生成最终输出

### 关键设计

#### 1. 文字-背景分离模块

使用两组 ViT Encoder-Decoder 分别提取背景和文字：

$$\text{Background} = G_{\text{back}}(E_{\text{deback}}(x))$$
$$\text{Source Text-Image} = G_{\text{text}}(E_{\text{detext}}(x))$$

所有 ViT 配置：patch_size=16, d_model=512, layers=8, heads=8, d_ff=2048。

#### 2. 图像翻译模块

分两阶段训练：

**阶段 1 — 向量量化 (VQ)**：使用 ViT Encoder 将文字图像编码，通过 codebook（大小 8192，维度 32）量化为离散 token 序列：

$$z_i = q(E(x_i)) = \arg\min_{e_k \in q} \|E(x_i) - e_k\|_2$$

**阶段 2 — 翻译**：将图像翻译转化为 code 序列到 code 序列的变换。关键设计包括：
- **Code Encoder**: 编码源 code 序列
- **Pivot Decoder**: 利用 TIT (Text-Image Translation) 辅助任务注入语义信息，输出 $H_{\text{pivot}}^D$ 同时用于辅助文本翻译和后续 code 解码
- **Linear Adapter**: 将 Pivot Decoder 输出适配到 Code Decoder 输入
- **Code Decoder**: 自回归生成目标 code 序列

推理时先解码辅助文本，获得完整 Pivot 表示后再解码目标 code 序列。

#### 3. 文字-背景融合模块

$$\text{Target Image} = G_{\text{fuse}}(E_{\text{back}}(i_b) + E_{\text{text}}(i_t))$$

两个 ViT Encoder 分别编码背景和目标文字图像，特征相加后通过 ViT Decoder 生成最终图像。

### 损失函数/训练策略

- **分离模块**: $\mathcal{L}_{\text{sep}} = \mathcal{L}_{\text{img}}(i_b, \hat{i_b}) + \mathcal{L}_{\text{img}}(i_t, \hat{i_t})$，其中 $\mathcal{L}_{\text{img}} = \|y - \hat{y}\|^2 + 0.1 \cdot \mathcal{L}_{\text{Perceptual}}$
- **VQ 阶段**: $\mathcal{L}_{\text{VQ}} = \|y - \hat{y}\|^2 + 0.1 \cdot \mathcal{L}_{\text{Perceptual}} + \|\text{sg}[z_q] - E(x)\|_2^2$（含 commitment loss + EMA 更新）
- **翻译阶段**: $\mathcal{L}_{\text{trans}} = \mathcal{L}_{\text{code}} + \mathcal{L}_{\text{TIT}}$（均使用 label-smoothing=0.1 的交叉熵）
- **融合模块**: 同样使用 $\mathcal{L}_{\text{img}}$
- **训练步数**: 分离 25K, VQ 50K, 预训练 100K + 微调 50K, 融合 15K

## 实验关键数据

### 数据集

IIMT30k 数据集基于 Multi30k 构建，包含三种字体（TNR, Arial, Calibri），各约 25K 训练 / 864 验证 / 2740 测试对。图像尺寸 $48 \times 512$。

### 主实验

| 系统 | BLEU (De→En Test) | COMET (De→En Test) | FID (De→En Test) ↓ | FID (En→De Test) ↓ |
|---|---|---|---|---|
| VQGAN | 0.6 | 24.4 | 21.3 | 20.7 |
| TIT-Render | 12.1 | 49.6 | 133.2 | 119.0 |
| McTIT-Render | 11.7 | 47.3 | 137.5 | 117.5 |
| Translatotron-V | 1.6 | 24.8 | 10.1 | 17.5 |
| **DebackX** | **12.8** | **50.0** | **9.0** | **8.7** |

DebackX 在翻译质量上略优于 TIT-Render 系列（BLEU 12.8 vs 12.1），在视觉效果上大幅领先（FID 9.0 vs 133.2）。

### 消融实验

| 设置 | Pivot | Deback | BLEU (De→En Test) |
|---|---|---|---|
| #1 完整模型 | ✓ | ✓ | 5.9 |
| #2 去 Pivot | ✗ | ✓ | 1.4 |
| #3 去 Deback | ✓ | ✗ | 1.1 |
| #4 全去 | ✗ | ✗ | 0.6 |

### 关键发现

- **预训练效果显著**: 仅用 IIMT30k 训练 BLEU=5.9，加 IWSLT 预训练提升至 12.8，加 WMT14 预训练达 17.5
- **多字体适应**: 混合三种字体训练后字体一致性达 96.5%–97.3%
- **OCR 误差影响**: 即使 ground truth 图像，OCR 识别也仅达 BLEU 67.5–81.0，说明真实评估上限受限
- **GPT-4o 对比**: GPT-4o 能生成有意义翻译但无法正确处理布局和字体

## 亮点与洞察

- **解耦设计精妙**: 将复杂的端到端 IIMT 任务分解为三个可独立训练的子模块，每个模块目标清晰
- **预训练优势独特**: 文字图像容易用平行语料批量构造，使预训练数据可大规模扩展（从 100K 到 1M），这是其他 IIMT 方法不具备的
- **Pivot Decoder 双重用途**: 既为辅助 TIT 任务提供语义监督，又为 Code Decoder 提供语义引导，设计巧妙

## 局限与展望

- 所有子模块均使用最基础的 ViT，未探索更先进的视觉架构（如 DiT, SwinTransformer）
- 多阶段训练成本高，难以端到端优化
- 数据集虽模拟真实字幕但仍是合成的，与真正的自然场景文字（任意角度、任意位置）仍有差距
- 仅测试了德英翻译方向，未验证到其他语言对的泛化能力

## 相关工作与启发

- Translatotron-V (Lan et al., 2024) 在简单背景上有效但无法处理复杂场景 → 启发: 复杂场景需要显式的背景-文字解耦
- OCR-VQGAN (Rodríguez et al., 2023) 使用 OCR 预训练特征计算文本感知损失 → 启发: 分离后的文字图像可以降低 OCR 错误率
- AnyText/GlyphDraw 等文本生成工作 → 将来可结合扩散模型提升生成质量

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次系统性地处理复杂背景 IIMT，背景分离+融合的框架设计新颖
- **实验充分度**: ⭐⭐⭐⭐ — 包含多模型对比、消融、预训练研究、多字体实验、GPT-4o 对比
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，图文并茂，实验设计合理
- **价值**: ⭐⭐⭐⭐ — 填补了真实场景 IIMT 研究的空白，有明确的应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Exploring In-context Example Generation for Machine Translation](exploring_in-context_example_generation_for_machine_translation.md)
- [\[ACL 2025\] LEMONADE: A Large Multilingual Expert-Annotated Abstractive Event Dataset for the Real World](lemonade_a_large_multilingual_expert-annotated_abstractive_event_dataset_for_the.md)
- [\[ACL 2025\] Cross-Lingual Representation Alignment Through Contrastive Image-Caption Tuning](cross-lingual_representation_alignment_through_contrastive_image-caption_tuning.md)
- [\[ACL 2025\] Machine Translation Models are Zero-Shot Detectors of Translation Direction](machine_translation_models_are_zero-shot_detectors_of_translation_direction.md)
- [\[NeurIPS 2025\] Exploring the Translation Mechanism of Large Language Models](../../NeurIPS2025/multilingual_mt/exploring_the_translation_mechanism_of_large_language_models.md)

</div>

<!-- RELATED:END -->
