---
title: >-
  [论文解读] Robustness in Both Domains: CLIP Needs a Robust Text Encoder
description: >-
  [NeurIPS 2025][图像生成][CLIP] 提出 LEAF (Levenshtein Efficient Adversarial Finetuning)，首个针对 CLIP 文本编码器的对抗微调方法，在字符级文本扰动下显著提升零样本分类、文本-图像检索和图像生成的鲁棒性，同时保持图像域性能。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "CLIP"
  - "文本编码器鲁棒性"
  - "对抗微调"
  - "字符级攻击"
  - "Levenshtein距离"
---

# Robustness in Both Domains: CLIP Needs a Robust Text Encoder

**会议**: NeurIPS 2025  
**arXiv**: [2506.03355](https://arxiv.org/abs/2506.03355)  
**代码**: 有 (github.com/LIONS-EPFL/LEAF, huggingface.co/LEAF-CLIP)  
**领域**: 图像生成  
**关键词**: CLIP, 文本编码器鲁棒性, 对抗微调, 字符级攻击, Levenshtein距离

## 一句话总结

提出 LEAF (Levenshtein Efficient Adversarial Finetuning)，首个针对 CLIP 文本编码器的对抗微调方法，在字符级文本扰动下显著提升零样本分类、文本-图像检索和图像生成的鲁棒性，同时保持图像域性能。

## 研究背景与动机

CLIP 模型广泛用于检索、LMM、文本到图像生成等下游任务，但对抗攻击可导致 CLIP embedding 显著偏移：

**图像域鲁棒性已有进展**：TeCoA 和 FARE 分别通过监督和无监督方式对抗微调图像编码器

**文本域鲁棒性空白**：文本编码器的鲁棒性完全未被探索

**双域防御必要性**：仅保护一个域不足以抵御实际攻击场景

核心动机：CLIP 需要**同时在图像域和文本域**具有对抗鲁棒性。

## 方法详解

### 整体框架

LEAF 扩展了 FARE 目标到文本域：

**TextFARE 目标**：
$$\min_{\theta} \sum_{i=1}^n \max_{S_i': d_{Lev}(S_i, S_i') \leq k \wedge S_i' \in \mathcal{C}(S_i)} \|f_{\theta^{CLIP}}(S_i) - f_{\theta}(S_i')\|_2^2$$

即：优化文本编码器参数 $\theta$，使得在 Levenshtein 距离 $\leq k$ 的扰动下，编码器输出尽可能接近原始文本的编码。

### 关键设计

**LEAF 攻击算法**（高效的训练时攻击）：
1. **位置选择**：随机选取 $\rho$ 个位置，替换为测试字符，选择损失最高的位置
2. **字符选择**：在选定位置随机尝试 $\rho$ 个字符，选择损失最高的替换

关键优势：每个句子仅需评估常数 $\rho$ 次扰动（与句子长度无关），支持 batch 并行。
- Charmer（基线攻击）：需要 $O(2|S|+1+n_{Charmer} \cdot |\Gamma|)$ 次评估
- LEAF：仅需 $2 \times B \times \rho$ 次评估（$B$为batch size）

**语义约束**：
- 采用 Chanakya et al. (2024) 的约束：扰动后不允许产生新的英语单词
- 使用NLTK词典检查
- 约束对保持图像域性能至关重要

**解耦训练**：
- 文本编码器和图像编码器独立微调
- FARE 微调图像编码器，LEAF 微调文本编码器
- 可自由组合使用

### 损失函数 / 训练策略

- 在 DataComp-small 前80K样本上训练30个epoch
- Batch size 128，AdamW优化器，学习率 $10^{-5}$
- $k=1$（单字符扰动），$\rho=50$
- 带语义约束训练

## 实验关键数据

### 主实验

零样本分类（ImageNet + AG-News）：

| 鲁棒编码器 | ImageNet | ImageNet | AG-News | AG-News |
| 图像 / 文本 | Clean Acc. | Adv. Acc. | Clean Acc. | Adv. Acc. |
|-----------|----------|---------|---------|---------|
| ✗ / ✗ (CLIP-L/14) | 76.4 | 0.0 | 74.4 | 44.7 |
| ✓ / ✗ (FARE) | 74.7 | 47.6 | 78.7 | 44.5 |
| ✗ / ✓ (LEAF) | 73.4 | 0.0 | 73.9 | 60.1 |
| ✓ / ✓ (FARE+LEAF) | 72.6 | 46.0 | 78.0 | **63.2** |

OpenCLIP-ViT-H/14 结果：

| 鲁棒编码器(图像/文本) | ImageNet Adv. | AG-News Adv. |
|-------------------|-------------|-------------|
| ✗ / ✗ | 0.0 | 37.6 |
| ✓ / ✗ | 48.4 | 37.5 |
| ✓ / ✓ | 46.3 | **53.3** |

### 消融实验

训练超参数影响（ViT-L/14，$k=1$）：

| $\rho$ | 约束 | ImageNet Clean | AG-News Adv. |
|--------|------|---------------|-------------|
| 1 (随机) | ✓ | 74.7 | 54.4 (+9.9) |
| 10 | ✓ | 74.8 | 59.9 |
| 50 | ✓ | 72.6 | 63.2 (+18.7) |
| 50 | ✗ | 65.5 | 66.3 |

训练速度对比：

| 攻击方法 | 每batch时间(s) | AG-News Adv. |
|---------|-------------|-------------|
| Charmer-20 | 118.19 | 基线 |
| Charmer-1 | 15.17 | 略低 |
| LEAF ($\rho$=20) | 1.83 | 接近 |
| LEAF ($\rho$=50) | 3.23 | 接近 |

文本到图像检索（MS-COCO, $k=2$, 平均3个目标）：

| 模型 | 鲁棒? | R@1 Clean | R@1 Adv. | R@5 Clean | R@5 Adv. |
|------|------|---------|---------|---------|---------|
| CLIP-L/14 | ✗ | 49.11 | 30.66 | 73.79 | 52.76 |
| CLIP-L/14 | ✓ | 48.71 | 40.22 | 73.71 | 65.09 |

### 关键发现

1. **LEAF加速一个数量级**：1.83s vs 118.19s per batch，性能几乎无损
2. **双域鲁棒必要性**：只有同时使用鲁棒图像和文本编码器才能在两个域都鲁棒
3. **语义约束至关重要**：无约束训练严重损害图像域性能（Clean从74.7降至65.5）
4. **鲁棒模型更可解释**：鲁棒文本编码器的embedding更容易通过优化反演回文本
5. **对大距离扰动也有效**：$k=1$训练可泛化到 $k=5$ 的扰动

## 亮点与洞察

1. **填补文献空白**：首次系统研究CLIP文本编码器的对抗鲁棒性
2. **高效且有效**：LEAF的batch并行设计使对抗训练在文本域变得实用
3. **即插即用**：鲁棒编码器可直接替换SD/SDXL中的原始编码器
4. **鲁棒性≈可解释性**：鲁棒模型的embedding反演质量更高

## 局限与展望

1. 图像和文本编码器独立微调，联合对抗攻击（同时扰动两个域）未测试
2. 仅研究字符级攻击，token级鲁棒性未涉及（因token攻击常改变语义）
3. 未训练最大的EVA-CLIP模型（计算限制）
4. RAG等其他CLIP应用场景未测试
5. 随计算预算增加，联合训练两个编码器可能效果更好

## 相关工作与启发

- **FARE** (Schlarmann et al. 2024)：CLIP图像编码器的无监督对抗微调
- **TeCoA** (Mao et al. 2023)：监督式CLIP图像对抗微调
- **Charmer** (Abad Rocamora et al. 2024)：字符级文本对抗攻击
- 启发：解耦训练+高效攻击算法是使对抗鲁棒性走向实用的关键路径

## 评分

- 新颖性：⭐⭐⭐⭐ (首次研究CLIP文本编码器鲁棒)
- 技术深度：⭐⭐⭐⭐ (高效攻击算法设计精巧)
- 实验充分性：⭐⭐⭐⭐⭐ (分类/检索/生成/反演多任务)
- 实用价值：⭐⭐⭐⭐⭐ (模型已开源，直接可用)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] T2SMark: Balancing Robustness and Diversity in Noise-as-Watermark for Diffusion Models](t2smark_balancing_robustness_and_diversity_in_noise-as-watermark_for_diffusion_m.md)
- [\[CVPR 2025\] GlyphMastero: A Glyph Encoder for High-Fidelity Scene Text Editing](../../CVPR2025/image_generation/glyphmastero_a_glyph_encoder_for_high-fidelity_scene_text_editing.md)
- [\[ICCV 2025\] Your Text Encoder Can Be An Object-Level Watermarking Controller](../../ICCV2025/image_generation/your_text_encoder_can_be_an_object-level_watermarking_controller.md)
- [\[CVPR 2025\] FineLIP: Extending CLIP's Reach via Fine-Grained Alignment with Longer Text Inputs](../../CVPR2025/image_generation/finelip_extending_clips_reach_via_fine-grained_alignment_with_longer_text_inputs.md)
- [\[NeurIPS 2025\] Towards Robust Zero-Shot Reinforcement Learning](towards_robust_zero-shot_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
