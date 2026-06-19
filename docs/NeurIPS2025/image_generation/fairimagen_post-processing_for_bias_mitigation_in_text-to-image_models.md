---
title: >-
  [论文解读] FairImagen: Post-Processing for Bias Mitigation in Text-to-Image Models
description: >-
  [NeurIPS 2025][图像生成][公平性] 提出 FairImagen 后处理去偏框架，通过在 CLIP prompt 嵌入空间应用 FairPCA 投影去除人口统计信息，结合经验噪声注入和跨人口统计联合去偏，在不重训模型的前提下显著提升文本到图像生成的公平性。 Stable Diffusion 等文本到图像模型在生…
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "公平性"
  - "偏见缓解"
  - "扩散模型"
  - "FairPCA"
  - "文本到图像"
---

# FairImagen: Post-Processing for Bias Mitigation in Text-to-Image Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.21363](https://arxiv.org/abs/2510.21363)  
**代码**: [fuzihaofzh/FairImagen](https://github.com/fuzihaofzh/FairImagen)  
**领域**: 图像生成  
**关键词**: 公平性, 偏见缓解, Stable Diffusion, FairPCA, 文本到图像

## 一句话总结

提出 FairImagen 后处理去偏框架，通过在 CLIP prompt 嵌入空间应用 FairPCA 投影去除人口统计信息，结合经验噪声注入和跨人口统计联合去偏，在不重训模型的前提下显著提升文本到图像生成的公平性。

## 背景与动机

Stable Diffusion 等文本到图像模型在生成过程中会复制甚至放大社会偏见——例如"CEO"倾向生成白人男性，"nurse"生成女性。现有去偏方法分三类：
- **Prompt 方法**：每张图都需手动改写，费力且不通用
- **微调方法**：需要访问模型内部参数，计算成本高
- **后处理方法**（SDID、TBIE等）：轻量但现有方案存在语义漂移、方向粗糙、多属性泛化差等问题

作者聚焦后处理方向，旨在提供一种简单、可扩展、模型无关的公平性解决方案。

## 核心问题

1. 如何在 prompt 嵌入空间中精确去除人口统计信息而保留语义内容？
2. 去偏后输出过于"中性化"怎么办？
3. 如何同时处理多个受保护属性（性别+种族）而避免过度修剪？

## 方法详解

### 模块 1：Prompt 嵌入提取

对 prompt $p$ 使用 CLIP 编码器提取 token 级嵌入 $E_p \in \mathbb{R}^{T \times D}$ 和池化嵌入 $\bar{E}_p \in \mathbb{R}^D$。按受保护属性分组构建嵌入矩阵 $X$ 和分组指示矩阵 $Z$。

### 模块 2：Fair Representation 变换（FairPCA）

经典 PCA 优化：

$$\arg\min_{P \in \mathbb{R}^{D \times d}: P^T P = I} \sum_{i=1}^{n} \|\mathbf{x}_i - PP^T \mathbf{x}_i\|_2^2$$

FairPCA 加入公平正则项：

$$\min_{P^T P = I} -\text{Tr}(P^T \Sigma_X P) + \lambda \|BP\|_F^2$$

其中 $B = Z^T X \in \mathbb{R}^{G \times D}$ 为分组特征矩阵，$\lambda$ 控制重建质量和公平性的权衡。约束 $P$ 在 $\mathcal{N}(B)$ 中确保投影后的表示与任何区分不同群体的方向正交。

推理时对嵌入进行投影：$\bar{E}_p' = PP^T \bar{E}_p$，$E_p' = E_p PP^T$。

### 模块 3：经验噪声注入

防止输出过度中性化。计算每个群体 $g$ 的偏差方向：

$$\nu_g = \frac{1}{|X^{(g)}|} \sum_{\bar{E}_p \in X^{(g)}} \bar{E}_p - \bar{E}$$

构建经验分布 $\mathcal{D}_g = \{\nu_g^T \bar{E}_p : \bar{E}_p \in X^{(g)}\}$，采样 $\delta \sim \mathcal{D}_g$ 施加扰动：

$$\bar{E}_p'' = \bar{E}_p' + \epsilon \cdot \delta \cdot \nu_g$$

$\epsilon$ 为可调噪声缩放参数。

### 模块 4：跨人口统计联合去偏

不同于对各属性逐一投影（导致过度修剪），构建联合属性空间的笛卡尔积。如性别 $\{M, F\}$ × 种族 $\{W, A, B\}$ = 6 个复合群体，在此联合空间上一次性应用 FairPCA。

## 实验关键数据

### 性别去偏

| 方法 | Fairness↑ | Accuracy↑ | MUSIQ↑ | Avg↑ |
|------|-----------|-----------|--------|------|
| Base（Stable Diffusion） | 0.167 | 0.785 | 0.574 | 0.509 |
| SDID | 0.507 | 0.776 | 0.553 | 0.612 |
| CDA | 0.547 | 0.772 | 0.549 | 0.623 |
| **FairImagen** | **0.560** | 0.771 | 0.541 | **0.624** |
| FairPrompt (上界) | 0.732 | 0.766 | 0.586 | 0.695 |

### 种族去偏

| 方法 | Fairness↑ | Accuracy↑ | MUSIQ↑ | Avg↑ |
|------|-----------|-----------|--------|------|
| Base | 0.193 | 0.785 | 0.574 | 0.517 |
| SDID | 0.370 | 0.770 | 0.537 | 0.559 |
| TBIE | 0.366 | 0.762 | 0.532 | 0.553 |
| **FairImagen** | **0.389** | 0.760 | 0.536 | **0.562** |
| FairPrompt (上界) | 0.444 | 0.752 | 0.566 | 0.587 |

### 性别+种族联合去偏

| 方法 | Gender Fair↑ | Race Fair↑ | Accuracy↑ | MUSIQ↑ | Avg↑ |
|------|-------------|-----------|-----------|--------|------|
| Base | 0.163 | 0.193 | 0.785 | 0.574 | 0.508 |
| TBIE | 0.400 | 0.286 | 0.776 | 0.546 | 0.574 |
| **FairImagen** | **0.537** | **0.320** | 0.753 | 0.544 | **0.611** |
| FairPrompt (上界) | 0.690 | 0.478 | 0.747 | 0.574 | 0.671 |

关键消融发现：
- 隐藏维度减小 → 公平性提升但 Accuracy/MUSIQ 下降
- 噪声参数 e-noise 增大 → 公平性提升，尤其在联合去偏中效果显著

## 亮点

- ⭐ FairPCA 的公平-语义权衡有明确数学形式，$\lambda$ 提供精确控制旋钮
- ⭐ 跨人口统计联合去偏方法（笛卡尔积构造）避免了逐属性投影的过度修剪
- ⭐ 完全无需重训模型，适配任意 off-the-shelf 扩散模型
- 经验噪声注入有效避免了过度中性化（如生成女性化的男性）
- 在历史性别确定的 prompt（如"中世纪铁匠"）上保持语义一致性，不盲目"校正"

## 局限与展望

- Accuracy 和 MUSIQ 有一定下降（Accuracy 从 0.785 降至 0.771），存在公平-保真 trade-off
- FairPCA 假设偏见是线性可分的，非线性偏见可能残留在高维空间
- 训练 FairPCA 投影矩阵需要一组带属性标注的 prompt，构建成本虽低但非零
- 评估依赖 DeepFace 分类器检测人口属性，分类器本身可能有偏差
- 仅在 Stable Diffusion 3 上验证，其他架构（DALL-E、Imagen 等）待确认

## 与相关工作的对比

| 特性 | Prompt方法 | 微调方法 | SDID | TBIE | FairImagen |
|------|-----------|---------|------|------|-----------|
| 无需训练 | ✓ | ✗ | ✓ | ✓ | ✓ |
| 黑盒兼容 | ✓ | ✗ | ✓ | ✓ | ✓ |
| 低人力 | ✗ | ✓ | ✓ | ✓ | ✓ |
| 多属性同时去偏 | ✗ | ✓ | ✗ | ✗ | ✓ |
| 保持语义保真 | ✓ | ✓ | 弱 | 弱 | 中 |

## 启发与关联

- FairPCA 方法可推广到视频生成、3D 生成等其他多模态生成任务的去偏
- 经验噪声注入的思想（沿偏差方向施加受控扰动）可用于数据增强
- 联合属性空间构造方法（笛卡尔积）可泛化到年龄、残障等更多受保护属性
- 在公平性和"历史准确性"之间的平衡是一个值得深入讨论的伦理话题

## 评分

- 新颖性: ⭐⭐⭐⭐ (FairPCA + 经验噪声 + 联合去偏的组合新颖)
- 实验充分度: ⭐⭐⭐⭐⭐ (多场景、多基线、多消融、定性分析全面)
- 写作质量: ⭐⭐⭐⭐ (方法描述清晰，逻辑流畅)
- 价值: ⭐⭐⭐⭐ (实用性强，即插即用的公平性工具)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Multi-Group Proportional Representation for Text-to-Image Models](../../CVPR2025/image_generation/multi-group_proportional_representations_for_text-to-image_models.md)
- [\[CVPR 2025\] Implicit Bias Injection Attacks against Text-to-Image Diffusion Models](../../CVPR2025/image_generation/implicit_bias_injection_attacks_against_text-to-image_diffusion_models.md)
- [\[AAAI 2026\] How Bias Binds: Measuring Hidden Associations for Bias Control in Text-to-Image Compositions](../../AAAI2026/image_generation/how_bias_binds_measuring_hidden_associations_for_bias_control_in_text-to-image_c.md)
- [\[NeurIPS 2025\] Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)
- [\[CVPR 2025\] ConceptGuard: Continual Personalized Text-to-Image Generation with Forgetting and Confusion Mitigation](../../CVPR2025/image_generation/conceptguard_continual_personalized_text-to-image_generation_with_forgetting_and.md)

</div>

<!-- RELATED:END -->
