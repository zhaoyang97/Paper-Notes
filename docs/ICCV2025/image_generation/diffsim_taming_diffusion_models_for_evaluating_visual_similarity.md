---
title: >-
  [论文解读] DiffSim: Taming Diffusion Models for Evaluating Visual Similarity
description: >-
  [ICCV 2025][图像生成][扩散模型] DiffSim 首次发现预训练扩散模型（Stable Diffusion）的注意力层特征可用于测量视觉相似度，提出 Aligned Attention Score (AAS) 在 U-Net 的 self-attention / cross-attention 层中对齐两张图像特征后计算余弦相似度，在人类感知一致性、风格相似度和实例一致性等多个 benchmark 上达到 SOTA。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "扩散模型"
  - "视觉相似度"
  - "注意力对齐"
  - "感知指标"
  - "风格相似度"
---

# DiffSim: Taming Diffusion Models for Evaluating Visual Similarity

**会议**: ICCV 2025  
**arXiv**: [2412.14580](https://arxiv.org/abs/2412.14580)  
**代码**: [https://github.com/showlab/DiffSim](https://github.com/showlab/DiffSim)  
**领域**: Image Generation / Visual Similarity  
**关键词**: 扩散模型, 视觉相似度, 注意力对齐, 感知指标, 风格相似度

## 一句话总结

DiffSim 首次发现预训练扩散模型（Stable Diffusion）的注意力层特征可用于测量视觉相似度，提出 Aligned Attention Score (AAS) 在 U-Net 的 self-attention / cross-attention 层中对齐两张图像特征后计算余弦相似度，在人类感知一致性、风格相似度和实例一致性等多个 benchmark 上达到 SOTA。

## 研究背景与动机

- **定制化生成**（custom generation）的兴起使得评估生成图像与参考图像的相似性越来越重要
- **传统感知指标**（LPIPS、SSIM等）主要在像素/patch层面比较低级颜色和纹理，无法捕捉布局、姿态、语义内容等中高层相似性
- **CLIP / DINO**虽常用于语义相似度度量，但将图像高度压缩为 $1 \times 768$ 维特征向量计算余弦相似度，丢失了大量外观细节
- **DreamSim** 等方法通过在人工标注数据上训练来对齐人类偏好，但域外泛化能力有限
- **关键洞察**：
    - ReferenceNet 等方法证明 U-Net self-attention 层特征能维持外观一致性
    - Custom Diffusion 证明 cross-attention 的 to_K / to_V 矩阵对学习概念至关重要
    - IP-Adapter 证明将 IP token 注入 cross-attention 可实现一致性生成
    - → 这些层的特征本身就编码了视觉外观信息，可直接用于相似度评估

## 方法详解

### 整体框架

DiffSim 提出两种实现方式：
1. **DiffSim-S**：利用 self-attention 层进行特征对齐和相似度计算
2. **DiffSim-C**：利用 IP-Adapter Plus 在 cross-attention 层进行特征对齐

两者的核心都是 **Aligned Attention Score (AAS)**。

### 关键设计

1. **Aligned Attention Score (AAS)**：

    - 传统方法假设特征像素对齐，但实际中风格/姿态差异使这一假设失效
    - AAS 利用注意力机制实现隐式对齐：
    $\text{AAS}(L_A, L_B) = \cos(\text{attn}(Q_A, K_A, V_A), \text{attn}(Q_A, K_B, V_B))$
    - 即：用图像A的 Query 对图像B的 Key/Value 做注意力，再与图像A自注意力的输出计算余弦相似度
    - 双向对称计算：$\text{Similarity}(L_A, L_B) = \text{AAS}(L_A, L_B) + \text{AAS}(L_B, L_A)$
    - 每张图像都既作为 query 又作为 key，保证了度量的完整性

2. **DiffSim-S（Self-Attention 版本）**：

    - 将两张图像加噪至时间步 $t$，送入 U-Net 的第 $n$ 个 self-attention 层
    - 提取两个图像的潜表示 $z_{t,\text{self},n}^A$ 和 $z_{t,\text{self},n}^B$
    - 在该层的 self-attention 中计算 AAS
    - **浅层+高噪声时间步**适合评估低级/风格相似度，**深层+低噪声时间步**适合评估语义相似度
    - 通过简单调整 layer 和 timestep 即可实现不同粒度的相似度度量

3. **DiffSim-C（Cross-Attention 版本）**：

    - 使用 IP-Adapter Plus 从 CLIP 图像编码器提取 patch 级特征，注入 U-Net cross-attention
    - 交换两张图像在 IP-Adapter 和 U-Net 中的角色
    - 在 cross-attention 层计算 AAS
    - 特别适合实例级相似度评估（如 CUTE 数据集）

4. **AAS 对 CLIP / DINO 的扩展**：

    - AAS 不限于扩散模型，也可应用于 CLIP 和 DINOv2 的 self-attention 层
    - 衍生出 **CLIP AAS** 和 **DINO v2 AAS**，在部分任务上显著提升原始模型性能
    - 尤其在低级相似度（TID2013）和风格相似度（Sref）上提升明显

### 损失函数 / 训练策略

DiffSim 是 **完全无监督的**，不需要任何微调或训练。预训练的 Stable Diffusion 1.5 直接使用，仅通过选择合适的 layer 和 timestep 配置即可适配不同任务。

## 实验关键数据

### 主实验 (表格)

| 模型 | NIGHTS | Dreambench++ | CUTE | IP | TID2013 | Sref | InstantStyle |
|------|--------|--------------|------|-----|---------|------|--------------|
| LPIPS | 71.13% | 62.33% | 63.17% | 84.01% | **94.50%** | 87.85% | 93.15% |
| CLIP | 82.26% | 70.54% | 72.71% | 91.70% | 90.33% | 84.60% | 82.90% |
| DINOv2 | 85.24% | **72.25%** | **77.27%** | 85.35% | 91.50% | 87.20% | 86.10% |
| CLIP AAS | 82.18% | 67.23% | 71.78% | 88.03% | 96.33% | 94.45% | 97.80% |
| DINO v2 AAS | 86.47% | 70.01% | 77.04% | 90.38% | 95.83% | 95.90% | 96.65% |
| **DiffSim** | **86.52%** | 71.50% | 76.17% | **91.84%** | 94.17% | **97.40%** | **99.05%** |
| Ensemble | 89.43% | 72.15% | 77.78% | 94.92% | 95.00% | 91.70% | 88.30% |

> DiffSim 在人类感知一致性（NIGHTS）、实例相似度（IP）、风格相似度（Sref、InstantStyle）上均达到最佳。

### 消融实验 (表格)

| 设置 | Sref | IP | NIGHTS |
|------|------|-----|--------|
| **DiffSim (AAS)** | **97.40%** | **92.04%** | **86.82%** |
| Diffusion feature (无AAS) | 78.80% | 62.47% | 66.75% |
| CLIP AAS | 71.15% | 87.36% | 80.54% |
| CLIP features (无AAS) | 66.50% | 82.54% | 75.84% |
| DINO v2 AAS | 78.50% | 90.38% | 86.41% |
| DINO v2 feature (无AAS) | 76.90% | 87.56% | 81.00% |

> **AAS 是核心**：去掉 AAS 后，直接用扩散模型特征计算余弦相似度，Sref 从 97.4% 暴降到 78.8%。AAS 对所有模型（CLIP、DINO、Diffusion）都带来显著提升。

### 关键发现

- **时间步影响**：高 $t$（~900）适合风格评估，低 $t$（~500）适合语义/实例评估
- **层选择影响**：U-Net 中部层（Down2、Up0）关注实例级，浅层（Down0、Up1）关注风格
- **分辨率影响**：512→768 对高分辨率原始图像略有提升，但默认 512 已足够
- **架构对比**：DiffSim-S SD1.5 整体最优，DiffSim-C 在 CUTE 上略好
- **Ensemble 有效**：CLIP + DINO v2 + DiffSim 的硬投票提升了多个 benchmark

## 亮点与洞察

- **首次将扩散模型用于图像相似度评估**，开辟了新方向
- **AAS 设计优雅**：通过交换 Q/K/V 实现隐式对齐，避免了显式的特征对齐步骤，且可推广到任意基于注意力的架构
- **无需训练、无需标注**是极大的实用优势，即插即用
- **多粒度相似度**：仅通过调整 layer + timestep 即可覆盖从低级到高级的相似度需求
- **AAS 对 CLIP/DINO 的增强** 是一个额外贡献，证明 AAS 的通用性
- 新提出的 **Sref bench**（508种风格，由人类艺术家挑选）和 **IP bench**（299个IP角色）弥补了评估空白

## 局限与展望

- **背景过度关注**：DiffSim 有时过度强调背景特征，忽略小目标的关键细节（如小猫图像检索出了背景相似的狗图像）。裁剪前景可缓解
- 基于 SD1.5，迁移到更新的扩散模型（SDXL、SD3、Flux）可能进一步提升
- 推理速度较慢（需要前向扩散+U-Net推理），不如CLIP/DINO轻量
- layer 和 timestep 的最优选择需要网格搜索，不够自动化
- 对于非常抽象的语义相似度，CLIP 可能仍有优势

## 相关工作与启发

- **ReferenceNet / IP-Adapter**：证明了 self-attention/cross-attention 层编码了外观信息，是 DiffSim 的直接灵感
- **DreamSim**：集成 CLIP+DINO 特征并在人工标注上训练，是训练式方案的代表
- **CSD**：用多标签对比学习提取风格描述符
- **LPIPS**：经典的感知相似度指标，在低级相似度上仍有竞争力
- 启发：生成模型的内部表示不仅能生成图像，还能理解图像关系

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次发现扩散模型的注意力层可直接用于相似度评估，AAS 设计简洁高效
- **实验充分度**: ⭐⭐⭐⭐⭐ — 7个benchmark、多种baseline、详细消融、新数据集、应用扩展
- **写作质量**: ⭐⭐⭐⭐ — 方法分析清晰，实验充分，但部分公式较密集
- **价值**: ⭐⭐⭐⭐⭐ — 开辟新方向，代码开源，两个新benchmark，AAS可推广到其他模型

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SliderSpace: Decomposing the Visual Capabilities of Diffusion Models](sliderspace_decomposing_the_visual_capabilities_of_diffusion_models.md)
- [\[ICCV 2025\] VSC: Visual Search Compositional Text-to-Image Diffusion Model](vsc_visual_search_compositional_text-to-image_diffusion_model.md)
- [\[ICML 2025\] Taming Diffusion for Dataset Distillation with High Representativeness (D³HR)](../../ICML2025/image_generation/taming_diffusion_for_dataset_distillation_with_high_representativeness.md)
- [\[CVPR 2026\] DiP: Taming Diffusion Models in Pixel Space](../../CVPR2026/image_generation/dip_taming_diffusion_models_in_pixel_space.md)
- [\[CVPR 2025\] The Art of Deception: Color Visual Illusions and Diffusion Models](../../CVPR2025/image_generation/the_art_of_deception_color_visual_illusions_and_diffusion_models.md)

</div>

<!-- RELATED:END -->
