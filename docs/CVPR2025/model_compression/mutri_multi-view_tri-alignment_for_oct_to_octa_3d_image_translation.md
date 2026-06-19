---
title: >-
  [论文解读] MuTri: Multi-view Tri-alignment for OCT to OCTA 3D Image Translation
description: >-
  [CVPR 2025][模型压缩][OCT] 本文提出MuTri，首次将向量量化（VQ）引入OCT到OCTA的3D体积翻译任务，通过两阶段训练——先预训练OCT和OCTA重建VQVAE提供多视图先验，再用对比语义对齐（3D OCT/OCTA视图）和血管结构对齐（2D OCTA投影图视图）三视图指导翻译VQVAE的码本学习，在三个数据集上全面超越SOTA。
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "OCT"
  - "OCTA"
  - "向量量化"
  - "对比学习"
  - "3D图像翻译"
  - "视网膜血管"
---

# MuTri: Multi-view Tri-alignment for OCT to OCTA 3D Image Translation

**会议**: CVPR 2025  
**arXiv**: [2504.01428](https://arxiv.org/abs/2504.01428)  
**代码**: [https://github.com/xmed-lab/MuTri](https://github.com/xmed-lab/MuTri)  
**领域**: 模型压缩  
**关键词**: OCT, OCTA, 向量量化, 对比学习, 3D图像翻译, 视网膜血管

## 一句话总结

本文提出MuTri，首次将向量量化（VQ）引入OCT到OCTA的3D体积翻译任务，通过两阶段训练——先预训练OCT和OCTA重建VQVAE提供多视图先验，再用对比语义对齐（3D OCT/OCTA视图）和血管结构对齐（2D OCTA投影图视图）三视图指导翻译VQVAE的码本学习，在三个数据集上全面超越SOTA。

## 研究背景与动机

**领域现状**：光学相干断层扫描血管成像（OCTA）能提供视网膜微血管的3D成像，对糖尿病视网膜病变、青光眼等疾病诊断至关重要。但OCTA需要专用传感器和昂贵设备，远不如OCT普及。

**现有痛点**：(1) 早期方法（AdjacentGAN、MultiGAN）仅处理2D B-scan或投影图，丢失3D深度信息；(2) 最新3D方法TransPro在连续无限空间中直接学OCT→OCTA映射，映射不确定性大；(3) TransPro依赖预训练的血管分割模型，而高质量血管分割标注获取成本极高；(4) 朴素VQVAE因OCT-OCTA巨大域差导致码本利用率低、翻译质量差。

**核心矛盾**：OCT（结构信息）和OCTA（血流信息）域差巨大——直接用VQVAE学映射时，编码器输出的unquantized特征难以有效量化到OCTA码本，码本坍塌导致大量码字闲置。

**本文目标** (1) 在离散有限空间（而非连续无限空间）中学习OCT→OCTA映射以降低不确定性；(2) 解决VQVAE在跨域翻译中的码本利用率低问题；(3) 不依赖额外的血管分割标注。

**切入角度**：利用预训练的OCT和OCTA重建模型提供高质量的领域特征作为多视图指导信号，通过对比学习最大化翻译模型与预训练模型之间的互信息，驱动码本探索更多码字。

**核心 idea**：预训练OCT和OCTA重建VQVAE提供三视图先验（3D OCT + 3D OCTA + 2D OCTA投影图），用对比语义对齐和血管结构对齐分别从语义和结构层面指导翻译VQVAE的码本学习。

## 方法详解

### 整体框架

**Stage 1**：分别训练OCT和OCTA的VQVAE重建模型。OCT VQVAE将OCT体积编码→量化→解码重建OCT；OCTA VQVAE同理。两者分别拥有码本 $\mathcal{Z}_{oct}$ 和 $\mathcal{Z}_{octa}$。

**Stage 2**：训练翻译VQVAE，输入OCT体积，输出OCTA体积。三个对齐损失利用Stage 1预训练模型的特征作为指导：(a) 对比语义对齐从3D OCT视图对齐unquantized特征；(b) 对比语义对齐从3D OCTA视图对齐quantized特征；(c) 血管结构对齐从2D OCTA投影图视图对齐血管结构。

### 关键设计

1. **对比语义对齐（Contrastive-inspired Semantic Alignment, CSA）**：
    - 功能：最大化翻译模型与预训练OCT/OCTA模型间的互信息，驱动码本探索
    - 核心操作：将翻译模型和预训练模型的特征图切成非重叠patch，经投影后做对比学习——同位置patch为正样本，不同位置为负样本
    - OCT视图对比：$\mathcal{L}_{OCT}$ 对齐翻译模型的unquantized特征与预训练OCT模型的unquantized特征
    - OCTA视图对比：$\mathcal{L}_{OCTA}$ 对齐翻译模型的quantized特征与预训练OCTA模型的quantized特征
    - 理论保证：最小化 $\mathcal{L}_{OCT}$ 等价于最大化互信息的下界 $I(\mathbf{P}, \mathbf{Q}) \geq \log(\frac{W}{S} \cdot \frac{H}{S} - 1) - \mathbb{E}\mathcal{L}_{OCT}$

2. **血管结构对齐（Vessel Structure Alignment, VSA）**：
    - 功能：利用2D OCTA投影图的血管结构先验，补充3D对齐缺失的细节
    - 核心操作：将预训练OCTA模型和翻译模型各自的2D OCTA投影图切成patch，计算patch间余弦相似度矩阵 $\mathcal{C}^{octa}$ 和 $\mathcal{C}^{oct2octa}$，最小化两者差异：$\mathcal{L}_{proj} = \sum \|\mathcal{C}^{octa} - \mathcal{C}^{oct2octa}\|$
    - 设计动机：真实OCTA投影图存在扫描不稳定导致的血管不连续伪影，而预训练OCTA模型的重建投影图更平滑连续。用后者作为指导避免翻译模型过拟合到这些伪影
    - 相比TransPro的血管分割辅助任务，VSA不需要任何额外标注

3. **两阶段VQVAE训练策略**：
    - 功能：先学习各域的离散表示空间，再利用它们指导跨域翻译
    - Stage 1损失：标准VQVAE损失 = 重建损失 + 码本损失 + commitment损失
    - 码本利用率提升：朴素VQVAE的利用率仅~30%，MuTri的CSA将利用率提升到~80%+

### 损失函数

$$\mathcal{L}_{stage2} = \mathcal{L}_{VQVAE}^{oct2octa}(\mathcal{Z}, E_{oct2octa}, D_{oct2octa}) + \lambda(\mathcal{L}_{OCT} + \mathcal{L}_{OCTA} + \mathcal{L}_{proj})$$

$\lambda=0.5$，$\tau=0.1$（对比温度），均通过敏感性分析确定，模型对两者均不敏感。

## 实验关键数据

### 主实验：三个数据集（Tables 1 & 2）

| 方法 | OCTA-3M PSNR(dB)↑ | OCTA-6M PSNR(dB)↑ | OCTA2024 PSNR(dB)↑ |
|------|-------------------|-------------------|-------------------|
| Pix2Pix3D | 31.58 | 30.66 | 41.87 |
| VQ-I2I | 31.72 | 29.54 | 41.25 |
| Palette (Diffusion) | 32.42 | 30.02 | 41.40 |
| TransPro | 32.56 | 30.53 | 42.69 |
| **MuTri** | **34.10** | **33.08** | **43.38** |

### 消融实验（Table 3，OCTA-6M）

| CSA | VSA | MAE↓ | PSNR(dB)↑ | SSIM(%)↑ |
|-----|-----|------|-----------|---------|
| ✗ | ✓ | 0.0765 | 32.31 | 89.53 |
| ✓ | ✗ | 0.0748 | 32.87 | 89.19 |
| ✓ | ✓ | **0.0741** | **33.08** | **90.04** |

### OCTA2024投影图指标

| 方法 | MAE↓ | PSNR(dB)↑ | SSIM(%)↑ |
|------|------|-----------|---------|
| TransPro | 0.01056 | 37.85 | 87.61 |
| **MuTri** | **0.00870** | **39.65** | **90.31** |

### 关键发现

- 在所有三个数据集的所有指标上全面SOTA
- OCTA-3M比TransPro提高**1.54 dB PSNR**，OCTA-6M提高**2.55 dB**
- CSA是最关键组件，单独贡献0.56 dB；VSA在此基础上再贡献0.21 dB
- 大规模OCTA2024数据集上提升更显著（因为更多数据能训练更好的预训练模型）
- 眼科医师诊断分析确认MuTri翻译的OCTA更接近真实OCTA的病变模式

## 亮点与洞察

1. **离散vs连续空间**：将跨模态翻译从连续空间转到离散码本空间，用码本的有限性约束映射不确定性，对医学影像翻译是重要的方法论转变
2. **预训练模型作教师**：不直接蒸馏，而是通过对比学习让翻译模型与预训练模型的特征互信息最大化，驱动码本探索新码字
3. **投影图的巧妙利用**：2D投影图是3D体积沿深度平均，天然突出血管结构。用patch-level相似度矩阵做结构对齐，比逐像素损失更鲁棒
4. **首个大规模数据集OCTA2024**：846对OCT-OCTA体积，推动领域发展

## 局限性

1. VQVAE的码本大小和patch大小作为超参需要手动调优
2. 两阶段训练增加了总体训练时间和复杂度
3. 编码器/解码器仅用ResBlock堆叠，未探索更强backbone（如Swin Transformer）
4. 仅在视网膜OCT/OCTA上验证，泛化到其他3D医学翻译任务（如CT→MRI）需进一步验证

## 相关工作与启发

- **TransPro** [Sun et al., MIA]：3D OCT→OCTA的直接前身，用预训练血管分割模型做单视图指导。MuTri用三视图替代且不依赖分割标注
- **VQ-I2I** [ECCV]：将VQ引入2D图像翻译，MuTri将VQ扩展到3D医学翻译并解决码本坍塌
- **CUT** [Park et al.]：patch-wise对比学习用于无配对翻译的先驱，MuTri借鉴但用于跨任务（重建vs翻译）的特征对齐
- **SynthSeg的多模态预训练思路**：预训练各域模型提供先验的策略可推广到其他跨模态任务

## 评分

- ⭐ 创新性：8/10 — VQ+三视图对齐的框架新颖，对比学习驱动码本探索有理论支撑
- ⭐ 实验完备性：8/10 — 三数据集+消融+敏感性+临床诊断分析全面
- ⭐ 实用价值：7/10 — 解决了实际临床需求（低成本OCT设备升级），但两阶段训练略重
- ⭐ 总体：8/10 — 医学影像跨模态翻译的扎实工作，方法论可推广性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Cross-View Distillation and Adaptive Masking for Incomplete Multi-View Multi-Label Classification](../../CVPR2026/model_compression/cross-view_distillation_and_adaptive_masking_for_incomplete_multi-view_multi-lab.md)
- [\[CVPR 2026\] Parallax to Align Them All: An OmniParallax Attention Mechanism for Distributed Multi-View Image Compression](../../CVPR2026/model_compression/parallax_to_align_them_all_an_omniparallax_attention_mechanism_for_distributed_m.md)
- [\[ACL 2025\] Magnet: Multi-turn Tool-use Data Synthesis and Distillation via Graph Translation](../../ACL2025/model_compression/magnet_multi-turn_tool-use_data_synthesis_and_distillation_via_graph_translation.md)
- [\[CVPR 2025\] InsTaG: Learning Personalized 3D Talking Head from Few-Second Video](instag_learning_personalized_3d_talking_head_from_few-second_video.md)
- [\[CVPR 2025\] IterIS: Iterative Inference-Solving Alignment for LoRA Merging](iteris_iterative_inference-solving_alignment_for_lora_merging.md)

</div>

<!-- RELATED:END -->
