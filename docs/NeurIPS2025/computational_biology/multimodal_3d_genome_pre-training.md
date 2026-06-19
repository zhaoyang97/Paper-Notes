---
title: >-
  [论文解读] Multimodal 3D Genome Pre-training
description: >-
  [NeurIPS 2025][计算生物][3D基因组] 提出MIX-HIC——首个面向3D基因组的多模态基础模型，通过跨模态交互块和跨模态映射块融合Hi-C接触图和表观基因组信号，在超过127万对样本上预训练，在Hi-C预测、染色质环检测和CAGE-seq表达预测三个下游任务上全面超越SOTA。 三维基因组组织（如染色质环、…
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "3D基因组"
  - "Hi-C"
  - "表观基因组"
  - "多模态预训练"
  - "基础模型"
---

# Multimodal 3D Genome Pre-training

**会议**: NeurIPS 2025  
**arXiv**: [2504.09060](https://arxiv.org/abs/2504.09060)  
**代码**: [github.com/myang998/MIX-HIC](https://github.com/myang998/MIX-HIC)  
**领域**: 计算生物  
**关键词**: 3D基因组, Hi-C, 表观基因组, 多模态预训练, 基础模型

## 一句话总结
提出MIX-HIC——首个面向3D基因组的多模态基础模型，通过跨模态交互块和跨模态映射块融合Hi-C接触图和表观基因组信号，在超过127万对样本上预训练，在Hi-C预测、染色质环检测和CAGE-seq表达预测三个下游任务上全面超越SOTA。

## 研究背景与动机

三维基因组组织（如染色质环、拓扑关联域等）对基因调控和细胞功能起关键作用。Hi-C技术可以量化染色质间的3D交互，而表观基因组信号（ATAC-seq、DNase-seq等）则反映染色质的开放状态。理解二者的关系对于揭示基因表达调控机制至关重要。

现有方法面临三大核心挑战：

**第一，异构数据融合困难。** Hi-C接触图是2D矩阵形式的空间交互数据，表观基因组是1D序列信号，二者特征天生异构。简单地将两种模态特征对齐到共享空间主要捕获模态不变知识（如基因调控机制），但忽略了模态特有特征（如表观基因组揭示的精确化学修饰和染色质状态），导致信息损失。论文通过定理1严格证明：完美的特征对齐会引入至少 $\Gamma_q$ 的信息差距，使预测误差低于直接使用原始数据。

**第二，泛化能力不足。** 现有方法通常针对单一任务优化，难以适应生成（Hi-C预测）和回归（表达预测）等多样化下游任务。

**第三，数据稀缺问题。** Hi-C实验成本高昂，实际中常面临模态缺失的情况。需要模型能从已有模态推断缺失模态的语义。

本文的切入角度是：构建首个多模态3D基因组基础模型，通过(1)分离学习模态不变和模态特有表示解决信息损失问题，(2)跨模态映射块实现缺失模态补全，(3)超百万规模配对数据预训练获得强泛化能力。

## 方法详解

### 整体框架
MIX-HIC采用双编码器架构。预训练阶段包含特征提取块、跨模态交互块和跨模态映射块三个核心组件。微调阶段通过模态融合块和任务特定解码器适配不同下游任务。支持三种输入模式：双模态输入（MIX-HIC-Bimodal）、非预训练双模态（MIX-HIC-NonPre）和单模态推断（MIX-HIC-Infer）。

### 关键设计

1. **特征提取块（双编码器）**: 

    - Hi-C编码器：基于ViT架构，将 $50 \times 50$ 的Hi-C接触图切分为patch（patch size=2），经三层Transformer编码器逐渐下采样，生成Hi-C嵌入 $X_M^B \in \mathbb{R}^{\alpha_3 \times C_3}$
    - 表观基因组编码器：处理5000长度的ATAC-seq和DNase-seq信号，先用卷积+池化提取初始嵌入，再经三层Transformer编码，得到表观嵌入 $X_E^B \in \mathbb{R}^{\beta_3 \times C_3}$
    - 设计动机：Transformer架构能同时建模Hi-C的空间交互和表观基因组的序列依赖，两种模态使用独立编码器保留各自特征空间

2. **跨模态交互块**: 

    - 功能：分离学习模态不变特征和模态特有特征
    - 核心思路：用4个独立的全连接网络分别从 $X_M^B$ 和 $X_E^B$ 提取模态不变表示 $X_M^I, X_E^I$ 和模态特有表示 $X_M^S, X_E^S$。通过对比学习损失 $\mathcal{L}_{\text{con}}$ 拉近模态不变表示：$\mathcal{L}_{\text{con}} = \frac{1}{2}(\mathcal{L}_{\text{pair}}(\hat{X_E^I}, \hat{X_M^I}) + \mathcal{L}_{\text{pair}}(\hat{X_M^I}, \hat{X_E^I}))$。通过正交损失 $\mathcal{L}_{\text{orth}} = \frac{1}{2}(\langle \hat{X_M^S}, \hat{X_M^I} \rangle + \langle \hat{X_E^S}, \hat{X_E^I} \rangle)$ 确保特有信息与不变信息正交互补。
    - 设计动机：定理1证明了完美对齐的信息损失，因此必须同时保留共享知识和模态独有信息

3. **跨模态映射块**: 

    - 功能：学习模态间的隐式语义关系，支持缺失模态推断
    - 核心思路：将各模态的不变+特有表示拼接为 $X_M^{\text{Concat}}$ 和 $X_E^{\text{Concat}}$，通过1D自适应池化对齐长度后，用全连接层学习映射 $X_{\text{M2E}}$ 和 $X_{\text{E2M}}$。映射损失：$\mathcal{L}_{\text{mapping}} = \frac{1}{2}(\|X_{\text{M2E}} - X_E^{\text{Concat}}\|_2^2 + \|X_{\text{E2M}} - X_M^{\text{Concat}}\|_2^2)$
    - 设计动机：Hi-C实验成本高，经常面临模态缺失。该模块使模型在仅有表观基因组数据时也能推断Hi-C特征，解决数据稀缺问题

### 损失函数 / 训练策略
预训练总损失：$\mathcal{L}_{\text{pretrain}} = \mathcal{L}_{\text{con}} + \mathcal{L}_{\text{orth}} + \mathcal{L}_{\text{mapping}}$

微调阶段使用任务特定损失：染色质环检测用BCE损失，Hi-C预测和CAGE-seq表达预测用MSE损失。数据经RPGC（CAGE-seq）和KR（Hi-C）归一化后取log变换。

预训练数据集包含4个细胞系（HepG2, HCT116, IMR90, WTC11），清洗后保留1,275,948对样本——这是迄今最大的3D基因组配对数据集。

## 实验关键数据

### 主实验

**Hi-C接触图预测 ($R^2$)**

| 方法 | GM12878 | K562 | 提升 |
|------|---------|------|------|
| Epiphany | 0.7970 | 0.6547 | - |
| EPCOT-LSTM | 0.7993 | 0.7840 | - |
| C.Origami | 0.7958 | 0.7055 | - |
| **MIX-HIC-Infer** | **0.8724** | **0.8001** | **+9.3% / +2.1%** |

**染色质环检测**

| 方法 | GM12878 F1 | K562 F1 | GM12878 AUROC | K562 AUROC |
|------|-----------|---------|---------------|------------|
| Peakachu | 0.8015 | 0.7900 | 0.8766 | 0.8834 |
| DLoopCaller | 0.8250 | 0.7932 | 0.9046 | 0.8924 |
| **MIX-HIC-Bimodal** | **0.8420** | **0.8267** | **0.9209** | **0.9194** |

**CAGE-seq表达预测 ($R^2$)**

| 方法 | GM12878 | K562 |
|------|---------|------|
| EPCOT-Transformer | 0.8578 | 0.8230 |
| EPI-Graph | 0.7965 | 0.8211 |
| **MIX-HIC-Bimodal** | **0.8833** | **0.9077** |

### 消融实验

**损失函数组件消融（AUROC，染色质环检测）**

| $\mathcal{L}_{\text{con}}$ | $\mathcal{L}_{\text{orth}}$ | $\mathcal{L}_{\text{mapping}}$ | GM12878 | K562 |
|:---:|:---:|:---:|---------|------|
| ✓ | - | - | 0.9136 | 0.9099 |
| ✓ | ✓ | - | 0.9183 | 0.9156 |
| ✓ | ✓ | ✓ | 0.9209 | 0.9194 |

**模态组合消融（Hi-C预测，$R^2$）**

| 配置 | 预训练 | GM12878 | K562 | 说明 |
|------|--------|---------|------|------|
| 仅Epi | 否 | 0.8481 | 0.7709 | 单模态基线 |
| Epi+推断Hi-C | 是 | 0.8724 | 0.8001 | 跨模态映射有效 |
| Epi+Hi-C | 否 | 0.8614 | 0.8755 | 非预训练双模态 |
| Epi+Hi-C | 是 | 0.8833 | 0.9077 | 预训练完整版本最优 |

### 关键发现
- Hi-C预测任务上相比次优方法提升9.3%，改进最为显著，证明预训练在捕获跨模态语义关系方面的价值
- 少样本实验显示：仅用10%训练数据，MIX-HIC-Bimodal即可达到约0.9 AUROC，与完整数据训练的其他SOTA方法相当
- 跨细胞系评估中MIX-HIC保持最优，表明其在新细胞类型上有良好泛化
- 非预训练的双模态版本在K562环检测上反而不如仅用Hi-C单模态，证实了定理1关于简单对齐会丢信息的论断

## 亮点与洞察
- 这是3D基因组领域的首个多模态基础模型，开创了将基础模型范式引入该领域的先河。超过127万对的预训练数据规模远超同领域现有工作。
- 定理1从理论上证明了完美模态对齐不如分离学习模态不变/特有表示，这一洞察不局限于生物信息学领域，对所有多模态融合任务都有启示意义。

## 局限与展望
- Hi-C分辨率固定为5kb，更高分辨率（如1kb）的建模需要更大的数据和模型
- 预训练仅覆盖4个细胞系，扩展到更多细胞类型和物种可进一步增强泛化性
- 跨模态映射尽管有效但提升幅度相对有限（约0.5% AUROC），更强的模态补全策略值得探索

## 相关工作与启发
- **vs Epiphany**: Epiphany仅用表观基因组预测Hi-C，MIX-HIC-Infer通过预训练学到的跨模态语义在相同设定下提升9.3%
- **vs EPCOT**: EPCOT在不同数据集上性能波动大，MIX-HIC通过大规模预训练获得稳健表现
- **vs RefHiC**: RefHiC的预训练局限于小规模数据和单一任务，MIX-HIC的百万级预训练使其具备多任务泛化能力

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 3D基因组领域首个多模态基础模型，定理1提供了有价值的理论洞察
- 实验充分度: ⭐⭐⭐⭐⭐ 3个下游任务、2个细胞系、少样本/跨细胞系/消融实验非常完整
- 写作质量: ⭐⭐⭐⭐ 结构严谨，理论与实验结合紧密
- 价值: ⭐⭐⭐⭐⭐ 为3D基因组研究提供了新范式，数据集和代码完全开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Contact-Guided 3D Genome Structure Generation of E. coli via Diffusion Transformers](../../ICLR2026/computational_biology/contact-guided_3d_genome_structure_generation_of_e_coli_via_diffusion_transforme.md)
- [\[ICLR 2026\] Zatom-1: A Multimodal Flow Foundation Model for 3D Molecules and Materials](../../ICLR2026/computational_biology/zatom-1_a_multimodal_flow_foundation_model_for_3d_molecules_and_materials.md)
- [\[CVPR 2025\] Synthetic Visual Genome](../../CVPR2025/computational_biology/synthetic_visual_genome.md)
- [\[NeurIPS 2025\] Manipulating 3D Molecules in a Fixed-Dimensional E(3)-Equivariant Latent Space](manipulating_3d_molecules_in_a_fixed-dimensional_e3-equivariant_latent_space.md)
- [\[NeurIPS 2025\] Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling](towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)

</div>

<!-- RELATED:END -->
