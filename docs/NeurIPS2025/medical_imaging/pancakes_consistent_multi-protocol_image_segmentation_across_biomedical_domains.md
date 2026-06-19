---
title: >-
  [论文解读] Pancakes: Consistent Multi-Protocol Image Segmentation Across Biomedical Domains
description: >-
  [NeurIPS 2025][医学图像][多协议分割] 提出 Pancakes 框架，给定来自未见过领域的生物医学图像集合，自动生成多个合理分割协议（protocol）的标签图，且同一协议下不同图像的标签具有语义一致性：——同一标签在所有图像中指代相同的解剖结构。 生物医学图像可以按多种方式分割——根据组织类型、血管分区、解…
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "多协议分割"
  - "语义一致性"
  - "基础模型"
  - "生物医学图像"
  - "无监督分割"
---

# Pancakes: Consistent Multi-Protocol Image Segmentation Across Biomedical Domains

**会议**: NeurIPS 2025  
**arXiv**: [2512.13534](https://arxiv.org/abs/2512.13534)  
**代码**: 暂无  
**领域**: 医学图像 / 图像分割  
**关键词**: 多协议分割, 语义一致性, 基础模型, 生物医学图像, 无监督分割

## 一句话总结

提出 Pancakes 框架，给定来自未见过领域的生物医学图像集合，自动生成多个合理分割协议（protocol）的标签图，且同一协议下不同图像的标签具有**语义一致性**——同一标签在所有图像中指代相同的解剖结构。

## 研究背景与动机

生物医学图像可以按多种方式分割——根据组织类型、血管分区、解剖区域、病理等不同协议。现有自动分割方法存在两个根本性问题：

**单协议绑定**：传统模型（如 nnU-Net）只支持训练时指定的单一协议，换协议需要重新标注和训练。

**交互式方法的负担**：SAM、ScribblePrompt 等虽可指定新协议，但需要用户为每张图像提供点击/涂鸦/示例等交互输入，在处理大量图像时极其费力。

**跨图像不一致**：现有基础模型（SAM、UnSAM 等）虽能全自动分割每张图像，但生成的标签在不同图像间**语义不一致**——同一标签可能在不同图像中指代完全不同的结构，同一结构在不同图像中被赋予不同标签ID，使群体分析无法进行。

Pancakes 提出了一个全新的能力：给定一组新领域的图像，自动发现并生成多个合理的分割协议，每个协议在图像集合内保持语义一致。用户无需指定协议，而是从模型发现的多个候选协议中选择最符合需求的那个。

## 方法详解

### 整体框架

Pancakes 的输入是图像 $x$，输出是分布参数 $\phi$，从中可采样多个协议的标签图。分为三步：
1. U-Net 网络 $f_{\theta_f}(x) = \phi$ 估计每个像素上的分布参数
2. 基于随机变量 $r_m = (M, K)$ 采样 $M$ 个协议，每个含 $K$ 个标签
3. 浅层卷积网络 $h_{\theta_h}(\phi || v_m) = \hat{y}_m$ 从分布参数生成具体标签图

### 关键设计

1. **协议采样机制**：使用类位置编码的向量表示来区分不同协议和标签。对于协议 $m$ 和标签 $k$，构建向量 $v_{m,k} = u_m || u_k$，其中 $u_t^{2j} = \sin(z_{t,2j} + \pi/2)$，$z_{t,j} = \frac{t\pi}{T} 2^{2j\pi/J}$。周期 $T$ 由采样的 $M$ 和 $K$ 决定。这个设计使得不同协议和标签的编码彼此可区分，同时支持灵活的协议数和标签数。关键是 $\phi$ 只计算一次（主网络前向传播），然后可高效地对多个 $r_m$ 生成不同的标签图。

2. **集合一致性损失**：训练时每次采样一组图像 $\{x_s\}$ 和对应的二值标签 $\{y_s\}$，生成 $M$ 个协议的标签图。损失函数通过在集合内取平均 Dice 分数来鼓励跨图像一致性：

$$d_{m,k}(\{\hat{y}_{s,m,k}\}, \{y_s\}) = \mathbb{E}_s[\mathcal{L}_{\text{Dice}}(\hat{y}_{s,m,k}, y_s)]$$

$$\mathcal{L}_{\text{seg}} = \min_{m,k} d_{m,k}(\{\hat{y}_{s,m,k}\}, \{y_s^t\})$$

取 min 操作（而非期望）鼓励多样性——至少有一个协议-标签组合匹配真值，而其他协议可自由发现不同的分割方式。跨集合平均则确保标签 $k$ 在协议 $m$ 中对所有图像指代同一区域。

3. **合成数据增强**：基于 Anatomix 和 TotalSegmentator，合成 120K 训练对。从分割图采样二值标签，对同一标签图施加不同的仿射/弹性变换模拟同集合图像，再赋予强度值生成合成图像。这增加了训练的多样性并改善了对新领域的泛化。

### 损失函数 / 训练策略

- 输入维度 $B \times S \times C \times H \times W$，展平为 $(B \times S) \times C \times H \times W$ 以兼容 2D 卷积
- 训练时随机采样：$K \in [5, 40]$，$M \in [5, 15]$，$S \in [2, 5]$
- 数据增强分两类：in-task（独立增加集合多样性）和 task（全集合一致，增加协议多样性）
- AdamW 优化器，学习率 0.0001
- 跨域训练数据来自 Megamedical（包含数十种生物医学数据集）
- 推理时 $K$ 和 $M$ 由用户选择，SoftMax 保证非重叠标签

## 实验关键数据

### 主实验：7 个未见数据集上的 Set Dice 对比

Pancakes 在所有 7 个 held-out 数据集上超越基线，通常领先 20+ Dice points。

| 方法 | 参数量 | S=1 推理时间 | S=3 推理时间 |
|------|--------|-------------|-------------|
| Pancakes | **0.22M** | **0.10s** | **0.12s** |
| SAM | 641M | 3.13s | 2.94s |
| ScribblePrompt | 93.7M | 1.99s | 1.85s |
| MedSAM | 93.7M | 2.12s | 1.94s |
| UnSAM | 23M | 0.54s | 0.45s |

### 准确性 vs 一致性分析

| 设置 | Pancakes 表现 | 基线表现 |
|------|-------------|---------|
| S=1（仅准确性） | 与 SAM 相当，优于其他 | SAM 最优 |
| S=3（准确性+一致性） | **性能不下降** | **所有基线大幅下降** |
| S=5（更大集合） | 仍然稳定 | 进一步恶化 |

### 消融实验：合成数据的影响（M=16）

| 训练数据 | S=1 | S=2 | S=3 | S=5 |
|---------|-----|-----|-----|-----|
| 真实 + 合成 | **73.2** | **67.3** | **67.4** | **68.4** |
| 仅真实 | 71.1 | 65.8 | 65.7 | 67.4 |
| 仅合成 | 56.3 | 45.8 | 44.3 | 42.7 |

### M 和 K 的影响

| 配置 | 关键发现 |
|------|---------|
| M 增大 | 性能普遍提升，更多协议 → 更可能覆盖真值 |
| K 增大 | 效果较复杂，K=20 附近最优；更大 K 产生更精细的结构分割 |
| 交互式初始化 | Pancakes 初始化 + ScribblePrompt：点击次数减半（3-4次 vs 5-8次） |

### 关键发现

- **一致性是核心优势**：Pancakes 是唯一在集合大小增大时性能不下降的方法，因为所有基线都不保证跨图像语义一致性
- 合成数据与真实数据联合训练效果最佳（$p < 0.05$）
- 协议空间近似平滑——嵌入空间中相近的协议产生相似的分割
- 仅 0.22M 参数，比 SAM（641M）快 30 倍

## 亮点与洞察

- **问题定义创新**：提出"多协议一致分割"这一新任务，填补了现有基础模型的重要空白
- **min-Dice 损失的巧妙**：通过取最优协议-标签组合的 min，而非平均，既鼓励多样性又避免回归均值
- **极致高效**：0.22M 参数的全卷积架构达到了 641M 参数 SAM 的准确度，推理速度快 30+倍
- **两种实际应用场景**：(1) 新协议的快速分割——选择最匹配的候选协议；(2) 探索性群体分析——发现与临床结局关联的候选分割区域
- **与交互式分割的互补**：可作为 ScribblePrompt 等的初始化，减少人工交互次数

## 局限与展望

- 生成的协议是"模型发现的合理分割"，不一定对应临床上已有的标准协议
- 训练依赖 Megamedical 数据集，对极端新领域的泛化能力待验证
- 2D 处理，未扩展到 3D 体积分割
- 协议的语义解释需要专家参与——标签编号本身无含义
- 未评估潜在的社会偏差

## 相关工作与启发

- **UniverSeg / Tyche**: 上下文分割模型，需要示例指定协议
- **SAM / SAM2**: 通用交互分割，但跨图像不一致
- **SynthSeg**: 合成数据训练的分割方法
- **Anatomix**: 合成生物医学数据生成
- 启发：一致性（Consistency）是群体分析中被低估的关键属性；位置编码类技术可用于参数化离散选择空间

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 全新的问题定义，多协议一致分割前所未有
- **实验充分度**: ⭐⭐⭐⭐ 7 个 held-out 数据集 + 消融 + 效率分析，全面
- **写作质量**: ⭐⭐⭐⭐⭐ 问题动机阐述极为清晰，图表信息丰富
- **价值**: ⭐⭐⭐⭐⭐ 解决了基础模型在生物医学应用中的核心痛点，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Noise-Consistent Siamese-Diffusion for Medical Image Synthesis and Segmentation](../../CVPR2025/medical_imaging/noise-consistent_siamese-diffusion_for_medical_image_synthesis_and_segmentation.md)
- [\[CVPR 2025\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](../../CVPR2025/medical_imaging/biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [\[NeurIPS 2025\] Orochi: Versatile Biomedical Image Processor](orochi_versatile_biomedical_image_processor.md)
- [\[NeurIPS 2025\] LoMix: Learnable Weighted Multi-Scale Logits Mixing for Medical Image Segmentation](lomix_learnable_weighted_multi-scale_logits_mixing_for_medical_image_segmentatio.md)
- [\[ICCV 2025\] MultiverSeg: Scalable Interactive Segmentation of Biomedical Imaging Datasets with In-Context Guidance](../../ICCV2025/medical_imaging/multiverseg_scalable_interactive_segmentation_of_biomedical_imaging_datasets_wit.md)

</div>

<!-- RELATED:END -->
