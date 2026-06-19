---
title: >-
  [论文解读] Efficient and Versatile Robust Fine-Tuning of Zero-shot Models
description: >-
  [ECCV 2024][语义分割][鲁棒微调] R-Adapter 通过在 CLIP 模型中插入轻量级 adapter 模块并结合三种自集成策略（Adapter Dropping、权重累积、权重缩放重参数化），在仅微调 13% 参数的前提下同时实现了 ID 高精度和 OOD 强鲁棒性，并首次将鲁棒微调扩展到分类之外的跨模态检索和开放词汇分割任务。
tags:
  - "ECCV 2024"
  - "语义分割"
  - "鲁棒微调"
  - "参数高效微调"
  - "自集成"
  - "CLIP"
  - "分布外泛化"
---

# Efficient and Versatile Robust Fine-Tuning of Zero-shot Models

**会议**: ECCV 2024  
**arXiv**: [2408.05749](https://arxiv.org/abs/2408.05749)  
**代码**: 有 (cvlab.postech.ac.kr/research/R-Adapter)  
**领域**: 图像分割 / 视觉语言模型微调  
**关键词**: 鲁棒微调, 参数高效微调, 自集成, CLIP, 分布外泛化

## 一句话总结

R-Adapter 通过在 CLIP 模型中插入轻量级 adapter 模块并结合三种自集成策略（Adapter Dropping、权重累积、权重缩放重参数化），在仅微调 13% 参数的前提下同时实现了 ID 高精度和 OOD 强鲁棒性，并首次将鲁棒微调扩展到分类之外的跨模态检索和开放词汇分割任务。

## 研究背景与动机

### 核心矛盾

大规模图像-文本预训练模型（如 CLIP）具备零样本分类能力和跨分布泛化能力，但在下游任务上微调时面临两个相互矛盾的挑战：

**鲁棒性下降**：全量微调会破坏预训练知识，导致分布外（OOD）数据精度大幅下降。例如 CLIP ViT-B/16 零样本 OOD 平均准确率 58.4%，全量微调后降至 52.8%——ID 提升 18.4%，OOD 仅提升 2.0%

**计算代价过高**：全量微调需更新所有参数（如 ViT-L/14 有 305M 参数），内存和存储开销随模型规模增长不可持续

### 现有方法的局限

| 方法类型 | 代表 | 优点 | 不足 |
|---------|------|------|------|
| 鲁棒微调 (WiSE-FT, Mask-Fill) | 权重空间插值/正则化 | OOD 鲁棒 | 需全量微调（80M+ 参数）；仅限分类任务 |
| 参数高效微调 PEFT (AdaptFormer, MaPLe) | 仅调少量参数 | 高效 | OOD 鲁棒性差，分布偏移下性能骤降 |

**关键洞察**：还没有方法能同时解决这两个问题。鲁棒微调方法效率低，PEFT 方法不鲁棒。R-Adapter 的核心思路是：**在 PEFT 框架中引入自集成机制，以低成本实现权重空间集成的鲁棒性增益**。

### 集成的启发

权重空间集成（如 WiSE-FT 线性插值预训练和微调权重）已被证明能提升 OOD 泛化。但传统方法需存储两个完整模型。R-Adapter 的创新在于：通过 adapter 的重参数化，将权重空间集成压缩到**单个模型**中，仅需存储 adapter 权重（约 13% 参数量）。

## 方法详解

### 整体框架

R-Adapter 构建在 CLIP 的 Vision Transformer 和 Text Transformer 之上：
- 在每一层 Transformer 的 MHA 和 FFN 之后各插入一个 adapter 模块
- 冻结所有预训练参数，仅训练 adapter 权重
- 结合三种自集成策略提升 OOD 鲁棒性
- 使用 MPM-NCE 损失函数替代标准 InfoNCE

### 关键设计

#### 1. **Adapter 模块设计**

采用简化版 Houlsby Adapter（去除非线性层和偏置项），结构为残差线性变换：

$$h(X) = X W_{\text{adp}} + X$$

其中 $W_{\text{adp}} \in \mathbb{R}^{d \times d}$ 是 adapter 权重矩阵。对于全量数据使用全秩结构，少样本学习可采用低秩分解 $W_{\text{adp}} = BA$（$B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}, r \ll d$）。

由于 adapter 没有非线性层，可以通过**重参数化**与前一层预训练权重合并，推理时零额外计算开销：

$$W_{\text{rep}} = W_{\text{org}}(W_{\text{adp}} + I), \quad b_{\text{rep}} = b_{\text{org}}(W_{\text{adp}} + I)$$

**设计动机**：线性 adapter 使重参数化成为可能，这是后续权重空间集成的基础——无需存储两个独立模型。

#### 2. **三种自集成策略**

**(a) Dynamic Ensemble by Adapter Dropping**

训练时以概率 $p$ 随机关闭 adapter 模块：

$$h(X) = \frac{\gamma}{1-p} \cdot X W_{\text{adp}} + X, \quad \gamma \sim \text{Bernoulli}(1-p)$$

推理时不应用 dropping。这等价于隐式地对多个子网络（不同 adapter 开启/关闭组合）进行集成。

**与 Dropout/Drop-path 的本质区别**：Dropout 制造特征稀疏性，Drop-path 减少模型深度，而 Adapter Dropping 独特地在**预训练特征**和**adapter 适配特征**之间随机切换，确保预训练知识始终得到保留。消融实验显示单独使用 AD 即提升 OOD 1.9%（E3 vs E0）。

**(b) Temporal Ensemble by Accumulation**

通过指数移动平均（EMA）累积 adapter 权重的历史版本：

$$\tilde{W}_{\text{adp}} \leftarrow m \cdot \tilde{W}_{\text{adp}} + (1-m) \cdot W_{\text{adp}}$$

推理时使用累积权重 $\tilde{W}_{\text{adp}}$。这是一种**时间维度**的集成，以几乎零额外成本捕获训练过程中所有模型的综合信息。

**设计动机**：仅对 adapter 参数做 EMA（而非整个模型），内存效率极高。

**(c) Weight-space Ensemble by Re-scaling**

通过缩放系数 $\alpha$ 在预训练权重和微调权重之间做线性插值：

$$W_{\text{ens}} = \alpha \tilde{W}_{\text{rep}} + (1-\alpha) W_{\text{org}} = W_{\text{org}}(\alpha \tilde{W}_{\text{adp}} + I)$$

**核心优势**：传统 WiSE-FT 需要存储两个完整模型才能做权重插值，而 R-Adapter 仅需存储 adapter 权重，通过重参数化在**单个模型内**实现等价的权重空间集成。$\alpha$ 控制预训练知识保留程度（单独使用 RS 提升 OOD 5.8%，E5 vs E0）。

#### 3. **MPM-NCE 损失函数**

针对下游视觉语言任务设计，解决标准 InfoNCE 的两个问题：

**(a) 多正样本软标签**：在分类任务中同一类别有多个文本模板，InfoNCE 只考虑单正样本对会导致语义冲突。MPM-NCE 使用软标签分配：

$$\tilde{y}_{ij} = \frac{(1-\epsilon) \cdot y_{ij}}{|P(i)|} + \frac{\epsilon \cdot (1-y_{ij})}{B - |P(i)|}$$

其中 $P(i)$ 是样本 $i$ 的正样本集合，$\epsilon$ 是标签平滑噪声。

**(b) 角度 margin**：对负样本对施加 margin $\delta$ 增强判别性：

$$\mathcal{L}(\mathcal{B}) = -\sum_{i,j=1}^{B} \left(\tilde{y}_{ij} \log \frac{e^{(f_i \cdot g_j + \delta_{ij})/\tau}}{\sum_{k=1}^{B} e^{(f_i \cdot g_k + \delta_{ik})/\tau}} + \tilde{y}_{ji} \log \frac{e^{(f_j \cdot g_i + \delta_{ji})/\tau}}{\sum_{k=1}^{B} e^{(f_k \cdot g_i + \delta_{ki})/\tau}}\right)$$

其中 $\delta_{ij} = 0$（正样本对）或 $\delta$（负样本对），$\tau = 0.01$。

### 训练策略

- 优化器：AdamW（无 weight decay）
- 训练 10 epochs，学习率 $5 \times 10^{-4}$，cosine schedule + 500 步 warmup
- 超参数：$p=0.2$（drop 概率），$m=0.999$（EMA 动量），$\delta=0.05$（margin），$\alpha=0.5$（分类任务）

## 实验关键数据

### 主实验

ImageNet 分类 + 5 个 OOD 数据集（CLIP ViT-B/16）：

| 方法 | 可训练参数 | IN (ID) | OOD avg | IN-V2 | IN-R | IN-Sketch | ObjectNet | IN-A |
|------|-----------|---------|---------|-------|------|-----------|-----------|------|
| Zero-Shot | 0 | 68.3 | 58.4 | 61.9 | 77.6 | 48.3 | 54.0 | 50.1 |
| Fine-Tuning | 86.7M | 80.7 | 52.8 | 70.4 | 64.0 | 45.1 | 49.1 | 35.2 |
| WiSE-FT | 86.7M | 81.7 | 63.0 | 72.8 | 78.7 | 53.9 | 57.3 | 52.2 |
| Mask-Fill | 86.7M | 82.4 | 63.3 | 73.4 | 78.1 | 53.4 | 57.9 | 53.5 |
| **R-Adapter (本文)** | **20.5M** | 82.0 | **64.8** | 73.6 | **79.1** | **53.9** | **59.7** | **57.5** |

R-Adapter 以 **1/4 的参数量** 超越所有鲁棒微调方法的 OOD 性能（+1.5% OOD avg），在分布偏移最大的 IN-A 上优势尤为显著（+4.0%）。

### 消融实验

各组件对 ImageNet 分类的贡献（ViT-B/32）：

| 配置 | ID | OOD avg | 说明 |
|------|-------|---------|------|
| E0: 基础 Adapter | 77.5 | 47.7 | 基线 |
| E3: + Adapter Dropping | 77.8 (+0.3) | 49.6 (+1.9) | 动态集成显著提升 OOD |
| E5: + Re-scaling | 76.5 (-1.0) | 53.5 (+5.8) | 权重插值大幅提升 OOD |
| E7: AD + AC + RS | 76.6 (-0.9) | 53.7 (+6.0) | 三策略组合 |
| E9: + MPM-NCE | 77.5 (±0) | 53.9 (+6.2) | 损失函数改进 |
| E10: + Label Smooth | **77.7 (+0.2)** | **54.3 (+6.6)** | 最终配置 |
| B1: AdaptFormer | 77.2 | 48.5 | 现有 PEFT 基线 |
| B2: RepAdapter | 77.2 | 48.3 | 现有 PEFT 基线 |

超参数敏感性分析（ViT-B/32）：

| margin δ | ID | OOD | 说明 |
|----------|-----|-----|------|
| 0 | 77.1 | 54.0 | 无 margin |
| 0.05 (默认) | 77.7 | 54.3 | 最优平衡 |
| 0.1 | 77.8 | 53.8 | margin 过大反而降低 OOD |
| w/ Single Positive | 77.2 | 47.0 | 单正样本 + margin 效果差 |

### 关键发现

1. **Adapter Dropping ≠ Dropout**：Dropout (E1, +1.1 OOD) 和 Drop-path (E2, +0.2 OOD) 的效果远不如 Adapter Dropping (E3, +1.9 OOD)，因为后者在预训练知识和适配知识间切换
2. **Re-scaling 是 OOD 鲁棒性的关键**：单独使用 RS 即带来 +5.8% OOD 提升，但会轻微降低 ID (-1.0%)，需与 AD/AC 组合平衡
3. **MPM-NCE + Label Smoothing 的协同效应**：InfoNCE + LS 会因语义冲突降低 ID，但 MPM-NCE 的多正样本机制避免了此问题，两者结合同时提升 ID 和 OOD
4. **方法跨任务通用**：R-Adapter 在少样本分类（超越 CLIPood +1.2% OOD avg）、跨模态检索和开放词汇分割上均有效
5. **对 α 不敏感**：相比 WiSE-FT，R-Adapter 在不同 $\alpha$ 值下性能曲线更平稳

## 亮点与洞察

1. **概念统一**：首次将参数高效微调和鲁棒微调统一到一个框架中，证明两者可以互补而非对立
2. **重参数化实现轻量集成**：通过 adapter 的线性结构 + 重参数化，将 WiSE-FT 的双模型权重插值压缩为单模型内操作，存储成本从 2× 模型降至 adapter 参数量
3. **benchmark 扩展**：首次将鲁棒微调评估从分类扩展到跨模态检索和开放词汇分割，推动了该领域的系统性评估
4. **MPM-NCE 的通用性**：多正样本 + margin 的损失设计思路可广泛应用于存在多对多映射关系的视觉语言任务

## 局限与展望

1. 全秩 adapter 在 ViT-L 上有 64.5M 参数（仍较大），低秩版本性能有所下降
2. $\alpha$ 值需要针对不同任务手动调优（分类 0.5、检索 0.8、分割 0.4）
3. 未探索更复杂的 adapter 结构（如带门控机制），可能进一步提升 ID-OOD 平衡
4. 评估以 CLIP 为主，未验证在其他 VLM（如 ALIGN、SigLIP）上的效果
5. OOD 评估限于自然分布偏移，未涉及对抗性攻击场景

## 相关工作与启发

- **WiSE-FT** 的权重插值思想在此被优雅地融入 PEFT 框架
- Adapter Dropping 策略可启发其他 PEFT 方法（如 LoRA）的鲁棒性改进
- MPM-NCE 损失在 CLIP-based 的开放世界检测/分割中有应用潜力

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 三种自集成策略各具创意，重参数化实现权重集成的思路优雅
- **实验充分度**: ⭐⭐⭐⭐⭐ — 跨 4 个 ViT 规模、5 种任务、详尽的消融和超参数分析
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，数学推导完整，但部分公式符号略显繁重
- **实用价值**: ⭐⭐⭐⭐⭐ — 参数高效 + 鲁棒性好，适合实际部署场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Robust 3D Shape Reconstruction in Zero-Shot from a Single Image in the Wild](../../CVPR2025/segmentation/robust_3d_shape_reconstruction_in_zero-shot_from_a_single_image_in_the_wild.md)
- [\[NeurIPS 2025\] Fast and Fluent Diffusion Language Models via Convolutional Decoding and Rejective Fine-tuning](../../NeurIPS2025/segmentation/fast_and_fluent_diffusion_language_models_via_convolutional_decoding_and_rejecti.md)
- [\[ICML 2025\] InfoSAM: Fine-Tuning the Segment Anything Model from An Information-Theoretic Perspective](../../ICML2025/segmentation/infosam_fine-tuning_the_segment_anything_model_from_an_information-theoretic_per.md)
- [\[ICCV 2025\] ZIM: Zero-Shot Image Matting for Anything](../../ICCV2025/segmentation/zim_zero-shot_image_matting_for_anything.md)
- [\[ECCV 2024\] Early Preparation Pays Off: New Classifier Pre-tuning for Class Incremental Semantic Segmentation](early_preparation_pays_off_new_classifier_pre-tuning_for_class_incremental_seman.md)

</div>

<!-- RELATED:END -->
