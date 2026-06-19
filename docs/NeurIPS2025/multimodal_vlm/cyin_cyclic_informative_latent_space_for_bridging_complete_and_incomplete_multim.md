---
title: >-
  [论文解读] CyIN: Cyclic Informative Latent Space for Bridging Complete and Incomplete Multimodal Learning
description: >-
  [NeurIPS 2025][多模态VLM][不完整多模态学习] 提出 CyIN 框架，通过 token 级和 label 级信息瓶颈（IB）构建信息化潜空间，结合循环跨模态翻译重建缺失信息，在单一统一模型中同时优化完整和不完整多模态学习。 1. 领域现状 多模态学习在整合语言、视觉、音频等信息方面快速发展…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "不完整多模态学习"
  - "信息瓶颈"
  - "循环翻译"
  - "变分近似"
  - "模态缺失"
---

# CyIN: Cyclic Informative Latent Space for Bridging Complete and Incomplete Multimodal Learning

**会议**: NeurIPS 2025  
**arXiv**: [2602.04920](https://arxiv.org/abs/2602.04920)  
**代码**: [GitHub](https://github.com/RH-Lin/CyIN)  
**领域**: 多模态VLM  
**关键词**: 不完整多模态学习, 信息瓶颈, 循环翻译, 变分近似, 模态缺失

## 一句话总结

提出 CyIN 框架，通过 token 级和 label 级信息瓶颈（IB）构建信息化潜空间，结合循环跨模态翻译重建缺失信息，在单一统一模型中同时优化完整和不完整多模态学习。

## 研究背景与动机

### 1. 领域现状

多模态学习在整合语言、视觉、音频等信息方面快速发展，Transformer-based 模型通过多模态融合在情感分析、情绪识别等任务取得优异表现。绝大多数方法假设训练和推理时所有模态都完整存在。

### 2. 现有痛点

- **模态缺失问题**：真实场景中传感器故障、数据采集不完整导致模态随机缺失，预训练多模态模型性能急剧下降
- **现有方法不足**：
    - 对齐方法（对比学习、CCA）：不充分利用缺失信息
    - 生成方法（VAE、扩散模型）：容易引入与任务无关的噪声
    - 需要为每种缺失组合训练单独模型，泛化性差
- **完整 vs 不完整的权衡**：大多数方法在增强不完整鲁棒性时牺牲完整性能

### 3. 核心矛盾

如何在**单一统一模型**中同时保持完整多模态性能不下降，又能鲁棒地处理各种动态模态缺失场景？

### 4. 本文目标

构建统一的信息化潜空间，使完整模态的融合和缺失模态的重建都能从中受益。

### 5. 切入角度

将信息瓶颈（Information Bottleneck）理论引入多模态融合，在 token 级和 label 级进行循环信息压缩，过滤噪声、保留任务相关特征，并在纯化的信息空间中进行缺失模态翻译。

### 6. 核心 idea

用双层 IB 循环纯化多模态表征为信息化瓶颈潜变量，在该空间中既能高效融合又能准确翻译缺失信息。

## 方法详解

### 整体框架

CyIN 包含三个核心组件：
1. **Token 级信息瓶颈**：在 token 嵌入层面进行模态内/模态间循环压缩
2. **Label 级信息瓶颈**：利用标签监督注入高层语义
3. **跨模态循环翻译**：正向+反向传播重建缺失模态

### 关键设计

#### 模块1：Token 级信息瓶颈（Token-level IB）

**功能**：在低层感知级别过滤冗余噪声、提取任务相关特征。

**核心思路**：对源模态 $F_S$ 和目标模态 $F_T$ 的 token 嵌入，通过 IB 编码器压缩为瓶颈潜变量 $B_S$：

$$\mathcal{L}_{tib}^{S \to T} \approx \frac{1}{L} \sum_i^L \left\{ KL(\mathcal{N}(\mu_B^i, (\sigma_B^i)^2) \| \mathcal{N}(0, \mathbf{I})) + \beta \mathbb{E}_{b_S}[\|f_T - D_T(b_S)\|^2] \right\}$$

第一项压缩信息（KL 正则化），第二项保留目标模态的重建能力。

**循环交互**：
- **模态内**（$F_S = F_T$）：学习模态特有特征，ℒ$_{tib}^{S \to S}$
- **模态间**（$F_S \neq F_T$）：学习模态共享特征，ℒ$_{tib}^{S \to T}$ + ℒ$_{tib}^{T \to S}$

最终：$\mathcal{L}_{tib} = \mathbb{E}_{S \cup T}[\mathcal{L}_{tib}^{S \to S} + \frac{1}{2}(\mathcal{L}_{tib}^{S \to T} + \mathcal{L}_{tib}^{T \to S})]$

**设计动机**：变分近似 + 重参数化技巧实现可微的信息压缩，循环交互确保模态内独特性和模态间一致性都被保留。

#### 模块2：Label 级信息瓶颈（Label-level IB）

**功能**：注入任务高层语义监督信号。

**核心思路**：以 ground truth 标签 $y_{gt}$ 作为目标，对每个模态的表征进行 IB 压缩：

回归任务：
$$\mathcal{L}_{lib} \approx \mathbb{E}_S \left\{ \frac{1}{N} \sum_i^N KL(\mathcal{N}(\mu_B^i, (\sigma_B^i)^2) \| \mathcal{N}(0, \mathbf{I})) + \beta \mathbb{E}_{B_S}[\|y_{gt} - P_S(B_S)\|] \right\}$$

分类任务：
$$\mathcal{L}_{lib} \approx \mathbb{E}_S \left\{ \frac{1}{N} \sum_i^N KL(\cdot \| \cdot) - \beta \mathbb{E}_{B_S}[\sum^V y_{gt} \log P_S(B_S)] \right\}$$

**设计动机**：token-level IB 关注低层感知，label-level IB 确保瓶颈潜变量与下游任务强相关。

#### 模块3：跨模态循环翻译

**功能**：在信息化潜空间中重建缺失模态的表征。

**正向传播**：使用级联残差自编码器（CRA）翻译：
$$B_{S \to T}^{rec} = \Gamma_{S \to T}(B_S), \quad \mathcal{L}_{rec}^{S \to T} = \|B_T - B_{S \to T}^{rec}\|^2$$

**反向传播**（Back-Translation 技巧）：
$$B_S^{cyc} = \Gamma_{T \to S}(B_{S \to T}^{rec}), \quad \mathcal{L}_{cyc}^{T \to S} = \|B_S - B_S^{cyc}\|^2$$

**多模态缺失泛化**：当多个模态保留时，利用高斯混合特性直接求和各翻译器输出：
$$B_i^{rec} = \sum_{j \neq i}^{|u|} \Gamma_{j \to i}(B_j)$$

无需为每种缺失组合专门训练翻译器。

### 损失函数/训练策略

总损失：
$$\mathcal{L}_{total} = \mathcal{L}_{task} + \frac{1}{\beta}(\mathcal{L}_{tib} + \mathcal{L}_{lib}) + \gamma \mathcal{L}_{tran}$$

**两阶段训练**：
- 第一阶段 $\gamma = 0$：在完整数据上构建稳定的信息化潜空间
- 第二阶段 $\gamma > 0$：引入翻译损失，逐步训练跨模态翻译器

## 实验关键数据

### 主实验：4 数据集完整+不完整学习

**MOSI 数据集（情感回归）**：

| 设置 | 模型 | Acc7↑ | F1↑ | MAE↓ | Corr↑ |
|------|------|------|------|------|------|
| 完整 | MMIN | 43.2 | 85.0 | 0.744 | 0.782 |
| 完整 | IMDer | 43.8 | 85.7 | 0.724 | 0.796 |
| 完整 | **CyIN** | **48.0** | **86.3** | **0.712** | **0.801** |
| 固定缺失(avg) | MMIN | 31.3 | 68.4 | 1.093 | 0.433 |
| 固定缺失(avg) | IMDer | 31.4 | 70.6 | 1.043 | 0.533 |
| 固定缺失(avg) | **CyIN** | **32.8** | **72.2** | **1.037** | **0.599** |
| 随机缺失(avg) | MMIN | 33.3 | 70.9 | 1.014 | 0.584 |
| 随机缺失(avg) | **CyIN** | **35.0** | **75.7** | **0.943** | **0.650** |

**IEMOCAP 数据集（情感分类）**：

| 设置 | GCNet | IMDer | **CyIN** |
|------|------|------|------|
| 完整 Acc/wF1 | 63.0/63.0 | 64.4/64.8 | **66.1/66.0** |
| 固定缺失 Acc/wF1 | 52.8/51.9 | 54.7/54.4 | **57.4/56.6** |
| 随机缺失 Acc/wF1 | 55.3/55.3 | 55.8/56.1 | **57.5/57.5** |

**MELD 数据集（多方情感识别）**：

| 设置 | IMDer wF1 | LNLN wF1 | **CyIN** wF1 |
|------|------|------|------|
| 完整 | 59.7 | 57.1 | **59.8** |
| 固定缺失 | 49.8 | 44.7 | **49.4** |
| 随机缺失 | 49.5 | 49.0 | **50.5** |

### 消融实验

**信息瓶颈组件消融**（从论文补充材料推断）：
- Token-level IB 提升模态间交互质量
- Label-level IB 注入任务监督，显著提升不完整场景性能
- 循环翻译的正向+反向传播均不可或缺
- 两阶段训练比端到端训练更稳定

### 关键发现

1. **完整+不完整统一优化**：CyIN 是唯一在完整和各种不完整设置下同时达到最优或几乎最优的方法
2. **鲁棒性优势突出**：在高缺失率（MR=0.7）下，CyIN 相比基线保持更小的性能下降
3. **翻译质量可视化**：t-SNE 显示跨模态翻译的潜变量与原始模态潜变量高度聚集
4. **单一模型泛化**：无需为每种缺失组合训练专门模型，一个模型处理所有情况
5. **对弱模态也有效**：即使信息量少的模态（如视觉/音频），CyIN 也能有效利用其信息

## 亮点与洞察

1. **信息瓶颈与多模态的深度结合**：Token-level + Label-level 双层 IB 从感知和语义两个层面纯化表征，比简单对比学习更有效过滤噪声
2. **循环翻译设计精巧**：
    - 正向翻译重建缺失信息
    - 反向翻译（back-translation）通过循环一致性约束提升翻译质量
    - 高斯混合特性优雅解决多模态缺失场景
3. **在信息化空间翻译**：在纯化后的瓶颈空间（而非原始特征空间）做重建，难度更低、质量更高
4. **两阶段训练**：先建立稳定的信息空间，再训练翻译器，避免训练初期不稳定影响整体质量
5. **实用性强**：对不完整场景的处理无需事先知道哪些模态缺失

## 局限与展望

1. 实验仅覆盖语音/文本/视频三模态情感分析场景，缺乏在更多模态组合（如图文、图像-点云）上的验证
2. CRA（级联残差自编码器）作为翻译器设计相对简单，可能用更强的生成模型（扩散模型等）替代
3. $\beta$ 和 $\gamma$ 超参数对不同数据集的敏感性未充分讨论
4. Token-level IB 要求源和目标 token 序列等长或需对齐，可能限制对序列长度差异大的模态应用
5. 缺少与近期扩散模型基不完整多模态方法的对比
6. 信息瓶颈的压缩比（由 $\beta$ 控制）如何最优选择缺乏理论指导

## 相关工作与启发

- **与 MMIN 的关系**：MMIN 使用模态缺失模拟+对比学习；CyIN 用 IB 提供更原理性的信息控制
- **与 GCNet 的关系**：基于图的缺失重建方法；CyIN 的翻译在更紧凑的瓶颈空间进行
- **与 IMDer 的关系**：IMDer 隐式建模缺失影响；CyIN 显式构建跨模态翻译器
- **信息瓶颈理论的应用启发**：IB 不仅用于学习紧凑表征，更可作为控制跨模态信息流的工具
- **循环一致性的启发**：类似 CycleGAN 的 back-translation 思想，但应用于潜空间而非像素空间

## 评分

⭐⭐⭐⭐ (4/5)

框架设计完整，信息瓶颈与多模态融合/缺失重建的结合自然。实验覆盖4个数据集在完整和多种不完整设置下全面验证。理论直觉清晰但缺少更大规模/更多模态的验证。循环翻译在信息化空间中进行是核心创新。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Anchor-Guided Gradient Alignment for Incomplete Multimodal Learning](../../CVPR2026/multimodal_vlm/anchor-guided_gradient_alignment_for_incomplete_multimodal_learning.md)
- [\[NeurIPS 2025\] VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning](vt-fsl_bridging_vision_and_text_with_llms_for_few-shot_learning.md)
- [\[NeurIPS 2025\] AntiGrounding: Lifting Robotic Actions into VLM Representation Space for Decision Making](antigrounding_lifting_robotic_actions_into_vlm_representatio.md)
- [\[ICML 2026\] SLQ: Bridging Modalities via Shared Latent Queries for Retrieval with Frozen MLLMs](../../ICML2026/multimodal_vlm/slq_bridging_modalities_via_shared_latent_queries_for_retrieval_with_frozen_mllm.md)
- [\[NeurIPS 2025\] Multimodal Negative Learning](multimodal_negative_learning.md)

</div>

<!-- RELATED:END -->
