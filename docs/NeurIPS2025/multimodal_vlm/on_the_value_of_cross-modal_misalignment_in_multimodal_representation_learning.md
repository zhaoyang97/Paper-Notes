---
title: >-
  [论文解读] On the Value of Cross-Modal Misalignment in Multimodal Representation Learning
description: >-
  [NeurIPS 2025][多模态VLM][多模态对比学习] 提出潜变量模型将跨模态失配形式化为选择偏差和扰动偏差两种机制，理论证明MMCL学到的表征恰好捕获与两种偏差无关的不变语义子集，统一了"失配有害/有益"两种对立观点。 1. 领域现状 多模态对比学习（MMCL）如 CLIP 通过图像-文本对齐学习联合表征…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "多模态对比学习"
  - "跨模态失配"
  - "潜变量模型"
  - "可辨识性"
  - "不变表征学习"
---

# On the Value of Cross-Modal Misalignment in Multimodal Representation Learning

**会议**: NeurIPS 2025  
**arXiv**: [2504.10143](https://arxiv.org/abs/2504.10143)  
**代码**: [项目主页](https://yichaocai.com/misalignment.github.io)  
**领域**: 多模态VLM  
**关键词**: 多模态对比学习, 跨模态失配, 潜变量模型, 可辨识性, 不变表征学习

## 一句话总结

提出潜变量模型将跨模态失配形式化为选择偏差和扰动偏差两种机制，理论证明MMCL学到的表征恰好捕获与两种偏差无关的不变语义子集，统一了"失配有害/有益"两种对立观点。

## 研究背景与动机

### 1. 领域现状

多模态对比学习（MMCL）如 CLIP 通过图像-文本对齐学习联合表征，在零样本分类、跨模态检索等任务取得巨大成功。其核心假设是配对的图像-文本语义完全一致。

### 2. 现有痛点

真实世界数据集广泛存在**跨模态失配**（cross-modal misalignment）。研究显示大规模视频-文本数据集中超过 50% 的配对存在语义不一致。文本作为图像的描述天然容易在语义上不完整或引入错误信息。

### 3. 核心矛盾

学术界对失配有两种对立看法：
- **缓解派**：失配是噪声干扰，导致多模态模型"幻觉"，应被消除（如 SigLIP, BLIP 等过滤策略）
- **利用派**：在风格相关信息上刻意引入失配（如随机文本增强），反而能增强零样本和对抗鲁棒性

### 4. 本文目标

如何从理论上调和这两种对立观点？什么条件下失配有害、什么条件下有益？给实际应用提供可操作的指导。

### 5. 切入角度

构建一个显式建模跨模态失配的潜变量生成模型（LVM），对 MMCL 框架进行可辨识性（identifiability）分析，从因果推断视角给出统一理论。

### 6. 核心 idea

MMCL 学到的表征恰好是图文共享且不受选择偏差和扰动偏差影响的语义子集的块可辨识变换，失配自然过滤掉不稳定语义从而起到正则化作用。

## 方法详解

### 整体框架

提出包含三类潜变量的生成模型 $\mathcal{Z} = \mathcal{S} \times \mathcal{M}_x \times \mathcal{M}_t$：
- **语义变量** $\mathbf{s} \in \mathbb{R}^{n_s}$：描述图文共有的语义内容（物体形状、颜色等）
- **图像特有变量** $\mathbf{m}_x$：相机噪声、背景等非语义因素
- **文本特有变量** $\mathbf{m}_t$：写作风格、语气等非语义因素

关键创新：允许语义变量 $\mathbf{s}$ 之间存在**任意因果结构**，不像先前工作要求独立或固定图结构。

### 关键设计

#### 模块1：选择偏差（Selection Bias）$\theta$

**功能**：决定哪些语义信息被保留在文本中。

**核心思路**：选择偏差 $\theta$ 映射到非空语义子集 $\mathbb{I}_\theta \in \mathcal{P}^+(\mathbb{I}_\mathbf{s})$，其补集 $\mathbb{I}_\theta^c$ 中的语义变量在文本中被完全忽略（如描述物体时省略纹理信息）。

**设计动机**：真实文本描述天然只覆盖图像的部分语义，选择偏差精确刻画了这种信息丢失。

#### 模块2：扰动偏差（Perturbation Bias）$\rho$

**功能**：在被选择的语义子集上引入错误标注。

**核心思路**：定义可扰动子集 $\mathbb{I}_\rho \subseteq \mathbb{I}_\theta$，随机抽取 $A \subseteq \mathbb{I}_\rho$，将其中的语义变量替换为随机值：
$$p_{\tilde{\mathbf{s}}_{\mathbb{I}_\theta} | \mathbf{s}, A} = \delta(\tilde{\mathbf{s}}_{\mathbb{I}_\theta \setminus A} - \mathbf{s}_{\mathbb{I}_\theta \setminus A}) \cdot p_{\tilde{\mathbf{s}}_A | \mathbf{s}_A}$$

**设计动机**：模拟标注过程中的错误（如将"黑色"误标为"红色"）。关键洞察是此类扰动不像因果干预那样传播到下游变量，因为它作用于观测层而非潜变量层。

#### 模块3：生成过程

图像生成：$\mathbf{x} = g_x(\mathbf{s}, \mathbf{m}_x)$，其中 $g_x$ 是微分同胚映射。
文本生成：$\mathbf{t}^{(\theta)} = g_{t^{(\theta)}}(\tilde{\mathbf{s}}_{\mathbb{I}_\theta}, \mathbf{m}_t)$，其中 $g_{t^{(\theta)}}$ 同样是微分同胚。

### 核心理论结果

**定理 4.1（失配下语义变量的可辨识性）**：在温和假设下，最小化 $\mathcal{L}_{\text{SymAlignMaxEnt}}$ 目标，编码器 $f_x, f_t$ 将**块辨识**（block-identify）未受偏差影响的语义子变量 $\mathbf{s}_{\mathbb{I}_\rho^c}$。换言之，学到的表征恰好是 $\mathbf{s}_{\mathbb{I}_\rho^c}$ 的可逆变换，维度 $n = |\mathbb{I}_\theta| - |\mathbb{I}_\rho|$。

**推论 4.1**（完美对齐 → 全语义恢复）：当 $\theta = 2^{n_s} - 1$ 且 $\rho = 1$（无偏差），MMCL 恢复全部 $n_s$ 个语义变量。

**推论 4.2**（定向失配 → 不变表征）：当 $\mathbb{I}_{var} = \mathbb{I}_\theta^c \cup \mathbb{I}_\rho$ 恰好等于对分布偏移敏感的语义子集时，MMCL 自动恢复对分布偏移鲁棒的不变语义 $\mathbf{s}_{\mathbb{I}_{inv}}$。

### 损失函数/训练策略

采用 MMCL 的渐近目标函数进行理论分析：
$$\mathcal{L}_{\text{SymAlignMaxEnt}} = \mathbb{E}[\|f_x(\mathbf{x}) - f_t(\mathbf{t})\|_2] - \frac{1}{2}(H(f_x(\mathbf{x})) + H(f_t(\mathbf{t})))$$
第一项最小化配对样本距离（对齐），第二项最大化表征熵（均匀性）。

## 实验关键数据

### 主实验

**数值模拟**（10维语义变量 + 5维模态特有变量）：

| 设置 | 保留语义 $|\mathbb{I}_\theta|$ | 未偏差语义 R² | 偏差语义 R² | 模态特有 R² |
|------|------|------|------|------|
| 独立潜变量 | 变化 1→10 | ≈1.0 | ≈0.0 | ≈0.0 |
| 依赖潜变量 | 变化 1→10 | ≈1.0 | 部分可预测 | ≈0.0 |

**MPI3D-Complex 真实数据集**（7个独立离散因子）：

| Bias 类型 | 设置 | 保留因子 MCC | 缺失因子 MCC |
|------|------|------|------|
| Selection | 1因子→5因子 | ≥0.8 → ≈1.0 | =0.0 |
| Perturbation | 0→4因子扰动 | ≈1.0 | =0.0 |

**Causal3DIdent**（含因果图结构的10维潜变量）：

| 设置 | shape (MCC) | x_pos (MCC) | color (R²) | s_pos (R²) |
|------|------|------|------|------|
| 全选+无扰动 | ≈1.0 | ≈1.0 | ≈1.0 | ≈1.0 |
| 仅 shape | ≈1.0 | =0 | =0 | =0 |

### 消融实验

**下游任务（数值模拟）**：

| 编码维度 | ID回归 R² | ID分类 Acc | OOD分类 Acc |
|------|------|------|------|
| 保留全部10维语义 | ≈1.0 | ≈1.0 | 下降 |
| 去除分布敏感维度 | 下降 | 下降 | **最佳** |

### OpenCLIP 案例研究

对 LAION-400M 预训练的 OpenCLIP 进行 146 个视觉概念的零样本评估：

| 概念组 | 覆盖率 | F1 Score |
|------|------|------|
| Animal | 高 (>1%) | 高 |
| Object | 高 (1.63%) | 高 |
| Texture | 低 (0.07%) | 低 |
| Emotion | 低 (0.04%) | 低 |
| Stereotype | 极低 (0.0003%) | 极低 |

### 关键发现

1. 理论预测与实验高度一致：未偏差语义 $R^2 \approx 1$，偏差语义 $R^2 \approx 0$
2. 在有因果依赖的潜变量场景下，部分偏差语义因统计依赖可部分预测，但程度有限
3. 模态特有变量在所有设置中一致被排除
4. 定向失配可增强 OOD 鲁棒性，验证了推论 4.2
5. OpenCLIP 在低覆盖概念上系统性失败，验证了选择偏差理论

## 亮点与洞察

1. **统一视角**：首次从可辨识性理论统一"失配有害 vs 有益"两种对立观点，给出清晰的条件划分
2. **灵活的潜变量模型**：允许语义变量间任意因果结构，比先前要求独立或固定结构的工作更通用
3. **实用洞察**：
    - 大规模预训练需要详尽、一致的标注以保留全部语义
    - OOD 场景下可通过控制文本失配来自然实现不变表征学习
    - 审阅和策展文本比操作潜变量更精确可控
4. **扰动非干预**：区分了观测层面的文本扰动与因果干预，后者会沿因果图传播而前者不会

## 局限与展望

1. 理论基于渐近 MMCL 目标（SymAlignMaxEnt），与实际有限样本 InfoNCE 之间存在差距
2. 假设生成函数为微分同胚，真实图像/文本生成可能不严格满足
3. 仅分析块可辨识性，未给出组件级别的精确辨识条件
4. OpenCLIP 实验中概念覆盖率与 F1 的因果关系需更严格验证
5. 实际应用中如何精确控制文本的选择/扰动偏差仍需探索

## 相关工作与启发

- **与 Daunhawer et al. (ICLR 2023) 的关系**：在完美对齐假设下证明 MMCL 可辨识语义变量，本文将其推广到失配场景
- **与不变表征学习的联系**：推论 4.2 表明 MMCL + 定向失配是实现 IRM/REx 等不变学习的新途径
- **与文本增强的联系**：CLAP 等通过随机文本增强提升鲁棒性的工作可被本文理论解释为引入了扰动偏差

## 评分

⭐⭐⭐⭐⭐ (5/5)

理论贡献扎实，统一了跨模态失配的两种对立观点，实验验证从合成数据到真实 CLIP 模型，兼具理论深度和实际指导意义。是多模态表征学习理论分析的重要工作。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MIDAS: Misalignment-based Data Augmentation Strategy for Imbalanced Multimodal Learning](midas_misalignment-based_data_augmentation_strategy_for_imbalanced_multimodal_le.md)
- [\[NeurIPS 2025\] Structure-Aware Fusion with Progressive Injection for Multimodal Molecular Representation Learning](structure-aware_fusion_with_progressive_injection_for_multimodal_molecular_repre.md)
- [\[NeurIPS 2025\] Guiding Cross-Modal Representations with MLLM Priors via Preference Alignment](guiding_cross-modal_representations_with_mllm_priors_via_preference_alignment.md)
- [\[NeurIPS 2025\] Continual Multimodal Contrastive Learning](continual_multimodal_contrastive_learning.md)
- [\[ICLR 2026\] SpectralGCD: Spectral Concept Selection and Cross-modal Representation Learning for Generalized Category Discovery](../../ICLR2026/multimodal_vlm/spectralgcd_spectral_concept_selection_and_cross-modal_representation_learning_f.md)

</div>

<!-- RELATED:END -->
