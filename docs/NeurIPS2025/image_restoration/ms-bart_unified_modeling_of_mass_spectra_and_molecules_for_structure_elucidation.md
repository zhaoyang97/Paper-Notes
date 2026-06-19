---
title: >-
  [论文解读] MS-BART: Unified Modeling of Mass Spectra and Molecules for Structure Elucidation
description: >-
  [NeurIPS 2025][图像恢复][质谱分析] 提出 MS-Bart，通过统一词表将分子指纹和分子结构（SELFIES）映射到共享的 token 空间，在 400 万指纹-分子对上进行多任务预训练，再通过实验谱微调和化学反馈对齐，实现从质谱到分子结构的高效生成。 从质谱（MS）数据推断分子结构是分析化学的核心任务…
tags:
  - "NeurIPS 2025"
  - "图像恢复"
  - "质谱分析"
  - "分子指纹"
  - "预训练-微调-对齐"
  - "BART"
  - "结构解析"
---

# MS-BART: Unified Modeling of Mass Spectra and Molecules for Structure Elucidation

**会议**: NeurIPS 2025  
**arXiv**: [2510.20615](https://arxiv.org/abs/2510.20615)  
**代码**: [有](https://github.com/OpenDFM/MS-BART)  
**领域**: 科学AI / 分子结构解析  
**关键词**: 质谱分析, 分子指纹, 预训练-微调-对齐, BART, 结构解析

## 一句话总结

提出 MS-Bart，通过统一词表将分子指纹和分子结构（SELFIES）映射到共享的 token 空间，在 400 万指纹-分子对上进行多任务预训练，再通过实验谱微调和化学反馈对齐，实现从质谱到分子结构的高效生成。

## 研究背景与动机

从质谱（MS）数据推断分子结构是分析化学的核心任务，广泛应用于药物发现、环境生化和材料科学。现有方法面临两大瓶颈：

**标注谱数据稀缺**：高质量的实验质谱-分子配对数据获取成本高昂，限制了数据驱动模型的训练。

**质谱信号复杂多变**：同一分子在不同碰撞能量、加合物类型或仪器设置下的谱图差异巨大（如 Figure 1 所示），甚至相同条件下也有波动。这使得直接在原始谱上预训练非常困难。

**关键洞察**：分子指纹（fingerprint）是对质谱信息的抽象编码，表示化学子结构的存在性。与原始谱不同，指纹对实验条件不变，且可通过 RDKit 从分子结构可靠计算。这为构建大规模预训练数据提供了途径。

现有方法（如 Spec2Mol、MSNovelist、DiffMS）通常将谱和分子结构视为独立模态，容易产生**分子幻觉**（molecular hallucination）——生成化学有效但偏离真实结构的分子。

## 方法详解

### 整体框架

MS-Bart 遵循 NLP 中的 **预训练→微调→对齐** 范式：
1. **预训练**：在 400 万计算指纹-分子对上进行多任务学习
2. **微调**：在实验质谱数据上适应真实分布
3. **对齐**：通过化学反馈引导模型减少分子幻觉

### 关键设计

1. **统一词表（Unified Vocabulary）**

   **功能**：将质谱（以指纹表示）和分子结构统一到同一 token 空间。

   **核心思路**：
    - **指纹 token**：4096 位 Morgan 指纹中，每个激活位 $FP_i = 1$ 转换为 token `<fp{i:04d}>`（如 `<fp0123>`）
    - **分子 token**：使用 185 个 SELFIES token 编码分子结构，保证化学有效性
    - 特殊分隔 token `<fps_sep>` 连接两种模态

   **设计动机**：统一词表使得模型可以在同一序列空间中同时学习谱表示和分子表示，实现真正的跨模态学习。

2. **多任务预训练**

   **功能**：通过 4 种自监督/跨模态任务学习指纹和分子的表示。

   **核心思路**：
    - **SELFIES 去噪**：随机掩码 30% 的 SELFIES token 并恢复
    - **指纹→分子翻译**：从指纹 token 生成完整 SELFIES
    - **混合去噪**：将指纹和被掩码的 SELFIES 以不同顺序拼接，预测完整 SELFIES

   所有任务统一使用交叉熵损失：$\mathcal{L}_{ce} = -\sum_i \log P(y_i | y_{<i}, X; \theta)$

   **设计动机**：单独的去噪任务与结构解析对齐不佳，翻译任务提供核心跨模态能力，混合任务将两者优势结合。

3. **化学反馈对齐（Contrastive Alignment）**

   **功能**：通过排序损失引导模型为结构越相似的分子赋予越高概率。

   **核心思路**：给定指纹，生成 $n$ 个候选分子，以 Tanimoto 相似度作为偏好评分。引入对比排序损失：

    $\mathcal{L}_{rank}(C) = \sum_i \sum_{j>i} \max(0, P_\theta(S_j) - P_\theta(S_i) + \gamma_{ij})$

   其中 $\gamma_{ij} = (j-i) \times \gamma$ 为间隔。总损失 $\mathcal{L} = \mathcal{L}_{ce} + \alpha \mathcal{L}_{rank}$。

   **设计动机**：纯交叉熵训练的模型可能为所有化学有效但不一定正确的分子分配相似概率，排序损失强制模型学会区分"更接近正确"的分子。

### 损失函数 / 训练策略

- 骨干网络：BART-Base，所有参数从头初始化
- 预训练：最大序列长度 512，400 万无标注分子
- 微调/对齐：输入输出 token 长度 256
- 对齐阶段冻结编码器，仅更新解码器
- MIST 模型将实验谱转换为预测指纹，阈值 $\epsilon = 0.2$（NPLIB1）/ $0.11$（MassSpecGym）

## 实验关键数据

### 主实验

在 NPLIB1 和 MassSpecGym 两个公开基准上与多种基线对比。

| 数据集 | 方法 | Top-1 Acc | Top-1 MCES↓ | Top-1 Tanimoto↑ | Top-10 Acc | Top-10 MCES↓ | Top-10 Tanimoto↑ |
|--------|------|-----------|-------------|-----------------|------------|--------------|------------------|
| NPLIB1 | DiffMS | 8.34% | 11.95 | 0.35 | 15.44% | 9.23 | 0.47 |
| NPLIB1 | **MS-Bart** | 7.45% | **9.66** | **0.44** | 10.99% | **8.31** | **0.51** |
| NPLIB1 | MS-Bart (Gold FP) | 73.50% | 2.14 | 0.90 | 79.12% | 1.60 | 0.94 |
| MassSpecGym | DiffMS | **2.30%** | 18.45 | **0.28** | **4.25%** | 14.73 | **0.39** |
| MassSpecGym | **MS-Bart** | 1.07% | **16.47** | 0.23 | 1.11% | **15.12** | 0.28 |

MS-Bart 在 NPLIB1 上相似度指标全面领先：MCES 改善 19.16%，Tanimoto 改善 25.71%。

### 消融实验

| 预训练策略 | Top-1 Acc | Top-1 MCES↓ | Top-1 Tanimoto↑ |
|-----------|-----------|-------------|-----------------|
| 无预训练 | 1.71% | 12.93 | 0.27 |
| 仅去噪 (Sd) | 0.37% | 14.41 | 0.24 |
| 仅翻译 (Trans) | 6.23% | 9.37 | 0.42 |
| 混合去噪 (Hybrid) | 5.13% | 9.96 | 0.41 |
| **完整 MS-Bart** | **7.45%** | **9.66** | **0.44** |

### 关键发现

- **预训练至关重要**：无预训练性能远低于完整模型，但仍优于直接编码原始谱的基线方法
- **多任务互补**：单独去噪反而降低性能（与结构解析不对齐），翻译任务贡献最大，两者结合效果最佳
- **化学反馈有效**：从 Pretrain → Pretrain-FT → Pretrain-FT-Rank，Top-1 准确率从 0%→1.07%→最终值，Tanimoto 相似度持续提升
- **Gold Fingerprint 实验**揭示了巨大潜力：使用真实指纹作为输入时准确率达 73.5%，说明瓶颈在 MIST 的指纹预测质量
- 推理速度比 DiffMS（扩散模型）快一个数量级

## 亮点与洞察

- **统一词表设计**精妙：将质谱的离散化表示（指纹 token）和分子的字符串表示（SELFIES token）融入同一序列空间
- **指纹作为桥梁**的思路很聪明：避免了直接模拟不同实验条件下质谱的复杂性
- 采用 NLP 领域成熟的"预训练→微调→对齐"范式，方法论迁移自然
- Gold Fingerprint 实验明确了未来改进方向：提升谱→指纹的预测模型

## 局限与展望

- 当前瓶颈在于 MIST 指纹预测的质量，MS-Bart 本身的分子生成能力已经很强
- 在 MassSpecGym 上未过滤 [M+Na]+ 测试数据导致部分性能下降
- 预训练数据中过滤了与测试集 MCES 距离 <2 的分子（比 DiffMS 更严格），导致精确匹配指标偏低
- SELFIES 词表仅 185 个 token，可能限制了复杂分子的表达能力

## 相关工作与启发

- 与 NMR 领域的预训练→微调范式类似（Yao et al. 2023），但针对质谱的特殊挑战提出了指纹抽象
- DiffMS 使用隐式指纹表示 + 离散扩散模型，方法路线不同但互补
- CSI:FingerID → MSNovelist 的流水线思路与 MS-Bart（MIST→统一模型）相似，但 MS-Bart 通过预训练获得了更强的分子生成先验

## 评分

- 新颖性: ⭐⭐⭐⭐ — 统一词表+多任务预训练+化学反馈对齐的三阶段设计新颖
- 实验充分度: ⭐⭐⭐⭐ — 两个基准、多种消融、Gold FP 分析有说服力
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，NLP→化学的范式迁移阐述得当
- 价值: ⭐⭐⭐⭐ — 为质谱结构解析提供了可扩展的预训练范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Self-supervised Dynamic Heterogeneous Degradation Modeling for Unified Zero-Shot Image Restoration](../../CVPR2026/image_restoration/self-supervised_dynamic_heterogeneous_degradation_modeling_for_unified_zero-shot.md)
- [\[NeurIPS 2025\] Latent Harmony: Synergistic Unified UHD Image Restoration via Latent Space Regularization and Controllable Refinement](latent_harmony_synergistic_unified_uhd_image_restoration_via_latent_space_regula.md)
- [\[ICCV 2025\] UniPhys: Unified Planner and Controller with Diffusion for Flexible Physics-Based Character Control](../../ICCV2025/image_restoration/uniphys_unified_planner_and_controller_with_diffusion_for_flexible_physics-based.md)
- [\[ICLR 2026\] SoFlow: Solution Flow Models for One-Step Generative Modeling](../../ICLR2026/image_restoration/soflow_solution_flow_models_for_one-step_generative_modeling.md)
- [\[CVPR 2026\] DVAR: Dynamic Visual Autoregressive Modeling for Image Super-Resolution](../../CVPR2026/image_restoration/dvar_dynamic_visual_autoregressive_modeling_for_image_super-resolution.md)

</div>

<!-- RELATED:END -->
