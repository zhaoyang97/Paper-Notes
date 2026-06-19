---
title: >-
  [论文解读] GAPrompt: Geometry-Aware Point Cloud Prompt for 3D Vision Model
description: >-
  [ICML 2025][3D视觉][参数高效微调] 提出 GAPrompt，针对预训练 3D 视觉模型的几何感知 PEFT 方法，通过可学习点云提示 (Point Prompt)、点偏移提示器 (Point Shift Prompter) 和提示传播 (Prompt Propagation) 三个模块协同利用点云几何信息，仅训练 2.19% 参数即可匹配甚至超越全量微调。
tags:
  - "ICML 2025"
  - "3D视觉"
  - "参数高效微调"
  - "点云"
  - "几何感知"
  - "提示学习"
  - "Transformer"
---

# GAPrompt: Geometry-Aware Point Cloud Prompt for 3D Vision Model

**会议**: ICML 2025  
**arXiv**: [2505.04119](https://arxiv.org/abs/2505.04119)  
**代码**: [GitHub](https://github.com/zhoujiahuan1991/ICML2025-GAPrompt)  
**领域**: 3D Vision  
**关键词**: 参数高效微调, 点云, 几何感知, 提示学习, 3D Transformer

## 一句话总结

提出 GAPrompt，针对预训练 3D 视觉模型的几何感知 PEFT 方法，通过可学习点云提示 (Point Prompt)、点偏移提示器 (Point Shift Prompter) 和提示传播 (Prompt Propagation) 三个模块协同利用点云几何信息，仅训练 2.19% 参数即可匹配甚至超越全量微调。

## 研究背景与动机

**领域现状**：预训练 3D 视觉模型（Point-MAE、Point-BERT、ReCon、Point-FEMAE 等）在点云理解任务上表现强劲，但全量微调成本高昂且存在灾难性遗忘风险。参数高效微调 (PEFT) 方法在 NLP 和 2D 视觉中已取得成功（VPT、Adapter Tuning、LoRA 等），但直接迁移到 3D 点云效果欠佳。

**现有痛点**：
1. 2D PEFT 方法依赖随机初始化的 token prompt，但点云数据的稀疏性和不规则性使这些 prompt 难以对齐，收敛困难
2. 现有 3D PEFT 方法如 IDPT（用 EdgeConv 动态生成 prompt）计算开销大，DAPT（动态 adapter）和 Point-PEFT 都主要在 token 特征层面操作，无法捕获点云固有的几何信息

**核心矛盾**：点云的核心信息在于其空间几何结构（形状、空间分布），但现有 PEFT 方法均在编码后的 token 空间操作，丢失了原始的几何信号。

**本文目标** 如何在参数高效微调框架中显式且有效地利用点云的几何信息，以弥合 PEFT 与全量微调之间的性能差距。

**切入角度**：同时在输入空间（点云层面）和特征空间（token 层面）注入几何感知信号。

**核心 idea**：用可学习点云和实例自适应形状特征从输入空间到特征空间全链路增强几何感知能力。

## 方法详解

### 整体框架

GAPrompt 冻结预训练骨干，引入三个轻量化模块：
1. **Point Prompt**：可学习点云直接拼接到输入空间
2. **Point Shift Prompter**：从原始点云提取全局形状特征并生成实例级点偏移
3. **Prompt Propagation**：将形状信息注入 Transformer 的特征提取过程

流程：原始点云 $\mathbf{x}$ → Point Shift Prompter 生成偏移后的 $\tilde{\mathbf{x}}$ 和形状特征 $\mathbf{f}$ → $[\tilde{\mathbf{x}}; \mathcal{P}]$ 编码为 input tokens → 形状特征增强 prompt tokens → Prompt Propagation 注入 → 冻结的 Transformer blocks → 分类头

### 关键设计

1. **Point Prompt（点云提示）**:
    - 功能：作为可学习的辅助点云与原始数据共同编码，引导模型关注细粒度几何细节
    - 核心思路：初始化 $P$ 个可学习3D点 $\mathcal{P} \in \mathbb{R}^{P \times 3}$（均匀分布 $z \sim U(-r, +r)$），拼接到原始点云形成 $[\tilde{\mathbf{x}}; \mathcal{P}] \in \mathbb{R}^{(S+P) \times 3}$，一起通过 token embedding 编码。训练中这些点会自动移动到几何信息丰富的区域
    - 设计动机：与 token-level prompt 相比，在点云输入空间操作更自然地保留了 3D 结构信息

2. **Point Shift Prompter（点偏移提示器）**:
    - 功能：提取全局形状特征并生成每个实例独有的点坐标偏移
    - 核心思路：
        - **层次化下采样**：参考 PointNet++，通过多分辨率 FPS+KNN 对原始点云层次化聚合：$\mathbf{x}_{j+1} = \text{FPS}(\mathbf{x}_j)$，$\mathbf{n}_j = \text{KNN}(\mathbf{x}_j, \mathbf{x}_{j+1})$
        - **形状特征提取**：轻量 PointNet 编码各层级特征 $\tilde{\mathbf{d}}_j = \text{PointNet}(\mathbf{x}_j)$，最终 reshape 为全局形状向量 $\mathbf{f} = \text{Reshape}(\tilde{\mathbf{d}}_k) \in \mathbb{R}^D$
        - **点偏移生成**：通过上采样传播特征回原始点，经 Shift Head 生成偏移 $\tilde{\mathbf{x}} = \text{Shift-Head}([\tilde{\mathbf{d}}_1^n, \tilde{\mathbf{d}}_1])$
        - **特征增强**：$\mathbf{f}$ 用于增强 prompt tokens $\mathbf{p}_i = \mathbf{p}'_i + \mathbf{f} \cdot \beta_p$ 和 adapter $\mathbf{h}_{i+1} = \hat{\mathbf{h}}_i + \text{Adapter}(\hat{\mathbf{h}}_i + \mathbf{f} \cdot \beta_a)$
    - 设计动机：不同实例几何结构差异大，固定 prompt 无法适应，需要实例自适应调整

3. **Prompt Propagation（提示传播）**:
    - 功能：将增强后的 prompt tokens 信息主动注入 Transformer 中间特征
    - 核心思路：在每个 Transformer block 中，对 input tokens 做 FPS+KNN 找局部邻域，将 prompt tokens 随机注入 center/neighbor 位置（Prompt Injection），再通过 PointNet++ 式特征传播扩散到所有 tokens：$\tilde{\mathbf{h}}_i = \text{Propagate}(\text{Inject}(\mathbf{h}_i^c, \mathbf{h}_i^n, \mathbf{p}_i))$。注入采用 Permutation 方式，引入类似 dropout 的随机性
    - 设计动机：仅通过 attention 被动扩散 prompt 信息效果有限，主动传播确保几何信息深入渗透每层特征。该机制无参数增加

### 损失函数 / 训练策略

- 标准分类交叉熵损失
- 冻结骨干，仅训练新增模块 + 分类头
- 超参数：$\beta_a = 0.5$，$\beta_p = 0.5$，$P = 20$（ScanObjectNN），AdamW，lr=5e-4，cosine schedule，400 epochs，单卡 RTX 4090

## 实验关键数据

### 主实验：ScanObjectNN + ModelNet40 分类

| 骨干 | 方法 | 参数量(M) | OBJ_BG | OBJ_ONLY | PB_T50_RS | ModelNet |
|------|------|----------|--------|----------|-----------|---------|
| Point-MAE | Full FT | 22.1 (100%) | 90.02 | 88.29 | 85.18 | 93.2 |
| Point-MAE | +IDPT | 1.7 (7.69%) | 91.22 | 90.02 | 84.94 | 93.3 |
| Point-MAE | +DAPT | 1.1 (4.97%) | 90.88 | 90.19 | 85.08 | 93.5 |
| Point-MAE | +Point-PEFT | 0.7 (3.17%) | 89.33 | 88.98 | 84.42 | 94.2 |
| Point-MAE | **+GAPrompt** | **0.6 (2.71%)** | **91.91** | **90.19** | **85.57** | **94.2** |
| Point-FEMAE | Full FT | 27.4 (100%) | 95.18 | 93.29 | 90.22 | 94.0 |
| Point-FEMAE | +IDPT | 1.7 (6.20%) | 92.94 | 90.88 | 88.38 | 93.4 |
| Point-FEMAE | +DAPT | 1.1 (4.01%) | 93.98 | 92.25 | 88.51 | 93.2 |
| Point-FEMAE | +Point-PEFT | 0.7 (2.55%) | 94.32 | 92.94 | 89.35 | 94.3 |
| Point-FEMAE | **+GAPrompt** | **0.6 (2.19%)** | **95.53** | **93.63** | **90.67** | **94.5** |
| PointGPT-L | Full FT | 360.5 (100%) | 97.20 | 96.60 | 93.40 | 94.1 |
| PointGPT-L | **+GAPrompt** | **2.0 (0.55%)** | **98.97** | **96.73** | **94.31** | **96.2** |

### 与 NLP/2D PEFT 方法对比（PB_T50_RS，Point-MAE）

| 方法 | 参数量(M) | Accuracy |
|------|----------|----------|
| Full FT | 22.1 | 85.18 |
| Linear Probing | 0.3 | 75.99 |
| VPT | 0.4 | 81.09 |
| Adapter Tuning | 0.9 | 83.93 |
| LoRA | 0.9 | 81.74 |
| SSF | 0.4 | 82.58 |
| **GAPrompt** | **0.6** | **85.57** |

### 消融实验（PB_T50_RS，Point-FEMAE）

| Point Prompt | PS-Prompter | Prompt Propagation | Acc. |
|:---:|:---:|:---:|:---:|
| ✓ | - | - | 87.85 |
| ✓ | ✓ | - | 89.34 (+1.49) |
| ✓ | ✓ | ✓ | **90.67** (+1.33) |

| Shift Head | Prompt增强 | Adapter增强 | Acc. |
|:---:|:---:|:---:|:---:|
| ✓ | - | - | 88.23 |
| ✓ | ✓ | - | 89.71 |
| ✓ | ✓ | ✓ | **90.67** |

### 关键发现

- GAPrompt 在所有四个骨干上均是 PEFT 最佳，在 Point-MAE 和 PointGPT-L 上超越全量微调
- 基于 PointGPT-L 达 ModelNet40 96.2%（仅 0.55% 参数），刷新 SOTA
- Point Shift Prompter 贡献最大（+1.49%），是几何信息核心来源
- 可视化：偏移后点云边界更清晰、更紧凑；[CLS] 注意力精确聚焦到物体关键部位
- 训练后 Point Prompt 自动移入点云内部空间

## 亮点与洞察

1. **几何感知是 3D PEFT 的核心差异化因素**：Token 级操作无法触及点云核心——空间几何结构
2. **实例自适应设计**：每个点云有独特的形状特征和偏移，这对形状差异大的分类至关重要
3. **极致参数效率**：0.6M 参数（2.19%）超越 27.4M 全量微调，FLOPs 基本不增加
4. **Prompt Propagation 零参数**：利用空间距离做特征插值，无额外参数但贡献显著

## 局限与展望

- 仅验证分类任务，3D 检测/分割等复杂任务未涉及
- Point Shift Prompter 使用简单 PointNet，更强编码器（KPConv 等）可能进一步提升
- 超参数（$\beta_a$、$\beta_p$、$P$）对不同数据集需分别调优
- 对不同预训练策略的适配性差异未分析

## 相关工作与启发

- **VPT (Jia et al., 2022)**：2D Prompt Tuning，GAPrompt 是其 3D 几何感知版本
- **IDPT (Zha et al., 2023)**：EdgeConv 动态 prompt，计算开销大（FLOPs 7.2G vs GAPrompt 5.0G）
- **DAPT (Zhou et al., 2024)**：动态 adapter，仍在 token 空间
- 启发：几何感知 prompt 方法可推广到 3D 检测/分割，Point Shift 思想也可用于 3D 数据增强

## 评分

- 新颖性: ⭐⭐⭐⭐ 在输入空间引入可学习点云 + 实例自适应偏移是独特的 3D PEFT 思路
- 实验充分度: ⭐⭐⭐⭐ 四个骨干、详尽消融和对比，但缺少检测/分割任务
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，可视化有说服力
- 价值: ⭐⭐⭐⭐ 为 3D PEFT 提供了明确的"几何感知"范式，代码开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Egocentric Action-aware Inertial Localization in Point Clouds with Vision-Language Guidance](../../ICCV2025/3d_vision/egocentric_action-aware_inertial_localization_in_point_clouds_with_vision-langua.md)
- [\[ICCV 2025\] UPP: Unified Point-Level Prompting for Robust Point Cloud Analysis](../../ICCV2025/3d_vision/upp_unified_point-level_prompting_for_robust_point_cloud_analysis.md)
- [\[AAAI 2026\] DANCE: Density-Agnostic and Class-Aware Network for Point Cloud Completion](../../AAAI2026/3d_vision/dance_density-agnostic_and_class-aware_network_for_point_cloud_completion.md)
- [\[CVPR 2026\] AnyPcc: Compressing Any Point Cloud with a Single Universal Model](../../CVPR2026/3d_vision/anypcc_compressing_any_point_cloud_with_a_single_universal_model.md)
- [\[ICCV 2025\] CstNet: Constraint-Aware Feature Learning for Parametric Point Cloud](../../ICCV2025/3d_vision/constraint-aware_feature_learning_for_parametric_point_cloud.md)

</div>

<!-- RELATED:END -->
