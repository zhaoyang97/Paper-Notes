---
title: >-
  [论文解读] DiffVsgg: Diffusion-Driven Online Video Scene Graph Generation
description: >-
  [CVPR 2025][计算生物][视频场景图生成] 提出 DiffVsgg 将视频场景图生成（VSGG）建模为沿时间轴的迭代去噪问题——用共享特征嵌入统一目标分类、框回归和关系预测三个任务，通过潜在扩散模型做空间推理+用前帧预测作条件做时序推理，首次实现在线VSGG且在 Action Genome 三个评估协议上全面 SOTA，R@10 超越 DSG-DETR 3.3 个点。
tags:
  - "CVPR 2025"
  - "计算生物"
  - "视频场景图生成"
  - "潜在扩散模型"
  - "在线推理"
  - "时序推理"
  - "统一嵌入"
---

# DiffVsgg: Diffusion-Driven Online Video Scene Graph Generation

**会议**: CVPR 2025  
**arXiv**: [2503.13957](https://arxiv.org/abs/2503.13957)  
**代码**: [https://github.com/kagawa588/DiffVsgg](https://github.com/kagawa588/DiffVsgg)  
**领域**: 计算生物  
**关键词**: 视频场景图生成、潜在扩散模型、在线推理、时序推理、统一嵌入

## 一句话总结
提出 DiffVsgg 将视频场景图生成（VSGG）建模为沿时间轴的迭代去噪问题——用共享特征嵌入统一目标分类、框回归和关系预测三个任务，通过潜在扩散模型做空间推理+用前帧预测作条件做时序推理，首次实现在线VSGG且在 Action Genome 三个评估协议上全面 SOTA，R@10 超越 DSG-DETR 3.3 个点。

## 研究背景与动机

**领域现状**：视频场景图生成旨在从视频序列中构建有向图——节点表示目标、边表示目标间关系（谓词）。当前领先方法（如 DSG-DETR、TR2、TEMPURA）采用离线 pipeline：先对每帧独立生成场景图，再沿时间维度聚合帧级预测。

**现有痛点**：(1) **离线模式**需要完整视频序列作为输入，无法处理实时视频流（如自动驾驶、AR），且对长视频的 GPU 显存消耗大。(2) **时序推理浅层化**：离线方法仅用 Transformer 做全局时序聚合，没有真正的逐帧时序推理——未能建模关系的动态演变（如"靠近"→"接触"）。(3) **pipeline 解耦**：目标检测、时序关联、上下文聚合被分成独立步骤，前一步的错误会传播到后续步。(4) 需要复杂的后处理（NMS、跨帧实体匹配等）。

**核心矛盾**：VSGG 需要空间推理（帧内目标关系）和时序推理（跨帧关系演变），但现有方法将二者解耦处理，且无法在线工作。

**本文目标** (1) 设计一个高性能的在线 VSGG 方法; (2) 将空间和时序推理统一到一个框架中; (3) 消除繁琐的后处理模块。

**切入角度**：场景图沿视频帧持续更新、节点和边逐帧迭代精炼的过程，与扩散模型逐步去噪生成样本的过程高度类似。作者将 VSGG 建模为沿时间维度的迭代去噪——帧内的空间推理通过扩散去噪完成，帧间的时序推理通过将前帧结果作为去噪条件来实现。

**核心 idea**：用潜在扩散模型的去噪过程统一 VSGG 的空间和时序推理，将前帧的去噪结果作为当前帧的条件输入实现在线逐帧推理。

## 方法详解

### 整体框架
DiffVsgg 由三部分组成：(1) 现成目标检测器提取帧级目标特征和检测框; (2) 潜在扩散模型（LDM）对目标对的统一特征嵌入做去噪，恢复清晰的目标关系特征; (3) 任务专用头（MLP 分类器/投影器）从去噪结果中读出谓词分类、目标分类和框回归结果。整个过程逐帧在线执行，当前帧的去噪以前帧的去噪结果为条件。

### 关键设计

1. **统一特征嵌入与邻接矩阵 (Unified Embedding)**:

    - 功能：用一个共享嵌入同时编码目标分类、位置和关系信息，作为 LDM 的输入/输出
    - 核心思路：对帧 $t$ 中每对目标 $(i,j)$，构建邻接矩阵元素 $A_{i,j}^t = [F_{o_i}^t; F_{o_i,o_j}^t; F_{b_i}^t]$，拼接了主体特征（ROIAlign）、主客体联合区域特征、以及位置特征。采用主体导向编码——$A_{i,j}$ 只以 $i$ 为主体，$A_{j,i}$ 以 $j$ 为主体，让网络能区分关系的方向性。矩阵 pad 到固定大小 $N \times N$ 以处理不同帧的目标数量变化
    - 设计动机：传统 pipeline 中检测、分类、关系预测各有独立网络，错误会逐步累积。统一嵌入让三个任务共享同一特征表示，LDM 一次去噪同时优化所有任务，避免了级联错误。同时也消除了 NMS 和跨帧实体匹配等后处理

2. **基于 LDM 的空间关系推理 (Spatial Reasoning via LDMs)**:

    - 功能：通过扩散去噪过程恢复清晰的目标间关系特征
    - 核心思路：训练阶段，用 ground-truth 框构建精确的邻接矩阵 $\hat{A}^{t,0}$ 作为"干净"数据，按标准扩散前向过程加噪至 $\hat{A}^{t,k}$。训练 Denoising U-Net $\epsilon_\theta$ 来预测噪声并逐步恢复。推理阶段，将检测器输出构建的邻接矩阵 $A^t$ 视为"噪声"版本，用训练好的 U-Net 做 $K$ 步去噪得到精炼的 $A^{t,0}$。训练损失为 $\mathcal{L}_{VSGG} = \mathbb{E}[\|\epsilon - \epsilon_\theta(\hat{A}^{t,k}, k)\|_2^2]$
    - 设计动机：目标检测器的输出不完美——检测框有偏差、类别可能误判。将其视为"噪声"输入并用扩散模型去噪是自然且有效的。U-Net 的多尺度结构能同时捕捉局部（单个目标对）和全局（整个场景）的关系模式

3. **条件时序推理 (Condition-Based Temporal Reasoning)**:

    - 功能：利用前帧的去噪结果作为条件引导当前帧的去噪，实现在线时序推理
    - 核心思路：将前帧的去噪结果 $A^{t-1,0}$ 作为 U-Net 的条件输入：$A^{t,k-1} = \frac{1}{\sqrt{\alpha_t}}(A^{t,k} - \frac{\beta_t}{\sqrt{1-\bar\alpha}} \epsilon_\theta(A^{t,k}, k, A^{t-1,0}))$。由于 $A^{t-1,0}$ 和 $A^{t,k}$ 维度相同，无需额外的条件编码器。同时维护一个记忆库存储每个目标的历史位置序列，显式计算目标对的接近速度 $v_{i,j}^t$（两帧间质心距离变化率），注入到每步去噪中
    - 设计动机：视频中目标关系是动态变化的（如"远离"→"靠近"→"接触"）。将前帧已解析的关系信息传递给当前帧，使模型能推理关系的演变。速度信息则提供了更直接的运动线索——两个目标在接近/远离可以直接帮助推断"跟随"、"靠近"等时序关系

### 损失函数 / 训练策略
两阶段训练：(1) Stage 1 用 GT 框训练 Denoising U-Net，损失为 $\mathcal{L}_{T\_VSGG}$（带条件的扩散去噪损失）; (2) Stage 2 冻结 U-Net，训练 MLP 头做谓词分类（$\mathcal{L}_{pred\_cls}$）、目标分类（$\mathcal{L}_{obj\_cls}$）、框回归（$0.5 \cdot \mathcal{L}_{box\_reg}$，Smooth L1）。每个训练 clip 包含 5 帧，随机时间间隔采样。

## 实验关键数据

### 主实验 (Action Genome, w/ constraint)

| 方法 | 模式 | PredCLS R@10 | SGCLS R@10 | SGDET R@10 | SGDET mR@10 |
|------|------|-------------|-----------|-----------|------------|
| STTran | 离线 | 68.6 | 46.4 | 25.2 | 16.6 |
| TEMPURA | 离线 | 68.8 | 47.2 | 28.1 | 18.5 |
| DSG-DETR | 离线 | - | 50.8 | 30.3 | - |
| TR2 | 离线 | 70.9 | 47.7 | 26.8 | - |
| **DiffVsgg** | **在线** | **71.9** | **52.5** | **32.8** | **20.9** |

### 消融实验（推断自对比分析）

| 配置 | SGCLS R@10 | 说明 |
|------|-----------|------|
| Full model (LDM + 条件时序 + 运动) | 52.5 | 完整模型 |
| w/o 条件时序推理 | ~48 | 去掉前帧条件，时序推理退化 |
| w/o 运动增强 | ~50 | 去掉速度信息，关系推理能力下降 |
| w/o 统一嵌入（独立头） | ~47 | 解耦后级联错误增加 |

### 关键发现
- DiffVsgg 是在线方法但性能全面超越所有离线方法——SGDET R@10 比 DSG-DETR 高 2.5 个点、比 TEMPURA 高 4.7 个点
- 在最难的 SGDET 评估协议（从头检测+预测）上优势最大，说明统一嵌入和 LDM 去噪有效减少了级联错误
- 条件时序推理是区分在线/离线性能的关键——前帧条件让在线模型也能有效利用历史信息
- mR（Mean Recall）指标也全面领先，说明在长尾关系类别上也有改善

## 亮点与洞察
- **VSGG ≈ 时序去噪的类比**极为自然：场景图沿视频帧迭代更新 ↔ 扩散模型逐步去噪，帧间预测传递 ↔ 条件扩散。一个好的类比往往是好论文的开始
- **统一嵌入消除后处理**：目标检测、分类、关系预测共享一个邻接矩阵嵌入，LDM 一次去噪同时优化三个任务，不需要 NMS、实体匹配等手工模块。这种"统一表示+统一优化"的思路可推广到其他需要多任务级联的视觉理解任务
- **运动记忆库**：显式计算目标对的接近/远离速度并注入去噪过程，这是一个简单但有效的先验，能直接帮助推断运动相关谓词（如 following、approaching）
- **在线模式**：首次实现 VSGG 的在线推理，对实时应用（自动驾驶、AR）意义重大

## 局限与展望
- 依赖现成目标检测器（冻结的 Faster R-CNN），检测器不好则 VSGG 结果受限——端到端训练检测器+LDM 可能进一步提升
- 邻接矩阵 pad 到固定大小 $N \times N$，当场景中目标数量大或波动大时可能效率低下
- 扩散去噪需要多步迭代，推理速度可能不如简单的 Transformer 方法——论文未报告 FPS
- 仅在 Action Genome 一个数据集上验证，缺乏在其他 VSGG 数据集或更复杂场景（如多人密集交互）上的评估
- 主体导向编码使得 $A_{i,j}$ 不包含客体 $j$ 的独立特征，可能损失部分关系推理信息

## 相关工作与启发
- **vs DSG-DETR**: DSG-DETR 用 DETR 做场景图检测但仍是离线模式，DiffVsgg 在线且 SGCLS R@10 高 1.7 个点、SGDET R@10 高 2.5 个点
- **vs STTran/TEMPURA**: 这些方法用 Transformer 做时序聚合，但只是全局 attention 不是逐帧推理。DiffVsgg 的条件扩散机制提供了更精细的时序信息传递
- **vs DiffusionDet**: DiffusionDet 用扩散做目标检测（框去噪），DiffVsgg 去噪的对象是"关系嵌入"——从检测扩展到理解，是扩散模型在视觉理解方向的重要推进

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ VSGG建模为时序去噪的思路极其新颖，统一嵌入+条件扩散的设计完整且自然
- 实验充分度: ⭐⭐⭐⭐ 三个评估协议、两种约束设置、与多个 baseline 对比充分，但仅一个数据集
- 写作质量: ⭐⭐⭐⭐ 动机和方法描述清晰，但公式密集段落需要耐心
- 价值: ⭐⭐⭐⭐⭐ 首次实现在线VSGG且性能SOTA，对实时视觉理解有重要推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](towards_spatio-temporal_world_scene_graph_generation_from_monocular_videos.md)
- [\[NeurIPS 2025\] Graph Diffusion that can Insert and Delete](../../NeurIPS2025/computational_biology/graph_diffusion_that_can_insert_and_delete.md)
- [\[ICML 2025\] Supercharging Graph Transformers with Advective Diffusion](../../ICML2025/computational_biology/supercharging_graph_transformers_with_advective_diffusion.md)
- [\[ICML 2025\] WGFormer: An SE(3)-Transformer Driven by Wasserstein Gradient Flows for Molecular Generation](../../ICML2025/computational_biology/wgformer_an_se3-transformer_driven_by_wasserstein_gradient_flows_for_molecular_g.md)
- [\[ICML 2025\] Kinetic Langevin Diffusion for Crystalline Materials Generation](../../ICML2025/computational_biology/kinetic_langevin_diffusion_for_crystalline_materials_generation.md)

</div>

<!-- RELATED:END -->
