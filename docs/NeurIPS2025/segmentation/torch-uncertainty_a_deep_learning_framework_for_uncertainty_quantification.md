---
title: >-
  [论文解读] Torch-Uncertainty: A Deep Learning Framework for Uncertainty Quantification
description: >-
  [NeurIPS 2025 Spotlight][语义分割][不确定性量化] Torch-Uncertainty 是首个统一、可扩展、领域通用且以评估为中心的 PyTorch/Lightning 不确定性量化 (UQ) 框架，集成了 6 大类 UQ 方法、26 种评估指标和 27 个即插即用数据集，覆盖分类、分割、回归等任务，并提供了完整的基准测试结果。
tags:
  - "NeurIPS 2025 Spotlight"
  - "语义分割"
  - "不确定性量化"
  - "PyTorch框架"
  - "深度集成"
  - "校准"
---

# Torch-Uncertainty: A Deep Learning Framework for Uncertainty Quantification

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2511.10282](https://arxiv.org/abs/2511.10282)  
**代码**: [GitHub](https://github.com/ENSTA-U2IS-AI/Torch-Uncertainty)  
**领域**: 图像分割  
**关键词**: 不确定性量化, PyTorch框架, 深度集成, 语义分割, 校准

## 一句话总结

Torch-Uncertainty 是首个统一、可扩展、领域通用且以评估为中心的 PyTorch/Lightning 不确定性量化 (UQ) 框架，集成了 6 大类 UQ 方法、26 种评估指标和 27 个即插即用数据集，覆盖分类、分割、回归等任务，并提供了完整的基准测试结果。

## 研究背景与动机

深度神经网络在计算机视觉和 NLP 等领域表现出色，但在预测不确定性量化方面存在严重不足，限制了其在医疗、自动驾驶、金融等高风险场景中的部署。不确定性量化 (UQ) 方法虽然已有大量研究，但存在三个核心痛点：

**碎片化实现**：现有 UQ 库（TorchCP、BLiTZ、Bayesian-Torch 等）各自只覆盖少量 UQ 方法家族，缺乏统一工具来无缝评估和集成不同方法

**方法不可组合**：大多数库的架构较为僵化，难以将多种 UQ 技术灵活组合（如构建"Laplace 近似的集成"或"MC Dropout 模型的集成"）

**评估不完整**：现有框架缺乏系统的多维度鲁棒性评估（校准、OOD 检测、选择性分类、分布偏移鲁棒性等）

Torch-Uncertainty 通过三大设计原则来填补这一空缺：**领域通用性**（支持从单模态视觉到时序数据的多种模态）、**模块化 UQ 设计**（每种方法独立实现，可自由组合）、**评估中心**（内置最全面的指标套件，训练时跟踪 UQ 指标并自动存储最佳检查点）。

## 方法详解

### 整体框架

Torch-Uncertainty 基于 PyTorch 和 Lightning 构建，核心架构包括：任务特定的 Routine（ClassificationRoutine、RegressionRoutine、SegmentationRoutine 等）、模块化的 UQ 方法实现、完整的评估指标和数据集集合。框架遵循"训练→后处理→评估"的统一流程。

### 关键设计

1. **TU Routines（任务路由）**：每种任务路由封装了训练循环、UQ 感知的验证指标、后处理和可视化。核心创新是在验证阶段同时跟踪多个 UQ 指标（如 ECE、NLL、Brier score），并通过 `CompoundCheckpoint` 高效保存多指标组合下的最佳模型。设计动机：不同指标的最优模型往往出现在不同 epoch，单一指标保存会遗漏最佳状态。SegmentationRoutine 还通过像素子采样高效计算分割指标。

2. **UQ 方法的可组合性**：所有方法在统一的任务路由内实现，用户可通过选择适当的层（`torch_uncertainty.layers`）、模型包装器（`torch_uncertainty.models.wrappers`）和后处理方法来自由组合不同 UQ 技术。例如可以轻松构建 Laplace 近似的集成、或 MC Dropout 模型的集成。支持的 6 大方法族包括：集成方法（Deep Ensembles、Packed-Ensembles、MIMO 等）、贝叶斯方法（变分 BNN、SWAG、SGLD 等）、后验校准（Temperature Scaling、MC Dropout、Laplace 近似等）、数据增强（TTA）、确定性 UQ（Evidential Networks）、区间/共形预测，共计 20+ 种具体方法。

3. **评估指标覆盖**：实现了 26 种指标覆盖 7 个任务类别——分类（Accuracy、Brier、NLL）、OOD 检测（AUROC、AUPR、FPR95）、选择性分类（AURC、AUGRC、Coverage@Risk、Risk@Coverage）、校准（ECE、aECE）、多样性、回归/深度、分割（mIoU、mAcc、pixAcc）。同时提供效率指标（参数量、FLOPs）。这是同类库中最全面的指标覆盖。

### 损失函数 / 训练策略

- 各任务使用标准损失（分类用交叉熵、分割用像素级交叉熵等）
- 关键创新在于训练过程中持续跟踪多个验证指标，并自动保存各指标下的最佳检查点
- 支持 Mixup 等数据增强作为内置选项
- 后处理方法（Temperature Scaling 等）作为可选步骤在训练后应用

## 实验关键数据

### 主实验 — 分类基准 (ViT-B/16, ImageNet-1K)

| 方法 | Accuracy (%) | ECE (%) | FarOOD AUROC (%) | Risk@80Cov (%) |
|------|-------------|---------|-----------------|---------------|
| Single Model | 80.67 | 0.01 | 90.75 | 9.81 |
| + Temperature Scaling | 80.67 | 0.01 | 90.44 | 9.79 |
| Deep Ensemble | **82.19** | 0.03 | **92.05** | **8.54** |
| + Temperature Scaling | 82.19 | **0.01** | 91.18 | 8.49 |
| Packed Ensemble | 79.23 | 0.01 | 89.84 | 10.88 |
| MiMo | 80.59 | 0.02 | 89.13 | 9.63 |

### 消融实验 — 语义分割基准 (UNet, MUAD)

| 方法 | mIoU (%) | mAcc (%) | pixAcc (%) | ECE (%) | NLL |
|------|---------|---------|-----------|---------|-----|
| Baseline UNet | 71.55 | 87.65 | 93.59 | 0.51 | 0.18 |
| + MC Dropout | 68.80 | 85.99 | — | — | — |
| Deep Ensembles | **最佳** | — | — | — | — |
| Packed Ensembles | 接近DE | — | — | 更好 | — |
| BatchEnsemble | — | — | — | — | — |
| MIMO | — | — | — | — | — |

### 关键发现

- **Deep Ensemble 全面领先**：在精度、校准和 OOD 检测上均为最优，Temperature Scaling 进一步改善校准（ECE 从 0.03% 降至 0.01%）
- **紧凑集成有竞争力**：Packed Ensemble 和 MIMO 以更低成本接近 Deep Ensemble 性能
- **分割任务中的发现**：Deep Ensembles 在分割精度上最佳，但 Packed Ensembles 在校准上可能更优（归因于其内置的增强效果）
- **最优 checkpoint epoch 因指标而异**：不同验证指标的最佳模型出现在不同训练阶段，验证了多指标保存策略的必要性

## 亮点与洞察

- **最全面的 UQ 框架**：6 大方法族、26 种指标、27 个数据集的统一集成，是目前覆盖面最广的 UQ 工具
- **可组合设计**：用户可轻松构建混合 UQ 方法（如 Laplace 近似的集成），这在其他库中需要大量额外实现
- **工程质量极高**：98% 的单元测试覆盖率、ruff 代码规范、Discord 社区、HuggingFace 预训练模型和 Zenodo 数据集
- **实用价值**：降低了 UQ 研究和部署的门槛，研究者可专注方法创新而非数据和评估的基础设施建设

## 局限与展望

- 分割基准仅使用 UNet + MUAD，未覆盖更大规模模型（如 Mask2Former）和更主流的数据集（如 Cityscapes、ADE20K）
- 当前不支持高斯过程方法（因难以扩展到大模型）
- 框架偏向视觉任务，对 NLP/多模态任务的支持有待扩展
- 缺乏对最新基础模型（如 SAM、CLIP）的 UQ 集成

## 相关工作与启发

- 与 Lightning-UQ-Box 最接近，但 Torch-Uncertainty 在方法覆盖（20+ vs 更少）、指标数量（26 vs 9）、数据集支持上全面领先
- 与 TorchCP（仅共形预测）、BLiTZ/Bayesian-Torch（仅贝叶斯）等专注型库形成互补
- 提供了标准化的 UQ 基准，有助于推动领域内的公平比较和可重复研究

## 评分

- **新颖性**: ⭐⭐⭐ 框架集成工作，方法本身非全新，贡献在于系统化整合
- **实验充分度**: ⭐⭐⭐⭐ 分类基准详尽，分割基准相对简单，覆盖多维度评估
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，库设计和对比表格一目了然
- **价值**: ⭐⭐⭐⭐⭐ 对 UQ 社区有极高实用价值，填补了统一工具的空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] On the Generalization of Representation Uncertainty in Earth Observation](../../ICCV2025/segmentation/on_the_generalization_of_representation_uncertainty_in_earth_observation.md)
- [\[CVPR 2026\] Hyperbolic Prototype Learning with Uncertainty-Aware Consistency for Continual Test-Time Segmentation](../../CVPR2026/segmentation/hyperbolic_prototype_learning_with_uncertainty-aware_consistency_for_continual_t.md)
- [\[ICML 2026\] Segment Anything with Robust Uncertainty-Accuracy Correlation](../../ICML2026/segmentation/segment_anything_with_robust_uncertainty-accuracy_correlation.md)
- [\[NeurIPS 2025\] OmniSegmentor: A Flexible Multi-Modal Learning Framework for Semantic Segmentation](omnisegmentor_a_flexible_multi-modal_learning_framework_for_semantic_segmentatio.md)
- [\[CVPR 2026\] Uncertainty-Aware Modality Fusion for Unaligned RGB-T Salient Object Detection](../../CVPR2026/segmentation/uncertainty-aware_modality_fusion_for_unaligned_rgb-t_salient_object_detection.md)

</div>

<!-- RELATED:END -->
