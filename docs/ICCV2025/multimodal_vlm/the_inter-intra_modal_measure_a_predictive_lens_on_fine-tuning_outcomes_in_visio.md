---
title: >-
  [论文解读] The Inter-Intra Modal Measure: A Predictive Lens on Fine-Tuning Outcomes in Vision-Language Models
description: >-
  [ICCV 2025][多模态VLM][迁移性估计] 提出 Inter-Intra Modal Measure（IIMM）——一个仅需单次前向推理即可预测视觉-语言双编码器模型微调后性能增益和灾难性遗忘程度的指标，通过量化模态内图像嵌入相似性和模态间错误标签对齐程度，在 4 个基础模型和 5 种微调策略下展现出强线性预测能力（$R^2 > 0.85$）。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "迁移性估计"
  - "微调预测"
  - "模态间隙"
  - "对比学习"
  - "灾难性遗忘"
---

# The Inter-Intra Modal Measure: A Predictive Lens on Fine-Tuning Outcomes in Vision-Language Models

**会议**: ICCV 2025  
**arXiv**: [2407.15731](https://arxiv.org/abs/2407.15731)  
**代码**: [GitHub](https://github.com/mit-ll/IIMM)  
**领域**: 多模态VLM  
**关键词**: 迁移性估计, 微调预测, 模态间隙, 对比学习, 灾难性遗忘

## 一句话总结

提出 Inter-Intra Modal Measure（IIMM）——一个仅需单次前向推理即可预测视觉-语言双编码器模型微调后性能增益和灾难性遗忘程度的指标，通过量化模态内图像嵌入相似性和模态间错误标签对齐程度，在 4 个基础模型和 5 种微调策略下展现出强线性预测能力（$R^2 > 0.85$）。

## 研究背景与动机

视觉-语言基础模型（如 CLIP）的微调面临两个实际问题：

**微调是否值得**：计算成本高昂，但某些特化任务（如卫星/医学图像分类）微调后提升有限甚至下降——能否在微调前就预测效果？

**学习 vs 遗忘的权衡**：微调提升目标任务性能的同时会破坏图文嵌入空间对齐，导致灾难性遗忘（off-target 任务性能下降）——能否预测遗忘程度？

现有迁移性度量（transferability metrics）的不足：
- **H-Score、LEEP、NCE** 等经典方法依赖源数据和任务标签，不适用于对比式自监督预训练的基础模型
- **TransRate、SFDA** 等近期方法仅考虑单模态编码器，忽略了模态间交互
- 大多数方法只评估 rank correlation（排名相关性），无法直接**预测性能变化量**

核心洞察：对比学习的嵌入空间几何包含了关于微调潜力的丰富信息——**模态内嵌入越聚集 + 模态间错误对齐越高 → 微调改善空间越大，但遗忘风险也越大**。

## 方法详解

### 整体框架

IIMM 从对比学习的 InfoNCE 损失出发，将嵌入空间几何分解为两个可计算的组分：
1. 模态间错误对齐度（$S_{inter}$）
2. 模态内均匀性（$S_{intra}$）

### 关键设计

1. **从 InfoNCE 损失推导 IIMM**：

   InfoNCE 损失优化两个目标：正样本对齐 + 负样本分离。IIMM 捕获这两个效果的当前状态：

    - **模态间错误对齐**（Inter-modal misalignment）：
    $S_{inter} = \frac{1}{|X|} \sum_{x \in X} \frac{1}{|Y|-1} \sum_{y' \in Y \setminus \{y(x)\}} x^\top y'$
   高 $S_{inter}$ 意味着图像嵌入与错误文本标签的相似度高，说明模型尚未充分分离负样本——微调空间大但遗忘风险高

    - **模态内均匀性**（Intra-modal uniformity）：
    $S_{intra} = \frac{2}{|X|(|X|-1)} \sum_{1 \leq i < j \leq |X|} x_i^\top x_j$
   高 $S_{intra}$ 意味着图像嵌入过度聚集，表示有进一步分离/重组的空间

    - **IIMM 定义**：
    $\text{IIMM} = \frac{1}{2}(S_{inter} + S_{intra})$
   取值范围 $[-1, 1]$，高值预示高学习增益但也高遗忘风险

2. **理论保证——Wasserstein 距离界**：

   证明了 IIMM 的变化被微调前后嵌入分布的 1-Wasserstein 距离约束：
    $|\text{IIMM}(Q) - \text{IIMM}(P)| \leq \frac{\delta_A}{2} + \delta_B$
   
   物理含义：只有嵌入空间发生显著重组时 IIMM 才会显著变化，保证了指标的**稳定性和鲁棒性**

3. **使用方式**：

    - 对目标任务数据做一次前向推理，计算 IIMM
    - 预先在几个标准视觉基准上微调并拟合 IIMM-to-gain 的线性模型
    - 对新任务直接用线性模型预测微调增益，无需实际微调

### 损失函数 / 训练策略

实验中的微调设置：
- 30 epoch SGD，batch size 128
- 网格搜索：lr ∈ {1e-2, 1e-3, 1e-4, 1e-5}，weight decay ∈ {1e-3, 1e-4, 1e-5}
- LoRA rank ∈ {2, 4, 8, 16}，CLIP-Adapter reduction ∈ {4, 8, 16, 32}
- 在 9 个视觉分类数据集上分别微调和测试

## 实验关键数据

### 主实验（IIMM vs 基线指标的 Kendall's τ 相关性）

IIMM 与"微调增益/零样本误差"的相关性：

| 指标 | CoCa τ | EVA-02 τ | CLIP τ | SigLIP τ |
|------|--------|----------|--------|---------|
| **IIMM** | **0.89** | **0.83** | **0.78** | **0.78** |
| GBC | 0.78 | 0.72 | 0.78 | 0.83 |
| TransRate | 0.67 | 0.61 | 0.67 | 0.72 |
| EMMS | 0.61 | 0.28 | 0.61 | 0.44 |
| NCTI | -0.06 | -0.59 | 0.07 | -0.06 |
| H-Score | 0.11 | -0.78 | 0.50 | -0.67 |

### 线性拟合强度（IIMM → gain over zero-shot error）

| 模型 | $R^2$ | p-value |
|------|-------|---------|
| CoCa | 0.92 | <10⁻³ |
| EVA-02 | 0.89 | <10⁻³ |
| CLIP | 0.92 | <10⁻³ |
| SigLIP | 0.92 | <10⁻³ |

对比 GBC（第二强指标）：$R^2$ 仅 0.59-0.72

### 嵌入空间成分分析

| 嵌入特征 | CoCa $R^2$ | CLIP $R^2$ | SigLIP $R^2$ | 说明 |
|---------|-----------|-----------|-------------|------|
| 模态内图像距离 | 0.88 | 0.81 | 0.88 | 最具预测力的单一特征 |
| 错误标签对齐 | 0.82 | 0.62 | 0.80 | 第二重要 |
| 模态内文本距离 | 0.37 | 0.57 | 0.17 | 基本无信息量 |

### 关键发现

- **IIMM 在所有基础模型上 $R^2 > 0.89$**，显著优于所有单模态迁移性指标
- **IIMM 与遗忘呈负线性关系**：IIMM 越高，微调后 off-target 任务平均准确率下降越多（除 CLIP 外 $R^2$ 均较高）
- **PEFT 方法的差异化表现**：
    - CLIP-Adapter 的局部化调整（冻结主干）减弱了 IIMM 与增益的关系，但也大幅减轻了遗忘
    - LoRA 和注意力权重微调在 IIMM 高时呈现严重遗忘
- **模态内图像嵌入相似度是最具预测力的组分**（$R^2$ = 0.81-0.88），但与模态间错误对齐结合后效果更强（$R^2$ = 0.89-0.92），验证了多模态信息的必要性
- **正确标签对齐和文本内距离基本无信息**——只有"负样本"方向有用

## 亮点与洞察

- **理论与实践的平衡**：从 InfoNCE 损失出发的推导既优雅又有物理意义，Wasserstein 距离界提供了理论保障
- **极低计算开销**：仅需目标数据的一次前向推理（无需微调），是真正实用的工具
- **有界且极值有意义**：IIMM ∈ [-1,1]，不像 GBC 那样无界——高 IIMM 直接告诉你"微调值得但要小心遗忘"
- **揭示了 PEFT 方法的遗忘特征**：LoRA 在 IIMM 高时可能非常危险，而 CLIP-Adapter 的"保守"设计反而是优势

## 局限与展望

- 预测线性模型需要先在几个基准上微调来拟合——未来可否找到完全不需要参考微调结果的方法？
- 实验仅涵盖分类任务，未验证在检测、分割、VQA 等其他下游任务上的有效性
- 仅测试了 ViT-B 级别的模型，在更大规模模型（ViT-L/H）上的表现未知
- 两个子分量的 等权平均 是否最优？虽然做了凸组合实验但未深入讨论

## 相关工作与启发

- IIMM 的思路可推广到其他对比学习模型（如 CLAP 音频-文本、ImageBind 多模态）
- 模态间隙（modality gap）研究的延续和实用化——从观察现象到预测工具
- 为 PEFT 方法选择提供了定量依据：IIMM 高的任务应选择保守的微调策略（如 Adapter）

## 评分

- 新颖性：⭐⭐⭐⭐⭐ （首个专为视觉-语言双编码器设计的微调结果预测指标）
- 技术深度：⭐⭐⭐⭐⭐ （从对比学习损失推导+Wasserstein 理论界+全面实验验证）
- 实验充分度：⭐⭐⭐⭐ （4 模型 × 5 微调方法 × 9 数据集，但仅限分类任务）
- 实用价值：⭐⭐⭐⭐⭐ （一次前向推理即可预测微调效果，极具实用性）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving](fine-grained_evaluation_of_large_vision-language_models_in_autonomous_driving.md)
- [\[ICCV 2025\] FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models](fedmvp_federated_multimodal_visual_prompt_tuning_for_vision-language_models.md)
- [\[CVPR 2025\] VladVA: Discriminative Fine-tuning of LVLMs](../../CVPR2025/multimodal_vlm/vladva_discriminative_fine-tuning_of_lvlms.md)
- [\[ICCV 2025\] From Holistic to Localized: Local Enhanced Adapters for Efficient Visual Instruction Fine-Tuning](from_holistic_to_localized_local_enhanced_adapters_for_efficient_visual_instruct.md)
- [\[CVPR 2026\] Reevaluating the Intra-Modal Misalignment Hypothesis in CLIP](../../CVPR2026/multimodal_vlm/reevaluating_the_intra-modal_misalignment_hypothesis_in_clip.md)

</div>

<!-- RELATED:END -->
