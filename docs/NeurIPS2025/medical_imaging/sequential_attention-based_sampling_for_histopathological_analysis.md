---
title: >-
  [论文解读] Sequential Attention-based Sampling for Histopathological Analysis
description: >-
  [NeurIPS 2025][医学图像][全切片图像分析] 提出 SASHA 框架，结合层次注意力多实例学习 (HAFED) 与深度强化学习 (RL)，仅采样 10-20% 的高分辨率 patch 即可达到全分辨率 SOTA 方法的分类性能，推理速度提升 4-8 倍，WSI 压缩率超 16 倍。 全切片图像 (WSI) 是数…
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "全切片图像分析"
  - "深度强化学习"
  - "多实例学习"
  - "注意力采样"
  - "病理诊断"
---

# Sequential Attention-based Sampling for Histopathological Analysis

**会议**: NeurIPS 2025  
**arXiv**: [2507.05077](https://arxiv.org/abs/2507.05077)  
**代码**: [GitHub](https://github.com/coglabiisc/SASHA)  
**领域**: 医学图像  
**关键词**: 全切片图像分析, 深度强化学习, 多实例学习, 注意力采样, 病理诊断

## 一句话总结

提出 SASHA 框架，结合层次注意力多实例学习 (HAFED) 与深度强化学习 (RL)，仅采样 10-20% 的高分辨率 patch 即可达到全分辨率 SOTA 方法的分类性能，推理速度提升 4-8 倍，WSI 压缩率超 16 倍。

## 研究背景与动机

全切片图像 (WSI) 是数字病理学的核心数据，但其尺寸高达**千兆像素级别**，包含数千个 patch。直接在高分辨率下处理所有 patch 面临巨大的计算和存储瓶颈。更关键的是，诊断信息通常仅集中在**少部分区域**（如肿瘤区域），大部分 patch 是非诊断性的正常组织，全部处理极其低效。

多实例学习 (MIL) 框架是当前主流方案，将 WSI 分为 patch bag 后用注意力加权聚合。但现有 MIL 方法（ABMIL、TransMIL、ACMIL）仍需处理所有 patch 的高分辨率特征。RLogist 首次尝试用深度 RL 选择性采样 patch，但存在三大问题：

**准确率差距大**：仅采样部分 patch 时比全分辨率 SOTA 低 10-15%

**特征表示弱**：使用 ImageNet 预训练的 ResNet-50，非病理诊断最优

**训练不稳定**：RL 策略网络与分类网络同时训练导致收敛困难

SASHA 的核心思路是：用**标签感知的层次注意力特征蒸馏器 (HAFED)** 学习高质量诊断特征，用**目标化状态更新 (TSU)** 利用 patch 间的特征相关性高效传播信息，并**分阶段训练降低 RL 学习难度**。

## 方法详解

### 整体框架

SASHA 将 WSI 分析建模为**马尔可夫决策过程 (MDP)**：RL 智能体从低分辨率全景出发，每一步选择一个 patch 放大到高分辨率，提取特征后更新 WSI 状态，最终基于积累的状态进行分类。框架包括三个关键组件：HAFED、TSU 和 PPO 策略网络。

### 关键设计

1. **层次注意力特征蒸馏器 (HAFED)**：两阶段注意力模型。第一阶段（Feature Aggregator）：对每个低分辨率 patch 内的 $k$ 个高分辨率 sub-patch 进行注意力加权聚合，将 $U \in \mathbb{R}^{N \times k \times d}$ 压缩为 $V \in \mathbb{R}^{N \times d}$，使高低分辨率特征维度对齐。第二阶段（Classifier）：跨 $N$ 个 patch 进行注意力聚合生成 slide 级别嵌入 $h \in \mathbb{R}^d$ 用于分类。采用**多注意力分支**（M 个头），配合 ACMIL 的相似性损失和标签损失训练，确保不同分支捕获不同的诊断特征。

   关键特点：HAFED 是**标签感知的**特征提取器，不同于 RLogist 使用的通用 ImageNet 特征。训练阶段处理所有 patch 的高分辨率特征，推理阶段仅处理 RL 选择的 patch。

2. **目标化状态更新 (TSU)**：初始状态 $S_0 = Z$（低分辨率特征）。每当 RL 选择 patch $a_t$ 并获得高分辨率特征 $V(a_t)$ 后，**仅更新与 $a_t$ 特征相似的 patch**：

    - 计算余弦相似度：$C = \{i: \cos\angle(S_t(i), S_t(a_t)) \geq \tau\}$
    - 对 $C$ 中的 patch 通过 MLP 更新：$S_{t+1}(i) = f_S([S_t(i), S_t(a_t), V(a_t)])$
    - 被采样 patch 直接替换：$S_{t+1}(a_t) = V(a_t)$
    - 已访问 patch 被 mask，避免重复采样

   与 RLogist 的**全局更新**（对所有 patch 都更新）相比，TSU 避免了不相关 patch 的信息污染，消融实验显示全局更新导致准确率下降 12.7%。

3. **PPO 强化学习策略**：从离散动作空间 $\{1,2,...,N\}$ 中采样 patch 索引，中间奖励为分类器的负交叉熵 $r_t = -CE(y, \hat{y_t})$。使用 GAE 计算优势函数。**关键训练策略**：先训练 HAFED 分类器至收敛，然后**冻结分类器权重**后再训练 RL 策略，避免两者同时训练的收敛问题。

### 预处理和特征提取

使用 CLAM 工具进行组织分割和 patch 划分（256×256×3），采用 WSI 预训练的 ViT 编码器提取 patch 特征嵌入 $d$ 维向量。低分辨率特征 $Z \in \mathbb{R}^{N \times d}$，高分辨率 $U \in \mathbb{R}^{N \times k \times d}$。

## 实验关键数据

### 主实验

| 方法 | 采样率 | CAMELYON16 Acc | CAMELYON16 AUC | TCGA-NSCLC Acc | TCGA-NSCLC AUC |
|------|--------|---------------|----------------|---------------|----------------|
| ACMIL | 100% | 0.941±0.015 | 0.970±0.011 | 0.906±0.025 | 0.959±0.006 |
| HAFED (ours) | 100% | **0.963±0.008** | **0.980±0.003** | **0.923±0.011** | **0.966±0.015** |
| RLogist-0.1 | 10% | 0.824 | 0.829 | 0.828 | 0.892 |
| **SASHA-0.1** | **10%** | **0.901±0.021** | **0.918±0.014** | **0.897±0.023** | **0.956±0.023** |
| RLogist-0.2 | 20% | 0.862 | 0.879 | 0.839 | 0.903 |
| **SASHA-0.2** | **20%** | **0.953±0.017** | **0.979±0.008** | **0.912±0.010** | **0.963±0.014** |

### 消融实验（CAMELYON16, 20% 采样）

| 变体 | Accuracy | AUC | F1 | 说明 |
|------|----------|-----|-----|------|
| SASHA Default | 0.964 | 0.980 | 0.953 | 完整模型 |
| ResNet-50 特征 | 0.860 | 0.817 | 0.780 | 换用ImageNet预训练ResNet |
| CONCH 编码器 | 0.930 | 0.950 | 0.905 | 换用医学预训练编码器 |
| 单注意力分支 | 0.899 | 0.964 | 0.851 | HAFED降为单头 |
| 全局更新 | 0.837 | 0.824 | 0.779 | TSU换为RLogist式全局更新 |
| 随机策略 | 0.516 | 0.550 | 0.553 | RL策略换为随机选择 |

### 关键发现

- SASHA-0.2 仅用 20% patch 即达到全分辨率 HAFED 的 98.9% 准确率（0.953 vs 0.963）
- 推理速度：SASHA-0.1 约 14 秒/WSI，SASHA-0.2 约 26 秒/WSI，vs HAFED 的 117 秒/WSI（4-8x 加速）
- WSI 压缩率超 16x，因为 HAFED 将高低分辨率特征统一压缩到 N×d
- RL 智能体选择的 patch 具有显著更高的肿瘤组织比例和注意力得分（p<0.001），证明策略具有可解释性
- 采样比例越高，模型校准误差 (ECE) 越低，SASHA-0.2 的 ECE 甚至低于全分辨率的 ACMIL 和 DTFD

## 亮点与洞察

- 将病理学家的"先扫描后放大"工作流直接建模为 MDP，设计自然合理
- TSU 的局部更新策略基于一个简单但有效的直觉：一个 patch 的高分辨率信息主要与特征相似的 patch 相关
- 分阶段训练（先 HAFED 后 RL）巧妙解决了 RL 训练不稳定问题
- 可解释性分析证明 RL 确实学到了有意义的采样策略

## 局限与展望

- 训练阶段仍需处理所有 patch 的高分辨率特征，训练时间未减少
- 仅在二分类任务上充分验证，多分类结果放在附录
- 观测预算对校准的影响值得更深入研究
- TSU 阈值 $\tau$ 是固定超参数，可考虑自适应学习

## 相关工作与启发

- 与 ZoomMIL 相比，SASHA 使用 RL 自适应选择而非基于固定 top-k 注意力的采样，更灵活
- HAFED 的多分支注意力借鉴了 ACMIL 的设计，但增加了层次结构
- WSI 压缩率指标是一个有意义的新评估维度，对实际临床部署重要

## 评分

- 新颖性: ⭐⭐⭐⭐ RL+MIL结合不完全新，但HAFED+TSU的设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 两个benchmark+全面消融+可解释性+校准分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富
- 价值: ⭐⭐⭐⭐⭐ 解决WSI分析的实际效率问题，4-8x推理加速很有临床价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Boltzmann Attention Sampling for Image Analysis with Small Objects](../../CVPR2025/medical_imaging/boltzmann_attention_sampling_for_image_analysis_with_small_objects.md)
- [\[NeurIPS 2025\] MTBBench: A Multimodal Sequential Clinical Decision-Making Benchmark in Oncology](mtbbench_a_multimodal_sequential_clinical_decision-making_benchmark_in_oncology.md)
- [\[NeurIPS 2025\] Posterior Sampling by Combining Diffusion Models with Annealed Langevin Dynamics](posterior_sampling_by_combining_diffusion_models_with_annealed_langevin_dynamics.md)
- [\[NeurIPS 2025\] RadZero: Similarity-Based Cross-Attention for Explainable Vision-Language Alignment in Chest X-ray](radzero_similarity-based_cross-attention_for_explainable_vision-language_alignme.md)
- [\[NeurIPS 2025\] EndoBench: A Comprehensive Evaluation of Multi-Modal Large Language Models for Endoscopy Analysis](endobench_a_comprehensive_evaluation_of_multi-modal_large_language_models_for_en.md)

</div>

<!-- RELATED:END -->
