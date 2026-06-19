---
title: >-
  [论文解读] Teaching Large Language Models to Regress Accurate Image Quality Scores Using Score Distribution
description: >-
  [CVPR 2025][多模态VLM][图像质量评估] 提出DeQA-Score，通过将质量分数的**高斯分布离散化为soft label**（替代Q-Align的one-hot label），大幅减少离散化信息损失（10-35倍），并引入基于Thurstone模型的**fidelity loss**实现多IQA数据集联合训练，在分数回归任务上全面超越基线。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "图像质量评估"
  - "MLLM"
  - "分数回归"
  - "分布离散化"
  - "soft label"
  - "多数据集联合训练"
---

# Teaching Large Language Models to Regress Accurate Image Quality Scores Using Score Distribution

**会议**: CVPR 2025  
**arXiv**: [2501.11561](https://arxiv.org/abs/2501.11561)  
**代码**: [https://depictqa.github.io/deqa-score/](https://depictqa.github.io/deqa-score/)  
**领域**: 多模态VLM  
**关键词**: 图像质量评估, MLLM, 分数回归, 分布离散化, soft label, 多数据集联合训练

## 一句话总结
提出DeQA-Score，通过将质量分数的**高斯分布离散化为soft label**（替代Q-Align的one-hot label），大幅减少离散化信息损失（10-35倍），并引入基于Thurstone模型的**fidelity loss**实现多IQA数据集联合训练，在分数回归任务上全面超越基线。

## 研究背景与动机
MLLM在图像质量描述（语言化评价）方面表现出色，但在精确的质量**分数回归**上仍不如传统IQA方法。核心障碍在于：质量分数本质上是连续值（通常建模为高斯分布），而MLLM生成的是离散token——这种**分布鸿沟**需要对分数进行离散化。

**现有痛点**：Q-Align等方法将均值分数离散化为one-hot标签（5个等级：bad/poor/fair/good/excellent），存在三大问题：
1. **信息损失严重**：离散化误差为soft label的10-35倍
2. **破坏图像间关系**：质量相近但跨越边界的图像被分到不同等级（如MOS=3.38→fair, MOS=3.49→good）
3. **假设等级独立**：one-hot假设5个等级正交，但实际上"fair"与"good"的语义距离比"fair"与"excellent"近

**核心idea**：不离散化分数的均值，而是离散化分数的**整个分布**——将高斯分布在5个等级中心点处积分，得到一个保留分布特征的soft label。

## 方法详解

### 整体框架
DeQA-Score基于mPLUG-Owl2架构（CLIP ViT-L + Q-Former视觉抽象器 + LLaMA-2-7B）。训练时对响应模板"The quality of this image is \<level\>"中的level token使用KL散度损失（与soft label对齐），其他token用标准交叉熵。推理时从5个等级的预测概率恢复连续分数分布的均值和方差。

### 关键设计

1. **基于分布的Soft Label构建**
    - 功能：将连续的质量分数分布离散化为5个等级上的概率分布，最小化信息损失
    - 核心思路：质量分数$x \sim \mathcal{N}(\mu, \sigma^2)$，以5个中心点$c_i \in \{1,2,3,4,5\}$为等级中心，宽度$d=1$，计算分数落入每个等级的概率：$p_i^{raw} = \int_{c_i-d/2}^{c_i+d/2} f(x)dx$。然后通过线性变换$p_i = \alpha p_i^{raw} + \beta$进行后调整，约束$\sum p_i = 1$且$\sum p_i c_i = \mu$，确保离散分布的期望精确等于原始MOS
    - 设计动机：one-hot label的离散化误差（L1 Error）约0.3，而soft label仅约0.01-0.02。soft label天然保留图像间相对关系：质量相近的图像得到相似的分布，质量不同的图像分布差异明显
    - 损失函数：level token处使用KL散度 $\mathcal{L}_{kl} = -\sum_i p_i \log(p_i^{pred}/p_i)$

2. **基于Thurstone模型的Fidelity Loss（多数据集联合训练）**
    - 功能：解决不同IQA数据集分布差异问题，使模型能有效联合训练多个数据集
    - 核心思路：从同一数据集采样两张图A和B，利用预测的分数分布计算A优于B的概率 $p^{pred}(A>B) = \Phi\left(\frac{\mu_A^{pred}-\mu_B^{pred}}{\sqrt{(\sigma_A^{pred})^2+(\sigma_B^{pred})^2}}\right)$，与ground-truth概率比较，使用fidelity loss训练
    - 设计动机：不同数据集的绝对分数不可比（同一分数在不同数据集代表不同质量），但**数据集内的排序关系**是可靠的。fidelity loss只学习排序而非绝对值，天然适合联合训练。关键：只有能预测分数*分布*（而非单个分数）的模型才能使用此loss——这正是soft label的独特优势
    - 最终损失：$\mathcal{L} = \mathcal{L}_{fd} + \gamma(\mathcal{L}_{ce} + \mathcal{L}_{kl})$，其中$\gamma=0.05$

3. **从离散到连续的分布恢复**
    - 功能：推理时从5个等级的预测概率恢复连续的质量分数分布
    - 核心思路：$\mu^{pred} = \sum_i p_i^{pred} c_i$，$(\sigma^{pred})^2 = \sum_i p_i^{pred}(c_i - \mu^{pred})^2$，得到恢复分布$\mathcal{N}(\mu^{pred}, (\sigma^{pred})^2)$
    - 设计动机：不仅输出一个分数，还能输出分数的不确定性，与人工标注的分布高度一致

## 实验关键数据

### 单数据集训练分数回归（Tab. 3, 在KonIQ上训练）

| 方法 | KonIQ (PLCC/SRCC) | LIVE-Wild | AGIQA-3K |
|------|-------------------|-----------|----------|
| Q-Align (one-hot) | 0.941/0.940 | 0.853/0.860 | 0.772/0.735 |
| **DeQA-Score (soft)** | **0.953/0.941** | **0.892/0.879** | **0.809/0.729** |
- 域内提升1.3% PLCC，域外（LIVE-Wild）提升4.6% PLCC

### 多数据集联合训练分数回归（Tab. 4）

| 训练集 | 方法 | KonIQ | SPAQ | KADID | PIPAL |
|-------|------|-------|------|-------|-------|
| KonIQ+SPAQ+KADID+PIPAL | Q-Align | 0.926/0.932 | 0.917/0.920 | 0.950/0.954 | 0.702/0.671 |
| KonIQ+SPAQ+KADID+PIPAL | **DeQA-Score** | **0.958/0.946** | **0.932/0.929** | **0.963/0.961** | **0.724/0.690** |
- 联合训练下优势更加明显，验证了fidelity loss的有效性

### 离散化精度对比（Tab. 1）

| 指标 | One-hot (Q-Align) | Soft Label |
|------|-------------------|------------|
| L1 Error (KonIQ) | 0.302 | **0.008** (37.75×更精确) |
| PLCC/SRCC与MOS | 0.961/0.952 | **1.000/1.000** |

### 消融实验（Tab. 6, 在KonIQ+SPAQ+KADID上训练）

| 配置 | KonIQ PLCC | KADID PLCC | LIVE-Wild PLCC | 说明 |
|------|-----------|-----------|---------------|------|
| One-hot（Q-Align基线） | 0.945 | 0.935 | 0.887 | 原始方法 |
| Soft label only | 0.954 | 0.951 | 0.896 | +soft label即有显著提升 |
| Soft label + Fidelity | **0.957** | **0.955** | **0.900** | Fidelity loss在多数据集下额外贡献 |

### 关键发现
- Soft label vs. one-hot：在所有8个测试集上全面领先
- Fidelity loss在多数据集联合训练中贡献显著，单数据集训练时作用有限（因为不需要跨数据集对齐）
- 后调整(post-adjustment)的$\alpha$和$\beta$平均值接近1和0，说明截断误差很小
- 分数分布预测与人工标注的JS散度仅0.014（KonIQ），Q-Align为0.415，差距悬殊
- 等级文本的语义顺序很重要：打乱或反转等级文本后性能明显下降

## 亮点与洞察
- **方法优雅**：用简单的分布离散化替代one-hot，在几乎不增加复杂度的情况下大幅提升精度
- **分布预测能力**：MLLM首次能预测与人工标注高度一致的质量分数分布（JS散度仅0.001-0.022），而非只是一个点估计
- **Soft label解锁fidelity loss**：两个设计形成协同——soft label使模型能预测分布，分布预测使fidelity loss可用，fidelity loss使多数据集联合训练有效
- **通用性**：soft label的思想可推广到任何需要MLLM回归连续值的任务（如深度估计、年龄估计等）

## 局限性
- 仅使用5个离散等级，虽然离散化精度已足够高，但更多等级可能进一步提升精度
- 基础模型mPLUG-Owl2相对较旧，在更强基础模型（如LLaVA-Next）上的效果未验证
- Fidelity loss需要从同一数据集采样图片对，对超大数据集的采样策略选择未深入探讨
- 分数回归是"底层感知"任务，与MLLM的"高层理解"能力之间的权衡值得进一步研究

## 相关工作与启发
- Q-Align（one-hot离散化的先驱）→ 本文直接改进其核心离散化策略
- UNIQUE（fidelity loss的提出者）→ 本文将其思想首次移植到MLLM框架
- DepictQA系列 → 关注语言描述而非分数回归，本文填补了后者的空白
- 启发：MLLM的离散token输出限制不一定是致命的——巧妙的离散化策略可以在几乎无损的情况下桥接连续和离散空间

## 评分
- 新颖性: ⭐⭐⭐⭐ 分布离散化为soft label的思路简洁优雅，解决了一个根本性的信息损失问题
- 实验充分度: ⭐⭐⭐⭐⭐ 9个IQA数据集、单/多数据集训练、分布预测、消融、等级文本实验全面到位
- 写作质量: ⭐⭐⭐⭐ 公式推导清晰，与Q-Align的对比直观，图表设计好
- 价值: ⭐⭐⭐⭐ 通用方法可推广到任何MLLM连续值回归任务，soft label+fidelity loss组合具有启发性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] On the Out-of-Distribution Generalization of Multimodal Large Language Models](on_the_out-of-distribution_generalization_of_large_multimodal_models.md)
- [\[CVPR 2025\] COUNTS: Benchmarking Object Detectors and Multimodal Large Language Models under Distribution Shifts](counts_benchmarking_object_detectors_and_multimodal_large_language_models_under_.md)
- [\[CVPR 2025\] RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics](robospatial_teaching_spatial_understanding_to_2d_and_3d_vision-language_models_f.md)
- [\[AAAI 2026\] DisCode: Distribution-Aware Score Decoder for Robust Automatic Evaluation of Image Captioning](../../AAAI2026/multimodal_vlm/discode_distribution-aware_score_decoder_for_robust_automatic_evaluation_of_imag.md)
- [\[ICCV 2025\] A Quality-Guided Mixture of Score-Fusion Experts Framework for Human Recognition](../../ICCV2025/multimodal_vlm/a_qualityguided_mixture_of_scorefusion_experts_framework_for.md)

</div>

<!-- RELATED:END -->
