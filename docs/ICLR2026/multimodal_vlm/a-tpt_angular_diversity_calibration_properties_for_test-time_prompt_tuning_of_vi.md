---
title: >-
  [论文解读] A-TPT: Angular Diversity Calibration Properties for Test-Time Prompt Tuning of Vision-Language Models
description: >-
  [ICLR 2026][多模态VLM][测试时提示调优] 提出 A-TPT 框架，通过最大化归一化文本特征在单位超球面上的最小成对角距离来促进角度多样性，解决测试时提示调优 (TPT) 中 VLM 预测过度自信导致的校准不良问题，在自然分布偏移和医学数据集上均优于现有 TPT 校准方法。 领域现状：TPT 通过在推理时用未标…
tags:
  - "ICLR 2026"
  - "多模态VLM"
  - "测试时提示调优"
  - "CLIP"
  - "校准"
  - "角度多样性"
  - "超球面均匀分布"
---

# A-TPT: Angular Diversity Calibration Properties for Test-Time Prompt Tuning of Vision-Language Models

**会议**: ICLR 2026  
**arXiv**: [2510.26441](https://arxiv.org/abs/2510.26441)  
**代码**: 即将开源  
**领域**: 多模态VLM / 校准  
**关键词**: 测试时提示调优, CLIP, 校准, 角度多样性, 超球面均匀分布  

## 一句话总结
提出 A-TPT 框架，通过最大化归一化文本特征在单位超球面上的最小成对角距离来促进角度多样性，解决测试时提示调优 (TPT) 中 VLM 预测过度自信导致的校准不良问题，在自然分布偏移和医学数据集上均优于现有 TPT 校准方法。

## 研究背景与动机

**领域现状**：TPT 通过在推理时用未标注样本优化可学习提示向量来适应 VLM（如 CLIP）到新任务，提升准确率但常导致校准误差增大（过度自信）。

**现有痛点**：C-TPT 通过最大化平均文本特征分散度 (ATFD) 来改善校准，但特征仍可能聚集；O-TPT 用正交性约束强制角度分离，但当类数>嵌入维度（如 ImageNet-1k 的 1000 类 vs CLIP 512 维）时强制正交数学上不可能，导致特征反而聚集。

**核心矛盾**：分散度 (L2 距离) 和正交性约束都不能保证超球面上特征的均匀角度分布——前者可能使所有特征偏向一个方向而与质心远，后者在高类数时失效。

**本文目标** 提出一种能在 $N > |D|$ 和 $N < |D|$ 两种情况下都有效促进文本特征角度多样性的 TPT 校准方法。

**切入角度**：将校准问题与 Tammes 问题（超球面上最优点布置）联系——最大化最小成对角距离保证特征均匀分布。

**核心 idea**：通过最大化归一化文本特征间的最小成对角距离（而非平均分散度或正交性），在超球面上实现均匀分布，显著改善 VLM 推理时的校准性能。

## 方法详解

### 整体框架
A-TPT 想解决的是 TPT 在推理时把准确率拉高、却把校准搞坏（预测过度自信）这个副作用。它沿用 TPT 的设置——在每个测试样本上对可学习提示向量做无标注优化——但在原本的熵最小化目标之外，额外加了一项角度多样性正则。整体流程是：把可学习提示拼到每个类名前生成文本特征 → 对图像做多视图数据增强、用熵最小化更新提示以适应当前样本 → 同时约束所有类的归一化文本特征在单位超球面上的**最小成对角距离**尽可能大，逼它们均匀铺开而不是挤成一团。最后用这组更分散的文本特征做分类，从而在不掉准确率的前提下压低校准误差。

### 关键设计

**1. 角度多样性正则：用最大化最小角距离替代分散度/正交约束**

前作的两条路都不能保证超球面上特征真正均匀分布：C-TPT 的 ATFD 最大化各特征到质心的 L2 距离，但所有特征完全可以一起偏向同一个方向、彼此挨得很近；O-TPT 强制特征两两正交，可一旦类数超过嵌入维度（如 ImageNet-1k 的 1000 类对 CLIP 的 512 维）正交在数学上根本不可能，约束失效后特征反而聚集。A-TPT 把这个问题对应到 **Tammes 问题**（在超球面上摆 N 个点使它们尽量散开）：对所有类对 $(i,j)$ 计算角距离 $\theta_{ij} = \arccos(\text{sim}(t_i, t_j))$，优化目标里包含 $\max \min_{i \neq j} \theta_{ij}$，也就是把"最近的那一对特征"撑得越开越好，等价于 Tammes 问题的数值近似。这样做的好处是它对维度不挑：$N > |D|$ 时正交虽不可行，但"让最小角距离最大"仍是良定义的、能找到有限维空间里 N 个点的最佳布置；$N < |D|$ 时它又能充分用满超球面空间，不像正交约束那样浪费维度。

**2. 与 TPT 的集成：作为正则项挂到熵最小化目标上**

角度多样性损失不替换 TPT 原有目标，而是作为正则项叠加进去，联合优化总损失 $\mathcal{L} = \mathcal{L}_{entropy} + \lambda \mathcal{L}_{angular}$，其中 $\lambda$ 平衡适应能力与校准约束。这样 $\mathcal{L}_{entropy}$ 仍负责让提示适应当前测试样本、保住 TPT 的准确率收益，$\mathcal{L}_{angular}$ 则在同一步优化里把文本特征推向均匀分布、改善校准。两项在每个测试样本的推理时一起做梯度下降，互不替代——这也是 A-TPT 能在不掉准确率的前提下压低校准误差的原因。

### 损失函数 / 训练策略
在每个测试样本上对可学习提示向量做梯度下降（无标注数据），先用数据增强生成多个视图、最小化预测熵，同时叠加角度多样性正则 $\mathcal{L}_{angular}$。温度参数 $\tau=0.01$ 固定。

## 实验关键数据

### 主实验

**CLIP ViT-B/16 在 Caltech101 上的 ECE↓ (期望校准误差):**

| 方法 | Caltech101 ECE↓ |
|------|-----------------|
| Zero-shot CLIP | 5.66 |
| TPT | 6.18 |
| **A-TPT** | **2.23** |

A-TPT 在 Caltech101、OxfordPets、DTD、EuroSAT、ImageNet 等多个自然分布偏移数据集上均取得最低 ECE，并泛化到医学数据集；逐数据集与 C-TPT / O-TPT 的完整对比见原文。

### 消融实验

下表为各校准约束相对 TPT 的趋势对比（准确率均基本保持，差异主要体现在 ECE）：

| 配置 | ECE 趋势 | 说明 |
|------|---------|------|
| TPT (无校准) | 最高 | 准确但过度自信 |
| + ATFD (C-TPT) | 略降 | 仅最大化与质心的 L2 分散，改善有限 |
| + 正交性 (O-TPT) | 略降 | $N > |D|$ 时正交不可行而失效 |
| + 角度多样性 (A-TPT) | **最低** | 最大化最小成对角距离，全面最优 |

### 关键发现
- 经验分析表明角距离 (AD) 与 ECE 呈负相关——AD 越大校准越好，验证了角度多样性的理论合理性
- 在 $N > |D|$ 时（如 ImageNet-1k 的 1000 类 vs 512 维），O-TPT 的正交约束完全失效（不可能让 1000 个向量成对正交），A-TPT 仍然有效
- t-SNE 可视化清晰展示了角度多样的特征不仅分散且与类标签对齐良好
- A-TPT 在不牺牲准确率的情况下显著降低 ECE，且泛化到医学数据集

## 亮点与洞察
- **Tammes 问题的机器学习应用**：将超球面最优布点问题与 VLM 校准连接是巧妙的跨领域思路。最小成对角距离最大化比 L2 分散度和正交性更本质地刻画了"均匀分布"
- **校准与准确率的解耦**：在相同准确率组内，校准性能的差异主要由角度多样性驱动——这个发现为理解 VLM 校准提供了新视角
- **理论与实践的一致性**：超球面上的均匀分布保留最大信息量（Wang & Isola, 2020），与校准改善一致

## 局限与展望
- 最大化最小角距离的优化是非凸的，可能收敛到局部最优
- 数值优化每个测试样本的开销——TPT 本身的推理时成本已不小，增加正则项进一步增加
- 实验主要在分类任务上，检测/分割等下游任务未验证
- 温度参数 $\tau=0.01$ 固定不变，自适应温度可能进一步改善

## 相关工作与启发
- **vs C-TPT**: C-TPT 用 ATFD 最大化与质心的 L2 距离；A-TPT 直接优化最小成对角距离，更直接地保证均匀性
- **vs O-TPT**: O-TPT 用正交约束但在 $N > |D|$ 时数学上不可行；A-TPT 的 Tammes 问题框架天然处理这种情况
- **vs Wang & Isola 2020**: 他们证明了对比学习中均匀性的重要性；A-TPT 将此洞察应用到 TPT 校准中

## 评分
- 新颖性: ⭐⭐⭐⭐ Tammes 问题与 VLM 校准的连接是新颖的，但核心思想（最大化最小距离）相对直观
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集（含医学）、多 backbone、详尽的可视化分析和理论支撑
- 写作质量: ⭐⭐⭐⭐ 动机阐述充分，可视化有说服力
- 价值: ⭐⭐⭐⭐ 为 VLM 测试时校准提供了实用的改进方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] SoC: Semantic Orthogonal Calibration for Test-Time Prompt Tuning](../../CVPR2026/multimodal_vlm/soc_semantic_orthogonal_calibration_for_test-time_prompt_tuning.md)
- [\[CVPR 2026\] Improving Calibration in Test-Time Prompt Tuning for Vision-Language Models via Data-Free Flatness-Aware Prompt Pretraining](../../CVPR2026/multimodal_vlm/improving_calibration_in_test-time_prompt_tuning_for_vision-language_models_via_.md)
- [\[ICLR 2026\] Flatness-Guided Test-Time Adaptation for Vision-Language Models](flatness_guided_test-time_adaptation_for_vision-language_models.md)
- [\[CVPR 2026\] Towards Calibrating Prompt Tuning of Vision-Language Models](../../CVPR2026/multimodal_vlm/towards_calibrating_prompt_tuning_of_vision-language_models.md)
- [\[ICLR 2026\] Bilateral Information-aware Test-time Adaptation for Vision-Language Models](bilateral_information-aware_test-time_adaptation_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
