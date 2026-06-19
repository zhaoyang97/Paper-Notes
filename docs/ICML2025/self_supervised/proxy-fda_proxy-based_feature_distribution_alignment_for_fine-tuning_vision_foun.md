---
title: >-
  [论文解读] Proxy-FDA: Proxy-based Feature Distribution Alignment for Fine-tuning Vision Foundation Models without Forgetting
description: >-
  [ICML 2025][自监督学习][robust fine-tuning] 提出结构级特征正则化方法 Proxy-FDA：通过迁移预训练特征空间的最近邻图到微调特征空间，并用轻量代理生成器合成新特征增强分布覆盖，在不牺牲下游精度的前提下实现所有微调任务的正向迁移。 领域现状：CLIP、DINOv2 等视觉基础模型在预训练中…
tags:
  - "ICML 2025"
  - "自监督学习"
  - "robust fine-tuning"
  - "concept forgetting"
  - "feature distribution alignment"
  - "proxy learning"
  - "vision foundation model"
---

# Proxy-FDA: Proxy-based Feature Distribution Alignment for Fine-tuning Vision Foundation Models without Forgetting

**会议**: ICML 2025  
**arXiv**: [2505.24088](https://arxiv.org/abs/2505.24088)  
**代码**: 待确认  
**领域**: 自监督学习  
**关键词**: robust fine-tuning, concept forgetting, feature distribution alignment, proxy learning, vision foundation model

## 一句话总结

提出结构级特征正则化方法 Proxy-FDA：通过迁移预训练特征空间的最近邻图到微调特征空间，并用轻量代理生成器合成新特征增强分布覆盖，在不牺牲下游精度的前提下实现所有微调任务的正向迁移。

## 研究背景与动机

**领域现状**：CLIP、DINOv2 等视觉基础模型在预训练中学习了丰富的真实世界概念表征，但在下游任务微调后，往往会遗忘其他任务的概念识别能力——概念遗忘（concept forgetting）。

**现有痛点**：L2SP 在权重空间做 L2 正则，LDIFS 在特征空间做逐点匹配。然而逐点约束**过于严格且盲目**——未考虑特征邻域结构。特征局部邻域中编码了超越类标签的丰富知识（如两个狗品种共享的"白色"属性），逐点匹配无法保留这些结构性信息。

**核心矛盾**：逐点约束既过强（限制了特征移动自由度）又不够（未保护邻域结构编码的细粒度知识），导致知识保留不充分。

**本文目标** (1) 设计结构级特征正则化，保留特征分布的局部拓扑；(2) 解决下游数据有限导致 FDA 覆盖不足的问题。

**切入角度**：作者发现 OTDD（最优传输数据集距离，考虑局部结构）与概念遗忘的相关性远强于 L2 特征距离——理论上暗示保留分布结构比逐点匹配更有效。

**核心 idea**：用最近邻图迁移保留特征邻域的拓扑结构，并通过代理特征生成增强数据多样性，实现结构级的遗忘防御。

## 方法详解

### 整体框架

在标准微调损失 $\mathcal{L}_{\text{task}}$ 之外加入 FDA 正则项：$\mathcal{L} = \frac{1}{B}\sum_{i=1}^{B}(\mathcal{L}_{\text{task}}^i + \lambda\mathcal{L}_{\text{FDA}}^i)$。FDA 通过迁移预训练特征空间的 kNN 图到微调特征空间实现。Proxy-FDA 进一步引入代理生成器，用合成特征增强 FDA 覆盖。

### 关键设计

1. **Feature Distribution Alignment（FDA）**:

    - 功能：保留预训练特征空间中的局部邻域结构
    - 核心思路：对每个预训练特征点 $\hat{x}_i$，在 batch 内构建 $K$-近邻集 $R_i$ 和余弦相似度 $\hat{w}_{ij}$，将邻域索引和相似度直接迁移到微调特征空间。使用 Sigmoid 对比损失（源自 SigLIP）：$\mathcal{L}_{\text{FDA}}^i = \frac{1}{|X|-1}\sum_{j\neq i}\log(1+e^{w_{ij}(-\cos(x_i,x_j)/\tau+b)})$，其中 $w_{ij}$ 对邻居取正的预训练相似度，对非邻居取负值
    - 设计动机：FDA 保留的是**关系**而非**绝对位置**——允许特征在微调中移动，只要邻域拓扑不变。跨类邻域传递了超越标签的知识

2. **Batch 构建与 Hard Class Mining**:

    - 功能：确保 batch 内邻域足够丰富
    - 核心思路：类均衡采样（$m=16$ 类 × $n=4$ 样本/类 = 64 batch），hard class mining 优先选择特征空间中相近的类。$K > n$ 确保每个邻域包含多个类
    - 设计动机：$K > n$ 保证邻域跨越类边界，FDA 传递跨类知识（如"白色"跨不同白狗品种）。若跨类相似度不高，FDA 自动退化为类语义对齐

3. **Proxy 代理生成器**:

    - 功能：在数据有限场景中增强 FDA 的数据多样性
    - 核心思路：轻量网络（1 个注意力层 + 2 个卷积层，仅 23.6K 参数），以邻居集 $X_i^+$ 和非邻居集 $X_i^-$ 为条件，通过自适应池化生成两组代理特征 $P_i^+$、$P_i^-$ 及其估计相似度。代理学习损失包含约束代理位于真实特征流形上的对比项和鼓励多样性的方差损失 $\mathcal{L}_{\text{var}}$
    - 设计动机：下游数据有限时 batch 真实特征不足以描述复杂分布。代理合成未见的数据点——包括**未见类概念**——在邻域边界处提供细粒度正则化。在线联合训练确保代理适配当前特征分布

### 损失函数

Proxy-FDA 将代理拼接到真实特征和相似度中扩展 FDA：$\mathcal{L}_{\text{Proxy-FDA}}^i = \mathcal{L}_{\text{FDA}}^i(\{[X_i^+, P_i^+], [X_i^-, P_i^-]\}, \{[\hat{w}_i^+, \hat{w}_i^{p+}], [\hat{w}_i^-, \hat{w}_i^{p-}]\})$。代理生成器梯度来自 $\mathcal{L}_{\text{proxy}} = \mathcal{L}_{P_i^+} + \mathcal{L}_{P_i^-}$（各含对比项+方差项，权重 $\alpha$）。

## 实验关键数据

### End-to-end 微调（CLIP ViT-B/32，10 个分类数据集，Table 1）

| 方法 | 平均 $\mathcal{A}_{\text{LP}}$↑ | 平均 $\Delta_{\text{LP}}$↑ |
|------|------|------|
| Naive FT | 91.90 | -4.37 |
| LP-FT | 91.55 | -2.59 |
| L2SP（权重正则） | 90.69 | +0.29 |
| LDIFS（逐点特征正则） | 91.66 | +0.86 |
| FDA（结构正则） | 91.86 | +1.39 |
| **Proxy-FDA** | **91.82** | **+1.54** |

### Few-shot Prompt Tuning（CLIP ViT-B/16，11 数据集，16-shot，Table 2）

| Prompt 方法 | +Proxy-FDA | $\mathcal{A}_{\text{Base}}$ | $\mathcal{A}_{\text{New}}$ | $\Delta_{\text{New}}$↑ | $\mathcal{A}_H$ |
|------------|-----------|------|------|------|------|
| CoOp | ✗ | 82.69 | 63.22 | -10.99 | 71.66 |
| CoOp | ✓ | 83.16 | 73.67 | -0.55 | 78.13 |
| PromptSRC | ✗ | 84.26 | 76.10 | +1.88 | 79.97 |
| PromptSRC | ✓ | 84.47 | 77.45 | +3.23 | 80.81 |

### 关键发现

1. Proxy-FDA 在所有 10 个微调任务上实现正向迁移（$\Delta_{\text{LP}} > 0$），Naive FT 和 LP-FT 全为负
2. 结构级正则（Proxy-FDA +1.54）显著优于逐点正则（LDIFS +0.86）
3. Proxy-FDA 对 CoOp 的 $\mathcal{A}_{\text{New}}$ 提升 10.45 分（63.22→73.67）
4. OTDD 与遗忘的相关性强于 L2 距离——Proxy-FDA 有时 L2 更大但 OTDD 更小且遗忘更少
5. 跨架构有效：CLIP/FLAVA/DINOv2/MAE 均保持正向迁移
6. 计算开销可控：FDA +7-9%，Proxy-FDA +17-21% 微调时间，推理无开销

## 亮点与洞察

- **结构级 vs 点级正则的质变**：从保留"特征位置"到保留"特征关系"，概念升级带来一致实验提升
- **代理是分布增强工具而非样本替代品**：不同于度量学习中的 proxy（类原型），这里是实例级合成特征，目的是增加邻域边界处的数据密度
- **OTDD 相关性分析**为结构对齐方法提供了理论依据
- **普适性强**：end-to-end、few-shot、continual learning、captioning、VQA 多种 setting 均有效

## 局限性

- kNN 图质量依赖 batch 构建策略，对极端类不平衡可能退化
- 代理仅从当前 batch 生成，未利用外部数据或记忆库
- 邻域大小 $K$ 和类数 $m$ 需针对数据集调整
- 超大规模模型（ViT-L/ViT-G）上的验证尚缺

## 相关工作与启发

- LDIFS（Mukhoti et al., 2024）：直接改进的逐点特征正则基线
- 关系知识蒸馏（Park et al., 2019 RKD）：Proxy-FDA 可看作面向鲁棒微调的关系蒸馏
- SigLIP（Zhai et al., 2023）：FDA 损失函数来源，Sigmoid 形式支持可变正负样本数
- OTDD（Alvarez-Melis & Fusi, 2020）：遗忘诊断指标，其与遗忘的强相关性支撑了结构对齐的合理性

## 评分

⭐⭐⭐⭐ — 结构级特征对齐思路新颖有效，从逐点到邻域结构的概念升级清晰优雅。OTDD 相关性分析有洞察力，代理生成器设计轻巧实用。实验覆盖 end-to-end/few-shot/continual/多模态，说服力强。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models](../../ICCV2025/self_supervised/loftup_learning_a_coordinatebased_feature_upsampler_for_visi.md)
- [\[NeurIPS 2025\] Implicit Modeling for Transferability Estimation of Vision Foundation Models](../../NeurIPS2025/self_supervised/implicit_modeling_for_transferability_estimation_of_vision_foundation_models.md)
- [\[ICML 2025\] Towards Benchmarking Foundation Models for Tabular Data With Text](towards_benchmarking_foundation_models_for_tabular_data_with_text.md)
- [\[ICML 2025\] Test-Time Canonicalization by Foundation Models for Robust Perception](test-time_canonicalization_by_foundation_models_for_robust_perception.md)
- [\[ICML 2025\] Foundation Model Insights and a Multi-Model Approach for Superior Fine-Grained One-shot Subset Selection](foundation_model_insights_and_a_multi-model_approach_for_superior_fine-grained_o.md)

</div>

<!-- RELATED:END -->
