---
title: >-
  [论文解读] The Boundaries of Fair AI in Medical Image Prognosis: A Causal Perspective
description: >-
  [NeurIPS 2025][医学图像][公平性] FairTTE是首个系统研究医学影像中时间-事件(TTE)预测公平性的综合框架，利用因果分析量化五种偏差来源，通过训练超过20000个模型揭示了现有公平性方法的局限性，特别是在分布偏移下公平性难以维持的根本挑战。 机器学习模型在医学影像分析中日益广泛应用…
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "公平性"
  - "时间-事件预测"
  - "因果分析"
  - "医学影像"
  - "分布偏移"
---

# The Boundaries of Fair AI in Medical Image Prognosis: A Causal Perspective

**会议**: NeurIPS 2025  
**arXiv**: [2510.08840](https://arxiv.org/abs/2510.08840)  
**代码**: [https://github.com/pth1993/FairTTE](https://github.com/pth1993/FairTTE)  
**领域**: 医学影像 / AI公平性  
**关键词**: 公平性, 时间-事件预测, 因果分析, 医学影像, 分布偏移

## 一句话总结
FairTTE是首个系统研究医学影像中时间-事件(TTE)预测公平性的综合框架，利用因果分析量化五种偏差来源，通过训练超过20000个模型揭示了现有公平性方法的局限性，特别是在分布偏移下公平性难以维持的根本挑战。

## 研究背景与动机

机器学习模型在医学影像分析中日益广泛应用，但模型可能对特定社会群体产生偏见。现有公平性研究主要集中在**诊断任务**（分类、分割），而忽视了**预后任务**——预测疾病的结局或进展时间（即时间-事件预测/生存分析）。

TTE预测中的公平性面临独特挑战：

**数据稀缺**：缺乏同时包含医学图像、时间-事件结局和敏感属性的公开数据集

**偏差机制不明**：不了解医学影像中的偏差如何具体影响TTE预测的公平性

**缺乏统一指标**：TTE预测没有被广泛接受的公平性度量标准

**删失问题**：TTE数据中广泛存在的右删失(right-censoring)使标准评估指标不适用

核心切入角度：用结构因果模型(SCM)刻画TTE数据的生成过程，系统分解偏差来源，理解为什么现有方法经常失败，从因果视角揭示公平TTE预测的根本边界。

## 方法详解

### 整体框架

FairTTE框架包含三个层次：(1) 基于因果推理的偏差分析理论框架；(2) 跨模态的大规模医学影像数据集集合；(3) 集成SOTA TTE预测模型和公平性算法的评估管线。

### 关键设计

1. **因果结构模型(Causal Structure)**：

    - 引入未观测的潜在健康状态 $Z$，将特征 $X$ 分解为：$X_Z$（与Z相关的目标特征）和 $X_A$（与敏感属性A相关的特征）
    - 无偏场景：$A$ 仅影响 $X_A$，不影响其他变量，学习不变特征 $X_Z$ 即可实现公平
    - 有偏场景：$A$ 可能通过多条因果路径影响 $T$、$C$、$X_Z$ 等，导致 $P(t|x_z, a) \neq P(t|x_z, a')$
    - 因果图清晰地揭示了"为何学习不变表征不总是足够"的原因

2. **五种偏差来源的量化分解**：

    - 基于贝叶斯分解，将标签函数分解为：$P(y,\delta|x,a) = \underbrace{PMI(x_z,y)}_{图像-时间互信息} \cdot \underbrace{PMI(x_z,\delta)}_{图像-删失互信息} \cdot \underbrace{P(y|\delta,a)}_{TTE分布} \cdot \underbrace{P(\delta|a)}_{删失率}$
    - 五种偏差来源：(i) 图像特征分布差异 (ii) $X_Z$与$Y$互信息差异 (iii) $X_Z$与$\Delta$互信息差异 (iv) TTE分布差异 (v) 删失率差异
    - 每种偏差可独立量化（Wasserstein距离、归一化互信息等）

3. **公平性理论分析**：

    - **Theorem 1**：公平性误差上界为 $\mathcal{F}_{Er}(h) \leq \max_{a,a'}(\eta(\mathcal{H},f_a,f_{a'}) + \mathcal{D}(\mathcal{H},D_a,D_{a'}))$
    - 其中 $\eta$ 是最小联合预测误差（当标签函数 $f_a \neq f_{a'}$ 时难以减小），$\mathcal{D}$ 是子群分布距离（可通过公平表征学习减小）
    - **Proposition 2**：在协变量偏移条件下，若表征Z对目标充分，则学习公平表征可实现公平TTE预测
    - 核心洞察：当因果路径使不同群体的标签函数本身不同时，公平变得根本困难

### 损失函数 / 训练策略
- TTE预测模型：DeepHit（离散时间竞争风险）、Nnet-survival、PMF
- 公平性算法：5种SOTA方法覆盖三类策略——预处理(SR)、训练中(DI, FRL, DRO)、后处理(CSA)
- 模型选择：在验证集上选择公平性最优模型，允许预测性能最多下降5%
- 骨干网络：2D EfficientNet(AREDS/MIMIC-CXR)、3D ResNet-18(ADNI)
- 10个随机种子确保稳定性

## 实验关键数据

### 主实验

**跨数据集/敏感属性的性能差距(Ct^d)**

| 数据集-敏感属性 | DeepHit最优组 | DeepHit最差组 | 差距 | 最佳公平算法效果 |
|------|------|------|----------|------|
| ADNI-Age | ~80% | ~66% | 14.19 | DI: -66.22% (差距) |
| AREDS-Race | ~88% | ~77% | 11.09 | DRO: -37.21% (差距) |
| MIMIC-CXR-Age | ~79% | ~73% | 5.93 | DRO: -95.74% (差距) |
| AREDS-Sex | ~82% | ~81% | 1.32 | SR: -85.27% (差距) |

### 消融实验

| 实验设置 | 关键发现 | 说明 |
|------|---------|------|
| 预训练 vs 从头训 | 预训练提高精度(尤其ADNI)，但不改善公平性 | 18/24设置中p>0.05 |
| 分布偏移(Y噪声) | IBS显著恶化，排序指标影响小 | 直接影响误差度量 |
| 分布偏移(Δ翻转) | Ct^d和AUCt^d严重恶化 | 破坏可比对数量 |
| 分布偏移(X噪声) | 所有指标恶化 | 特征质量下降 |
| 偏差量化相关性 | 高偏差度 ↔ 大性能差距 | AREDS-race最典型 |

### 关键发现
- **偏差无处不在**：所有数据集/敏感属性组合中均观察到显著的组间性能差异
- **年龄和种族的偏差比性别更严重**：跨所有数据集一致
- **现有公平方法效果有限**：没有方法在所有设置中一致优于基础模型DeepHit；在某些设置中甚至加剧不公平
- **公平性与准确性的权衡**：公平性提升常伴随预测准确性下降
- **预训练无法自动改善公平性**：提高精度但公平性指标基本不变
- **分布偏移下公平性更难维持**：不同类型的偏移以不同方式影响公平性
- 偏差来源的量化结果与模型不公平程度高度相关，验证了因果分解框架的实用性

## 亮点与洞察
- 首次系统地将因果分析引入医学影像TTE预测的公平性研究
- 五种偏差来源的分解为理解"为何不公平"提供了精细化的诊断工具
- Theorem 1清晰地解释了为什么在标签函数本身依赖敏感属性时（如年龄确实影响存活时间），公平性有根本的不可实现性
- 大规模实验（20000+模型）确保了结论的可靠性
- 公平性与分布偏移的内在联系是深刻的洞察：实现公平等价于让模型泛化到只有公平因果路径的测试分布

## 局限与展望
- 仅考虑单一风险和非信息性右删失，未覆盖竞争风险和信息性删失
- 采用组公平性定义（组间性能差距），未考虑个体公平性
- 在某些临床场景中，敏感属性确实是重要的风险因素（如年龄与死亡率），严格的组公平性可能不适合
- 因果图中的因果方向需要临床专家的领域知识来确定
- 未探索如何区分"公平的因果路径"与"不公平的因果路径"

## 相关工作与启发
- MedFair(Zong et al., 2023)和FairSeg(Tian et al., 2024)关注诊断任务的公平性
- 本文将公平性研究扩展到预后领域，填补了重要空白
- 因果推理的视角为公平性提供了比纯统计方法更深层的理解
- 分布偏移下的公平性是一个重要但未被充分研究的方向
- 启发方向：开发能同时解决多种偏差来源、且在分布偏移下鲁棒的公平算法

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统研究TTE预测公平性，因果分解框架新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 3数据集×多敏感属性×3模型×5算法×10种子，规模惊人
- 写作质量: ⭐⭐⭐⭐ 框架清晰，理论与实验紧密结合
- 价值: ⭐⭐⭐⭐⭐ 为医学影像公平性研究开辟了新方向，洞察深刻且可指导实践

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] FairGRPO: Fair Reinforcement Learning for Equitable Clinical Reasoning](fairgrpo_fair_reinforcement_learning_for_equitable_clinical_reasoning.md)
- [\[AAAI 2026\] DualFete: Revisiting Teacher-Student Interactions from a Feedback Perspective for Semi-supervised Medical Image Segmentation](../../AAAI2026/medical_imaging/dualfete_revisiting_teacher-student_interactions_from_a_feedback_perspective_for.md)
- [\[NeurIPS 2025\] Doctor Approved: Generating Medically Accurate Skin Disease Images through AI-Expert Feedback](doctor_approved_generating_medically_accurate_skin_disease_images_through_ai-exp.md)
- [\[NeurIPS 2025\] Dynamic Causal Discovery in Alzheimer's Disease through Latent Pseudotime Modelling](dynamic_causal_discovery_in_alzheimers_disease_through_latent_pseudotime_modelli.md)
- [\[CVPR 2025\] CholecTrack20: A Multi-Perspective Tracking Dataset for Surgical Tools](../../CVPR2025/medical_imaging/cholectrack20_a_multi-perspective_tracking_dataset_for_surgical_tools.md)

</div>

<!-- RELATED:END -->
