---
title: >-
  [论文解读] CLIMB: Class-Imbalanced Learning Benchmark on Tabular Data
description: >-
  [NeurIPS 2025][预训练][class imbalance] 提出 Climb——迄今最全面的表格数据类别不平衡学习基准，涵盖 73 个真实数据集和 29 种 CIL 算法，通过大规模实验揭示了朴素重平衡往往无效、集成方法至关重要、数据质量比不平衡本身更影响性能等实用洞察。 - 类别不平衡是表格数据的核心挑战：金…
tags:
  - "NeurIPS 2025"
  - "预训练"
  - "class imbalance"
  - "tabular data"
  - "benchmark"
  - "ensemble learning"
  - "resampling"
---

# CLIMB: Class-Imbalanced Learning Benchmark on Tabular Data

**会议**: NeurIPS 2025  
**arXiv**: [2505.17451](https://arxiv.org/abs/2505.17451)  
**代码**: [ZhiningLiu1998/imbalanced-ensemble](https://github.com/ZhiningLiu1998/imbalanced-ensemble)  
**领域**: LLM评测  
**关键词**: class imbalance, tabular data, benchmark, ensemble learning, resampling

## 一句话总结

提出 Climb——迄今最全面的表格数据类别不平衡学习基准，涵盖 73 个真实数据集和 29 种 CIL 算法，通过大规模实验揭示了朴素重平衡往往无效、集成方法至关重要、数据质量比不平衡本身更影响性能等实用洞察。

## 研究背景与动机

- **类别不平衡是表格数据的核心挑战**：金融欺诈检测、网络入侵识别、医疗诊断等关键场景中，少数类代表罕见但重要的结果，标准分类器在此场景下表现退化严重
- **现有基准碎片化严重**：已有研究大多局限于特定领域（商业/金融/医疗/教育），数据集不平衡程度相似，算法覆盖面窄，缺乏跨范式（欠采样/过采样/代价敏感/集成）的系统比较
- **表格数据有独特难点**：与图像/文本不同，表格数据特征异质、样本量小、缺乏局部相关性，树模型仍是主流选择；不平衡进一步加剧少数类泛化困难
- **方法选择缺乏指导**：实践者面对数十种 CIL 方法时缺少可靠的选择依据，不同评价指标可能导致矛盾结论
- **效率与鲁棒性未被系统研究**：已有工作聚焦准确率，较少分析运行时间和噪声/缺失值等数据质量因素的影响
- **缺乏高质量开源工具**：此前没有统一 API、完善文档、严格测试覆盖的 CIL benchmark 开源库

## 方法详解

### 整体框架：Climb Benchmark
构建一个包含数据集、算法、评估协议三位一体的综合基准平台，配套高质量 Python 开源库（统一 scikit-learn 风格 API、95% 测试覆盖率、详细文档），支持公平、可复现、可扩展的 CIL 方法评估。

### 关键设计一：73 个真实不平衡表格数据集
- **功能**：从 OpenML 精心筛选 73 个自然类别不平衡数据集，覆盖多领域、不平衡比率从 2.1 到 577.9
- **核心思路**：设定七项严格筛选标准——真实数据且自然不平衡、有学习难度（排除 AUPRC > 0.95 的简单集）、IR > 2、无缺失值、满足 i.i.d. 假设、非确定性函数、有文档记录
- **设计动机**：排除人工构造和简单数据集，确保评测能真实反映实际应用中的挑战；按 IR 分为 low/medium/high/extreme 四档便于分层分析

### 关键设计二：29 种 CIL 算法的统一实现
- **功能**：实现涵盖六大范式的 29 种代表性算法——欠采样（RUS/CC/IHT/NearMiss）、清洗（Tomek Links/ENN/RENN/AllKNN/OSS/NCR）、过采样（ROS/SMOTE/BorderlineSMOTE/SVMSMOTE/ADASYN）、欠采样集成（SPE/BC/BRF/EE/RUSBoost/UnderBagging）、过采样集成（OverBoost/SMOTEBoost/OverBagging/SMOTEBagging）、代价敏感集成（CS/AdaCost/AdaUBoost/AsymBoost）
- **核心思路**：统一 API 设计 + 层次化模块抽象，通过继承和多态支持便捷扩展
- **设计动机**：消除不同实现之间的不公平比较，提供首个跨越全部主流 CIL 范式的统一对比

### 关键设计三：严格的评估协议
- **功能**：标准化预处理（数值特征标准化、类别特征编码）、5 折分层划分、Optuna 超参搜索（每个算法-数据集对 100 次试验）、三种评价指标（AUPRC/macro-F1/BAC）
- **核心思路**：决策树作为统一基分类器、集成大小统一为 100，确保不同方法间的公平对比
- **设计动机**：避免单次随机划分带来的偶然性；多指标评价揭示不同侧面（AUPRC 重视精确率、BAC 重视召回率平衡），防止单一指标带来的误导性结论

### 关键设计四：鲁棒性控制实验
- **功能**：分别引入标签噪声（minority 类翻转 10%/20%/30%）、缺失值（均值填补 10%/20%/30%）、额外不平衡（进一步移除 minority 样本使 IR 翻倍/3倍/5倍）
- **核心思路**：逐一引入单一干扰因素，分离各因素对 CIL 性能的影响
- **设计动机**：实际数据往往同时伴随噪声和缺失值，需要了解这些因素相对于不平衡本身的影响程度

## 损失函数与训练

以决策树为基分类器；集成方法统一 100 棵基学习器；代价敏感方法按类频率反比设置误分类代价。使用 Optuna 进行贝叶斯超参搜索，每个配置 100 次试验。整个基准涉及约 80 万次超参搜索试验、超 1000 万个基模型的训练。

## 实验关键数据

### Table 2：主基准结果（按不平衡程度分组的 AUPRC/F1/BAC）

| 不平衡分组 | Base AUPRC | 最优方法 | 最优 AUPRC | 提升 |
|---|---|---|---|---|
| Low (IR<5, 28集) | 51.0 | SPE | 59.3 | +8.3 |
| Medium (IR∈[5,10), 24集) | 50.9 | SPE | 64.6 | +13.7 |
| High (IR∈[10,50), 15集) | 34.9 | SPE | 47.1 | +12.2 |
| Extreme (IR>50, 6集) | 42.6 | SPE | 57.5 | +14.9 |

核心发现：**Self-paced Ensemble (SPE) 在所有不平衡等级的 AUPRC 上均排名第一或前二**；朴素欠采样（RUS/CC/NearMiss）在多数场景下反而降低性能；集成方法整体显著优于非集成方法。

### 鲁棒性分析关键发现

| 干扰因素 | 等效影响 |
|---|---|
| 10% 标签噪声 | ≈ 500% IR 增加带来的性能下降 |
| 30% 缺失特征 | ≈ 500% IR 增加带来的性能下降 |

这一发现表明**数据质量可能比类别不平衡本身更重要**。

## 亮点

- **覆盖面最全**：73 个数据集 × 29 种算法 × 6 大范式，是该领域迄今最大规模的系统评测
- **五条实用洞察高度凝练**：(1) 朴素重平衡经常有害 (2) 集成是 CIL 的关键 (3) 评价指标选择影响结论 (4) 欠采样集成在性能-效率上最佳平衡 (5) 数据质量可能比不平衡更重要
- **开源库质量高**：统一 API、95% 测试覆盖、详细文档，具有真实工程价值
- **数据质量 vs 不平衡的量化对比**是新颖且实用的发现

## 局限性

- 基分类器仅使用决策树，未验证深度学习模型（如 TabNet、FT-Transformer）在 CIL 场景下的表现
- 排除了含缺失值的数据集，限制了对真实"脏数据"场景的覆盖
- 鲁棒性实验中各干扰因素独立引入，未研究噪声+缺失+不平衡同时存在的复合效应
- 仅考虑二分类和传统多分类，未涵盖多标签不平衡和开放集等更复杂场景
- 数据集均来自 OpenML，可能存在选择偏差

## 与相关工作的对比

与 Zhu et al. (2018)、Xiao et al. (2021)、Khushi et al. (2021)、Kim & Hwang (2022) 等已有经验研究相比，Climb 在三个维度上全面超越：(1) 算法数量从 4-21 扩展到 29 且覆盖全部主流范式（首次纳入代价敏感）；(2) 数据集从 2-31 扩展到 73 且跨多领域多 IR；(3) 首次提供配套高质量开源工具。与 TableShift 等表格数据基准相比，Climb 专注于类别不平衡这一正交挑战。

## 评分

- 新颖性: ⭐⭐⭐ — 方法层面无新算法，但基准构建全面、洞察有价值
- 实验充分度: ⭐⭐⭐⭐⭐ — 80 万次超参搜索、1000 万基模型、多维度分析，极其充分
- 写作质量: ⭐⭐⭐⭐ — 结构清晰、RQ 驱动的分析框架好，表格信息密度高
- 价值: ⭐⭐⭐⭐ — 对 CIL 研究者和实践者都有直接参考意义，开源库有持续影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] ZEUS: Zero-shot Embeddings for Unsupervised Separation of Tabular Data](zeus_zero-shot_embeddings_for_unsupervised_separation_of_tabular_data.md)
- [\[NeurIPS 2025\] Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training](nemotron-climb_clustering-based_iterative_data_mixture_bootstrapping_for_languag.md)
- [\[NeurIPS 2025\] LCDB 1.1: A Database Illustrating Learning Curves Are More Ill-Behaved Than Previously Thought](lcdb_11_a_database_illustrating_learning_curves_are_more_ill-behaved_than_previo.md)
- [\[CVPR 2025\] Precise Event Spotting in Sports Videos: Solving Long-Range Dependency and Class Imbalance](../../CVPR2025/llm_pretraining/precise_event_spotting_in_sports_videos_solving_long-range_dependency_and_class_.md)
- [\[NeurIPS 2025\] Composition and Alignment of Diffusion Models using Constrained Learning](composition_and_alignment_of_diffusion_models_using_constrai.md)

</div>

<!-- RELATED:END -->
