---
title: >-
  [论文解读] BCWildfire: A Long-term Multi-factor Dataset and Deep Learning Benchmark for Boreal Wildfire Risk Prediction
description: >-
  [AAAI 2026][LLM评测][野火风险预测] 本文构建了一个覆盖加拿大BC省2.4亿公顷、跨度25年的多模态野火风险预测数据集BCWildfire，包含38个驱动因子，并对CNN/Linear/Transformer/Mamba四大范式的时序预测模型进行了系统评测，揭示了当前模型在野火预测中的性能上限和关键影响因子。
tags:
  - "AAAI 2026"
  - "LLM评测"
  - "野火风险预测"
  - "时间序列预测"
  - "多模态数据集"
  - "深度学习基准"
  - "北方森林"
---

# BCWildfire: A Long-term Multi-factor Dataset and Deep Learning Benchmark for Boreal Wildfire Risk Prediction

**会议**: AAAI 2026  
**arXiv**: [2511.17597](https://arxiv.org/abs/2511.17597)  
**代码**: [https://github.com/SynUW/BCWildfire](https://github.com/SynUW/BCWildfire)  
**领域**: LLM评测  
**关键词**: 野火风险预测, 时间序列预测, 多模态数据集, 深度学习基准, 北方森林

## 一句话总结

本文构建了一个覆盖加拿大BC省2.4亿公顷、跨度25年的多模态野火风险预测数据集BCWildfire，包含38个驱动因子，并对CNN/Linear/Transformer/Mamba四大范式的时序预测模型进行了系统评测，揭示了当前模型在野火预测中的性能上限和关键影响因子。

## 研究背景与动机

野火在全球范围内的频率、规模和强度不断升级，尤其在碳储量丰富的北方森林（boreal region）中风险尤为严峻。数据驱动方法（特别是深度学习）在野火预测中展现了潜力，但部署受制于高质量数据集的匮乏。

现有数据集存在三大痛点：

**地理覆盖有限**：多数数据集聚焦美国和地中海区域的局部火灾蔓延建模，缺乏对碳密集型北方生态系统的覆盖

**时间跨度不足**：绝大多数数据集仅支持1天或7天的短期回溯窗口，无法建模由燃料累积、持续干旱等长期因素驱动的大尺度野火风险

**驱动因子不全**：异质火灾驱动因子（气象、植被、地形、人类活动）的整合仍然不充分

核心矛盾在于：野火风险是长期累积的多因子交互过程，但现有数据集和模型主要关注短期、局部的火灾蔓延预测。本文的切入角度是构建一个长时间序列、大空间覆盖、多模态的标准化数据集，将野火风险预测重新定义为时间序列预测问题，从而利用时序预测领域的最新进展。

## 方法详解

### 整体框架

BCWildfire是一个时间序列预测格式的数据集基准，包含数据构建和模型评测两部分。数据覆盖BC省及周边区域（2782×1302 km²），时间跨度2000-2024年，空间分辨率1km，时间分辨率每天，包含38个多模态协变量。

### 关键设计

1. **多模态驱动因子设计（38个变量）**:

    - 功能：整合5大类野火驱动因子
    - 核心思路：燃料条件（LAI/FPAR/NDVI/EVI等MODIS产品）、气象因子（ERA5-Land温度/降水/风速/土壤湿度+MODIS热辐射产品）、地形因子（ASTER DEM的坡度/坡向/山影+水体距离）、人类活动（MODIS土地利用+距基础设施距离）、火灾检测（MOD/MYD14A1主动火产品）
    - 设计动机：ERA5-Land空间分辨率粗（~11km），不能捕捉热异常细节，因此额外结合MODIS 1km热辐射产品补充；深层土壤湿度（0-289cm）被纳入以反映长时间尺度的环境湿度累积效应

2. **数据预处理与类不平衡策略**:

    - 功能：统一坐标系、分辨率，处理云遮挡和类不平衡
    - 核心思路：所有数据统一到WGS84坐标系和1km/日分辨率；用QC波段+历史填充处理云遮挡（避免数据泄露）；采用欠采样策略，排除正样本周围60km/3天缓冲区内的负样本，按土地覆盖类型保持正负样本比例（训练集1:2，测试集1:1）
    - 设计动机：野火是极端稀有事件，直接训练会导致严重的类不平衡；随机选取负样本可能采到高风险区域，需要空间缓冲排除

3. **时序预测基准评测**:

    - 功能：评测6个SOTA时序模型的次日野火风险预测性能
    - 核心思路：以前10天的驱动因子为输入、预测次日野火发生概率，将其建模为二分类时序预测问题
    - 设计动机：将野火预测从传统的空间CNN Task转向时间序列预测范式，利用长期累积因子的时序建模能力

### 损失函数 / 训练策略

使用二元交叉熵损失（BCE Loss）训练所有模型。Adam优化器，2张NVIDIA A6000 GPU，batch size 128，训练50 epoch，学习率1×10⁻⁵。数据划分：2000-2020训练，2021-2022验证，2023-2024测试。

## 实验关键数据

### 主实验

| 模型 | 类型 | Precision | Recall | F1 | PR-AUC |
|------|------|-----------|--------|------|--------|
| SCINet | CNN | 84.77 | 88.05 | 86.38 | 94.46 |
| TSMixer | Linear | 85.69 | 90.39 | 87.97 | 96.24 |
| CrossLinear | Linear | 88.04 | 87.59 | 87.81 | 96.07 |
| Crossformer | Transformer | 88.74 | 87.49 | 88.11 | 96.28 |
| FEDformer | Transformer | 82.95 | 91.18 | 86.87 | 94.93 |
| S_Mamba | Mamba | 84.21 | 86.44 | 85.31 | 94.83 |

### 消融实验（位置编码效果）

| 模型 | 无位置编码 F1 | 有位置编码 F1 | Recall提升 |
|------|-------------|-------------|-----------|
| Crossformer | 88.11 | 88.70 | +2.03% |
| FEDformer | 86.87 | 87.43 | -1.36% |
| S_Mamba | 85.31 | 87.46 | +3.65% |
| TSMixer | 87.97 | 88.12 | -1.10% |

### 关键发现

- **性能天花板**：即使最佳模型recall也低于92%，precision在83-88%的狭窄区间内，说明野火预测的内在随机性和类不平衡是根本挑战
- **Transformer优势**：Crossformer和FEDformer在recall和稳定性上优于CNN和Mamba模型，反映其建模长程时序依赖的优势
- **SHAP因子分析**：火灾检测信号最重要（负SHAP表示无火信号指示潜在新着火），其次是28-100cm土壤湿度和地表潜热通量，雪覆盖有显著负影响（抑制着火）
- **空间位置编码**有效提升多数模型性能，S_Mamba recall提升3.65%

## 亮点与洞察

- 将野火风险预测重新定义为时序预测问题是一个有价值的视角转换，使得时序预测领域的大量模型可以直接应用
- 25年日分辨率、38变量、240M公顷的数据集规模在野火领域具有显著优势
- SHAP分析揭示了一个反直觉发现：着火区域往往具有略高的深层（100-289cm）土壤含水量，说明野火常发生在生态上湿润、燃料丰富的环境中

## 局限与展望

- 当前仅评测了次日预测，未充分利用25年长期数据进行更长期的趋势建模
- 数据集为BC省区域特定，模型泛化到其他北方森林区域的能力未验证
- 仅使用时序信息，未结合空间邻域信息（如CNN的空间建模能力），可能丢失局部空间传播模式
- 合成(CARLA)数据集可与真实气象数据结合，但二者之间可能存在gap

## 相关工作与启发

- 与SeasFire Cube等现有数据集相比，BCWildfire的关键差异在于支持长回溯窗口的时序预测
- Mamba模型（S_Mamba）在精度和效率之间取得了良好平衡，适合实际部署
- 未来可探索时空联合建模（如将BEV范式引入野火预测）和集成预测方法

## 评分
- 新颖性: ⭐⭐⭐⭐ （数据集贡献大，方法层面无新模型）
- 实验充分度: ⭐⭐⭐⭐ （评测全面但缺少与传统方法的对比）
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐ （填补北方森林长期野火预测数据集空白）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction](../../ACL2026/llm_evaluation/sciimpact_a_multi-dimensional_multi-field_benchmark_for_scientific_impact_predic.md)
- [\[NeurIPS 2025\] MEMTRACK: Evaluating Long-Term Memory and State Tracking in Multi-Platform Dynamic Agent Environments](../../NeurIPS2025/llm_evaluation/memtrack_evaluating_long-term_memory_and_state_tracking_in_multi-platform_dynami.md)
- [\[AAAI 2026\] DiCaP: Distribution-Calibrated Pseudo-labeling for Semi-Supervised Multi-Label Learning](dicap_distribution-calibrated_pseudo-labeling_for_semi-supervised_multi-label_le.md)
- [\[ICLR 2026\] Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs](../../ICLR2026/llm_evaluation/beyond_a_million_tokens_benchmarking_and_enhancing_long-term_memory_in_llms.md)
- [\[ICLR 2026\] AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning](../../ICLR2026/llm_evaluation/anessuite_a_comprehensive_benchmark_and_dataset_suite_for_anesthesiology_reasoni.md)

</div>

<!-- RELATED:END -->
