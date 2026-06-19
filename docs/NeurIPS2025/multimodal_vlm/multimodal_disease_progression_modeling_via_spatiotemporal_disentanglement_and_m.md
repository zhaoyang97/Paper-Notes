---
title: >-
  [论文解读] Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement and Multiscale Alignment
description: >-
  [NeurIPS 2025 Spotlight][多模态VLM][疾病进展建模] 提出 DiPro 框架，通过区域感知的时空解耦（分离静态解剖与动态病理特征）和多时间尺度对齐（局部-全局融合 CXR 与 EHR），解决了纵向胸部X光序列的冗余问题和跨模态时间错位挑战，在疾病进展识别和 ICU 预测任务上达到 SOTA。
tags:
  - "NeurIPS 2025 Spotlight"
  - "多模态VLM"
  - "疾病进展建模"
  - "多模态融合"
  - "时空解耦"
  - "纵向CXR"
  - "电子健康记录"
---

# Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement and Multiscale Alignment

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2510.11112](https://arxiv.org/abs/2510.11112)  
**代码**: [GitHub](https://github.com/Chenliu-svg/DiPro)  
**领域**: 医学图像  
**关键词**: 疾病进展建模, 多模态融合, 时空解耦, 纵向CXR, 电子健康记录

## 一句话总结

提出 DiPro 框架，通过区域感知的时空解耦（分离静态解剖与动态病理特征）和多时间尺度对齐（局部-全局融合 CXR 与 EHR），解决了纵向胸部X光序列的冗余问题和跨模态时间错位挑战，在疾病进展识别和 ICU 预测任务上达到 SOTA。

## 研究背景与动机

纵向多模态临床数据对疾病进展建模至关重要，但面临两个核心挑战：

**临床图像序列的冗余性**：连续胸部X光（CXR）中静态解剖结构（如慢性心脏扩大、稳定的骨骼畸形）占据主导，掩盖了临床更重要的微妙病理变化（如新发浸润、演变中的水肿）。现有方法（如 CheXRelNet、SDPL）将所有成像特征统一处理，没有区分长期解剖基线与演变中的病理变化

**跨模态时间错位**：EHR 提供连续高频测量（如每小时生命体征），而 CXR 仅提供稀疏、不规则的快照，存在内在的时间粒度不匹配。现有多模态方法（如 MedFuse、DrFuse）仅使用最新一张 CXR，丢弃了纵向信息；纵向方法（如 UTDE、UMSE）依赖刚性插值或固定时间嵌入，缺乏自适应的跨模态对齐机制

DiPro 基于两个关键观察：(1) CXR 序列中的疾病进展通过局部区域的病理变化展现；(2) EHR 和影像数据在不同时间粒度上呈现互补的动态。

## 方法详解

### 整体框架

DiPro 包含三个模块：时空解耦（STD）从连续 CXR 对中分离静态解剖与动态病理特征；进展感知增强（PAE）通过反转 CXR 顺序来强化动态特征的方向敏感性；多尺度多模态融合（MMF）在局部（时间间隔级）和全局（序列级）两个尺度对齐 CXR 与 EHR。

### 关键设计

1. **时空解耦（STD）**：对每个解剖区域 $\mathbf{R}_{t_i}^r$，使用预训练 ResNet-50 提取特征后拼接相邻帧特征，分别通过静态投影头 $f_s$ 和动态投影头 $f_d$ 得到 $\mathbf{S}_i^r$ 和 $\mathbf{D}_i^r$。通过两个辅助损失约束解耦质量：(a) **正交解耦损失**减小静态与动态特征的余弦相似度：$\mathcal{L}_{\text{orth}} = \frac{1}{(T-1)R}\sum_i\sum_r (\text{sim}(\mathbf{S}_i^r, \mathbf{D}_i^r))^2$；(b) **时序一致性损失**约束静态特征在时间上保持稳定：$\mathcal{L}_{\text{temp}} = \frac{1}{N}\sum_r\sum_i \|\mathbf{S}_i^r - \mathbf{S}_{i+1}^r\|_2^2$。设计动机：不同临床语义的特征混合会稀释进展信号，显式分离可让模型聚焦于有意义的病理变化。

2. **进展感知增强（PAE）**：核心思想简洁而巧妙——反转 CXR 对的输入顺序应该反转进展方向但保持静态信息不变。通过反转输入得到 $\widetilde{\mathbf{D}}_i^r$ 和 $\widetilde{\mathbf{S}}_i^r$，并用 $K$ 个疾病特定分类头预测进展状态：原始方向预测 $y_i^{r,k}$，反转方向预测 $-y_i^{r,k}$。训练损失为：$\mathcal{L}_{\text{PAE}} = \sum_{r,k}[\text{CE}(\hat{y}, y) + \text{CE}(\tilde{y}, -y)] + \lambda_{\text{static}}\sum_r \|\mathbf{S}_i^r - \widetilde{\mathbf{S}}_i^r\|_2^2$。设计动机：通过时间反转的等变约束，强制动态特征编码进展方向，同时进一步验证静态特征的时间不变性。

3. **多尺度多模态融合（MMF）**：分三个层次融合 CXR 和 EHR：

    - **局部融合**：对每个 CXR 时间间隔 $[t_i, t_{i+1}]$，设计相对时间嵌入 $T_{t_j} = f_{\text{TE}}([t_j - t_i, t_{i+1} - t_j, \sigma((t_j-t_i)(t_{i+1}-t_j))])$，通过带中心聚焦注意力掩码的交叉注意力从全局 EHR 表示中提取间隔特定的 EHR 特征 $\mathbf{E}_i^{\text{local}}$，再与动态 CXR 特征做交叉注意力融合
    - **全局融合**：收集所有局部融合特征，通过交叉注意力让全局 EHR 表示关注所有时间间隔的融合特征，再做自注意力增强
    - **静态融合**：将人口学信息与静态 CXR 特征拼接，通过交叉注意力与动态/全局特征融合生成最终预测。设计动机：多尺度融合在时间间隔级别捕获局部 EHR-CXR 交互，在序列级别捕获全局进展趋势，桥接了不同时间粒度的模态。

### 损失函数 / 训练策略

总训练损失为多项加权组合：$\mathcal{L} = \lambda_{\text{pred}} \cdot \text{CE}(\hat{\mathbf{y}}, \mathbf{y}) + \lambda_{\text{orth}}\mathcal{L}_{\text{orth}} + \lambda_{\text{temp}}\mathcal{L}_{\text{temp}} + \lambda_{\text{PAE}}\mathcal{L}_{\text{PAE}}$

使用 MIMIC 数据集族（MIMIC-IV EHR + MIMIC-CXR 影像 + Chest ImaGenome 区域标注），选择 ≥2 张 CXR 的 ICU 住院记录。

## 实验关键数据

### 主实验

疾病进展识别（七种胸部疾病的宏平均）：

| 方法 | 类型 | Precision | Recall | F1 | AUPRC |
|---|---|---|---|---|---|
| SDPL | 单模态CXR | 0.408 | 0.406 | 0.393 | 0.417 |
| CheXRelNet | 单模态CXR | 0.395 | 0.392 | 0.389 | 0.394 |
| DiPro (单模态) | 单模态CXR | 0.475 | 0.452 | **0.453** | **0.468** |
| UTDE | 多模态 | 0.481 | 0.462 | 0.449 | 0.472 |
| DrFuse | 多模态 | 0.442 | 0.461 | 0.429 | 0.438 |
| DiPro (多模态) | 多模态 | **0.484** | **0.471** | **0.466** | **0.478** |

ICU 预测任务（纵向CXR+EHR设置）：

| 方法 | 死亡率 AUPRC | 死亡率 AUROC | 住院时间 Kappa | 住院时间 ACC |
|---|---|---|---|---|
| UMSE | 0.712 | 0.891 | 0.204 | 0.410 |
| MedFuse | 0.716 | 0.881 | 0.210 | 0.412 |
| UTDE | 0.710 | 0.887 | 0.195 | 0.400 |
| DiPro | **0.742** | **0.897** | **0.248** | **0.440** |

### 消融实验

| 配置 | Disease Prog. F1 | Mortality AUPRC | LOS ACC | 说明 |
|---|---|---|---|---|
| DiPro (完整) | 0.466 | 0.742 | 0.440 | 所有模块预期工效果 |
| A1: 去MMF | 0.460 | 0.724 | 0.416 | MMF对ICU预测贡献最大 |
| A2: 去PAE | 0.433 | 0.730 | 0.432 | PAE对进展识别贡献显著 |
| A3: 仅STD | 0.439 | 0.694 | 0.404 | 无融合策略性能较差 |
| A4: 基线 | 0.362 | 0.721 | 0.425 | STD带来21.3%的F1相对提升 |
| DiPro-: 自动bbox | 0.457 | 0.736 | 0.430 | 替换为自动检测区域仍有效 |

### 关键发现

- **单模态 DiPro 已超越所有基线**：仅使用 CXR 的 DiPro 在 F1 上比 SDPL 提升 15.3%，说明时空解耦本身就极具价值
- **EHR 融合带来一致增益**：多模态 DiPro 比单模态进一步提升 2.9% F1，确认了多模态融合的有效性
- **注意力权重与临床知识吻合**：心脏轮廓区域在心脏扩大检测中获得最高注意力，肺门区域在肺水肿检测中突出，右侧结构在死亡率预测中更受关注——这些都与已知的临床知识高度一致
- DiPro 是首个将 EHR 数据整合到 CXR 疾病进展识别任务中的工作

## 亮点与洞察

- **CXR 序列 "静-动" 解耦的临床合理性**：解剖结构的时间一致性和病理变化的时间变异性是放射学诊断的基本原则，DiPro 将这一临床直觉形式化
- **PAE 的时间反转技巧**：简单但有效的约束——反转输入顺序应反转进展方向，为时序特征学习提供了优雅的自监督信号
- **中心聚焦注意力掩码**：处理 EHR-CXR 时间错位的注意力掩码设计精妙，通过 sigmoid 近似实现了软时间窗口选择
- **可解释性**：注意力权重分析提供了决策过程的可视化解释，有助于临床可信度

## 局限与展望

- 依赖 Chest ImaGenome 的区域注释（解剖 bbox），虽然消融显示自动 bbox 也有效，但仍是额外依赖
- 排除了仅有单张 CXR 的住院记录，引入采样偏差
- 七种疾病的进展标签来自 Chest ImaGenome 的自动注释而非专家标注，存在标签噪声
- 仅在 MIMIC（单中心）数据上验证，多中心泛化能力未知
- 静态/动态解耦假设解剖结构短期不变，但在长期随访中某些结构（如心脏大小）可能逐渐变化

## 相关工作与启发

- 与视频表示学习的联系：STD 模块可类比为动态纹理与静态背景分离，但加上了医学领域的正交约束和时间一致性
- 与多导联时序融合的关系：MMF 的局部-全局融合范式可推广到其他异步多模态场景（如 MRI+EEG、CT+基因组）
- 对临床预测的启发：显式建模疾病进展方向比仅提取差异特征更有利于风险分层

## 评分

- **新颖性**: ⭐⭐⭐⭐ 时空解耦+PAE+多尺度融合的组合有新意，但各单独组件并非全新
- **实验充分度**: ⭐⭐⭐⭐⭐ 大规模公开数据集、多任务评估、丰富的消融研究、注意力可视化、鲁棒性分析，非常完整
- **写作质量**: ⭐⭐⭐⭐ 框架图清晰，但模块较多使得方法部分较长
- **价值**: ⭐⭐⭐⭐⭐ 对纵向多模态临床预测有重要实践价值，直接面向ICU临床决策支持

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning](../../ICCV2025/multimodal_vlm/causal_disentanglement_and_cross-modal_alignment_for_enhanced_few-shot_learning.md)
- [\[ECCV 2024\] A Multimodal Benchmark Dataset and Model for Crop Disease Diagnosis](../../ECCV2024/multimodal_vlm/a_multimodal_benchmark_dataset_and_model_for_crop_disease_di.md)
- [\[ICML 2025\] Universal Retrieval for Multimodal Trajectory Modeling](../../ICML2025/multimodal_vlm/universal_retrieval_for_multimodal_trajectory_modeling.md)
- [\[ICLR 2026\] Unified Vision-Language Modeling via Concept Space Alignment](../../ICLR2026/multimodal_vlm/unified_vision-language_modeling_via_concept_space_alignment.md)
- [\[NeurIPS 2025\] Aligning by Misaligning: Boundary-aware Curriculum Learning for Multimodal Alignment](aligning_by_misaligning_boundaryaware_curriculum_learning_fo.md)

</div>

<!-- RELATED:END -->
