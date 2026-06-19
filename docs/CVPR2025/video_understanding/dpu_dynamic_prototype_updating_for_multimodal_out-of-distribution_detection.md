---
title: >-
  [论文解读] DPU: Dynamic Prototype Updating for Multimodal Out-of-Distribution Detection
description: >-
  [CVPR 2025][视频理解][多模态OOD检测] 提出**Dynamic Prototype Updating (DPU)**框架，通过**Cohesive-Separate对比训练**建立稳健表示空间、**动态原型逼近**自适应更新类中心、**Pro-ratio差异增强**按样本到原型的距离调节多模态预测差异的放大强度，作为即插即用模块在5个数据集×9种基础OOD方法上全面提升性能，Far-OOD检测提升最高达**80%**。
tags:
  - "CVPR 2025"
  - "视频理解"
  - "多模态OOD检测"
  - "动态原型更新"
  - "类内变异"
  - "对比学习"
  - "差异增强"
---

# DPU: Dynamic Prototype Updating for Multimodal Out-of-Distribution Detection

**会议**: CVPR 2025  
**arXiv**: [2411.08227](https://arxiv.org/abs/2411.08227)  
**代码**: [https://github.com/lili0415/DPU-OOD-Detection](https://github.com/lili0415/DPU-OOD-Detection)  
**领域**: 视频理解  
**关键词**: 多模态OOD检测, 动态原型更新, 类内变异, 对比学习, 差异增强

## 一句话总结

提出**Dynamic Prototype Updating (DPU)**框架，通过**Cohesive-Separate对比训练**建立稳健表示空间、**动态原型逼近**自适应更新类中心、**Pro-ratio差异增强**按样本到原型的距离调节多模态预测差异的放大强度，作为即插即用模块在5个数据集×9种基础OOD方法上全面提升性能，Far-OOD检测提升最高达**80%**。

## 研究背景与动机

OOD检测旨在识别偏离训练分布的样本，对自动驾驶、医疗影像、机器人等安全关键场景至关重要。传统OOD检测主要针对单模态（如图像），而多模态（视频+光流+音频）数据中不同模态的预测差异可以作为区分ID/OOD的强信号。

**现有痛点**：
- Dong et al.发现多模态间的**预测差异（prediction discrepancy）**是ID vs OOD的关键信号（ID样本跨模态预测一致，OOD样本不一致），并通过均匀放大差异提升检测性能
- 但该方法假设**同一类的所有样本都是完全一致的**（perfect cohesion），忽略了现实中的**类内变异**
- 当对类中心附近的样本（本身跨模态预测高度一致）强行放大差异时，会**破坏其一致性**，导致模型混乱，ID准确率下降

**核心矛盾**：需要放大OOD的跨模态差异以增强检测，但均匀放大会伤害类内高一致性样本。如何**自适应地**根据样本在类内的位置决定差异放大强度？

**核心idea**：动态维护每个类的原型（class prototype），样本距原型越远则放大差异越强（可能是边界/异常样本），距原型越近则保持低差异（核心样本不应被干扰）。

## 方法详解

### 整体框架

DPU包含三个顺序组件（见Figure 2）：(1) **CSCT**（Step 1）：通过鲁棒边际对比学习和方差正则化构建类内紧凑、类间分离的表示空间；(2) **DPA**（Step 2）：根据batch内方差动态更新类原型，降低离群点的影响；(3) **PDI**（Step 3）：按样本-原型相似度自适应调节差异放大强度，并通过自适应离群点合成进一步增强区分能力。DPU是模型无关的即插即用框架，可与任意OOD检测算法结合。

### 关键设计

1. **Cohesive-Separate Contrastive Training (CSCT)**
    - **功能**：构建类内紧凑（cohesive）、类间分离（separate）的表示空间，同时量化类内方差
    - **核心思路**：
     - **鲁棒边际对比学习** $\mathcal{L}_{rmcl}$：基于InfoNCE损失，使用arc-cosine距离并添加角度margin $m$ 增强类间区分灵敏度
     - **方差表示控制** $\mathcal{L}_{irm}$：最小化每个正集（同类样本集）损失值的方差，确保同类样本获得一致的表示
     - 总损失：$\mathcal{L}_{csct} = \mathcal{L}_{rmcl} + \lambda \cdot \mathcal{L}_{irm}$
    - **设计动机**：基于不变风险最小化（IRM）范式，对比学习提供类间分离，方差约束提供类内稳定性，两者为后续的动态原型学习奠定基础

2. **Dynamic Prototype Approximation (DPA)**
    - **功能**：动态、自适应地更新每个类的原型表示，使原型真正代表类中心
    - **核心思路**：对每个batch中同类样本计算平均嵌入 $H_{av_k}^y$，通过带方差感知的移动平均更新原型：
     $$P_{ty_k}^y = \beta P_{ty_k}^y + (1-\beta) \cdot \frac{1}{\gamma + \text{Var}(\mathcal{L}^j) N^y} \cdot (H_{av_k}^y - P_{ty_k}^y)$$
     更新率与方差成反比——方差低（样本一致）时更新快，方差高（含离群点）时更新慢
    - **设计动机**：传统方法对所有样本等权计算类中心，离群点会拉偏原型。方差感知的动态更新使原型在面对噪声batch时保持稳定

3. **Pro-ratio Discrepancy Intensification (PDI) + 自适应离群点合成**
    - **功能**：按样本到原型的距离自适应调节跨模态差异放大强度
    - **核心思路**：
     - 放大率 = $\mu \cdot (1 - \text{Sigmoid}(F_i^v \cdot (P_{ty_v}^y)^T))$，即样本到原型越远（相似度越低），放大率越高
     - 差异度量使用Hellinger距离，对两个模态的预测分布计算：$\mathcal{L}_{pdi} = -\text{IntensificationRate} \cdot \text{Discr}(\hat{p}_i^{k_1}, \hat{p}_i^{k_2})$
     - **自适应离群点合成**：融合不同类的原型生成合成OOD样本 $\bar{P}_{fuse} = \eta \bar{P}_{y_1} + (1-\eta) \bar{P}_{y_2}$，最大化合成样本间的差异和熵
    - **损失函数**：$\mathcal{L}_{aos} = -(\text{Discr}(\bar{P}_{y_1}^{fuse}, \bar{P}_{y_2}^{fuse}) + E(\bar{P}_{y_1}^{fuse}) + E(\bar{P}_{y_2}^{fuse}))$
    - **设计动机**：核心样本的跨模态预测本就一致，强行放大会混淆分类器；边缘样本更可能出现模态不一致，放大其差异更有意义。离群点合成提供了额外的训练信号

## 实验关键数据

### Near-OOD检测（HMDB51为ID，25/26类划分）

| 方法 | FPR95↓ | AUROC↑ | ID ACC↑ |
|------|--------|--------|---------|
| MSP | 44.66 | 87.74 | 89.32 |
| MSP+A2D | 38.78 | 88.37 | 90.64 |
| **MSP+DPU** | **34.20** | **89.15** | **92.16** |
| Energy | 43.36 | 87.46 | 89.32 |
| **Energy+DPU** | **35.07** | **89.52** | **92.16** |
| Mahalanobis | 40.31 | 85.28 | 89.32 |
| **Mahalanobis+DPU** | **36.17** | **89.53** | **92.16** |

### Far-OOD检测（HMDB51 vs Kinetics600）

DPU在所有9种基础OOD方法上均实现显著提升，FPR95降低最高达**80%**（图1示例）

### 跨数据集验证

在UCF101（50/51划分）、Kinetics600（129/110划分）、EPIC-Kitchen上，DPU对所有基础方法均实现一致提升：
- UCF101：FPR95平均降低约1-2个百分点
- Kinetics600：FPR95平均降低约2-3个百分点
- EPIC-Kitchen：FPR95平均降低约3-5个百分点

### 消融实验

- 移除CSCT：AUROC下降2.1%
- 移除DPA（使用固定原型）：AUROC下降1.5%
- 移除PDI（使用均匀放大）：FPR95上升4.2%
- 移除离群点合成：AUROC下降0.8%
- 三个组件缺一不可，PDI的贡献最为关键

### 关键发现

- 均匀差异放大（A2D）在提升OOD检测的同时可能**降低**ID准确率（某些配置下），而DPU在提升OOD检测的同时**始终提升**ID准确率
- Near-OOD检测中，所有指标平均提升约10%；Far-OOD检测中提升更加显著
- 方差感知的原型更新使原型更稳定，比简单移动平均高出约1% AUROC

## 亮点与洞察

- **首次关注多模态OOD中的类内变异问题**：揭示了均匀差异放大的根本缺陷，提供了更精细的解决方案
- **模型无关的即插即用设计**：在MSP、Energy、Maxlogit、Mahalanobis等9种主流OOD方法上一致有效，体现了极强的通用性
- **动态原型更新机制**：方差感知的自适应更新率是一个精巧的设计，在噪声batch中自动减缓更新，在一致batch中快速适应
- ID准确率的提升表明DPU不仅帮助OOD检测，还改善了ID分类——因为更好的类内紧凑性和类间分离性对分类本身有益

## 局限性

- 多模态输入限于视频+光流+音频，对于更多模态的组合（如文本、深度等）的效果未验证
- DPA中的超参数 $\beta$、$\gamma$ 需对每个数据集调优
- 原型空间的维度较高（与特征维度一致），大量类别时原型存储和更新的开销可能变得不可忽略
- 仅在分类模型backbone（如SlowFast、I3D）上验证，对基于VLM的OOD检测适用性待考察

## 相关工作与启发

- **Dong et al.**首次提出多模态OOD基准和跨模态预测差异的观察，DPU在此基础上解决了均匀放大的缺陷
- **原型学习**（Prototypical Networks）在少样本学习中广泛使用，DPU将其引入OOD检测场景并增加了动态更新机制
- **不变风险最小化（IRM）**的思想启发了CSCT中的方差约束设计
- 启发：在其他需要"自适应强度"的场景（如数据增强、对抗训练）中，基于原型距离的自适应机制值得借鉴

## 评分

⭐⭐⭐⭐ — 对多模态OOD检测中类内变异问题的观察敏锐且有洞察力。三步框架设计逻辑清晰、环环相扣。在5个数据集×9种基础方法的大规模实验中全面验证了通用性，Far-OOD 80%的提升令人印象深刻。即插即用特性使其具有很强的实用价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning](../../CVPR2026/video_understanding/worldmm_dynamic_multimodal_memory_agent_for_long_video_reasoning.md)
- [\[CVPR 2025\] DynFocus: Dynamic Cooperative Network Empowers LLMs with Video Understanding](dynfocus_dynamic_cooperative_network_empowers_llms_with_video_understanding.md)
- [\[CVPR 2025\] STOP: Integrated Spatial-Temporal Dynamic Prompting for Video Understanding](stop_integrated_spatial-temporal_dynamic_prompting_for_video_understanding.md)
- [\[CVPR 2025\] Localizing Events in Videos with Multimodal Queries](localizing_events_in_videos_with_multimodal_queries.md)
- [\[ICCV 2025\] DisTime: Distribution-based Time Representation for Video Large Language Models](../../ICCV2025/video_understanding/distime_distribution-based_time_representation_for_video_large_language_models.md)

</div>

<!-- RELATED:END -->
