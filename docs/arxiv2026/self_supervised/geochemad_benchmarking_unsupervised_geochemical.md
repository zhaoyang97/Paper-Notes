# GeoChemAD: Benchmarking Unsupervised Geochemical Anomaly Detection for Mineral Exploration

**会议**: arXiv 2026  
**arXiv**: [2603.13068](https://arxiv.org/abs/2603.13068)  
**作者**: Yihao Ding, Yiran Zhang, Chris Gonzalez, Eun-Jung Holden, Wei Liu
**代码**: 待确认  
**领域**: 自监督/表示学习  
**关键词**: geochemad, benchmarking, unsupervised, geochemical, anomaly  

## 一句话总结
地球化学异常检测在矿产勘探中起着至关重要的作用，因为与区域地球化学基线的偏差可能表明有矿化。
## 背景与动机
Geochemical anomaly detection plays a critical role in mineral exploration as deviations from regional geochemical baselines may indicate mineralization.. Existing studies suffer from two key limitations: (1) single region scenarios which limit model generalizability; (2) proprietary datasets, which makes result reproduction unattainable.

## 核心问题
现有研究存在两个主要局限性：（1）单一区域场景限制了模型的普遍性； (2) 专有数据集，导致结果再现无法实现。
## 方法详解

### 整体框架
- In this work, we introduce \textbf{GeoChemAD}, an open-source benchmark dataset compiled from government-led geological surveys, covering multiple regions, sampling sources, and target elements.
- The dataset comprises eight subsets representing diverse spatial scales and sampling conditions.
- Furthermore, we propose \textbf{GeoChemFormer}, a transformer-based framework that leverages self-supervised pretraining to learn target-element-aware geochemical representations for spatial samples.
- The proposed dataset and framework provide a foundation for reproducible research and future development in this direction.

### 关键设计
待深读后补充。

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 在这项工作中，我们引入了 \textbf{GeoChemAD}，这是一个由政府主导的地质调查编译而成的开源基准数据集，涵盖多个区域、采样源和目标元素。
- 为了建立强大的基线，我们重现并基准化了一系列无监督异常检测方法，包括统计模型、生成和基于变压器的方法。
- 大量实验表明，GeoChemFormer 在所有八个子集中始终实现卓越且稳健的性能，在异常检测精度和泛化能力方面均优于现有的无监督方法。
## 局限性 / 可改进方向
- 待深读后补充

## 与相关工作的对比
待深读后补充。

## 与我的研究方向的关联
- 待分析

## 评分
- 新颖性: ⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐
- 对我的价值: ⭐⭐⭐
