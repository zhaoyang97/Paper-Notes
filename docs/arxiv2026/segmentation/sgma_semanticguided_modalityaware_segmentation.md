# SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data

**会议**: arXiv 2026  
**arXiv**: [2603.02505](https://arxiv.org/abs/2603.02505)  
**作者**: Lekang Wen, Liang Liao, Jing Xiao, Mi Wang
**代码**: 待确认  
**领域**: 语义分割  
**关键词**: sgma, semantic-guided, modality-aware, segmentation, remote  

## 一句话总结
多模态语义分割集成了来自不同传感器的互补信息，用于遥感地球观测。
## 背景与动机
Multimodal semantic segmentation integrates complementary information from diverse sensors for remote sensing Earth observation.. However, practical systems often encounter missing modalities due to sensor failures or incomplete coverage, termed Incomplete Multimodal Semantic Segmentation (IMSS).

## 核心问题
然而，实际系统经常遇到由于传感器故障或覆盖不完整而丢失模态的情况，称为不完整多模态语义分割（IMSS）。
## 方法详解

### 整体框架
- To address these limitations, we propose the Semantic-Guided Modality-Aware (SGMA) framework, which ensures balanced multimodal learning while reducing intra-class variation and reconciling cross-modal inconsistencies through semantic guidance.
- SGMA introduces two complementary plug-and-play modules: (1) Semantic-Guided Fusion (SGF) module extracts multi-scale, class-wise semantic prototypes that capture consistent categorical representations across modalities, estimates per-modality robustness based on prototype-feature alignment, and performs adaptive fusion weighted by robustness scores to mitigate intra-class variation and cross-modal heterogeneity; (2) Modality-Aware Sampling (MAS) module leverages robustness estimations from SGF to dynamically reweight training samples, prioritizing challenging samples from fragile modalities to address modality imbalance.

### 关键设计
1. **关键组件1**: SGMA introduces two complementary plug-and-play modules: (1) Semantic-Guided Fusion (SGF) module extracts multi-scale, class-wise semantic prototypes that capture consistent categorical representations across modalities, estimates per-modality robustness based on prototype-feature alignment, and performs adaptive fusion weighted by robustness scores to mitigate intra-class variation and cross-modal heterogeneity; (2) Modality-Aware Sampling (MAS) module leverages robustness estimations from SGF to dynamically reweight training samples, prioritizing challenging samples from fragile modalities to address modality imbalance.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 跨多个数据集和骨干网的广泛实验表明，SGMA 始终优于最先进的方法，尤其是在脆弱模式方面有显着的改进。
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
- 对我的价值: ⭐⭐⭐⭐
