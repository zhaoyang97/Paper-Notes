# Towards Faithful Multimodal Concept Bottleneck Models

**会议**: arXiv 2026  
**arXiv**: [2603.13163](https://arxiv.org/abs/2603.13163)  
**作者**: Pierre Moreau, Emeline Pineau Ferrand, Yann Choho, Benjamin Wong, Annabelle Blangero et al.
**代码**: 待确认  
**领域**: 多模态/VLM  
**关键词**: faithful, multimodal, concept, bottleneck, models  

## 一句话总结
概念瓶颈模型 (CBM) 是可解释的模型，通过人类可解释的概念层进行预测。

## 背景与动机
Concept Bottleneck Models (CBMs) are interpretable models that route predictions through a layer of human-interpretable concepts.. While widely studied in vision and, more recently, in NLP, CBMs remain largely unexplored in multimodal settings.

## 核心问题
概念瓶颈模型 (CBM) 是可解释的模型，通过人类可解释的概念层进行预测。

## 方法详解

### 整体框架
- For their explanations to be faithful, CBMs must satisfy two conditions: concepts must be properly detected, and concept representations must encode only their intended semantics, without smuggling extraneous task-relevant or inter-concept information into final predictions, a phenomenon known as leakage.
- In this work, we introduce f-CBM, a faithful multimodal CBM framework built on a vision-language backbone that jointly targets both aspects through two complementary strategies: a differentiable leakage loss to mitigate leakage, and a Kolmogorov-Arnold Network prediction head that provides sufficient expressiveness to improve concept detection.

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
- 现有方法将概念检测和泄漏缓解视为单独的问题，并且通常以牺牲预测准确性为代价来改进问题。
- 在这项工作中，我们引入了 f-CBM，这是一个建立在视觉语言主干上的忠实多模态 CBM 框架，它通过两种互补策略共同针对这两个方面：用于减轻泄漏的可微泄漏损失，以及提供足够表达能力以改进概念检测的 Kolmogorov-Arnold 网络预测头。
- 实验表明，f-CBM 在任务准确性、概念检测和减少泄漏之间实现了最佳权衡，同时无缝应用于图像和文本或纯文本数据集，使其具有跨模态的多功能性。

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
