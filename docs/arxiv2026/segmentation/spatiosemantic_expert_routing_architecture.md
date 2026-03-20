# Spatio-Semantic Expert Routing Architecture with Mixture-of-Experts for Referring Image Segmentation

**会议**: arXiv 2026  
**arXiv**: [2603.12538](https://arxiv.org/abs/2603.12538)  
**作者**: Alaa Dalaq, Muzammil Behzad
**代码**: 待确认  
**领域**: 语义分割 / LLM/NLP  
**关键词**: spatio-semantic, expert, routing, architecture, mixture-of-experts  

## 一句话总结
参考图像分割旨在为由自然语言表达描述的图像区域产生像素级掩模。
## 背景与动机
Referring image segmentation aims to produce a pixel-level mask for the image region described by a natural-language expression.. Although pretrained vision-language models have improved semantic grounding, many existing methods still rely on uniform refinement strategies that do not fully match the diverse reasoning requirements of referring expressions.

## 核心问题
为了解决这些限制，我们提出了 SERA，一种用于参考图像分割的空间语义专家路由架构。
## 方法详解

### 整体框架
- To address these limitations, we propose SERA, a Spatio-Semantic Expert Routing Architecture for referring image segmentation.
- SERA introduces lightweight, expression-aware expert refinement at two complementary stages within a vision-language framework.
- First, we design SERA-Adapter, which inserts an expression-conditioned adapter into selected backbone blocks to improve spatial coherence and boundary precision through expert-guided refinement and cross-modal attention.
- We then introduce SERA-Fusion, which strengthens intermediate visual representations by reshaping token features into spatial grids and applying geometry-preserving expert transformations before multimodal interaction.

### 关键设计
1. **关键组件1**: First, we design SERA-Adapter, which inserts an expression-conditioned adapter into selected backbone blocks to improve spatial coherence and boundary precision through expert-guided refinement and cross-modal attention.
2. **关键组件2**: In addition, a lightweight routing mechanism adaptively weights expert contributions while remaining compatible with pretrained representations.
3. **关键组件3**: To make this routing stable under frozen encoders, SERA uses a parameter-efficient tuning strategy that updates only normalization and bias terms, affecting less than 1% of the backbone parameters.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 尽管预训练的视觉语言模型已经改进了语义基础，但许多现有方法仍然依赖于统一的细化策略，而这些策略并不完全满足指称表达的多样化推理要求。
- 首先，我们设计了 SERA-Adapter，它将表达条件适配器插入到选定的主干块中，通过专家指导的细化和跨模式注意来提高空间一致性和边界精度。
- 标准参考图像分割基准的实验表明，SERA 始终优于强基线，在需要精确空间定位和精确边界描绘的表达方面尤其明显。
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
