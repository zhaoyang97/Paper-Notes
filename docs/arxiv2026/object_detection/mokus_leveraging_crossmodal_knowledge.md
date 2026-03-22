# MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization

**会议**: arXiv 2026  
**arXiv**: [2603.12743](https://arxiv.org/abs/2603.12743)  
**作者**: Chenyang Zhu, Hongxiang Li, Xiu Li, Long Chen
**代码**: 待确认  
**领域**: 目标检测  
**关键词**: mokus, leveraging, cross-modal, knowledge, transfer  

## 一句话总结
概念定制通常将稀有标记绑定到目标概念。

## 背景与动机
Concept customization typically binds rare tokens to a target concept.. Unfortunately, these approaches often suffer from unstable performance as the pretraining data seldom contains these rare tokens.

## 核心问题
概念定制通常将稀有标记绑定到目标概念。不幸的是，这些方法通常会遇到性能不稳定的问题，因为预训练数据很少包含这些稀有标记。

## 方法详解

### 整体框架
- Consequently, we introduce Knowledge-aware Concept Customization, a novel task aiming at binding diverse textual knowledge to target visual concepts.
- Therefore, we propose MoKus, a novel framework for knowledge-aware concept customization.
- Our framework relies on a key observation: cross-modal knowledge transfer, where modifying knowledge within the text modality naturally transfers to the visual modality during generation.
- Inspired by this observation, MoKus contains two stages: (1) In visual concept learning, we first learn the anchor representation to store the visual information of the target concept.

### 关键设计
1. **关键组件1**: Inspired by this observation, MoKus contains two stages: (1) In visual concept learning, we first learn the anchor representation to store the visual information of the target concept.
2. **关键组件2**: To further comprehensively evaluate our proposed MoKus on the new task, we introduce the first benchmark for knowledge-aware concept customization: KnowCusBench.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 为了进一步全面评估我们在新任务上提出的 MoKus，我们引入了第一个知识感知概念定制基准：KnowCusBench。
- 广泛的评估表明 MoKus 的性能优于最先进的方法。
- 我们还展示了我们的方法实现世界知识基准改进的能力。

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
