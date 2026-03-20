# MotionAnymesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins

**会议**: arXiv 2026  
**arXiv**: [2603.12936](https://arxiv.org/abs/2603.12936)  
**作者**: WenBo Xu, Liu Liu, Li Zhang, Dan Guo, RuoNan Liu
**代码**: 待确认  
**领域**: 语义分割 / 3D视觉  
**关键词**: motionanymesh, physics-grounded, articulation, simulation-ready, digital  

## 一句话总结
将静态 3D 网格转换为可交互的铰接资产对于实体 AI 和机器人模拟至关重要。
## 背景与动机
Converting static 3D meshes into interactable articulated assets is crucial for embodied AI and robotic simulation.. However, existing zero-shot pipelines struggle with complex assets due to a critical lack of physical grounding.

## 核心问题
然而，由于严重缺乏物理接地，现有的零样本管道难以应对复杂的资产。
## 方法详解

### 整体框架
- To bridge this gap, we propose MotionAnymesh, an automated zero-shot framework that seamlessly transforms unstructured static meshes into simulation-ready digital twins.
- Our method features a kinematic-aware part segmentation module that grounds VLM reasoning with explicit SP4D physical priors, effectively eradicating kinematic hallucinations.
- Furthermore, we introduce a geometry-physics joint estimation pipeline that combines robust type-aware initialization with physics-constrained trajectory optimization to rigorously guarantee collision-free articulation.

### 关键设计
1. **关键组件1**: Specifically, ungrounded Vision-Language Models (VLMs) frequently suffer from kinematic hallucinations, while unconstrained joint estimation inevitably leads to catastrophic mesh inter-penetration during physical simulation.
2. **关键组件2**: Our method features a kinematic-aware part segmentation module that grounds VLM reasoning with explicit SP4D physical priors, effectively eradicating kinematic hallucinations.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 大量实验表明，MotionAnymesh 在几何精度和动态物理可执行性方面均显着优于最先进的基线，为下游应用提供高度可靠的资产。
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
