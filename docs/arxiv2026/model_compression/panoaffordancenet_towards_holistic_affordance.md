# PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments

**会议**: arXiv 2026  
**arXiv**: [2603.09760](https://arxiv.org/abs/2603.09760)  
**作者**: Guoliang Zhu, Wanjun Jia, Caoyang Shao, Yuheng Zhang, Zhiyong Li
**代码**: [https://github.com/GL-ZHU925/PanoAffordanceNet](https://github.com/GL-ZHU925/PanoAffordanceNet)  
**领域**: 模型压缩/高效推理  
**关键词**: panoaffordancenet, holistic, affordance, grounding, 360°  

## 一句话总结
全局感知对于 360° 空间中的具体主体至关重要，但当前的可供性基础仍然主要以对象为中心并仅限于透视图。
## 背景与动机
Global perception is essential for embodied agents in 360° spaces, yet current affordance grounding remains largely object-centric and restricted to perspective views.. To bridge this gap, we introduce a novel task: Holistic Affordance Grounding in 360° Indoor Environments.

## 核心问题
全局感知对于 360° 空间中的具体主体至关重要，但当前的可供性基础仍然主要以对象为中心并仅限于透视图。
## 方法详解

### 整体框架
- To bridge this gap, we introduce a novel task: Holistic Affordance Grounding in 360° Indoor Environments.
- We propose PanoAffordanceNet, an end-to-end framework featuring a Distortion-Aware Spectral Modulator (DASM) for latitude-dependent calibration and an Omni-Spherical Densification Head (OSDH) to restore topological continuity from sparse activations.
- By integrating multi-level constraints comprising pixel-wise, distributional, and region-text contrastive objectives, our framework effectively suppresses semantic drift under low supervision.

### 关键设计
1. **关键组件1**: Furthermore, we construct 360-AGD, the first high-quality panoramic affordance grounding dataset.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 大量实验表明，PanoAffordanceNet 的性能显着优于现有方法，为具体智能的场景级感知建立了坚实的基线。
- 源代码和基准数据集将在 https://github.com/GL-ZHU925/PanoAffordanceNet 上公开提供。
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
