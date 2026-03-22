# O3N: Omnidirectional Open-Vocabulary Occupancy Prediction

**会议**: arXiv 2026  
**arXiv**: [2603.12144](https://arxiv.org/abs/2603.12144)  
**作者**: Mengfei Duan, Hao Shi, Fei Teng, Guoqiang Zhao, Yuheng Zhang
**代码**: [https://github.com/MengfeiD/O3N](https://github.com/MengfeiD/O3N)  
**领域**: 3D视觉  
**关键词**: o3n, omnidirectional, open-vocabulary, occupancy, prediction  

## 一句话总结
通过全方位感知来理解和重构3D世界是自主智能体和具身智能发展的必然趋势。

## 背景与动机
Understanding and reconstructing the 3D world through omnidirectional perception is an inevitable trend in the development of autonomous agents and embodied intelligence.. However, existing 3D occupancy prediction methods are constrained by limited perspective inputs and predefined training distribution, making them difficult to apply to embodied agents that require comprehensive and safe perception of scenes in open world exploration.

## 核心问题
然而，现有的 3D 占用预测方法受到有限的视角输入和预定义的训练分布的限制，使得它们难以应用于开放世界探索中需要全面且安全地感知场景的实体代理。

## 方法详解

### 整体框架
- Understanding and reconstructing the 3D world through omnidirectional perception is an inevitable trend in the development of autonomous agents and embodied intelligence.
- To address this, we present O3N, the first purely visual, end-to-end Omnidirectional Open-vocabulary Occupancy predictioN framework.
- O3N embeds omnidirectional voxels in a polar-spiral topology via the Polar-spiral Mamba (PsM) module, enabling continuous spatial representation and long-range context modeling across 360°.
- The Occupancy Cost Aggregation (OCA) module introduces a principled mechanism for unifying geometric and semantic supervision within the voxel space, ensuring consistency between the reconstructed geometry and the underlying semantic structure.

### 关键设计
1. **关键组件1**: To address this, we present O3N, the first purely visual, end-to-end Omnidirectional Open-vocabulary Occupancy predictioN framework.
2. **关键组件2**: O3N embeds omnidirectional voxels in a polar-spiral topology via the Polar-spiral Mamba (PsM) module, enabling continuous spatial representation and long-range context modeling across 360°.
3. **关键组件3**: The Occupancy Cost Aggregation (OCA) module introduces a principled mechanism for unifying geometric and semantic supervision within the voxel space, ensuring consistency between the reconstructed geometry and the underlying semantic structure.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 对多个模型的大量实验表明，我们的方法不仅在 QuadOcc 和 Human360Occ 基准上实现了最先进的性能，而且还表现出卓越的跨场景泛化性和语义可扩展性，为通用 3D 世界建模铺平了道路。

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
