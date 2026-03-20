# $M^2$-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs

**会议**: arXiv 2026  
**arXiv**: [2603.09737](https://arxiv.org/abs/2603.09737)  
**作者**: Kaixin Lin, Kunyu Peng, Di Wen, Yufan Chen, Ruiping Liu
**代码**: [https://github.com/qixi7up/M2-Occ](https://github.com/qixi7up/M2-Occ)  
**领域**: 3D视觉  
**关键词**: $m^2$-occ, resilient, semantic, occupancy, prediction  

## 一句话总结
语义占用预测可实现自动驾驶的密集 3D 几何和语义理解。
## 背景与动机
Semantic occupancy prediction enables dense 3D geometric and semantic understanding for autonomous driving.. However, existing camera-based approaches implicitly assume complete surround-view observations, an assumption that rarely holds in real-world deployment due to occlusion, hardware malfunction, or communication failures.

## 核心问题
然而，现有的基于摄像头的方法隐含地假设了完整的环视观察，但由于遮挡、硬件故障或通信故障，这种假设在现实世界的部署中很少成立。
## 方法详解

### 整体框架
- We study semantic occupancy prediction under incomplete multi-camera inputs and introduce $M^2$-Occ, a framework designed to preserve geometric structure and semantic coherence when views are missing.
- First, a Multi-view Masked Reconstruction (MMR) module leverages the spatial overlap among neighboring cameras to recover missing-view representations directly in the feature space.
- Second, a Feature Memory Module (FMM) introduces a learnable memory bank that stores class-level semantic prototypes.
- We introduce a systematic missing-view evaluation protocol on the nuScenes-based SurroundOcc benchmark, encompassing both deterministic single-view failures and stochastic multi-view dropout scenarios.

### 关键设计
1. **关键组件1**: First, a Multi-view Masked Reconstruction (MMR) module leverages the spatial overlap among neighboring cameras to recover missing-view representations directly in the feature space.
2. **关键组件2**: Second, a Feature Memory Module (FMM) introduces a learnable memory bank that stores class-level semantic prototypes.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 我们在基于 nuScenes 的 SurroundOcc 基准上引入了系统性缺失视图评估协议，涵盖确定性单视图故障和随机多视图丢失场景。
- 在安全关键的缺失后视设置下，$M^2$-Occ 将 IoU 提高了 4.93%。
- 这些增益是在不影响全视图性能的情况下实现的。
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
