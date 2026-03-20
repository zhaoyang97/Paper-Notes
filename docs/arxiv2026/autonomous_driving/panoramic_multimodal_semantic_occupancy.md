# Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots

**会议**: arXiv 2026  
**arXiv**: [2603.13108](https://arxiv.org/abs/2603.13108)  
**作者**: Guoqiang Zhao, Zhe Yang, Sheng Wu, Fei Teng, Mengfei Duan
**代码**: [https://github.com/SXDR/PanoMMOcc](https://github.com/SXDR/PanoMMOcc)  
**领域**: 自动驾驶/机器人 / 3D视觉  
**关键词**: panoramic, multimodal, semantic, occupancy, prediction  

## 一句话总结
全景图像为四足机器人的感知提供整体 360° 视觉覆盖。
## 背景与动机
Panoramic imagery provides holistic 360° visual coverage for perception in quadruped robots.. However, existing occupancy prediction methods are mainly designed for wheeled autonomous driving and rely heavily on RGB cues, limiting their robustness in complex environments.

## 核心问题
然而，现有的占用预测方法主要针对轮式自动驾驶而设计，严重依赖 RGB 线索，限制了其在复杂环境中的鲁棒性。
## 方法详解

### 整体框架
- However, existing occupancy prediction methods are mainly designed for wheeled autonomous driving and rely heavily on RGB cues, limiting their robustness in complex environments.
- To bridge this gap, (1) we present PanoMMOcc, the first real-world panoramic multimodal occupancy dataset for quadruped robots, featuring four sensing modalities across diverse scenes.
- (2) We propose a panoramic multimodal occupancy perception framework, VoxelHound, tailored for legged mobility and spherical imaging.
- Specifically, we design (i) a Vertical Jitter Compensation (VJC) module to mitigate severe viewpoint perturbations caused by body pitch and roll during mobility, enabling more consistent spatial reasoning, and (ii) an effective Multimodal Information Prompt Fusion (MIPF) module that jointly leverages panoramic visual cues and auxiliary modalities to enhance volumetric occupancy prediction.

### 关键设计
1. **关键组件1**: To bridge this gap, (1) we present PanoMMOcc, the first real-world panoramic multimodal occupancy dataset for quadruped robots, featuring four sensing modalities across diverse scenes.
2. **关键组件2**: Specifically, we design (i) a Vertical Jitter Compensation (VJC) module to mitigate severe viewpoint perturbations caused by body pitch and roll during mobility, enabling more consistent spatial reasoning, and (ii) an effective Multimodal Information Prompt Fusion (MIPF) module that jointly leverages panoramic visual cues and auxiliary modalities to enhance volumetric occupancy prediction.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- （3）我们建立了基于PanoMMOcc的基准，并提供详细的数据分析，以便能够在具有挑战性的具体场景下系统地评估感知方法。
- 大量实验表明，VoxelHound 在 PanoMMOcc 上实现了最先进的性能（mIoU 中+4.16%}）。
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
