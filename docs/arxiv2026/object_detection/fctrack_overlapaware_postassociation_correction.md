# FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking

**会议**: arXiv 2026  
**arXiv**: [2603.12758](https://arxiv.org/abs/2603.12758)  
**作者**: Cheng Ju, Zejing Zhao, Akio Namiki
**代码**: 待确认  
**领域**: 目标检测  
**关键词**: fc-track, overlap-aware, post-association, correction, online  

## 一句话总结
可靠的多目标跟踪（MOT）对于在复杂和动态环境中运行的机器人系统至关重要。

## 背景与动机
Reliable multi-object tracking (MOT) is essential for robotic systems operating in complex and dynamic environments.. Despite recent advances in detection and association, online MOT methods remain vulnerable to identity switches caused by frequent occlusions and object overlap, where incorrect associations can propagate over time and degrade tracking reliability.

## 核心问题
尽管最近在检测和关联方面取得了进展，但在线 MOT 方法仍然容易受到频繁遮挡和对象重叠引起的身份切换的影响，其中不正确的关联可能会随着时间的推移而传播并降低跟踪可靠性。

## 方法详解

### 整体框架
- We present a lightweight post-association correction framework (FC-Track) for online MOT that explicitly targets overlap-induced mismatches during inference.
- The proposed method suppresses unreliable appearance updates under high-overlap conditions using an Intersection over Area (IoA)-based filtering strategy, and locally corrects detection-to-tracklet mismatches through appearance similarity comparison within overlapped tracklet pairs.
- By preventing short-term mismatches from propagating, our framework effectively mitigates long-term identity switches without resorting to global optimization or re-identification.
- Specifically, our framework FC-Track produces only 29.55% long-term identity switches, which is substantially lower than existing online trackers.

### 关键设计
1. **关键组件1**: The proposed method suppresses unreliable appearance updates under high-overlap conditions using an Intersection over Area (IoA)-based filtering strategy, and locally corrects detection-to-tracklet mismatches through appearance similarity comparison within overlapped tracklet pairs.
2. **关键组件2**: Specifically, our framework FC-Track produces only 29.55% long-term identity switches, which is substantially lower than existing online trackers.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 我们在 MOT17 测试集上实现了 81.73 MOTA、82.81 IDF1 和 66.95 HOTA，运行速度为 5.7 FPS；在 MOT20 测试集上实现了 77.52 MOTA、80.90 IDF1 和 65.67 HOTA，运行速度为 0.6 FPS。
- 同时，我们的框架在 MOT20 基准测试中保持了最先进的性能。

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
