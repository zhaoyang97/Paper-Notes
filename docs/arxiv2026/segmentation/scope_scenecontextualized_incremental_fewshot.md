# SCOPE: Scene-Contextualized Incremental Few-Shot 3D Segmentation

**会议**: arXiv 2026  
**arXiv**: [2603.06572](https://arxiv.org/abs/2603.06572)  
**作者**: Vishal Thengane, Zhaochong An, Tianjin Huang, Son Lam Phung, Abdesselam Bouzerdoum
**代码**: [https://github.com/Surrey-UP-Lab/SCOPE](https://github.com/Surrey-UP-Lab/SCOPE)  
**领域**: 语义分割 / 3D视觉  
**关键词**: scope, scene-contextualized, incremental, few-shot, segmentation  

## 一句话总结
增量少样本（IFS）分割旨在随着时间的推移仅从少量注释中学习新类别。
## 背景与动机
Incremental Few-Shot (IFS) segmentation aims to learn new categories over time from only a few annotations.. Although widely studied in 2D, it remains underexplored for 3D point clouds.

## 核心问题
尽管在 2D 领域得到了广泛的研究，但对于 3D 点云的研究仍然不足。
## 方法详解

### 整体框架
- Existing methods suffer from catastrophic forgetting or fail to learn discriminative prototypes under sparse supervision, and often overlook a key cue: novel categories frequently appear as unlabelled background in base-training scenes.
- We introduce SCOPE (Scene-COntextualised Prototype Enrichment), a plug-and-play background-guided prototype enrichment framework that integrates with any prototype-based 3D segmentation method.
- When novel classes arrive with few labelled samples, relevant background prototypes are retrieved and fused with few-shot prototypes to form enriched representations without retraining the backbone or adding parameters.
- Experiments on ScanNet and S3DIS show that SCOPE achieves SOTA performance, improving novel-class IoU by up to 6.98% and 3.61%, and mean IoU by 2.25% and 1.70%, respectively, while maintaining low forgetting.

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
- ScanNet 和 S3DIS 上的实验表明，SCOPE 实现了 SOTA 性能，将新颖类 IoU 提高了高达 6.98% 和 3.61%，平均 IoU 分别提高了 2.25% 和 1.70%，同时保持了较低的遗忘。
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
