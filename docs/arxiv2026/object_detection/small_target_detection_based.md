# Small Target Detection Based on Mask-Enhanced Attention Fusion of Visible and Infrared Remote Sensing Images

**会议**: arXiv 2026  
**arXiv**: [2603.06925](https://arxiv.org/abs/2603.06925)  
**作者**: Qianqian Zhang, Xiaolong Jia, Ahmed M. Abdelmoniem, Li Zhou, Junshe An
**代码**: 待确认  
**领域**: 目标检测 / LLM/NLP  
**关键词**: small, target, detection, mask-enhanced, attention  

## 一句话总结
遥感图像中的目标通常体积小、纹理弱、容易受到复杂背景的干扰，给通用算法的高精度检测带来挑战。
## 背景与动机
Targets in remote sensing images are usually small, weakly textured, and easily disturbed by complex backgrounds, challenging high-precision detection with general algorithms.. Building on our earlier ESM-YOLO, this work presents ESM-YOLO+ as a lightweight visible infrared fusion network.

## 核心问题
遥感图像中的目标通常较小，纹理较弱，容易受到复杂背景的干扰，这对通用算法的高精度检测提出了挑战。在我们早期的ESM-YOLO的基础上，这项工作提出了ESM-YOLO+作为一种轻量级可见红外融合网络。
## 方法详解

### 整体框架
- Building on our earlier ESM-YOLO, this work presents ESM-YOLO+ as a lightweight visible infrared fusion network.
- (1) A Mask-Enhanced Attention Fusion (MEAF) module fuses features at the pixel level via learnable spatial masks and spatial attention, effectively aligning RGB and infrared features, enhancing small-target representation, and alleviating cross-modal misalignment and scale heterogeneity.
- (2) Training-time Structural Representation (SR) enhancement provides auxiliary supervision to preserve fine-grained spatial structures during training, boosting feature discriminability without extra inference cost.

### 关键设计
1. **关键组件1**: (1) A Mask-Enhanced Attention Fusion (MEAF) module fuses features at the pixel level via learnable spatial masks and spatial attention, effectively aligning RGB and infrared features, enhancing small-target representation, and alleviating cross-modal misalignment and scale heterogeneity.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 在 VEDAI 和 DroneVehicle 数据集上进行的大量实验验证了 ESM-YOLO+ 的优越性。
- 该模型在 VEDAI 上实现了 84.71% mAP，在 DroneVehicle 上实现了 74.0% mAP，同时大大降低了模型复杂性，参数比基线少了 93.6%，GFLOPs 比基线低了 68.0%。
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
