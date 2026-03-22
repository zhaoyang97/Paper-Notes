# SDF-Net: Structure-Aware Disentangled Feature Learning for Opticall-SAR Ship Re-identification

**会议**: arXiv 2026  
**arXiv**: [2603.12588](https://arxiv.org/abs/2603.12588)  
**作者**: Furui Chen, Han Wang, Yuhan Sun, Jianing You, Yixuan Lv
**代码**: [https://github.com/cfrfree/SDF-Net](https://github.com/cfrfree/SDF-Net)  
**领域**: 目标检测  
**关键词**: sdf-net, structure-aware, disentangled, feature, learning  

## 一句话总结
光学和合成孔径雷达 (SAR) 图像之间的跨模式船舶重新识别 (ReID) 从根本上受到无源光学成像和相干有源雷达传感之间严重辐射差异的挑战。

## 背景与动机
Cross-modal ship re-identification (ReID) between optical and synthetic aperture radar (SAR) imagery is fundamentally challenged by the severe radiometric discrepancy between passive optical imaging and coherent active radar sensing.. While existing approaches primarily rely on statistical distribution alignment or semantic matching, they often overlook a critical physical prior: ships are rigid objects whose geometric structures remain stable across sensing modalities, whereas texture appearance is highly modality-dependent.

## 核心问题
光学和合成孔径雷达 (SAR) 图像之间的跨模式船舶重新识别 (ReID) 从根本上受到无源光学成像和相干有源雷达传感之间严重辐射差异的挑战。

## 方法详解

### 整体框架
- In this work, we propose SDF-Net, a Structure-Aware Disentangled Feature Learning Network that systematically incorporates geometric consistency into optical--SAR ship ReID.
- Built upon a ViT backbone, SDF-Net introduces a structure consistency constraint that extracts scale-invariant gradient energy statistics from intermediate layers to robustly anchor representations against radiometric variations.
- At the terminal stage, SDF-Net disentangles the learned representations into modality-invariant identity features and modality-specific characteristics.

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
- 对 HOSS-ReID 数据集的大量实验表明，SDF-Net 始终优于现有的最先进方法。

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
