# Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography

**会议**: arXiv 2026  
**arXiv**: [2603.11627](https://arxiv.org/abs/2603.11627)  
**作者**: Yichi Zhang, Le Xue, Wenbo Zhang, Lanlan Li, Feiyang Xiao
**代码**: 待确认  
**领域**: 语义分割 / 医学图像  
**关键词**: developing, foundation, models, universal, segmentation  

## 一句话总结
正电子发射断层扫描（PET）是一种关键的核医学成像方式，可可视化放射性示踪剂分布以量化体内生理和代谢过程，在疾病管理中发挥着不可替代的作用。
## 背景与动机
Positron emission tomography (PET) is a key nuclear medicine imaging modality that visualizes radiotracer distributions to quantify in vivo physiological and metabolic processes, playing an irreplaceable role in disease management.. Despite its clinical importance, the development of deep learning models for quantitative PET image analysis remains severely limited, driven by both the inherent segmentation challenge from PET's paucity of anatomical contrast and the high costs of data acquisition and annotation.

## 核心问题
尽管其临床重要性，用于定量 PET 图像分析的深度学习模型的开发仍然受到严重限制，这是由于 PET 缺乏解剖对比度所带来的固有分割挑战以及数据采集和注释的高成本所驱动的。
## 方法详解

### 整体框架
- Despite its clinical importance, the development of deep learning models for quantitative PET image analysis remains severely limited, driven by both the inherent segmentation challenge from PET's paucity of anatomical contrast and the high costs of data acquisition and annotation.
- To bridge this gap, we develop generalist foundational models for universal segmentation from 3D whole-body PET imaging.
- We first build the largest and most comprehensive PET dataset to date, comprising 11041 3D whole-body PET scans with 59831 segmentation masks for model development.
- Based on this dataset, we present SegAnyPET, an innovative foundational model with general-purpose applicability to diverse segmentation tasks.

### 关键设计
1. **关键组件1**: We first build the largest and most comprehensive PET dataset to date, comprising 11041 3D whole-body PET scans with 59831 segmentation masks for model development.
2. **关键组件2**: Built on a 3D architecture with a prompt engineering strategy for mask generation, SegAnyPET enables universal and scalable organ and lesion segmentation, supports efficient human correction with minimal effort, and enables a clinical human-in-the-loop workflow.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 对多中心、多示踪剂、多疾病数据集的广泛评估表明，SegAnyPET 在广泛的分割任务中实现了强大的零样本性能，凸显了其推进分子成像临床应用的潜力。
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
