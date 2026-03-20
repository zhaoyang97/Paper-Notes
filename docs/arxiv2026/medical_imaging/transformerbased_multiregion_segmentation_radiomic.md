# Transformer-Based Multi-Region Segmentation and Radiomic Analysis of HR-pQCT Imaging for Osteoporosis Classification

**会议**: arXiv 2026  
**arXiv**: [2603.09137](https://arxiv.org/abs/2603.09137)  
**作者**: Mohseu Rashid Subah, Mohammed Abdul Gani Zilani, Thomas L. Nickolas, Matthew R. Allen, Stuart J. Warden
**代码**: 待确认  
**领域**: 医学图像 / 语义分割  
**关键词**: transformer-based, multi-region, segmentation, radiomic, analysis  

## 一句话总结
骨质疏松症是一种骨骼疾病，通常使用双能 X 射线吸收测定法 (DXA) 进行诊断，该技术可量化面积骨矿物质密度，但忽略骨微结构和周围软组织。
## 背景与动机
Osteoporosis is a skeletal disease typically diagnosed using dual-energy X-ray absorptiometry (DXA), which quantifies areal bone mineral density but overlooks bone microarchitecture and surrounding soft tissues.. High-resolution peripheral quantitative computed tomography (HR-pQCT) enables three-dimensional microstructural imaging with minimal radiation.

## 核心问题
骨质疏松症是一种骨骼疾病，通常使用双能 X 射线吸收测定法 (DXA) 进行诊断，该技术可量化面积骨矿物质密度，但忽略骨微结构和周围软组织。
## 方法详解

### 整体框架
- We introduce a fully automated framework for binary osteoporosis classification using radiomics features extracted from anatomically segmented HR-pQCT images.
- To our knowledge, this work is the first to leverage a transformer-based segmentation architecture, i.e., the SegFormer, for fully automated multi-region HR-pQCT analysis.

### 关键设计
1. **关键组件1**: To our knowledge, this work is the first to leverage a transformer-based segmentation architecture, i.e., the SegFormer, for fully automated multi-region HR-pQCT analysis.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- SegFormer 模型同时描绘了胫骨和腓骨的皮质骨和小梁骨以及周围的软组织，平均 F1 得分为 95.36%。
- 使用肌腱组织特征实现了最佳图像级性能，准确度为 80.08%，受试者工作特征曲线下面积 (AUROC) 为 0.85，优于基于骨骼的模型。
- 在患者层面，用软组织放射组学取代标准生物学、DXA 和 HR-pQCT 参数，将 AUROC 从 0.792 提高到 0.875。
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
