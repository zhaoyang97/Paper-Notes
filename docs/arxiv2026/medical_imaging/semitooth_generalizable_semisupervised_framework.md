# SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation

**会议**: 投稿中  
**arXiv**: [2603.11616](https://arxiv.org/abs/2603.11616)  
**代码**: 未提及  
**领域**: 医学图像  
**关键词**: semitooth, generalizable, semi-supervised, framework, multi-source  

## 一句话总结
在本文中，我们提出了 SemiTooth，一种用于多源牙齿分割的通用半监督框架。

## 核心问题
然而，全注释数据的获取困难以及不同机构之间多源数据获取的可变性带来了挑战，导致 CBCT 切片的利用率低、体素水平不一致以及特定领域的差异。

## 关键方法
1. 在本文中，我们提出了 SemiTooth，一种用于多源牙齿分割的通用半监督框架
2. 然后，我们设计了一个多教师和多学生的框架，即SemiTooth，它促进了多源数据的半监督学习

## 亮点 / 我学到了什么
- 此外，为多个教师引入了更严格的加权置信约束，以提高多源精度。在MS3Toothset上进行了大量实验，验证了SemiTooth框架的可行性和优越性，在半监督和多源牙齿分割场景上实现了SOTA性能。
- 然后，我们设计了一个多教师和多学生的框架，即SemiTooth，它促进了多源数据的半监督学习

## 局限性 / 可改进方向
- 需要详细阅读原文以确认具体局限
- 潜在扩展方向待进一步分析

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 详见原文 | 详见原文 | — | — | — |

## 与我的研究方向的关联
- `2d_to_3d_medical_distill`
- `medical_bias_audit`
- `medical_dynamic_routing`

## 评分
- 新颖性: ⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐
- 对我的价值: ⭐⭐⭐⭐
