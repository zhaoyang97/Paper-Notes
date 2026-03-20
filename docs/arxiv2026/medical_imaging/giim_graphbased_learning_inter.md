# GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis

**会议**: arXiv 2026  
**arXiv**: [2603.09446](https://arxiv.org/abs/2603.09446)  
**作者**: Tran Bao Sam, Hung Vu, Dao Trung Kien, Tran Dat Dang, Van Ha Tang
**代码**: 待确认  
**领域**: 医学图像  
**关键词**: giim, graph-based, learning, inter-, intra-view  

## 一句话总结
计算机辅助诊断 (CADx) 在医学成像中变得至关重要，但自动化系统往往难以复制临床解释的微妙过程。
## 背景与动机
Computer-aided diagnosis (CADx) has become vital in medical imaging, but automated systems often struggle to replicate the nuanced process of clinical interpretation.. Expert diagnosis requires a comprehensive analysis of how abnormalities relate to each other across various views and time points, but current multi-view CADx methods frequently overlook these complex dependencies.

## 核心问题
计算机辅助诊断 (CADx) 在医学成像中变得至关重要，但自动化系统往往难以复制临床解释的微妙过程。
## 方法详解

### 整体框架
- To address these gaps, we reframe the diagnostic task as one of relationship modeling and propose GIIM, a novel graph-based approach.
- Our framework is uniquely designed to simultaneously capture both critical intra-view dependencies between abnormalities and inter-view dynamics.

### 关键设计
1. **关键组件1**: Specifically, they fail to model the crucial relationships within a single view and the dynamic changes lesions exhibit across different views.
2. **关键组件2**: Furthermore, it ensures diagnostic robustness by incorporating specific techniques to effectively handle missing data, a common clinical issue.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 我们通过对不同成像方式（包括 CT、MRI 和乳房 X 线摄影）的广泛评估，证明了这种方法的通用性。
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
