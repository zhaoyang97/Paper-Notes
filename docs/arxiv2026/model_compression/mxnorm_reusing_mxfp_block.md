# MXNorm: Reusing MXFP block scales for efficient tensor normalisation

**会议**: arXiv 2026  
**arXiv**: [2603.13180](https://arxiv.org/abs/2603.13180)  
**作者**: Callum McLean, Luke Y. Prince, Alexandre Payot, Paul Balança, Carlo Luschi
**代码**: 待确认  
**领域**: 模型压缩/高效推理  
**关键词**: mxnorm, reusing, mxfp, block, scales  

## 一句话总结
矩阵乘法性能长期以来一直是扩展深度学习工作负载的主要瓶颈，这刺激了使用越来越低精度数字格式的新型加速器的设计。
## 背景与动机
Matrix multiplication performance has long been the major bottleneck to scaling deep learning workloads, which has stimulated the design of new accelerators that use increasingly low-precision number formats.. However, improvements in matrix multiplication performance have far outstripped improvements in performance on reductions and elementwise computations, which are still being performed in higher precision.

## 核心问题
矩阵乘法性能长期以来一直是扩展深度学习工作负载的主要瓶颈，这刺激了使用越来越低精度数字格式的新型加速器的设计。
## 方法详解

### 整体框架
- Matrix multiplication performance has long been the major bottleneck to scaling deep learning workloads, which has stimulated the design of new accelerators that use increasingly low-precision number formats.
- In this work, we propose MXNorm, a drop-in replacement for RMSNorm that estimates the RMS using only the block scales calculated as part of the MXFP8 cast and enables a 32x decrease in the size of reduction needed for normalization.

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
- 然而，矩阵乘法性能的改进远远超过了约简和元素计算性能的改进，而这些计算仍在以更高的精度执行。
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
