# RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images

**会议**: arXiv 2026  
**arXiv**: [2603.12215](https://arxiv.org/abs/2603.12215)  
**作者**: Bin Wan, Runmin Cong, Xiaofei Zhou, Hao Fang, Yaoqi Sun
**代码**: 待确认  
**领域**: 目标检测 / LLM/NLP  
**关键词**: rdnet, region, proportion-aware, dynamic, adaptive  

## 一句话总结
由于对象大小的巨大变化、自注意力机制的计算成本以及基于 CNN 的提取器在捕获全局上下文和远程依赖性方面的局限性，遥感图像中的显着对象检测 (SOD) 面临着重大挑战。

## 背景与动机
Salient object detection (SOD) in remote sensing images faces significant challenges due to large variations in object sizes, the computational cost of self-attention mechanisms, and the limitations of CNN-based extractors in capturing global context and long-range dependencies.. Existing methods that rely on fixed convolution kernels often struggle to adapt to diverse object scales, leading to detail loss or irrelevant feature aggregation.

## 核心问题
由于对象大小的巨大变化、自注意力机制的计算成本以及基于 CNN 的提取器在捕获全局上下文和远程依赖性方面的局限性，遥感图像中的显着对象检测 (SOD) 面临着重大挑战。

## 方法详解

### 整体框架
- We propose the Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network (RDNet), which replaces the CNN backbone with the SwinTransformer for global context modeling and introduces three key modules: (1) the Dynamic Adaptive Detail-aware (DAD) module, which applies varied convolution kernels guided by object region proportions; (2) the Frequency-matching Context Enhancement (FCE) module, which enriches contextual information through wavelet interactions and attention; and (3) the Region Proportion-aware Localization (RPL) module, which employs cross-attention to highlight semantic details and integrates a Proportion Guidance (PG) block to assist the DAD module.

### 关键设计
1. **关键组件1**: Salient object detection (SOD) in remote sensing images faces significant challenges due to large variations in object sizes, the computational cost of self-attention mechanisms, and the limitations of CNN-based extractors in capturing global context and long-range dependencies.
2. **关键组件2**: We propose the Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network (RDNet), which replaces the CNN backbone with the SwinTransformer for global context modeling and introduces three key modules: (1) the Dynamic Adaptive Detail-aware (DAD) module, which applies varied convolution kernels guided by object region proportions; (2) the Frequency-matching Context Enhancement (FCE) module, which enriches contextual information through wavelet interactions and attention; and (3) the Region Proportion-aware Localization (RPL) module, which employs cross-attention to highlight semantic details and integrates a Proportion Guidance (PG) block to assist the DAD module.
3. **关键组件3**: By combining these modules, RDNet achieves robustness against scale variations and accurate localization, delivering superior detection performance compared with state-of-the-art methods.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 为了解决这些问题，这项工作旨在增强对尺度变化的鲁棒性并实现精确的对象定位。
- 通过组合这些模块，RDNet 实现了针对尺度变化的鲁棒性和精确定位，与最先进的方法相比，提供了卓越的检测性能。

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
