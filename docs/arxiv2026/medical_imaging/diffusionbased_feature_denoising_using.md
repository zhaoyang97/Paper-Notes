# Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification

**会议**: arXiv 2026  
**arXiv**: [2603.13182](https://arxiv.org/abs/2603.13182)  
**作者**: Hiba Adil Al-kharsan, Róbert Rajkó
**代码**: 待确认  
**领域**: 医学图像 / 隐私/安全/公平  
**关键词**: diffusion-based, feature, denoising, nnmf, robust  

## 一句话总结
磁共振成像（也称为 MRI）的脑肿瘤分类在计算机辅助诊断系统中发挥着敏感的作用。

## 背景与动机
Brain tumor classification from magnetic resonance imaging, which is also known as MRI, plays a sensitive role in computer-assisted diagnosis systems.. In recent years, deep learning models have achieved high classification accuracy.

## 核心问题
然而，它们对对抗性扰动的敏感性已成为医疗应用中重要的可靠性问题。

## 方法详解

### 整体框架
- Initially, MRI images are preprocessed and converted into a non-negative data matrix, from which compact and interpretable NNMF feature representations are extracted.
- To improve adversarial robustness, a diffusion-based feature-space purification module is introduced.
- The experimental results show that the proposed framework achieves competitive classification performance while significantly enhancing robustness against adversarial perturbations.The findings presuppose that combining interpretable NNMF-based representations with a lightweight deep approach and diffusion-based defense technique supplies an effective and reliable solution for medical image classification under adversarial conditions.

### 关键设计
1. **关键组件1**: Statistical metrics, including AUC, Cohen's d, and p-values, are used to rank and choose the most discriminative components.
2. **关键组件2**: To improve adversarial robustness, a diffusion-based feature-space purification module is introduced.
3. **关键组件3**: The experimental results show that the proposed framework achieves competitive classification performance while significantly enhancing robustness against adversarial perturbations.The findings presuppose that combining interpretable NNMF-based representations with a lightweight deep approach and diffusion-based defense technique supplies an effective and reliable solution for medical image classification under adversarial conditions.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 近年来，深度学习模型取得了很高的分类精度。
- 为了提高对抗鲁棒性，引入了基于扩散的特征空间净化模块。
- 实验结果表明，所提出的框架实现了有竞争力的分类性能，同时显着增强了对抗性扰动的鲁棒性。研究结果表明，将可解释的基于 NNMF 的表示与轻量级深度方法和基于扩散的防御技术相结合，为对抗性条件下的医学图像分类提供了有效且可靠的解决方案。

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
