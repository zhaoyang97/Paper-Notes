# Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation

**会议**: arXiv 2026  
**arXiv**: [2603.12547](https://arxiv.org/abs/2603.12547)  
**作者**: Fares Bougourzi, Fadi Dornaika, Abdenour Hadid
**代码**: 待确认  
**领域**: 语义分割 / 医学图像  
**关键词**: decoding, matters, efficient, mamba-based, decoder  

## 一句话总结
深度学习在医学图像分割方面取得了显着的成功，在描绘肿瘤和组织方面通常达到专家级的准确性。
## 背景与动机
Deep learning has achieved remarkable success in medical image segmentation, often reaching expert-level accuracy in delineating tumors and tissues.. However, most existing approaches remain task-specific, showing strong performance on individual datasets but limited generalization across diverse imaging modalities.

## 核心问题
深度学习在医学图像分割方面取得了显着的成功，在描绘肿瘤和组织方面通常达到专家级的准确性。
## 方法详解

### 整体框架
- In this paper, we propose a decoder-centric approach for generalized 2D medical image segmentation.
- The proposed Deco-Mamba follows a U-Net-like structure with a Transformer-CNN-Mamba design.
- The encoder combines a CNN block and Transformer backbone for efficient feature extraction, while the decoder integrates our novel Co-Attention Gate (CAG), Vision State Space Module (VSSM), and deformable convolutional refinement block to enhance multi-scale contextual representation.
- Additionally, a windowed distribution-aware KL-divergence loss is introduced for deep supervision across multiple decoding stages.

### 关键设计
1. **关键组件1**: The encoder combines a CNN block and Transformer backbone for efficient feature extraction, while the decoder integrates our novel Co-Attention Gate (CAG), Vision State Space Module (VSSM), and deformable convolutional refinement block to enhance multi-scale contextual representation.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 深度学习在医学图像分割方面取得了显着的成功，在描绘肿瘤和组织方面通常达到专家级的准确性。
- 对各种医学图像分割基准的广泛实验产生了最先进的性能和强大的泛化能力，同时保持了适度的模型复杂性。
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
