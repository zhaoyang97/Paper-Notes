# ARCHE: Autoregressive Residual Compression with Hyperprior and Excitation

**会议**: arXiv 2026  
**arXiv**: [2603.10188](https://arxiv.org/abs/2603.10188)  
**作者**: Sofia Iliopoulou, Dimitris Ampeliotis, Athanassios Skodras
**代码**: 待确认  
**领域**: 模型压缩/高效推理  
**关键词**: arche, autoregressive, residual, compression, hyperprior  

## 一句话总结
基于学习的图像压缩的最新进展表明，通过联合学习紧凑的潜在表示和概率熵模型，端到端优化可以大大优于传统编解码器。

## 背景与动机
Recent progress in learning-based image compression has demonstrated that end-to-end optimization can substantially outperform traditional codecs by jointly learning compact latent representations and probabilistic entropy models.. However, many existing approaches achieve high rate-distortion efficiency at the expense of increased computational cost and limited parallelism.

## 核心问题
然而，许多现有方法以增加计算成本和有限的并行性为代价实现了高率失真效率。

## 方法详解

### 整体框架
- Recent progress in learning-based image compression has demonstrated that end-to-end optimization can substantially outperform traditional codecs by jointly learning compact latent representations and probabilistic entropy models.
- This paper presents ARCHE - Autoregressive Residual Compression with Hyperprior and Excitation, an end-to-end learned image compression framework that balances modeling accuracy and computational efficiency.
- The proposed architecture unifies hierarchical, spatial, and channel-based priors within a single probabilistic framework, capturing both global and local dependencies in the latent representation of the image, while employing adaptive feature recalibration and residual refinement to enhance latent representation quality.
- Visual comparisons confirm sharper textures and improved color fidelity, particularly at lower bit rates, demonstrating that accurate entropy modeling can be achieved through efficient convolutional designs suitable for practical deployment.

### 关键设计
1. **关键组件1**: Without relying on recurrent or transformer-based components, ARCHE attains state-of-the-art rate-distortion efficiency: it reduces the BD-Rate by approximately 48% relative to the commonly used benchmark model of Balle et al., 30% relative to the channel-wise autoregressive model of Minnen & Singh and 5% against the VVC Intra codec on the Kodak benchmark dataset.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 基于学习的图像压缩的最新进展表明，通过联合学习紧凑的潜在表示和概率熵模型，端到端优化可以大大优于传统编解码器。
- 然而，许多现有方法以增加计算成本和有限的并行性为代价实现了高率失真效率。
- 在不依赖循环或基于变压器的组件的情况下，ARCHE 获得了最先进的率失真效率：相对于 Balle 等人常用的基准模型，它将 BD 率降低了约 48%，相对于 Minnen & Singh 的通道自回归模型降低了 30%，相对于 Kodak 基准数据集上的 VVC Intra 编解码器降低了 5%。

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
