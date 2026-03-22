# Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods

**会议**: arXiv 2026  
**arXiv**: [2603.13077](https://arxiv.org/abs/2603.13077)  
**作者**: Yihang Zhou, Chao Lin, Hideki Kikumoto, Ryozo Ooka, Sibo Cheng
**代码**: 待确认  
**领域**: 模型压缩/高效推理  
**关键词**: rooftop, wind, field, reconstruction, sparse  

## 一句话总结
实时屋顶风速分布对于无人机和城市空中交通系统、风控系统和屋顶利用的安全运行具有重要意义。

## 背景与动机
Real-time rooftop wind-speed distribution is important for the safe operation of drones and urban air mobility systems, wind control systems, and rooftop utilization.. However, rooftop flows show strong nonlinearity, separation, and cross-direction variability, which make flow field reconstruction from sparse sensors difficult.

## 核心问题
然而，屋顶流表现出很强的非线性、分离性和横向变化性，这使得稀疏传感器的流场重建变得困难。

## 方法详解

### 整体框架
- This study develops a learning-from-observation framework using wind-tunnel experimental data obtained by Particle Image Velocimetry (PIV) and compares Kriging interpolation with three deep learning models: UNet, Vision Transformer Autoencoder (ViTAE), and Conditional Wasserstein GAN (CWGAN).

### 关键设计
1. **关键组件1**: The results also show that sensor configuration, optimization, and training strategy should be considered jointly for reliable deployment.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 本研究使用粒子图像测速 (PIV) 获得的风洞实验数据开发了一个从观察中学习的框架，并将克里金插值与三种深度学习模型进行了比较：UNet、视觉变换器自动编码器 (ViTAE) 和条件 Wasserstein GAN (CWGAN)。
- 与Kriging插值相比，深度学习模型将SSIM提高了32.7%，FAC2提高了24.2%，NMSE提高了27.8%。
- 混合风向训练进一步提高了性能，与单向训练相比，SSIM 提升高达 173.7%，FAC2 提升 16.7%，MG 提升 98.3%。

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
