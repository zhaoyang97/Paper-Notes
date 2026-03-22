# Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation

**会议**: arXiv 2026  
**arXiv**: [2603.04887](https://arxiv.org/abs/2603.04887)  
**作者**: Hong Liu, Dong Wei, Qian Dai, Xian Wu, Yefeng Zheng
**代码**: 待确认  
**领域**: 医学图像 / 语义分割  
**关键词**: federated, modality-specific, encoders, partially, personalized  

## 一句话总结
大多数现有的用于医学图像分析的联邦学习（FL）方法仅考虑模内异质性，限制了它们在多模态成像应用中的适用性。

## 背景与动机
Most existing federated learning (FL) methods for medical image analysis only considered intramodal heterogeneity, limiting their applicability to multimodal imaging applications.. In practice, some FL participants may possess only a subset of the complete imaging modalities, posing intermodal heterogeneity as a challenge to effectively training a global model on all participants' data.

## 核心问题
在实践中，一些 FL 参与者可能只拥有完整成像模式的一个子集，这给跨模式异质性带来了对所有参与者数据有效训练全局模型的挑战。

## 方法详解

### 整体框架
- This work proposes a new FL framework with federated modality-specific encoders and partially personalized multimodal fusion decoders (FedMEPD) to address the two concurrent issues.
- Implementation-wise, a server with full-modal data employs a fusion decoder to fuse representations from all modality-specific encoders, thus bridging the modalities to optimize the encoders via backpropagation.
- Moreover, multiple anchors are extracted from the fused multimodal representations and distributed to the clients in addition to the model parameters.
- Conversely, the clients with incomplete modalities calibrate their missing-modal representations toward the global full-modal anchors via scaled dot-product cross-attention, making up for the information loss due to absent modalities.

### 关键设计
1. **关键组件1**: Specifically, FedMEPD employs an exclusive encoder for each modality to account for the intermodal heterogeneity.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- FedMEPD 在 BraTS 2018 和 2020 多模式脑肿瘤分割基准上得到验证。
- 结果表明，它优于多种最新的多模态和个性化 FL 方法，并且其新颖的设计是有效的。

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
