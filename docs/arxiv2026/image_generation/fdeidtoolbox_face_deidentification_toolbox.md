# FDeID-Toolbox: Face De-Identification Toolbox

**会议**: arXiv 2026  
**arXiv**: [2603.13121](https://arxiv.org/abs/2603.13121)  
**作者**: Hui Wei, Hao Yu, Guoying Zhao
**代码**: [https://github.com/infraface/FDeID-Toolbox](https://github.com/infraface/FDeID-Toolbox)  
**领域**: 扩散模型/生成  
**关键词**: fdeid-toolbox, face, de-identification, toolbox  

## 一句话总结
人脸去识别（FDeID）旨在从面部图像中删除个人身份信息，同时保留与任务相关的实用属性，例如年龄、性别和表情。

## 背景与动机
Face de-identification (FDeID) aims to remove personally identifiable information from facial images while preserving task-relevant utility attributes such as age, gender, and expression.. It is critical for privacy-preserving computer vision, yet the field suffers from fragmented implementations, inconsistent evaluation protocols, and incomparable results across studies.

## 核心问题
这些挑战源于任务固有的复杂性：FDeID跨越多个下游应用（例如年龄估计、性别识别、表达分析），需要跨三个维度（例如隐私保护、效用保存和视觉质量）进行评估，使得现有代码库难以使用和扩展。

## 方法详解

### 整体框架
- To address these issues, we present FDeID-Toolbox, a comprehensive toolbox designed for reproducible FDeID research.

### 关键设计
1. **关键组件1**: Our toolbox features a modular architecture comprising four core components: (1) standardized data loaders for mainstream benchmark datasets, (2) unified method implementations spanning classical approaches to SOTA generative models, (3) flexible inference pipelines, and (4) systematic evaluation protocols covering privacy, utility, and quality metrics.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 我们的工具箱采用模块化架构，由四个核心组件组成：(1) 适用于主流基准数据集的标准化数据加载器，(2) 涵盖 SOTA 生成模型经典方法的统一方法实现，(3) 灵活的推理管道，以及 (4) 涵盖隐私、效用和质量指标的系统评估协议。
- 通过实验，我们证明 FDeID-Toolbox 能够在一致的条件下对不同的 FDeID 方法进行公平且可重复的比较。

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
