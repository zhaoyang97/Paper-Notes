# Imagine and Seek: Improving Composed Image Retrieval with an Imagined Proxy

**会议**: CVPR 2025  
**arXiv**: 见CVF  
**代码**: 待确认  
**领域**: NLP理解  
**关键词**: 待补充

## 一句话总结
> 基于摘要：The Zero-shot Composed Image Retrieval (ZSCIR) requires retrieving images that match the query image and the relative captions.Current methods focus on projecting the query image into the text feature space, subsequently combining them with features of query texts for retrieval. However, retrieving 

## 研究背景与动机
1. **领域现状**：本文研究的问题属于 NLP理解 方向。The Zero-shot Composed Image Retrieval (ZSCIR) requires retrieving images that match the query image and the relative captions.Current methods focus on projecting the query image into the text feature space, subsequently combining them with features of query texts for retrieval.
2. **现有痛点**：现有方法存在局限性——效率、精度或泛化性方面有改进空间。
3. **核心矛盾**：需要在效果与效率/泛化性之间找到更好的平衡。
4. **本文要解决什么？** 针对上述问题，作者提出了新方法。
5. **切入角度**：从新的技术视角或观察出发。
6. **核心idea一句话**：However, retrieving images only with the text features cannot guarantee detailed alignment due to the natural gap between images and text.In this paper, we introduce Imagined Proxy for CIR, a training

## 方法详解

### 整体框架
本文提出的方法概述如下（基于摘要信息）：

However, retrieving images only with the text features cannot guarantee detailed alignment due to the natural gap between images and text.In this paper, we introduce Imagined Proxy for CIR, a training-free method that creates a proxy image aligned with the query image and text description, enhancing query representation in the retrieval process.We first leverage the large language model's generalization capability to generate an image layout, and then apply both the query text and image for conditional generation.The robust query features are enhanced by merging the proxy image, query image, and text semantic perturbation. Our newly proposed balancing metric integrates text-based and proxy retrieval similarities, allowing for more accurate retrieval of the target image while incorporating image-side information into the process.Experiments on three public datasets demonstrate that our method significantly improves retrieval performances.

### 关键设计

1. **核心模块**:
   - 做什么：解决上述痛点的关键技术组件
   - 核心思路：详见论文方法部分
   - 设计动机：提升性能或效率

### 损失函数 / 训练策略
详见论文全文（缓存不足，无法提取具体训练细节）。

## 实验关键数据

### 主实验
基于摘要的实验信息：We achieve state-of-the-art (SOTA) results on the CIRR dataset with a Recall@K of 70.07 at K=10. Additionally, we achieved an improvement in Recall@10 on the FashionIQ dataset, rising from 45.11 to 45.74, and improved the baseline performance in CIRCO with a mAPK@10 score, increasing from 32.24 to 34.26.

| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 详见论文 | - | - | - | - |

### 消融实验
| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 完整模型 | 最优 | 完整方法 |
| 去除核心模块 | 下降 | 验证核心贡献 |

### 关键发现
- 本文方法在目标任务上取得显著改进
- 各核心模块均对最终性能有贡献

## 亮点与洞察
- 问题定义清晰，方法针对性强
- 核心设计思路可能可以迁移到相关场景

## 局限性 / 可改进方向
- 需要阅读全文才能深入分析方法细节和局限
- 泛化性和可扩展性有待进一步验证

## 相关工作与启发
- 本文在该领域的既有方法基础上做出了改进

## 评分
- 新颖性: ⭐⭐⭐ 基于摘要初评，有一定创新
- 实验充分度: ⭐⭐⭐ 需读全文验证
- 写作质量: ⭐⭐⭐ 基于摘要初评
- 价值: ⭐⭐⭐ 在该领域有贡献
