# Unleashing Video Language Models for Fine-grained HRCT Report Generation

**会议**: arXiv 2026  
**arXiv**: [2603.12469](https://arxiv.org/abs/2603.12469)  
**作者**: Yingying Fang, Huichi Zhou, KinHei Lee, Yijia Wang, Zhenxuan Zhang
**代码**: 待确认  
**领域**: 医学图像 / LLM/NLP  
**关键词**: unleashing, video, language, models, fine-grained  

## 一句话总结
从高分辨率计算机断层扫描 (HRCT) 生成精确的诊断报告对于临床工作流程至关重要，但由于 3D 体积内的高度病理多样性和空间稀疏性，它仍然是一个艰巨的挑战。

## 背景与动机
Generating precise diagnostic reports from High-Resolution Computed Tomography (HRCT) is critical for clinical workflow, yet it remains a formidable challenge due to the high pathological diversity and spatial sparsity within 3D volumes.. While Video Language Models (VideoLMs) have demonstrated remarkable spatio-temporal reasoning in general domains, their adaptability to domain-specific, high-volume medical interpretation remains underexplored.

## 核心问题
从高分辨率计算机断层扫描 (HRCT) 生成精确的诊断报告对于临床工作流程至关重要，但由于 3D 体积内的高度病理多样性和空间稀疏性，它仍然是一个艰巨的挑战。

## 方法详解

### 整体框架
- In this work, we present AbSteering, an abnormality-centric framework that steers VideoLMs toward precise HRCT report generation.
- Specifically, AbSteering introduces: (i) an abnormality-centric Chain-of-Thought scheme that enforces abnormality reasoning, and (ii) a Direct Preference Optimization objective that utilizes clinically confusable abnormalities as hard negatives to enhance fine-grained discrimination.

### 关键设计
1. **关键组件1**: Specifically, AbSteering introduces: (i) an abnormality-centric Chain-of-Thought scheme that enforces abnormality reasoning, and (ii) a Direct Preference Optimization objective that utilizes clinically confusable abnormalities as hard negatives to enhance fine-grained discrimination.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 虽然视频语言模型 (VideoLM) 在一般领域表现出了卓越的时空推理能力，但它们对特定领域、大容量医学解释的适应性仍有待探索。
- 我们的结果表明，在这种范式的指导下，通用 VideoLM 具有向大容量医学成像的强大可移植性。
- 值得注意的是，AbSteering 的性能优于最先进的特定领域 CT 基础模型，后者经过大规模 CT 预训练，实现了卓越的检测灵敏度，同时减轻了幻觉。

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
