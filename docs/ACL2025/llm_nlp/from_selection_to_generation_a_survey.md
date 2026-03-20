---
title: "From Selection to Generation: A Survey of LLM-based Active Learning"
conference: "ACL 2025"
arxiv: "2502.11767"
domain: "nlp_generation"
keywords: ["active learning", "LLM", "data selection", "data generation", "annotation", "survey"]
---

# From Selection to Generation: A Survey of LLM-based Active Learning

## 一句话总结

首篇系统综述 LLM 时代的主动学习（Active Learning），提出以 Querying（选择/生成）和 Annotation（标注）为核心的分类体系，全面梳理 LLM 如何变革传统主动学习的选择-标注流程。

## 研究背景与动机

1. 传统主动学习通过选择最有信息量的数据点进行标注来提高模型效率，但仅限于从固定未标注池中选择
2. LLM 的出现使得主动学习范式发生根本变化：不仅可以选择数据，还可以生成全新的数据实例
3. LLM 可以扮演标注者角色，大幅降低人工标注成本
4. LLM-based AL 还可以帮助减少训练成本（如 SFT），而非仅仅减少标注成本
5. 现有综述主要关注传统主动学习技术，缺乏对 LLM 驱动 AL 方法的系统性梳理
6. 在高质量数据和高效模型训练日益重要的 LLM 时代，需要一份全面的 LLM-based AL 综述

## 方法详解

### 分类体系

围绕主动学习的两大核心组件构建分类法：

**一、Querying（获取数据实例）**

| 类别 | 说明 |
|------|------|
| Traditional Selection | 经典不确定性/多样性采样（Least Confidence, BADGE, BALD 等） |
| LLM-based Selection | 用 LLM 评估或排序未标注样本的信息量（ActiveLLM, SelectLLM, Ask-LLM） |
| LLM-based Generation | 用 LLM 生成全新数据实例，突破固定数据池限制 |
| Hybrid | 结合选择和生成的混合策略（NoiseAL, CAL） |

**二、Annotation（标注数据实例）**

| 类别 | 说明 |
|------|------|
| Human Annotation | 传统人工标注，质量高但成本高 |
| LLM-based Annotation | 用 LLM 作为标注者，成本远低于人工 |
| Hybrid | 人机协作标注，平衡效率与准确性 |

### 完整 AL 循环

算法描述为五步迭代流程：
1. **Initialize**: 用 LLM 标注初始数据集或生成起步数据，解决冷启动问题
2. **Query**: 用 LLM 选择或生成最有信息量的数据实例
3. **Annotate**: 用 LLM/人工/混合方式标注
4. **Train**: 用新标注数据更新目标模型
5. **Stop**: 达到预算限制或模型收敛时终止

### 关键方法梳理

**LLM-based Selection 代表方法**:
- **ActiveLLM**: 完全无监督地用 LLM 评估不确定性和多样性
- **SelectLLM**: 直接 prompt LLM 评估和排序未标注样本
- **ActivePrune**: 用 LLM 剪枝大规模未标注池，减少传统采集函数的计算负担
- **Ask-LLM**: prompt LLM 评估训练样本的质量

**LLM-based Generation 代表方法**:
- **池内生成**: 结合 k-NN + 困惑度策略改进少样本选择（Margatina et al., 2023）
- **池外生成**: APE 框架用 Query-by-Committee + CoT 合成新 prompt；rejection sampling 确保生成质量
- **混合生成**: NoiseAL 用小 LLM 初筛 + 大 LLM 标注；CAL 用密度聚类 + LLM 查询修正偏差

**LLM-based Annotation 范式**:
- LLM 直接标注 vs 人机协作标注
- 挑战：LLM 标注存在偏差和不一致性
- 动态任务路由：简单样本给 LLM，困难样本给人工

### 扩展讨论

- **Stopping**: LLM 时代需同时考虑 API 调用成本和人工标注成本
- **AL 影响 LLM 学习范式**: AL 应用于 ICL 样本选择、SFT 数据选择、RLHF 偏好数据收集
- **应用领域**: 文本分类、问答、情感分析、去偏、翻译、实体匹配等

## 实验关键数据

作为综述论文，本文不包含原创实验，但汇总了各方法的关键结果：

### 代表性方法效果

| 方法 | 任务 | 核心发现 |
|------|------|---------|
| ActiveLLM | 文本分类 | 无监督 LLM 选择可匹配传统 AL 方法 |
| SelectLLM | 少样本学习 | LLM 排序 + k-NN 优于随机和不确定性选择 |
| Ask-LLM | 数据质量评估 | LLM 评分可有效过滤低质量训练数据 |
| Rejection Sampling | 数据生成 | 仅保留达标样本，确保生成质量 |
| NoiseAL | 混合策略 | 小 LLM 选择 + 大 LLM 标注的两阶段流程有效 |

### 关键发现汇总

1. LLM-based selection 在少样本和冷启动场景中优势明显
2. LLM 生成数据可以有效扩展训练集，特别是在标注数据稀缺时
3. 混合人机标注在成本-质量权衡上最优
4. 传统不确定性采样在 LLM few-shot 设置下可能表现不佳
5. LLM-based AL 的停止条件需要同时考虑 API 成本和标注成本

## 亮点与洞察

- **范式转移**: 从"从固定池中选择"到"可以无限生成新数据"，是 AL 范式的根本性扩展
- **分类法清晰**: Querying × Annotation 的二维分类简洁直观，覆盖了现有方法的组合空间
- **LLM 的多重角色**: LLM 同时作为 selector、generator、annotator、evaluator，多重角色在 AL 循环中的协同
- **成本模型变化**: 传统 AL 优化标注成本，LLM-based AL 还需优化 API 调用成本和计算成本
- **冷启动解决**: LLM 生成初始数据可天然解决传统 AL 的冷启动问题

## 局限性

- 综述覆盖面广但深度有限，部分方法的描述较简略
- 缺少跨方法的统一实验对比（各方法在不同设置下评测）
- LLM-based AL 的理论基础（如 PAC 学习框架的扩展）讨论不足
- 未充分讨论 LLM 标注质量的系统性评估方法
- 对多语言和低资源语言场景的覆盖有限

## 相关工作

- 传统 AL 综述（Settles, 2009; Ren et al., 2021; Zhan et al., 2022）
- LLM 数据生成（Yang et al., 2024; Mukherjee et al., 2024）
- LLM 作为标注器（Wang et al., 2024; Kholodna et al., 2024）
- 少样本学习选择策略（Margatina et al., 2023）
- 混合人机标注系统

## 评分

- **新颖性**: ★★★★☆ — 首篇 LLM-based AL 系统综述，分类法有贡献
- **技术深度**: ★★★☆☆ — 综述性质，方法描述为主，缺少新算法
- **实验充分性**: ★★☆☆☆ — 无原创实验，靠引用文献数据
- **实用价值**: ★★★★☆ — 对该领域研究者是好的入门和索引，分类法可指导方法设计
