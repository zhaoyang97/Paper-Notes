# Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning

**会议**: arXiv 2026  
**arXiv**: [2603.12816](https://arxiv.org/abs/2603.12816)  
**作者**: Gyutae Oh, Jungwoo Bae, Jitae Shin
**代码**: 待确认  
**领域**: 模型压缩/高效推理  
**关键词**: residual, sodap, self-organizing, domain-adaptive, prompting  

## 一句话总结
持续学习（CL）会遭受灾难性遗忘，而在领域增量学习（DIL）中，这种情况会更加严重，因为任务标识符不可用，并且存储过去的数据也不可行。
## 背景与动机
Continual learning (CL) suffers from catastrophic forgetting, which is exacerbated in domain-incremental learning (DIL) where task identifiers are unavailable and storing past data is infeasible.. While prompt-based CL (PCL) adapts representations with a frozen backbone, we observe that prompt-only improvements are often insufficient due to suboptimal prompt selection and classifier-level instability under domain shifts.

## 核心问题
持续学习（CL）遭受灾难性遗忘，这种情况在域增量学习（DIL）中加剧，其中任务标识符不可用并且存储过去的数据是不可行的。虽然基于提示的 CL（PCL）采用冻结主干的表示，但我们观察到，由于提示选择次优和域转移下分类器级别的不稳定，仅提示的改进通常是不够的。
## 方法详解

### 整体框架
- While prompt-based CL (PCL) adapts representations with a frozen backbone, we observe that prompt-only improvements are often insufficient due to suboptimal prompt selection and classifier-level instability under domain shifts.
- We propose Residual SODAP, which jointly performs prompt-based representation adaptation and classifier-level knowledge preservation.
- Our framework combines $α$-entmax sparse prompt selection with residual aggregation, data-free distillation with pseudo-feature replay, prompt-usage--based drift detection, and uncertainty-aware multi-loss balancing.

### 关键设计
待深读后补充。

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 虽然基于提示的 CL (PCL) 采用冻结骨干网来适应表示，但我们观察到，由于提示选择次优和域转移下分类器级别的不稳定，仅提示的改进通常是不够的。
- 在没有任务 ID 或额外数据存储的三个 DIL 基准测试中，Residual SODAP 实现了最先进的 AvgACC/AvgF 0.850/0.047 (DR)、0.760/0.031（皮肤癌）和 0.995/0.003 (CORe50)。
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
