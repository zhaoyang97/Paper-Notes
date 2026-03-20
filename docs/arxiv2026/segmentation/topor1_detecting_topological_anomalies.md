# Topo-R1: Detecting Topological Anomalies via Vision-Language Models

**会议**: arXiv 2026  
**arXiv**: [2603.13054](https://arxiv.org/abs/2603.13054)  
**作者**: Meilong Xu, Qingqiao Hu, Xiaoling Hu, Shahira Abousamra, Xin Yu et al.
**代码**: 待确认  
**领域**: 语义分割 / LLM/NLP  
**关键词**: topo-r1, detecting, topological, anomalies, vision-language  

## 一句话总结
拓扑正确性对于血管、神经纤维和道路网络等管状结构至关重要。
## 背景与动机
Topological correctness is crucial for tubular structures such as blood vessels, nerve fibers, and road networks.. Existing topology-preserving methods rely on domain-specific ground truth, which is costly and rarely transfers across domains.

## 核心问题
拓扑正确性对于血管、神经纤维和道路网络等管状结构至关重要。现有的拓扑保持方法依赖于特定领域的地面实况，这种方法成本高昂且很少跨域传输。
## 方法详解

### 整体框架
- To bridge this gap, we develop an automated data-curation pipeline that synthesizes diverse topological anomalies with verifiable annotations across progressively difficult levels, thereby constructing the first large-scale, multi-domain benchmark for this task.
- We then introduce Topo-R1, a framework that endows VLMs with topology-aware perception via two-stage training: supervised fine-tuning followed by reinforcement learning with Group Relative Policy Optimization (GRPO).
- Central to our approach is a topology-aware composite reward that integrates type-aware Hungarian matching for structured error classification, spatial localization scoring, and a centerline Dice (clDice) reward that directly penalizes connectivity disruptions, thereby jointly incentivizing semantic precision and structural fidelity.

### 关键设计
1. **关键组件1**: To bridge this gap, we develop an automated data-curation pipeline that synthesizes diverse topological anomalies with verifiable annotations across progressively difficult levels, thereby constructing the first large-scale, multi-domain benchmark for this task.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 视觉语言模型（VLM）是自然的候选者；然而，我们发现最先进的 VLM 几乎是随机执行的，缺乏识别密集结构中稀疏连接错误所需的细粒度、拓扑感知的感知能力。
- 为了弥补这一差距，我们开发了一个自动化数据管理管道，该管道可以在逐渐困难的级别上综合不同的拓扑异常和可验证的注释，从而为该任务构建第一个大规模、多领域的基准。
- 大量实验表明，Topo-R1 建立了无注释拓扑质量评估的新范例，在所有评估协议中始终优于通用 VLM 和监督基线。
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
