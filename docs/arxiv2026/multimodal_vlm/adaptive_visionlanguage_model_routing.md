# Adaptive Vision-Language Model Routing for Computer Use Agents

**会议**: 投稿中  
**arXiv**: [2603.12823](https://arxiv.org/abs/2603.12823)  
**代码**: 未提及  
**领域**: 多模态VLM  
**关键词**: adaptive, vision-language, model, routing, computer  

## 一句话总结
我们提出了自适应 VLM 路由（AVR），这是一个在 CUA 协调器和 VLM 池之间插入轻量级语义路由层的框架。

## 核心问题
然而，不同 VLM 的接地精度差异很大，而当前的 CUA 系统通常将每个动作路由到单个固定模型，无论难度如何。

## 关键方法
1. 我们提出了 \textbf{Adaptive VLM Routing} (AVR)，这是一个在 CUA 协调器和 VLM 池之间插入轻量级语义路由层的框架
2. 然而，不同 VLM 的接地精度差别很大，而当前的 CUA 系统通常将每个动作路由到单个固定模型，无论难度如何
3. 对于每个工具调用，AVR 都会根据多模态嵌入估计操作难度，探测小型 VLM 来测量置信度，并将操作路由到预测精度满足目标可靠性阈值的最便宜模型
4. 对于具有先前 UI 交互记忆的 \textit{warm} 代理，检索到的上下文进一步缩小了小型模型和大型模型之间的能力差距，允许在不升级的情况下处理许多操作

## 亮点 / 我学到了什么
- 详见原文实验部分


## 局限性 / 可改进方向
- 需要详细阅读原文以确认具体局限
- 潜在扩展方向待进一步分析

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 详见原文 | 详见原文 | — | — | — |

## 与我的研究方向的关联
- `asymmetric_multimodal_scaling`
- `beyond_amradio`
- `panoramic_spatial_reasoning`

## 评分
- 新颖性: ⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐
- 对我的价值: ⭐⭐⭐⭐
