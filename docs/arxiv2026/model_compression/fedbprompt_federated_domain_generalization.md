# FedBPrompt: Federated Domain Generalization Person Re-Identification via Body Distribution Aware Visual Prompts

**会议**: 投稿中  
**arXiv**: [2603.12912](https://arxiv.org/abs/2603.12912)  
**代码**: 未提及  
**领域**: 模型压缩  
**关键词**: fedbprompt, federated, domain, generalization, person  

## 一句话总结
为了解决这个问题，我们提出了联合身体分布感知视觉提示（FedBPrompt），引入可学习的视觉提示来引导 Transformer 将注意力集中到以行人为中心的区域。
## 核心问题
虽然 Vision Transformer (ViT) 被广泛采用，但其全球注意力往往无法区分具有高度相似背景或不同观点的行人——FedDG-ReID 中跨客户端分布变化加剧了这一挑战。
## 关键方法
1. 为了解决这个问题，我们提出了联合身体分布感知视觉提示（FedBPrompt），引入可学习的视觉提示来引导 Transformer 关注以行人为中心的区域
2. 为了减轻高昂的通信成本，我们设计了一种基于提示的微调策略（PFTS），该策略冻结 ViT 主干并仅更新轻量级提示，从而在保持适应性的同时显着降低通信开销
## 亮点 / 我学到了什么
- 大量实验表明，BAPM 有效增强了特征辨别和跨域泛化，而 PFTS 仅在几轮聚合内就实现了显着的性能提升
- 为了减轻高昂的通信成本，我们设计了一种基于提示的微调策略（PFTS），该策略冻结 ViT 主干并仅更新轻量级提示，从而在保持适应性的同时显着降低通信开销
## 局限性 / 可改进方向
- 需要详细阅读原文以确认具体局限
- 潜在扩展方向待进一步分析

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 详见原文 | 详见原文 | — | — | — |

## 与我的研究方向的关联
- `agfu_foundation_pruning`
- `attention_aware_quant`
- `task_aware_token_compression`

## 评分
- 新颖性: ⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐
- 对我的价值: ⭐⭐⭐⭐
