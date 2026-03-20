# One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers

**会议**: arXiv 2026  
**arXiv**: [2603.12245](https://arxiv.org/abs/2603.12245)  
**作者**: Moayed Haji-Ali, Willi Menapace, Ivan Skorokhodov, Dogyun Park, Anil Kag
**代码**: 待确认  
**领域**: 扩散模型/生成 / LLM/NLP  
**关键词**: one, model, many, budgets, elastic  

## 一句话总结
扩散变换器 (DiT) 实现了高生成质量，但将 FLOP 锁定到图像分辨率，限制了原则上的延迟质量权衡，并在输入空间令牌之间统一分配计算，浪费了对不重要区域的资源分配。
## 背景与动机
Diffusion transformers (DiTs) achieve high generative quality but lock FLOPs to image resolution, limiting principled latency-quality trade-offs, and allocate computation uniformly across input spatial tokens, wasting resource allocation to unimportant regions.. We introduce Elastic Latent Interface Transformer (ELIT), a drop-in, DiT-compatible mechanism that decouples input image size from compute.

## 核心问题
扩散变换器 (DiT) 实现了高生成质量，但将 FLOP 锁定到图像分辨率，限制了原则上的延迟质量权衡，并在输入空间标记之间统一分配计算，从而将资源分配浪费到不重要的区域。我们引入了弹性潜在接口变换器 (ELIT)，这是一种直接的、与 DiT 兼容的机制，可将输入图像大小与计算解耦。
## 方法详解

### 整体框架
- We introduce Elastic Latent Interface Transformer (ELIT), a drop-in, DiT-compatible mechanism that decouples input image size from compute.
- Our approach inserts a latent interface, a learnable variable-length token sequence on which standard transformer blocks can operate.
- By training with random dropping of tail latents, ELIT learns to produce importance-ordered representations with earlier latents capturing global structure while later ones contain information to refine details.

### 关键设计
1. **关键组件1**: We introduce Elastic Latent Interface Transformer (ELIT), a drop-in, DiT-compatible mechanism that decouples input image size from compute.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 扩散变换器 (DiT) 实现了高生成质量，但将 FLOP 锁定到图像分辨率，限制了原则上的延迟质量权衡，并在输入空间令牌之间统一分配计算，浪费了对不重要区域的资源分配。
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
