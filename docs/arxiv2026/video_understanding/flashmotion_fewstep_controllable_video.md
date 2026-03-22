# FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance

**会议**: arXiv 2026  
**arXiv**: [2603.12146](https://arxiv.org/abs/2603.12146)  
**作者**: Quanhao Li, Zhen Xing, Rui Wang, Haidong Cao, Qi Dai
**代码**: 待确认  
**领域**: 视频理解 / 扩散模型/生成  
**关键词**: flashmotion, few-step, controllable, video, generation  

## 一句话总结
轨迹可控视频生成的最新进展取得了显着的进展。

## 背景与动机
Recent advances in trajectory-controllable video generation have achieved remarkable progress.. Previous methods mainly use adapter-based architectures for precise motion control along predefined trajectories.

## 核心问题
然而，所有这些方法都依赖于多步骤去噪过程，导致大量的时间冗余和计算开销。

## 方法详解

### 整体框架
- To bridge this gap, we introduce FlashMotion, a novel training framework designed for few-step trajectory-controllable video generation.
- For evaluation, we introduce FlashBench, a benchmark for long-sequence trajectory-controllable video generation that measures both video quality and trajectory accuracy across varying numbers of foreground objects.

### 关键设计
1. **关键组件1**: We first train a trajectory adapter on a multi-step video generator for precise trajectory control.
2. **关键组件2**: Finally, we finetune the adapter using a hybrid strategy that combines diffusion and adversarial objectives, aligning it with the few-step generator to produce high-quality, trajectory-accurate videos.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 轨迹可控视频生成的最新进展取得了显着的进展。
- 为了进行评估，我们引入了 FlashBench，它是长序列轨迹可控视频生成的基准，可测量不同数量前景对象的视频质量和轨迹精度。
- 对两种适配器架构的实验表明，FlashMotion 在视觉质量和轨迹一致性方面都超越了现有的视频蒸馏方法和之前的多步模型。

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
