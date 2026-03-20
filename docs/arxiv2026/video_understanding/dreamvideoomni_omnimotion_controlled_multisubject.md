# DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning

**会议**: arXiv 2026  
**arXiv**: [2603.12257](https://arxiv.org/abs/2603.12257)  
**作者**: Yujie Wei, Xinyu Liu, Shiwei Zhang, Hangjie Yuan, Jinbo Xing
**代码**: 待确认  
**领域**: 视频理解 / 强化学习  
**关键词**: dreamvideo-omni, omni-motion, controlled, multi-subject, video  

## 一句话总结
虽然大规模扩散模型彻底改变了视频合成，但实现对多主体身份和多粒度运动的精确控制仍然是一个重大挑战。
## 背景与动机
While large-scale diffusion models have revolutionized video synthesis, achieving precise control over both multi-subject identity and multi-granularity motion remains a significant challenge.. Recent attempts to bridge this gap often suffer from limited motion granularity, control ambiguity, and identity degradation, leading to suboptimal performance on identity preservation and motion control.

## 核心问题
虽然大规模扩散模型彻底改变了视频合成，但实现对多主体身份和多粒度运动的精确控制仍然是一个重大挑战。
## 方法详解

### 整体框架
- In this work, we present DreamVideo-Omni, a unified framework enabling harmonious multi-subject customization with omni-motion control via a progressive two-stage training paradigm.
- To ensure robust and precise controllability, we introduce a condition-aware 3D rotary positional embedding to coordinate heterogeneous inputs and a hierarchical motion injection strategy to enhance global motion guidance.
- Furthermore, to resolve multi-subject ambiguity, we introduce group and role embeddings to explicitly anchor motion signals to specific identities, effectively disentangling complex scenes into independent controllable instances.
- In the second stage, to mitigate identity degradation, we design a latent identity reward feedback learning paradigm by training a latent identity reward model upon a pretrained video diffusion backbone.

### 关键设计
1. **关键组件1**: In the first stage, we integrate comprehensive control signals for joint training, encompassing subject appearances, global motion, local dynamics, and camera movements.
2. **关键组件2**: To ensure robust and precise controllability, we introduce a condition-aware 3D rotary positional embedding to coordinate heterogeneous inputs and a hierarchical motion injection strategy to enhance global motion guidance.
3. **关键组件3**: In the second stage, to mitigate identity degradation, we design a latent identity reward feedback learning paradigm by training a latent identity reward model upon a pretrained video diffusion backbone.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 在我们精心策划的大规模数据集和用于多主体和全运动控制评估的综合性 DreamOmni Bench 的支持下，DreamVideo-Omni 在生成具有精确可控性的高质量视频方面表现出了卓越的性能。
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
- 对我的价值: ⭐⭐⭐
