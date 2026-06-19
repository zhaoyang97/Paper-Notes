---
title: >-
  [论文解读] Learning from Streaming Video with Orthogonal Gradients
description: >-
  [CVPR 2025][视频生成][流式视频学习] 针对流式视频学习中连续帧高度相关导致梯度冗余、模型崩溃的问题，提出正交梯度优化器（Orthogonal Optimizer），通过将当前梯度投影到历史梯度的正交分量来去相关，可无缝集成到 SGD/AdamW 中，在 DoRA、VideoMAE、未来预测三个场景下均显著恢复了从打乱训练到顺序训练的性能损失。
tags:
  - "CVPR 2025"
  - "视频生成"
  - "流式视频学习"
  - "正交梯度"
  - "优化器设计"
  - "时序相关性"
  - "非IID训练"
---

# Learning from Streaming Video with Orthogonal Gradients

**会议**: CVPR 2025  
**arXiv**: [2504.01961](https://arxiv.org/abs/2504.01961)  
**代码**: 无  
**领域**: 视频生成  
**关键词**: 流式视频学习, 正交梯度, 优化器设计, 时序相关性, 非IID训练

## 一句话总结
针对流式视频学习中连续帧高度相关导致梯度冗余、模型崩溃的问题，提出正交梯度优化器（Orthogonal Optimizer），通过将当前梯度投影到历史梯度的正交分量来去相关，可无缝集成到 SGD/AdamW 中，在 DoRA、VideoMAE、未来预测三个场景下均显著恢复了从打乱训练到顺序训练的性能损失。

## 研究背景与动机

1. **领域现状**：当前视频表示学习的标准做法是将长视频切成短片段，随机打乱后训练，以满足 SGD 等优化器对 IID（独立同分布）数据的假设。
2. **现有痛点**：当视频只能以连续流的形式获取时（如机器人在线学习、隐私保护场景不存储视频），IID 假设被打破，模型性能急剧下降甚至崩溃。作者在 DoRA 上展示，使用 AdamW 顺序训练时 ImageNet kNN 准确率从 74.4% 暴跌至 1.8%。
3. **核心矛盾**：连续视频帧之间变化极慢，导致相邻 batch 的梯度高度相似（余弦相似度趋近 1），优化器在同一方向上过度更新，无法学到新信息。
4. **本文目标** 如何在不存储/打乱视频的前提下，从顺序视频流中学到好的视觉表示？
5. **切入角度**：既然问题本质在于梯度相关性，那就在优化器层面去相关——只保留当前梯度相对于历史梯度的正交分量进行更新。
6. **核心 idea**：用正交投影把冗余的梯度信息剔除，只用"新信息"更新模型，使顺序训练也能逼近 IID 训练效果。

## 方法详解

### 整体框架
输入是一个连续的长视频流，模型按时间顺序依次处理视频片段。在每个训练步骤中，计算当前 batch 的梯度后，不直接用该梯度更新参数，而是先将其投影到历史梯度方向的正交分量上，再将这个去相关后的梯度送入标准优化器流程。该修改是一个即插即用的几何变换，可以应用于任意优化器。

### 关键设计

1. **正交梯度计算（Orthogonal Gradient Projection）**:

    - 功能：去除当前梯度中与历史梯度方向相同的冗余分量，只保留新信息。
    - 核心思路：给定当前梯度 $g_t$ 和历史梯度 $g_{t-1}$，正交分量为 $u_t = g_t - \text{proj}_{g_{t-1}}(g_t)$。投影通过余弦距离和向量范数实现：$\text{proj}_{g_{t-1}}(g_t) = \frac{g_t \cdot g_{t-1}}{g_{t-1} \cdot g_{t-1}} g_{t-1}$。当数据近似 IID 时 $\cos(g_{t-1}, g_t) \approx 0$，正交梯度约等于原始梯度，不影响正常训练；当数据高度相关时 $\cos(g_{t-1}, g_t) \approx 1$，正交分量很小，避免在同一方向过度优化。
    - 设计动机：从几何角度直接解决梯度冗余问题，计算开销极低（只需向量点积和范数），且对 IID 场景无副作用。

2. **指数移动平均动量（EMA Momentum for Robust Orthogonalization）**:

    - 功能：用 EMA 平滑历史梯度，减少单步噪声对正交投影的干扰。
    - 核心思路：维护原始梯度的 EMA：$c_t = \beta c_{t-1} + (1-\beta) g_t$（默认 $\beta=0.9$），然后用 $c_{t-1}$ 代替 $g_{t-1}$ 做正交投影：$u_t = g_t - \text{proj}_{c_{t-1}}(g_t)$。注意 EMA 是对原始梯度 $g_t$ 计算，而非正交分量 $u_t$，这样 EMA 保留了完整的梯度方向信息。
    - 设计动机：直接用单步梯度做正交投影对噪声敏感，类似标准优化器中"动量"的思想，用 EMA 平滑可以更稳定地捕捉梯度的主方向。

3. **多优化器适配（Orthogonal-SGD / Orthogonal-AdamW）**:

    - 功能：将正交梯度技术集成到 SGD 和 AdamW 两种主流优化器中。
    - 核心思路：在标准优化器算法中插入两行：(1) 计算正交梯度 $u_t$；(2) 更新 EMA $c_t$。然后将 $g_t$ 替换为 $u_t$ 送入后续的动量/二阶矩估计步骤。对于 AdamW，正交梯度 $u_t$ 作为输入进入一阶矩 $m_t$ 和二阶矩 $v_t$ 的更新，其余流程不变。
    - 设计动机：正交修改与优化器的具体设计正交，理论上可适用于任何基于梯度的优化器，实际验证了 SGD 和 AdamW 两种最常用的。

### 损失函数 / 训练策略
正交优化器不改变损失函数，它是在优化器层面的修改。三个实验场景分别使用各自原有的自监督损失函数（DoRA 用 DINO 风格的师生蒸馏损失，VideoMAE 用 masked autoencoder 重建损失，未来预测用像素级预测损失）。

## 实验关键数据

### 主实验

**DoRA 单视频预训练 (WTvenice → ImageNet)：**

| 初始化 | 优化器 | Linear Probe Top1 | kNN Top1 |
|--------|--------|-------------------|----------|
| DINO_ImageNet | AdamW | 6.1 | 1.8 |
| DINO_ImageNet | **Orth-AdamW** | **64.5** | **51.8** |
| Random | AdamW | 3.5 | 0.8 |
| Random | **Orth-AdamW** | **8.2** | **3.1** |

**VideoMAE 多视频预训练 (SSV2 → SSV2)：**

| 视频处理方式 | 优化器 | Linear-probe Top1 | Attn-probe Top1 |
|-------------|--------|-------------------|-----------------|
| Shuffled clips | AdamW | 19.0 | 54.9 |
| Shuffled clips | Orth-AdamW | 21.0 | 54.7 |
| Sequential (batch-along-time) | AdamW | 16.4 | 46.1 |
| Sequential (batch-along-time) | **Orth-AdamW** | **18.4** | **48.0** |
| Sequential (batch-along-video) | AdamW | 9.5 | 30.3 |
| Sequential (batch-along-video) | **Orth-AdamW** | **10.4** | **32.6** |

### 消融实验

| 配置 | kNN Top1 (DINO init) | 说明 |
|------|---------------------|------|
| IID训练 (shuffled) | 74.4 | 上界参考 |
| Sequential + AdamW | 1.8 | 顺序训练崩溃 |
| Sequential + Orth-AdamW | 51.8 | 正交梯度大幅恢复 |
| Sequential + Orth-SGD | ~更低 | SGD 收敛慢，但仍优于标准 SGD |

### 关键发现
- **正交梯度在极端场景下效果最显著**：DINO 初始化 + 顺序训练时，标准 AdamW 直接崩溃（1.8%），Orth-AdamW 恢复到 51.8%，差距达 50 个百分点。
- **对 IID 训练无害**：当数据已打乱时，正交优化器与标准优化器性能接近，具备良好的兼容性。
- **batch-along-video 比 batch-along-time 更难**：因为 batch 内样本来自不同视频的相邻帧，batch 间时间相关性极高。
- **梯度余弦相似度可视化**直观验证了方法有效性：正交优化器训练过程中，连续梯度的余弦相似度从接近 1 逐渐降低到接近 0，趋近 IID 训练的分布。

## 亮点与洞察
- **极简但有效的优化器改进**：只需在任意优化器中添加两行代码（正交投影 + EMA 更新），实现开销极低，但效果惊人。这种"在优化器层面解决数据分布问题"的思路非常优雅。
- **IID 兼容性设计**：正交梯度在 IID 场景下自动退化为原始梯度，不需要手动切换，这是一个很好的"无害"设计原则。
- **可迁移性强**：这个技巧不仅适用于视频流，还可以迁移到任何非 IID 场景——如持续学习、在线学习、联邦学习中数据分布非均匀的情况。

## 局限与展望
- **性能仍未完全恢复**：虽然从 1.8% 恢复到 51.8%（DINO init），但距 IID 训练的 74.4% 还有不小差距，说明仅靠优化器去相关还不够。
- **只考虑了一步历史**：正交投影只相对于 EMA（近似一步历史方向），对于更长程的梯度相关模式可能不够。
- **未探索共轭梯度等更强方法**：作者提到共轭梯度法理论上可能更优，但因计算成本更高而未深入。
- **对计算开销分析不够**：虽然声称开销低，但维护额外的 EMA 状态和做向量投影的具体 overhead 未量化。

## 相关工作与启发
- **vs OGD（持续学习中正交梯度）[Farajtabar et al.]**: 他们在任务切换后做正交投影以避免灾难性遗忘，本文将其扩展到单任务流式视频中，且在每个训练步都做正交投影并引入 EMA 平滑。
- **vs Baby Learning [Orhan et al.]**: 他们研究流式视频训练但主要关注不同优化器的对比和 replay buffer，未显式处理梯度相关性。本文直接在梯度层面解决问题，更本质。
- **vs Replay Buffer 方法**: Replay buffer 通过存储历史数据近似 IID，但需要额外存储和访问。正交优化器不需要存储数据，更适合隐私保护场景。

## 评分
- 新颖性: ⭐⭐⭐⭐ 将正交梯度从持续学习迁移到流式视频学习，idea 清晰简洁
- 实验充分度: ⭐⭐⭐⭐ 三个场景、多种初始化、多种 batching 策略，梯度可视化直观
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，图示直观，算法表述规范
- 价值: ⭐⭐⭐⭐ 实用性强的优化器改进，对在线学习和流式处理有广泛参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Teller: Real-Time Streaming Audio-Driven Portrait Animation with Autoregressive Motion Generation](teller_real-time_streaming_audio-driven_portrait_animation_with_autoregressive_m.md)
- [\[CVPR 2025\] Learning Temporally Consistent Video Depth from Video Diffusion Priors](learning_temporally_consistent_video_depth_from_video_diffusion_priors.md)
- [\[CVPR 2025\] 4Real-Video: Learning Generalizable Photo-Realistic 4D Video Diffusion](4real-video_learning_generalizable_photo-realistic_4d_video_diffusion.md)
- [\[ICLR 2026\] Streaming Autoregressive Video Generation via Diagonal Distillation](../../ICLR2026/video_generation/streaming_autoregressive_video_generation_via_diagonal_distillation.md)
- [\[ICCV 2025\] Disentangled World Models: Learning to Transfer Semantic Knowledge from Distracting Videos for Reinforcement Learning](../../ICCV2025/video_generation/disentangled_world_models_learning_to_transfer_semantic_knowledge_from_distracti.md)

</div>

<!-- RELATED:END -->
