---
title: >-
  [论文解读] PROFIT: A Specialized Optimizer for Deep Fine Tuning
description: >-
  [NeurIPS 2025][机器人][微调优化器] PROFIT 将微调视为时间维度上的多任务学习问题，通过将新任务梯度对"回归平衡点"方向做正交化投影，实现了无需额外数据或参数的抗遗忘微调优化器。 微调预训练模型已成为深度学习的主流范式——从自动驾驶到 LLM，从零训练的成本越来越高。然而现有的 SGD、Adam、Ada…
tags:
  - "NeurIPS 2025"
  - "机器人"
  - "微调优化器"
  - "灾难性遗忘"
  - "梯度正交化"
  - "多任务学习"
  - "近端微调"
---

# PROFIT: A Specialized Optimizer for Deep Fine Tuning

**会议**: NeurIPS 2025  
**arXiv**: [2412.01930](https://arxiv.org/abs/2412.01930)  
**代码**: 暂无  
**领域**: 优化  
**关键词**: 微调优化器, 灾难性遗忘, 梯度正交化, 多任务学习, 近端微调

## 一句话总结

PROFIT 将微调视为时间维度上的多任务学习问题，通过将新任务梯度对"回归平衡点"方向做正交化投影，实现了无需额外数据或参数的抗遗忘微调优化器。

## 研究背景与动机

微调预训练模型已成为深度学习的主流范式——从自动驾驶到 LLM，从零训练的成本越来越高。然而现有的 SGD、Adam、AdamW 等优化器都是为**从零训练**设计的，对初始化状态不做假设。微调的场景完全不同：起点已经是一个收敛良好的模型，我们需要利用这一先验。

微调的核心挑战是**灾难性遗忘**：在新数据上训练时模型快速丢失旧知识。现有解决方案各有局限：
- **LWF**（Learning Without Forgetting）需要存储旧模型响应并做蒸馏，增加了数据流水线复杂度。
- **LoRA** 等参数高效方法减少计算但不提升精度。
- **冻结骨干+小学习率** 限制了模型的适应能力。

作者的关键洞察：既然微调从一个收敛点出发，模型偏离该点后自然有"回归"的倾向——偏离方向 $\Delta$ 的负方向 $-\Delta$ 就是旧损失函数的近似梯度方向。这构成了一个天然的"时间多任务学习"问题：任务一（旧模型保持）与任务二（新数据适应）可能存在梯度冲突。

## 方法详解

### 整体框架

PROFIT（PROximal FIne Tuning）是一个优化器包装器，接受两个标准优化器：主优化器 $\mathbf{O}$ 和参考优化器 $\mathbf{O}^{(\text{ref})}$。每一步训练的流程为：

1. 保存当前参数 $\theta_{\text{ref}} \leftarrow \theta$
2. 用参考优化器在新数据上走 $n_{\text{ref}}$ 步小步，到达 $\theta'$
3. 计算偏移 $\Delta = \theta' - \theta_{\text{ref}}$
4. 采一个新批次计算梯度 $\mathbf{g}$
5. 如果 $\langle \Delta, \mathbf{g} \rangle < 0$（即梯度方向冲突），将 $\mathbf{g}$ 正交化到 $\Delta$
6. 恢复参数 $\theta \leftarrow \theta_{\text{ref}}$，沿正交化后的 $\mathbf{g}$ 走一步

### 关键设计

1. **时序梯度正交化**: 借鉴多任务学习中的 PCGrad 方法。$\Delta$ 代表旧任务的隐含梯度（因为回到收敛点的方向就是旧损失的下降方向），$\mathbf{g}$ 代表新任务梯度。当二者冲突时（点积 $\omega < 0$），将 $\mathbf{g}$ 投影到 $\Delta$ 的正交补空间：$\mathbf{g} \leftarrow \mathbf{g} - \frac{\langle \mathbf{g}, \Delta \rangle}{\|\Delta\|^2} \Delta$。关键是只正交化 $\mathbf{g}$，不动 $\Delta$，因为旧数据可能不可访问，$\Delta$ 的信息更珍贵。

2. **参考步骤机制**: 参考优化器 $\mathbf{O}^{(\text{ref})}$（推荐 SGD）用小学习率探索损失曲面，$n_{\text{ref}}$ 步的累积位移 $\Delta$ 编码了局部曲面形状信息。$\lambda_{\text{ref}}$ 控制探索速度，通常设在 $\lambda_{\text{main}}/10$ 到 $\lambda_{\text{main}}/10000$ 之间。实验中所有设置使用 $n_{\text{ref}}=1$。

3. **理论保证**:

    - **定理 3.1（旧数据正确性）**: 在收敛点附近，损失曲面 $L(\mathbf{x}) \approx L(\mathbf{x}_0) + 0.5(\mathbf{x}-\mathbf{x}_0)^T \mathbf{H}_0 (\mathbf{x}-\mathbf{x}_0)$，因此 $-\Delta$ 是有效的梯度下降方向，PROFIT 的一步能降低旧数据损失。
    - **定理 3.2（稳定点）**: PROFIT 仅在损失曲面在 $\theta$ 和 $\theta'$ 之间完全线性时产生零更新——这在高维深度网络中几乎不可能发生。
    - **定理 3.4（收敛性）**: PROFIT 收敛到主优化器 $\mathbf{O}$ 的收敛点或定理 3.2 描述的稳定点。

### 损失函数 / 训练策略

- 使用任务原始损失函数，无需修改。
- 假设预训练和微调数据来自**近端分布**（distributional overlap）。
- 对于非近端场景（如 ImageNet→VTAB），提供了"warmup"方案：先用 AdamW 微调 10 个 epoch 缩小分布距离，再切换到 PROFIT。
- 内存开销约增加 25%（用于存储 $\theta_{\text{ref}}$），但可通过利用动量缓冲区近似消除。

## 实验关键数据

### 主实验

| 任务 | 方法 | 旧数据指标 | 新数据指标 |
|------|------|-----------|-----------|
| 2D回归(MLP) | 全模型微调 | 0.705 误差 | 0.504 误差 |
| 2D回归(MLP) | Head微调 | 0.110 误差 | 0.572 误差 |
| 2D回归(MLP) | **PROFIT** | **0.046** 误差 | **0.501** 误差 |
| CIFAR10→100 (ViT-Tiny) | Adam | 56.00% / 58.99% | — |
| CIFAR10→100 (ViT-Tiny) | Lookahead | 55.64% / 61.35% | — |
| CIFAR10→100 (ViT-Tiny) | **PROFIT** | **58.53%** / **62.20%** | — |
| CIFAR10→100 (ViT-Small) | Adam | 58.60% / 63.93% | — |
| CIFAR10→100 (ViT-Small) | **PROFIT** | **59.02%** / **65.44%** | — |

### 消融实验

| 设置 | ADE@8s | FDE@8s | 说明 |
|------|--------|--------|------|
| Waymo Car→Car 基线 | 1.327m | 2.581m | 未微调 |
| Car→Car 全模型微调 | 1.322m | 2.548m | 小幅提升 |
| Car→Car **PROFIT** | **1.299m** | **2.489m** | 更优 |
| Car→Ped 全模型微调 | 0.621m | 1.242m | 领域迁移 |
| Car→Ped Head微调 | 0.724m | 1.544m | 差于全模型 |
| Car→Ped **PROFIT** | **0.579m** | **1.145m** | 显著提升 |

| 任务 | AdamW | PROFIT | 说明 |
|------|-------|--------|------|
| DriveLM VQA 准确率 | 62.21% | **67.88%** | VLM微调 |
| DriveLM Final Score | 56.98 | **59.16** | 综合评分 |

### 关键发现

- PROFIT 在所有骨干网络（ResNet-18, ViT-Tiny, ViT-Small）和所有对比优化器上都取得了更好的 CIFAR10/100 权衡。
- 在 VTAB-1K 非近端场景下，朴素 PROFIT 失效（Clevr-Count 仅 12.6%），但加上 AdamW warmup 后在 19 个任务中的 15 个超过了全模型微调。
- Car→Ped 跨领域微调中，PROFIT 的 FDE@8s 提升了 7.8%（1.242m→1.145m），远超其他方法。

## 亮点与洞察

- **视角独特**: 将微调重新定义为时序多任务学习，开辟了一个新的优化器设计方向。
- **实用性极强**: 仅需包装现有优化器，无需改模型架构、无需旧数据、无需额外参数。
- **理论简洁有力**: 基于 Hessian 正定性的正确性证明直观且有说服力。
- 提出"PROFIT 可作为模型维护的标准操作"——即使在相同数据上继续训练也能带来提升。

## 局限与展望

- **近端假设限制**: 要求预训练和微调数据分布相似，非近端场景需要额外的 warmup 步骤。
- **内存开销**: 存储 $\theta_{\text{ref}}$ 增加约 25% 显存，大模型全参数微调时可能成为瓶颈。
- **额外前向传播成本**: $n_{\text{ref}}$ 步参考计算增加训练时间（虽然通常 $n_{\text{ref}}=1$）。
- $\lambda_{\text{ref}}$ 的最优值高度依赖于具体问题的损失曲面，缺乏自动调参机制。

## 相关工作与启发

- 核心灵感来自 PCGrad（多任务梯度冲突解决），但从空间维度扩展到时间维度。
- 与 LWF 的数据驱动锚点思路异曲同工，但避免了存储旧数据/模型快照。
- 与 LoRA 互补而非竞争：PROFIT 侧重提升微调精度，LoRA 侧重效率。

## 评分

- 新颖性: ⭐⭐⭐⭐ 微调作为时序多任务学习的视角新颖，正交化设计简洁优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 从2D玩具到CIFAR到VLM到自动驾驶，跨度非常大
- 写作质量: ⭐⭐⭐⭐ 动机清晰，图示直观，理论与实验衔接自然
- 价值: ⭐⭐⭐⭐⭐ 即插即用的微调优化器，实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](autovla_a_vision-language-action_model_for_end-to-end_autonomous_driving_with_ad.md)
- [\[NeurIPS 2025\] Time Reversal Symmetry for Efficient Robotic Manipulations in Deep Reinforcement Learning](time_reversal_symmetry_for_efficient_robotic_manipulations_in_deep_reinforcement.md)
- [\[ICLR 2026\] Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning](../../ICLR2026/robotics/cosmos_policy_fine-tuning_video_models_for_visuomotor_control_and_planning.md)
- [\[ICML 2025\] Learning to Stop: Deep Learning for Mean Field Optimal Stopping](../../ICML2025/robotics/learning_to_stop_deep_learning_for_mean_field_optimal_stopping.md)
- [\[ACL 2025\] CHEER-Ekman: Fine-grained Embodied Emotion Classification](../../ACL2025/robotics/cheer-ekman_fine-grained_embodied_emotion_classification.md)

</div>

<!-- RELATED:END -->
