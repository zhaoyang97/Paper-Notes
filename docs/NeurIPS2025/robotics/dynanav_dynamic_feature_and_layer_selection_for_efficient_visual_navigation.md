---
title: >-
  [论文解读] DynaNav: Dynamic Feature and Layer Selection for Efficient Visual Navigation
description: >-
  [NeurIPS 2025][机器人][visual navigation] 提出 DynaNav，通过可训练的硬特征选择器和基于贝叶斯优化的 early-exit 机制，根据场景复杂度动态调整特征与层的使用，在视觉导航中实现 2.26× FLOPs 降低、42.3% 推理时间减少，同时保持甚至提升导航性能。
tags:
  - "NeurIPS 2025"
  - "机器人"
  - "visual navigation"
  - "dynamic inference"
  - "early exit"
  - "feature selection"
  - "efficient deployment"
---

# DynaNav: Dynamic Feature and Layer Selection for Efficient Visual Navigation

**会议**: NeurIPS 2025  
**arXiv**: [2509.21930](https://arxiv.org/abs/2509.21930)  
**代码**: 待确认  
**领域**: 机器人  
**关键词**: visual navigation, dynamic inference, early exit, feature selection, efficient deployment  

## 一句话总结

提出 DynaNav，通过可训练的硬特征选择器和基于贝叶斯优化的 early-exit 机制，根据场景复杂度动态调整特征与层的使用，在视觉导航中实现 2.26× FLOPs 降低、42.3% 推理时间减少，同时保持甚至提升导航性能。

## 背景与动机

视觉导航是机器人和具身 AI 的核心能力。近年来，ViNT、NoMaD 等基础模型展示了跨平台、跨环境的泛化能力，但它们依赖深层 Transformer 解码器，计算开销大，难以部署在边缘设备上。同时，这些模型以"黑盒"方式运行，缺乏可解释性。

作者从人类视觉系统获得启发：人脑不会对所有视觉任务都激活全部神经元，而是根据任务复杂度动态调配资源。由此提出两个关键研究问题：

1. 是否每个导航场景都需要激活所有 Transformer 层？
2. 解码过程中哪些特征最重要？能否识别出对导航最关键的区域或像素？

## 核心问题

如何在不牺牲导航性能的前提下，大幅降低视觉导航基础模型的计算开销，同时提升模型的可解释性？

## 方法详解

### 整体框架

DynaNav 基于 EfficientNet-B0 编码器 + Transformer 解码器架构，在此基础上引入两个动态机制：

1. **动态特征选择器**（Dynamic Feature Selector）：在特征进入 Transformer 解码器前生成稀疏掩码
2. **动态层推理**（Dynamic Layer Inference）：基于 early-exit 策略根据场景复杂度提前终止计算

### 特征提取

使用两个 EfficientNet-B0 实例：
- 一个处理连续观测帧序列 $\mathbf{o}_{t-p:t}$，提取 $\psi(\mathbf{o}_i) \in \mathbb{R}^{H \times W \times C}$
- 另一个采用 early fusion 策略处理当前观测与目标图像的拼接 $\phi([\mathbf{o}_t; \mathbf{o}_s])$

### 动态硬特征选择器

核心创新之一。选择器 $f(\cdot)$ 是一个基于 Gumbel-Softmax 的分类网络：

1. 将编码特征通过 MLP 投影到 $\mathbb{R}^{H \times W \times C \times 2}$ 空间
2. 对每个像素的每个通道进行像素级 Gumbel-Softmax 操作，计算选择概率
3. 生成二值掩码 $\mathbf{m}_i \in \mathbb{R}^{H \times W}$，过滤掉与导航预测无关的像素

温度参数 $\tau$ 控制选择的"硬度"。训练过程中选择器逐步学会过滤冗余特征，可视化的显著性图可以直观展示模型关注的区域，增强可解释性。

### 动态 Transformer 层推理（Early Exit）

**Feature-Aware Early Exit 策略**：

- 在每个中间解码层使用动作一致性条件判断是否提前退出：$|h(\mathbf{x}_i) - h(\mathbf{x}_{i-1})|_2 \leq \eta_i$
- 更激进的策略：当目标状态与当前观测的 L2 差异低于阈值时，直接跳过整个 Transformer 解码器
- 特征选择器与 early exit 集成，利用被掩码的特征数量辅助判断退出条件

**自适应阈值优化**：采用贝叶斯优化（Bayesian Optimization）确定最优 early-exit 阈值 $\eta = \{\eta_1, \eta_2, \dots, \eta_N\}$，优化目标为最大化预测动作与真实动作的余弦相似度，同时满足三个约束：

- 推理时间约束：平均推理时间 $\leq \mathcal{T}_{\max}$
- GPU 显存约束：峰值显存 $\leq G_{\max}$
- FLOPs 约束：平均计算量 $\leq F_{\max}$

### 训练目标

训练损失为动作预测和航点距离预测的联合似然：

$$\mathcal{L} = \mathbb{E}[\log p(\mathbf{a}_t^{\text{gt}} | \mathbf{a}_t) + \lambda \log p(\mathbf{w}_t^{\text{gt}} | \mathbf{w}_t)]$$

训练时随机触发中间层的 early exit 以增强鲁棒性。

## 实验关键数据

### 基准数据集（四个真实世界数据集）

在 Recon、Go-Stanford、SACSoN、SCAND 上对比 ViNT 和 NoMaD：

| 指标 | DynaNav vs ViNT |
|------|----------------|
| FLOPs | 降低约 58%（平均 1.93 vs 4.37 × 10⁹） |
| 推理时间 | 降低 42.3%（0.228s vs 0.379s/traj） |
| 显存 | 降低 32.8%（13.35 vs 19.07 Gb） |
| 动作相似度 | 提升 0.83%（平均） |
| 航点相似度 | 提升 0.28%（平均） |

NoMaD 精度略高于 DynaNav（约 0.2%），但 FLOPs 是 DynaNav 的约 4 倍。

### CARLA 仿真

在 Town02（简单）、Town03（中等）、Town10（困难）三个场景测试：

- DynaNav 与 ViNT 成功率相当（0.727 vs 0.724 / 0.664 vs 0.659 / 0.588 vs 0.589）
- FLOPs 减少超过 2 倍
- 随环境复杂度提升，DynaNav 的 FLOPs 自动增加（1.58 → 1.70 → 1.93 × 10⁹），验证了动态推理的合理性

### 消融实验

- 单独使用特征选择器：性能提升，效率略有改善
- 单独使用动态解码器：效率提升明显，精度略降
- 两者结合：效率和精度同时最优
- 贝叶斯优化对阈值确定至关重要，无 BO 时 early exit 效果受损
- 特征选择器提高了 early exit 的跳层频率（2-4 层跳跃更频繁）

## 亮点

1. **首次将动态网络机制引入视觉导航**：是 dynamic inference 在导航领域的开创性工作
2. **效率与性能双赢**：2.26× FLOPs 降低的同时性能不降反升，打破了效率-精度的传统取舍
3. **特征选择增强可解释性**：可视化掩码清晰展示模型关注区域，发现模型并不简单聚焦于观测与目标间最大的共同物体
4. **特征选择与 early exit 协同**：稀疏特征使 early exit 更稳定，跳层频率更高
5. **自适应复杂度感知**：室内简单场景自动用更少计算，室外复杂场景自动分配更多资源

## 局限与展望

1. **额外优化开销**：贝叶斯优化需要训练后额外步骤，增加了人工成本和流程复杂度
2. **阈值泛化性**：CARLA 实验中三个场景用统一阈值，未探索针对不同环境的自适应阈值
3. **编码器未优化**：EfficientNet 编码器保持静态，未探索编码阶段的动态推理
4. **仅限 RGB 输入**：未考虑深度、激光雷达等多模态传感器输入的动态选择
5. **未来方向**：将贝叶斯优化与训练并行进行，实现真正端到端的动态推理系统

## 与相关工作的对比

| 方法 | 特点 | 局限 |
|------|------|------|
| ViNT | 大规模跨平台训练的导航基础模型 | 静态推理，所有层全部激活，计算开销大 |
| NoMaD | 扩散策略 + 目标掩码，精度高 | FLOPs 约为 DynaNav 的 4 倍，无法实时仿真 |
| GNM | 异构 RGB 数据集学习导航策略 | 泛化能力不足，成功率显著低于 DynaNav |
| DeeR-VLA | 多模态 LLM 的动态推理 | 仍需激活多层，计算节省有限 |
| **DynaNav** | 动态特征选择 + early exit + 贝叶斯优化 | 需额外优化步骤，阈值泛化有限 |

## 启发与关联

1. 动态推理思路可推广到其他具身 AI 任务（抓取、操作等），根据任务难度调整计算量
2. Gumbel-Softmax 硬特征选择方案可用于其他需要可解释稀疏注意力的视觉任务
3. 贝叶斯优化确定 early-exit 阈值的方法可迁移到 LLM 推理加速
4. 特征选择与 early exit 的协同设计思路值得在 VLM / 多模态模型中探索

## 评分
- 新颖性: 8/10（首次将动态网络引入视觉导航，特征选择与 early exit 的协同设计有创意）
- 实验充分度: 8/10（四个真实数据集 + CARLA 仿真 + 详细消融，但缺少更多仿真场景和真实机器人部署）
- 写作质量: 7/10（结构清晰，但部分公式较多，early exit 策略的描述可更直观）
- 价值: 8/10（对边缘部署的导航模型有直接实用价值，开创了导航模型效率优化的新方向）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction](../../CVPR2026/robotics/force_transferable_visual_jailbreaking_attacks_via_feature_over_reliance_correct.md)
- [\[NeurIPS 2025\] NeSyPr: Neurosymbolic Proceduralization For Efficient Embodied Reasoning](nesypr_neurosymbolic_proceduralization_for_efficient_embodied_reasoning.md)
- [\[NeurIPS 2025\] EfficientNav: Towards On-Device Object-Goal Navigation with Navigation Map Caching and Retrieval](efficientnav_towards_on-device_object-goal_navigation_with_navigation_map_cachin.md)
- [\[NeurIPS 2025\] BEAST: Efficient Tokenization of B-Splines Encoded Action Sequences for Imitation Learning](beast_efficient_tokenization_of_b-splines_encoded_action_sequences_for_imitation.md)
- [\[NeurIPS 2025\] Sample-Efficient Tabular Self-Play for Offline Robust Reinforcement Learning](sample-efficient_tabular_self-play_for_offline_robust_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
