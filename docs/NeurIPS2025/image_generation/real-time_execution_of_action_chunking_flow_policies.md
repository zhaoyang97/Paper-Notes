---
title: >-
  [论文解读] Real-Time Execution of Action Chunking Flow Policies
description: >-
  [NeurIPS 2025][图像生成][实时推理] 提出 Real-Time Chunking (RTC)，将异步动作分块执行建模为修复（inpainting）问题，通过冻结已执行动作并"修复"其余部分，实现扩散/流策略的实时平滑执行，无需重新训练。 现代 VLA（视觉-语言-动作模型）在机器人控制中日益强大…
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "实时推理"
  - "动作分块"
  - "流匹配"
  - "修复引导"
  - "VLA"
---

# Real-Time Execution of Action Chunking Flow Policies

**会议**: NeurIPS 2025  
**arXiv**: [2506.07339](https://arxiv.org/abs/2506.07339)  
**代码**: [项目页面](https://pi.website/research/real_time_chunking)  
**领域**: 扩散/流模型 / 机器人控制  
**关键词**: 实时推理, 动作分块, 流匹配, 修复引导, VLA

## 一句话总结

提出 Real-Time Chunking (RTC)，将异步动作分块执行建模为修复（inpainting）问题，通过冻结已执行动作并"修复"其余部分，实现扩散/流策略的实时平滑执行，无需重新训练。

## 研究背景与动机

现代 VLA（视觉-语言-动作模型）在机器人控制中日益强大，但面临一个根本矛盾：**模型越大越好，但延迟也越高**。一个 3B 参数的 $\pi_0$ VLA 仅 KV cache 预填充就需 46ms，而控制频率要求 20ms 一帧。远程推理还额外增加网络延迟。

**动作分块（Action Chunking）**——模型一次输出多步动作——部分缓解了延迟问题，但引入了新的核心难题：

**块边界不连续**: 相邻动作块可能跳到分布的不同模式，导致突变的、超出训练分布的运动

**同步推理的暂停**: 默认方法在执行完当前块后停下等待新块生成，引入可见的暂停，改变机器人动力学

**朴素异步策略失败**: 简单切换到新块会导致极高加速度；时间集成（Temporal Ensembling）通过平均多个预测，但在多模态分布中平均值可能不是有效动作

作者通过图示清晰展示了问题：假设当前块计划从上方绕过障碍物，新块计划从下方，延迟 7 步后跳转会产生超出分布的剧烈加速度。

## 方法详解

### 整体框架

RTC 将实时执行建模为**修复问题**：在生成新动作块时，"冻结"因推理延迟必定已执行的动作前缀，然后"修复"其余部分使其与前缀一致。算法在后台线程持续运行推理循环，主线程每 $\Delta t$ 消费一个动作。

### 关键设计

1. **基于 $\Pi$GDM 的流匹配修复**: 在每个去噪步骤中添加基于梯度的引导项，鼓励生成结果匹配已知的目标值（冻结的动作）。修正后的速度场为：
    $\mathbf{v}_{\Pi\text{GDM}}(\mathbf{A}_t^\tau, \mathbf{o}_t, \tau) = \mathbf{v}(\mathbf{A}_t^\tau, \mathbf{o}_t, \tau) + \min\left(\beta, \frac{1-\tau}{\tau \cdot r_\tau^2}\right)(\mathbf{Y} - \widehat{\mathbf{A}_t^1})^\top \text{diag}(\mathbf{W}) \frac{\partial \widehat{\mathbf{A}_t^1}}{\partial \mathbf{A}_t^\tau}$
   其中 $\widehat{\mathbf{A}_t^1} = \mathbf{A}_t^\tau + (1-\tau)\mathbf{v}(\mathbf{A}_t^\tau, \mathbf{o}_t, \tau)$ 是最终去噪结果的估计。引导权重裁剪 $\beta$ 是作者的改进，防止少步去噪时的不稳定性。

2. **软掩码（Soft Masking）**: 这是保证跨块连续性的关键创新。仅使用前 $d$ 个动作做硬掩码修复信号太弱，容易策略切换。软掩码利用所有 $H-s$ 个重叠动作，权重从 1 指数衰减到 0：
    $\mathbf{W}_i = \begin{cases} 1 & \text{if } i < d \\ c_i \frac{e^{c_i}-1}{e-1} & \text{if } d \leq i < H-s \\ 0 & \text{if } i \geq H-s \end{cases}$
   其中 $c_i = \frac{H-s-i}{H-s-d+1}$。直觉上，越远的未来动作应赋予越少的注意力权重。

3. **异步执行系统**: 使用互斥锁和条件变量实现线程安全：

    - `GetAction`: 控制器每 $\Delta t$ 调用，返回当前块中的下一个动作
    - `InferenceLoop`: 后台线程持续推理，用过去延迟的滑动窗口保守估计下一次延迟
    - 新块一就绪就原子切换，执行周期 $s = \max(d, s_{\min})$

### 损失函数 / 训练策略

RTC 是纯推理时算法，**不需要任何训练或重新训练**。它适用于任何使用扩散或流匹配的动作分块策略。引导项通过反向传播计算向量-雅可比积，是唯一的额外计算开销。

## 实验关键数据

### 仿真实验：Kinetix 12 个动态任务

| 方法 | d=0 求解率 | d=2 求解率 | d=4 求解率 | 对延迟的鲁棒性 |
|------|-----------|-----------|-----------|---------------|
| Naive Async | ~48% | ~42% | ~33% | 差 |
| TE (时间集成) | ~35% | ~33% | ~30% | 最差 |
| BID (双向解码) | ~51% | ~46% | ~38% | 中等 |
| RTC (硬掩码) | ~52% | ~48% | ~42% | 较好 |
| **RTC (软掩码)** | **~54%** | **~50%** | **~43%** | **最佳** |

### 真实世界实验：6 个双臂操作任务 ($\pi_{0.5}$ VLA)

| 方法 | 平均吞吐量（无延迟）↑ | 平均吞吐量（+100ms）↑ | 平均吞吐量（+200ms）↑ |
|------|----------------------|----------------------|----------------------|
| 同步推理 | ~0.35 | ~0.28 | ~0.22 |
| TE (稀疏) | ~0.36 | 不可用（触发保护停止） | 不可用 |
| TE (密集) | ~0.33 | 不可用（触发保护停止） | 不可用 |
| **RTC** | **~0.40** | **~0.40** | **~0.40** |

### 关键发现

- **RTC 对延迟完全鲁棒**: 注入 +200ms 延迟后性能无退化，而同步推理线性退化，TE 方法因抖动过大触发机器人保护停止
- **速度与质量双提升**: RTC 不仅执行更快（移除推理暂停后仍比同步快 20%），还因减少失误和重试而更早完成任务
- **点火柴任务**: 最需要精确度的任务中，RTC 成功率大幅领先，因为该任务无重试机会
- 软掩码在低延迟时比硬掩码更有效，高延迟时差异缩小

## 亮点与洞察

- **问题建模的优雅转化**: 将实时控制中的异步块拼接问题转化为已有成熟方法的修复问题，理论根基扎实
- **纯推理时方案**: 无需改变训练流程，适用于所有扩散/流策略，包括已部署的大模型如 $\pi_{0.5}$
- **软掩码的类比**: 指数衰减权重模拟了对未来不确定性的递增折扣，与控制理论中的预测时域折扣类似
- **真实世界验证的彻底性**: 480 个 episode、28 小时纯执行时间、6 种任务包括移动操作，且注入了不同延迟

## 局限与展望

- RTC 引导项需要反向传播计算 VJP，增加了约 28% 的推理延迟（97ms vs 76ms）
- 仅适用于扩散和流匹配策略，不适用于自回归或 VQ 策略
- 真实世界实验未涵盖腿式运动等更动态的场景（仅在仿真中测试）
- 软掩码的衰减函数选择（指数 vs 线性 vs 余弦）的消融仅在附录中提及

## 相关工作与启发

- **Diffuser** 首次将扩散修复用于强化学习约束，但非基于引导，且未考虑实时控制
- **BID** 通过拒绝采样保持块连续性，但需要 32 个批样本并行，计算量远大于 RTC
- **一致性策略和流式扩散策略** 通过蒸馏减少去噪步数，但单次前向传播的延迟仍无法消除
- 分层 VLA（System 1/2 设计）与 RTC 正交，可能组合使用

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 将修复引导引入实时机器人控制，问题建模精准
- **实验充分度**: ⭐⭐⭐⭐⭐ 仿真+真实世界、多种延迟条件、480 episodes
- **写作质量**: ⭐⭐⭐⭐⭐ 问题阐述极清晰，图示直观，算法伪代码完备
- **价值**: ⭐⭐⭐⭐⭐ 直接解决大模型机器人部署的核心瓶颈，即插即用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition](../../ICLR2026/image_generation/compose_your_policies_improving_diffusion-based_or_flow-based_robot_policies_via.md)
- [\[NeurIPS 2025\] Failure Prediction at Runtime for Generative Robot Policies](failure_prediction_at_runtime_for_generative_robot_policies.md)
- [\[NeurIPS 2025\] Value Gradient Guidance for Flow Matching Alignment](value_gradient_guidance_for_flow_matching_alignment.md)
- [\[CVPR 2026\] StreamDiT: Real-Time Streaming Text-to-Video Generation](../../CVPR2026/image_generation/streamdit_real-time_streaming_text-to-video_generation.md)
- [\[CVPR 2025\] SemanticDraw: Towards Real-Time Interactive Content Creation from Image Diffusion](../../CVPR2025/image_generation/semanticdraw_towards_real-time_interactive_content_creation_from_image_diffusion.md)

</div>

<!-- RELATED:END -->
