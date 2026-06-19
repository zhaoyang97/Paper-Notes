---
title: >-
  [论文解读] MP1: MeanFlow Tames Policy Learning in 1-step for Robotic Manipulation
description: >-
  [AAAI 2026][图像生成][机器人操作] 首次将 MeanFlow 范式引入机器人学习领域，结合 3D 点云输入和 Dispersive Loss，实现仅需一次网络前向传播（1-NFE）即可生成动作轨迹，在机器人操作任务中以 6.8ms 推理延迟达到 SOTA 成功率。 领域现状 机器人操作（Robot Manipu…
tags:
  - "AAAI 2026"
  - "图像生成"
  - "机器人操作"
  - "MeanFlow"
  - "单步推理"
  - "流匹配"
  - "分散损失"
---

# MP1: MeanFlow Tames Policy Learning in 1-step for Robotic Manipulation

**会议**: AAAI 2026  
**arXiv**: [2507.10543](https://arxiv.org/abs/2507.10543)  
**代码**: [github.com/LogSSim/MP1](https://github.com/LogSSim/MP1)  
**领域**: 图像生成 / 机器人操作  
**关键词**: 机器人操作, MeanFlow, 单步推理, 流匹配, 分散损失

## 一句话总结

首次将 MeanFlow 范式引入机器人学习领域，结合 3D 点云输入和 Dispersive Loss，实现仅需一次网络前向传播（1-NFE）即可生成动作轨迹，在机器人操作任务中以 6.8ms 推理延迟达到 SOTA 成功率。

## 研究背景与动机

### 领域现状

机器人操作（Robot Manipulation）领域中，生成模型已成为策略学习的主流方法。扩散模型（如 Diffusion Policy, DP3）能有效处理多模态动作分布，但需要多步（~10步）去噪迭代，推理延迟约 130ms。流匹配方法（如 FlowPolicy）通过一致性约束实现单步采样，缩短推理时间至约 12ms，但仍依赖额外的约束假设。

### 现有痛点

**扩散模型推理慢**：多步去噪过程导致推理延迟高达 ~130ms，难以满足实时控制需求（如力控操作、高频反馈回路）。

**流匹配方法受限于一致性约束**：FlowPolicy 等方法通过强制一致性损失实现 1-NFE 推理，但这引入了额外的结构假设，可能限制策略的表达能力。

**ODE 求解器误差**：传统 Flow Matching 在推理时需要积分瞬时速度场，数值 ODE 求解器引入的误差会降低轨迹精度。

### 核心矛盾

如何在**不引入任何一致性约束或 ODE 求解器**的前提下实现真正的单步动作生成，同时保持甚至超越多步方法的任务成功率？

### 核心 Idea

MeanFlow 范式通过直接学习区间平均速度场（而非瞬时速度场），利用 "MeanFlow Identity" 这一微分恒等式将训练转化为简单的回归目标。推理时只需一次前向传播即可从噪声直接得到动作轨迹，完全消除 ODE 求解器误差和一致性约束。同时引入 Dispersive Loss 作为训练时正则化器，增强特征空间判别性，提升少样本泛化能力。

## 方法详解

### 整体框架

MP1 以 3D 点云和机器人状态作为输入，通过视觉编码器和状态编码器提取条件特征 $\mathbf{c} = (\mathbf{f}_v, \mathbf{f}_s)$，输入集成了 MeanFlow 的 UNet 网络生成动作轨迹。训练时结合 CFG 回归损失 $\mathcal{L}_{cfg}$ 和 Dispersive Loss $\mathcal{L}_{disp}$，推理时仅需一次前向传播。

### 关键设计

#### 1. **MeanFlow 策略生成（核心创新）**

**功能**：学习区间平均速度场替代瞬时速度场，实现无 ODE 求解器的单步动作生成。

**核心思路**：标准 Flow Matching 学习瞬时速度场 $v(z_t, t)$，推理需要积分 ODE。MeanFlow 则学习区间 $[r, t]$ 上的**平均速度场**：

$$u(z_t, r, t) \triangleq \frac{1}{t-r} \int_r^t v(z_\tau, \tau) d\tau$$

直接从积分定义学习是不可行的，但通过对 $t$ 求导可得 **"MeanFlow Identity"**：

$$u(z_t, r, t) = v(z_t, t) - (t-r) \frac{d}{dt} u(z_t, r, t)$$

基于该恒等式，训练目标变为简单的回归：

$$\mathcal{L}(\theta) = \mathbb{E}_{t,r,x,\epsilon} \|u_\theta(z_t, r, t) - sg(u_{tgt})\|_2^2$$

其中 $u_{tgt} = v_t - (t-r)(v_t \partial_z u_\theta + \partial_t u_\theta)$，$sg(\cdot)$ 表示 stop-gradient。

**推理公式**：
$$\mathbf{A}_0 = \mathbf{A}_1 - u_\theta^{cfg}(\mathbf{A}_1, 0, 1 | \mathbf{c})$$

从噪声 $\mathbf{A}_1 \sim \mathcal{N}(0, I)$ 一步直接得到动作轨迹 $\mathbf{A}_0$。

**设计动机**：MeanFlow 消除了三个限制：(1) 多步去噪（扩散模型）、(2) ODE 求解器误差（FlowMatching）、(3) 一致性约束（FlowPolicy）。

#### 2. **Classifier-Free Guidance (CFG) 集成**

**功能**：在 MeanFlow 框架中引入 CFG 增强条件控制能力。

**核心思路**：训练时按混合比例 $\omega$ 融合条件和无条件预测构造引导速度：

$$\tilde{v}_t \triangleq \omega v_t(\mathbf{A}_t | \mathbf{A}_0, \mathbf{c}) + (1-\omega) u_\theta^{cfg}(\mathbf{A}_t, t, t | \emptyset)$$

用引导速度替代普通瞬时速度来构造 MeanFlow 训练目标。

**设计动机**：CFG 在图像生成中已证明能显著提升条件控制质量，但传统 CFG 需要多步推理。MeanFlow 的框架允许在不牺牲 1-NFE 的前提下集成 CFG，因为 MeanFlow 学的是平均速度而非瞬时速度。

#### 3. **Dispersive Loss（分散损失）**

**功能**：训练时正则化策略网络的中间特征空间，防止不同状态的特征坍缩，提升泛化能力。

**核心思路**：类似于"无正样本对的对比学习"，对批次内不同样本的中间表示施加排斥力：

$$\mathcal{L}_{Disp}(\theta) = \log \mathbb{E}_{i,j \in \mathcal{B}} \left[\exp\left(-\frac{\|\mathbf{z}_{\mathbf{A},i} - \mathbf{z}_{\mathbf{A},j}\|_2^2}{\tau}\right)\right]$$

其中 $\mathbf{z}_{\mathbf{A},i}$ 是 UNet 下采样块的输出特征。$\tau=1$ 为温度超参。

**设计动机**：纯回归目标只为每个状态匹配正确轨迹，不显式约束特征空间结构。这导致"特征坍缩"——不同场景状态映射到相似的潜在点，使策略无法区分需要不同动作的微妙状态差异。Dispersive Loss 强制特征分散，隐式增强了策略对场景细微变化的敏感性，特别在少样本学习中效果显著。

**关键优势**：仅在训练时计算，推理时零额外开销，不影响 1-NFE 速度。

### 损失函数 / 训练策略

总损失：
$$\mathcal{L}_{total}(\theta) = \mathcal{L}_{cfg}(\theta) + \lambda \mathcal{L}_{Disp}(\theta)$$

$\lambda = 0.5$ 平衡两项贡献。训练参数：10 个专家演示、batch size 128、AdamW 优化器、学习率 0.0001、Adroit 训练 3000 epochs、Meta-World 训练 1000 epochs。

## 实验关键数据

### 主实验

37个任务（3 Adroit + 34 Meta-World）的成功率和推理速度：

| 方法 | NFE | Adroit Hammer | Adroit Door | Adroit Pen | MW Easy(21) | MW Medium(4) | MW Hard(4) | MW V.Hard(5) | 平均 |
|------|-----|-------|------|-----|------|--------|------|---------|------|
| DP3 | 10 | 100±0 | 56±5 | 46±10 | 87.3±2.2 | 44.5±8.7 | 32.7±7.7 | 39.4±9.0 | 68.7±4.7 |
| FlowPolicy | 1 | 98±1 | 61±2 | 54±4 | 84.8±2.2 | 58.2±7.9 | 40.2±4.5 | 52.2±5.0 | 71.6±3.5 |
| **MP1** | **1** | **100±0** | **69±2** | **58±5** | **88.2±1.1** | **68.0±3.1** | **58.1±5.0** | **67.2±2.7** | **78.9±2.1** |

MP1 比 DP3 高 **10.2%**，比 FlowPolicy 高 **7.3%**，同时标准差更低（2.1% vs 3.5%）。

推理速度对比：

| 方法 | NFE | 平均推理/ms |
|------|-----|-------------|
| DP3 | 10 | 132.2±11.2 |
| Simple DP3 | 10 | 97.0±9.2 |
| FlowPolicy | 1 | 12.6±1.5 |
| **MP1** | **1** | **6.8±0.1** |

MP1 推理仅 **6.8ms**，比 FlowPolicy 快 **~2×**，比 DP3 快 **~19×**。

真实世界实验（5个任务，每个20次评估）：

| 任务 | MP1 | FlowPolicy | DP3 |
|------|-----|-----------|-----|
| Hammer | **90% / 18.6s** | 70% / 22.3s | 70% / 31.1s |
| Drawer Close | **100% / 8.8s** | 90% / 15.7s | 80% / 20.2s |
| Heat Water | **90% / 23.4s** | 60% / 31.1s | 70% / 38.8s |
| Stack Block | **80% / 27.2s** | 50% / 29.6s | 60% / 35.1s |
| Spoon | **90% / 22.6s** | 80% / 26.7s | 70% / 28.3s |

### 消融实验

Dispersive Loss 消融（选取 10 个任务）：

| 配置 | Adroit Door | Adroit Pen | MW Coffee Pull | MW Disassemble | MW Pick Place Wall | 说明 |
|------|------------|-----------|---------------|----------------|-------------------|------|
| MP1（完整） | **58±5** | **69±2** | **92.3±3.7** | **74.0±1.4** | **64.3±1.2** | 标准配置 |
| MP1 - Disp Loss | 55±6 | 68±4 | 90.7±2.1 | 72.7±0.5 | 60.3±2.4 | 移除分散损失 |

MeanFlow Ratio 消融：

| Flow Ratio | Adroit Pen | MW Dial Turn | MW Coffee Pull | MW Assembly | 平均 |
|-----------|-----------|-------------|---------------|------------|------|
| 0（= FM） | 53±5 | 81±1 | 62±4 | 97±2 | 72.6±3.0 |
| **0.50** | **58±5** | **90±2** | **92±4** | **98±1** | **82.4±2.6** |
| 1.0 | 0±0 | 0±0 | 12±5 | 0±0 | 2.4±1.0 |

当 $r=t$（ratio=0）退化为标准 Flow Matching 时，性能显著下降；ratio=1.0 时完全失败。

### 关键发现

1. **MeanFlow 优于 Flow Matching**：ratio=0（标准 FM）→ ratio=0.5（MeanFlow）成功率从 72.6% 提升至 82.4%，验证了平均速度场的优势。
2. **Dispersive Loss 对困难任务帮助更大**：在 Meta-World Push 任务上，Dispersive Loss 带来 23.3 个百分点的提升（50.7→74.0）。
3. **少样本学习能力强**：仅用 10 个演示即可获得高成功率，增加到 20 个时提升收敛但最终性能差异不大。
4. **真实世界迁移成功**：在 5 个真实机器人任务上全面领先，且完成时间更短。

## 亮点与洞察

1. **MeanFlow 在机器人领域的首次应用**：从图像生成领域成功迁移，展现了该范式的通用性。
2. **真正的 1-NFE**：与 FlowPolicy 不同，MP1 不需要一致性约束，不需要 ODE 求解器，理论上更干净。
3. **Dispersive Loss 的巧妙设计**：作为训练时正则化器，不增加推理开销，却显著提升少样本泛化。这实际上是一种特征空间的均匀化约束，可推广到其他策略学习场景。
4. **6.8ms 推理延迟**：已达到实时控制级别（~150Hz），对高频控制任务具有实际意义。

## 局限与展望

1. **仅验证 Adroit 和 Meta-World**：虽然包含 37 个任务，但任务类型相对有限（主要是桌面操作），未验证在 locomotion、navigation 等领域。
2. **真实世界实验规模有限**：5 个任务、每个 10 次评估，统计意义有限。
3. **10 个演示的设定**：仍然需要人工演示，非零样本学习。
4. **UNet 骨干网络**：使用传统 UNet 而非 Transformer 架构（如 DiT），可能限制了在更复杂任务上的表达能力。

## 相关工作与启发

- **MeanFlow（图像生成）**：MP1 的核心理论基础，证明了平均速度场可以实现真正的单步生成。
- **DP3**：提供了 3D 点云 + 扩散模型的机器人策略框架基础。
- **FlowPolicy**：最直接的对比对象，展示了 MeanFlow 相对一致性约束方法的优势。
- **Dispersive Loss（Disperse 论文）**：提供了特征空间正则化的思路，在少样本学习中效果显著。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — MeanFlow 在机器人领域的首次应用，Dispersive Loss 的引入
- **实验充分度**: ⭐⭐⭐⭐⭐ — 37个仿真任务 + 5个真实任务，多维度消融分析
- **写作质量**: ⭐⭐⭐⭐ — 方法推导清晰，实验对比全面
- **价值**: ⭐⭐⭐⭐⭐ — 6.8ms 推理 + SOTA 成功率，对实时机器人控制有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] OMP: One-step Meanflow Policy with Directional Alignment](../../ICML2026/image_generation/omp_one-step_meanflow_policy_with_directional_alignment.md)
- [\[AAAI 2026\] EfficientFlow: Efficient Equivariant Flow Policy Learning for Embodied AI](efficientflow_efficient_equivariant_flow_policy_learning_for_embodied_ai.md)
- [\[NeurIPS 2025\] Two-Steps Diffusion Policy for Robotic Manipulation via Genetic Denoising](../../NeurIPS2025/image_generation/two-steps_diffusion_policy_for_robotic_manipulation_via_genetic_denoising.md)
- [\[CVPR 2026\] Temporal Equilibrium MeanFlow: Bridging the Scale Gap for One-Step Generation](../../CVPR2026/image_generation/temporal_equilibrium_meanflow_bridging_the_scale_gap_for_one-step_generation.md)
- [\[ICCV 2025\] A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation](../../ICCV2025/image_generation/a0_affordance_aware_hierarchical_model_robotic_manipulation.md)

</div>

<!-- RELATED:END -->
