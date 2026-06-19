---
title: >-
  [论文解读] Adaptive Preconditioners Trigger Loss Spikes in Adam
description: >-
  [ICML 2026][优化/理论][Adam优化器] 这篇论文把 Adam 训练中的 loss spike 归因于二阶矩预条件器与当前梯度平方的滞后解耦，并用预条件 Hessian 的梯度方向曲率解释和预测 spike 的发生。 领域现状：神经网络训练中经常出现 loss spike，尤其是在使用 Adam 训练 Tran…
tags:
  - "ICML 2026"
  - "优化/理论"
  - "Adam优化器"
  - "loss spike"
  - "预条件 Hessian"
  - "二阶矩估计"
  - "训练稳定性"
---

# Adaptive Preconditioners Trigger Loss Spikes in Adam

**会议**: ICML 2026  
**arXiv**: [2506.04805](https://arxiv.org/abs/2506.04805)  
**代码**: 无  
**领域**: 优化 / Adam训练稳定性  
**关键词**: Adam优化器, loss spike, 预条件 Hessian, 二阶矩估计, 训练稳定性  

## 一句话总结
这篇论文把 Adam 训练中的 loss spike 归因于二阶矩预条件器与当前梯度平方的滞后解耦，并用预条件 Hessian 的梯度方向曲率解释和预测 spike 的发生。

## 研究背景与动机
**领域现状**：神经网络训练中经常出现 loss spike，尤其是在使用 Adam 训练 Transformer 或较大模型时，loss 会突然上冲再恢复。已有解释主要从 loss landscape 的尖锐性入手，例如 lower-loss-as-sharper 和 Edge of Stability 现象，认为模型进入更尖锐区域后会触发不稳定。

**现有痛点**：单纯用 landscape 几何解释 Adam 的 spike 不够充分。论文展示了一个很直接的反例：在一维二次函数这种曲率恒定的场景下，普通 GD 在稳定学习率下平滑收敛，而 Adam 即使学习率远低于 GD 的稳定阈值，仍会出现明显 spike。这说明 spike 不一定来自“低损失区域变尖锐”，也可能来自优化器自身状态变量的动态。

**核心矛盾**：Adam 的自适应步长本来应该在梯度变大时增大二阶矩估计 $v_t$，从而降低有效步长；但当 $v_t$ 被历史项主导时，它可能继续衰减，无法及时追踪当前梯度平方 $g_t^2$。于是预条件后的有效曲率被不断放大，训练进入持续不稳定区间。

**本文目标**：作者希望回答三个问题：Adam 的稳定性应该由什么量控制；二阶矩估计为什么会在 spike 前失效；以及能否构造比最大 Hessian 特征值更精确的 spike 预警指标。

**切入角度**：论文从局部二次近似出发，把 Adam 的更新看成对 Hessian 做了空间预条件和动量预条件。这个角度把“优化器内部状态”显式纳入稳定性分析，因此可以解释一维二次函数和真实 Transformer 中共同出现的 spike 机制。

**核心 idea**：用 Adam 预条件 Hessian 的梯度方向曲率，而不是原始 Hessian 最大特征值，刻画 loss spike 的真正触发条件。

## 方法详解
这篇论文不是提出一个新优化器，而是给 Adam 的 loss spike 建立机制解释、预测指标和抑制建议。整体逻辑是：先用局部二次模型推导 Adam 的稳定条件，再分析二阶矩 $v_t$ 与梯度平方 $g_t^2$ 的解耦如何让稳定条件持续失效，最后用多个尺度的实验验证这个机制。

### 整体框架
输入是一条使用 Adam 训练得到的优化轨迹，作者沿着这条轨迹观察梯度、二阶矩、Hessian 以及预条件 Hessian 的变化。分析先在 GD 上回顾局部稳定阈值 $\lambda_{\max}(H_t)<2/\eta$ 作为参照，再顺着三个核心机制层层推进：其一，把 Adam 的动量项和自适应分母都折进 Hessian，得到预条件 Hessian $\hat H_t$，稳定判据随之换成 $\lambda_{\max}(\hat H_t)<2/\eta$；其二，二阶矩 $v_t$ 与当前梯度平方 $g_t^2$ 发生解耦——梯度在涨、分母却自顾自衰减，使 $\hat H_t$ 的特征值被持续推高，这正是 spike 区别于普通 Edge of Stability 振荡的分水岭；其三，用梯度方向曲率 $\lambda_{\mathrm{grad}}(\hat H_t)$ 取代最大特征值作为更精确的 spike 预警判据。最后在一维二次函数、FNN、CNN 和 Transformer 上验证这套机制，并给出降低 $\beta_2$、增大 $\epsilon$ 两种有理论依据的抑制手段。

在 Adam 中，更新含有一阶矩 $m_t$ 和二阶矩 $v_t$。如果暂时忽略动量，Adam 近似等价于在局部 Hessian $H_t$ 前乘上对角矩阵 $\mathrm{diag}(1/(\sqrt{\hat v_t}+\epsilon))$。论文进一步把动量项也纳入，得到综合的 Adam 预条件 Hessian $\hat H_t = \frac{1}{1-\beta_1^t}\frac{1-\beta_1}{1+\beta_1}\mathrm{diag}(1/(\sqrt{\hat v_t}+\epsilon))H_t$。当这个矩阵的有效曲率长期超过 $2/\eta$ 时，训练就有进入 spike 的风险。

### 关键设计

**1. Adam 预条件 Hessian：把自适应分母和动量都折进局部曲率**

GD 的局部稳定性由原始 Hessian 的最大特征值决定，只要 $\lambda_{\max}(H_t)<2/\eta$ 就不会发散。但 Adam 在 $H_t$ 前面乘了一层对角缩放——二阶矩分母 $\mathrm{diag}(1/(\sqrt{\hat v_t}+\epsilon))$ 再加上动量修正，合起来构成预条件 Hessian $\hat H_t$，于是判据换成 $\lambda_{\max}(\hat H_t)<2/\eta$。这一步把"优化器内部状态"显式塞进了稳定性条件：即便原始 Hessian 一动不动，只要 $\sqrt{\hat v_t}$ 变小，预条件曲率就被放大、跨过稳定边界。它正好解释了那个反例——一维二次函数曲率恒定，GD 平滑收敛，Adam 在远低于 GD 阈值的学习率下却照样 spike，根因不是几何突然变尖，而是 Adam 自己改了坐标尺度。

**2. 二阶矩与梯度平方的解耦：spike 为什么会持续而不是一闪而过**

正常情况下梯度变大应该把 $v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2$ 顶上去，从而压低有效步长、形成负反馈。问题出在当前梯度项相对历史项太小的时候：$v_t$ 近似按 $\beta_2 v_{t-1}$ 自顾自地衰减，于是梯度已经在涨、分母却还在缩，$\hat H_t$ 的特征值被双重推高。这个滞后正是 spike 与普通 Edge of Stability 振荡的分水岭——若 $v_t$ 能及时响应，系统只在阈值附近抖动；一旦 $v_t$ 跟不上，稳定性违背会持续累积，loss 就鼓成一个尖峰。这也直接给出了缓解手段：调小 $\beta_2$ 让二阶矩更快追梯度，从源头削弱解耦。

**3. 梯度方向曲率：用更新方向上的曲率代替最大特征值来预警**

下一步 loss 升不升，取决于实际更新方向上的二阶项，而不是所有方向里最陡的那一个。高维模型里最大曲率方向常常和梯度方向并不对齐，单看 $\lambda_{\max}$ 会频繁误报。作者因此定义梯度方向曲率 $\lambda_{\mathrm{grad}}(H_t)=\nabla L(\theta_t)^T H_t \nabla L(\theta_t)/\|\nabla L(\theta_t)\|^2$，并在 Adam 里替换成预条件版 $\lambda_{\mathrm{grad}}(\hat H_t)$，只有当这个量也越过 $2/\eta$ 时 spike 才真正出现。因为它直接对应本次更新带来的 loss 变化，所以比"全方向最大曲率"更贴近 spike 的 onset，误报大幅减少。

### 损失函数 / 训练策略
论文的训练目标沿用各实验任务本身的损失，没有引入新损失。主要实验策略是沿训练轨迹计算 Hessian-vector product，从而估计 $\lambda_{\max}$、$\lambda_{\mathrm{grad}}$ 和预条件版本。抑制策略方面，作者验证了两类直观干预：增大 Adam 的 $\epsilon$ 可以抬高分母下界，降低有效曲率；降低 $\beta_2$ 可以让二阶矩更快响应当前梯度，从根源上缓解解耦。

## 实验关键数据

### 主实验
论文以图和轨迹分析为主，没有常规“数据集-指标-SOTA”表。下面按实验场景整理最关键的主结果。

| 实验场景 | 观测指标 | 本文关键结果 | 对照/基线现象 | 结论 |
|--------|------|------|----------|------|
| 一维二次函数 | loss 与有效学习率 | Adam 在小学习率下仍出现 spike，且 $\eta/\sqrt{\hat v_t}$ 到固定阈值附近触发 | GD 在同一稳定区间平滑收敛 | spike 可由 Adam 内部状态触发 |
| 二层 FNN 拟合 $\sin x+\sin 4x$ | $\lambda_{\max}$ 与 $\lambda_{\mathrm{grad}}$ | Adam 出现 77 个 spike；spike 只在 $\lambda_{\mathrm{grad}}(\hat H_t)>2/\eta$ 时发生 | $\lambda_{\max}(\hat H_t)$ 有 1010 个越界时刻，误报更多 | 梯度方向曲率更精确 |
| 50 维函数近似 FNN | spike 时刻 | $\lambda_{\max}(\hat H_t)$ 在 epoch 179 越界，但 loss 到 epoch 184 才 spike | 原始 $\lambda_{\max}(H_t)$ 很快稳定 | 需要看梯度对齐后的曲率 |
| 88 层 Transformer | sustained predictor | 7 次 loss spike 均与 sustained $\lambda_{\mathrm{grad}}(\hat H_t)$ 越界对应 | 原始单步指标受 mini-batch 噪声干扰 | 随机训练中需用持续越界判据 |
| 187M LLaMA 结构 Transformer | spike 频率与 $\beta_2$ | 默认 $\beta_2=0.999$ 下多次 spike；降低 $\beta_2$ 后 spike 减少 | 大模型中仍能观察到梯度方向曲率越界 | 机制可扩展到真实语言模型训练 |

### 消融实验
这里的“消融”对应论文对预测指标和超参数干预的分析。

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅看 $\lambda_{\max}(H_t)$ | 高维场景中会提前越界 | 最大曲率方向未必参与当前更新，不能直接说明 loss 会升高 |
| 看 $\lambda_{\max}(\hat H_t)$ | 能反映 Adam 预条件带来的风险，但仍有大量越界时刻 | 捕捉到二阶矩衰减放大有效曲率，但仍缺少方向信息 |
| 看 $\lambda_{\mathrm{grad}}(\hat H_t)$ | FNN/Adam 中只在该量越过 $2/\eta$ 时出现 spike | 与单步 loss 增长条件直接对应，误报更少 |
| 增大 $\epsilon$ 到 0.1 | FNN 实验中可消除 spike | 抬高分母下界，阻止预条件曲率继续放大 |
| 降低 $\beta_2$ 到 0.9 | Transformer 和 LLaMA 实验中 spike 频率下降 | 二阶矩更快追踪当前梯度，削弱 $v_t$ 与 $g_t^2$ 的解耦 |

### 关键发现
- 这篇论文最重要的实验证据是“原始 Hessian 不够解释，预条件 Hessian 才解释 Adam”。二次函数、FNN、CNN 和 Transformer 都呈现出 $v_t$ 衰减导致有效曲率上升的同一模式。
- 最大特征值是风险信号，但不是触发信号。只有当梯度方向也进入高曲率不稳定区，loss 才会真正上升。
- 降低 $\beta_2$ 的解释很清楚：不是魔法调参，而是让二阶矩估计更快跟上梯度变化，避免分母在梯度上升时继续下降。

## 亮点与洞察
- 论文把“loss spike 是 optimizer state 的动态失配”讲得很清楚。它没有停留在经验观察，而是把 Adam 的二阶矩写进稳定性阈值，让 spike 可以在二次函数上被解释。
- 梯度方向曲率是一个很有用的诊断视角。很多训练监控只看 loss、梯度范数或最大 Hessian 特征值，但这篇论文提醒我们，真正决定下一步 loss 是否上升的是更新方向上的曲率。
- 对大模型训练的实际启发是，较低 $\beta_2$ 可能不仅影响收敛速度，也是在降低 loss spike 风险。这为一些 LLM 训练实践中使用 $\beta_2=0.95$ 或更低提供了机制解释。

## 局限与展望
- 理论最严格的部分建立在一维二次函数和局部二次近似上，高维非凸网络中的结论主要依赖实验验证。预条件器、真实 loss landscape 和随机 mini-batch 噪声之间可能还有更复杂的耦合。
- Hessian-vector product 级别的指标计算在 200M 参数以上模型中仍然昂贵，难以直接作为常规训练监控工具。未来需要更便宜的近似指标。
- spike 并不总是坏事，附录中还讨论了 neutral、benign、malignant、catastrophic 等类型。如何区分“该抑制的 spike”和“可能有利于 basin transition 的 spike”仍是开放问题。

## 相关工作与启发
- **vs Edge of Stability**: EoS 解释 GD 中最大 Hessian 特征值靠近 $2/\eta$ 后的非单调下降；本文把这个框架推广到 Adam 的预条件 Hessian，并强调持续越界才会形成 spike。
- **vs lower-loss-as-sharper**: LLAS 从 loss landscape 形状解释 spike；本文指出即便 landscape 曲率不变，Adam 的 $v_t$ 也能改变有效曲率，因此 optimizer state 本身就是独立机制。
- **vs Adam 收敛性分析**: 传统 Adam 理论关注收敛或不收敛；本文更像训练动力学诊断，解释 spike 的发生、持续和恢复阶段。
- **对训练实践的启发**: 监控二阶矩衰减、梯度方向曲率或其低成本 proxy，可能比只监控 loss 更早发现训练不稳定；调低 $\beta_2$ 或增大 $\epsilon$ 也可作为有理论解释的稳定化手段。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从 Adam 内部预条件器动态解释 loss spike，视角非常清晰。
- 实验充分度: ⭐⭐⭐⭐☆ 覆盖从二次函数到 187M Transformer，但数值指标多以图示为主，缺少更大规模模型的系统表格。
- 写作质量: ⭐⭐⭐⭐☆ 机制链条完整，不过公式和图示较密，需要一定优化背景才能快速读懂。
- 价值: ⭐⭐⭐⭐⭐ 对大模型训练稳定性和 Adam 超参数选择都有直接启发。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks](the_implicit_bias_of_adam_and_muon_on_smooth_homogeneous_neural_networks.md)
- [\[ICML 2026\] Towards Understanding Adam Convergence on Highly Degenerate Polynomials](towards_understanding_adam_convergence_on_highly_degenerate_polynomials.md)
- [\[NeurIPS 2025\] Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking Loss with Representation Consistency](../../NeurIPS2025/optimization/abstain_mask_retain_core_time_series_prediction_by_adaptive.md)
- [\[ICML 2026\] Multi-Objective Bayesian Optimization via Adaptive ε-Constraints Decomposition](multi-objective_bayesian_optimization_via_adaptive_varepsilon-constraints_decomp.md)
- [\[CVPR 2026\] Globscope: Toward a Global View of the Loss Landscape](../../CVPR2026/optimization/globscope_toward_a_global_view_of_the_loss_landscape.md)

</div>

<!-- RELATED:END -->
