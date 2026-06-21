---
title: >-
  [论文解读] Entropy Regularizing Activation: Boosting Continuous Control, Large Language Models, and Image Classification with Activation as Entropy Constraints
description: >-
  [ICLR 2026][强化学习][熵正则化] ERA（Entropy Regularizing Activation）通过在网络输出层附加专门设计的激活函数来施加熵下界约束，无需修改损失函数，一套框架同时提升连续控制 RL、LLM 推理和图像分类的性能。 领域现状：最大熵强化学习（如 SAC）将熵奖励直接加入优化目标…
tags:
  - "ICLR 2026"
  - "强化学习"
  - "熵正则化"
  - "激活函数"
  - "最大熵强化学习"
  - "策略熵约束"
  - "LLM 对齐"
---

# Entropy Regularizing Activation: Boosting Continuous Control, Large Language Models, and Image Classification with Activation as Entropy Constraints

**会议**: ICLR 2026  
**代码**: https://nothingbutbut.github.io/era  
**领域**: 强化学习  
**关键词**: 熵正则化、激活函数、最大熵强化学习、策略熵约束、LLM 对齐  

## 一句话总结

ERA（Entropy Regularizing Activation）通过在网络输出层附加专门设计的激活函数来施加熵下界约束，无需修改损失函数，一套框架同时提升连续控制 RL、LLM 推理和图像分类的性能。

## 研究背景与动机

**领域现状**：最大熵强化学习（如 SAC）将熵奖励直接加入优化目标，已成为连续控制的主流范式；在 LLM RL（GRPO 等）中，维持策略熵以防止探索崩溃同样是核心挑战。  
**现有痛点**：① 将熵 bonus 加入损失函数会扭曲主目标的优化景观，二者相互干扰，导致 SAC 在高维人形机器人任务上表现欠佳；② LLM 对齐中直接加熵 bonus 不稳定，KL-Cov/Clip-Cov 等启发式方法缺乏理论保证，且仅适用于离策略设定；③ 先前的投影方法（均匀混合）对所有维度施加相同正则化，在高维动作空间中扩展性差。  
**核心矛盾**：熵约束与主目标优化耦合在一起——一旦把熵项写进损失，主目标梯度就被污染。  
**本文目标**：设计一种与领域无关、无侵入、有理论保证的熵约束范式，完全解耦熵约束与主目标。  
**核心 idea**：不修改损失函数，而是在网络的最后一层输出上施加一个专门设计的激活函数 $g(\cdot)$，将裸输出的分布参数 $z$ 变换为 $z' = g(z)$，使变换后的策略 $\pi_{z'}$ 的期望熵满足 $\mathbb{E}_{s}[H(\pi_{\theta}(\cdot|s))] \geq H_0$，从而在架构层面内建熵下界。

## 方法详解

### 整体框架

ERA 定义了一个通用的输出激活框架：对于参数化策略 $f_\theta(s)$ 输出分布参数 $z$，在最后一层之后插入激活 $g: \mathcal{Z} \to \mathcal{Z}$，得到 $z' = g(z)$，最终策略为 $\pi_\theta(\cdot|s) = \pi_{g(f_\theta(s))}(\cdot|s)$。$g(\cdot)$ 的设计保证熵下界，同时对主目标损失完全透明——损失函数中不需要出现任何熵项。ERA 为连续空间（有界高斯策略）、离散空间（Softmax 策略）和 LLM RL（GRPO）分别给出具体实例化。

```mermaid
flowchart LR
    A[策略网络 f_θ] --> B[裸输出 z]
    B --> C{ERA 激活 g}
    C -->|z' = g(z)| D[变换后分布参数]
    D --> E[动作采样/分类/token 生成]
    D --> F[满足 H(π) ≥ H₀]
    G[主目标损失<br/>无熵项] --> A
```

### 关键设计

**1. 连续控制：基于有界高斯的熵下界激活**  
连续控制中策略通常对高斯采样后施加 $\tanh$ squash 或截断操作。有界策略的熵等于原始高斯熵减去一个非负偏置项：$H_\pi = H_\text{Gaussian} - \mathbb{E}[\text{bias}]$。因此，对最终策略施加熵下界 $H_0$ 等价于对底层高斯标准差施加更高的约束。ERA 通过如下激活函数同时满足熵下界 $H_0$ 和标准差范围约束 $[\sigma_\text{min}, \sigma_\text{max}]$：

$$\sigma'_i = \exp\!\left[\max\!\left(\log\sigma_\text{max} + \bigl(H'_0 - D\log\sqrt{2\pi e} - D\log\sigma_\text{max}\bigr)\frac{e^{\hat\sigma_i}}{\sum_j e^{\hat\sigma_j}},\; \log\sigma_\text{min}\right)\right]$$

其中 $H'_0 = H_0 + \hat\delta$ 是目标熵加上对 bounding bias 的补偿项（$\hat\delta$ 可固定或通过辅助损失 $\mathcal{L}(\hat\delta) = \mathbb{E}_s[\hat\delta(H[\pi(\cdot|s)] - H_0)]$ 自动学习）。由于熵约束已内建于激活，SAC 的 actor/critic 目标可直接去掉熵项，策略专注于最大化奖励。

**2. 离散分类：Softmax 熵下界激活**  
对于 Softmax 策略的分类任务，ERA 将预激活 logit $z$ 变换为 $z'$，使输出分布熵不低于 $H_0$：

$$z'_i = \hat{h}^{-1}\!\left[\max\!\left(\log\frac{\tau}{\tau} + \left(C_{H_0} - n\log\frac{\tau}{\tau}\right)\frac{1}{D-1}\left(1 - \frac{e^{z_i}}{\sum_j e^{z_j}}\right),\; 0\right)\right]$$

其中 $\hat{h}^{-1}(x) \approx -\frac{1}{4} - \sqrt{2(-1-\ln x)} + \frac{3}{4}\ln x$，$C_{H_0} = e^{H_0 - 1}$，$\tau \geq e$ 为固定超参。与标签平滑相比，ERA 是输入依赖的自适应正则化，而非全局均匀平滑，表达能力更强。

**3. LLM RL：基于 forking token 的后采样激活**  
LLM 中动作空间极大，大多数 token 接近确定性；对所有 token 强制高熵会破坏语言结构。ERA 的 LLM 实例化在采样后的模型更新阶段工作，仅对具有最高熵的 top-20% "forking token" 的 logit 施加激活：

$$z'_i = \begin{cases} kz_i & H_\text{resp} < \omega_\text{low},\; A_t > 0 \\ z_i & (\omega_\text{low} \leq H_\text{resp} \leq \omega_\text{high},\; A_t < 0)\;\text{or}\; A_t > 0 \\ \frac{1}{k}z_i & H_\text{resp} > \omega_\text{high},\; A_t > 0 \end{cases}$$

$k > 1$，$H_\text{resp}$ 为该 response 内 top-20% 高熵 token 的平均熵，$\omega_\text{low}/\omega_\text{high}$ 为上下界阈值。熵过低时锐化（$kz$）使模型"感知"自己在过度利用从而促进探索；熵过高时平坦化（$\frac{1}{k}z$）避免无效发散。同时对修改 token 的 advantage 施加对应缩放 $A'_t$，平衡修改与未修改 token 的梯度幅度。该设计兼容在策略设定（无需重要性采样比或 KL loss），且推理时策略不变。

**4. 理论保证**  
三种实例化均附有严格的熵下界证明（附录 B.1~B.3）：激活函数 $g(\cdot)$ 的构造保证了变换后分布参数对应的策略期望熵满足 $\mathbb{E}_s[H(\pi_\theta(\cdot|s))] \geq H_0$，这是先前启发式方法（clip-higher、KL-Cov 等）所不具备的性质。

## 实验关键数据

### 主实验

**连续控制**（归一化得分，aggregated IQM）：

| 任务集 | 算法 | Baseline | ERA-Augmented | Δ |
|--------|------|----------|---------------|---|
| HumanoidBench (6 tasks) | SAC | 0.59 | 0.84 | +42% |
| DMC Dog & Humanoid (6 tasks) | TD-MPC2 | 0.57 | 0.88 | +54% |
| HumanoidBench (8 tasks) | FastSAC | 0.56 | 0.81 | +45% |
| MuJoCo Gym (4 tasks) | PPO | 0.63 | 0.82 | +30% |

**LLM 推理**（Qwen2.5-Math-7B，avg.@16）：

| 基准 | GRPO | ERA | Δ |
|------|------|-----|---|
| AIME'24 | 34.4 | 36.0 | +4.7% |
| AIME'25 | 12.3 | 21.0 | +70.7% |
| AMC'23 | 69.5 | 76.6 | +10.4% |
| MATH500 | 80.6 | 85.4 | +6.0% |
| Minerva | 36.8 | 40.1 | +9.0% |
| OlympiadBench | 40.6 | 46.8 | +15.3% |
| **平均** | 45.7 | **51.0** | **+11.6%** |

**图像分类**（ResNet-50，ImageNet Top-1）：

| 设置 | Baseline | ERA | Δ |
|------|----------|-----|---|
| 无数据增强 | 74.75 | 75.44 | +0.69% |
| 有数据增强 | 76.93 | 77.30 | +0.37% |

### 消融实验

| 配置 | 指标 | 说明 |
|------|------|------|
| SAC-ERA（不同熵目标 $H_0$）| IQM 始终优于 SAC | 对熵目标超参不敏感，无需精细调参 |
| SAC w/o 熵项（但无 ERA）| 低于 SAC-ERA | 去掉熵 bonus 不足，需 ERA 保证探索下界 |
| Qwen2.5-Math-1.5B + ERA vs GRPO | avg +14.1% | 泛化到更小模型 |
| GSPO + ERA vs GSPO（7B）| avg +6.9% | 与不同 RL 算法兼容 |
| ImageNet，不同 $H_0$ | Top-1 保持稳定 | 对熵超参鲁棒 |

### 关键发现

- ERA 始终将策略熵维持在非零下界，而 GRPO baseline 出现典型的 entropy collapse；熵稳定与推理性能提升高度相关。
- 在 HumanoidBench 这类高维任务上提升尤为显著（>30%），因为这些任务对探索质量最敏感。
- ERA 计算开销 <7%，可直接叠加在已有算法上，无需改变其他组件。

## 亮点与洞察

- **解耦思路干净**：把熵约束从损失函数移到网络架构（激活函数），是一个概念上非常优雅的迁移——primary loss 专注奖励最大化，熵保证由结构承担。
- **理论支撑扎实**：三种场景均有严格的熵下界证明，区别于此前的经验性启发方法。
- **跨域统一**：同一范式覆盖连续控制、离散分类、LLM RL 三个看似不同的领域，揭示了"输出分布熵控制"的通用性。
- **对超参鲁棒**：实验表明在较宽的 $H_0$ 范围内性能稳定，减轻了调参负担。
- **与现有方法互补**：在已有数据增强、标签平滑的基础上仍有提升，说明 ERA 填补的是不同的正则化空白。

## 局限与展望

- LLM 实例化的 $\omega_\text{low}/\omega_\text{high}/k$ 等超参仍需针对不同模型手动设置，自动化调参机制有待研究。
- forking token 的 top-20% 截断是启发式选择，更细粒度的 token 重要性估计可能进一步提升效果。
- 目前只在数学推理任务上验证了 LLM 性能，是否推广到代码生成、多模态等任务尚未探索。
- 对于多智能体或部分可观测场景，熵约束的适配方式还需进一步研究。

## 相关工作与启发

- **vs SAC（最大熵 RL）**：SAC 将熵 bonus 写入 Q-target 和 actor loss，ERA 完全移除这两项熵项，用激活取代，在高维任务上优势明显。
- **vs Akrour et al. (2019)/Otto et al. (2021)（投影方法）**：同属"不修改目标函数"的思路，但其均匀混合（各维度等同正则化）在高维空间扩展性差；ERA 通过 softmax 加权实现维度感知的梯度引导。
- **vs KL-Cov/Clip-Cov（LLM 熵控制）**：这些方法依赖重要性采样比，仅适用于离策略，且无理论下界；ERA 兼容在策略设定且有证明。
- **vs 标签平滑（分类正则化）**：标签平滑是全局均匀的固定正则化；ERA 是自适应的、输入依赖的，具有更强的表达能力。

## 评分

- 新颖性: ⭐⭐⭐⭐ 输出激活实现熵约束的视角新颖，且统一了三个不同领域，泛化性强  
- 实验充分度: ⭐⭐⭐⭐ 覆盖连续控制（5种算法/多环境）、LLM 推理（6基准/2模型/2算法）、图像分类，消融详尽  
- 写作质量: ⭐⭐⭐⭐ 动机清晰，理论与实验结合紧密，公式推导完整  
- 价值: ⭐⭐⭐⭐ 轻量（<7% overhead）、即插即用、有理论保证，工程落地价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] AutoTool: Automatic Scaling of Tool-Use Capabilities in RL via Decoupled Entropy Constraints](autotool_automatic_scaling_of_tool-use_capabilities_in_rl_via_decoupled_entropy_.md)
- [\[ICLR 2026\] GoldenStart: Q-Guided Priors and Entropy Control for Distilling Flow Policies](goldenstart_q-guided_priors_and_entropy_control_for_distilling_flow_policies.md)
- [\[ICLR 2026\] Learning Massively Multitask World Models for Continuous Control](learning_massively_multitask_world_models_for_continuous_control.md)
- [\[ICLR 2026\] Relative Entropy Pathwise Policy Optimization](relative_entropy_pathwise_policy_optimization.md)
- [\[ICLR 2026\] WIMLE: Uncertainty-Aware World Models with IMLE for Sample-Efficient Continuous Control](wimle_uncertainty-aware_world_models_with_imle_for_sample-efficient_continuous_c.md)

</div>

<!-- RELATED:END -->
