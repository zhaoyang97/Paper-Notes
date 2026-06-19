---
title: >-
  [论文解读] How to Build a Consistency Model: Learning Flow Maps via Self-Distillation
description: >-
  [NeurIPS 2025][模型压缩][Flow Map] 提出统一的自蒸馏（Self-Distillation）框架来直接学习 flow map（即 consistency model 的一般化形式），通过 tangent condition 将任意蒸馏方案转化为无需预训练教师的直接训练算法，并导出三大算法族（Eulerian / Lagrangian / Progressive），其中 Lagrangian 方法避免了空间梯度和自举引导，训练最稳定、性能最优。
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "Flow Map"
  - "Consistency Model"
  - "Self-Distillation"
  - "加速推理"
  - "Flow Matching"
---

# How to Build a Consistency Model: Learning Flow Maps via Self-Distillation

**会议**: NeurIPS 2025  
**arXiv**: [2505.18825](https://arxiv.org/abs/2505.18825)  
**代码**: [nmboffi/flow-maps](https://github.com/nmboffi/flow-maps)  
**领域**: 图像生成  
**关键词**: Flow Map, Consistency Model, Self-Distillation, 加速推理, Flow Matching

## 一句话总结
提出统一的自蒸馏（Self-Distillation）框架来直接学习 flow map（即 consistency model 的一般化形式），通过 tangent condition 将任意蒸馏方案转化为无需预训练教师的直接训练算法，并导出三大算法族（Eulerian / Lagrangian / Progressive），其中 Lagrangian 方法避免了空间梯度和自举引导，训练最稳定、性能最优。

## 研究背景与动机
**领域现状**：基于 flow/diffusion 的生成模型在视觉、语言、蛋白质结构预测等领域取得了顶尖效果，但推理时需反复求解 ODE/SDE，计算开销大，限制了实时应用。

**加速推理需求**：Consistency Model 通过学习 flow map（概率流方程的解算子）实现一步或少步采样，推理加速 10–100 倍。但已有方法缺乏统一的数学表述，训练不稳定，工程复杂度高。

**蒸馏 vs 直接训练**：蒸馏方法（如 progressive distillation）需要先训练一个教师模型再训练学生，是两阶段流程，学生性能受限于教师质量。直接训练方法（如 consistency training）则面临优化不稳定、需要精心设计退火策略和梯度裁剪等技巧。

**核心矛盾**：缺乏一个统一的数学框架来揭示如何高效地学习 flow map，现有方法被视为相互独立的算法，设计原则不清晰。

**切入角度**：利用 tangent condition（flow map 在对角线 $s=t$ 处的时间导数恢复速度场 $b_t$）这一关键关系，将蒸馏方案转化为自蒸馏，用一个网络同时学速度场和 flow map。

## 核心问题
- **是否存在一套原则性的 flow map 训练方法论，能像标准 flow matching 一样简洁？**
- 能否在无预训练教师的情况下高效稳定地训练 flow map？
- 不同数学表征（Eulerian / Lagrangian / Semigroup）对训练稳定性和性能的影响如何？

## 方法详解

### 理论基础：Flow Map 的三种等价刻画

设 $X_{s,t}(x)$ 为概率流 $\dot{x}_t = b_t(x_t)$ 的 flow map，满足跳跃条件 $X_{s,t}(x_s) = x_t$。作者给出三种等价刻画：

1. **Lagrangian 条件**（ODE 视角）：
   $$\partial_t X_{s,t}(x) = v_{t,t}(X_{s,t}(x))$$
   flow map 在 $t$ 方向上的变化率等于终点处的速度场——这是一个关于 $t$ 的 ODE。

2. **Eulerian 条件**（PDE 视角）：
   $$\partial_s X_{s,t}(x) + \nabla X_{s,t}(x)\, v_{s,s}(x) = 0$$
   涉及 flow map 对空间的 Jacobian $\nabla X_{s,t}$——这是一个关于 $s$ 的 PDE。

3. **Semigroup 条件**（组合性）：
   $$X_{u,t}(X_{s,u}(x)) = X_{s,t}(x)$$
   两步跳跃可以被一步跳跃替代。

### Tangent Condition：连接速度场与 Flow Map

核心数学洞察：
$$\lim_{s \to t} \partial_t X_{s,t}(x) = b_t(x)$$

即 flow map 在对角线 $s=t$ 处的切线方向恰好恢复速度场。这意味着 flow map 中"隐含"了一个速度模型。

基于此，作者采用参数化：
$$\hat{X}_{s,t}(x) = x + (t-s)\, \hat{v}_{s,t}(x)$$

其中 $v_{s,t}$ 几何意义为 ODE 轨迹上 $x_s$ 到 $x_t$ 连线的"斜率"，满足 $v_{t,t}(x) = b_t(x)$。

### 自蒸馏框架

总损失由两部分组成：
$$\mathcal{L}_{\text{SD}}(\hat{v}) = \mathcal{L}_b(\hat{v}) + \mathcal{L}_D(\hat{v})$$

- **对角损失** $\mathcal{L}_b$：在 $s=t$ 处用标准 flow matching 学速度场
  $$\mathcal{L}_b = \int_0^1 \mathbb{E}_{x_0,x_1}\left[|\hat{v}_{t,t}(I_t) - \dot{I}_t|^2\right] dt$$

- **离对角损失** $\mathcal{L}_D$：在 $s \neq t$ 处用某种蒸馏目标学 flow map，有三种选择：

| 方法 | 损失来源 | 是否涉及空间梯度 | 是否涉及自举 |
|------|----------|------------------|-------------|
| **LSD**（Lagrangian） | Lagrangian ODE 残差 | ❌ | ❌ |
| **ESD**（Eulerian） | Eulerian PDE 残差 | ✅（Jacobian） | ❌ |
| **PSD**（Progressive） | Semigroup 组合一致性 | ❌ | ✅（小步→大步） |

### LSD（Lagrangian Self-Distillation）——最优方法

$$\mathcal{L}_{\text{LSD}} = \int_0^1 \int_0^t \mathbb{E}\left[|\partial_t \hat{X}_{s,t}(I_s) - \hat{v}_{t,t}(\hat{X}_{s,t}(I_s))|^2\right] ds\, dt$$

- 比较 flow map 的时间导数与速度场在 flow map 终点处的预测
- **不需要空间梯度**（避免 Jacobian 计算），训练梯度更平稳
- **不需要自举**（不依赖从小步组合出大步），避免误差积累
- 使用 stop-gradient 使信息从对角线（速度场）单向流向离对角线（flow map）

### 关键工程细节

1. **自适应损失加权**：引入可学习权重 $w_{s,t}$（类似 EDM2 扩展到两时间维度）:
   $$\mathcal{L} = \mathbb{E}_{p_{s,t}}\left[e^{-w_{s,t}} \cdot \ell_{s,t} + w_{s,t}\right]$$
   $w_{s,t}$ 估计损失的对数方差，使不同 $(s,t)$ 对的梯度贡献归一化。

2. **时间采样**：使用混合分布 $p_{s,t} = \eta\, U_d + (1-\eta)\, U_{od}$，其中 $\eta=0.75$ 将 75% 的 batch 用于对角线 flow matching，25% 用于离对角线蒸馏。因为蒸馏目标更昂贵（需要多次网络求值和 JVP），$\eta$ 也可用于控制训练开销。

3. **PSD 缩放预处理**：PSD 损失经重写后可消除 $(t-s)^2$ 因子，避免不同时间步的有效学习率差异过大。

### 与现有方法的统一

通过选择不同的蒸馏目标和教师，该框架可恢复：
- **Consistency Models**（Song et al., 2023）→ Eulerian 视角的特殊情况
- **Consistency Trajectory Models**（Kim et al., 2024）→ 两时间 Eulerian 蒸馏
- **Shortcut Models**（Frans et al., 2024）→ 离散化 semigroup 条件
- **Progressive Distillation**（Salimans & Ho, 2022）→ PSD 的蒸馏版本
- **Mean Flow** / **Align Your Flow** → Eulerian 的特定实例

## 实验关键数据

在 Checkerboard（2D 合成）、CIFAR-10、CelebA-64、AFHQ-64 上比较 LSD、ESD、PSD-M（midpoint）、PSD-U（uniform），固定训练时间公平对比：

| 数据集 | 方法 | 1-step | 2-step | 4-step | 8-step | 16-step |
|--------|------|--------|--------|--------|--------|---------|
| CIFAR-10 (FID↓) | **LSD** | **8.10** | **4.37** | **3.34** | **3.33** | **3.57** |
| | PSD-M | 12.81 | 8.43 | 5.96 | 5.07 | 4.64 |
| | PSD-U | 13.61 | 7.95 | 6.03 | 5.32 | 5.16 |
| CelebA-64 (FID↓) | **LSD** | **12.22** | **5.74** | **3.18** | **2.18** | **1.96** |
| | PSD-M | 19.64 | 11.75 | 7.89 | 6.06 | 5.09 |
| AFHQ-64 (FID↓) | **LSD** | **11.19** | **7.78** | **7.00** | **5.89** | **5.61** |
| | PSD-U | 14.50 | 10.73 | 10.99 | 12.02 | 11.47 |

- **ESD 在图像数据集上训练不稳定**，梯度范数远大于 LSD/PSD，最终发散，无法报告 FID
- **LSD 在所有数据集、所有步数上均为最优**
- LSD 在 CIFAR-10 上 4-step 即可达到 FID 3.34，接近 8-step 的 3.33，说明性能早早饱和
- Checkerboard 上 LSD 在 $N \geq 4$ 时 KL 即饱和（0.07），其他方法需 16 步才逼近

## 理论保证

对 LSD 和 ESD 证明了 Wasserstein 界：
- **LSD**：$W_2^2(\hat{\rho}_1, \rho_1) \leq 4 e^{1+2\hat{L}} \varepsilon$
- **ESD**：$W_2^2(\hat{\rho}_1, \rho_1) \leq 2e(1+e^{2\hat{L}}) \varepsilon$

其中 $\varepsilon$ 为损失值，$\hat{L}$ 为 $v_{t,t}$ 的 Lipschitz 常数。PSD 由于误差累积和分布偏移问题，无法获得类似保证。

## 亮点
1. **统一框架**：将 consistency model、progressive distillation、shortcut model 等方法统一纳入一个数学体系，揭示共同设计原则
2. **自蒸馏消除教师依赖**：通过 tangent condition，用一个网络同时学速度场和 flow map，不需要两阶段训练
3. **Lagrangian 方法的优越性**：LSD 避免空间梯度和自举，在所有实验中稳定性和性能均最优，解释了 consistency model 训练不稳定的根本原因（Eulerian 视角引入 Jacobian）
4. **理论与实践一致**：LSD 有更紧的 Wasserstein 界，实验上也确实更优；PSD 无法获得理论保证，实验上也确实更弱
5. **即插即用**：框架可直接扩展到条件生成和 CFG 引导

## 局限与展望
1. **计算开销未充分探索**：由于计算资源限制，未能对所有设计选择（参数化、架构、stop-gradient 方案）进行系统消融
2. **仅做了无条件生成**：未在 class-conditional 或 text-to-image 场景下验证，CFG 引导留给了未来工作
3. **分辨率有限**：实验最高仅到 64×64，未在 256/512 分辨率或 latent space 上验证
4. **单模型容量压力**：用一个网络同时表示速度场和 flow map，可能需要更大的网络容量；论文提到了双模型和高阶参数化但未实验
5. **LSD 的少步性能仍有提升空间**：1-step FID（CIFAR-10: 8.10）相比 SOTA consistency model 仍有差距

## 与相关工作的对比

| 方法 | 是否需要教师 | 数学基础 | 训练稳定性 | 多步推理 |
|------|-------------|---------|-----------|---------|
| Consistency Model (CT) | ❌ | Eulerian PDE | 差（需精心工程） | ❌（单时间） |
| Consistency Distillation (CD) | ✅ | Eulerian PDE | 较好 | ❌ |
| Progressive Distillation | ✅ | Semigroup | 较好 | ✅ |
| Shortcut Models | ❌ | 离散 Semigroup | 中等 | ✅ |
| **本文 LSD** | **❌** | **Lagrangian ODE** | **最优** | **✅** |

本文的关键贡献在于揭示了 Eulerian 视角（Consistency Model 所用）的不稳定性源于空间 Jacobian 计算，而 Lagrangian 视角完全规避了这一困难。

## 训练与推理细节

### 训练流程（Algorithm 1）

1. 每个 mini-batch 按比例 $\eta$ 分为对角线样本和离对角线样本
2. **对角线部分**：采样 $M_d = \lfloor \eta M \rfloor$ 对 $(x_0, x_1)$ 和时间 $t$，计算 interpolant $I_t$ 和目标 $\dot{I}_t$，用标准 flow matching 损失训练 $v_{t,t}$
3. **离对角线部分**：采样 $M_o = M - M_d$ 对和时间对 $(s,t)$，计算 interpolant $I_s$，用所选蒸馏目标（LSD/ESD/PSD）训练 $v_{s,t}$
4. 两部分损失相加后统一反传，自适应权重 $w_{s,t}$ 同步更新

### Stop-gradient 策略

- LSD 中对速度场预测 $\hat{v}_{t,t}(\hat{X}_{s,t}(I_s))$ 施加 stop-gradient，确保信息从对角线（有外部监督信号 $\dot{I}_t$）单向流向离对角线
- 这等价于将自身的对角线预测视为"冻结教师"，因此称为"自蒸馏"
- PSD 中对组合步的"教师"侧（$\hat{X}_{u,t}(\hat{X}_{s,u}(I_s))$）施加 stop-gradient
- ESD 中对 $v_{s,s}(I_s)$ 施加 stop-gradient

### 推理：灵活多步采样

- **1-step**：$\hat{x}_1 = \hat{X}_{0,1}(x_0) = x_0 + v_{0,1}(x_0)$，单次前向即完成采样
- **N-step**：在 $[0,1]$ 上均匀分 $N$ 段，逐步组合 $\hat{X}_{t_i, t_{i+1}}$，用更多计算换更高质量
- 多步推理时无需额外训练，模型天然支持任意步数——这是双时间参数化 $v_{s,t}$ 的核心优势

## 启发与关联
1. **对 flow-based 生成模型加速的启示**：Lagrangian 视角可能成为未来 consistency model 训练的标准范式，替代当前 Eulerian 方案
2. **跨模态应用**：框架完全不依赖图像特性，可直接扩展到语言（discrete flow matching）、视频、3D 等领域
3. **与 VeCoR 等流正则化方法互补**：LSD 从训练目标层面加速推理，VeCoR 等方法从速度场质量层面改善生成质量，两者可结合
4. **自蒸馏思想的广泛性**：tangent condition 的"从对角线向离对角线传播信号"的思路可能启发其他需要学习解算子的 PDE 问题

## My Notes

### 为什么 Lagrangian 优于 Eulerian？——从梯度角度的直觉

Eulerian 条件 $\partial_s X_{s,t} + \nabla X_{s,t} \cdot v_{s,s} = 0$ 要求计算 flow map 对输入空间的 Jacobian $\nabla X_{s,t}$。在神经网络中，这个 Jacobian 是一个 $d \times d$ 矩阵（$d$ 为数据维度），即使用 JVP 高效计算，其梯度仍然涉及二阶导数（Hessian-vector product），导致梯度范数不稳定。CIFAR-10 实验中 ESD 的梯度范数比 LSD 大数个量级，最终训练发散。

相比之下，Lagrangian 条件 $\partial_t X_{s,t}(x) = v_{t,t}(X_{s,t}(x))$ 只需要 flow map 对时间 $t$ 的导数（通过 JVP 高效计算）和速度场在 flow map 终点处的求值——完全不涉及空间 Jacobian。这从根本上解释了为什么 consistency training（Eulerian 视角）需要精心工程而 LSD 可以"开箱即用"。

### 与 Shortcut Models 的深层联系

Shortcut Models（Frans et al., 2024）本质上是 semigroup 条件的离散化：只在有限的 step size 集合 $\{0, \Delta, 2\Delta, ...\}$ 上训练，通过 self-consistency 从小步推大步。本文的 PSD 是其连续化推广，但实验表明 PSD 性能不如 LSD，因为 semigroup 条件存在误差累积——大步的精度依赖于小步的精度，形成链式依赖。

这启示我们：**避免自举（bootstrapping）是训练稳定性的关键**。LSD 每个 $(s,t)$ 对独立训练，不依赖其他步长的预测质量，因此更鲁棒。

### 单模型 vs 双模型的 trade-off

本文用单个网络 $v_{s,t}$ 同时表示速度场（$s=t$）和 flow map（$s \neq t$），优势是参数共享和训练效率，但要求网络容量足够大以同时拟合两种不同的功能。论文提到了双模型方案和高阶参数化 $X_{s,t}(x) = x + (t-s)b_s(x) + \frac{1}{2}(t-s)^2 \psi_{s,t}(x)$（类似 ODE 求解器的二阶展开），但未实验。这是一个值得探索的方向，特别是在高分辨率生成中网络容量可能成为瓶颈时。

### 潜在扩展方向

- **Latent space 应用**：与 Latent Diffusion / SDXL 结合，在 latent space 中训练 flow map，有望在 text-to-image 任务上实现 1-4 步高质量生成
- **CFG 蒸馏**：论文已给出 CFG flow map 的理论公式 $v_{t,t}(x;\alpha,c) = q_t(x;\alpha,c)$，但未实验。将 guidance scale $\alpha$ 作为额外条件加入训练是直接的扩展
- **与 InstaFlow / SDXL-Turbo 对比**：这些方法用 progressive distillation 实现少步生成，本文的 LSD 可作为直接训练的替代方案，消除对教师模型的依赖

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 统一框架 + 三大算法族 + Lagrangian 新方法，理论深度和广度突出
- 实验充分度: ⭐⭐⭐⭐ — 多数据集公平对比，但分辨率受限、缺少条件生成实验
- 写作质量: ⭐⭐⭐⭐⭐ — 数学推导严谨，系统性强，图示清晰
- 价值: ⭐⭐⭐⭐⭐ — 为 consistency model 家族提供了原则性的设计指南，具有长期影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Robustifying Learning-Augmented Caching Efficiently without Compromising 1-Consistency](robustifying_learning-augmented_caching_efficiently_without_compromising_1-consi.md)
- [\[NeurIPS 2025\] Learning to Better Search with Language Models via Guided Reinforced Self-Training](learning_to_better_search_with_language_models_via_guided_reinforced_self-traini.md)
- [\[NeurIPS 2025\] SCAN: Self-Denoising Monte Carlo Annotation for Robust Process Reward Learning](scan_self-denoising_monte_carlo_annotation_for_robust_process_reward_learning.md)
- [\[NeurIPS 2025\] Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning](mixture_of_noise_for_pre-trained_model-based_class-incremental_learning.md)
- [\[NeurIPS 2025\] Understanding Differential Transformer Unchains Pretrained Self-Attentions](understanding_differential_transformer_unchains_pretrained_self-attentions.md)

</div>

<!-- RELATED:END -->
