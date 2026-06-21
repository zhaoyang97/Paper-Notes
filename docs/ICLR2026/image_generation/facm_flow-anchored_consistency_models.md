---
title: >-
  [论文解读] FACM: Flow-Anchored Consistency Models
description: >-
  [ICLR 2026][图像生成][Consistency Models] 把 Flow Matching 当作"锚"和 Consistency Model 的"捷径"目标混在一个模型里联合训练，用一个"扩展时间区间"技巧把两个任务解耦到不同时间域，从根上治好了连续时间一致性模型的训练崩溃问题，在 ImageNet 256×256 上 NFE=1/2 分别刷到 FID 1.70/1.32，并能扩展到 14B 文生图模型。
tags:
  - "ICLR 2026"
  - "图像生成"
  - "Consistency Models"
  - "Flow Matching"
  - "少步生成"
  - "蒸馏"
  - "训练稳定性"
  - "JVP"
  - "FSDP"
---

# FACM: Flow-Anchored Consistency Models

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=k9BpW1c4in](https://openreview.net/forum?id=k9BpW1c4in)  
**代码**: [https://github.com/ali-vilab/FACM](https://github.com/ali-vilab/FACM)  
**领域**: 图像生成 / 少步采样 / 一致性模型  
**关键词**: Consistency Models, Flow Matching, 少步生成, 蒸馏, 训练稳定性, JVP, FSDP  

## 一句话总结
把 Flow Matching 当作"锚"和 Consistency Model 的"捷径"目标混在一个模型里联合训练，用一个"扩展时间区间"技巧把两个任务解耦到不同时间域，从根上治好了连续时间一致性模型的训练崩溃问题，在 ImageNet 256×256 上 NFE=1/2 分别刷到 FID 1.70/1.32，并能扩展到 14B 文生图模型。

## 研究背景与动机
**领域现状**：扩散/流匹配模型生成质量很高，但要几十上百步才能出图，实时应用扛不住。一致性模型（Consistency Models, CM）想把这条 ODE 轨迹"抄近路"——从轨迹上任意一点 $x_t$ 一步直接映射到终点 $x_1$。离散时间版本受离散化误差困扰，连续时间版本理论上能绕开这个误差，却长期被严重的**训练不稳定**所困，动不动就崩。

**现有痛点**：近期两条路线都只治标不治本。一条是 sCM 靠正则化加架构改造（pixel normalization）来稳住连续时间训练，但这些归一化层的改动让它很难套到大规模预训练模型上；另一条 Flow Mapping 系（MeanFlow、IMM 等）通过重新设计捷径目标——比如建模"平均速度"或加多时间步自一致约束——来求稳，但它们都靠一个**过度耦合的单一目标**同时去学"流"和"捷径"，没法显式解耦两个任务，导致对流的轨迹保真度被牺牲。

**核心矛盾**：连续时间 CM 的训练目标 $T = v + (1-t)\frac{dF_\theta}{dt}$ 是**自指**的——目标里既含真实瞬时速度 $v$，又含模型对自己导数的估计。问题在于 CM 损失只对最终预测 $F_\theta$ 施加监督，没有任何机制保证模型内部学到的动态忠实于底层瞬时速度场 $v(x_t,t)$。一旦缺这个"锚"，模型输出开始漂移，导数项 $(1-t)\frac{dF_\theta}{dt}$ 就会反客为主、压过真值速度 $v$ 的监督信号，满足一致性恒等式不再收敛到边界条件，训练目标变得噪声满天飞，形成放大误差的恶性循环直至崩溃。

**本文目标**：从根上消除连续 CM 的不稳定，做到架构无关、可扩展到超大模型。

**核心 idea**（**Flow-Anchoring**）：作者主张不稳定不是捷径目标本身的缺陷，而是"孤立地只训捷径"造成的——模型丢失了它在底层速度场里的锚点。解法很直接：**重新显式地把瞬时速度场的监督（即一个 Flow Matching 任务）加回来，当作主捷径目标的动态锚**，让模型的梯度场行为良好，从而稳住 CM 目标里的导数项。

## 方法详解

### 整体框架
FACM 不需要任何专门架构，只用一个**混合目标**的训练策略：总损失就是 Flow Matching 损失（锚，提供稳定性）加 Consistency Model 损失（加速器，学一步捷径）的直接相加 $L_{\text{FACM}} = L_{\text{FM}} + L_{\text{CM}}$。关键问题是怎么让同一个模型分清"现在该预测瞬时速度还是平均速度"——作者用一个**扩展时间区间**条件信号把两个任务解耦到 $[0,1]$ 和 $[1,2]$ 两个时间域，再配上一个稳定的插值式 CM 目标和可扩展的 Chain-JVP 实现，整套在单个训练循环里跑通。

```mermaid
flowchart TD
    A[采样 x0, x1, t<br/>构造 xt=(1-t)x0+t·x1] --> B[按 t 定义两个条件<br/>cCM=t, cFM=2-t]
    B --> C[FM 分支: Fθ(xt, cFM)<br/>回归瞬时速度 v ——锚]
    B --> D[CM 分支: Fθ(xt, cCM) + JVP<br/>算一致性残差 g 并 clamp]
    D --> E[松弛插值目标<br/>vtar=(1-α)·sg(FCM)+α·T(F)]
    C --> F[L_FM]
    E --> G[L_CM]
    F --> H[L_total = L_FM + L_CM]
    G --> H
```

### 关键设计
**1. Flow-Anchoring：用 FM 任务当锚根治不稳定。** 作者先把捷径目标的物理意义点透——用 OT-FM 参数化 $f_\theta(x_t,t)=x_t+(1-t)F_\theta(x_t,t)$，要实现一步捷径 $f_\theta=x_1$，网络必须学到 $F_\theta(x_t,t)=\frac{x_1-x_t}{1-t}$，这正是从 $x_t$ 到终点的**平均速度** $v(x_t,t)$。对它求导能推出恒等式 $v(x_t,t)=v(x_t,t)+(1-t)\frac{dv}{dt}$，形式上和连续 CM 目标、MeanFlow 恒等式一致。问题就出在孤立训这个目标时模型对瞬时速度 $v$ 没有任何直接监督，导数项会失控。FACM 的对策就是把 Flow Matching 损失加回来作为锚：

$$L_{\text{FM}}(\theta) = \mathbb{E}\big[\|F_\theta(x_t, c_{\text{FM}}) - v\|_2^2 + L_{\cos}(F_\theta(x_t, c_{\text{FM}}), v)\big]$$

其中 $v$ 在蒸馏时由非训练的教师 $F_\delta$ 给出（含可选 CFG 项 $v=v_{\text{base}}+w(v_{\text{cond}}-v_{\text{uncond}})$），从头训练时直接用 $x_1-x_0$。这个锚保证模型梯度场良好，让 CM 目标里那个原本失控的导数项稳下来——稳定性由 Flow-Anchoring 原理本身保证，与具体加权、损失函数选择无关。

**2. 扩展时间区间：把两个任务解耦到不同时间域。** 难点在于同一个模型怎么知道现在该预测瞬时速度（FM）还是平均速度（CM）。作者的巧解是**概念上把时间域翻倍**：CM 任务照常在 $t\in[0,1]$ 上，条件直接取 $c_{\text{CM}}=t$；要在同一个点 $x_t$ 上做 FM 任务，就把时间映射到另一个区间 $[1,2]$，令 $c_{\text{FM}}=2-t$。这样两个条件解耦、对称、易区分，且无需任何架构改动。更妙的是它天然保证边界连续性——CM 目标在 $t\to1$ 时恰好收敛到 FM 目标：

$$\lim_{t\to1^-}\Big(v + (1-t)\frac{dF_\theta(x_t,t)}{dt}\Big) = v$$

所以两个学习区间之间是平滑过渡。作者也试过另一种"辅助时间条件"方案（给模型加第二个时间变量 $r$，$c_{\text{CM}}=(t,1)$、$c_{\text{FM}}=(t,t)$），但消融显示扩展区间因为用了 $[0,1]$ 与 $[1,2]$ 这种高度可分的时间域，条件信号更清晰，效果更好。

**3. 松弛式定点迭代 CM 目标：让捷径学习稳定收敛。** CM 损失被理解为一个定点问题 $F_\theta = T(F_\theta)$，算子 $T(F)\triangleq v+(1-t)\frac{dF}{dt}$。直接逼近 $T(F)$ 容易梯度爆炸，作者改成先算 stop-gradient 模型的一致性残差 $g = F_{\theta^-} - T(F_{\theta^-})$ 并把它 clamp 到 $[-1,1]$ 抑制极端梯度，再构造**插值目标**：

$$v_{\text{tar}} = (1-\alpha(t))F_{\theta^-} + \alpha(t)\,T(F_{\theta^-})$$

这相当于在"当前模型输出"和"理想一致性目标"之间做一步松弛迭代，配合权重 $\beta(t)$ 用 norm L2 损失 $L_{\text{CM}}=\mathbb{E}[\beta(t)\cdot L_{\text{norm}}(F_\theta(x_t,c_{\text{CM}}), v_{\text{tar}})]$。作者强调插值、beta 加权、残差 clamp 这些都是为了加速收敛，而非稳定性的前提。

**4. Chain-JVP：让 JVP 兼容 FSDP，扩展到 14B 模型。** CM 损失里的总导数 $\nabla_t F_\theta$ 要靠 Jacobian-Vector Product（JVP）算，但标准 JVP 要求把整个模型参数 $\theta$ materialize 到设备上，在 FSDP 下会触发 all-gather 把全部参数重建到每张卡，造成内存暴涨，10B+ 参数训练直接不可行。作者利用链式法则把 JVP 按模块顺序拆开 $J_{F_\theta}(z)\cdot v = J_{f_L}\cdot(\cdots J_{f_1}(z_0)\cdot v)$，每次只算一个模块的 JVP 并嵌进 FSDP 逻辑，因此**任一时刻只 materialize 一个模块的参数**，峰值内存取决于最大模块而非整个模型，省的内存随参数量增长越省越多，速度与单机 JVP 一致。这是把 FACM 扩到 Wan 2.2（14B）文生图、把推理从 2×40 步压到 2-8 步的工程关键。

## 实验关键数据

### 主实验（少步生成 SOTA）

ImageNet 256×256（class-conditional）：

| 方法 | Params | NFE | FID (↓) |
|---|---|---|---|
| LightningDiT (多步基线) | 675M | 250×2 | 1.35 |
| MeanFlow | 676M | 1 | 3.43 |
| **FACM (Ours)** | 675M | **1** | **1.70** |
| MeanFlow | 676M | 2 | 2.20 |
| IMM | 675M | 1×2 | 7.77 |
| **FACM (Ours)** | 675M | **2** | **1.32** |

CIFAR-10（unconditional）：FACM 在 NFE=1 拿到 FID 2.69（优于 sCM 2.85、MeanFlow 2.92），NFE=2 拿到 1.87（优于 IMM 1.98、sCM 2.06）。少步模型甚至超过了部分需要上百次评估的多步基线。

### 消融实验

不同骨干（ImageNet 256×256, NFE=2，FID↓），证明架构无关：

| Backbone | 多步基线 | sCM† | MeanFlow† | **FACM** |
|---|---|---|---|---|
| SiT-XL/2 | 2.06 | 2.83 | 2.27 | **2.07** |
| REPA | 1.42 | 2.25 | 1.88 | **1.52** |
| DiT-XL/2 | 2.27 | 2.91 | 2.62 | **2.31** |
| LightningDiT | 1.35 | 1.94 | 1.74 | **1.32** |

稳定化策略（同一 LightningDiT 教师蒸馏，NFE=1）：sCM 不加 pixel norm 直接崩（✗），加了也只到 FID 3.04；MeanFlow 2.75；FACM 辅助条件版 1.97，**扩展区间版 1.81**，都稳定。

关键技术拆解（Epoch 10, NFE=1）：纯 MeanFlow / 0% FM 直接崩溃（FID 372-391）；加 75% FM 降到 43;；加 Flow-Anchoring 扩展时间区间骤降到 **4.31**；再加插值 $\alpha$ 到 3.42。

### 关键发现
- **FM 锚是稳定性的充分条件**：去掉 FM 目标就崩，加上就稳，与加权/损失选择无关。
- **对 FM 损失权重极鲁棒**：finetuned 设定下 $\lambda_{\text{FM}}$ 低到 $10^{-8}$ 仍稳，non-finetuned 需 $\geq0.1$，因此直接取 $\lambda_{\text{FM}}=1.0$ 求和即可，免去调参。
- **教师越强越好**：FACM 性能随教师质量单调提升，说明它是高保真轨迹压缩而非有损妥协。

## 亮点与洞察
- **诊断深刻**：把连续 CM 不稳定归因为"目标自指 + 丢失流锚导致导数项压过真值速度"，比"加正则/改架构"的工程经验解释更触及本质，结论"孤立训捷径会灾难性遗忘瞬时速度场"很有说服力。
- **解法极简又通用**：FM + CM 直接相加、扩展时间区间不改架构，所以能无缝套到任意预训练骨干上，这点比必须改归一化层的 sCM 实用得多。
- **任务解耦 vs 过度耦合的对比清晰**：MeanFlow 把瞬时速度当成 $r=t$ 的边界特例，监督信号被稀释；FACM 用不同时间域让锚始终拿到干净、不被稀释的监督，思路上是对 Flow Mapping 系的一记正面回应。
- **Chain-JVP 把理论落到 14B 规模**：解决 JVP 与 FSDP 的内存冲突是真正让方法"可用"的一步，不是 toy。

## 局限与展望
- 默认走两阶段（先预训 FM 教师再蒸馏），从头训练版本（CIFAR-10 FID 2.27）与蒸馏版仍有差距，端到端从头训的潜力待挖。
- 主要在 FID 这一个指标和 ImageNet/CIFAR 上验证，文生图 14B 规模虽展示了可扩展性，但缺更系统的大规模定量评测（如人类偏好、prompt 一致性）。
- 扩展时间区间把 $[1,2]$ 留给 FM，等于占用了一段条件空间，对那些已经把时间条件用满的模型是否有副作用没充分讨论。
- 多步采样仍依赖标准 CM 采样流程，NFE 增大到 8 步时相比顶尖多步流匹配的质量上限对比可以更细。

## 相关工作与启发
- **连续时间一致性模型 sCM**：靠 pixel normalization 等架构改造稳定训练，FACM 把它当主要对照，指出其架构依赖限制了在大预训练模型上的适配性。
- **Flow Mapping / MeanFlow / IMM / Shortcut**：通过重设计捷径目标（平均速度、多时间步自一致）求稳，FACM 论证它们"过度耦合、稀释流监督"是没治本。
- **Flow Matching（Lipman et al.）**：被 FACM 重新当成稳定锚使用，是整个方法的理论支点。
- **启发**：当一个自指/自蒸馏目标训练不稳时，与其加各种正则去压梯度，不如**显式补回它隐含依赖但没被直接监督的那个量**——这里是瞬时速度场。把"锚任务"和"加速任务"解耦到可分的条件域，是个能迁移到其他自一致/自蒸馏训练（如某些 RL、自洽推理蒸馏）的通用配方。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 从"目标自指丢锚"角度重新诊断连续 CM 不稳定，并用 FM 锚 + 扩展时间区间这一极简且架构无关的方案根治，洞察和解法都很有原创性。
- **实验充分度**: ⭐⭐⭐⭐ 多骨干、多数据集、稳定化策略对比、组件消融、权重鲁棒性、教师质量敏感性都覆盖，且扩到 14B 文生图；扣分在指标偏单一（FID 为主）、大规模端缺系统定量。
- **写作质量**: ⭐⭐⭐⭐⭐ 推导清晰、动机层层递进，把"为什么不稳"和"为什么这样修"讲得很透，图表对照到位。
- **价值**: ⭐⭐⭐⭐⭐ 同时拿下 ImageNet 少步 SOTA（NFE=1/2 FID 1.70/1.32）和可扩展到 14B 的工程实现，理论与实用兼具，对少步生成社区影响大。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] CMT: Mid-Training for Efficient Learning of Consistency, Mean Flow, and Flow Map Models](cmt_mid-training_for_efficient_learning_of_consistency_mean_flow_and_flow_map_mo.md)
- [\[ICLR 2026\] Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency](large_scale_diffusion_distillation_via_score-regularized_continuous-time_consist.md)
- [\[ICLR 2026\] The Intricate Dance of Prompt Complexity, Quality, Diversity, and Consistency in T2I Models](the_intricate_dance_of_prompt_complexity_quality_diversity_and_consistency_in_t2.md)
- [\[ICLR 2026\] TempFlow-GRPO: When Timing Matters for GRPO in Flow Models](tempflow-grpo_when_timing_matters_for_grpo_in_flow_models.md)
- [\[ICLR 2026\] UniEdit-Flow: Unleashing Inversion and Editing in the Era of Flow Models](uniedit-flow_unleashing_inversion_and_editing_in_the_era_of_flow_models.md)

</div>

<!-- RELATED:END -->
