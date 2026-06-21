---
title: >-
  [论文解读] Beyond Outliers: A Study of Optimizers Under Quantization
description: >-
  [ICLR2026][模型压缩][量化] 作者第一次系统地把"优化器选择"和"量化鲁棒性"放到一起研究：在 50M–1.5B 的 LLM 上用 6 种优化器训练，发现传统的离群值指标（MMR、Kurtosis）根本预测不了量化后的精度，转而提出一套可解析的 ABC 误差传播分解和新指标 $R_L$，并得出反直觉结论——离群值最严重的 Shampoo 反而在 PTQ/QAT 下掉点最少、参数效率最高。
tags:
  - "ICLR2026"
  - "模型压缩"
  - "量化"
  - "优化器"
  - "误差传播"
  - "离群值"
  - "缩放律"
---

# Beyond Outliers: A Study of Optimizers Under Quantization

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=mVldAuDAn5](https://openreview.net/forum?id=mVldAuDAn5)  
**代码**: 待确认  
**领域**: 模型压缩  
**关键词**: 量化, 优化器, 误差传播, 离群值, 缩放律

## 一句话总结
作者第一次系统地把"优化器选择"和"量化鲁棒性"放到一起研究：在 50M–1.5B 的 LLM 上用 6 种优化器训练，发现传统的离群值指标（MMR、Kurtosis）根本预测不了量化后的精度，转而提出一套可解析的 ABC 误差传播分解和新指标 $R_L$，并得出反直觉结论——离群值最严重的 Shampoo 反而在 PTQ/QAT 下掉点最少、参数效率最高。

## 研究背景与动机
**领域现状**：新优化器（Muon、Shampoo、SOAP、PSGD、Scion）层出不穷，量化（PTQ 与 QAT）也成了大模型部署的标配。但这两条线几乎是各管各的——优化器 benchmark（如 Wen et al. 2025、Semenov et al. 2025）一律在全精度下比，量化研究则默认用 AdamW 训练，没人系统问过"换个优化器，模型抗量化的能力会不会变"。

**现有痛点**：联合量化（权重和激活都压到低比特）之所以难，公认的罪魁是输入矩阵里的**离群特征（outlier features）**。社区因此发明了一堆指标来量化离群程度——最大-中位数比 MMR、峰度 Kurtosis——并据此做旋转、架构改造等"去离群"手段。但这些指标全是"静态地看某一层有多少离群值"，没人验证过：训练用的优化器到底怎么影响这些离群值，以及离群值多是不是真的等于量化后掉点多。

**核心矛盾**：folklore（口口相传的经验）认为"离群值越大 → 量化越崩"，可这条经验从没在"不同优化器"这个维度上被检验过。一旦检验，作者发现它直接崩了：离群值最严重的优化器反而最抗量化。这说明用单层离群值预测全网量化误差，从根上就漏掉了**误差如何在层间累积、放大、传播**这件事。

**本文目标**：拆成三个递进的子问题——(1) 同样验证 loss、不同优化器训出的模型，PTQ 后表现一样吗？(2) QAT 对优化器选择有多敏感，全精度最好的优化器在量化下还最好吗？(3) 这些结论能不能外推到更大模型？

**切入角度**：与其继续盯着"单层有多少离群值"，不如把量化误差当成一个**在前向传播中逐层演化的信号**，去解析地追踪它怎么从第 1 层传到第 $L$ 层。这个视角天然能解释为什么单层指标失效。

**核心 idea**：用一套可证明的 ABC 误差分解（把每层量化误差拆成"累积误差 + 本层误差 + 二者交互"）代替静态离群值指标，得到一个真正能预测量化掉点的新指标 $R_L$，并用它揭示不同优化器在误差传播上的本质差异。

## 方法详解

### 整体框架
这篇是分析/研究向的工作，"方法"不是提出一个新模型，而是搭一套**统一的实验台 + 一套误差传播的理论工具**，再用工具去解释实验里的反直觉现象。整条研究链是：先在全精度下把 6 个优化器都调到各自最优（建立公平 baseline）→ 对训到同一 loss 的模型做 PTQ，观察哪个优化器最抗压、顺便检验离群值指标灵不灵 → 当指标失效时，引入 ABC 分解与新指标 $R_L$ 从理论上解释为什么 → 再做 QAT 并拟合缩放律，确认结论能外推到大模型。

实验底座统一为 OLMo2 架构（RoPE、RMSNorm、QKNorm、重排 pre-norm，外加输入输出权重绑定和 ReLU² 激活），覆盖 50M / 125M / 350M / 500M / 760M / 1.5B 六个尺寸，全部在 ClimbMix（400B token 高质量混合数据）上按 Chinchilla 最优配比训练。PTQ 用 4-bit 对称 AbsMax 逐行 round-to-nearest（W4A4，只压线性层）；QAT 用 SOTA 的 QuEST（前向对输入和权重做 Hadamard 变换 + 最优裁剪比，反向保持高精度并掩蔽超出裁剪值的梯度）。

### 关键设计

**1. 公平的全精度 baseline 协议：先把每个优化器各自调到最优，再谈量化**

要比较"优化器对量化的影响"，前提是不能让某个优化器因为没调好而背锅。作者设计了两段式调参：第一步在最小的 50M 模型上，对每个优化器做**逐维一维 sweep**（从原论文推荐超参出发，一个超参一个超参地扫），定下来的配置直接套到所有尺寸；第二步固定优化器超参后，对每个"模型×优化器"组合扫 8 个学习率并完整训练，挑最优。1.5B 因为太贵，只跑 AdamW（baseline）、Muon（全精度最强）、Shampoo（最抗量化）三个代表。

为了让 PTQ 比较有意义，作者引入 **Common Loss（CL）**：在不超过 20× token/参数比的预算内，所有优化器都能达到的最低验证 loss。所有模型都训到这个 CL 才做 PTQ——这样量化前大家起点一样，量化后的精度差就纯粹反映"抗量化能力"而非"训练好坏"。全精度结果是 Muon 几乎在所有尺寸都最强，且 Muon 的 MMR 也最低，看起来完全符合"低离群=好"的直觉，为后面的反转埋下伏笔。

**2. ABC 误差分解：把每层的相对量化误差拆成累积、本层、交互三项**

这是全文的理论核心，目的是解释"为什么单层离群值指标预测不了量化掉点"。把网络看成 $L$ 个模块 $f_\ell(\cdot)$ 的串联，激活 $h_\ell = f_\ell(h_{\ell-1})$；量化后激活变成 $h^q_\ell = f^q_\ell(h^q_{\ell-1})$。注意量化带来**两重变化**：输入从 $h_{\ell-1}$ 变成 $h^q_{\ell-1}$（前面层误差传过来的），函数从 $f_\ell$ 变成 $f^q_\ell$（本层新引入的扰动）。作者证明激活差 $\Delta h_\ell := h^q_\ell - h_\ell$ 可严格写成 $\Delta h_\ell = a_\ell + b_\ell$，其中 $a_\ell$ 是"输入变化"的平均（前层误差），$b_\ell$ 是"函数变化"的平均（本层误差）。

为了把它压成一个可解释的标量，取相对 L2 范数并平方，借助余弦定理得到精确分解：

$$R_\ell := \left(\frac{\|\Delta h_\ell\|}{\|h_\ell\|}\right)^2 = A_\ell + B_\ell + C_\ell,\quad A_\ell = \left(\frac{\|a_\ell\|}{\|h_\ell\|}\right)^2,\ B_\ell = \left(\frac{\|b_\ell\|}{\|h_\ell\|}\right)^2,\ C_\ell = \frac{2\langle a_\ell, b_\ell\rangle}{\|h_\ell\|^2}$$

$A_\ell$ 是从前 $\ell-1$ 层**累积放大**过来的误差，$B_\ell$ 是本层**新引入**的误差（这一项才对应传统离群值指标想刻画的东西），$C_\ell$ 是两者的交互项。实测发现：绝大多数情况下 $A_\ell$ 远大于 $B_\ell$ 和 $C_\ell$。这就一句话解释了反直觉现象——**即便 MMR 能很好预测某一层"本层"引入的误差 $B_\ell$，全网总误差 $R_\ell$ 主要由累积误差 $A_\ell$ 决定**，所以盯着单层离群值（≈ $B_\ell$）当然预测不了最终掉点。图中那些规律性的"凹陷"对应 RMSNorm 层，它们会衰减而非放大量化误差。

**3. 新指标 $R_L$ 与增益分解：用网络末端的相对误差预测量化掉点，并归因到优化器**

既然 $R_\ell$ 追踪的是量化网络相对原网络的逐层偏离，那么**网络输出端的 $R_L$** 自然应该和 loss 退化强相关。实测 $R_L$ 与 PTQ 后平均零样本精度的相关系数 $\rho = -0.89$，远超 MMR（$\rho=0.62$）和 Kurtosis（$\rho=0.70$）。代价是 $R_L$ 需要四次前向（$f_L(h_{L-1})$、$f_L(h^q_{L-1})$、$f^q_L(h_{L-1})$、$f^q_L(h^q_{L-1})$ 各一次），比 MMR 贵，但准得多。

更进一步，为了看一个模块如何把前层误差 $R_{\ell-1}$ 转成累积误差 $A_\ell$，定义**增益** $G_\ell := A_\ell / R_{\ell-1}$。对线性层（$h^q_\ell = (W_\ell + \varepsilon^W_\ell)h^q_{\ell-1} + \varepsilon^h_\ell$），增益可进一步分解为 $G_\ell = G_{1,\ell}\,G_{2,\ell}$：**谱比** $G_{1,\ell} = \left(\|W_\ell + \tfrac12\varepsilon^W_\ell\|_* / \|W_\ell\|_*\right)^2$ 衡量量化对权重谱范数的改变，**对齐比** $G_{2,\ell} = (\cos\phi_\ell / \cos\psi_\ell)^2$ 衡量"激活变化与量化后权重的对齐程度"相对"原激活与原权重对齐程度"的比值。分析发现谱比对所有优化器都接近 1，所以增益几乎全由对齐比决定。这套工具一举把"为什么 Muon 抗量化差"归因到机制层面：**Muon 的线性层增益最高**（误差放大最猛），所以尽管它 MMR 低，量化后照样崩；而 AdamW、Shampoo 的增益最低，激活变化与量化权重更不对齐，误差传不大。

### 损失函数 / 训练策略
QAT 用 QuEST 做 4-bit 前向量化，反向用 Straight-Through Estimator 估计 round 的梯度并保持高精度。每个"模型×优化器"对在相同算力预算（Chinchilla 最优、固定 20× token/参数比）下用各自最优的全精度超参训练。缩放律沿用 Hoffmann et al. (2022) 的形式，并引入 Kumar et al. (2024) 的**逐优化器参数效率** $\rho$ 作为额外超参（全精度 $\rho=1$，$\rho_{4bit}$ 表示 4-bit QAT 的参数效率），固定 $D/N=20$ 后拟合等算力形式：

$$L = \frac{A'}{(N\cdot\rho)^\alpha} + E$$

直观含义是：4-bit QAT 训练的 $N$ 参数模型，性能等价于全精度训练的 $\rho_{4bit}\cdot N$ 参数模型——$\rho$ 越高越抗量化。

## 实验关键数据

### 主实验

PTQ（W4A4）后的平均零样本精度（PIQA + HellaSwag + ARC-Easy），所有模型量化前都训到同一 CL，故量化后精度即抗压能力。AdamW vs 全精度最强的 Muon vs 最抗量化的 Shampoo：

| 模型 | AdamW | Muon | Shampoo | 现象 |
|------|-------|------|---------|------|
| 350M | 49.23 | 47.42 | **53.93** | Shampoo 领先 Muon 6.5 点 |
| 500M | 55.17 | 50.60 | **55.65** | Muon 严重掉队 |
| 760M | 59.22 | 50.00 | **59.26** | Muon 比全精度掉了 ~14.6 点 |
| 1.5B | 62.51 | 47.75 | **63.88** | Muon 崩到 47.75 |

反直觉点：Shampoo 的 MMR（离群值）最高，按 folklore 应该掉最惨，实际却最抗量化；Muon 的 MMR 最低，却掉点最猛。

### 消融实验

不同离群/误差指标与 PTQ 后精度的相关性（760M），直接说明新指标的价值：

| 指标 | 与精度相关系数 $\rho$ | 是否能预测量化掉点 |
|------|----------------------|--------------------|
| MMR（最大-中位数比） | 0.62 | 弱，几乎无关 |
| Kurtosis（峰度） | 0.70 | 弱 |
| **$R_L$（本文 ABC 末端误差）** | **−0.89** | 强相关 |

QAT（4-bit QuEST）相对全精度 baseline 的精度退化（括号内为掉点，越小越好）：

| 模型 | AdamW | Muon | Shampoo |
|------|-------|------|---------|
| 760M | 62.22 (−2.63) | 62.32 (−3.57) | 62.76 (**−0.46**) |
| 1.5B | 66.82 (−1.63) | 67.08 (−2.11) | 67.34 (**−1.20**) |

缩放律参数效率 $\rho_{4bit}$：Shampoo 最高（0.879），其次 AdamW（0.863）、Scion（0.856）、Muon（0.852），PSGD 最低（0.739）。PTQ 恢复最好的几个优化器恰好也是参数效率最高的几个。

### 关键发现
- **离群值范式失效**：MMR/Kurtosis 这类单层离群指标无法预测不同优化器下的 PTQ 表现，因为它们只刻画本层误差 $B_\ell$，而总误差由累积误差 $A_\ell$ 主导。
- **误差传播形态因优化器而异**：AdamW、Scion 训出的网络 $R_\ell$ 在末端有明显尖峰，PSGD、Muon 则更"平"；Muon 线性层增益最高，解释了它低离群却高掉点。
- **全精度排名不能预测量化排名**：Muon 全精度最强，量化下却最差；Shampoo 全精度平平，却在 PTQ、QAT、缩放律三处都最抗量化、参数效率最高。

## 亮点与洞察
- **把"经验法则"做成可证伪的科学**：作者没停在"换优化器有影响"，而是给出 $\Delta h_\ell = a_\ell + b_\ell$ 的严格分解，用余弦定理推出 $R_\ell = A_\ell+B_\ell+C_\ell$，让"为什么离群值指标失效"有了精确数学解释——这套 ABC 框架本身是可复用的诊断工具，可移植到任意 PTQ 方案、任意模块去做误差归因。
- **$R_L$ 是个实用的量化友好度探针**：四次前向就能算，相关性从 0.6 量级跳到 0.89，比起 MMR 是质变。任何做 PTQ 选模型/选优化器的工程，都可以用它在量化前预判哪个 checkpoint 更抗压。
- **最大的"啊哈"是反直觉的 Shampoo**：离群值最严重的优化器反而最抗量化，直接颠覆了社区多年"去离群=保精度"的主流假设，提示研究重心应从"压离群值"转向"控制误差传播增益 $G_\ell$"。
- **增益视角指出了新优化器方向**：既然 PSGD/Muon 内部本就在解约束优化问题，作者建议把"$G_\ell$ 要小"作为额外约束塞进去，有望直接训出"量化友好"的优化器——这是个具体可落地的迁移思路。

## 局限与展望
- **只测了 4-bit 一个位宽**：作者自己承认还没覆盖 8-bit、6-bit，以及 micro-scaling 等其他 4-bit 数据格式，结论在其他位宽下是否成立未知。
- **增益分解只对线性层完整推导**：$G_\ell = G_{1,\ell}G_{2,\ell}$ 这套漂亮的谱比×对齐比分解只适用于线性层，self-attention（MHSA）整体当作单个单元处理，没有展开，限制了对注意力模块误差行为的理解。
- **PTQ 方案较单一**：主实验是 RTN，虽附录给了 GPTQ、Llama 2 的类似结果，但更多 SOTA PTQ 方案下 ABC 分解的形态仍待验证。
- **优化器子集有限**：六个优化器里 PSGD/Scion/SOAP 在 1.5B 上是 N/A（没训），最大规模的结论实际只建立在 AdamW/Muon/Shampoo 三者上，外推到全部优化器需谨慎。
- **改进思路**：把 $R_L$ 直接作为训练期正则项、或把 $G_\ell$ 约束嵌进优化器更新规则，去主动训练抗量化模型，是作者点出但尚未实现的方向。

## 相关工作与启发
- **vs 离群值/去离群方法（QuaRot、SmoothQuant、KurTail 等）**：它们都在"压低离群值（MMR/Kurtosis）"上下功夫，假设离群值是量化崩坏的根因；本文用 ABC 分解证明单层离群值只对应可忽略的 $B_\ell$，真正主导的是累积误差 $A_\ell$，因此"去离群"未必直接保精度，视角更本质但需要更贵的 $R_L$ 计算。
- **vs 优化器 benchmark（Wen et al. 2025、Semenov et al. 2025）**：它们系统比较了优化器，但全在全精度下，结论是 Muon 类方法更优；本文加上量化维度后发现全精度排名会反转，补上了"优化器×量化交互"这块此前没人碰的空白。
- **vs QAT 缩放律工作（Kumar et al. 2024）**：本文沿用其参数效率 $\rho$ 概念，但首次把 $\rho$ 做成**逐优化器**的量，揭示 Shampoo 在 4-bit 下参数效率最高，把缩放律从"量化整体掉多少"细化到"哪个优化器更抗量化"。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 第一个系统研究优化器×量化交互，并用可证明的 ABC 分解颠覆离群值范式
- 实验充分度: ⭐⭐⭐⭐ 覆盖 6 优化器×6 尺寸×PTQ/QAT，但 1.5B 仅 3 个优化器、位宽只测 4-bit
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导与实验现象一一对应，反直觉结论有机制解释，逻辑闭环
- 价值: ⭐⭐⭐⭐⭐ 既给出实用预测指标 $R_L$，又指明"量化友好优化器"这一新方向，对部署侧影响大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] A Granular Study of Safety Pretraining under Model Abliteration](../../NeurIPS2025/model_compression/a_granular_study_of_safety_pretraining_under_model_abliteration.md)
- [\[CVPR 2026\] TWEO: Transformers Without Extreme Outliers Enables FP8 Training And Quantization For Dummies](../../CVPR2026/model_compression/tweo_transformers_without_extreme_outliers_enables_fp8_training_and_quantization.md)
- [\[ICLR 2026\] Study of Training Dynamics for Memory-Constrained Fine-Tuning (TraDy)](study_of_training_dynamics_for_memory-constrained_fine-tuning.md)
- [\[ICLR 2026\] AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in LVLMs](agilepruner_an_empirical_study_of_attention_and_diversity_for_adaptive_visual_to.md)
- [\[ICLR 2026\] Beyond Student: An Asymmetric Network for Neural Network Inheritance](beyond_student_an_asymmetric_network_for_neural_network_inheritance.md)

</div>

<!-- RELATED:END -->
