---
title: >-
  [论文解读] Beyond Uniformity: Regularizing Implicit Neural Representations through a Lipschitz Lens
description: >-
  [ICLR 2026][Implicit Neural Representation] 把 INR 的 Lipschitz 正则化从"刚性的统一 1-Lipschitz 约束"重构为"可估计、可非均匀分配的 Lipschitz 预算"框架，用任务先验导出全局预算 $K$ 并按层智能分配，从而在平滑性与表达力之间取得更好平衡。
tags:
  - "ICLR 2026"
  - "Implicit Neural Representation"
  - "Lipschitz 正则化"
  - "谱归一化"
  - "可变形配准"
  - "图像修复"
---

# Beyond Uniformity: Regularizing Implicit Neural Representations through a Lipschitz Lens

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=REEdaR0zqj](https://openreview.net/forum?id=REEdaR0zqj)  
**代码**: [https://lipschitz-inrs.github.io](https://lipschitz-inrs.github.io)  
**领域**: 隐式神经表示 / 谱正则化 / 逆问题  
**关键词**: Implicit Neural Representation, Lipschitz 正则化, 谱归一化, 可变形配准, 图像修复  

## 一句话总结
把 INR 的 Lipschitz 正则化从"刚性的统一 1-Lipschitz 约束"重构为"可估计、可非均匀分配的 Lipschitz 预算"框架，用任务先验导出全局预算 $K$ 并按层智能分配，从而在平滑性与表达力之间取得更好平衡。

## 研究背景与动机
**领域现状**：隐式神经表示（INR）把信号建模成坐标到值的连续函数，已广泛用于压缩、新视角合成以及配准、MRI 重建等逆问题。但 INR 缺乏内在正则化，表达力与平滑性之间存在固有矛盾——网络越能拟合高频细节，就越容易过拟合、产生不光滑的解。

**现有痛点**：一种有原则的隐式正则化手段是 Lipschitz 连续性：约束网络的 Lipschitz 常数即可限制其对输入扰动的敏感度。但现有方法几乎都把网络强行约束为 **1-Lipschitz**，并且把这个"1"的预算**均匀地**摊到每一层、每个激活、每个嵌入上。这带来两个被长期回避的开放问题：(1) 对一个具体任务，到底该用多大的总 Lipschitz 预算 $K$？(2) 这个预算应该如何在网络各组件间分配？

**核心矛盾**：统一的 1-Lipschitz 约束既缺乏任务针对性（不同任务对平滑度的需求天差地别），又缺乏结构灵活性（首层做特征提取本该更"放得开"，末层本该收得更紧），结果是表达力被不必要地压死。

**本文目标**：把 Lipschitz 正则化重新表述为一个**灵活的 Lipschitz 预算框架**，回答上面两个问题——既给出从任务先验/数据/信号理论导出 $K$ 的方法，又给出把 $K$ 非均匀分配到各组件的策略。

**核心 idea**：**预算化 + 非均匀分配**——利用 Lipschitz 的乘积合成性质 $K \le \prod_i \mathrm{Lip}(\phi_i)\mathrm{Lip}(W_i)$，把"全局预算 $K$"看作可在对数空间自由切分的总和，再用可解释的领域知识或信号先验确定 $K$ 的大小。

## 方法详解

### 整体框架
方法分三步：先借助 Lipschitz 的合成公式把网络整体 Lipschitz 拆成各层（线性权重、激活、坐标嵌入）的乘积；再用任务先验导出一个有意义的全局预算 $K$（医学配准用组织可压缩性、修复用信号带宽 oracle）；最后把 $K$ 在对数空间按某种策略（均匀或四种非均匀）分配到各组件，并用谱归一化/Björck 正交化把各层的 Lipschitz 钉到分配值。

```mermaid
flowchart LR
    A[各组件 Lipschitz 推导<br/>线性层/激活/嵌入] --> B[导出全局预算 K<br/>领域/数据/信号先验]
    B --> C[预算分配策略<br/>均匀 vs 非均匀]
    C --> D[谱约束实现<br/>谱归一化/Björck/SLL]
    D --> E[逆问题求解<br/>SDF/配准/修复]
```

### 关键设计

**1. 组件级 Lipschitz 推导：把"1-Lipschitz"拆成可计费的零件**
框架的地基是给 INR 的每一类组件都算出闭式 Lipschitz 常数，这样总预算才能被精确拆分。线性层的 Lipschitz 由权重谱范数（最大奇异值）给出 $\mathrm{Lip}(W_i)=\sigma_{\max}(W_i)$；坐标嵌入也有解析值，例如位置编码 $\mathrm{Lip}(\gamma_p)=\pi\sqrt{(4^L-1)/3}$、随机傅里叶特征 $\mathrm{Lip}(\gamma_f)=2\pi\sqrt{\lambda_{\max}(\sum_j b_j b_j^\top)}$；激活则分两类——ReLU、GroupSort、MaxMin 等天然 1-Lipschitz，而 $\sin(\omega x)$ 的 Lipschitz 是 $\omega$、Gaussian 激活 $e^{-x^2/2a^2}$ 的 Lipschitz 是 $1/(a\sqrt{e})$，依超参而变。把这些零件标好"价格"，乘积合成公式 $K=\mathrm{Lip}(f_\theta)\le \prod_{i=1}^{L}\mathrm{Lip}(\phi_i)\mathrm{Lip}(W_i)$ 就成了可操作的预算账本。

**2. 任务驱动的预算估计：让 $K$ 来自可解释的物理/信号先验**
不同于盲选 $K=1$，本文给出三条估计 $K$ 的路线，核心是把领域知识翻译成 Lipschitz 上界。**领域驱动**：在肺部可变形配准里，临床证据表明组织应变接近 2.0 是失效阈值，于是直接令 $K_B=2$，从而保证位移场的局部拉伸/压缩在解剖学上可信、不折叠不撕裂；CT 重建可用相邻体素的最大合理梯度（空气到组织的 Hounsfield 跳变）来定 $K$。**数据驱动**：当有代表性参考图（如修复用的 CelebA）时，用一个基于 L2 范数梯度的 oracle 估计信号的局部变化，给出 Lipschitz 的上下界。**信号理论驱动**：缺先验时退而用带宽/采样率（音频 44.1 kHz、心电 ≈150 Hz）算出保守上界。三条路线都把"该有多平滑"这件事从拍脑袋变成可解释的量。

**3. 非均匀预算分配：在对数空间重新切分 $K$**
给定总预算 $K_B$，把分配问题写成 $\prod_{i=1}^M K_i = K_B$（等价于对数空间求和约束 $\sum_i \log K_i = \log K_B$），本文比较了五种策略：(A) **均匀**，$K_i=\sqrt[M]{K_B}$，每个组件等额；(B) **全给首层**，$K_1=K_B$、其余为 1，呼应"首层权重对网络敏感度影响最大"的观察；以及三种单调递减的参数化策略——(C) **线性** 在预算空间从 $s_0$ 直线降到末层 $K_{\min}$，用二分/牛顿法解出 $s_0$；(D) **指数** 在对数空间做前重型斜坡 $u_i=\log K_{\min}+\frac{\log K_B-M\log K_{\min}}{\sum_j(1-t_j)}(1-t_i)$；(E) **余弦退火** 取 $K_i=K_{\min}(1+\alpha g(t_i))$，$g(t_i)=\tfrac{1+\cos(\pi t_i)}{2}$，由单调方程解出唯一振幅 $\alpha$。这些非均匀策略让早层"放得开"以学更丰富的特征、末层"收得紧"以保平滑，在固定全局预算下重新分配表达力。

**4. 实现层面的谱约束工具箱**
要把分配值真正钉到各层，本文系统对比了谱约束的实现方式：标准谱归一化（幂迭代估最大奇异值）较松，而 Björck 正交化（迭代逼近 $W^\top W=I$）和 SLL（Araujo et al. 2023）能给出更紧的 Lipschitz 上界、更充分地"用满"预算；激活上则用梯度范数保持型（MaxMin、Householder）替代 ReLU 来逼近网络的 Lipschitz 容量上界。一个关键观察是：网络越接近用满它被分配的 Lipschitz 预算（用经验估计的 Lipschitz 常数 $K_m$ 度量），感知质量越好。

## 实验关键数据

### 主实验：三类任务

| 任务 | 数据/设置 | 关键结论 |
|------|-----------|----------|
| 1-Lipschitz SDF | Stanford bunny，Chamfer 距离 | Björck/SLL 比标准谱归一化更锐利；MaxMin/Householder 激活优于 ReLU；越用满预算质量越好 |
| 可变形配准 | 肺 CT (Castillo)，$K_B=2$，TRE / 折叠率 | 非均匀（如指数）分配在保持折叠率可比的同时降低 TRE；推荐 spectral ReLU FFN 与 Björck/SLL 的 SIREN 为稳定配置 |
| 图像修复 | CelebA，FFN + SLL | 非均匀分配带来统计显著提升，性能在 oracle 估计预算附近达到峰值 |

### 消融实验

| 消融维度 | 现象 |
|----------|------|
| 1-Lipschitz 下的分配策略 (SDF) | 非均匀相对均匀提升微乎其微——单位预算太紧，掐死了非均匀的收益 |
| 预算偏离 oracle (修复) | 预算偏离 oracle 估计时性能下降，高预算（FFN）尤其明显，证明 oracle 给出了有意义的上界近似 |
| 不同架构的自调节 (修复) | SIREN/FFN/Gauss 偏离 oracle 时性能衰减曲线陡峭程度不同，反映各自不同的"诱导 Lipschitz 调节偏置" |
| 归一化×架构稳定性 (配准) | FFN+谱归一化、SIREN+标准谱归一化会训练不稳；Björck/SLL 更稳 |

### 关键发现
- **单位预算是天花板**：在 1-Lipschitz（$K=1$）下非均匀分配几乎无收益，必须放开到 $K$-Lipschitz 才能让分配策略发挥作用。
- **存在最优预算且可估计**：修复任务性能在 oracle 预算附近达峰，偏离则下降——说明"该用多大 $K$"是一个有真实最优解、且可被先验估计的量。
- **非均匀分配是可控旋钮**：在配准中，分配策略提供了在 TRE（表达力）与折叠率（平滑性）之间权衡的连续控制手段。

## 亮点与洞察
- **范式转换的清晰度**：把"1-Lipschitz"这个被默认接受的刚性约束，重新解读为"预算 + 分配"两个可调维度，一下子把两个被长期回避的开放问题（多大 $K$、怎么分）摆上台面并给出可操作答案。
- **可解释性是真卖点**：$K$ 不再是炼丹超参，而是组织应变阈值、信号带宽这类有物理/信号意义的量，让正则化强度变得可解释、可迁移。
- **统一视角的附加价值**：作者还指出"权重缩放初始化"（Yeom et al. 2024）这类经验有效技巧，本质就是在缩放各线性层的 Lipschitz 上界，从而为 Fourier 分析和 NTK 理论提供了一个互补的 Lipschitz 解释视角。

## 局限与展望
- **分配仍是超参搜索**：论文坦承全局预算如何最优分配仍是开放问题，实际操作仍建议把分配策略当超参做网格搜索，并未给出闭式最优解。
- **预算估计依赖先验**：领域驱动估计需要可靠的物理先验（如组织应变阈值），数据驱动需代表性参考样本，缺二者时只能退回保守的信号理论估计，可能偏松。
- **任务覆盖有限**：实验集中在 SDF、配准、修复（超分在附录），是否能推广到 NeRF、大规模生成等更复杂 INR 场景尚待验证。
- **训练稳定性敏感**：某些归一化×架构组合（如 FFN+谱归一化）会训练不稳，框架对实现选择较敏感。

## 相关工作与启发
- **谱约束 / 1-Lipschitz 网络**：Coiffier & Béthune 的 1-Lipschitz 神经距离场、Miyato 的谱归一化、Björck 正交化、Araujo 的 SLL 层是本文的工具基础；本文的贡献是把它们从"统一 1-Lipschitz"推广到"任务化 $K$ + 非均匀分配"。
- **INR 表达力-平滑性权衡**：Ramasinghe 等指出 INR 缺乏隐式正则化，本文正是从 Lipschitz 角度系统补上这块。
- **与 NTK / Fourier 分析互补**：本文主张 Lipschitz lens 可以解释权重缩放等经验技巧的有效性，为现有 INR 理论框架提供第三种互补视角，这一点对想统一理解 INR 行为的研究者颇有启发。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把刚性 1-Lipschitz 重构为可估计、可非均匀分配的预算框架，并把抽象的正则化强度对接到可解释的物理/信号先验，视角新颖。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 SDF、医学配准、图像修复三类任务，含分配策略、预算偏离、架构自调节等多维消融与显著性检验。
- 写作质量: ⭐⭐⭐⭐ 理论推导（组件 Lipschitz、五种分配策略）严谨，并提炼出面向从业者的实操指南，结构清晰。
- 价值: ⭐⭐⭐⭐ 为 INR 逆问题提供了可操作、可解释的正则化设计原则，对配准/修复等应用有直接指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] On the Lipschitz Continuity of Set Aggregation Functions and Neural Networks for Sets](on_the_lipschitz_continuity_of_set_aggregation_functions_and_neural_networks_for.md)
- [\[ICLR 2026\] Refine Now, Query Fast: A Decoupled Refinement Paradigm for Implicit Neural Fields](refine_now_query_fast_a_decoupled_refinement_paradigm_for_implicit_neural_fields.md)
- [\[ICLR 2026\] Exploring State-Space Models for Data-Specific Neural Representations](exploring_state-space_models_for_data-specific_neural_representations.md)
- [\[NeurIPS 2025\] Improving Decision Trees through the Lens of Parameterized Local Search](../../NeurIPS2025/others/improving_decision_trees_through_the_lens_of_parameterized_local_search.md)
- [\[ACL 2025\] Implicit Reasoning in Transformers is Reasoning through Shortcuts](../../ACL2025/others/implicit_reasoning_in_transformers_is_reasoning_through_shortcuts.md)

</div>

<!-- RELATED:END -->
