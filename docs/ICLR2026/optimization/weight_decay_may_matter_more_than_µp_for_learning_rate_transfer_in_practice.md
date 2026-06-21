---
title: >-
  [论文解读] Weight Decay May Matter More Than µP for Learning Rate Transfer in Practice
description: >-
  [ICLR 2026][优化/理论][µP] 这篇论文用一个"相对更新"统一框架重新审视大模型训练里的学习率迁移，发现 µP 赖以成立的对齐假设在实际训练中很快失效，真正在大部分训练时间里稳住跨宽度特征学习、让学习率得以迁移的其实是**独立形式的 weight decay**；而 µP 的学习率缩放实质只起到了一个"隐式学习率 warmup"的作用，可以用更强的显式 warmup 大体替代。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "µP"
  - "学习率迁移"
  - "weight decay"
  - "AdamW"
  - "特征学习"
---

# Weight Decay May Matter More Than µP for Learning Rate Transfer in Practice

**会议**: ICLR 2026  
**论文**: Published as a conference paper at ICLR 2026  
**代码**: 无（缓存未提供）  
**领域**: 优化 / 大模型预训练  
**关键词**: µP, 学习率迁移, weight decay, AdamW, 特征学习

## 一句话总结
这篇论文用一个"相对更新"统一框架重新审视大模型训练里的学习率迁移，发现 µP 赖以成立的对齐假设在实际训练中很快失效，真正在大部分训练时间里稳住跨宽度特征学习、让学习率得以迁移的其实是**独立形式的 weight decay**；而 µP 的学习率缩放实质只起到了一个"隐式学习率 warmup"的作用，可以用更强的显式 warmup 大体替代。

## 研究背景与动机
**领域现状**：要在超大规模模型（尤其 LLM）上训练，逐个调超参代价高得吓人。Maximal Update Parameterization（µP，最大更新参数化）提供了一条捷径——在小模型上调好最优学习率，再按一套缩放规则迁移到大模型上，理论上能让"最优学习率"跨宽度保持不变。µP 已经成为很多开源 LLM 训练配方的基石。

**现有痛点**：但实践中反复出现一个让人困惑的现象（Wortsman 2024、Wang & Aitchison 2025、Bergsma 2025 等都观察到）——µP 只有在搭配 **independent weight decay（独立权重衰减）** 时，学习率才迁移得好（见原文 Figure 1：independent WD 下不同宽度的最优学习率重合，standard WD 下则散开）。这很反直觉：standard weight decay 才是"符合 µP 理论缩放"的那个，independent WD 反而违背 µP，却效果更好。

**核心矛盾**：µP 的缩放规则依赖一组很强的几何假设，核心是层输入 $X$ 与权重 $W$、与梯度更新 $\Delta W$ 之间的**对齐**关系（alignment）。µP 假设权重对齐 $\alpha_W = \Theta(1/\sqrt{C})$、更新对齐 $\alpha_{\Delta W}=\Theta(1)$。问题是：这些假设在真实训练里到底成不成立？如果不成立，那把学习率迁移做对的到底是谁？

**本文目标**：(1) 搭一个能同时刻画 µP 与 weight decay 的统一框架；(2) 实测 µP 的对齐假设在训练全程的演化；(3) 厘清 µP 缩放与 weight decay 各自的真实角色；(4) 验证能否用显式 warmup 替代 µP。

**切入角度**：作者不盯着"绝对更新量"，而改用 **相对更新**（相对权重更新 $\|\Delta W\|/\|W\|$、相对表示变化 $\|\Delta Y\|/\|Y\|$）来度量。原因是神经网络对一类重缩放变换（normalization、可学增益、齐次激活）天然不变，固定的绝对更新量在这些等价状态下影响各异，而固定的相对更新量影响始终一致——相对量更能反映"这一步更新到底有多大影响"，而且它天然把 weight decay 纳入了进来。

**核心 idea**：用"维持跨宽度相对表示变化恒定"这一目标，把 µP 和 weight decay 放进同一杆秤——你会发现 µP 的对齐假设只在训练最初几步成立，之后是 independent weight decay 在替它兜底，而 µP 缩放剩下的唯一实际价值是"隐式 warmup"。

## 方法详解
这是一篇分析 + 大规模实证的论文，没有可画 pipeline 的系统结构，核心是一条推理链：先用相对更新框架统一两者，再实测对齐假设失效，再说明 independent WD 如何接管，最后把 µP 归结为 warmup 并验证可替代。下面用文字 + 公式把这条链讲清。

### 整体框架
全篇围绕一个**单层线性变换** $Y = WX$（$X\in\mathbb{R}^{C\times B}$ 输入、$W\in\mathbb{R}^{K\times C}$ 权重、$Y\in\mathbb{R}^{K\times B}$ 输出）展开。一次权重更新 $W\mapsto W+\Delta W$ 会让输出变化 $\Delta Y = \Delta W X$。优化器（如 Adam）容易控制更新的绝对大小 $\|\Delta W\|$，但真正决定一步更新影响的是表示变化 $\|\Delta Y\|$，而后者难控——这正是 µP 学习率缩放想干的事。

把二者联系起来的桥梁是**对齐（alignment）**。定义更新对齐

$$\alpha_{\Delta W} := \frac{\|\Delta Y\|}{\|\Delta W\|\,\|X\|} = \frac{\|\Delta W X\|}{\|\Delta W\|\,\|X\|}\in[0,1]$$

它本质是输入样本 $x_b$ 与更新行 $\Delta w_k$ 之间余弦相似度的加权均方根：对齐越高，同样大小的 $\|\Delta W\|$ 带来越大的 $\|\Delta Y\|$。类似地定义权重对齐 $\alpha_W := \|Y\|/(\|W\|\,\|X\|)$。于是相对表示变化可以拆成：

$$\frac{\|\Delta Y\|}{\|Y\|} = \frac{\alpha_{\Delta W}}{\alpha_W}\cdot\frac{\|\Delta W\|}{\|W\|}$$

其中 $\alpha_{\Delta W}/\alpha_W$ 叫**对齐比（alignment ratio）**。要让 $\|\Delta Y\|/\|Y\|$ 跨宽度恒定，µP 就必须用学习率精确抵消对齐比随宽度的变化。整篇论文的所有结论，都是在这条等式上"看哪一项在变、谁在补偿"。

µP 的理论假设是：初始化时 $\alpha_W\approx 1/\sqrt{C}$、$\alpha_{\Delta W}=\Theta(1)$，于是对齐比 $\propto \sqrt{C}$。当宽度 $C\mapsto mC$，为抵消它，µP 规定相对更新 $\propto 1/\sqrt{m}$，对应 Adam 学习率 $\eta=\eta_{\text{base}}/m$。

### 关键设计

**1. 相对更新统一框架：把 µP 和 weight decay 放进同一杆秤**

µP 的研究历来用"绝对表示变化 $\|\Delta Y\|_{\text{RMS}}=\Theta(1)$"来表述，而 weight decay 的研究历来用"相对更新"。这篇论文的第一步是把 µP 也改写成相对量（上面那条 $\|\Delta Y\|/\|Y\| = (\alpha_{\Delta W}/\alpha_W)\cdot\|\Delta W\|/\|W\|$），从而让两者可比。关键事实是：weight decay 并非主要起正则作用，而是充当"第二学习率"，调制相对更新的稳态大小。Kosson 等给出 AdamW 稳态下权重范数收敛到 $\|W\|\approx\sqrt{KC\,\eta/\lambda}$，配合 AdamW 的更新归一化 $\|\Delta W\|\propto\eta\sqrt{KC}$，得到核心关系：

$$\|W\|\propto\sqrt{KC\,\eta/\lambda},\qquad \frac{\|\Delta W\|}{\|W\|}\propto\sqrt{\eta\lambda}$$

这条式子点破了关键：在稳态下，学习率 $\eta$ 和 weight decay $\lambda$ 对相对更新的影响**完全对称，只有乘积 $\eta\lambda$ 重要**。这就为后面"weight decay 能替代/抵消 µP 缩放"埋下了伏笔——既然只有 $\eta\lambda$ 重要，那调 $\lambda$ 和调 $\eta$ 在稳态下等效。

**2. independent weight decay 最终抵消（neutralize）µP 缩放**

AdamW 有两种 weight decay 写法：PyTorch 默认每步把权重乘 $1-\eta\lambda$，Loshchilov & Hutter 原版乘 $1-\lambda$。当 µP 把学习率缩成 $\eta\mapsto\eta/m$ 时，两种处理 $\lambda$ 的方式后果完全不同：

$$(\eta,\lambda)\mapsto(\eta/m,\ \lambda)\quad\text{(standard scaling)}$$
$$(\eta,\lambda)\mapsto(\eta/m,\ m\lambda)\quad\text{(independent scaling)}$$

independent scaling 让乘积 $\eta\lambda$ 保持不变，于是按设计点 1 的结论，**稳态相对更新完全不被缩放**——这恰恰和 µP 规定的"相对更新 $\propto 1/\sqrt{m}$"相矛盾。换句话说，independent weight decay 会在训练后期**直接覆盖（override）掉 µP 的更新缩放**；而 standard weight decay 才老老实实跟着 µP 走。原文 Figure 3 实测证实：independent WD 下不同宽度的相对表示变化重合，standard WD 下则在后期严重发散。这就解释了那个反直觉现象——"违背 µP"的 independent WD 反而迁移得好，因为 µP 的理论缩放从训练中期起就是错的，需要被抵消。

**3. weight decay 在补 µP 失效的对齐假设的窟窿**

为什么需要抵消？因为 µP 的对齐假设很快崩。原文 Figure 4 实测：对齐比从初始的"依赖宽度"值迅速跌到约 $1$，违反了 µP 的 $\Theta(\sqrt{C})$ 假设。一旦对齐比 $\approx 1$，由相对表示变化等式可知，要让 $\|\Delta Y\|/\|Y\|$ 跨宽度相等，就必须让相对权重更新 $\|\Delta W\|/\|W\|$ 跨宽度相等——这正是 independent weight decay（保持 $\eta\lambda$ 恒定）所做的，而 µP 的 standard 缩放反而把大网络的相对更新压小，导致后期失配。

更新对齐为何会变成依赖宽度？作者在单神经元 SGD 上给出机理（原文 Eq. 9）：输出变化 $\Delta y_i$ 含一个"自贡献项"和 $B-1$ 个"干扰项"，随机对齐让每个干扰项弱 $1/\sqrt{C}$，但 $B-1$ 项随机求和又放大约 $\sqrt{B}$。于是当 **batch size $B\gg C$**（宽度）时干扰项主导，更新对齐重新依赖 $C$，µP 假设破裂。这在实践中极常见：LLaMA 实验里每 batch 的有效"样本数"是 token 总数 1,048,576，远超网络宽度 $C\le 3\times 2048$。这也解释了为何 µP 这类缩放在 Transformer 流行后才被需要——良好归一化的 CNN 在合理规模下可能用不上（见 ResNet 实验）。

**4. 把 µP 重新定性为"隐式学习率 warmup"，并用显式 warmup 替代**

既然实际训练由"对齐比 $\approx 1$"主导、相对更新该跨宽度恒定，那 µP 缩放在后期就是多余甚至有害的，它唯一剩下的好处发生在训练**早期**。independent 缩放 $(\eta,\lambda)\mapsto(\eta/m,m\lambda)$ 用"更低学习率 + 更高 weight decay"达到同一 $\eta\lambda$，使得早期相对更新更小、再随权重范数逼近稳态而逐渐恢复——这正是一个 warmup 的形状（原文 Figure 5，缩放因子 $s_t$ 从 $1/m$ 指数式逼近 $1$）。简化常学习率设定下作者给出闭式（原文 Eq. 11）：

$$s_t = \sqrt{\frac{1+(\rho_0^2/\rho_\infty^2-1)a^{2t}}{1+(m^2\rho_0^2/\rho_\infty^2-1)a^{2t}}}\ \xrightarrow{\ \rho_0=\rho_\infty\ }\ \frac{1}{\sqrt{1+(m^2-1)a^{2t}}}$$

其中 $a:=1-\eta\lambda$ 为每步衰减乘子，$\rho_t$ 为权重 RMS，$\rho_\infty=\sqrt{\eta/(2\lambda)}$ 为稳态预测值。基于这个洞察，作者构造两种乘性 warmup 因子去替代 µP（指数递增型 Eq. 12、decay-away 型 Eq. 13），都把初始学习率压成 $1/m$ 再回到 $1$。原文 Figure 6 显示：在已有 10% 线性 warmup 之上叠加这种额外 warmup，能取得和 µP+independent WD 相近、甚至更稳的学习率迁移——印证 warmup 才是 µP 的主要实际好处。

### 损失函数 / 训练策略
不涉及新损失函数。训练目标是标准的 next-token prediction（LLaMA 在 DCLM 上预训练，宽度 128–2048，约 20B tokens；另有 ResNet/ImageNet 旁证）。核心可操作结论是超参缩放策略：用 independent weight decay（保持 $\eta\lambda$ 恒定），或在不做 µP 缩放时用更强的显式 warmup（指数递增 / decay-away），来稳定跨宽度的相对特征更新。

## 实验关键数据

### 主实验：学习率迁移质量

| 设置 | weight decay 形式 | 学习率迁移 | 说明 |
|------|------|------|------|
| µP（LLaMA 预训练，20B tokens，宽度至 2048）| Independent WD | 好（各宽度最优 LR 重合）| µP 真正有效的唯一组合（Fig.1）|
| µP | Standard WD | 差（长训后期发散）| 跟随 µP 理论缩放反而迁移失败 |
| µP | No WD | 介于两者之间 | 尺度不变权重永久增长，相对更新最终丢掉对峰值 LR 的依赖 |
| 无 µP，10% / 50% 线性 warmup | — | 较差 | 长线性 warmup 也补不齐 µP 的收益（Fig.6 中）|
| 无 µP，额外指数 warmup（Eq.12/13）| — | 好 | 显式 warmup 可大体替代 µP 缩放，高 LR 下略逊（Fig.6 右）|

### 机理分析：对齐假设是否成立

| 测量量（LLaMA，宽度 128 vs 2048）| µP 假设 | 实测结果 |
|------|------|------|
| 权重对齐 $\alpha_W$ | $\Theta(1/\sqrt{C})$ | 仅训练极早期吻合，之后偏离（ResNet 上更明显）|
| 更新对齐 $\alpha_{\Delta W}$ | $\Theta(1)$（与宽度无关）| 随时间显著变化、变得依赖宽度（$B\gg C$ 所致）|
| 对齐比 $\alpha_{\Delta W}/\alpha_W$ | $\Theta(\sqrt{C})$ | 迅速跌到 $\approx 1$，失去宽度依赖（Fig.4）|
| 相对表示变化 $\|\Delta Y\|/\|Y\|$ 跨宽度 | 应恒定 | Independent WD 维持住，Standard WD 后期发散（Fig.3）|

### 关键发现
- **谁在做学习率迁移**：训练绝大部分时间里是 independent weight decay（保持 $\eta\lambda$）而非 µP 缩放，在稳定跨宽度特征学习。µP 的理论缩放只在最初几步对。
- **假设何时破裂**：µP 对齐假设的失效根因是 batch size $B$ 相对宽度 $C$ 太大（$B\gg C$ 时更新对齐重新依赖 $C$），这在 LLM 训练里是常态（有效"样本数"= token 数 $\approx 10^6 \gg C$）。
- **µP 的真实价值**：等价于一个 $1/m$ 起步、指数式恢复到 $1$ 的隐式 warmup；对长训练而言这是 µP 学习率缩放剩下的唯一实际好处。
- **可替代性**：显式指数 warmup 能大体替代 µP；但 independent 缩放在高学习率下稳定性略好。ResNet 上则基本不需要这个额外 warmup（可能因为卷积网络归一化良好）。

## 亮点与洞察
- **统一杆秤很巧**：把 µP（历来用绝对表示变化表述）和 weight decay（历来用相对更新表述）都翻译成"相对量"，一条 $\|\Delta Y\|/\|Y\|=(\alpha_{\Delta W}/\alpha_W)\cdot\|\Delta W\|/\|W\|$ 等式就让"谁在补偿谁"一目了然，这是全篇推理的支点。
- **解释了一个长期反直觉现象**：为什么"违背 µP 理论"的 independent weight decay 反而迁移更好——因为 µP 理论缩放从训练中期起就错了，independent WD 恰好把它抵消回正确轨道。
- **$B\gg C$ 这个判据可迁移**：把"µP 何时失效"归到一个可计算的条件（batch/宽度比），让实践者能预判自己的设置是否落在 µP 假设成立的区间，并解释了 µP 为何在 Transformer 时代才被需要。
- **指向更优替代**：discussion 指出 Muon/Scion 这类矩阵级优化器能保持恒定的低更新对齐，可能天然绕开这些复杂性、减少对 warmup 的需求——给出后续方向。

## 局限与展望
- **只研究 AdamW**：作者明确只覆盖 AdamW（因其是 LLM 主流，且 SGD 版 µP 对隐层不缩放学习率、难以暴露差异）。SGD、矩阵级优化器（Muon/Scion）下的动态可能显著不同，仅作猜想。
- **简化模型不够准**：Eq. 11 的闭式 warmup 形状在实测中高估了达到稳态所需时间，作者归因于动量与梯度时间相关性，未在简化模型中建模。
- **替代方案需调参**：用显式 warmup 替代 µP 在高学习率下稳定性略逊，且 decay-away/指数 warmup 的长度需要调，不是即插即用。
- **ResNet 上有混淆因素**：ImageNet 上较大学习率带来的正则收益会干扰"训练损失 vs 验证损失最优相对变化不一致"的判断，结论的普适性打了折扣。
- **arXiv 号缺失**：缓存未给 arXiv 编号，引用时⚠️ 以原文为准。

## 相关工作与启发
- **vs Wang & Aitchison (2025)**：他们也论证 µP 需要 independent WD，但走的是"把 AdamW 近似成对历史更新的 EMA、时间尺度应跨宽度恒定"的路子，且未形式化论证。本文指出 EMA 视角忽略了"后续更新依赖更早更新"这一点（正是它让优化无法一步完成），改从"特征变化速率"直接把 weight decay 和 µP 的目标挂钩。
- **vs Everett et al. (2024)**：本文沿用其逐层学习率缩放和对齐定义，但 Everett 只考虑权重对齐，本文同时刻画更新对齐与权重对齐，才看清对齐比的演化。
- **vs Yang et al. (2023)**：借用其更易理解的 µP 表述与"局部变化"视角来隔离出核心对齐假设，但把分析从"无限宽 + 初始化"推进到"有限宽 + 全程训练"的实证。
- **vs Kosson et al. (2024a,b)**：直接建立在其 weight decay 框架与"相对更新 / warmup"观察之上，是本文相对更新统一框架的直接基础。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 推翻了"µP 负责学习率迁移"的主流认知，给出 weight decay 才是主角的统一解释
- 实验充分度: ⭐⭐⭐⭐ 大规模 LLaMA + ResNet 多宽度、多 WD 形式系统对照，但主要在两类网络上
- 写作质量: ⭐⭐⭐⭐⭐ 推理链清晰，公式与实测图一一对应，把反直觉现象讲透
- 价值: ⭐⭐⭐⭐⭐ 对大模型训练超参缩放有直接、可操作的指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Cautious Weight Decay](cautious_weight_decay.md)
- [\[ICLR 2026\] WSM: Decay-free Learning Rate Schedule via Checkpoint Merging for LLM Pre-training](wsm_decay-free_learning_rate_schedule_via_checkpoint_merging_for_llm_pre-trainin.md)
- [\[ICLR 2026\] Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate](convex_dominance_in_deep_learning_i_a_scaling_law_of_loss_and_learning_rate.md)
- [\[ICLR 2026\] Seesaw: Accelerating Training by Balancing Learning Rate and Batch Size Scheduling](seesaw_accelerating_training_by_balancing_batch_size_and_learning_rate_schedulin.md)
- [\[ICML 2026\] Limits of Convergence-Rate Control for Open-Weight Safety](../../ICML2026/optimization/limits_of_convergence-rate_control_for_open-weight_safety.md)

</div>

<!-- RELATED:END -->
