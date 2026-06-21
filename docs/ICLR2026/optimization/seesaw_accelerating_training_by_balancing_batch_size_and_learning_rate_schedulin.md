---
title: >-
  [论文解读] Seesaw: Accelerating Training by Balancing Learning Rate and Batch Size Scheduling
description: >-
  [ICLR 2026][优化/理论][batch size 调度] 本文从理论上证明了「学习率衰减」与「batch size 增大」在 SGD（及作为 Adam 代理的 normalized SGD）下的有限样本等价性，并据此提出即插即用的 Seesaw 调度器——每当余弦调度本应把学习率减半时，改为把学习率乘以 $1/\sqrt{2}$ 同时把 batch size 翻倍，在等 FLOPs 下匹配余弦衰减的 loss 曲线，却把串行墙钟时间缩短约 36%。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "batch size 调度"
  - "学习率衰减"
  - "训练加速"
  - "normalized SGD"
  - "critical batch size"
---

# Seesaw: Accelerating Training by Balancing Learning Rate and Batch Size Scheduling

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Nj0XBF2o7z](https://openreview.net/forum?id=Nj0XBF2o7z)  
**领域**: LLM效率 / 优化器与训练加速  
**关键词**: batch size 调度, 学习率衰减, 训练加速, normalized SGD, critical batch size

## 一句话总结
本文从理论上证明了「学习率衰减」与「batch size 增大」在 SGD（及作为 Adam 代理的 normalized SGD）下的有限样本等价性，并据此提出即插即用的 Seesaw 调度器——每当余弦调度本应把学习率减半时，改为把学习率乘以 $1/\sqrt{2}$ 同时把 batch size 翻倍，在等 FLOPs 下匹配余弦衰减的 loss 曲线，却把串行墙钟时间缩短约 36%。

## 研究背景与动机
**领域现状**：大模型预训练的 wall-clock 时间已经长达数月，而 batch ramp（训练过程中逐步增大 batch size）被认为是缩短墙钟时间的有效手段——更大的 batch 能利用更多并行计算、按比例减少所需的串行优化步数。LLaMA、Nemotron、OLMo、Apertus 等主流大模型训练都采用了某种 batch ramp。

**现有痛点**：这些 batch ramp 方案几乎都是**启发式手调**的，没有理论依据。没人知道这些启发式离最优有多远，也不知道学习率和 batch size 应该按什么比例联动。对 SGD 而言有个经典直觉「batch 翻倍 ≈ 学习率减半」，但对 Adam 这类自适应优化器，正确的联动规则并不清楚。

**核心矛盾**：batch size 不能无限增大——超过临界 batch size（critical batch size, CBS）后样本效率下降，加速收益消失。因此真正的问题是：在**不牺牲性能**的前提下，能把多少「串行步数」换成「并行 batch」？这本质是在「数据效率」和「并行度」之间找最优调度。

**本文目标**：(1) 给 batch size 调度一个严格的理论框架；(2) 推导出适用于 Adam 训练的实用调度规则；(3) 量化能达到的最大加速比。

**切入角度**：作者从一个朴素观察出发——「用学习率 $\eta/2$、batch $B$ 走两步」和「用学习率 $\eta$、batch $2B$ 走一步」在一阶泰勒展开下完全等价（确定性部分相同、噪声项方差也对得上）。如果这种等价能被严格证明并推广到 Adam，就能把学习率衰减「翻译」成 batch ramp。

**核心 idea**：把标准调度器中「学习率减半」这一步，**等价替换**为「学习率 ×$1/\sqrt{2}$ + batch ×2」，在保持 loss 动力学不变的同时减少串行步数。

## 方法详解

### 整体框架
Seesaw 不是一个新优化器，而是一个套在现有调度器（如余弦衰减）外面的**等价改写规则**。它的逻辑链是三层：先在 SGD 上严格证明「学习率衰减 ↔ batch ramp」的有限样本等价（Theorem 1）；再借助 normalized SGD（NSGD，Adam 的可分析代理）和「方差主导」假设把等价推广到自适应优化器（Corollary 1），得到一条等价曲线 $\alpha\sqrt{\beta}=\text{const}$；最后把这条等价关系工程化成 Algorithm 1——遍历输入调度器（余弦近似成步进衰减）会衰减学习率的那些时间点，在每个点把学习率衰减因子从 $\alpha$ 改成 $\sqrt{\alpha}$、同时把 batch 乘以 $\alpha$。

输入是一个已有调度器（给出它在哪些 token 数处会把学习率按因子 $\alpha$ 衰减），输出是一张 `(step, η, B)` 的新调度表：学习率衰减更慢、batch 同步增大，等 FLOPs 下 loss 曲线不变，但串行步数显著减少。

### 关键设计

**1. SGD 下学习率衰减与 batch ramp 的有限样本等价**

本文针对「现有 batch ramp 全是启发式、缺理论依据」这一痛点，给出（据作者所知）第一个**非渐近**（finite-sample）的等价性证明。考虑在带加性噪声的线性回归上跑 mini-batch SGD，总样本量为 $D$。基准过程（base）用一个阶梯式 batch ramp：在若干时间点把 batch 翻倍、学习率固定。对照过程（alternative）在相同时间点改为把学习率减半、batch 固定，并调整步数使总处理样本量仍为 $D$。Theorem 1 证明：base 过程的 excess risk 与 alternative 过程的 excess risk 相差只在一个常数因子之内。

这一等价的直觉来自一阶泰勒展开。设光滑损失 $L(x)$、$g_0=\nabla L(x_0)$，则「$(\eta,2B)$ 走一步」与「$(\eta/2,B)$ 走两步」的损失分别为
$$L(x_1)=L(x_0)-\eta g_0^\top(g_0+\xi')+O(\eta^2),\quad \mathrm{Cov}(\xi')=\frac{\sigma^2}{2B}I_d$$
$$L(x_2)=L(x_0)-\frac{\eta}{2}g_0^\top(2g_0+\xi_0+\xi_1)+O(\eta^2),\quad \mathrm{Cov}(\xi_i)=\frac{\sigma^2}{B}I_d$$
两者在确定性部分和噪声项上都到 $O(\eta^2)$ 等价——大 batch 摊薄了噪声方差，小学习率走两步把噪声平均掉，二者效果相同。本文把这个直觉升级成严格的风险界，这是整个 Seesaw 的地基。

**2. 推广到 Normalized SGD 与方差主导假设**

SGD 的等价比例是「学习率减 $\alpha$ ↔ batch 增 $\alpha$」，但 LLM 用的是 Adam，比例不一样。为此本文把 Adam 更新式逐步简化到 NSGD（normalized SGD），这是文献中常用的 Adam 可分析代理：取 $\beta_1=\beta_2=0$、用整参数更新近似逐坐标更新、把分母换成梯度平方范数的真实期望，得到
$$\theta_t=\theta_t-\eta\frac{g_t}{\sqrt{\mathbb{E}\|g_t\|^2}}$$
关键在于分母 $\mathbb{E}\|g_t\|^2$ 可分解为「mean + variance」两部分，其中 variance 随 batch 以 $O(1/B)$ 衰减。本文引入 **Assumption 3：方差主导**——假设梯度平方范数主要由加性噪声方差贡献。在这个假设下，NSGD 更新（在常数因子意义上）退化为带「重标定学习率」的 SGD，从而把 Theorem 1 的风险等价推广到 NSGD（Corollary 1）。

结论形式很优雅：对 NSGD，「学习率衰减因子 $\alpha$」与「batch 增大因子 $\beta$」等价的充要条件是 **乘积 $\alpha\sqrt{\beta}$ 保持不变**。代入 $\beta=2$（batch 翻倍）即得 $\alpha=\sqrt{2}$——这正是 Seesaw 用 $1/\sqrt{2}$ 而非 $1/2$ 衰减学习率的来源，与 SGD 的 $1/2$ 形成对照。

**3. Seesaw 调度器：余弦调度的即插即用替换**

有了 $\alpha\sqrt{\beta}=\text{const}$ 的等价曲线，本文把它工程化成 Algorithm 1。理论是对步进衰减建立的，实践中先把余弦衰减**近似成步进衰减**：取衰减因子 $\alpha$，记录余弦调度在哪些 token 数处会把学习率衰减 $\alpha$ 倍，把这些时间点 $S$ 作为输入。然后在每个点 $t\in S$ 执行
$$\eta\leftarrow \eta/\sqrt{\alpha},\qquad B\leftarrow B\cdot\alpha$$
输出新的 `(step, η, B)` 调度表。它是对现有余弦调度器的 drop-in 替换，不需要改优化器、不需要额外的 warmup checkpoint 搜索。

这正是与最接近的工作 Merrill et al. (2025) 的区别：后者通过从 checkpoint 出发、搜索能让 loss 维持 $\epsilon$-接近的最大 batch 倍数 $k^\star$ 来定规则（$B_{t+1}=2B_t,\ \eta_{t+1}=\sqrt{2}\eta$）。本文论证该规则在固定步数后会因不满足收敛约束而失稳发散（见 Lemma 4），而 Seesaw 由 NSGD on quadratics 严格推导、且自带防发散约束。

**4. 最激进的 ramp 约束与 36% 加速上限**

并不能在任意时刻把 batch 增到任意大就指望风险还匹配。Lemma 4 量化了这一点，给出**最激进**（不发散）的方案是 $\alpha=\sqrt{\beta}$（Remark 1）。结合等价曲线 $\alpha\sqrt{\beta}=\text{const}$，在基准 $\alpha=2,\beta=1$（$\alpha\sqrt{\beta}=2$）下，最激进可取 $\alpha=\sqrt{2},\beta=2$，再激进（如 $\alpha=1,\beta=4$）就会因失稳而偏离基准 loss 曲线。

在最激进极限下，可推出相对余弦衰减的理论加速上限（Lemma 1）：基准是 $T$ 步、恒定 batch、余弦学习率 $\eta(t)=\eta_0\cos(\frac{\pi t}{2T})$；在连续极限下保持 $\alpha=\sqrt{\beta}$ 的等价 batch ramp 过程，总串行步数等于归一化学习率曲线的积分
$$\int_0^T \frac{\eta(t)}{\eta_0}\,dt=\int_0^T \cos\!\Big(\frac{\pi t}{2T}\Big)\,dt=\frac{2T}{\pi}$$
于是串行墙钟时间最多减少 $1-\frac{2}{\pi}\approx 36.3\%$。之所以不到 50%，是因为余弦调度下训练进展大多发生在**早期高学习率阶段**，此时 batch 必须较小、并行度受限；Seesaw 主要在后期猛加并行，早期那段串行瓶颈仍在。

### 损失函数 / 训练策略
模型在 OLMo 代码库上以 Chinchilla 比例（$D=20N$）预训练，前 10% token 做学习率 warmup，之后按余弦或 Seesaw 衰减。优化器为 AdamW（$\lambda=0$ 无权重衰减，$\beta_1=0.9,\beta_2=0.95,\epsilon=10^{-8}$），训练时启用 z-loss。学习率在 $\{0.001,0.003,0.01,0.03\}$、初始 batch 在 $\{128,256,512,1024\}$ 上扫描，序列长度 $L=1024$，数据集为 C4（T5 tokenizer）。步进近似余弦时取衰减因子 $\alpha=1.1$。

## 实验关键数据

### 主实验
在 150M / 300M / 600M 模型上、按各自临界 batch size（CBS：150M≈256、300M≈512、600M≈1024，单位 ×$L$ tokens）训练，比较 Seesaw 与余弦衰减在等 FLOPs 下的最终验证 loss（取余弦调度最优学习率，$\alpha=1.1$）：

| 模型 | B=128 | B=256 | B=512 | B=1024 |
|------|-------|-------|-------|--------|
| 150M（余弦） | 3.0282 | 3.0353 | 3.0696 | 3.1214 |
| 150M（Seesaw） | 3.0208 | 3.0346 | 3.0687 | 3.1318 |
| 300M（余弦） | 2.8531 | 2.8591 | 2.8696 | 2.9369 |
| 300M（Seesaw） | 2.8452 | 2.8561 | 2.8700 | 2.9490 |
| 600M（余弦） | - | 2.6904 | 2.6988 | 2.7128 |
| 600M（Seesaw） | - | 2.6883 | 2.6944 | 2.7132 |

在 CBS 处训练时，两种调度器的最终 loss 在三个规模上都高度吻合（差异在小数点后第三位），而 Seesaw 的串行墙钟时间缩短约 36%，逼近理论上限。

### 消融实验
沿等价线 $\alpha\sqrt{\beta}=2$ 取不同 $(\alpha,\beta)$，验证「最激进方案 $\alpha=\sqrt{\beta}$」约束（150M，固定 batch，Chinchilla 规模）：

| 配置 $(\alpha,\beta)$ | 是否满足 $\alpha\ge\sqrt{\beta}$ | 现象 |
|------|------|------|
| $(2,\ 1)$ | 是（基准步进） | 匹配基准 loss |
| $(2^{3/4},\ \sqrt{2})$ | 是 | 匹配基准 |
| $(\sqrt{2},\ 2)$ | 临界（最激进） | 仍可匹配 |
| $(2^{1/4},\ 2^{3/2})$ | 否 | 偏离基准、失稳 |
| $(1,\ 4)$ | 否 | 明显偏离、发散 |

### 关键发现
- **方差主导假设是 Seesaw 成立的命门**：Assumption 3 要求梯度平方范数被加性噪声方差主导。当 batch 增到足够大（超过 CBS，如 1024/2048/4096/8192），噪声方差 $O(1/B)$ 被压到很小、mean 项开始主导，Seesaw 就匹配不上余弦曲线，且 batch 越大偏差越大。
- **超出 CBS 后存在根本性不可能**：作者用 1D NGD 玩具例子说明——在二次损失 $L(x)=\frac{1}{2}hx^2$ 上，NGD 更新 $x_{t+1}=x_t+\eta h\,\mathrm{sign}(x_t)$ 会停在 minimizer 周围 $O(\eta)$ 的稳定环上，必须**衰减学习率**才能逼近最优。大 batch 接近 NGD 区，继续加 batch 不改变动力学，因此过了某个 batch 后，任何固定学习率的 batch ramp 都无法复现学习率衰减的效果。
- **激进度有硬上界**：$\alpha=\sqrt{\beta}$ 是不发散的最激进选择，再贪心地增大 batch 会因失稳而掉点。

## 亮点与洞察
- **把工程启发式升级成定理**：batch ramp 在工业界用了多年却一直靠手调，本文给出第一个有限样本等价证明，并明确指出 Adam 下正确的联动是 $\alpha\sqrt{\beta}$ 恒定（而非 SGD 的 $\alpha\beta$），$1/\sqrt{2}$ 这个看似奇怪的系数有了严格出处。
- **NSGD 作为 Adam 代理 + 方差主导假设**这套分析路径很可迁移：先把自适应优化器化简到可分析的 NSGD，再用一个可验证的训练区假设把问题打回 SGD，是研究自适应优化器调度的好范式。
- **drop-in 替换**：Seesaw 不碰优化器、不需额外搜索，直接套在现有余弦调度上就能拿 ~36% 墙钟加速，落地成本极低。
- **诚实地给出失效边界**：用 1D NGD 玩具例子讲清「过了 CBS 为何根本不可能」，而不是假装方法万能。

## 局限与展望
- **依赖方差主导假设**：Seesaw 只在 batch 不超过 CBS、噪声方差主导分母时才匹配余弦衰减；一旦 batch 进入 mean 主导区就失效，限制了能换取的最大并行度。
- **理论建立在线性回归 / 二次损失上**：等价定理是对带噪线性回归和 NSGD on quadratics 证明的，到真实 Transformer 上是经验外推，严格性有 gap。
- **实验规模有限**：最大只到 600M、Chinchilla 比例、C4 数据集，是否在更大模型 / 更长训练 / 不同数据分布上保持 ~36% 加速尚待验证。
- **改进思路**：能否在线估计「方差 vs mean 占比」来自适应决定何时停止 batch ramp，从而在 CBS 边界附近榨取更多加速，是自然的下一步。

## 相关工作与启发
- **vs Merrill et al. (2025)**: 他们靠从 checkpoint 搜索最大不掉点 batch 倍数 $k^\star$ 来定规则（$B_{t+1}=2B_t,\eta_{t+1}=\sqrt{2}\eta$），本文证明该规则在固定步数后会失稳发散；Seesaw 由 NSGD on quadratics 严格推导、自带 $\alpha=\sqrt{\beta}$ 防发散约束，且是无需搜索的即插即用替换。
- **vs 线性缩放规则 (Smith et al., 2017)**: 他们经验观察到 SGD 下「线性增大 batch ≈ 减小学习率」，本文给出非渐近的有限样本证明，并把比例推广到自适应优化器的 $\alpha\sqrt{\beta}$。
- **vs McCandlish et al. (2018) 的噪声尺度**: 他们用基于 Hessian 的指标刻画 CBS（需访问 Hessian，大规模不可行）并观察到训练中噪声尺度上升；本文的理论预测与「噪声尺度随训练上升」一致，但无需 Hessian。
- **vs Malladi et al. (2022) / 平方根缩放**: 他们用 SDE 研究自适应算法下学习率随 batch 的缩放，本文复用了其一阶等价直觉，但把它落成可执行的调度器并给出风险界。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个学习率衰减↔batch ramp 的有限样本等价证明，并推广到 Adam 代理。
- 实验充分度: ⭐⭐⭐⭐ 三规模 + 等价线消融 + 失效边界都覆盖，但最大仅 600M。
- 写作质量: ⭐⭐⭐⭐⭐ 理论直觉、算法、加速上限层层递进，失效分析诚实。
- 价值: ⭐⭐⭐⭐⭐ 即插即用、~36% 墙钟加速，对大模型预训练有直接工程价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] WSM: Decay-free Learning Rate Schedule via Checkpoint Merging for LLM Pre-training](wsm_decay-free_learning_rate_schedule_via_checkpoint_merging_for_llm_pre-trainin.md)
- [\[ICLR 2026\] Predictive Differential Training Guided by Training Dynamics](predictive_differential_training_guided_by_training_dynamics.md)
- [\[ICLR 2026\] Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate](convex_dominance_in_deep_learning_i_a_scaling_law_of_loss_and_learning_rate.md)
- [\[ICLR 2026\] Shuffling the Data, Stretching the Step-Size: Sharper Bias in Constant Step-Size SGD](shuffling_the_data_extrapolating_the_step_sharper_bias_in_constant_step-size_sgd.md)
- [\[ICLR 2026\] Weight Decay May Matter More Than µP for Learning Rate Transfer in Practice](weight_decay_may_matter_more_than_µp_for_learning_rate_transfer_in_practice.md)

</div>

<!-- RELATED:END -->
