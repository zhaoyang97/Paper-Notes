---
title: >-
  [论文解读] Grokking: From Abstraction to Intelligence
description: >-
  [ICML 2026][可解释性][grokking] 本文从结构简化（奥卡姆剃刀）的视角统一解释 grokking 现象：训练过程中模型经历因果中介度退化、流形坍缩到 $\mathbb{Z}_{97}$ 圆环、谱能量向稀疏 Fourier 模集中、BDM 算法复杂度急剧下降这四种同步发生的"内部凝聚"，并用一个可解析的奇异特征机（SFM）证明这等价于自由能驱动的相变。
tags:
  - "ICML 2026"
  - "可解释性"
  - "grokking"
  - "奥卡姆剃刀"
  - "奇异学习理论"
  - "Kolmogorov 复杂度"
  - "模ular 算术"
---

# Grokking: From Abstraction to Intelligence

**会议**: ICML 2026  
**arXiv**: [2603.29262](https://arxiv.org/abs/2603.29262)  
**代码**: 无  
**领域**: 可解释性 / 涌现机制  
**关键词**: grokking, 奥卡姆剃刀, 奇异学习理论, Kolmogorov 复杂度, 模ular 算术

## 一句话总结
本文从结构简化（奥卡姆剃刀）的视角统一解释 grokking 现象：训练过程中模型经历因果中介度退化、流形坍缩到 $\mathbb{Z}_{97}$ 圆环、谱能量向稀疏 Fourier 模集中、BDM 算法复杂度急剧下降这四种同步发生的"内部凝聚"，并用一个可解析的奇异特征机（SFM）证明这等价于自由能驱动的相变。

## 研究背景与动机
**领域现状**：grokking（在模 $p$ 算术等小数据集上，训练精度饱和后很久才出现测试精度突增）已经成为研究大模型涌现现象的"果蝇实验"。现有解释主要分两类：电路级机理分析（哪些 attention head 在干什么）和正则化/初始化尺度分析（weight decay、init scale 与延迟泛化的关系）。

**现有痛点**：这些工作以描述为主、缺乏预测能力。它们或者依赖某个具体任务的电路分析，难以跨架构推广；或者只观察到某个相关指标的变化，并不解释"为什么在第 $T$ 步发生相变"。当问"grokking 究竟何时发生、为何发生"时，整个领域还没有统一答案。

**核心矛盾**：以往工作把 grokking 当作一个**局部电路**或**优化动力学**事件来研究，忽略了一个全局视角——模型的整体结构是否在自发地朝某种"最小描述长度"的解演化。如果存在这种全局简化倾向，那么 grokking 只是该倾向跨过某个能量阈值时的可观测后果，而不是一个独立现象。

**本文目标**：(1) 提供一组与架构无关的全局度量来追踪 grokking 过程中的结构演化；(2) 在一个解析可控的代理模型上证明这种结构演化等价于自由能/Kolmogorov 复杂度的最小化；(3) 把延迟泛化解释为一次"信息压缩相变"。

**切入角度**：作者把 grokking 视作模型在固定训练精度约束下不断"瘦身"——奥卡姆剃刀。在 SLT（奇异学习理论）的语言里，这对应于后验质量从大 RLCT $\lambda$ 的奇点流向小 $\lambda$ 的奇点；在 Kolmogorov 复杂度的语言里，这对应于权重描述长度的下降；在 Fourier 视角下，这对应于模型从全频段杂乱响应坍缩到稀疏的 group character。这三种语言其实是同一件事的不同投影。

**核心 idea**：grokking $=$ 在保持训练损失为零的等位面上，沿着"参数有效维度"下降方向的自发滑动，且滑动方向由 SLT 的自由能 $F_n \approx n\mathcal{L} + \lambda\ln n$ 决定。

## 方法详解

### 整体框架
论文要回答的是"grokking 究竟何时、为何发生"，做法是把一个无法解析的真实 Transformer 和一个能手算的代理模型并排放，让两者在同一组复杂度语言下相互印证。实证这条腿在 $p=97$ 的模 $\{+,-,\times,\div\}$ 任务上训一个 48 层 GPT-2 风格 Transformer，在初始化 / 记忆 / 涌现 / 泛化四个关键 step（$0.1\text{k}/1\text{k}/10\text{k}/100\text{k}$）分别做因果中介分析、嵌入流形的 PCA+Fourier 谱分析、以及对量化权重的 BDM 复杂度估计；理论这条腿构造一个奇异特征机（SFM），直接在 Fourier 域用复权矩阵拟合任务并显式带 $\ln n$ 稀疏先验，使 RLCT $\lambda$ 和 Kolmogorov 复杂度都能写成闭式。两条腿的落点是同一个相变：实证看到的三种"塌缩"指标，对应理论上 $\lambda$ 从 $p^2/2$ 降到 $p/2$。

### 关键设计

**1. 因果中介分析（CMA）+ skip-ablation：把"哪一层在干活"变成因果实验**

以往的电路解释只看 attention pattern 或 logit lens，分不清相关与因果，无法断言某个 head 是否真在因果通路上。作者改用 activation patching：构造两条同结构、不同操数的输入 $\mathbf{s}_1,\mathbf{s}_2$，把 $\mathbf{s}_2$ 的某个 head 激活嫁接进 $\mathbf{s}_1$ 得到 $\tilde{\mathbf{s}}$，再用因果中介得分 $\text{CMS}(h)=[\mathcal{M}_\theta(y_2\mid\tilde{\mathbf{s}})-\mathcal{M}_\theta(y_1\mid\tilde{\mathbf{s}})]-[\mathcal{M}_\theta(y_2\mid\mathbf{s}_1)-\mathcal{M}_\theta(y_1\mid\mathbf{s}_1)]$ 度量这次嫁接把正确答案的 logit 拨动了多少。沿训练时间看，这个量画出一条清晰的退化轨迹：step=1k 时高 CMS 的 head 杂乱散布在 0–47 全层，step=10k 整体变暗，到 step=100k 只剩 0–15 与 32–47 两端凝聚、中间 16–31 层熄灭。配套的 skip-ablation 把这种"凝聚"坐实为可观测量——直接跳过 16–31 层，精度几乎不掉，说明这些层已被 residual 旁路。这条从扁平噪声到两端凝聚的轨迹，就是 grokking 的结构指纹。

**2. 谱定域 + BDM 算法复杂度：两个互补的"变简单了多少"代理**

只看频域稀疏性会被 weight decay 的幅值缩水骗，只看 PCA 又看不到算法层面的结构，所以作者同时上两把尺子。频域这一把对 embedding 矩阵 $W_E$ 做二维 DFT 得谱密度 $S[k,l]$，再算 Gini 系数 $G(\mathbf{s})$ 和 inverse participation ratio $P(\mathbf{s})=\sum_i s_i^4(\sum_i s_i^2)^{-2}$，两者同时升高即表示能量从弥散收向少数 Fourier 模。算法这一把先把所有层权重经 quartile 量化映射到 4 字母表，再按 $4\times 4$ 子块用 CTM 查表配 BDM 公式 $K_{\text{BDM}}(\theta)=\sum_l\sum_b(\text{CTM}(b)+\log_2 n_b)$ 估全局算法复杂度——先量化正是为了剥掉 weight decay 带来的幅值变化，只留下真正的结构性重组。三类指标在 1k–10k 区间几乎同步陡降，共同支撑"grokking $=$ 结构简化"的结论。

**3. 奇异特征机（SFM）+ Occam Gate：把相变写成闭式**

真实 Transformer 上算不出 RLCT，于是作者造一个简化到极致却仍会 grok 的代理：把输入 $(u,v)$ 直接编码成 Fourier 张量 $\mathbf{x}_{\text{spec}}=\chi(u)\otimes\chi(v)$，模型只学一个复权矩阵 $\mathbf{W}\in\mathbb{C}^{p\times p}$，目标取 MAP 风格的 $\min_\mathbf{W}\tfrac12\sum_i\|y_i-\langle\mathbf{W},\mathbf{x}_{\text{spec}}^{(i)}\rangle_F\|^2+\beta\ln n\cdot\|\mathbf{W}\|_0$。动力学是两步迭代：先做残差与基函数的相关（drift），再用 Occam Gate $W_{kl}^{(t+1)}=\mathbb{I}(|\tilde W_{kl}^{(t)}|>\tau)\cdot\tilde W_{kl}^{(t)}$ 把信噪比低于 $\tau=\sqrt{2\beta\ln n/n}$ 的频率分量直接抹掉，正是这个 $\ln n$ 阈值在扮演奥卡姆剃刀。在这个模型上一切可解析：记忆期 $\lambda_{\text{mem}}\approx p^2/2$，泛化期支撑集坍缩到对角使 $\lambda_{\text{gen}}\approx p/2$，自由能交叉点 $n^*\approx-\frac{\beta(p^2-p)}{\epsilon_{\text{gen}}}W_{-1}(-\frac{\epsilon_{\text{gen}}}{\beta(p^2-p)})$。作者用"激活 support 大小 $/2$"作为 $\lambda$ 的上界代理，并证明它与 $K_{SFM}(\mathbf{W})\propto\lambda(\mathbf{W})\cdot(2\log_2 p+C_{\text{float}})$ 成正比，从而把 SLT 与 AIT 耦合到同一个可见对象上；同时明确声明 SFM 只是"假说生成型代理"，不是对 SGD-Transformer 的等价证明。

### 训练策略
真实 Transformer 用标准交叉熵 + AdamW，48 层 GPT-2、$d_{\text{model}}=512$、8 头、fp32、A100、100k 步、5 seed 平均；SFM 优化上式 $\mathcal{J}(\mathbf{W})$，每步走 drift + Occam Gate 两小步，由 $\beta\ln n$ 控制相变阈值，$n_{\text{eff}}$ 虽与训练 step 成正比但被明确解释为启发式映射。

## 实验关键数据

### 主实验

| 训练 step | CMA 高响应 head 分布 | 嵌入流形 | 谱集中度 (Gini, IPR) | BDM 复杂度 |
|----------|--------------------|----------|---------------------|-----------|
| 0.1k | 全层稀疏 | 高熵球团 | 极低 | 高 plateau |
| 1k（记忆） | 全层弥散 | 高维点云 | 仍低 | 高 plateau |
| 10k（涌现） | 中部开始变暗 | 开始收缩 | 急剧上升 | 急剧下降 |
| 100k（泛化） | 仅 0–15、32–47 | 1D 圆环（同构 $\mathbb{Z}_{97}$） | 高位稳定 | 最低 plateau |

| 现象 | 实证（Transformer） | 理论（SFM） |
|------|--------------------|------|
| 有效维度 | 层级旁路、中部可被跳过 | $\lambda$ 从 $p^2/2$ 降到 $p/2$ |
| 算法复杂度 | BDM 急降 + 块状结构出现 | $K_{SFM}\propto \lambda\cdot(2\log_2 p+C_{\text{float}})$ |
| 几何对称 | embedding 1D 环 | 支撑集塌缩到对角（加/减） |

### 消融实验

| 配置 | 现象 | 说明 |
|------|------|------|
| 跳过 head 0–15 | 精度崩溃 | 早层是必经路径 |
| 跳过 head 16–31 | 精度几乎不变 | 中层"功能冗余"，可被 residual 旁路 |
| 跳过 head 32–47 | 精度崩溃 | 末层负责输出格式化 |
| 量化前看 sparsity | 看似下降 | 但混入 weight decay 幅值缩水 |
| 量化后看 BDM | 真正下降 | 排除幅值效应后仍下降 → 结构性重组 |

### 关键发现
- 三类不同语言（电路冗余 / 谱稀疏 / 算法复杂度）的"塌缩"在时间轴上几乎同步发生，强烈暗示它们是同一事件的不同投影。
- 中部 16–31 层可旁路这一点说明所谓"涌现的符号结构"不是均匀分布在整个模型里，而是凝聚在两端的少数层；这与"实现 FMA 只需要 1D group 编码 + 输出投影"的理论预言一致。
- SFM 中相变阈值 $n^*$ 与 $\beta(p^2-p)/\epsilon_{\text{gen}}$ 成 $W_{-1}$ 关系，定性地复现了"高 weight decay → grokking 提前"的经验规律。
- 对乘除运算，SFM 的"对角"图像并不严格成立（需要离散对数重排），作者非常诚实地标注了这点局限。

## 亮点与洞察
- **复杂度的三重统一**：把 SLT 的 $\lambda$、AIT 的 KC 和谱稀疏在同一个 case 上对齐，是这篇论文最大的"啊哈"——以前这三套语言是分头说话的。
- **可旁路性作为可观测量**：用 skip-ablation 直接把"层是否必要"变成 yes/no 实验，比传统的 attention pattern 解释力更强，且这个 trick 可迁移到任何后训练分析（如 LLM 的功能性剪枝）。
- **量化后再算 BDM**：避免把 weight decay 的幅值变化误读成结构变化，是处理 grokking 数据的一个干净 trick。任何想用复杂度代理证明"模型变简单"的工作都该照抄。
- **SFM 不假装自己是 Transformer**：作者明确把 SFM 定位为"假说生成器"，不去吹"我们证明了 grokking 等价于 SLT 相变"，这种克制反而让结论更可信。

## 局限与展望
- SFM 的对角支撑图像只对加减法严格成立；乘除需要离散对数重排，作者只给定性说明，没有把 $\times,\div$ 的 SFM 解严格写出来。
- $n_{\text{eff}}(t)$ 与训练步数的映射是启发式的，自由能交叉点 $n^*$ 的预测无法在真实 Transformer 上做定量校验。
- 所有结论都基于 $p=97$ 的玩具任务，是否能推广到 LLM 上"知识涌现"是另一个量级的问题——文章自己承认"phase transition 在 SGD 上的语言只是描述性的"。
- BDM 的量化粒度（4×4 块、4 字母）有不少超参没消融。

## 相关工作与启发
- **vs Liu et al. (Omnigrok)**: 他们关注 weight decay 与 grokking 的因果，本文把这层因果嵌进 SLT 自由能框架里，给出了一个统一的"为什么 weight decay 有用"的解释（$\beta\ln n$ 项控制阈值）。
- **vs Nanda 等的电路机理工作**: 他们做的是 case-by-case 的电路逆向工程，本文用 CMA 给出了一个跨任务可计算的"哪一层在干活"指标，跳出对特定 head 的过拟合。
- **vs Mallinar et al. (non-NN grokking)**: 他们说 average gradient outer product 也能 grok，本文的 SFM 进一步剥离掉 NN 结构本身，把现象归结到"有 $\ln n$ 稀疏先验 + 全局可观测复杂度"这一最小集合，强化了 grokking 与架构无关的结论。
- **启发**：可旁路性测试 + 量化后复杂度 + 谱稀疏率，这套"三件套"诊断可迁移到任何"模型在训练中变简单"的研究，比如 LLM 的 emergent abilities 或 diffusion 的 mode collapse。

## 评分
- 新颖性: ⭐⭐⭐⭐ 第一次把 SLT/AIT/spectral 三套语言对齐到 grokking 这件事上，但具体度量都是已有工具
- 实验充分度: ⭐⭐⭐ 在 $p=97$ 这一个任务上做得很扎实，但只 1 个 prime、1 套架构，跨任务/跨尺度验证缺失
- 写作质量: ⭐⭐⭐⭐ 数学和实证两条腿叙述清晰，且作者对 SFM 局限的标注非常克制可信
- 价值: ⭐⭐⭐⭐ 给后续"涌现/相变"研究提供了一套通用诊断工具和一个可手算的 toy model

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test](../../ICLR2026/interpretability/grokking_in_llm_pretraining_monitor_memorization-to-generalization_without_test.md)
- [\[ICML 2026\] AI Engram: In Search of Memory Traces in Artificial Intelligence](ai_engram_in_search_of_memory_traces_in_artificial_intelligence.md)
- [\[ICML 2025\] Explaining, Fast and Slow: Abstraction and Refinement of Provable Explanations](../../ICML2025/interpretability/explaining_fast_and_slow_abstraction_and_refinement_of_provable_explanations.md)
- [\[NeurIPS 2025\] The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?](../../NeurIPS2025/interpretability/the_non-linear_representation_dilemma_is_causal_abstraction_enough_for_mechanist.md)
- [\[ICML 2026\] BLOCK-EM: Preventing Emergent Misalignment via Latent Blocking](block-em_preventing_emergent_misalignment_via_latent_blocking.md)

</div>

<!-- RELATED:END -->
