---
title: >-
  [论文解读] Towards a Theoretical Understanding of In-Context Learning: Stability and Non-i.i.d. Generalisation
description: >-
  [ICLR 2026][学习理论][算法稳定性] 本文在不假设 token 正交、不假设 i.i.d. 采样的现实条件下，用「算法稳定性 + 分布差异度量」两把工具，为非线性 Transformer 在 ICL（in-context learning）下的下一 token 预测推导出泛化误差界，揭示了优化配置与损失平滑度如何共同决定稳定性、训练/测试分布对齐如何决定可泛化性，并证明自回归预测长度若不加约束会导致误差累积乃至泛化崩溃。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "上下文学习"
  - "算法稳定性"
  - "分布偏移"
  - "泛化界"
  - "误差累积"
---

# Towards a Theoretical Understanding of In-Context Learning: Stability and Non-i.i.d. Generalisation

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Y8oiuzaAxl](https://openreview.net/forum?id=Y8oiuzaAxl)  
**代码**: 无  
**领域**: 学习理论 / 上下文学习  
**关键词**: 上下文学习, 算法稳定性, 分布偏移, 泛化界, 误差累积

## 一句话总结
本文在不假设 token 正交、不假设 i.i.d. 采样的现实条件下，用「算法稳定性 + 分布差异度量」两把工具，为非线性 Transformer 在 ICL（in-context learning）下的下一 token 预测推导出泛化误差界，揭示了优化配置与损失平滑度如何共同决定稳定性、训练/测试分布对齐如何决定可泛化性，并证明自回归预测长度若不加约束会导致误差累积乃至泛化崩溃。

## 研究背景与动机
**领域现状**：ICL 让大模型仅凭一段示例 prompt 就能在下游任务上预测、无需微调参数，这一能力近年吸引了大量理论分析。已有工作（如 Li et al. 2023、Huang et al. 2024、Chen et al. 2024、Wu et al. 2024）试图给出 ICL 的泛化界或刻画其训练动力学，部分还证明了预训练注意力模型能逼近最优岭回归的 Bayes 风险。

**现有痛点**：这些分析普遍依赖**理想化的数据假设**才能让证明走通——要么强加「成对正交的 token 模式」（Huang et al. 2024、Li et al. 2024a），要么假设「token 独立同分布采样」（Chen et al. 2024、Wu et al. 2024），要么干脆是优化无关（optimization-independent）的界。这些假设在真实场景里几乎不成立：现实文本 token 高度相关、训练与推理分布常常不一致。

**核心矛盾**：要让理论同时满足「非线性多头多层 Transformer + 依赖优化过程 + 容许分布偏移 + 不要求特殊输入结构 + 不要求正交」这一整套现实条件，每多一条都让证明难度陡增；既有工作只能各放松一两条（见原文 Table 1），没人能全要。

**本文目标**：在尽量贴近实际的非 i.i.d. 场景下，量化两件事——(1) 用 mini-batch 梯度下降训练的 Transformer 何时稳定；(2) 训练分布与目标 prompt 分布的差异如何影响泛化——并把二者合成一个可收敛的 ICL 泛化误差界。

**切入角度**：作者不去假设任何潜在的「概念几何」结构，而是借用统计学习理论里两件成熟工具：**算法稳定性**（algorithmic stability，刻画算法对训练数据扰动的敏感度）与**差异度量**（discrepancy measure，刻画训练分布与目标分布的偏离），把它们改造到「自回归下一 token 预测 + Transformer」的语境里。

**核心 idea**：泛化误差 ≈ 经验风险 + 分布差异 disc(q) + 稳定性项 β，三者各自可控、合起来给出 $O(N^{-1/2})$ 收敛率；而自回归用「自己预测的 token」喂给下一步会让误差逐 token 累积，因此预测长度必须被约束到至多对数增长。

## 方法详解

### 整体框架
本文是纯理论工作，"方法"即一条环环相扣的证明链：**问题建模 → 稳定性界（Theorem 1）→ 差异度量界（Theorem 2–3）→ 泛化误差界（Theorem 4）→ 误差累积约束（Theorem 5）**。

问题设定上，给定 $N$ 个样本 $\{(X_i, C_i)\}$，其中 $X_i$ 是查询、$C_i=(C_i^1,\dots,C_i^{N_c})$ 是长度 $N_c$ 的输出序列（关键在于**允许各样本服从不同分布**）。一条 ICL prompt $P_i=[D_i, X_i]$ 由示例集 $D_i$ 拼上查询构成；自回归预测第 $j$ 个 token 时，prompt 为 $P_{i,j}=[P_i, C_i^1,\dots,C_i^{j-1}]$。实践中下一 token 用的是模型**自己估计**的中间 token，记 $\hat P_{i,j}=[\hat P_{i,j-1}, T(\hat P_{i,j-1})]$，这正是误差累积的根源。模型 $T$ 是标准 $L$ 层非线性 Transformer（多头自注意力 + ReLU-MLP），用 mini-batch GD（Algorithm 1）以加权经验风险

$$\hat L(T)=\sum_{i=1}^{N}\frac{q_i}{N_c}\sum_{j=1}^{N_c}\ell\big(T(p_{i,j-1})_{*,:},\,c_i^j\big)$$

训练，其中 $q_i$ 是训练样本权重。目标是控制总体风险与经验风险之差 $L(T_S)-\hat L(T_S)$。三个核心设计正对应证明链里的三块基石：稳定性、差异度量、合成的泛化界（含误差累积推论）。

### 关键设计

**1. mini-batch GD 依赖的算法稳定性界：把优化配置、损失平滑度与稳定性绑在一起**

既有 ICL 泛化界大多优化无关，看不出「迭代步数、batch 大小、学习率」这些实际旋钮如何影响泛化。本文针对该痛点，定义了一种适配自回归 ICL 的**均匀稳定性** $\beta$：若把训练集 $S$ 中第 $i$ 个样本换成同分布独立样本得到 $S^i$，则对所有查询位置平均的损失变化被 $\beta$ 控制。Theorem 1 在有界性假设（Assumption 1，输入/响应/各参数矩阵范数有界）下，取学习率 $\eta_k=k^{-\alpha}$，给出依赖 Lipschitz 平滑常数 $\gamma$ 的分段界：

$$\beta \lesssim \begin{cases}\dfrac{B M_\ell L_\ell^{2/(\alpha(1+\gamma))}\,Q^{\gamma/(1+\gamma)}}{N^{\gamma\alpha}}, & \gamma\le \frac{1+\sqrt{1-4\alpha(1-\alpha)}}{2\alpha},\\[2mm]\dfrac{B M_\ell L_\ell^{2/(\alpha(1+\gamma))}\,Q^{(\alpha\gamma^2+1-\alpha)/(1+\gamma)}}{N^{\gamma\alpha}}, & \text{otherwise.}\end{cases}$$

这个界给出三条可操作结论：(1) 损失景观**足够平滑**（$\gamma$ 小）时稳定性可控，迭代步数 $Q$ 可随样本量 $N$ 多项式增长（Corollary 1，捕捉「多迭代降经验风险 vs 多迭代沿优化路径放大扰动、恶化稳定性」的权衡）；(2) **非平滑**（$\gamma$ 大）时稳定性随 $Q$ 急剧退化，尤其学习率小时更糟，故应把 $Q$ 限制到 $O(\ln N)$ 对数级（Corollary 2）；(3) 不论平滑与否，适当选步长都能让稳定性收敛率达到 $O(N^{-1})$。同时界里 $\beta$ 随 batch 大小 $B$ 增大而变差，这与「小 batch SGD 泛化更好」的经验观察一致。深度 $L$ 因 $M_\ell$ 等量随 $L$ 指数增长，故需深度至多随 $N$ 对数增长才稳定。

**2. 假设无关的分布差异度量：量化训练分布与目标分布的错位**

现实中训练域与目标域常不一致，需要一个**不施加分布假设**的指标来度量二者偏离。本文把 Kuznetsov & Mohri 的 discrepancy 改造成「与假设空间无关」的形式：

$$\mathrm{disc}(q):=\frac{1}{N_c}\sum_{j=1}^{N_c}\Big[E_{N+1,j}-\sum_{i=1}^{N}q_i E_{i,j}\Big],\quad E_{i,j}=\mathbb{E}\big[\ell(T_S(P_{i,j-1})_{*,:},C_i^j)\,\big|\,\{(p_m,c_m)\}_{m=1}^{i-1}\big].$$

它衡量目标任务分布与（按权重 $q$ 加权的）训练分布之间的失配程度，并在两种场景下被量化：i.i.d. 时（Theorem 2）$\mathrm{disc}(q)\le 2\beta\|q\|_2 N\sqrt{\log(2/\delta)}$，只要 $\beta\|q\|_2 N\to 0$（如均匀权重 $q_i=1/N$ 时即 $\beta=o(N^{-1/2})$）差异就渐近消失——这把差异度量直接挂到了设计 1 的稳定性 $\beta$ 上。非 i.i.d. 时（Theorem 3），在「至少部分训练域与目标域相关、存在有效 prompt 使失配 $\le\epsilon$」的前提下，给出基于**序列 Rademacher 复杂度** $R_N(\{\ell\circ T\})$ 的上界，里面出现权重差 $\|q-v\|$ 项——它为「微调通过把训练样本重加权到与目标相关的样本」提供了理论解释，也说明假设空间越复杂（Rademacher 复杂度越高）对分布偏移越敏感，故正则化（如权重范数约束）有助于对齐。

**3. 合成泛化误差界与误差累积约束：把稳定性 + 差异度量拼成可收敛的界，并限制预测长度**

有了前两块，Theorem 4 直接给出泛化误差的合成界：

$$L(T_S)\le \frac{1}{N_c}\sum_{i=1}^{N}\sum_{j=1}^{N_c}q_i\,\ell(T_S(p_{i,j})_{*,:},c_i^j)+\mathrm{disc}(q)+\|q\|_1\beta+2\|q\|_2 M_\ell\sqrt{2\log(4/\delta)}.$$

即「经验风险 + 差异 + 稳定性项 + 置信项」。其渐近行为（Corollary 3–4，汇总于原文 Table 2）显示：i.i.d. 下不论损失是否平滑，只要按 $|B|=O(N^{\zeta_1})$、$Q=O(N^{\zeta_2})$ 且 $2\zeta_1+2\zeta_2\gamma/(1+\gamma)=1$（平滑）或 $Q=O(\ln N)$（非平滑）调参，都能达到最快收敛率 $O(N^{-1/2})$；非 i.i.d. 下则需「合理重加权样本（$\|q-v\|$ 小）+ 好的 ICL prompt」才能泛化，并归结为一个含 $\lambda_1\|s-q\|_2^2+\lambda_2\|q\|_2^2$ 正则的两阶段优化（先 DC 规划/梯度法求样本权重 $q$，再训模型）。最后 Theorem 5（误差累积，详见原文 Appendix G）指出：用模型自估 token 自回归时误差逐步前传累积，为保证泛化，**下一 token 预测的长度应至多随样本量对数增长**，否则误差陡增、泛化崩溃。

### 损失函数 / 训练策略
训练用 Algorithm 1 的 mini-batch GD：每步随机采 batch $B$，按 $\theta_q=\theta_{q-1}-\frac{\eta_{q-1}}{|B|}\sum_{i\in B}\nabla_\theta\hat L(T)$ 更新，学习率 $\eta_k=k^{-\alpha}$ 衰减。非 i.i.d. 的实用建议是解 Remark 5 的两阶段加权目标（式 4）：先优化后三项求最优样本权重 $q$，再优化首项学模型参数；可用 DC programming 或梯度法。

## 实验关键数据

实验为纯验证性质（论文是理论工作），用 12 层 8 头的 GPT-2（HuggingFace 实现）在 H20 GPU 上跑，沿用 Li et al. 2023 的设置。任务为 $d=10$ 维线性回归，序列按递推 $c_l^i=\beta_{l-1}^i c_{l-1}^i+\epsilon$ 生成，$N\in\{50,100,200,400,800,1600\}$，均匀权重 $q_i=1/N$，$|B|=N^{1/2}$，$Q=200$，$\alpha=1$，序列长度 $N_c\in\{1,\dots,9\}$，独立生成 1000 个测试样本评测。

### 主实验（渐近收敛验证）

| 现象 | 实验配置 | 观察 | 对应理论 |
|------|---------|------|---------|
| 泛化误差随 $N$ 收敛 | 序列长 1、2，$N$ 从 50→1600 | 误差随 $N\to\infty$ 单调下降并渐近消失 | Corollary 3（i.i.d.，$O(N^{-1/2})$） |
| 误差随长度累积 | 固定 $N$，$N_c$ 从 1→9 | 误差随序列长度近似多项式上升；超过约 $\ln N$ 阈值后陡增 | Theorem 5（误差累积） |

### 消融 / 分析实验

| 配置 | 关键观察 | 说明 |
|------|---------|------|
| 增大序列长度 $N_c$ | 误差陡升点随 $N$ 增大而后移 | 阈值 $\approx\ln N$，验证「预测长度应对数增长」 |
| 不同 $N$ 下对比 | 大 $N$ 容忍更长预测 | 样本越多、可安全自回归的步数越长 |
| 非 i.i.d. 扩展（Appendix I） | 与理论界一致 | 验证分布差异与 prompt 重加权的影响 |

### 关键发现
- **误差累积的对数阈值是最直观的可验证结论**：序列长度一旦超过约 $\ln N$，泛化误差从多项式增长转为陡升，且该阈值随样本量增大而右移——精确印证 Theorem 5「预测长度至多对数增长」的约束。
- **小 batch 更稳**：稳定性界里 $\beta$ 随 batch 大小增大而恶化，与经验上「小 batch SGD 泛化更好」一致，给该现象一个稳定性视角的理论支撑。
- **分布对齐是非 i.i.d. 泛化的关键**：$\|q-v\|$ 越小（训练重加权越贴近目标）泛化越好，这也解释了微调为何有效。

## 亮点与洞察
- **一次性放松了全部理想化假设**：相比既有工作只能放松一两条，本文同时做到多头多层 + 优化依赖 + 容许分布偏移 + 无特殊输入结构 + 无正交假设（原文 Table 1 中唯一全 ✓），把 ICL 泛化理论推到更贴近真实场景。
- **把「稳定性」和「差异度量」拼成一个界**：$\beta$ 同时出现在稳定性项和差异度量界里，使优化旋钮（$Q,|B|,\alpha$）、损失平滑度 $\gamma$、分布对齐 $\|q-v\|$ 这些原本割裂的量被统一进同一个泛化表达式，可迁移到其他自回归生成的泛化分析。
- **误差累积给「别让模型自回归太长」一个硬约束**：$\ln N$ 阈值把「生成越长越不准」从经验直觉变成可量化的理论边界，对长序列生成的可靠性设计有直接启发。
- **为微调与样本重加权提供理论解释**：$\|q-v\|$ 项把「微调=把训练样本重加权到与目标相关」讲清楚了，并导出可用 DC programming 求解的两阶段加权训练目标。

## 局限与展望
- **稳定性界依赖有界性假设**（Assumption 1，输入/参数范数有界），虽然作者声称可放松到 sub-Gaussian 轻尾分布，但完整证明仍在该假设框架内。
- **实验仅在合成线性回归任务 + GPT-2 上验证**，未触及真实语言任务，理论与大规模实际 ICL 之间仍有距离。
- **界中常数随深度 $L$ 指数增长**（$M_\ell$ 等），要求深度至多随 $N$ 对数增长，对很深的现代大模型而言界可能偏松。
- 作者展望用梯度稳定性等更精细工具收紧界，并把加权训练策略落到实际算法设计中。

## 相关工作与启发
- **vs Li et al. (2023)**：他们给出优化无关的 ICL 泛化界（i.i.d. 输入或动力系统轨迹），本文把优化过程（mini-batch GD 的 $Q,|B|,\alpha$）显式纳入，并扩展到非 i.i.d.。
- **vs Huang et al. (2024) / Li et al. (2024a)**：他们依赖成对正交 token 模式且多为单/浅层注意力，本文是无正交假设的多头多层 Transformer。
- **vs Chen et al. (2024) / Wu et al. (2024)**：他们假设 token 独立同分布采样，本文显式处理分布偏移，用差异度量刻画训练/目标失配。
- **vs Bu et al. (2024, 2025)**：他们从概念几何/任务向量角度做机制解释，本文是互补路线——不假设任何潜在概念结构，直接用稳定性 + 差异度量构建分布偏移感知的泛化框架。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次在一个分析里同时放松正交、i.i.d.、单层等全部理想化假设，并把稳定性与差异度量统一。
- 实验充分度: ⭐⭐⭐ 作为理论工作实验是验证性的，仅合成线性回归 + GPT-2，规模有限但与理论吻合。
- 写作质量: ⭐⭐⭐⭐ 证明链条清晰，Table 1/2 对比与汇总到位，假设与结论交代明确。
- 价值: ⭐⭐⭐⭐ 为 ICL 泛化、误差累积约束、微调/样本重加权提供了可操作的理论依据。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Understanding In-Context Learning on Structured Manifolds: Bridging Attention to Kernel Methods](understanding_in-context_learning_on_structured_manifolds_bridging_attention_to_.md)
- [\[ICLR 2026\] Understanding the Dynamics of Forgetting and Generalization in Continual Learning via the Neural Tangent Kernel](understanding_the_dynamics_of_forgetting_and_generalization_in_continual_learnin.md)
- [\[ICLR 2026\] Transformers with Endogenous In-Context Learning: Bias Characterization and Mitigation](transformers_with_endogenous_in-context_learning_bias_characterization_and_mitig.md)
- [\[ICLR 2026\] Pretrain–Test Task Alignment Governs Generalization in In-Context Learning](pretraintest_task_alignment_governs_generalization_in_in-context_learning.md)
- [\[ICLR 2026\] On learning linear dynamical systems in context with attention layers](on_learning_linear_dynamical_systems_in_context_with_attention_layers.md)

</div>

<!-- RELATED:END -->
