---
title: >-
  [论文解读] Understanding the Role of Training Data in Test-Time Scaling
description: >-
  [ICLR2026][Reasoning][测试时缩放] 从理论上分析训练数据属性如何影响 test-time scaling 的效果，证明 CoT 推理等价于伪牛顿法迭代，提出基于特征协方差最小特征值的任务难度度量，揭示"更多思考不一定更好"的 overthinking 现象机制，并给出多任务训练中最优任务选择策略——训练集应多样、相关且困难。
tags:
  - "ICLR2026"
  - "Reasoning"
  - "测试时缩放"
  - "chain-of-thought"
  - "上下文学习"
  - "task hardness"
  - "过度思考"
  - "training data selection"
---

# Understanding the Role of Training Data in Test-Time Scaling

**会议**: ICLR2026  
**arXiv**: [2510.03605](https://arxiv.org/abs/2510.03605)  
**代码**: 无  
**领域**: LLM推理  
**关键词**: 测试时缩放, chain-of-thought, 上下文学习, task hardness, 过度思考, training data selection

## 一句话总结
从理论上分析训练数据属性如何影响 test-time scaling 的效果，证明 CoT 推理等价于伪牛顿法迭代，提出基于特征协方差最小特征值的任务难度度量，揭示"更多思考不一定更好"的 overthinking 现象机制，并给出多任务训练中最优任务选择策略——训练集应多样、相关且困难。

## 研究背景与动机

**领域现状**：Test-time scaling（如 OpenAI o1、DeepSeek R1）通过在推理时分配更多计算资源生成更长的 CoT 来提升推理能力，已在数学竞赛、编程等任务上取得显著成功。

**核心问题**：尽管实践效果显著，训练数据在什么条件下支持 test-time scaling 仍不清楚——具体地：
   - 增加 test-time 计算是否**总是**能提升下游推理表现？
   - 增加 test-time 计算能否降低训练时的计算需求？
   - 什么是"困难"训练样本？它们为何对 test-time scaling 有益？

**现有工作不足**：先前关于训练数据多样性和难度的研究大多是经验性的，缺乏严格的理论框架解释 test-time scaling 的机制。

**本文切入角度**：在 linear regression 的 in-context learning 框架下，从理论和实验两个维度回答上述三个问题。

## 方法详解

### 整体框架

本文把 test-time scaling 放进一个可解析的玩具世界：用线性回归的上下文学习（in-context learning, ICL）作任务，用单层线性自注意力（linear self-attention, LSA）作模型，再把测试时的多步思维链（chain-of-thought, CoT）看成在这个模型上反复迭代。整篇论文沿一条因果链展开：先把"从 prompt 反推权重"写成可优化的回归问题，证明 LSA 的全局最优解只取决于一个由训练数据协方差决定的枢纽矩阵 $\Gamma$；再证明测试时跑 CoT 等价于用 $\Gamma^{-1}$ 做伪牛顿迭代，于是"思考好不好"被翻译成"这个迭代收不收敛、收得快不快"；接着用协方差谱给任务难度下定义、推出 test-time scaling law，量化"多思考"和"多训练数据"如何互换；最后说明这套迭代只在训练协方差 $\Gamma$ 与测试协方差 $\Sigma$ 对齐时才有效，由此既解释了 overthinking（越想越糟）、又导出该挑什么训练数据。所有结论都从同一个协方差矩阵的谱性质里推出来，把原本只能靠经验观察的现象变成可证明的定理。

> 本文是纯理论分析（线性模型 + 谱性质推导），核心是一串环环相扣的定理而非数据流水线，故不画框架图；下面四个关键设计即按上述因果链顺序展开。

### 关键设计

**1. 可解析的玩具世界：ICL 权重预测与 LSA 全局最优**

要严格分析 test-time scaling，先得有一个能算出闭式解的设定。每个 prompt 是 $P_\tau = (x_{\tau,1}, y_{\tau,1}, \ldots, x_{\tau,n}, y_{\tau,n})$，标签由隐藏权重生成 $y_{\tau,i} = \langle w_\tau, x_{\tau,i}\rangle$，其中 $x_{\tau,i}\sim\mathcal{N}(0,\Lambda)$、$w_\tau\sim\mathcal{N}(0,I_d)$，模型要从 prompt 里反推出 $w_\tau$。输入被排成一个嵌入矩阵，最后一列专门留给权重估计位 $\hat w_0$：

$$E_\tau = \begin{bmatrix} X_\tau & 0 \\ y_\tau & 0 \\ 0_{d \times n} & \hat{w}_0 \\ 0_{1 \times n} & 1 \end{bmatrix}$$

训练目标就是让最后一列预测出真权重，最小化均方误差 $L(\theta) = \tfrac{1}{2}\mathbb{E}\big[\| f_{\text{LSA}}(E_\tau;\theta)_{[:,-1]} - (0_d, 0, w_\tau, 1)\|^2\big]$。Theorem 3.1 证明，在合适初始化下常数步长的梯度下降会收敛到全局最优 $V_* = -\Gamma^{-1}/c$，关键是这个最优解只依赖一个由数据协方差决定的矩阵 $\Gamma := (1 + \tfrac{1}{n})\Lambda + \tfrac{1}{n}\text{tr}(\Lambda) I_d$。$\Gamma$ 是整篇论文的枢纽——它把训练数据属性（协方差 $\Lambda$、prompt 长度 $n$）一次性打包，后面所有定理都通过它来传导。

**2. CoT 等价于伪牛顿法**

设定搭好后，第一个问题是"测试时多跑几步 CoT 到底在算什么"。Proposition 3.2 表明，测试时跑 $k$ 步 CoT，本质上是让权重估计按 $w_{i+1} = w_i - \tfrac{1}{m}\Gamma^{-1} X_{\text{test}} X_{\text{test}}^\top (w_i - w_{\text{test}})$ 递推。这恰好是对测试损失 $\ell(w) = \tfrac{1}{2m}\|y_{\text{test}} - X_{\text{test}}^\top w\|^2$ 做**伪牛顿法**——用训练得到的 $\Gamma^{-1}$ 去近似真 Hessian 的逆 $\Lambda^{-1}$。展开 $k$ 步得到闭式 $w_{k+1} = \big(I - (I - \tfrac{1}{m}\Gamma^{-1} X_{\text{test}} X_{\text{test}}^\top)^k\big) w_{\text{test}}$。这个等价关系把"链式思考"从黑盒动作翻译成收敛行为可分析的优化迭代，后面所有好坏判断都建立在"这个迭代收不收敛、收得快不快"之上——而收敛速度恰由枢纽 $\Gamma$ 与测试数据的匹配程度决定。

**3. 任务难度与 test-time scaling law**

有了"CoT = 迭代"，就能量化什么任务难、思考多久才划算。Theorem 3.3 给出无 CoT 直接 ICL 的误差上界 $\mathbb{E}\|\hat{w} - w_{\text{test}}\|^2 \leq \tfrac{d}{n^2}(1 + \tfrac{\text{tr}(\Lambda)}{\lambda_{\min}(\Lambda)})^2 + \tfrac{d}{m}(1 + \tfrac{\text{tr}(\Lambda)}{\lambda_{\min}(\Lambda)})$，其中反复出现的比值被提炼成任务难度 $\text{Hard}(\Lambda) := \tfrac{\text{tr}(\Lambda)}{\lambda_{\min}(\Lambda)}$。直观上 $\Lambda$ 的每个特征向量是一种"技能"、特征值是技能强度：容易任务只靠少数几种势均力敌的技能（特征值相近，$\lambda_{\min}$ 不小），困难任务依赖多种技能且分布长尾（存在极小特征值，把比值顶得很大）。它不靠人工标注，而是直接从数据分布的几何结构里读出难度。把这个难度和伪牛顿迭代的收敛速度合在一起，Corollary 3.5 给出 $k$ 步 CoT 后的误差

$$\mathbb{E}\|w_{k+1} - w_{\text{test}}\|^2 \leq d\,\Big(1 + \tfrac{n}{1 + \text{Hard}(\Lambda)}\Big)^{-2k}(1 + o(1))$$

这条 scaling law 一次读出三个结论：固定目标误差 $\varepsilon$ 时，增大思考步数 $k$ 可换取更短的训练 prompt 长度 $n$（训练算力和推理算力可互补）；任务越难（$\text{Hard}(\Lambda)$ 越大），底数越接近 1、收敛越慢，需要越长的 CoT 才达标；整个过程复杂度只是 $O(kd^2)$。

**4. 分布对齐决定成败：overthinking 与最优任务选择**

前面 scaling law 的收敛有个前提——训练协方差 $\Gamma$ 要跟测试协方差 $\Sigma$ 对齐；这条前提一旦破了，就同时解释了"越想越糟"和"该练什么数据"。思考的净效果实际由 $\text{tr}\big((I - \Gamma^{-1/2}\Sigma\Gamma^{-1/2})^{2k}\big)$ 控制：当目标任务的某些技能方向（$\Sigma$ 的特征向量）在训练数据里覆盖不足、对应的 $\Gamma$ 在该方向很弱时，括号里矩阵在该方向的特征值大于 1，该项随 $k$ 指数级放大——多思考不是修正误差，而是把训练没学到的方向越推越偏。这就是 overthinking 的数学根源：不是模型"想累了"，而是迭代在未覆盖子空间上发散。反过来，既然表现取决于训练协方差能否覆盖并对齐目标谱，多任务训练就该优先挑能补上缺失方向的任务。Proposition 4.3 证明最优采样概率 $\{\pi_\ell\}$ 会把至少一半概率分配给"困难"任务（$\sigma_{\min}(\Lambda_\ell)$ 小的那些），而求最优配比本身是一个高效可解的二次规划：

$$\min_{\{\pi_\ell\}} \left\| I - \Sigma^{-1} \sum_{\ell} \Lambda_\ell \pi_\ell \right\|_F^2 \quad \text{s.t.} \sum_\ell \pi_\ell = 1,\ \pi_\ell \geq 0$$

目标函数让训练任务的加权协方差去逼近测试协方差 $\Sigma$，于是"好训练集"的三条标准自然浮现：**多样**（覆盖目标的所有技能方向）、**相关**（加权后与目标谱对齐）、**困难**（包含长尾、贡献小特征值方向的样本）。

## 实验关键数据

### LSA 模型验证

| 设定 | 结论 |
|------|------|
| 训练 prompt 长度 $n=10,20,30$ | 增大 $k$ 可弥补较短的训练上下文，$n=10$ 在 $k=20$ 时达到 $n=30$ 直接预测的误差水平 |
| 训练协方差倾斜 ($\lambda_i \propto 1/i$) | 训练/测试分布不匹配时，$k$ 增大后测试误差先降后升——overthinking 出现 |
| overthinking 时大 $n$ 反而更差 | 与非 overthinking 情况相反——更长训练上下文在倾斜分布下"学得更偏" |

### GPT-2（9.5M 参数）验证

| 实验 | 结果 |
|------|------|
| 训练 $n=20,30,40$，变化 $k$ | 与 LSA 趋势一致：更长 CoT 允许用更短训练上下文达到同等性能 |
| 倾斜协方差 + 全等测试 | GPT-2 同样出现 overthinking：大约 $k>10$ 后误差上升 |

### 任务选择实验

| 任务类型 | $(\alpha, B)$ | 平均选择概率 |
|----------|--------------|-------------|
| Easy-Short | (0.2, 20) | 最低 |
| Hard-Short | (0.8, 20) | 中等 |
| Easy-Long | (0.2, 100) | 中等偏低 |
| **Hard-Long** | **(0.8, 100)** | **最高** |

### 真实推理基准（Qwen 2.5-7B）

| 模型 | CoT 长度 [0, 1k) | CoT 长度 [1k, 2k] |
|------|------------------|--------------------|
| Qwen-Base | 30.39% | 27.2% |
| Qwen-GCD（训练对齐） | **75%**（+44.6） | **38.4%**（+11.2） |
| Qwen-Poly（训练不对齐） | 29%（-1.4） | 20.83%（**-6.4**） |

训练数据对齐时更多 thinking 有帮助，不对齐时更多 thinking 有害——完美验证了理论预测。

## 亮点与洞察
- **CoT = 伪牛顿法**：将 test-time CoT 与优化算法建立了精确的数学对应，提供了理解推理过程的新视角
- **Overthinking 的理论解释**：首次从理论上解释了为什么更多推理有时会损害性能——训练数据未覆盖的技能方向在迭代中被放大
- **任务难度的特征谱定义**：$\text{Hard}(\Lambda) = \text{tr}(\Lambda)/\lambda_{\min}(\Lambda)$ 是一个简洁而有洞察力的度量
- **训练-测试计算的可替代性**：严格证明了 test-time compute 可以补偿训练时 context length 的不足——为实践中的资源分配提供了理论指导

## 局限与展望
- **理论局限于线性模型**：主要分析限于 linear regression + LSA，对非线性任务和深层 Transformer 的推广需进一步工作
- **GPT-2 实验仍在合成数据上**：真实推理基准实验（Qwen）只涉及两个特定任务（GCD 和多项式根），覆盖面有限
- **任务难度定义依赖协方差谱**：实际 NLP 任务中"技能"和"特征分布"难以直接测量，理论到实践的 gap 明显
- **未考虑 RL 训练场景**：当前分析基于 SFT/ICL，o1/R1 类模型的 RL 训练范式下理论是否成立未知

## 相关工作与启发
- **vs Snell et al. (2024) / Muennighoff et al. (2025)**：先前工作是经验性地展示 test-time scaling 效果，本文提供了理论基础
- **vs Su et al. (2025) (overthinking)**：先前经验观察到 LLM 在简单问题上 overthink，本文给出了数学解释
- **vs Huang et al. (2025a)**：先前分析限于各向同性特征（$\Lambda = I$）且在训练时用 CoT，本文扩展到一般协方差并分析纯 test-time CoT
- **启发**：在实际系统设计中，应根据训练数据覆盖度动态调整 test-time 计算量——对训练数据充分覆盖的任务加大 compute，对覆盖不足的任务适度限制

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次从理论上系统地分析训练数据与 test-time scaling 的关系，overthinking 和任务选择的理论都是新贡献
- 实验充分度: ⭐⭐⭐⭐ LSA/GPT-2 合成实验充分验证理论，Qwen 真实基准实验为亮点，但真实任务覆盖面可更广
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨清晰，直觉解释到位，从单任务到多任务层层递进
- 价值: ⭐⭐⭐⭐⭐ 对理解和改进 test-time scaling 有重要指导意义，任务选择策略可直接用于 RL reasoning 训练

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] ContextPRM: Leveraging Contextual Coherence for multi-domain Test-Time Scaling](contextprm_leveraging_contextual_coherence_for_multi-domain_test-time_scaling.md)
- [\[ICLR 2026\] Plan and Budget: Effective and Efficient Test-Time Scaling on Reasoning LLMs](plan_and_budget_effective_and_efficient_test-time_scaling_on_reasoning_large_lan.md)
- [\[ICLR 2026\] ATTS: Asynchronous Test-Time Scaling via Conformal Prediction](atts_asynchronous_test-time_scaling_via_conformal_prediction.md)
- [\[ACL 2025\] Rethinking the Role of Prompting Strategies in LLM Test-Time Scaling: A Perspective of Probability Theory](../../ACL2025/llm_reasoning/rethinking_the_role_of_prompting_strategies_in_llm_test-time_scaling_a_perspecti.md)
- [\[ICLR 2026\] Native Reasoning Models: Training Language Models to Reason on Unverifiable Data](native_reasoning_models_training_language_models_to_reason_on_unverifiable_data.md)

</div>

<!-- RELATED:END -->
