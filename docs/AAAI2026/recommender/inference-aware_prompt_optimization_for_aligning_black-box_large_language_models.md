# Inference-Aware Prompt Optimization for Aligning Black-Box Large Language Models

**会议**: AAAI 2026  
**arXiv**: [2508.10030](https://arxiv.org/abs/2508.10030)  
**代码**: [https://iapo-aaai25.github.io/](https://iapo-aaai25.github.io/)  
**领域**: 对齐RLHF / Prompt优化  
**关键词**: Inference-Aware Optimization, Prompt Optimization, Best-of-N Sampling, Contextual Bandits, Black-Box Alignment  

## 一句话总结
揭示 prompt 选择与推理策略（Best-of-N、Majority Voting）之间存在非平凡交互关系，提出 IAPO 框架将 prompt 设计与推理规模联合优化为上下文最优臂识别问题，并设计 PSST 固定预算训练算法，在 6 个任务上相比推理无关方法提升最高 50%。

## 研究背景与动机

1. **领域现状**：黑盒 LLM 的对齐主要依赖两类方法——prompt 优化（通过改写/追加指令来引导输出）和推理扩展策略（Best-of-N 采样、Majority Voting 生成多候选选最优）。两者分别取得了显著成功。
2. **现有痛点**：现有 prompt 优化方法完全不考虑部署时的推理策略，即在单次生成（$N=1$）下优化 prompt，然后在部署时直接套用 BoN 或 MV。这种脱耦会导致次优甚至错误的 prompt 选择。
3. **核心矛盾**：最优 prompt 会随推理策略和预算而变化。作者发现在 MATH 上，Prompt A 在 $N=1$ 准确率 65% 优于 Prompt B 的 62%，但 MV 下 $N=10$ 时 Prompt B 升至 ~77% 而 Prompt A 降至 ~63%。这是因为 MV 放大了单查询正确率分布的非线性效应。
4. **本文要解决什么？** 如何在有限计算预算下，联合优化 prompt 和推理规模，同时考虑用户对多目标的偏好权衡？
5. **切入角度**：将问题建模为上下文最优臂识别（contextual best-arm identification），其中每个"臂"是 (prompt, 推理规模 $N$) 的组合，"上下文"编码用户偏好和预算约束。
6. **核心idea一句话**：推理感知的 prompt 优化——让训练阶段模拟推理策略（BoN/MV）的非线性聚合效应，从而选出在实际部署配置下最优的 prompt-规模组合。

## 方法详解
### 整体框架
IAPO（Inference-Aware Prompt Optimization）框架定义臂 $a = (p, N) \in \mathcal{A} = \mathcal{P} \times [N_{\max}]$，上下文 $c = (w_1, \dots, w_{K+1})$ 编码多目标权重和预算偏好。策略 $\pi: \mathcal{C} \to \mathcal{A}$ 在观测上下文后选择最优臂。优化目标为最大化 Average Contextual Return $\text{ACR}(\pi) = \mathbb{E}_{c}[Q^\alpha(c, \pi(c))]$，其中 $Q^\alpha$ 是在推理策略 $\alpha \in \{\text{BoN}, \text{MV}\}$ 下的期望回报。

### 关键设计
1. **推理策略感知的效用函数**
   - 做什么：将 BoN 和 MV 的聚合逻辑显式建模到效用函数中。
   - 核心思路：BoN 效用 $R_x^{\text{BoN}} = \max_{i \leq N} \sum_k w_k O_k + w_{K+1} \sum_i O_{K+1}$（最大化加权任务奖励减推理成本），MV 效用基于投票计数和正确答案概率。两者都是 inference-agnostic 效用的非仿射变换。
   - 设计动机：Proposition 2 证明仅在仿射变换下推理无关策略才是最优的；BoN/MV 不满足此条件，因此必须推理感知优化。

2. **PSST 算法（Prompt Scaling via Sequential Trimming）**
   - 做什么：固定预算下的臂淘汰算法，学习最优 IAPO 策略。
   - 核心思路：进行 $R = \lceil \log_2 |\mathcal{A}| \rceil$ 轮，每轮均匀分配预算。利用三种结构特性加速：(1) 跨上下文信息共享——一次拉臂的结果可用于估计所有上下文的 $Q$ 值；(2) 跨规模嵌套复用——拉 $(p, N_i)$ 可自动生成 $\lfloor N_i/N_j \rfloor$ 个 $(p, N_j)$ 的样本；(3) 非对称成本感知——只为每个 prompt 的最大存活规模分配预算。每轮按估计 $Q$ 值排序，淘汰各上下文中最差一半臂。
   - 设计动机：直接用 Sequential Halving 不利用 IAPO 结构，样本复杂度多 $O(|\mathcal{C}| N_{\max})$ 倍。PSST 的结构感知分配使误差概率以指数速率衰减。

3. **Top-K Screening 启发式**
   - 做什么：在 PSST 前用小部分预算（$\rho T$）在 $N=1$ 下快速筛选 prompt，保留 $K$ 个最优 prompt 再跑完整 PSST。
   - 核心思路：分配 $T_0 = \lfloor \rho T \rfloor$（$\rho=0.2$）均匀采样评估所有 prompt 在单次生成下的表现，每个上下文保留前 $K$ 个，剩余预算 $T' = T - T_0$ 在缩小的臂空间上运行 PSST。
   - 设计动机：低预算下全空间搜索效率低；但过激剪枝（小 $K$）可能丢弃在大 $N$ 下才优秀的 prompt（可构造反例），因此适合预算受限场景而非关键高频部署。

### 损失函数 / 训练策略
无显式损失函数——PSST 是基于采样的免超参数探索算法。训练阶段批量查询 LLM API，收集 $(x, a, \mathbf{o}_{1:N})$ 数据，通过蒙特卡洛估计 $Q^\alpha(c, a)$。支持 stockpiling（累积历史轮次数据）进一步减少外层 $R$ 因子。所有训练使用 Llama-3.3-70B-Instruct 作为黑盒 LLM，在 8×A100 GPU 上用 vLLM 生成，约 2000 GPU 小时。环境构建完后所有实验可在 CPU 上快速运行。

## 实验关键数据

### 主实验

| 环境 | 策略 $\alpha$ | $|\mathcal{P}|$ | $N_{\max}$ | $|\mathcal{C}|$ | PSST vs Uniform 提升 | PSST vs UCB 提升 |
|------|---------|------|------|------|------|------|
| Synth-Bernoulli | MV | 32 | 32 | 3 | 显著 | 显著（$p<0.05$） |
| MATH | MV | 25 | 32 | 3 | 显著 | 显著 |
| CommonsenseQA | MV | 48 | 32 | 3 | 显著 | 显著 |
| Synth-Categorical | BoN | 32 | 32 | 27 | 显著 | 显著 |
| Helpful-Harmless | BoN | 20 | 32 | 27 | 显著 | 显著 |
| Summarization | BoN | 20 | 32 | 27 | 显著 | 显著 |

在所有 6 个环境中，PSST 及 Top-K screening 在 Wilcoxon 检验下显著优于所有 baseline（$p < 0.05$），仅需 5K 推理调用即可找到良好策略。

### 消融实验

| 配置 | 相对 ACR | 说明 |
|------|---------|------|
| IAPO + PSST (Full) | 最优 | 联合优化 prompt + 推理规模 |
| TRIPLE (N=1) | -50% | 仅优化 prompt，不用推理扩展 |
| TRIPLE (N=Random) | -30% | 优化 prompt，随机分配 N |
| PSST+K1 (近似脱耦) | -25% | 先选 prompt 再调 N，在 Summarization 上尤差 |
| PSST+K4 | 接近最优 | 中等剪枝，大多数任务表现好 |
| PSST+K8 | 接近最优 | 轻度剪枝 |

### 关键发现
- 推理感知 vs 推理无关：IAPO 相比推理无关方法最高提升 50%（仅优化 prompt）和 25%（脱耦优化），验证了联合优化的必要性。
- PSST+K1 的失败模式：它会被"欺骗性 prompt"困住——这些 prompt 在 $N=1$ 下表现好但在大 $N$ 下不 scale。
- Top-K screening 在低预算时表现更好（快速收敛），但高预算下完整 PSST 更优。
- 理论保证：Theorem 1 给出 PSST 的有限预算误差概率上界，复杂度相比朴素 Sequential Halving 降低 $O(|\mathcal{C}| N_{\max})$ 倍。

## 亮点与洞察
- **Prompt-Inference 交互的理论刻画**：Proposition 2 精确给出推理无关策略最优的充要条件（仿射变换），揭示 BoN/MV 本质上不满足此条件——这是推理感知优化的理论基石，可推广到任何非线性聚合策略。
- **跨规模嵌套复用**：拉一次 $(p, N_{\max})$ 可免费得到所有 $N < N_{\max}$ 的样本，这个 trick 大幅降低预算需求，可迁移到其他需要评估不同配置的 bandit 问题。
- **免超参数设计**：PSST 无需调参，批量查询可利用 API 折扣，对实际部署非常友好。
- **上下文感知的策略学习**：IAPO 不只输出一个最优配置，而是为每个上下文（用户偏好+预算）输出对应的最优策略，实现了个性化对齐。

## 局限性 / 可改进方向
- 当前仅考虑 BoN 和 MV 两种推理策略，未涉及更复杂的 tree search 或 parallel thinking。
- Prompt 集合 $\mathcal{P}$ 需要预先给定，未与 prompt 生成/搜索方法（如 OPRO）结合。
- 上下文空间 $\mathcal{C}$ 是离散有限的，连续偏好空间需要函数逼近。
- 未考虑部署时分布漂移的影响。

## 相关工作与启发
- **vs TRIPLE-SH (Shi et al. 2024)**：TRIPLE 将 prompt 优化建模为 BAI 但推理无关、仅单目标；IAPO 引入上下文化多目标和推理感知，PSST 利用 IAPO 结构实现更高效优化。
- **vs BonBon/BOND (Gui 2024, Sessa 2025)**：这些方法通过微调将 BoN 策略蒸馏为单次解码，但需要白盒访问模型权重；IAPO 完全黑盒操作。
- **vs Inference-Aware Fine-Tuning (Chow et al. 2025)**：白盒微调方法在训练时优化 BoN 探索-利用权衡，与 IAPO 互补——可先 IAPO 选 prompt，再白盒微调。
- **vs MORL-Prompt (Jafari et al. 2024)**：多目标 prompt 优化但推理无关，不考虑 BoN/MV 的非线性聚合效应。
- **vs GenARM/DEAL (Xu et al. 2025)**：这些推理时对齐方法需要 logit 访问，不适用于纯黑盒场景。IAPO 仅需最终输出文本和外部评分器。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 推理感知的 prompt 优化是全新问题设定，理论刻画令人信服
- 实验充分度: ⭐⭐⭐⭐ 6 个环境（含 2 合成 + 4 真实任务），200 次独立运行，统计显著性检验完善
- 写作质量: ⭐⭐⭐⭐ 动机案例直观有力，理论与实验结合紧密
- 价值: ⭐⭐⭐⭐⭐ 对实际 LLM 部署中 prompt + 推理规模的联合选择有重要实用指导
