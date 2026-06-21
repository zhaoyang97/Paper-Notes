---
title: >-
  [论文解读] Random Policy Valuation is Enough for LLM Reasoning with Verifiable Rewards
description: >-
  [ICLR 2026][Reasoning][RLVR] 作者发现数学推理的 RLVR 其实是一个"确定性转移 + 树状结构 + 二值终局奖励"的简化 MDP，在这种结构下只需评估一个**固定的均匀随机策略**的 Q 值、再按 softmax 采样，就能跳过 PPO/GRPO 那套"评估-改进"循环和一堆启发式 trick，得到既高质量（pass@1 +8.2、pass@256 +16.8）又高多样性（+20.5%）的推理策略。
tags:
  - "ICLR 2026"
  - "Reasoning"
  - "RLVR"
  - "随机策略评估"
  - "均值算子"
  - "推理多样性"
  - "数学推理"
---

# Random Policy Valuation is Enough for LLM Reasoning with Verifiable Rewards

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ujLgLz6QQa](https://openreview.net/forum?id=ujLgLz6QQa)  
**代码**: https://github.com/tinnerhrhe/ROVER  
**领域**: LLM推理 / 可验证奖励RL  
**关键词**: RLVR, 随机策略评估, 均值算子, 推理多样性, 数学推理

## 一句话总结
作者发现数学推理的 RLVR 其实是一个"确定性转移 + 树状结构 + 二值终局奖励"的简化 MDP，在这种结构下只需评估一个**固定的均匀随机策略**的 Q 值、再按 softmax 采样，就能跳过 PPO/GRPO 那套"评估-改进"循环和一堆启发式 trick，得到既高质量（pass@1 +8.2、pass@256 +16.8）又高多样性（+20.5%）的推理策略。

## 研究背景与动机
**领域现状**：用可验证奖励做 LLM 后训练（RLVR）现在几乎都建立在 PPO 及其派生（GRPO、REINFORCE++、DAPO）之上。这些方法都遵循"广义策略迭代"（GPI）范式：交替地评估当前策略的价值、再据此改进策略，循环直到收敛。

**现有痛点**：PPO 原本是为电子游戏、机器人控制这类**通用控制问题**设计的——状态转移可能随机、奖励结构复杂、状态空间是带环的图。把它搬到 LLM 推理上会水土不服：训练动态不稳定、策略熵塌缩（diversity collapse），导致探索空间越来越窄。为了缓解这些病，工程上又堆了一大堆启发式：clip、KL 正则、数据筛选……每一项都要针对具体任务小心调参，复杂度直线上升。

**核心矛盾**：GPI 的根本问题在于评估目标是**非平稳**的——策略在不停变，评估对象也跟着变，于是训练既不稳又容易熵塌。而作者注意到：数学推理任务的底层 MDP 其实比通用控制简单得多。每一步动作（生成一个 token）都确定性地展开成一个新分支，每个部分序列恰好只有一个父状态，整张可达图是一棵**树**，奖励只在终局给二值信号（答对=1，答错=0）。是不是在拿过于复杂的工具去解一个结构上更简单（只是规模大）的问题？

**本文目标**：在这个特化的树状 MDP 下，找到一个**极简但有效**的 RLVR 算法，同时保住质量和多样性，且不需要 GPI 循环和那些 trick。

**切入角度**：经典 RL 普遍认为"均值算子"（评估均匀策略）对一般控制问题是无用的——它对所有动作一视同仁地取平均，不偏向最优动作，给不出有用指引（Asadi & Littman, 2017）。但作者证明：在树状、确定性、二值终局奖励这个特化结构下，**均匀随机策略的 Q 值配上贪心选择就已经最优**。

**核心 idea**：与其往 PPO/GRPO 上加 trick，不如只评估"最简单的策略"（均匀随机），用它的 Q 值做 softmax 采样——一次策略评估就够，不需要迭代改进。

## 方法详解

### 整体框架
ROVER（Random Policy Valuation for Diverse Reasoning）把数学推理形式化为一个有限步、确定性转移、树状状态空间、二值终局奖励的 MDP $M$：状态是已生成的 token 串，动作是词表 $V$ 里的下一个 token，转移是拼接（确定性），折扣 $\gamma=1$，奖励 $r(x,y)\in\{0,1\}$ 只在生成结束时由验证器给出。

ROVER 的整条逻辑是：**先在理论上证明"评估固定均匀策略 + 贪心"即最优 → 再把贪心换成 softmax 采样以保住多样性 → 最后把抽象 Q 值用 LLM 自身参数落地成可扩展算法**。它彻底抛弃了 GPI 的"评估↔改进"循环，只保留一次对均匀策略的评估，因此没有 value network、没有 clip、没有 KL 正则、没有非平稳目标。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["输入：prompt x<br/>+ 验证器二值奖励"] --> B["随机策略评估<br/>均值算子求 Q^πu"]
    B --> C["Softmax 采样<br/>按 Q 值换贪心保多样性"]
    C --> D["LLM 内蕴 Q 参数化<br/>相对 Q = ρ(logπθ − logπθold)"]
    D --> E["低方差奖励<br/>组内中心化 + 广播到每 token"]
    E -->|回归 sg[Q̂] 更新 θ| B
    E --> F["输出：高质量 + 高多样性推理策略"]
```

### 关键设计

**1. 随机策略评估：用均匀策略的 Q 值直接贪心就能拿到最优**

这一步直击"GPI 非平稳目标导致不稳 + 熵塌"的痛点。作者从最简单的均匀随机策略 $\pi_u(a|s)=1/|A|$ 出发，用带**均值算子**的广义 Bellman 更新估它的 Q 值；在确定性转移、$\gamma=1$ 下更新简化为 $\hat{Q}^{\pi_u}(s,a)\leftarrow r(s,a)+\frac{1}{|A|}\sum_{a'\in A}\hat{Q}^{\pi_u}(s',a')$。Theorem 1 证明：在确定性、树状、二值终局奖励（$R(s)\in\{0,R\},R>0$）的 MDP 里，贪心策略 $\pi_{\text{greedy}}(s)=\arg\max_a Q^{\pi_u}(s,a)$ **就是最优策略**。

为什么这个本被认为"无用"的均值算子在这里突然有效？因为 $Q^{\pi_u}(s,a)$ 在这个结构下有一个清晰含义：**它等于在 $s$ 处采取 $a$、之后一路均匀随机走到终局、最终答对的概率**。$Q^{\pi_u}(s,a)=0$ 说明从 $(s,a)$ 出发没有任何后续能到正确解；值越大说明这个分支里"通往成功的路径越密"。于是按 Q 值贪心，等价于剪掉所有死路、优先走最有希望的分支。整个过程只需要对一个**固定**策略做一次评估，没有 off-policy 修正、没有迭代改进，也就根本不存在非平稳目标的问题。

**2. Softmax 采样：把贪心换成软采样，在保性能的前提下救回多样性**

纯贪心虽然最优，但是确定性的，会塌缩到单一模式（图 5(e)：Q-learning 和 ROVER-greedy 都只覆盖一个最优解），而推理任务需要多样性来撑起 pass@k 和泛化。作者利用"$Q^{\pi_u}$ 正比于成功概率"这一性质，把贪心换成 softmax 采样：

$$\pi_s(a|s)=\frac{\exp(Q^{\pi_u}(s,a)/\rho)}{\sum_{a'}\exp(Q^{\pi_u}(s,a')/\rho)}$$

这样动作被按"估计成功概率"比例选中，能同时探索多条有效路径，而不是一条道走到黑；并且 softmax 天然契合现代 LLM 解码框架，易于工程接入。Theorem 2 给出了性能保证：$V^{\pi_s}(s_0)\ge R\left(1-\sum_{s\in P}\Pr_{\pi_s}(s|s_0)\frac{N(s)}{N(s)+\exp(\max_a Q^{\pi_u}(s,a)/\rho)}\right)$，其中 $N(s)$ 是零值动作数、$P$ 是同时存在最优/次优动作的关键状态集。温度 $\rho$ 就是质量-多样性的旋钮：$\rho\to0$ 退回贪心、与最优策略的差距消失，$\rho$ 越大采样越多样。玩具实验里 $\rho=1$ 能同时覆盖全部 4 个最优模式且保持 100% 成功率。

**3. LLM 内蕴 Q 参数化：把抽象 Q 直接挂到策略参数上，省掉 value network 还稳住训练**

理论很漂亮，但 LLM 的状态/动作空间巨大、horizon 很长（又宽又深的树），从头训一个 Q 网络代价太大。作者注意到 Q 值和策略可以通过 $\rho\log\pi_\theta(a|s)$ 内蕴地联系起来（它刻画了同一状态下对各动作的相对偏好），于是**直接用 LLM 自身参数 $\theta$ 表示 Q 函数**，不再需要单独的 value 网络。

但直接用 $\rho\log\pi_\theta$ 作 Q 学习目标会漂移、易发散。为此作者引入**相对 Q 函数**——只衡量相对一个固定基线的改进：$Q(s_t,a_t)=\rho\big(\log\pi_\theta(a_t|s_t)-\log\pi_{\theta_{old}}(a_t|s_t)\big)$，其中 $\pi_{\theta_{old}}$ 是每个 epoch 用来采样的行为策略，充当稳定锚点。这让初始 Q 值居中在 0，模型学的是"相对上一版策略的变化"而非绝对值，从而抑制了 Q 更新的波动。

**4. 低方差奖励：组内中心化 + 广播到每个 token，让稀疏奖励变稠密又稳**

二值终局奖励信号稀疏、方差高，难以稳定地估 Q。作者对每个 prompt 采样 $n$ 个回答，把组内平均奖励减掉，得到中心化奖励 $\tilde{r}(x,y_i)=r(x,y_i)-\frac{1}{n}\sum_{i=1}^n r(x,y_i)$（类似 GRPO 的优势估计但**去掉了标准差归一化**这一项）。这降低了估计方差、丰富了对价值地形的采样。同时为了给长推理链做信用分配，把这个中心化奖励**广播到生成的每一个 token**，提升训练效率。最终损失是把参数化 Q 回归到目标 $\hat{Q}$（stop-gradient）：$L_{\text{ROVER}}=\frac{1}{\sum_i|y_i|}\sum_i\sum_t\|Q(a_t|s_t),\text{sg}[\hat{Q}(a_t|s_t)]\|^2$。

### 一个例子：玩具树状 MDP
作者设计了一个 tabular 环境验证理论：从空状态出发，每步从 $\{A,B,C,D\}$ 选一个动作拼到序列后，只有 4 个特定终局（ACD、BDC、CAB、DBA）奖励为 1。结果（图 5）很说明问题：标准 Q-learning（ε-greedy）虽然最优但只收敛到单一模式 ACD；ROVER-greedy 给最优动作赋了最高 Q 值，但因贪心也只塌缩到单一模式 BDC；而 ROVER（$\rho=1$）给**全部 4 个最优动作赋了同样高的 Q 值**，成功覆盖全部 4 个模式且保持 100% 成功率。这把"贪心保最优、softmax 保多样"的取舍可视化得很清楚。

### 损失函数 / 训练策略
每个 epoch：固定 $\pi_{\theta_{old}}\leftarrow\pi_\theta$ 并采样一批 prompt；对每个 prompt rollout $n$ 个回答算中心化奖励 $\tilde{r}$；对每个状态算 $Q(a_{t+1}|s_{t+1})=\rho(\log\pi_\theta-\log\pi_{\theta_{old}})$，再回填目标 $\hat{Q}(a_t|s_t)\leftarrow\tilde{r}+\frac{1}{|V|}\sum_{a_{t+1}\in V}Q(a_{t+1}|s_{t+1})$；最后用 AdamW 优化上面的 MSE 损失。全程没有 clip、KL 正则、value 网络；温度 $\rho=1$ 全任务统一、不做任务特定调参。

## 实验关键数据

### 主实验
在 Qwen3-4B/8B-Base 上训练于 DeepScaler 数据集，pass@1 跨数学与 O.O.D 基准对比（节选 Qwen3-8B-Base）：

| 方法 | AIME24 | AIME25 | HMMT25 | OlympiadBench | AMC23 | MATH500 | GPQA-d | Avg. |
|------|--------|--------|--------|---------------|-------|---------|--------|------|
| Base | 11.5 | 8.8 | 0.8 | 34.7 | 48.1 | 68.8 | 29.1 | 28.8 |
| GRPO | 16.8 | 15.1 | 4.8 | 48.6 | 66.9 | 81.9 | 43.8 | 39.7 |
| DAPO | 20.8 | 15.2 | 3.6 | 49.0 | 67.9 | 84.3 | 46.6 | 41.1 |
| REINFORCE++ | 19.4 | 16.7 | 7.1 | 47.6 | 63.5 | 83.6 | 46.3 | 40.6 |
| **ROVER** | **30.6** | **22.7** | **14.6** | **56.4** | **74.8** | **89.6** | **50.2** | **48.4** |

ROVER 在所有模型规模上一致超过最强基线（8B 平均 pass@1 比次优高 +7.3，AIME 子集 +8.2）；越难的任务优势越大——AIME24 相对最优基线 +47.1%、AIME25 +35.9%、HMMT25 几乎翻倍。O.O.D 的 GPQA-diamond 上也最优，说明泛化能力强。

### 消融实验

| 配置 | 现象 | 说明 |
|------|------|------|
| ROVER ($\rho=1$) | 质量+多样性 Pareto 前沿最优 | 全实验统一温度，无需调参 |
| $\rho=3$ | 收敛慢、欠利用 | 温度过高过度探索 |
| $\rho=0.1$ | 熵过早塌缩、探索受限 | 温度过低提前利用 |
| $\rho=0.001$ | 近确定性采样→训练严重不稳 | 极端贪心，test score 崩 |
| GRPO w/o KL | 熵塌缩 | 基线代表性失败模式 |
| GRPO w/ KL=0.01 | pass@1 明显更差 | 过度正则反而受害 |

### 关键发现
- **高熵是性能主驱动**：ROVER 的熵随训练优雅衰减但始终显著高于基线（基线要么塌缩、要么剧烈震荡），持续探索是它同时拿下质量和多样性的根本原因。
- **pass@k 不饱和**：基线在 pass@1 提升后很快饱和、大 $k$ 时甚至跌破 base model（DAPO 在 AIME25 $k>4$ 就变差）；ROVER 随 $k$ 增大持续上扬，pass@256 比最强基线高 +16.8，最难的 HMMT25 上仍在加速。
- **多样性最高**：用 NoveltyBench 的"distinct strategies"指标，ROVER 相对 GRPO +6.8%、相对三基线均值 +20.5%；Countdown 任务上对同一问题找出 17 种不同解法，而 GRPO 只有 3 种。
- **test-time 扩展最好**：maj@k 随 $k$ 稳健上升，基线因模式塌缩而自信地收敛到相似错误解、加样本也涨不动。

## 亮点与洞察
- **把"问题结构"当一等公民**：不在 PPO 上继续加 trick，而是退一步问"这个 MDP 到底长什么样"，发现树状+确定性+二值奖励的结构让被judged为"无用"的均值算子重新生效——是结构洞察驱动算法简化的范例。
- **Q 值的概率解释非常优雅**：$Q^{\pi_u}(s,a)$ = 从此处随机走到底答对的概率，让"贪心剪死路"有了直白的几何/概率直觉，也直接导出 softmax 采样的合理性。
- **复用 LLM 自身参数化 Q + 相对基线锚点**：省掉 value 网络的同时用 $\pi_{\theta_{old}}$ 当锚把 Q 居中在 0，这个"学相对改进而非绝对值"的技巧可迁移到其他需要稳定 critic 的 RLVR 设定。
- **多样性是"免费的副产品"而非靠奖励工程**：相比那些靠复杂任务相关奖励工程或事后采样硬凑多样性的方法，ROVER 的多样性直接来自 softmax + 高熵保持，且带性能保证。

## 局限与展望
- **理论强依赖特化结构**：最优性证明严格依赖确定性转移、树状、二值终局奖励。一旦奖励是连续/中间奖励、或状态图带环（非树），Theorem 1 不再成立，方法适用范围受限。
- **均值算子的"概率解释"依赖均匀随机后验**：现实中 token 空间巨大，$\frac{1}{|V|}\sum_{a'}$ 对整个词表求平均只是近似，这个近似在极长 horizon 下的误差累积没有充分讨论。
- **温度 $\rho$ 虽统一设 1 但敏感**：消融显示 $\rho$ 偏离 1 会明显伤性能（尤其 $\rho=0.001$ 直接崩），统一取 1 在更多样的任务/模型上是否依旧稳健仍需验证。
- **主要在数学/Countdown 上验证**：虽有 GPQA-diamond 的 O.O.D 结果，但更广义的可验证奖励任务（代码、定理证明等）上的表现尚待考察。

## 相关工作与启发
- **vs GRPO/PPO/DAPO**：它们走 GPI 的"评估↔改进"循环，靠 clip/KL/数据筛选稳住训练，目标非平稳易熵塌；ROVER 只评估固定均匀策略一次，无这些 trick，从根上避开非平稳，质量和多样性双赢。其中心化奖励像 GRPO 的优势估计但去掉了标准差归一化。
- **vs Q-learning（ε-greedy）**：两者在玩具 MDP 上都能拿最优，但 Q-learning 因贪心塌缩到单一模式；ROVER 用 softmax 在保最优性的前提下覆盖全部最优模式。
- **vs 经典"均值算子无用论"（Asadi & Littman, 2017）**：经典 RL 认为均值算子不适合一般控制；本文首次给出 LLM 数学推理这一特化场景下均值算子配贪心即最优的**理论**证明，把此前仅有的经验观察（Laidlaw et al., 2023; He et al., 2025b）落到理论基础上。
- **vs 多样性 RL 方法**：传统方法靠复杂奖励工程或事后采样技巧凑多样性、且无保证；ROVER 的多样性是 softmax over $Q^{\pi_u}$ 的自然结果，带 Theorem 2 的性能下界保证，且实现极简。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用"问题结构简化"重新激活均值算子，并给出 LLM 推理场景下的首个最优性证明，视角独到。
- 实验充分度: ⭐⭐⭐⭐ 多模型多基准、pass@1/pass@k/maj@k/多样性/O.O.D 都覆盖，但主战场集中在数学类任务。
- 写作质量: ⭐⭐⭐⭐⭐ 理论—直觉—工程三层递进清晰，玩具例子把核心取舍可视化得很到位。
- 价值: ⭐⭐⭐⭐⭐ 极简却强，挑战"RLVR 必须复杂"的默认认知，对后训练方法设计有方向性启发。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Reference-guided Policy Optimization for Molecular Optimization via LLM Reasoning](reference-guided_policy_optimization_for_molecular_optimization_via_llm_reasonin.md)
- [\[ICLR 2026\] Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning](slow-fast_policy_optimization_reposition-before-update_for_llm_reasoning.md)
- [\[ICLR 2026\] Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning](temperature_as_a_meta-policy_adaptive_temperature_in_llm_reinforcement_learning.md)
- [\[ICLR 2026\] Asymmetric Proximal Policy Optimization: Mini-Critics Boost LLM Reasoning](asymmetric_proximal_policy_optimization_mini-critics_boost_llm_reasoning.md)
- [\[ICLR 2026\] On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning](on_the_design_of_kl-regularized_policy_gradient_algorithms_for_llm_reasoning.md)

</div>

<!-- RELATED:END -->
