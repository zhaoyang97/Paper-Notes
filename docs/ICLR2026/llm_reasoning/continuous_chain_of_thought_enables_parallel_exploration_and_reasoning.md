# Continuous Chain of Thought Enables Parallel Exploration and Reasoning

**会议**: ICLR 2026  
**arXiv**: [2505.23648](https://arxiv.org/abs/2505.23648)  
**代码**: https://github.com/alperengozeten/CoT2  
**领域**: LLM推理 / 模型压缩  
**关键词**: 连续思维链, 并行推理, 多轨迹追踪, GRPO, 信息论

## 一句话总结
CoT2 提出用连续值 token（词表 embedding 的凸组合）替代离散 token 进行链式推理，使模型能在单次推理中并行追踪多条推理路径，理论证明等价于 K 次 self-consistency/best-of-N 采样，并通过 GRPO 强化学习进一步提升性能。

## 研究背景与动机

1. **领域现状**：现代 LLM 的 CoT 推理通过自回归采样离散 token 实现，配合 self-consistency（多次采样取多数投票）或 best-of-N 解码来提升准确率。
2. **现有痛点**：
   - 离散采样每步最多传递 $\log_2(v)$ 比特信息，而每个 token embedding 可存储 $O(d)$ 比特——信息利用严重不足
   - 一旦采样某个 token，模型就"承诺"了某条推理路径，无法探索替代方案
   - self-consistency/best-of-N 需要多次前向传播，推理成本线性增长
3. **核心矛盾**：离散采样的决策不可逆性导致单条推理链容易"滚雪球"累积错误，而弥补手段（多次采样）又带来巨大计算开销
4. **本文要解决什么？**
   - 如何让模型在单次推理中同时追踪多条推理路径？
   - 连续 token 的并行追踪能力有多强？与离散多次采样有何理论关系？
   - 如何训练和推理连续 token 模型？
5. **切入角度**：将 LM 在每步的 softmax 输出不进行离散采样，而是直接作为连续 token（所有词表 embedding 的加权组合）送入下一步。这个"叠加态"自然编码了多条路径的信息。
6. **核心idea一句话**：连续 token 是词表 embedding 的凸组合，天然实现并行路径追踪，其效果理论上等价于 K 条独立离散 CoT 的聚合——一次前向传播顶 K 次采样。

## 方法详解

### 整体框架
给定输入 $\bm{X}$，模型自回归生成 $m$ 个 token：前 $m-1$ 步输出连续 token $\bm{z}_t = \bm{E}^\top \bm{\alpha}_t$（softmax 分布与 embedding 矩阵的乘积），最后一步采样离散答案 token。训练分为 CSFT（连续监督微调）和 GRPO-based RL 两阶段。

### 关键设计

1. **连续监督微调 (CSFT)**:
   - 做什么：用"多轨迹叠加"作为中间步的监督信号
   - 核心思路：给定 Budget $B$ 条最优轨迹，在每个中间步 $t$ 的监督分布 $\alpha_{t,g}^* = \frac{1}{B}\sum_{\pi \in \Pi_B} \mathbf{1}\{g_t(\pi)=g\}$——即 $B$ 条轨迹在步 $t$ 经过的状态的经验分布。最终步用 one-hot（正确答案）。用交叉熵/KL 散度训练模型拟合这些软标签。
   - 设计动机：$B=1$ 退化为离散 CoT（one-hot）；$B=|\mathcal{T}|$ 追踪所有轨迹（最大并行）。Budget 提供了并行度和模型容量之间的灵活控制。

2. **Budget-Embedding Dimension 权衡**:
   - 做什么：量化并行度与 embedding 维度的理论关系
   - 核心思路：信息论下界 $d = \Omega(B\log(v/B))$，即要可靠解码 $B$ 条轨迹的叠加，embedding 维度需要 $\Omega(B\log(v/B))$。当 $d$ 足够大时，增大 $B$ 单调提升性能；当 $d$ 不够时，存在最优 $B$ 的 sweet spot。
   - 设计动机：解释了为什么 $d=16$ 时 $B=8$ 优于 $B=16$（容量不足），而 $d=32$ 时 $B=16$ 最优。

3. **单层 Transformer 构造 (Proposition 1)**:
   - 做什么：证明单层 Transformer 可用 CoT2 解决 MNNS（最小非负和）问题
   - 核心思路：用三角函数 embedding 将所有 $2^k$ 个状态编码在不重叠的（sin, cos）表示中，注意力层扩展状态（加减新数字），MLP 层读取和过滤。每步并行追踪指数增长的状态数，最终步选出最小非负和。
   - 设计动机：MNNS 本质是子集和问题，需要搜索 $2^m$ 种可能——离散 CoT 必须"选择"一条路径，而 CoT2 可以同时追踪所有路径。

4. **Multi-Token Sampling (MTS) + GRPO**:
   - 做什么：为 CoT2 引入可控的随机性，使 RL 方法可用
   - 核心思路：每步采样 $K$ 个离散 token 并平均：$\bm{z}_t = \frac{1}{K}\sum_{r=1}^K \bm{e}_{i_r}$。这给出了 $\bm{\alpha}_t$ 的无偏但有噪估计。Proposition 3 证明 MTS 的估计误差等价于 $K$ 条独立离散 CoT 的聚合，即样本复杂度降低 $K$ 倍。
   - 设计动机：Base CoT2 是确定性的（无随机性），无法直接计算 policy ratio 用于 GRPO。MTS 引入可控噪声，使 GRPO 的 policy ratio $r_t^{(i)}(\theta)$ 可以定义和计算。

### 损失函数 / 训练策略
- **CSFT 阶段**：$\mathcal{L}_{CSFT} = \sum_{t=1}^m D(\bm{\alpha}_t^* \| \bm{\alpha}_t)$，中间步用软标签的交叉熵，最终步用标准 CE
- **GRPO 阶段**：标准 GRPO clipped surrogate + KL 正则化，稀疏奖励（正确=1，错误=0）
- Teacher forcing 用于 CSFT（即使推理时是自回归的），效果优于 self-feeding

## 实验关键数据

### 主实验（MNNS 任务，4 位数字 1-99）

| 方法 | d=16 acc | d=24 acc | d=32 acc |
|------|---------|---------|---------|
| No-CoT | ~15% | ~15% | ~15% |
| Discrete CoT (B=1) | ~55% | ~70% | ~75% |
| COCONUT | ~45% | ~60% | ~65% |
| **CoT2 (B=16)** | ~60% | **~95%** | **~98%** |

### Pass@k 比较（d=24, MNNS）

| 方法 | Pass@1 | Pass@4 | Pass@8 | Pass@16 |
|------|--------|--------|--------|---------|
| Discrete CoT | ~70% | ~82% | ~88% | ~93% |
| **CoT2** | **~95%** | ~96% | ~97% | ~98% |

### 关键发现
- **CoT2 单次推理 ≈ 离散 CoT 多次采样**：CoT2 的 Pass@1 就达到离散 CoT Pass@16 的水平
- **Budget-Dimension 甜蜜点存在**：$d=16$ 时 $B=8$ 最优（$B=16$ 太多容量不够），$d=32$ 时 $B=16$ 最优
- **GRPO 在 CoT2 上有效**：RL 微调使模型学会优先追踪相关推理路径，降低连续 token 的熵
- **CoT2 比 COCONUT 更好**：有外部搜索监督信号时，直接拟合多轨迹分布比隐状态替换更有效
- **理论与实验高度一致**：$d=\Omega(B\log(v/B))$ 的下界在实验中被验证

## 亮点与洞察
- **信息论视角的深刻洞察**：离散 token 每步最多 $\log_2 v$ 比特，而连续 token 可以打包 $B \cdot \log_2(v/B)$ 比特——这个信息论论证非常优雅地解释了为什么连续 token 更强大。
- **"一次前向 ≈ K 次采样"的理论保证 (Proposition 3)** 是非常有力的结果——直接将 CoT2 与 self-consistency 建立了量化等价关系，赋予了连续 token 清晰的实际意义。
- **将 RL 扩展到连续动作空间用于 LLM**：传统 GRPO/PPO 在离散 token 空间操作，CoT2 的 MTS 策略巧妙地通过"采样+平均"在连续空间中引入可控噪声，使 policy gradient 方法可用。

## 局限性 / 可改进方向
- 仅在合成任务（MNNS、ProntoQA、ProsQA）上验证，未在真实 NLP 任务或大规模 LLM 上测试
- Assumption 1（Markov 性 + 线性叠加）在实际 Transformer 中可能不严格成立
- 连续 token 无法直接解读为自然语言，丧失了 CoT 的可解释性
- 词表 embedding 矩阵 $\bm{E}$ 的正交性会影响叠加质量，实际中 embedding 可能高度相关
- 只有最后一步输出离散 token，如果需要多步离散输出（如长答案），需要扩展框架

## 相关工作与启发
- **vs COCONUT**: 同为连续思维链，但 COCONUT 用 LLM 的隐状态替换，没有显式的多轨迹监督；CoT2 通过 CSFT 直接拟合轨迹分布，效果更好
- **vs Self-Consistency**: self-consistency 需要 K 次采样取多数投票；CoT2 理论上一次前向等价于 K 次采样，推理效率提升 K 倍
- **vs Latent Reasoning (Coconut/Quiet-STaR)**: 这些方法着重于将推理内化到隐空间，但缺乏 CoT2 的信息论保证和多轨迹并行的明确形式化

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 信息论驱动的连续推理+并行追踪，理论贡献扎实
- 实验充分度: ⭐⭐⭐ 仅限合成任务，缺乏真实 LLM 规模验证
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，直觉解释到位，图表设计精美
- 价值: ⭐⭐⭐⭐ 对连续推理的理论理解贡献卓越，但实用性有待验证
