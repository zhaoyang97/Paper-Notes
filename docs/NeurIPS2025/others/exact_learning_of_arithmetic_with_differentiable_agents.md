# Exact Learning of Arithmetic with Differentiable Agents

**会议**: NeurIPS 2025 (MATH-AI Workshop)  
**arXiv**: [2511.22751](https://arxiv.org/abs/2511.22751)  
**代码**: [GitHub](https://github.com/dngfra/differentiable-exact-algorithmic-learner)  
**领域**: 神经符号计算 / 长度泛化  
**关键词**: differentiable FST, exact learning, arithmetic, length generalization, Turing-complete

## 一句话总结
提出可微有限状态转换器（DFST），一种图灵完备且端到端可微的模型族，在 2D 符号网格上通过观察专家算术计算的中间步骤（Policy-Trajectory Observations）训练，仅用 20 个样本（最长 3 位数加法）即可完美泛化到 3850 位二进制加法、2450 位十进制加法，未发现任何错误。

## 研究背景与动机
1. **领域现状**：长度泛化（length generalization）是算术推理的核心挑战。Transformer 在简单算术任务上难以精确泛化到更长输入（通常 2-3x），即使使用各种技巧（位置编码优化、scratchpad、index hints）也只能达到高但非完美的精确匹配精度。
2. **现有痛点**：现有计算通用模型族要么存在**计算时间爆炸**问题（RNN 无界精度、Transformer 增长上下文窗口），要么存在**梯度无信息**问题（非可微的自适应计算时间机制）。这意味着它们虽然理论上图灵完备，但实际上无法通过梯度方法有效训练。
3. **核心矛盾**：Gold 定理证明没有算法能仅从输入-输出样本精确学习递归函数类。但如果能观察计算的中间步骤，问题可以从学习递归函数简化为学习有限状态转换器——这是可行的。
4. **核心 idea**：设计一种同时满足三个条件的模型：（1）图灵完备，（2）常数精度/常数时间输出生成，（3）端到端可微。关键在于将可微控制单元与外部环境解耦——DFST 在 2D 网格上自由移动并读写符号。

## 方法详解

### 整体框架
专家 Grid Agent（手工编写的算术算法，在 2D 网格上操作）→ 生成 Policy-Trajectory Observations（观察序列：当前符号 + 动作 = 写入符号 + 移动方向）→ DFST 通过 MSE 损失进行 next-action prediction 训练 → 训练后的 DFST 在网格上自主执行算术运算。

### 关键设计

1. **Grid Agent 与符号宇宙**
   - 做什么：在 $\mathbb{Z}^2$ 无界 2D 网格上定义计算环境，Grid Agent 有有限状态集 $Q$ 和转移函数 $\delta: Q \times \Sigma \to Q \times \Sigma \times \mathcal{M}$（$\mathcal{M} = \{U,D,L,R,S\}$）
   - 核心思路：Grid Agent 每步只能感知当前位置的符号、写入一个符号、移动一格。这是 Turing 机在平面上的自然推广，因此计算通用
   - 设计动机：模拟人类老师在黑板上演算的过程——学生观察粉笔的移动和书写，但无法直接访问老师的"内部状态"

2. **可微有限状态转换器（DFST）**
   - 做什么：学习 Grid Agent 的策略，将离散的状态转移参数化为可微张量运算
   - 核心思路：DFST 由三个可训练张量 $A \in \mathbb{R}^{|\Sigma| \times d \times d}$（状态转移）、$B \in \mathbb{R}^{|\Sigma| \times |\Sigma| \times d}$（符号输出）、$C \in \mathbb{R}^{|\Sigma| \times |\mathcal{M}| \times d}$（移动输出）+ 初始隐态 $h_0$ 组成。更新规则：$h_{t+1} = A(x_t) h_t$，$\hat{s}_t = B(x_t) h_t$，$\hat{m}_t = C(x_t) h_t$。输出通过 argmax 离散化
   - **通用性定理**：任何 $d$ 状态的 Grid Agent 都能被维度为 $d$ 的 DFST 精确模拟（通过将状态转移 one-hot 编码到张量中）
   - 设计动机：线性结构使得可以用并行扫描（Blelloch scan）实现 $O(\log T)$ 并行训练，无非线性 → 损失景观更简单

3. **Policy-Trajectory Observations (PTOs)**
   - 做什么：用包含中间计算步骤的监督数据替代纯输入-输出样本
   - 核心思路：每个训练样本是完整的计算轨迹 $(x_t, s_t, m_t)_{t=0}^T$——当前观察到的符号、写入的符号、移动方向。这比 scratchpad 更结构化，因为它直接对应执行策略
   - 设计动机：Gold 定理排除了从 I/O 样本精确学习递归函数的可能性；但从 PTOs 学习可以将问题简化为学习 FST（Papazov et al. 理论保证）

### 训练策略
- 损失：MSE on next-action prediction（符号 token + 移动 token）
- 优化器：Adam + cosine annealing，无 warmup
- 初始化：状态转移矩阵 $A$ 初始化为单位矩阵（identity init），$B, C$ 初始化为 0
- 精度：float32
- 训练时长：加法 500K 迭代，乘法 3M 迭代

## 实验关键数据

### 主实验（DFST vs Neural GPU）

| 任务 | DFST 样本数 | DFST 参数量 | 训练最大位数 | Robust LG | Prob. LG | Neural GPU 样本数 |
|------|-----------|-----------|-----------|----------|---------|----------------|
| add2 | 20 | 1,020 | 3 | ≥3850 | ≥3850 | ~200K |
| add10 | 225 | 8,900 | 3 | ≥2450 | ≥2450 | N/A |
| mult2 | 750 | 5,280 | 5 | ≥600 | ≥600 | ~200K |
| mult10 | 10,000 | 162,108 | 5 | ≥180 | ≥180 | N/A |

注：所有"≥"表示受 GPU 内存限制无法测试更长输入，未发现任何计算错误。

### 关键发现
- DFST 用 20 个样本（最大 3 位数）学会完美的二进制加法，泛化到 3850 位（1283x）——这是破纪录的长度泛化
- 与 Neural GPU 对比：Neural GPU 需要 ~200K 样本且泛化高度依赖随机种子，在对称数字上会失败；DFST 在对称数字上也完美
- DFST 训练只需简单的 cosine scheduler，Neural GPU 需要课程学习、梯度噪声注入、relaxation-pull 等复杂技巧
- 更长的训练时间持续提升 RLG（Figure 2-5），未观察到饱和
- 验证精确学习本身是不可判定的（等价于判定两个图灵机等价），只能通过随机测试建立信心

## 亮点与洞察
- **中间监督的威力**：从 "不可能"（Gold 定理）到 "轻松实现" 的关键转变在于训练数据的形式——观察计算过程 vs 只看结果。这对 LLM 的 chain-of-thought 训练有深刻启示：结构化的中间步骤可能比更多的 I/O 样本有价值得多
- **环境-控制解耦**：DFST 的核心创新不在于模型本身的结构，而在于让模型能在外部环境中自由移动——这打破了 sequential model 只能单向移动的限制，是实现图灵完备且可微的关键
- **极简设计的力量**：纯线性运算 + 简单训练策略就能实现精确算法学习，无需任何花哨技巧

## 局限性 / 可改进方向
- 需要手工设计 Grid Agent（专家算法），未解决从零发现算法的问题
- DFST 隐层维度需要匹配目标 Grid Agent 的状态数，需要先验知识
- 乘法的泛化倍数（36x for mult2, 36x for mult10）远不如加法（1283x），可能因为乘法算法状态更复杂
- 仅验证了算术任务，是否能推广到排序、搜索等其他算法任务未知
- 精确学习的验证本身不可判定——无法证明模型在所有输入上都正确

## 相关工作与启发
- **vs Transformer + scratchpad**：Transformer 只能达到 2-3x 长度泛化且精度不完美；DFST 实现 1000x+ 完美泛化，关键差异在于计算通用性和训练数据形式
- **vs Neural GPU**：Neural GPU 是最接近的工作，也能在二进制算术上实现长泛化，但高度依赖随机种子且在 hard cases 上失败
- **对 LLM 的启示**：Chain-of-thought 本质上是一种 PTO——让模型观察中间推理步骤。但 Transformer 不是图灵完备的（有限精度下），这可能是其算术泛化上限的根本原因

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ DFST 的设计理念（图灵完备+可微+环境解耦）非常优雅，理论基础扎实
- 实验充分度: ⭐⭐⭐ 仅算术任务，但在该范围内结果惊艳
- 写作质量: ⭐⭐⭐⭐ 形式化框架清晰，与先前工作的对比到位
- 价值: ⭐⭐⭐⭐ 对精确算法学习的理论和实践均有重要启示
