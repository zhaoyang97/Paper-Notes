# What One Cannot, Two Can: Two-Layer Transformers Provably Represent Induction Heads on Any-Order Markov Chains

**会议**: NeurIPS 2025  
**arXiv**: [2508.07208](https://arxiv.org/abs/2508.07208)  
**代码**: [GitHub](https://github.com/cekbote/two-layer-induction-heads)  
**领域**: LLM NLP / Transformer理论 / 上下文学习  
**关键词**: induction heads, in-context learning, Markov chains, transformer depth, expressiveness

## 一句话总结
理论证明两层单头 Transformer 足以表示任意 $k$ 阶马尔可夫过程的条件 $k$-gram 模型（即 $k$ 阶 induction head），给出了 Transformer 深度与马尔可夫阶数关系的最紧已知刻画，关键在于利用 MLP 中的 ReLU 和 LayerNorm 非线性来补偿减少的层数。

## 研究背景与动机
1. **领域现状**：In-context learning (ICL) 是 Transformer 的核心能力，通过输入上下文中的 induction head 电路实现"复制-匹配"机制。理论上，ICL 可建模为在马尔可夫序列上学习条件 $k$-gram
2. **现有痛点**：
   - 一层单头 Transformer 无法高效表示一阶 induction head（除非隐藏维度指数级大）[Sanford et al.]
   - 两层单头可处理一阶 [Bietti, Edelman, Nichani]，但高阶需要头数随 $k$ 线性增长 [Nichani]
   - 已知最佳结果：三层单头可处理任意 $k$ 阶 [Rajaraman et al.]
   - **遗留问题**：两层单头能否处理任意 $k$ 阶？
3. **核心矛盾**：三层构造用三个注意力层分别处理"当前上下文"、"前移上下文"和"匹配"三步操作；减到两层时，一个注意力层要同时完成两步操作，信息量不够
4. **切入角度**：注意到 MLP（含 ReLU + LayerNorm）可以从注意力输出中反向提取缺失的信息——让 MLP 承担第二个注意力头的角色
5. **核心idea一句话**：用 MLP 的非线性（ReLU 过滤 + LayerNorm 归一化）从第一层注意力输出中分离关键 token 嵌入，替代三层构造中的第二个注意力操作

## 方法详解

### 整体框架
输入是 $k$ 阶马尔可夫链生成的序列 $s_{0:T}$，目标是预测下一个 token 的分布（即条件 $k$-gram 估计器）。构造 2 层 1 头 Transformer，其中第一层注意力 + MLP 计算两个关键表征 $\mathbf{u}_n$ 和 $\mathbf{v}_n$，第二层注意力作为 $k$ 阶 induction head 完成匹配。

### 关键设计

1. **第一层注意力：提取前移上下文 $\mathbf{v}_n$**
   - 做什么：对位置 $n$，注意力集中在 $\{n-1, n-2, \ldots, n-k\}$，计算前移上下文表征
   - 核心思路：注意力权重 $\operatorname{attn}_{n,i} = 3^i / \sum_{j=1}^k 3^j$，得到 $\mathbf{v}_n = \frac{\sum_{i=1}^{k} 3^i \cdot e_{x_{n-i}}^S}{\sum_{i=1}^{k} 3^i}$
   - 设计动机：$3^i$ 的指数加权让不同位置的 one-hot 编码可以被唯一分离（类似进制编码）

2. **MLP 非线性重构：提取当前上下文 $\mathbf{u}_n$**
   - 做什么：从 $\mathbf{v}_n$ 中分离出 $e_{x_{n-k}}^S$（最远位置的 token），结合跳跃连接中的当前 token $e_{x_n}^S$，重构当前上下文
   - 核心思路：
     - 将完整词表的 one-hot 编码嵌入 MLP 权重矩阵
     - ReLU + 适当 bias 过滤，只保留与 $x_{n-k}$ 对齐的分量
     - LayerNorm 归一化，得到干净的 $e_{x_{n-k}}^S$
     - 线性组合 $\mathbf{u}_n = \frac{e_{x_n}^S}{\sum 3^i} + \frac{3 e_{x_{n-k}}^S \mathbf{v}_n}{\sum 3^i} - \frac{3^k e_{x_{n-k}}^S}{\sum 3^i}$
   - 设计动机：**这是论文最核心的贡献**——三层构造用第二个注意力头获取 $\mathbf{u}_n$，本文证明 MLP 的非线性组件足以替代，不需要额外的注意力操作

3. **第二层注意力：$k$ 阶 Induction Head**
   - 做什么：在位置 $T$，通过余弦相似度 $\langle \mathbf{u}_T, \mathbf{v}_n \rangle / (\|\mathbf{u}_T\| \|\mathbf{v}_n\|)$ 匹配历史中上下文相同的位置
   - 核心思路：当 $\mathbf{u}_T = \mathbf{v}_n$ 时相似度为 1（意味着位置 $n$ 之前的 $k$ 个 token 与当前位置完全匹配），softmax 温度趋于无穷时逼近条件 $k$-gram
   - 设计动机：与三层构造的第三层相同，只是输入的 $\mathbf{u}_n$ 和 $\mathbf{v}_n$ 来源不同

### 梯度下降分析（简化一阶情况）
- 对一阶马尔可夫链，分两阶段训练：
  - **阶段1**：训练位置编码标量 $\mathbf{p}$，学习正确的注意力模式
  - **阶段2**：冻结位置编码，训练注意力缩放标量 $a_2$，使 induction head 收敛
- 证明交叉熵损失在 $T_1 + T_2$ 步后收敛到最优值附近

## 实验关键数据

### 注意力模式验证（$k=3$ 阶马尔可夫链）

| 层 | 观察到的注意力模式 | 理论预期 |
|---|---|---|
| Layer 1 | 注意力权重在前 $k=3$ 条下对角线单调递增，超过后骤降至 0 | $\operatorname{attn} \propto 3^i$ 在 $i \leq k$ |
| Layer 2 | 注意力集中在向前 $k$ 个 token 匹配当前上下文的位置 | 逼近条件 $k$-gram 的理想模式 |

### 参数效率对比

| 方法 | 层数 | 头数/层 | 参数量级 |
|------|------|--------|---------|
| Nichani et al. | 2层 | $k$头 | $(k+(k+1)^2)(S+T)^2$ |
| Rajaraman et al. | 3层 | 1头 | $15(6S+3)^2 + (6S+3)(3T+2S+10)$ |
| **本文** | **2层** | **1头** | $9(6S+3)^2 + (6S+3)(2T+2S+9)$ |

- 本文在 $k$ 和 $T$ 同时增大时参数最紧凑
- 比特精度要求 $\Omega(\log T + k)$，逼近误差 $\mathcal{O}(1/T)$

### 关键发现
- **单层不可能**：实验验证单层 Transformer 在相同参数量下无法解决 induction head 任务
- **非线性是关键**：去掉 ReLU 或 LayerNorm 后构造失败，证明 MLP 非线性不仅是辅助的
- **鲁棒性**：对输入序列添加噪声后两层模型仍然工作

## 亮点与洞察
- **MLP 的新角色**：传统认为 Transformer 中 MLP 主要做特征变换，本文证明它可以替代整个注意力层——通过非线性操作从混合表征中分离单个 token 嵌入
- **深度-宽度权衡的精确刻画**：2层2头 vs 3层1头 vs 2层1头，形成完整的帕累托前沿
- **$3^i$ 进制编码技巧**：用指数加权让 one-hot 编码的混合体可唯一解码，这个技巧可迁移到其他需要在固定维度中编码变长序列信息的场景

## 局限性 / 可改进方向
- 梯度下降分析仅覆盖一阶情况，高阶情况的学习动态仍未解决
- 构造是存在性证明（constructive），但实际训练出的网络权重不一定采用这种构造
- 仅验证了马尔可夫序列，真实自然语言的分布复杂得多
- 理论要求比特精度 $\Omega(\log T + k)$，实际浮点精度是否足够需要更多分析

## 相关工作与启发
- **vs Rajaraman et al. (3层1头)**：本文将层数从 3 降到 2，关键贡献是证明 MLP 可以补偿一个注意力层
- **vs Nichani et al. (2层k头)**：本文将头数从 $k$ 降到 1，用 MLP 非线性替代多头并行处理
- **vs Sanford et al. (1层下界)**：本文结果与 1 层不可能性结合，完整说明了 2 层是表示 induction head 的最小深度
- 对模型压缩有启发：如果注意力头可以被 MLP 替代，能否在推理时用更少的注意力头+更强的 MLP？

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将深度从3降到2的紧致性结果，MLP替代注意力头的洞察新颖
- 实验充分度: ⭐⭐⭐ 主要是理论工作，实验为验证性质，覆盖面有限
- 写作质量: ⭐⭐⭐⭐⭐ 理论展开清晰，证明思路直观易懂
- 价值: ⭐⭐⭐⭐ 对理解Transformer表达能力和ICL机制有重要贡献
