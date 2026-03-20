# Optimal Transport-Based Token Weighting for Enhanced Preference Optimization (OTPO)

**会议**: ACL 2025 / **arXiv**: [2505.18720](https://arxiv.org/abs/2505.18720) / **代码**: [GitHub](https://github.com/Mimasss2/OTPO) / **领域**: LLM对齐-偏好优化 / **关键词**: DPO, optimal transport, token weighting, preference optimization, length bias

## 一句话总结

提出 OTPO，利用无平衡最优传输（Unbalanced OT）在 chosen/rejected 回复的 token 语义表示之间建立匹配关系，从传输计划的边际分布导出每个 token 的重要性权重，替换 DPO 中的均匀权重，使偏好优化聚焦于语义关键 token，在 AlpacaEval2 上 LC WR 比 DPO 提升最高 10.9%。

## 研究背景与动机

**领域现状**：Direct Preference Optimization (DPO) 已成为 LLM 对齐的主流方法，通过直接优化 chosen/rejected 回复之间的对数似然差来隐式学习奖励。后续工作如 SimPO、SamPO、LDDPO 等针对其长度偏差问题提出了各种启发式修正。

**核心痛点**：标准 DPO 对序列中所有 token 赋予相同权重（$\omega_i = 1$），但人类评价偏好时关注的是回复中语义关键的部分（如论点质量、信息准确性），而非填充词或格式标记。这种均匀权重导致：
1. **噪声 token 主导优化**：不相关的 token 在损失中占据过多比例，干扰梯度更新方向
2. **长度偏差**：更长回复天然有更多 token 贡献损失，DPO 系统性偏向选择长回复
3. **捷径学习**：模型可能通过增加长度而非提升质量来提升奖励差

**已有方法的不足**：SimPO 用 $\omega_i = 1/|y|$ 做长度归一化；SamPO 随机采样等量 token；LDDPO 对超出公共长度的部分降权。这些方法都是启发式的，缺乏对 token 级语义重要性的建模，无法区分哪些 token 真正承载了偏好差异信息。

**核心 idea**：chosen 和 rejected 回复在回答同一问题时，语义上对应的部分（如都在表达"法国首都是巴黎"）才是偏好优化应聚焦的区域。这种跨序列的 token 语义匹配恰好是最优传输（OT）擅长解决的问题——OT 寻找两个分布之间的最小代价对齐方案，自然产生每个 token 被匹配的"质量"，即权重。

## 方法详解

### 整体框架

OTPO 在 DPO 训练流程中增加一个无梯度的 token 权重计算步骤：
1. 前向传播获取 chosen/rejected 序列每个 token 的最后一层 hidden state
2. 计算 token 对之间的欧氏距离，构造代价矩阵 $M \in \mathbb{R}^{|y_c| \times |y_r|}$
3. 求解无平衡最优传输问题，获得传输计划 $\Gamma^*$
4. 从 $\Gamma^*$ 的行/列边际分布导出 token 权重
5. 用加权对数概率比替换 DPO 中的均匀对数概率比

### 关键设计一：Token 级 DPO 分解与统一加权框架

将 DPO 的奖励差 $\Delta_r$ 分解到 token 级别：

$$\Delta_r = \sum_{i=1}^{|y_c|} \omega_c^i q_c^i - \sum_{j=1}^{|y_r|} \omega_r^j q_r^j, \quad q_*^i = \log \frac{\pi_\theta(y_*^i | x, y_*^{<i})}{\pi_{\text{ref}}(y_*^i | x, y_*^{<i})}$$

这个框架统一了现有方法：DPO 是 $\omega_i=1$；SimPO 是 $\omega_i=1/|y|$；SamPO 是随机子集上 $\omega_i=1$ 其余为 0；LDDPO 是公共长度内 $\omega_i=1$、超出部分 $\omega_i=\alpha$。OTPO 则从数据驱动地学习最优权重。

### 关键设计二：Token-Pair 级分解与 OT 的引入动机

进一步将奖励差分解到 chosen-rejected token 对级别：

$$\Delta_r = \sum_i \sum_j \Gamma_{i,j} (q_c^i - q_r^j)$$

其中 $\Gamma_{i,j}$ 是分配给 token 对 $\{y_c^i, y_r^j\}$ 的权重。token 级权重通过边际求和恢复：$\omega_c^i = \sum_j \Gamma_{i,j}$，$\omega_r^j = \sum_i \Gamma_{i,j}$。这保证了 chosen/rejected 的总权重相等（$\sum_i \omega_c^i = \sum_j \omega_r^j$），天然消除长度偏差。

### 关键设计三：代价矩阵与无平衡最优传输

**代价矩阵**：用最后一层 hidden state 的欧氏距离度量 token 间语义距离，$M_{ij} = \|h_c^i - h_r^j\|_2$。选择理由：(1) 最后一层表示包含最丰富的语义信息；(2) 欧氏距离构成合法的度量空间，满足 OT 的理论要求。

**UOT 优化目标**：

$$\Gamma^* = \arg\min_\Gamma \sum_{i,j} \Gamma_{i,j} M_{i,j} + \epsilon_1 \Omega(\Gamma) + \epsilon_2 (\text{KL}(\Gamma\mathbf{1}, \mathbf{1}_{|y_c|}) + \text{KL}(\Gamma^T\mathbf{1}, \mathbf{1}_{|y_r|}))$$

三项分别是：(1) 传输代价最小化——语义相似的 token 对获得更大传输量；(2) 熵正则 $\Omega(\Gamma) = \sum_{i,j} \Gamma_{i,j} \log \Gamma_{i,j}$——控制传输计划的稀疏性，使用 Sinkhorn 算法高效求解；(3) KL 边际松弛——允许边际偏离均匀分布，处理长度不等的情况（这是用 UOT 而非标准 OT 的核心原因）。

### 关键设计四：权重归一化

传输计划求解后，归一化到预设的总权重预算 $\tau$：

$$\omega_c^* = \frac{\Gamma\mathbf{1}}{|\Gamma|}\tau, \quad \omega_r^* = \frac{\Gamma^\top\mathbf{1}}{|\Gamma|}\tau$$

其中 $\tau = \min(|y_c|, |y_r|)$，即公共长度。这个选择确保总权重与较短回复对齐，既避免长回复获得不公平的总权重优势，也保留足够的权重预算覆盖关键语义内容。

### 损失函数 / 训练策略

最终损失仍为 DPO 形式，但用加权奖励差替换：

$$\mathcal{L}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_c, y_r) \sim D} [\log \sigma(\beta \Delta_{\hat{r}})]$$

OT 求解是 no-gradient 的（权重作为常数参与损失计算），不增加反向传播负担。OT 的时间复杂度 $O(n^2)$ 相对于 Transformer 的 $O(ln^2d + lnd^2)$ 可忽略不计。

## 实验关键数据

### 主实验：AlpacaEval2（4 个设置，2 个模型 × 2 个数据集）

| 模型 | 数据集 | 方法 | LC WR (%) | WR (%) | 长度 |
|------|--------|------|-----------|--------|------|
| Llama-3-8B-Instruct | UltraFeedback | DPO | 48.14 | 51.52 | 2168 |
| | | SimPO | 47.56 | 40.72 | 1756 |
| | | SamPO | 52.17 | 46.31 | 1806 |
| | | LDDPO | 52.10 | 51.72 | 2036 |
| | | **OTPO** | **53.37** | 47.58 | 1791 |
| Llama-3-8B-Instruct | HelpSteer2 | DPO | 27.91 | 27.45 | 1945 |
| | | LDDPO | 28.55 | 28.54 | 1956 |
| | | **OTPO** | **29.64** | **29.54** | 1991 |
| Llama-3.2-3B-Instruct | UltraFeedback | DPO | 26.02 | 27.96 | 2094 |
| | | **OTPO** | **26.97** | **28.61** | 2075 |
| Llama-3.2-3B-Instruct | HelpSteer2 | DPO | 19.99 | 20.54 | 1970 |
| | | **OTPO** | **20.50** | **21.25** | 2000 |

OTPO 在所有 4 个设置的 LC WR 上均取得最优，相对 DPO 提升 0.5%~5.2%，相对最强 baseline 提升 1.0%~3.8%。结果在 99% 置信度下显著优于其他方法。

### 消融实验（Llama-3-8B + UltraFeedback）

| 消融项 | LC WR (%) | WR (%) | 长度 | 分析 |
|--------|-----------|--------|------|------|
| DPO（基线） | 48.14 | 51.52 | 2168 | — |
| **OTPO（完整）** | **53.37** | 47.58 | 1791 | — |
| OT → 均匀权重 | 52.60 | 46.36 | 1796 | 仅长度公平，LC WR 下降 0.77 |
| OT → 余弦相似度 | 53.28 | 46.09 | 1757 | 接近但缺乏 token-pair 交互 |
| 归一化: None | 26.38 | 26.07 | 1939 | 梯度剧烈波动，性能崩塌 |
| 归一化: Mean | 52.79 | 46.69 | 1791 | 短序列权重上缩比例波动大 |
| 归一化: Max | 49.85 | 44.77 | 1808 | 同上，更严重 |
| 归一化: Length（各自长度） | 48.51 | 52.12 | 2167 | 恢复长度偏差，退化接近 DPO |

**关键发现**：(1) OT 权重比均匀权重和相似度权重都更优，说明 token-pair 级别的匹配比单 token 级别更精确；(2) 归一化策略至关重要，不归一化直接崩塌，min 归一化最稳定。

### 人工评估（50 个问题，Llama-3-8B + UltraFeedback）

| 方法 | 专家1 WR | 专家2 WR |
|------|----------|----------|
| DPO | 0.46 | 0.50 |
| SimPO | 0.56 | 0.54 |
| LDDPO | 0.56 | 0.48 |
| **OTPO** | **0.62** | **0.64** |

两位专家一致认为 OTPO 最优，尽管两人偏好相关性仅 0.37（说明偏好多样性高）。

### 摘要任务（TL;DR，Qwen-2.5-3B）

OTPO 在摘要任务上超过最强 baseline SamPO 达 8.6% win rate，因为 OTPO 自然倾向于关注关键语义内容、生成更简洁的摘要。

## 亮点与洞察

1. **统一框架的理论贡献**：将 DPO/SimPO/SamPO/LDDPO 统一为 token 加权方案的特例，OTPO 是最一般的上下文感知版本——这个视角比各个方法孤立的启发式修正更有说服力
2. **"共识即重要"的核心直觉**：chosen 和 rejected 中语义重叠的部分更可能是与问题相关的核心回答，类似多数投票原理——这个观察简单但深刻
3. **无监督权重学习**：不需要额外标注或外部模型（如 APO 需要 rewriter），仅从模型自身的表示空间中提取 token 重要性
4. **长度控制的副产品**：OTPO 通过归一化到 $\min(|y_c|, |y_r|)$ 天然消除长度偏差，生成更简洁的回复（平均长度 1791 vs DPO 的 2168）
5. **计算开销可忽略**：OT 求解的 $O(n^2)$ 时间和 $O(n^2)$ 内存相比 Transformer 的前后向传播完全可忽略

## 局限性 / 可改进方向

1. **未验证迭代 on-policy 设置**：当前仅测试了 off-policy 和一次性 on-policy，实际中迭代 on-policy 可能带来更大提升
2. **模型规模有限**：仅在 3B 和 8B 模型上验证，未扩展到更大模型（如 70B+），可扩展性待验证
3. **OT 假设的局限**：核心假设"语义相似的 token 是重要的"在某些场景可能不成立——例如 chosen 比 rejected 多了一段关键推理过程，这段内容在 rejected 中完全不存在，OT 会给它分配低权重
4. **表示空间的选择**：仅测试了最后一层 hidden state + 欧氏距离，未广泛探索中间层、CLS 表示或其他距离函数的效果
5. **评估依赖 GPT judge**：GPT-4 评估可能存在偏差，论文依赖 AlpacaEval2 单一 benchmark 的 LC WR 作为主要指标
6. **跨语言泛化**：仅在英文数据上验证，非英语场景效果未知

## 相关工作与启发

- **SimPO/SamPO/LDDPO**：本文的前置工作，从不同角度处理 DPO 的长度偏差，OTPO 将它们统一并超越
- **APO (Anchored PO)**：用外部 reviser 改写无关部分来创建最小对比数据，与 OTPO 动机相似（聚焦关键差异）但方法更重（需要额外 LLM）
- **TDPO**：通过 token 级 KL 散度隐式实现 token 加权，与 OTPO 的显式加权形成对比
- **Wasserstein GAN**：OT 在生成模型中的经典应用，启发了用 OT 度量分布差异的思路
- **AOT (Alignment via OT)**：在无配对偏好数据上用 OT 实现分布级别的对齐，与 OTPO 的 token 级 OT 互补

**对后续研究的启发**：token 级重要性建模是 DPO 改进的关键方向；OT 作为分布匹配工具在 LLM 对齐中有更广阔的应用空间（如 reward model 训练中的样本加权）。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — OT 用于 DPO token 加权是新颖的组合，统一框架有理论价值
- **实验充分度**: ⭐⭐⭐⭐ — 4 个设置主实验 + 详细消融 + 人工评估 + 效率分析，但仅限小模型
- **写作质量**: ⭐⭐⭐⭐ — 从统一框架到 OT 的引入逻辑清晰，Sankey 图可视化直观
- **价值**: ⭐⭐⭐⭐ — 提供了 DPO token 加权的原则化方法，实用且计算开销小，即插即用

### 关键发现

- OTPO 在 AlpacaEval2 上比 DPO 提升 7.7% LC WR，比最佳启发式方法 LDDPO 仍高 2.6%
- UOT 比标准 OT 更适合偏好优化——长度不等序列之间的部分传输比强制全匹配更合理
- token 权重可视化显示：OTPO 自动学会关注包含关键论点、事实信息的 token，忽略连接词和格式 token
- 随训练进行权重分布趋于稳定，说明 OT 发现了稳定的语义结构
- 框架与基础偏好优化方法正交，理论上可与 KTO/IPO 等其他方法结合

## 亮点与洞察

- **最优传输与偏好优化的理论优雅性**：OT 天然建模两个离散分布间的最优匹配，恰好对应 chosen/rejected token 之间的语义对齐——这不是强行嫁接，而是问题结构与数学工具的完美匹配
- **统一框架的解释力**：将多种启发式修正统一为 token 权重的特例，既有理论洞察又有实践指导——未来新的偏好优化方法可以直接在权重设计空间中探索
- **即插即用**：OTPO 只需在 DPO 训练循环中加一个 UOT 求解步骤，额外计算开销可控

## 局限性 / 可改进方向

- UOT 求解器（Sinkhorn 迭代）增加了每步训练的计算成本，对于超长序列可能成为瓶颈
- 代价矩阵基于当前模型的 hidden states，训练初期模型表示质量不高时权重可能不准确
- 仅在 UltraFeedback 数据集上验证，数据多样性有限
- 未探索与 RLHF/PPO 的结合——token 权重是否也能改进在线偏好优化
- 权重可视化的可解释性分析还不够深入，缺乏定量的语义对齐质量评估

## 相关工作与启发

- **vs SimPO**：SimPO 用长度归一化是 OTPO 权重的特例，无法捕捉 token 级语义差异
- **vs SamPO**：SamPO 用采样概率加权考虑了 token 的生成难度，但忽略了跨序列的语义对应关系
- **vs LDDPO**：LDDPO 显式建模长度差异，但仍是序列级调整，不是 token 级
- **vs TDPO**：TDPO 也做 token 级 DPO 但用规则定义权重，OTPO 用 OT 自动发现最优权重
- **启发**：OT 在 NLP 中的应用越来越广泛（文档匹配、跨语言对齐），偏好优化是一个新的成功应用场景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ OT 处理 token 权重有理论优雅性和统一视角
- 实验充分度: ⭐⭐⭐⭐ 效果显著，消融充分
- 写作质量: ⭐⭐⭐⭐ 理论推导与实验结合紧密
- 价值: ⭐⭐⭐⭐⭐ 统一框架 + 强效果 + 即插即用
