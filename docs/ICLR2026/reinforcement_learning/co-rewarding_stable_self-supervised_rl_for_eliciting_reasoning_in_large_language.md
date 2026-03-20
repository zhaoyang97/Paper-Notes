# Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in Large Language Models

**会议**: ICLR 2026  
**arXiv**: [2508.00410](https://arxiv.org/abs/2508.00410)  
**代码**: https://github.com/tmlr-group/Co-rewarding  
**领域**: LLM推理 / 模型压缩  
**关键词**: 自监督RL, 无标签推理, 训练崩溃, GRPO, 对比学习

## 一句话总结
Co-rewarding 提出自监督 RL 框架，通过数据侧（对比改写问题的跨视角一致性）和模型侧（EMA 教师模型提供伪标签）两种互补监督方式，解决自奖励 RL 中的训练崩溃问题，在无人工标签条件下多项数学推理基准上达到甚至超过 RLVR（有标签）的性能。

## 研究背景与动机

1. **领域现状**：RLVR（带可验证奖励的强化学习，如 DeepSeek-R1 的 GRPO）是提升 LLM 推理能力的主流方法，但依赖人工标注的 ground-truth 答案作为奖励信号。
2. **现有痛点**：
   - GT 标注成本高、不易扩展，尤其对复杂任务
   - 自奖励方法（self-certainty、entropy-based、majority voting）可替代 GT，但频繁出现**训练崩溃**
   - 崩溃原因：奖励信号来自模型自身的单一视角输出 → 形成"自一致幻觉" → reward hacking
3. **核心矛盾**：自监督信号与当前策略纠缠——模型通过最小化熵或最大化一致性获得高奖励，但实际只是收敛到了平凡解（重复字符串、一致但错误的答案）
4. **本文要解决什么？**
   - 如何在不使用 GT 标签的情况下获得稳定的 RL 训练？
   - 如何打破"单一视角"下的自一致幻觉？
   - 能否达到有 GT 标签的 RLVR 水平？
5. **切入角度**：受自监督学习（SimCLR、BYOL、DINO）启发——真正的推理能力应体现为跨视角/跨时间的不变性（invariance），而非单一输出的确定性。
6. **核心idea一句话**：通过数据侧的"改写问题交叉验证"和模型侧的"EMA 教师伪标签"引入互补监督视角，增加 reward hacking 的难度从而防止训练崩溃。

## 方法详解

### 整体框架
基于 GRPO 优化，核心创新在于修改 advantage 估计中的奖励来源：不再使用 GT 标签或模型自身的单视角输出，而是引入另一视角的伪标签作为交叉参考。三个实例化版本：I（数据侧）、II（模型侧）、III（两者结合）。

### 关键设计

1. **Co-rewarding-I（数据侧：对比一致性）**:
   - 做什么：为原始问题和语义改写问题生成交叉伪标签
   - 核心思路：给定原始问题 $x$ 及其改写 $x'$，分别从策略模型采样 $G$ 个 rollout。对 $x$ 的 rollout 用多数投票得到伪标签 $y_v$，对 $x'$ 同样得到 $y_v'$。然后**交叉使用**：$y_v'$ 用来评估 $x$ 的 rollout，$y_v$ 评估 $x'$ 的 rollout。advantage $\hat{A}_i = \frac{r(y_v', y_i) - \text{mean}(...)}{\text{std}(...)}$
   - 设计动机：语义等价的改写问题应得到相同答案（analogy-invariance）。交叉验证使得模型难以通过在单一输入上产生一致但错误的答案来获得高奖励——因为改写问题的输出会"检验"原始问题的答案。

2. **Co-rewarding-II（模型侧：自蒸馏）**:
   - 做什么：用 EMA 更新的教师模型提供伪标签
   - 核心思路：维护一个 EMA 教师 $\tilde{\pi}_{ref}^{(k)} \leftarrow \alpha^{(k)} \tilde{\pi}_{ref}^{(k-1)} + (1-\alpha^{(k)}) \pi_{\theta_{old}}^{(k)}$，EMA 权重按余弦退火从 $\alpha_{start}$ 到 $\alpha_{end}$。教师模型生成 rollout 并通过多数投票产生伪标签 $\tilde{y}_v$，然后用此伪标签评估策略模型的 rollout。
   - 设计动机：教师模型通过 EMA 更新，与当前在线策略"时间解耦"——它的伪标签不会立即受到策略变化的影响，打破了自奖励的即时反馈循环。类似 BYOL/DINO 中的 momentum teacher。

3. **Co-rewarding-III（两者结合）**:
   - 做什么：同时使用改写问题和 EMA 教师，全面解耦
   - 核心思路：教师模型对改写问题生成 rollout → 产生伪标签 → 监督策略模型对原始问题的 rollout（反之亦然）。数据和模型两个维度都引入了互补视角。
   - 设计动机：数据侧和模型侧的互补性——I 解决数据视角单一问题，II 解决监督与策略纠缠问题。

### 损失函数 / 训练策略
- 基于 GRPO：$\mathcal{J}(\theta) = \text{clipped surrogate objective} - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$
- 奖励函数：二元（正确=1, 错误=0），用交叉伪标签评估
- EMA 教师更新：余弦退火调度，初期快速更新（跟上策略改进），后期慢速（稳定监督）
- 改写生成：用 LLM 对原始数学问题进行语义等价改写

## 实验关键数据

### 主实验（MATH 训练集，Qwen3-8B-Base）

| 方法 | MATH500 | GSM8K | AMC | IFEval | MMLU-Pro |
|------|---------|-------|-----|--------|----------|
| Before RL | 72.4 | 27.8 | 20.9 | 50.9 | 52.9 |
| GT-Reward (RLVR) | 82.6 | 87.3 | 54.2 | 52.8 | 57.1 |
| Self-Certainty | 80.2 | 80.7 | 50.8 | 51.0 | 54.2 |
| Majority-Voting | 79.8 | 89.8 | 49.1 | 51.8 | 56.9 |
| **Co-rewarding-I** | 81.2 | **93.7** | 51.2 | 55.8 | **60.0** |
| **Co-rewarding-II** | 80.8 | 92.4 | 53.5 | **60.7** | 57.5 |
| **Co-rewarding-III** | **81.4** | 91.0 | **54.1** | 53.7 | 59.1 |

### 消融 / 训练稳定性

| 配置 | 训练崩溃？ | 数学推理平均提升 |
|------|----------|--------------|
| Self-Certainty | 经常崩溃 | +3% |
| Entropy | 偶尔崩溃 | +2% |
| Majority-Voting | 有时崩溃 | +4% |
| **Co-rewarding** | **不崩溃** | **+7.49% (Llama-3.2-3B)** |

### 关键发现
- **GSM8K 94.01%**：Co-rewarding 在 GSM8K 上用 Qwen3-8B 达到 94.01% Pass@1，超过了使用 GT 标签的 RLVR（87.26%）——无标签居然比有标签更好
- **训练稳定性**：所有自奖励基线在训练过程中都出现过崩溃（validation loss 突然飙升），Co-rewarding 的训练曲线始终稳定
- **平均提升 +3.31%**：在多个数学推理基准上，Co-rewarding 比最佳自奖励基线平均高 3.31%，在 Llama-3.2-3B 上高达 +7.49%
- **跨任务迁移**：仅在 MATH 上训练，在 Code（LiveCodeBench）和 Instruction Following（IFEval）上也有显著提升
- **三个版本各有优势**：I 在 GSM8K 上最强，II 在 IFEval 上最强，III 整体最均衡

## 亮点与洞察
- **自监督学习哲学的优雅迁移**：将 SimCLR 的"双视角一致性"和 BYOL/DINO 的"momentum teacher"迁移到 LLM RL 训练中，概念清晰、类比精准。这提示了一个更广泛的方法论：自监督学习中的成功范式可以系统性地迁移到 RL 中。
- **无标签超越有标签的现象值得关注**：GSM8K 上 94.01% vs 87.26%（GT-Reward），可能的解释是自监督信号提供了更多样的探索，而 GT 标签的二元奖励可能过度约束了策略。
- **EMA 教师 + 改写交叉验证的组合很实用**：计算开销可控（EMA 不需额外优化器），改写可以离线生成，整体方案比 RLVR 更容易扩展到无标注数据。

## 局限性 / 可改进方向
- 改写质量影响 Co-rewarding-I 的效果，需要高质量的改写模型
- EMA 教师的超参数（$\alpha_{start}, \alpha_{end}$）需要调优
- 仅在数学推理任务上验证，对 NLP 推理、代码生成等场景的效果待探究
- Co-rewarding-III 需要同时维护 EMA 教师和改写数据，内存开销较大
- 理论分析不够深入——为什么交叉验证能防止崩溃的形式化保证尚缺

## 相关工作与启发
- **vs Self-Certainty (Zhao et al.)**: 单一视角的确定性信号→容易崩溃；Co-rewarding 引入多视角→稳定
- **vs RLVR (GT-Reward)**: RLVR 依赖人工标注限制扩展；Co-rewarding 无标签但在多个设置下匹敌或超越
- **vs Majority-Voting (Shafayat et al.)**: 同样用多数投票，但仅在单一问题上→还是单视角；Co-rewarding 通过改写或教师引入真正的互补视角

## 评分
- 新颖性: ⭐⭐⭐⭐ 自监督学习→RL的概念迁移很精彩，但具体技术（改写+多数投票+EMA）并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型系列(Qwen3/Llama)、多基准、训练稳定性可视化、充分消融
- 写作质量: ⭐⭐⭐⭐ 概念阐述清晰，三个版本的递进关系好，但公式较多
- 价值: ⭐⭐⭐⭐⭐ 无标签RL训练的稳定性问题是实际中的痛点，本文提出了可行的解决方案
