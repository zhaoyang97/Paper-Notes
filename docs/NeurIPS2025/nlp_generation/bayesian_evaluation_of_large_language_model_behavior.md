# Bayesian Evaluation of Large Language Model Behavior

**会议**: NeurIPS 2025  
**arXiv**: [2511.10661](https://arxiv.org/abs/2511.10661)  
**代码**: 待确认  
**领域**: LLM 评估 / 贝叶斯统计  
**关键词**: Bayesian inference, LLM evaluation, uncertainty quantification, sequential sampling, Thompson sampling, binary metrics

## 一句话总结
提出基于 Beta-Binomial 贝叶斯模型的 LLM 行为评估框架，通过对每个 prompt 的随机生成结果建模 $\theta_m$ 后验分布，量化评估指标的统计不确定性，并引入 Thompson sampling 等序贯采样策略以更少的 API 调用获得更窄的置信区间。

## 研究背景与动机

1. **领域现状**：LLM 评估通常在固定 benchmark 上用确定性解码（greedy）生成一次回复，计算准确率/拒绝率等指标。但实际部署中 LLM 使用随机解码（temperature>0、top-p 等），同一 prompt 每次生成的结果可能不同。
2. **现有痛点**：(a) 确定性评估忽略了 LLM 输出的随机性，无法区分"99% 概率拒绝"和"55% 概率拒绝"的 prompt；(b) 评估指标通常不附带不确定性量化，无法判断两个模型的性能差异是否显著；(c) 多次采样评估成本高（API 调用收费），但简单地均匀分配采样次数效率低。
3. **核心矛盾**：准确评估 LLM 的随机行为需要每个 prompt 多次采样，但 API 成本限制了总采样数。如何在有限预算下最大化评估精度？
4. **本文要解决什么**：(1) 为 LLM 二值行为评估提供带不确定性量化的贝叶斯框架；(2) 通过序贯采样策略降低评估成本。
5. **切入角度**：将每个 prompt 的二值行为建模为 Bernoulli 试验（参数 $\theta_m$ 未知），用 Beta 先验 + Binomial 似然进行贝叶斯推断，得到 $\theta_m$ 后验分布。在此基础上推导聚合指标（均值、阈值计数等）的后验分布。
6. **核心 idea 一句话**：用 Beta-Binomial 共轭模型为每个 prompt 的随机二值行为建模，再用 Thompson sampling 把采样预算优先分配给不确定性最大的 prompt。

## 方法详解

### 整体框架
输入是 M 个 benchmark prompt，LLM 系统 $\pi$ 对每个 prompt $x^{(m)}$ 进行 $n_m$ 次随机生成，每次输出 $y$ 经二值判断器（如毒性检测器或偏好比较器）映射为 $b(y) \in \{0,1\}$。统计每个 prompt 的正例次数 $r_m$，用 Beta 后验推断 $\theta_m$，再对聚合函数 $W = g(\theta_1,...,\theta_M)$ 做后验推断。

### 关键设计

1. **Beta-Binomial 后验推断**：
   - 做什么：估计每个 prompt 的二值行为概率 $\theta_m$ 及其不确定性
   - 核心思路：$\theta_m \sim Beta(\alpha_m, \beta_m)$ 先验，观测 $r_m$ 次正例后后验为 $Beta(\alpha_m + r_m, \beta_m + n_m - r_m)$。共轭性保证闭式更新，计算高效
   - 设计动机：Beta-Binomial 是二值数据不确定性建模的经典方案，每个 prompt 独立建模，后验可增量更新

2. **聚合指标的后验分布**：
   - 做什么：从个体 $\theta_m$ 后验推导评估指标（如均值 $W_{mean}$、超阈值数 $W_{>\nu}$）的分布
   - 核心思路：$W_{mean} = \frac{1}{M}\sum \theta_m$ 通过 Monte Carlo 采样近似后验分布；$W_{>\nu} = \sum I(\theta_m > \nu)$ 服从 Poisson Binomial 分布，可精确计算
   - 设计动机：用户关心的往往不是单个 prompt 而是整体指标，贝叶斯框架自然地将个体不确定性传播到聚合指标

3. **序贯 Thompson Sampling**：
   - 做什么：在有限 API 调用预算下，动态决定下一次采样哪个 prompt
   - 核心思路：每轮利用当前 $\theta_m$ 后验进行 Thompson sampling——对每个 prompt 采样一个 $\theta_m$ 值，基于采样值选择对聚合指标不确定性贡献最大的 prompt 进行下一次生成。还探索了 maximum variance 等策略
   - 设计动机：不同 prompt 的 $\theta_m$ 不确定性不同（有些接近 0 或 1，几次就够；有些接近 0.5，需要更多次），序贯策略智能分配预算

### 训练策略
本文不涉及模型训练，是纯推断框架。全部实验基于现有 LLM API（GPT-4o-mini、GPT-4.1-nano）和 LM-as-judge（GPT-4.1-mini）进行推理。对于偏好实验使用 temperature=1.0 和 top-p=0.9，对于拒绝率实验同样使用随机解码以确保捕获输出级别的随机性。先验设为 Beta(1,1) 均匀分布。

## 实验关键数据

### Case Study 1: Pairwise Preference（GPT-4o-mini vs GPT-4.1-nano）
| 方法 | 结果 |
|------|------|
| Greedy decoding | Model A preferred on 41/80 prompts |
| Bayesian (n=50) | $W_{mean}$ 95% CI: (51%, 53%) |
| Bayesian (n=50) | 23 prompts where Model A preferred with >75% prob |

### Case Study 2: Jailbreak Refusal Rate
| 评估 | 说明 |
|------|------|
| 批量评估 | 揭示不同 prompt 的拒绝概率 $\theta_m$ 差异巨大 |
| 序贯 vs 批量 | Thompson sampling 用更少采样次数达到相同精度 |

### 消融：序贯采样策略对比
| 策略 | 效率 | 说明 |
|------|------|------|
| Uniform (batch) | baseline | 每个 prompt 采样相同次数 |
| Max Variance | 较优 | 优先采样后验方差最大的 prompt |
| Thompson Sampling | 最优 | 探索-利用平衡，整体聚合指标收敛最快 |

### 关键发现
- Greedy decoding 评估会隐藏重要信息：两个 prompt 在 greedy 下都选 Model A，但随机解码下一个是 99% 另一个是 55%
- 序贯 Thompson sampling 比均匀采样在相同总预算下将聚合指标的 credible interval 缩小约 20-30%
- Beta 后验的增量更新特性使得序贯方案几乎零额外计算开销

## 亮点与洞察
- **将 LLM 评估正式化为统计推断**：不是 ad-hoc 的"多跑几次取平均"，而是严格的贝叶斯框架，后验置信区间有统计学保证
- **序贯采样降成本**：Thompson sampling 的引入将"智能评估"变为可能——不再盲目均匀采样，而是把有限的 API 调用集中在最有信息量的 prompt 上
- **黑盒兼容**：框架不需要 LLM 内部任何信息（无需 logits、权重或架构），纯基于观测输出

## 局限性 / 可改进方向
- **仅支持二值指标**：当前框架限于 $b(y) \in \{0,1\}$，无法直接处理连续评分（如 1-5 分质量评分）或多类别判断
- **判断器假设确定性**：假设二值判断器 $b(y)$ 是确定性的，但实际中 LLM-as-judge 本身也有随机性，引入额外不确定性源
- **独立性假设**：各 prompt 的 $\theta_m$ 独立建模，忽略了相似 prompt 之间可能共享的行为模式（如同一主题的 prompt 拒绝率可能相关）
- **计算成本**：每个 prompt 多次采样需要大量 API 调用（80 prompts × 50 samples = 4000 次），对闭源模型成本不低
- **聚合函数的选择**：$W_{mean}$ 和 $W_{>\nu}$ 只是最基础的聚合，更复杂的评估需求（如条件评估、分组比较）未覆盖
- **改进方向**：(1) 扩展到连续评分的 Beta 回归或正态模型；(2) 用层次贝叶斯模型跨 prompt 共享信息；(3) 将判断器不确定性也纳入模型；(4) 开发自适应停止准则减少不必要的采样

## 相关工作与启发
- **vs Scholten et al. (2025)**：同样关注 output-level uncertainty，但用频率学派方法；本文贝叶斯方法自然支持序贯更新和决策
- **vs Miller (2024)**：讨论了重复生成减少方差，但未提供正式的统计模型；本文给出完整的贝叶斯推断框架
- **vs Hariri et al. (2025)**：贝叶斯方法用于多类别评估；本文聚焦二值行为 + 序贯采样
- **启发**：这个框架可直接迁移到任何需要评估随机系统（如 agent）行为的场景。Thompson sampling 序贯采样思路也可用于 benchmark 构建——识别最"有区分力"的 prompt

## 评分
- 新颖性: ⭐⭐⭐ Beta-Binomial 模型本身不新，但将其系统性地应用于 LLM 评估并引入序贯采样有创新
- 实验充分度: ⭐⭐⭐⭐ 两个实际案例（偏好对比 + 越狱拒绝），batch/sequential 双模式对比
- 写作质量: ⭐⭐⭐⭐⭐ 面向统计学受众的教学式写作，公式推导完整清晰
- 价值: ⭐⭐⭐⭐ 为 LLM 评估提供了统计学上严格的不确定性量化工具，实用性强
