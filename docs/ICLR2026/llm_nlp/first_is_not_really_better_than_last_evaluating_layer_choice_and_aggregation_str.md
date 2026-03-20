# First is Not Really Better Than Last: Evaluating Layer Choice and Aggregation Strategies in Language Model Data Influence Estimation

**会议**: ICLR 2026  
**arXiv**: [2511.04715](https://arxiv.org/abs/2511.04715)  
**代码**: 有  
**领域**: LLM/NLP (其他)  
**关键词**: Influence Functions, Data Attribution, Layer Analysis, LLM Interpretability, Training Data Quality

## 一句话总结
通过理论和实验证明先前工作所推崇的"第一层（embedding）最适合做 influence estimation"的结论是不可靠的，发现中间 attention 层才是更好的估计层，并提出 Rank 和 Vote 两种新的跨层聚合策略以及 Noise Detection Rate (NDR) proxy 指标，显著改善了 LLM 中有害训练样本的检测效果。

## 研究背景与动机

1. **领域现状**：Influence function 是评估训练数据对模型决策影响的重要工具（TracIn、DataInf、Cosine 等）。由于现代 LLM 参数量巨大，通常只在部分层上计算 influence 以保证可行性。
2. **现有痛点**：Yeh et al. (2022) 基于"cancellation effect"假设得出结论认为第一层（word embedding）最适合做 influence estimation。然而这一结论只在小规模模型（RoBERTa）和单一方法（TracIn）上验证过，且 cancellation effect 本身的可靠性从未被严格检验。
3. **核心矛盾**：cancellation effect 指标 $C(W)$ 通过参数子集的 norm 聚合来衡量梯度抵消，但这种聚合会掩盖个别参数上的极端抵消，导致指标无法可靠预测层的实际 influence 性能。同时，标准的均值聚合策略也可能因对冲效应而降低判别能力。
4. **本文要解决什么？** (RQ1) cancellation effect 是否可靠；(RQ2) 哪些层最适合 influence estimation；(RQ3) 如何更好地跨层聚合 influence 分数；(RQ4) 是否存在不需要重训练就能评估 influence 方法效果的 proxy 指标。
5. **切入角度**：从理论证明 cancellation effect 的反例出发，然后在多模型多数据集上做大规模实验，系统评估层选择和聚合策略。
6. **核心 idea 一句话**：中间 attention 层比 embedding 层更适合影响力估计，Vote 聚合比均值聚合显著更好，NDR 可作为无需重训练的可靠 proxy 指标。

## 方法详解

### 整体框架

输入是带噪声的训练数据集 + 验证数据集 + 预训练 LLM。五阶段流程：(1) 注入合成噪声到训练数据；(2) 在噪声数据上 fine-tune，选最佳 checkpoint；(3) 在所有可调层上计算 influence 值；(4) 将模型分为 WE（embedding）、4 组 attention 层、CL（分类头），聚合每个训练样本的 influence；(5) 删除 30% 最低 influence 样本后重训，用测试准确率评估效果。

### 关键设计

1. **Cancellation Effect 的理论反驳**
   - 做什么：证明 Yeh et al. 的 cancellation effect 作为层选择指标是不可靠的。
   - 核心思路：Theorem 5.1 构造了一个反例：存在验证点 $\bar{x}_3$ 使得包含高 cancellation 权重 $\omega$ 的 influence 分数 $\Delta I_{\theta,\omega}$ 比只用低 cancellation 权重 $\theta$ 的 $\Delta I_{\theta}$ 对噪声/干净样本的区分度更大。Spearman 相关性分析也显示 $C$ 与下游性能几乎无关（$\rho$ 接近 0）。
   - 设计动机：直接挑战该领域的基础假设，为重新评估层选择打开大门。

2. **Rank 聚合策略**
   - 做什么：用排名代替原始 influence 分数来跨层聚合，消除极端值的支配效应。
   - 核心思路：每个验证样本和每个层对训练样本按 influence 分数排序，将排名求和（$\operatorname{Rank}(I') = \sum_{x',l} \sum_{y} \mathbb{I}(I'(y,...) < I'(\cdot,...))$），且只考虑被正确预测的验证样本。
   - 设计动机：标准均值聚合中，不同层的 influence 量纲差异大，少数极端值可能主导结果。排名消除了量纲影响。

3. **Vote（位置投票）聚合策略**
   - 做什么：每个验证样本/层对排名最低的 $k$ 个训练样本投票，投票数按排名递减。
   - 核心思路：$\operatorname{Vote}_k(I') = -\sum_{x',l} \max(k - \operatorname{rank}, 0)$，选择 $k$ 等于要过滤的训练样本数。只有排在最底部的样本获得投票，避免中间排名的噪声影响。
   - 设计动机：Rank 方法仍受极低/极高排名的影响，Vote 通过截断只关注最有害的样本。实验表明 $k \in [10, 50]$ 效果最好。

4. **Noise Detection Rate (NDR) proxy 指标**
   - 做什么：提出一个不需要重训练就能评估 influence 方法的 proxy 指标。
   - 核心思路：NDR 衡量在 influence 排名末尾 $k\%$ 的样本中噪声样本占比。AUC 衡量噪声在整个排名中的分布偏斜度。与 cancellation effect 的相关性几乎为零不同，NDR 与实际下游性能有 0.5-0.9 的 Spearman 相关性。
   - 设计动机：避免每次评估新 influence 方法都需要昂贵的重训练实验。

### 损失函数 / 训练策略

使用标准 cross-entropy fine-tuning（LoRA），在最低验证 loss 处选择 checkpoint。每个配置 10 个 seed 重复。成对比较配置优劣，用 win rate 和 Pareto front 分析。

## 实验关键数据

### 主实验

| 模型 | 最佳层 | 最差层 | 最佳方法+层 | Win Rate |
|------|--------|--------|------------|----------|
| RoBERTa-Large | Attn 18-23 | CL | DataInf + Attn 18-23 | 0.70 |
| Qwen-2.5 1.5B | Attn 07-13 | CL | Cosine + Attn 07-13 | Top-1 |
| Mistral 7B | Attn 08-15 | CL | DataInf + Attn 08-15 | 0.71 |
| Llama-3.2 1B | Attn 04-07 | CL | DataInf + Attn 04-07 | 0.64 |

最佳层比最差层（CL）在过滤后准确率上高 10-15%。

### 消融实验

| 聚合策略 | 效果 | 说明 |
|---------|------|------|
| Mean (baseline) | 基准 | 标准跨层均值 |
| Rank | 中等提升 | 消除量纲差异，某些场景优于 Vote |
| Vote (k=10-50) | 显著提升 | TracIn CL win rate 从 0.10 提升到 top-1；DataInf 00-07 win rate 达 0.84 |

| Proxy 指标 | 与下游性能相关性 (Spearman ρ) |
|-----------|---------------------------|
| Cancellation Effect C | -0.3 到 0.2（弱/无） |
| NDR@30% (Mean) | 0.4 到 0.7（中等到强） |
| NDR@30% (Vote) | 0.8 到 0.9（强） |

### 关键发现
- **中间 attention 层一致性地优于 embedding 层和分类头**，跨模型跨方法成立，推翻了 Yeh et al. 的结论
- **分类头（CL）在所有模型上都是最差的选择**，可能因为 CL 对噪声过于敏感
- **Vote 聚合能将原本表现差的配置提升到 top-1**（如 TracIn CL 在 Mistral 上从排名 12 提升到 1）
- **DataInf 和 Cosine 通常优于 TracIn**，尤其在更大模型的中间层
- **Llama-3.2 1B 是最难的模型**——所有 influence 方法都没超过随机过滤，可能与该模型的训练特性有关

## 亮点与洞察
- **理论反例+大规模实验双重证据**推翻领域经典假设，方法论值得学习：先用定理构造反例证明"可以错"，再用实验证明"确实错"。
- **Vote 聚合极其简单但效果显著**：只需要选择一个 $k$ 参数，就能把原本不可用的层（如 CL）变得可用，说明 influence estimation 的瓶颈可能不在方法本身而在聚合策略。
- **NDR 作为 proxy 指标有很大实用价值**：研究者可以不做重训练就快速评估 influence 方法的效果，大幅降低实验成本。
- 层间 influence 分数的相关性分析揭示了三个层组（early/middle/late），这个结构与知识编辑（ROME/MEMIT）的发现一致。

## 局限性 / 可改进方向
- 实验仅在 GLUE benchmark 上测试，未涉及生成任务或 in-context learning 场景
- Llama-3.2 1B 上 influence function 全面失效，文章未给出令人信服的解释
- Vote 的超参 $k$ 需要搜索，且在 Cosine 方法上会降低性能
- 噪声注入方式（标签翻转）较为简单，未测试更复杂的数据质量问题（如 backdoor attack）
- 只测试了 LoRA fine-tuning，全参数 fine-tuning 的行为可能不同

## 相关工作与启发
- **vs Yeh et al. (2022)**: 直接挑战其核心结论。Yeh et al. 用小模型 + 单方法得出 embedding 最好的结论，本文在 4 个模型 + 5 种方法 + 8 个数据集上证明中间层更好。
- **vs ROME/MEMIT**: 知识编辑方法也发现中间 MLP 层编码最多事实信息，与本文发现中间层 influence 最强一致，从不同角度验证了"中间层最信息丰富"的假设。
- **vs Li et al. (2025)**: 他们发现 influence function 在 LLM 上表现差，但用默认设置（均值聚合 + 所有层）。本文表明选对层 + 换 Vote 聚合可以显著改善结果。

## 评分
- 新颖性: ⭐⭐⭐⭐ 推翻经典假设 + Rank/Vote 聚合 + NDR proxy 指标，多点创新
- 实验充分度: ⭐⭐⭐⭐⭐ 4 模型 × 5 方法 × 8 数据集 × 10 seeds，极其充分
- 写作质量: ⭐⭐⭐⭐ RQ 驱动的结构清晰，但篇幅较长
- 价值: ⭐⭐⭐⭐ 对 influence function 研究有直接指导意义，尤其是层选择和聚合策略
