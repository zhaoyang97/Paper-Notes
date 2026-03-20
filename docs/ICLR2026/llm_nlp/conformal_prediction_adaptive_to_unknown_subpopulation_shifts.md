# Conformal Prediction Adaptive to Unknown Subpopulation Shifts

**会议**: ICLR 2026  
**arXiv**: [2506.05583](https://arxiv.org/abs/2506.05583)  
**代码**: 待确认  
**领域**: LLM/NLP  
**关键词**: conformal prediction, distribution shift, subpopulation shift, uncertainty quantification, LLM hallucination

## 一句话总结
针对子群体偏移（subpopulation shift）下标准 conformal prediction 失效的问题，提出三种自适应算法：利用学习的 domain classifier 加权校准数据（Algorithm 1/2）或利用嵌入相似度加权（Algorithm 3），在不完美甚至无 domain 标签的情况下仍能保证覆盖率，并应用于视觉分类和 LLM 幻觉检测。

## 研究背景与动机

1. **领域现状**：Conformal prediction (CP) 为黑盒 ML 模型提供不确定性量化，保证在可交换数据下的边际覆盖率 $\Pr(Y_{\text{test}} \in C_\alpha(X_{\text{test}})) \geq 1-\alpha$。
2. **现有痛点**：实际中测试环境的子群体混合比例常与校准数据不同（subpopulation shift），导致标准 CP 在某些测试环境下严重欠覆盖或过覆盖。现有解决方案要么需要已知的分布偏移（Tibshirani et al.），要么使用 worst-case 阈值（max CP，严重过覆盖），要么需要完美的 group 标签（group-conditional CP）。
3. **核心矛盾**：Group-conditional CP 在理论上能解决子群体偏移问题，但它需要精确的 group 成员信息。Theorem 2.1 证明不完美的 domain classifier 会导致覆盖保证严重退化——如果 classifier 准确率为 $\gamma$，覆盖可以低至 $\max(0, \gamma - \alpha)$。
4. **本文要解决什么**：在 domain 标签未知或不完美的情况下，设计有理论保证的自适应 CP 算法。
5. **切入角度**：不要求完美 domain classifier，而是利用 multicalibration / multiaccuracy 等更弱的假设来保证覆盖率；甚至完全不需要 domain classifier，仅用嵌入相似度加权。
6. **核心 idea 一句话**：用不完美的 domain classifier 或嵌入相似度来自适应加权校准数据，在未知子群体偏移下保持 conformal prediction 的覆盖保证。

## 方法详解

### 整体框架

测试环境 $\mathbb{P}_{\text{test}} = \sum_{k=1}^K \lambda_k \mathbb{P}_k$，其中 $\lambda_k$ 未知且与校准数据不同。核心思路：根据测试数据点估计其所属 domain 的概率分布 $\hat{\lambda}$，然后用 $\hat{\lambda}$ 加权不同 domain 的校准分数来计算自适应阈值。

三种算法递进放松假设：
1. **Algorithm 1**：有 domain classifier + 逐点加权（需 multicalibrated classifier）
2. **Algorithm 2**：有 domain classifier + 批量平均加权（需 multiaccurate classifier，更弱要求）
3. **Algorithm 3**：无 domain classifier，用嵌入相似度加权（无理论保证但实验有效）

### 关键设计

1. **Algorithm 1: Weighted CP with domain classifier (逐点)**:
   - 做什么：对每个测试点 $X_{\text{test}}$，用 domain classifier $c(X_{\text{test}})$ 预测其属于各 domain 的概率 $\hat{\lambda}$
   - 核心思路：$\hat{q}_\alpha \leftarrow \min_{\hat{q}} \sum_{k=1}^K \frac{\hat{\lambda}_k m_k(\hat{q}_\alpha)}{n_k + 1} \geq (1-\alpha)$，其中 $m_k(\hat{q})$ 是 domain $k$ 中分数不超过 $\hat{q}$ 的校准样本数
   - 理论保证（Theorem 3.1）：若 $c$ 是贝叶斯最优分类器，则保证 $\Pr(Y_{\text{test}} \in C_\alpha(X_{\text{test}})) \geq 1-\alpha$
   - Theorem 3.3 放松到 multicalibrated classifier 仍成立

2. **Algorithm 2: Weighted CP with batch averaging**:
   - 做什么：用测试集上 domain 预测概率的平均值替代逐点估计
   - 核心思路：$\hat{\lambda} = \text{mean}_{i=1}^{n_{\text{test}}} c(X_{\text{test}}^i)$，然后同样计算加权阈值
   - 理论保证（Theorem 3.5）：仅需 multiaccurate classifier（比 multicalibrated 更弱的条件）即可保证覆盖率
   - 设计动机：Multiaccuracy 比 multicalibration 计算和样本复杂度更低，更容易满足

3. **Algorithm 3: Similarity-weighted CP (无 domain classifier)**:
   - 做什么：完全不需要 domain 标签或 domain classifier，用嵌入空间的相似度加权校准数据
   - 核心思路：
     - 按与测试点的嵌入相似度排序，保留 top $\beta$ 比例的校准数据
     - 用 softmax 加权：$\gamma_i = h(z(X_{\text{test}}), z(X_i'))$，$m = \text{Softmax}(\{\gamma_i/\sigma\})$
     - 计算加权分位数作为阈值
   - 设计动机：语义相似的数据更可能来自相同 domain，用相似度近似 domain 归属

4. **Theorem 2.1 (关键理论贡献)**:
   - 做什么：证明 group-conditional CP 在 domain classifier 不完美时的覆盖退化
   - 核心结论：存在某些分布使得覆盖率退化至 $\max(0, \gamma - \alpha)$，其中 $\gamma$ 是 classifier 的条件准确率
   - 这从根本上说明为什么不能简单地把不完美 classifier 插入 group-conditional CP

### 损失函数 / 训练策略

- Domain classifier 训练：冻结预训练模型主体，只训练最后 3 层 FC (2048→1024→512→K)，Adam + CE loss
- 训练后用 Multi-domain temperature scaling 校准
- LLM 幻觉检测：用 GPT-4o 作为正确性评估器，LLaMA-3-8B 作为生成模型

## 实验关键数据

### 主实验

ImageNet 上 100 个测试环境的覆盖率分布（26 domains, ViT, LAC score, α=0.05）：

| 方法 | 平均覆盖率 | 标准差 | 评价 |
|------|-----------|--------|------|
| 目标覆盖率 | 0.950 | - | 理想值 |
| Standard CP | ~0.95 | **高** | 部分环境严重欠覆盖 |
| Max CP | ~0.99 | 低 | 严重过覆盖 |
| Conditional Calibration | ~0.94 | 中 | 某些环境欠覆盖 |
| Algorithm 1 (A1) | ~0.95 | **低** | 紧贴目标 |
| Algorithm 2 (A2) | ~0.95 | **低** | 紧贴目标 |
| Oracle | ~0.95 | **最低** | 理想上界 |
| Algorithm 3 (A3) | ~0.95 | 低 | 无 domain info 也有效 |

### 消融实验

| 实验维度 | 结果 |
|---------|------|
| 不同 score function (HPS/APS/LAC) | 一致有效 |
| 不同模型 (ResNet50/ViT/CLIP-ViT) | 一致有效 |
| 不同偏移程度 (α'=0.1/0.5/1.0) | 偏移越大，A1-A3 vs Standard CP 差距越大 |
| LLM 幻觉检测 (LLaMA-3-8B) | A3 显著降低 recall 的标准差 |

### 关键发现
- **Standard CP 在偏移下不可靠**：100 个测试环境中有相当比例严重欠覆盖，标准差很大。
- **Max CP 过于保守**：虽然保证覆盖但严重过覆盖（~0.99 vs 目标 0.95），导致预测集过大，实用性差。
- **A1/A2 紧贴目标**：覆盖率均值和标准差都接近 oracle，说明 multicalibrated/multiaccurate classifier 假设在实践中成立。
- **A3 无需 domain 信息也有效**：仅用嵌入相似度加权即可在大部分环境下保持覆盖，是最实用的方法。
- **LLM 幻觉检测**：Standard CP 的 recall 在不同测试环境下波动大，A3 显著降低波动，更可靠。

## 亮点与洞察
- **Theorem 2.1 揭示了 group-conditional CP 的根本缺陷**：不完美 group 信息会导致覆盖保证退化，这不是小问题而是可能完全失效（覆盖率可低至 $\gamma - \alpha$）。这为本文的方法提供了强动机。
- **从 Bayes-optimal → multicalibrated → multiaccurate 的假设放松链**：优雅地逐步弱化对 domain classifier 的要求，同时保持覆盖保证。实践者可根据自身 classifier 质量选择合适的算法。
- **Algorithm 3 的无监督自适应**：不需要任何 domain 信息，仅用嵌入相似度即可实现类似效果——这使得方法可以直接应用于任何预训练模型。

## 局限性 / 可改进方向
- **理论未利用样本独立性**：当前理论未利用不同 domain 样本间的独立性，导致轻微偏移时有些过覆盖。
- **Algorithm 3 无理论保证**：虽然实验有效但缺乏像 A1/A2 那样的形式化覆盖保证。
- **Score function 选择无指导**：没有提供关于何种 score function 在何种场景下最优的理论或实证指导。
- **LLM 实验局限**：仅在短答题 QA 上测试幻觉检测，未涉及更复杂的生成任务。
- **单一风险控制**：扩展到同时控制多种风险（幻觉+毒性+谄媚等）是重要的未来方向。

## 相关工作与启发
- **vs Tibshirani et al. (2020)**：需要已知的 covariate likelihood ratio，在高维 ML 中不可行。本文方法仅需学习 domain classifier 或用相似度近似。
- **vs Gibbs et al. (2024, Conditional Calibration)**：也用两阶段 domain classifier 方法，但假设完美 group 信息。本文的 Theorem 2.1 证明这一假设的脆弱性，并提出更鲁棒的替代。
- **vs Max/Robust CP (Cauchois et al.)**：保证覆盖但过于保守。本文自适应地匹配实际测试分布而非 worst case。

## 评分
- 新颖性: ⭐⭐⭐⭐ multicalibration/multiaccuracy + CP 的结合新颖，Theorem 2.1 有实际意义。
- 实验充分度: ⭐⭐⭐⭐ 视觉+LLM 双领域验证，多模型多 score function 多偏移度；A3 缺理论分析。
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，假设链条逻辑自洽；Algorithm 伪代码易读。
- 价值: ⭐⭐⭐⭐ 对 CP 在实际部署中的可靠性有直接贡献，LLM 幻觉检测应用有潜力。
