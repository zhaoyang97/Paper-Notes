# Predicate-Conditional Conformalized Answer Sets for Knowledge Graph Embeddings

**会议**: ACL 2025 (Findings)
**arXiv**: [2505.16877](https://arxiv.org/abs/2505.16877)
**代码**: 有
**领域**: 图学习/知识图谱
**关键词**: knowledge graph embedding, conformal prediction, conditional coverage, uncertainty quantification, link prediction

## 一句话总结
提出 CondKGCP——基于谓词条件的 conformal prediction 方法用于知识图谱嵌入的不确定性量化，通过合并相似谓词增大校准集+双重校准（score+rank）减小预测集大小，在保证谓词级条件覆盖率的同时输出更紧凑的答案集，在多个KGE基准上优于5个baseline。

## 研究背景与动机

1. **领域现状**：KGE 方法（TransE/RotatE等）通过向量空间编码实体和关系进行链接预测，但缺乏预测不确定性量化。Zhu et al. (2025) 提出 KGCP 将 conformal prediction 应用于 KGE，生成保证覆盖真实答案的预测集。
2. **现有痛点**：KGCP 只提供**边际覆盖保证**（averaged over all queries），但不同谓词的不确定性差异巨大——如医学KG中"contraindicated_for"（禁忌症）比"has_symptom"（症状）需要更严格的保证。边际覆盖可能导致某些谓词覆盖率远低于目标。
3. **核心矛盾**：谓词级条件覆盖需要对每个谓词单独做 conformal prediction（Mondrian CP），但 KG 中大多数谓词只有极少三元组→校准集太小→阈值不稳定→预测集过大或覆盖失败。
4. **本文要解决什么**：在 KGE 中实现谓词条件覆盖保证，同时保持紧凑的预测集。
5. **切入角度**：合并向量表示相似的谓词扩大校准集 + 引入 rank 信息做双重校准。
6. **核心 idea**：CondKGCP = 谓词合并（增加校准数据）+ 双重校准（score 阈值 ∩ rank 阈值），理论保证+实验验证。

## 方法详解

### 整体框架
给定 KGE 模型 $M_\theta$，查询 $q = \langle h, r, ? \rangle$，目标覆盖率 $1-\epsilon$。输出：包含真实答案的紧凑实体集合 $\hat{C}(q)$。

### 关键设计

1. **谓词合并 (Predicate Merging)**
   - 做什么：将校准数据稀少的谓词合并到向量表示最相似的大谓词所在分区
   - 核心思路：Algorithm 1 将谓词集 $R$ 分为"数据充足"（$|\mathcal{T}_{cal}[\{r\}]| \geq \phi$）和"数据稀少"两组，稀少谓词按曼哈顿距离找最近的充足谓词合并
   - 设计动机：相似向量的谓词有相似的 nonconformity score 分布→合并后校准更稳定
   - 理论：合并后每个分区 $g$ 的校准集 $\mathcal{T}_g$ 足够大以确定可靠阈值

2. **双重校准 (Dual Calibration)**
   - 做什么：同时用 score 阈值和 rank 阈值约束预测集
   - 核心思路：$\hat{C}_{CondKGCP}(q) = \{e \in E_q[S \leq \hat{s}_{\epsilon'}] : \text{rank}(q,e) \leq \hat{k}(g)\}$
     - **Score 校准**：找 nonconformity score 的 $\epsilon'$-分位数作为阈值
     - **Rank 校准**：找最小 $k$ 使 $P(\text{rank} > k | \text{pred} \in g) < \epsilon$
   - 设计动机：单纯 score 校准在子组级别会产生过大预测集；加入 rank 约束排除高 score 但低 rank 的噪声实体
   - 误差率分配：$\epsilon = \epsilon_{rank} + \epsilon_{score}$，保证总覆盖率

3. **理论保证**
   - Theorem 2: CondKGCP 的条件覆盖概率紧紧围绕 $1-\epsilon$（上下界差 $O(1/|\mathcal{T}_g|)$）
   - Theorem 3: 双重校准的预测集大小在一定条件下严格小于单纯 score 校准

## 实验关键数据

### 条件覆盖率 vs 预测集大小 (FB15k-237, $\epsilon=0.1$)

| 方法 | 平均条件覆盖率 | 覆盖率标准差↓ | 平均预测集大小↓ |
|------|-------------|------------|--------------|
| KGCP（边际） | 0.90 | 0.15 | 245 |
| MCP（Mondrian） | 0.91 | 0.08 | 892 |
| ClusterCP | 0.90 | 0.09 | 456 |
| **CondKGCP** | **0.91** | **0.05** | **312** |

### 多数据集对比

| 数据集 | CondKGCP 覆盖率 | CondKGCP 集大小 | Best Baseline 覆盖率 | Best Baseline 集大小 |
|--------|---------------|---------------|--------------------|--------------------|
| FB15k-237 | 0.91 | **312** | 0.91 | 456 |
| WN18RR | 0.90 | **89** | 0.90 | 178 |
| YAGO3-10 | 0.91 | **205** | 0.90 | 367 |

### 关键发现
- **条件覆盖显著优于边际覆盖**：KGCP 的覆盖率标准差 0.15 → CondKGCP 仅 0.05，意味着所有谓词的覆盖率都接近目标
- **预测集比 Mondrian CP 小 2-3 倍**：谓词合并有效增加了校准数据量
- **双重校准关键**：去掉 rank 校准后预测集增大 40-60%
- **合并策略优于随机合并**：基于向量相似度的合并比随机分组的覆盖率标准差低 50%
- **对不同 KGE 模型鲁棒**：在 TransE/RotatE/ComplEx 上都有效

## 亮点与洞察
- **从"平均覆盖"到"条件覆盖"的关键升级**：对医学KG等高风险应用，"平均90%覆盖"不够——每个谓词都需90%才安全。CondKGCP 填补了这个gap
- **谓词合并的优雅设计**：利用 KGE 已有的向量表示做相似度，不需要额外模型——当然前提是KGE向量确实编码了谓词语义相似性
- **双重校准（score ∩ rank）**：两个独立维度的约束比单一阈值更有效——rank 排除了"score低但语义不相关"的噪声实体
- **对不确定性量化在 KG 领域的推动**：将 conformal prediction 从分类扩展到 KGE 的链接预测，是一个有价值的跨领域迁移

## 局限性 / 可改进方向
- **i.i.d 假设**：conformal prediction 要求校准集与测试集同分布，KG 的时态演化可能违反此假设
- **谓词合并的超参 $\phi$**：需要手动选择最小校准集大小阈值
- **仅限 1-hop 链接预测**：未扩展到多跳查询（如 EPFO queries）
- **计算成本**：需要对每个谓词分区分别校准，谓词数多时开销大
- **曼哈顿距离的选择**：其他相似度函数（如余弦、欧氏）可能在某些 KGE 上更合适

## 相关工作与启发
- **vs KGCP (Zhu et al., 2025)**：KGCP 只有边际覆盖，CondKGCP 提升到谓词条件覆盖——适用于高风险场景
- **vs Ding et al. (2024)**：他们在分类中做子组 conformal prediction，CondKGCP 将此思路迁移到 KGE 链接预测
- **vs Shi et al. (2024)**：他们用 rank 信息减小分类预测集，CondKGCP 将 rank 校准引入 KGE

## 评分
- 新颖性: ⭐⭐⭐⭐ 谓词合并+双重校准的组合在 KGE 中全新，理论保证有深度
- 实验充分度: ⭐⭐⭐⭐⭐ 3 个数据集 × 3 个 KGE 模型 × 5 个 baseline + 消融
- 写作质量: ⭐⭐⭐⭐⭐ 定义严谨，Algorithm 1 清晰，理论→实验衔接自然
- 价值: ⭐⭐⭐⭐ 对 KG 不确定性量化有重要推进，医学等高风险应用有直接意义
