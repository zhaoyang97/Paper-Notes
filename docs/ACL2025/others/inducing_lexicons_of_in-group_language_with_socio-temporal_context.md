---
title: >-
  [论文解读] Inducing Lexicons of In-Group Language with Socio-Temporal Context
description: >-
  [ACL 2025][词汇归纳] 提出 LISTN（Lexicon Induction with Socio-Temporal Nuance）框架，利用动态词嵌入和用户嵌入联合建模社区语言的社会结构和时间演化，在反女性在线社区（manosphere）的群体内词汇归纳任务上达到 0.77 的平均精度，显著超越现有方法。
tags:
  - "ACL 2025"
  - "词汇归纳"
  - "群体内语言"
  - "动态嵌入"
  - "社会时序建模"
  - "manosphere"
---

# Inducing Lexicons of In-Group Language with Socio-Temporal Context

**会议**: ACL 2025  
**arXiv**: [2409.19257](https://arxiv.org/abs/2409.19257)  
**代码**: [GitHub](https://github.com/unimelb-nlp/listn)  
**领域**: 其他  
**关键词**: 词汇归纳, 群体内语言, 动态嵌入, 社会时序建模, manosphere

## 一句话总结

提出 LISTN（Lexicon Induction with Socio-Temporal Nuance）框架，利用动态词嵌入和用户嵌入联合建模社区语言的社会结构和时间演化，在反女性在线社区（manosphere）的群体内词汇归纳任务上达到 0.77 的平均精度，显著超越现有方法。

## 研究背景与动机

**群体内语言（in-group language）** 是社会群体的重要标志：一方面用于对外混淆（让外部观察者难以理解），另一方面用于对内凝聚（信号化群体认同）。由于这类非正式语言演化迅速，使用最新的词汇创新是群体归属的强信号。

现有的群体内词汇构建方法存在两个主要问题：

**手工构建昂贵且容易过时**: 例如 Rowe & Saif (2016) 在研究 ISIS 语言时使用了 7 年前的词典

**计算方法忽视社会和时间维度**: Lucy & Bamman (2021) 用上下文词嵌入，Farrell et al. (2020) 用话题模型，但都只利用语言信息，忽视了群体的动态社会结构

本文聚焦的 **manosphere**（反女性在线社区）是一个特别合适的研究对象：(1) 语言创新极其活跃（如 foid、AWALT 等）；(2) 子群体结构复杂且持续演化（Incels、MRA、MGTOW、PuA、TRP）；(3) 与现实世界暴力事件关联，是紧迫的社会关切。

## 方法详解

### 整体框架

LISTN 分为两步：
1. **表示学习**: 使用 Cerberus 架构训练动态词嵌入和用户嵌入（联合分解用户-内容矩阵和用户-用户邻接矩阵）
2. **词汇归纳**: 基于低秩重构的评分方法，计算每个词在不同时间步对不同子群体的相关性

### 关键设计

#### 1. 表示学习: Cerberus 动态矩阵分解

**功能**: 在每个时间步 $t$，联合分解用户-内容矩阵 $C_t$ 和用户-用户邻接矩阵 $A_t$，得到动态的用户嵌入 $U_t$ 和词嵌入 $W_t$。

**核心思路**:
- **Content 矩阵** $C_t$: 用 PPMI（正值点互信息）构建，衡量用户 $i$ 使用词 $j$ 的频率相对于背景语料的偏离程度
- **Adjacency 矩阵** $A_t$: 捕获两个用户在同一讨论串中的互动频率
- 联合分解: $C_t \approx U_t \cdot W_t^T$ 且 $A_t \approx U_t \cdot V_t^T$
- 时序正则化: 惩罚连续时间步之间嵌入的大变化，保证时间对齐

**设计动机**: 将用户和词映射到同一空间，自然地将社会结构（谁和谁互动）与语言内容（谁用什么词）整合。时序正则化确保嵌入变化反映的是真实的社会-语义演化，而非噪声。

**实现细节**: 使用 PyTorch 重新实现，采用广义矩阵分解（GMF）公式，支持批量更新和稀疏矩阵处理。训练数据为来自 50 个 manosphere subreddit 的 400 万+ 发言，覆盖 2018 年 4-12 月共 9 个月。

#### 2. 词汇归纳方法: 6 种 LISTN 变体

给定用户嵌入 $u_{i,t}$ 和词嵌入 $w_{j,t}$，词 $j$ 对用户 $i$ 在时间 $t$ 的相关性为 $r(i,j,t) = u_{i,t} \cdot w_{j,t}^T$。

六种聚合方式：

| 方法 | 计算方式 | 思路 |
|------|---------|------|
| **Community centroid** | 所有用户均值 × 词向量 | 全社区层面 |
| **Category centroid** | 各子社区（Incel/MRA/...）均值取max | 考虑子群体专业化 |
| **Subreddit centroid** | 各subreddit均值取max | 更细粒度 |
| **Cluster (K=5/20/100)** | K-means聚类后各簇均值取max | 数据驱动的子群体发现 |
| **Bootstrap** | 已知词典的最近邻 | 词典扩展 |
| **Bias** | 分解模型的词偏置项 | 全局流行度 |

**时间聚合**: 取词在所有时间步中的最大得分，可以捕获进出流行期的词。

**设计动机**: 使用 max 而非 mean 聚合，因为一个词只要在某个子群体中高度相关就应被视为群体内语言。不同粒度的方法探索了"最佳子群体划分"的效果。

#### 3. 评估框架

**任务定义**: 将词汇归纳框架为二分类任务——判断一个词是否为群体内词汇创新（lexical innovation）。

**测试集构建**: 
1. 初步评分：用所有方法对现有 5 个 manosphere 词典（483 个已知词）评分
2. 取最佳基线和最佳 LISTN 方法的 top-1000 词
3. 由作者和一位社会心理学博士专家独立标注（Cohen's Kappa = 0.726）
4. 最终测试集：1803 个词，944 正 / 859 负

**评价指标**: Average Precision (AP) 和 AUROC，优先关注 AP。

### 基线方法

- **word2vec bootstrap**: 训练 word2vec 后做已知词的最近邻扩展
- **PMI 变体**: PPMI/NPMI 在不同粒度（社区/subreddit/类别/月份）上计算

## 实验关键数据

### 主实验：词汇归纳性能

| 方法 | AP | AUROC |
|------|-----|-------|
| Random | 0.52 | 0.50 |
| word2vec bootstrap | 0.5563 | 0.5427 |
| NPMI-category (最佳基线) | 0.6790 | 0.6647 |
| LISTN-CA Cluster-5 | 0.7620 | 0.7403 |
| **LISTN-C Cluster-5** | **0.7679** | **0.7363** |
| LISTN-C Category | 0.7272 | 0.6809 |

### 关键方法对比

| 聚合粒度 | LISTN-CA AP | LISTN-C AP |
|----------|-----------|-----------|
| Community (全局) | 0.6228 | 0.6297 |
| Category (5类) | 0.7231 | 0.7272 |
| Subreddit (52个) | 0.5519 | 0.5891 |
| Cluster-5 | **0.7620** | **0.7679** |
| Cluster-20 | 0.7069 | 0.7554 |
| Cluster-100 | 0.6950 | 0.7040 |
| Bootstrap | 0.5349 | 0.5276 |
| Bias | 0.6190 | 0.6016 |

### 关键发现

1. **LISTN-C (仅内容) ≥ LISTN-CA (内容+邻接)**: 出人意料地，加入用户互动信息并未提升词汇归纳（P=0.723 不显著）。但 LISTN-C 仍包含社会信息——使用相同词的用户被表示为相似
2. **Cluster-5 最优**: K=5 的聚类优于 category 级别（也是约 5 个类别）——数据驱动的子群体划分比平台定义的更有效
3. **粒度过细反而有害**: Subreddit 级别表现最差，可能因为偶尔在某 subreddit 发言的用户引入噪声
4. **Bootstrap/Bias 方法最差**: 不考虑社区结构的方法表现差，说明子群体专业化是关键
5. **NPMI > PPMI**: 与 Lucy & Bamman (2021) 的结论一致；但月度 NPMI 并不比全局 NPMI 更好，说明简单的时间分片不足以捕获时序动态

## 嵌入分析的额外发现

### 词表示的时间稳定性

- 群体内词汇的嵌入比同频率的一般词汇更稳定（CEV 更低）——群体内词汇作为社会符号有使用规范，成员会"正确"使用它们
- 词频与 CEV 呈强负相关（ρ = -0.77），但在频率 > 10,000 后变化趋近零
- 低频高稳定的词包括药物名称（lamictal、seroquel）——技术性术语不受社会变化影响

### 子群体语言专业化

| 群体对 | Spearman ρ | 解释 |
|--------|-----------|------|
| PuA ↔ TRP | 0.729 | 共同关注诱惑，术语大量共享 |
| MGTOW ↔ MRA | 0.654 | MGTOW 由 MRA 成员创建 |
| MGTOW ↔ PuA | 0.282 | 意识形态冲突（追求 vs 回避女性） |
| Incels ↔ 所有其他 | <0 (min -0.240) | 最独特的语言，独立起源 |

这些发现与社会学文献完全吻合，验证了方法的有效性。

## 亮点与洞察

- **首次在词汇归纳中整合社会和时间维度**: 现有方法只看语言特征，LISTN 同时建模"谁用什么词"和"谁和谁互动"
- **嵌入空间可解释性强**: 不仅能归纳词汇，还能分析词的时间稳定性和子群体专业化，产出了有社会学价值的洞察
- **测试集构建严谨**: 结合现有词典 + 专家标注，Cohen's Kappa 0.726
- **输出有直接应用价值**: 发布 455 个新 manosphere 术语及其子社区相关性评分

## 局限与展望

1. **每个词只有一个评分**: 不处理一词多义或 dogwhistle（在群体内外有不同含义的词）
2. **仅评估单 token 词**: 排除了多词表达（如 "all women are like that" → AWALT）
3. **训练数据为 2018 年**: 群体内语言快速演化，词典的时效性有限
4. **Reddit 数据访问受限**: 难以获取更新的数据来验证泛化性
5. **可扩展方向**: 结合 LLM 进行 dogwhistle 检测；多语言群体内语言研究；与 NPMI 方法组合（两者关注频谱的不同区域）

## 相关工作与启发

- **Lucy & Bamman (2021)**: 用 BERT 嵌入 + 统计特征做词汇归纳，不考虑时间和社会因素
- **Farrell et al. (2020)**: 用话题模型和 word2vec，发现子群体间术语使用差异
- **Danescu-Niculescu-Mizil et al. (2013)**: 发现不采用词汇创新预示用户离开社区
- **Stewart & Eisenstein (2018)**: 研究语言创新传播的语言和社会因素
- **启发**: 群体内语言是研究社会动态的独特窗口——语言的使用模式编码了群体结构、凝聚力和演化方向

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次将社会-时间上下文融入词汇归纳，方法设计有理论深度
- **实验充分度**: ⭐⭐⭐⭐ — 6 种方法变体 + 多种基线 + 专家标注测试集 + 嵌入分析；但只在一个社区（manosphere）上验证
- **写作质量**: ⭐⭐⭐⭐⭐ — 问题动机清晰，分析与社会学文献的对话很有价值，伦理考量到位
- **价值**: ⭐⭐⭐⭐ — 方法通用（不限于特定群体或语言），产出的词典有直接研究价值，嵌入分析揭示了有意义的社会语言学洞察

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Autalic: A Dataset for Anti-Autistic Ableist Language In Context](autalic_a_dataset_for_anti-autistic_ableist_language_in_context.md)
- [\[ACL 2025\] Attention Entropy is a Key Factor for Parallel Context Encoding](attention_entropy_parallel_encoding.md)
- [\[ACL 2025\] Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)
- [\[ACL 2025\] The Time Scale of Redundancy between Prosody and Linguistic Context](the_time_scale_of_redundancy_between_prosody_and_linguistic_context.md)
- [\[ACL 2025\] GA-S3: Comprehensive Social Network Simulation with Group Agents](ga-s3_comprehensive_social_network_simulation_with_group_agents.md)

</div>

<!-- RELATED:END -->
