---
title: >-
  [论文解读] LotusFilter: Fast Diverse Nearest Neighbor Search via a Learned Cutoff Table
description: >-
  [CVPR 2025][信息检索/RAG][多样化最近邻搜索] 提出LotusFilter，通过离线预计算每个向量的邻近关系构建截断表(cutoff table)，在线阶段用贪心集合删除实现多样化过滤，将传统 $O(DS^2)$ 的多样化搜索降至 $O(T+S+KL)$，过滤仅需0.02ms/query，内存仅为传统方法的1/40。
tags:
  - "CVPR 2025"
  - "信息检索/RAG"
  - "多样化最近邻搜索"
  - "截断表"
  - "OrderedSet数据结构"
  - "超参数学习"
  - "RAG检索去重"
---

# LotusFilter: Fast Diverse Nearest Neighbor Search via a Learned Cutoff Table

**会议**: CVPR 2025  
**arXiv**: [2506.04790](https://arxiv.org/abs/2506.04790)  
**代码**: [https://github.com/matsui528/lotf](https://github.com/matsui528/lotf)  
**领域**: 向量检索 / 多样化搜索  
**关键词**: 多样化最近邻搜索、截断表、OrderedSet数据结构、超参数学习、RAG检索去重

## 一句话总结
提出LotusFilter，通过离线预计算每个向量的邻近关系构建截断表(cutoff table)，在线阶段用贪心集合删除实现多样化过滤，将传统 $O(DS^2)$ 的多样化搜索降至 $O(T+S+KL)$，过滤仅需0.02ms/query，内存仅为传统方法的1/40。

## 研究背景与动机

**领域现状**：近似最近邻搜索(ANNS)是RAG、推荐系统、图像检索等应用的核心组件。HNSW、IVF等现代ANNS方法已能在百万级数据上实现亚毫秒级搜索，但它们返回的Top-K结果往往高度相似——例如用猫的图片搜索，可能返回同一只猫的多张近似照片；用医疗问题检索RAG文档，可能返回几乎相同的段落。

**现有痛点**：多样化最近邻搜索(DNNS)是解决此问题的经典方向，但现有方法存在三个核心瓶颈：(1) 从 $S$ 个候选中选 $K$ 个子集是NP-hard问题，朴素枚举代价为 $\binom{S}{K}$；(2) 需要计算候选间的两两距离，复杂度至少 $O(DS^2)$，在高维场景下极慢；(3) 过滤阶段需要访问原始向量，若向量未全部驻留内存则涉及慢速磁盘IO。

**核心矛盾**：现代ANNS方法（如HNSW）通常只存储压缩后的向量表示，传统DNNS方法无法直接作为其后处理模块使用，导致"快速搜索+慢速多样化"的瓶颈。例如GMM方法需要 $O(DKS)$ 复杂度并访问原始高维向量，对 $D=1536$ 的OpenAI embedding来说，仅过滤一个查询就需要13.4ms，是搜索本身时间的15倍以上。用户需要的是一个纯后处理方案，能黑盒接入任意ANNS引擎。

**本文目标**：设计一个极轻量的后处理过滤器，在不访问原始向量的前提下快速实现搜索结果多样化，且可与任意ANNS方法即插即用。

**切入角度**：将运行时的距离计算完全转移到离线阶段——为每个数据库向量预先记录其"太近的邻居"列表，运行时只需做简单的集合删除操作。

**核心 idea**：预计算一张截断表记录每个向量的近邻ID集合，查询时贪心地弹出最近候选并删除其近邻，用 $O(1)$ 的集合操作替代 $O(D)$ 的距离计算。

## 方法详解

### 整体框架
LotusFilter分为离线预处理和在线过滤两个阶段。

**离线阶段**：(1) 用任意ANNS方法构建搜索索引 $\mathcal{I}$；(2) 对每个数据库向量 $\mathbf{x}_n$ 做范围搜索，收集距离平方小于阈值 $\varepsilon$ 的邻居ID集合 $\mathcal{L}_n$，构成截断表；(3) 通过bracketing优化学习最优 $\varepsilon^*$。

**在线阶段**：(1) 用ANNS获取 $S$ 个候选 $\mathcal{S}$；(2) 将 $\mathcal{S}$ 加载为OrderedSet数据结构；(3) 贪心循环：弹出距查询最近的候选 $k$ 加入结果集 $\mathcal{K}$，从 $\mathcal{S}$ 中删除 $\mathcal{L}_k$ 中的所有ID；(4) 重复直到 $|\mathcal{K}|=K$。整个过滤过程不涉及任何浮点运算，仅操作整数ID集合。

### 关键设计
1. **截断表(Cutoff Table)预计算**:
    - 功能：离线存储每个向量的近邻关系，运行时避免任何距离计算
    - 核心思路：对每个 $\mathbf{x}_n$，通过范围搜索找到 $\mathcal{L}_n = \{i \mid \|\mathbf{x}_n - \mathbf{x}_i\|_2^2 < \varepsilon, n \neq i\}$。截断表是一个"整数数组的数组"，每条记录存储距离小于 $\varepsilon$ 的邻居ID。平均长度为 $L = \frac{1}{N}\sum_{n=1}^N |\mathcal{L}_n|$，内存消耗为 $64LN$ bit（使用64位整数）
    - 设计动机：传统DNNS方法在过滤阶段需实时计算 $O(S^2)$ 对距离，代价 $O(DS^2)$。截断表将所有距离比较预先完成，运行时仅需 $O(1)$ 的哈希表查找判断两个向量是否"太近"。对 $N=9\times10^5$ 的OpenAI数据集，截断表仅占136 MiB，构建时间约54秒

2. **OrderedSet数据结构**:
    - 功能：高效支持过滤循环中的Pop（弹出最小元素）和Remove（删除指定元素）操作
    - 核心思路：同时维护原始数组 $\mathbf{v}$ 和其对应的哈希集合 $\mathcal{V}$，以及头指针 $c$。Remove操作只在 $\mathcal{V}$ 中删除元素（浅删除），代价 $O(1)$；Pop操作从 $c$ 位置沿数组扫描直到找到仍在 $\mathcal{V}$ 中的元素，代价 $O(\Delta)$，其中 $\Delta \leq L$（因为两次Pop之间最多删除 $L$ 个元素）。使用boost::unordered_flat_set实现常数时间哈希操作
    - 设计动机：朴素数组实现Remove需 $O(V)$ 线性搜索；朴素哈希集合无法实现Pop（无序）；优先队列的删除代价为 $O(\log V)$。OrderedSet以额外内存为代价，同时获得 $O(1)$ Remove和 $O(L)$ Pop，使整个过滤循环达到 $O(KL)$

3. **基于Bracketing的阈值 $\varepsilon$ 学习**:
    - 功能：自动选择最优截断阈值 $\varepsilon^*$，平衡搜索相关性与多样性
    - 核心思路：将目标函数 $f(\mathcal{K}) = \frac{1-\lambda}{K}\sum_{k \in \mathcal{K}}\|\mathbf{q}-\mathbf{x}_k\|_2^2 - \lambda\min_{i,j \in \mathcal{K},i\neq j}\|\mathbf{x}_i-\mathbf{x}_j\|_2^2$ 视为 $\varepsilon$ 的单变量函数，使用训练查询集 $\mathcal{Q}_{\text{train}}$ 求解 $\varepsilon^* = \arg\min_\varepsilon \mathbb{E}_{\mathbf{q} \in \mathcal{Q}_{\text{train}}}[f^*(\varepsilon, \mathbf{q})]$。采用bracketing方法递归缩小搜索区间
    - 设计动机：$\varepsilon$ 过小则多样性不足，过大则截断表膨胀且过度裁剪导致候选不足。该参数对数据分布敏感，手动调参不现实。通过从数据库中抽取前1000个向量作为训练查询，可以低成本地自动学习最优值

### 损失函数 / 训练策略
目标函数包含两项：相关性项 $\frac{1-\lambda}{K}\sum_{k \in \mathcal{K}}\|\mathbf{q}-\mathbf{x}_k\|_2^2$（越小越好）和多样性项 $-\lambda\min_{i,j \in \mathcal{K},i\neq j}\|\mathbf{x}_i-\mathbf{x}_j\|_2^2$（越小表示最近对越远，多样性越好）。参数 $\lambda \in [0,1]$ 控制两者权衡：$\lambda=0$ 退化为标准NNS，$\lambda=1$ 退化为MAX-MIN多样化问题。LotusFilter提供理论保证：过滤后结果集中任意两个向量距离平方 $\|\mathbf{x}_i - \mathbf{x}_j\|_2^2 \geq \varepsilon$，即多样性项有界。此外设有safeguard模式：若过度裁剪导致候选集耗尽，立即终止过滤并将剩余候选直接加入结果集，确保 $|\mathcal{K}|=K$。

## 实验关键数据

### 主实验
**OpenAI数据集** ($N=900\text{K}$, $D=1536$, $\lambda=0.3$, $K=100$, $S=500$):

| 方法 | 综合得分 $f$ ↓ | 搜索时间(ms) | 过滤时间(ms) | 总时间(ms) | 内存开销(bit) |
|------|-------------|------------|------------|----------|-------------|
| 纯ANNS(无过滤) | 0.200 | 0.855 | - | 0.855 | - |
| K-means聚类 | 0.223 | 0.941 | 6.94 | 7.88 | $4.42\times10^{10}$ |
| GMM | **0.177** | 0.977 | 13.4 | 14.4 | $4.42\times10^{10}$ |
| **LotusFilter** | **0.171** | 1.00 | **0.02** | **1.03** | $1.14\times10^9$ |

**预处理时间与截断表规模**:

| $N$ | $\lambda$ | 学习的 $\varepsilon^*$ | 平均长度 $L$ | 训练时间(s) | 构建时间(s) |
|-----|---------|-----------------|------------|-----------|-----------|
| $9\times10^3$ | 0.3 | 0.39 | 8.7 | 96 | 0.16 |
| $9\times10^4$ | 0.3 | 0.33 | 10.1 | 176 | 3.8 |
| $9\times10^5$ | 0.3 | 0.27 | 18.4 | 1020 | 54 |
| $9\times10^5$ | 0.5 | 0.29 | 29.3 | 1087 | 54 |

### 消融实验
**初始候选数 $S$ 的影响** ($K=100$, $\lambda=0.3$, OpenAI数据集):

增大 $S$ 意味着ANNS返回更多候选供过滤，理论上可获得更优解但运行时间线性增长。实验显示 $S$ 从200增至500时 $f$ 显著下降，之后收益递减：

| $S$ | 综合得分 $f$ ↓ | 总时间(ms) | 趋势 |
|-----|-------------|----------|------|
| 200 | ~0.185 | ~0.9 | 候选少，多样性受限 |
| 500 | 0.171 | 1.03 | 最佳平衡点 |
| 1000 | ~0.168 | ~1.2 | 继续提升但收益递减 |
| 2000 | ~0.166 | ~1.5 | 运行时间增加明显 |

### 关键发现
- LotusFilter以0.02ms/query过滤时间达到最优综合得分(0.171)，总时间仅为GMM的1/14、聚类的1/8
- 内存开销仅 $1.14\times10^9$ bit(136 MiB)，是传统方法(需存原始向量 $4.42\times10^{10}$ bit)的**1/40**
- 截断表平均长度 $L$ 在实验范围内最大仅30，保证了 $O(KL)$ 过滤的实际效率
- 定性实验(MS MARCO)：查询"扁桃体炎是发生在扁桃体上的喉咙感染"，纯NNS返回的Top-5中有3条几乎一字不差的"链球菌性咽喉炎是喉咙和扁桃体的细菌感染"，LotusFilter成功去重后返回了扁桃体炎、乳突炎、海绵状皮炎等多样结果
- 定性实验(Revisited Paris)：查询巴黎蓬皮杜中心照片，NNS返回5张几乎相同构图，LotusFilter返回不同角度和距离拍摄的多样化结果
- Top-1的Recall不受影响——第一轮循环必然保留初始搜索的最近邻
- 随着 $N$ 增大，$\varepsilon^*$ 逐渐减小而 $L$ 增大，说明数据越密集时需要更精细的截断阈值
- $\lambda$ 从0.3增至0.5时，$L$ 从18.4增至29.3，说明用户要求更高多样性时截断表相应膨胀但仍可控

## 亮点与洞察
- **预计算替代实时计算**：将 $O(DS^2)$ 的在线两两距离计算完全转移为离线截断表构建，运行时仅做整数集合操作。这种"以空间换时间+离线换在线"的思路在高维场景下收益极大，因为传统方法的代价与 $D$ 成正比而LotusFilter与 $D$ 无关
- **OrderedSet数据结构设计**：通过同时维护数组（保序）和哈希集合（快删），用额外内存换取Pop $O(L)$ + Remove $O(1)$的组合。该设计精准匹配了贪心过滤的操作模式，比优先队列或跳表更简洁高效
- **纯后处理的架构优势**：LotusFilter不修改ANNS索引、不访问原始向量，可作为黑盒模块插入任何ANNS管线。这意味着如果多样化效果不佳，直接关闭过滤器即可回退，零侵入性设计使得工程落地成本极低
- **Safeguard机制的实用性**：当 $\varepsilon$ 较大时可能过度裁剪候选集，safeguard模式提前终止过滤并直接填充剩余候选，保证结果数始终为 $K$。这种graceful degradation设计在工程实践中非常重要

## 局限与展望
- 全局单一阈值 $\varepsilon$ 对数据分布不均匀的场景可能不够灵活，密集区域可能过度裁剪、稀疏区域过滤不足，未来可探索区域自适应阈值
- 学习 $\varepsilon$ 时需预先确定 $K$，若运行时 $K$ 变化则 $\varepsilon^*$ 可能非最优，限制了动态查询场景的灵活性
- 仅有目标函数 $f$ 的多样性项下界保证，缺少对总目标的理论最优性证明，无法量化与最优解的差距
- 截断表内存 $64LN$ bit对超大规模数据集(如 $N>10^8$)可能不可忽略，需要探索压缩截断表的方案
- 低维向量场景下简单方法（如GMM）可能更优，LotusFilter的优势主要体现在高维（如 $D>768$）
- 尚未做端到端RAG评测（如LLM-as-a-judge），不确定多样化搜索对下游LLM生成质量的实际提升幅度
- 名称"LotusFilter"源于过滤过程中以每个向量为圆心删除邻近点，形似荷叶覆盖水面的视觉意象

## 相关工作与启发
- **vs GMM (Greedy Max-Min)**: GMM通过贪心迭代选择与已选集合距离最远的点，多样性最强但需 $O(DKS)$ 实时计算。LotusFilter在近似多样性的同时，过滤时间仅为GMM的1/670(0.02ms vs 13.4ms)
- **vs MMR (Maximal Marginal Relevance)**: MMR无法直接利用现代ANNS方法，需在全库上计算，实际运行极慢。LotusFilter作为纯后处理模块完美兼容HNSW/IVF等ANNS引擎
- **vs Learned Index (学习索引)**: LotusFilter延续了learned data structure的范式（如Learned B-tree通过学习数据分布优化索引），但创新地将学习应用于多样化阈值而非索引结构本身
- **vs Hirata et al. (现代ANNS+DNNS)**: 唯一将现代ANNS应用于diverse inner product search的工作，但仍需在线距离计算。LotusFilter通过彻底消除在线距离计算实现了质的飞跃

## 评分
- 新颖性: ⭐⭐⭐⭐ 预计算截断表+贪心过滤的思路简洁优雅，但核心idea并非全新（类似于Bloom Filter的预计算思想）
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集覆盖文本/图像场景，定量+定性评估完备，含复杂度分析、消融、预处理时间详细报告
- 写作质量: ⭐⭐⭐⭐ 算法描述清晰，理论推导严谨，从问题定义到数据结构设计的逻辑链条完整
- 价值: ⭐⭐⭐⭐⭐ 直击RAG系统重复检索的核心痛点，0.02ms过滤+即插即用设计使其极具工程落地价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] LEMUR: Learned Multi-Vector Retrieval](../../ICML2026/information_retrieval/lemur_learned_multi-vector_retrieval.md)
- [\[ACL 2025\] Drama: Diverse Augmentation from Large Language Models to Smaller Dense Retrievers](../../ACL2025/information_retrieval/drama_diverse_augmentation_from_large_language_models_to_smaller_dense_retriever.md)
- [\[ACL 2025\] FlashBack: Efficient Retrieval-Augmented Language Modeling for Fast Inference](../../ACL2025/information_retrieval/flashbackefficient_retrieval-augmented_language_modeling_for_long_context_infere.md)
- [\[CVPR 2026\] Intra-class Distribution-guided Generative Hashing with Neighbor Refinement for Cross-modal Retrieval](../../CVPR2026/information_retrieval/intra-class_distribution-guided_generative_hashing_with_neighbor_refinement_for_.md)
- [\[ACL 2025\] ARise: Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search](../../ACL2025/information_retrieval/arise_risk_adaptive_search.md)

</div>

<!-- RELATED:END -->
