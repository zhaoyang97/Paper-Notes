# Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering

**会议**: ACL 2025  
**arXiv**: [2503.01606](https://arxiv.org/abs/2503.01606)  
**代码**: 无  
**领域**: LLM效率  
**关键词**: open-domain QA, embedding-level reranking, contrastive learning, exploratory embedding, entropy selection

## 一句话总结
EmbQA 提出嵌入级 ODQA 框架，用轻量线性层和无监督对比学习优化查询表示实现段落重排序，并引入基于序统计量的探索性嵌入扩展候选答案多样性，配合熵选择机制自动选答，在 4 个 ODQA 数据集上以更低计算成本超越 SuRe 等 prompt 级方法。

## 研究背景与动机
1. **领域现状**：ODQA 通常采用 retriever-reader 流水线，先从大规模语料检索相关段落，再用 LLM 作为 reader 生成答案。prompt 级框架（SuRe、Self-Verification、CoT）通过多轮提示来改善答案质量
2. **现有痛点**：
   - Retriever 返回大量候选段落，但含正确答案的段落排名靠后（top-k recall 低）
   - 现有 prompt 级 reranking 方法（逐段让 LLM 评分）计算量大，且受限于 context window 只能处理少量段落
   - Reader 端依赖多轮 prompt（摘要、自验证、候选选择），每轮都需完整 LLM 推理，效率低且不稳定
3. **核心矛盾**：提升 ODQA 准确率通常需要更多轮 prompt 交互，但每轮交互都带来显著计算开销
4. **切入角度**：用嵌入级操作（轻量线性层 + 单 token embedding）替代 prompt 级多轮交互
5. **核心idea一句话**：在 embedding 空间操作来同时优化检索重排和答案生成多样性，避免多轮 prompt 的计算开销

## 方法详解

### 整体框架
EmbQA 分两阶段：(1) **Retriever 重排**：先用标准检索获取 top-N 段落，让 LLM 生成 K=2 个候选答案，然后用候选答案引导无监督对比学习训练两个线性层 $W_1, W_2$，生成新的查询嵌入 $\mathbf{e}_{q_{new}} = W_1 \mathbf{e}_y + W_2 \mathbf{e}_q$，用新查询重新排序段落；(2) **Reader 生成**：注入一个基于序统计量筛选的探索性嵌入 $\mathbf{e}_r$ 来扩展语义空间，生成更多样的候选答案，最后用熵值最低的候选作为最终答案。

### 关键设计

1. **嵌入级重排序（Self-Refinement Driven Reranking）**:
   - 做什么：用 LLM 生成的候选答案引导查询表示优化，实现段落重排
   - 核心思路：冻结检索器参数，只训练两个线性层 $W_1, W_2$。将候选答案嵌入和原始查询嵌入线性组合得到新查询 $\mathbf{e}_{q_{new}}$，用无监督对比学习训练——含候选答案的段落为正样本，不含的为负样本（5:1 采样）
   - 设计动机：相比 prompt 级 reranking（逐段让 LLM 打分），只训练两个矩阵的开销极小，且能遍历整个知识库而非仅处理 top-k

2. **探索性嵌入（Exploratory Embedding）**:
   - 做什么：在推理时向 query 中注入一个随机采样的 token 级嵌入，引导模型探索不同语义方向
   - 核心思路：从标准正态分布采样 $\mathbf{e}_r \in \mathbb{R}^D$，与 query 和检索上下文拼接后输入 LLM。为保证多样性，利用序统计量——将 $\mathbf{e}_r$ 经 LLM 后的隐层表示 $\mathbf{h}_r$ 按降序排列，计算 $S_{\mathbf{e}_r} = \sum_{i=1}^p \Delta_{(i)}^2$（相邻元素差的平方和），只保留 $S$ 低于阈值 $T$ 的嵌入
   - 设计动机：基于 Jain et al. 的理论，最小化向量间内积等价于最大化正交性，而 $S$ 是其 Gaussian 近似下的高效代理指标。只需一个 token 大小的嵌入即可激活 LLM 中的不同知识路径

3. **熵选择机制（Entropy-Based Selection）**:
   - 做什么：用输出 logit 的熵值自动选择最终答案，无需额外 prompt 轮次
   - 核心思路：$\hat{a} = \arg\min_{\hat{y}} \text{Entropy}(\hat{y})$，选择熵最低（最确信）的候选
   - 设计动机：低熵 ↔ 高置信度，替代 SuRe 中需要多轮 LLM 推理的摘要+投票策略

## 实验关键数据

### 主实验
以 LLaMA 3.1 8B + BM25 为基准：

| 方法 | HotpotQA EM | 2Wiki EM | NQ EM | WebQ EM | Avg EM | Avg F1 |
|------|------------|----------|-------|---------|--------|--------|
| Retrieval Only | 25.4 | 16.6 | 26.0 | 22.2 | 22.6 | 30.6 |
| Chain-of-Thought | 27.0 | 15.4 | 27.2 | 28.8 | 24.6 | 33.2 |
| Self-Verification | 32.8 | 21.0 | 28.0 | 27.2 | 27.4 | 38.0 |
| SuRe | 38.8 | 23.8 | 36.6 | 34.4 | 33.4 | 45.3 |
| **EmbQA** | **42.0** | **27.4** | **42.2** | **38.2** | **37.5** | **49.7** |

EmbQA 比 SuRe 平均 EM 高 +4.1，F1 高 +4.4

### 跨检索器/跨模型
| 模型 + 检索器 | SuRe Avg EM | EmbQA Avg EM | 提升 |
|-------------|------------|-------------|------|
| LLaMA3.1 + BM25 | 33.4 | 37.5 | +4.1 |
| LLaMA3.1 + DPR | 28.6 | 31.9 | +3.3 |
| LLaMA3.1 + Contriever | 32.1 | 35.3 | +3.2 |
| Mistral + BM25 | 29.2 | 31.3 | +2.1 |

### 消融实验
| 配置 | Avg EM | 说明 |
|------|--------|------|
| EmbQA Full | 37.5 | 完整模型 |
| w/o Reranking | 34.2 | 去掉嵌入级重排，掉 3.3 |
| w/o Exploratory Embedding | 35.1 | 去掉探索性嵌入，掉 2.4 |
| w/o Entropy Selection | 36.0 | 去掉熵选择，掉 1.5 |

### 关键发现
- **嵌入级重排是最关键模块**：去掉后掉 3.3 EM，因为检索质量直接影响下游所有模块
- **探索性嵌入有效**：引入单个 token 嵌入就能显著扩展候选多样性
- **跨检索器一致有效**：BM25、DPR、Contriever 三种检索器上均优于 SuRe
- **效率优势明显**：嵌入级操作比 prompt 级（SuRe 需多轮 LLM 推理）快数倍

## 亮点与洞察
- **嵌入级 vs Prompt 级的范式迁移**是核心洞察：很多 prompt 级操作（重排、验证、选择）可以用轻量的嵌入操作替代，大幅降低计算量。这个思路可迁移到其他 RAG 任务
- **单 token 嵌入激活不同知识路径**的想法很有趣——本质上是在 latent space 做可控扰动来增加输出多样性，比 temperature sampling 更有理论依据
- **序统计量筛选嵌入**提供了一个有原理支撑的多样性度量，比随机采样更可靠

## 局限性 / 可改进方向
- 对比学习中正负样本用"是否包含候选答案"判定，可能引入假负例（段落含正确信息但未被候选答案匹配）
- 探索性嵌入的阈值 $T$ 需要手动调节，不同数据集可能需要不同值
- 只在 7-8B 模型上验证，更大模型上嵌入空间的行为可能不同
- 序统计量的 Gaussian 假设在 LLM hidden states 上可能不完全成立

## 相关工作与启发
- **vs SuRe**: SuRe 用摘要+投票的 prompt 级策略，多轮 LLM 推理；EmbQA 用嵌入操作+熵选择，快且稳定
- **vs RPG/KnowTrace**: 同为 prompt 级方法，EmbQA 在四个数据集上均优于它们
- **vs Prompt 级 reranking**: EmbQA 只训练两个线性层（秒级），而 prompt 级需要逐段落 LLM 推理（分钟级）

## 评分
- 新颖性: ⭐⭐⭐⭐ 嵌入级框架替代 prompt 级是有意义的范式转变，探索性嵌入设计有理论支撑
- 实验充分度: ⭐⭐⭐⭐⭐ 3 个 LLM × 3 个检索器 × 4 个数据集，含详细消融
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述详细，理论分析有深度
- 价值: ⭐⭐⭐⭐ 实用的 RAG 效率提升方案，嵌入级操作思路可广泛迁移
