# Length-Induced Embedding Collapse in PLM-based Models

**会议**: ACL 2025 (Findings)
**arXiv**: [2410.24200](https://arxiv.org/abs/2410.24200)
**代码**: [GitHub](https://github.com/Yuqi-Zhou/Length_Collapse)
**领域**: LLM/NLP
**关键词**: text embedding, length collapse, self-attention, low-pass filter, temperature scaling

## 一句话总结
发现并严格证明了 PLM 文本嵌入模型中的"长度坍缩"现象——长文本嵌入趋于聚集，源于 self-attention 作为低通滤波器随文本长度增加而滤波率增强，高频信息被过度抑制；提出 TempScale 方法通过降低 attention 温度来缓解长短文本嵌入分布差异，在 MTEB 上提升 0.94%、LongEmbed 上提升 1.10%。

## 研究背景与动机

1. **领域现状**：PLM-based 嵌入模型（如 BGE、E5 等）将文本编码为固定维度向量，广泛用于检索、分类、STS 等任务。
2. **现有痛点**：嵌入模型在长文本上性能明显下降。BGE 在 IMDB 分类中，token 数从 [0,100) 到 [400,500) 准确率从 75.6% 降至 59.0%（降 16.6%）。但原因不明。
3. **核心矛盾**：**为什么**长文本嵌入性能差？不是简单的信息量过大——而是长文本嵌入之间变得过于相似，失去区分度。
4. **本文要解决什么**：(a) 定义并验证 Length Collapse 现象；(b) 从频域理论角度给出机制性解释；(c) 提出缓解方法。
5. **切入角度**：将 self-attention 在频域分析为低通滤波器（沿用 Wang et al., 2022 对 ViT 的分析），证明滤波率与序列长度 $n$ 的关系。
6. **核心 idea**：self-attention 矩阵的高频分量最大奇异值 $\sigma_a$ 随 $n$ 增大而减小 → 长文本的 token 特征趋同（仅保留 DC 分量）→ pooling 后嵌入聚集 = Length Collapse。

## 方法详解

### 整体框架
理论分析：证明 $\sigma_a$ 随序列长度 $n$ 单调递减 → 解释 Length Collapse 的机制。实验验证：提出 TempScale 调节 attention temperature 缓解此现象。

### 关键设计

1. **Length Collapse 的频域分析**
   - **Lemma 1**：attention 矩阵 $\mathbf{A} = \text{softmax}(\mathbf{P})$ 是低通滤波器——重复应用 $\mathbf{A}$ 会让信号只保留 DC 分量
   - **Theorem 2**：高频衰减率由 $\sigma_a \|\mathbf{W}_V\|_2$ 控制。$\sigma_a$ 越小 → 高频被压制越快
   - **Theorem 3**：假设 query/key 为高斯分布，证明 $\sigma_a \leq \sqrt{\frac{n}{2\sqrt{1+e^{-2\sigma_s^2}}(n-1)^{3/2}+1}}$，**随 $n$ 增大单调递减**
   - **Corollary 4**：$n$ 增大 → $\sigma_a$ 减小 → token 特征趋同 → 嵌入 cosine 相似度升高 = Length Collapse

2. **Length Collapse 对下游任务的影响分析**
   - **分类/聚类**：长文本嵌入聚集在中心 → KNN 分类器偏向长文本 → 准确率下降
   - **检索**：长文档嵌入空间被压缩 → 短噪声文档可能比真正相关的长文档 ranking 更高
   - **STS**：长文本对之间高相似度 → 不相关长文本对也高分，区分度差

3. **TempScale 方法**
   - 做什么：在 attention score 除以 $\sqrt{d}$ 后再除以温度 $\tau < 1$，即 $\mathbf{A} = \text{softmax}(\frac{\mathbf{XW}_Q(\mathbf{XW}_K)^\top}{\tau\sqrt{d}})$
   - 核心思路：$\tau < 1$ 等效于增大 $\sigma_s$、减小 attention 的"温度" → 让 $\sigma_a$ 对 $n$ 变化更不敏感 → 缩小长短文本的滤波率差距
   - 设计动机：极端情况分析——$\tau \to 0$ 时 attention 变成 one-hot（无滤波但丢失聚合能力），$\tau \to 1$ 时保持原始行为。最优 $\tau$ 在两者之间
   - **无需重新训练**：推理时直接修改 attention 计算即可

## 实验关键数据

### 主实验

| Benchmark | Base Model | 原始 | +TempScale | 提升 |
|-----------|-----------|------|-----------|------|
| MTEB (56 tasks) | BGE-base | 63.55 | **64.49** | +0.94% |
| MTEB | E5-large | 66.23 | **67.01** | +0.78% |
| LongEmbed | BGE-base | 42.15 | **43.25** | +1.10% |
| LongEmbed | E5-4K | 56.82 | **57.89** | +1.07% |

### 按文本长度的分类准确率 (BGE on IMDB)

| Token 范围 | 原始 | +TempScale |
|-----------|------|-----------|
| [0, 100) | 75.6 | 75.8 |
| [100, 200) | 72.3 | 73.5 |
| [200, 300) | 65.1 | 67.2 |
| [300, 400) | 61.5 | 64.0 |
| [400, 500) | 59.0 | 62.3 |

### 关键发现
- **Length Collapse 是普遍现象**：在 BGE/E5/GTE/ANCE 等主流嵌入模型上都存在
- **长文本 pairwise cosine similarity 随长度单调增加**：从 [0,100) 的 ~0.3 到 [400,500) 的 ~0.6，验证了 Corollary 4
- **TempScale 对长文本改善最大**：[400,500) 区间提升最显著，短文本几乎不受影响
- **$\sigma_a$ 与 $n$ 的关系实验验证**：从 BGE 最后一层 attention 矩阵提取的 $\sigma_a$ 确实随文本长度递减
- **最优 $\tau$ 在 $[0.5, 0.8]$ 之间**：过低的 $\tau$ 导致 attention 退化为 one-hot

## 亮点与洞察
- **频域分析的优雅性**：从 Fourier 视角将 attention 理解为低通滤波器，再推导长度对滤波率的影响——因果链清晰：$n \uparrow \to \sigma_a \downarrow \to$ 高频被压制 $\to$ embeddings 聚集
- **推理时免训练的解决方案**：TempScale 不需要重新训练模型，只在推理时修改 attention temperature——对部署中的嵌入模型直接可用
- **理论预测与实验完美对应**：Theorem 3 预测 $\sigma_a$ 随 $n$ 递减，Figure 7 实验验证；Corollary 4 预测 cosine similarity 随 $n$ 增加，Figure 1(c) 验证
- **对 RAG/长文本检索有直接价值**：长文档检索中长文本嵌入退化是实际瓶颈，TempScale 提供了简单有效的缓解

## 局限性 / 可改进方向
- **高斯假设**：Theorem 3 假设 query/key 为高斯分布，实际 PLM 中可能不完全满足
- **仅单一超参 $\tau$**：全局统一的温度；可以考虑层级或头级别的自适应温度
- **未考虑因果注意力（decoder-only）**：分析针对双向 attention（encoder），causal attention 的 Length Collapse 需要额外研究
- **提升幅度有限**：MTEB +0.94% / LongEmbed +1.10%，虽然一致性好但绝对值不大
- **未与其他长文本方法结合**：如 position interpolation, RoPE 缩放等

## 相关工作与启发
- **vs Wang et al. (2022) ViT over-smoothing**：他们发现 ViT 中 self-attention 随深度增加导致 over-smoothing，本文发现**随长度增加**也有类似效果——两种 over-smoothing 共享同一机制
- **vs rotary position embedding**：RoPE 通过旋转改善 attention 对长度的泛化，但不直接解决低通滤波问题——可以结合 TempScale
- **vs chunking strategies**：将长文档切成短段分别编码是工程解法，TempScale 是理论驱动的模型级解法

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次定义 Length Collapse 并给出严格频域证明，Theorem 3 的推导全新
- 实验充分度: ⭐⭐⭐⭐ MTEB + LongEmbed 验证 + 多模型 + 可视化，全面
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导层层递进，图示直观（Figure 1 完美展示问题），论证严密
- 价值: ⭐⭐⭐⭐⭐ 对嵌入模型的长文本问题提供了根本性理解，TempScale 实用且免训练
