# Dynamic Chunking and Selection for Reading Comprehension of Ultra-Long Context in Large Language Models

**会议**: ACL 2025
**arXiv**: [2506.00773](https://arxiv.org/abs/2506.00773)
**代码**: [GitHub](https://github.com/ECNU-Text-Computing/DCS)
**领域**: llm_efficiency
**关键词**: long-context QA, dynamic chunking, chunk selection, reading comprehension, semantic segmentation

## 一句话总结

提出 Dynamic Chunking and Selection (DCS)，通过基于语义相似度的动态分块和问题感知分类器的块选择，解决长文本固定分块导致的语义断裂问题，在 12 个长文本 QA 数据集上以 Llama3 为基座实现 single-hop 平均 35.50（+28.6%）和 multi-hop 平均 29.07（+20.0%）的提升，且在 256k token 输入下保持鲁棒。

## 研究背景与动机

1. **领域现状**：LLM 在长文本阅读理解中面临两大瓶颈——位置编码限制上下文窗口长度、二次方注意力计算复杂度约束实际可处理长度；同时 LLM 倾向于关注输入首尾（"lost in the middle"现象）。
2. **现有痛点**：当前长文本处理方法（InfLLM、StreamingLLM 等）通常将输入按**固定长度**切分为 chunks，但固定截断往往在句子中间切断，破坏语义完整性。如图 1 所示，"Deep learning"被切分到两个 chunk 中，导致 LLM 无法理解完整含义并给出错误答案。
3. **核心矛盾**：固定长度分块的简单性 vs. 语义连贯性的必要性——要么保持分块简单但丢失语义，要么需要更智能的分块策略。
4. **本文要解决什么**：如何在不改变 LLM 架构的前提下，通过语义感知的动态分块 + 问题相关的块选择，压缩长文本输入并保持回答准确性。
5. **切入角度**：用 Sentence-BERT 编码句子级语义，利用相邻句子间的语义距离自适应确定分块边界；再训练一个轻量分类器根据问题选择相关块。
6. **核心 idea**：基于语义相似度的动态分块保留语义完整性 + 问题感知分类器精选相关块，两个简单模块协同解决长文本理解。

## 方法详解

### 整体框架

DCS 包含两个核心模块：**(1) Dynamic Chunking**——基于语义相似度的自适应分块；**(2) Chunk Selection**——基于问题感知分类器的块筛选。最终将筛选后的块按原始顺序拼接送入 LLM 生成答案。

### 关键设计

**1. Dynamic Chunking（动态分块）**

- 将长文本按标点分句，得到句子序列 $[s_0, s_1, \ldots, s_{n-1}]$
- 用邻居合并扩展上下文：$s'_i = s_{i-1} \oplus s_i \oplus s_{i+1}$
- 用 Sentence-BERT（paraphrase-multilingual-MiniLM-L12-v2）编码获取 embedding
- 计算相邻句子间余弦距离 $\text{dis}(i) = 1 - \text{sim}(i, i+1)$
- 选取距离最大的 top-$(1-\alpha)$ 比例的位置作为分块边界
- 迭代细化：确保每个 chunk 不超过预定义长度 $l$（默认 512 tokens）
- 对过小的 chunk 进行合并，使其尽量接近目标长度

**2. Chunk Selection（块选择）**

- 对每个 chunk $c_i$ 与 question 拼接，送入 LLM 提取特征
- 特征提取策略：取边界 token 的 hidden state + 注意力加权的 context/question 表示，共 6 个 $d$ 维向量
- 用 3 层 MLP 分类器预测每个 chunk 的问题相关性概率 $T_i$
- 根据压缩比 $\alpha_c = l_C / l_T$ 选择 top-k 个最相关的 chunk
- 最终按原始顺序拼接选中的 chunk，配合 initial info 和 question 送入 LLM

### 损失函数

分类器使用二元交叉熵损失训练：

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log \sigma(h_\theta(H_i)) + (1-y_i)\log(1-\sigma(h_\theta(H_i)))]$$

训练数据基于 AdversarialQA，使用负采样策略构建正负样本。

## 实验关键数据

### 主实验（Llama-3-8B-Instruct）

| 方法 | Single-hop Avg | Multi-hop Avg | 总 Avg |
|------|---------------|--------------|--------|
| Llama3 原始 | 27.60 | 24.22 | - |
| + StreamingLLM | 24.26 | 22.51 | - |
| + LM-Infinite | 24.20 | 22.79 | - |
| + InfLLM | 27.15 | 23.58 | - |
| **+ DCS (Ours)** | **35.50** | **29.07** | - |

Single-hop 代表性成绩：Loogle_SD 45.10（原始 21.25），Factrecall 29.89（原始 15.50）。Multi-hop：Musique 28.90（原始 21.72），HotpotwikiQA 25.40（原始 14.22）。

### 消融实验

| 组件 | Single-hop | Multi-hop | Avg |
|------|-----------|----------|-----|
| Llama3 + DC (动态分块) | 38.10 | 38.06 | 38.08 |
| Llama3 + FC (固定分块) | 36.66 | 37.26 | 36.96 |

动态分块一致优于固定分块，平均提升 1.12。

### 超参数选择

| Chunk 长度 $l$ | Avg |
|---------------|-----|
| 256 | 37.33 |
| **512** | **38.08** |
| 768 | 36.87 |
| 1024 | 36.72 |

| 阈值 $\alpha$ | Avg |
|--------------|-----|
| 55 | 37.43 |
| **60** | **38.08** |
| 65 | 37.67 |
| 70 | 36.97 |

### 关键发现

- DCS 在 Mistral 上提升 5.8%（single-hop）和 7.6%（multi-hop），Vicuna 上提升 24.9% 和 7.3%
- 在 16k→256k 输入长度范围内，DCS 性能衰减明显小于 baseline，尤其 64k 以上优势拉大
- MLP 分类器显著优于余弦相似度选择方案

## 亮点与洞察

1. **直觉清晰且有效**：用语义距离确定分块边界，避免固定切割的语义断裂，想法自然且论证充分
2. **训练开销极低**：只需训练一个 3 层 MLP 分类器，不需要微调 LLM 本身
3. **鲁棒性强**：在 256k 超长文本上依然保持稳定性能，远超 baseline 的衰减曲线
4. **即插即用**：可搭配任意 LLM（Llama3/Mistral/Vicuna），无需修改模型架构
5. **迭代细化策略**巧妙：先粗分再合并，确保每个 chunk 既不超长也不过碎

## 局限性 / 可改进方向

1. **分类器依赖基座 LLM 的特征提取**：需要对每个 chunk-question pair 做一次前向传播获取 hidden states，当 chunk 数量多时开销不小
2. **仅在 QA 任务上验证**：未涉及摘要、翻译等其他长文本任务
3. **分类器训练数据来自短文本**（AdversarialQA），能否推广到更多领域值得探讨
4. **基座模型规模有限**（7B-8B），未验证更大模型或商业 API 上的效果
5. **Sentence-BERT 编码质量**对最终分块效果有直接影响，不同领域文本可能需要调整

## 相关工作与启发

- 与 **InfLLM** 相比：InfLLM 用固定分块+记忆单元索引，DCS 用动态分块+分类器选择，方向互补
- 与 **HMT** 相比：HMT 模拟人类分层记忆，DCS 更简洁直接
- 与 **Token Eviction**（H2O/TOVA）相比：token 级别淘汰破坏原始语义结构，chunk 级别操作更保真
- **chunk 级别处理是长文本理解的有效粒度**——既不像 token 级太细碎，也不像整段太粗糙

## 评分

- 新颖性: ⭐⭐⭐ — 动态分块思路不新，但语义距离驱动+问题感知分类器的组合设计有效
- 实验充分度: ⭐⭐⭐⭐ — 12 个数据集、3 个 LLM、超参消融、ablation、超长文本测试，覆盖全面
- 写作质量: ⭐⭐⭐⭐ — 公式推导清晰，图示直观，动机阐述到位
- 价值: ⭐⭐⭐⭐ — 简单有效的长文本处理方案，实用性强，适合工程落地
