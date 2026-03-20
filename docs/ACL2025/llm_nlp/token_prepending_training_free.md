# Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs

**会议**: ACL 2025  
**arXiv**: [2412.11556](https://arxiv.org/abs/2412.11556)  
**代码**: [https://github.com/fuyuchenIfyw/token_prepending.git](https://github.com/fuyuchenIfyw/token_prepending.git)  
**领域**: LLM NLP / 句子表示学习  
**关键词**: Sentence Embedding, Token Prepending, Causal Attention, Training-Free, LLM  

## 一句话总结

提出 Token Prepending (TP) 技术，通过在每层将解码得到的句子嵌入前置到句子开头，使因果注意力机制下的早期 token 也能感知完整句子信息，无需训练即可显著提升 LLM 的句子嵌入质量。

## 研究背景与动机

1. **领域现状**:
   - 句子嵌入在信息检索、推荐系统、情感分析、文档聚类等领域有着广泛应用
   - 近年来 LLM 展现出强大的语义理解能力，研究者开始尝试从 LLM 中直接提取句子嵌入而无需额外微调
   - 已有方法如 PromptEOL、MetaEOL、Pretended CoT 等通过提示工程引导 LLM 将句子信息编码到最后一个 token 的嵌入中

2. **现有痛点**:
   - 当前主流 LLM 为 decoder-only 架构，使用因果注意力机制（causal attention），导致句子中前面的 token 无法关注到后面的 token
   - 即使最后一个 token 可以关注所有前面的 token，但前面 token 的编码本身就是有偏的（缺少后向依赖信息），这种偏差会级联传递到最终解码的 token
   - 已有方法如 Echo Embeddings 通过重复输入来解决，但显著增加序列长度和推理成本

3. **核心矛盾**:
   - decoder-only LLM 的因果注意力限制了双向信息流动，但 training-free 的句子嵌入方法需要依赖 causal attention 的框架来工作
   - 如何在不改变模型参数、不引入训练的前提下，让因果注意力机制也能实现类似双向注意力的效果？

4. **本文要解决什么？**
   - 在不引入额外训练和显著推理开销的情况下，解决 decoder-only LLM 中因果注意力导致的前向 token 信息编码偏差问题

5. **切入角度**:
   - 从 LLM 的层间信息传递入手，在模型内部层级间插入操作，将每层解码得到的句子嵌入 token 前置到下一层输入的句子开头

6. **核心idea一句话**:
   - 将上一层最后一个 token（包含完整句子信息）的表示前置到下一层句子的开头位置，让因果注意力下所有 token 都能感知完整句子语义

## 方法详解

### 整体框架

TP 方法修改了 LLM 的层间传播过程，主要由三部分组成：
- **Initial Token Prepending**：在输入层，将一个自定义的 `<PST>` 占位 token 前置到句子开头
- **Intermediate Token Prepending**：在前几层，用上一层最后一个 token（SET）的表示替换 `<PST>` 的表示
- **Early Exit Strategy**：从中间层而非最后一层输出句子嵌入

### 关键设计

1. **Initial Token Prepending (初始前置)**:
   - 做什么：在 embedding 层输出后，将一个随机初始化的 `<PST>` token 插入到句子文本的前面位置
   - 核心思路：`<PST>` 作为句子嵌入信息的占位符，使后续 token 可以通过因果注意力机制关注到它
   - 设计动机：在第一层还没有句子嵌入可用，因此使用随机初始化的占位符

2. **Intermediate Token Prepending (中间层前置)**:
   - 做什么：在第 2 层到第 k 层之间，每一层都用上一层最后一个 token 的隐状态（即 SET 的表示）来替换 `<PST>` 位置的表示
   - 核心思路：f(h^{l-1}) 函数将第 n 个位置（最后 token）的表示复制到 i* 位置（PST 位置），使所有句子 token 都能通过因果注意力看到包含完整句子信息的嵌入
   - 设计动机：只在前 k 层执行 TP 操作，因为实验发现在所有层执行反而效果不好；前几层做完 TP 后，句子 token 已经感知了足够的全局信息

3. **Early Exit Strategy (早退策略)**:
   - 做什么：从中间层而非最后一层提取句子嵌入
   - 核心思路：LLM 最后几层主要用于 token 生成，语义信息反而较少
   - 设计动机：中间层保留了更丰富的语义表示，适合作为句子嵌入

### 损失函数 / 训练策略

- **完全无需训练**：TP 不引入任何新的可学习参数，仅在推理时修改层间信息传递
- 仅增加一个 token 到原始序列，推理开销几乎可以忽略不计（约 1.04× 的时间开销）

## 实验关键数据

### 主实验

在 7 个 STS 任务上使用 LLaMA2-7B 的结果：

| 方法 | STS Avg | 推理时间 |
|------|---------|----------|
| PromptEOL | 70.03 | 1.00× |
| **PromptEOL + TP** | **77.19 (↑7.16)** | 1.04× |
| MetaEOL | 75.96 | 8.17× |
| **MetaEOL + TP** | **77.91 (↑1.95)** | 8.29× |
| Pretended CoT | 76.86 | 1.18× |
| **Pretended CoT + TP** | **78.02 (↑1.16)** | 1.22× |

- TP 对 PromptEOL 的提升最为显著，STS-B 提升 9.01 个点
- 对已经较强的 Pretended CoT 也有一致的提升
- 推理时间开销极小，PromptEOL + TP 仅为 1.04×

### 关键发现

1. **TP 在前几层操作效果最佳**：实验表明只在早期层执行 TP 优于在所有层执行
2. **早退策略有效**：从中间层提取嵌入优于从最后一层提取
3. **跨模型泛化**：TP 在 LLaMA2-7B、Mistral-7B 等多个 LLM 上均有效
4. **即插即用**：TP 可与多种现有 prompt 方法无缝结合
5. **Echo Embeddings 对比**：Echo 需要将序列重复一遍（1.67× 开销），而 TP 仅增加一个 token（1.04× 开销），且效果更好

## 亮点与洞察

- **优雅的设计思路**：通过层间信息传递的微小修改，巧妙地在因果注意力框架下实现了类似双向信息流动的效果
- **真正的 training-free**：不需要任何训练数据和微调过程，保留了 LLM 的通用能力
- **几乎零成本**：相比 Echo 的重复输入方法，TP 只增加一个 token，推理开销可忽略
- **通用性强**：作为 plug-and-play 技术，可以叠加在各种 prompt 方法之上

## 局限性 / 可改进方向

1. 目前只在 7B 规模的模型上验证，更大规模模型的效果有待验证
2. `<PST>` token 的随机初始化可能不是最优选择，更好的初始化策略值得探索
3. TP 操作的最优层数 k 需要针对不同模型进行调优
4. 主要在 STS 任务上验证，其他下游任务（如检索、聚类）的效果有待进一步验证
5. 早退层的选择也需要根据模型调整，缺乏自动化选择机制

## 相关工作与启发

- **PromptEOL** (Jiang et al., 2023)：首个使用 prompt 从 LLM 提取句子嵌入的工作，本文的 baseline
- **Echo Embeddings** (Springer et al., 2024)：通过重复输入实现后向依赖，思路类似但开销大
- **MetaEOL** (Lei et al., 2024)：使用 ChatGPT-4 设计元任务 prompt，从多角度考虑句子表示
- 对未来句子嵌入研究的启发：在不修改模型参数的条件下，可以通过巧妙的推理时干预来显著提升表示质量

## 评分

- 新颖性: ⭐⭐⭐⭐ — 层间前置的思路新颖，但本质上是一种推理时 trick
- 实验充分度: ⭐⭐⭐⭐ — 多个 STS 基准、多种 prompt 方法、多个模型，缺少更多下游任务
- 写作质量: ⭐⭐⭐⭐ — 方法描述清晰，图示直观
- 价值: ⭐⭐⭐⭐ — training-free 且即插即用，实用价值较高
