# A Partition Cover Approach for Tokenization

**会议**: NeurIPS 2025  
**arXiv**: [2501.06246](https://arxiv.org/abs/2501.06246)  
**代码**: [https://github.com/PreferredAI/pcatt/](https://github.com/PreferredAI/pcatt/)  
**领域**: LLM / NLP / Tokenization  
**关键词**: tokenization, BPE, partition cover, NP-hard, greedy algorithm  

## 一句话总结
将分词（tokenization）问题重新建模为**分区覆盖（partition cover）**优化问题，证明其为NP-hard，并提出多项式时间的贪心算法GreedTok，在压缩率和1B参数LLM预训练下游任务上均优于BPE。

## 背景与动机
分词是LLM的基础组件，负责将文本编码为固定词表大小的token序列。当前主流方法是BPE（Byte-Pair Encoding），它把分词视为压缩问题，通过反复执行成对合并来构建词表。另一个方法是Unigram，它从大候选集自上而下剪枝。然而，BPE的成对合并机制是一种**人为的算法约束**——合并过程中产生的许多中间token可能永远不会出现在最终编码中，浪费了词表容量。Unigram则因其似然目标会过度偏好选择完整单词，牺牲了有信息量的子词token。目前对分词问题的理论理解也很有限，缺乏清晰的优化目标和复杂度分析。

## 核心问题
能否跳出BPE的成对合并和Unigram的剪枝范式，从更一般的优化视角来理解分词问题？具体来说：（1）分词问题的计算复杂度是什么？（2）能否设计不受合并序列约束、同时具有良好实际性能的分词算法？

## 方法详解

### 整体框架
作者提出了分词的**分区覆盖（Partition Cover）**形式化：给定语料库$\mathcal{C}=(\mathbf{W}, \text{count})$、候选token集合$\mathbf{T}$和预算$k$，目标是选择子集$\mathbf{S} \subseteq \mathbf{T}$（$|\mathbf{S}| \leq k$），使得用$\mathbf{S}$加上基础单字符token集$\mathbf{B}$对语料所有单词进行最优分割后，总token数最小。

核心思路是将"两个相邻字符是否被同一个token覆盖"作为基本决策变量，而不是像BPE那样约束token必须通过逐步合并产生。这样的形式化**包含了所有基于合并的方案**（它们都是合法解），但合法解空间更大——不限于自底向上合并能产生的词表。

### 关键设计

1. **NP-hard证明**：通过从顶点覆盖（vertex cover）问题的简单归约证明分词是NP-hard的。对于图$\mathcal{G}=(V,E)$的每条边$\{V_i,V_j\}$，构造单词$(@,V_i,@,V_j,@)$，候选token为$\{(@,V_i,@)\}$的形式。这样选token覆盖单词等价于选顶点覆盖边。每个单词长度为5，每个token长度为3，所以被覆盖时分区数为3，否则为5。证明清晰直观。

2. **混合整数规划（MIP）等价形式**：引入二元变量$m_{i,i+1}^W$表示单词$W$中第$i$和第$i+1$个字符是否被同一token覆盖，将最小化partition转化为等价的最大化cover问题。关键观察是$|W| = \text{partition} + \text{cover}$，所以最小化partition等价于最大化cover。MIP包含了token的选择约束、覆盖一致性约束、不重叠约束等。

3. **与加权最大覆盖（WMC）的关联**：MIP自然松弛为加权最大覆盖问题——去掉不重叠约束和连续性约束后。WMC有经典的贪心$(1-1/e)$近似算法GreedWMC。由于WMC的最优值不低于Tok的最优值，可以用GreedWMC的解作为GreedTok近似比的经验上界。

4. **GreedTok算法**：两步走——（1）选token集合$\mathbf{S}$：从$\mathbf{S}=\emptyset$开始，每轮贪心选择使MIP目标函数增量最大的token加入，直到$|\mathbf{S}|=k$；（2）编码：对给定文本，找出$\mathbf{S}$中所有匹配位置，按token被加入$\mathbf{S}$的顺序排列，逐个尝试覆盖，后加入的更长token可以覆盖之前的短token决策。算法复杂度选词为$O(|\mathbf{T}|\cdot k \cdot \sum|W|)$，实际通过lazy evaluation大幅加速；编码复杂度为$O(|W|^2 \log|W|)$。

### 训练策略
- GreedTok的目标函数**既不是子模的，也不是超模的**（通过反例证明），所以标准的子模最大化近似分析不适用
- 但经验上，随着$k$增大，GreedTok的目标值与GreedWMC之比$d_{\text{inst}}$趋向1，实际中达到至少$0.9(1-1/e)$近似比

## 实验关键数据

### 压缩实验
| 数据集 | 指标 (avg tokens/word) | GreedTok | BPE | 提升 |
|--------|------|------|----------|------|
| UN (k=5000) | tokens/word | 1.163 | 1.194 | 2.54% |
| arXiv (k=5000) | tokens/word | 1.226 | 1.263 | 2.89% |
| wiki (k=10000) | tokens/word | 1.283 | 1.301 | 1.37% |
| PubMed (k=10000) | tokens/word | 1.206 | 1.225 | 1.52% |

GreedTok平均优于BPE 2.88%，优于Unigram 3.43%。

### 预训练实验（1B参数LLM）
| 模型 | Avg Accuracy | Wikitext bits/byte |
|------|-------------|-------------------|
| BPEM (BPE) | 62.0 | 0.7066 |
| GTEP (GreedTok, 同比例数据) | 62.5 | 0.7028 |
| GTET (GreedTok, 同token数) | **63.2** | **0.6989** |

GreedTok用同样token数的文本训练（DCLM约20%数据），在11个benchmark的平均accuracy和bits/byte上均优于BPE。GreedTok使整个DCLM数据集的token总数减少约18%。

### 消融实验要点
- token集分析表明GreedTok的token集同时继承了BPE的高压缩率和Unigram的高token质量（更多完整单词、Unigram似然更接近）
- 近似比$d_{\text{inst}}$在$k \geq 5000$时接近0.95-1.0，说明GreedTok在实际NLP场景中几乎找到了最优解
- 在RefinedWeb上40次独立采样实验也验证了$d_{\text{inst}} \to 1$的趋势

## 亮点
- **视角新颖**：把分词从"压缩/合并"视角转换为"覆盖"视角，建立了与顶点覆盖、加权最大覆盖等经典组合优化问题的清晰联系
- **NP-hard证明极其简洁**：通过顶点覆盖的直接归约，比并发工作WBP(2024)的MAX-2SAT归约简单得多
- **MIP形式化的灵活性**：可以自然地向目标函数中注入新的约束（如公平性、下游任务目标），这是BPE/Unigram做不到的
- **实际可用**：C++实现+Python绑定+HuggingFace集成，编码速度700-800K words/s/thread，wiki 10K词表11分钟构建

## 局限性 / 可改进方向
- GreedTok选词的理论复杂度$O(|\mathbf{T}|\cdot k \cdot \sum|W|)$高于BPE的$O(k \cdot \sum|W|)$，虽然lazy evaluation实际大幅加速，但对超大规模语料仍有计算瓶颈（14.3M unique words时34分钟/160GB RAM）
- 缺少GreedTok的正式近似比证明（仅有经验$0.9(1-1/e)$保证）
- 预训练实验仅限于1B参数模型和BPE对比，未验证更大模型或与Unigram的预训练对比
- 未测试低资源语言和多语言场景
- 编码速度虽可用但仍慢于BPE/tiktoken的优化实现

## 与相关工作的对比
- **vs BPE**：BPE被限制在成对合并路径中，浪费词表容量在中间token上；GreedTok无此约束，压缩率提升~3%，预训练效果也更好
- **vs Unigram**：Unigram自上而下剪枝会偏好整词token；GreedTok自下而上构建，兼具Unigram的token质量和BPE的压缩能力
- **vs WBP(2024)同期工作**：WBP也证明了分词NP-hard，但通过MAX-2SAT归约，证明更复杂；本文的顶点覆盖归约更简洁直观
- **vs KV(2024)**：KV证明BPE的底层合并问题APX-complete，但限于合并框架；本文形式化更一般

## 启发与关联
这篇论文展示了从组合优化角度重新审视NLP基础组件的价值。MIP形式化为在分词中注入额外目标（如下游任务信号、公平性约束）提供了自然框架。这个方法论思路——把深度学习pipeline中"约定俗成"的组件重新形式化为优化问题——可能对其他NLP/多模态基础组件也有启发，比如视觉tokenizer的设计。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从覆盖问题角度重新定义分词，视角独特且有深度
- 实验充分度: ⭐⭐⭐⭐ 压缩实验充分，预训练实验规模合理但仅限1B模型
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，证明简洁优雅，结构组织良好
- 价值: ⭐⭐⭐⭐ 为分词领域提供了新理论基础和实用算法，MIP框架有扩展潜力
