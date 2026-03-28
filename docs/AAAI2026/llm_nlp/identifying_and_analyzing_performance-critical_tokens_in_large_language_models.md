# Identifying and Analyzing Performance-Critical Tokens in Large Language Models

**会议**: AAAI 2026  
**arXiv**: [2401.11323](https://arxiv.org/abs/2401.11323)  
**代码**: https://github.com/ybai-nlp/PCT_ICL  
**领域**: NLP理解  
**关键词**: In-Context Learning, 性能关键token, 注意力消融, 模板token, 信息聚合

## 一句话总结
通过representation-level和token-level两种消融实验，发现LLM在ICL中直接依赖的"性能关键token"是模板和停用词token（如"Answer:"），而非人类会关注的内容token（如实际文本），并揭示了LLM通过将内容信息聚合到这些关键token的表示中来间接利用内容。

## 研究背景与动机
1. **领域现状**：ICL已成为LLM的主流few-shot学习方法，但对LLM如何从demonstration中学习和泛化仍理解不足。已有研究表明ICL对prompt的微小变化（如demo顺序）非常敏感。
2. **现有痛点**：之前的研究要么只关注最后一个token（function vector），要么只关注label word，缺乏对prompt中**所有token角色**的系统性研究。
3. **核心矛盾**：人类在学习analogy时关注"内容words"（如名词/形容词），但LLM是否也如此？
4. **本文要解决什么？** 系统性地识别ICL prompt中哪些token的表示直接影响性能（性能关键token），并分析其特征
5. **切入角度**：将ICL prompt的token分为三类（content/stopword/template），通过消融各类token的表示来测量对性能的影响
6. **核心idea一句话**：LLM并不直接依赖内容token的表示，而是依赖模板和停用词token——后者聚合了前者的信息

## 方法详解

### 整体框架
将ICL prompt token分三类，设计两种消融实验，在多个LLM和数据集上系统评估：
- **Token分类**：Template（如"Article:", "Answer:"）、Stopword（标点和功能词）、Content（实际文本内容）
- **Representation-level消融**：从test example的attention中mask掉特定类型token的表示
- **Token-level消融**：直接从prompt中删除特定类型的token

### 关键设计

1. **Representation-level消融**:
   - 做什么：测试哪类token的表示被test example直接依赖
   - 核心思路：修改attention mask，使test example只能attend到特定类型的token表示，然后测量分类准确率变化。如果只保留template+stopword的表示就能维持性能，说明这些才是性能关键token
   - 关键发现：**masking content tokens几乎不影响性能**，但masking template/stopword tokens导致性能大幅下降。这说明LLM直接从template/stopword的表示中获取任务信息

2. **Token-level消融**:
   - 做什么：测试哪类token的信息对整个prompt的信息流至关重要
   - 核心思路：直接从prompt中删除特定类型token。如果删除content token导致性能大幅下降（即使representation-level消融显示它不是直接依赖），说明content token的信息被间接传递了
   - 关键发现：**删除content token确实大幅降低性能**——这与representation-level消融的结果看似矛盾。解释：content token的信息被LLM聚合到了template/stopword token的表示中

3. **性能关键token的三个特征**:
   - **词汇含义**：与任务相关的词（如分类任务中的"Answer:"）更可能是性能关键token
   - **重复性**：在prompt中反复出现的token更可能是性能关键的（如每个demo都有的template）
   - **结构线索**：标记demo边界的token（如换行符、分隔符）提供关键的结构信息

### 损失函数 / 训练策略
纯分析工作，无训练。使用Llama 3B-33B、Llama 2、Mistral 7B、Gemma 3 4B。6个分类数据集，每个15 seeds × 500 test examples。

## 实验关键数据

### 主实验——表示级消融（Representation-Level Ablation）

从标准ICL（Standard ICL）中mask掉某类token的表示（仅影响test example的attention），测量准确率变化：

| 设置 | AGNews | SST2 | DBPedia | 平均△ |
|------|--------|------|---------|-------|
| Standard ICL（7B） | 85.0 | 93.2 | 66.7 | 基准70.7 |
| − content | 82.4 | 85.5 | 64.2 | **-3.8** |
| − stopword | 84.8 | 88.0 | 65.7 | -2.5 |
| − template | 0.9 | 61.0 | 12.9 | **-32.8** |

从零样本（Zero-shot）中加入某类token的表示：

| 设置 | 33B模型 平均△ |
|------|-------------|
| Zero-shot基线 | 54.6 |
| + content | **-6.7**（反而下降） |
| + stopword | +17.7 |
| + template | **+24.6** |

### Token级消融（Token-Level Ablation）
| 配置 | 效果 | 说明 |
|------|------|------|
| 只mask content表示 | 性能几乎不变（-3.8%） | content不是test example的直接依赖 |
| 只mask template表示 | 性能暴跌（-32.8%） | template是test example的直接依赖 |
| 从prompt中删除content token | 性能大幅下降 | content信息是必需的——但间接提供 |
| 切断content→template信息流 | 性能下降 | 验证了信息聚合机制 |

### 关键发现
- **反直觉核心发现**：LLM不像人类那样直接"关注"内容token，而是将内容信息聚合到模板token的表示中，再从模板表示中做决策
- **Template token的平均贡献是content的6.5倍**：+template带来平均+24.6%提升，+content仅-6.7%（甚至负面影响），量化差距极大
- **Content token的"间接贡献"假说**：表示级消融显示content不重要，token级消融显示content必需——这个"矛盾"证明content通过聚合到template表示来间接贡献
- 三个特征（词汇含义/重复性/结构线索）在所有模型大小上都一致，说明这是LLM的通用机制
- 结果对不同instruction prompt和不同模型（Llama/Llama 2/Mistral/Gemma 3）都鲁棒

## 亮点与洞察
- **"LLM和人类的attention完全不同"**这个发现非常有启发性：人类关注内容词，LLM关注模板/结构词。这挑战了"LLM像人一样理解text"的假设
- **信息聚合假说**很优雅：content → template的信息流解释了两种消融结果的"矛盾"——content重要但不直接参与，它通过聚合到template表示间接影响
- **对prompt engineering的实际意义**：模板的设计（而非内容的措辞）可能对ICL性能影响更大

## 局限性 / 可改进方向
- 主要在分类任务上验证，生成任务的结果在附录中但不够深入
- 消融方法假设可以clean地分离token类型，但实际中有些token同时兼具多种角色
- 未研究instruction-tuned模型是否有不同的依赖模式

## 相关工作与启发
- **vs Function Vector研究**: 它们只关注最后一个token的表示，本文扩展到prompt中所有token
- **vs Label Word研究**: 之前认为label word是信息锚点，本文发现template/stopword也是，甚至更关键
- **对ICL理论的启示**：ICL的working mechanism不能简单归结为"从demo内容中学"，结构和格式本身承载了关键的任务specification信息

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "模板比内容更关键"的发现非常反直觉且有深度
- 实验充分度: ⭐⭐⭐⭐⭐ 8种LLM、6个数据集、多种消融方式、15 seeds
- 写作质量: ⭐⭐⭐⭐ 逻辑链清晰（分类→消融→发现→解释→特征分析）
- 价值: ⭐⭐⭐⭐⭐ 对理解ICL机制和优化prompt engineering都有重要启示
