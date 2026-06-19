---
title: >-
  [论文解读] Inferring Functionality of Attention Heads from their Parameters
description: >-
  [ACL 2025][注意力头] 提出MAPS框架，通过将注意力头参数投影到词汇空间构建token映射矩阵$M$，无需任何推理或训练即可推断注意力头实现的功能，在6个LLM上验证了20种关系操作的映射准确性，并开发自动化pipeline发现了大量此前未被识别的注意力头功能。 问题背景 注意力头是LLM的核心构建模块…
tags:
  - "ACL 2025"
  - "注意力头"
  - "可解释性"
  - "参数分析"
  - "LLM内部机制"
  - "词汇空间投影"
---

# Inferring Functionality of Attention Heads from their Parameters

**会议**: ACL 2025  
**arXiv**: [2412.11965](https://arxiv.org/abs/2412.11965)  
**作者**: Amit Elhelo, Mor Geva (Tel Aviv University)  
**代码**: [github.com/amitelhelo/MAPS](https://github.com/amitelhelo/MAPS)  
**领域**: 其他  
**关键词**: 注意力头, 可解释性, 参数分析, LLM内部机制, 词汇空间投影  

## 一句话总结

提出MAPS框架，通过将注意力头参数投影到词汇空间构建token映射矩阵$M$，无需任何推理或训练即可推断注意力头实现的功能，在6个LLM上验证了20种关系操作的映射准确性，并开发自动化pipeline发现了大量此前未被识别的注意力头功能。

## 研究背景与动机

### 问题背景
注意力头是LLM的核心构建模块，理解其功能对模型可解释性至关重要。现有研究主要通过分析注意力头在推理时的行为（attention pattern、输出投影、因果干预）来理解其功能，但这种方法存在固有缺陷。

### 已有工作的不足
- **覆盖不完整**：依赖特定输入的分析可能遗漏注意力头在其他输入上的功能，因为同一个head在不同输入下可能表现不同
- **计算代价高**：全面分析需要在大量输入上执行模型推理，计算成本高且训练数据可能不可用
- **解释困难**：分析激活模式通常不直观，可能产生误导性结论
- **局限于特定电路**：词汇空间投影方法此前仅用于研究特定电路中的少数head或单一操作，未被系统化应用

### 核心动机
能否直接从注意力头的参数推断其功能，完全绕过模型推理？本文将词汇空间解释方法扩展为通用框架MAPS，系统化地回答两类问题：(a) 给定操作，映射模型中哪些head实现了它；(b) 给定head，推断其显著功能。

## 方法详解

### 核心思想：注意力头的词汇空间解释
基于Elhage等人的formulation，将注意力头的$W_{VO}$矩阵通过embedding和unembedding矩阵投影到词汇空间：

$$M = E \cdot W_{VO} \cdot U \in \mathbb{R}^{|\mathcal{V}| \times |\mathcal{V}|}$$

矩阵$M$中每个元素$M[s,t]$表示该head将source token $s$ 映射到target token $t$ 的强度分数。

### MAPS框架的两种分析方式

**方式A：预定义关系映射（Predefined Relations）**

给定一组token对数据集$\mathcal{D}_R$表达关系$R$（如国家到首都），计算head实现该关系的分数：

$$\phi_R(M) = \frac{1}{|\mathcal{D}_R|} \sum_{(s,t) \in \mathcal{D}_R} \mathbb{1}[t \in \text{topk}(\mathbf{m}_s)]$$

即检查target token是否出现在source token对应行的top-k映射中。阈值$\tau=15\%$用于分类head是否实现某关系。同时支持抑制操作（suppression），通过考虑$-\mathbf{m}_s$的top-k实现。

**方式B：显著操作推断（Salient Operations）**

1. 用显著性分数$\sigma_t(W_{VO}) = \|e_t W_{VO}\| / \|e_t\|$找出变换最显著的top-k token
2. 对每个显著token收集其top-n映射目标
3. 用GPT-4o自动描述这些映射中的模式

该方法比直接取$M$中最高分的映射更可靠，因为后者受token embedding范数影响，可能偏向少数token。

### 关系类型设计
构建了4类共20种关系的数据集：
- **算法类**：copying、name copying、word到首字母/末字母、年份到下一年
- **知识类**：国家到首都、国家到语言、物体到上位类、产品到公司、作品到地点
- **语言类**：反义词、形容词到比较级/最高级、名词到代词、动词到过去式、同音词、近义词、合成词
- **翻译类**：英到法、英到西

## 实验关键数据

### 实验1：与推理输出的相关性验证

在Llama-3.1 8B上，MAPS的静态估计分数$\phi_R(M)$与推理时动态分数$\phi_R^*(h)$的Pearson相关系数：

| 类别 | 关系 | 无上下文相关性 | 有上下文相关性 |
|------|------|-------------|-------------|
| 算法 | Copying | 0.76 | 0.73 |
| 算法 | Name copying | 0.95 | 0.95 |
| 算法 | Word到首字母 | 0.90 | 0.78 |
| 知识 | 国家到首都 | 0.85 | 0.85 |
| 知识 | 国家到语言 | 0.76 | 0.62 |
| 语言 | 反义词 | 0.90 | 0.86 |
| 语言 | 形容词到比较级 | 0.85 | 0.86 |
| 语言 | 动词到过去式 | 0.91 | 0.86 |
| 翻译 | 英到法 | 0.71 | 0.68 |
| 翻译 | 英到西 | 0.82 | 0.81 |

绝大多数关系达到0.71-0.95的强至极强相关性，表明MAPS能准确估计head的推理时行为。

### 实验2：因果效应验证（Ablation）

在Pythia 12B上，移除MAPS识别的关系head vs 移除随机head对模型准确率的影响：

| 关系 | 基线准确率 | 移除关系head | 移除随机head | 控制任务-移除关系head |
|------|-----------|-------------|-------------|-------------------|
| 形容词到比较级 | 0.91 | 0.20 | 0.82 | 0.63 |
| Copying | 1.00 | 0.68 | 1.00 | 0.88 |
| 国家到首都 | 0.97 | 0.00 | 0.95 | 0.90 |
| 国家到语言 | 1.00 | 0.08 | 0.96 | 0.89 |
| Name copying | 1.00 | 0.24 | 1.00 | 0.92 |
| Word到首字母 | 0.91 | 0.34 | 0.87 | 0.74 |
| 年份到下一年 | 0.92 | 0.00 | 0.87 | 0.79 |

所有关系中移除MAPS识别的head导致准确率下降大于32%，而移除随机head仅下降小于13%，证明MAPS识别的head与模型行为存在因果关系。

## 亮点

- **零推理开销**：完全从参数推断注意力头功能，无需模型训练或推理，计算效率极高
- **系统化框架**：首次将词汇空间投影方法扩展为通用的、可同时支持"操作定位"和"功能发现"的框架，在6个LLM和20种关系上进行了大规模验证
- **发现新head**：在GPT-2 small和medium中分别发现了25和46个此前未被识别但实现类似操作的head，扩展了现有电路分析的覆盖范围
- **架构洞察**：揭示了多个有价值的架构偏差——小模型倾向于在单个head上编码更多关系；Llama-3.1的分组注意力中同组head常实现相同或相似关系；关系head普遍集中在中间和上层
- **自动化pipeline**：结合GPT-4o实现了注意力头功能的自动描述，在中上层达到60%-96%的覆盖率，人工评估80%正确

## 局限与展望

- **仅分析$W_{VO}$**：忽略了$W_{QK}$矩阵（负责注意力计算/上下文化），未能完整刻画head的选择性行为
- **词汇空间限制**：只能捕获可用token对表达的操作，无法处理成语、位置特征等更抽象的计算
- **早期层覆盖不足**：早期层head在词汇空间中的可解释性较低（20%-60%），可能因为它们计算通用特征而非词汇级操作
- **忽略bias项**：$W_V$和$W_O$的bias被省略，可能影响估计精度
- **多token实体泛化有限**：虽然实验表明单token估计可泛化到多token输入，但仍存在少量相关性下降的情况
- **自动描述质量**：GPT-4o的功能描述虽然80%正确，但仍有误判空间

## 与相关工作的对比

- **Wang et al. (2023), McDougall et al. (2024)**：在特定电路（如IOI）中用词汇投影验证已有head功能，MAPS将其扩展为通用框架并发现大量新head
- **Gould et al. (2024)**：仅研究单一关系（copying）的跨模型分布，MAPS支持20种关系的系统化映射
- **Voita et al. (2019), Clark et al. (2019)**：基于attention pattern分析head功能，MAPS完全基于参数无需推理
- **Millidge & Black (2022)**：用LLM解释参数的奇异向量，但不考虑输入-输出映射关系，无法估计head功能
- **Hernandez et al. (2024)**：证明head的关系操作可用线性函数近似，MAPS进一步展示这些关系编码在参数的映射中
- **Merullo et al. (2024a)**：在GPT-2 medium中发现多功能head，MAPS通过系统化分析扩展并量化了这一发现

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将参数级词汇投影方法系统化为通用框架，思路自然但应用规模和深度有显著突破
- 实验充分度: ⭐⭐⭐⭐⭐ — 6个模型x20种关系，相关性/因果/泛化三重验证，含人工评估
- 写作质量: ⭐⭐⭐⭐⭐ — 结构清晰，图表信息量大，动机和方法阐述准确到位
- 价值: ⭐⭐⭐⭐ — 为LLM可解释性提供了高效实用的工具，架构洞察有启发性，但对更复杂计算的解释力有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] The Hidden Attention of Mamba Models](the_hidden_attention_of_mamba_models.md)
- [\[ACL 2025\] Segment-Based Attention Masking for GPTs](segment-based_attention_masking_for_gpts.md)
- [\[ACL 2025\] Attention Entropy is a Key Factor for Parallel Context Encoding](attention_entropy_parallel_encoding.md)
- [\[ACL 2025\] Unique Hard Attention: A Tale of Two Sides](unique_hard_attention_a_tale_of_two_sides.md)
- [\[NeurIPS 2025\] Normalization in Attention Dynamics](../../NeurIPS2025/others/normalization_in_attention_dynamics.md)

</div>

<!-- RELATED:END -->
