---
title: >-
  [论文解读] Reasoning About the Unsaid: Misinformation Detection with Omission-Aware Graph Inference
description: >-
  [AAAI 2026][社会计算][虚假信息检测] 提出OmiGraph，首个基于"遗漏感知"的虚假信息检测框架，通过构建遗漏感知图、利用LLM推理遗漏意图、以及遗漏导向的消息传递与聚合机制，从"未说出的内容"中提取欺骗模式，在双语数据集上平均提升+5.4% F1和+5.3% ACC。 虚假信息的欺骗手段主要分为两类：（1）…
tags:
  - "AAAI 2026"
  - "社会计算"
  - "虚假信息检测"
  - "遗漏感知"
  - "图神经网络"
  - "大语言模型"
  - "信息操纵"
---

# Reasoning About the Unsaid: Misinformation Detection with Omission-Aware Graph Inference

**会议**: AAAI 2026  
**arXiv**: [2512.01728](https://arxiv.org/abs/2512.01728)  
**代码**: [GitHub](https://github.com/ICTMCG/OmiGraph)  
**领域**: 社会计算  
**关键词**: 虚假信息检测, 遗漏感知, 图神经网络, 大语言模型, 信息操纵

## 一句话总结

提出OmiGraph，首个基于"遗漏感知"的虚假信息检测框架，通过构建遗漏感知图、利用LLM推理遗漏意图、以及遗漏导向的消息传递与聚合机制，从"未说出的内容"中提取欺骗模式，在双语数据集上平均提升+5.4% F1和+5.3% ACC。

## 研究背景与动机

虚假信息的欺骗手段主要分为两类：（1）**委托式欺骗（commission）**——显式地编造虚假内容；（2）**遗漏式欺骗（omission）**——隐式地省略关键信息，让读者在信息不完整的情况下得出错误结论。

现有的虚假信息检测方法几乎全部聚焦于前者，即从"已说出的内容"中提取欺骗信号：
- **基于风格/情感的方法**（如DualEmo）：分析语言风格和情感信号；
- **基于常识冲突的方法**（如MD-PCC）：利用外部知识检测内容与事实的矛盾；
- **基于证据验证的方法**（如RAV、RAFTS）：通过外部证据核实陈述的真实性。

然而，**遗漏式欺骗在实际中普遍且更具隐蔽性**。心理学研究表明，当信息被选择性呈现时，人们更容易被欺骗。例如，一条关于抗议活动的新闻故意省略了背景原因和因果关系，加剧了警民冲突的表面印象，但现有方法难以发现这种"故意不说"的欺骗策略。

检测遗漏式欺骗面临三个核心挑战：

**隐式信号恢复**：被省略的信息本身在目标文章中不存在，无法直接观察；

**动态遗漏关系**：陈述内容与被省略内容之间的关系千变万化，可能是正常的编辑选择，也可能是恶意的因果掩盖；

**遗漏模式建模**：需要有效整合被省略的内容及其关系，建立整体性的欺骗感知。

## 方法详解

### 整体框架

OmiGraph由三个核心组件组成（如Figure 2所示）：
1. **遗漏感知图构建**：利用上下文环境恢复被省略的信息，构建图结构；
2. **遗漏导向关系建模**：推理图内和图间节点的遗漏关系；
3. **遗漏引导的消息传递与聚合**：提取遗漏欺骗模式用于检测。

OmiGraph设计为**即插即用的增强模块**，可以与任何现有虚假信息检测器配合使用。

### 关键设计

1. **遗漏感知图构建（Omission-Aware Graph Construction）**

   核心洞察：同一事件的不同新闻报道自然提供了互补的视角，可以作为恢复被省略信息的资源。

   **上下文环境构建**：给定目标新闻$n_{\text{tgt}}$，从发布时间$T$天内的候选新闻池$\mathcal{P}$中，使用BERT计算语义相似度，选取Top-$K$个语义相近的新闻构成上下文环境$\mathcal{C}_{\text{ctx}}$：
    $\mathcal{C}_{\text{ctx}} = \{n_u \mid n_u \in \text{TopK}(\cos(\mathbf{h}_{\text{tgt}}, \mathbf{h}_u), n_u \in \mathcal{P})\}$

   **细粒度节点初始化**：将目标新闻和上下文新闻拆解为句子级的原子片段，作为图节点$\mathcal{V}$。这种细粒度拆分使得遗漏边界的精确识别成为可能，而非粗粒度的整篇新闻表示。图边分为源内边$\mathcal{E}_{\text{intra}}$（同一新闻内部的片段连接）和源间边$\mathcal{E}_{\text{inter}}$（目标新闻与上下文新闻之间的连接）。

2. **遗漏导向关系建模（Omission-Oriented Relation Modeling）**

   **源内关系（Intra-source）**：建模同一新闻内部片段间的语义依赖关系，揭示内部片段如何相互作用以维持叙事连贯或促进欺骗。通过可学习的边嵌入实现：
    $\mathbf{e}_{\text{intra}}^{ij} = \text{MLP}(\mathbf{h}_i \| \mathbf{h}_j \| \text{diff}(\mathbf{h}_i - \mathbf{h}_j))$

   **源间关系（Inter-source）**：关键创新点——利用**大语言模型推理动态遗漏意图**。不同于预定义关系类型，OmiGraph利用LLM的上下文理解能力来推理"为什么特定信息被省略"：
    $\mathbf{e}_{\text{inter}}^{ij} = \text{PLM}(\mathcal{M}(s_{\text{tgt}}^i, s_{\text{ctx}}^j))$

   LLM返回自由文本形式的遗漏意图描述（如"为了淡化行为背后的政治动机"），再通过预训练语言模型编码为边属性。这种设计使得框架能够**不依赖预定义的关系类型**而动态捕捉各种遗漏模式。

3. **遗漏引导的消息传递与聚合（Omission-Guided Message Passing）**

   **局部注意力消息传递**：利用边编码的遗漏关系引导信息传播。边属性通过类型特定的可学习嵌入增强后，参与注意力权重计算：
    $\alpha_{ij} = \text{softmax}((\mathbf{h}_i^{(l-1)} + \hat{\mathbf{e}}_t^{ij}) \cdot (\mathbf{h}_j^{(l-1)} + \hat{\mathbf{e}}_t^{ij}))$

   **全局聚合**：引入超级根节点$\mathbf{h}_{\text{root}}$作为全局信息的中央聚合器，避免仅靠多层局部消息传递导致的过平滑和过压缩问题：
    $\mathbf{h}_{\text{root}}^{(l)} = \mathbf{h}_{\text{root}}^{(l-1)} + \sum_i \text{softmax}(\mathbf{W}\mathbf{h}_i^{(l-1)} + b) \cdot \mathbf{h}_i^{(l-1)}$

   然后全局信息通过残差融合回传到各节点，确保片段级遗漏模式在整体叙事结构中被情境化理解。

### 损失函数 / 训练策略

最终将目标新闻的图节点特征进行均值池化得到$\mathbf{h}_{\text{omi}}$，与传统检测器的特征$\mathbf{h}_{\text{com}}$融合后预测：
$$\hat{y} = \text{fuse}(\mathbf{h}_{\text{omi}} \| \mathbf{h}_{\text{com}})$$

使用标准二分类交叉熵损失优化：
$$\mathcal{L}_{\text{cls}} = -y \log(\hat{y}) - (1-y)\log(1-\hat{y})$$

训练细节：AdamW优化器，batch size 64，学习率$2 \times 10^{-5}$，特征维度256，MLP隐层大小$[128, 128]$。英文使用bert-base-uncased，中文使用bert-base-chinese。LLM使用GPT-4o-mini推理遗漏意图。

## 实验关键数据

### 主实验

OmiGraph作为增强模块应用于多种基线检测器（"+ Ours"表示加入OmiGraph）：

| 基线方法 | 英文macF1 | +OmiGraph | 提升 | 中文macF1 | +OmiGraph | 提升 |
|----------|-----------|-----------|------|-----------|-----------|------|
| BERT | 0.7111 | **0.7530** | +4.19% | 0.7851 | **0.8407** | +5.56% |
| DualEmo | 0.7194 | **0.7557** | +3.63% | 0.7958 | **0.8417** | +4.59% |
| MSynFD | 0.7317 | **0.7608** | +2.91% | 0.8054 | **0.8496** | +4.42% |
| LLM | 0.5556 | **0.7259** | +17.03% | 0.6992 | **0.8336** | +13.44% |
| PCoT | 0.6508 | **0.7062** | +5.54% | 0.8020 | **0.8383** | +3.63% |
| NEP | 0.7274 | **0.7596** | +3.22% | 0.8288 | **0.8585** | +2.97% |
| RAV | 0.7189 | **0.7433** | +2.44% | 0.7930 | **0.8354** | +4.24% |
| RAFTS | 0.6016 | **0.6771** | +7.55% | 0.7427 | **0.7870** | +4.43% |

所有提升均在$p<0.005$水平下统计显著。

### 消融实验

| 配置 | 英文macF1 | 中文macF1 | 说明 |
|------|-----------|-----------|------|
| OmiGraph完整版 | **0.7530** | **0.8407** | 完整模型 |
| w/o Seg（无细粒度分割） | ~0.735 | ~0.825 | 粗粒度表示阻碍遗漏推理 |
| w/o Textual（无LLM遗漏意图） | ~0.730 | ~0.820 | 结构连接无法替代语义推理 |
| w/o Intra（无源内关系） | ~0.740 | ~0.830 | 内部依赖提供重要上下文线索 |
| w/o GlobalAgg（无全局聚合） | ~0.742 | ~0.832 | 全局叙事理解对系统性遗漏检测重要 |

### 关键发现

1. **LLM基线提升最大**（+17.03%英文 / +13.44%中文）：说明即使是强大的语言模型，在缺乏遗漏建模的情况下也难以有效检测虚假信息。
2. **对已有外部信息方法仍有提升**：即使NEP等已使用外部新闻，OmiGraph通过"信息完整性缺口"而非"事实矛盾"维度提供互补收益。
3. **遗漏类型分布差异**：虚假信息在比较遗漏和利益相关方遗漏上比例更高，而真实新闻在复杂性遗漏上更高——这揭示了不同的编辑动机。
4. **LLM模拟可行性**：在无外部新闻语料的场景下，使用LLM模拟上下文环境仍能获得有竞争力的性能，且token成本低于PCoT等方法。
5. **超参数鲁棒性**：上下文节点数$k$变化时性能稳定；GNN层数在英文$l=2$、中文$l=3$时最优。

## 亮点与洞察

1. **开创性的"遗漏"视角**：首次将虚假信息检测从"what is said"扩展到"what is unsaid"，这是一个被长期忽视但至关重要的欺骗维度。心理学上有坚实的理论支撑。
2. **LLM推理遗漏意图的创新用法**：不预定义关系类型，而是让LLM自由生成遗漏意图的文本描述，再编码为边属性，巧妙结合了LLM的推理能力和GNN的结构学习能力。
3. **即插即用的框架设计**：OmiGraph可以增强任何现有检测器，不改变原有模型结构，具有极强的实用性和适应性。
4. **八类遗漏类型的系统化分析**：通过大规模数据分析总结出的遗漏类型分类法为后续研究提供了有价值的理论基础。

## 局限与展望

1. **依赖外部新闻语料**：标准版需要大规模的同期新闻语料库（英文100万+、中文58万+篇），获取成本高。虽然提出了LLM模拟方案，但性能有所下降。
2. **LLM调用成本**：源间关系推理需要对每对片段调用LLM，在片段数增多时token消耗可能成为瓶颈。
3. **时效性约束**：上下文环境基于发布前$T$天的新闻池构建，对于突发事件可能无法找到足够的参考新闻。
4. **仅处理文本模态**：当前仅分析文本内容的遗漏，未涉及多模态新闻中的图像/视频遗漏。
5. **是否存在遗漏的判定标准**：当前由LLM判断遗漏是否存在，缺乏更精确的形式化定义。

## 相关工作与启发

- 本文填补了虚假信息检测领域"遗漏式欺骗"方向的空白，与已有的委托式检测方法形成互补。
- **图结构建模新闻关系**的方法可推广到谣言溯源、新闻可信度评估等相关任务。
- **LLM作为关系推理引擎**的设计思路值得在其他需要隐式推理的NLP任务中借鉴。
- 八类遗漏类型的分析为未来自动化的"新闻信息完整性评估"工具提供了基础。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次提出遗漏感知的虚假信息检测框架，视角独特且重要
- **技术深度**: ⭐⭐⭐⭐ — LLM+GNN的组合设计合理，消息传递机制完整
- **实验充分度**: ⭐⭐⭐⭐⭐ — 双语数据集、8种基线、消融全面、案例分析清晰
- **实用性**: ⭐⭐⭐⭐ — 即插即用设计实用，但依赖外部语料和LLM有成本
- **写作质量**: ⭐⭐⭐⭐⭐ — 问题定义清晰，motivation层层递进，案例引人入胜

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] T2Agent: A Tool-augmented Multimodal Misinformation Detection Agent with Monte Carlo Tree Search](t2agent_a_tool-augmented_multimodal_misinformation_detection_agent_with_monte_ca.md)
- [\[ICML 2026\] IDO: Incongruity-Aware Distribution Optimization for Multimodal Fake News Detection](../../ICML2026/social_computing/ido_incongruity-aware_distribution_optimization_for_multimodal_fake_news_detecti.md)
- [\[ACL 2026\] LiveFact: A Dynamic, Time-Aware Benchmark for LLM-Driven Fake News Detection](../../ACL2026/social_computing/livefact_a_dynamic_time-aware_benchmark_for_llm-driven_fake_news_detection.md)
- [\[AAAI 2026\] From Imitation to Discrimination: Toward A Generalized Curriculum Advantage Mechanism Enhancing Cross-Domain Reasoning Tasks](from_imitation_to_discrimination_toward_a_generalized_curriculum_advantage_mecha.md)
- [\[ACL 2025\] BiasGuard: A Reasoning-Enhanced Bias Detection Tool for Large Language Models](../../ACL2025/social_computing/biasguard_a_reasoning-enhanced_bias_detection_tool_for_large_language_models.md)

</div>

<!-- RELATED:END -->
