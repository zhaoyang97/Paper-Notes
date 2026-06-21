---
title: >-
  [论文解读] Critical Confabulation: Can LLMs Hallucinate for Social Good?
description: >-
  [ICLR 2026][幻觉检测][批判性虚构] 本文把"幻觉"重新框定为一种可用资源：提出 **critical confabulation（批判性虚构）**，让 LLM 在证据约束下"填补"历史档案中被结构性抹除的空白，并用一个基于未出版黑人历史语料的"叙事完形填空"任务系统评估了 19 个模型，证明受控、良定义的幻觉可以服务于知识生产而不坍缩成虚假。
tags:
  - "ICLR 2026"
  - "幻觉检测"
  - "批判性虚构"
  - "受控幻觉"
  - "叙事完形填空"
  - "数据污染审计"
  - "数字人文"
---

# Critical Confabulation: Can LLMs Hallucinate for Social Good?

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=wGFD7ITicm](https://openreview.net/forum?id=wGFD7ITicm)  
**代码**: 待确认（camera-ready 随论文发布，BWTC 数据需 ARTFL 授权）  
**领域**: 幻觉 / 计算人文 / 叙事理解  
**关键词**: 批判性虚构, 受控幻觉, 叙事完形填空, 数据污染审计, 数字人文  

## 一句话总结
本文把"幻觉"重新框定为一种可用资源：提出 **critical confabulation（批判性虚构）**，让 LLM 在证据约束下"填补"历史档案中被结构性抹除的空白，并用一个基于未出版黑人历史语料的"叙事完形填空"任务系统评估了 19 个模型，证明受控、良定义的幻觉可以服务于知识生产而不坍缩成虚假。

## 研究背景与动机
**领域现状**：LLM 幻觉通常被当成纯粹的失败模式来消除，但近期工作发现其中一类被称为 **confabulation（虚构填充）** 的行为——用自洽的故事去"填补"缺失信息、且与现实保持高度逼真——其实具有叙事价值，已在计算创意、叙事暴露疗法、文化遗产数字叙事等场景显示出社会效用。

**现有痛点**：人文领域的 **critical fabulation（批判性虚构，Hartman 2008）** 是一种用思辨叙事修复历史档案不公的成熟方法论，专门为那些因系统性压迫从未获得记录特权的"隐形人物"（hidden figures）发声。但它高度依赖学者对密集史料的逐字细读，劳动密集，无法规模化覆盖浩瀚档案。

**核心矛盾**：严格事实性 vs 叙事补全。把"档案是否留存"当作真理的代理，本质上是对"什么得以幸存"这一有偏标准的过拟合，会进一步沉默档案中的隐形人物；但放任 LLM 无约束幻觉又会把思辨坍缩成虚假，丧失历史保真度。

**本文目标**：在严格的证据边界内，把 LLM 已有的虚构行为操作化为可规模化的批判性虚构工作流，既能识别档案中的潜在空白，又能给出多个证据受限的可能性（而非断言单一真理），辅助人文学者扩充历史知识。

**核心 idea**：**【受控幻觉作为资源】** 将批判性虚构形式化为**开放式叙事完形填空**——给定某隐形人物的事件时间线，遮蔽其中一个事件，要求模型在已知上下文约束下重建被遮蔽事件，用叙事嵌入相似度判定是否"足够接近"，从而把"幻觉是缺陷"翻转为"幻觉是可优化的能力"。

## 方法详解

### 整体框架
整个流程分两层目标：**known unknowns（空白重建）** 与更难的 **unknown unknowns（空白检测）**，本文聚焦前者。系统先用未出版的黑人历史档案 BWTC 作为模型"没见过"的真值，经过双重数据污染审计筛出干净语料，再抽取隐形人物的事件时间线作 ground truth，最后通过遮蔽-重建的完形填空任务、配合不同"诱导幻觉"提示词来评估模型。

```mermaid
flowchart LR
    A[BWTC 档案语料 B] --> B1[双重数据污染审计]
    B1 -->|String Search + 行为探针| C[剔除 SEEN 文档<br/>仅保留 Bunseen]
    C --> D[隐形人物挖掘<br/>NER + Aho-Corasick 长尾过滤]
    D -->|156 个 hidden figures| E[GPT-o3 抽取<br/>事件时间线 T_n + 事件类型]
    E --> F[遮蔽一个事件 → C n,m]
    F --> G[19 个 LLM 在受控提示下重建 ê_m]
    G --> H[story-emb 余弦相似度 ≥ ε*<br/>判定 correct]
```

### 关键设计

**1. 任务形式化：把批判性虚构变成可度量的叙事完形填空。** 对每个隐形人物 $n$，从其相关档案构造按时间排序的事件时间线 $T(n)=\langle(t_1,e_1),\dots,(t_{m(n)},e_{m(n)})\rangle$，每个元素是一个时间戳 $t_i$ 加一句话事件 $e_i$。模拟历史空白的方式是把第 $m$ 个事件替换成字面量 `[MASK]`，得到 $C(n,m)=\langle(t_1,e_1),\dots,(t_m,\text{[MASK]}),\dots\rangle$。模型 $f_\theta$ 必须在其余时间线片段和固定指令下重建 $e_m$，当生成事件与真值的嵌入相似度满足 $\text{sim}_{\text{emb}}(\hat e_m,e_m)\ge\epsilon$ 时记为正确。这一设计把模糊的"思辨叙事"转化成可复现、可比较、可优化的指标，同时刻意只遮蔽一个事件以保留充足的证据约束，让幻觉"有边界"。

**2. 两阶段数据污染审计：保证"未见历史"假设不被记忆污染。** 评估的关键前提是模型没见过这些档案，否则证据约束的虚构就退化成记忆背诵。作者只用训练数据公开的 OLMO-2 全开放模型做主审计：先用 Boyer–Moore 子串匹配把 BWTC 每句话与 OLMO-2 完整训练集逐句对比，文档匹配数 $\text{matches}(d)=\sum_{x\in O}\sum_{s\in S(d)}\text{BM}(x,s)$ 达到 $\ge 100$ 即标 SEEN（共 21%）；再用一个行为探针做交叉验证——取前 20 句作上下文让 OLMO-2 续写，若标签可信则 SEEN 文档的续写应更贴近真值，即 $\text{mean}_{d\in B_{\text{seen}}}[\text{sim}_i(d)]>\text{mean}_{d\in B_{\text{unseen}}}[\text{sim}_i(d)]$。结果证实 SEEN 的平均相似度 0.3009 高于 UNSEEN 的 0.2782，且优势随续写位置 $p_1\to p_5$ 单调衰减（符合记忆优势在观测上下文后最强的预期），在多种统计检验下方向稳健。最终保守地剔除全部 SEEN 文档、仅在 $B_{\text{unseen}}$ 上分析。

**3. 隐形人物挖掘 + 证据受限的真值抽取。** 即便剔除 SEEN，模型参数化知识仍可能对特定人名构成先验，于是再做一轮 Aho–Corasick 多模式匹配：从 $B_{\text{unseen}}$ 用 NLTK 抽取最多前 1 万个 PERSON 名，只保留频率 $<51$、且至少出现在 3 个文档中的长尾人名，构建自动机扫描 OLMO-2 训练集计数 $c(n)$，$c(n)\ge100$ 记 SEEN-IN-O，配合人工过滤掉仅顺带提及/共指的名字，最终得到 156 个干净的隐形人物。每个人物的所有相关文档被一并送入长上下文抽取器 GPT-o3，在严格"源约束"指令下产出按时间排序、带显式引用的事件时间线，每个事件是一句主动语态、$\le 30$ 词的句子，并标注 `{AGENTIVE, RELATIONAL, OBSERVATIONAL, COGNITIVE, ROLE}` 五类之一的事件类型。

**4. 叙事专用评估 + 受控诱导提示。** 评估不用通用语义嵌入（会被主题相似度混淆），而用强调故事线结构的 narrative embedding 模型 **story-emb** 算余弦相似度，并在标注验证集上扫描阈值、以 macro-F1 最大化选出全局操作阈值 $\epsilon^\star=73.13$（macro-F1=0.805），证明该距离是人类叙事逼真度判断的合理代理。提示侧则在一个统一基线提示之上，叠加 6 个来自前人工作、有意或无意"诱导模型增加幻觉/创造性输出"的系统/指令模板（如 Null-Shot、HaluEval、LLM-Discussion、Eccentric Automatic Prompts 等），并可选地给出被遮蔽事件的类型作为轻量结构监督，以系统考察"什么提示能把幻觉用好"。

## 实验关键数据

### 主实验表格（叙事完形填空准确率，节选）

| 模型 | 基线(无类型) | 最佳(无类型) | 基线(有类型) | 峰值(有类型) |
|------|------|------|------|------|
| GPT-5-chat | 51.0 | 57.4 | 55.5 | **59.7** |
| Qwen3-4B-0725 | 47.1 | 50.9 | 50.2 | 53.8 |
| OLMo-2-7B | 33.8 | 49.1 | 38.5 | **58.8** |
| OLMo-2-32B | 44.9 | 46.0 | 48.4 | 50.8 |
| Qwen3-4B | 44.6 | 55.0 | 46.5 | 56.8 |
| GPT-4o | 42.6 | 45.5 | 45.4 | 49.1 |
| gemma-2-27b | 40.1 | 40.1 | 42.1 | 46.1 |

- GPT-5-chat 是总体领头羊，也是唯一在多数提示上超过 50% 的模型，峰值 59.7%；小模型 OLMo-2-7B、Qwen3-4B "以小搏大"，反超许多更大基线。
- 提供 **EVENT_TYPE 提示几乎对所有 (模型,提示) 对带来 +2~10 点稳定提升**。

### 消融实验表格（采样温度随机性消融，相对确定性基线的平均变化）

| 温度设定 | 平均准确率变化 |
|------|------|
| 确定性(T=0) | 基准 |
| 低 (T=0.2) | −0.3 |
| 中 (T=0.7) | −0.8 |
| 高 (T=1.2) | −2.3 |

- 性能对采样随机性整体稳健，模型/提示排名几乎完全保持；OLMO-2 家族中越大的变体在高温下掉点越多。

### 关键发现
- **任务可行但很难**：多数模型停在 50% 以下，强提示下峰值近 60%。
- **审计无显著记忆优势**：OLMO-2 与可比的未审计同行无显著差异（平均 p=0.354），即没观察到记忆带来的优势。
- **事件类型梯度**：模型最擅长 "role"（44.8%，偏传记信息）→ "relational" → "agentive" → "observational"，最差 "cognitive"（24.9%，内在状态/观点缺乏可观测锚点）。
- **结构敏感性**：事件描述越长准确率略升（$\rho=0.09$），但时间线越长准确率越低（$\rho=-0.173$）；时间线开头事件最易重建（0.45）、结尾最难（0.337）。
- **错误高度聚集**：仅 2.2% 的事件被全部 19 个模型答对，59.1% 被至少 10 个模型答错，错误集的 Jaccard 重叠达 0.6~0.9，说明模型不是各自随机失败，而是集中栽在同一批"难"事件/人物上。

## 亮点与洞察
- **范式翻转**：把幻觉从"要消除的缺陷"重构为"可优化的资源"，并罕见地把人文理论（Hartman 的批判性虚构）严谨地操作化为 NLP 可度量任务，跨学科桥接得很扎实。
- **审计严谨**：用全开放训练数据的 OLMO-2 做主体、双重污染审计（精确子串 + 行为探针）+ 隐形人物的二次去污，把"模型没见过这段历史"这个评估命门钉得很死，方法论上可作其他"未见数据"评估的范本。
- **评估贴题**：用叙事专用嵌入 story-emb 而非通用语义嵌入，并用人工标注调阈值，避免主题相似度混淆，体现对"叙事逼真度 ≠ 语义相近"的细腻把握。
- **社会价值取向明确**：附带人文使命声明与伦理声明，强调不是发明新历史、不坍缩思辨为事实，而是服务于档案沉默的"恢复技术"。

## 局限与展望
- **预备性强**：性能对提示和输入结构高度敏感，结论的稳健性受限。
- **单语言单文化**：实验只覆盖英语 + 黑人历史一个文化传统，泛化性未知。
- **缺训练/推理期优化**：当前全是零样本评估，尚未设计显式优化"良定义、证据约束虚构"的训练或推理方法。
- **伦理风险**：作者自陈需建立溯源追踪与伦理护栏，避免重建反过来"复合档案暴力"；未来方向含更鲁棒的开放式叙事评估、跨语言/文化扩展、provenance 机制。

## 相关工作与启发
- **有用的幻觉**：延续 Jiang et al. 2024、Sui et al. 2024 等"confabulation 有价值"的脉络，但首次给出了系统的证据约束评估框架。
- **AI 文本修复**：把 AlphaGeometry 式的古文修复（Assael et al. 2022/2025）、变音符号还原等领域特定修复，拓展到文化/历史叙事这一更开放的语义空间。
- **数据污染检测**：方法上呼应 Oren et al. 2024 的行为探针思路，并指出 MIA 在 OOD 下不可靠（Maini/Duan et al. 2024），故选择全开放模型做审计——这对任何"声称模型没见过测试数据"的评估都是有益提醒。
- **启发**：对从事评估的研究者，本文示范了"怎样把一个模糊的人文/创造性目标转化为有真值、可复现、可消融的基准"，尤其是污染审计与叙事专用度量这两个环节值得借鉴。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 把人文理论"批判性虚构"操作化为 NLP 任务、并把幻觉翻转为可优化资源，概念与问题设定都很原创。
- **实验充分度**: ⭐⭐⭐⭐ 19 个模型 × 多提示 × 类型/温度/长度/位置多维分析，双重污染审计扎实；但限于单语言单文化、零样本，缺少训练/推理期方法验证。
- **写作质量**: ⭐⭐⭐⭐⭐ 跨学科动机讲得清晰有力，研究问题 R1-R3 组织得当，理论与工程衔接自然。
- **价值**: ⭐⭐⭐⭐ 为"有益幻觉"与数字人文打开了一个可量化的研究方向，社会意义与方法论价值兼具，落地仍待后续工作推进。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Honest Lying: Understanding Memory Confabulation in Reflexive Agents](../../ICML2026/hallucination/honest_lying_understanding_memory_confabulation_in_reflexive_agents.md)
- [\[ICLR 2026\] HalluGuard: Demystifying Data-Driven and Reasoning-Driven Hallucinations in LLMs](halluguard_demystifying_data-driven_and_reasoning-driven_hallucinations_in_llms.md)
- [\[ACL 2026\] Vocabulary Hijacking in LVLMs: Unveiling Critical Attention Heads by Excluding Inert Tokens to Mitigate Hallucination](../../ACL2026/hallucination/vocabulary_hijacking_in_lvlms_unveiling_critical_attention_heads_by_excluding_in.md)
- [\[CVPR 2026\] FINER: MLLMs Hallucinate under Fine-grained Negative Queries](../../CVPR2026/hallucination/finer_mllms_hallucinate_under_fine-grained_negative_queries.md)
- [\[ICLR 2026\] GHOST: Hallucination-Inducing Image Generation for Multimodal LLMs](ghost_hallucination-inducing_image_generation_for_multimodal_llms.md)

</div>

<!-- RELATED:END -->
