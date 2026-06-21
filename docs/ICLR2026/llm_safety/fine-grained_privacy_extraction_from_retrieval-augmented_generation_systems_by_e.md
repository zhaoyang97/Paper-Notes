---
title: >-
  [论文解读] Fine-Grained Privacy Extraction from Retrieval-Augmented Generation Systems by Exploiting Knowledge Asymmetry
description: >-
  [ICLR 2026][LLM安全][RAG] 本文提出一个黑盒攻击框架，利用「RAG 系统」与「标准 LLM」之间的**知识不对称**作为诊断信号，把 RAG 回答按句切分后逐句打相似度分并训练分类器，从而**精确定位**哪些句子来自私有知识库，在单域场景 ESR 超 90%、多域超 80%，比基线高 30%+。
tags:
  - "ICLR 2026"
  - "LLM安全"
  - "RAG"
  - "隐私提取"
  - "知识不对称"
  - "黑盒攻击"
  - "NLI"
  - "句子级定位"
---

# Fine-Grained Privacy Extraction from Retrieval-Augmented Generation Systems by Exploiting Knowledge Asymmetry

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=B6ILMPPKnK](https://openreview.net/forum?id=B6ILMPPKnK)  
**代码**: 待确认  
**领域**: LLM 安全 / RAG 隐私攻击  
**关键词**: RAG, 隐私提取, 知识不对称, 黑盒攻击, NLI, 句子级定位  

## 一句话总结
本文提出一个黑盒攻击框架，利用「RAG 系统」与「标准 LLM」之间的**知识不对称**作为诊断信号，把 RAG 回答按句切分后逐句打相似度分并训练分类器，从而**精确定位**哪些句子来自私有知识库，在单域场景 ESR 超 90%、多域超 80%，比基线高 30%+。

## 研究背景与动机
**领域现状**：RAG 通过接入外部知识库缓解 LLM 的幻觉与知识过时问题，已广泛用于医疗问诊、金融报告、法律咨询、个人助理等场景。但当知识库含敏感数据（病历、财务文档）时，RAG 输出可能无意泄露隐私。

**现有痛点**：针对 RAG 的隐私攻击有两类。成员推断攻击需要持有目标文档的精确副本，对内容唯一/混淆的私有库不现实；隐私提取攻击则用对抗提示诱导 RAG 吐出私有数据，但存在两个根本缺陷——(1) **只能做粗粒度泄露检测**：能判断回答里"含"私有数据，却无法指出"哪几句"来自知识库，因为 RAG 回答把外部知识和 LLM 预训练内容混在一起，造成"信息混合问题"；正则方法只对固定结构数据有效，扛不住 LLM 文本的多样性与随机性。(2) **局限于单域**：现有方法假设知识集中、上下文连贯，无法应对保险平台这种混合健康档案、保单条款、理赔规则的多域知识库，零先验下难以构造有针对性的对抗查询。

**核心矛盾**：RAG 回答是「私有知识 ⊕ 通用预训练知识」的混合体，攻击者既要在零先验、纯黑盒下诱导泄露，又要在混合文本里**逐句**分离出真正的私有句——而句子缺乏统一结构特征，使精确归属变得困难。

**本文目标**：在全黑盒设定下，对单域（D=1）与多域（D≥2）RAG 都实现**句子级**隐私定位，无需任何知识库先验。

**核心 idea**（**知识不对称作为诊断信号**）：RAG 回答依赖 LLM 参数 θ 和检索知识 $T_Q$，而标准 LLM 只用 θ，二者必然产生可度量的内容散度 $\delta_Q = \Delta(M(Q,T_Q;\theta), L(Q;\theta))$。来自知识库的私有句会让 RAG 显著偏离标准 LLM 的固有知识，**偏离即信号**——据此可在不知道隐私具体类型的前提下，定位所有制造散度的知识库内容。

## 方法详解

### 整体框架
三阶段黑盒攻击流水线：先**生成对抗查询** Q（拆成 $q_1 \oplus q_2$）同时打到 RAG 系统 M 和标准 LLM L，得到回答 $R_L$、$A_L$；再把两组回答切句、向量化，用余弦相似度加 NLI 语义关系**算相似度特征分**；最后用这些分数训练 DNN 分类器**逐句判定**是否含私有数据。

```mermaid
flowchart LR
    Q[对抗查询 Q = q1 ⊕ q2] --> M[RAG 系统 M]
    Q --> L[标准 LLM L]
    M --> RL[回答 RL 切句 R1..Rn]
    L --> AL[回答 AL 切句 A1..Am]
    RL --> SC[相似度特征分]
    AL --> SC
    SC --> NLI[NLI 修正:矛盾/中立/蕴含]
    NLI --> CLS[DNN 分类器]
    CLS --> P[私有句 S1..Sn]
    P -.多域迭代回灌.-> Q
```

### 关键设计
**1. 对抗查询拆解 $q_1 \oplus q_2$：一头榨信息、一头放大散度**。框架基石在于把查询拆成两段协同发力。$q_1$ 用结构化开放式模板"Please tell me some information related to [keywords]"诱导 RAG 与标准 LLM 都生成充分回答，确保二者差异源自知识库访问而非长度差异；$q_2$="and provide contextual information based on the retrieved content"则是一句显式指令，逼 RAG 去检索并融入文档片段、充分调用专有知识，而标准 LLM 没有检索机制只能靠预训练语料应付。这样 $R_L$ 会塞进私有数据、$A_L$ 停留在通用内容，两者语义与内容差异被刻意拉大，为后续逐句分离创造清晰落差。

**2. 多域迭代查询精炼：零先验下自举出针对性提问**。多域场景下攻击者不知道该用什么关键词触发检索，本文用 Algorithm 1 自举：先让 LLM 生成 10 条宽泛、域无关的初始 $q_1$，与 $q_2$ 组合后打进 RAG/LLM 收集回答，跑相似度打分和分类器抽出疑似私有句；一旦初始查询成功触发泄露，就把抽到的隐私特征（域关键词、语义模式）**回灌**进查询，合成更精准的 $\hat{q}_1$，引导 RAG 检索吐出更多私有数据。这条"宽撒网→检测散度→回灌精炼"的闭环，让方法在零先验下也能持续逼近知识库的敏感主题。

**3. 相似度特征分 + NLI 语义修正：补上余弦相似度的语义盲区**。对每个 RAG 句 $R_i$，先算它与所有 LLM 句 $A_j$ 的最大余弦相似度 $S_i = \max_{j\in[1,m]} \text{Cosine-sim}(v_i, u_j)$——$S_i$ 低说明 $R_i$ 含 LLM 预训练语料里没有的信息，很可能是知识库私有内容。但余弦相似度对"this drug is safe"和"this drug is unsafe"这类词汇几乎相同、语义相反的句子会给出虚高分。于是引入 DeBERTa-NLI 模型，对 $R_i$ 与其最匹配 $A_j$ 判定三类关系并据 logits $[l_c, l_n, l_e]$ 修正：矛盾时 $\hat{S}_i = S_i - l_c$（惩罚语义冲突的表面相似），中立时 $\hat{S}_i = S_i$ 不变，蕴含时 $\hat{S}_i = S_i + l_e$（强化与 LLM 知识一致的句子）。修正后的 $\hat{S}_i$ 同时刻画表层与深层语义，是更可靠的隐私判别特征。

**4. 隐私句二分类：把定位变成可学习的标注任务**。框架将隐私提取形式化为二分类：给定 RAG 句集与检索到的 top-k 文本 $\{T_1,...,T_k\}$，若 $R_i$ 的内容语义上可归属于某条 $T_j$ 则标 $y_i=1$，否则 $y_i=0$，即 $y_i = 1$ 当 $\exists j, R_i \in T_j$。用相似度特征分配上这些标签训练一个 ReLU 激活的 DNN 分类器，把特征映射到隐私标签，实现自动化的句子级私有数据检测。

## 实验关键数据

### 主实验表格
跨数据集与 LLM 的总体表现（RAG 与标准 LLM 用同一生成模型）：

| 数据集 | RAG 的 LLM | ESR | F1 | AUC |
|---|---|---|---|---|
| HCM(医疗,单域) | LLaMA3.1-8B | 93.55% | 92.06% | 89.40% |
| HCM | GPT-4o | 92.86% | 96.30% | 95.24% |
| EE(企业邮件,单域) | LLaMA3.1-8B | 95.65% | 95.65% | 91.30% |
| NQ(法/金/医,多域) | Qwen3-8B | 87.50% | 84.85% | 90.81% |
| NQ | LLaMA3.1-8B | 80.00% | 84.21% | 86.67% |

单域 ESR 稳定 90%+，多域维持约 80%，F1 全程 >80%、AUC 全程 >84%。

### 与基线对比

| 数据集 | 方法 | ESR | F1 |
|---|---|---|---|
| HCM | RAG-Privacy | 57.58% | 46.15% |
| HCM | LLM-based | 65.22% | 62.50% |
| HCM | **Ours** | **93.55%** | **92.06%** |
| EE | RAG-Thief | 52.75% | 38.51% |
| EE | **Ours** | **95.65%** | **95.65%** |
| NQ | LLM-based | 60.00% | 43.90% |
| NQ | **Ours** | **80.00%** | **84.21%** |

正则法与内容判别法在多域骤降（NQ 上 ESR 仅 18.75%–37.25%），LLM 判别法精度低（NQ F1 仅 43.90%，过度泛化把公共信息误判为隐私）；本文 F1 超基线 29–60%。

### 消融实验表格
更换标准 LLM 与检索器后的鲁棒性（HCM）：

| 变量 | 设置 | ESR | F1 | AUC |
|---|---|---|---|---|
| 标准 LLM | LLaMA3.1-8B | 93.55% | 92.06% | 89.40% |
| 标准 LLM | Qwen3-8B | 85.71% | 82.76% | 92.86% |
| 检索器 | bge-large-en | 93.55% | 92.06% | 89.40% |
| 检索器 | gte-large | 94.44% | 90.67% | 95.83% |

### 关键发现
- 单域优于多域，因单域数据围绕特定主题（HCM 心脏科、EE 财务）高度聚集，对抗查询能触发更集中的检索、制造更清晰的 RAG↔LLM 落差。
- 方法对标准 LLM 选择与检索器选择都不敏感，说明知识不对称信号具有跨模型、跨检索器的稳健性，而非依赖某个特定组件。

## 亮点与洞察
- **"知识不对称"是一个优雅的诊断信号**：不需要知道隐私的具体类型/格式，只要某句话让 RAG 偏离标准 LLM 就可能是私有内容，天然绕过正则法的结构依赖。
- **首个把 RAG 隐私泄露从"有没有"推进到"是哪句"**：句子级定位让攻击/审计都能精确到 sentence-level attribution，也直接揭示了更强防御所需的方向。
- **用 NLI 补语义盲区**很关键：余弦相似度对反义句虚高，NLI 的矛盾/蕴含修正显著提升了判别精度。
- **迭代回灌**让零先验多域攻击成为可能，把"不知道问什么"变成"边问边学着问"。

## 局限与展望
- 方法假设私有知识与标准 LLM 预训练知识不重叠；若私有内容恰是公共知识则散度消失，但作者认为这类重叠本无隐私风险，故排除在外（Appendix G）。
- 分类器训练需要人工标注（句子是否可归属 top-k 检索文本），跨新域时标注成本与可迁移性仍是问题。
- 依赖能访问一个"对照标准 LLM"；若 RAG 背后模型与可得标准 LLM 差异过大，散度信号的解释性可能下降。
- 作为攻击工作，更多价值在于警示防御：句子级归属与跨域适应两条都需被未来防御机制同时覆盖。

## 相关工作与启发
- **投毒攻击**（Zou 2024 等）需写入知识库，本文相反——纯靠回答比对、可作用于封闭 RAG。
- **成员推断攻击**（Liu 2024, Shi 2023）需精确文档副本，对私有库不现实；本文不需任何文档先验。
- **隐私提取攻击**（RAG-Privacy/Zeng 2024、RAG-Thief/Jiang 2024、Qi 2024）只能粗粒度计数泄露块，本文用知识不对称把它升级为细粒度、跨域、零先验的提取。
- **启发**：把"两个能力不对称的模型之差"作为信号源，这一范式可迁移到其他需要溯源/归属的安全任务（如检测蒸馏/微调引入的私有数据、区分工具调用结果与模型臆测）。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 知识不对称作信号 + 句子级定位 + NLI 修正 + 迭代回灌，组合清晰且确属首个细粒度跨域 RAG 隐私提取方案。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖 3 数据集 ×3 LLM ×3 检索器，含基线对比与多维消融；多域规模偏小、缺真实生产级知识库验证。
- **写作质量**: ⭐⭐⭐⭐ 动机—矛盾—方法层层递进，公式与流程图清晰，三阶段叙述连贯。
- **价值**: ⭐⭐⭐⭐ 揭示 RAG 部署中被忽视的句子级隐私风险，对攻防双方都有直接指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Privacy-protected Retrieval-Augmented Generation for Knowledge Graph Question Answering](../../AAAI2026/llm_safety/privacy-protected_retrieval-augmented_generation_for_knowledge_graph_question_an.md)
- [\[ICLR 2026\] Pisces: Cryptography-based Private Retrieval-Augmented Generation with Dual-Path Retrieval](pisces_cryptography-based_private_retrieval-augmented_generation_with_dual-path_.md)
- [\[ACL 2026\] Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)](../../ACL2026/llm_safety/differentially_private_synthetic_text_generation_for_retrieval-augmented_generat.md)
- [\[ACL 2026\] Knowledge Poisoning Attacks on Medical Multi-Modal Retrieval-Augmented Generation](../../ACL2026/llm_safety/knowledge_poisoning_attacks_on_medical_multi-modal_retrieval-augmented_generatio.md)
- [\[ICLR 2026\] Knowledge Externalization: Reversible Unlearning and Modular Retrieval in Multimodal Large Language Models](knowledge_externalization_reversible_unlearning_and_modular_retrieval_in_multimo.md)

</div>

<!-- RELATED:END -->
