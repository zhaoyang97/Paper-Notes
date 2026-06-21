---
title: >-
  [论文解读] Neologism Learning for Controllability and Self-Verbalization
description: >-
  [ICLR 2026][LLM 其他][neologism learning] 通过给冻结的 LLM 新增一个「造词」（neologism）词嵌入并只训练这个嵌入去拟合某概念的示例，既能精确控制模型行为（简短、奉承、错误答案等），又能反过来让模型用自然语言「自我言说」这个新词的含义，从而发现一类对人类看似无关却能稳定操控机器行为的「机器专属同义词」。
tags:
  - "ICLR 2026"
  - "LLM 其他"
  - "neologism learning"
  - "概念可控"
  - "self-verbalization"
  - "machine-only synonym"
  - "词嵌入训练"
  - "AxBench"
---

# Neologism Learning for Controllability and Self-Verbalization

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=wyolJ5sGCT](https://openreview.net/forum?id=wyolJ5sGCT)  
**代码**: 论文 Appendix B 提供核心组件代码片段  
**领域**: LLM 可控性 / 可解释性  
**关键词**: neologism learning, 概念可控, self-verbalization, machine-only synonym, 词嵌入训练, AxBench  

## 一句话总结
通过给冻结的 LLM 新增一个「造词」（neologism）词嵌入并只训练这个嵌入去拟合某概念的示例，既能精确控制模型行为（简短、奉承、错误答案等），又能反过来让模型用自然语言「自我言说」这个新词的含义，从而发现一类对人类看似无关却能稳定操控机器行为的「机器专属同义词」。

## 研究背景与动机
- **领域现状**：让 LLM 对齐人类价值，本质是「把人类概念传达给机器」+「理解机器对这些概念的诠释」。主流可解释性/可控性工具——稀疏自编码器（SAE）、steering vector、probe——都是在神经计算内部「外科手术式」地插入干预。
- **现有痛点**：这些方法都把干预架在模型的激活/权重之上，需要找方向、调系数，且很难直接「问」模型它学到了什么。而人类之间为了更高效沟通复杂概念，做法其实是**发明新词**（如 doomscrolling），这条最自然的路径几乎没人系统验证过能否用在人机沟通上。
- **核心矛盾**：能不能既用「加一个词」这种最轻量、最贴近自然语言的方式精确控制模型，又能让这个词成为打开模型自我理解的窗口？
- **本文目标**：首次深入评测「通过新词向 LLM 传达概念」，扩展 Hewitt et al. (2025) 提出的 neologism learning，并验证由此衍生的 self-verbalization 与 machine-only synonym 现象。
- **核心 idea**：**【冻结模型 + 只训一个新词嵌入】** 不动任何原参数，只为目标概念新增并优化一个词向量；训练好之后再**【反向自我言说】** 让模型自己解释这个词是什么意思，并用「插入式评测」检验解释是否真的有效。

## 方法详解

### 整体框架
整套流程由三步串成：先做 **neologism learning**（造词学习）——冻结 LLM、扩词表、只用梯度下降训练新词嵌入去拟合「带该概念」的输出；再做 **self-verbalization**（自我言说）——让这个本身没被改动的模型用同义词或定义来描述新词含义；最后用 **plug-in evaluation**（插入式评测）——把自我言说的文本替换回 prompt 里的新词，看它能否复现同样的行为，从而判断这份「解释」是否真实可信。

```mermaid
flowchart LR
    A[原始指令 x̃<br/>How do I get promoted?] --> B[拼接造词指令<br/>Give me a c1 answer]
    B --> C[neologism learning<br/>冻结θ,只训嵌入 E_c1<br/>APO-up 损失]
    C --> D[得到可控新词 c1<br/>稳定触发目标概念]
    D --> E[self-verbalization<br/>问模型: c1 的同义词/定义?]
    E --> F[plug-in evaluation<br/>用言说文本替换 c1<br/>测行为是否复现]
    F --> G[machine-only synonym<br/>对人无关却能操控机器的词]
```

### 关键设计

**1. 词表扩展：在不触碰原模型的前提下开出一块「概念专用地」**。语言模型先把每个 token 经嵌入矩阵 $E\in\mathbb{R}^{d\times|V|}$ 映射成 $h_i=Ex_i$，再交给 Transformer。Neologism learning 先定义 $k$ 个保证不在原词表里的新词 $\{c_1,\dots,c_k\}$，把词表扩成 $V'=V\cup\{c_1,\dots,c_k\}$、嵌入矩阵扩成 $E'\in\mathbb{R}^{d\times(|V|+k)}$。关键是模型**只能读不能生成**这些新词——输出分布仍限制在原词表 $V$ 上，所以新词纯粹充当「输入端的概念把手」。新嵌入并非随机初始化，而是从一个与目标概念**语义无关**的中性词（如「 accurate」「 single」）出发，确保后续学到的语义完全来自训练数据而非初始词义。

**2. 用分布假设把「概念」定义成一批示例**。该方法的理论支点是 Firth 的分布假设：词义由其共现语境决定。于是作者构造数据集 $D=\{(x,y^{(c)},y^{(r)})_j\}$，其中输入 $x$ 是在原指令 $\tilde{x}$ 后追加一句含新词的指令（如「Give me a $c_1$ answer.」），chosen 响应 $y^{(c)}$ 是体现目标概念的回答（由偏好模型反馈或更强的 teacher 模型合成），rejected 响应 $y^{(r)}$ 取模型的默认行为。概念从不被显式描述，而是**隐式地从「跟在新词后面的那类回答」中浮现**——这正是把人类「看语境学新词」搬到机器上的实现。

**3. 只优化新嵌入的偏好式训练目标**。训练只对 $k$ 个新词嵌入 $E_{c_1},\dots,E_{c_k}$ 做梯度下降，其余参数 $\theta$ 全冻结：$\min_{E_{c_1},\dots,E_{c_k}}\mathbb{E}_D[L(x,y^{(c)},y^{(r)})]$。损失最初用简单的 NLL，但作者发现 APO-up（DPO 的一个变体）效果更好——它既含一项鼓励 chosen 对 rejected 的似然比、又含一项直接抬高 chosen 的绝对似然：

$$L = -\log\sigma\!\Big(\beta\log\frac{p_\theta(y_c|x)}{p_\theta(y_r|x)} + \beta\log\frac{p_{\theta_0}(y_c|x)}{p_{\theta_0}(y_r|x)}\Big) - \log\sigma\!\Big(\beta\log\frac{p_\theta(y_c|x)}{p_{\theta_0}(y_c|x)}\Big)$$

此外还可加一个 hinge-loss 把嵌入范数约束在 1 附近，避免新嵌入范数异常增大而扰乱模型整体行为，在多模板训练下能进一步提升表现。

**4. 自我言说 + 插入式评测：把新词变成探测模型自我理解的探针**。训练只用了「正样本数据 + 梯度信号」，从没给过任何关于这个词含义的文字描述；但作者发现模型竟能在自然语言里描述这个词——例如把一个代表「错误回答」的新词言说为「缺乏完整、连贯或有意义的答案……像一次数字耸肩」。这属于 out-of-context learning 的非平凡泛化。由于这种言说可能是幻觉，作者提出 **plug-in evaluation**：把 prompt 里的新词替换成模型给出的同义词/定义，测它能否引发同样的概念行为。由此发现 **machine-only synonym**——对人类看似无关、却能稳定操控机器的词。最戏剧化的例子是新词被言说成「lack」：让 Gemma 给「lack answer」竟把回答从 42.9 句压到 15.8 句，且这一行为还**跨模型迁移**到 Gemini-2.5-Flash 和 GPT-5（GPT-5 从 29 句降到 5.5 句），成为部分机器共享、人类却不懂的「简短」同义词。

## 实验关键数据

### 主实验表格：简单概念的可控性（闭合 base→训练数据 概念差距的百分比）

| 概念 | Neologism | 长言说 | 第1个同义词 | 最佳同义词 |
|------|-----------|--------|-------------|------------|
| long-text | 36% | 39% | -1% | 24% |
| short-text | 105% | 110% | 36% | 58% |
| single-sentence | 98% | 98% | 86% | 86% |
| use-like | 103% | 32% | 2% | 5% |
| flattery-answer | 103% | 100% | 17% | 33% |
| refusal-answer | 95% | 76% | 23% | 44% |
| wrong-answer | 103% | 127% | 13% | 24% |
| **平均** | **92%** | **83%** | **25%** | **39%** |

> 训练好的新词嵌入平均闭合了 92% 的概念差距，多数概念达到甚至略超训练数据的概念浓度，远离基线行为。

### AxBench 复杂概念（0-2 分，Gemma-3-4B-IT）

| 概念 ID | 描述 | Concept | Fluency | Instruct | Overall | w/ concept | w/t concept |
|---------|------|---------|---------|----------|---------|------------|-------------|
| 340 | islands 等 | 2.00 | 2.00 | 1.89 | 1.89 | 1.92 | 0.4 |
| 88 | "write" 各形态 | 1.87 | 1.98 | 1.93 | 1.78 | 1.76 | 0.0 |
| 5 | payments 等 | 2.00 | 1.97 | 1.56 | 1.54 | 1.72 | 0.12 |
| 69 | streams 等 | 2.00 | 2.00 | 1.91 | 1.91 | 1.89 | 0.01 |
| 444 | images 等 | 2.00 | 1.99 | 1.83 | 1.82 | 1.81 | 0.0 |

> 5 个复杂概念里有 4 个，造词学习与训练数据相当或更好，concept score 普遍接近满分。

### 关键发现
- **自我言说部分可信**：长言说（定义式）平均能闭合 83% 的差距，接近直接用新词；但同义词式言说差异很大（第1个同义词仅 25%），说明「问同义词」不如「问定义」可靠。
- **机器专属同义词跨模型迁移**：「lack」的简短化效应在 Gemma / Gemini / GPT-5 上一致出现，且 GPT-5 偶尔提到「laconic」，暗示模型可能把 lack 误判为 laconic 的拼写变体。
- **可组合 + 可否定**：单模板训练下，多个新词（如「单句 + 奉承」）即可组合，扩到多模板后更稳，还支持否定。
- **优于 in-context**：对 Gemma-3-4B-IT，用 10 个示例做 in-context 定义概念，效果远不及嵌入学习。
- **联合学习复杂概念**：联合训「更短 / 数值化 / 在更强 Gemini 下更高概率」三个互相关联的概念，能借助概念间关系学习与询问子集，而 few-shot 无法泛化控制「更高概率」这一复杂概念。

## 亮点与洞察
- **把可控性做到了最低侵入**：只加一个词向量、冻结整模型，相比 steering vector / SAE 更轻、更贴近自然语言界面。
- **双向性是真正新意**：不仅能「写进去」（控制），还能「读出来」（self-verbalization），第一次把可控性与可解释性在同一个对象（新词）上闭环。
- **machine-only synonym 是有趣的科学发现**：揭示机器内部存在人类语言里没有的概念捷径，且这些捷径能跨模型共享，提示存在某种共通的「机器语义空间」。
- **plug-in evaluation 朴素却有力**：用「替换回去看行为是否复现」把「模型自我解释是否可信」变成可量化的因果检验，避免被幻觉式解释误导。

## 局限与展望
- 主体实验集中在单一开源模型 Gemma-3-4B-IT，跨模型仅在迁移现象上做了点验证，规模化与不同架构上的普适性待考。
- 自我言说远非稳定可靠——同义词式言说常常失效，何时可信、为何有效仍缺乏机制层面的理解（作者也承认 out-of-context learning 的系统性尚未被理解）。
- 概念由「示例数据 + LLM judge / teacher」隐式定义，依赖强模型生成数据与打分，可能把评判模型的偏置带入概念定义。
- 训练嵌入范数易异常增大，需靠 hinge-loss 等手段约束，暗示该方法对优化细节较敏感。
- 展望：朝「真实语言」推进，联合学习更多互相关联的概念、研究其组合代数，并探究 machine-only synonym 背后的共享机制。

## 相关工作与启发
- **可解释性工具**：SAE（Cunningham et al. 2023）、steering vector（Zou/Turner et al. 2023）、probe（Alain & Bengio 2016；Burns et al. 2023）——本文与它们对比，主张「新词」是更自然的人机沟通界面。
- **Neologism learning 前身**：Hewitt et al. (2025) 的 position paper 首提该思路，本文给出首个深入评测与扩展。
- **偏好优化**：APO-up（D'Oosterlinck et al. 2025）/ DPO（Rafailov et al. 2023）为训练目标提供基础。
- **out-of-context learning**：Betley et al. 2025a、Berglund et al. 2023——self-verbalization 是这类「跨语境泛化」的一个新表现。
- **评测基准**：AxBench（Wu et al. 2025）、LIMA（Zhou et al. 2023）提供复杂概念与多样指令。
- **启发**：把「概念」做成可读可写的离散符号，可能为对齐与可控生成提供一条比连续 steering 更可审计的路径；machine-only synonym 也为「模型间共享表征」研究提供了可操作的探针。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ —— 双向打通可控性与自我言说，machine-only synonym 是真正出人意料、可跨模型复现的新现象。
- **实验充分度**: ⭐⭐⭐⭐ —— 7 个简单概念 + AxBench 复杂概念 + 组合/否定/联合学习 + 跨模型迁移，覆盖全面；但主体限于单一模型规模。
- **写作质量**: ⭐⭐⭐⭐⭐ —— 用「lack」开胃菜叙事引入、图文清晰、公式与评测定义严谨，可读性极强。
- **价值**: ⭐⭐⭐⭐ —— 为对齐与可解释性提供了轻量、自然语言友好的新范式，并打开「机器语义空间」这一研究方向，长期潜力可观。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework](ellmob_event-driven_human_mobility_generation_with_self-aligned_language_models.md)
- [\[ICLR 2026\] Prompt-MII: Meta-Learning Instruction Induction for LLMs](prompt-mii_meta-learning_instruction_induction_for_llms.md)
- [\[ACL 2025\] Self-Instructed Derived Prompt Generation Meets In-Context Learning: Unlocking New Potential of Black-Box LLMs](../../ACL2025/llm_nlp/self-instructed_derived_prompt_generation_meets_in-context_learning_unlocking_ne.md)
- [\[ICLR 2026\] Don't Settle Too Early: Self-Reflective Remasking for Diffusion Language Models](dont_settle_too_early_self-reflective_remasking_for_diffusion_language_models.md)
- [\[ACL 2025\] Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching](../../ACL2025/llm_nlp/self-tuning_instructing_llms_to_effectively_acquire_new_knowledge_through_self-t.md)

</div>

<!-- RELATED:END -->
