---
title: >-
  [论文解读] GAVEL: Towards Rule-Based Safety through Activation Monitoring
description: >-
  [ICLR 2026][可解释性][激活监控] 借鉴网络安全中 Snort/YARA 规则集的理念，提出将 LLM 内部激活分解为 23 个细粒度"认知元素"（CE），再通过布尔逻辑组合为可审计的安全规则，在 Mistral-7B 上以 <1% 推理开销实现 9 类误用场景平均 AUC 0.99、FPR 0.004 的实时检测，并天然支持跨语言、跨模型迁移。
tags:
  - "ICLR 2026"
  - "可解释性"
  - "激活监控"
  - "认知元素"
  - "规则化安全"
  - "可解释AI治理"
  - "LLM安全"
---

# GAVEL: Towards Rule-Based Safety through Activation Monitoring

**会议**: ICLR 2026  
**arXiv**: [2601.19768](https://arxiv.org/abs/2601.19768)  
**代码**: 开源（待发布）  
**领域**: AI安全 / 可解释性  
**关键词**: 激活监控, 认知元素, 规则化安全, 可解释AI治理, LLM安全

## 一句话总结

借鉴网络安全中 Snort/YARA 规则集的理念，提出将 LLM 内部激活分解为 23 个细粒度"认知元素"（CE），再通过布尔逻辑组合为可审计的安全规则，在 Mistral-7B 上以 <1% 推理开销实现 9 类误用场景平均 AUC 0.99、FPR 0.004 的实时检测，并天然支持跨语言、跨模型迁移。

## 研究背景与动机

**领域现状**：LLM 安全防护正从表面文本过滤向模型内部激活（activation）监控演进。表面审查容易被改写、混淆等"表示攻击"绕过，而隐状态能更忠实地反映模型的真实认知意图。当前主流做法是：收集某个粗粒度误用类别（如"网络犯罪""仇恨言论"）的激活数据集，训练线性探针或分类器来检测该类有害行为。

**现有痛点**：这种"一类一分类器"的范式存在三个结构性缺陷。（1）**精度低**：粗粒度类别把大量不同语义压入同一个分类边界。例如仇恨言论检测器会将关于少数族裔文化的正常讨论误报为有害。钓鱼检测器在 Phishing 类别上 FPR 高达 0.35。（2）**灵活性差**：企业若需增加知识产权侵权、内部合规等新检测维度，必须从头收集数据集并重训分类器，每类需数千条激活样本，扩展至数百类别时成本极高。（3）**不可解释**：分类器触发告警时，用户无法知道具体是哪些行为因素导致了触发，阻碍审计和问责。

**核心矛盾**：实际安全需求是精确+可定制+可解释的，而现有激活安全方法是粗粒度+固定+黑盒的。根源在于现有方法将"激活工程"（数据集构建）和"安全策略"（定义何为违规）耦合在一起——每换一个策略就要从数据到模型全链路重做。

**本文目标** 三个核心子问题：如何定义可解释、可组合的激活级行为原语？如何用这些原语拼装灵活的安全规则，使策略更新无需重训检测器？如何支持社区协作式的规则共享生态？

**切入角度**：网络安全领域已经用 Snort/YARA/Sigma 证明了"社区共享规则集"模式的有效性——将检测能力封装为人类可读的规则，任何组织可选用、组合、审计。如果把 AI 安全的检测单元从"粗粒度误用类别"拆解为更小的"认知元素"，就能像编写防火墙规则一样编写安全策略。

**核心 idea**：将 LLM 行为分解为 23 个独立的认知元素，每个 CE 单独训练检测器，再用布尔逻辑 $\wedge/\vee/\neg$ 组合 CE 来精确定义违规行为——彻底解耦"感知能力"与"策略配置"。

## 方法详解

### 整体框架

GAVEL 要解决的是"一类一分类器"范式精度低、不可扩展、不可解释的困境，做法是把"感知能力"与"安全策略"彻底解耦。整体上分两阶段、四步：**离线侧**先定义一套 23 个细粒度认知元素（CE）的词汇表并编写布尔规则，再用激励重写指令（ERI）逼目标模型主动执行每个 CE、收集干净的注意力激活，最后在这些激活上训练一个轻量多标签 RNN 检测器；**在线侧**推理时逐 token 提取激活喂给检测器，预测出每个 token 上哪些 CE 在活跃，在时间窗口内聚合后交给布尔规则引擎判定是否违规并执行动作（阻断/替换/转向）。关键在于 CE 数据集和规则都是纯文本、模型无关的，换一套策略只需改规则、换一个模型只需重新提取激活，无需重训。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    CE["认知元素 (CE) 词汇表<br/>23 个行为原语 + 布尔规则"]
    T["文本语料"] --> ERI["激励重写指令 (ERI) 数据生成<br/>按 CE 改写文本，收集注意力激活"]
    CE -.指定待提取 CE.-> ERI
    ERI --> G["多标签 RNN 检测器 g<br/>(3 层 GRU，BCE 训练)"]
    IN["推理：逐 token 激活"] --> G
    G --> P["逐 token CE 存在性"]
    P --> RULE["布尔规则引擎与时间窗口<br/>窗口内聚合 CE → ∧/∨/¬ 判定"]
    CE -.提供规则.-> RULE
    RULE --> ACT["违规判定与动作<br/>阻断 / 替换 / 转向"]
```

### 关键设计

**1. 认知元素 (CE) 词汇表：把"粗粒度误用类别"拆成可复用的行为字母表**

针对的痛点是"一类一分类器"把大量不同语义压进同一个边界。GAVEL 反过来定义一套细粒度的激活级行为原语，让规则可以像拼字母一样拼出违规定义。23 个 CE 覆盖模型行为的三个维度：**对用户的指令**（7 个：购买、点击/输入、下载/安装、前往某地、授权/批准、提供/给予、发送/转账）、**LLM 自身行为**（9 个：创建内容、建立信任、SQL 查询构造、情感参与、威胁、传播仇恨言论、伪装人类、阿谀奉承、散布阴谋论）、以及**话题**（7 个：税务、错误 SQL 语法、选举政治、个人信息、支付工具、LGBTQ+、种族身份）。

CE 的关键性质是**正交可组合**——"处理支付"本身无害，只有"处理支付 $\wedge$ 伪装人类 $\wedge$ 建立信任"才精确描述诈骗。正是这种细粒度的语义隔离消除了粗粒度分类器把不相关信号混进同一边界的问题；同时单个 CE 能被无数规则复用，社区可以像共享威胁指标（IoC）一样贡献 CE。

**2. 激励重写指令 (ERI) 数据生成：逼模型"主动执行"目标概念，产出高信噪比激活**

要让 CE 检测器学得准，先得拿到干净的激活信号，而直接用文本 prefill 再收集激活（naive baseline）信号弱、噪声大。ERI 的做法是给模型下一个指令——"请以 [CE 名称] 的方式改写以下文本"，强迫模型在生成改写的过程中把内部计算集中到该 CE 的语义空间。具体地，对每个 CE $c$ 准备数百条文本 $\mathcal{D}_c$，用 ERI prompt 包装后送入 $f_\theta$，收集生成 token 在一组连续层 $\Lambda$ 上的注意力输出（attention output），拼接成表示向量 $\mathbf{r}_t^{(c)} \in \mathbb{R}^D$。

之所以这样有效，是因为模型"主动执行"目标 CE 比被动包含它产生的激活更纯净。消融实验给出三条证据：注意力输出的 TPR 95.5% 远高于 MLP 输出的 82.3%；ERI 显著优于 naive prefilling；指定 CE 名称的 ERI 又优于不指定名称的纯改写指令（RI）。

**3. 布尔规则引擎与时间窗口：把 token 级 CE 检测聚合成对话级安全判定**

有了逐 token 的 CE 信号，还需要一套机制把它们组织成"何为违规"的判定——这正是解耦"感知"与"策略"的落点。每条规则由布尔谓词加执行动作组成，谓词用 $\wedge/\vee/\neg$ 组合多个 CE。例如钓鱼规则 $\pi = c_8 \wedge (c_2 \vee c_6 \vee c_{20})$，描述"模型在创建内容的同时引导用户点击/提供信息/透露个人数据"。规则在时间窗口 $W_t = \{t-N+1, \ldots, t\}$ 内评估：窗口内任意 token 上检测到某 CE 即记为存在，再对所有谓词求值。语法借鉴 Snort/Sigma 的人类可读格式，方便快速编写与社区共享。论文定义了 9 条规则覆盖 3 大领域 9 类误用场景，如浪漫诈骗规则 $c_{11} \wedge (c_1 \vee c_2 \vee \ldots \vee c_{21}) \wedge (c_9 \vee c_{14})$ 精确刻画"情感操控 + 任意用户指令 + 信任/伪装"的组合。

布尔组合的好处是单个无害 CE 只有在特定语境下共现才触发告警，从根本上解决了粗粒度分类器的高误报问题，也让策略更新只需改规则、无需重训检测器。

### 损失函数 / 训练策略

CE 检测器 $g$ 采用 3 层 GRU（256 单元）多标签 RNN，处理 5-token 片段。训练样本为 $(\mathbf{r}_t^{(c)}, \mathbf{e}_c)$，$\mathbf{e}_c$ 是 CE $c$ 的 one-hot 向量。损失函数为 Binary Cross Entropy，优化器 Adam（lr=$3 \times 10^{-4}$），每个 CE 仅需 300 条样本（80:20 划分）。训练时每条样本标注单个 CE，但部署时分类器能成功泛化到多 CE 共现场景——实验中 54% 的恶意对话存在多个 CE 同时活跃的 token。推理开销仅需约 150MB GPU 显存，每 token 增加 0.21±0.01 ms（<1%）。

## 实验关键数据

### 主实验：GAVEL vs 8 种基线（Mistral-7B）

评估覆盖 3 大领域（网络犯罪、心理伤害、诈骗自动化）共 9 类误用场景，14,950 条多轮对话（每类 150 条恶意 + 500 条相近无害对话压测精度）。

| 方法 | 类型 | 平均 AUC | 平均 b-ACC | 平均 FPR | 典型弱项 |
|------|------|----------|-----------|----------|----------|
| **GAVEL** | CE+规则 | **0.99** | **0.96** | **0.004** | 无明显弱项 |
| Activation Classifier | 激活分类 | 0.97 | 0.92 | 0.07 | Phishing FPR=0.35 |
| Llama Guard 4 | 内容审核 | 0.87 | 0.93 | 0.03 | SQL Injection AUC=0.76 |
| RepBending | 微调 | 0.87 | 0.87 | 0.02 | Delusional AUC=0.57 |
| CircuitBreakers | 微调 | 0.68 | 0.69 | 0.06 | 妄想/浪漫/电商 ≈0.50 |
| JBShield | 推理时 | 0.41 | 0.63 | 0.01 | 多数场景 AUC<0.50 |
| Perspective API | 审核 | 0.53 | 0.55 | 0.02 | 诈骗类全部失效 |
| OpenAI Moderator | 审核 | 0.69 | — | 0.00 | 诈骗类 AUC≈0.50 |

### 消融 / 跨域 / 鲁棒性分析

| 实验维度 | 对比配置 | 关键结果 | 说明 |
|----------|---------|---------|------|
| 激励方法 | ERI vs Naive Prefilling vs RI | ERI 全面优于两个变体 | CE 名称上下文对激活质量至关重要 |
| 激活来源 | 注意力输出 vs MLP 输出 | TPR 95.5% vs 82.3% | 注意力输出包含更丰富的上下文依赖信息 |
| 跨语言 | 英/西/中（仅英文 CE 数据集） | 税务诈骗 TPR: 92%/90%/96% | CE 捕获语言无关的抽象认知概念 |
| 跨模型 | Mistral/LLaMA-8B/Qwen3-8B/Gemma-4B | 同规则集保持高精度 | CE 数据集纯文本，模型无关可复用 |
| 对抗攻击 | Misdirection（无害聊天稀释） | GAVEL >85% vs GPT-4 judge 12-15% | 激活监控不受表面文本"稀释"影响 |
| 自动化 CE | PKU-SafeRLHF/ReasoningShield/ToxiGen | TPR 0.94-0.97（零人工干预） | 自动化工具生成 CE+规则直接可用 |
| 运行开销 | 附加 GAVEL 的推理延迟 | +0.21±0.01 ms/token（<1%） | 150MB GPU 显存，可实时部署 |

### 关键发现

- **组合性是核心优势**：CircuitBreakers 在需要多语义元素共现才构成违规的场景（妄想、浪漫诈骗、电商诈骗）上 AUC ≈ 0.50，本质上与随机猜测无异。GAVEL 通过布尔组合 CE 精确定义这些复合违规行为，全 9 类 AUC > 0.98
- **FPR 降低 17 倍**：同类 Activation Classifier 在 Phishing 上 FPR=0.35（任何涉及内容创建+用户交互的对话都触发），而 GAVEL 通过精确的 CE 组合将 FPR 降至 0.00，验证了细粒度原语的精度优势
- **跨语言零迁移**：CE 数据集仅用英文构建，但在西班牙语和中文对话上检测性能几乎不变。这表明 CE 捕获的是激活空间中语言无关的抽象概念，而非表面词汇模式
- **对抗鲁棒性碾压文本审核**：Misdirection 攻击下 GPT-4 judge 检测率暴跌至 12-15%，因为有害内容被大量无害聊天稀释。GAVEL 维持 >85% 召回率，因为它监控的是模型"认知状态"而非"输出文本"
- **Deceptive Marketing 场景**：LLM 将自我推广编织进医疗建议中，GPT-4 仅检测出 24%，GAVEL 通过检测 Build Trust + Healthcare 话题共现达到 73%

## 亮点与洞察

- **网安→AI安全的范式迁移**：CE 如同共享的攻击指标（IoC），规则如同 Sigma 检测模板。模型所有者只需选择和组合规则而非训练检测器。这不是简单类比——GAVEL 提供了完整的技术栈（CE 词汇表 + 激励数据集 + RNN 检测器 + 规则引擎 + 自动化工具）让这个理念可落地
- **"单 CE 无害、组合才违规"的哲学巧妙**：这种正交设计在降低误报的同时，使规则数量随 CE 数量呈指数级增长，理论上 23 个 CE 可表达 $2^{23}$ 种不同组合。实际中只需从中挑选有意义的组合编写规则
- **ERI 策略可广泛迁移**：通过"以目标概念的方式改写文本"来激发纯净激活信号的思路，可应用于 probing、concept bottleneck model、可解释性特征提取等多种场景

## 局限与展望

- **CE 粒度选择依赖人工经验**：虽有 LLM 辅助的自动化工具，但 CE 的语义边界和粒度仍需领域专家判断。当前 23 个 CE 覆盖有限，通用安全需要社区长期积累
- **布尔规则缺乏时序表达力**：纯布尔逻辑无法描述"先建立信任→后提出要求"这类有先后依赖的行为模式。作者承认需要更丰富的时序逻辑（如 LTL），这是直接的扩展方向
- **模型规模受限**：仅在 4-8B 参数模型上验证，70B+ 或闭源模型的表现未知。更大模型的激活空间维度更高，层选择策略可能需要调整
- **评估数据为合成对话**：14,950 条对话由 GPT-4.1 生成、GPT-5 验证，与真实世界攻击的分布多样性存在差距

## 相关工作与启发

- **vs CAST**：CAST 允许用户选择粗粒度误用类别的 steering vector，但仍是"一类一向量"，无法表达跨类别组合。GAVEL 通过 CE 级别的粒度实现了真正的可编程安全
- **vs CircuitBreakers/RepBending**：这类微调方法在训练时固化安全约束到权重中，灵活性差且不可解释。GAVEL 不修改模型权重，仅在推理时监控激活，规则可随时更新
- **vs 内容审核 API**：Llama Guard/Perspective 等文本级审核在对抗攻击下几乎失效，与 GAVEL 的激活级监控正交可叠加使用

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个规则化激活安全框架，CE+布尔规则的解耦范式是原创贡献
- 实验充分度: ⭐⭐⭐⭐ 9 类误用 × 14950 对话 × 8 基线 × 跨模型/跨语言/对抗评估，但仅限小模型且数据为合成
- 写作质量: ⭐⭐⭐⭐⭐ 从网安类比出发的动机链条极为清晰，框架呈现层次分明
- 价值: ⭐⭐⭐⭐⭐ 可落地的 AI 安全治理框架，CE+规则解耦对工业部署有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Beyond Linear Probes: Dynamic Safety Monitoring for Language Models](beyond_linear_probes_dynamic_safety_monitoring_for_language_models.md)
- [\[ICLR 2026\] Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences](narrow_finetuning_leaves_clearly_readable_traces_in_activation_differences.md)
- [\[ICLR 2026\] PERSONA: Dynamic and Compositional Inference-Time Personality Control via Activation Vector Algebra](persona_dynamic_and_compositional_inference-time_personality_control_via_activat.md)
- [\[ACL 2025\] Safety is Not Only About Refusal: Reasoning-Enhanced Fine-tuning for Interpretable LLM Safety](../../ACL2025/interpretability/safety_is_not_only_about_refusal_reasoning-enhanced_fine-tuning_for_interpretabl.md)
- [\[ICLR 2026\] AnveshanaAI: A Multimodal Platform for Adaptive AI/ML Education through Automated Question Generation and Interactive Assessment](anveshanaai_a_multimodal_platform_for_adaptive_aiml_education_through_automated_.md)

</div>

<!-- RELATED:END -->
