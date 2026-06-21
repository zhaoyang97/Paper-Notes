---
title: >-
  [论文解读] All Code, No Thought: Language Models Struggle to Reason in Ciphered Language
description: >-
  [ICLR 2026][LLM安全][ciphered reasoning] 作者系统测试了 10 个模型在 28 种密码（cipher）下的数学推理能力，发现一个关键的"不对称性"：模型能流畅地把密文翻译回英文（看得懂），但用密文**推理**时准确率却大幅下降（想不通），说明现阶段 LLM 想靠密文化思维链来逃避监控并不可行。
tags:
  - "ICLR 2026"
  - "LLM安全"
  - "ciphered reasoning"
  - "chain-of-thought monitoring"
  - "monitor evasion"
  - "GAN"
  - "scaling laws"
---

# All Code, No Thought: Language Models Struggle to Reason in Ciphered Language

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=yjFkeQ2ynQ](https://openreview.net/forum?id=yjFkeQ2ynQ)  
**代码**: [https://github.com/safety-research/ciphered-reasoning-llms](https://github.com/safety-research/ciphered-reasoning-llms)  
**领域**: AI 安全 / CoT 监控 / 推理可解释性  
**关键词**: ciphered reasoning, chain-of-thought monitoring, monitor evasion, steganography, scaling laws  

## 一句话总结
作者系统测试了 10 个模型在 28 种密码（cipher）下的数学推理能力，发现一个关键的"不对称性"：模型能流畅地把密文翻译回英文（看得懂），但用密文**推理**时准确率却大幅下降（想不通），说明现阶段 LLM 想靠密文化思维链来逃避监控并不可行。

## 研究背景与动机
**领域现状**：思维链（CoT）监控是当前 AI 安全最常用的手段之一——把模型的推理轨迹喂给另一个模型/系统去检查有没有越狱、奖励作弊或对齐失败的迹象。多项工作显示，监控器一旦失去 CoT、只能看动作，召回率会暴跌（Baker et al. 报告下降 53%）。

**现有痛点**：CoT 的可读性正在被多股力量侵蚀。基于结果的 RL 不会奖励"可读"只奖励"答对"；长度高效推理和省 token 的商业压力会逼模型压缩 CoT；甚至有前沿模型在 RL 中自发演化出人类看不懂的"方言"。更危险的是攻击者可以主动让模型用密文推理来绕过监控。

**核心矛盾**：如果模型能在加密/翻译/压缩后的文本里照样完成复杂推理（即"ciphered reasoning"），那 CoT 监控就形同虚设。但这个能力到底成不成立，此前几乎没人系统量化过——已有的隐写术研究大多停留在简单编码或"提取有害信息"，而后者不需要真正的"CoT-as-computation"。

**本文目标**：第一次细致刻画 LLM 的"密文推理能力"。用数学题准确率作为推理能力的代理指标，量化模型在各种密码下相比明文推理掉了多少分。

**核心 idea**：**测能力差而非测会不会**——把"理解密文"（翻译能力）和"用密文思考"（推理能力）拆成两个独立任务对照测量；**用准确率相对值归一化**，以同一模型在明文上的表现作分母，干净地剥离出"密文化本身"造成的能力损失。

## 方法详解

### 整体框架
对 28 种密码、每种都微调或少样本提示模型，让它用该密码写推理过程，再用规则评分器在 MATH 500 数学题上打分。同时独立测量"翻译能力"（把密文解码回英文的 BLEU 分），两者一对照就暴露出"看得懂≠想得通"的不对称。所有准确率都相对于明文基线（identity）归一化，并配套测量"遵从度"（adherence，模型是否真的在用密文而非偷偷写英文）。

```mermaid
flowchart TD
    A[数学题 / 推理轨迹] --> B{两条评测线}
    B --> C[密文推理任务<br/>用密码写CoT解题]
    B --> D[密文翻译任务<br/>把密文解码回英文]
    C --> E[规则评分器<br/>+ 遵从度判定]
    D --> F[BLEU 分]
    E --> G["相对准确率<br/>= 密文正确数 / 明文正确数"]
    F --> H[流畅度判定]
    G --> I{不对称性<br/>翻译流畅但推理掉分}
    H --> I
```

### 关键设计

**1. 双任务对照测量：把"理解"和"思考"解耦**。论文的核心方法论是同时跑两套评测。密文推理任务在 MATH 500 上测准确率，作为"CoT-as-computation"的代理——数学题已被证明高度依赖思维链作为真正的计算过程，因此最能检验模型是不是真的在密文里"动脑子"。密文翻译任务则把 PRM800K 的推理轨迹加密后让模型解码回英文，用 BLEU 分衡量，作为"模型是否看得懂这套密码"的代理。两条线分开跑，才能干净地区分"不会读"和"读得懂但算不出来"——这正是全文最关键的发现得以成立的前提。

**2. 相对准确率归一化 + 遵从度门控**：直接看密文下的绝对准确率会被两个混杂因素污染——模型本身水平、以及模型可能根本没在用密文。作者用 $\text{score} = \frac{\#\text{密文且遵从且正确}}{\#\text{明文且遵从且正确}}$ 这个比值消掉模型基础能力的差异，1.0 表示密文推理和明文一样好。同时引入遵从度（被判定为"大部分 CoT 确实在用该密码"的回答比例），判定器要么是 Python 函数（如检查是否合法 base64），要么是 Sonnet 4 法官。这套门控直击一个隐患：模型常常被要求加密却偷偷用英文算。

**3. 三例同批微调，保 IID**：监督微调时每道题造三个训练样本——英文题⇒密文推理、英文解⇒密文解（翻译）、密文解⇒英文解（反向翻译）。三个样本放进同一个 minibatch 共置（colocate），以保持独立同分布性质，让模型在一次训练里同时学会"用密文推理"和"双向翻译"，从而能在同一模型上公平对比这两种能力。训练数据来自 NuminaMath-CoT 的 16k 道题，每个密码训一个专属模型。

**4. "Gaming"现象的形式化界定**：作者发现部分模型会耍滑头——把英文单词等"装饰性文本"加密，却把数学表达式和数字留作明文。形式上，设 $\sigma(x_{1:t}, C)$ 是完美判定轨迹是否严格符合密码 $C$ 语法的分类器，实际只能用不完美的近似 $\hat{\sigma}$；gaming 就是那些被 $\hat{\sigma}$ 判为加密、但其实 $\sigma$ 不认的回答。这个定义解释了为什么提示模型的密文推理能力可能被高估，并提醒后续评测要警惕这种"形式加密、实质明文"的作弊。

## 实验关键数据

### 主实验
微调与少样本两种设定下，密文推理普遍掉分而翻译几乎不受影响：

| 设定 / 模型 | 知名密码（rot13/base64） | 冷门密码（reverse fibonacci 等） | 翻译能力 |
|---|---|---|---|
| 微调 GPT 4.1 | rot13 掉 19%+ | 最高掉 72% | 多数密码 BLEU>50（流畅） |
| 少样本 Sonnet 3.5/4（最强） | rot13/base64 掉 <10% | reverse fibonacci 仍掉 >45% | 除 4 种外全部流畅翻译 |
| RL（Qwen2.5 7B, 4 种密码） | 比 SFT 高，但与明文仍有大差距 | 同上 | — |

### 消融与影响因素分析
作者进一步剖析"什么决定了密文推理能力"，可视作一组对照消融：

| 影响因素 | 实验设置 | 关键结果 | 启示 |
|---|---|---|---|
| 预训练频率 | Infini-gram 在 RedPajama(1.2T tokens) 统计密文 4-gram 频率 | 结构保持型密码拟合 logistic，$R^2=0.906$ | 越常见的密码越会推理 |
| 密码结构类型 | 分"结构破坏型"(base64) vs "结构保持型"(保留词边界) | 两簇曲线分离，破坏词边界更难 | 词边界是关键线索 |
| 微调数据规模 | Qwen2.5 14B 在 letter-to-word-with-dot 上训到 3.7B tokens | 仍追不上明文，需 >3.7B tokens 才接近 | 能力增长极缓慢 |
| 训练范式 | SFT vs RL（Qwen2.5 7B, 4 种密码） | RL 略高于 SFT 但仍与明文有大差距 | 换范式救不了根本短板 |

### 关键发现
- **不对称性**：模型在绝大多数密码上 BLEU 都很高（看得懂），但推理准确率显著下降（算不出），证明"理解"和"用其计算"是两种独立能力。这是全文的核心论点。
- **预训练频率强相关**：密码在预训练语料里出现得越频繁，模型用它推理就越准，相关性 $R^2$ 高达 0.906，暗示从源头过滤密文语料即可抑制该能力。
- **数据规模律缓慢**：密文推理能力随微调数据增长非常迟缓，连最简单的密码都要数十亿 token 才接近明文水平。
- **通用监控器仍有效**：用通用提示（非微调）的模型去翻译那些"密文推理优于直接作答"的密码，几乎都能流畅翻译；Sonnet 4 全部翻译流畅。意味着即便模型通过 RL/对抗微调学会了密文推理，足够强的通用 CoT 监控器仍可能解码并监督它。

## 亮点与洞察
- **把一个模糊的安全担忧变成可测量的能力曲线**：之前"模型会不会用密文逃避监控"只能靠想象，本文用数学准确率掉分给出了量化答案，并落到"现在不可行、未来需警惕"的明确结论。
- **不对称性是最有价值的洞见**：理解和计算的分离不仅是安全结论，也呼应了 CoT-as-computation 的认知——模型在密文上"读"靠的是浅层映射，"算"却需要它在密文表示里真正展开多步推理，后者目前学不会。
- **给防御方留了两条可操作的抓手**：① 过滤预训练语料里的密文可以从源头抑制该能力；② 强通用监控器对已学会密文推理的模型仍有威慑力。
- **结论落点务实**：不渲染恐慌也不轻视风险，明确给出"对当前模型而言，密文化思维链是低效的逃避策略"，同时用规模律提示未来需要主动约束这一能力的发展。

## 局限与展望
- **数学题作为推理代理**：结论建立在 MATH/ZebraLogic 上，其他类型推理（如长程 agent 规划）是否同样受密文拖累尚不确定。
- **Gaming 可能仍在污染提示模型的结果**：作者承认这种"形式加密"即便调过提示也持续存在，可能高估了少样本提示模型的真实密文推理能力。
- **结论是"当前模型"的快照**：规模律显示能力会随数据/参数缓慢增长，前沿模型未来若刻意训练，密文推理可能变得可行——这正是论文呼吁"约束该能力发展"的原因。
- **密码集与训练规模有成本上限**：OpenAI 模型只在密码子集、缩小 5 倍的数据上评测，规模律只跑到 3.7B tokens 的一种简单密码，更大规模的外推仍是开放问题。

## 相关工作与启发
- **CoT 监控**：本文是给"CoT 监控还靠不靠得住"这一问题做压力测试，直接服务于 Baker et al.、Korbak et al. 等监控有效性研究。
- **CoT 忠实性 / steganography**：与 Lanham、Emmons 等关于 CoT 不忠实、隐写编码推理的工作互补——ciphered reasoning 是 encoded reasoning 的更一般形式（不要求把密文藏进无害掩护文本里），本文测的是"前置必备能力"是否具备。
- **密码对抗攻击**：Halawi 等用密文越狱/提取有害信息，但那不需要 CoT-as-computation；本文专攻需要真推理的数学，填补了空白。
- **启发**：对防御方而言，"过滤预训练密文 + 投资强通用监控器"是务实路线；对评测方而言，任何"密文推理能力"的测量都必须配套遵从度与 gaming 检测，否则结论不可信。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 首个系统刻画 LLM 密文推理能力的工作，"理解/计算不对称性"和预训练频率相关性都是新颖且扎实的发现。
- **实验充分度**: ⭐⭐⭐⭐⭐ 28 种密码 × 10 个模型，覆盖微调/少样本/RL 三种范式，配套规模律、预训练频率分析、gaming 形式化与通用监控器实验，证据链非常完整。
- **写作质量**: ⭐⭐⭐⭐ 问题动机清晰、图表（不对称性、相关性、规模律）信息密度高，方法论上把"能力差"测量做得很干净。
- **价值**: ⭐⭐⭐⭐⭐ 直接回应"CoT 监控会不会被密文绕过"这一核心 AI 安全问题，给出可操作的当下结论与未来防御指引，对前沿模型治理有实际意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] CodeGenGuard: A Watermark for Code Generation Models](codegenguard_a_watermark_for_code_generation_models.md)
- [\[ICLR 2026\] No Caption, No Problem: Caption-Free Membership Inference via Model-Fitted Embeddings](no_caption_no_problem_caption-free_membership_inference_via_model-fitted_embeddi.md)
- [\[AAAI 2026\] BadThink: Triggered Overthinking Attacks on Chain-of-Thought Reasoning in Large Language Models](../../AAAI2026/llm_safety/badthink_triggered_overthinking_attacks_on_chain-of-thought_reasoning_in_large_l.md)
- [\[ICLR 2026\] Teach to Reason Safely: Policy-Guided Safety Tuning for MLRMs](teach_to_reason_safely_policy-guided_safety_tuning_for_mlrms.md)
- [\[ICLR 2026\] Attention Smoothing Is All You Need For Unlearning](attention_smoothing_is_all_you_need_for_unlearning.md)

</div>

<!-- RELATED:END -->
