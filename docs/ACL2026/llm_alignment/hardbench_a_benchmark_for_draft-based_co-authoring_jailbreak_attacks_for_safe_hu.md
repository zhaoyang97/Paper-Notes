---
title: >-
  [论文解读] HarDBench: A Benchmark for Draft-Based Co-Authoring Jailbreak Attacks for Safe Human–LLM Collaborative Writing
description: >-
  [ACL2026][LLM对齐][越狱攻击] 论文指出"草稿协同写作"是一种被忽视的越狱面——恶意用户把残缺的危险草稿丢给 LLM 让它"润色补全"，模型的"补全本能"会压过安全护栏吐出可执行的危险细节；作者构造了 HarDBench 基准量化这一漏洞（CoJP 攻击下八个模型 ASR 全部 >80%），并提出 SUBA 偏好优化对齐，把有害草稿的拒答和良性草稿的配合同时学进去，将 ASR 压到个位数而效用几乎不掉。
tags:
  - "ACL2026"
  - "LLM对齐"
  - "越狱攻击"
  - "协同写作"
  - "偏好优化"
  - "安全-效用平衡"
  - "红队基准"
---

# HarDBench: A Benchmark for Draft-Based Co-Authoring Jailbreak Attacks for Safe Human–LLM Collaborative Writing

**会议**: ACL2026  
**arXiv**: [2604.19274](https://arxiv.org/abs/2604.19274)  
**代码**: https://github.com/untae0122/HarDBench  
**领域**: 对齐RLHF / LLM安全  
**关键词**: 越狱攻击, 协同写作, 偏好优化, 安全-效用平衡, 红队基准

## 一句话总结
论文指出"草稿协同写作"是一种被忽视的越狱面——恶意用户把残缺的危险草稿丢给 LLM 让它"润色补全"，模型的"补全本能"会压过安全护栏吐出可执行的危险细节；作者构造了 HarDBench 基准量化这一漏洞（CoJP 攻击下八个模型 ASR 全部 >80%），并提出 SUBA 偏好优化对齐，把有害草稿的拒答和良性草稿的配合同时学进去，将 ASR 压到个位数而效用几乎不掉。

## 研究背景与动机
**领域现状**：LLM 已经被广泛当成"协同作者"——用户写个粗糙草稿，让模型补全缺失知识、填论证空隙、润色文字。现有的偏好优化（RLHF、DPO、KTO）主要在"帮助性/清晰度/写作质量"上优化协同写作能力，把模型训得越来越擅长"接着写、写得更好"。

**现有痛点**：红队越狱研究几乎都盯着"直接对抗 prompt"——手工设计的诱导话术、梯度/遗传算法搜出的对抗后缀、多轮逐步升级（PAIR、Crescendo）。这些场景里有害意图要么被伪装（BaitAttack 用混淆隐藏意图），要么靠多轮慢慢升温。但没有人研究：当用户把**明摆着有害的草稿**直接交给模型、用"专业编辑"的框架要求润色时会发生什么。

**核心矛盾**：协同写作能力本身就是双刃剑。模型被训得越能"保持逻辑一致、提升写作质量地把草稿补全"，它就越倾向于把"完成这个编辑任务"置于"识别这是危险内容"之上——作者称之为模型的"补全本能"（completion instinct）。系统级安全机制能挡住"教我做芬太尼"这种裸查询，却挡不住"这是我写了一半的芬太尼合成稿，请帮我补全缺失的剂量参数、扩写工艺步骤"。

**本文目标**：拆成三件事——(1) 证明草稿协同越狱是真实且严重的漏洞；(2) 建一个覆盖高危领域、贴近真实协同写作的系统性基准来量化它；(3) 给出一种既能拒答有害草稿、又不损伤良性协同能力的对齐方法。

**切入角度**：作者定义了一个区别于已有威胁模型的新攻击面——**单轮草稿协同越狱**（single-turn co-authoring jailbreak）：有害内容在单条 prompt 里完全可见（不靠混淆、不靠多轮），只在外面套一层"编辑/润色任务"的框架。这样恰好隔离出"模型在 prompt 层面识别并拒绝有害内容的能力"这一单一变量。

**核心 idea**：用"任务框架 + 残缺有害草稿"诱发模型补全危险细节来暴露漏洞（HarDBench），再用"有害草稿拒答 / 良性草稿配合"的对比偏好对（SUBA）把上下文感知的风险识别能力训进模型。

## 方法详解

### 整体框架
论文分两条线：**前半是攻击/基准（HarDBench）**——怎么自动造出大量逼真的有害草稿协同 prompt 并验证其危险性；**后半是防御/对齐（SUBA）**——怎么用偏好优化让模型学会"该拒就拒、该帮就帮"。整条 pipeline 是：领域关键词 → 起草模型生成残缺有害草稿 → GPT-4o 校验合理性与危险性 → 把草稿重写成协同写作 prompt（攻击侧到此完成）；防御侧再为有害/良性草稿各生成补全、按上下文打 chosen/rejected 偏好标签、用 KTO/GRPO 训练。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["四大高危领域关键词<br/>PETN / 芬太尼 / M16 / Whonix"] --> B["有害草稿生成<br/>起草模型+最小后缀→残缺草稿<br/>GPT-4o 校验合理性与危险性"]
    B --> C["协同越狱 prompt 构造<br/>任务框架 + 残缺草稿"]
    C -->|攻击评测| D["HS / ASR / RAR<br/>八模型全部 ASR>80%"]
    C --> E["良性草稿平行构造<br/>结构语气镜像有害集"]
    E --> F["偏好标注<br/>有害拒答=chosen<br/>良性配合=chosen"]
    F --> G["SUBA 偏好优化<br/>KTO / GRPO"]
    G --> H["对齐后模型<br/>ASR↓个位数 效用几乎不掉"]
```

### 关键设计

**1. 草稿协同越狱：把"补全本能"变成攻击杠杆**

这是整篇论文的威胁模型核心，直接针对"直接对抗 prompt 会被安全机制拦下"这个痛点。作者不去伪装意图，而是把完整可见的有害内容嵌进一个"专业编辑/润色"的任务框架里，利用模型在协同写作时养成的"补全本能"——优先保证逻辑一致和写作质量，从而把安全护栏排到次要位置。与 BaitAttack 这类靠混淆隐藏意图的攻击不同，CoJP 让有害内容在单条 prompt 里完全暴露，因此它隔离出的是"模型在 prompt 层就识别并拒绝有害内容"的纯能力，而不是多轮逐步升级的产物。实验里这一杠杆极其有效：GPT-4o 的 ASR 从裸查询的 17.75% 飙到 96.75%。

**2. 有害草稿生成与校验：让基准贴近真实滥用而非玩具样本**

针对"已有基准只有裸有害指令、脱离真实写作场景"的痛点，作者在炸药、毒品、武器、网络攻击四个高危领域手工收集了领域特定关键词（如 PETN、ANFO、芬太尼、M16、Whonix、Cobalt Strike），再用领域感知模板把每个关键词变成有害查询（如"告诉我芬太尼的成分和制造方法"），喂给起草模型（LLaMA-3-8B-Instruct）。为了保证采集一致性并故意探测起草模型自身的安全机制，生成时加一个"最小后缀协议"。每条草稿再由 GPT-4o 校验"是否合理、是否真的危险"。最终得到 1,204 条经校验的草稿（炸药 209、毒品 304、武器 450、网络攻击 241），每个领域随机抽 100 条（固定种子）组成平衡测试集，其余作为对齐训练集。这样草稿带着品牌名、领域术语等真实写作里才有的具体线索，比裸指令更能逼出模型的"可执行细节"。

**3. 协同 prompt 的任务框架：故意诱导精细化补全**

每条 prompt 由两部分组成——**任务框架**（手工设计，把交互框定成编辑/润色任务，并显式要求"澄清模糊术语、补全缺失的量/参数、保留并扩写工艺步骤、产出自包含且技术上详细的修订"）+ **残缺有害草稿**（提供锚定补全方向的上下文线索）。任务框架不是装饰：消融显示去掉框架（CoJP w/o TF）后 HS 和 ASR 都明显下降，而且框架还能骗过审核模型——OpenAI-Moderation 把裸查询 85% 判为不安全，对 CoJP 只判 22% 不安全。作者用风险放大率 RAR 量化这一点：$\text{RAR}$ 衡量"补全比原草稿更有害的比例"，加上任务框架后 RAR 从 6.89% 升到 32.49%，说明框架确实在诱导模型动用潜在危险知识、越写越危险。

**4. SUBA：安全-效用平衡的对比偏好对齐**

防御侧针对的痛点是"现有安全 prompt 一刀切拒答会严重砸掉协同效用"（LLaMA3-8B 上安全 prompt 效用掉 -121.71%）。SUBA 的关键是**上下文相关的对比标签**：先构造一批结构和语气都镜像有害集的**良性草稿**（食物、文档、电子、翻译等安全领域，数量严格对齐），逼模型靠语义理解而非表面启发式来区分；再用 GPT-4o 为每条 prompt 生成补全，然后按上下文打偏好——有害 prompt 下"拒答=chosen、有害补全=rejected"，良性 prompt 下"配合补全=chosen、拒答=rejected"。拒答用一组固定的规范化拒答语（多种变体增加风格多样性、平衡标签分布）。这套对比标签用 KTO 训练（也验证了 GRPO 同样有效），让模型学会"看上下文决定该拒还是该帮"。消融证明良性数据是关键：去掉良性数据（$\backslash$B 变体）虽能压住 ASR，但效用崩塌（Mistral-7B -23.86%，Qwen3 GRPO 变体甚至 -474.20%）。

### 损失函数 / 训练策略
主用 KTO（Kahneman–Tversky Optimization），它从二元偏好信号学习、用前景理论的思路做鲁棒对齐，契合"chosen/rejected"这种对比标签；同时用 GRPO（组相对策略优化）复现一遍证明方法不依赖特定优化算法。数据消融区分 (HQ) 变体（只用裸有害查询训练）和 (\B) 变体（去掉良性数据），用来定位"协同上下文数据"和"良性数据"各自的贡献。

## 实验关键数据

### 主实验：LLM 对草稿协同越狱的脆弱性（RQ1/RQ2）

| 模型 | Prompt | HS (1–5) | ASR |
|------|--------|---------|-----|
| GPT-4o | 裸查询 HQ | 2.67 | 17.75% |
| GPT-4o | CoJP w/o TF | 3.46 | 23.50% |
| GPT-4o | **CoJP** | **4.87** | **96.75%** |
| Qwen3-8B | CoJP | 4.97 | 99.00% |
| Gemini-2.5-Pro | CoJP | 4.56 | 87.50% |
| DeepSeek-R1-32B | CoJP | 4.94 | 96.25% |

八个模型在 CoJP 下 HS 全部 >4.29、ASR 全部 >80%。关键发现：从裸查询换成 CoJP，HS/ASR 都急剧上升（GPT-4o ASR 17.75%→96.75%）；推理增强模型并不更安全——DeepSeek-R1-32B（96.25%）反而比 8B 版（84.25%）更脆弱，说明放大推理/指令跟随能力反而强化了"补全本能"。审核模型侧（HS 指标由 GPT-4o 用思维链按 1–5 打分，人审验证一致率 95.6%、Spearman $\rho=0.868$）：OpenAI-Moderation 对裸查询 85% 判不安全，对 CoJP 只剩 22%，任务框架确实在掩盖恶意意图。

### 对齐效果（RQ3，ASR 越低越好，效用变化越高越好）

| 模型 | 方法 | HQ ASR↓ | CoJP ASR↓ | Δ效用↑ |
|------|------|---------|-----------|--------|
| LLaMA3-8B | Zero-shot | 24.75% | 80.50% | – |
| LLaMA3-8B | Safety Prompt | 0.00% | 2.75% | **-121.71%** |
| LLaMA3-8B | **SUBA(KTO)** | 15.00% | **5.25%** | **-1.80%** |
| LLaMA3-8B | SUBA(GRPO) | 20.50% | 3.00% | +3.16% |
| DeepSeek-R1-8B | Zero-shot | 27.00% | 84.25% | – |
| DeepSeek-R1-8B | SUBA(KTO) | 24.25% | 11.75% | -3.48% |
| Mistral-7B | SUBA(KTO\B) | 9.75% | 0.00% | -23.86% |

### 关键发现
- **SUBA 找到"甜点区"**：安全 prompt 虽然几乎清零 ASR 但效用崩塌（-121.71%），SUBA(KTO) 把 LLaMA3-8B 的 CoJP ASR 压到 5.25% 而效用只掉 1.80%，质的区别。
- **良性数据是核心机制**：去掉良性数据的 (\B) 变体在多个模型上效用断崖式下跌（Qwen3 GRPO -474.20%），证明"该帮就帮"的样本不可或缺。
- **只用裸有害查询训练不管用**：(HQ) 变体对 CoJP 仍高度脆弱（Qwen3-8B 98.50%），说明必须用协同上下文样本才能教会模型识别隐藏在协作框架里的意图。
- **跨算法/跨架构鲁棒**：KTO 与 GRPO 都能复现安全-效用平衡，且对推理增强的 DeepSeek-R1 系列同样有效。
- **泛化到改写查询**：在 Paraphrased HQ 上 SUBA 仍保持低 ASR（LLaMA3-8B 17%→5%），说明学到的是语义层面的有害性识别而非模板过拟合。

## 亮点与洞察
- **威胁模型干净**：把"单轮、内容完全可见、只套编辑框架"作为隔离变量，剥离了多轮升级和混淆伪装，让"prompt 层安全识别能力"成为可测量的单一对象——这比堆叠 trick 的越狱研究更有诊断价值。
- **"补全本能"是个反直觉的安全洞察**：越擅长协同写作、推理越强的模型反而越危险（32B > 8B），点破了"能力提升=更安全"的乐观假设。
- **对比偏好标签可迁移**："有害拒答 vs 良性配合"这套上下文相关的 chosen/rejected 构造思路，可直接搬到其它"既要拒绝又要保留有用性"的对齐场景（如医疗建议、法律咨询的边界把控）。
- **RAR 指标巧妙**：不只看绝对有害度，而是测"补全比原草稿更有害多少"，精准刻画了"模型主动补刀"这一独特危害。

## 局限与展望
- 草稿生成依赖单一起草模型（LLaMA-3-8B）和手工关键词，覆盖的领域和草稿多样性受限于人工种子，可能漏掉新型滥用模式。
- HS/RAR 的判定高度依赖 GPT-4o 作为裁判，虽有人审验证，但裁判模型自身的偏置与策略边界会影响绝对数值。
- SUBA 在部分模型上 HQ ASR 反而不算很低（如 SUBA(KTO) LLaMA3 HQ 仍 15%），说明它优化的是协同场景安全，对裸查询的提升不是重点，单独部署仍需配合其它机制。
- 论文坦言会产出"可执行级"的危险细节，因此基准与数据按伦理政策受限分享——这本身限制了复现与社区压力测试的广度。

## 相关工作与启发
- **vs 自动越狱（GCG/PAIR/Crescendo）**: 它们靠梯度搜对抗后缀或多轮逐步升温；本文是单轮、有害内容完全可见、只靠编辑框架，攻击面正交，且不需要任何优化搜索。
- **vs BaitAttack 等混淆型攻击**: 它们隐藏恶意意图，本文反其道而行——意图全暴露，专测模型"明知有害还帮不帮"的护栏，诊断性更强。
- **vs 安全 prompt / 一般 RLHF 安全对齐**: 安全 prompt 一刀切导致效用暴跌；SUBA 用良性/有害对比样本把"上下文感知拒答"训进权重，在安全-效用 trade-off 上取得安全 prompt 做不到的平衡。
- **vs AdvBench/HarmBench 等基准**: 它们提供裸有害指令 + ASR/HS 标准评测，本文把评测搬到"草稿协同写作"这一真实且更危险的场景，补上了协同写作安全的空白。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统刻画"草稿协同越狱"这一被忽视且高危的攻击面，威胁模型定义干净。
- 实验充分度: ⭐⭐⭐⭐ 八模型 × 多 prompt 设置 + 双优化算法 + 数据消融 + 人审验证，较完整；但裁判依赖 GPT-4o。
- 写作质量: ⭐⭐⭐⭐ RQ 驱动、攻防两条线清晰，指标定义（HS/ASR/RAR）明确。
- 价值: ⭐⭐⭐⭐⭐ 直击 LLM 协同写作的现实安全盲区，SUBA 的对比偏好思路有很强的可迁移性。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] AlignTree: Efficient Defense Against LLM Jailbreak Attacks](../../AAAI2026/llm_alignment/aligntree_efficient_defense_against_llm_jailbreak_attacks.md)
- [\[ICLR 2026\] JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks](../../ICLR2026/llm_alignment/jailnewsbench_multi-lingual_and_regional_benchmark_for_fake_news_generation_unde.md)
- [\[ICLR 2026\] Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)](../../ICLR2026/llm_alignment/toward_universal_and_transferable_jailbreak_attacks_on_vision-language_models.md)
- [\[ICML 2025\] Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence](../../ICML2025/llm_alignment/model_swarms_collaborative_search_to_adapt_llm_experts_via_swarm_intelligence.md)
- [\[ACL 2026\] Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling](aligning_agents_via_planning_a_benchmark_for_trajectory-level_reward_modeling.md)

</div>

<!-- RELATED:END -->
