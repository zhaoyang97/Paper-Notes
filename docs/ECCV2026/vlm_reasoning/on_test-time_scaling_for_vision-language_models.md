---
title: >-
  [论文解读] On Test-Time Scaling for Vision-Language Models
description: >-
  [ECCV 2026][VLM Reasoning][测试时扩展] 这是第一篇系统研究「LLM 上的测试时扩展（test-time scaling）方法能否直接搬到大型视觉语言模型（LVLM）上」的实证工作，横跨 13 个模型、9 种扩展策略、6 个 benchmark，颠覆了「小模型不受益」的旧结论——小而强的指令模型受益最大（最高约 +30%，可达到甚至超过大模型），同时揭示了「算力过量反而伤害感知任务」和「视觉信息在推理链早期就被编码进文本 token」两个关键现象。
tags:
  - "ECCV 2026"
  - "VLM Reasoning"
  - "测试时扩展"
  - "视觉语言模型"
  - "思维链"
  - "注意力分析"
  - "系统性评估"
---

# On Test-Time Scaling for Vision-Language Models

**会议**: ECCV 2026  
**arXiv**: [2606.28864](https://arxiv.org/abs/2606.28864)   
**代码**: 无  
**领域**: 多模态VLM / LLM推理 / 测试时扩展  
**关键词**: 测试时扩展, 视觉语言模型, 思维链, 注意力分析, 系统性评估

## 一句话总结
这是第一篇系统研究「LLM 上的测试时扩展（test-time scaling）方法能否直接搬到大型视觉语言模型（LVLM）上」的实证工作，横跨 13 个模型、9 种扩展策略、6 个 benchmark，颠覆了「小模型不受益」的旧结论——小而强的指令模型受益最大（最高约 +30%，可达到甚至超过大模型），同时揭示了「算力过量反而伤害感知任务」和「视觉信息在推理链早期就被编码进文本 token」两个关键现象。

## 研究背景与动机

**领域现状与痛点**：测试时扩展指的是在推理阶段多花算力（不改权重）来换取更强的性能，它已成为「参数扩展」之外的另一条提升路线，在 LLM 上被证明常常比单纯把模型做大更划算。这类方法分两派：一派要额外训练（SFT + RL，训出所谓「thinking model」），另一派完全零训练、靠 prompting / sampling / aggregation / 迭代精炼来「激活」指令模型里本已存在的推理能力。本文聚焦零训练这一派。然而这套方法在 LVLM 上「灵不灵」几乎无人系统回答过。更麻烦的是，此前文献留下了一个几乎成为共识的负面结论：多篇 LLM 工作报告小模型（<10B）不受益于测试时扩展、甚至 prompt 它推理会掉分；针对 VLM 的近期工作（如在 SmolVLM2 上的评测、以及对 LLaVA 系列的 CoT 研究）同样报告「思维链 prompting 在 LVLM 上失效」。于是研究者干脆假设「LLM 的测试时扩展方法迁移不到 VLM」，转而去设计 VLM 专用方案（如 Compositional CoT 从图里抽场景图再喂回 prompt）。

**核心矛盾与本文目标**：本文要挑战的正是这个「LLM 方法不能迁移到 VLM」的假设。作者提出一个朴素但根本的问题——为 LLM 开发的常规测试时扩展方法，能不能直接用在 LVLM 上？他们的目标不是再造一个新方法，而是做一次真正覆盖多模型尺度、多方法、多任务类型（感知 vs 推理）的对照实验，把「是否有效、何时有效、有效到什么程度」讲清楚，并深入到模型内部（注意力动力学、推理链信息充分性）去解释现象背后的机制。这决定了本文是一篇**空白填补 / 系统研究型**工作，而非提出单一 pipeline 的方法论文。

本文的核心 idea 是：把 9 种主流零训练测试时扩展策略搬到 13 个开源 LVLM 上做地毯式评测，用统一的「思考提示 + 可靠答案抽取」协议消除混淆因素，从而得到三条可落地的结论——(1) 小而强的模型受益最大（与 LLM 文献相反），(2) 算力过量会让 LVLM「失焦」、在感知任务上掉分，(3) 视觉信息在推理链早期被「前置编码」进文本 token、之后推理几乎不再看图。之前工作得到相反结论，本文归因于所用模型（SmolVLM2、LLaVA-OneVision）指令遵从差、直接答题而忽略思考提示，更像专用 VQA 系统而非会聊天的指令模型。

## 方法详解

### 整体框架

本文没有单一 pipeline，而是一套**统一评测协议 + 三层分析**：先用同一套 prompt 模板把 9 种测试时扩展方法（多数来自 LLM 文献，少数为 VLM 专用）套在 13 个模型上跑 6 个 benchmark，得到主结果；再从三个角度深挖机制——**token 预算**（截断到 300 token 看性能如何变）、**注意力动力学**（推理链生成过程中对图像 token 的注意力如何演化，并用 KV-cache 丢弃做因果干预验证）、**推理链质量**（用 LLM-as-judge 测「只读推理链能否答对」的信息充分性，并做句子级的时序分析）。

统一协议里两个关键工程点保证了结论可信：**思考提示（think-prompt）机制**——所有方法都实现为在 post-prompt 前插一段 think-prompt，最终输入为 `{question};{think-prompt};{post-prompt}`，模型先输出推理链 $r$ 再给答案（例如 CoT 的 think-prompt 就是「think step-by-step」）；**可靠答案抽取**——若第一次没抽出规范格式的答案，就用 `{question};{r};{answer-only};{post-prompt}` 二次请求，强制模型基于已生成的 $r$ 只给答案。这一步很重要：它让「截断到 300 token 后仍能抽出答案」成为可能，也排除了「掉分是因为答案没抽出来」这个混淆。

### 关键设计

**1. 九种测试时扩展方法的统一套用：把 LLM 武器库整体搬到 LVLM**

本文的第一大贡献不是发明方法，而是把九种策略放到同一协议下公平对照，覆盖「单次前向多耗 token」「多次采样投票」「多次精炼」三大类算力形态。具体：① CoT（单次前向，多耗思考 token）；② S-CoT（结构化 CoT，引导问题分解→信息组织→逻辑推理→总结）；③ Plan-and-Solve（先规划再执行）；④ Self-Consistency（采样 $b$ 条 CoT 投票取众数，$b$ 次前向）；⑤ Self-Aggregation（保留全部 $b$ 条链拼接后让模型聚合，$b{+}1$ 次前向）；⑥ Self-Refinement（先出一条 CoT 再迭代精炼 $k$ 轮，$k{+}1$ 次前向）；⑦ Describe-Answer（先让模型详细描述图像再据文本描述作答，2 次前向）；⑧ Compositional CoT / CCoT（先生成场景图再作答，VLM 专用，2 次前向）；⑨ Prompt Repetition（把问题重复一遍，think-prompt = question，制造类似双向注意力的效果，单次前向、不产生思考 token）。把它们并列跑，才能得出「简单 CoT 在运行时/性能权衡上最优、且常规方法显著优于 VLM 专用的 CCoT」这种横向结论。

**2. Token 预算分析：定位「感知任务为何被算力伤害」**

为回答「为什么额外算力会伤害纯感知 benchmark」，作者把最大输出长度从 1024 砍到 300 token 重跑全部实验，报告 $\Delta = \text{score}_{300} - \text{score}_{1024}$。因为有可靠答案抽取，即使推理链在 300 token 被截断也能拿到最终答案，这个对照才干净。结论很反直觉：与 LLM 文献「模型在写思维链之前就已经知道答案」相反，本文发现减少 token 预算会掉分——说明**推理链真的在驱动答案、多花 token 生成中间步骤确实有用**，但这只在推理 benchmark 成立；在 RealWorldQA、常在 HallusionBench 这类感知/幻觉任务上，增加 token 预算反而伤害或至少无益。据此提出「失焦」假说：给 LVLM 过量算力去 overthink，它会编造貌似合理的叙事、做错误假设、累积链式错误，最终幻觉、偏离正确答案。作者进一步指出可以训一个简单的二分类器判断任务是否真需要推理，从而省下大量算力。

**3. 注意力动力学 + KV-cache 因果干预：揭示「视觉信息前置编码」**

这是全文机制层最硬的一块。作者把每个生成 token $g_y$ 的注意力分解为三部分：对图像 token 的 $a^{img}=\sum a_{l}^{h}[1{:}N_m]$、对文本 prompt 的 $a^{pmt}=\sum a[N_m{+}1{:}N_m{+}T]$、对已生成 token 的 $a^{gen}=\sum a[N_m{+}T{+}1{:}N_m{+}T{+}(y{-}1)]$（$N_m$ 为进入 LLM 的压缩视觉 token 数，Qwen 系列 $N_m = N/4$）。由于三者归一，$a^{pmt}=1-(\sum a^{img}+\sum a^{gen})$ 可直接推出。绘出 $a^{img}$、$a^{gen}$ 随生成步的曲线（跨层跨头跨样本平均，并额外画逐 token 的 $\max[a^{img}]$ 上界）后发现：**图像注意力在开头短暂达峰、随后快速衰减，而对已生成 token 的注意力持续上升并最终主导**。即视觉 grounding 是「前置」的——推理链早期有一个短窗口把图像信息吸收进文本 token 的隐藏表示，之后模型主要靠这份隐藏表示和前文推理，几乎不再回看图像。为把相关性坐实为因果，作者做 KV-cache 丢弃干预：每数据集取 300 样本，在生成第 {20, 50, 100, 200, 300} 步时把图像 token 的 Key/Value 缓存从所有层彻底移除再继续生成，测各丢弃步的准确率。结果早期丢图像会显著掉分（因为正处于「前置编码」关键期），约 200 步之后再丢几乎不影响准确率——这既验证了前置编码，也解释了设计 2 的失焦：长链越到后面越依赖自己生成的文本 token 去推理，错误由此累积放大。

**4. 推理链信息充分性的 LLM-as-Judge 双指标**

为评估生成的多模态推理链「质量/信息是否够」，作者用一个外部 LLM 当裁判，并用人工评估验证其可靠性。两个互补指标：**Rationale Sufficiency（充分性）**——裁判只拿到问题 $q$ 和推理链 $r$（看不到图像、外部知识、真值），仅凭 $r$ 的文本内容预测答案 $\hat{x}_r$，若只读推理链就能答对，说明 $r$ 信息充分；把这个 rationale accuracy 与 LVLM 自身准确率对比。**Rationale Dynamics（动力学）**——把 $r$ 拆成有序句子序列 $\{r_1,\dots,r_S\}$，逐句判定其信息是支持正答、支持负答还是不确定，再聚合出「最早决策点、推理步间的立场翻转/矛盾、中间不确定模式、最终定论位置」等统计量。这套指标让「推理链到底有没有承载决策信号、信息在链中如何逐步积累」变得可测量，而不只是看最终对错。

### 一个例子：4B 模型如何追平 32B

以 Qwen3-VL-4B 为例走一遍主结论：它在 LogicVista 的 baseline 是 40.85，加最简单的 CoT 后升到 57.59——而 8 倍大的 Qwen3-VL-32B baseline 才 44.42，即 4B+CoT 反超 32B 约 +13%；换 Self-Consistency 差距更大（62.95）。在 HallusionBench、WeMath 上同样出现「4B 加 CoT/S-CoT 就超过 32B baseline」。Pareto 前沿显示：约 9 秒算力预算下，4B 在 LogicVista、WeMath 上直接跑赢 32B，在 HallusionBench 上追平、MMStar 上接近。这就是 Takeaway 1 的落地含义——可以部署更便宜、更快、更省显存的小 VLM，靠测试时扩展拿到接近甚至超过大模型的精度。

## 实验关键数据

评测覆盖 13 个模型（Qwen2.5-VL 的 7B/32B/72B，Qwen3-VL 的 2B/4B/8B/32B，InternVL-3.5 的 2B/4B/8B/38B，Molmo2 的 4B/8B），6 个 benchmark（MMStar、RealWorldQA、HallusionBench、WeMath、LogicVista、A-OKVQA），最大 token 长度 1024，总实验成本约 6000 美元，runtime 在单张 NVIDIA H200（140GB）上测（72B 用两张）。

### 主实验：Qwen 系列测试时扩展结果（节选）

下表节选自 Table 1，展示小模型受益最大、以及方法在感知任务（RealWorldQA）上易掉分的对比（分数越高越好，runtime 单位秒）：

| 模型 | 方法 | MMStar | RealWorldQA | HallusionBench | WeMath | LogicVista | Runtime |
|------|------|--------|-------------|----------------|--------|-----------|---------|
| Qwen3-VL-2B | Baseline | 51.69 | 65.10 | 63.72 | 35.46 | 32.59 | 0.1 |
| Qwen3-VL-2B | Self-Consistency | 63.83 | 64.31↓ | 71.08 | **64.25** | 46.88 | 40.2 |
| Qwen3-VL-4B | Baseline | 61.71 | 71.76 | 70.66 | 58.22 | 40.85 | 0.1 |
| Qwen3-VL-4B | CoT | 68.59 | 69.80↓ | 75.50 | 71.55 | **57.59** | 5.9 |
| Qwen3-VL-4B | Self-Consistency | 71.07 | 72.55 | 74.97 | 74.25 | 62.95 | 44.8 |
| Qwen3-VL-32B | Baseline | 72.66 | 77.25 | 74.76 | 66.03 | 44.42 | 0.2 |
| Qwen2.5-VL-72B | Baseline | 68.65 | 69.67 | 74.97 | 72.99 | 47.54 | 0.3 |

关键发现：
- **小模型受益最大，且能反超大模型**：Self-Consistency 把 Qwen3-VL-2B 的 WeMath 从 35.46 拉到 64.25（约 +29%）；Qwen3-VL-4B 仅用 CoT，其 LogicVista(57.59)、WeMath(71.55)、HallusionBench(75.50) 就全面超过 8 倍大的 Qwen3-VL-32B baseline（44.42 / 66.03 / 74.76）。这与 LLM 文献「小模型不受益」完全相反。
- **感知任务常被伤害**：RealWorldQA（纯感知、重空间理解）上多数方法相对 baseline 掉分（表中 ↓），而在 WeMath/LogicVista 这类推理 benchmark 上普遍大涨；说明测试时扩展的收益高度依赖任务「是否真需要推理」。
- **简单 CoT 性价比最高**：CoT 单次前向、runtime 仅 ~6 秒却拿到接近 Self-Consistency（~45 秒、多次采样）的多数收益，是运行时/性能权衡的最优选择；常规 LLM 方法整体显著优于 VLM 专用的 CCoT。
- **VLM 专用方法未必更好**：CCoT、Describe-Answer 在很多模型/任务上不如朴素 CoT，甚至掉到 baseline 以下。

### 分析实验：Token 预算截断到 300（$\Delta = \text{score}_{300}-\text{score}_{1024}$，节选）

下表节选自 Table 2，负值表示截断后掉分（即长链有用），正值表示截断反而更好（即长链有害）：

| 模型 | 方法 | MMStar | RealWorldQA | HallusionBench | WeMath | LogicVista |
|------|------|--------|-------------|----------------|--------|-----------|
| Qwen3-VL-4B | CoT | -2.07 | +0.79 | -0.42 | **-11.32** | **-7.37** |
| Qwen3-VL-4B | S-CoT | -2.80 | +3.79 | -0.11 | -14.89 | -13.61 |
| Qwen3-VL-8B | CoT | -0.55 | -0.79 | -2.10 | -6.04 | -0.44 |
| Qwen2.5-VL-72B | Self-Refinement | -0.15 | **+10.07** | +1.68 | +2.76 | -0.23 |
| Qwen2.5-VL-72B | Self-Aggregation | -1.52 | +6.14 | +1.58 | -1.10 | -0.67 |

关键发现：
- **推理任务：长链驱动答案**。WeMath、LogicVista 上截断到 300 token 普遍大幅掉分（4B+CoT 在 WeMath 掉 11.32、LogicVista 掉 7.37），证明中间推理步骤真在起作用，而非「模型早就知道答案」。
- **感知任务：短答案更好**。RealWorldQA 上截断反而常常持平或涨（4B+S-CoT 涨 3.79，72B+Self-Refinement 涨 10.07），印证「感知任务偏好简洁输出、过量 token 让模型失焦幻觉」的假说。
- **KV-cache 丢弃干预**：早期（第 20~50 步）丢弃图像 token 显著掉分，约 200 步后再丢几乎无影响——坐实「视觉信息前置编码」。
- **推理链充分性**：多数 benchmark 上「只读推理链」的裁判准确率紧贴 LVLM 自身准确率，说明推理链承载了决策相关信号；模型越大/越新，二者差距越小（可解释性越好）。HallusionBench 的句子级时序显示不确定性早现、约链的中段(∼0.5)开始出现对正答的支持、约 0.75 处信息已充分定论，推理链是「渐进组织」而非一开始就下结论。

## 亮点与洞察
- **一次「证伪」式的系统研究，价值高于新方法**：它推翻了「LLM 测试时扩展迁移不到 VLM」和「小模型不受益」两个被广泛引用的假设，且精准定位到旧结论的根因——所用模型（SmolVLM2、LLaVA-OneVision）指令遵从差、直接答题忽略思考提示，更像专用 VQA 系统。这提醒社区：负面结论常常是「模型选得不对」而非「方法本身不行」。
- **「视觉前置编码」是可迁移的机制洞察**：注意力早峰晚衰 + KV-cache 丢弃因果验证，共同说明推理链后期几乎不看图。这直接启发一个高价值工程 trick——**约 200 步后可安全丢弃图像 token 的 KV 缓存来省显存/算力**（论文在补充材料给了 GFLOPs 节省），对长链 VLM 推理部署很实用。
- **「按需扩展」的部署哲学**：既然感知任务被算力伤害、推理任务才受益，就该训一个轻量二分类器先判断「这题要不要推理」，只对需要的样本开测试时扩展，能省下大量算力。这把「无脑加算力」变成「条件性加算力」。
- **可靠答案抽取是被低估的实验卫生**：二次请求强制抽答案，既让 300-token 截断实验干净，又排除了「掉分因为格式没对上」的混淆——很多前人负面结论可能就栽在这一步。

## 局限与展望
- **只覆盖零训练一派**：本文明确聚焦零训练（prompting/sampling）的测试时扩展，未涉及 SFT+RL 训出的 thinking model，两派在同等预算下孰优仍未知。
- **闭源模型验证有限**：仅在 GPT-5.2（⚠️ 模型名疑似占位/未来命名，以原文为准）上于 MMStar、RealWorldQA 做了感知掉分的抽查，发现该现象在强闭源模型上「较弱但仍存在」，覆盖不全面。
- **max token 固定 1024、benchmark 集中**：token 预算只对比了 1024 vs 300 两档，粒度较粗；6 个 benchmark 虽多样但仍偏学术，与真实产品分布的差距未量化。
- **「失焦」仍是假说**：虽有注意力+干预支撑，但「overthink→错误假设→累积→幻觉」的完整因果链条尚未做逐环节消融，更细的干预（如只截断特定类型的推理句）值得补。
- **改进思路**：把「是否需要推理」的二分类器真正训出来并端到端评估节省的算力；把 KV-cache 丢弃做成自适应（按注意力衰减自动决定丢弃时机）而非固定步数。

## 相关工作与启发
- **vs CCoT / DCoT / DDCoT（VLM 专用测试时扩展）**：它们分别抽场景图、抽 bounding-box 视觉线索、把识别与推理拆成串行 prompt。本文实验证明这些专用方法在多数模型/任务上不如朴素 CoT，说明「为 VLM 定制」的必要性被高估了——先把通用 LLM 方法用对，往往就够。
- **vs kaya2026efficient（在 SmolVLM2 上报告失败）**：同样研究小 VLM 的测试时扩展但得出相反（失败）结论。本文归因于其模型指令遵从差，强调「有能力的指令遵从模型才受益最大」，是对该负面结论的直接反驳与解释。
- **vs LLM 文献（Lanham2023 等报告小 LLM 不受益、且模型答题前已知答案）**：本文在 LVLM 上观察到相反现象——小模型受益最大、且减 token 会掉分（说明推理链真在驱动答案）。这提示视觉模态可能改变了推理链的作用方式。
- **vs 早期 VLM 推理（NLX-GPT / UniNLX）**：早期把答案和单条推理短语一起生成，本文则是在现代大模型上系统研究「多花推理算力」的效果与内在机制，关注点从「生成解释」转向「用推理换精度并解释其边界」。

## 评分
- 新颖性: ⭐⭐⭐⭐ 不是新方法，但「第一篇系统评测 + 证伪两个流行假设 + 前置编码机制洞察」的组合足够新且有分量。
- 实验充分度: ⭐⭐⭐⭐⭐ 13 模型 × 9 方法 × 6 benchmark，外加 token 预算、注意力干预、LLM-as-judge 三层分析和人工验证，覆盖极广。
- 写作质量: ⭐⭐⭐⭐ 三条 takeaway 清晰、机制分析层层递进；表格信息密度高，个别符号（如注意力索引）需对照才好读。
- 价值: ⭐⭐⭐⭐⭐ 直接给出「小模型+简单 CoT 可替代大模型」「感知任务别加算力」「200 步后可丢图省算力」等可落地的部署指导，对研究和工业界都实用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2026\] E-TTS: A New Embodied Test-Time Scaling Framework for Robotic Manipulation](e-tts_a_new_embodied_test-time_scaling_framework_for_robotic_manipulation.md)
- [\[CVPR 2026\] Scaling Test-Time Robustness of Vision-Language Models via Self-Critical Inference Framework](../../CVPR2026/vlm_reasoning/scaling_test-time_robustness_of_vision-language_models_via_self-critical_inferen.md)
- [\[CVPR 2026\] UniT: Unified Multimodal Chain-of-Thought Test-time Scaling](../../CVPR2026/vlm_reasoning/unit_unified_multimodal_chain-of-thought_test-time_scaling.md)
- [\[ECCV 2026\] GAIA: A Data Flywheel System for Training GUI Test-Time Scaling Critic Models](gaia_a_data_flywheel_system_for_training_gui_test-time_scaling_critic_models.md)
- [\[ECCV 2026\] ScAle: Attention Head Scaling as a Minimal Adapter for Spatial Reasoning in Vision–Language Models](scale_attention_head_scaling_as_a_minimal_adapter_for_spatial_reasoning_in_visio.md)

</div>

<!-- RELATED:END -->
