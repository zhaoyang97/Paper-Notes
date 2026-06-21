---
title: >-
  [论文解读] Do LLM Agents Know How to Ground, Recover, and Assess? Evaluating Epistemic Competence in Information-Seeking Agents
description: >-
  [ICLR 2026][LLM评测][搜索智能体] 提出 SeekBench——首个面向 LLM 搜索智能体的**过程级**评测框架，把"会不会用证据"拆成接地（groundedness）、纠错（recovery）、校准（calibration）三种认知能力并设计可量化指标（RQI / ERF / CE），用 190 条专家标注轨迹校准出一套高一致性标注 schema，再借 LLM-as-judge 把评测扩到 28,493 条轨迹，揭示出只看答案准确率根本看不到的行为缺陷。
tags:
  - "ICLR 2026"
  - "LLM评测"
  - "搜索智能体"
  - "过程级评测"
  - "认知能力"
  - "证据状态"
  - "LLM-as-judge"
---

# Do LLM Agents Know How to Ground, Recover, and Assess? Evaluating Epistemic Competence in Information-Seeking Agents

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=r0L9GwlnzP](https://openreview.net/forum?id=r0L9GwlnzP)  
**代码**: https://github.com/SHAO-Jiaqi757/SeekBench  
**领域**: LLM评测 / 搜索智能体 / Benchmark  
**关键词**: 搜索智能体、过程级评测、认知能力、证据状态、LLM-as-judge

## 一句话总结
提出 SeekBench——首个面向 LLM 搜索智能体的**过程级**评测框架，把"会不会用证据"拆成接地（groundedness）、纠错（recovery）、校准（calibration）三种认知能力并设计可量化指标（RQI / ERF / CE），用 190 条专家标注轨迹校准出一套高一致性标注 schema，再借 LLM-as-judge 把评测扩到 28,493 条轨迹，揭示出只看答案准确率根本看不到的行为缺陷。

## 研究背景与动机

**领域现状**：近来一大批工作用强化学习（RL）训练 LLM 搜索智能体来做开放域问答——智能体反复"识别信息缺口 → 检索外部证据 → 在证据上推理 → 决定下一步动作或给出答案"，靠 RL 隐式学到决策策略。评判这些智能体好坏，几乎清一色用**最终答案**的 exact match / F1。

**现有痛点**：答案对不代表过程对。一个智能体完全可能在"忽略冲突来源、没意识到歧义、证据不足就抢答"的情况下蒙对答案，拿到高 benchmark 分数却毫无可信的推理过程。论文开篇的图 1 就给了一个反例：两个智能体都答出"455,000"，一个是瞎蒙、一个是认真核实，答案级指标对它们一视同仁。

**核心矛盾**：信息检索任务有个区别于代码/数学的本质难点——它**没有客观验证器**。代码能跑、证明能验，但"搜来的一段文本到底支不支持这个结论"没法自动判对错。于是评测要么退化成只看最终答案，要么需要昂贵的人工读轨迹，二者都没法在过程层面、大规模地回答"智能体是否真的在合理地获取、评估、运用知识"。

**本文目标**：把抽象的"认知能力（epistemic competence）"操作化成**可在轨迹上测量**的属性，并且既要标得准（高标注者一致性）、又要扩得开（能自动跑几万条）。

**切入角度**：作者借用社会科学里的内容分析法（Content Analysis）和心理测量学里的构念效度（construct validity）——先观察大量真实轨迹、归纳出可观测的行为标签，再从行为里推断出潜在能力构念，最后把构念翻译成量化指标。这条"可观测特征 → 潜在能力 → 量化指标"的链路保证了指标不是拍脑袋定的。

**核心 idea**：用一个 $E=C+Q$ 的**证据状态**作为统一支点，把接地、纠错、校准三种能力都定义成证据状态上的可测量函数，再用 LLM-as-judge 复刻专家标注，实现过程级、大规模、可解释的智能体评测。

## 方法详解

### 整体框架

SeekBench 把一个搜索智能体的多轮轨迹形式化为 $T=\langle \tau_1,\tau_2,\dots,\tau_T\rangle$，其中非终止轮 $\tau_t=\langle r_t, s_t, e_t\rangle$ 由推理 $r_t$、检索 $s_t$、证据 $e_t$ 组成，终止轮 $\tau_T=\langle r_T, a_T\rangle$ 给出答案 $a_T$。整套框架分三阶段递进搭建：**第一阶段**用内容分析法迭代构造并校准一套标注 schema，把每一步轨迹打上"功能类型 + 质量属性"两个维度的标签，用 190 条专家标注轨迹（1,800+ 步）把候选的 12 个标注字段收敛到 8 个高一致性特征（Cohen's $\kappa>0.8$）；**第二阶段**从标注数据里观察到的三类行为差异（推理是否有据、面对烂结果会不会换策略、答得太早还是太晚），通过潜在构念推断归纳出三种核心认知能力；**第三阶段**把每种能力翻译成一个量化指标（RQI / ERF / CE），全部建立在一个统一的"证据状态"定义上。最后用一个 LLM-as-judge 流水线复刻这套标注，把评测从 190 条手工轨迹扩展到 28,493 条轨迹、283,950 步。

整篇论文的产出不是一个新模型，而是"一套标注规范 + 三个指标 + 一个自动评判器"，所以下面的关键设计围绕"怎么把模糊的认知能力变成能算的数"展开。

### 关键设计

**1. 双维度标注 schema：把每一步既标"在干什么"又标"干得好不好"**

过程级评测的第一道坎是：同样是"推理步"，有的在识别信息缺口、有的在总结已查到的内容、有的在规划下一步搜索，功能完全不同；只打一个"reasoning"标签等于没标。作者因此给 schema 设了两个正交维度：**功能类型**刻画这一步的认知目的（推理步细分为 InformationSynthesis 证据整合、PlanFormation 搜索策略制定、StateAssessment 知识缺口识别；搜索步细分为初始探索、重复、跟进、精炼），**质量属性**刻画这一步的认知质量（如推理是否被证据支撑、证据是否清晰充分）。这套"做什么 × 做得多好"的结构正是后面能把能力拆开测的前提。schema 不是一次定死的：三位专家分三轮独立编码 190 条轨迹，对一致性低（$\kappa<0.5$）的特征要么删（太稀少/太模糊）要么并（概念重叠），最终把 12 个字段压到 8 个；还用 GPT-5 生成对抗性边界样例（如同时含事实断言和规划的推理步）来逼问定义的互斥性，确保标签不会两可。

**2. 证据状态 $E=C+Q$：所有指标共用的统一支点**

三种能力如果各定义各的，指标之间没法对话。作者用一个极简的证据状态把它们串起来。对轨迹 $i$ 的第 $t$ 轮检索到的证据，标注两个二值量：清晰度 $C_{i,t}\in\{0,1\}$（证据是否无歧义、可解读）和充分度 $Q_{i,t}\in\{0,1\}$（证据是否含有足够回答问题的信息，注意这里的"质量"特指充分性而非泛泛的好坏）。证据状态定义为

$$E_{i,t} := C_{i,t}+Q_{i,t}\in\{0,1,2\}$$

$E=0$ 是差证据（既不清晰也不充分），$E=1$ 是部分证据（清晰、充分二者居其一），$E=2$ 是好证据（既清晰又充分）。论文用一个 Black Sabbath 主唱是谁的例子把四种 $(C,Q)$ 组合讲得很具体：检索结果含糊其辞 → $C=0,Q=0$；点题但只给了吉他手 → $C=1,Q=0$；给了两个主唱名字造成歧义 → $C=0,Q=1$；一致地确认是 Ozzy Osbourne → $C=1,Q=1$。接地、纠错、校准三个指标后面全部以 $E_{i,t}$ 为坐标轴来定义，这也是整套框架"自洽"的关键。

**3. 三个能力对应三个量化指标：RQI 测接地、ERF 测纠错、CE 测校准**

这是框架的核心产出，三个指标各自盯住一种认知失败。

*接地（Groundedness）—— 推理质量指数 RQI*：给每个推理步打一个二值接地标签 $G_{i,t}\in\{0,1\}$（其事实内容是否被检索证据支撑），模型级 RQI 就是轨迹级接地分的平均 $\text{RQI}_{\text{model}}:=\mathbb{E}_{i}[\mathbb{E}_{t}[G_{i,t}]]$。更妙的是它可按证据状态分解 $\text{RQI}_i=\sum_{k=0}^{2}P_t(E_{i,t}=k)\cdot\mathbb{E}_t[G_{i,t}\mid E_{i,t}=k]$，从而能区分"接地差是因为没查到好证据"还是"有好证据也不用"；还能按推理类型（IS/PF/SA）拆开看智能体在哪类推理上塌方。

*纠错（Recovery）—— 证据纠错函数 ERF*：先定义恢复事件为智能体首次进入好证据状态、或首次给出正确答案的那一轮 $T_{\text{recover},i}:=\min\{t: E_{i,t}=2 \text{ 或 } correct_i=1\}$，再定义 ERF 为到第 $t$ 轮为止已成功恢复的轨迹比例 $\text{ERF}(t):=\frac{1}{N}\sum_{i}\mathbb{I}(T_{\text{recover},i}\le t)$。曲线越陡说明智能体越快从烂证据里爬出来。由于很多轨迹在恢复前就结束了（右删失数据），论文进一步用 Kaplan–Meier 生存分析来稳健估计不同动作类型（精炼/跟进/重复）的恢复速度。

*校准（Calibration）—— 校准误差 CE*：理想策略是"当且仅当证据好（$E=2$）时才作答"，记 $\pi^*(k):=\mathbb{I}[k=2]$。CE 衡量智能体实际作答行为偏离理想策略多远：

$$\text{CE}_i := \sum_{k=0}^{2} P(E_{i,t}=k)\,\bigl|P(answer_{i,t}=1\mid E_{i,t}=k)-\pi^*(k)\bigr|$$

模型级 CE 再对轨迹取平均。它一举抓住两种相反的失败：$k=0$ 时作答率高 = 过度自信（证据不足就抢答），$k=2$ 时作答率低 = 过度谨慎（好证据在手却不答）。完美校准的智能体 $\text{CE}=0$。

**4. LLM-as-judge 流水线：把专家标注扩到几万条轨迹**

专家标注准但贵，190 条到顶。作者用前面校准好的 schema 做提示词，让 LLM 自动标注轨迹，从而把评测规模拉到 28,493 条轨迹、283,950 步。关键是这个评判器经过了和人类的一致性验证：整体人类标注一致性 $\kappa=0.811$，LLM 评判器里 GPT-5 达 $\kappa=0.754$、GPT-4.1-mini 达 $\kappa=0.731$。再做成本-效益分析后，GPT-4.1-mini 以每条轨迹 \$0.0087、2.48s 的低成本拿到强一致性，被选作大规模评测的主力评判器。正是这一步让"过程级评测"从昂贵的人工活变成了可落地的自动化管线。

### 一个完整示例

拿"Black Sabbath 的主唱是谁"这道题走一遍框架。智能体第一轮检索回来两段含糊文本（"乐队主唱位置换过很多人"），标注为 $C=0,Q=0$，证据状态 $E=0$——差证据。若它此刻就抢答，校准指标会把这一步记为过度自信（$k=0$ 却作答）。一个有纠错能力的智能体会精炼查询（REFINE），第二轮拿到"Ozzy Osbourne 是 Black Sabbath 的原主唱"且多源一致，$C=1,Q=1$，进入 $E=2$——此时恢复事件触发，ERF 在这一轮把该轨迹计入"已恢复"。它在 $E=2$ 时才作答，校准上是正确时机；如果它给出的推理一句句都引自检索证据，接地标签 $G=1$，RQI 也得分。同一条轨迹，三个指标各自从不同侧面读出了它的认知质量——而答案级 F1 只会告诉你"答对了"。

## 实验关键数据

评测对象是基于 Qwen-2.5-7B-Instruct 的多种智能体：Base、Few-shot，以及 RL 训练的 SEARCH-R1、RESEARCH、ASEARCHER、DEEPRESEARCHER，外加 CoT/ReAct 提示策略；数据集为 7 个 QA benchmark（NQ / TriviaQA / PopQA 单跳，HotpotQA / 2Wiki / MusiQue / Bamboogle 多跳），并在 GAIA 上评测带网页浏览的 32B 智能体。总计 28,493 条轨迹、283,950 步。

### 主实验：答案级排名 vs 过程级能力

| 维度 | 关键发现 |
|------|---------|
| 答案级 F1 排名 | ASEARCHER > Search-R1 > RESEARCH > Few-shot ≈ DEEPRESEARCHER > Base |
| 接地 RQI（模型级） | **Few-shot 最高（0.27）**，反超所有 RL 智能体——RL 优化了答案对错，却没培养出"用证据撑住推理"的能力 |
| 接地按推理类型 | 信息整合是相对强项（ASEARCHER 达 0.56）；**计划制定是所有智能体的最大短板（普遍 < 0.2）**；状态评估在 Few-shot 上提升明显（0.28），显示更强的元认知 |
| 纠错 ERF | F1 最高的 ASEARCHER 恢复表现也最好，F1 最低的 DEEPRESEARCHER 恢复最差——纠错能力和最终性能正相关；REFINE / FOLLOW-UP 恢复最快，REPEAT 几乎无效 |

### 校准分析（CE，越低越好）

| 模型 | 过度自信 ↓ | 过度谨慎 ↓ | 校准误差 CE ↓ |
|------|-----------|-----------|--------------|
| Base | 0.631 | 0.030 | 0.329 |
| Few-shot | 0.511 | 0.024 | 0.317 |
| RL-trained | **0.353** | 0.085 | **0.309** |

RL 训练把过度自信作答从 63.1% 压到 35.3%，CE 也最低——RL 确实教会了模型"证据够了才答"。这恰好和接地结论（RL 反而损害推理接地）形成对照，说明 **RL 的影响是按能力分项的**：它改善校准、却没改善接地。

### 关键发现

- **证据质量直接驱动答案正确率**：RL 智能体在好证据（$E=2$）下作答正确率 31.6%，没证据支撑时只有 8.4%——这是证据状态有效性的直接验证。
- **答案级指标会同时低估和高估**：只看 F1 既看不到 Search-R1 的信息整合强项（IS 上 RQI=0.63），也看不到 Base 其实推理不弱。
- **能力互补可被利用**：用一个智能体收集证据、喂给另一个生成答案的"智能体合成"实验中，Search-R1 是最强合成器（平均 +2.61 F1），而 Base 被搭配时获得最高 F1 增益（+2.42）——说明纯准确率评测高估了 RL 训练的收益。框架还能提供推理时反馈信号，不训练就让 ASEARCHER-7B 提升 8.4% F1。

## 亮点与洞察

- **"证据状态"这个支点选得极聪明**：$E=C+Q$ 只用两个二值标签，却把接地/纠错/校准三种看似无关的能力统一到同一坐标系下，三个指标因此能互相对话、还能各自按 $E$ 分解去定位失败根因。这是整篇框架自洽的关键，值得迁移到任何"过程可拆解"的智能体评测里。
- **把社会科学方法论搬进 LLM 评测**：用内容分析法迭代收敛 schema（12→8 字段、删/并低一致性特征）、用构念效度把潜在能力翻译成可测指标，给"主观、无客观验证器"的评测问题提供了一套严谨范式，而不是拍脑袋造指标。
- **最反直觉的"啊哈"**：RL 训练同时**改善校准**又**损害接地**——能力是分项的，不能笼统说"RL 让智能体更好"。这一发现直接解释了为什么答案级指标会误导研发方向。
- **可复用 trick**：用 Kaplan–Meier 生存分析处理"很多轨迹没恢复就结束"的右删失数据，给变长轨迹的恢复速度估计提供了统计上稳健的工具，比直接平均靠谱得多。

## 局限与展望

- **依赖 LLM-as-judge**：尽管 $\kappa>0.73$ 算强一致性，但大规模标注全靠 GPT-4.1-mini，评判器自身的偏差/盲区会被放大进结论；接地、清晰、充分这些判断本身就有主观性，专家间 $\kappa$ 也只是 0.81 而非接近 1。
- **证据状态的二值粒度偏粗**：清晰度和充分度都压成 0/1，$E$ 只有三档，难以刻画"部分相关""轻微歧义"这类中间态；同一题的多文档冲突也只能笼统记为不清晰。
- **评测对象集中在 Qwen-2.5-7B 系**：主结论建立在同一底座的几种智能体上，换底座/换规模是否仍成立（尤其"RL 损害接地"这条）需要更广验证；GAIA 上的 32B 网页智能体只是补充。
- **指标对比需带 caveat**：不同 benchmark 难度、不同轨迹长度预算下的 RQI/ERF/CE 不宜直接比大小，论文也强调框架是"过程级诊断"而非给智能体排座次。
- **改进方向**：把证据状态做成连续/多档、给评判器加多模型投票降偏差、把这套指标变成训练信号（论文已初步展示推理时反馈能 +8.4% F1，下一步可做训练时联合优化接地与校准）。

## 相关工作与启发

- **vs 答案级评测（exact match / F1 / LLM-as-Judge）**：主流评测只看 $a_T$，本文论证这会同时低估和高估智能体能力（Base 推理被低估、RL 收益被高估），转而评测整条轨迹 $T$ 的过程质量。
- **vs 推理忠实度（faithfulness）/ 黄金推理链对齐 / 图依赖建模等过程分析**：这些工作大多针对**最终答案**的推理一致性，或局限于中间步的某一面（如真值追踪、检索分离），没有形式化出"接地/纠错/校准"这组对搜索智能体至关重要的认知能力。本文用精确的数学定义 + 大规模协议补上这一空白。
- **vs 一般智能体 benchmark**：多数 benchmark 给的是任务成功率，本文给的是**为什么成功/失败**的过程级诊断，且配套了可自动化的 LLM 评判管线，能直接指导"用谁收集证据、用谁生成答案"这类系统设计。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个搜索智能体过程级评测框架，证据状态统一三能力的设计干净有力
- 实验充分度: ⭐⭐⭐⭐⭐ 28,493 条轨迹 / 7 个 benchmark / 6 种智能体，含专家一致性、成本分析、合成与反馈实验
- 写作质量: ⭐⭐⭐⭐ 方法论链路（观测→构念→指标）讲得清楚，指标定义严谨；部分结论依赖附录
- 价值: ⭐⭐⭐⭐⭐ 揭示答案级指标的系统性盲区，为可信信息检索智能体的研发提供可落地的诊断工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] ResearchRubrics: A Benchmark of Prompts and Rubrics For Evaluating Deep Research Agents](researchrubrics_a_benchmark_of_prompts_and_rubrics_for_evaluating_deep_research_.md)
- [\[ICLR 2026\] From Reproduction to Replication: Evaluating Research Agents with Progressive Code Masking](from_reproduction_to_replication_evaluating_research_agents_with_progressive_cod.md)
- [\[ICLR 2026\] HackWorld: Evaluating Computer-Use Agents on Exploiting Web Application Vulnerabilities](hackworld_evaluating_computer-use_agents_on_exploiting_web_application_vulnerabi.md)
- [\[ICLR 2026\] Can LLMs Refuse Questions They Do Not Know? Measuring Knowledge-Aware Refusal in Factual Tasks](can_llms_refuse_questions_they_do_not_know_measuring_knowledge-aware_refusal_in_.md)
- [\[ICLR 2026\] CyberGym: Evaluating AI Agents' Real-World Cybersecurity Capabilities at Scale](cybergym_evaluating_ai_agents_real-world_cybersecurity_capabilities_at_scale.md)

</div>

<!-- RELATED:END -->
