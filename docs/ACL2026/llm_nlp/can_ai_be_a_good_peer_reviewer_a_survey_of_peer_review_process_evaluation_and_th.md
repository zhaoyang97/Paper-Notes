---
title: >-
  [论文解读] Can AI Be a Good Peer Reviewer? A Survey of Peer Review Process, Evaluation, and the Future
description: >-
  [ACL 2026][LLM 其他][peer review] 作者系统综述了 LLM 时代 AI 辅助 peer review 全流程的方法：把"review 生成"分为 fine-tuning / agent / RL / 生成增强 四大范式，把"after-review"分为 rebuttal / meta-review / paper revision 三类，再给出"human / reference-based / LLM-based / aspect-oriented"四象限评测分类法，最后从 novelty、自动评测、跨域、多模态、伦理 6 个方向讨论未来。
tags:
  - "ACL 2026"
  - "LLM 其他"
  - "peer review"
  - "LLM agent"
  - "RL"
  - "评测"
  - "AI4Research"
---

# Can AI Be a Good Peer Reviewer? A Survey of Peer Review Process, Evaluation, and the Future

**会议**: ACL 2026  
**arXiv**: [2604.27924](https://arxiv.org/abs/2604.27924)  
**代码**: https://github.com/formula12/Awesome-Peer-Review (有, 论文清单)  
**领域**: LLM NLP / 学术写作 / 综述  
**关键词**: peer review, LLM agent, RL, 评测, AI4Research

## 一句话总结
作者系统综述了 LLM 时代 AI 辅助 peer review 全流程的方法：把"review 生成"分为 fine-tuning / agent / RL / 生成增强 四大范式，把"after-review"分为 rebuttal / meta-review / paper revision 三类，再给出"human / reference-based / LLM-based / aspect-oriented"四象限评测分类法，最后从 novelty、自动评测、跨域、多模态、伦理 6 个方向讨论未来。

## 研究背景与动机

**领域现状**：peer review 已经成为 AI4Research 中最活跃的子方向之一——从 2018 PeerRead 数据集起步，到 2023 GPT-4 端到端写完整 review，再到 2024–2026 的 multi-agent panel (MARG、AgentReview、DeepReview) 和 RL-aligned 方法 (Remor、CycleResearcher、ReviewRL)，几年内方法谱爆炸。

**现有痛点**：(1) 已有综述要么把 peer review 当成 AI4Research 的一个小章节（覆盖太浅，如 Chen et al. 2025），要么没跟上最新一波 agent/RL 浪潮 (Zhuang et al. 2025)；(2) 评测侧极度割裂——human / ROUGE / LLM-judge / aspect-level 四套体系互不衔接，缺乏统一对比；(3) "after-review"任务（rebuttal、meta-review、revision）相比 review generation 关注度低，缺乏系统梳理；(4) 学术伦理风险（affiliation bias、AI-modified content estimated at 6.5-16.9%）讨论分散。

**核心矛盾**：peer review 是多阶段、subjective、低数据量任务——既要靠 LLM 提速，又要防止 LLM 引入偏见、伪洞察、shallow critique。社区缺少一个把"方法 + 评测 + 伦理"三轴打通的图谱。

**本文目标**：(1) 给出一个面向 AI agent 视角的 peer review 全流程方法分类；(2) 系统化 4 类评测方法及其 pros/cons；(3) 总结跨年代 (pre-2023 vs post-2023) 数据集差异；(4) 列出 6 个明确的未来方向，特别是"超越 review generation 本身"。

**切入角度**：把 peer review 切成两个轴——纵轴是流程阶段（review / rebuttal / meta-review / revision），横轴是方法范式（foundation → fine-tuning → agent → RL → enhancement）；评测沿"是否用 reference / 是否调用 LLM judge"分四象限。

**核心 idea**：用统一 taxonomy（图 1）把 30+ 个 system、20+ 个 dataset 织成一张可检索的网，方便研究者快速定位"我做的是哪一类、跟谁对比"。

## 方法详解

### 整体框架
这篇综述用"流程纵轴 × 方法横轴"两条线索把 AI 辅助 peer review 的散乱文献织成一张可检索的图谱（图 1）。纵轴是任务阶段——从 review generation 走到 after-review（rebuttal / meta-review / paper revision）；横轴是方法范式——foundation → fine-tuning → agent → RL → enhancement 的演进谱系。在此之上再叠一条评测轴，沿"是否用 reference / 是否调用 LLM judge"把 30+ 个 system、20+ 个 dataset 的评价方法切成四象限，最后落到 6 个明确的未来方向。读者由此能快速定位"我做的是哪一类、该跟谁对比、用什么指标评"。

### 关键设计

**1. Review Generation 的 5 范式分类：把"模型直接写 review"收敛成一条有内在演进逻辑的谱系**

以往文献交叉引用混乱，常把"agent"和"RL"混为一谈，作者显式拆出五个范式并让每一步都解决前一步的痛点。Foundation approaches（pre-2023，PeerRead/NLPeer/MOPRD）只做 accept 预测、score 回归、multi-doc summary 等子任务；Fine-tuning（OpenReviewer-Llama8B、REVIEWER2 两阶段、LimGen）针对"zero-shot 太正面"和格式不符问题。Agent-based 进一步分两支——task decomposition（MARG leader-worker / SWIF²T 四 agent / DeepReview 三阶段 / MAMORX 多模态 / DIAGPaper 弱点诊断）是工具化地写更好的 review，process simulation（AgentReview / ReviewMT 把 review 当多轮对话）则是研究 panel 动态本身。RL 范式按 reward 设计细分（Remor 用 GRPO + Human-aligned Peer Review Reward 解决 shallow critique、CycleResearcher 让 review-as-reward 构成 research-review-refine 闭环、ReviewRL 用 composite reward 同时优化质量与一致性、REM-CTX 用辅助上下文做 correspondence-aware reward）。Enhancement 再分 RAG（ReviewRobot 三 KG、novelty 检索 + 重排序）、iterative refinement（ReviewEval 自/外双循环、RbtAct/GoodPoint/ActReview 用 rebuttal 当监督）与 structure control（TreeReview 动态问题树、AutoRev 文档图、RevGAN style 控制）。拆开后，"想做 agent 选哪支、想做 RL 选哪类 reward"一目了然。

**2. 四类评测方法 + pros/cons 矩阵：解决"评测 review 系统该用什么指标"的长期混乱**

过去的工作要么只跑一种评测（被另一类方法否掉），要么跑全套却不交代为什么，作者用一张 4×2 的 pros/cons 矩阵（Table 2）让选择有据可依。Human-centric（Robertson 2023 GPT-4 pilot、Liang 2023 N=308 user study、Reviewer Arena pairwise）最直接但 expensive 且 subjective；Reference-based（ROUGE / BERTScore / hit rate / MCQ accuracy）scalable 但 shallow，会惩罚措辞不同却有效的反馈；LLM-based（多 judge ensemble、unsupervised judge）可扩展灵活，却带 position/verbosity bias 且依赖 prompt；Aspect-oriented（ReviewCritique 23 类细粒度错误标注、STRICTA 把 review 拆成 reasoning step graph、focus-level 评估、adversarial review injection）细粒度可诊断但 annotation 成本高、多维 profile 难直接横比。矩阵的价值在于把"想发现具体失败模式就用 aspect-oriented、想快跑 ablation 就用 reference-based"这类决策显式化。

**3. 跨年代数据集对比 + after-review 三类增强机制：把"review 之外"的任务和数据 mapped 到具体子任务**

作者明确指出 after-review 比 review generation 更 challenging（要处理 argument/response 的动态博弈），但数据集和 benchmark 远不充足，于是这一节本身就充当"做哪个 sub-task"的导航。Rebuttal generation 从单轮（Cheng 2020 APE 抽 argument pair、Purkayastha JITSUPEER 当 attitude-root）演进到多轮（ReviewMT、Re2 把 rebuttal 当对话），再到"verify-then-write"的 evidence 框架（DRPG、Paper2Rebuttal），近期还加入 author-in-the-loop 信号（DEFEND）。Meta-review 从 MetaGen 的 extract-then-write，到 MReD 加 sentence functional label，到 PeerSum 引入 RAMMER 处理 review-rebuttal 层级，到 ORSUM 跨 venue 多阶段 introspection，最近 Purkayastha 2026 把它当 document-grounded dialogue。Paper revision from reviews 则由 ARIES + CASIMIR + arXivEdits 三项工作奠基，是目前最 underexplored 的子方向。配合跨年代数据集对比（pre-2023 vs post-2023，Table 7），这一节把"任务—数据"的空白点清晰标注出来。

### 损失函数 / 训练策略
综述本身不做训练，此处整理文中代表性的训练范式以便横向参照。SFT 路线以 OpenReviewer（79k expert reviews 微调 Llama-8B）和 REVIEWER2（两阶段：aspect prompt → review text）为代表；RL 路线包括 Remor（GRPO + 多目标奖励：criticism / relevance / actionable suggestion）、CycleResearcher（SimPO + dual-agent）、ReviewRL（rule-based composite reward + retrieval augmentation）；iterative refinement 则有 RbtAct（用 rebuttal 当 implicit supervision）和 ActReview（rubric-guided RL 把"review-to-revision"形式化）。

## 实验关键数据

### 主实验
作者总结的方法对比表（节选关键 system，Table 1）：

| 方法 | 范式 | 数据集 | 关键贡献 |
|------|------|------|------|
| PeerRead (NAACL 2018) | Foundation | PeerRead | 第一个大规模 paper+review 数据集 |
| OpenReviewer (NAACL 2025) | Fine-tuning | 79k expert reviews | Llama-8B 微调，捕获多种 critique pattern |
| REVIEWER2 (2024) | Fine-tuning | 27k papers / 99k reviews | 两阶段 aspect prompt → review |
| MARG | Agent (Dec.) | – | leader-worker，iterative refinement |
| DeepReview (ACL 2025) | Agent (Dec.) | DeepReview-13K | 3 阶段：novelty / multi-dim / reliability |
| AgentReview (EMNLP 2024) | Agent (Sim.) | – | 模拟 panel，研究 authority bias |
| ReviewMT | Agent (Sim.) | ReviewMT | rebuttal 当 dialogue，SFT > zero-shot |
| Remor | RL (GRPO) | PeerRT | 多目标 HPRR reward |
| CycleResearcher (ICLR 2025) | RL (SimPO) | Review-5k + Research-14k | research-review-refine 闭环 |
| ReviewRL (EMNLP 2025) | RL (rule) | ICLR 2025 papers | composite-reward + grounded |
| ReviewRobot (INLG 2020) | RAG (KG) | – | 三 KG：paper / cited / background |
| TreeReview (EMNLP 2025) | Structure | venue benchmark | 动态问题树 |

### 关键数据点（survey 引用的实证结论）
- Robertson 2023 pilot：GPT-4 review 平均 helpfulness 和人类一致，但 variance 更高。
- Liang 2023 N=308：用户认为 LLM 反馈"有价值"。
- Liang 2024 corpus-level：6.5%-16.9% 顶级 AI 会议 review 文本可能被 LLM 大幅修改，截止日期前激增。
- von Wedel 2024：LLM 在 single-blinded 时偏好高 rank 机构作者（affiliation bias）。
- ReviewCritique：23 种细粒度错误标注（"Unstated Statement"、"Missing Reference" 等）。

### 关键发现
- **方法演进的清晰轨迹**：foundation (sub-task) → fine-tuning (full review) → agent (decomposed roles) → RL (alignment-aware) → enhancement (RAG + iterative + structure)；每一步都解决前一步的限制（zero-shot 太正面 → fine-tune；fine-tune 太单线 → agent；agent 缺 alignment → RL；RL 缺 grounding → enhancement）。
- **评测瓶颈在 LLM-as-judge 的 bias**：position/verbosity bias 让 single-judge 不可靠，至少 2 judge majority vote 才稳。
- **数据集 NLP/ML 占绝对主导**：PeerRead、NLPeer、ReviewMT 均来自 NLP/ML 会议；MOPRD 是少数多学科尝试，跨域泛化未充分验证。
- **after-review 是最 underexplored 区域**：rebuttal/meta-review/revision 任务数据集稀少，但实际 utility 更高（rebuttal 直接帮作者、revision 直接改论文）。
- **伦理风险已实证存在**：Liang 2024 量化估计 + von Wedel 2024 偏见证据，社区已无法回避"LLM-in-review 是否合规"的问题。

## 亮点与洞察
- 把"agent-based"显式拆成 task decomposition vs process simulation 两支非常清晰——前者是工具化（生成更好的 review），后者是研究化（理解 panel 动态本身）；以前文献混在一起讨论很容易踩坑。
- "evaluation 4 象限" Table 2 是这篇 survey 最有复用价值的产出——比单纯列方法更有持续指导意义；建议未来论文写 evaluation section 直接引用这张表来 justify 自己的选择。
- 把 RL 范式按 reward 设计分类（multi-objective HPRR / review-as-reward / composite rule-based / context-aware）也很实用——后续做 RL-based review 的人可以快速定位"我的 reward 属于哪一类、有哪些已知坑"。
- §5 提出的 "Beyond Review Generation"（rebuttal/meta-review/revision）方向直接命名为"underdeveloped"——这种 explicit prioritization 在 survey 里很少见，对资源分配有帮助。
- 把 SchNovel / NovBench / OpenReviewer / Reviewer2 等关于"novelty 评估"的工作单独列为 future direction 1，提示了"LLM 能写流畅 review 但判不准真正的 novelty" 这一持续 gap。

## 局限与展望
- 综述写作时间是 2026 早期，但 peer review 工作发表节奏极快（每月新出几篇），分类法需持续维护（作者也在 limitation 中承认）。
- 大部分讨论集中在 NLP / ML 会议数据，对生物医学 / 自然科学 review 的差异覆盖薄。
- 没有把 commercial review 系统（Elsevier AI、Editor Assistant 等）纳入对比——专有系统的能力对学术开源工作而言是"暗物质"。
- 评测 pros/cons 矩阵 (Table 2) 偏定性，缺少跨 system 的统一定量对比表（如同一组论文上各方法的 ROUGE/judge/aspect score 同时报告）；这对后续 benchmark 建设是 hint 也是缺憾。
- 改进思路：(1) 建一个 unified leaderboard 同时跑 4 类评测；(2) 把 review、rebuttal、meta-review、revision 当端到端 pipeline 评测而非孤立子任务；(3) 系统化地测试 commercial system 和开源 system 的 gap；(4) 跨域（医学、化学）peer review benchmark。

## 相关工作与启发
- **vs Chen et al. 2025 (AI4Research survey)**: 那篇覆盖更广（research idea → writing → review → analysis），peer review 是其中一章；本文专门、且深入到 agent/RL 最新工作，对 reviewing 子领域更适合做 entry point。
- **vs Zhuang et al. 2025**: 同样综述 LLM-for-peer-review，但缺 RL 和 evaluation 系统化；本文在两轴都补足。
- **vs Drozdz & Ladomery 2024 (BJBS)**: 那篇是医学视角，关注政策和伦理；本文是 NLP 视角，关注方法和数据集，互补阅读。
- **启发**：(1) 这种"流程 × 方法范式"双轴分类法可迁移到其他 AI4Science 子领域（如 AI-for-Literature-Review、AI-for-Hypothesis-Generation）；(2) "evaluation 4 象限"模板可推广到任何 long-form generation 的评测综述；(3) explicit 标注 "underexplored" 子任务的写法值得 survey 作者参考——比平铺直叙更能引导社区注意力。

## 评分
- 新颖性: ⭐⭐⭐ 综述本身不追求方法新颖，分类法 (5 范式 + 4 评测 + 流程纵轴) 是组合式新颖；evaluation 4 象限表是最有原创价值的产出。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 30+ system + 20+ dataset，跨 2018-2026 八年；附录给出方法 backbone 表、解析工具表、数据集时间线表，对系统对比帮助很大。
- 写作质量: ⭐⭐⭐⭐ 章节组织清晰，每个范式都配代表系统简介 + 关键贡献；图 1 taxonomy tree 和图 2/3 流程图很直观；术语用"Dec./Sim."标记 agent 类型节省阅读时间。
- 价值: ⭐⭐⭐⭐ 对刚进 AI-for-peer-review 方向的研究者，是最实用的 entry point；公开 Awesome 仓库可持续维护；6 个未来方向（尤其 after-review、novelty 评估、multimodal）能直接转成下一篇论文的 motivation 段。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Stop Automating Peer Review Without Rigorous Evaluation](../../ICML2026/llm_nlp/stop_automating_peer_review_without_rigorous_evaluation.md)
- [\[AAAI 2026\] Position on LLM-Assisted Peer Review: Addressing Reviewer Gap through Mentoring and Feedback](../../AAAI2026/llm_nlp/position_on_llm-assisted_peer_review_addressing_reviewer_gap_through_mentoring_a.md)
- [\[ICML 2026\] Position: The ML Community Must Build an AI-Augmented Peer-Review Ecosystem](../../ICML2026/llm_nlp/position_the_ml_community_must_build_an_ai-augmented_peer-review_ecosystem.md)
- [\[ACL 2026\] Big AI is Accelerating the Metacrisis: What Can We Do?](big_ai_is_accelerating_the_metacrisis_what_can_we_do.md)
- [\[ACL 2026\] From Fallback to Frontline: When Can LLMs be Superior Annotators of Human Perspectives?](from_fallback_to_frontline_when_can_llms_be_superior_annotators_of_human_perspec.md)

</div>

<!-- RELATED:END -->
