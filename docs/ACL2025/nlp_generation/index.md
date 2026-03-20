---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✍️ 文本生成

**💬 ACL2025** · 共 **15** 篇

**[An Empirical Study of Many-to-Many Summarization with Large Language Models](an_empirical_study_of_manytomany_summarization.md)**

:   首次系统研究LLM在多对多摘要（M2MS）任务上的表现，整合8个数据集构建涵盖5个领域6种语言的47.8K样本基准，评测18个LLM发现零样本LLM可媲美微调传统模型，指令微调后显著超越，但事实性问题仍是关键瓶颈。

**[Improving the Calibration of Confidence Scores in Text Generation Using the Output Distribution's Characteristics](calibration_confidence_text_gen.md)**

:   针对文本生成中多个有效输出导致传统置信度指标失效的问题，提出两种任务无关的置信度度量——"比率"（头部vs中部概率比）和"尾部稀薄度"（分布尾部薄厚），仅依赖模型输出概率即可改善 BART/Flan-T5 在摘要、翻译、问答任务上的置信度校准。

**[CLEME2.0: Towards Interpretable Evaluation by Disentangling Edits for Grammatical Error Correction](cleme2_gec_evaluation.md)**

:   本文提出 CLEME2.0，一种可解释的 GEC 参考评估指标，通过将编辑解耦为四类（正确纠正 TP、错误纠正 FPne、欠纠正 FN、过纠正 FPun）并结合编辑加权技术，在 GJG15 和 SEEDA 两个人工评判数据集上达到了与人工判断最高相关性的 SOTA 结果。

**[CoCoLex: Confidence-guided Copy-based Decoding for Grounded Legal Text Generation](cocolex_legal_text_gen.md)**

:   提出 CoCoLex，一种无需训练的解码策略，通过置信度引导的动态插值将模型词表分布与上下文复制分布结合，鼓励从检索上下文中直接复制 token，在五个法律文本生成基准上显著提升生成文本对源文档的忠实性。

**[Dehumanizing Machines: Mitigating Anthropomorphic Behaviors in Text Generation Systems](dehumanizing_machines_anthropomorphic.md)**

:   系统研究如何缓解文本生成系统的拟人化行为——编制基于文献和众包的干预手段清单，发展概念框架来表征干预空间、区分干预类型和评估干预效果，为减少用户对 AI 的过度依赖和情感依附提供理论和实证基础。

**[Document-Level Text Generation with Minimum Bayes Risk Decoding using Optimal Transport](doc_level_mbr_optimal_transport.md)**

:   提出 MBR-OT，将最优传输（Wasserstein距离）引入最小贝叶斯风险（MBR）解码，实现用句子级效用函数评估文档级输出质量，在文档级机器翻译、文本简化和密集图像描述任务上显著优于标准 MBR 解码。

**[Multi-document Summarization through Event Relation Graph Reasoning for Framing Bias Mitigation](event_graph_bias_mitigation_summarization.md)**

:   提出基于多文档事件关系图的中立化摘要方法，通过构建包含文内事件关系（时间/因果/子事件/共指）、跨文档事件共指和事件级道德观点的知识图，以图文本化（硬提示）和图提示调优（软提示）两种方式引导 LLM 生成去偏见的中立摘要，在内容保持和偏见消除上均优于基线。

**[FEA-Bench: A Benchmark for Evaluating Repository-Level Code Generation for Feature Implementation](feabench_repo_code_gen.md)**

:   提出 FEA-Bench——首个评估 LLM 在仓库级代码库中实现新特性（Feature Implementation）能力的基准，包含来自 83 个 GitHub 仓库的 1401 个任务实例，每个实例配有单元测试。最强模型 DeepSeek-R1 仅解决约 10% 的任务，揭示了仓库级增量开发对当前 LLM 的巨大挑战。

**[GiFT: Gibbs Fine-Tuning for Code Generation](gift_gibbs_fine_tuning_code_gen.md)**

:   提出 Gibbs Fine-Tuning（GiFT），受 Gibbs 采样启发，通过"代码→描述→代码"的迭代翻译从边际分布而非条件分布中采样自生成代码，结合困惑度引导的长尾数据选择，在 APPS+/MBPP+/CodeInsight 上比标准自训练提升最高 9.8%。

**[Has Machine Translation Evaluation Achieved Human Parity?](mt_eval_human_parity.md)**

:   首次将人类基线引入 WMT Metrics Shared Task 的排名，发现最先进的自动指标经常与人类评估者排名持平甚至更高，但论证了现在声称"人类对等"为时尚早，并讨论了衡量 MT 评估进步的根本困难。

**[Odysseus Navigates the Sirens' Song: Dynamic Focus Decoding for Factual and Diverse Open-Ended Text Generation](odysseus_dynamic_focus_decoding.md)**

:   提出动态聚焦解码（DFD），通过追踪 LLM 各层间分布差异（KL 散度）来识别知识密集型解码步骤，自适应调整温度——知识密集步用低温保事实性，非知识密集步用高温促多样性——在七个数据集上同时提升事实性和多样性。

**[Personality-Guided Code Generation Using Large Language Models](personality_guided_code_gen.md)**

:   让 GPT-4o 为每个编程任务生成适配的 MBTI 人格类型和描述，再让 LLM 以该人格角色扮演程序员生成代码，在 28 个 LLM-数据集组合中 23 个取得 pass rate 提升，最高达 12.9%，且可与 CoT 等策略叠加使用。

**[Tree-of-Evolution: Tree-Structured Instruction Evolution for Code Generation in Large Language Models](tree_of_evolution_code_gen.md)**

:   提出Tree-of-Evolution (ToE)——一种树结构的代码指令合成框架，通过多路径进化和质量驱动优化来克服现有方法（如Code Evol-Instruct/OSS-Instruct）的单向合成和随机生成限制，仅用75K合成数据微调base model即可达到或超越Qwen2.5-Coder-Instruct（数百万样本微调）的性能。

**[What Is That Talk About? A Video-to-Text Summarization Dataset for Scientific Presentations](video_text_summarization.md)**

:   构建了 VISTA 数据集（18,599 个 AI 会议演讲视频-摘要对），首次系统性基准测试科学视频到文本摘要任务，并提出基于计划（plan-based）的框架，通过显式建模摘要结构来提升生成质量和事实一致性。

**[Writing Like the Best: Exemplar-Based Expository Text Generation](writing_like_best_exemplar.md)**

:   定义"基于范例的说明文生成"新任务——给定一篇关于源主题的范例文本，生成关于目标主题的说明文，提出 Recurrent Plan-then-Adapt（RePA）框架，通过逐段模仿规划+检索增强自适应生成+双记忆机制，在 Wikipedia/RoleEE/USNews 三个数据集上显著优于 GPT-4 和 o1 基线。
