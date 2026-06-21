---
title: >-
  [论文解读] WebDS: An End-to-End Benchmark for Web-based Data Science
description: >-
  [多模态VLM] 提出首个端到端 Web 数据科学基准 WebDS（870 个任务，29 个网站，10 个领域），当前最强 Agent（BrowserUse + GPT-4o）仅完成 15% 的任务，而人类达到 90%，揭示了真实数据科学工作流中 Agent 的巨大性能差距。 现实中的数据科学任务涉及复杂的 Web 交互：在…
tags:
  - "多模态VLM"
---

# WebDS: An End-to-End Benchmark for Web-based Data Science

## 元信息
- **会议**: ICLR 2026
- **arXiv**: [2508.01222](https://arxiv.org/abs/2508.01222)
- **代码**: [WebDS Benchmark](https://webdsbenchmark.github.io/)
- **领域**: 多模态大模型 / Web Agent / 数据科学
- **关键词**: web agent, 数据科学, benchmark, 端到端评估, 多步推理

## 一句话总结

提出首个端到端 Web 数据科学基准 WebDS（870 个任务，29 个网站，10 个领域），当前最强 Agent（BrowserUse + GPT-4o）仅完成 15% 的任务，而人类达到 90%，揭示了真实数据科学工作流中 Agent 的巨大性能差距。

## 研究背景与动机

现实中的数据科学任务涉及复杂的 Web 交互：在互联网上寻找合适的数据、从不同位置综合多模态数据、生成汇总分析。然而现有基准存在两个关键缺陷：

**Web Agent 基准**（如 WebVoyager、WebArena）聚焦简单交互（发帖、购物），不要求多样化的工具使用能力和数据分析

**数据科学基准**（如 InfiAgent-DABench、DSBench）集中于静态结构化数据集，不涵盖从数据获取到分析的端到端工作流

**核心矛盾**：真实数据科学工作流通常从浏览 Web 开始，跨多个网站导航和综合信息，但这一关键环节被现有基准忽略。例如，BrowserUse 在 WebVoyager 上达到 80%，但在 WebDS 上仅 15%。

## 方法详解

### 整体框架

WebDS 不是一个模型，而是一个完整刻画"真实数据科学管道"的基准。它先把一项 web 数据科学任务形式化为三段串行映射 $f = f_\alpha \circ f_a \circ f_d$：$f_d$ 在真实网站上浏览、导航并抓取原始数据，$f_a$ 把抓到的多模态数据转换成分析产出（报告、可视化或预测），$f_\alpha$ 再给出一个**可选**的下游动作（如发帖、删除评论），无动作时输出空。现有基准要么只覆盖前半段（纯 web 浏览），要么只覆盖后半段（在静态结构化数据上做分析），而 WebDS 第一次把整条端到端管道一起测。

基准的全部构造都围绕这条管道展开，要回答四个问题：任务从哪来（专家访谈而非研究者臆造）、怎么覆盖真实复杂度（七维属性标注 + 由属性机械推导的难度分级）、用什么环境跑（live 与 dockerized 双轨）、又如何客观评分（二元 + 五级 + 人工兜底的三级协议）。下面四个关键设计就分别对应这四步。

### 关键设计

**1. 专家驱动的任务来源：让任务源自真实工作流而非研究者臆造**

多数 web agent 基准的任务是研究者凭空设计的简单交互（发帖、购物），与真实数据从业者每天做的事相去甚远，测出来的高分并不代表能干真活。WebDS 改为先对 8 名记者、数据科学家和领域专家做访谈，从他们的实际工作里归纳出两类核心任务：一类要产出下游产品（分析报告、可视化图表），另一类要解答关键的分析性问题。再据此从 100 个候选网站中筛出 29 个数据丰富、且数据表征互不重复的网站（CDC、政府数据门户、新闻媒体等），覆盖 10 个高风险领域，由 8 位标注者手工编写并经二次复核，最终得到 870 个任务，数据形态同时含结构化（CSV、表格）与非结构化（文本、图形）。这样基准触及的就是真实工作中"先找对数据、再做跨源综合"的难点，而不是好刷分的玩具任务。

**2. 七维属性标注与能力组合难度分级：把模糊的"难"拆成可定位、可复现的能力维度**

只看一个总成功率，没法说明 agent 究竟卡在哪一步。WebDS 给每个任务打上 7 种属性标签——问答（QA）/ 行动（Action）、单跳 / 多跳、结构化 / 非结构化、是否需 Python·SQL 等外部工具、是否需网站导航、是否跨网站——于是失败能被定位到具体能力短板。按这套标签统计，任务分布为：单跳问答 344、多跳问答 117、单跳行动 97、多跳行动 134、行动加工具 139。难度分级则**不靠主观打分**，而是由任务是否触及"多跳、非文本、行动、工具、多网站"这些硬能力机械推导，保证客观可复现：

$$\text{Difficulty} = \begin{cases} \text{Easy (247)} & \text{不含多跳/非文本/行动/工具，且单网站} \\ \text{Medium (275)} & \text{恰好包含上述一项，且单网站} \\ \text{Hard (348)} & \text{包含两项及以上，或跨多网站} \end{cases}$$

难度直接等于任务所需硬能力的数量，因此 agent 在三档上的得分曲线能直接读出它随复杂度上升的退化情况（论文实测 easy 档的平均得分约为 medium/hard 的 2.5 倍）。

**3. 双轨部署：在真实性与可复现性之间两头兼顾**

真实网站会随时间变页面、改数据，难以复现；可一旦容器化又会丢掉真实 web 那种多层交互与访问限制的复杂度，二者难以兼得。WebDS 因此提供两条评估轨道：WebDS-live 让 agent 直接在真实网站上交互，最大程度保留真实 web 的混乱与多变，并完整录下页面状态、下载产物与动作轨迹以便事后审计；WebDS-dockerized 则把子集网站容器化、冻结其内容与结构，换来确定性执行与严格可复现的对照实验。为防过拟合，基准还拆出 470 个公开验证任务与 400 个私有测试任务，排行榜用私有集的轮换子集评分。

**4. 全轨迹三级评估协议：让端到端的开放产出也能被客观打分**

数据科学任务的产出常是报告或图表这类开放结果，单一对错判定不够用；先前方法（如 WebVoyager）又只看最后 15 张截图给二元结论，看不到中间过程。WebDS 叠加三层评估：对有参考答案的任务做自动二元评估，由 LLM 比对输出与标准答案给出 SUCCESSFUL / UNSUCCESSFUL；对开放任务扩展 WebVoyager 的 LLM-as-Judge，关键改动是不再只看终态，而是把整条轨迹按 $(\text{观测}, \text{动作}, \text{下个观测})$ 三元组逐步总结后给出 1–5 的整数评分并附失败分析；最后用人工验证兜底，对 400 对任务-轨迹做独立评审，自动评分与人工判断达到 93% 一致率，证明这套打分足够可信。

## 实验

### 主要结果

| Agent | 框架 | SR% |
|-------|------|-----|
| GPT-4o + BrowserUse | BrowserUse | 13.2% |
| GPT-4o + AgentOccam | AgentOccam | 4.8% |
| **GPT-5.1 + BrowserUse（live 最佳）** | BrowserUse | **22.2%** |
| **人类基线** | 浏览器 | **90% (±3%)** |

> 同一 GPT-4o + BrowserUse 在 WebVoyager 上达 81.1%，到 WebDS 只剩 13.2%；live 轨道上表现最好的 BrowserUse（GPT-5.1）也仅 22.2%，且多数模型成功率低于 2%。

### 关键发现

1. **巨大的人机差距**：live 轨道最强 agent 仅 22.2%（GPT-4o 配置 13.2%），人类约 90%，差距高达 ~68–77 个百分点
2. **增加模型容量无显著提升**：GPT-4o、GPT-4o-mini 和 Qwen2.5-72B 表现相似
3. **新型失败模式**：
    - **信息锚定失误**：锚定知识与潜在知识矛盾
    - **重复行为**：在多跳任务中陷入循环
    - **走捷径**：跳过必要的数据获取步骤
4. **难度梯度明显**：Agent 在 Easy 任务上得分约为 Medium/Hard 的 2.5 倍
5. **跨基准差距**：WebVoyager 上 81.1% vs WebDS 上 13.2%（同一 Agent）

### 对比 WebVoyager / WebArena

| 特征 | WebVoyager | WebArena | WebDS |
|------|-----------|----------|-------|
| 多跳 | ✗ | ✓ | ✓ |
| 结构化数据 | ✗ | ✗ | ✓ |
| 非结构化数据 | ✗ | ✗ | ✓ |
| 多网站 | ✗ | ✓ | ✓ |
| 工具使用 | ✗ | ✓ | ✓ |
| 端到端数据科学 | ✗ | ✗ | ✓ |

## 亮点

- 首个端到端 Web 数据科学基准，弥合了 Web 交互与数据科学能力之间的鸿沟
- 870 个人工编写的高质量任务，粒度覆盖 7 种属性和 3 种难度
- 双轨设计（live + dockerized）兼顾真实性与可复现性
- 完整轨迹评估 + 细粒度评分，超越简单的二元判定
- 量化了巨大的人机差距，为社区指明方向

## 局限性

- 当前仅覆盖 29 个网站，领域代表性有限
- 容器化部署仅为子集，部分任务依赖 live 网站可能随时间变化
- 人工标注成本高，870 个任务规模可能不足以覆盖所有真实场景
- 评估仍依赖 LLM-as-Judge，对复杂分析报告的质量评判可能不够精确
- 未深入分析不同类型工具使用的能力差异

## 相关工作

- **数据分析基准**：SQuAD、HotpotQA（结构化 QA），InfiAgent-DABench、DSBench（数据科学 agent），Spider 2.0（企业 SQL）
- **Web Agent 基准**：WebArena（功能正确性），WebVoyager（最终截图），Mind2Web（动作序列）
- **端到端工作流**：GAIA（多模态推理），AssistantBench（Web 辅助）— 均不专注数据科学管道

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首个端到端 Web 数据科学基准，问题定义新颖
- **技术深度**: ⭐⭐⭐⭐ — 任务设计严谨，评估体系全面
- **实验充分度**: ⭐⭐⭐⭐ — 9 个 SOTA agent + 人类基线，多维度分析
- **实用价值**: ⭐⭐⭐⭐⭐ — 揭示 Agent 在真实数据科学中的关键不足，指导未来发展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Why Reinforcement Fine-Tuning Preserves Prior Knowledge Better: A Data Perspective](why_reinforcement_fine-tuning_enables_mllms_preserve_prior_knowledge_better_a_da.md)
- [\[AAAI 2026\] FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models](../../AAAI2026/multimodal_vlm/ft-ncfm_an_influence-aware_data_distillation_framework_for_efficient_vla_models.md)
- [\[ICLR 2026\] VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations](visjudge-bench_aesthetics_and_quality_assessment_of_visualizations.md)
- [\[ICCV 2025\] SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency](../../ICCV2025/multimodal_vlm/scan_bootstrapping_contrastive_pre-training_for_data_efficiency.md)
- [\[ACL 2025\] Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach](../../ACL2025/multimodal_vlm/multimodal_coreference_resolution_for_chinese_social_media_dialogues_dataset_and.md)

</div>

<!-- RELATED:END -->
