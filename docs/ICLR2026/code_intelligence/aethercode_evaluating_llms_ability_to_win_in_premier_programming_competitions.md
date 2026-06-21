---
title: >-
  [论文解读] AetherCode: Evaluating LLMs' Ability to Win In Premier Programming Competitions
description: >-
  [ICLR2026][代码智能][竞赛编程] AetherCode 是首个系统性从 IOI、ICPC 等顶级编程竞赛收集 456 道高难度题目、并用「自动生成 + 67 位专家人工标注」混合方法把每道题的测试用例做到 100% TPR / 100% TNR 的代码推理 benchmark，结果显示即便最强的 o4-mini-high 也只有 35.5% 的 Pass@1，揭穿了「LLM 已征服竞赛编程」的错觉。
tags:
  - "ICLR2026"
  - "代码智能"
  - "竞赛编程"
  - "代码推理"
  - "测试用例质量"
  - "IOI/ICPC"
  - "LLM Benchmark"
---

# AetherCode: Evaluating LLMs' Ability to Win In Premier Programming Competitions

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=lSM6MtjQcM](https://openreview.net/forum?id=lSM6MtjQcM)  
**代码**: 待确认（论文提及有 online leaderboard）  
**领域**: 代码智能 / LLM 评测 / 竞赛编程 Benchmark  
**关键词**: 竞赛编程, 代码推理, 测试用例质量, IOI/ICPC, LLM Benchmark

## 一句话总结
AetherCode 是首个系统性从 IOI、ICPC 等顶级编程竞赛收集 456 道高难度题目、并用「自动生成 + 67 位专家人工标注」混合方法把每道题的测试用例做到 100% TPR / 100% TNR 的代码推理 benchmark，结果显示即便最强的 o4-mini-high 也只有 35.5% 的 Pass@1，揭穿了「LLM 已征服竞赛编程」的错觉。

## 研究背景与动机
**领域现状**：竞赛编程被普遍当作衡量 LLM 推理与编码能力的关键标尺。近一两年模型在 MBPP、HumanEval 上的 Pass@1 已超过 90%，在 LiveCodeBench 上也超过 80%，于是业界普遍认为「竞赛编程已被攻克」。

**现有痛点**：作者指出这种乐观判断主要源于现有 benchmark 本身两个硬伤。其一是**难度与覆盖面不足**——HumanEval/MBPP 都是排序、翻转列表这类基础题；而所谓「竞赛级」的 LiveCodeBench、CodeELO、LiveCodeBench Pro 几乎只取自 LeetCode、AtCoder、CodeForces。LeetCode 题偏简单且往往只需实现单个函数；CodeForces 一场比赛 2-3 小时内只有 5-7 题，限制了出题人的设计空间，缺少需要大规模复杂实现的题目。其二是**测试用例质量带来的评测偏差**——一套不完整的测试集会漏判错误提交（尤其是边界处理出错或在极端数据下超时的解法）。HumanEval/MBPP 只有少量手写用例；CodeContests、EvalPlus 用随机变异这类朴素方法生成；甚至有研究发现 CodeContests 里大量用例违反题目约束，把正确解都判错了。

**核心矛盾**：要做出区分度高的 benchmark，必须同时解决「题目够难够广」和「判题够准够全」两件事；而这两件事都极度依赖竞赛领域的专家经验，是过去自动化流水线绕不过去的坎。有些工作（CodeELO、LiveCodeBench Pro）试图直接调用 CodeForces 官方判题服务来「白嫖」高质量测试用例，但这既有合规风险（CodeForces 明令禁止爬其判题接口），又受提交频率限制，无法灵活实验。

**本文目标**：构建一个**开源、自包含、测试用例高质量**的竞赛编程 benchmark，让 LLM 与顶尖人类选手之间的真实差距重新暴露出来。

**切入角度**：与其从在线题库抓题，不如直接面向 IOI、ICPC 这类全球顶级线下竞赛——它们题目更难、设计空间更大、覆盖的算法门类更全；并且把判题质量当作 benchmark 的一等公民来对待，用「测试集当二分类器」的视角去量化它好不好。

**核心 idea**：从顶级竞赛收题 + 用「生成-验证 Agent + 67 位专家」混合管线把测试用例打磨到零假阳零假阴，做出一个真正区分得开模型的代码推理标尺。

## 方法详解

### 整体框架
AetherCode 的工作本质是一条**数据集构建（curation）流水线**，而非一个模型，目标是产出一套高难度题目 + 高质量判题的 benchmark。整条管线分三大环节（对应论文 Fig.1）：(a) **题面处理**——把从顶级竞赛收集来的 PDF 题面转成 Markdown+LaTeX 并人工校对；(b) **题目分类**——由竞赛专家给每题打上算法类型、难度、时间/主办方等多维标签；(c) **测试用例构建**——用 Generator-Validator(G-V) Agent 自动生成，再叠加专家人工标注与审计，最终用收集来的人类提交验证质量。最终产物 AetherCode v1 含 456 道题（76 道 OI、380 道 ICPC），平均每题 47.15 个测试用例，覆盖 10 个一级算法门类、144 个二级标签。

### 关键设计

**1. 顶级竞赛选题：用 IOI/ICPC 而非在线题库撑开难度与覆盖面**

针对「现有 benchmark 题太简单、来源太窄」的痛点，AetherCode 是首个系统性从全球顶级竞赛收题的 benchmark。题源分两大系列：面向中学生的 OI 系列（以 IOI 为顶点，含中国 NOI、美国 USACO 等国家级赛事，选手 5 小时独立解 3 题、主要用 C++）和面向大学生的 ICPC 系列（3 人一队、一台机、5 小时解 10-13 题，还纳入了 CCPC 等大型赛事）。这些线下竞赛给出题人留足了设计空间，因而天然包含需要大规模复杂实现、深度算法推理的题目，难度被标为 ★★★，与 APPS、CodeContests 同级但题目更新且更聚焦顶级赛事。对每道题，作者不仅收题面，还配套收集了**超过 30,000 份人类解法**（每题至少 5 份正确、20 份错误），这批解法是后续衡量测试用例质量的关键素材。

**2. 多维分类体系：让评测能按难度、算法门类、时间切片做细粒度分析**

光收题不够，作者给每题打上一套多维标签，使 benchmark 能做细粒度诊断而非只给一个总分。**难度**分 Easy/Medium/Hard/Extreme 四档，关键在于这个难度**完全从人类视角判定**而非按 LLM 表现——具体做法是同一场比赛内按「成功解出该题的人数」排序，跨比赛则靠专家评定，最后把整体难度排名三等分成 Easy/Medium/Hard；而**没有任何人类选手在赛中解出**的题被单列为 Extreme。这样设计是为了研究「LLM 眼中的难度」和「人类眼中的难度」有何差异。此外还标注了时间/主办方/赛事范围（区域级/国家级/世界级）等**时序与情境维度**（其中比赛日期用于去污染 decontamination），并排除依赖图像输入的题、显式标记需要 special judge 的题。**算法分类**则是 10 个一级门类（算法基础、搜索、动态规划、字符串、数学、数据结构、图论、计算几何、常用技巧、树上问题）+ 144 个二级标签的层次体系，一题可属多类以反映其跨学科性。

**3. 测试集即二分类器：用 TPR/TNR 把「判题质量」从拍脑袋变成可度量**

这是全文最有方法论价值的一点。过去衡量测试用例质量只看数量（默认越多越好），但数量不等于质量——老数据集里大量用例违反约束，随机堆数据也覆盖不到边界。作者转而把**一道题的整套测试集看成一个二分类器**：它要把正确解和错误解分开。于是用收集来的大量正确/错误提交去评测这个分类器，定义两个核心指标：

$$\text{TPR} = \frac{\text{通过的正确解数}}{\text{正确解总数}}, \qquad \text{TNR} = \frac{\text{被拒的错误解数}}{\text{错误解总数}}$$

其中 TPR 衡量测试用例的**正确性**（高 TPR 表示不会把正确解误杀），TNR 衡量测试用例的**全面性/覆盖度**（即能否「hack 掉」错误解）。这个视角把模糊的「用例够不够好」变成两个可直接计算的数字，也直接定义了后续管线要冲击的目标：100% TPR + 100% TNR。

**4. G-V Agent + 专家标注的混合判题构建：从 89.9% TNR 补到 100%**

有了量化目标，作者用**混合管线**把测试用例做到极致。第一步是 **Generator-Validator(G-V) Agent 自动构建**：这是个双 agent 系统，generator 写一个测试数据生成器程序产出随机数据和各类边界数据，validator 写一个校验器程序确保生成的数据符合题目约束。为防 validator 自身出错，还加了人工 in-the-loop 复核校验器程序。G-V agent 单独能达到 89.9% 的 TNR，叠加对 validator 的人工核验后 TPR 达 100%。但自动阶段自己拿不到 100% TNR，于是第二步引入**专家标注**：招募 67 位竞赛专家（多数 Codeforces rating 2000+，少数超过 2600、为 International Grandmaster），让他们专门针对收集到的各类错误解构造能 hack 掉它们的定向测试用例，与自动生成的合并。对于错误解少于 50 份、单看 TNR 不足以保证鲁棒性的题，再由一支**精英审计队**（每人至少 3 枚 ICPC 金牌、2 年以上出题经验）做人工质检，补缺失边界并额外编写错误/低效解来反向验证用例覆盖度；对接受多解的题则提供并审校 special judge。最终在收集的解法集上达到 **100% TPR + 100% TNR**——据作者所知是首个达到此标准的 benchmark。

## 实验关键数据

评测覆盖 11 个推理模型和 6 个非推理模型，所有模型最大输出 32,768 token，每题采样 4 次取平均。

### 主实验：模型整体表现（Pass@1，单位 %）

| 模型 | Easy | Medium | Hard | Extreme | Pass@1 | Pass@4 |
|------|------|--------|------|---------|--------|--------|
| o4-mini-high（推理） | 65.3 | 32.1 | 8.0 | 3.8 | 35.5 | 46.6 |
| Gemini-2.5-Pro（推理） | 60.1 | 28.6 | 8.5 | 2.5 | 32.7 | 46.0 |
| Seed-1.6-Thinking | 53.9 | 20.2 | 4.7 | 0 | 26.6 | 38.5 |
| DeepSeek-R1-0528 | 46.2 | 16.0 | 3.8 | 0 | 22.3 | 32.4 |
| Claude-4-Opus-thinking | 30.0 | 5.2 | 1.0 | 0 | 12.4 | 18.2 |
| GPT-4.1（非推理） | 23.9 | 5.7 | 1.1 | 0 | 10.5 | 15.3 |
| GPT-4o（非推理） | 11.6 | 1.0 | 0.2 | 0 | 4.4 | 7.0 |

### 分算法门类表现（Pass@1，节选）

| 模型 | 基础 | 字符串 | 动态规划 | 数学 | 计算几何 | 树上问题 |
|------|------|--------|----------|------|----------|----------|
| o4-mini-high | 38.1 | 35.6 | 27.7 | 31.8 | 27.1 | 7.3 |
| Gemini-2.5-Pro | 36.1 | 29.8 | 24.6 | 31.5 | 18.1 | 7.3 |
| Qwen3-32B | 19.7 | 18.3 | 10.9 | 14.1 | 6.9 | 0 |

### 关键发现
- **模型间分层明显，benchmark 区分度高**：o4-mini-high 与 Gemini-2.5-Pro 构成断层式的第一梯队，也是仅有的两个能碰「Extreme（无人类解出）」题的模型；在各难度档上都对其他模型保持显著领先，说明 AetherCode 区分度强。即便最强模型的总 Pass@1 也只有 35.5%，且 Hard 档骤降到 8%、Extreme 几乎归零，印证了「LLM 远未征服竞赛编程」的核心论点。
- **推理模型全面碾压非推理模型**：Qwen3-32B 这类推理模型参数更少却胜过多个非推理模型；更关键的是，非推理模型即便采样到 Pass@4 仍追不上推理模型的水平——说明对竞赛编程这种复杂任务，非推理模型的解空间探索能力受限，靠多采样也难找到正解，弱模型尤甚。
- **顶级模型探索潜力更大**：从 Pass@1 到 Pass@4，o4-mini-high 提升 11.1%（35.5→46.6）、Gemini-2.5-Pro 提升 13.3%（32.7→46.0），而较弱的 Qwen3-32B 只涨 7.6%——多次采样对强模型增益更大，反映其能生成更多样、更高质量的候选解。
- **抽象题型是共同短板**：所有模型在「基础算法」「字符串」这类模式化任务上普遍较好，但在「计算几何」「树上问题」这类高度抽象题型上集体掉链子（树上问题多数模型只有个位数甚至 0）；非推理模型的瓶颈还进一步蔓延到动态规划、数学等需要深度逻辑的领域。GPT-4.1 虽是非推理模型里总分最高，但数学题明显偏弱。

## 亮点与洞察
- **「测试集即二分类器 + TPR/TNR」是可迁移的方法论**：把判题质量从「用例数量」这个伪指标，转成「用真实正确/错误解去测分类性能」，给整个代码评测领域提供了一套可量化、可复现的质量审计框架，任何 benchmark 都能拿来自检。
- **用人类视角而非模型表现定义难度**，并单列 Extreme 档，使 benchmark 能直接对比「人觉得难」和「模型觉得难」的错位，比按模型分数自动分档信息量大得多。
- **30,000+ 份真实人类解法既是评测素材又是质量保障**：错误解被当成「攻击样本」来检验测试集覆盖度，把人类社区的对抗经验注入了 benchmark 构建，这种「拿错误解反向打磨判题」的思路很巧妙。
- **诚实地揭穿虚高**：在多数榜单都已饱和（90%+）的当下，本文用一个新标尺把 SOTA 拉回 35%，提醒社区现有「竞赛级」评测的天花板早已不够用。

## 局限与展望
- **规模与时间覆盖有限**：v1 仅 456 题，且集中在 2024（400 题）与 2025（56 题），OI 题只有 76 道、偏向 ICPC；长期看需要持续更新以对抗数据污染并扩大题量。
- **判题的 100% TPR/TNR 是相对于「已收集解法集」而言**：零假阳/假阴是在现有正确/错误解语料上度量的，面对未来全新的、更刁钻的错误解，覆盖度未必仍是 100%，需要持续补充对抗样本。
- **重度依赖专家人力**：67 位高 rating 专家 + 金牌审计队的混合管线质量很高，但成本与可扩展性是隐忧，难以快速推广到其他领域或低资源语种竞赛。
- **评测设定较单一**：统一 32,768 token 上限、纯文本输入输出、排除图像题，未覆盖交互式题、需要工具/编译反馈的 agentic 解题等更接近真实竞赛的场景。

## 相关工作与启发
- **vs HumanEval / MBPP**: 它们是基础编码任务、手写少量用例，SOTA 已 90%+ 饱和；AetherCode 取自顶级竞赛、难度 ★★★、用例经混合管线打磨，把模型分数拉回 35% 量级，区分度天差地别。
- **vs LiveCodeBench / CodeELO / LiveCodeBench Pro**: 它们题源局限于 LeetCode/AtCoder/CodeForces，且 CodeELO、LCB Pro 直接调 CodeForces 判题服务（有合规风险、受频率限制）；AetherCode 来源更广（IOI/ICPC/CCPC）、且**自包含**高质量开源测试用例，无需依赖外部判题服务。
- **vs CodeContests / EvalPlus**: 它们用随机变异生成用例、被发现存在约束违反等正确性问题；AetherCode 用「测试集即分类器」的 TPR/TNR 框架直接量化并冲到 100%，从根上解决判题偏差。

## 评分
- 新颖性: ⭐⭐⭐⭐ 「测试集即二分类器」的质量度量框架 + 首个系统采集 IOI/ICPC 的 benchmark，方法论扎实但单点创新有限
- 实验充分度: ⭐⭐⭐⭐⭐ 17 个模型 × 难度 × 算法门类 × Pass@N 的全矩阵评测，诊断维度丰富
- 写作质量: ⭐⭐⭐⭐ 论证链清晰、问题定位精准，benchmark 构建细节交代充分
- 价值: ⭐⭐⭐⭐⭐ 给已饱和的代码评测提供了高区分度新标尺，TPR/TNR 框架可被广泛复用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Can LLMs Compress (and Decompress)? Evaluating Code Understanding and Execution via Invertibility](../../ACL2026/code_intelligence/can_llms_compress_and_decompress_evaluating_code_understanding_and_execution_via.md)
- [\[ACL 2026\] CodeWiki: Evaluating AI's Ability to Generate Holistic Documentation for Large-Scale Codebases](../../ACL2026/code_intelligence/codewiki_evaluating_ai39s_ability_to_generate_holistic_documentation_for_large-s.md)
- [\[ACL 2025\] TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs](../../ACL2025/code_intelligence/texpert_a_multi-level_benchmark_for_evaluating_latex_code_generation_by_llms.md)
- [\[ICLR 2026\] SpotIt: Evaluating Text-to-SQL Evaluation with Formal Verification](spotit_evaluating_text-to-sql_evaluation_with_formal_verification.md)
- [\[ICLR 2026\] Multi-LCB: Extending LiveCodeBench to Multiple Programming Languages](multi-lcb_extending_livecodebench_to_multiple_programming_languages.md)

</div>

<!-- RELATED:END -->
