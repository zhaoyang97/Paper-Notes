---
title: >-
  [论文解读] Distilled Pretraining: A modern lens of Data, In-Context Learning and Test-Time Scaling
description: >-
  [ICLR2026][预训练][蒸馏预训练] 这篇论文系统拆解了"用蒸馏做预训练（DPT）"在现代 LLM 范式下的得失：发现蒸馏能显著增强测试时缩放（pass@k 多样性），却同时损害上下文学习（削弱归纳头），并用一个 bigram 沙盒证明这两件相反的事其实出自同一机制——蒸馏只对高熵分布有帮助、对低熵确定性映射无益甚至有害，最后给出 token routing 等可落地的预训练设计建议。
tags:
  - "ICLR2026"
  - "预训练"
  - "蒸馏预训练"
  - "测试时缩放"
  - "上下文学习"
  - "归纳头"
  - "bigram 沙盒"
---

# Distilled Pretraining: A modern lens of Data, In-Context Learning and Test-Time Scaling

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=PNm2dl7HcY](https://openreview.net/forum?id=PNm2dl7HcY)  
**代码**: 待确认  
**领域**: LLM预训练 / 知识蒸馏  
**关键词**: 蒸馏预训练, 测试时缩放, 上下文学习, 归纳头, bigram 沙盒

## 一句话总结
这篇论文系统拆解了"用蒸馏做预训练（DPT）"在现代 LLM 范式下的得失：发现蒸馏能显著增强测试时缩放（pass@k 多样性），却同时损害上下文学习（削弱归纳头），并用一个 bigram 沙盒证明这两件相反的事其实出自同一机制——蒸馏只对高熵分布有帮助、对低熵确定性映射无益甚至有害，最后给出 token routing 等可落地的预训练设计建议。

## 研究背景与动机
**领域现状**：知识蒸馏（用教师的软标签训学生）以前主要用在后训练和模型压缩，早期 LLM（GPT-2/3、Llama 1/2）的预训练里几乎不用。但最近一年它在预训练阶段强势回归——Llama-3.2、Gemma-3 都靠预训练蒸馏出小模型，Llama-4-Maverick 干脆整个从 Llama-4-Behemoth 蒸馏而来。背后的现实是：超大模型部署太贵，未来只会当"教师"，真正上线的小模型很可能完全靠蒸馏预训练。

**现有痛点**：尽管蒸馏在预训练里越来越重要，它的"科学"却几乎没人研究。Gemma-3、Llama-3.2 在标准 benchmark 上确实涨点，但这些教师通常比学生见过多得多的数据。于是一个根本问题没人回答：蒸馏的收益到底是教师"额外数据"带来的，还是蒸馏本身有独特好处？等撞上数据墙（教师学生看同样多数据）后蒸馏还灵不灵？

**核心矛盾**：更要命的是，现代 LLM 的能力早已超出"标准语言建模"。测试时缩放（多次采样取最优）和上下文学习（从 prompt 里现学）才是当下前沿，而预训练蒸馏对这两种新范式的影响完全是空白。作者怀疑：让标准 benchmark 涨点的同一套蒸馏，未必对这些新能力同样友好。

**本文目标**：拆成三个子问题——(1) 数据受控（IsoData，教师学生同等数据量）下蒸馏还有没有用；(2) 蒸馏对测试时缩放和上下文学习各自是什么影响；(3) 如果有相反的影响，背后是不是同一个机制。

**切入角度**：作者用"分布熵"这把尺子看蒸馏——软标签的价值在于把概率分散到多个合理答案上，所以它天然只对"有多个合法续写"的高熵情形有信息增量，对"答案唯一"的低熵确定映射则几乎没东西可教。上下文学习靠的归纳头恰恰是低熵复制操作，测试时缩放靠的多样性恰恰是高熵分布——一刀切开。

**核心 idea**：用"高熵 vs 低熵分布"这一个轴，统一解释蒸馏为何同时帮了测试时缩放、害了上下文学习，并据此设计选择性蒸馏（token routing）。

## 方法详解

### 整体框架
这是一篇分析型论文，"方法"是一条层层递进的实证 + 理论调查链，而非一个新模型。整条链是：先做 **IsoData 实验**（教师 8B 训 1T tokens，学生 1B 在同样的 1T tokens 上训，有/无蒸馏）排除"教师多看数据"这个混淆变量，确认蒸馏在标准任务上仍然有用；然后把同一批模型放到**两个现代视角**下测——上下文学习（看归纳头驱动的复制能力）和测试时缩放（看 pass@k 多样性），得到一组看似矛盾的现象：蒸馏伤 ICL、却助 pass@k；接着用一个**可解析的 bigram 沙盒**把现象还原到最小单元，证明高熵行/低熵行的学习差异就是共同根因，并配上样本复杂度命题；最后据此提出 **token routing** 这一缓解手段并验证。读这篇笔记，关键是抓住"高熵 vs 低熵"这条贯穿始终的主线。

### 关键设计

**1. IsoData 蒸馏实验：把"教师额外数据"这个混淆变量摘掉**

现有论文报告的蒸馏收益始终洗不清一个嫌疑：教师见过远比学生多的数据，涨点也许只是数据多。作者直接构造"等数据"对照——先用 1T tokens 训一个 8B 教师，再让 1B 学生在**完全相同的 1T tokens** 上训，一组带蒸馏一组不带。结果在 HellaSwag、NaturalQuestions、MBPP 等标准语言建模任务上，即便数据完全受控，蒸馏学生仍普遍优于从零训练。这条结论很关键：它说明蒸馏在数据受限（撞上数据墙）的未来仍有价值，不是靠教师偷看更多数据，从而为后面"蒸馏到底教了什么"的追问扫清了干扰。

**2. 现代双视角度量：上下文学习（归纳头）与测试时缩放（pass@k）分开看**

作者刻意不只看平均 benchmark，而是把蒸馏放到两个对当代 LLM 至关重要、却方向相反的能力上称重。**上下文学习**这一侧，用三类专测"从上下文复制"的任务——基于上下文的 QA（DROP、RACE）、大海捞针（babilong）、反事实上下文 QA（正确答案与模型参数化知识冲突，逼模型只信上下文）；这些任务的核心是 Olsson et al. 提出的归纳头，即把序列中先前出现的 token 复制到后面的低熵确定映射。**测试时缩放**这一侧，用 pass@k：给模型对每题 $k$ 次尝试，只要有一个对就算赢。pass@1 只要 top-1 排对，pass@k 则要求把概率质量铺到多个合理答案上——它考的是"生成多样性"而非单点精度。两把尺子一起用，作者得到核心反差：随着数据从 125B 缩放到 IsoData 的 1T，蒸馏在 ICL 上的优势逐渐消失、在大海捞针和反事实 QA 上甚至反超变落后；而在 pass@k 上，DPT-50（蒸馏权重 50%）尽管 GSM8K/MATH 的 pass@1 略低，pass@16 却明显更高（GSM 上 28% vs 23%），DPT-90 更是在只用一半数据的情况下 pass@16 全面超过"双倍数据标准预训练（SPT-2x）"——多样性收益约等于多看一倍数据。

**3. bigram 沙盒 + 样本复杂度分析：用一个可解析模型统一两个相反现象**

为了把"为何同一套蒸馏一边帮一边害"说透，作者退到一个最小可解析模型——bigram（一阶马尔可夫链），转移矩阵 $\pi \in \mathbb{R}^{k\times k}$，每一行 $\pi_{i\cdot}$ 是给定当前 token 后下一 token 的分布。按行的熵把行分成两类：高熵行像"I go to"（office/gym/restaurant 各占 1/3），低熵行像"2+3="（答案几乎确定为 5）。实验发现：高熵行上蒸馏学生用更少样本就逼近真分布，低熵行上有无蒸馏没区别。理论上作者给出命题——每行至多 $p$ 个非零项时，蒸馏的样本复杂度 $S_{\text{distill}} = O(k\log k)$，而标准训练 $S_{\text{standard}} \approx \frac{p}{\epsilon^2} S_{\text{distill}}$（$\epsilon$ 是逼近误差上界）。代入即可解释一切：高熵行 $p=O(k)$，标准训练要 $O(k^2\log k)$、蒸馏只要 $O(k\log k)$，蒸馏大占便宜；低熵行 $p$ 为常数，两者都是 $O(k\log k)$，蒸馏无增益。归纳头正是低熵行——把 bigram 改造成带"触发 token"的复制结构：

$$\tilde\pi_{ji} = \begin{cases} \pi_{ji} & i \neq t \\ \mathbb{I}(j=c) & i = t \end{cases}$$

即触发 token $t$ 出现后下一 token 确定是该序列的复制目标 $c$。这种确定映射下，完美教师的软标签就是 one-hot，和 hard label 一模一样、蒸馏无新信息；而真实教师并不完美，会给干扰 token 分一点概率质量，把本该干净的 one-hot 目标"软化"成略带噪声的分布，反而拖慢甚至阻碍归纳头的形成。pass@k 那边则相反：最优 pass@k 要求准确估计底层概率分布（而非只把 top-1 排对就够的 pass@1），蒸馏的软标签恰好在高熵情形下帮学生更好地拟合多峰分布，于是多样性更强、测试时缩放更好。一正一反，根子是同一条"高熵有料、低熵无料且易被噪声污染"的原理。

**4. Token Routing：按教师熵选择性施加蒸馏损失，补回 ICL**

既然 ICL 掉点的根源是"对低熵 token 强行用软标签反而注入噪声"，缓解办法就顺理成章：对低熵位置干脆别用蒸馏损失。回顾蒸馏目标

$$h^\dagger \in \arg\min_{h}\ \frac{1}{n}\Big[(1-\alpha)\sum_{i}\ell\big(y_i,\sigma(h(x_i))\big) + \alpha\sum_{i}\ell\big(s_i,\sigma(h(x_i))\big)\Big],\quad s_i=\sigma\!\big(h_{\text{teacher}}(x_i)/T\big)$$

它在 hard-label 项与 teacher-label 项间用 $\alpha$ 插值。token routing 的做法是：对每条输入先算教师软标签，挑出熵最低的 $x\%$ 位置，**只在这些位置丢掉蒸馏损失项、退回纯 ground-truth 监督**，其余高熵位置照常蒸馏。取 $x=15\%$ 时，大海捞针和反事实 QA 上明显回血、部分追平标准预训练，而标准语言建模任务不掉点——反过来印证了"蒸馏增益来自高熵软标签"。$x=30\%$ 则 ICL 不再有额外收益且标准 benchmark 退化，说明该跳过的就是那一小撮低熵 token。

### 损失函数 / 训练策略
核心目标即上文 Hinton 式软标签蒸馏，权重 $\alpha$ 控制蒸馏强度（论文取 50% 的 DPT-50 与 90% 的 DPT-90）；token routing 是对该损失的**位置级门控**——按教师输出熵把最低 $x\%$ 位置切回 hard-label。主体实验为 1B 学生 / 8B（或 Llama-3.1-8B）教师，数据从 125B 缩放到 1T（IsoData）。

## 实验关键数据

### 主实验

| 视角 | 任务/指标 | 关键现象 |
|------|-----------|----------|
| 标准语言建模（IsoData） | HellaSwag / NaturalQA / MBPP 等平均 | 教师学生同看 1T tokens 时蒸馏仍普遍优于从零训练 |
| 测试时缩放 | GSM8K pass@16 | DPT 27–28% vs SPT 23%，pass@1 相近甚至更低 |
| 测试时缩放 | GSM/MATH/MBPP pass@16 | DPT-90（一半数据）全面超过 SPT-2x（双倍数据） |
| 上下文学习（IsoData） | 大海捞针 / 反事实 QA | 数据缩放到 1T 后蒸馏优势消失，甚至反超变落后 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 标准预训练 SPT | ICL / pass@k 基线 | 不带蒸馏 |
| DPT（vanilla） | ICL↓，pass@k↑ | 全位置蒸馏，呈现核心 trade-off |
| DPT + token routing(15%) | ICL 回血，标准任务不掉 | 低熵 15% token 退回 hard-label |
| DPT + token routing(30%) | ICL 无额外增益、标准 benchmark 退化 | 跳过比例过大 |

### 关键发现
- 蒸馏帮 pass@k、害 ICL，二者源自同一机制：软标签只对高熵分布有信息，对低熵确定映射无益且会被教师噪声污染。
- bigram 样本复杂度命题给出定量解释：高熵行蒸馏从 $O(k^2\log k)$ 降到 $O(k\log k)$，低熵行两者持平。
- token routing 用 15% 这个量级最甜——既补回 ICL 又不伤标准任务，30% 则过犹不及。

## 亮点与洞察
- **用一根"熵"轴统一两个相反现象**：把"蒸馏帮什么、害什么"归结为高熵 vs 低熵分布，是这篇论文最漂亮的洞察——既解释了 pass@k 提升，也解释了归纳头受损，避免了"就事论事"的零散结论。
- **IsoData 设计**：先把"教师多看数据"这个最常见的混淆变量摘干净，后面的因果归因才站得住，是分析型工作值得借鉴的实验卫生习惯。
- **bigram 沙盒可迁移**：把复杂的归纳头/ICL 现象退化到可解析的一阶马尔可夫链 + 触发 token 复制结构，既能跑实验又能证命题，是研究 LLM 机制的好范式。
- **token routing 的工程价值**：按教师熵做位置级门控，几乎零成本就能缓解蒸馏的 ICL 副作用，对真正靠蒸馏预训练小模型的团队是直接可用的建议。

## 局限与展望
- 主体实验规模为 1B 学生 / 8B 教师，是否在更大模型、更长训练上同样成立需进一步验证。
- token routing 是"preliminary"的概念验证，熵阈值/比例的最优选择、与其他数据筛选的协同尚未充分探索。
- bigram 沙盒虽优雅，但与真实 Transformer 归纳头之间仍是"类比"，命题为非正式（informal）版本，严格性以附录证明为准（⚠️ 细节以原文为准）。
- pass@k 作为"多样性/测试时缩放"的代理指标本身有局限，RLVR 等下游是否同样受益论文只做了定性讨论。

## 相关工作与启发
- **vs Llama-3.2 / Gemma-3 的预训练蒸馏**：它们都用 Hinton 式加权软标签且教师远多于学生数据，只报标准 benchmark 涨点；本文先在 IsoData 下证明蒸馏本身有效，再揭示它对 ICL/测试时缩放的相反作用，把"涨点"拆成了机制。
- **vs Cha & Cho (2025) 的合成数据蒸馏**：他们研究 hard-label（直接采样教师生成的数据），结论是多样性受限；本文聚焦 soft-label 蒸馏，因软标签能铺开高熵分布而得到相反的"多样性提升"结论，差异根源在于从教师采样多样合成数据本身很难。
- **vs Busbridge et al. (2025)**：后者认为在算力匹配设置下蒸馏未必有用；本文主张应在数据受限设置下评估，把教师 logit 计算成本算进去并非合适比较口径。
- **vs 容量失配类工作（Cho & Hariharan / Mirzadeh / Beyer 等）**：他们指出"更大教师未必更好"并提出缓解方案；本文的实践者指南（教师选择、token routing 等）与之互补。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用熵轴统一解释蒸馏对两种现代范式的相反作用，视角新且自洽
- 实验充分度: ⭐⭐⭐⭐ IsoData + 双视角 + bigram + routing 形成闭环，但规模偏小、routing 偏初步
- 写作质量: ⭐⭐⭐⭐⭐ 现象—机制—缓解三段递进清晰，沙盒与命题相互印证
- 价值: ⭐⭐⭐⭐⭐ 对"未来靠蒸馏预训练小模型"这一现实趋势给出可落地的设计洞察

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] TNT: Improving Chunkwise Training for Test-Time Memorization](tnt_improving_chunkwise_training_for_test-time_memorization.md)
- [\[ICLR 2026\] Beyond Length: Quantifying Long-Range Information for Long-Context LLM Pretraining Data](beyond_length_quantifying_long-range_information_for_long-context_llm_pretrainin.md)
- [\[ICLR 2026\] Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining](scaling_laws_revisited_modeling_the_role_of_data_quality_in_language_model_pretr.md)
- [\[ICCV 2025\] ETA: Energy-based Test-time Adaptation for Depth Completion](../../ICCV2025/llm_pretraining/eta_energy-based_test-time_adaptation_for_depth_completion.md)
- [\[ICLR 2026\] Reformulation for Pretraining Data Augmentation](reformulation_for_pretraining_data_augmentation.md)

</div>

<!-- RELATED:END -->
