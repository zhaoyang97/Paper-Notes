---
title: >-
  [论文解读] An Information-Theoretic Criterion for Efficient Data Synthesis
description: >-
  [ICML2026][LLM推理][合成数据] 这篇论文用数据处理不等式解释合成数据为何有时有效、有时导致模型坍塌：只有当训练闭环持续引入稳定外部信号时，合成数据才是 information-open；而高 meta-level 的验证信号比实例级模仿更高效、更容易泛化。 领域现状：大模型训练越来越依赖合成数据…
tags:
  - "ICML2026"
  - "LLM推理"
  - "合成数据"
  - "信息论"
  - "数据处理不等式"
  - "外部验证器"
  - "奖励黑客"
---

# An Information-Theoretic Criterion for Efficient Data Synthesis

**会议**: ICML2026  
**arXiv**: [2605.16379](https://arxiv.org/abs/2605.16379)  
**代码**: 无  
**领域**: LLM预训练  
**关键词**: 合成数据、信息论、数据处理不等式、外部验证器、奖励黑客  

## 一句话总结
这篇论文用数据处理不等式解释合成数据为何有时有效、有时导致模型坍塌：只有当训练闭环持续引入稳定外部信号时，合成数据才是 information-open；而高 meta-level 的验证信号比实例级模仿更高效、更容易泛化。

## 研究背景与动机
**领域现状**：大模型训练越来越依赖合成数据。真实高质量文本逐渐接近供给瓶颈，而数学、代码、工具使用和长程推理等任务又需要比普通网页语料更强的监督信号。成功案例包括 verifier-guided synthesis、RLVR、程序测试反馈、形式证明检查器和固定 rubric 评估器。

**现有痛点**：合成数据是一把双刃剑。用模型自己的输出反复自训练，常会造成分布收缩、长尾模式丢失和能力退化；但带 verifier 或环境反馈的合成管线又能带来巨大增益。现有经验很多，却缺少一个统一解释：什么时候合成数据注入了新信息，什么时候只是把模型已有分布循环一遍。

**核心矛盾**：模型生成的数据本身并不会凭空增加关于真实任务的信息。若训练样本完全来自当前模型，并且没有任何独立于当前模型状态的反馈，数据处理不等式意味着任务相关信息只会保持或下降。可是现实中 RLVR、代码测试、证明检查又确实能让模型进步，这说明成功管线并不是封闭系统。

**本文目标**：作者希望给出一个信息论准则，判断合成数据管线是否有效；进一步解释在有效管线之间，为什么某些外部信号能以很少样本产生跨域泛化，而另一些信号需要大量数据却收益有限。

**切入角度**：论文把训练过程写成随机变量关系：真实任务结构为 $X$，已有数据为 $D$，模型状态为 $Z$，外部信号为 $S$。若 $X\to D\to Z$ 或 $X\to Z_t\to D_t^{syn}\to Z_{t+1}$ 构成信息闭环，则 DPI 给出信息单调不增；若引入稳定的 $S$，上界变为 $I(X;D,S)$ 或 $I(X;Z_t,S)$。

**核心 idea**：合成数据是否有效不取决于“是不是模型生成”，而取决于训练循环是否 information-open，以及外部信号是否以高 meta-level 方式注入任务相关信息。

## 方法详解

### 整体框架
这篇论文不提新训练算法，而是给合成数据管线一套信息论的"判定标准"。核心做法是把整个训练过程写成随机变量之间的关系——真实任务结构 $X$、数据 $D$、模型状态 $Z$、外部信号 $S$——然后看数据处理不等式（DPI）允许信息怎么流动：先区分 information-closed 与 information-open 两种管线解释模型坍塌，再用"meta-level"刻画外部信号进入 SFT、RFT、RLVR 时的样本效率差异，最后用 task-relevant partition 把监督信号拆成有效任务信息与类内噪声，从而把泛化、多样性和 reward hacking 统一在同一框架里。

把训练抽象成把数据 $D$ 和内部随机性 $R$ 映射到模型状态 $Z=f_{train}(D,R)$。若训练程序没有关于任务结构 $X$ 的额外侧信息，就有马尔可夫链 $X\to D\to Z$，DPI 立即给出 $I(X;Z)\leq I(X;D)$——闭环自训练无法系统性增加模型关于真实任务的信息。对迭代合成数据更尖锐：若 $D_t^{syn}$ 完全由 $Z_t$ 生成，且没有 verifier、环境、固定 judge 或新数据，那么 $X\to Z_t\to D_t^{syn}\to Z_{t+1}$，于是 $I(X;Z_{t+1})\leq I(X;Z_t)$，信息逐轮单调不增；实际退化（采样误差、优化误差、分布收缩）还会让不等式严格。关键转折在于引入外部信号 $S$：此时正确的观察对象变成 $(D,S)$ 或 $(Z_t,S)$，上界松成 $I(X;Z)\leq I(X;D,S)=I(X;D)+I(X;S\mid D)$，真正新增的信息全部来自条件互信息 $I(X;S\mid D)$。

### 关键设计

**1. Information-open criterion：用条件互信息判定合成管线能否持续涨点**

合成数据是双刃剑——纯自训练会坍塌、带 verifier 的管线却能大涨，但过去缺一个统一判据说清两者差在哪。本文给出的标准很干净：只要外部信号 $S$ 满足 $I(X;S\mid D)>0$（迭代场景是 $I(X;S\mid Z_t)>0$），它就携带了当前模型/数据之外的任务相关信息，这条管线就是 information-open。反过来，没有这样的 $S$ 时，随机采样只能提供多样的候选，却给不出"哪个候选更贴近真实任务"的选择方向，循环重新闭合、DPI 接管、信息只减不增。这一刀同时解释了两件经验事实：闭环 self-training 为什么必然坍塌，以及代码测试、证明检查、环境奖励、固定 rubric 为什么能让合成数据真正有用——它们都是不随学生模型漂移的稳定外部信号。

**2. Meta-level information injection：信号区分"行为等价类"还是"具体参考"决定样本效率**

光是 information-open 还不够，不同的有效管线样本效率差很多，差异来自信号注入的"层级"。高 meta-level 信号只区分行为上重要的等价类，比如"答案是否正确""程序是否通过测试"；低 meta-level 信号则要求复现某一个具体参考答案。当一道题有 $M$ 个同样可接受的答案时，高层信号把它们视为同一类，而实例级 SFT 要额外花最多 $\log M$ bit 去识别某个表面形式，这部分容量对任务毫无价值。这正解释了为什么 RLVR 的二元 correctness 能跨数学、代码、逻辑泛化（它只约束"什么算对"），而单参考答案模仿常把模型容量耗在无关的措辞和格式细节上。

**3. 监督信息分解：把任务信号与类内噪声拆开，给 reward hacking 一个信息论解释**

给定 task-relevant partition $\pi$（按"任务上等价"把输出分块），任意监督信号都满足分解

$$I(Y;S\mid Q)=I([Y]_\pi;S\mid Q)+I(Y;S\mid [Y]_\pi,Q)$$

第一项 $I([Y]_\pi;S\mid Q)$ 是真正的任务相关增益，第二项是类内增益（同一等价类内部的表面差异）。定义效率 $\eta_\pi$ 为任务相关项占比：信号只依赖 $[Y]_\pi$ 时 $\eta_\pi=1$，完全高效。reward hacking 的机制由此变得清晰——如果存在一个更粗、更易学的伪信号（如长度、风格）恰好和奖励相关，它在梯度学习的 simplicity bias 下信息效率更高，模型就会理性地优先收敛到这个伪分区，于是训练 reward 上升、真实正确率停滞。这把 reward hacking 从"模型偷懒钻空子"改写成"模型学到了训练信号中信息效率最高的成分"，对应的解法也随之明确：不是训练更久，而是主动去相关伪特征（打散长度/风格/来源），或让真正的意图信号成为最高效的那一个。

### 损失函数 / 训练策略
论文用同一框架对比三种信号注入方式：SFT 最大化固定数据对的 log-likelihood，本质是抬高某个参考答案的概率；RFT 由当前模型生成候选，再用外部 acceptance test 过滤，对被接受样本做 SFT；RLVR 则把 verifier reward 直接放进 policy gradient，用 advantage 加权模型自己生成的输出。作者强调三者真正的分水岭不是"数据是否合成"，而是外部信号以什么方式进入梯度与样本选择——这恰好对应前面的 information-open 与 meta-level 两个维度。

## 实验关键数据

### 主实验
本文主要是理论与案例分析论文，没有传统 benchmark 表。主结果可以概括为以下准则与案例证据。

| 场景 | 指标 | 本文判据 / 观察 | 之前常见理解 | 提升 |
|------|------|------------------|--------------|------|
| 闭环 self-training | $I(X;Z_{t+1})\leq I(X;Z_t)$ | 无外部信号时任务信息不能增加，坍塌是预期结果 | 把坍塌视为经验性训练不稳定 | 给出 DPI 级别解释 |
| 带 verifier / environment 的合成数据 | $I(X;S\mid Z_t)>0$ | 外部信号提供新增任务信息，循环保持 information-open | 只说“过滤提升数据质量” | 明确新增信息来源 |
| SFT vs RLVR | meta-level | SFT 学具体参考，RLVR 学正确性等价类 | 只比较算法形式 | 解释 RLVR 样本效率和跨域泛化 |
| JudgeRLVR 案例 | 二元 correctness signal | 仅用正确/错误训练的 judge 可跨 math/code/logic 泛化 | 需要 domain-specific rubric | 高 meta-level 信号减少无关差异 |
| 奖励黑客案例 | 训练 reward 上升但真实正确率停滞 | 模型学到长度/风格这个更粗伪信号 | reward hacking 是模型“钻空子” | 信息效率视角给出可诊断机制 |

### 消融实验
论文中的分析性消融主要围绕“是否外部”“信号粒度”和“伪相关是否被去除”。

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无外部信号的自训练 | 信息闭环，$I(X;S\mid Z_t)=0$ | 只能探索已有分布，无法判断哪些样本更贴近任务 |
| 固定 evaluator / verifier | 信息开放，$I(X;S\mid Z_t)>0$ | evaluator 不随学生漂移，能持续提供选择压力 |
| 高 meta-level 二元信号 | $\eta_\pi=1$ | 只区分正确/错误或满足/不满足，把类内表面差异全部忽略 |
| 实例级参考答案 | 需要识别类内具体元素 | 当可接受答案很多时，监督容量被消耗在无关表面形式上 |
| 长度/风格与正确性相关 | reward 上升、真实正确率停滞 | 伪信号比正确性更粗更易学，模型优先收敛到伪分区；重平衡数据源后问题消失 |

### 关键发现
- 合成数据的核心瓶颈是 verification capacity。数学、代码、形式推理进展快，是因为这些领域更容易提供稳定、高 meta-level 的外部信号。
- 随机性和外部信号缺一不可。随机采样提供候选多样性，外部信号提供筛选方向；只有采样没有信号会坍塌，只有信号没有候选也无法探索。
- 多样性优于重复数据的原因可以用 partition coverage 解释：同一 prompt-output 对反复出现几乎不提供新任务信息，而覆盖新分区块的样本会提供 fresh bits。

## 亮点与洞察
- 论文把“合成数据是否有效”从经验配方提升为信息边界问题。这个视角很实用，因为它迫使我们问：管线里到底哪个变量携带了当前模型之外的任务信息。
- meta-level 的表述很有解释力。许多看似不同的成功案例，如代码单元测试、Lean proof checker、格式奖励、SAT solver evaluator，本质都是把大量表面不同的输出压缩到少数行为等价类。
- 对 reward hacking 的解释比常见说法更可操作：如果伪特征比真实目标更高效，模型学习伪特征就是理性的优化结果。因此数据构造时必须主动打散长度、风格、来源模型等粗粒度伪相关。
- 论文也给了训练设计启发：与其追求更多合成样本，不如先投资稳定 verifier、固定 judge、可执行环境和多样 prompt coverage。

## 局限与展望
- 框架主要是定性解释，不能直接预测某个合成数据管线需要多少样本、多少 verifier 精度或多少轮训练才能生效。
- “模型优先收敛到信息效率最高信号”的 thesis 依赖梯度学习的 simplicity bias 和案例支持，并不是由互信息分解本身严格推出。
- 外部信号被当作给定变量，但现实中 verifier 会有误差、覆盖盲区和可被优化攻击的问题；如何设计随模型能力提升仍可靠的 verifier 是开放问题。
- 对开放式生成、安全偏好和创造性任务，高质量高 meta-level 信号很难构造，可能需要人类标注、固定 rubric、模型 judge 与行为测试的混合方案。

## 相关工作与启发
- **vs 模型坍塌研究**: 既有工作观察到 self-consuming generative models 会退化，本文用 DPI 把它解释为 information-closed loop 的必然趋势。
- **vs RLVR / verifiable reward**: RLVR 的成功不只是强化学习技巧，而是 verifier 作为外部信号直接进入梯度，使训练循环保持 information-open。
- **vs SFT 合成数据**: SFT 可以有效蒸馏固定数据，但当多个答案都可接受时，它会把监督容量花在复现具体参考上；本文指出这正是低 meta-level 的低效来源。
- **vs Sutton Bitter Lesson**: 论文把“让计算自己发现结构”解释为用 meta-level 约束而非实例级知识硬编码：信号只规定什么算对，不规定必须长什么样。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用 information-open 和 meta-level injection 统一解释合成数据、RLVR、坍塌和 reward hacking，视角很强。
- 实验充分度: ⭐⭐⭐⭐☆ 案例和理论分析充分，但缺少统一可复现实验 benchmark 与定量预测。
- 写作质量: ⭐⭐⭐⭐⭐ 概念链条清楚，从 DPI 到监督信息分解再到实际案例，论证连贯。
- 价值: ⭐⭐⭐⭐⭐ 对设计 LLM 合成数据管线、verifier 和奖励模型都有很高参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Efficient PRM Training Data Synthesis via Formal Verification](../../ACL2026/llm_reasoning/efficient_prm_training_data_synthesis_via_formal_verification.md)
- [\[ACL 2026\] Efficient Process Reward Modeling via Contrastive Mutual Information](../../ACL2026/llm_reasoning/efficient_process_reward_modeling_via_contrastive_mutual_information.md)
- [\[ACL 2026\] Self-Reinforcing Controllable Synthesis of Rare Relational Data via Bayesian Calibration](../../ACL2026/llm_reasoning/self-reinforcing_controllable_synthesis_of_rare_relational_data_via_bayesian_cal.md)
- [\[ACL 2026\] MathAgent: Adversarial Evolution of Constraint Graphs for Mathematical Reasoning Data Synthesis](../../ACL2026/llm_reasoning/mathagent_adversarial_evolution_of_constraint_graphs_for_mathematical_reasoning_.md)
- [\[ICLR 2026\] DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning](../../ICLR2026/llm_reasoning/designer_design-logic-guided_multidisciplinary_data_synthesis_for_llm_reasoning.md)

</div>

<!-- RELATED:END -->
