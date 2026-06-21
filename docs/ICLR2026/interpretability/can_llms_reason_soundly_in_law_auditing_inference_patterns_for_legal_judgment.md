---
title: >-
  [论文解读] Can LLMs Reason Soundly in Law? Auditing Inference Patterns for Legal Judgment
description: >-
  [ICLR2026][可解释性][交互作用解释] 这篇论文不再只看法律 LLM 的判决"答案对不对"，而是把模型对每个判决的打分**忠实拆解成一组输入短语之间的 AND/OR 交互模式**，再让 16 位法律专家把短语标注为"相关 / 无关 / 禁用"，从而量化出"模型到底是凭哪些逻辑下判断的"；结果发现即便四个主流（含法律专用）LLM 的判决结果正确，**超过一半的推理交互其实是无关甚至错误的依据**——比如把别人的犯罪行为算到被告头上、或被职业身份带偏。
tags:
  - "ICLR2026"
  - "可解释性"
  - "交互作用解释"
  - "AND-OR 逻辑模型"
  - "法律判决"
  - "推理可靠性"
  - "身份偏见"
---

# Can LLMs Reason Soundly in Law? Auditing Inference Patterns for Legal Judgment

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=5T0BXtJxzN](https://openreview.net/forum?id=5T0BXtJxzN)  
**代码**: 待确认  
**领域**: 可解释性 / 法律 LLM 审计  
**关键词**: 交互作用解释, AND-OR 逻辑模型, 法律判决, 推理可靠性, 身份偏见

## 一句话总结
这篇论文不再只看法律 LLM 的判决"答案对不对"，而是把模型对每个判决的打分**忠实拆解成一组输入短语之间的 AND/OR 交互模式**，再让 16 位法律专家把短语标注为"相关 / 无关 / 禁用"，从而量化出"模型到底是凭哪些逻辑下判断的"；结果发现即便四个主流（含法律专用）LLM 的判决结果正确，**超过一半的推理交互其实是无关甚至错误的依据**——比如把别人的犯罪行为算到被告头上、或被职业身份带偏。

## 研究背景与动机
**领域现状**：法律判决预测是 LLM 的一个典型高风险落地场景，已有 LawBench、LegalBench、SaulLM、BAI-Law 等专门 benchmark 和法律专用模型。主流评估方式就是看生成结果（判决类别）准不准——准确率高就算可靠。

**现有痛点**：作者发现"输出 token 正确"和"推理过程正确"是两回事。即使顶尖 LLM 生成了正确的判决 token，它在算这个 token 时所依赖的内部逻辑可能是有问题的。在法律场景里，这种问题逻辑会直接左右模型在多个"看上去都说得通"的判决之间的选择，而这恰恰是本该由法官自由裁量的部分——一旦模型用了带偏见的依据，就会引入不公平和风险，却被"结果正确"完全掩盖。

**核心矛盾**：传统评测只看语言生成结果（一个标量分数 / 类别），无法回答"模型是用什么逻辑得到这个结果的"。而要审计推理过程，又面临一个根本难题：LLM 的预测能不能被**忠实地**分解成一组人能读懂、又能数学上保证等价的推理单元？如果分解本身不忠实，审计就是空中楼阁。

**本文目标**：(1) 找到一种有理论保证的方式把 LLM 判决分解为离散推理模式；(2) 用法律领域知识定义"这条推理可靠 vs 不可靠"；(3) 量化出有多少推理是不可靠的、它们对判决影响多大；(4) 归纳出常见的几类表征缺陷。

**切入角度**：作者团队此前在可解释性理论上证明了——深度网络的输出分数可以被一组**输入特征之间的交互作用（interaction）忠实表示**，并且这种表示具有"普适匹配性"和"稀疏性"两条数学保证。于是他们把这套交互解释工具搬到法律 LLM 上，把每个交互当成模型用的一条"推理模式"。

**核心 idea**：用 **AND-OR 交互作用**把判决打分忠实拆成一组离散推理模式，再借法律专家对短语的"相关/无关/禁用"标注，把这些交互切成**可靠效应**与**不可靠效应**两堆，最后用一个比值指标 $s_{\text{reliable}}$ 衡量"模型的判决依据有多少和人类专家对得上"。

## 方法详解

### 整体框架
方法是一条"**先忠实拆解、再领域裁决、最后量化**"的审计流水线，不是要训一个更好的模型，而是给已有 LLM 的判决做"逻辑体检"。给定一段法律案情 $x$（被切成 $n$ 个语义短语）和模型对某个判决（如 "Assault"）的输出分数 $v(x)$，流程是：

1. **交互提取**：把 $v(x)$ 拆成一个 AND-OR 逻辑模型 $h(x)$，它由一组带权重的交互模式 $I_S^{\text{AND}}, I_S^{\text{OR}}$ 组成，理论上能在输入被任意 mask 的 $2^n$ 种状态下都匹配 $v$，所以每个交互可被视作模型真正用到的一条推理模式。
2. **短语标注**：让法律专家把案情里的短语分成相关 $R$ / 无关 $I$ / 禁用 $F$ 三类。
3. **可靠性分解**：根据每个交互涉及哪些短语，把它的数值效应切成可靠效应（$R_S^{\text{AND}}, R_S^{\text{OR}}$）和不可靠效应（$U_S^{\text{AND}}, U_S^{\text{OR}}$）。
4. **指标量化**：用可靠效应占比 $s_{\text{reliable}}$、交互复杂度（阶数）分布、冲突抵消度 $s_{\text{conflict}}$ 三个角度刻画推理质量，并在具体案例上可视化出几类表征缺陷。

这里的关键在于：分解是"事后审计"，模型本身不动；而能这么审计的底气，全在交互表示的两条数学保证（普适匹配 + 稀疏）。

### 关键设计

**1. 用 AND-OR 交互作用把判决打分忠实拆成推理模式**

要审计推理，前提是分解必须忠实——否则审计的是分析者的臆想而非模型的逻辑。作者用 AND-OR 逻辑模型来表示输出分数。先把判决打分定义为生成目标 token 序列的对数几率之和：$v(x)=\sum_{t=1}^{m}\log\frac{p(y=y_t\mid x, Y_t^{\text{prev}})}{1-p(y=y_t\mid x, Y_t^{\text{prev}})}$，把概率变成可加的标量分数。然后构造逻辑模型 $h(x_{\text{mask}})=h(b)+\sum_{S\in\Omega^{\text{AND}}}\mathbb{1}^{\text{AND}}(S\mid x_{\text{mask}})\,I_S^{\text{AND}}+\sum_{S\in\Omega^{\text{OR}}}\mathbb{1}^{\text{OR}}(S\mid x_{\text{mask}})\,I_S^{\text{OR}}$：AND 触发函数只有当集合 $S$ 里的短语**全部出现**才取 1，OR 触发函数只要 $S$ 里**任一短语出现**就取 1，$I_S$ 是对应的标量权重（AND 权重即合作博弈里的 Harsanyi dividend）。

它有效是因为两条定理保证：**普适匹配性**（Theorem 1）证明只要把权重按 $I_S^{\text{AND}}=\sum_{T\subseteq S}(-1)^{|S|-|T|}v^{\text{and}}(x_T)$ 这样设置，对输入的全部 $2^n$ 种 mask 状态都有 $v(x_T)=h(x_T)$——也就是这组交互不是近似拟合，而是在指数多个扰动样本上都精确等价；**稀疏性**则保证真正有显著数值的交互只有 $O(n^p)$ 个（经验上 $1.5\le p\le2.0$），其余权重几乎为零可剪掉。两条合起来，让"用少数几条可读的交互去代表模型推理"在数学上站得住，这正是本文区别于普通 attribution（如纯 Shapley/显著图）的地方：它给的是可枚举、可验证忠实度的离散推理单元。

**2. 相关 / 无关 / 禁用三类短语标注，把"对错"判据落到领域知识上**

有了交互还不够——交互本身没有"对错"标签，得有人类法律知识来当裁判。作者请 16 位法律专家与志愿者（含两位执业 10 年以上的资深专家）把案情短语 $N$ 划成三个互斥子集 $R\cup I\cup F=N$：**相关短语 $R$** 是直接构成判决理由的（如对 Andy 判 "Assault"，"chased / with an axe / bit / slightly injured" 是直接依据）；**无关短语 $I$** 描述被告但不直接决定判决（如 "morning / had an argument"）；**禁用短语 $F$** 是敏感且会误导的（最典型是描述**另一个人**行为后果的短语，如 "hit / with a shovel / injuring / death" 描述的是 Bob 而非 Andy）。这一步把抽象的"推理可靠性"锚定到了可操作的短语类别上，是后面所有量化的判据来源。作者也坦承这三类并不能穷举所有问题短语，目的只是证明"不可靠推理大量存在"，而非穷尽。

**3. 把交互效应分解为可靠 / 不可靠，并定义可靠占比指标**

第三步把短语标注"投影"到交互效应上，按 AND/OR 各自的语义分别裁决。对 **AND 交互**（必须 $S$ 里短语全在才触发），判据是二值的：当且仅当 $S$ 不含任何禁用短语且至少含一个相关短语时才算可靠，否则整条交互判为不可靠——
$$\text{if } S\cap F=\varnothing,\ S\cap R\neq\varnothing:\ R_S^{\text{AND}}=I_S^{\text{AND}},\ U_S^{\text{AND}}=0;\quad \text{otherwise } R_S^{\text{AND}}=0,\ U_S^{\text{AND}}=I_S^{\text{AND}}$$
对 **OR 交互**（任一短语出现即触发），因为效应是被多个短语共享的，作者按"均匀分摊"（类似 Shapley 的分配方式）把效应按相关短语占比切开：$R_S^{\text{OR}}=\frac{|S\cap R|}{|S|}I_S^{\text{OR}}$，$U_S^{\text{OR}}=\left(1-\frac{|S\cap R|}{|S|}\right)I_S^{\text{OR}}$。这样判决分数就被写成可靠效应和与不可靠效应和两部分。最后定义核心指标 **可靠交互效应占比**：
$$s_{\text{reliable}}=\frac{\sum_{S\in\Omega^{\text{AND}}}|R_S^{\text{AND}}|+\sum_{S\in\Omega^{\text{OR}}}|R_S^{\text{OR}}|}{\sum_{S\in\Omega^{\text{AND}}}|I_S^{\text{AND}}|+\sum_{S\in\Omega^{\text{OR}}}|I_S^{\text{OR}}|}\in[0,1]$$
$s_{\text{reliable}}$ 越接近 1，说明模型的判决依据越贴合专家逻辑。这个指标的价值在于：它把"模型推理对不对"从一句定性吐槽变成了一个可比较的标量。

**4. 交互复杂度与冲突抵消度量，揭示"长链推理"的虚假表象**

光看可靠占比还看不出推理的"形状"，作者再加两个度量。其一是 **交互复杂度（阶数）**，即一条交互涉及的短语数 $|S|$；统计各阶正负交互强度 $A^{(o),\text{pos}}, A^{(o),\text{neg}}$ 的分布后发现，无论看全部交互还是只看可靠交互，模型都**强烈偏好低阶交互**——即只用少数几个局部短语做启发式猜测，本质接近 bag-of-words，而非真的把所有案情要素综合分析。这直接挑战了"CoT 提示让 LLM 具备长链推理"的流行假设。其二是 **冲突抵消度** $s_{\text{conflict}}=1-\frac{\sum_{op}|\sum_{S\in\Omega^{op}}I_S^{op}|}{\sum_{op}\sum_{S\in\Omega^{op}}|I_S^{op}|}$，衡量正效应（支持证据）与负效应（反证据）相互抵消的比例；实验显示超 60% 的交互效应被互相抵消，说明模型判决里存在大量自相矛盾的噪声模式，越可靠的大模型抵消度越低。

### 一个完整示例
以 SaulLM 对 Andy 判 "Assault" 的案情为例（"Andy 持斧追砍并咬伤 Charlie 致轻伤；Bob 用铁锨打 Charlie 致死"）。对 Andy 而言，"chased / with an axe / bit / slightly injured" 是相关短语 $R$，"morning / had an argument" 是无关短语 $I$，而描述 **Bob** 行为的 "hit / with a shovel / injuring / death" 是禁用短语 $F$。审计结果：模型确实用到了一些可靠交互，如 AND 交互 $S_1=\{\text{slightly injured}\}$、$S_2=\{\text{bit}\}$ 和 OR 交互 $S_3=\{\text{bit, slightly injured}\}$，分别贡献可靠效应 $R_{S_1}^{\text{AND}}=0.47$、$R_{S_2}^{\text{AND}}=0.33$、$R_{S_3}^{\text{OR}}=0.10$。但它**同时把 Bob 的行为算到了 Andy 头上**：AND 交互 $S_4=\{\text{death}\}$、$S_5=\{\text{with a shovel}\}$、$S_6=\{\text{injuring}\}$ 贡献了不可靠效应 $U_{S_4}^{\text{AND}}=-1.04$、$U_{S_5}^{\text{AND}}=0.93$、$U_{S_6}^{\text{AND}}=0.19$。最终 SaulLM 对这一判决只有 $s_{\text{reliable}}=41.5\%$ 的可靠占比——判决虽然正确，但近六成依据是错的，模型实际上是把"武器/致死"这类敏感词和判决结果记在了一起，而没真正分清"谁做了什么"。

## 实验关键数据

### 主实验
评测四个模型：通用模型 Qwen2.5-14B-Base、Deepseek-R1-Distill-Qwen-14B，法律专用模型 SaulLM-7B-Instruct（英文法律语料）、BAI-Law-13B（中文法律语料）。英文任务用 LexGLUE 的 ECtHR、LegalBench 的 Learned Hand Crime；中文任务用 CAIL2018、LeCaRD、LEVEN。每任务随机取 100 样本，每样本由资深专家挑 10 个信息短语、再由 16 人标注短语类别。

| 维度 | 关键结果 | 说明 |
|------|---------|------|
| 可靠占比 $s_{\text{reliable}}$ | 无论通用还是法律专用 LLM 都偏低，**过半交互不可靠** | 判决正确 ≠ 推理正确 |
| 单案例（Assault, SaulLM） | $41.5\%$ | 近六成依据来自禁用/无关短语 |
| 单案例（Intentional Injury, BAI-Law） | $44.5\%$ | 同案略高但仍不可靠为主 |
| 交互复杂度 | 强烈偏好低阶交互 | 本质接近 bag-of-words，质疑"长链推理" |
| 冲突抵消度 $s_{\text{conflict}}$ | $>60\%$ 效应被互相抵消 | 判决内部存在大量矛盾噪声 |

### 分析实验（身份偏见案例）

| 配置（受害者职业） | 判决结果 | 关键变化 |
|------|---------|------|
| a judge / a lawyer / a policeman | Robbery（维持） | 法律相关职业把判决"撑住" |
| a volunteer / a programmer | Not mentioned（改判） | 非法律职业导致判决崩塌 |
| judge→volunteer 时 $S_5=\{\text{[occupation], a day's work, belongings}\}$ | 可靠效应 $0.22\to0.06$ | 含职业短语的交互显著左右判决 |

### 关键发现
- **结果正确掩盖了推理错误**：四个模型在判决类别上都能答对，但拆开看，超过一半的推理交互依赖无关或禁用短语——这是被传统准确率评估完全漏掉的风险。
- **"实体张冠李戴"是普遍缺陷**：模型倾向把武器、致死等敏感 token 与判决绑定记忆，而非分清"谁施加了哪个行为"，导致把 Bob 的行为算到 Andy 头上。
- **职业身份会左右判决**：把受害者职业从"法官"换成"志愿者/程序员"，判决直接从 Robbery 变成 Not mentioned，暴露出明显的职业偏见，且这种身份偏见很可能推广到年龄、性别、学历等其他属性。
- **长链推理是假象**：低阶交互占主导说明模型在做局部启发式猜测，而非综合所有案情要素，CoT 表象下的"推理"被证伪。

## 亮点与洞察
- **把"评对错"从结果层提到推理层**：传统评测只问"判决类别对不对"，本文问"凭什么这么判"，并用有数学保证的交互分解让这个问题变得可量化、可验证——这是方法论上的一次抬升。
- **理论保证是地基而非装饰**：普适匹配性（在 $2^n$ 个 mask 样本上精确等价）+ 稀疏性（仅 $O(n^p)$ 条显著交互）让"用少数交互代表模型推理"不再是启发式，而这正是审计可信的前提，可迁移到金融、医疗等其他高风险领域。
- **"相关/无关/禁用"标注范式很巧**：用一个简单的三分类把人类领域知识接进交互分解，AND 用二值判据、OR 用均匀分摊，干净地把效应切成可靠/不可靠两堆——这套接口思路可复用到任何"有专家可标注关键证据"的任务。
- **冲突抵消度量是个被忽视的视角**：>60% 效应互相抵消揭示了判决内部的自相矛盾，这种"噪声推理"的度量为衡量模型可靠性提供了新角度。

## 局限与展望
- **不是 benchmark 而是风险警示**：作者明确说当前 LLM 的表征质量太差，以至于无法支撑"用交互质量给不同模型排名"的 benchmark 评测，所以本文定位是揭示缺陷的存在性，而非给出可比的排行榜。
- **依赖人工标注、规模有限**：每任务仅 100 样本、每样本 10 个短语，且短语类别由专家手工标注；法律条文繁多、各国法律差异大，无法穷举，三类短语也不是问题短语的完整枚举。
- **短语粒度与标注主观性**：把案情切成短语并三分类本身带有判断空间，不同专家对"相关 vs 无关"的边界可能不一致，可靠占比会受标注口径影响。
- **改进方向**：把审计信号反过来用于训练/对齐（如惩罚不可靠交互、抑制禁用短语效应），或扩展到 finance/healthcare 等其他高风险任务做跨域可靠性体检。

## 相关工作与启发
- **vs 传统语言生成评测（准确率/LegalBench 等）**：他们只看输出结果是否正确，本文看输出背后的推理交互是否正确，揭示了"结果对、过程错"这一被长期忽视的盲区。
- **vs 普通归因解释（Shapley / 显著图）**：普通 attribution 给每个 token 一个重要性分但无忠实度保证，本文用 AND-OR 交互给出可枚举、在指数多扰动上精确匹配的离散推理单元，并能区分 AND/OR 两种逻辑。
- **vs 作者团队此前交互理论工作（Harsanyi 交互、稀疏概念、generalizable interaction primitives）**：本文是把这套理论从"解释通用 DNN/概念涌现"落地到"审计法律 LLM 推理可靠性"的应用，新增了相关/无关/禁用标注与可靠占比指标这层领域接口。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把审计从结果层抬到推理层，并用有数学保证的交互分解 + 领域标注落地，视角新颖
- 实验充分度: ⭐⭐⭐⭐ 覆盖 4 模型 6 数据集中英双语、含复杂度/抵消/偏见多维分析，但每任务仅 100 样本且依赖人工标注
- 写作质量: ⭐⭐⭐⭐ 理论铺陈清晰、案例可视化直观，但交互记号较密，门槛偏高
- 价值: ⭐⭐⭐⭐⭐ 直指高风险场景"结果对掩盖推理错"的真实风险，对法律/金融/医疗等可靠性评估有现实意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Reforming the Mechanism: Editing Reasoning Patterns in LLMs with Circuit Reshaping](reforming_the_mechanism_editing_reasoning_patterns_in_llms_with_circuit_reshapin.md)
- [\[ICLR 2026\] The Achilles' Heel of LLMs: How Altering a Handful of Neurons Can Cripple Language Abilities](the_achilles_heel_of_llms_how_altering_a_handful_of_neurons_can_cripple_language.md)
- [\[ICML 2026\] Interpretability Can Be Actionable](../../ICML2026/interpretability/interpretability_can_be_actionable.md)
- [\[ACL 2026\] Evian: Towards Explainable Visual Instruction-tuning Data Auditing](../../ACL2026/interpretability/evian_towards_explainable_visual_instruction-tuning_data_auditing.md)
- [\[ICLR 2026\] GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](gepa_reflective_prompt_evolution_can_outperform_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
