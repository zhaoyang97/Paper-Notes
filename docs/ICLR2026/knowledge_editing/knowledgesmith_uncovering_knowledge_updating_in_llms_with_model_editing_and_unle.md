---
title: >-
  [论文解读] KnowledgeSmith: Uncovering Knowledge Updating in LLMs with Model Editing and Unlearning
description: >-
  [ICLR 2026][知识编辑][机器遗忘] 本文提出 KnowledgeSmith，把"知识编辑"和"机器遗忘"统一为同一个约束优化问题，并用知识图谱自动生成跨层级（根/中间/叶）、跨数据规模的大规模评测基准，系统揭示了 LLM 知识更新中的传播不对称、一致性-容量权衡、学科依赖等一系列反直觉现象。
tags:
  - "ICLR 2026"
  - "知识编辑"
  - "机器遗忘"
  - "知识图谱"
  - "一致性-容量权衡"
  - "传播不对称"
  - "评测基准"
---

# KnowledgeSmith: Uncovering Knowledge Updating in LLMs with Model Editing and Unlearning

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=znnA2Opw6v](https://openreview.net/forum?id=znnA2Opw6v)  
**代码**: [https://github.com/AIFrontierLab/KnowledgeSmith](https://github.com/AIFrontierLab/KnowledgeSmith)  
**领域**: 知识编辑 / 机器遗忘 / LLM 知识更新机制  
**关键词**: 知识编辑, 机器遗忘, 知识图谱, 一致性-容量权衡, 传播不对称, 评测基准  

## 一句话总结
本文提出 KnowledgeSmith，把"知识编辑"和"机器遗忘"统一为同一个约束优化问题，并用知识图谱自动生成跨层级（根/中间/叶）、跨数据规模的大规模评测基准，系统揭示了 LLM 知识更新中的传播不对称、一致性-容量权衡、学科依赖等一系列反直觉现象。

## 研究背景与动机
**领域现状**：让 LLM "保持最新"主要靠两条技术路线——知识编辑（精准改写某条事实，如 ROME/MEMIT/AlphaEdit）和机器遗忘（广泛抹除某类信息，如 ReLearn）。两者各有大量方法积累，但长期被当作两个独立问题分别研究。

**现有痛点**：① 绝大多数评测只针对**孤立事实**，忽略真实知识是图状互联的——改了"里昂是法国首都"，"埃菲尔铁塔在法国首都"也该跟着变，但现有基准测不出这种级联；② 编辑 vs 遗忘在**数据规模**上的差异不清楚（编辑往往小数据就够，遗忘则不然）；③ 缺乏一个统一框架把两者放在一起比较其在传播、稳定性、泛化上的 trade-off。

**核心矛盾**：编辑追求**精准注入**但易产生副作用（over-spread，殃及无关节点）；遗忘强调**广泛抹除**但常常改不动目标（under-spread，漏改）。这对"可塑性"与"稳定性"的张力，缺乏受控、可扩展、结构化的工具去刻画。

**本文目标**：建立一个统一的理论视角 + 自动化的结构化基准，回答"LLM 到底是怎么更新知识的，它像人类一样会级联传播吗"。

**核心 idea**：**【统一视角】** 编辑和遗忘是同一约束优化问题的两个实例，差别只在目标分布 $q_\text{target}$ 的选择；**【KG 驱动评测】** 把任意 KG 相关数据集自动转成跨层级、跨规模的干预基准，从而能受控地观察更新如何在知识层级中传播。

## 方法详解

### 整体框架
KnowledgeSmith 由两部分构成：一是**统一优化视角**，把编辑/遗忘写成带保持约束的同一个目标函数；二是**自动基准生成管线**，从知识图谱出发，在根/中间/叶三个层级、从 1 到百万级数据规模上生成探针，测量直接效果与传播效果。整套流程对编辑器和遗忘器都是 method-agnostic 的（实验用 AlphaEdit 做编辑、ReLearn 做遗忘）。

```mermaid
flowchart LR
    A[静态知识图谱<br/>4 领域: 生物/经济/历史/物理] --> B[实体-关系选择<br/>分为 root/inter/leaf 三层]
    B --> C[模板化问题生成<br/>6 类探针]
    C --> D[四选一 QA 构造<br/>百万级样本]
    D --> E[统一优化干预<br/>编辑 or 遗忘]
    E --> F[跨层级/跨规模评测<br/>传播·一致性·鲁棒性·失败模式]
```

### 关键设计

**1. 把编辑与遗忘统一成一个约束优化问题：分歧只在目标分布。** 设模型 $f_\theta$ 给出条件分布 $p_\theta(y\mid x)$，一次更新请求由待改项 $e$ 和作用域 $c$ 给出，得到 $\theta'=T(\theta;e,c)$。文章定义两类探针——应当改变的正探针 $Q^+$ 和应当保持的保持探针 $Q^-$，把"改对目标、不伤无关"写成统一目标：

$$\theta' = \arg\min_{\theta'}\; \mathcal{L}_\text{task}(\theta';Q^+) + \lambda_\text{pres}\,\mathcal{L}_\text{pres}(\theta';Q^-) + \lambda_\text{reg}\,R(\theta',\theta)$$

其中 $\mathcal{L}_\text{task}$ 让 $Q^+$ 逼近目标分布 $q_\text{target}$，$\mathcal{L}_\text{pres}$ 抑制 $Q^-$ 上的漂移，$R$ 正则化参数改动量（如 $\lVert\Delta\rVert_2^2$、Fisher 范数、低秩约束）。**编辑**就是 $q_\text{target}$ 编码一个事实纠正（"巴黎是德国首都"）；**遗忘**就是 $q_\text{target}$ 取中性分布 $q_\text{neutral}$（"巴黎是\[MASK\]首都"）。ROME/MEMIT、MEND、GRACE、LoRA 编辑、影响函数遗忘、认证删除全都能归入这个式子的不同实例化——这给了"公平对比"一个统一标尺。

**2. 用知识图谱把孤立事实评测升级成层级传播评测。** 现有基准只测单点事实，测不出"改一处会不会级联"。本文锚定在一个 GPT-4o 生成、人工校验的层级 KG 上，把节点分成 **root（领域级宽概念）/ intermediate（子主题）/ leaf（具体实体）** 三层，对每层分别施加干预，再观察直接节点与结构相关节点的变化。这样单张 KG 就能展开成动态基准：既测目标本身改没改对，又测它在 multi-hop、反向关系等结构依赖上传播得对不对。这是本文能观察"传播不对称"的前提。

**3. 六类探针 + 自动 QA 管线，把任意 KG 数据集转成百万级标准化基准。** 管线三步：实体-关系选择（按三层采样保留层级结构）→ 模板化问题生成（每个三元组生成多种问法，人工校语法与事实）→ 四选一 QA 构造（MMLU 风格，实体替换+改写产出超百万样本，全部对 KG 做校验）。六类探针各自对应一种行为：direct（目标是否被更新）、reverse（关系方向是否守住）、conflict（是否出现矛盾，兼测对抗鲁棒）、multi-hop（是否沿链式关系正确传播）、comparison（更新后是否被一致偏好）、contextual（无关/OOD 知识是否被保住）。其中 direct/reverse/multi-hop/comparison 属 $Q^+$，contextual 属 $Q^-$，conflict 横跨两者。本文实例化为生物/经济/历史/物理四个领域，每分支编辑与遗忘各 10,000 样本，合计约 36 万训练样本。

**4. 提出三个新诊断指标刻画"改过头/没改动/自相矛盾"。** 为量化传播不对称，定义 **CCR（Collateral Change Ratio，附带改变率）** 捕捉编辑的 over-spreading，**RR（Residual Retention，残余保留率）** 捕捉遗忘的 under-spreading；为捕捉残余信念之外的失败，定义 **conflict rate（冲突率）**——模型在不同上下文下同时支持互相矛盾断言（既说"巴黎是德国首都"又说"巴黎是法国首都"）的比例。这三个指标补足了传统只看"目标是否改对"的盲区，让一致性崩塌、矛盾涌现这些隐藏不稳定性变得可测。

**5. SVD 几何视角解释编辑与遗忘的机制差异。** 对参数矩阵 $W=U\Sigma V^\top$，干预后 $W'=U'\Sigma'V'^\top$，把改动分解为**缩放效应**（奇异值 $\Sigma'/\Sigma$ 的放大/衰减）与**旋转效应**（子空间 $\text{span}(U,V)$ 的重定向）。实验发现编辑表现为"局部旋转 + 轻度缩放"，保留了大部分表示几何、只重定向特定事实方向；遗忘则在超过临界数据规模后出现**突变式相位转变**。这从几何上解释了为什么编辑平滑局部、遗忘剧烈整体。

## 实验关键数据

覆盖 6 个 LLM 家族、1B–123B 共 13 个模型（LLaMA-3、Qwen-3、QwQ-32B、Mistral、Gemma、DeepSeek-R1-Qwen3-8B），编辑用 AlphaEdit、遗忘用 ReLearn，数据规模从 1 到 10,000。

### 主实验：传播与鲁棒性

| 现象 | 编辑 (Editing) | 遗忘 (Unlearning) |
|------|---------------|-------------------|
| 传播方向 | over-spread（殃及相关节点，低层更明显） | under-spread（漏改，传不到目标之外） |
| 即时可塑性 | 小模型快但不稳 | 大模型需更多数据但更稳 |
| ID 准确率 | 高（经济可达 50–60%） | 低（≤30%） |
| OOD 准确率 | 受损（牺牲全局稳定） | 强（63–82%，保住无关知识） |
| 计算成本(1000样本/H100) | ~6h | ~0.2h |

### 一致性-容量权衡（表示相似度，log-min-max 归一化）

| 指标 | 设置 | k=1 | k=10 | k=100 | k=1000 | k=10000 |
|------|------|-----|------|-------|--------|---------|
| KL | 遗忘 | 0.014 | 0.392 | 0.805 | 0.838 | 0.883 |
| KL | 编辑 | 0.140 | 0.522 | 0.606 | 0.647 | 0.652 |
| CKA | 遗忘 | 0.917 | 0.861 | 0.566 | 0.576 | 0.692 |
| CKA | 编辑 | 0.958 | 0.852 | 0.801 | 0.714 | 0.714 |

数据规模超过模型容量后，direct 探针饱和/下降而 reverse 探针仍高 → **一致性崩塌**；崩塌点在低层（叶/中间）比根层来得更早。

### 失败模式统计（开放问答中观测占比）

| 失败模式 | 编辑 | 遗忘 |
|----------|------|------|
| Under-forgetting (RR) | 20% | 35% |
| Over-spreading (CCR) | 35% | 15% |
| Conflict emergence | 30% | 12% |
| Knowledge drift | 18% | 10% |
| Instruction-following drop | 22% | 18% |
| Hallucination increase | 5% | 4% |

### 关键发现
- **传播不对称**：编辑改过头、遗忘改不动；层级分支结构对更新效果设了天花板，越高/越中心的节点越难改。
- **学科依赖**：历史领域最"抗改"，即使大量样本也几乎不动，说明评测必须做 subject-aware，CounterFact/ZsRE 一视同仁是有偏的。
- **方法对比**：LoRA 微调最不稳（k=1000 时 ID 准确率掉到 12.5%），编辑兼顾稳定性与低数据效率，遗忘保守但稳——这解释了为何持续更新更应选编辑/遗忘而非 LoRA。

## 亮点与洞察
- **统一视角干净有力**：把编辑和遗忘归为同一约束优化、差别仅在 $q_\text{target}$，是一个简洁且能"装下"几乎所有现有方法的理论框架，给后续公平对比立了标尺。
- **从"测点"到"测网"**：用 KG 把孤立事实评测升级成层级传播评测，是该工作最有价值的方法论贡献，让 over-spread/under-spread 这类现象首次可被量化观察。
- **一系列反直觉结论**：LLM 并不像人类那样级联更新知识、存在一致性-容量权衡、历史比物理更难改——这些都对"知识更新该怎么设计"有实际指导意义。
- **规模诚意足**：13 个模型 × 4 领域 × 6 类探针 × 跨 5 个数量级数据规模，结论的普适性较强。

## 局限与展望
- **领域仅 4 个**（生物/经济/历史/物理），虽兼顾 STEM 与人文，但法律、医学等高价值领域尚未验证，结论的跨域外推待考。
- **KG 由 GPT-4o 生成**，尽管有外部校验与人工抽检，仍可能引入生成偏差或事实噪声，影响"真值"可靠性。
- **方法只选了两个代表**（AlphaEdit + ReLearn），虽声称 method-agnostic，但不同编辑/遗忘算法是否都呈现同样的不对称与崩塌规律，需要更多 baseline 佐证（附录有部分补充）。
- 文章是**诊断性/分析性工作**，揭示了问题但没有给出"如何同时兼顾可塑性与一致性"的解法，留给后续——例如基于层级感知的传播正则、subject-aware 的更新预算分配。

## 相关工作与启发
- **知识编辑**：ROME/MEMIT（定位并修改 MLP 权重）、MEND（辅助网络重定向）、GRACE（梯度更新+约束漂移）、AlphaEdit（本文采用的 SOTA 编辑器）。
- **机器遗忘**：负梯度微调、影响函数/Fisher 加权删除、认证删除、ReLearn（本文采用的遗忘器）。
- **核心启发**：① 评测应从"孤立事实"走向"结构化传播"，KG 是天然脚手架；② 编辑与遗忘不该割裂研究，统一优化视角能揭示共享的失败机理；③ 不同知识域的"可改性"差异巨大，未来的更新算法应做 subject-aware 与 hierarchy-aware 的资源分配。对做持续学习、模型对齐、事实纠错的研究者，本文提供了可直接复用的基准生成管线和诊断指标。

## 评分
- 新颖性: ⭐⭐⭐⭐ 统一优化视角虽不算颠覆，但"KG 驱动的层级传播基准 + CCR/RR/conflict rate 诊断指标"组合很新，首次把传播不对称做成可量化现象。
- 实验充分度: ⭐⭐⭐⭐ 13 模型 × 4 领域 × 5 个数量级数据规模，外加表示分析、SVD 几何、鲁棒性与失败模式，覆盖面扎实；扣分在领域仅 4 个、方法仅 2 个代表。
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰、统一框架推导利落、findings 提炼成 5 条主结论易读；图表略密集需对照附录。
- 价值: ⭐⭐⭐⭐ 给"LLM 知识更新机制"提供了可复用的评测工具与一批反直觉洞察，对编辑/遗忘/持续学习社区有实际指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] MobiEdit: Resource-efficient Knowledge Editing for Personalized On-device LLMs](mobiedit_resource-efficient_knowledge_editing_for_personalized_on-device_llms.md)
- [\[ICLR 2026\] MoEEdit: Efficient and Routing-Stable Knowledge Editing for Mixture-of-Experts LLMs](moeedit_efficient_and_routing-stable_knowledge_editing_for_mixture-of-experts_ll.md)
- [\[NeurIPS 2025\] MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs](../../NeurIPS2025/knowledge_editing/memoir_lifelong_model_editing_with_minimal_overwrite_and_informed_retention_for_.md)
- [\[ICLR 2026\] Fine-tuning Done Right in Model Editing](fine-tuning_done_right_in_model_editing.md)
- [\[ICLR 2026\] Energy-Regularized Sequential Model Editing on Hyperspheres](energy-regularized_sequential_model_editing_on_hyperspheres.md)

</div>

<!-- RELATED:END -->
