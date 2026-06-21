---
title: >-
  [论文解读] Mixture of Cognitive Reasoners: Modular Reasoning with Brain-Like Specialization
description: >-
  [ICLR 2026][可解释性][功能特化] 把预训练 LLM 的每一层拆成「语言 / 逻辑 / 社交 / 世界知识」四个对应人脑认知网络的专家模块，再用一套三阶段课程训练把这种脑式功能特化"逼"出来，得到既可解释、可在推理时按专家路由进行行为调控、又不损失推理性能的模块化语言模型 MICRO。 领域现状：认知神经科学发现…
tags:
  - "ICLR 2026"
  - "可解释性"
  - "功能特化"
  - "脑启发架构"
  - "Mixture-of-Experts"
  - "可控生成"
  - "因果消融"
---

# Mixture of Cognitive Reasoners: Modular Reasoning with Brain-Like Specialization

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=m3jztlHDmG](https://openreview.net/forum?id=m3jztlHDmG)  
**代码**: [cognitive-reasoners.epfl.ch](https://cognitive-reasoners.epfl.ch)  
**领域**: 可解释性 / 模块化语言模型 / 认知神经科学启发  
**关键词**: 功能特化, 脑启发架构, Mixture-of-Experts, 可控生成, 因果消融  

## 一句话总结
把预训练 LLM 的每一层拆成「语言 / 逻辑 / 社交 / 世界知识」四个对应人脑认知网络的专家模块，再用一套三阶段课程训练把这种脑式功能特化"逼"出来，得到既可解释、可在推理时按专家路由进行行为调控、又不损失推理性能的模块化语言模型 MICRO。

## 研究背景与动机
**领域现状**：认知神经科学发现人脑的复杂行为来自一组高度特化的脑网络协作——语言网络、多重需求(逻辑)网络、心智理论(社交)网络、默认模式(世界知识)网络各司其职。反观 LLM，其内部组织是高度非结构化的：虽然有研究发现某些神经元/子网络会选择性激活，但这种特化是**隐式的**，难以解释也难以控制。

**现有痛点**：标准 dense Transformer 与常规稀疏 MoE 都没有把"功能"与"模块"显式对齐——MoE 的专家划分由负载均衡损失驱动，得到的是数据驱动的、语义不明的分工，你无法指着某个专家说"它负责社交推理"，更无法通过开关它来调控模型行为。

**核心矛盾**：可解释/可控的显式特化，与不牺牲整体性能、不破坏大规模指令微调收益之间的张力。一个被强行划分模块的模型很可能在大规模端到端训练中把特化"洗掉"，或者因为模块间干扰而掉点。

**本文目标**：构造一个**显式按脑认知网络划分专家**的语言模型，让专家(1)可解释且因果有意义(消融某专家会在其对应领域基准上大幅掉点)，(2)可在推理时通过路由调控(偏向社交而非逻辑)，(3)在推理基准(GSM8K/BBH)和人类行为对齐(CogBench)上不输甚至超过同规模 baseline。

**核心 idea**：**[脑启发归纳偏置]** 先用极少量(3055 条)按认知领域精心构造的数据给专家和路由器播下特化的种子，再让大规模指令微调在这个已被"塑形"的架构上展开——早期的弱监督归纳偏置足以让功能分解持久存活到训练结束。

## 方法详解

### 整体框架
MICRO 从一个预训练 Transformer backbone 出发，把每一层的整个 block 克隆 N=4 份得到四个专家(类似 parameter upcycling)，并加一个 MLP 路由器对每个 token 做 top-1 分配，使活跃参数量与原模型相当。作者把这种"克隆整个 block(含注意力)"的设计称为 **mixture-of-blocks (MOB)**，区别于只在 FFN 处分专家、共享注意力的常规 MOE——他们发现 MOB 才能在所有规模下都诱导出清晰的功能特化(更低的路由熵、领域一致的路由模式)。四个专家分别对齐语言网络、多重需求(逻辑)网络、心智理论(社交)网络、默认模式(世界知识)网络。真正让这个划分"活"起来的是一套三阶段训练课程。

```mermaid
flowchart LR
    A[预训练 Transformer<br/>每层克隆 4 份专家] --> B[Stage 1<br/>仅训专家<br/>MiCRoSFT 3055 条<br/>token 级确定性路由标签]
    B --> C[Stage 2<br/>冻结模型仅训路由器<br/>同数据 soft top-2]
    C --> D[Stage 3<br/>端到端 SFT<br/>Tülu-3 939k 条]
    D --> E[MICRO<br/>可解释 / 可消融调控 / 不掉点]
```

### 关键设计

**1. Mixture-of-Blocks 而非常规 MoE：把整块计算特化，而不只是 FFN。** 常规稀疏 MoE 把专家限制在 FFN 子层、共享同一套注意力，作者发现这种设计在特定规模下诱导不出稳定的脑式特化。MICRO 改为克隆**整个 Transformer block**(注意力+FFN 全部独立)，每个 token 在每一层由路由器 top-1 选一个专家;为保持效率与原模型相当的活跃参数量,采用类似 Switch Transformer 的 top-1 路由。一个细节是注意力的处理:token 可以注意到序列中所有在先 token,但用的是**当前专家自己产生的 key/value 表示**,而只有被分配给该专家的 token 才继续走该专家的 FFN——这保证了专家既能共享上下文、又在前馈通路上保持独立特化。实验证明 MOB 的路由熵更低、领域一致性更强,是脑式特化能否出现的关键。

**2. 三阶段特化课程：先播种、再校准路由、最后大规模端到端。** 这是诱导并固化特化的核心。**Stage 1(诱导特化)** 只训练专家参数,用 M=3055 条的 MiCRoSFT 数据,每条带 token 级路由标签 $r_{i,t}\in\{1,\dots,N\}$ 把 token 钉死到指定专家做确定性路由,目标是下一 token 预测,让每个专家先获得各自领域的初始归纳偏置。**Stage 2(校准路由器)** 冻结整个模型只训路由器,仍用同一份数据;此阶段改用 top-2 专家的**软混合**而非 top-1,作者发现这样过渡更平滑、路由决策更鲁棒,让路由器在专家已特化的前提下学会"该把哪个 token 派给谁"。**Stage 3(端到端 SFT)** 在 Tülu-3(939k 条)上端到端微调整个模型,虽然这一阶占了绝大部分训练预算,但前两阶播下的特化基本被保留,且专家在各自领域上持续变强——印证了"早期弱归纳偏置→持久功能分解"的核心假设。

**3. MiCRoSFT 数据构造：用 o1 生成推理链 + GPT-4o 句级伪标注。** 特化的种子全靠这 3055 条数据的质量。作者先选了 19 个对应非语言专家(逻辑/社交/世界)认知领域的现有推理数据集,确保覆盖各脑网络已知会调用的多样功能;从三组各随机采 1000 条,用 OpenAI o1 生成详细的逐步推理回答;再用 GPT-4o 把生成推理链中的**每个句子**伪标注归到四个专家之一,句内 token 继承该句的专家标签用于 Stage 1 的确定性路由。语言专家的样本则由 GPT-5 直接生成语法类问答。正是这种"句级语义对齐"的标注让确定性路由有了语义依据。

**4. 神经科学定位器 + 因果消融:用脑科学的工具反过来验证专家。** 这不是训练设计而是验证特化是否"真"的方法,但很关键。一方面用**因果消融**:逐个移除专家观察各领域基准的变化——移除 Logic 专家会让 MATH/GSM8K 大幅掉点(证明其对数值推理因果必要),而移除 Social 专家在这些数学任务上反而小涨(说明它在此处是干扰)。另一方面把神经科学里用来定位人脑语言网络、多重需求网络、心智理论网络的**功能定位器(localizer)** 直接套到 MICRO 上,看 top-10% 选择性单元落在哪个专家:多重需求定位器成功偏向 Logic 专家,语言定位器在浅层偏向 Language 专家、深层偏向 World 专家;ToM 定位器在小模型上效果差但随规模改善,暗示社交能力要先"涌现"才能被定位。

## 实验关键数据

设置:在三个模型家族五个规模上后训练——LLAMA-3.2-{1B,3B}、SMOLLM2-{135M,360M}、OLMO-2-1B;主文报告 LLAMA-3.2-{1B,3B}。两个关键对照:**MOB**(有模块化但无脑式特化)与 **DENSE**(无模块化),均用 2×MiCRoSFT + 1×Tülu-3 的等量数据后训练以保证公平。

### 主实验(推理 & 行为对齐)

| 维度 | 基准 | 结论 |
|------|------|------|
| 推理性能 | GSM8K(0-shot CoT)、Minerva-MATH、MMLU、MMLU-Pro、BBH | MICRO **匹配或超过** MOB baseline;消融掉最不相关专家(数学任务下的 Social)进一步涨点 |
| 人类行为对齐 | CogBench(7 个认知心理学实验、10 个行为指标) | MICRO-LLAMA-1B 的对齐分(SBRE)**优于** MOB 与 Dense |
| 规模差异 | — | LLAMA-3.2-1B 从脑式特化中显著获益;3B 仅在部分基准上相对 baseline 显著 |

行为对齐用作者新提出的有界相对误差相似度 $S_{BRE}=1-\frac{1}{n}\sum_i \mathrm{BRE}_i$,其中 $\mathrm{BRE}_i=|s_i-1|/\max(1,s_i)$,使指标在 $s_i>1$(超人类)时仍被约束在 $[0,1]$,避免超人类分数虚高。

### 消融实验(专家因果性)

| 消融对象 | MATH / GSM8K | 含义 |
|----------|--------------|------|
| 移除 Logic 专家 | 大幅掉点 | 逻辑专家对数值推理因果必要 |
| 移除 Social 专家 | 略有上升 | 社交专家在数学任务上是干扰项 |
| 移除 Language 专家 | 所有基准显著掉点 | 语言专家是底层语言锚定,普遍依赖 |
| MMLU/BBH 各子类 | 部分依赖单一专家,部分需重叠贡献 | BBH 这类混合任务跨多个认知域 |

### 关键发现
- **路由语义连贯**:社交类样本路由到 Social 专家、算术任务路由到 Logic 专家;路由概率与人类标注相关(Social 专家的选择性与"心智状态内容"评分 $r=0.7$)。
- **层级组织自发涌现**:浅层聚焦语言锚定、深层逐渐委派给领域专家——这种层级未被训练显式强加,却与认知神经科学证据一致。
- **特化在大规模训练后存活**:Stage 3 各 checkpoint 上专家使用模式保持一致,证明 3055 条种子数据的归纳偏置足够持久。
- **可推理时调控**:只保留 Social 专家时输出偏社交,只保留 Logic 专家时逻辑推理主导。

## 亮点与洞察
- **把"可解释性"从事后探针变成架构先验**:多数可解释性工作是训练后去探测隐式特化;MICRO 反过来把脑认知网络的划分直接焊进架构并用课程训练逼出来,得到的是**设计即可解释**的专家。
- **3055 条数据撬动持久特化**:用极少量按认知领域对齐的弱监督播种,就能让功能分解存活过 939k 条的大规模 SFT,是"早期归纳偏置 > 数据量"的漂亮例证。
- **脑科学工具与 ML 双向验证**:不只是"灵感来自大脑",而是真的把神经科学的功能定位器和消融范式搬过来验证专家与脑网络的对应,给认知科学提供了可检验的计算假设载体。
- **MOB vs MOE 的洞察**:克隆整块(含注意力)比只分 FFN 更能诱导稳定特化——对模块化架构设计是个有价值的经验结论。

## 局限与展望
- **规模上限未验证**:尚未在 8B 以上 backbone 上验证,增加更多专家对当前架构的影响也未知。
- **ToM 定位偏弱**:心智理论专家的神经科学定位效果差(可能受限于仅 10 对对比刺激的小样本集),社交能力似乎要先涌现才能被可靠定位。
- **专家划分依赖人脑四网络的先验**:作者也指出框架可推广到任意有意义的划分(技术领域/自然语言),还可加入抽象形式推理、直觉物理等新发现的脑网络模块。
- **跨语言网络的神经对齐受限于数据**:验证非语言专家是否真的对齐对应脑区活动,目前缺乏合适的 fMRI 数据集(现有多为 blocked design、难做 item 级分析)。

## 相关工作与启发
- **模块化语言模型**:从稀疏 MoE(Shazeer 2017)、ModuleFormer(用负载均衡+集中损失诱导无标签特化)到多模态/多语言的领域解耦。MICRO 的差异在于**首个显式诱导脑式特化、把专家对齐到成熟认知网络**的模块化 LM。
- **脑启发模型**:此前工作多聚焦视觉皮层层级或语言网络的脑对齐(Schrimpf 2021 等),MICRO 把脑式功能特化扩展到逻辑/社交/世界知识多个认知域。
- **启发**:对做可控生成与可解释性的人来说,"用少量领域对齐数据 + 课程训练把语义模块焊进架构"是一条值得借鉴的路径;对认知科学,则提供了一个可做消融/定位实验的计算模型平台。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 首个显式诱导脑式功能特化的模块化 LM,把神经科学的定位/消融范式真正用于验证专家,概念与方法都新。
- **实验充分度**: ⭐⭐⭐⭐ 跨 3 家族 5 规模、推理+行为对齐双线评估,有 MOB/Dense 双对照与因果消融、定位器验证;但 8B 以上规模、更多专家数未覆盖,ToM 验证偏弱。
- **写作质量**: ⭐⭐⭐⭐⭐ 神经科学动机—架构—课程—验证的叙事环环相扣,图示(路由/消融/定位)清晰,SBRE 等细节交代到位。
- **价值**: ⭐⭐⭐⭐⭐ 同时服务 ML 可解释/可控与认知科学可检验假设两端,且不牺牲性能,是连接两个领域的高价值工作。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] ERMoE: Eigen-Reparameterized Mixture-of-Experts for Stable Routing and Interpretable Specialization](../../CVPR2026/interpretability/ermoe_eigen-reparameterized_mixture-of-experts_for_stable_routing.md)
- [\[ICLR 2026\] Understanding Cross-Layer Contributions to Mixture-of-Experts Routing in LLMs](understanding_cross-layer_contributions_to_mixture-of-experts_routing_in_llms.md)
- [\[ICML 2026\] Equilibrium Reasoners: Learning Attractors Enables Scalable Reasoning](../../ICML2026/interpretability/equilibrium_reasoners_learning_attractors_enables_scalable_reasoning.md)
- [\[ICLR 2026\] On The Geometry and Topology of Representations: the Manifolds of Modular Addition](on_the_geometry_and_topology_of_representations_the_manifolds_of_modular_additio.md)
- [\[ICLR 2026\] Explainable Mixture Models through Differentiable Rule Learning](explainable_mixture_models_through_differentiable_rule_learning.md)

</div>

<!-- RELATED:END -->
