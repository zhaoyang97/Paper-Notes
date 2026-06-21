---
title: >-
  [论文解读] Beyond Markovian Drifts: Action-Biased Geometric Walks with Memory for Personalized Summarization
description: >-
  [ICLR 2026][推荐系统][个性化摘要] 本文提出"结构化游走假设"(SWH)质疑个性化摘要中通用的马尔可夫漂移假设(MDH)，并给出轻量编码-解码模型 Walk2Pers——把用户偏好演化建模成带双记忆通道、可分解为幅度与方向(连续 vs 新颖)的动作偏置几何游走，在三个基准上显著超越专用摘要器与大模型。
tags:
  - "ICLR 2026"
  - "推荐系统"
  - "个性化摘要"
  - "用户偏好建模"
  - "几何随机游走"
  - "双记忆通道"
  - "动作条件化"
---

# Beyond Markovian Drifts: Action-Biased Geometric Walks with Memory for Personalized Summarization

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=HvOKarTubb](https://openreview.net/forum?id=HvOKarTubb)  
**代码**: 待确认  
**领域**: 个性化推荐 / 个性化摘要  
**关键词**: 个性化摘要, 用户偏好建模, 几何随机游走, 双记忆通道, 动作条件化  

## 一句话总结
本文提出"结构化游走假设"(SWH)质疑个性化摘要中通用的马尔可夫漂移假设(MDH)，并给出轻量编码-解码模型 Walk2Pers——把用户偏好演化建模成带双记忆通道、可分解为幅度与方向(连续 vs 新颖)的动作偏置几何游走，在三个基准上显著超越专用摘要器与大模型。

## 研究背景与动机
- **领域现状**：个性化文档摘要要帮读者聚焦"感兴趣内容"，而这是一个主观且随时间变化的量。新闻推荐与摘要领域的主流做法（图扩散 RWR/Personalized PageRank、NAML/NRMS/EBNR 这类短记忆神经编码器、提示式中型 LLM 摘要器）几乎都假设用户偏好沿交互图做**无记忆或短记忆的随机游走**——即每个新状态主要取决于最近一次交互。
- **现有痛点**：作者把这些做法统一归纳为**马尔可夫漂移假设(MDH)**：历史被压成最后状态/种子向量/隐藏态/提示窗口，长程动作动态被覆盖。图扩散没有动作语义、神经编码器把长历史压成浅记忆、LLM 受提示长度硬约束且无持久的强化/抑制。在 PENS 这类含点击/跳过日志的数据上，用户兴趣随时间在细粒度子话题间漂移，长历史塞进提示反而让 SOTA LLM 性能退化。
- **核心矛盾**：偏好演化既需要**持久、不对称地记住"喜欢什么(点击强化)/不喜欢什么(跳过抑制)"**，又需要区分演化是"沿原轨迹延续(连续性)"还是"转向新方向(新颖性)"——而 MDH 这两点都做不到，它把一切收敛到一步漂移。
- **本文目标**：检验 MDH 在个性化摘要任务上是否成立，并给出一个比 MDH 更忠实、且可解释、轻量的替代建模。
- **核心 idea**：**结构化游走假设(SWH)**——把每次交互(click/skip/summarize)引起的偏好状态更新分解为 **(i) 幅度**(动作推动强度) + **(ii) 朝向**(连续 vs 新颖)，并辅以**双记忆通道**(强化/抑制)和**摘要请求专属漂移项**，理论上逼近一阶动作条件化核，实践上以 Walk2Pers 实例化。

## 方法详解

### 整体框架
方法先把用户历史抽象成两层结构：底层 **用户交互图(UIG)** 记录 user/document/summary 三类节点与 click/skip/summarize/summGen 动作边；上层把每个交互压成"行为对偶" b-cell `b=⟨动作, 尾节点⟩`并用 nextBehavior 边串成轨迹。Walk2Pers 是一个 T5-base 编码-解码框架：SWH-Encoder 沿 b 层做带记忆的几何游走得到上下文化嵌入，Predictor 预测下一个 b-node，Inverse Approximator 抽出潜在摘要意图(s-node)，Contextualizer 用交叉注意力融合摘要意图、用户历史与查询文档，最后微调 T5 解码器(仅顶层)生成个性化摘要。

```mermaid
flowchart LR
  A[用户交互日志<br/>click/skip/summarize] --> B[UIG → b-cell 轨迹]
  B --> C[SWH-Encoder<br/>双记忆+几何步]
  C --> D[Predictor<br/>预测下一 b-node]
  D --> E[Inverse Approximator<br/>抽潜在 s-node]
  E --> F[Contextualizer<br/>交叉注意力融合历史/文档]
  F --> G[T5 解码器<br/>顶层微调]
  G --> H[个性化摘要]
```

### 关键设计

**1. 结构化游走的统一更新式：把"一步漂移"换成"几何步+记忆+漂移"三元分解。** MDH 假设下一状态只依赖前一状态 $e^{(t+1)}_{b,u}=f(e^{(t)}_{b,u}, a^{(t)}, q)+\epsilon^{(t)}$，长历史被折叠进近因先验 $q$。SWH 把这条更新式重写成一个可解释的加性家族：

$$e^{(t+1)}_{b,u}=e^{(t)}_{b,u}+\underbrace{\mathrm{mag}(a^{(t)})\big(\cos\theta(a^{(t)})\,u^{(t)}+\sin\theta(a^{(t)})\,o^{(t)}\big)}_{\Phi:\ \text{几何步, 连续 vs 新颖}}+\underbrace{\Psi(h^+_t,h^-_t)}_{\Psi:\ \text{双记忆}}+\underbrace{\delta\cdot\mathbb{I}[a^{(t)}=\text{summGen}]}_{\Delta:\ \text{摘要漂移}}$$

其中 $\mathrm{mag}(\cdot)$ 控制步长(单次点击=小步，重复点击=大步)，旋转角 $\theta$ 在动量轴 $u^{(t)}$(延续既有兴趣)与正交新颖轴 $o^{(t)}$(转向新方向)之间插值——$\theta\approx0$ 偏连续、$\theta$ 大偏新颖。这个分解借鉴了 JODIE 的轨迹动态嵌入与 RotatE/ChronoR 的角度关系建模，但加上了显式动作偏置与记忆，把随机游走泛化成受状态-动作-记忆控制的扩散。

**2. 双记忆通道 Ψ 与摘要漂移 Δ：不对称地强化兴趣、抑制反感。** Walk2Pers 用两条记忆通道线性组合表示历史 $h^{(t)}=\omega^{(t)}h^{(+,t)}+(1-\omega^{(t)})h^{(-,t)}$($\omega$ 可学习)，正通道累积点击的强化信号、负通道累积跳过的抑制信号，更新规则刻意做成不对称：

$$h^{(+,t_i)}=h^{(+,t_{i-1})}+m^{(t_i)}\odot c^{(t_i)}_{tl};\qquad h^{(-,t_i)}=h^{(-,t_{i-1})}\odot(1-m^{(t_i)})+c^{(t_i)}_{tl}$$

门控 $m^{(t_i)}=\mathrm{SoftMax}(W_h h^{(t_{i-1})}+W_c c^{(t_i)}_{tl})$。当动作是 summGen(请求摘要)时触发漂移向量 $\Delta^{(t)}=(I-e^{(t-1)}_{tl})\cdot e^{(t)}_{tl}$，把偏好状态推向更凝练的表示，因为 summarize→summGen 是更强的正信号。这样反复点击气候政策会让正通道累积、靠近动量轴；反复跳过娱乐内容会压低负通道、下调该方向；密集报告后发起摘要请求则触发 Δ。

**3. 动作偏置 b-cell 与几何步落地，双目标监督让游走"既忠实又前瞻"。** 每个 d/s 节点用 T5-base 初始化，动作编成 4 维 one-hot，b-cell 把动作门控与尾节点内容融合 $c^{(t_i)}_{tl}=\tanh(f^{(a,t_i)}\odot e^{(t_i)}_{tl})$，其中 $f^{(a,t_i)}=\mathrm{AGD}(e_a,t_i)\odot h^{(t)}$ 借用基线 AGD 的动作门(click 增强、skip 削弱、summGen 锚定到摘要节点)。几何步按式(5) $\widetilde{e}^{(t)}_{b,u}=e^{(t)}_{b,u}+\mathrm{mag}^{(t)}(\cos\theta^{(t)}u^{(t-1)}+\sin\theta^{(t)}o^{(t)})$ 实现连续-新颖权衡。编码器用两个互补目标监督：下一节点预测头 $\mathcal{L}_{next}$ 让游走具预测性，位置分类对齐项 $\mathcal{L}_{align}$ 保证轨迹中每个中间 b-node 都能从上下文化嵌入恢复，合成 $\mathcal{L}_{enc}=\alpha\mathcal{L}_{align}+(1-\alpha)\mathcal{L}_{next}$($\alpha=0.6$ 防级联累积)。

**4. 用户感知解码(T5-UCA)：让同一文档透过不同用户的偏好棱镜被改写。** 解码器复用 T5-base，提供两个变体：T5-CA 用交叉注意力把查询文档嵌入与潜在摘要意图(s-node)上下文化，注入"摘要意图"但文档表示与用户历史无关；T5-UCA 在此基础上再用用户轨迹状态门控查询文档嵌入——压制与负记忆 $h^-$ 对齐的方面(反复跳过的话题)、放大与正记忆 $h^+$ 对齐的方面，产出"用户感知文档向量"，使 Alice 和 Bob 看到的同一篇文档被不同改写。解码总目标 $\mathcal{L}_{dec}=\mathrm{Average}(\mathcal{L}_{gen},\mathcal{L}_{enc})$，先端到端训 6 轮、再冻结编码器只微调 T5 解码顶 6 层 18 轮。

## 实验关键数据

### 主实验表格
下一 b-node 预测任务(PENS，151 候选)，衡量对真实用户行为的预测能力：

| 类别 | 模型 | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|
| MDH | NAML | 0.498 | 0.001 | 0.0004 | 0.0007 |
| MDH | NRMS | 0.499 | 0.0009 | 0.0002 | 0.0004 |
| MDH | EBNR | 0.499 | 0.0009 | 0.0003 | 0.0005 |
| MDH | SMD (本文) | 0.415 | 0.094 | 0.052 | 0.065 |
| MDH | AGD (本文) | 0.446 | 0.113 | 0.069 | 0.073 |
| SWH | Walk2Pers-Enc. w/o 几何步 | 0.474 | 0.121 | 0.082 | 0.132 |
| SWH | **Walk2Pers-Enc. Full** | **0.532** | **0.23** | **0.198** | **0.249** |

PENS 个性化摘要(PerSEval 三指标，越高越个性化)：

| 类别 | 模型 | PSE-JSD | PSE-SU4 | PSE-METEOR |
|---|---|---|---|---|
| Oracle(注入线索) | BigBird-Pegasus | 0.253 | 0.143 | 0.168 |
| LLM(2-shot 历史) | DeepSeek-14B | 0.248 | 0.094 | 0.097 |
| LLM(2-shot 历史) | Gemini-2.5-Flash | 0.222 | 0.104 | 0.124 |
| 专用摘要器 ~MDH | GTP | 0.024 | 0.017 | 0.019 |
| 专用摘要器 ~MDH | PENS-NAML-T1 | 0.021 | 0.014 | 0.016 |
| MDH 编码器(本文) | AGD + T5-UCA | 0.286 | 0.214 | 0.248 |
| SWH(本文) | **Walk2Pers Full + T5-UCA** | **0.452** | **0.383** | **0.449** |

### 消融实验表格
关键组件逐步剥离(摘自上表内的变体对比)：

| 变体 | PSE-JSD | PSE-SU4 | PSE-METEOR |
|---|---|---|---|
| AGD + T5-UCA(纯 MDH) | 0.286 | 0.214 | 0.248 |
| Walk2Pers w/o 几何步 + T5-UCA | 0.306 | 0.334 | 0.321 |
| Walk2Pers Full + T5-CA | 0.418 | 0.341 | 0.422 |
| Walk2Pers Full + T5-UCA | 0.452 | 0.383 | 0.449 |

跨域泛化(OpenAI-Reddit，非新闻多域)：Walk2Pers Full+T5-UCA 取得 0.339/0.303/0.350，比最佳 2-shot LLM(DeepSeek-14B 0.243/0.095/0.109)平均高约 0.19。

### 关键发现
- **RQ1：MDH 不够。** 短记忆神经编码器(NAML/NRMS/EBNR)的 AUC 徘徊在随机水平(≈0.5)、排序指标几近归零(MRR≤0.001)，说明压缩隐藏态几乎不携带预测信号。
- **RQ2：SWH 组件系统性增益且互补。** 加双记忆+漂移(w/o 几何步)已超最佳 AGD(AUC +0.028、nDCG@10 +0.059)；再加几何幅度-朝向步带来大跳(相对 AGD：AUC/MRR/nDCG@5/10 分别 +0.086/+0.117/+0.176)。
- **RQ3：端到端超越专用摘要器与 LLM。** 相对 GTP/SP 等微调 MDH 专用器平均绝对增益约 0.41，相对所有 LLM 平均增益约 0.22(对最佳 2-shot DeepSeek-14B 高 0.20/0.29/0.35)，提示链式提示甚至落后于 MDH 基线。
- **跨任务可迁移：** 仅为摘要任务训练的 Walk2Pers 编码器迁到 MIND 新闻推荐榜，超最佳基线(Fastformer+PLM-NR-Ensemble) MRR +1.2、nDCG@5 +1.8、nDCG@10 +3.5。

## 亮点与洞察
- **把"假设"作为研究对象**：不直接堆模型，而是显式命名并检验主流隐含假设(MDH)，再提出可证伪的替代假设(SWH)，方法论上干净。
- **几何分解带来可解释性**：幅度(推动强度)与朝向(连续/新颖)的显式分解，让"偏好为何这么变"可读，区别于不透明的隐藏态。
- **不对称双记忆**：把"喜欢"和"不喜欢"放进两条带不同更新规则的通道，正面契合点击=强化、跳过=抑制的直觉，且能跨长历史持久保留。
- **轻量却强**：T5-base 级别编码-解码框架在个性化指标上大幅压过 13B-235B 的 LLM 与 oracle 注入，说明结构先验在该任务上比规模更关键。

## 局限与展望
- 评测主要绑定 PerSEval 这一与人类相关性强的个性化指标，但绝对摘要质量(ROUGE 等)与可读性的权衡讨论较少，强个性化是否牺牲流畅度未充分展开。
- 动作集合固定在 click/skip/summarize/summGen 四类 one-hot，更细粒度或隐式反馈(停留时长、滚动)如何纳入几何步未探讨。
- 几何步与角度参数化引入额外超参($\alpha$、$\omega$、$\theta$ 的学习)，对训练稳定性与跨数据集迁移的敏感度仅在附录提及。
- PersonalSum-EN 仅 700 条且经机器翻译(M2M-100)，多语种与小样本场景的结论需谨慎外推。

## 相关工作与启发
- **个性化摘要评测**：EGISES、PerSEval(Dasgupta 等 2024)推动个性化感知指标，本文采用 PerSEval 因其与人类判断强相关。
- **个性化摘要模型**：GSUM/CTRLSum/TMWIN/Tri-Agent 依赖静态画像，PENS/GTP/SCAPE 引入动态新闻推荐编码器，但都落在 MDH 范畴；few-shot LLM 受提示长度与记忆约束。
- **图随机游走推荐**：S-Walk(重构转移核)、D-RDW(多样化重启路径缓解流行度偏置)作为轻量图基线，但仍是无记忆扩散。
- **动态嵌入与角度关系建模**：JODIE 的轨迹嵌入、RotatE/ChronoR 的旋转关系给了几何步分解的灵感。
- **启发**：在序列推荐/会话推荐里，"显式记忆通道 + 几何方向分解"或可替代纯注意力/GRU 的隐式聚合，尤其在需要长程不对称反馈(强化/抑制)的场景。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 把主流隐含假设显式化并提出可证伪替代(SWH)，几何幅度-朝向分解+不对称双记忆的组合在个性化摘要里少见。
- **实验充分度**: ⭐⭐⭐⭐ 三数据集+下一节点预测/端到端摘要/跨域/跨任务多角度验证，对照 MDH 基线、专用摘要器、6 个 LLM 与 oracle，消融清晰；个性化指标单一略减分。
- **写作质量**: ⭐⭐⭐⭐ 假设-公式-实例化层层递进，MDH↔SWH 对照表与统一更新式表达清楚。
- **价值**: ⭐⭐⭐⭐ 用轻量模型大幅超越大模型，且编码器可迁移到新闻推荐，对个性化序列建模有实际启发。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents](../../ACL2026/recommender/from_recall_to_forgetting_benchmarking_long-term_memory_for_personalized_agents.md)
- [\[ICLR 2026\] More Than What Was Chosen: LLM-based Explainable Recommendation Beyond Noisy User Preferences](more_than_what_was_chosen_llm-based_explainable_recommendation_beyond_noisy_user.md)
- [\[ICLR 2026\] Low-pass Personalized Subgraph Federated Recommendation](low-pass_personalized_subgraph_federated_recommendation.md)
- [\[ICLR 2026\] ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation](propersim_developing_proactive_and_personalized_ai_assistants_through_user-assis.md)
- [\[AAAI 2026\] Evaluating LLMs for Police Decision-Making: A Framework Based on Police Action Scenarios](../../AAAI2026/recommender/evaluating_llms_for_police_decision-making_a_framework_based_on_police_action_sc.md)

</div>

<!-- RELATED:END -->
