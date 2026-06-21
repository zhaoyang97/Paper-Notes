---
title: >-
  [论文解读] TangleScore: Tangle-Guided Purge and Imprint for Unstructured Knowledge Editing
description: >-
  [ICLR 2026][知识编辑][非结构化知识] 本文提出一个无需依赖具体编辑算法、只由「模型 + 知识样本」决定的内在难度指标 **TangleScore**，用它度量某条知识有多「难改」，并据此设计 **PIPE**（先清除旧知识、再印刻新知识的两阶段编辑框架），在四个不同规模 LLM、两个非结构化编辑基准上把泛化性能平均提升 6.49%。
tags:
  - "ICLR 2026"
  - "知识编辑"
  - "非结构化知识"
  - "编辑难度量化"
  - "知识遗忘"
  - "泛化性"
---

# TangleScore: Tangle-Guided Purge and Imprint for Unstructured Knowledge Editing

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=TbLkgJCGfc](https://openreview.net/forum?id=TbLkgJCGfc)  
**代码**: https://github.com/famoustourist/TangleScore  
**领域**: 知识编辑 / LLM  
**关键词**: 知识编辑, 非结构化知识, 编辑难度量化, 知识遗忘, 泛化性

## 一句话总结
本文提出一个无需依赖具体编辑算法、只由「模型 + 知识样本」决定的内在难度指标 **TangleScore**，用它度量某条知识有多「难改」，并据此设计 **PIPE**（先清除旧知识、再印刻新知识的两阶段编辑框架），在四个不同规模 LLM、两个非结构化编辑基准上把泛化性能平均提升 6.49%。

## 研究背景与动机

**领域现状**：知识编辑（knowledge editing）是替代昂贵重训练、轻量修正 LLM 内部过时/错误事实的主流路线。ROME、MEMIT、AlphaEdit 等 locate-then-edit 方法在**结构化知识**（主谓宾三元组）上已相当成熟，能精准地把某层权重改掉来注入新事实。

**现有痛点**：但现实中约 80% 的知识是**非结构化**的——自由文本、信息密集的长段落，而非干净的三元组。当把结构化方法搬到非结构化场景时，编辑结果出现一个明显裂缝：**accuracy（能背出目标答案）和 generalization（换个问法还能答对）之间差距很大**。更糟的是，作者做 knowledge-wise 分析发现，这种性能崩塌**不是随机的**，而是在不同编辑方法、不同模型上呈现一致的模式——某些特定样本就是「改不动」。

**核心矛盾**：作者把根因归结为**编辑扰动强度与「目标知识—模型」耦合程度之间的失配**。当模型对某条旧知识的内部依赖极深、根深蒂固时，现有方法只是把新答案「贴」上去、并没有真正覆盖掉旧知识的内部表征；模型于是把新知识和一个固定回答模式死记硬背地绑定，一旦换个 paraphrase 就露馅。问题是：现有方法对所有样本一视同仁，用同样的强度去编辑，难样本被「编辑不足」，简单样本又可能被「编辑过度」。

**本文目标**：(1) 找到一个能**事先量化**某条知识「难不难改」的指标；(2) 用这个指标去**自适应调节**编辑强度，对难样本下更猛的手、对简单样本更克制。

**切入角度**：作者观察到失败案例的共同点是「模型输出仍被原知识强烈牵制」。于是假设：存在一个只由 base 模型与知识本身决定、与所用编辑算法无关的**内在抗编辑性**，而且它和编辑后的泛化性能强相关。

**核心 idea**：提出 **TangleScore** 量化这种「知识与模型的纠缠度 / 抗编辑性」，再用它驱动一个**先 purge 后 imprint** 的两阶段编辑框架 PIPE——按难度自适应地决定「该遗忘多狠、该印刻多重」。

## 方法详解

### 整体框架

PIPE（Purge-Imprint Patch Editing）把「编辑一条知识」拆成可被难度调制的两步。给定待编辑样本 $(x_e, y_e)$，**先算 TangleScore** 判断这条知识有多抗改；TangleScore 越高，说明旧知识缠得越紧，**purge rate（清除率）就开得越大**；随后通过**知识清除函数**用梯度上升主动「遗忘」掉模型对旧 key 向量的依赖；清空之后再通过**知识印刻函数**把新知识写进去，且印刻时同样用 TangleScore 调制「学新 vs 保旧」的权重，最后两阶段联合优化，端到端完成对这条知识的编辑。整套方法只编辑选定的第 7 层。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["待编辑知识<br/>(xe, ye)"] --> B["TangleScore<br/>量化内在编辑难度"]
    B --> C["自适应清除率<br/>难度越高 purge 越猛"]
    C --> D["知识清除函数<br/>梯度上升遗忘旧 key"]
    D --> E["知识印刻函数<br/>写入新知识 + 保旧能力"]
    B -.->|σ(TS) 调制 α| E
    E --> F["联合优化<br/>编辑后模型 fϕ"]
```

### 关键设计

**1. TangleScore：用「内部表征位移 ÷ 输出语义差」量化样本的内在抗编辑性**

这是全文的地基，针对的痛点是「现有方法不知道哪条知识难改，只能盲目一刀切」。作者要的指标必须满足：只由 base 模型 + 知识样本决定、与编辑算法无关，且能预测编辑后的泛化好坏。它由两部分组合而成。其一是**内部表征位移** $D_{semantic}$：把原知识与新知识分别拼上定制 prompt 喂进模型，取隐层表征 $r_{old}, r_{new}$ 做平均池化后算余弦距离 $D_{semantic} = 1 - \frac{r_{old}\cdot r_{new}}{\|r_{old}\|\,\|r_{new}\|}$，值越大说明注入新知识后内部表征震动越大、抵抗越强。其二是**输出响应的语义鸿沟**：比较编辑前后模型答案的差异，但不用对 token 级失配敏感、易误判的 KL 散度，而采用可微、稳定的**最优传输 Sinkhorn 距离**，在嵌入空间做整体语义对齐。最终 TangleScore 定义为二者之比：

$$\text{TANGLESCORE}(M) = \frac{D_{semantic}}{\text{Sinkhorn}(\text{Ans}_{old}, \text{Ans}_{new})}.$$

这个**比值形式**刻画的是「模型响应从旧答案切换到新答案的转变率」——分子是内部要付出的表征代价，分母是输出端实际拉开的距离。值越高表示模型在编辑过程中「惯性」越大、越难改。作者用实验验证了它的两个关键性质：编辑前后 TangleScore 分布几乎不变（证明它是样本的**内在属性**，不随模型状态漂移），且 ROUGE/BERT-Score 随 TangleScore 升高而**单调下降**（证明它确实预测泛化性能）。

**2. TangleScore 驱动的自适应清除率：难样本下猛手、简单样本求克制**

有了难度量化，下一步是把它接进编辑强度。痛点在于：用统一的清除策略，难样本会因力度不够而残留旧知识，简单样本又会因力度过头而过拟合。作者让清除率 $PR$ 随 TangleScore 指数级变化：

$$PR = \lambda_{max}^{\gamma\,\text{TANGLESCORE}},\quad \gamma = \frac{\log\lambda_{max}}{\log\lambda_{min}},$$

其中 $\lambda_{max}=0.001$、$\lambda_{min}=0.0001$ 由前期探索实验定下。高 TangleScore（难改）对应更高 $PR$、更激进的遗忘，以压制顽固的旧内容；低 TangleScore（易改）对应更低 $PR$，避免过度扰动导致过拟合。这一步把「难度→强度」的映射显式化，是 PIPE「自适应」二字的来源。

**3. 梯度有界的知识清除函数：用饱和损失稳定地「忘掉」旧 key**

清除阶段要主动让模型遗忘旧知识，作者借鉴 unlearning 的梯度上升思路——反转梯度方向、抬高待遗忘信息的损失。但直接对 MSE 做梯度上升有个致命问题：MSE 对输出的梯度 $\nabla_{\hat y}L_{MSE}=2(\hat y - y)$ 与误差成正比，预测越偏离目标梯度越大，容易数值不稳甚至梯度爆炸。作者改成一个**惩罚「靠近旧知识」而非「鼓励大偏离」的有界函数**：把绝对差裁剪到 $[0,1]$ 保证梯度受控，结合前面的清除率 $PR$，第 $i$ 条知识的清除损失为

$$L_{purge} = \sum_{i=1}^{u} PR \cdot \big(\text{Clamp}(1 - |f_\phi(h_{q,i}) - \tilde k_i^{orig}|)\big)^2,$$

其中 $f_\phi(h_{q,i})$ 是当前 key 输出、$\tilde k_i^{orig}$ 是待清除的原始 key 向量。当模型输出仍贴近旧 key 时损失很高、给出强遗忘信号；当预测远离旧 key 时梯度饱和、不再无界增长。这样既清得干净又训得稳。

**4. TangleScore 调制的知识印刻函数：边写新知识边按难度保住旧能力**

清空之后要把新知识印进去，核心挑战是「学新的同时尽量不破坏已有能力」，否则顺序编辑会越改越崩。作者设计一致性损失 $L_{consistency}$ 在「知识印刻」与「稳定性保持」之间逐 token 加权：

$$L_{consistency} = \sum_{i=1}^{u}\sum_{j=1}^{n-1}\big[\alpha\,\|f_\phi(h_{q,i,j}) - k_{q,i,j}\|^2 + (1-\alpha)\,\|f_\phi(h_{q,i,j}) - f_\theta(h_{q,i,j})\|^2\big],$$

第一项把输出拉向新的目标 key、第二项把输出钉在原模型 $f_\theta$ 上以保旧。关键在权重 $\alpha$ **不是静态/二值**，而是再次由 TangleScore 经 sigmoid 动态调制：$\alpha = \sigma(\text{TANGLESCORE})$。难样本（高 TangleScore）让 $\alpha\to1$，损失偏向知识迁移、用力学新；简单样本（低 TangleScore）让 $\alpha\to0$，偏向保留原能力。再叠加一个保证正确生成新知识的辅助损失 $L_{learn}=\sum_i \|f_\phi(h_{q,i}) - \tilde k_i\|^2$，合成 $L_{imprint}=L_{consistency}+L_{learn}$。

### 损失函数 / 训练策略
为做到端到端编辑，PIPE 把清除与印刻两阶段联合优化，最终目标为

$$f_\phi = \arg\min_\phi (L_{purge} + L_{imprint}).$$

所有编辑统一作用在第 7 层（经消融验证该层最佳）。

## 实验关键数据

### 主实验：非结构化知识编辑（UNKEBench）

斜杠前后分别为原始问题 / paraphrase 问题的结果；FC 为 FactScore（多跳理解），MMLU 衡量通用能力保持。

| 模型 / 方法 | BERT-Score | ROUGE-L | FactScore | MMLU |
|------|------|------|------|------|
| LLaMA3-8B · UNKE | 97.41 / 90.06 | 97.86 / 77.72 | 41.56 | 29.45 |
| LLaMA3-8B · AnyEdit | 98.62 / 91.56 | 95.15 / 79.60 | 48.48 | 28.52 |
| **LLaMA3-8B · PIPE** | **98.64 / 91.71** | **98.44 / 84.07** | **50.91** | **29.51** |
| LLaMA2-7B · UNKE | 96.56 / 90.73 | 95.34 / 76.11 | 40.99 | 29.07 |
| **LLaMA2-7B · PIPE** | **98.57 / 91.73** | **97.39 / 78.47** | **50.12** | 29.65 |
| Qwen2.5-7B · UNKE | 96.84 / 89.92 | 93.64 / 73.89 | 40.12 | 31.28 |
| **Qwen2.5-7B · PIPE** | **97.42 / 91.76** | **97.05 / 80.18** | **42.47** | **31.78** |

结构化方法（ROME/MEMIT/RECT/AlphaEdit）在 UNKEBench 上 ROUGE-L 仅 40 上下，远低于专为非结构化设计的方法。PIPE 相比 UNKE/AnyEdit 主要赢在 **paraphrase 问题的泛化**（ROUGE-L 78→84 类提升）和 **FactScore（多跳理解）**，同时 MMLU 基本不掉，说明它没拿通用能力换编辑效果。论文报告平均泛化提升 **6.49%**。

### 结构化知识编辑（KEBench）

| 方法 | Ori-Acc | Para-Acc | Src-Acc | Tgt-Acc |
|------|------|------|------|------|
| UNKE | 93.59 | 85.34 | 89.28 | 62.56 |
| **PIPE** | **95.89** | **88.23** | **94.47** | **70.11** |

TangleScore/PIPE 不止用于非结构化：在结构化基准上同样把 Ori-Acc、Para-Acc 都推高，且无关知识的 Src/Tgt 准确率也更好，说明它是一个更统一的编辑方案。

### 关键发现
- **TangleScore 是内在属性**：编辑前后其分布几乎不变（Figure 2、5a），证明「某条知识难不难改」由模型+知识决定、不随编辑过程漂移；而且换 AlphaEdit、RECT 等不同编辑器结论一致。
- **难度强相关泛化**：ROUGE-1/2/L 与 BERT-Score 都随 TangleScore 升高单调下降——难样本恰是旧方法泛化崩塌之处，正好是 PIPE 发力的地方。
- **purge 确实压住了旧知识**：把旧答案拼回 prompt 测「模型仍输出旧知识的概率」，PIPE 编辑后该分布整体右移（Figure 5b），即更有效地抑制了对旧知识的依赖。
- **稳定性来自有界损失**：清除函数若直接反转 MSE 会梯度爆炸，裁剪 + 饱和设计是训练能跑稳的关键。

## 亮点与洞察
- **把「编辑难度」做成可事先计算的内在量**：TangleScore 与编辑算法解耦、只看模型和知识本身，这让「编辑前就知道该用多大力」成为可能，是一个可迁移到任意编辑器的诊断工具。
- **比值式难度定义很巧**：内部表征位移做分子、输出语义差（Sinkhorn）做分母，量的是「转变率」而非单一距离，既抓住内部抗性又抓住外部实际改变量。
- **同一个难度信号驱动两端**：清除率与印刻权重 $\alpha=\sigma(\text{TS})$ 都由 TangleScore 调制，难样本「猛遗忘 + 偏学新」、简单样本「轻遗忘 + 偏保旧」，逻辑自洽。
- **先清除再印刻的解耦思路**：把「覆盖旧知识」和「写入新知识」拆成两步，正面回应了「旧知识没被真正覆盖、只是被新答案盖住」这一非结构化编辑的核心病灶。

## 局限与展望
- 作者承认 PIPE 目前**不支持连续/顺序的知识更新流**，只在受控设置下表现好。
- 仅限**文本编辑**，缺乏多模态支持，跨模态更新是后续方向。
- 需要在**更大模型、更多样数据集**上进一步验证泛化性。
- （自评）TangleScore 依赖隐层表征与 Sinkhorn 计算，相对纯权重方法多了一道前置开销；$\lambda_{max}/\lambda_{min}$、编辑层（第 7 层）等都是按经验选的，跨架构是否稳健有待验证。

## 相关工作与启发
- **vs UNKE**: UNKE 把单层所有参数一次性更新来吸收非结构化知识，但对所有样本同等强度，难样本泛化崩塌；PIPE 用 TangleScore 自适应调强度、并显式加 purge 阶段，主要赢在 paraphrase 泛化与多跳理解。
- **vs AnyEdit**: AnyEdit 把长文知识拆成块、逐块编辑关键 token，侧重「怎么分解长知识」；PIPE 不分解，而是从「理解知识难度」入手，二者关注点正交，可设想结合。
- **vs ROME / MEMIT / AlphaEdit**: 这些 locate-then-edit 方法为结构化三元组设计，搬到非结构化文本上 ROUGE/泛化大幅掉点；PIPE 证明同一套 TangleScore + purge/imprint 框架在结构化与非结构化两边都能用。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ TangleScore 把「编辑难度」做成与算法无关的内在可计算量，并用它统一驱动 purge 与 imprint，角度新颖。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 4 个模型、2 个非结构化 + 1 个结构化基准，含 batch/sequential 鲁棒性与多项分析，唯顺序编辑尚未支持。
- 写作质量: ⭐⭐⭐⭐ 从诊断现象到提出指标再到方法，逻辑链清晰；部分公式细节需对照附录。
- 价值: ⭐⭐⭐⭐ 给非结构化知识编辑提供了「先量化难度再自适应编辑」的实用范式与诊断工具。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] FABLE: Fine-grained Fact Anchoring for Unstructured Model Editing](../../ACL2026/knowledge_editing/fable_fine-grained_fact_anchoring_for_unstructured_model_editing.md)
- [\[CVPR 2026\] Attribution-Guided Model Rectification of Unreliable Neural Network Behaviors](../../CVPR2026/knowledge_editing/attribution-guided_model_rectification_of_unreliable_neural_network_behaviors.md)
- [\[ACL 2025\] ChainEdit: Propagating Ripple Effects in LLM Knowledge Editing through Logical Rule-Guided Chains](../../ACL2025/knowledge_editing/chainedit_propagating_ripple_effects_in_llm.md)
- [\[ICLR 2026\] SUIT: Knowledge Editing with Subspace-Aware Key-Value Mappings](suit_knowledge_editing_with_subspace-aware_key-value_mappings.md)
- [\[ICLR 2026\] MoEEdit: Efficient and Routing-Stable Knowledge Editing for Mixture-of-Experts LLMs](moeedit_efficient_and_routing-stable_knowledge_editing_for_mixture-of-experts_ll.md)

</div>

<!-- RELATED:END -->
