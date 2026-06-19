---
title: >-
  [论文解读] Dynamic Important Example Mining for Reinforcement Finetuning
description: >-
  [CVPR 2026][Reasoning][强化微调] DIEM 在 RFT（GRPO/PPO 等）的每一步训练里，用「单样本梯度与 batch 总梯度的内积」实时估计每条样本对当前策略改进的边际贡献，再解一个保持梯度模长不变的约束优化问题给样本重加权，几乎零额外开销（+1.3% 时间）就让多模态推理 benchmark 平均提升 1–6 个点。
tags:
  - "CVPR 2026"
  - "Reasoning"
  - "强化微调"
  - "数据选择"
  - "梯度对齐"
  - "样本重加权"
  - "课程学习"
---

# Dynamic Important Example Mining for Reinforcement Finetuning

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Tan_Dynamic_Important_Example_Mining_for_Reinforcement_Finetuning_CVPR_2026_paper.html)  
**代码**: https://github.com/hrtan/DIEM  
**领域**: LLM推理 / 强化微调（RFT）  
**关键词**: 强化微调, 数据选择, 梯度对齐, 样本重加权, 课程学习  

## 一句话总结
DIEM 在 RFT（GRPO/PPO 等）的每一步训练里，用「单样本梯度与 batch 总梯度的内积」实时估计每条样本对当前策略改进的边际贡献，再解一个保持梯度模长不变的约束优化问题给样本重加权，几乎零额外开销（+1.3% 时间）就让多模态推理 benchmark 平均提升 1–6 个点。

## 研究背景与动机
**领域现状**：强化微调（Reinforcement Fine-Tuning, RFT）已经成为提升大模型（尤其是多模态大模型）推理能力的主流后训练范式——用 GRPO、PPO 这类策略梯度算法，让模型直接从 reward 信号里学习，而不是单纯模仿监督数据。RFT 的效果高度依赖「训练数据怎么用」：选哪些样本、给多大权重，直接决定优化稳定性和最终的泛化推理能力。

**现有痛点**：绝大多数数据中心式（data-centric）的 RFT 方法把样本重要性当成**固定的**。静态方法（如 LIMR 看 reward 趋势、HVS 看 reward 方差）在训练开始前一次性挑好子集；动态方法（如 PCL 训一个辅助 value model 评难度、SPEED-RL 用 pass rate 优先中等难度题）虽然会随训练调整顺序，但都依赖**外部启发式指标**。

**核心矛盾**：RFT 本质是非平稳（non-stationary）的——同一条样本在训练早期和后期对策略的价值完全不同。而上述启发式指标有两个根本缺陷：① 它们站在策略**外部**打分，反映不了策略自身此刻对这条样本的「偏好/契合度」；② 它们量不出样本对这一步策略更新的**真实边际贡献**。把一个会随训练漂移的量当常数用，自然会导致次优更新。

**本文目标**：把数据选择从「一次性预处理」变成「嵌进优化循环、随策略动态自调」的内生组件，并且要做到（a）有理论支撑、（b）几乎不增加计算开销、（c）能即插即用兼容各种 RFT 算法。

**切入角度**：与其外部打分，不如直接问——「如果把样本 $z$ 从这一步更新里拿掉，整个 batch 的总 reward 会变好还是变差？」这个「留一法」式的边际贡献天然贴合策略本身，但精确算它要做 $|B_t|$ 次完整梯度更新，不可行。作者的关键观察是：这个边际贡献可以用**已经算好的梯度**做一阶近似，不花额外的反传成本。

**核心 idea**：用「单样本梯度 $G_z$ 与 batch 总梯度 $G_{B_t}$ 的内积」作为样本即时重要性的低成本代理，再把重加权写成「最大化加权重要性、同时锁死总梯度模长」的约束优化，闭式求解后即可重塑梯度方向，零额外训练。

## 方法详解

### 整体框架
DIEM 不改动任何 RFT 算法的主干，而是在每个优化步的标准前向-反传之后、参数更新之前，插入两个轻量步骤：先**测每条样本的动态重要性**，再**按重要性重加权**得到新的梯度方向，最后用这个加权梯度更新策略。输入是当前 minibatch $B_t$ 在标准 RFT 里已经算出的「每样本梯度 $G_z$ + batch 总梯度」，输出是重加权后的总梯度 $G_{\text{weighted}}$。因为它只复用现成梯度、只对一个 $N\times N$（$N$ 为 minibatch 大小）的小矩阵求逆，所以整步开销可忽略。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["minibatch B_t<br/>标准 RFT 前向+反传"] --> B["每样本梯度 G_z<br/>+ batch 总梯度 G_Bt"]
    B --> C["梯度对齐重要性估计<br/>Î(z)=η⟨G_z, G_Bt⟩"]
    C --> D["约束重加权<br/>max IᵀW s.t. 模长不变<br/>闭式解 W*=P⁻¹I·√C/√(IᵀP⁻¹I)"]
    D -->|裁掉负权重 max(0,·)| E["加权梯度<br/>G_weighted = W*ᵀG"]
    E --> F["策略更新 θ_{t+1}"]
```

### 关键设计

**1. 留一式边际贡献：把「样本价值」定义成可解释的策略改进量**

要动态地、贴合策略本身地衡量样本价值，作者先给出一个**精确但昂贵**的定义：样本 $z$ 在第 $t$ 步的重要性 $I_t(z)$，等于「用整个 batch $B_t$ 更新一步后的总 reward」减去「把 $z$ 抽掉、用 $B_t\setminus\{z\}$ 更新后的总 reward」：

$$I_t(z) = J\!\left(\theta^{\text{update}}_{B_t},\, B_t\right) - J\!\left(\theta^{\text{update}}_{B_t\setminus\{z\}},\, B_t\right)$$

$I_t(z)>0$ 说明把 $z$ 放进更新让策略在当前 batch 上变好（有益），$<0$ 说明 $z$ 反而把策略往坏处带（此刻有害），接近 0 则可忽略。这个定义的好处是它不再是「难度/熵」这类外部启发式，而是直接、有方向地度量「这条样本对策略改进的真实边际效用」，且天然随训练步变化——这正是后面重加权的依据。

**2. 梯度对齐估计器：把昂贵的留一法降成一次内积**

直接算 $I_t(z)$ 要对 batch 里每条样本各做一次完整梯度更新与评估（至少 $|B_t|$ 次），在高吞吐训练里根本跑不动。DIEM 给出一阶近似（Proposition 1）：样本的即时价值 $\approx$ 它的梯度方向有多「拥护」batch 的集体更新方向，用内积度量：

$$\hat I_t(z) = \eta_t\,\big\langle G^{(t)}_z,\, G^{(t)}_{B_t} \big\rangle$$

其中 $G^{(t)}_z$ 是单样本 $z$ 贡献的策略梯度，$G^{(t)}_{B_t}$ 是整个 batch 的聚合梯度，二者在标准反传里都已经算出来了，所以这一步几乎免费。直觉很清楚：内积大正值 = 该样本梯度与集体方向高度一致，是「代表性、高效用」样本，能加速收敛；负值 = 梯度指向集体的反方向，是噪声或当下有害样本。作者还给了误差界（Proposition 2）：在 log-likelihood 满足 $\ell$-Lipschitz、advantage 上界 $A_{\max}$ 的温和假设下，$|I_t(z)-\hat I_t(z)|\le O(\eta_t\ell^2 + 2\eta_t\ell A_{\max})$，且不依赖凸性或「接近稳定点」的假设，因而适配 RFT 早期高度非凸、非平稳的训练区。⚠️ 完整推导在附录，此处以原文公式为准。

**3. 模长守恒的约束重加权：放大高价值样本但不改更新步幅**

有了重要性向量 $I\in\mathbb{R}^N$，怎么用？最朴素的做法（直接按 $I$ 缩放）会改变总梯度的模长，等于偷偷改了有效学习率，破坏优化稳定性。DIEM 把重加权写成一个约束优化：最大化加权效用，同时强制重加权后的总梯度 L2 模长等于原始未加权梯度的模长：

$$\max_{W}\; I^\top W \quad \text{s.t.}\quad \big\|W^\top G\big\|_2 = \big\|\mathbf{1}^\top G\big\|_2$$

目标项 $I^\top W$ 隐式地把权重倾斜给高影响样本；约束项锁死更新步幅，保证「方向变了、步长不变」。用拉格朗日乘子法解驻点，得到拟闭式解（$P=GG^\top$ 是梯度的 Gram 矩阵，$C=\|\mathbf{1}^\top G\|^2$ 是原总梯度模长平方）：

$$W^* = \frac{P^{-1}I\,\sqrt{C}}{\sqrt{I^\top P^{-1}I}}$$

只需对 $N\times N$ 的小矩阵 $P$ 求一次逆，开销相对动辄几分钟的一步 RFT 完全可忽略。解出的 $W^*$ 可能含负值（样本真有害，或被估计误差误判为低效用），于是做非负后处理 $W^*\leftarrow\max(0, W^*)$，只让建设性贡献进入更新，最终加权梯度 $G_{\text{weighted}}=W^{*\top}G$ 拿去更新策略 $\theta_{t+1}=\theta_t+\eta_t G_{\text{weighted}}$。这一步把「测重要性」和「稳定优化」缝合在一起，是 DIEM 区别于简单 reweight 的核心。

### 损失函数 / 训练策略
DIEM 不引入新 loss，沿用底层 RFT 算法（GRPO 为主，advantage 采用组内标准化 $A(s,a_i)=\frac{r_i-\text{mean}(r)}{\text{std}(r)}$，带 PPO 式 clip 与 KL 正则）。它唯一改变的是每步用 $W^*$ 重塑梯度，整体流程见 Algorithm 1：算每样本梯度 → 内积得 $I$ → 解 $W^*$ → 裁负 → 加权更新。整个机制对策略更新规则（PPO/GRPO/TRPO/Reinforce++）是 agnostic 的。

## 实验关键数据

### 主实验
在 Qwen2.5-VL-7B / 32B 上，用 MM-Eureka 语料随机采 52K 多模态样本做 RFT，对比静态（LIMR、HVS）与动态（PCL、SPEED-RL）数据选择基线，在 6 个多模态推理 benchmark 上评测（平均分）：

| 基座 | 方法 | MathVista | MathVerse | MMMU | Average |
|------|------|-----------|-----------|------|---------|
| 7B | Vanilla RFT | 74.1 | 51.7 | 57.1 | 59.1 |
| 7B | SPEED-RL（次优） | 74.9 | 47.1 | 59.2 | 60.0 |
| 7B | **DIEM** | **76.9** | **53.0** | 59.2 | **61.8** |
| 32B | Vanilla RFT | 75.6 | 52.7 | 69.0 | 64.9 |
| 32B | SPEED-RL（次优） | 75.1 | 53.9 | 70.6 | 65.6 |
| 32B | **DIEM** | **76.9** | **58.0** | **71.9** | **67.3** |

7B 上 DIEM 平均 61.8%，比 Vanilla RFT +3.6 点、比最强基线 SPEED-RL +1.8 点，甚至略超 GPT-4o（60.9%）；32B 上达 67.3%，6 个 benchmark 全部第一。

跨 RFT 算法泛化（MathVerse / Qwen2.5-VL-32B），DIEM 即插即用且全面领先：

| 方法 | GRPO | PPO | Reinforce++ | TRPO |
|------|------|-----|-------------|------|
| Vanilla RFT | 52.7 | 53.6 | 51.2 | 50.5 |
| PCL | 54.1 | 55.0 | 54.8 | 55.1 |
| **DIEM** | **58.0** | **57.3** | **56.9** | **57.2** |

GRPO 下比次优 PCL 高 3.9 点（58.0 vs 54.1），其余优化器上也稳定领先。

### 消融实验
MathVerse / Qwen2.5-VL-32B，完整模型 58.0：

| 配置 | 性能 | 说明 |
|------|------|------|
| DIEM（完整） | 58.0 | — |
| 重要性换 Random value | 53.0 | 掉 5.0，证明分数有效 |
| 重要性换 Pass@k score | 53.2 | 掉 4.8 |
| 重要性换 Difficulty score (PCL) | 52.1 | 掉 5.9，最大跌幅 |
| 重要性换 Pass@k 到中位数距离 | 54.9 | 改善但仍差 3.1 |
| 重加权换 NULL-operation（去掉） | 55.4 | 掉 2.6 |
| 重加权换 Softmax 归一化 | 56.4 | 掉 1.6 |

### 关键发现
- **梯度对齐分数 > 所有启发式指标**：把动态影响分换成 Random/Pass@k/Difficulty 都掉 4.8–5.9 点；即便把这些启发式分数转成「到中位数距离」也只补回部分（最高到 54.9），仍打不过 DIEM，说明「基于策略内部梯度」的打分本质上优于外部启发式。
- **模长守恒重加权不可省**：直接去掉重加权（NULL）掉 2.6 点，换成普通 Softmax 归一化也掉 1.6 点——专门设计的约束重加权才能把重要性分数的收益最大化。
- **几乎零开销**：Vanilla RFT(GRPO) 70.3 小时，加 DIEM 仅 71.2 小时（+约 1.3%），而 PCL 79.1h、SPEED-RL 94.6h、LIMR/HVS 都要 122.0h——因为 DIEM 不训练任何代理模型，纯复用已算梯度。
- **自发涌现课程学习**：用 Pass@k 把样本分 Hard/Medium/Easy，追踪 DIEM 分配的权重轨迹（Fig.3）发现——训练早期 Easy/Medium 权重高，随训练推进 Easy 权重快速下降、Hard 权重持续上升（带震荡）。这种「由易到难」的课程是模型动态自发形成的，而非 PCL/SPEED-RL 那样靠人工硬编码启发式。

## 亮点与洞察
- **「免费的边际贡献估计」是最巧的地方**：留一法本来要 $O(|B_t|)$ 次完整更新，作者用一阶泰勒近似把它折成单样本梯度与 batch 梯度的一次内积，而这两个量在标准反传里早就算好了——等于零成本拿到 influence function，且误差界比监督学习里的经典 influence 结果还紧。
- **「方向变、模长不变」的约束设计可迁移**：把重加权约束在「保持总梯度 L2 模长」上，等于解耦了「调整更新方向」与「控制有效学习率」——这个思路可以迁移到任何想重加权梯度但又怕破坏优化稳定性的场景（如带噪标签训练、多任务梯度合并）。
- **课程学习从「人工设计」变「自发涌现」**：DIEM 没写任何 easy-to-hard 规则，课程却自己长出来，说明「贴合策略本身的边际贡献」天然就编码了「此刻该学什么」。

## 局限与展望
- **一阶近似的适用边界**：误差界依赖学习率 $\eta_t$ 和 Lipschitz 常数 $\ell$，在大学习率或 reward 极不平滑时近似可能变差，论文未给这种极端区的实证（⚠️ 推导细节在附录，未在正文充分展开）。
- **Gram 矩阵求逆的可扩展性**：闭式解需对 $N\times N$ 的 $P$ 求逆，minibatch 不大时开销可忽略，但若未来想在极大 batch 或 token 级粒度上做重加权，$P^{-1}$ 的成本与数值稳定性需要重新评估。
- **评测域偏窄**：实验全在多模态数学/通用推理 benchmark 上，未验证纯文本 LLM 推理（如代码、长链数学证明）或对齐类 RLHF 任务上的表现，泛化主张主要靠跨优化器实验支撑。
- **负权重裁剪的粗糙性**：$\max(0,W^*)$ 把所有负权样本一刀切到 0，可能误伤「被估计误差误判」的有用样本，更细的软处理或许还能再压一点。

## 相关工作与启发
- **vs 静态选择（LIMR / HVS）**：它们在 RFT 前一次性按 reward 趋势/方差挑子集，假设样本重要性全程恒定；DIEM 每步动态重估，直接打掉「常数重要性」这个根本假设，且省掉了它们训练代理模型的 122 小时开销。
- **vs 动态启发式（PCL / SPEED-RL）**：PCL 训外部 value model 评难度、SPEED-RL 用 pass rate 优先中等难度，二者都是策略**外部**的启发式；DIEM 用策略**内部**的梯度对齐分，更贴合「策略此刻真正偏好什么」，消融里换成这些启发式分数都掉 4.8–5.9 点。
- **vs 经典 influence function**：传统 influence 度量常需凸性假设或要求模型接近稳定点；DIEM 的梯度对齐估计器不依赖这些，专为 RFT 早期非凸、非平稳区设计，误差界反而更紧。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把 influence function 的留一思想一阶近似成「免费内积」并配上模长守恒重加权，角度新且自洽，但单项技术都能在已有文献找到根。
- 实验充分度: ⭐⭐⭐⭐ 两种规模基座 × 6 benchmark × 4 种 RFT 算法 + 速度测试 + 课程可视化，覆盖全面；略欠纯文本/对齐任务验证。
- 写作质量: ⭐⭐⭐⭐ 动机—定义—近似—约束解的逻辑链清晰，公式与算法表完整，可惜核心推导都甩给附录。
- 价值: ⭐⭐⭐⭐ 即插即用、近零开销、稳定涨点 1–6%，对做 RFT 数据中心优化的人有直接实用价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models](../../ICLR2026/llm_reasoning/dynamics-predictive_sampling_for_active_rl_finetuning_of_large_reasoning_models.md)
- [\[CVPR 2026\] Scaling Agentic Reinforcement Learning for Tool-Integrated Reasoning in VLMs](scaling_agentic_reinforcement_learning_for_tool-integrated_reasoning_in_vlms.md)
- [\[ICML 2026\] DyCon: Dynamic Reasoning Control via Evolving Difficulty Modeling](../../ICML2026/llm_reasoning/dycon_dynamic_reasoning_control_via_evolving_difficulty_modeling.md)
- [\[ACL 2026\] DELTA: Dynamic Layer-Aware Token Attention for Efficient Long-Context Reasoning](../../ACL2026/llm_reasoning/delta_dynamic_layer-aware_token_attention_for_efficient_long-context_reasoning.md)
- [\[ACL 2026\] Step-GRPO: Internalizing Dynamic Early Exit for Efficient Reasoning](../../ACL2026/llm_reasoning/step-grpo_internalizing_dynamic_early_exit_for_efficient_reasoning.md)

</div>

<!-- RELATED:END -->
