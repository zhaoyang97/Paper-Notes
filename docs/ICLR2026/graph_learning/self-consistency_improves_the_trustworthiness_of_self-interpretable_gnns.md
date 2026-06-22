---
title: >-
  [论文解读] Self-Consistency Improves the Trustworthiness of Self-Interpretable GNNs
description: >-
  [ICLR 2026][图学习][图神经网络] 自解释 GNN（SI-GNN）训练时优化的是交叉熵+简洁正则，评测时却用忠实度（faithfulness），两者错位；本文指出忠实度本质上等价于"解释自洽性"，于是加一个把第一遍解释回灌后再生成的第二遍解释对齐起来的自洽（SC）损失，做一次模型无关的微调，就能在一致性、准确度、忠实度、信息量四个维度上同时提升解释质量。
tags:
  - "ICLR 2026"
  - "图学习"
  - "图神经网络"
  - "解释忠实度"
  - "自洽性"
  - "解释冗余"
  - "微调"
---

# Self-Consistency Improves the Trustworthiness of Self-Interpretable GNNs

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=hxGdAUn3sB](https://openreview.net/forum?id=hxGdAUn3sB)  
**代码**: https://github.com/ICDM-UESTC/SelfConsistencyXGNN  
**领域**: 图学习 / 可解释性  
**关键词**: 自解释 GNN, 解释忠实度, 自洽性, 解释冗余, 微调  

## 一句话总结
自解释 GNN（SI-GNN）训练时优化的是交叉熵+简洁正则，评测时却用忠实度（faithfulness），两者错位；本文指出忠实度本质上等价于"解释自洽性"，于是加一个把第一遍解释回灌后再生成的第二遍解释对齐起来的自洽（SC）损失，做一次模型无关的微调，就能在一致性、准确度、忠实度、信息量四个维度上同时提升解释质量。

## 研究背景与动机

**领域现状**：GNN 预测很强但像黑箱，难以在高风险或科学场景落地。自解释 GNN（SI-GNN）的思路是在模型里内置一个解释器 $h_{G_s}$，端到端地同时学预测和解释——给每条边打重要性分数 $\alpha_{ij}$，选出一个子图 $G_s \subseteq G$ 作为解释。代表方法按选子集的策略分四类：基于注意力的 GAT、基于因果的 CAL、基于尺寸约束的 SMGNN、基于互信息约束的 GSAT。

**现有痛点**：评测解释质量常用**忠实度**——把解释子图 $G_s$ 重新喂回模型，如果预测不变就认为这个解释是忠实的（它不依赖人工标注的 ground-truth，所以适用面广）。问题在于：SI-GNN 训练时的目标是交叉熵 $L_{CE}$ 加一个简洁正则 $R(G_s)$，**根本没有一项在优化忠实度**。优化什么和评测什么对不上。

**核心矛盾**：忠实度的检验过程（把解释回灌、看预测是否稳定）隐含地要求模型对同一实例反复抽取的解释要稳定一致。如果解释器真的抓住了决定性结构，那它第二次也应该高亮同一个子图。换句话说，**忠实度本质上依赖"自洽性"**：解释稳定 → 预测稳定 → 满足忠实度。既然如此，忠实度就可以通过一个让前后两次解释对齐的损失被直接优化。

**本文目标**：拆成两个子问题——(i) 忠实度这个概念性质能不能在训练时显式优化？(ii) 就算能，它真的能改善解释质量吗？

**切入角度**：作者做了实证分析（Figure 1），发现不做忠实度训练时，SI-GNN 的第一遍和第二遍解释会显著不同，而且这种"自不一致"主要发生在 ground-truth 标为**不重要**的特征上，重要特征反而稳定。这一现象和近期工作（Tai et al., 2025）发现的"解释冗余"对上了：简洁约束不够时，多余预算会让解释器不负责任地给不重要的边也打高分。既然自不一致也集中在不重要边上，那么**治自不一致 ≈ 治冗余 ≈ 提质量**。

**核心 idea**：在 SI-GNN 的标准目标之上，加一个自洽（SC）损失，最小化前后两遍解释的差异，做一次不改架构的微调。

## 方法详解

### 整体框架
方法解决的是"训练目标里没有忠实度"这件事，做法是把忠实度翻译成可微的自洽损失，然后用两步式微调把它注入任意现成的 SI-GNN。整体流程是：先按标准目标把 SI-GNN 训到收敛，冻结 GNN 编码器（保证表示学习不被后续损失破坏，SC 损失只去影响解释器的行为）；然后进入自洽微调——给定图 $G$，解释器产生第一遍解释 $G_s^{(1)}$，把 $G_s^{(1)}$ 再喂回模型得到第二遍解释 $G_s^{(2)}$，用一个对齐损失逼着两遍一致。微调时只更新解释器和分类器。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["输入图 G"] --> B["标准训练 SI-GNN<br/>交叉熵 + 简洁正则"]
    B --> C["冻结 GNN 编码器<br/>只留解释器/分类器可调"]
    C --> D["第一遍解释 G_s^(1)<br/>解释器打边分 α_ij"]
    D -->|把 G_s^(1) 回灌| E["第二遍解释 G_s^(2)"]
    E --> F["自洽损失对齐<br/>L_SC = |G_s^(1) − G_s^(2)|"]
    F --> G["输出：更可信的解释"]
```

### 关键设计

**1. 自洽微调框架：把"忠实度"翻译成可微的双过程对齐损失**

痛点很直接：忠实度是个"回灌后预测不变"的检验性质，不可微，没法当训练目标。本文的关键观察是，这个检验隐含要求解释器对同一实例自洽——所以只要逼着解释器两遍输出一致，就在间接优化忠实度。具体地，第一遍解释器产生 $G_s^{(1)}$，把它回灌进（被冻结编码器的）模型得到第二遍 $G_s^{(2)}$，自洽损失就是两者的 L1 差异：

$$L_{SC} = |G_s^{(1)} - G_s^{(2)}|.$$

最终目标是在标准 SI-GNN 损失上叠一项：$L_{FT} = L_{GE} + \eta \cdot L_{SC}$，$\eta$ 是权重。整个解释器里，边 $e_{ij}$ 的重要性分数由 $w_{ij} = \text{MLP}([v_i; v_j])$、$\alpha_{ij} = \sigma(w_{ij})$ 给出，并用 Gumbel–Sigmoid 做可微采样。这个设计之所以好用，是因为它**不动模型架构、只加一项损失**，因此能无缝套到注意力/因果/尺寸/互信息四类 SI-GNN 上；而且"先 pretrain 再冻编码器"这一步保证 SC 损失只重塑解释器的打分行为，不会把已经学好的表示搅乱。

**2. 近似不动点（near-fixed levels）：解释为什么 SC 会把分数收敛到几个稳定档位**

光说"对齐两遍"还不够，要解释清楚它对边分数到底做了什么。把第二遍在每条边上的映射记作 $T(\alpha) = \sigma(g(\alpha))$，其中 $g(\cdot)$ 是第二遍的 pre-activation。直觉上强制自洽等价于要求 $T(\alpha) \approx \alpha$，即边分数跨两遍保持稳定。作者用"近似不动点"刻画这件事：给定容差 $\varepsilon$，$\alpha$ 是 $\varepsilon$-近似不动点当且仅当 $|T(\alpha) - \alpha| \le \varepsilon$，等价于 pre-activation 落进一个 logit 窗口 $g(a^*) \in [\text{logit}(a^*-\varepsilon),\ \text{logit}(a^*+\varepsilon)]$。这个窗口宽度近似为 $\Delta g \approx \frac{2\varepsilon}{a^*(1-a^*)}$，在内部点是有限窗口（需要架构和优化轨迹刚好命中），而当 $a^* \to 0$ 或 $1$ 时窗口退化成单边阈值——对应 sigmoid 的饱和平区，所以 **0/1 这两个极值档位比中间档位更容易达到**。结论是：SC 不是把所有分数压到一个值，而是把它们驱往少数几个近似不动的稳定档位，重要边被分类损失推向 1，不重要边则停在某个低而稳定的档位。

**3. 与简洁正则的相互作用：靠 CR 强度决定不重要边收敛到哪个档位**

设计 2 说明 SC 提供"稳定性"，但稳到哪个档位由谁定？答案是简洁正则（CR）。把联合损失对边分数求梯度，分成三股力量：

$$\frac{\partial L}{\partial \alpha_{ij}} \approx \underbrace{\frac{\partial L_{CE}}{\partial \alpha_{ij}}}_{\text{分类}} + \beta \cdot \underbrace{\frac{\partial R(G_s)}{\partial \alpha_{ij}}}_{\text{简洁}} + \eta \cdot \underbrace{\frac{\partial L_{SC}}{\partial \alpha_{ij}}}_{\text{稳定}}.$$

分类项把重要边推向 1；简洁项视 CR 形式而定（SMGNN 鼓励稀疏，把不重要边推向 0；GSAT 鼓励独立，推向 0.5）；稳定项把分数拉向近似不动点。三者博弈出三个区间：$\beta$ 太弱时 CR 几乎不起作用，不重要边可以自由停在任意可行档位；$\beta$ 适中时 CR 主动压制不重要边（SMGNN 压到 0、GSAT 压到 0.5），重要边仍被分类项守在 1 附近——**此时 CR 和 SC 合力把不重要边推向又低又稳的分数，解释质量最好**；$\beta$ 太强时 CR 压过分类，重要边和不重要边一起塌缩（SMGNN 都到 0、GSAT 都到 0.5），重要边和不重要边再也分不开，质量反而崩。这也解释了为什么 SC 对本身没有 CR 的 GAT/CAL 单独用并不稳，必须先补上 CR（实验里的 +CR+SC 配置）。

### 损失函数 / 训练策略
两步式：Step 1 用标准目标 $L_{GE}$（各 backbone 的交叉熵+简洁正则，见第 2 节四类公式）训到收敛并冻结编码器；Step 2 用 $L_{FT} = L_{GE} + \eta \cdot L_{SC}$ 微调解释器与分类器。$\eta$ 控制自洽强度，$\beta$ 控制简洁正则强度，二者需配合（$\beta$ 适中区最佳）。

## 实验关键数据

数据集：合成的 BA-2MOTIFS 加三个真实分子数据集 3MR、BENZENE、MUTAGENICITY。指标四项：SHD↓ 衡量解释一致性、AUC↑ 衡量解释准确度、ACC↑ 衡量下游信息量、FID↓ 衡量忠实度。对比对象除原始 backbone 外，还有后处理的解释集成 EE（Explanation Ensemble）。

### 主实验（SMGNN / GSAT，含 CR）

| 方法 | BA-2MOTIFS SHD↓ | BA-2MOTIFS AUC↑ | BENZENE SHD↓ | BENZENE AUC↑ | MUTAG SHD↓ | MUTAG FID↓ |
|------|------|------|------|------|------|------|
| SMGNN | 10.44 | 99.32 | 16.06 | 84.38 | 12.65 | 1.72 |
| SMGNN+EE | 4.99 | 99.59 | 8.55 | 91.38 | 6.21 | – |
| SMGNN+SC | 3.48 | 99.87 | 7.19 | 90.07 | 2.51 | 0.61 |
| SMGNN+SC+EE | **1.52** | **99.90** | **4.14** | **92.20** | **1.44** | – |
| GSAT | 4.58 | 98.44 | 6.93 | 90.66 | 10.08 | 1.11 |
| GSAT+SC | 2.73 | 99.30 | 2.32 | 92.80 | 2.38 | 0.17 |
| GSAT+SC+EE | **1.19** | **99.35** | **1.14** | **93.53** | **1.06** | – |

SC 在四个维度上对原始 backbone 全面提升；多数情况 SC 优于 EE，且快约 5×、兼容所有标准指标；SC 与 EE 互补，叠加后进一步涨点。

### 消融 / 分析

| 配置 | 现象 | 说明 |
|------|------|------|
| GAT+SC（无 CR） | MUTAG SHD 0.04 但 AUC 暴跌到 81.79 | 没有 CR 时 SC 不稳，重要/不重要边一起塌 |
| GAT+CR+SC | AUC 回到 99.87、SHD 3.48 | 补上 CR 后 SC 才稳定有效 |
| CAL+SC（无 CR） | BENZENE AUC 88.74，仍不及 +CR | 同上，因果 backbone 也需 CR 配合 |
| CAL+CR+SC | AUC 89.87、SHD 6.25 | CR+SC 联合最稳 |
| $\beta$ 适中 vs 太强 | 太强时重要/不重要边都塌缩 | 印证三股梯度博弈，$\beta$ 需落在适中区 |

自洽性本身也被直接验证：Table 2 里 SMGNN 的 $G_s^{(1)}$ 与 $G_s^{(2)}$ 的余弦相似度从 99.68% 升到 99.98%、L1 距离从 16.87% 降到 1.84%（BA-2MOTIFS），PCA 图（Figure 7）里前后两遍表示的连线明显变短。

### 关键发现
- SC 之所以提升忠实度（FID），是因为它直接缩小前后两遍解释的差异，一致的解释 → 一致的表示 → 一致的预测，FID 自然降低。
- 解释质量的提升几乎全来自把**不重要边**驱往低而稳定的档位（Figure 3/4），重要边一直被分类损失守在 1 附近，这与"自不一致集中在不重要特征"的动机首尾呼应。
- SC 的有效性强依赖 CR 强度：无 CR 的 GAT/CAL 单加 SC 会让重要/不重要边一起坍塌（如 GAT+SC 在 MUTAG 上 AUC 掉到 81.79），必须先补 CR。

## 亮点与洞察
- **把不可微的评测指标翻译成可微训练目标**：忠实度本来是"回灌看预测变不变"的检验流程，作者抓住"它隐含要求自洽"这一点，用一个回灌-对齐的 L1 损失把它变得可优化——这种"用自洽性桥接评测与训练"的思路可迁移到其他训练-评测错位的可解释性任务。
- **模型无关、即插即用**：不改架构、只加一项损失加一步微调，四类主流 SI-GNN 全部适用，工程代价极低却全维度提升。
- **理论把"为什么有效"讲透**：近似不动点 + 三股梯度博弈，清楚解释了 SC 把分数收敛到少数稳定档位、CR 强度决定档位位置、$\beta$ 过强会坍塌——不是纯实证 trick。

## 局限与展望
- 只在边（结构特征）的实例级解释上验证，没覆盖节点特征 / 图级别其他解释形式。
- 数据集规模偏小（一个合成 + 三个分子），未在大规模或更复杂图任务上检验可扩展性。
- 效果对 CR 强度 $\beta$ 敏感，需要落在"适中区"，论文给的是定性结论，实际部署时 $\beta$/$\eta$ 的自动选取仍待解决。
- 自洽只是忠实度的必要条件而非充分条件——一个稳定但错误的解释也能自洽，论文靠 CR+真实标注实验间接缓解，但理论上仍有"稳定地错"的空间。

## 相关工作与启发
- **vs Explanation Ensemble (EE, Tai et al., 2025)**：EE 是后处理、多次集成来抑冗余；本文是训练期、单次微调注入自洽。SC 多数情况质量更好、快约 5×、兼容所有标准指标，且两者互补可叠加。
- **vs 解释冗余分析 (Tai et al., 2025)**：那篇关注跨模型（不同随机种子重训）的解释不一致并归因于冗余；本文揭示单个模型内部的"自不一致"，同样集中在不重要边，于是用显式自洽约束在训练时治冗余。
- **vs 标准 SI-GNN（GAT/CAL/SMGNN/GSAT）**：它们各自定义不同的简洁正则但都没优化忠实度；本文是正交的一层，叠在任意 backbone 之上补齐"训练-评测对齐"。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把忠实度等价为自洽性、用回灌对齐损失显式优化，视角新颖
- 实验充分度: ⭐⭐⭐⭐ 四 backbone × 四数据集 × 四指标全覆盖，理论有可视化佐证；但数据集规模偏小
- 写作质量: ⭐⭐⭐⭐⭐ 从动机到理论到实验逻辑闭环，近似不动点的分析很清晰
- 价值: ⭐⭐⭐⭐ 模型无关、即插即用、低成本提升可信度，可解释 GNN 落地价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] HarmonyGNNs: Harmonizing Heterophily and Homophily in GNNs via Self-Supervised Node Encoding](harmonygnns_harmonizing_heterophily_and_homophily_in_gnns_via_self-supervised_no.md)
- [\[AAAI 2026\] Self-Adaptive Graph Mixture of Models](../../AAAI2026/graph_learning/self-adaptive_graph_mixture_of_models.md)
- [\[AAAI 2026\] Self-Correction Distillation for Structured Data Question Answering](../../AAAI2026/graph_learning/self-correction_distillation_for_structured_data_question_answering.md)
- [\[ECCV 2024\] SENC: Handling Self-collision in Neural Cloth Simulation](../../ECCV2024/graph_learning/senc_handling_self-collision_in_neural_cloth_simulation.md)
- [\[ICLR 2026\] Structurally Human, Semantically Biased: Detecting LLM-Generated References with Embeddings and GNNs](structurally_human_semantically_biased_detecting_llm-generated_references_with_e.md)

</div>

<!-- RELATED:END -->
