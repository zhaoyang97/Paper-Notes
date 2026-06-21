---
title: >-
  [论文解读] GradPruner: Gradient-guided Layer Pruning Enabling Efficient Fine-Tuning and Inference for LLMs
description: >-
  [ICLR 2026][模型压缩][层剪枝] GradPruner 用 LoRA 微调最初 1% 步：累积的梯度算出每层重要性（IGIA-Matrix）来做层剪枝，再把被剪层"同符号合并"进保留层，从而在下游任务上同时省训练和推理：剪掉 40% 参数只掉 0.99% 精度。 领域现状：LLM 在医疗、金融等垂直领域往往要在下…
tags:
  - "ICLR 2026"
  - "模型压缩"
  - "层剪枝"
  - "结构化剪枝"
  - "LoRA 微调"
  - "梯度重要性"
  - "层合并"
---

# GradPruner: Gradient-guided Layer Pruning Enabling Efficient Fine-Tuning and Inference for LLMs

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=bxzJorqyYM](https://openreview.net/forum?id=bxzJorqyYM)  
**代码**: [https://github.com/secretflow/ACoLab/tree/main/PaperCode/GradPrune](https://github.com/secretflow/ACoLab/tree/main/PaperCode/GradPrune)  
**领域**: 模型压缩 / LLM 结构化剪枝  
**关键词**: 层剪枝, 结构化剪枝, LoRA 微调, 梯度重要性, 层合并  

## 一句话总结
GradPruner 用 LoRA 微调**最初 1% 步**累积的梯度算出每层重要性（IGIA-Matrix）来做层剪枝，再把被剪层"同符号合并"进保留层，从而在下游任务上同时省训练和推理：剪掉 40% 参数只掉 0.99% 精度。

## 研究背景与动机
**领域现状**：LLM 在医疗、金融等垂直领域往往要在下游数据上微调才有好表现，但全量微调既慢又贵；结构化剪枝虽能提升推理效率，却普遍是"先用校准数据找重要参数、再训练/蒸馏恢复"的两段式流程，反而要额外的时间和显存。

**现有痛点**：现有剪枝方法多依赖前向传播（如中间层激活）在校准数据上评估参数重要性，而 LLM 在垂直领域本身表现就弱，这种判断会引入显著偏差；而少数面向"高效训练+推理"的工作各有硬伤——APT 只能配合 LoRA 微调，SAT 在不同训练步动态改结构、最后一步又把模型恢复成稠密形态，无法加速推理。

**核心矛盾**：剪得越多训练/推理越快但精度越掉，剪得越少精度好但速度慢——需要在"省时省内存地评估重要性"与"尽可能多剪层又不掉精度"之间找到平衡。作者发现微调初期 loss 会在前 1% 步骤里急剧下降，说明模型很快抓住了下游任务知识，且不同参数学习能力差异明显。

**本文目标**：不增加额外训练时间与显存的前提下，针对具体下游数据与模型衡量参数重要性，最大限度保留原结构地剪枝，并同时支持全量与 LoRA 微调。

**核心 idea**：**[早期梯度即重要性]** 用 LoRA 微调最初极少步骤的累积梯度构建 IGIA-Matrix 评估层重要性做剪枝，再用**[同符号层合并]** 把被剪层的关键参数稀疏后并入保留层，进一步提高剪枝率而不掉精度。

## 方法详解

### 整体框架
GradPruner 分三步：先用少量 LoRA 微调采集前 t 步（t≪T）的梯度，据此为每个线性层算出"初始梯度信息累积矩阵"IGIA-Matrix；再把每层内所有线性层的 IGIA-Matrix 求和得到层重要性分数并剪掉低分层；最后把被剪层稀疏化后按符号合并进前面保留的层，以更高剪枝率守住精度。

```mermaid
flowchart LR
    A[下游数据 D] --> B[LoRA 微调前 t 步<br/>采集 ∇W_A, ∇W_B]
    B --> C[模拟 W 梯度<br/>∇W = ∇W_B · ∇W_A]
    C --> D[IGIA-Matrix F_W<br/>= 梯度平方均值]
    D --> E[层重要性 = 层内各线性层<br/>IGIA 之和]
    E --> F[剪掉低分层]
    F --> G[被剪层按 IGIA top-p% 稀疏化]
    G --> H[同符号合并进前一保留层]
```

### 关键设计

**1. IGIA-Matrix：用 LoRA 初始梯度模拟 W 的重要性。** 这是全方法的基石。冻结原权重 $W$、只训练 LoRA 旁路 $W_A,W_B$，跑满总步数 $T$ 太贵，所以只取前 $t$ 步（$t\ll T$）的逐步梯度 $\nabla_{W_A}L$ 与 $\nabla_{W_B}L$。由于 LoRA 微调后能与原参数合并，作者把两路梯度相乘对齐回 $W$ 的维度，得到第 $i$ 步对 $W$ 的"模拟梯度" $\nabla_W L(x,y)^{sim}_i = \nabla_{W_B}L_i \cdot \nabla_{W_A}L_i$。随后对前 $t$ 步的模拟梯度取平方求均值得到该线性层的 IGIA-Matrix：$F_W = \frac{1}{t}\sum_{i=1}^{t}\big(\nabla_W L(x,y)_i\big)^2$。平方既消除符号、又放大学习更剧烈（即对下游任务更关键）的参数，使早期阶段的重要性度量就能逼近全量训练后的结果——这点由论文的"梯度敏感性分析"佐证：早期 top-20 重要层与全训后的标签高度吻合。

**2. 层级粒度剪枝：守住整体结构。** 拿到每个线性层的 IGIA-Matrix 后，把第 $j$ 层内所有 $M$ 个线性层、每层 $H$ 个参数的重要性累加成层分数 $\text{Layer}_j=\sum_{k=1}^{M}\sum_{l=1}^{H}F_{W_{kl}}$，再剪掉分数最低的若干层。选层级（而非神经元/通道级）有两个动机：一是尽量不破坏模型整体架构，二是消融显示剪掉重要层的参数会显著拖垮下游精度。但层剪枝有上限——单独剪超过约 30% 的层（Llama3.1-8B 上约剪 10 层后再多剪一层）精度就断崖式下跌，这正是引出第三步的原因。

**3. 同符号层合并：把"该剪的"变成"可并的"，突破剪枝率上限。** 与其直接丢弃被剪层，不如把它们的有用信息并回保留层。指定保留层 $W_1$、待剪层 $\{W_2,...,W_n\}$，分两步走：① **稀疏化**——用 IGIA-Matrix 作判据只保留待剪层 top-p% 的参数、其余置零得到 $\{\hat W_2,...,\hat W_n\}$（$W_1$ 更关键，不稀疏以免伤精度）；② **按符号合并**——同一位置的参数在不同层可能正负号相反，直接相加会相互抵消缩小数值，因此以 $W_1$ 的符号 $\gamma$ 为基准，只把 $\hat W$ 中符号与之一致的元素加进来：当符号冲突时保留 $W_1$ 原值，当某待剪层符号匹配时才执行 $(W_j)_{kl}+\hat{(W_{j+n})}_{kl}$。被剪层只与紧邻的前一保留层合并。靠这一步，在剪 10 层基础上再多剪 1~3 层仍能逼近稠密模型精度。

## 实验关键数据

### 主实验（40% 稀疏剪枝，八数据集平均分）

| 方法 | Llama3.1-8B (FFT) | Llama3.1-8B (LoRA) | Mistral-7B (FFT) | Mistral-7B (LoRA) |
|---|---|---|---|---|
| Dense Model（上界） | 0.784 | 0.794 | 0.781 | 0.790 |
| LLMPruner | 0.734 | 0.733 | 0.730 | 0.728 |
| LaCo | 0.736 | 0.740 | 0.738 | 0.737 |
| MINITRON | 0.734 | 0.734 | 0.734 | 0.731 |
| SAT | 0.750 | 0.745 | 0.748 | 0.743 |
| APT | — | 0.759 | — | 0.750 |
| FT(Llama3.2-3B) | 0.777 | 0.774 | — | — |
| **GradPruner** | **0.782** | **0.786** | **0.770** | **0.780** |

GradPruner 在四种设置下都领先所有基线，相对稠密模型平均只掉 0.99%（FFT/Llama3.1），且剪枝后的 8B 模型精度超过直接微调的 Llama3.2-3B，比 LLMPruner/LaCo/MINITRON 平均高约 5 个百分点。

### 效率（归一化到 Dense Model，越小越好）

| 方法 | 训练时间 | 训练显存 | 推理时间 | 推理显存 |
|---|---|---|---|---|
| Dense Model | 100% | 100% | 100% | 100% |
| LLMPruner | 78.3% | 284.4% | 67.4% | 65.3% |
| LaCo | 73.8% | 64.4% | 59.7% | 61.3% |
| SAT | 75.5% | 79.4% | 98.9% | 103.6% |
| **GradPruner (FFT)** | **62.4%** | **65.8%** | **61.5%** | **60.9%** |

GradPruner 训练时间/显存约省 36%、推理时间/显存约省 39%，与 LaCo 相当；而 SAT 因末步恢复稠密结构推理几乎不省，APT 因蒸馏训练时间反而高达 158%。

### 消融实验（Llama3.1-8B / FFT，合并层数对精度的影响）

| 合并层数 | GradPruner | w/o Merging |
|---|---|---|
| 1 | 0.785 | 0.775 |
| 2 | 0.786 | 0.767 |
| 3 | 0.782 | 0.741 |

### 关键发现
- **层合并是高剪枝率的关键**：不合并时随剪层增多精度急跌（剪 13 层只剩 0.741），加入合并后稳定逼近稠密模型；
- **单纯层剪枝有硬上限**：Llama3.1-8B 上剪到约 10 层影响很小，再多就显著掉点；
- **稀疏率非越高越好**：在 50%~90% 区间扫描，过高或过低都伤精度，存在甜点区；
- **早期梯度足够可靠**：梯度敏感性分析证明前 1% 步骤识别的重要层与全训后高度一致。

## 亮点与洞察
- **把"剪枝"前移进微调早期**：传统剪枝靠校准数据的前向激活判重要性，在垂直领域会因模型本身弱而失真；GradPruner 改用任务自身的早期梯度，既贴合下游数据又顺手省下评估开销，是一个很自然但少有人系统利用的切入点。
- **LoRA 双路梯度相乘模拟 W 梯度**：避免直接对冻结的大矩阵 $W$ 求梯度，用低秩旁路的梯度乘积近似，是兼顾"省显存"与"重要性可解释"的巧思。
- **符号感知合并**：层合并不是简单相加，而是抓住"符号冲突会相互抵消"这一痛点，只并同号元素，把被剪层从"扔掉"变成"有用残值回收"，直接突破层剪枝 30% 的天花板。
- **同时优化训练与推理**：多数剪枝工作只盯推理，本文把训练时间/显存也压下来，对垂直领域反复微调的实际场景更友好。

## 局限与展望
- **粒度较粗**：只做层级剪枝，未触及更细的通道/注意力头级，剪枝率提升空间受层数离散性限制；
- **依赖 LoRA 早期梯度的可迁移性**：方法建立在"前 1% 步梯度能代表全程重要性"这一经验观察上，对训练极不稳定或 loss 不快速下降的任务是否成立未充分讨论；
- **合并的近似误差**：同符号合并丢弃异号元素并稀疏化待剪层，是有损操作，高剪枝率下误差累积如何随模型规模变化缺乏理论刻画；
- **评测口径**：医疗/金融 QA 用 BertScore+ROUGE-L 的相似度评估，与真实临床/合规需求仍有差距，且仅在 7B/8B 量级验证，更大模型上的表现待考。

## 相关工作与启发
- **结构化剪枝**：LLMPruner（基于梯度删耦合结构）、LaCo（后层塌缩进前层）、MINITRON（深度/宽度/注意力/MLP 联合剪枝 + 蒸馏恢复）代表"先剪后训/蒸馏"的两段式路线，GradPruner 则把重要性评估融进微调本身。
- **高效训练+推理剪枝**：APT（动态增删显著调参，仅限 LoRA）、SAT（阶梯式遗漏率调度，但末步恢复稠密）是最直接对标，本文针对它们"只支持 LoRA / 不加速推理"的短板做改进。
- **梯度/Fisher 重要性**：用全量训练后的梯度信息评估参数重要性（Matena & Raffel, Daheim et al.）已有先例，IGIA-Matrix 的贡献在于证明只需早期梯度即可，并把它产品化进剪枝-合并流程。
- **启发**：把"模型适配下游任务的早期动力学"作为压缩/选择的信号，可能比静态校准更适配垂直领域；同符号合并的思路也可迁移到模型融合、任务向量等场景。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — "早期 LoRA 梯度建 IGIA-Matrix + 同符号层合并"组合新颖，把剪枝评估融进微调早期、同时省训练与推理的角度切得准。
- **实验充分度**: ⭐⭐⭐⭐ — 2 个 LLM × 8 数据集 × FFT/LoRA 双设置，含精度、效率、合并层数、剪枝率、梯度敏感性多维消融，较扎实；但仅 7B/8B 量级。
- **写作质量**: ⭐⭐⭐ — 动机清晰、框架图到位，但公式(2)符号写作有瑕疵（$\nabla_{W_B}\cdot\nabla_{W_B}$ 与文意 $W_B\cdot W_A$ 不一致），细节表述偶有笔误。
- **价值**: ⭐⭐⭐⭐ — 面向垂直领域微调的高效压缩有很强实用性，代码开源，剪 40% 参数仅掉 0.99% 精度的结果对工程落地有吸引力。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Reassessing Layer Pruning in LLMs: New Insights and Methods](reassessing_layer_pruning_in_llms_new_insights_and_methods.md)
- [\[ICLR 2026\] TRAC: Tensor-Train Based Across-Layer Compression for Parameter-Efficient Fine-Tuning](trac_tensor-train_based_across-layer_compression_for_parameter-efficient_fine-tu.md)
- [\[CVPR 2026\] LoPrune: Efficient Data Pruning for LoRA-Based Fine-Tuning of Vision Transformer](../../CVPR2026/model_compression/loprune_efficient_data_pruning_for_lora-based_fine-tuning_of_vision_transformer.md)
- [\[ICLR 2026\] Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning](sculpting_subspaces_constrained_full_fine-tuning_in_llms_for_continual_learning.md)
- [\[ACL 2026\] Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference](../../ACL2026/model_compression/adaptive_layer_selection_for_layer-wise_token_pruning_in_llm_inference.md)

</div>

<!-- RELATED:END -->
