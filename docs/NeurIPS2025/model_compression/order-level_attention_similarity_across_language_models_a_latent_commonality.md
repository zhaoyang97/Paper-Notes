---
title: >-
  [论文解读] Order-Level Attention Similarity Across Language Models: A Latent Commonality
description: >-
  [NeurIPS 2025][模型压缩][注意力机制] 提出 Order-Level Attention (OLA)——对 Attention Rollout 的阶次分解，发现不同语言模型在同阶 OLA 上存在显著相似性 (OLAS)，并且 OLA 隐式编码了句法知识，基于此提出 TOA 实现首个无需训练的跨LM适配器迁移。
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "注意力机制"
  - "语言模型相似性"
  - "跨模型迁移"
  - "句法知识"
  - "无训练适配器"
---

# Order-Level Attention Similarity Across Language Models: A Latent Commonality

**会议**: NeurIPS 2025  
**arXiv**: [2511.05064](https://arxiv.org/abs/2511.05064)  
**代码**: [有](https://github.com/jinglin-liang/OLAS)  
**领域**: 模型分析与压缩  
**关键词**: 注意力机制, 语言模型相似性, 跨模型迁移, 句法知识, 无训练适配器

## 一句话总结

提出 Order-Level Attention (OLA)——对 Attention Rollout 的阶次分解，发现不同语言模型在同阶 OLA 上存在显著相似性 (OLAS)，并且 OLA 隐式编码了句法知识，基于此提出 TOA 实现首个无需训练的跨LM适配器迁移。

## 研究背景与动机

**核心问题**：不同语言模型的上下文聚合模式是否存在共性？

虽然现有工作（Attention Rollout、归因分析等）分析了单个模型的注意力机制，但它们关注的是个体模型的特性分析，缺乏对**多个LM共性**的系统研究。如果LM间存在共同的表征空间，就能实现高效的跨模型知识迁移。

**直觉**：主流transformer LM都依赖注意力机制聚合上下文进行预测。考虑到相似的训练目标和注意力机制，不同LM在大规模语料上训练后可能收敛到对同一文本的最优注意力模式。

**Attention Rollout的问题**：直接分析Attention Rollout会遇到**Attention Sinks**现象——softmax不能产生精确的零注意力分数，当token已收集足够信息后，多余的注意力会泄漏到无关token上。这导致Rollout在不同文本上呈现相似的偏置模式，缺乏区分性。

**关键洞察**：Attention Sinks是因为N层LM产生 $2^N$ 条信息路径，高阶路径中过度聚合导致偏置。分别分析不同聚合次数的路径，低阶分量更有区分性。

## 方法详解

### 整体框架

1. **OLA定义**：将Attention Rollout按阶次分解为可比较的表示
2. **OLAS发现**：通过定性和定量实验验证跨LM的OLA相似性
3. **句法发现**：证明OLA隐式编码了句法依赖关系
4. **TOA应用**：利用OLAS实现无训练的跨模型适配器迁移

### 关键设计

#### 1. **Order-Level Attention (OLA) 的推导**

$N$ 层LM的Attention Rollout定义为：
$$\hat{A} = \prod_{i=1}^N (A^{(i)} + I)$$

展开为阶次分解：
$$\hat{A} = I + \sum_{i=1}^N A^{(i)} + \sum_{1 \leq i < j \leq N} A^{(j)}A^{(i)} + \cdots + A^{(N)}\cdots A^{(1)}$$

归一化后得到 $k$ 阶 OLA：
- 0阶：$\hat{A}^{(0)} = I$（纯残差连接）
- 1阶：$\hat{A}^{(1)} = \frac{1}{N}\sum_{i=1}^N A^{(i)}$（经过恰好1次注意力聚合的路径均值）
- $k$阶：$\binom{N}{k}$ 条路径的均值

Rollout可重写为：$\hat{A} = \sum_{i=0}^N \binom{N}{i} \cdot \hat{A}^{(i)}$

设计动机：OLA统一了不同层数、不同头数模型的注意力表示到同一语义空间（$k$阶=恰好聚合$k$次上下文），使跨模型比较成为可能。

#### 2. **OLAS的定量验证**

**方法一：视觉模型代理评估**

训练ResNet-18图像分类器，将源LM的OLA maps作为训练数据（同一文本→同一类别），在目标LM的OLA maps上测试。

**方法二：图像检索评估**

用SSIM相似度做跨模型OLA检索，Hits@1/Hits@5评估检索成功率。

#### 3. **OLA与句法知识的映射**

训练辅助网络仅从OLA预测句法依赖关系。一阶OLA在MLM上UAS超过80%，表明OLA隐式编码了丰富的句法知识。低阶OLA的句法特征比高阶OLA更显著。

### 损失函数 / 训练策略

TOA（Transferable OLA Adapter）的设计：
1. **训练阶段**：冻结源LM，以堆叠的1阶和2阶OLA为输入，训练下游任务适配器
2. **测试阶段**：直接将适配器转移到目标LM，无需任何参数更新或训练数据
3. 适配器接收的是OLA（统一表示）而非模型特定的hidden states，因此天然可迁移

## 实验关键数据

### 主实验

视觉模型代理评估（分类准确率%，CLM结果）：

| 方法 | Q-1b5 | Q-7b | G-2b | G-9b | L-3b | L-8b |
|---|---|---|---|---|---|---|
| Rollout | 27.9 | 7.7 | 52.6 | 26.0 | 66.1 | 59.7 |
| 1st OLA | **52.6** | **49.2** | **93.1** | **92.4** | **94.6** | **94.1** |
| 2nd OLA | 67.1 | 49.9 | 89.3 | 86.2 | 90.7 | 91.9 |
| ALTI | 22.6 | 15.5 | 69.3 | 71.8 | 85.6 | 79.8 |

关系抽取(RE)上的跨模型TOA迁移（准确率%）：

| 源→目标 | Q-1b5 | G-2b | L-3b | Zero-shot |
|---|---|---|---|---|
| TOA from L-3b | 30.49 | 33.49 | 35.57 | - |
| TOA from Q-1b5 | 34.90 | 30.95 | 31.08 | - |
| Zero-shot | 7.69 | 5.01 | 14.65 | 基准 |

### 消融实验

OLA句法依赖预测（UAS/LAS %）：

| LM | 1阶 | 2阶 | 3阶 | Rollout | 说明 |
|---|---|---|---|---|---|
| Bert-base | 81.29/72.16 | 72.86/61.05 | 66.44/53.17 | 46.20/30.69 | 低阶>>高阶 |
| Roberta-base | 80.00/70.44 | 72.68/60.10 | 36.99/18.67 | 35.77/17.94 | 同上 |
| Electra-base | 81.23/72.63 | 77.47/66.78 | 50.72/33.90 | 50.35/34.02 | 同上 |

图像检索评估（Hits@1/Hits@5 %，1阶OLA）：

| 源\目标 | Q-1b5 | G-2b | L-3b |
|---|---|---|---|
| Q-1b5 | - | 83.6/89.4 | 95.9/97.0 |
| L-3b | 92.9/96.1 | 94.1/96.5 | - |

### 关键发现

1. **低阶OLA相似性最强**：一阶OLA跨模型一致性最高，高阶包含更多Attention Sinks
2. **句法信息随阶次递减**：一阶OLA句法预测远优于高阶和Rollout
3. **OLAS是预训练产物**：参数扰动使OLAS消失，确认来自学到的知识而非实验偏差
4. **CLM间相似度高于MLM**：可能因CLM家族架构更统一
5. **无训练迁移有效**：TOA在RE任务上从7.69%零样本提升到34.90%（4.5x）

## 亮点与洞察

1. **揭示了LM间被忽视的共性**：不同架构、不同训练数据的LM在注意力聚合模式上存在统一的"语言学先验"
2. **Attention Sinks的阶次解析**：提供了理解Attention Sinks现象的全新视角——高阶路径是噪声的主要来源
3. **首个无训练跨LM适配器迁移**：打破了适配器与特定模型绑定的限制
4. **数学推导简洁优雅**：OLA自然地源于Attention Rollout的多项式展开

## 局限与展望

1. 仅验证了基础NLP任务（RE、NER、DP、POS），能否扩展到生成、推理等复杂任务
2. TOA仅使用OLA作为输入，未结合模型原始表示，可能损失了信息
3. OLA阶数选择（1阶+2阶堆叠）是启发式的，最优组合待研究
4. 可探索用OLA指导模型压缩、知识蒸馏等其他跨模型任务

## 相关工作与启发

- 与Moschella et al.的表示学习方法相比，OLAS提供了更直接的跨模型公共空间
- OLA的句法编码发现与探针研究的结论一致，但更结构化
- 可能启发NLP社区重新审视注意力机制在不同模型间的共同特性

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （OLA概念新颖，OLAS发现重要，TOA首创无训练跨LM迁移）
- 实验充分度: ⭐⭐⭐⭐⭐ （12个LM、定性+定量分析、控制实验、4个下游任务）
- 写作质量: ⭐⭐⭐⭐⭐ （逻辑清晰，从发现到应用的故事线完整）
- 价值: ⭐⭐⭐⭐ （对理解LM共性有重要意义，但TOA迁移效果还有提升空间）

发现不同语言模型在相同阶次的注意力分解（Order-Level Attention, OLA）上存在显著相似性（OLAS），并基于此提出 TOA 方法实现无需训练的跨模型 adapter 迁移。

## 研究背景与动机

不同 LM 在架构、训练数据等方面差异巨大，但它们都依赖注意力机制进行上下文聚合。一个自然的问题是：不同 LM 的上下文聚合模式是否存在共同点？现有研究主要关注单个模型或单个注意力头的分析，缺乏跨模型的系统性研究。如果能发现这种共性，就有望实现高效的跨模型知识迁移，避免在每个新模型上从头微调 adapter 的重复劳动。

## 方法详解

### 整体框架

本文的技术路线分为三步：（1）提出 OLA 作为统一的跨模型注意力表示；（2）通过定性（可视化）和定量（分类+检索）实验验证 OLAS 现象；（3）基于 OLAS 提出 TOA 实现无训练 adapter 迁移。

### 关键设计

1. **Order-Level Attention (OLA) 分解**: 从 Attention Rollout 出发，将信息流分解为多条路径。一个 N 层模型有 $2^N$ 条可能路径。Attention Rollout $\hat{A} = \prod_{i=1}^N (A^{(i)} + I)$ 可展开为：$\hat{A} = I + \sum_{i}A^{(i)} + \sum_{i<j}A^{(j)}A^{(i)} + \cdots$。第 k 阶 OLA 为 $\hat{A}^{(k)}$，即经过 k 次注意力聚合的路径效果的归一化。例如一阶 OLA 为 $\hat{A}^{(1)} = \frac{1}{N}\sum_{i=1}^N A^{(i)}$。这种分解消除了不同模型因层数差异导致的不可比性，赋予相同阶次的注意力以统一语义。

2. **OLAS 现象验证**: 
    - **定性分析**：可视化不同 LM（如 Qwen2-1.5b 和 Llama3.2-3b）对相同文本的 OLA，发现同阶 OLA 高度相似，而不同文本的 OLA 有明显区分度。高阶 OLA 的 attention sink 现象更严重，说明低阶 OLA 包含更有效的聚合信息。
    - **基于视觉分类模型的定量分析**：训练 ResNet-18 将源 LM 的 OLA 图分类为对应文本，然后在目标 LM 的 OLA 上测试。一阶 OLA 在 CLM 上超过 90% 的分类准确率。
    - **基于图像检索的定量分析**：使用 SSIM 度量 OLA 图之间的相似度。一阶 OLA 的 Hits@5 在 CLM 上最低 89%，最高超过 97%。

3. **OLA 与句法知识的隐式映射**: 实验表明仅使用 OLA 表示就能预测句法依存关系（Universal Dependencies），说明 OLA 内在编码了输入文本的句法知识。

4. **Transferable OLA Adapter (TOA)**: 将 OLA 作为统一的跨模型句法特征表示，在源 LM 上用 OLA 作输入训练 adapter 完成下游任务。由于 OLA 在不同 LM 间具有相似性，训练好的 adapter 可以直接迁移到未见过的目标 LM，无需任何参数更新或额外训练。

### 损失函数 / 训练策略

TOA adapter 训练使用标准分类/序列标注损失。源 LM 上的 OLA 图分类实验使用交叉熵损失：$\theta^* = \arg\min_\theta \mathbb{E}_{(a,i)\sim\mathcal{D}_{train}}[\mathcal{L}_{CE}(F_\theta(a), i)]$。

## 实验关键数据

### 主实验

| 任务 | 源→目标 | 基线(zero-shot) | TOA迁移 | 提升 |
|------|---------|----------------|---------|------|
| 关系抽取(RE) | LLaMA3-3B→Qwen2-1.5B | 7.69% | 34.90% | +27.2 |
| OLA视觉分类(CLM 1st) | L-3b,L-8b→Q-1b5 | - | 52.6% | 远超Rollout(27.9%) |
| OLA视觉分类(CLM 1st) | L-3b,L-8b→G-2b | - | 93.1% | 远超ALTI(69.3%) |
| OLA检索(CLM Hits@5) | L-3b→Q-1b5 | - | 96.1% | 极高检索成功率 |
| OLA视觉分类(MLM 1st) | R-b,R-l,E-b,E-l→B-b | - | 91.9% | 远超Rollout(44.3%) |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 1st order OLA | CLM分类≥49.2% | 所有配置下一阶 OLA 最强 |
| 2nd order OLA | CLM分类略低 | 二阶有更多 attention sink |
| 3rd order OLA | CLM分类继续下降 | 高阶 OLA 区分度降低 |
| Attention Rollout | CLM分类7.7-66.1% | 混合了低效高阶成分 |
| ALTI | CLM分类15.5-85.6% | 基于范数的方法偏向个体特征 |
| 参数扰动控制实验 | OLAS消失 | 确认OLAS是预训练模型的固有属性 |

### 关键发现

- OLAS 是一个普遍现象，在 12 个 LM（6个 CLM + 6个 MLM）上都得到验证
- 一阶 OLA 相似度最高且包含最有效的上下文聚合信息，高阶 OLA 的 attention sink 越严重
- OLA 内在编码了句法依存知识，为其作为跨模型统一表示提供了语言学基础
- TOA 可将源 LM 上训练的 adapter 直接迁移到完全不同架构的目标 LM，无需任何微调
- 参数扰动实验确认 OLAS 来源于预训练参数而非实验设计偏差

## 亮点与洞察

- 从 Attention Rollout 到 OLA 的阶次分解是巧妙的数学洞察：通过展开乘积为有序和，自然消除了不同层数模型的不可比性
- Attention Sink 问题的新解释：在 OLA 框架下，sink 被归因于高阶路径的过度聚合导致低效成分淹没有效信息
- 跨模型注意力共性的发现具有深远意义：暗示不同 LM 在大规模语料上训练后可能收敛到相似的最优注意力模式
- TOA 是首个实现无训练跨模型 adapter 迁移的方法，实用价值显著

## 局限与展望

- OLA 分解假设每层的注意力矩阵为多头平均，丢失了头间的差异信息
- TOA 的下游任务验证主要在 RE/NER/DP/POS 四个 NLP 任务上，未在生成任务或更复杂场景验证
- MLM 上的 OLAS 效果弱于 CLM，可能与 Bert 等模型的双向注意力机制有关
- 未探索如何利用 OLAS 进行更深层的跨模型知识蒸馏或模型融合

## 相关工作与启发

- Attention Rollout (Abnar & Zuidema, 2020) 是 OLA 的直接理论基础
- 与 Relative Representation (Moschella et al., 2023) 的工作互补：后者关注表示空间的对齐，本文关注注意力模式的对齐
- 跨语言 adapter 迁移（Pfeiffer et al., 2020）关注语言间迁移，而 TOA 关注模型间迁移，是正交的维度

## 补充讨论

- OLA 的计算复杂度：一阶 OLA 仅需对各层注意力矩阵求平均，计算开销极低；高阶 OLA 需要矩阵乘法组合但可通过缓存优化
- OLAS 在 CLM（Qwen、Gemma、LLaMA）上的表现显著优于 MLM（BERT、RoBERTa、ELECTRA），可能与自回归注意力的单向性更容易收敛到统一模式有关
- OLA 与句法依存的联系为注意力的可解释性研究提供了新视角：低阶 OLA 可能捕获局部句法结构，高阶 OLA 捕获长距离依赖
- TOA 在 RE 任务上将 Qwen2-1.5B 从 7.69% 提升到 34.90%，虽然绝对值不高但证明了跨模型知识迁移的可行性

## 方法细节补充

### OLA 的数学推导

Attention Rollout 的阶次分解本质上是多项式展开。对 $N$ 层模型：
$$\hat{A} = \prod_{i=1}^N (A^{(i)} + I) = \sum_{k=0}^N \binom{N}{k} \hat{A}^{(k)}$$
其中第 $k$ 阶 OLA $\hat{A}^{(k)}$ 是所有经过 $k$ 次注意力聚合的 $\binom{N}{k}$ 条路径效果的归一化平均。这一分解将不同层数模型的注意力统一到同一语义空间——相同阶次代表相同程度的上下文聚合深度。

### TOA 的输入与输出设计

TOA 使用堆叠的一阶和二阶 OLA 作为 adapter 输入特征。在训练阶段，源 LM 被冻结，只训练 adapter 参数。在测试阶段，adapter 直接应用到目标 LM 生成的 OLA 上，无需任何参数调整。这种设计利用了 OLAS 提供的自然跨模型对齐，避免了传统方法需要的特征空间变换或对齐训练。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统发现并验证 OLAS 现象，OLA 分解思路原创性强
- 实验充分度: ⭐⭐⭐⭐ 12 个模型的定性定量分析全面，控制实验排除了混杂因素
- 写作质量: ⭐⭐⭐⭐ 从现象发现到理论解释到应用的逻辑清晰
- 价值: ⭐⭐⭐⭐ OLAS 发现对理解 LM 内部机制有重要意义，TOA 有实际应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments](keydiff_key_similarity-based_kv_cache_eviction_for_long-context_llm_inference_in.md)
- [\[NeurIPS 2025\] Geometry of Decision Making in Language Models](geometry_of_decision_making_in_language_models.md)
- [\[NeurIPS 2025\] LittleBit: Ultra Low-Bit Quantization via Latent Factorization](littlebit_ultra_low-bit_quantization_via_latent_factorization.md)
- [\[NeurIPS 2025\] Disentangling Latent Shifts of In-Context Learning with Weak Supervision](disentangling_latent_shifts_of_in-context_learning_with_weak_supervision.md)
- [\[ACL 2026\] Establishing a Scale for Kullback–Leibler Divergence in Language Models Across Various Settings](../../ACL2026/model_compression/establishing_a_scale_for_kullback-leibler_divergence_in_language_models_across_v.md)

</div>

<!-- RELATED:END -->
