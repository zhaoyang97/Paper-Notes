---
title: >-
  [论文解读] GRACE: Graph-Based Contextual Debiasing for Fair Visual Question Answering
description: >-
  [ECCV 2024][社会计算][视觉问答] 提出 GRACE（GRAph-based Contextual DEbiasing），一种基于图结构的上下文去偏方法，通过无监督上下文图学习和基于图的多样化 in-context example 选择，解决知识增强 VQA 系统中大语言模型继承的数据偏差问题。
tags:
  - "ECCV 2024"
  - "社会计算"
  - "视觉问答"
  - "公平性去偏"
  - "图结构学习"
  - "上下文学习"
  - "知识增强VQA"
---

# GRACE: Graph-Based Contextual Debiasing for Fair Visual Question Answering

**会议**: ECCV 2024  
**DOI**: [10.1007/978-3-031-72643-9_11](https://doi.org/10.1007/978-3-031-72643-9_11)  
**代码**: [GitHub](https://github.com/SuperJohnZhang/ContextGraphKVQA)  
**领域**: 社会计算  
**关键词**: 视觉问答, 公平性去偏, 图结构学习, 上下文学习, 知识增强VQA

## 一句话总结

提出 GRACE（GRAph-based Contextual DEbiasing），一种基于图结构的上下文去偏方法，通过无监督上下文图学习和基于图的多样化 in-context example 选择，解决知识增强 VQA 系统中大语言模型继承的数据偏差问题。

## 研究背景与动机

大语言模型（LLMs）在知识增强视觉问答（knowledge-based VQA）中发挥了重要作用。通过条件化 in-context examples 和任务特定提示，LLMs 能全面理解输入问题并提供与上下文相关的答案。然而，这种依赖 in-context examples 的方式使 LLMs 容易继承上下文描述和示例中的数据集偏差。

**偏差来源分析**：

**语言先验偏差**：VQA 数据集中存在明显的答案分布偏斜。例如，对于"What sport is this?"类问题，"tennis"可能在训练集中占据不成比例的频率，导致模型学会捷径而非真正理解图像

**视觉-语言关联偏差**：某些视觉特征与特定答案之间存在虚假关联。例如，看到"厨房"场景就倾向回答"cooking"，而忽略实际问题内容

**社会偏差**：模型可能对不同性别、种族等群体产生不公平的预测差异

**In-context example 的偏差放大**：当 LLM 使用有偏的示例进行推理时，偏差不仅被继承还可能被放大

现有去偏方法主要关注训练层面（如数据增强、反事实训练），但较少关注如何在 LLM 的 in-context learning 范式下实现去偏。这一问题在知识增强 VQA 中尤为突出，因为模型需要检索外部知识和示例来辅助推理。

## 方法详解

### 整体框架

GRACE 由两个核心组件构成，形成一个端到端的去偏框架：

1. **无监督上下文图学习**（Unsupervised Context Graph Learning）：构建公平约束下的平衡上下文图
2. **基于图的多样化 Prompt 增强**（Graph-Based Diverse Prompt Enhancement）：利用上下文图选择语义相关且多样化的 in-context examples

整体流程：输入 VQA 样本 → 提取视觉和文本特征 → 基于上下文图检索多样化示例 → 构建去偏后的 prompt → LLM 进行推理生成答案。

### 关键设计

#### 组件一：无监督上下文图学习

该组件的核心目标是构建一个**平衡的上下文图**，使不同类别的示例能被公平地检索和使用。

**图构建过程**：

1. **节点定义**：每个 VQA 训练样本作为图中的一个节点，节点特征包含视觉特征（从图像编码器提取）和文本特征（从问题和答案编码）的融合表示
2. **边构建**：基于节点间的语义相似度建立边连接。采用 k-近邻策略，每个节点连接语义上最相近的 k 个节点
3. **公平性约束**：在图学习过程中引入公平性正则化项，确保不同答案类别、不同属性组（如性别）的节点在图中具有相似的连接模式和分布特征

**公平性约束的实现**：

定义公平性损失函数，惩罚图中不同组别间节点连接的不平衡性。具体而言，通过限制图的节点度分布和边权重分布在不同组别间的差异，防止高频答案类别的示例过度聚集、主导检索过程。

$$\mathcal{L}_{fair} = \sum_{g \in G} D_{KL}(P_{deg}^{g} \| P_{deg}^{uniform})$$

其中 $P_{deg}^{g}$ 为组别 $g$ 的度分布，$P_{deg}^{uniform}$ 为均匀分布。

**无监督学习**：图结构的学习不需要显式的偏差标注，通过对比学习和聚类约束实现自组织，使语义相似但答案多样的节点被连接在一起。

#### 组件二：基于图的多样化 Prompt 增强

在构建好上下文图后，该组件利用图结构来选择高质量的 in-context examples，同时考虑两个维度：

**语义相关性**：
- 给定一个查询 VQA 样本，在图中找到其最近的邻居节点
- 使用图上的随机游走或消息传递机制，探索与查询语义相关的样本空间
- 确保检索到的示例与查询有足够的语义共性

**多样性约束**：
- 在选择 in-context examples 时，引入多样性惩罚（diversity penalty），避免选择过于相似的示例
- 利用图的社区结构，从不同的社区/子图中均匀采样示例
- 确保所选示例覆盖不同的答案类型和推理模式

**Prompt 构建**：

最终的 prompt 由以下部分组成：
1. **任务描述**：定义 VQA 任务的基本要求
2. **去偏 In-context Examples**：通过图选择的多样化示例，每个示例包含图像描述、问题和答案
3. **查询问题**：当前需要回答的 VQA 样本

这种基于图的示例选择策略有效打破了高频答案的统治地位，使 LLM 在推理时能考虑更多样的答案可能性。

#### 推理流程与知识整合

GRACE 的推理过程整合了外部知识检索：

1. **视觉理解**：使用预训练视觉模型（如 BLIP-2）对图像进行理解，生成图像描述（caption）
2. **知识检索**：基于图像内容和问题，从知识库中检索相关知识片段
3. **图引导的示例选择**：在上下文图中检索去偏后的 in-context examples
4. **LLM 推理**：将知识、示例和查询整合为 prompt，输入 LLM 生成最终答案

### 损失函数 / 训练策略

**总体损失函数**由三部分组成：

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda_1 \mathcal{L}_{fair} + \lambda_2 \mathcal{L}_{graph}$$

其中：
- $\mathcal{L}_{task}$：VQA 任务损失，衡量答案预测的准确性
- $\mathcal{L}_{fair}$：公平性约束损失，确保图的均衡性
- $\mathcal{L}_{graph}$：图结构学习损失，包括对比学习和图重建目标
- $\lambda_1, \lambda_2$：平衡系数

**训练流程**：
1. 预训练阶段：无监督学习上下文图结构
2. 微调阶段：在具体 VQA 任务上微调图结构和示例选择策略

## 实验关键数据

### 主实验

GRACE 在三个基准数据集上进行了评估：

**表1：分布内评估 (OK-VQA)**

| 方法 | 整体准确率 | 去偏后提升 |
|------|-----------|-----------|
| 无去偏基线 | 基准 | - |
| 随机示例选择 | 基准+Δ₁ | 有限 |
| 相似度检索 | 基准+Δ₂ | 中等 |
| **GRACE** | **最优** | **显著提升** |

**表2：分布外泛化评估**

| 方法 | VQA-CP (OOD) | GQA-OOD | OK-VQA |
|------|-------------|---------|--------|
| 标准 LLM prompting | 受偏差影响大 | 受偏差影响大 | 基准 |
| 反事实去偏 | 部分改善 | 部分改善 | 可能降低 |
| 数据增强去偏 | 改善 | 改善 | 保持 |
| **GRACE** | **最优/次优** | **最优/次优** | **保持或提升** |

GRACE 在 OOD 数据集上表现尤为突出，说明基于图的去偏方法具有更好的泛化能力——不仅在训练分布内有效，在分布偏移场景下也能保持鲁棒。

### 消融实验

**组件消融研究**：

| 配置 | 公平约束 | 多样化选择 | VQA-CP | OK-VQA |
|------|---------|-----------|--------|--------|
| 基线 | ✗ | ✗ | 基准 | 基准 |
| +公平约束 | ✓ | ✗ | 提升++ | 保持 |
| +多样化选择 | ✗ | ✓ | 提升+ | 提升+ |
| **GRACE (完整)** | ✓ | ✓ | **提升+++** | **提升+** |

消融实验证明两个组件互补：公平约束主要改善 OOD 泛化能力，多样化选择同时改善分布内和分布外性能。

**性别公平性分析**：

| 指标 | 基线方法 | GRACE |
|------|---------|-------|
| 男性组准确率 | 高 | 保持 |
| 女性组准确率 | 相对低 | 提升 |
| 性别差距 (Gap) | 较大 | **显著缩小** |
| Equalized Odds | 不满足 | **更接近满足** |

GRACE 在减少性别组间性能差距方面表现出色，体现了其促进社会公平的潜力。

### 关键发现

1. **图结构有效建模上下文关系**：与简单的相似度检索相比，图结构能捕获更丰富的高阶语义关系，使示例选择更加合理
2. **公平约束是 OOD 泛化的关键**：在分布偏移场景下，公平约束带来的提升最为显著，说明数据偏差是 OOD 性能下降的主要原因之一
3. **多样化示例提升推理质量**：从不同社区采样的示例为 LLM 提供了更全面的推理视角，减少了答案的单一性
4. **去偏与性能可以共赢**：GRACE 在减少偏差的同时通常不损害甚至提升整体准确率，打破了"公平性-准确率权衡"的传统认知

## 亮点与洞察

1. **问题视角的新颖性**：首次系统性地研究 LLM 的 in-context learning 范式下 VQA 的偏差问题，填补了去偏方法在 ICL 设置中的空白
2. **图作为去偏工具的创新**：利用图的结构特性（社区、度分布、连通性）来实施公平性约束，比简单的统计方法更优雅且有效
3. **无监督去偏框架**：不需要显式的偏差类型标注，通过构建平衡图结构自动实现去偏，具有更好的通用性
4. **兼顾效果与伦理**：将社会公平性（如性别公平）纳入VQA评估维度，推动了该领域对伦理问题的关注
5. **即插即用的设计**：两个组件（图学习 + 多样化选择）具有良好的模块化特性，可与不同的 LLM 和 VQA 框架组合使用

## 局限与展望

1. **图构建成本**：大规模数据集上构建和维护上下文图可能带来较高的计算和存储开销
2. **公平性定义的局限**：当前主要关注性别维度的公平性，未来可扩展到种族、年龄等更多敏感属性
3. **动态上下文图**：当前图结构在训练后固定，未来可探索随推理过程动态更新的图结构
4. **多模态偏差的深度分析**：仅提供了语言层面的去偏，视觉编码器本身的偏差可能需要额外处理
5. **更大规模的 LLM 验证**：当前实验主要基于特定 LLM，在更大规模模型（如 GPT-4、LLaMA-65B）上的效果有待验证
6. **与因果推理方法的结合**：当前方法偏向统计去偏，未来可与因果推理框架结合，从根源上消除虚假关联

## 相关工作与启发

- **VQA 去偏方向**：与 CSS（反事实样本合成）、LMH（学习混合）、D-VQA（因果去偏）等方法相比，GRACE 的优势在于不需要修改模型训练过程，而是在推理时通过示例选择实现去偏
- **图神经网络在 VQA 中的应用**：此前图主要用于建模场景图、问题图或知识图谱，GRACE 创新性地将图用于建模"训练样本间关系"以辅助去偏
- **In-context Learning 研究**：GRACE 为 ICL 的示例选择策略提供了新方向——不仅考虑相关性，还需考虑公平性和多样性
- **启发**：
  1. 偏差问题不仅存在于模型参数中，还存在于推理过程的上下文组织中
  2. 图结构是处理公平性约束的自然工具，可推广到其他需要平衡性的场景
  3. 未来的 prompt engineering 应将公平性纳入设计考量

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4.0 | 将图学习引入ICL去偏是新颖的思路，问题导向性强 |
| 技术深度 | 3.5 | 框架设计合理，但图学习部分的理论深度可进一步加强 |
| 实验充分性 | 4.0 | ID/OOD数据集覆盖全面，包含公平性分析 |
| 表达清晰度 | 3.5 | 方法描述清晰，但部分细节需要参考代码 |
| **总体** | **3.5** | 将公平性问题与ICL架构结合，是有意义的交叉研究方向 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] GG-BBQ: German Gender Bias Benchmark for Question Answering](../../ACL2025/social_computing/gg-bbq_german_gender_bias_benchmark_for_question_answering.md)
- [\[ICLR 2026\] SAGE: Spatial-visual Adaptive Graph Exploration for Efficient Visual Place Recognition](../../ICLR2026/social_computing/sage_spatial-visual_adaptive_graph_exploration_for_efficient_visual_place_recogn.md)
- [\[ACL 2026\] Bayesian Social Deduction with Graph-Informed Language Models](../../ACL2026/social_computing/bayesian_social_deduction_with_graph-informed_language_models.md)
- [\[ICLR 2026\] Adaptive Debiasing Tsallis Entropy for Test-Time Adaptation](../../ICLR2026/social_computing/adaptive_debiasing_tsallis_entropy_for_test-time_adaptation.md)
- [\[NeurIPS 2025\] GraphKeeper: Graph Domain-Incremental Learning via Knowledge Disentanglement and Preservation](../../NeurIPS2025/social_computing/graphkeeper_graph_domain-incremental_learning_via_knowledge_disentanglement_and_.md)

</div>

<!-- RELATED:END -->
