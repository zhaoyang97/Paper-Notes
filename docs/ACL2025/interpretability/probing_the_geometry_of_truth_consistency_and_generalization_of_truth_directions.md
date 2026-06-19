---
title: >-
  [论文解读] Probing the Geometry of Truth: Consistency and Generalization of Truth Directions
description: >-
  [ACL 2025][可解释性][真值方向] 系统性研究LLM内部"真值方向"(truth direction)的一致性与泛化能力，发现只有能力较强的模型才稳定展现一致的真值方向，且基于简单原子陈述训练的真实性探针可泛化至逻辑变换、问答任务和上下文知识场景。 问题背景 LLM在大规模语料上训练后拥有丰富知识…
tags:
  - "ACL 2025"
  - "可解释性"
  - "真值方向"
  - "真实性探测"
  - "LLM内部表征"
  - "线性探针"
  - "泛化性"
---

# Probing the Geometry of Truth: Consistency and Generalization of Truth Directions

**会议**: ACL 2025  
**arXiv**: [2506.00823](https://arxiv.org/abs/2506.00823)  
**作者**: Yuntai Bao, Xuhong Zhang, Tianyu Du, Xinkui Zhao, Zhengwen Feng, Hao Peng, Jianwei Yin (浙江大学, 浙江师范大学)
**代码**: [GitHub](https://github.com/colored-dye/truthfulness_probe_generalization)  
**领域**: 可解释性  
**关键词**: 真值方向, 真实性探测, LLM内部表征, 线性探针, 泛化性

## 一句话总结

系统性研究LLM内部"真值方向"(truth direction)的一致性与泛化能力，发现只有能力较强的模型才稳定展现一致的真值方向，且基于简单原子陈述训练的真实性探针可泛化至逻辑变换、问答任务和上下文知识场景。

## 研究背景与动机

### 问题背景
LLM在大规模语料上训练后拥有丰富知识，但输出中常包含自信地陈述的不实信息。前期工作（Burns et al. 2022; Marks & Tegmark 2023）发现LLM内部存在一个线性特征——"真值方向"(truth direction)，可通过轻量级分类器（探针）从模型隐藏状态中判断陈述的真实性。

### 已有工作的不足
- 已有工作普遍**假设所有LLM都存在一致的真值方向**，缺乏对该假设的系统检验
- Levinstein & Herrmann (2024) 声称探针无法跨逻辑否定泛化，但将原因归咎于探针设计不够复杂
- 真实性探针从陈述句到问答场景的**语法形式泛化**尚未被充分研究
- 探针是否能从**参数化知识**泛化到**上下文知识**（如阅读理解、摘要生成）仍不明确

### 核心研究问题
- **RQ1**: LLM是否普遍将真实性表示为线性特征？
- **RQ2**: 是否需要复杂的探针技术来识别真值方向？
- **RQ3**: 真值方向在多大程度上可以泛化？

## 方法详解

### 探针形式化定义

给定Transformer语言模型，输入token序列 $t = (t_1, t_2, \ldots, t_n)$ 经过 $L$ 层处理后得到各层表征 $\boldsymbol{h}_i^{(l)} \in \mathbb{R}^d$。对于自回归模型，取第 $l$ 层最后一个token位置的表征 $\boldsymbol{h}_{-1}^{(l)}$ 作为探针输入。给定标注数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^M$，目标是学习探针 $\Phi$ 使分类误差最小化：

$$\underset{\Phi}{\arg\min} \frac{1}{M} \sum_{i=1}^{M} J(\Phi, \boldsymbol{h}_i, y_i)$$

### 设计1：几何导向探针

基于"真值方向假设"——真/假表征可被一个超平面分离，超平面法向量即为真值方向：

- **线性SVM探针**: 最大化分离边距，通过Platt scaling后验校准获得概率输出。使用NuSVC实现，$\nu=0.5$，5折交叉验证进行Platt scaling
- **Mass-Mean (MM) 探针**: 计算真/假两类表征的质心，以质心连线方向作为真值方向（Marks & Tegmark 2023）。分类时按样本到两个质心的距离投票

### 设计2：统计导向探针

不对几何结构做假设，直接最大化标签似然：

- **逻辑回归 (LR)**: 使用L-BFGS优化，作为通用基线
- **多层感知器 (MLP/SAPLMA)**: 隐层结构 $(512, 128, 64)$，tanh激活，Adam优化器训练至收敛

### 最优层选择策略
通过计算各层**类间方差与类内方差之比**来选择最优层。对于Llama-3.1-8B，第12层（零索引）方差比最高；对于Llama-2-7B，仅sp_en_trans主题出现明显峰值，说明弱模型的真值方向不一致。

### 数据构造
使用Bürger et al. (2024) 整理的事实陈述数据，涵盖6个主题：animal_class、cities、element_symb、facts、inventors、sp_en_trans。每个主题包含肯定陈述、否定陈述、逻辑合取和逻辑析取四种变体。

## 实验关键数据

### 实验1：真值方向一致性（跨逻辑否定泛化）

在肯定陈述上训练探针，在否定陈述上测试。模型能力从弱到强排列：

| 模型 | 泛化成功的主题数(/6) | 模型参数量 |
|------|---------------------|-----------|
| Llama-2-7B | 0/6 | 7B |
| Llama-2-7B-Chat | 0/6 | 7B |
| Llama-2-13B | 4/6 | 13B |
| Llama-2-13B-Chat | 4/6 | 13B |
| Llama-3.1-8B | 4/6 | 8B |
| Llama-3.1-8B-Instruct | 4/6 | 8B |
| Llama-3.1-70B | 5/6 | 70B |
| Llama-3.1-70B-Instruct | **6/6** | 70B |

结论：真值方向的一致性与模型能力正相关，只有最强模型在所有主题上都展现一致的真值方向。

### 实验2：问答任务泛化（MMLU & TriviaQA）

在原子事实陈述上训练，在MMLU和TriviaQA上测试（Llama-3.1-8B）：

| 数据集 | Prompt设置 | SVM AUROC↑ | SVM ECE↓ | SVM BS↓ |
|--------|-----------|------------|----------|---------|
| MMLU | zero-shot | ~0.60 | ~0.15 | ~0.24 |
| MMLU | TTTTT (5-shot全对) | ~0.65 | ~0.12 | ~0.22 |
| MMLU | TTFFF (含错误示例) | ~0.65 | ~0.12 | ~0.22 |
| TriviaQA | 5-shot | ~0.70 | ~0.15 | ~0.22 |
| TriviaQA | 20-shot | ~0.72 | ~0.10 | ~0.20 |

关键发现：含错误few-shot示例的效果与全对示例几乎一致——探针仅关注最后一个(Q,A)对的真实性，将前面的示例视为上下文。

### 实验3：选择性问答应用

使用TriviaQA 20-shot设置，SVM探针筛选LLM回答：

| 指标 | 数值 |
|------|------|
| 全部回答的准确率 | 55.29% |
| 探针判断为真的比例 | 80.26% |
| 筛选后子集准确率 | **64.06%** |

通过探针过滤，准确率提升约9个百分点。

### 实验4：上下文知识泛化

在原子事实陈述上训练，在需要上下文知识的任务上测试（Llama-3.1-8B）：

| 数据集 | 任务类型 | 泛化结果 |
|--------|---------|---------|
| SciQ | 上下文多选QA | AUROC > 0.5，成功泛化 |
| BoolQ | 上下文是非判断 | AUROC > 0.5，成功泛化 |
| XSum | 摘要忠实度检测 | AUROC > 0.5，成功泛化 |

## 核心发现

1. **并非所有LLM都有一致的真值方向**：Llama-2-7B完全无法跨否定泛化，而Llama-3.1-70B-Instruct在所有主题上完美泛化
2. **简单探针足以识别真值方向**：当模型能力足够时，LR、SVM、MLP、MM四种简单探针的表现差异可忽略
3. **真值方向是预训练产物**：随机初始化模型上的探针AUROC约0.5（随机水平），预训练权重上达1.0
4. **逻辑合取的泛化优于析取**：可能因为析取的真值计算对LLM更具挑战性
5. **探针对错误few-shot示例具有鲁棒性**：探针仅提取最终(Q,A)对的真实性，不受上下文中错误示例影响
6. **域外探针优于域内探针**：在MMLU上，用原子陈述训练的探针反而优于MMLU域内训练的探针

## 亮点

- **系统性回答三个核心问题**：不假设真值方向普遍存在，而是通过8个模型×4种探针×多种任务的大量实验给出有数据支撑的回答
- **挑战前人结论**：指出Levinstein & Herrmann (2024)的泛化失败不是探针问题，而是模型本身缺乏一致真值表征
- **实际应用演示**：选择性QA场景展示了探针的实用价值——无需修改模型即可提升回答可靠性
- **实验设计严谨**：随机化模型实验排除了探针"自造"真值方向的可能性；含错误示例的few-shot实验揭示了探针的鲁棒性机制

## 局限与展望

- **"真实性"定义模糊**：探针可能捕获的是人类广泛共识而非客观事实，对超人类AI系统的适用性存疑
- **仅测试短文本QA**：未涉及长文本QA、指令遵循等更复杂场景
- **因果关系不明**：未证明LLM是否在生成时实际使用了真值方向，仅展示了相关性
- **计算资源限制**：最大模型仅70B，未在GPT-4等更强模型上验证假设
- **仅覆盖Llama系列为主**：Mistral仅作为附录补充，缺乏更广泛的模型家族对比

## 与相关工作的对比

- **Burns et al. (2022, CCS)**: 无监督方法，面向yes/no QA；本文使用有监督探针，泛化测试范围更广
- **Marks & Tegmark (2023)**: 提出真值方向概念和MM探针；本文系统验证其一致性和泛化边界
- **Bürger et al. (2024, TTPD)**: 提出TTPD探针区分肯定/通用真值方向；本文发现强模型中该区分不必要
- **Levinstein & Herrmann (2024)**: 声称探针无法跨否定泛化；本文证明这是模型能力问题而非探针问题
- **Azaria & Mitchell (2023, SAPLMA)**: 使用MLP探针；本文表明简单线性探针同样有效
- **Sky et al. (2024)**: 检测上下文生成中的幻觉；本文将事实陈述探针泛化到上下文任务

## 评分

- 新颖性: ⭐⭐⭐⭐ — 系统性回答了真值方向的三个开放问题，挑战前人结论
- 实验充分度: ⭐⭐⭐⭐⭐ — 8个模型、4种探针、6类主题、多种下游任务的全面评估
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，研究问题明确，实验设计有层次
- 价值: ⭐⭐⭐⭐ — 对LLM可信度评估和安全对齐具有理论指导意义，选择性QA展示了实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Emergence of Linear Truth Encodings in Language Models](../../NeurIPS2025/interpretability/emergence_of_linear_truth_encodings_in_language_models.md)
- [\[ACL 2026\] How Context Shapes Truth: Geometric Transformations of Statement-level Truth Representations in LLMs](../../ACL2026/interpretability/how_context_shapes_truth_geometric_transformations_of_statement-level_truth_repr.md)
- [\[ACL 2025\] Probing Subphonemes in Morphology Models](probing_subphonemes_in_morphology_models.md)
- [\[NeurIPS 2025\] Representation Consistency for Accurate and Coherent LLM Answer Aggregation](../../NeurIPS2025/interpretability/representation_consistency_for_accurate_and_coherent_llm_answer_aggregation.md)
- [\[CVPR 2025\] Geometry-Guided Camera Motion Understanding in VideoLLMs](../../CVPR2025/interpretability/geometry-guided_camera_motion_understanding_in_videollms.md)

</div>

<!-- RELATED:END -->
