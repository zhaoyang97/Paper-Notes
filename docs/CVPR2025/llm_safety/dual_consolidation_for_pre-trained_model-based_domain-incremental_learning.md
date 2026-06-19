---
title: >-
  [论文解读] Dual Consolidation for Pre-Trained Model-Based Domain-Incremental Learning
description: >-
  [CVPR 2025][LLM安全][域增量学习] 提出Duct方法，通过表征合并（累加任务向量构建统一嵌入空间）和分类器合并（利用类别语义信息通过最优传输估计旧域分类器权重），在预训练模型基础上实现无样本存储的域增量学习，在四个基准上以1~7%的优势超越SOTA。 领域现状：域增量学习（DIL）要求模型在标签空间不变的情况…
tags:
  - "CVPR 2025"
  - "LLM安全"
  - "域增量学习"
  - "模型合并"
  - "分类器校准"
  - "任务向量"
  - "预训练模型"
---

# Dual Consolidation for Pre-Trained Model-Based Domain-Incremental Learning

**会议**: CVPR 2025  
**arXiv**: [2410.00911](https://arxiv.org/abs/2410.00911)  
**代码**: [https://github.com/Estrella-fugaz/CVPR25-Duct](https://github.com/Estrella-fugaz/CVPR25-Duct)  
**领域**: LLM评测  
**关键词**: 域增量学习、模型合并、分类器校准、任务向量、预训练模型

## 一句话总结
提出Duct方法，通过表征合并（累加任务向量构建统一嵌入空间）和分类器合并（利用类别语义信息通过最优传输估计旧域分类器权重），在预训练模型基础上实现无样本存储的域增量学习，在四个基准上以1~7%的优势超越SOTA。

## 研究背景与动机

**领域现状**：域增量学习（DIL）要求模型在标签空间不变的情况下适应不断变化的数据域（如不同天气/风格），同时不遗忘旧域知识。基于预训练模型（PTM）的方法通过冻结backbone +可训练prompt来编码域特定知识（如L2P、CODA-Prompt）。

**现有痛点**：遗忘发生在两个层面——（1）表征层面：顺序更新prompt/backbone使特征偏向最新域；（2）分类器层面：分类器与持续变化的特征空间不匹配。现有prompt-based方法虽冻结backbone，但可训练的prompt池仍会被覆盖，且分类器偏置未被解决。

**核心矛盾**：DIL需要一个适合所有已见域的统一嵌入空间，但流式数据到达使我们无法同时访问所有域的训练数据来构建这个空间。

**本文目标** 在不存储旧样本的前提下，同时解决PTM-based DIL中的表征遗忘和分类器遗忘问题。

**切入角度**：借鉴模型合并（model merging）思想，将每个域独立微调产生的"任务向量"累加到预训练权重上，构建覆盖所有域的统一表征；然后利用类别语义关系通过最优传输估计旧域分类器在新嵌入空间中的权重。

**核心 idea**：用任务向量累加合并表征空间、用语义信息驱动的最优传输对齐旧分类器，从表征和分类器双层面抵抗域增量学习中的遗忘。

## 方法详解

### 整体框架
每个新域到来时：（1）从预训练模型初始化，在新域数据上微调得到域专家模型；（2）提取任务向量$\delta_{\phi_i} = \phi_i - \phi_0$，按任务相似度加权累加得到统一backbone $\phi_i^m$；（3）在统一backbone上重训新域分类器；（4）用最优传输将新域分类器的类别语义关系迁移到旧域分类器的估计上。推理时仅用一个合并后的backbone+合并分类器。

### 关键设计

1. **表征合并（Representation Consolidation）**:

    - 功能：构建适合所有已见域的统一嵌入空间
    - 核心思路：每个域独立微调后得到任务向量$\delta_{\phi_k}$，统一backbone为$\phi_i^m = \phi_0 + \alpha_\phi \sum_{k=1}^{i} \text{Sim}_{0,k} \cdot \delta_{\phi_k}$。任务相似度$\text{Sim}_{0,k}$通过计算预训练模型和域专家模型在当前数据上的类中心余弦相似度来衡量。因为任务向量基于同一预训练权重且不同域间语义差距大，它们低相似度使累加有效
    - 设计动机：避免顺序更新导致的遗忘——每个域的任务向量独立于其他域计算，合并是非破坏性的。可增量执行：$\phi_i^m = \phi_{i-1}^m + \alpha_\phi \text{Sim}_{0,i} \delta_{\phi_i}$，只需存2个backbone

2. **分类器合并 - 新分类器重训**:

    - 功能：让当前域的分类器匹配合并后的嵌入空间
    - 核心思路：冻结合并backbone $\phi_i^m$，在当前域数据上重新训练分类器$W_n$。属于标准的线性探针步骤
    - 设计动机：合并后的嵌入空间与微调时的不同，分类器必须重新对齐

3. **分类器合并 - 旧分类器传输（Old Classifier Transport）**:

    - 功能：在无旧域样本的情况下估计旧域分类器在新嵌入空间中的权重
    - 核心思路：利用新域分类器中编码的类间语义关系来估计旧域分类器。具体地，构建一个基于类别嵌入的语义传输矩阵S，用最优传输（Sinkhorn算法）求解类别间的最优匹配关系，然后将这个关系作用于新域分类器得到旧域分类器的估计$\hat{W}_o = \mathcal{T}(W_n, S)$。最终旧分类器为历史分类器和估计值的加权合并
    - 设计动机："狮子"在不同域（剪贴画 vs 真实照片）的分类器权重有语义关联性，利用新域学到的类间关系可以推断旧域的分类器应该是什么样

### 损失函数 / 训练策略
微调阶段用标准交叉熵损失，分类器重训也用交叉熵。使用cosine分类器。域到来后独立微调15个epoch，SGD lr=0.001。推理时仅需单个合并backbone，无额外推理开销。

## 实验关键数据

### 主实验

| 数据集 | Duct $\bar{\mathcal{A}}$ | Duct $\mathcal{A}_B$ | CODA-Prompt | S-iPrompt | 提升 |
|--------|---------|---------|-------------|-----------|------|
| Office-Home | 86.27% | 86.91% | 85.07% | 80.51% | +1.8% |
| DomainNet | 67.16% | 67.01% | 59.99% | 60.46% | +7.0% |
| CORe50 | 91.95% | 94.47% | 91.57% | 83.38% | +2.9% |
| CDDB | 84.14% | 85.10% | 74.18% | 72.76% | +10.9% |

### 消融实验

| 配置 | CDDB $\mathcal{A}_B$ | 说明 |
|------|---------------------|------|
| Baseline（冻结backbone，类中心分类器） | ~63% | 无任何合并 |
| +表征合并 | ~79% | backbone合并大幅提升 |
| +新分类器重训 | ~83% | 对齐分类器有效 |
| +旧分类器传输 (Full Duct) | 85.10% | 恢复旧域知识+2% |

### 关键发现
- 表征合并是最关键模块，贡献最大的性能提升（~16%），说明统一嵌入空间对DIL至关重要
- Duct的遗忘度量最低（0.12），远低于Finetune和MEMO，说明双层合并有效抵抗遗忘
- 跨5种任务顺序的标准差小（如CORe50上±0.15），说明方法对域到达顺序鲁棒
- 推理时仅需单个backbone，无额外计算开销，不同于prompt方法需要prompt选择
- 即使不存储任何旧样本，也超越了存储样本的Replay方法（如CDDB上 85.1% vs 63.2%）

## 亮点与洞察
- **模型合并思路的创新应用**：将model merging从多任务学习迁移到增量学习，利用任务向量的低相似性保证合并的有效性。这个思路简洁且理论动机清晰
- **旧分类器传输的巧妙设计**：在无旧样本的情况下利用类间语义关系通过OT估计旧分类器，避免了replay buffer的需求
- **增量存储高效**：只需存储当前合并backbone和一个在训backbone，推理时只用一个模型，O(1)存储

## 局限与展望
- 任务向量累加的缩放系数$\alpha_\phi$是全局固定的，不同域可能需要不同的合并强度
- 旧分类器传输依赖语义信息的质量，如果域间语义关系差异极大可能不准确
- 仅在ViT-B/16上验证，缺少在更大/更小模型上的实验
- 任务向量方法假设所有域从同一预训练权重出发，对非共享初始化的场景不适用

## 相关工作与启发
- **vs L2P/CODA-Prompt**: 这些方法通过prompt池编码域知识但prompt本身会被覆盖；Duct直接在权重空间合并域信息，更稳定
- **vs S-iPrompt**: S-iPrompt用KNN检索域特定prompt，依赖准确的域识别；Duct在统一空间直接分类，不需要域识别步骤
- **vs Model Merging (Task Arithmetic)**: Duct将原本用于静态多任务的模型合并扩展到增量场景，增加了任务相似度加权和分类器传输，是model merging在CL中的首次系统应用

## 评分
- 新颖性: ⭐⭐⭐⭐ 双层合并框架设计优雅，模型合并+OT分类器传输的组合新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 四数据集、五任务顺序、多baseline、消融和遗忘度量
- 写作质量: ⭐⭐⭐⭐ 问题动机分析透彻，方法推导清晰
- 价值: ⭐⭐⭐⭐ 对PTM-based增量学习有重要推动，无样本存储即超越replay方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Low-Rank Adaptation in Multilinear Operator Networks for Security-Preserving Incremental Learning](low-rank_adaptation_in_multilinear_operator_networks_for_security-preserving_inc.md)
- [\[AAAI 2026\] LAMP: Learning Universal Adversarial Perturbations for Multi-Image Tasks via Pre-trained Models](../../AAAI2026/llm_safety/lamp_learning_universal_adversarial_perturbations_for_multi-image_tasks_via_pre-.md)
- [\[ACL 2025\] Exploring Forgetting in Large Language Model Pre-Training](../../ACL2025/llm_safety/exploring_forgetting_in_large_language_model_pre-training.md)
- [\[ICLR 2026\] Lifelong Learning with Behavior Consolidation for Vehicle Routing](../../ICLR2026/llm_safety/lifelong_learning_with_behavior_consolidation_for_vehicle_routing.md)
- [\[ICLR 2026\] Unmasking Backdoors: An Explainable Defense via Gradient-Attention Anomaly Scoring for Pre-trained Language Models](../../ICLR2026/llm_safety/unmasking_backdoors_an_explainable_defense_via_gradient-attention_anomaly_scorin.md)

</div>

<!-- RELATED:END -->
