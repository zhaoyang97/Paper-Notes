---
title: >-
  [论文解读] Don't Start Over: A Cost-Effective Framework for Migrating Personalized Prompts Between LLMs
description: >-
  [AAAI 2026 Oral][模型压缩][软提示迁移] 提出PUMA框架，通过轻量级适配器和分组用户选择策略，高效地将个性化软提示从源LLM迁移到不同架构的目标LLM，在三个大规模数据集上匹配甚至超越从头训练的性能，同时减少计算成本高达98%。 随着LLM在推荐系统、个人助手、自适应教育等领域的深入应用…
tags:
  - "AAAI 2026 Oral"
  - "模型压缩"
  - "软提示迁移"
  - "个性化"
  - "LLM"
  - "参数高效适配器"
  - "用户选择策略"
---

# Don't Start Over: A Cost-Effective Framework for Migrating Personalized Prompts Between LLMs

**会议**: AAAI 2026 Oral  
**arXiv**: [2601.12034](https://arxiv.org/abs/2601.12034)  
**代码**: [github](https://github.com/Kimagure7/Dont-Start-Over)  
**领域**: 模型压缩  
**关键词**: 软提示迁移, 个性化, LLM, 参数高效适配器, 用户选择策略

## 一句话总结
提出PUMA框架，通过轻量级适配器和分组用户选择策略，高效地将个性化软提示从源LLM迁移到不同架构的目标LLM，在三个大规模数据集上匹配甚至超越从头训练的性能，同时减少计算成本高达98%。

## 研究背景与动机

随着LLM在推荐系统、个人助手、自适应教育等领域的深入应用，**个性化**成为核心需求。软提示（Soft Prompt）作为一种轻量级个性化技术，形成了独特的"1+N"应用架构：一个通用基础模型（"1"）加上成千上万的用户个性化软提示（"N"）。

然而，这种架构存在一个**致命脆弱性**：

**模型升级必然发生**：基础模型会被升级（更强大的版本）或替换（更小更高效的版本）

**语义对齐断裂**：新模型的嵌入维度和语义空间通常不同（$d_t \neq d_s$），原有软提示完全失效

**重训代价高昂**：对数万甚至数百万用户的软提示全部从头训练，计算成本极其高昂

**核心问题**：能否以极低的计算成本将大规模用户的个性化软提示从源模型迁移到目标模型，同时保持个性化性能？

**与现有工作的区别**：已有的软提示转移研究聚焦于**任务级**转移（将一个公共提示从NLI任务转移到文本分类），是"一对一"问题。本文首次关注**用户级**迁移——将成千上万的私有用户提示从一个模型迁移到另一个模型，是"N对N"的挑战。

**本文的分解思路**：将问题分解为两个耦合子问题：(1) 语义不兼容——如何让目标模型理解源模型上训练的提示；(2) 迁移效率——如何扩展到数万用户的大规模迁移。

## 方法详解

### 整体框架

PUMA（Prompt-level User Migration Adapter）由两个核心组件构成（如图2所示）：

1. **迁移适配器**：端到端训练的轻量前馈网络，桥接源模型和目标模型的语义空间
2. **分组用户选择策略**：基于K-Means聚类和方差分层抽样选取代表性训练子集

### 关键设计

#### 1. **迁移适配器（Migration Adapter）**

将迁移函数 $\Phi$ 实现为带有残差连接和Layer Normalization的前馈网络：

$$p'_u = \Phi_\theta(p_u), \quad p_u \in \mathbb{R}^{l \times d_s}, \; p'_u \in \mathbb{R}^{l \times d_t}$$

关键设计选择：
- **端到端训练**：通过任务损失直接优化，确保转换保持下游任务实用性
- **参数高效**：仅训练适配器参数，目标模型和源提示均冻结
- **架构简洁**：残差连接+LayerNorm平衡表达能力与计算成本

优化目标：

$$\theta^* = \arg\min_\theta \sum_{(u,i,y) \in D} \mathcal{L}_{\text{task}}\left(M_t(T(\Phi_\theta(p_u), \phi(i))), y\right)$$

#### 2. **分组用户选择策略（核心效率创新）**

在数万用户中训练适配器代价仍然很高。核心洞察：理想的用户子集必须同时体现**偏好多样性**和**复杂度谱系**。

**两阶段选择过程**：

**阶段一：K-Means聚类以捕获多样性**
- 对源提示 $\{p_u\}_{u \in \mathcal{U}}$ 进行K-Means聚类，将用户分为k个群组
- 确保选择覆盖不同的学习偏好模式

**阶段二：方差分层抽样以捕获复杂度**
- 在每个聚类内，以历史输出方差分层
- 低方差用户 = 一致性偏好（如总是给高分），易建模
- 高方差用户 = 复杂偏好，建模困难
- 使用正态分布加权，对中等方差组赋予更高权重

#### 3. **高级迁移拓扑**

**链式迁移（Chained Migration）**：$M_A \rightarrow M_B \rightarrow M_C$
- 处理序列化模型更换
- 新迁移的提示成为下一步的源

**聚合迁移（Aggregated Migration）**：$[M_A, M_B] \rightarrow M_C$
- 融合多个源模型的个性化
- 通过拼接用户的源提示 $[p_u^A; p_u^B]$ 并映射到目标模型
- 综合多源信息产生更丰富的用户表征

### 损失函数 / 训练策略

**针对不同任务的损失函数**：

- **评分预测**（Amazon/Yelp）：混合损失 $0.8 \cdot \mathcal{L}_{MSE} + 0.2 \cdot \mathcal{L}_{CE}$
    - 从LLM输出提取5个离散评分token的logits
    - 交叉熵损失处理分类+MLP头回归连续评分值
- **CTR预测**（MIND）：标准二元交叉熵 $\mathcal{L}_{BCE}$，对"yes" token的logit计算

**训练细节**：
- 源提示长度 $l=1$，预训练15个epoch，学习率 $5 \times 10^{-4}$
- PUMA适配器训练4个epoch，FusedAdam优化器，学习率 $10^{-4}$，batch size 32
- 使用NVIDIA A100 GPU，PyTorch 2.5

## 实验关键数据

### 主实验

**PUMA vs 从头训练的性能对比**（LLaMA-2-1B → LLaMA-2-3B）：

| 方法 | Amazon RMSE↓ | Amazon MAE↓ | MIND AUC↑ | MIND uAUC↑ | Yelp RMSE↓ | Yelp MAE↓ |
|------|------------|-----------|---------|----------|----------|---------|
| 随机初始化 | 1.2352 | 1.1168 | 0.4917 | 0.4883 | 1.6671 | 1.4981 |
| 从头训练 | 0.9414 | 0.6296 | 0.5778 | 0.5289 | 1.1994 | 0.9269 |
| **PUMA** | **0.9135** | **0.5701** | **0.6546** | **0.6552** | **1.1073** | **0.8493** |

**效率对比**：

| 方法 | 每轮时间 | 轮数 | 总时间 | 加速比 |
|------|---------|------|--------|--------|
| 从头训练 | 3.00h | 8 | 24.0h | 1x |
| PUMA (2k用户) | 0.16h | 3 | 0.48h | **50x** |

### 消融实验

**用户选择策略对比**（固定预算：Amazon/Yelp 2000用户，MIND 1500用户）：

| 策略 | Amazon RMSE↓ | MIND uAUC↑ | Yelp RMSE↓ |
|------|------------|----------|----------|
| 随机抽样 | 0.9419 | 0.5861 | 1.1146 |
| 随机抽样 (6k) | 0.9320 | 0.6636 | 1.1128 |
| 方差分桶 | 0.9508 | 0.5888 | 1.1171 |
| K-Means分层 | 0.9546 | 0.5927 | 1.1152 |
| K-Means + FPS | 0.9355 | 0.5966 | 1.1147 |
| **K-Means + 方差分层 (PUMA)** | **0.9315** | **0.6344** | **1.1111** |

PUMA仅用2000用户就超过了随机抽样6000用户的效果！

**跨架构迁移（图4热力图）**：

| 源模型 → 目标模型 | 性能增益(Gain) | 说明 |
|-----------------|-------------|------|
| LLaMA → Qwen | >1.0 | 匹配或超越从头训练 |
| LLaMA → Phi-3 | >1.0 | 跨家族迁移有效 |
| Gemma → Phi-3 | <1.0 | 弱源→强目标略逊于从头训练 |
| Phi-3 → LLaMA | >1.0 | 强源→弱目标超越从头训练 |

**链式迁移稳定性**（Llama→Qwen→Gemma→StableLM→Phi-3）：
- 起始RMSE: 0.9348，终止RMSE: 0.9277
- 全链条优于各步的从头训练

**聚合迁移**（双源→Phi-3）：
- Llama+StableLM→Phi-3 RMSE: 0.9217（vs 单源Llama 0.9293，StableLM 0.9380）
- 多源融合产生"知识协同"，不同模型捕获互补的用户偏好

### 关键发现

1. **PUMA超越从头训练**：这出人意料——适配器学到的共享映射函数比独立学习k个用户表征更具泛化性
2. **98%的计算节省**：仅用2000/30000用户训练适配器，50x加速
3. **跨架构迁移有效**：LLaMA→Qwen等不同家族间迁移依然有效
4. **链式迁移稳定**：5步连续迁移后性能不衰退
5. **聚合迁移增强个性化**：多源融合优于单源，揭示"知识协同"原理
6. **迁移效能受源提示质量约束**：弱源→强目标可能略逊于从头训练

## 亮点与洞察

1. **问题定义精准且有实际价值**：大规模个性化提示的模型更迭问题是工业界的真实痛点
2. **PUMA超越从头训练的洞察**深刻——共享映射函数可能通过正则化效应发现更优的语义转换空间
3. **分组用户选择策略**结合多样性（K-Means）和复杂度（方差分层），用1/15的数据达到全量效果
4. **链式和聚合迁移**的成功将迁移从被动维护任务转变为主动增强个性化的战略机会
5. **实验设计全面**：三个数据集、五个模型、两种高级迁移拓扑

## 局限与展望

1. 仅在推荐系统任务上评估，其他个性化场景（对话、写作助手）的适用性未验证
2. 用户选择策略基于静态启发式，可基于强化学习训练选择策略进一步优化
3. 仅迁移用户提示，未考虑同时迁移物品嵌入——全面的知识传递需要两者兼顾
4. 冷启动问题未解决——新用户（目标系统中不存在于源系统的用户）的处理
5. 提示长度固定为1，更长的提示是否能带来更好的迁移效果值得探索

## 相关工作与启发

- **Soft Prompt Tuning** (Lester et al., 2021)：个性化的基础技术，PUMA解决其模型绑定问题
- **SPoT** (Vu et al., 2022)：任务级提示转移，本文扩展到用户级
- **ATTRIPOTION** (Asai et al., 2022)：多提示混合机制，启发了聚合迁移设计
- **Coreset Selection**方法：K-Means/FPS/梯度匹配等，本文设计了更适合此场景的方差分层方法
- 启发：个性化资产应被视为可迁移、可融合的"数字资产"，而非绑定在特定模型上的一次性参数；提示迁移可能成为LLM生态中的基础设施级技术

## 评分
- 新颖性: ⭐⭐⭐⭐⭐（首次形式化并解决用户级软提示跨模型迁移问题）
- 实验充分度: ⭐⭐⭐⭐⭐（三个数据集、五个模型、多种迁移拓扑，实验极为充分）
- 写作质量: ⭐⭐⭐⭐（条理清晰，问题阐述充分）
- 价值: ⭐⭐⭐⭐⭐（解决实际工业痛点，98%成本节省具有直接应用价值）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Highly Efficient and Effective LLMs with Multi-Boolean Architectures](../../ICLR2026/model_compression/highly_efficient_and_effective_llms_with_multi-boolean_architectures.md)
- [\[ACL 2025\] IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](../../ACL2025/model_compression/iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)
- [\[AAAI 2026\] Explore and Establish Synergistic Effects between Weight Pruning and Coreset Selection](explore_and_establish_synergistic_effects_between_weight_pruning_and_coreset_sel.md)
- [\[ICML 2026\] Detecting Fluent Optimization-Based Adversarial Prompts via Sequential Entropy Changes](../../ICML2026/model_compression/detecting_fluent_optimization-based_adversarial_prompts_via_sequential_entropy_c.md)
- [\[AAAI 2026\] HCF: Hierarchical Cascade Framework for Distributed Multi-Stage Image Compression](hcf_hierarchical_cascade_framework_for_distributed_multi-stage_image_compression.md)

</div>

<!-- RELATED:END -->
