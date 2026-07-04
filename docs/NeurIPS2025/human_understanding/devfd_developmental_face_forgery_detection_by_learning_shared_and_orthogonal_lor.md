---
title: >-
  [论文解读] DevFD: Developmental Face Forgery Detection by Learning Shared and Orthogonal LoRA Subspaces
description: >-
  [NEURIPS2025][人体理解][face forgery detection] 提出 DevFD——一种发展式 MoE 架构，用共享 Real-LoRA 建模真实人脸共性、正交 Fake-LoRA 序列逐步建模新伪造类型，并通过将正交梯度集成到正交损失中缓解灾难性遗忘，在持续学习人脸伪造检测中达到最高准确率和最低遗忘率。
tags:
  - "NEURIPS2025"
  - "人体理解"
  - "face forgery detection"
  - "continual learning"
  - "mixture of experts"
  - "LoRA"
  - "orthogonal subspace"
---

# DevFD: Developmental Face Forgery Detection by Learning Shared and Orthogonal LoRA Subspaces

**会议**: NEURIPS2025  
**arXiv**: [2509.19230](https://arxiv.org/abs/2509.19230)  
**代码**: 待确认  
**领域**: 人体理解  
**关键词**: face forgery detection, continual learning, mixture of experts, LoRA, orthogonal subspace

## 一句话总结
提出 DevFD——一种发展式 MoE 架构，用共享 Real-LoRA 建模真实人脸共性、正交 Fake-LoRA 序列逐步建模新伪造类型，并通过将正交梯度集成到正交损失中缓解灾难性遗忘，在持续学习人脸伪造检测中达到最高准确率和最低遗忘率。

## 背景与动机
- GAN、Diffusion 等生成模型快速迭代，人脸伪造技术不断涌现，现有检测器难以泛化到新伪造类型
- 简单地将新伪造数据加入训练集重训模型计算开销大，且会导致对旧伪造类型的灾难性遗忘
- **关键观察**：真实人脸数据丰富、采集方式单一（相机成像），跨数据集分布紧凑；而伪造人脸随技术不同呈现多个离散聚类（t-SNE 可视化验证）
- 这一不对称性启发作者分别建模：用单一共享专家学习真实人脸共性，用可扩展的正交专家序列逐步捕捉不同伪造类型的特征

## 核心问题
1. **持续学习场景下的灾难性遗忘**：学习新伪造类型时如何不丢失之前已学习的检测能力？
2. **真实/伪造人脸的分布不对称性**：如何利用真脸分布紧凑、假脸分布分散的特点设计更高效的架构？
3. **正交子空间约束的不完备性**：现有正交 LoRA 方法在训练早期正交性条件未满足时，梯度仍会干扰已建立的子空间

## 方法详解

### 整体框架
- 冻结预训练 ViT 骨干网络，在每个 Transformer block 的 FFN 层引入 LoRA 作为 MoE 的专家
- 架构随任务增长而发展（developmental）：每遇到新伪造类型，扩展一个新 LoRA 分支

### 1. 发展式 MoE 架构（Developmental MoE）
- **Real-LoRA**（$\mathbf{A}_0\mathbf{B}_0$）：共享专家，在所有任务中持续可训练，专门学习和精炼真实人脸的通用知识
- **Fake-LoRA 序列**（$\{\mathbf{A}_j\mathbf{B}_j\}_{j=1}^{t}$）：正交专家序列，每个新任务扩展一个分支，之前的分支冻结
- 输出公式：$\mathbf{e} = \text{FFN}(\mathbf{x}) + \sum_{j=0}^{t} \text{LLB}(\mathbf{A}_j\mathbf{B}_j\mathbf{x})$

### 2. 集成正交损失（Integrated Orthogonal Loss）
- **问题分析**：仅用子空间正交损失（$\|\mathbf{B}_t^T\mathbf{B}_i\|^2$）约束不够——在训练早期正交性条件不满足时，新任务梯度依然会破坏旧子空间的知识
- **正交梯度约束**：利用"线性层梯度更新方向位于输入向量张成的空间内"这一性质，对输入矩阵 $\mathbf{H}_t$ 做 SVD，取前 $r$ 个奇异值对应的 $(\mathbf{V}_t^T)_r$ 作为梯度空间的估计
- **集成损失**：$\mathcal{L}_{\text{ort}} = \frac{1}{t-1}\sum_{i=1}^{t-1}(\lambda_1\sum\|\mathbf{O}_{i,t}\|^2 + \lambda_2\sum\|\mathbf{G}_{i,t}\|^2)$
    - 第一项约束子空间正交性，第二项约束梯度空间正交性，双重保障防遗忘

### 3. 标签引导的局部平衡策略（Label-guided Localized Balancing, LLB）
- 构建响应矩阵 $\mathbf{I} \in \mathbb{R}^{(t+1)\times n}$，衡量每个专家对每个样本的响应强度
- 根据标签构建平衡系数矩阵 $\mathbf{C}$：专家类型与样本标签匹配时系数为 $1-\delta$（鼓励更大响应），不匹配时为 $1+\delta$（抑制响应）
- LLB 损失约束加权响应矩阵的离散度：$\mathcal{L}_{\text{llb}} = \sigma^2(\mathbf{I}\circ\mathbf{C}) / \mu(\mathbf{I}\circ\mathbf{C})$
- 推理时所有 LoRA 专家协同输出联合决策，不进行硬选择

### 总损失
$$\mathcal{L} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{ort}} + \lambda_3\mathcal{L}_{\text{llb}}$$

## 实验关键数据

### 数据集增量协议（FF++ → DFDC-P → DFD → CDF2）

| 方法 | 最终 Avg Acc(%) | 最终 AF(%) |
|------|----------------|-----------|
| LwF | 76.68 | 18.21 |
| DFIL | 85.49 | 7.01 |
| DMP | 89.99 | 4.08 |
| **DevFD** | **89.82** | **4.03** |

### 伪造类型增量协议（Hybrid → FR → FS → EFS）

| 方法 | 最终 Avg AUC(%) | 最终 AF(%) |
|------|-----------------|-----------|
| DFIL | 79.29 | 26.01 |
| HDP | 78.26 | 22.65 |
| SUR-LID | 94.33 | 2.99 |
| **DevFD** | **94.90** | **2.44** |

### 消融实验要点
- 仅用 $\mathcal{L}_{\text{cls}}$：最终 AF=23.93%；加入正交子空间约束后降至 10.06%；再加入正交梯度约束后降至 **7.91%**
- Real-LoRA 的引入同时提升准确率和降低遗忘率；将 Real-LoRA 替换为正交序列并未带来显著提升，验证了单一共享专家建模真脸的合理性
- Task10 长序列实验（DF40 数据集 10 个任务）中，DevFD 保持最高平均 AUC 和最低遗忘率，DFIL 和 HDP 则出现不稳定的灾难性遗忘

## 亮点
- **真假脸分布不对称性的洞察**：用 t-SNE 和正交梯度损失实验双重验证，为"共享 Real-LoRA + 正交 Fake-LoRA"的设计提供了充分理论和实验依据
- **正交梯度集成**：理论分析了仅靠子空间正交约束的不足，提出从梯度空间角度补充约束，有效缓解训练早期的信息干扰
- **严格数据隔离**：不使用任何 replay 机制，仍优于使用 replay 的 DFIL 和 SUR-LID，体现了架构设计本身的优势
- **标签引导的软路由**：通过平衡系数矩阵实现"鼓励匹配、抑制不匹配"，同时保持所有专家的协同推理

## 局限与展望
- 随着任务数增长 LoRA 分支线性增加，在极长任务序列下参数量和推理开销值得关注
- LoRA 秩 $r$ 的选择依赖网格搜索，不同任务可能需要不同秩的自适应分配
- 正交子空间的维度有限（$r \ll \min(d_O, d_I)$），理论上任务数量受限于特征空间维度
- 实验主要针对人脸伪造检测，方法在更一般的持续学习场景（如类增量学习）中的泛化性有待验证
- 标签引导策略在推理时无法使用标签信息，仅靠推理时的响应聚合可能在边界样本上效果受限

## 与相关工作的对比
- **vs O-LoRA / InfLoRA**：同样使用正交 LoRA，但 DevFD 额外引入梯度空间正交约束并区分 Real/Fake LoRA
- **vs DFIL / SUR-LID**：这些方法依赖 replay 机制，DevFD 在严格数据隔离下仍取得更优或相当的性能
- **vs MoEFFD**：MoEFFD 将 MoE+LoRA 用于伪造检测但非持续学习场景，DevFD 专为增量新伪造类型设计
- **vs DMP**：在数据集增量协议中两者性能相近（89.99 vs 89.82），但 DevFD 在伪造类型增量协议中优势更明显

## 启发与关联
- **发展式架构的通用性**：Real-LoRA + 正交 Fake-LoRA 的设计思路可推广到其他"正常样本稳定、异常样本多变"的检测场景（如异常检测、网络入侵检测）
- **梯度空间约束的启发**：将输入 SVD 作为梯度空间的估计是一个有趣的理论洞察，可用于其他持续学习方法中缓解遗忘
- **与 adapter tuning 的结合**：框架天然兼容各种参数高效微调方法，可探索将 LoRA 替换为 Adapter 或 Prefix Tuning 的变体

## 评分
- 新颖性: ⭐⭐⭐⭐ （正交梯度集成和真假脸不对称建模有理论贡献）
- 实验充分度: ⭐⭐⭐⭐ （两种增量协议+长序列+消融+动机验证实验全面）
- 写作质量: ⭐⭐⭐⭐ （理论推导清晰，图表直观）
- 价值: ⭐⭐⭐⭐ （持续学习人脸伪造检测是实际需求，方法有实用价值）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Learning Instance-Adaptive Low-Rank Orthogonal Subspaces for Clothes-Changing Person Re-Identification](../../ICML2026/human_understanding/learning_instance-adaptive_low-rank_orthogonal_subspaces_for_clothes-changing_pe.md)
- [\[CVPR 2026\] PRISM: Learning a Shared Primitive Space for Transferable Skeleton Action Representation](../../CVPR2026/human_understanding/prism_learning_a_shared_primitive_space_for_transferable_skeleton_action_represe.md)
- [\[ICCV 2025\] Bi-Level Optimization for Self-Supervised AI-Generated Face Detection](../../ICCV2025/human_understanding/bi-level_optimization_for_self-supervised_ai-generated_face_detection.md)
- [\[CVPR 2025\] FSFM: A Generalizable Face Security Foundation Model via Self-Supervised Facial Representation Learning](../../CVPR2025/human_understanding/fsfm_a_generalizable_face_security_foundation_model_via_self-supervised_facial_r.md)
- [\[NeurIPS 2025\] Part-Aware Bottom-Up Group Reasoning for Fine-Grained Social Interaction Detection](part-aware_bottom-up_group_reasoning_for_fine-grained_social_interaction_detecti.md)

</div>

<!-- RELATED:END -->
