---
title: >-
  [论文解读] Instruction-Grounded Visual Projectors for Continual Learning of Generative Vision-Language Models
description: >-
  [ICCV 2025][多模态VLM][持续学习] 提出 MVP（Mixture of Visual Projectors），一种基于指令上下文的视觉投影器混合专家框架，通过专家推荐策略和专家剪枝机制，使生成式 VLM 在持续学习新视觉-语言任务时避免灾难性遗忘，同时保持对不同指令类型的响应能力，在分类/描述/问答等任务上全面超越现有方法。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "持续学习"
  - "视觉-语言模型"
  - "Mixture-of-Experts"
  - "视觉投影器"
  - "指令感知"
---

# Instruction-Grounded Visual Projectors for Continual Learning of Generative Vision-Language Models

**会议**: ICCV 2025  
**arXiv**: [2508.00260](https://arxiv.org/abs/2508.00260)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 持续学习, 视觉-语言模型, Mixture-of-Experts, 视觉投影器, 指令感知

## 一句话总结

提出 MVP（Mixture of Visual Projectors），一种基于指令上下文的视觉投影器混合专家框架，通过专家推荐策略和专家剪枝机制，使生成式 VLM 在持续学习新视觉-语言任务时避免灾难性遗忘，同时保持对不同指令类型的响应能力，在分类/描述/问答等任务上全面超越现有方法。

## 研究背景与动机

生成式 VLM（如 InstructBLIP）在适应新任务时面临两个核心挑战：

**灾难性遗忘**：当模型在新任务上训练时，先前学到的知识会被覆盖。重新在所有数据上训练代价巨大，且预训练数据可能不可获取

**指令忽视问题（关键洞察）**：现有持续学习方法（如 EProj、GMM）通过更新共享的视觉投影器来学习新任务。但当多个任务共享相似的指令模板时，投影器会过度适配这些相似指令，导致模型在面对不同类型的指令时"忽视"文本指令、仅依赖视觉输入生成回复。例如，在问答任务上训练后，给模型一张图片和分类指令，模型仍然会生成问答式的回复

作者发现问题的根源在于：**共享视觉投影器无法根据不同指令上下文调整视觉信息的翻译方式**。一个投影器只能将视觉特征翻译成一种"语言"给 LLM，而不同任务需要不同的翻译方式。

## 方法详解

### 整体框架

MVP 由三个核心组件构成：
1. 混合视觉投影器（MoE）：多个投影器专家 + 路由器
2. 专家推荐策略：基于任务语义相似度推荐复用哪些专家
3. 专家剪枝机制：移除冗余激活的专家以防止负迁移

在推理时，通过自适应知识聚合（AKA）平衡 MoE 输出和预训练投影器输出。

### 关键设计

1. **混合视觉投影器（Mixture of Visual Projectors）**

   引入 $N_E$ 个投影器专家 $\{\mathcal{E}_j\}$ 和一个路由器 $\mathcal{R}$，路由器根据图像特征和指令嵌入共同决定激活哪些专家：

    $W = \text{Softmax}(\text{Top-}K(\mathcal{R}(\mathbf{x}_{i,\text{img}}^t, \mathbf{x}_{i,\text{text}}^t)))$

   聚合输出与预训练投影器的输出做平均：

    $\tilde{\mathbf{x}}_{i,\text{img}}^t = \frac{1}{K+1}\left(\mathcal{V}(\mathbf{x}_{i,\text{img}}^t) + \sum_{j=1}^{N_E} w_j \mathcal{E}_j(\mathbf{x}_{i,\text{img}}^t)\right)$

   设计动机：通过将路由条件化在指令嵌入上，不同类型的指令（分类/描述/问答）会激活不同的专家组合，从而实现指令感知的视觉翻译。保留预训练投影器的贡献确保零样本能力不丢失。

2. **专家推荐策略（Expert Recommendation）**

   计算新任务与所有先前任务之间的语义相似度，基于视觉和文本两个维度：

    $s^{t'} = \alpha \cdot \sigma(s_{\text{img}}^{t'}) + (1-\alpha) \cdot \sigma(s_{\text{text}}^{t'})$

   使用对比分布损失鼓励路由器复用与相似旧任务关联的专家，同时通过激活偏差减少（Activation Bias Reduction）抑制高频使用的专家：

    $\mathcal{L}_{bias} = \frac{1}{2}\left(1 + \frac{\langle \bar{L}^{1:t-1}, L^t \rangle}{\|\bar{L}^{1:t-1}\|_2 \cdot \|L^t\|_2}\right)$

   设计动机：一方面利用旧任务的相关知识加速新任务学习，另一方面避免所有任务只使用少数几个专家，确保学习容量。

3. **专家剪枝（Expert Pruning）**

   训练完每个任务后，学习一个稀疏向量 $E^t$，最小化剪枝前后输出的差异同时约束激活的专家数量：

    $\min_{E^t} \left\|\sum_{j=1}^{N_E}(w_j - e_j^t)\mathcal{E}_j(x_{i,\text{img}}^t)\right\|_F + \|\mathcal{M}^{1:t-1} + E^t\|_1$

   将 $E^t$ 阈值化为二值掩码 $\mathcal{M}^t$，然后用合成数据微调路由器以适应剪枝后的专家配置。

   设计动机：持续学习过程中，某些专家可能被冗余地累积激活。剪枝后重新初始化这些专家为预训练权重，为后续任务保留学习容量，同时减少负迁移风险。

### 损失函数 / 训练策略

总损失函数：
$$\mathcal{L} = \mathcal{L}_{ce} + \lambda_{rec}\mathcal{L}_{rec} + \lambda_{bias}\mathcal{L}_{bias}$$

实现细节：
- LLM：Vicuna-7B 或 LLaMa-2-7B
- 视觉编码器：ViT-g/14 + Q-Former（同 InstructBLIP）
- 专家数 $N_E=20$，每次激活 $K=2$ 个
- 语义评分权重 $\alpha=0.3$，损失权重 $\lambda_{rec}=\lambda_{bias}=1$
- 数据集：ImageNet-R（10子集×20类，分类）+ Flickr-30K（4子集，描述）+ COCO-QA（4子集，问答），共 18 个顺序任务
- 优化器：Adam（β1=0.9, β2=0.999），NVIDIA RTX 3090 GPU

## 实验关键数据

### 主实验

**18 个顺序任务学习后的 Last 指标（Vicuna 版本）**：

| 方法 | 分类 T1-T3 | 分类 T7-T10 | 描述（动物） | 描述（车辆） | QA（对象） | QA（颜色） |
|------|-----------|-----------|------------|------------|-----------|-----------|
| Zero-Shot | 67.79 | 62.87 | 77.08 | 73.43 | 68.52 | 62.62 |
| LwF | 4.38 | 5.83 | 70.16 | 68.60 | 70.69 | 76.46 |
| EWC | 5.61 | 5.83 | 67.00 | 66.47 | 65.72 | 73.69 |
| GMM | 1.33 | 1.62 | 62.96 | 64.70 | 42.29 | 45.11 |
| MoEAdapter | 70.01 | 65.55 | 76.53 | 73.92 | 66.76 | 41.53 |
| **MVP** | **82.81** | **87.52** | **79.60** | **77.79** | **79.26** | **80.30** |

MVP 在所有任务上均优于零样本基线，而竞争方法（LwF、EWC、GMM）在分类任务上几乎完全遗忘。

**LLaMa-2 版本 Last 平均性能差异（vs Zero-Shot）**：

| 方法 | 分类 Δ | 描述 Δ | QA Δ |
|------|--------|--------|------|
| LwF | +10.92 | +1.22 | -5.52 |
| MoEAdapter | +2.09 | -0.03 | -1.05 |
| **MVP** | **+20.78** | **+2.82** | **+7.15** |

### 消融实验

| 配置 | 分类 Avg | 描述 Avg | QA Avg | 说明 |
|------|---------|---------|--------|------|
| MVP 完整方法 | 85.87 (+21.46) | 77.75 (+2.75) | 76.34 (+24.72) | 最佳 |
| w/o AKA | 85.53 | 76.70 | 76.17 | 自适应聚合贡献稳定 |
| w/o AKA, Prune | 85.92 | 71.55 (-3.45) | 75.54 | 剪枝防止描述遗忘 |
| w/o AKA, Prune, $\mathcal{L}_{bias}$ | 3.31 (-61.10) | 70.70 | 73.99 | 偏差减少对分类至关重要 |
| w/o 所有（共享投影器） | 1.49 (-62.92) | 63.10 (-11.90) | 60.15 (+8.53) | 严重遗忘 |

$\mathcal{L}_{bias}$ 是最关键的组件——移除后分类从 85.92% 暴跌至 3.31%，因为没有偏差减少，路由器将所有样本集中到少数专家上。

### 关键发现

1. **指令忽视的严重性**：LwF、EWC、GMM 在学完问答任务后，分类准确率降至 ~5%，因为共享投影器被最后一个任务的指令模式覆盖
2. **跨任务知识迁移**：MVP 在学完分类后，描述任务性能略有提升，说明指令感知的翻译机制能利用分类知识辅助描述
3. **MoEAdapter 的局限**：虽然也使用 MoE，但它冻结旧专家、无推荐机制，导致在描述和 QA 任务上表现不如 MVP
4. **两种 VLM 上一致的优势**：在 Vicuna 和 LLaMa-2 两种不同架构的 VLM 上，MVP 都展现出显著优势

## 亮点与洞察

- **问题定义精准**：发现"指令忽视"这一被忽略的问题，并追溯到共享投影器的根本限制
- **MoE 在持续学习中的新应用**：不同于传统的冻结旧专家/扩展新专家策略，MVP 通过推荐和剪枝动态管理专家生命周期
- **自适应知识聚合的优雅性**：推理时根据输入与已学任务的相似度来调节 MoE 分支的贡献权重，对未见任务自动退化为零样本模式
- **专家剪枝的必要性**：消融实验清楚地展示了剪枝防止负迁移的效果——没有剪枝时描述任务下降 6.2%

## 局限与展望

- 实验使用的任务序列相对简单（分类→描述→问答），更复杂的任务混合/交错学习场景未验证
- 20 个专家的设置对于实际部署来说参数开销不小（虽然每次只激活 2 个）
- 专家推荐依赖存储各任务的平均嵌入，随着任务数量增长，存储和计算开销线性增加
- 仅在 7B 规模的 LLM 上测试，是否适用于更大或更小的模型未知
- 合成数据用于路由器微调可能引入分布偏差
- 任务边界假设（知道何时切换到新任务）在实际场景中可能不成立

## 相关工作与启发

本文桥接了 MoE 和 VLM 持续学习两个研究方向。与 MoEAdapter 相比，核心创新在于指令条件化的路由和基于语义相似度的专家推荐。与传统持续学习（LwF、EWC）相比，优势在于不需要知识蒸馏或参数重要性估计。专家剪枝机制可能对其他 MoE-based 持续学习场景也有借鉴意义。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 指令条件化的 MoE 视觉投影器是新颖的设计，推荐+剪枝机制有技术贡献
- **实验充分度**: ⭐⭐⭐⭐ 两种 VLM、消融研究、时间步性能曲线、定性分析，但任务设置偏简单
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，图示表达直观，但符号较多需要仔细阅读
- **价值**: ⭐⭐⭐⭐ 对 VLM 持续学习是有意义的推进，但实际部署场景的验证不足

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](smolora_exploring_and_defying_dual_catastrophic_forgetting_in_continual_visual_i.md)
- [\[CVPR 2025\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](../../CVPR2025/multimodal_vlm/continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)
- [\[ICLR 2026\] Enhanced Continual Learning of Vision-Language Models with Model Fusion](../../ICLR2026/multimodal_vlm/enhanced_continual_learning_of_vision-language_models_with_model_fusion.md)
- [\[ICCV 2025\] Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models](multi-cache_enhanced_prototype_learning_for_test-time_generalization_of_vision-l.md)
- [\[NeurIPS 2025\] Learning to Instruct for Visual Instruction Tuning](../../NeurIPS2025/multimodal_vlm/learning_to_instruct_for_visual_instruction_tuning.md)

</div>

<!-- RELATED:END -->
