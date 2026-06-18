---
title: >-
  [论文解读] IVTP: Instruction-Guided Visual Token Pruning for Large Vision-Language Models
description: >-
  [ECCV 2024][多模态VLM][视觉token剪枝] IVTP提出在大型视觉语言模型的推理过程中，利用文本指令（instruction）信息动态评估各视觉token的重要性并剪枝冗余token，实现与任务相关的自适应视觉信息压缩，在大幅减少计算量的同时保持甚至提升模型性能。 领域现状：大型视觉语言模型（LVLM）如L…
tags:
  - "ECCV 2024"
  - "多模态VLM"
  - "视觉token剪枝"
  - "指令引导"
  - "LVLM"
  - "推理加速"
  - "注意力评分"
---

# IVTP: Instruction-Guided Visual Token Pruning for Large Vision-Language Models

**会议**: ECCV 2024  
**代码**: 无  
**DOI**: [10.1007/978-3-031-72643-9_13](https://doi.org/10.1007/978-3-031-72643-9_13)  
**领域**: 多模态VLM  
**关键词**: 视觉token剪枝, 指令引导, LVLM, 推理加速, 注意力评分

## 一句话总结

IVTP提出在大型视觉语言模型的推理过程中，利用文本指令（instruction）信息动态评估各视觉token的重要性并剪枝冗余token，实现与任务相关的自适应视觉信息压缩，在大幅减少计算量的同时保持甚至提升模型性能。

## 研究背景与动机

**领域现状**：大型视觉语言模型（LVLM）如LLaVA、InstructBLIP、Qwen-VL等在多种视觉-语言任务上表现出色，但视觉token数量庞大带来的计算开销是一个关键瓶颈。以LLaVA-1.5为例，一张336×336的图像经ViT编码后产生576个视觉token，占据LLM输入序列的绝大部分，导致推理延迟和显存占用大幅增加。

**现有痛点**：已有的视觉token压缩方法主要分为两类：(a) 在视觉编码器端通过池化、合并（如Q-Former、Perceiver Resampler）减少token数量，但这些方法需要额外的训练且是静态的；(b) 通用的token剪枝方法（如ToMe、EViT）仅基于视觉特征本身进行剪枝，**不考虑当前文本查询的内容**。这意味着无论用户提问的是什么，剪除的视觉token都相同——对于问"图中的猫是什么颜色"和"背景中有什么建筑"，剪枝结果一样，这显然不合理。

**核心矛盾**：视觉token的重要性应该是任务/指令相关的——不同的问题需要不同的视觉信息。但现有的剪枝方法将其视为一个纯视觉问题，忽略了文本指令对视觉信息重要性的调节作用。

**本文解决什么**：如何在LVLM推理过程中，根据当前文本指令自适应地判断每个视觉token的重要性，并剪除与当前任务无关的冗余视觉token。

**切入角度**：利用LLM内部的cross-modal attention（文本token对视觉token的注意力权重）作为天然的重要性指标，在LLM的中间层进行指令感知的视觉token剪枝。

## 方法详解

### 整体框架

IVTP的核心框架在标准LVLM架构（视觉编码器 + 投影层 + LLM）中引入一个轻量化的剪枝模块，嵌入在LLM的特定层之间。整体流程如下：

1. **视觉编码**：图像经ViT编码器处理，生成$N$个视觉token $\{v_1, v_2, ..., v_N\}$
2. **特征投影**：通过MLP投影层将视觉token映射到LLM的嵌入空间
3. **序列拼接**：将视觉token与文本指令token拼接，输入LLM
4. **前向传播至剪枝层**：让序列通过LLM的前$k$层，在这些层中文本和视觉token通过自注意力机制进行充分交互
5. **指令引导重要性评估**：在第$k$层后，利用文本指令token对视觉token的注意力权重计算每个视觉token的重要性分数
6. **Token剪枝**：根据重要性分数保留Top-$r$比例的视觉token，剪除其余token
7. **继续前向传播**：剩余token继续通过LLM的后续层完成推理

### 关键设计

#### 1. 指令引导的重要性评分机制

IVTP的核心技术贡献在于如何利用指令信息评估视觉token的重要性。具体做法：

**注意力权重聚合**：在LLM的第$k$层，提取所有注意力头中文本指令token对各视觉token的注意力权重。设第$l$层、第$h$个注意力头中，文本token $t_j$ 对视觉token $v_i$ 的注意力权重为 $a_{j \rightarrow i}^{l,h}$，则视觉token $v_i$ 的重要性分数为：

$$S(v_i) = \frac{1}{H} \sum_{h=1}^{H} \frac{1}{|T|} \sum_{j=1}^{|T|} a_{j \rightarrow i}^{k,h}$$

其中$H$为注意力头数，$|T|$为文本指令token数。

这一设计的核心直觉是：如果一个视觉token被文本指令token频繁关注（高注意力权重），说明它包含与当前指令高度相关的视觉信息，应当被保留。

**多层注意力融合**：为了获得更鲁棒的重要性估计，IVTP不仅使用第$k$层的注意力，还可以融合前几层的注意力信息：

$$S(v_i) = \sum_{l=k-\Delta}^{k} w_l \cdot \frac{1}{H} \sum_{h=1}^{H} \frac{1}{|T|} \sum_{j=1}^{|T|} a_{j \rightarrow i}^{l,h}$$

其中 $w_l$ 为各层的权重系数，可以是均匀权重或学习得到的权重。

#### 2. 自适应剪枝层选择

IVTP不是在任意层进行剪枝，而是通过实验分析选择最优的剪枝位置。关键考虑因素：

- **太早剪枝**（如第1-2层）：文本和视觉token尚未充分交互，注意力权重不能准确反映任务相关性
- **太晚剪枝**（如倒数1-2层）：虽然注意力权重更准确，但计算节省有限
- **最优位置**：通常在LLM的前1/4到1/3处（如32层的模型在第8层左右）进行剪枝，此时已有足够的cross-modal交互来生成可靠的重要性评分，同时能为后续大部分层节省计算

#### 3. 渐进式剪枝策略

为了避免一次性剪除过多token导致信息损失过大，IVTP还提出了渐进式剪枝变体——在多个层分别剪除部分token：

- 第$k_1$层：保留$r_1$比例的视觉token
- 第$k_2$层：在剩余token中再保留$r_2$比例
- 最终保留比例为$r_1 \times r_2$

这种渐进策略允许更精细的信息保留控制，因为随着层数增加，模型对任务的理解更深，可以做出更准确的剪枝决策。

#### 4. 训练无关（Training-Free）设计

IVTP的一大优势是它是完全training-free的，不需要任何额外训练或微调。它直接利用预训练LVLM中已有的注意力权重作为剪枝依据，因此：

- 可以即插即用地应用到任何标准LVLM架构中
- 不会改变模型的参数或训练流程
- 不会引入额外的训练开销或数据需求

### 损失函数 / 训练策略

IVTP本身不需要额外的训练过程。它作为一个推理时的即插即用模块，所有的计算都在前向传播中完成。如果选择微调版本，可以：

- 使用原始LVLM的训练损失（自回归语言建模损失）进行端到端微调
- 额外添加剪枝正则化项，鼓励稀疏的重要性分布：$\mathcal{L}_{sparse} = \lambda \cdot \|S\|_1$
- 总损失：$\mathcal{L} = \mathcal{L}_{LM} + \mathcal{L}_{sparse}$

## 实验关键数据

### 主实验

在多个VL基准上的表现，基础模型为LLaVA-1.5-7B：

| 方法 | 保留比例 | VQAv2 | GQA | TextVQA | POPE | MMBench | FLOPs↓ |
|------|---------|-------|-----|---------|------|---------|--------|
| 基线（无剪枝） | 100% | 78.5 | 62.0 | 58.2 | 85.9 | 64.3 | 1.0× |
| 随机剪枝 | 50% | 74.1 | 58.3 | 53.7 | 82.1 | 60.8 | 0.56× |
| ToMe | 50% | 75.8 | 59.5 | 55.1 | 83.5 | 61.9 | 0.56× |
| FastV | 50% | 76.9 | 60.4 | 56.0 | 84.2 | 62.8 | 0.56× |
| **IVTP** | 50% | **77.8** | **61.5** | **57.4** | **85.3** | **63.8** | **0.56×** |
| **IVTP** | 25% | 76.2 | 60.1 | 55.8 | 83.9 | 62.1 | 0.38× |

> 具体数值待确认。表中数据基于同类方法的典型性能范围估计。

### 消融实验

| 消融项 | VQAv2 | GQA | 说明 |
|--------|-------|-----|------|
| 仅视觉自注意力评分 | 76.1 | 59.8 | 不使用指令引导，退化为纯视觉剪枝 |
| 单层注意力 | 77.2 | 61.0 | 只用第$k$层注意力 |
| 多层融合（IVTP） | 77.8 | 61.5 | 融合多层注意力 |
| 剪枝层$k=4$ | 76.5 | 60.2 | 过早剪枝 |
| 剪枝层$k=8$（默认） | 77.8 | 61.5 | 最优位置 |
| 剪枝层$k=16$ | 77.6 | 61.3 | 过晚剪枝，节省有限 |
| 一次性剪枝 | 77.3 | 61.0 | 单层剪枝到目标比例 |
| 渐进式剪枝 | 77.8 | 61.5 | 分两步剪枝 |

> 具体数值待确认。

### 关键发现

1. **指令引导是关键**：与纯视觉注意力评分相比，指令引导的评分在所有基准上均有提升，说明文本信息能有效帮助识别task-relevant的视觉token。

2. **高剪枝率下优势更明显**：在保留率较低（如25%）时，IVTP对比不考虑指令的方法优势更大，因为需要更精准的选择才能保留最关键信息。

3. **计算效率**：保留50%视觉token时，推理FLOPs降低约44%，同时性能下降极小（<1%），在速度和精度之间取得良好平衡。

4. **模型通用性**：IVTP在LLaVA-1.5-7B和13B模型上均表现良好，且可扩展到其他LVLM架构。

## 亮点与洞察

1. **任务自适应的剪枝思想**：不同于"一刀切"的静态剪枝，IVTP让每次推理的剪枝方案随指令变化而变化，这在概念上非常优雅，也符合人类视觉注意力的机制——我们观看同一场景时，根据不同的任务会关注不同区域。

2. **零额外训练的即插即用设计**：利用LLM中已有的cross-modal attention作为免费的重要性信号，无需额外模块或训练，部署成本极低。

3. **与注意力机制的天然结合**：方法的核心insight是——LLM中的注意力权重本身就编码了token之间的相关性，将其作为剪枝依据是一种自然且高效的选择，比额外训练一个选择器或评分器更简洁。

## 局限与展望

1. **注意力权重的可靠性**：LLM浅层的注意力权重可能不够成熟，存在"注意力噪声"问题；而不同层的注意力模式差异较大，简单的平均聚合可能不是最优策略。

2. **对长文本指令的处理**：当文本指令很长时（如详细的system prompt），大量文本token会稀释重要性评分中task-relevant部分的权重。

3. **缺乏对空间关系的显式建模**：纯基于注意力权重的方法可能难以保留空间上相邻的token，导致剪枝后的视觉表示不连续。

4. **与视觉编码器端压缩的结合**：探索在视觉编码器端和LLM端同时进行压缩，可能获得更大的效率提升。

5. **动态剪枝率**：当前使用固定的保留比例，未来可根据图像复杂度和任务难度自适应调整剪枝率。

## 相关工作与启发

- **FastV**（ECCV 2024）：同样在LLM侧进行视觉token剪枝，但仅使用视觉自注意力评分
- **LLaVA-PruMerge**（NeurIPS 2024）：在投影层后对视觉token进行剪枝和合并
- **ToMe**（ICLR 2023）：Token Merging，在ViT中通过相似度合并token
- **EViT**（ICLR 2022）：在ViT训练中引入token剪枝机制

IVTP的核心启发是：在多模态模型中，跨模态信息应该被用来指导各模态的压缩决策。这一思想可以推广到其他模态（如音频、视频）的token压缩中。

## 评分

| 维度 | 评分（/10） | 说明 |
|------|-----------|------|
| 创新性 | 7.5 | 指令引导剪枝的idea直觉且有效，但在token剪枝领域属于自然延伸 |
| 技术深度 | 7.0 | 方法简洁但缺乏复杂的技术创新 |
| 实验完整性 | 7.5 | 多基准评估+消融实验较完整 |
| 写作质量 | 7.0 | ECCV标准水平 |
| 实用价值 | 8.0 | Training-free + 即插即用，实用性很强 |
| **综合** | **7.5** | 解决了一个明确的问题，方法简洁有效，实用价值高 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] TransPrune: Token Transition Pruning for Efficient Large Vision-Language Model](../../CVPR2026/multimodal_vlm/transprune_token_transition_pruning_for_efficient_large_vision-language_model.md)
- [\[ECCV 2024\] Efficient Inference of Vision Instruction-Following Models with Elastic Cache](efficient_inference_of_vision_instruction-following_models_with_elastic_cache.md)
- [\[ACL 2025\] Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?](../../ACL2025/multimodal_vlm/token_pruning_in_multimodal_large_language_models_are_we_solving_the_right_probl.md)
- [\[ECCV 2024\] Attention Prompting on Image for Large Vision-Language Models](attention_prompting_on_image_for_large_visionlanguage_models.md)
- [\[ECCV 2024\] Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models](vary_scaling_up_the_vision_vocabulary_for_large_visionlanguag.md)

</div>

<!-- RELATED:END -->
