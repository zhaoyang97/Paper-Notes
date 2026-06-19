---
title: >-
  [论文解读] Enhancing Vision-Language Model Reliability with Uncertainty-Guided Dropout Decoding
description: >-
  [NeurIPS 2025][多模态VLM][VLM幻觉] 提出Dropout Decoding——将视觉token投影到文本空间后量化其认知不确定性，选择性遮掩高不确定性视觉token并通过多组遮掩结果的集成投票增强输出可靠性，无需额外训练即可显著减少LVLM的对象幻觉。 LVLM幻觉的严重性：大型视觉语言模型（LVLM）…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "VLM幻觉"
  - "不确定性量化"
  - "视觉token dropout"
  - "认知不确定性"
  - "集成解码"
---

# Enhancing Vision-Language Model Reliability with Uncertainty-Guided Dropout Decoding

**会议**: NeurIPS 2025  
**arXiv**: [2412.06474](https://arxiv.org/abs/2412.06474)  
**代码**: [https://github.com/kigb/DropoutDecoding](https://github.com/kigb/DropoutDecoding)  
**领域**: 多模态VLM  
**关键词**: VLM幻觉, 不确定性量化, 视觉token dropout, 认知不确定性, 集成解码

## 一句话总结
提出Dropout Decoding——将视觉token投影到文本空间后量化其认知不确定性，选择性遮掩高不确定性视觉token并通过多组遮掩结果的集成投票增强输出可靠性，无需额外训练即可显著减少LVLM的对象幻觉。

## 研究背景与动机

**LVLM幻觉的严重性**：大型视觉语言模型（LVLM）在图像描述、视觉问答等任务上展现了强大能力，但频繁产生幻觉——模型生成与图像内容不一致的错误描述，特别是对象幻觉（Object Hallucination）问题在实际部署中构成严重的可信度障碍。这些幻觉往往源于模型对某些视觉token的误解读——错误地将特定视觉patch的信息映射为不存在的对象或错误的属性。

**现有解决方案的不足**：训练阶段的方法（如针对特定任务的微调、RLHF）需要大量计算资源且难以泛化到新任务；推理阶段的方法（如OPERA修改beam search、VCD使用对比解码）多基于启发式设计，缺乏对"哪些视觉token不可靠"的原理性度量。更根本的问题是，这些方法没有直接回答一个核心问题：在数百到数千个视觉token中，哪些token携带的信息是可靠的，哪些是模型不确定、容易产生误解的？

**Dropout思想的迁移**：传统Dropout在训练时对模型参数施加随机遮掩以防止过拟合，但在预训练的LVLM中直接对参数做Dropout不可行。作者提出将Dropout的核心思想从参数空间迁移到输入token空间——在推理阶段对视觉输入token进行选择性遮掩，通过引入解码上下文的随机性来减少对噪声视觉token的过度依赖。

**核心切入点**：利用LVLM文本解码器的隐含能力——视觉token在解码器顶层的隐表示本身就编码了文本语义信息。通过将视觉token投影到文本词表空间，可以获得每个视觉token的"文本化"解读，进而量化其不确定性。认知不确定性（epistemic uncertainty，反映模型知识不足的部分）特别适合识别那些信息丰富但容易被误解的关键视觉token。

## 方法详解

### 整体框架
Dropout Decoding包含两个阶段：(1) **解码前**——对所有视觉token的不确定性进行量化和分解；(2) **解码时**——基于认知不确定性指导的token dropout + 多组集成 + 多数投票。整个过程在推理时完成，无需修改模型参数或额外训练。

### 关键设计

1. **视觉token的文本空间投影（Textual Interpretation）**:

    - 功能：将每个视觉token映射到文本词表空间，获得其"文本化"概率分布，揭示模型对该视觉patch的语义解读
    - 核心思路：利用LVLM解码器的logit lens方法，对第$i$个视觉token $x_i^v$获取其顶层隐表示$h_i^v = f_\theta(x_{\leq i}^v)$，然后通过文本词表投影矩阵得到文本化分布$q_i^{\text{proj}} = \text{softmax}(W_\mathcal{V} h_i^v)$。信息丰富的patch会投影出具体的词（如"Berlin"、"computer"），而无信息的背景patch投影出高频词（如"a"、"the"）
    - 设计动机：LVLM的解码器顶层隐表示天然接近文本词表投影，即使在视觉token位置上（模型并非被训练在此生成文本），这种投影仍能有效捕获语义信息。这提供了一种免监督的、基于模型自身能力的视觉token信息量评估手段

2. **不确定性分解与认知不确定性度量**:

    - 功能：将每个视觉token的总不确定性分解为偶然不确定性（数据固有）和认知不确定性（模型知识不足），发现认知不确定性是识别关键但易被误解视觉token的最佳指标
    - 核心思路：首先定义所有视觉token的平均文本化分布$q^{\text{proj}} = \frac{1}{N}\sum_{i}^{N} q_i^{\text{proj}}$作为基线。偶然不确定性$U_{\text{ale}}(i) = \mathbb{H}[q_i^{\text{proj}}]$为单个token分布的熵；认知不确定性$U_{\text{epi}}(i) = D_{\text{KL}}(q_i^{\text{proj}} \| q^{\text{proj}})$为单个token分布与全局平均分布的KL散度。总不确定性分解为$U_{\text{total}} = \mathbb{E}_i[U_{\text{ale}}(i) + U_{\text{epi}}(i)]$
    - 设计动机：直觉上，高认知不确定性意味着某个视觉token的文本化解读与整体图像的平均解读差异很大——它携带独特的、"令人惊讶的"信息。这恰恰是容易被模型误解但又至关重要的区域。实验证实，认知不确定性与视觉token的信息量正相关，而偶然不确定性和总不确定性则缺乏这种关联

3. **不确定性引导的Token Dropout + 集成投票**:

    - 功能：基于认知不确定性生成多组dropout掩码，对视觉token施加选择性遮掩，通过集成多组遮掩后的解码结果做多数投票，得到最终输出
    - 核心思路：根据归一化的认知不确定性构建dropout概率分布$P_{\text{dropout}}^{(k)}(x_i^v) = \gamma^{(k)} \frac{U_{\text{epi}}(i) - U_{\text{epi}}^{\min}}{U_{\text{epi}}^{\max} - U_{\text{epi}}^{\min}} + \delta^{(k)}$，其中$\gamma^{(k)}$和$\delta^{(k)}$控制dropout强度。独立采样$K$个二值掩码$M^{(k)}$，用每个掩码遮掩后的视觉上下文分别解码得到候选token $y_j^{(k)}$，最终通过多数投票选出最终输出token。可选地，在每步解码前先做一次初步前向传播产生初始预测$y_j^{\text{init}}$，保留与初始预测相关的视觉token不被dropout
    - 设计动机：单次解码可能因对某些误解读的视觉token过度依赖而产生幻觉；通过多组不同遮掩方案的集成，多样化了模型对视觉内容的视角，减少了单一误解的影响，类似于模型集成（ensemble）的降方差效果

### 损失函数 / 训练策略
无需训练，这是Dropout Decoding最大的实用优势之一。所有操作完全在推理时完成，不修改模型任何参数。主要超参数包括：dropout掩码数量$K$（推荐5-10组，平衡精度与效率）、dropout概率范围控制参数$\gamma^{(k)}$和$\delta^{(k)}$（用于调制不同dropout强度的ensemble成员，使得不同mask具有不同的遮掩比例，增加集成多样性）、以及可选的relevant token保留步骤中的top-$k$阈值。在实际实现中，$K$个mask的前向传播可以通过batch化处理并行执行，不需要顺序推理，从而部分缓解延迟开销。

## 实验关键数据

### 主实验

| 模型 | 方法 | CHAIR_S↓ | CHAIR_I↓ | THRONE $F^1_{\text{all}}$↑ | THRONE $P_{\text{all}}$↑ |
|------|------|---------|---------|-----------|-----------|
| LLaVA-1.5 | Greedy | 42.20 | 12.83 | 0.795 | 0.772 |
| LLaVA-1.5 | Beam Search | 46.33 | 13.90 | 0.790 | 0.759 |
| LLaVA-1.5 | OPERA | 41.47 | 12.37 | 0.802 | 0.782 |
| LLaVA-1.5 | VCD | 49.20 | 14.87 | 0.786 | 0.759 |
| LLaVA-1.5 | **Dropout Decoding** | **39.80** | **11.73** | **0.804** | **0.784** |
| InstructBLIP | Greedy | 27.87 | 7.90 | 0.809 | - |
| InstructBLIP | **Dropout Decoding** | **24.53** | **6.63** | **0.814** | - |
| LLaVA-NEXT | Greedy | 28.80 | 8.10 | 0.815 | - |
| LLaVA-NEXT | **Dropout Decoding** | **26.26** | **7.39** | **0.821** | - |

### 消融实验

| 配置 | CHAIR_S↓ | CHAIR_I↓ | 说明 |
|------|---------|---------|------|
| 偶然不确定性引导 | 43.10 | 13.20 | 效果差，不能有效识别关键token |
| 总不确定性引导 | 41.80 | 12.50 | 略有改善但不稳定 |
| **认知不确定性引导** | **39.80** | **11.73** | 最优，精准定位关键且易误解的token |
| K=1（单次dropout） | 41.30 | 12.40 | 集成不充分 |
| K=5 | 40.10 | 11.90 | 接近最优 |
| **K=10** | **39.80** | **11.73** | 最优集成数量 |
| 有relevant token保留 | 39.80 | 11.73 | CHAIR最优 |
| 无relevant token保留 | 40.20 | 11.85 | CHAIR略差但THRONE可能更好 |

### 关键发现
- VCD在InstructBLIP上反而大幅恶化（CHAIR_S从27.87→39.33），而Dropout Decoding在所有模型上一致有效
- 认知不确定性远优于偶然不确定性和总不确定性作为dropout引导信号
- InstructBLIP仅使用32个视觉token（信息密度高），LLaVA系列使用数百到上千个，方法对不同token数量规模均有效
- 多数投票的集成策略在平局时选择保留token最多的前向传播结果（信息更完整），这一细节对稳定性有贡献

## 亮点与洞察
- **Dropout概念的巧妙迁移**：将Dropout从训练时的参数正则化迁移到推理时的输入token空间——概念极其自然，但在LVLM场景下此前无人尝试。关键的创新在于用不确定性引导而非随机遮掩，使得dropout有了信息论基础
- **认知不确定性的直觉解释**：高认知不确定性的视觉token = 信息丰富但可能被误解的关键patch——这一发现为理解LVLM的视觉感知提供了新视角，也解释了为什么随机dropout效果差而定向dropout效果好
- **免训练的即插即用设计**：整个方法仅依赖LVLM自身的forward pass能力（logit lens + 文本投影），不引入外部模型，兼容任意LVLM架构

## 局限与展望
- 多次前向传播（K次dropout + 可选的初步预测）增加了推理延迟，约5-10倍计算开销，对实时交互场景不友好
- 依赖logit lens投影的质量——如果模型的视觉-文本对齐本身做得不好，投影出的文本化分布可能不准确，导致不确定性度量失效
- 对开放式生成任务（如创意写作、复杂推理）的适用性未验证，目前主要在描述性任务（图像描述、VQA）上测试
- 多数投票策略在生成多样性要求高的场景下可能不合适——集成倾向于输出"共识"答案，可能抑制创造性回复
- 当视觉token数量较少时（如InstructBLIP的32个），dropout可能移除关键信息导致信息损失；当token极多时（如LLaVA-NEXT的2880+个），不确定性计算的开销也相应增大

## 相关工作与启发
- **vs OPERA**: OPERA通过修改beam search的过度关注惩罚来减少幻觉，是操作层面的启发式方法；Dropout Decoding从信息论出发量化token级不确定性，更加原理性
- **vs VCD（Visual Contrastive Decoding）**: VCD通过对比有视觉输入和无视觉输入的输出分布来减少幻觉，但在某些模型上（如InstructBLIP）反而恶化；Dropout Decoding直接作用于视觉token子集，一致有效
- **vs HALC**: HALC使用外部视觉grounding模型定位相关区域，需要额外模型；Dropout Decoding仅靠LVLM自身能力完成不确定性评估
- **vs GAN-DIME/MI估计**: 从信息论角度，Dropout Decoding的认知不确定性度量本质上是在衡量单个视觉token携带的独特互信息

## 评分
- 新颖性: ⭐⭐⭐⭐ 将Dropout从参数空间迁移到输入token空间是巧妙的概念创新，不确定性引导使其有理论基础
- 实验充分度: ⭐⭐⭐⭐ 3个模型、CHAIR+THRONE双benchmark、详细消融，但缺乏效率分析和更多任务类型
- 写作质量: ⭐⭐⭐⭐⭐ Figure 1的投影可视化和不确定性分解极其直观，数学推导与直觉解释并重
- 价值: ⭐⭐⭐⭐ 推理时VLM可靠性增强的实用方法，但推理开销是实际部署的瓶颈

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Controlling Multimodal LLMs via Reward-guided Decoding](../../ICCV2025/multimodal_vlm/controlling_multimodal_llms_via_rewardguided_decoding.md)
- [\[NeurIPS 2025\] AdaLRS: Loss-Guided Adaptive Learning Rate Search for Efficient Foundation Model Pretraining](adalrs_lossguided_adaptive_learning_rate_search_for_efficien.md)
- [\[ICCV 2025\] Enhancing Few-Shot Vision-Language Classification with Large Multimodal Model Features](../../ICCV2025/multimodal_vlm/enhancing_few-shot_vision-language_classification_with_large_multimodal_model_fe.md)
- [\[CVPR 2026\] Uncertainty-guided Compositional Alignment with Part-to-Whole Semantic Representativeness in Hyperbolic Vision-Language Models](../../CVPR2026/multimodal_vlm/uncertainty-guided_compositional_alignment_with_part-to-whole_semantic_represent.md)
- [\[NeurIPS 2025\] Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors](learning_from_videos_for_3d_world_enhancing_mllms_with_3d_vision_geometry_priors.md)

</div>

<!-- RELATED:END -->
