---
title: >-
  [论文解读] MMUnlearner: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models
description: >-
  [ACL 2025 (Findings)][LLM安全][机器遗忘] 本文重新定义了多模态大语言模型（MLLM）时代的机器遗忘任务——仅擦除与特定实体关联的视觉模式而保留文本知识，并提出几何约束梯度上升方法MMUnlearner，通过权重显著性图选择性更新参数，在MLLMU-Bench和CLEAR两大基准上全面超越GA和NPO等基线。
tags:
  - "ACL 2025 (Findings)"
  - "LLM安全"
  - "机器遗忘"
  - "多模态大语言模型"
  - "梯度上升"
  - "权重显著性"
  - "隐私保护"
---

# MMUnlearner: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models

**会议**: ACL 2025 (Findings)  
**arXiv**: [2502.11051](https://arxiv.org/abs/2502.11051)  
**代码**: [https://github.com/Z1zs/MMUnlearner](https://github.com/Z1zs/MMUnlearner)  
**领域**: 多模态VLM  
**关键词**: 机器遗忘、多模态大语言模型、梯度上升、权重显著性、隐私保护

## 一句话总结

本文重新定义了多模态大语言模型（MLLM）时代的机器遗忘任务——仅擦除与特定实体关联的视觉模式而保留文本知识，并提出几何约束梯度上升方法MMUnlearner，通过权重显著性图选择性更新参数，在MLLMU-Bench和CLEAR两大基准上全面超越GA和NPO等基线。

## 研究背景与动机

**领域现状**：机器遗忘（Machine Unlearning, MU）旨在从已训练模型中选择性移除特定数据或知识的影响，以满足隐私法规（如GDPR的"被遗忘权"）。早期MU研究主要聚焦于分类模型，近年来扩展到LLM。对于多模态大语言模型（如LLaVA、InternVL等），MU尚处于起步阶段。

**现有痛点**：现有MU方法直接将单模态遗忘策略（如梯度上升GA、负偏好优化NPO）应用于MLLM时面临两个严重问题：(1) **灾难性遗忘**——在遗忘目标知识的同时，大量非目标知识也被破坏，模型整体能力严重退化；(2) **模态边界模糊**——MLLM的知识同时存在于视觉编码器和语言模型中，粗暴的参数更新无法区分"应该遗忘的视觉模式"和"应该保留的文本知识"。

**核心矛盾**：遗忘的彻底性与保留的完整性之间存在根本冲突。特别是在MLLM中，同一个实体的知识横跨视觉和文本两个模态——要求擦除"看到某人脸的识别能力"但保留"知道这个人是谁"的文本知识，这在参数空间中是高度纠缠的。

**本文目标**：(1) 重新定义MLLM时代的MU任务——仅擦除视觉模式，保留LLM backbone中的文本知识；(2) 设计一种能精确控制遗忘范围的方法。

**切入角度**：利用参数重要性的几何分析——通过计算权重显著性图（weight saliency map），识别哪些参数主要编码目标视觉知识、哪些编码非目标知识或文本知识，然后只更新与视觉模式相关的参数。

**核心 idea**：在梯度上升遗忘过程中，用由"保留概念"和"文本知识"共同约束的权重显著性图来屏蔽不应更新的参数，实现精确的选择性遗忘。

## 方法详解

### 整体框架

MMUnlearner的pipeline分为两个阶段：(1) **显著性图生成**：使用保留数据集和纯文本数据分别计算梯度，构建权重显著性图，标记每个参数对"保留知识"和"文本知识"的重要程度；(2) **约束梯度上升遗忘**：在遗忘阶段对需遗忘的数据执行梯度上升（增大loss以遗忘），但通过显著性图屏蔽掉对保留知识重要的参数，仅更新"安全"参数。输入是MLLM模型、需遗忘的实体数据、保留数据，输出是遗忘后的MLLM。

### 关键设计

1. **双约束权重显著性图 (Dual-Constrained Weight Saliency Map)**:

    - 功能：识别哪些参数可以安全更新用于遗忘，哪些必须保护
    - 核心思路：构建两个显著性图并取交集。第一个"保留概念显著性图"：在保留数据（非目标实体的VQA数据）上计算梯度，高梯度参数标记为"对保留概念重要"，需保护。第二个"文本知识显著性图"：在纯文本数据上计算梯度（不输入图像），高梯度参数标记为"对文本知识重要"，需保护。两个map取并集得到"受保护参数集"，其补集即为可安全更新的参数。最终的遗忘梯度与显著性掩码逐元素相乘：$\Delta w = m \odot \nabla_w L_{forget}$
    - 设计动机：单一约束不够——只用保留概念约束可能损坏与保留概念无关但与文本能力相关的参数（如语言生成能力）；加入文本知识约束确保LLM backbone的原有能力不退化

2. **几何约束梯度上升 (Geometry-Constrained Gradient Ascent)**:

    - 功能：在遗忘方向上更新参数的同时防止偏离保留知识的"安全区域"
    - 核心思路：标准梯度上升直接增大遗忘数据上的loss（$w \leftarrow w + \eta \nabla_w L_{forget}$），容易导致参数在遗忘方向上过度移动。本文引入几何约束——在参数更新后检查保留数据上的loss是否显著增加，如果超过阈值则回退或缩小步长。这类似于在参数空间中画一个"安全球"，只允许在球内更新
    - 设计动机：即使有显著性图屏蔽，低显著性参数也可能在大步长更新后影响保留性能。几何约束提供了额外的安全网

3. **模态感知的遗忘目标**:

    - 功能：只遗忘视觉模式，保留文本知识
    - 核心思路：遗忘loss仅在包含目标实体视觉信息的VQA样本上计算（如含某人照片的问答对），不涉及纯文本提及该实体的样本。评估指标也相应分为：视觉遗忘效果（看到该人照片时是否无法识别）、文本保留效果（被问到该人的纯文本问题时是否仍能正确回答）、整体能力保留（在通用VQA任务上的表现是否退化）
    - 设计动机：MLLM中的实体知识具有模态特异性——视觉识别能力和文本事实知识可以分别编码在不同参数子空间，模态感知遗忘能最大限度保留有用知识

### 损失函数 / 训练策略

遗忘阶段的优化目标为：$L = -L_{forget}(D_{forget}) + \lambda \cdot L_{retain}(D_{retain})$，其中第一项为梯度上升（增大遗忘数据loss），第二项为保留约束（保持保留数据loss不变）。实际参数更新通过显著性掩码过滤：$\Delta w = m \odot \nabla_w L$。训练超参包括学习率 $1 \times 10^{-5}$，batch size 4，训练1个epoch即可。

## 实验关键数据

### 主实验（MLLMU-Bench）

| 方法 | 遗忘效果↑ | 保留概念↑ | 文本知识↑ | 通用VQA↑ | 综合得分 |
|------|----------|----------|----------|----------|---------|
| Vanilla (原模型) | 0% | 100% | 100% | 100% | - |
| GA | 高 | 低 | 低 | 大幅下降 | 差 |
| GA_Diff | 中 | 中 | 中 | 下降 | 中 |
| KL_Min | 中 | 中高 | 中高 | 轻微下降 | 中 |
| NPO | 高 | 低 | 低 | 大幅下降 | 差 |
| **MMUnlearner** | **高** | **高** | **高** | **轻微下降** | **最优** |

### 消融实验

| 配置 | 遗忘效果 | 保留效果 | 说明 |
|------|----------|----------|------|
| Full MMUnlearner | 最优均衡 | 最优 | 双约束+几何约束 |
| w/o 文本知识显著性 | 不变 | 文本知识下降 | 文本约束关键 |
| w/o 保留概念显著性 | 不变 | 保留概念下降 | 概念约束关键 |
| w/o 任何显著性 (= GA) | 高遗忘 | 严重退化 | 无约束的破坏性 |
| 语言mask vs 视觉mask vs both | 视觉mask最优 | - | 视觉参数是遗忘核心 |

### 关键发现

- **GA和NPO在MLLM上的灾难性退化**：直接应用这些基线方法会导致保留概念和通用VQA能力大幅下降（有时降幅超过30%），说明MLLM的参数纠缠远比单模态模型严重
- **双约束缺一不可**：去掉文本知识约束后，虽然视觉遗忘效果不变，但LLM的文本问答能力下降；去掉保留概念约束后，其他视觉概念被误伤
- **视觉参数是遗忘主战场**：使用视觉模块的显著性mask（vision_mask）比语言模块的mask效果更好，印证了"视觉模式主要编码在视觉相关参数中"的假设
- 在CLEAR数据集上也取得一致的优势，验证了方法的泛化性

## 亮点与洞察

- **任务重定义有价值**：将MLLM的机器遗忘从"粗暴擦除"重新定义为"模态选择性擦除"（只删视觉，留文本），更符合实际需求。例如，要求模型忘记某人的脸但保留关于此人的公开文本知识，这比完全删除合理得多
- **显著性图的双约束设计很优雅**：通过保留概念和文本知识的交叉约束锁定"安全参数区域"，思路类似于持续学习中的EWC但更精细。这一框架可迁移到任何需要选择性修改多模态模型的场景
- **仅需1epoch、无额外训练数据**：遗忘效率高，不需要大量"替代数据"来做KD或对抗训练，实用性强

## 局限与展望

- **遗忘效果的可验证性**：当前评估主要基于VQA准确率下降，但模型可能在隐含层面仍保留了目标知识（如激活模式），更严格的遗忘验证手段是开放问题
- **扩展到更多遗忘场景**：当前聚焦于实体级别的视觉遗忘（如删除某人），对于概念级别（如删除"暴力内容"的识别能力）或样本级别的遗忘未充分验证
- **显著性图计算需要保留数据**：在某些场景下，获取与待遗忘数据"类似但不同"的保留数据集可能有困难
- **仅在LLaVA系列模型上验证**：对其他MLLM架构（如Qwen-VL、InternVL等）的适用性未知
- 未来可探索连续遗忘（sequential unlearning）场景——需要依次遗忘多个实体时，显著性图如何高效更新

## 相关工作与启发

- **vs Gradient Ascent (GA)**: GA是最简单的遗忘基线（直接增大遗忘数据loss），但在MLLM上会导致灾难性退化。MMUnlearner通过显著性约束解决了这一问题
- **vs NPO (Negative Preference Optimization)**: NPO将遗忘建模为偏好优化问题，但在MLLM上同样因缺乏模态感知而过度破坏非目标知识。本文的模态感知策略是关键区别
- **vs SRF (Unified Unlearning w/ Remain Geometry)**: 本文借鉴了SRF的几何约束思路，但增加了多模态特有的双约束显著性图设计
- 这项工作揭示了多模态模型中知识纠缠的复杂性，对MLLM的隐私保护和安全对齐有重要启示

## 评分

- 新颖性: ⭐⭐⭐⭐ 任务重定义有见地，双约束显著性图是有效创新
- 实验充分度: ⭐⭐⭐⭐ 两大基准数据集、多基线对比、详细消融和mask分析
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，方法描述完整
- 价值: ⭐⭐⭐⭐ 对MLLM隐私安全有实际推动，代码已开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models](manu_modality_aware_unlearning.md)
- [\[AAAI 2026\] Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models](../../AAAI2026/llm_safety/cross-modal_unlearning_via_influential_neuron_path_editing_i.md)
- [\[NeurIPS 2025\] PULSE: Practical Evaluation Scenarios for Large Multimodal Model Unlearning](../../NeurIPS2025/llm_safety/pulse_practical_evaluation_scenarios_for_large_multimodal_model_unlearning.md)
- [\[ACL 2025\] ReLearn: Unlearning via Learning for Large Language Models](relearn_unlearning_via_learning_for_large_language_models.md)
- [\[ACL 2026\] GAMBIT: A Gamified Jailbreak Framework for Multimodal Large Language Models](../../ACL2026/llm_safety/gambit_a_gamified_jailbreak_framework_for_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
