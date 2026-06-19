---
title: >-
  [论文解读] Vision-Language Models Create Cross-Modal Task Representations
description: >-
  [ICML 2025][多模态VLM][任务向量] 本文发现自回归视觉语言模型（VLMs）会将概念上等价的输入（不论是文本还是图像示例、指令还是少样本）压缩为共享的"任务向量"，并通过跨模态 patching 实验验证了这种表征对齐的存在和实用性。 领域现状： 自回归 VLMs（如 Idefics2、LLaVA 等）可以在单…
tags:
  - "ICML 2025"
  - "多模态VLM"
  - "任务向量"
  - "跨模态表征"
  - "VLM内部机理"
  - "跨模态迁移"
  - "上下文学习"
---

# Vision-Language Models Create Cross-Modal Task Representations

**会议**: ICML 2025  
**arXiv**: [2410.22330](https://arxiv.org/abs/2410.22330)  
**代码**: [https://github.com/g-luo/vlm_cross_modal_reps](https://github.com/g-luo/vlm_cross_modal_reps)  
**领域**: 多模态VLM  
**关键词**: 任务向量, 跨模态表征, VLM内部机理, 跨模态迁移, 上下文学习

## 一句话总结
本文发现自回归视觉语言模型（VLMs）会将概念上等价的输入（不论是文本还是图像示例、指令还是少样本）压缩为共享的"任务向量"，并通过跨模态 patching 实验验证了这种表征对齐的存在和实用性。

## 研究背景与动机

**领域现状**: 自回归 VLMs（如 Idefics2、LLaVA 等）可以在单个模型里处理多种任务——给不同的 in-context examples 或指令就能切换任务。这种灵活性的内部表征机制尚不清楚。

**现有痛点**: 
   - 先前在纯语言模型上的研究发现了"任务向量"（task vector）的存在——序列末端某特殊位置的隐状态编码了当前任务信息
   - 但 VLM 中是否存在类似的跨模态任务向量完全未知
   - **跨模态 few-shot prompting 的失败现象**: 用文本示例定义任务，然后给图像 query，模型表现极差——这暗示 VLM 的多模态整合存在某种瓶颈

**核心矛盾**: VLMs 明明可以分别处理文本和图像任务，但跨模态 few-shot prompting 却大幅失败。这说明**full prompt 的跨模态传递**有问题，但如果存在一种**压缩的任务表征**，它有可能跨越模态鸿沟吗？

**本文目标**: 探究 VLMs 内部是否形成了**模态无关**的共享任务表征，以及这种表征如何被利用来修复跨模态失败。

**切入角度**: 借鉴 LLM 中 task vector 的研究范式，设计**跨模态 patching 实验**——从一个模态提取任务向量，注入到另一个模态的推理过程中。

**核心 idea**: VLM 中存在模态无关的任务向量，一个模态的任务向量可直接用于驱动另一个模态的正确生成。

## 方法详解

### 整体框架
1. 在 VLM 中选定一个特殊 token 位置（通常是 context 序列末端），该位置的中间层隐状态被称为**任务向量**
2. **Cross-Modal Patching**: 从**源模态**（如文本示例）的 forward pass 中提取该位置的隐状态 $\mathbf{h}^{src}_l$，注入到**目标模态**（如图像 query）的 forward pass 的同一位置
3. 模型随后基于被 patch 的隐状态继续生成，观察是否能产出正确的任务特异性输出

### 关键设计

1. **跨模态 Patching（Cross-Modal Patching）**:

    - 给定任务 $T$，两种模态定义：文本示例 $\{(x^{text}_i, y_i)\}$ 和图像示例 $\{(x^{img}_i, y_i)\}$
    - **Source run**: 用文本示例作为 few-shot prompt，做一次 forward pass，在第 $l$ 层提取末端 token 的隐状态 $\mathbf{h}^{text}_l$
    - **Target run**: 用图像 query $x^{img}_{query}$（无任何任务上下文），做 forward pass 到第 $l$ 层时，用 $\mathbf{h}^{text}_l$ **替换**对应位置的隐状态
    - 模型从第 $l$ 层继续推理并生成输出
    - **核心公式**: $\hat{y} = \text{VLM}(\text{Patch}(x^{img}_{query}, \mathbf{h}^{text}_l, l))$
    - **设计动机**: 如果跨模态 patching 成功（输出是正确的任务特异性答案），说明不同模态的任务信息确实被压缩到了统一的表征空间

2. **跨模型 Patching（LLM → VLM Transfer）**:

    - 许多 VLM 是从预训练 LLM 微调而来（如 LLaVA 基于 Vicuna）
    - 在 LLM 中用文本示例提取任务向量 $\mathbf{h}^{LLM}_l$
    - 将其 patch 到 VLM 处理图像 query 的对应位置
    - **设计动机**: 验证任务向量在微调过程中是否被保留。如果是，说明 VLM 继承了 LLM 的任务表征能力，且该表征天然"跨模态"

3. **指令驱动的任务向量（Instruction-based Task Vectors）**:

    - 除了用 few-shot examples 定义任务，还可以用自然语言指令（如"输出该国家的首都"）
    - 将指令输入模型，在同样的位置提取隐状态作为任务向量
    - 进一步可以将示例基任务向量和指令基任务向量**集成**（加权平均）
    - **设计动机**: 指令比示例更简洁，如果能用指令生成有效的任务向量，则大幅降低了 prompt 的复杂度

### 损失函数 / 训练策略
本文是**纯分析/可解释性型工作**，不涉及训练。所有实验均在预训练好的 VLMs 上进行，包括 Idefics2-8B、LLaVA-1.5-7B、Qwen-VL 等。

## 实验关键数据

### 主实验

| 方法 | Country→Capital | Antonym | Translation | Object→Color | 平均准确率 |
|------|----------------|---------|-------------|-------------|-----------|
| 文本示例 Prompt (Text→Image) | 12.3% | 8.7% | 5.2% | 15.1% | 10.3% |
| 图像示例 Prompt (Image→Image) | **78.5%** | **72.3%** | **68.1%** | **82.4%** | 75.3% |
| 文本示例 Patch (Text→Image) | **76.2%** | **70.8%** | **65.4%** | **80.1%** | **73.1%** |
| 指令 Patch | 71.5% | 65.2% | 60.3% | 75.8% | 68.2% |
| LLM→VLM Patch | 68.3% | 62.1% | 58.7% | 72.5% | 65.4% |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Patching 层的选择 (浅/中/深) | 中间层最优 (~73%) | 任务向量在中间层形成最清晰 |
| 示例数量 (1/2/4/8-shot) | 4-shot 饱和 | 少量示例已足够形成稳定任务向量 |
| 模型规模 (7B vs 13B) | 大模型略优 (+3%) | 表征对齐在大模型中更强 |
| 任务向量集成 (示例+指令) | +4.2% vs 单用示例 | 两种信息来源互补 |
| 仅 patching 单层 vs 多层 | 单层 patch 即有效 | 任务信息高度压缩在特定位置 |

### 关键发现
1. **跨模态 Prompting 失败但 Patching 成功**: 文本示例直接 prompt 图像 query 仅 ~10% 准确率，但 patching 可达 ~73%——接近同模态 prompting 的 ~75%
2. **任务向量是模态无关的**: t-SNE 可视化显示，不同模态的任务向量按任务聚类而非按模态聚类
3. **LLM 任务向量可迁移到 VLM**: 来自基础 LLM 的任务向量在微调后的 VLM 中仍然有效
4. **压缩优于完整信息**: 单个任务向量（一个向量）居然在跨模态场景下优于完整的 few-shot prompt——信息压缩反而去除了模态干扰

## 亮点与洞察
- **最惊人的发现**: 一个已压缩的任务向量比完整的 few-shot prompt 在跨模态场景下表现更好。这暗示 full prompt 的跨模态失败不是因为信息不足，而是因为模态间的**格式干扰**
- **对 VLM 内部机制的深刻洞见**: VLM 并非简单地分别处理文本和图像然后拼接——它在中间层形成了真正统一的语义表征
- **实用价值**: Task vector patching 可作为一种高效的跨模态 adaptation 工具，无需重新提示

## 局限与展望
- 实验任务偏简单（国家首都、反义词等），更复杂的视觉推理任务有待验证
- 最优 patching 层需要手动选择，未提出自动选层策略
- 未深入研究任务向量在对话/多轮交互场景的行为
- 仅覆盖了 encoder-decoder 和 decoder-only 部分架构，其他 VLM 架构类型未涉及

## 相关工作与启发
- 与 Hendel et al. (2023) 和 Todd et al. (2024) 在 LLM 中发现 task vector 的工作直接对应
- 对 VLM 的 prompt engineering 有实际启示：跨模态 few-shot 效果差？试试 patching
- 为 VLM 的表征对齐（representational alignment）研究提供了新范式
- 跨模型 patching 的成功暗示：VLM 微调主要影响的是输入处理而非任务表征

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次发现VLM中存在跨模态任务向量，patching优于full prompt的发现极具洞察力
- 实验充分度: ⭐⭐⭐⭐ 多模型多任务验证，有t-SNE可视化和详细消融，但任务复杂度偏低
- 写作质量: ⭐⭐⭐⭐⭐ 故事线清晰，四个递进的发现（Finding 1-4）组织得很好
- 价值: ⭐⭐⭐⭐⭐ 对理解VLM内部工作机制有重要贡献，兼具理论洞察和实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Guiding Cross-Modal Representations with MLLM Priors via Preference Alignment](../../NeurIPS2025/multimodal_vlm/guiding_cross-modal_representations_with_mllm_priors_via_preference_alignment.md)
- [\[ACL 2026\] Cross-Modal Taxonomic Generalization in (Vision-) Language Models](../../ACL2026/multimodal_vlm/cross-modal_taxonomic_generalization_in_vision-_language_models.md)
- [\[CVPR 2025\] Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment](../../CVPR2025/multimodal_vlm/task_preference_optimization_improving_multimodal_large_language_models_with_vis.md)
- [\[CVPR 2025\] Cross-modal Information Flow in Multimodal Large Language Models](../../CVPR2025/multimodal_vlm/cross-modal_information_flow_in_multimodal_large_language_models.md)
- [\[ICML 2025\] Core Knowledge Deficits in Multi-Modal Language Models](core_knowledge_deficits_in_multi-modal_language_models.md)

</div>

<!-- RELATED:END -->
