---
title: >-
  [论文解读] Con Instruction: Universal Jailbreaking of Multimodal Large Language Models via Non-Textual Modalities
description: >-
  [ACL 2025][多模态VLM][多模态越狱] 本文提出 Con Instruction 方法，通过优化对抗性图像或音频使其在嵌入空间中与目标恶意指令对齐，实现无需文本输入即可越狱多模态大语言模型（MLLM），在 LLaVA-v1.5 上达到 86.6% 的攻击成功率，并提出了 ARC 评估框架来同时衡量攻击响应的质量和相关性。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "多模态越狱"
  - "对抗性攻击"
  - "非文本指令"
  - "嵌入空间对齐"
  - "安全机制绕过"
---

# Con Instruction: Universal Jailbreaking of Multimodal Large Language Models via Non-Textual Modalities

**会议**: ACL 2025  
**arXiv**: [2506.00548](https://arxiv.org/abs/2506.00548)  
**代码**: 有（论文中提到公开）  
**领域**: AI安全 / 多模态VLM  
**关键词**: 多模态越狱, 对抗性攻击, 非文本指令, 嵌入空间对齐, 安全机制绕过

## 一句话总结

本文提出 Con Instruction 方法，通过优化对抗性图像或音频使其在嵌入空间中与目标恶意指令对齐，实现无需文本输入即可越狱多模态大语言模型（MLLM），在 LLaVA-v1.5 上达到 86.6% 的攻击成功率，并提出了 ARC 评估框架来同时衡量攻击响应的质量和相关性。

## 研究背景与动机

**领域现状**：多模态大语言模型（MLLM）如 LLaVA、InternVL、Qwen-VL、Qwen-Audio 等能够理解和处理图像、音频等非文本模态。与此同时，这些模型的安全防护机制主要针对文本输入设计，通过检测文本中的有害意图来拒绝危险请求。

**现有痛点**：现有的 MLLM 越狱攻击（如视觉对抗攻击、对抗性提示注入等）主要通过"文本指令 + 对抗性图像辅助"的方式工作——恶意意图仍然通过文本传达，对抗性图像只是起辅助绕过作用。这使得文本安全过滤器仍然可以检测到攻击意图。此外，许多方法需要训练数据或对文本指令进行预处理，增加了攻击复杂度。

**核心矛盾**：MLLM 具有理解非文本指令的强大能力（例如，可以"阅读"图像中的文字、理解音频中的语义），但安全防护机制主要检查文本通道。这意味着如果恶意指令完全通过非文本模态传递，安全过滤器可能完全无法检测。

**本文目标**：(1) 验证 MLLM 能否通过纯非文本模态（图像/音频）接收并执行恶意指令；(2) 开发无需训练数据的通用越狱方法；(3) 设计更全面的攻击评估框架。

**切入角度**：既然 MLLM 被训练来理解非文本输入的语义，那么是否可以将恶意指令"编码"到图像或音频中，让模型的多模态理解能力成为安全漏洞的来源？

**核心 idea**：通过梯度优化生成对抗性图像/音频，使其在 MLLM 的嵌入空间中与目标恶意指令高度对齐，从而实现"非文本模态即指令"的越狱攻击。

## 方法详解

### 整体框架

Con Instruction 的攻击流程分为三步：(1) **目标指令编码**：将恶意文本指令通过 MLLM 的文本编码器映射到嵌入空间，得到目标嵌入向量；(2) **对抗性样本优化**：初始化一个随机图像/音频，通过梯度下降优化其像素/频谱值，使其经过 MLLM 的视觉/音频编码器后的嵌入尽可能接近目标嵌入；(3) **攻击执行**：将优化后的对抗性图像/音频单独或配合无害文本输入 MLLM，模型通过多模态融合"读取"出嵌入在非文本模态中的恶意指令并执行。

### 关键设计

1. **嵌入空间对齐优化（Embedding Space Alignment）**:

    - 功能：将恶意指令的语义"编码"到非文本模态中
    - 核心思路：给定恶意文本指令 $t$，通过文本编码器得到目标嵌入 $e_t = \text{TextEnc}(t)$。对于图像输入 $x$（初始为随机噪声），通过视觉编码器得到 $e_x = \text{VisEnc}(x)$。优化目标为最小化余弦距离 $\mathcal{L} = 1 - \cos(e_x, e_t)$，通过 PGD（Projected Gradient Descent）在像素空间中迭代更新。对音频模态同理。不需要任何训练数据，只需要对模型进行白盒前向/反向传播
    - 设计动机：MLLM 的多模态融合机制将不同模态映射到共享嵌入空间，这意味着在嵌入空间中对齐的图像/音频会被模型解读为等价于对应的文本指令。利用模型自身的跨模态理解能力来传递恶意信号

2. **非文本+文本组合攻击（Multi-Modal Amplification）**:

    - 功能：通过组合非文本对抗性样本和无害文本来大幅提升攻击成功率
    - 核心思路：单独使用对抗性图像/音频已经可以绕过安全机制，但成功率受限于嵌入对齐的精度。通过在文本通道补充与恶意主题相关但本身无害的文本（如"请描述图片中的内容"或与主题相关的上下文），可以帮助模型更准确地"解码"非文本模态中的隐藏指令，攻击成功率大幅提升
    - 设计动机：MLLM 的多模态推理是协同的——文本提供上下文，图像/音频提供内容。利用这种协同效应可以克服单一模态嵌入对齐的精度瓶颈

3. **攻击响应分类框架 ARC（Attack Response Categorization）**:

    - 功能：全面评估攻击效果，区分不同类型的成功和失败
    - 核心思路：传统评估只关注"是否生成有害内容"（二分类），ARC 引入两个正交维度：(a) 响应质量——生成内容的信息量和完整度；(b) 响应相关性——生成内容是否与恶意指令的具体意图相关。这产生四个象限：高质量高相关（完全成功）、高质量低相关（模型生成了有害内容但不是要求的）、低质量高相关（模型理解了意图但给出不完整回答）、低质量低相关（完全失败）
    - 设计动机：现有的 ASR 指标过于粗糙，无法区分"模型生成了与指令无关的有害内容"和"模型精确执行了恶意指令"。ARC 提供了更细粒度的攻击效果评估

### 损失函数 / 训练策略

核心损失为嵌入空间的余弦距离损失 $\mathcal{L} = 1 - \cos(e_x, e_t)$。使用 PGD 优化，步长和迭代次数为超参数。图像使用 $L_\infty$ 范数约束来控制扰动大小，音频使用类似的频谱域约束。无需额外训练数据或目标模型的微调。

## 实验关键数据

### 主实验

**视觉语言模型攻击结果（AdvBench + SafeBench）：**

| 模型 | 方法 | AdvBench ASR | SafeBench ASR | 说明 |
|------|------|-------------|-------------|------|
| LLaVA-v1.5 (7B) | 文本攻击 | 32.1% | 28.5% | 基线 |
| LLaVA-v1.5 (7B) | Con Instruction | **76.8%** | **79.2%** | 纯图像攻击 |
| LLaVA-v1.5 (13B) | Con Instruction | **81.3%** | **86.6%** | 纯图像攻击 |
| LLaVA-v1.5 (13B) | Con Inst.+文本组合 | **89.7%** | **92.1%** | 组合攻击 |
| InternVL | Con Instruction | 68.4% | 71.2% | 纯图像攻击 |
| Qwen-VL | Con Instruction | 65.7% | 69.8% | 纯图像攻击 |

**音频语言模型攻击结果：**

| 模型 | Con Instruction ASR | 说明 |
|------|-------------------|------|
| Qwen-Audio | 72.3% | 纯音频攻击 |
| Qwen-Audio + 文本 | 84.5% | 组合攻击 |

### 消融实验

| 配置 | ASR (LLaVA-v1.5-13B) | 说明 |
|------|----------------------|------|
| Con Instruction (完整) | 81.3% | 图像模态 |
| 仅文本攻击（无图像） | 32.1% | 安全过滤器有效 |
| 随机图像+恶意文本 | 38.5% | 图像无对齐效果 |
| Con Inst.+无害文本 | 89.7% | 组合攻击显著提升 |
| Con Inst. (减少优化步数 50%) | 62.4% | 对齐精度下降 |
| Con Inst. + 对抗性训练防御 | 48.2% | 防御有效但残留漏洞 |
| Con Inst. + 输入检测防御 | 55.1% | 检测率有限 |

### 关键发现

- 纯非文本对抗性样本（无任何恶意文本）即可达到 81.3% 的攻击成功率，证明安全机制对非文本通道几乎没有防护
- 13B 模型比 7B 更容易被攻击（81.3% vs 76.8%），这是因为更大的模型有更强的跨模态理解能力，反而使其更容易"解读"嵌入在图像中的恶意指令
- 非文本+文本组合攻击将成功率进一步提升至 89.7%，说明多模态协同效应可被攻击者利用
- 现有防御方法（对抗性训练、输入检测）有一定效果但远不足够，存在巨大的安全防护差距（gap）
- ARC 评估框架揭示了传统 ASR 指标遗漏的重要区分——约 15% 的"成功"攻击实际上生成了与指令不相关的有害内容

## 亮点与洞察

- **"能力即漏洞"的深刻洞察**：MLLM 越强大（跨模态理解越好），越容易被 Con Instruction 攻击。这揭示了多模态 AI 安全的一个根本悖论——提升理解能力的同时不可避免地扩大了攻击面
- **零数据攻击的实用性**：不需要任何训练数据或目标模型的微调，只需白盒访问进行梯度计算。这大大降低了攻击门槛，也意味着任何开源 MLLM 都面临这种威胁
- **ARC 评估框架的方法论贡献**：通过质量×相关性的二维评估，提供了比传统 ASR 更精确的攻击效果度量，对后续安全研究有标准化价值

## 局限与展望

- 攻击需要白盒访问（梯度计算），对闭源模型（如 GPT-4V）不直接适用。但可以通过迁移攻击（在开源模型上生成对抗样本，在闭源模型上测试）部分解决
- 此处生成的对抗性图像/音频在人类看来通常是无意义的噪声，容易被人工审核识别。但自动化系统中缺乏人工审核环节
- 对更大规模模型（如 LLaVA-Next-34B）的测试有限
- 防御探索初步，未深入研究基于嵌入空间监控的检测方法——例如检测非文本模态的嵌入是否异常接近有害文本的嵌入
- 未考虑多轮对话场景下的攻击持续性——模型是否能在后续轮次中"记住"非文本指令

## 相关工作与启发

- **vs Visual Adversarial Examples (Qi et al. 2024)**：Qi 等人的视觉对抗攻击仍然依赖文本通道传递部分恶意意图，Con Instruction 完全通过非文本模态传递指令，更彻底
- **vs GCG 等文本越狱**：GCG 操控文本 token，可以被困惑度过滤器检测。Con Instruction 操控的是图像像素/音频频谱，完全不在文本安全检测的范围内
- **vs Multimodal Prompt Injection**：提示注入通常将文本嵌入图像中（如 OCR 载体），Con Instruction 在嵌入空间而非像素空间中编码信息，更隐蔽
- **对安全研究的启发**：MLLM 安全需要"全模态"防护，不能仅依赖文本过滤器。嵌入空间监控可能是一个有效的防御方向

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统性探索纯非文本模态作为恶意指令载体的越狱攻击，视角独特
- 实验充分度: ⭐⭐⭐⭐ 覆盖了视觉和音频模态、多个模型、两个基准，但闭源模型测试不足
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，ARC 框架设计合理
- 价值: ⭐⭐⭐⭐⭐ 揭示了多模态 AI 安全的关键盲区，ARC 框架可作为标准化评估工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency](../../ICCV2025/multimodal_vlm/jailbreaking_multimodal_large_language_models_via_shuffle_inconsistency.md)
- [\[ACL 2025\] MEIT: Multimodal Electrocardiogram Instruction Tuning on Large Language Models for Report Generation](meit_multimodal_electrocardiogram_instruction_tuning_on_large_language_models_fo.md)
- [\[ACL 2025\] HiDe-LLaVA: Hierarchical Decoupling for Continual Instruction Tuning of Multimodal Large Language Model](hidellava_hierarchical_decoupling_for_continual_instruction.md)
- [\[ACL 2025\] Vision-Language Models Struggle to Align Entities across Modalities](vision-language_models_struggle_to_align_entities_across_modalities.md)
- [\[ICCV 2025\] IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves](../../ICCV2025/multimodal_vlm/ideator_jailbreaking_and_benchmarking_large_visionlanguage_m.md)

</div>

<!-- RELATED:END -->
