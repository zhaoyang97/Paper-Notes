---
title: >-
  [论文解读] Argus: Vision-Centric Reasoning with Grounded Chain-of-Thought
description: >-
  [CVPR 2025][LLM推理][视觉链式思维] Argus 提出了一种grounded visual CoT机制，通过让MLLM先预测与问题相关的bounding box（RoI），然后重新采样/编码该区域的视觉token作为推理上下文，实现了显式的目标导向视觉注意力，在7B/8B级MLLM中取得视觉推理和目标grounding双料SOTA。
tags:
  - "CVPR 2025"
  - "LLM推理"
  - "视觉链式思维"
  - "视觉注意力接地"
  - "多模态推理"
  - "RoI重新关注"
  - "混合视觉专家"
---

# Argus: Vision-Centric Reasoning with Grounded Chain-of-Thought

**会议**: CVPR 2025  
**arXiv**: [2505.23766](https://arxiv.org/abs/2505.23766)  
**代码**: [https://yunzeman.github.io/argus/](https://yunzeman.github.io/argus/) (项目页)  
**领域**: LLM推理  
**关键词**: 视觉链式思维, 视觉注意力接地, 多模态推理, RoI重新关注, 混合视觉专家

## 一句话总结

Argus 提出了一种grounded visual CoT机制，通过让MLLM先预测与问题相关的bounding box（RoI），然后重新采样/编码该区域的视觉token作为推理上下文，实现了显式的目标导向视觉注意力，在7B/8B级MLLM中取得视觉推理和目标grounding双料SOTA。

## 研究背景与动机

**领域现状**：现有MLLM在视觉-语言任务上表现出色，但在需要精确视觉聚焦的视觉中心（vision-centric）场景中表现不佳，如识别小物体的空间关系、读取图表中的特定数据等。

**现有痛点**：现有MLLM主要依赖隐式的self-attention机制来处理视觉token与语言token之间的交互，缺乏显式的目标导向视觉搜索能力。Cambrian-1和Eagle虽然研究了多视觉编码器的互补性，但仍然没有引入有意识的视觉注意力控制。

**核心矛盾**：认知科学区分了两种视觉注意力——刺激驱动的involuntary attention（自下而上，由图像中显著物体触发）和目标导向的voluntary attention（自上而下，由任务目标引导）。现有MLLM的ViT编码器对应前者，而LLM中的cross-attention隐式实现后者，但这种隐式目标导向注意力不够精确和可控。

**本文目标** (1) 如何在MLLM中引入显式的语言引导视觉注意力？(2) 这种更显式的视觉关注是否能提升推理任务表现？

**切入角度**：借鉴认知科学中的voluntary attention概念，利用object-centric grounding（预测bounding box）作为视觉CoT的中间信号——先让模型找到"应该看哪里"，再让模型"仔细看那里"进行推理。

**核心 idea**：用grounding预测的bounding box作为视觉CoT信号，通过RoI区域的视觉token重采样/重编码来实现显式的目标导向视觉注意力。

## 方法详解

### 整体框架

Argus的pipeline分为两个阶段的推理过程：(1) 给定图像和问题，模型首先通过混合视觉专家（MoVE）对图像进行初始编码，然后**预测与问题最相关的bounding box**（RoI坐标以文本形式输出）；(2) 根据预测的bounding box，从原图中**裁剪/采样对应区域**，将其视觉token作为额外的CoT上下文拼接到输入序列中，模型再基于这些聚焦的视觉信息生成最终答案。

### 关键设计

1. **混合视觉专家编码器（MoVE）**:

    - 功能：提取互补的视觉特征，最小化图像到token过程中的信息损失
    - 核心思路：同时使用CLIP ViT-L/14（448×448, 语义对齐）、ConvNeXt-XXL（1024×1024, 细粒度空间特征）和EVA-02-L/16（1024×1024, 检测导向特征）三个视觉专家，将三者的特征插值到统一的32×32空间分辨率后沿channel拼接（5120维），再通过MLP映射到LLM的4096维空间，产生1024个视觉token
    - 设计动机：不同视觉编码器各有所长——CLIP擅长语义对齐，ConvNeXt保留细节纹理，EVA-02擅长目标检测，三者互补可实现更全面的视觉理解

2. **RoI预测与视觉上下文重关注（Visual Context Re-engagement）**:

    - 功能：实现显式的目标导向视觉搜索，让模型"先找再看"
    - 核心思路：模型以文本坐标形式 $[x_{min}, y_{min}, x_{max}, y_{max}]$ 输出归一化bounding box，然后根据该box对图像进行RoI采样。论文系统比较了4种视觉注意力重关注策略：(a) 隐式self-attention（基线，不做额外处理）; (b) 隐式box指引（只输出box坐标文本作为CoT，不重新编码视觉token）; (c) 显式RoI重编码（裁剪区域作为新图像送入视觉编码器）; (d) 显式RoI重采样（从初始token缓存中检索与box重叠的patch token）
    - 设计动机：隐式方法对视觉token的关注控制有限；显式方法通过实际提取RoI区域的视觉token，强制模型聚焦关键区域。重采样效率更高（复用缓存token），重编码在小物体感知上更优（高分辨率处理）

3. **Grounded CoT训练策略**:

    - 功能：将grounding能力与推理能力协同训练
    - 核心思路：SFT阶段使用三类数据混合训练——Eagle1.8M（通用推理）、VCoT数据集（带bounding box标注的CoT推理，包含TextVQA/DocVQA/ScienceQA等）、GRIT+Shikra（大规模grounding数据）。训练格式为多轮对话：模型先输出`<roi-box>`坐标，用户提供`<visual-context>` token，模型再生成答案
    - 设计动机：grounding数据增强了模型的目标感知能力，从而提升bounding box预测质量，进而最大化CoT机制的效用

### 损失函数 / 训练策略

两阶段训练：(1) 预训练阶段用LLaVA-595K，冻结LLM，训练视觉编码器+MLP（32×A100, 4小时）；(2) SFT阶段全参数微调，1个epoch，batch=256，lr=2e-5，AdamW+cosine scheduler（64×A100, 28小时）。

## 实验关键数据

### 主实验

| 模型 | Vision-Centric Avg | V-Star | CV-Bench-2D | MMVP | TextVQA | ChartQA | MMMU | MMBench |
|------|-------------------|--------|-------------|------|---------|---------|------|---------|
| GPT-4o | 73.7 | 70.7 | 79.8 | 58.5 | 79.7 | 86.9 | 68.9 | 87.1 |
| Qwen2.5-VL | 72.6 | 72.8 | 80.0 | 53.1 | 84.9 | 85.2 | 58.6 | 86.5 |
| Eagle-X3-8B | 59.6 | 60.7 | 66.4 | 45.1 | 70.9 | 70.4 | 39.8 | 70.9 |
| Visual-CoT-7B | 54.4 | 49.7 | 61.5 | 35.7 | 70.0 | 69.7 | 37.2 | 67.3 |
| **Argus-X3-8B** | **65.3** | **78.5** | **68.5** | **45.5** | **73.6** | **74.8** | 40.4 | **72.9** |

Referring Grounding (RefCOCO):

| 模型 | RefCOCO-val | RefCOCO-testA | RefCOCO+-val | RefCOCOg-val |
|------|-------------|---------------|--------------|--------------|
| G-DINO-L (专家) | 90.6 | 93.2 | 82.8 | 86.1 |
| QwenVL-7B | 89.4 | 92.3 | 83.1 | 85.6 |
| **Argus-X3-8B** | **89.8** | **92.9** | **84.7** | **86.7** |

### 消融实验

| 视觉注意力策略 | V-Star | CV-Bench-2D | TextVQA | ChartQA |
|---------------|--------|-------------|---------|---------|
| 隐式Self-Att | 58.6 | 64.5 | 69.2 | 67.3 |
| 隐式Box指引 | 63.9 | 67.0 | 71.6 | 70.4 |
| 显式RoI重编码 | 68.1 | 67.4 | 71.4 | 71.8 |
| 显式RoI重采样 | 67.0 | 68.2 | 73.9 | 72.7 |

CoT和Grounding的叠加效果：

| 配置 | V-Star | CV-Bench-2D | TextVQA | ChartQA |
|------|--------|-------------|---------|---------|
| Baseline (Eagle-X3) | 55.3 | 64.9 | 66.3 | 63.0 |
| + CoT signals | 62.7 | 65.5 | 71.1 | 69.4 |
| ++ Grounding (Argus) | 67.0 | 68.2 | 73.9 | 72.7 |

### 关键发现

- 显式视觉RoI重关注（无论重采样还是重编码）一致优于隐式方法，证实了"先找再看"策略的有效性
- 重采样在大多数任务上优于重编码，因为保留了原始位置信息且避免了分辨率变换带来的分布偏移；但在V-Star（小目标感知）上重编码更优，因为可以用更大patch处理小区域
- 重采样计算效率显著更高：GMACs仅4355 vs 8711，额外token仅26 vs 1024，推理时间492ms vs 827ms
- 多RoI扩展（将单目标扩展为多目标推理）在V-Star上从68.1提升到78.5，在CV-Bench-2D上从64.2提升到69.6

## 亮点与洞察

- **视觉CoT的类比非常巧妙**：将认知科学中involuntary/voluntary attention的概念映射到MLLM的ViT编码（stimulus-driven）和RoI重关注（goal-directed），提供了清晰的理论动机
- **重采样策略的效率优势值得关注**：仅增加26个token就能显著提升推理效果，这比重编码1024个token高效得多，适合实际部署
- **Grounding与推理的正反馈循环**：grounding数据提升box预测准确度 → 更好的CoT信号 → 更好的推理结果，这种协同效应是方法成功的关键

## 局限与展望

- 仅验证了8B规模的LLM，未测试更大规模模型（如70B）能否进一步放大视觉CoT的收益
- 视觉CoT训练数据稀缺，现有数据主要来自文本理解和科学问答场景，缺乏更多样化的视觉推理CoT标注
- 多RoI扩展目前需要多步串行推理，效率有待优化
- MMMU和GQA上提升有限，作者归因于这些benchmark更依赖语言先验而非视觉信息

## 相关工作与启发

- **vs Eagle**: Argus在Eagle-X3的基础上增加了视觉CoT机制，在共享相同MoVE架构的情况下，V-Star上从60.7提升至78.5（+17.8），证明了显式视觉注意力的巨大价值
- **vs Visual-CoT**: Visual-CoT使用外部目标检测器提供RoI，而Argus将grounding能力内化到模型中，实现了端到端训练，性能全面超越
- **vs Cambrian-1**: 都强调vision-centric设计，但Cambrian-1关注编码器组合，Argus关注推理时的注意力机制，两者互补

## 评分

- 新颖性: ⭐⭐⭐⭐ 视觉CoT的formulation清晰优雅，但grounding+re-engage的思路在之前的工作中有所铺垫
- 实验充分度: ⭐⭐⭐⭐⭐ 消融实验全面系统，涵盖4种re-engagement策略、编码器容量、上下文扩展、多RoI等
- 写作质量: ⭐⭐⭐⭐⭐ 论文结构清晰，认知科学的类比引人入胜，图示直观
- 价值: ⭐⭐⭐⭐ 为MLLM的视觉推理提供了清晰的改进方向，重采样策略的实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Interleaved-Modal Chain-of-Thought](interleaved-modal_chain-of-thought.md)
- [\[CVPR 2026\] Revisiting the Necessity of Lengthy Chain-of-Thought in Vision-centric Reasoning Generalization](../../CVPR2026/llm_reasoning/revisiting_the_necessity_of_lengthy_chain-of-thought_in_vision-centric_reasoning.md)
- [\[CVPR 2025\] VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection](videoespresso_a_large-scale_chain-of-thought_dataset_for_fine-grained_video_reas.md)
- [\[ACL 2025\] Improve Vision Language Model Chain-of-thought Reasoning](../../ACL2025/llm_reasoning/improve_vlm_cot_reasoning.md)
- [\[CVPR 2025\] Style Evolving along Chain-of-Thought for Unknown-Domain Object Detection](style_evolving_along_chain-of-thought_for_unknown-domain_object_detection.md)

</div>

<!-- RELATED:END -->
