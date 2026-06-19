---
title: >-
  [论文解读] Cross-modal Information Flow in Multimodal Large Language Models
description: >-
  [CVPR 2025][多模态VLM][多模态信息流] 通过"attention knockout"方法系统性地追踪 MLLM 中视觉和语言信息的流动路径，发现视觉信息分两阶段（先全局后局部）融入语言表征，最终在中间层由问题位置传播到最后位置生成答案。 领域现状：自回归多模态大语言模型（MLLMs）在 VQA 等视觉-语言任…
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "多模态信息流"
  - "注意力机制可解释性"
  - "跨模态融合"
  - "LLaVA"
  - "VQA"
---

# Cross-modal Information Flow in Multimodal Large Language Models

**会议**: CVPR 2025  
**arXiv**: [2411.18620](https://arxiv.org/abs/2411.18620)  
**代码**: [https://github.com/FightingFighting/cross-modal-information-flow-in-MLLM](https://github.com/FightingFighting/cross-modal-information-flow-in-MLLM)  
**领域**: 多模态VLM  
**关键词**: 多模态信息流、注意力机制可解释性、跨模态融合、LLaVA、VQA

## 一句话总结
通过"attention knockout"方法系统性地追踪 MLLM 中视觉和语言信息的流动路径，发现视觉信息分两阶段（先全局后局部）融入语言表征，最终在中间层由问题位置传播到最后位置生成答案。

## 研究背景与动机

**领域现状**：自回归多模态大语言模型（MLLMs）在 VQA 等视觉-语言任务上取得了显著进展，但其内部工作机制——特别是视觉和语言信息如何交互融合——仍然是个黑箱。

**现有痛点**：已有的可解释性工作主要关注单一方面，如参数中的知识存储、视觉 token 冗余、安全机制定位等，但缺乏对两个模态之间信息流动路径的全面理解。我们不知道模型在哪些层、以什么方式将图像信息和文本信息整合起来。

**核心矛盾**：MLLM 中注意力模块是唯一允许不同位置隐表征之间通信的组件，但我们不清楚这种通信在不同层的功能分工——是所有层都在做融合，还是存在明确的阶段划分？

**本文目标** 三个具体问题：(1) 视觉信息如何从图像 token 传播到问题 token？(2) 这种传播是粗粒度全局的还是细粒度局部的？(3) 融合后的多模态信息如何最终到达最后位置生成答案？

**切入角度**：借鉴 NLP 领域 mechanistic interpretability 中的 attention knockout 方法——通过选择性地阻断特定层中特定 token 对之间的注意力边，观察答案预测概率的变化来反向推断信息流路径。

**核心 idea**：用 attention knockout 逐层阻断跨模态注意力边，揭示 MLLM 中视觉-语言信息融合的两阶段机制和三步预测流程。

## 方法详解

### 整体框架
以自回归 MLLM（如 LLaVA 系列）为研究对象，输入为 [图像 patch 特征, 问题 token] 的拼接序列。在每一层的 masked multi-head attention 中，通过修改注意力掩码矩阵，将特定 source 集合（如图像 token）到 target 集合（如问题 token）的注意力边设为 $-\infty$，从而阻断两组 token 之间的信息流动。然后对比阻断前后答案预测概率的相对变化 $p_c\% = ((p_2 - p_1)/p_1) \times 100$，来量化该信息流的重要性。

### 关键设计

1. **Attention Knockout 方法**:

    - 功能：选择性阻断特定 token 组之间的注意力连接
    - 核心思路：在注意力掩码矩阵 $M$ 中，对 source 集合 $\mathbb{S}$ 到 target 集合 $\mathbb{T}$ 的所有位置对 $(s,t)$ 设置 $M_{s,t} = -\infty$，使 softmax 后注意力权重为零。以滑动窗口 $k=9$ 层为单位进行阻断，避免单层效果太弱
    - 设计动机：注意力是 Transformer 中唯一的跨位置通信模块，阻断它等于切断信息流。比梯度方法更直接，能精确定位信息传递发生的层

2. **三组对照实验设计**:

    - 功能：分解不同模态对最终预测的贡献路径
    - 核心思路：设计三组 knockout 实验——(a) Image→Last vs Question→Last：测试哪个模态直接影响最终预测；(b) Image→Question：测试视觉信息是否间接通过问题 token 传播；(c) RelatedPatches→Question vs OtherPatches→Question：区分全局和局部视觉信息的贡献
    - 设计动机：自回归模型中图像在前、问题在后，图像无法 attend 问题，因此只有两种路径——图像直接到最后位置，或先到问题再到最后位置

3. **细粒度图像区域拆分**:

    - 功能：区分问题相关的图像区域和背景区域的信息贡献
    - 核心思路：利用 GQA 数据集的 bounding box 标注，将图像 patch 分为包含被提问对象的 $V_{obj}$ 和其余 $V_{oth}$，分别做 knockout 观察两类区域在不同层的信息流差异
    - 设计动机：验证模型是否真的"理解"了问题指向的对象，还是只依赖全局统计特征

### 损失函数 / 训练策略
本文为分析论文，不涉及训练。实验在 LLaVA-1.5-7b/13b、LLaVA-v1.6-Vicuna-7b、Llama3-LLaVA-NEXT-8b 四个预训练模型上进行，使用 GQA 数据集的 6 类 VQA 任务。

## 实验关键数据

### 主实验

| 实验设置 (LLaVA-1.5-13b) | 最大概率下降 | 关键层范围 | 说明 |
|--------------------------|-------------|-----------|------|
| Question ↛ Last | ~30% | 中间层 (15-25层) | 问题信息直接驱动最终预测 |
| Image ↛ Last | ~5% | - | 图像信息不直接影响最终预测 |
| Last ↛ Last | ~0% | - | 最后位置自身输入不含关键信息 |
| Image ↛ Question (第一阶段) | ~60% | 底层 (0-4层) | 全局视觉信息融入问题表征 |
| Image ↛ Question (第二阶段) | ~21% | 中间层 (~10层) | 目标相关视觉信息融入问题表征 |

### 消融实验

| 配置 | 概率变化 | 影响层 | 说明 |
|------|---------|-------|------|
| OtherPatches ↛ Question | 底层大幅下降 | 0-4层 | 第一阶段主要是全局/背景信息传播 |
| RelatedPatches ↛ Question | 中间层下降 | ~10层 | 第二阶段主要是目标相关信息传播 |
| 小模型 (7b) vs 大模型 (13b) | 7b 第一阶段流弱 | 0-4层 | 小模型全局信息整合能力较弱 |
| Logit lens 观测 | 小写答案中间层涌现 | ~15-20层 | 语义推理在中间层完成 |
| 大写→小写转换 | 高层进行格式修正 | 25-40层 | 语法修正在高层完成 |

### 关键发现
- **两阶段视觉融合**：底层（0-4层）进行全局视觉信息到问题 token 的传播，约占总信息流的 60%；中间层（~10层）进行问题相关的局部视觉信息传播，约占 21%。两阶段在所有模型和任务上一致
- **问题 token 是信息枢纽**：最终预测几乎完全依赖从问题位置流向最后位置的信息，而非图像直接贡献。这说明多模态融合的主战场在问题 token 的隐表征中
- **语义先于语法**：通过 logit lens 观察到，模型先在中间层生成小写形式的正确答案（语义推理），再在高层将首字母转为大写（语法修正），这是一个全新的发现

## 亮点与洞察
- **Attention knockout 方法简洁有效**：不需要训练探针或额外模型，直接通过阻断注意力边来追踪信息流，可推广到任何基于 Transformer 的多模态模型分析
- **两阶段融合发现具有指导意义**：底层做全局感知、中间层做局部精细融合的模式，暗示可以针对性地在不同层设计不同的视觉 token 压缩策略，而非全层统一处理
- **"问题 token 是融合枢纽"的发现**可以解释为什么 prompt engineering 和问题措辞对 VLM 性能影响很大——因为视觉信息最终需要"存储"在问题 token 的表征中

## 局限与展望
- 仅在 LLaVA 系列模型上验证，对 Qwen-VL、InternVL 等不同架构的 MLLM 是否有相同规律未知
- 实验限于 VQA 的单词/短语答案，对长文本生成任务（如图像描述）的信息流模式未探索
- Attention knockout 方法基于"阻断注意力等于切断信息流"的假设，但 FFN 层也可能在隐式传递信息（通过残差连接中累积的表征）
- 未探索如何利用发现的信息流模式来实际改进模型设计（如层级自适应的 token 压缩）

## 相关工作与启发
- **vs Logit Lens / Tuned Lens**：这些方法只能观察每层的输出分布变化，无法揭示模态间的信息传递路径。本文的 knockout 方法能直接定位跨模态信息流发生的层和方向
- **vs FastV / Token 压缩方法**：这些方法在经验上删减视觉 token，本文的发现为它们提供了理论依据——底层需要全局 token，中间层之后只需保留与问题相关的局部 token
- **vs Probing 方法**：Probing 检测表征中"包含什么信息"，本文揭示"信息从哪来、怎么来的"，两者互补

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统揭示 MLLM 中跨模态信息流的两阶段机制，方法虽借鉴已有 knockout 思路，但应用到多模态场景是新贡献
- 实验充分度: ⭐⭐⭐⭐ 4 个模型、6 类 VQA 任务、多组对照实验，结论一致性强，但缺少非 LLaVA 架构验证
- 写作质量: ⭐⭐⭐⭐⭐ 论文结构清晰，从简单到复杂层层推进，图表直观易懂
- 价值: ⭐⭐⭐⭐ 对多模态模型的可解释性研究有重要贡献，发现可直接指导 token 压缩和模型设计

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] VEENA: Interpreting and Enhancing Emotional Circuits in Large Vision-Language Models via Cross-Modal Information Flow](../../ICML2026/multimodal_vlm/interpreting_and_enhancing_emotional_circuits_in_large_vision-language_models_vi.md)
- [\[CVPR 2025\] On the Out-of-Distribution Generalization of Multimodal Large Language Models](on_the_out-of-distribution_generalization_of_large_multimodal_models.md)
- [\[CVPR 2025\] EventGPT: Event Stream Understanding with Multimodal Large Language Models](eventgpt_event_stream_understanding_with_multimodal_large_language_models.md)
- [\[CVPR 2025\] RAP: Retrieval-Augmented Personalization for Multimodal Large Language Models](rap_retrieval-augmented_personalization_for_multimodal_large_language_models.md)
- [\[CVPR 2025\] 4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models](4d_langsplat_4d_language_gaussian_splatting_via_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
