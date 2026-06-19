---
title: >-
  [论文解读] Uni-MuMER: Unified Multi-Task Fine-Tuning of Vision-Language Model for Handwritten Mathematical Expression Recognition
description: >-
  [NeurIPS 2025 Spotlight][多模态VLM][手写数学公式识别] 提出 Uni-MuMER，通过三种数据驱动任务（Tree-CoT、Error-Driven Learning、Symbol Counting）对开源 VLM 进行统一多任务微调，在 CROHME 和 HME100K 数据集上大幅超越专用轻量模型和零样本商用 VLM。
tags:
  - "NeurIPS 2025 Spotlight"
  - "多模态VLM"
  - "手写数学公式识别"
  - "视觉语言模型"
  - "多任务微调"
  - "Chain-of-Thought"
  - "错误驱动学习"
---

# Uni-MuMER: Unified Multi-Task Fine-Tuning of Vision-Language Model for Handwritten Mathematical Expression Recognition

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2505.23566](https://arxiv.org/abs/2505.23566)  
**代码**: [https://github.com/BFlameSwift/Uni-MuMER](https://github.com/BFlameSwift/Uni-MuMER)  
**领域**: 多模态VLM  
**关键词**: 手写数学公式识别, 视觉语言模型, 多任务微调, Chain-of-Thought, 错误驱动学习

## 一句话总结

提出 Uni-MuMER，通过三种数据驱动任务（Tree-CoT、Error-Driven Learning、Symbol Counting）对开源 VLM 进行统一多任务微调，在 CROHME 和 HME100K 数据集上大幅超越专用轻量模型和零样本商用 VLM。

## 研究背景与动机

手写数学公式识别（HMER）因符号布局的自由度和手写风格的多样性，是 OCR 领域的持久挑战。此前方法大多通过孤立的架构改进（树解码器、位置感知注意力等）来提升性能，但近年进展甚微——在 CROHME 数据集上从 CoMER 到 SSAN 仅提升约 3%。这些方法面临三个瓶颈：改进彼此独立难以整合、单一辅助任务难以覆盖 HMER 多维挑战、单域数据集训练缺乏可扩展性。

与此同时，预训练 VLM 在结构化识别任务上展现出意想不到的强能力，但商用模型使用不透明数据，难以系统性地指导改进。因此，如何赋能开源 VLM 在 HMER 上达到可比甚至更优性能成为关键问题。

## 方法详解

### 整体框架

Uni-MuMER 以 Qwen2.5-VL-3B 为骨干 VLM，**不修改任何架构**，通过全量微调将领域知识注入通用框架。统一训练四个任务：Vanilla HMER（基础识别）、Tree-Aware Chain-of-Thought（结构化推理）、Error-Driven Learning（错误纠正）、Symbol Counting（符号计数），输入为手写表达式图像加任务指令，输出对应的 LaTeX 序列。

### 关键设计

1. **Tree-Aware Chain-of-Thought (Tree-CoT)**: 将 LaTeX 表达式解析为抽象语法树（AST），通过深度优先遍历将树结构线性化为制表符缩进的文本表示。模型先生成序列化的树结构，再输出最终 LaTeX——显式地引导模型推理二维空间关系。心理学动机是将隐式的布局理解转化为显式的结构化 CoT，特别有助于复杂结构表达式。

2. **Error-Driven Learning (EDL)**: 采用"从错误中学习"范式。首先通过交叉折叠训练（将数据集分为多个 fold，交叉训练+多次采样）构建错误语料库，收集模型自身的错误预测。然后定义两个子任务：错误检测（用 `<error_start>/<error_end>` 标记错误位置，用 `<deleted>` 标记遗漏）和错误纠正（输入标记后的表达式，输出纠正日志和正确 LaTeX）。核心思想是让模型学会区分视觉相似字符（如 2↔z、0↔o）。

3. **Symbol Counting (SC)**: 在输出前添加符号计数字符串（如 `\frac:1,a:1,2:2,+:1`），迫使模型在生成 LaTeX 前先准确统计所有可见符号数量。灵感来自 CAN 的观察——模型常产生局部连贯但全局不一致的输出（重复或遗漏符号），SC 通过显式计数约束缓解长表达式中的符号幻觉。

### 损失函数 / 训练策略

所有数据集（CROHME、HME100K、MathWriting、Im2Latexv2）的三种数据驱动任务训练数据统一混合，仅训练一轮 epoch。Uni-MuMER† 使用约 386K 图像构建的约 1.6M 训练样本（原始+三任务衍生数据）。采用标准自回归交叉熵损失，无特殊损失设计。

## 实验关键数据

### 主实验

| 数据集 | 指标 | Uni-MuMER† | SSAN (Prev SOTA) | 提升 |
|--------|------|-----------|-----------------|------|
| CROHME 平均 | ExpRate | 79.74% | 63.43% (w/ aug) | +16.31% |
| CROHME 平均 | ExpRate@CDM | 82.86% | — | — |
| CROHME14 | ExpRate | 82.05% | 62.58% | +19.47% |
| HME100K | ExpRate | 72.66% | — | — |
| HME100K | ExpRate@CDM | 74.30% | — | — |

零样本对比：超越 Gemini2.5-flash（55.32%→79.74%，+24.42%）和 Qwen2.5-VL-72B（56.40%→79.74%）。

### 消融实验

| 配置 | CROHME 平均 ExpRate | 说明 |
|------|---------------------|------|
| Vanilla baseline | 68.64% | 仅 Vanilla HMER |
| + Tree-CoT | 70.85% (+2.21) | 结构化推理提升 |
| + EDL | 70.30% (+1.66) | 减少字符混淆 |
| + SC | 69.86% (+1.22) | 符号一致性提升 |
| Tree-CoT + EDL + SC | 73.29% (+4.65) | 三者互补，最优 |

### 关键发现

- **Tree-CoT 对复杂结构表达式提升最大**（约 5-6%），对简单表达式作用有限，说明其核心价值在于结构推理。
- **EDL 显著减少字符混淆**：top-5 字母-数字混淆从 5.25 降至 3.31（下降 37%），特别是 3↔z（-0.58）和 1↔n（-0.59）效果突出。
- **SC 改善长表达式一致性**：当符号重复 5 次以上时效果显著，但对简单表达式略有负面影响。
- 轻量模型 CoMER 使用相同外部数据（386K）性能反而下降，说明轻量架构无法有效利用大规模多样数据。

## 亮点与洞察

- **范式转移**：从"改架构"转向"改数据"——不修改 VLM 架构，完全依靠数据驱动的多任务学习注入领域知识，简洁优雅。
- **错误语料库构建方法巧妙**：交叉折叠+多次采样自动收集模型错误，无需人工标注，构建的错误数据规模与原始数据相当。
- 引入 CDM（Character Detection Matching）视觉评估指标，解决了 ExpRate 对 LaTeX 语法风格差异不公平惩罚的问题。
- 基于 vLLM 推理框架，速度优于传统专用方法，实用性强。- 多任务的增益呈现正交性（去掉任何一个都会降低性能），设计合理性得到充分验证。
- 在训练数据规模上，386K图像扩展到 1.6M样本的三任务数据增强策略具有参考价值。

## 局限与展望

- Qwen2.5-VL-3B 参数量相比轻量模型仍偏大，部署成本更高。
- 错误语料库构建需要多轮训练和采样，预处理开销较大。
- SC 在简单表达式上存在轻微性能下降，可考虑自适应启用机制（根据表达式长度动态开关）。
- 未探索更大规模 VLM（如 7B/72B）的微调潜力。
- 仅在 LaTeX 格式输出上验证，MathML 等其他标记语言格式的泛化性未知。
- Tree-CoT 依赖 AST 解析，对于无法解析为标准 AST 的非标准表达式可能失效。
- CDM 指标虽解决语法风格问题，但对渲染引擎依赖性强。

## 相关工作与启发

- 延续了 CAN 的符号计数思路，但将其整合进 VLM 框架更加自然。
- Tree-CoT 的 AST 序列化方案可推广至其他结构化输出任务（代码生成、化学式识别等）。
- EDL 的"模型自纠"范式与 Self-Refine、Constitutional AI 理念呼应，在 OCR 领域的具体化值得关注。- 计算效率上，基于 vLLM 推理框架推理速度优于传统方法，对实际应用部署友好。
- 作为开源工作，对 VLM 在 OCR 领域的应用研究有重要参考价值。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 数据驱动三任务统一微调范式新颖，Tree-CoT 和 EDL 设计巧妙
- **实验充分度**: ⭐⭐⭐⭐⭐ 多数据集、多基线对比充分，消融实验细致，每个模块贡献清晰
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，图示丰富，动机阐述充分
- **价值**: ⭐⭐⭐⭐ HMER 领域重大突破，范式转移有启发意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition](../../CVPR2026/multimodal_vlm/trivia_self-supervised_fine-tuning_of_vision-language_models_for_table_recogniti.md)
- [\[CVPR 2026\] Rethinking BCE Loss for Multi-Label Image Recognition with Fine-Tuning](../../CVPR2026/multimodal_vlm/rethinking_bce_loss_for_multi-label_image_recognition_with_fine-tuning.md)
- [\[CVPR 2026\] UNI-OOD: Unified Object- and Image-level Out-of-Distribution Detection via Cross-Context Attentive Vision-Language Modeling](../../CVPR2026/multimodal_vlm/uni-ood_unified_object-_and_image-level_out-of-distribution_detection_via_cross-.md)
- [\[ACL 2025\] Can Vision-Language Models Evaluate Handwritten Math?](../../ACL2025/multimodal_vlm/can_vision-language_models_evaluate_handwritten_math.md)
- [\[NeurIPS 2025\] Advancing Compositional Awareness in CLIP with Efficient Fine-Tuning](advancing_compositional_awareness_in_clip_with_efficient_fin.md)

</div>

<!-- RELATED:END -->
