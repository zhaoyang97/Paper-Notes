---
title: >-
  [论文解读] SDEval: Safety Dynamic Evaluation for Multimodal Large Language Models
description: >-
  [AAAI 2026][多模态VLM][MLLM安全] 提出首个 MLLM 安全动态评估框架 SDEval，通过文本动态（6种策略）、图像动态（2类策略）和跨模态动态（4种策略）从原始安全基准生成可控复杂度的变体样本，在 MLLMGuard 和 VLSBench 上使 InternVL-3-78B 安全率下降近 10%，有效缓解数据泄露并暴露模型安全漏洞。
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "MLLM安全"
  - "动态评估"
  - "数据泄露"
  - "越狱攻击"
  - "安全基准"
---

# SDEval: Safety Dynamic Evaluation for Multimodal Large Language Models

**会议**: AAAI 2026  
**arXiv**: [2508.06142](https://arxiv.org/abs/2508.06142)  
**代码**: [SDEval](https://github.com/hq-King/SDEval)  
**领域**: 多模态VLM  
**关键词**: MLLM安全, 动态评估, 数据泄露, 越狱攻击, 安全基准

## 一句话总结

提出首个 MLLM 安全动态评估框架 SDEval，通过文本动态（6种策略）、图像动态（2类策略）和跨模态动态（4种策略）从原始安全基准生成可控复杂度的变体样本，在 MLLMGuard 和 VLSBench 上使 InternVL-3-78B 安全率下降近 10%，有效缓解数据泄露并暴露模型安全漏洞。

## 研究背景与动机

**领域现状**：MLLM 在多模态理解方面取得巨大进展，但也面临生成有害内容的风险。社区已构建了多个安全评估基准（MLLMGuard、VLSBench、MMSafetyBench 等），用于评估模型抵御有害输出的能力。

**现有痛点**：
(1) **数据泄露严重**——大多数安全基准整合自开源数据集，极可能已被包含在 MLLM 训练集中，导致评估结果失真；
(2) **静态数据集复杂度固定**——手工构建的基准无法匹配 MLLM 的快速进步，难以精准评估模型性能上限；
(3) **攻击方法持续演进**——新型越狱攻击不断出现，固定基准无法及时覆盖新风险。

**核心矛盾**：已有动态评估方法（DyVal 等）仅针对模型能力评估，不适用于开放式回答的安全评估场景，且忽视了能力-安全平衡问题。

**本文切入角度**：设计一个通用、灵活的安全动态评估框架，从任意原始基准出发无限生成复杂度可调、数据污染率更低的变体样本。

## 方法详解

### 整体框架

SDEval 以原始安全基准样本 $P=(T, I)$ 为输入，通过动态策略集 $\mathfrak{D}$ 生成新的文本-图像对 $P'=(T', I')$。框架分为三个维度：文本动态、图像动态和跨模态动态。生成后由验证器 Agent 确保语义一致性，最后用评分器判断模型响应的有害程度。

### 关键设计

1. **文本动态策略（6种）**

    - **功能**：从人类绕过审核的策略出发，在不改变语义的前提下修改文本以增加安全识别难度
    - **核心策略**：
        - 词替换（同义词/上下文近似词替换 ≤5 个词）
        - 句子改写（保留核心概念、变换句式）
        - 添加描述（相关/无关描述干扰模型注意力）
        - 制造拼写错误（重复字母、特殊字符等不影响可读性的变形）
        - 多语言混合（中英俄法日韩多语言重构）
        - 思维链注入（添加"逐步回答"指令）
    - **设计动机**：模拟真实世界中用户通过语言变形绕过安全审查的行为

2. **图像动态策略（2类）**

    - **功能**：通过基础增强和生成式操作修改图像，降低数据泄露并测试模型视觉安全识别能力
    - **基础增强**：空间变换（随机填充 10%-20% + 翻转）和颜色变换（颜色反转 + 椒盐噪声）
    - **生成与操作**：Caption 引导的 SD3.5-Large 重新生成；使用 ICEdit 进行物体插入、文字插入和风格迁移
    - **质量保证**：GPT-4o 验证生成图像与原始图像的语义一致性
    - **设计动机**：生成式方式产生的图像与原始样本视觉差异大，能有效降低数据泄露率

3. **跨模态动态策略（4种）**

    - **功能**：探索文本-图像交互对安全的影响
    - **Text-to-Image**：将文本动态变体注入图像生成（采样文本扰动 → 生成 caption → SD 生成新图）
    - **Image-to-Text**：将图像动态变体注入文本（采样图像扰动 → GPT-4o 生成安全相关 caption → 前置于原始文本）
    - **FigStep 越狱**：将文本提示转为排版图片直接输入（绕过文本安全对齐）
    - **HADES 越狱**：将不安全关键词从文本迁移到图像中

### 评估体系

在 MLLMGuard 上使用 ASD（攻击成功程度↓）和 PAR（完美回答率↑）两个指标；在 VLSBench 上使用安全率 SR（安全拒绝 + 安全警告的比例↑）。

## 实验关键数据

### 主实验——MLLMGuard 动态评估

| 模型 | ASD↓(动态) | ASD(原始) | PAR↑(动态) | PAR(原始) |
|------|-----------|----------|-----------|----------|
| GPT-4o | 32.78 | 29.22 | 24.71 | 40.38 |
| Claude-4-Sonnet | 25.42 | 23.49 | 51.89 | 56.37 |
| InternVL-3-78B | 39.34 | 30.04 | 21.40 | 39.04 |
| Qwen-VL-2.5-7B | 40.17 | 29.46 | 33.96 | 44.04 |

### 消融实验——不同动态策略效果（InternVL-Chat-V1.5）

| 策略 | ASD↓ | ΔASD | PAR↑ | ΔPAR |
|------|------|------|------|------|
| 原始 | 32.21 | - | 40.19 | - |
| 词替换 | 38.71 | +6.30 | 26.94 | -13.25 |
| FigStep | 41.96 | +9.55 | 17.08 | -23.11 |
| 添加物体 | 39.41 | +7.00 | 26.45 | -13.74 |
| Text-to-Image | 35.10 | +2.69 | 24.36 | -15.83 |

### 能力评估影响（SDEval 对能力基准的影响）

| 模型 | MMVet(原始→动态) | MMBench(原始→动态) |
|------|-----------------|-------------------|
| GPT-4o | 68.8→67.5 (-1.3) | 83.4→81.8 (-1.6) |
| Qwen2.5VL-7B | 67.1→63.9 (-3.2) | 83.5→79.3 (-4.2) |

### 关键发现

- FigStep 是最有效的单一策略，ASD 上升约 10%，PAR 下降超 23%——表明视觉嵌入空间未与 LLM 的安全对齐
- 动态评估下所有 MLLM 安全率显著下降，说明模型更多是"记忆"安全答案而非真正理解不安全因素
- 安全性能与模型规模未呈现明显正相关——参数更多的模型可能因更好地理解指令反而更易执行有害请求
- SDEval 对能力基准的影响较小（下降 1-4 分），显示安全比能力更脆弱

## 亮点与洞察

1. **框架设计通用灵活**：三维度动态策略可组合应用于任意安全基准，且能与基准共存共演进
2. **数据泄露缓解有效**：生成式图像和文本变体与训练集重合度极低
3. **揭示安全-能力不平衡**：安全评估在动态扰动下波动远大于能力评估，暗示模型安全对齐深度不足
4. **FigStep 攻击效果惊人**：排版攻击直接绕过文本安全守卫，揭示视觉-语言安全对齐的根本缺陷

## 局限与展望

1. 动态策略依赖 GPT-4o 和 SD3.5 等外部模型，成本较高且引入额外偏差
2. 语义一致性验证依赖 GPT-4o 判断，可能存在漏检情况
3. 未深入分析不同策略组合的交互效应
4. 评估仅覆盖两个安全基准，未验证在更多场景（如毒性检测、偏见检测专项基准）上的泛化性

## 相关工作与启发

- 动态评估的思路可推广至其他 AI 安全领域（如 LLM 对齐评估、代码安全评估）
- FigStep 和 HADES 攻击的有效性表明，视觉-语言安全对齐是一个被严重忽视的安全盲区
- 安全与能力的不平衡发现为 AI 45° 定律提供了实证支持

## 评分

⭐⭐⭐⭐

- **新颖性** ⭐⭐⭐⭐：首个安全动态评估框架，三维度策略设计系统全面
- **实验充分度** ⭐⭐⭐⭐：覆盖 17 个 MLLM、2 个安全基准 + 2 个能力基准，消融详尽
- **写作质量** ⭐⭐⭐⭐：动机清晰，框架图易于理解
- **价值** ⭐⭐⭐⭐：为 MLLM 安全评估提供了可持续演进的方法论，对社区有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models](../../ICCV2025/multimodal_vlm/simplevqa_multimodal_factuality_evaluation_for_multimodal_large_language_models.md)
- [\[AAAI 2026\] VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models](vp-bench_a_comprehensive_benchmark_for_visual_prompting_in_m.md)
- [\[AAAI 2026\] SpeakerLM: End-to-End Versatile Speaker Diarization and Recognition with Multimodal Large Language Models](speakerlm_end-to-end_versatile_speaker_diarization_and_recognition_with_multimod.md)
- [\[NeurIPS 2025\] Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs](../../NeurIPS2025/multimodal_vlm/video-safetybench_a_benchmark_for_safety_evaluation_of_video_lvlms.md)
- [\[ICLR 2026\] Shuffle-R1: Efficient RL Framework for Multimodal Large Language Models via Data-centric Dynamic Shuffle](../../ICLR2026/multimodal_vlm/shuffle-r1_efficient_rl_framework_for_multimodal_large_language_models_via_data-.md)

</div>

<!-- RELATED:END -->
