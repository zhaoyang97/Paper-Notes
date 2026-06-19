---
title: >-
  [论文解读] MM-IFEngine: Towards Multimodal Instruction Following
description: >-
  [多模态VLM] 提出 MM-IFEngine 管线，系统性地生成高质量的图像-指令对数据（含 SFT 和 DPO 版本），并构建 MM-IFEval 基准，显著提升 MLLM 在多模态指令遵循任务上的表现。 多模态大语言模型（MLLM）在实际应用中需要精准地遵循用户给出的指令，例如以 JSON 格式输出、限定字数或包含关键…
tags:
  - "多模态VLM"
---

# MM-IFEngine: Towards Multimodal Instruction Following

- **会议**: ICCV 2025
- **arXiv**: [2504.07957](https://arxiv.org/abs/2504.07957)
- **领域**: 多模态VLM
- **关键词**: Instruction Following, MLLM, SFT, DPO, Benchmark, 约束生成, 多模态评估

## 一句话总结

提出 MM-IFEngine 管线，系统性地生成高质量的图像-指令对数据（含 SFT 和 DPO 版本），并构建 MM-IFEval 基准，显著提升 MLLM 在多模态指令遵循任务上的表现。

## 研究背景与动机

多模态大语言模型（MLLM）在实际应用中需要精准地遵循用户给出的指令，例如以 JSON 格式输出、限定字数或包含关键词等。然而，当前面临三大瓶颈：

**训练数据稀缺**：开源 MLLM 缺乏高质量的多模态指令遵循训练数据，模型在复杂约束下表现不足。

**现有基准过于简单**：MIA-Bench 等基准仅含简单原子指令（平均 2.6 个约束/题），且约束与视觉内容弱相关，导致模型准确率普遍超过 80%，无法区分优劣。

**评估策略不精确**：现有方法依赖 LLM-as-a-judge，但在字数统计、格式检查等需要精确验证的约束下，判断结果不可靠。

这三重限制使得 MLLM 的指令遵循能力提升陷入停滞，亟需从数据生成、基准构建和评估策略三方面同时突破。

## 方法详解

### 整体框架

MM-IFEngine 是一个端到端的图像-指令对生成流水线，分为三个阶段：

1. **图像筛选（Image Filter）**：从 CC3M、ALLaVA 等数据集中选取高质量图像，过滤低分辨率和语义贫乏的图片。对无标注的纯图像数据集使用 IC9600 和 RAM 指标筛选语义丰富的自然场景图像。
2. **任务生成（Task Generation）**：针对无 QA 对的图像，从预定义的 16 种任务描述池中采样，用 GPT-4o 为每张图生成适配的任务指令列表；对已有 QA 对的数据集（如 ALLaVA），用正则表达式和长度限制过滤含 few-shot 示例或选项的问题。
3. **约束整合（Constraints Integration）**：从 6 大类 32 子类的约束池中采样约束（文本长度、数学要求、格式、修辞逻辑、动作要求、关键词），由 LLM 生成具体约束内容并验证与任务指令的兼容性。

### 关键设计

**MM-IFInstruct-23k（SFT 数据集）**：
- 用 InternVL2.5-78B-MPO 生成响应，经后处理保留约束满足率 ≥ 80% 的样本
- 最终包含 23k 数据项，来源：CC3M 16k + ALLaVA 6k + MultiUI/Geo170k/ChartQA 4k
- 每个样本含 3-12 个约束，平均约束数远超现有数据集

**MM-IFDPO-23k（DPO 偏好数据集）**：
- 正样本直接使用高质量数据
- 负样本通过 Qwen2-VL-7B-Instruct 生成，设计四种设置：
    - 有图像但随机删除 1/3 约束
    - 有图像但随机删除 2/3 约束
    - 有图像但删除全部约束
    - 完整提示但无图像
- 消融实验表明删除 100% 约束的负样本效果最佳，因其最大化了正负样本间的语义差距

**MM-IFEval 基准**：
- 400 道题（300 Compose-Level + 100 Perception-Level）
- 32 种约束类别，平均每题 5.1 个约束
- Compose-Level：对输出格式、关键词等的组合约束
- Perception-Level：需要视觉感知能力，涵盖自然场景、UI 界面、图表、数学表达式

**混合评估策略**：
1. **规则验证（Rule-based）**：用预定义函数检查格式、字数等可精确验证的约束
2. **LLM 直接判断**：评估包含特定词汇等不需精确计数的约束
3. **LLM 比较判断**：对语气、风格等主观约束，生成有/无约束两版响应并比较

### 损失函数

- SFT 阶段：标准交叉熵损失
- DPO 阶段：标准 DPO 损失，KL 散度项保留模型的原始泛化能力

## 实验关键数据

### 主实验：指令遵循基准提升

| 模型 | MM-IFEval | MIA-Bench | IFEval | 平均 |
|------|-----------|-----------|--------|------|
| Qwen2-VL-7B 原始 | 42.0 | 80.5 | 47.4 | 56.6 |
| + MM-IFInstruct-23k (SFT) | 52.3 (+10.3) | 87.7 (+7.2) | 52.6 (+5.2) | 64.2 (+7.6) |
| + MM-IFDPO-23k (DPO) | **52.2 (+10.2)** | **88.1 (+7.6)** | **59.7 (+12.3)** | **66.7 (+10.1)** |
| LLaVA-NeXT-Llama3-8B 原始 | 39.7 | 83.3 | 50.7 | 57.9 |
| + MM-IFDPO-23k (DPO) | **49.3 (+9.6)** | **90.0 (+6.7)** | **69.1 (+18.4)** | **69.5 (+11.6)** |

### VQA 基准保持能力

| 模型 | MMMU | MMBench | MMStar | AI2D | OCRBench | 平均 |
|------|------|---------|--------|------|----------|------|
| Qwen2-VL-7B 原始 | 53.9 | 81.0 | 60.8 | 82.9 | 86.7 | 72.3 |
| + MM-IFDPO-23k | 54.0 | 81.3 | 58.5 | 83.3 | 86.8 | 72.4 |

DPO 训练后 VQA 性能几乎无损，得益于 KL 散度正则化。

### MM-IFEval 排行榜亮点

| 模型 | C-Level | P-Level | 平均 |
|------|---------|---------|------|
| GPT-4o | 71.5 | 44.0 | 64.6 |
| Qwen2-VL-72B | 53.4 | 43.0 | 50.8 |
| Qwen2-VL-7B + DPO | 55.2 | 43.0 | **52.2** |

7B 模型经 DPO 微调后超过原始 72B 模型，相对提升 24.3%。

### DPO 负样本策略消融

删除约束比例从 33% → 66% → 100% 逐步提升效果，表明拉大正负样本语义差距对 DPO 训练更有效。去掉图像的策略效果最弱。

## 亮点与洞察

1. **系统性解决方案**：同时解决数据、基准和评估三大痛点，形成完整闭环
2. **DPO 显著优于 SFT**：负样本通过移除约束构造，结合 KL 散度保留泛化能力，效果提升幅度翻倍
3. **小模型超越大模型**：7B 模型经 DPO 微调在 MM-IFEval 上超过 72B 原始模型，证明高质量指令遵循数据的价值
4. **Perception-Level 仍是难点**：即使 GPT-4o 在 P-Level 也仅 44.0，说明视觉约束理解远未解决
5. **约束粒度设计精妙**：6 大类 32 子类的分层约束体系兼顾覆盖度和可控性

## 局限性

1. Perception-Level 提升有限：DPO 微调主要改善 Compose-Level，P-Level 提升不明显
2. 数据生成依赖 GPT-4o，成本较高且可能引入其偏见
3. 基准规模较小（仅 400 题），统计显著性受限
4. 仅在 7-8B 规模模型上验证，未测试更大规模模型的增益

## 相关工作

- **LLM 指令遵循**：IFEval、CFBench、InFoBench 等文本指令遵循基准
- **多模态指令遵循**：MIA-Bench、VisIT-Bench 等多模态基准，但约束简单、评估粗糙
- **指令微调数据**：ShareGPT4V、ALLaVA 等合成数据集，但缺乏指令遵循专项数据

## 评分

- **创新性**: ⭐⭐⭐⭐ — 从数据生成、基准到评估策略的全栈创新，混合评估策略是亮点
- **实用性**: ⭐⭐⭐⭐⭐ — 数据集和评估工具已完全开源，可直接用于提升任意 MLLM
- **实验质量**: ⭐⭐⭐⭐ — 多基准全面验证，消融实验充分，但基准规模偏小
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，图表信息量大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)
- [\[ICCV 2025\] Hints of Prompt: Enhancing Visual Representation for Multimodal LLMs in Autonomous Driving](hints_of_prompt_enhancing_visual_representation_for_multimodal_llms_in_autonomou.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[ICCV 2025\] SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency](scan_bootstrapping_contrastive_pre-training_for_data_efficiency.md)
- [\[ICCV 2025\] SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning](sc-captioner_improving_image_captioning_with_self-correction_by_reinforcement_le.md)

</div>

<!-- RELATED:END -->
