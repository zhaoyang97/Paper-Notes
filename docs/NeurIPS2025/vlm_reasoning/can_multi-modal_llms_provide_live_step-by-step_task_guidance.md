---
title: >-
  [论文解读] Can Multi-Modal LLMs Provide Live Step-by-Step Task Guidance?
description: >-
  [NeurIPS 2025][多模态VLM][流式视频理解] 提出 Qualcomm Interactive Cooking 基准和 LiveMamba 模型，首次系统评估多模态 LLM 在实时流式视频中提供分步任务指导（包括指令下发、完成检测和错误反馈）的能力。 当前多模态 LLM 虽然具备强大的对话能力…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "流式视频理解"
  - "交互式指导"
  - "错误检测"
  - "Mamba"
  - "步骤引导"
---

# Can Multi-Modal LLMs Provide Live Step-by-Step Task Guidance?

**会议**: NeurIPS 2025  
**arXiv**: [2511.21998](https://arxiv.org/abs/2511.21998)  
**代码**: [GitHub](https://github.com/qualcomm-ai-research/livemamba)  
**领域**: 多模态VLM  
**关键词**: 流式视频理解, 交互式指导, 错误检测, Mamba, 步骤引导

## 一句话总结

提出 Qualcomm Interactive Cooking 基准和 LiveMamba 模型，首次系统评估多模态 LLM 在实时流式视频中提供分步任务指导（包括指令下发、完成检测和错误反馈）的能力。

## 研究背景与动机

当前多模态 LLM 虽然具备强大的对话能力，但大多局限于 **轮次制交互**（turn-based），即只在用户提问时才产生回复。然而，一个真正有用的 AI 助手需要能够在视频流中 **异步反应**：

**主动提供下一步指令**：在用户完成当前步骤后自动给出下一步操作

**检测指令完成**：判断用户是否成功执行了当前指令

**识别并报告错误**：在用户犯错时尽快发出警告

现有数据集（Epic-Kitchens、Ego4D、HowTo100M 等）主要记录专家操作或日常活动，**缺少用户犯错场景**，无法有效评估模型的交互式指导能力。作者利用 CaptainCook4D 数据集（包含用户错误操作）构建了首个包含精确时间戳的指令与反馈标注的基准。

## 方法详解

### 整体框架

LiveMamba 是一个轻量级流式多模态 LLM，由视觉编码器和语言模型骨干组成：

- **视觉编码器**: InternViT-300M-448px 提取每帧 $M$ 个视觉 token
- **Q-Former 适配器**: 通过 4 层交叉注意力将 $M$ 个 token 压缩为 $K$ 个 token
- **语言骨干**: Mamba-130M 循环模型，支持高效长序列推理
- **重规划模块**: 外部 Qwen3-32B 模型，在用户偏离计划时重新规划指令顺序

### 关键设计

**"When-to-Say" 机制**: 使用两个特殊 token:

- `<vision>`: 请求下一帧视频输入
- `<response>`: 在适当时机产生指令或反馈

模型在每帧输入后自主决定是继续观察还是发出回复，无需外部提示。

**迭代式重规划**: 当用户跳步或乱序执行时，LiveMamba 触发外部重规划器。重规划器接收初始计划、已完成步骤和反馈，选择最优的下一步指令。

### 数据增强策略

**指令完成增强 (ICAug)**: 将 Epic-Kitchens 和 Ego4D 的动作描述转换为指令-反馈格式，在动作起始时给出指令，结束时确认完成。

**反事实错误增强 (CFAug)**: 生成"可信的反事实错误"——将动作描述修改为合理的错误操作（如把"加 1 茶匙盐"改为"加 1 汤匙盐"），构造错误场景训练数据。

**时间抖动**: 对每条指令的起始时间戳加入 $\pm 30$ 秒的随机扰动，防止自回归模型的误差累积。

### 损失函数

模型使用标准自回归语言建模损失进行训练。预训练阶段仅训练 Q-Former，使视觉嵌入与文本嵌入对齐；微调阶段同时训练 Q-Former 和 Mamba 语言骨干。

## 实验关键数据

### 主实验

**零样本评估（流式模式，Main Set）**:

| 模型 | IC-Acc ↑ | Prec. ↑ | Rec. ↑ | F1 ↑ | BERT ↑ | ROUGE-L ↑ |
|------|----------|---------|--------|------|--------|-----------|
| Gemini-2.5-Flash | 23.1 | 0.01 | 0.22 | 0.02 | 0.410 | 0.342 |
| Qwen2.5-VL-7B | 18.9 | 0.18 | 0.01 | 0.02 | 0.299 | 0.219 |
| VideoLLaMA3-7B | 1.8 | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 |

**微调评估（流式模式，Main Set）**:

| 模型 | IC-Acc ↑ | Prec. ↑ | Rec. ↑ | F1 ↑ | BERT ↑ | ROUGE-L ↑ |
|------|----------|---------|--------|------|--------|-----------|
| LiveMamba (完整) | **31.5** | **0.17** | **0.10** | **0.13** | **0.651** | **0.561** |
| LiveMamba (w/o-CFAug) | 14.3 | 0.12 | 0.03 | 0.05 | 0.558 | 0.511 |
| LiveMamba (w/o-ICAug) | 7.8 | 0.05 | 0.01 | 0.01 | 0.605 | 0.542 |
| Videollm-online† | 7.6 | 0.04 | 0.01 | 0.01 | 0.434 | 0.412 |

**轮次制评估（Main Set）**:

| 模型 | IC-Acc ↑ | F1 ↑ | BERT ↑ | ROUGE-L ↑ |
|------|----------|------|--------|-----------|
| LiveMamba† | **51.0** | **0.19** | **0.631** | **0.535** |
| Qwen2.5-VL-7B | 38.9 | 0.06 | 0.348 | 0.230 |
| Qwen2-VL-7B | 19.4 | 0.11 | 0.398 | 0.293 |

### 消融实验

| 组件 | IC-Acc | 错误 F1 |
|------|--------|---------|
| 完整 LiveMamba | 31.5 | 0.13 |
| 去除指令完成增强 | 7.8 → 14.3 | 0.01 → 0.05 |
| 去除反事实增强 | 14.3 | 0.05 |
| 去除重规划 (Adv Set) | 10.9 | 0.16 |
| 带重规划 (Adv Set) | 12.6 | 0.19 |

### 关键发现

1. 所有零样本 MLLM 在流式交互指导任务上表现极差，最好的 Gemini-2.5-Flash 仅 23.1% IC-Acc
2. 反事实错误增强将错误 F1 从 0.05 提升至 0.13，说明高质量错误数据至关重要
3. LiveMamba 使用 Mamba-130M 骨干，实时处理速度为输入速度的 4 倍（8.1 fps vs 2 fps），延迟仅 1.1 秒

## 亮点与洞察

- **首个实时交互指导基准**: Qualcomm Interactive Cooking 填补了流式视频中分步指导评估的空白，包含 94 小时密集标注数据
- **轻量高效架构**: Mamba-130M 骨干使模型适合边缘设备部署（手机、智能眼镜）
- **反事实增强策略**: 通过自动生成可信错误场景解决错误训练数据稀缺问题
- **流式 + 轮次双评估**: 提供全面的评价视角，流式评估反映真实场景，轮次评估便于追踪单步进展

## 局限性

- 仅关注烹饪领域，泛化到其他任务场景尚未验证
- 细粒度错误检测仍然极具挑战（如区分 1 茶匙 vs 1 汤匙）
- Advanced Planning Set 上的指令完成准确率仍然很低（12.6%），复杂计划推理能力有待提高
- 重规划依赖外部大模型（Qwen3-32B），平均耗时 6.1 秒，影响实时性

## 相关工作与启发

- **VideoLLM-online**: 首个在线视频对话框架，但仅支持叙述而非交互指导
- **CaptainCook4D**: 提供包含用户错误的烹饪视频，为构建交互式基准提供了基础
- **Mamba 架构**: 循环模型在长序列上的高效推理优势在流式视频场景中得到了充分体现
- 启发：流式交互式 AI 助手需要"主动发言"能力，这与传统问答范式有本质区别

## 评分

- ⭐ 新颖性: 4/5 — 首次系统定义了实时交互指导任务，数据集和评估框架设计完善
- ⭐ 实验充分度: 4/5 — 零样本、微调、消融、轮次制多角度评估，但仅限单一领域
- ⭐ 写作质量: 4/5 — 问题定义清晰，实验组织有条理
- ⭐ 价值: 4/5 — 开辟了流式交互指导的新研究方向，基准具有长期价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Training-free Online Video Step Grounding](training-free_online_video_step_grounding.md)
- [\[ECCV 2024\] m&m's: A Benchmark to Evaluate Tool-Use for Multi-step Multi-modal Tasks](../../ECCV2024/multimodal_vlm/m_ampmaposs_a_benchmark_to_evaluate_tool-use_for_multi-step_multi-modal_tasks.md)
- [\[NeurIPS 2025\] Unveiling Chain of Step Reasoning for Vision-Language Models with Fine-grained Rewards](unveiling_chain_of_step_reasoning_for_visionlanguage_models.md)
- [\[ICCV 2025\] LLaVA-CoT: Let Vision Language Models Reason Step-by-Step](../../ICCV2025/multimodal_vlm/llava-cot_let_vision_language_models_reason_step-by-step.md)
- [\[ICCV 2025\] ToolVQA: A Dataset for Multi-step Reasoning VQA with External Tools](../../ICCV2025/multimodal_vlm/toolvqa_a_dataset_for_multistep_reasoning_vqa_with_external.md)

</div>

<!-- RELATED:END -->
