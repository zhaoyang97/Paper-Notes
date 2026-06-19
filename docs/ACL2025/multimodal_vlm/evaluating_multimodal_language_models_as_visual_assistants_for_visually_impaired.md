---
title: >-
  [论文解读] Evaluating Multimodal Language Models as Visual Assistants for Visually Impaired Users
description: >-
  [ACL 2025][多模态VLM][视觉辅助] 通过用户调查确定视障人群对 AI 视觉助手的核心需求与挑战，设计涵盖图像描述、多语言VQA、光学盲文识别、视频物体识别、视频问答五大用户中心任务的评估框架，系统评测 12 个 MLLM，揭示当前模型在文化理解、多语言支持、盲文阅读、辅助设备识别和幻觉控制方面的显著不足。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "视觉辅助"
  - "视障用户"
  - "MLLM评估"
  - "盲文识别"
  - "文化敏感性"
  - "多语言VQA"
  - "视频理解"
---

# Evaluating Multimodal Language Models as Visual Assistants for Visually Impaired Users

**会议**: ACL 2025  
**arXiv**: [2503.22610](https://arxiv.org/abs/2503.22610)  
**代码**: [https://github.com/MalvinaNikandrou/visual-assistant-eval](https://github.com/MalvinaNikandrou/visual-assistant-eval)  
**作者**: Antonia Karamolegkou, Malvina Nikandrou, Georgios Pantazopoulos, Danae Sanchez Villegas, Phillip Rust, Ruchira Dhar, Daniel Hershcovich, Anders Søgaard  
**机构**: University of Copenhagen, Heriot-Watt University  
**领域**: 多模态大模型 / 无障碍辅助技术  
**关键词**: 视觉辅助, 视障用户, MLLM评估, 盲文识别, 文化敏感性, 多语言VQA, 视频理解

## 一句话总结

通过用户调查确定视障人群对 AI 视觉助手的核心需求与挑战，设计涵盖图像描述、多语言VQA、光学盲文识别、视频物体识别、视频问答五大用户中心任务的评估框架，系统评测 12 个 MLLM，揭示当前模型在文化理解、多语言支持、盲文阅读、辅助设备识别和幻觉控制方面的显著不足。

## 研究背景与动机

**领域现状**：MLLM（如 GPT-4V、Qwen2-VL）已被集成到视障辅助服务（Be My Eyes、Aira等），但现有评估基准主要面向一般性视觉推理（VQA、MMLU等），缺少对无障碍应用场景的针对性评估。

**核心挑战**：
   - 视障用户拍摄的图片/视频质量差（模糊、取景不正、光线问题）
   - 用户无法自行验证模型输出的正确性，幻觉（hallucination）问题尤为致命
   - 多语言和多文化需求未被现有评估覆盖
   - 盲文识别和辅助设备识别等特殊需求几乎被忽略

**研究动机**：基于真实视障用户的调查反馈，设计以用户需求为中心的评估框架，全面揭示 MLLM 在辅助视障人群方面的能力边界。

## 方法详解

### 整体框架：用户驱动的五任务评估

本文分为两大部分：**用户调查**（理解需求）和**系统评估**（量化能力）。

### 用户调查设计

- 招募 106 名不同程度视力障碍的参与者（通过 Prolific 平台）
- 两阶段调查：开放式问题（使用场景、挑战经验）+ Likert 量表评分
- **关键发现**：
    - **87%** 用户已使用或愿意使用 AI 视觉助手
    - 最常见用途：描述、转录、翻译、识别（购物识别商品、理解化学/数学图表、选衣服、解读面部表情）
    - 最大挑战（TF-IDF 分析）：不准确/幻觉、手写识别困难、多语言支持不足、空间理解弱

### 五大评估任务

**任务1：图像描述（Image Captioning）**
- 数据集：VizWiz-Captions（500张）+ 文化扩展版（324张，60种文化）
- 评估指标：RefCLIPScore
- 考察维度：通用描述能力 vs 文化敏感描述能力

**任务2：图像问答（Image QA）**
- 数据集：VizWiz VQA 验证集 + 自建多语言扩展（34种语言）
- 翻译流程：自动翻译 + 人工质量检查
- 评估指标：VQA Accuracy

**任务3：光学盲文识别（Optical Braille Recognition）**
- 全新任务，贡献两个新数据集：
    - 句子级盲文转文字：10万句训练 + NTREX-128/FLORES-200 评估
    - 段落级跨脚本问答：SQuAD 改编（13万训练 + 1.19万评估）
- 盲文文本渲染为图像，并施加质量增强（模拟视障用户拍照缺陷）
- 评估指标：chrF++（转录）、F1/EM（问答）

**任务4：视频物体识别**
- 数据集：ORBIT（1036个视频片段，92类物体，含辅助设备类别）
- 区分通用物体和辅助物体（如盲文显示器、拐杖）
- 评估指标：LAVE 协议（LLM-as-Judge，1-3分）

**任务5：视频问答**
- 自建数据集：98个视频、882个问答对
- 三类问题：描述性（属性）、空间理解（位置关系）、对抗性（不存在物体）
- 对抗性问题测试模型是否会幻觉回答
- 评估指标：LAVE 协议

### 评估模型

12 个主流 MLLM，包括 Qwen2-VL、InternVL2.5、LLaVA-v1.6、MiniCPM-V-2.6、PaliGemma、Phi-3.5-Vision 等。

## 实验关键数据

### 图像描述

| 模型 | 原始VizWiz | 文化扩展版 |
|------|-----------|----------|
| PaliGemma | **81.0** | 55.0 |
| MiniCPM-V-2.6 | 78.0 | 74.8 |
| Qwen2-VL | 75.9 | **76.9** |
| LLaVA-v1.6 | 72.3 | 52.2 |

- 5/9 模型在文化场景下性能大幅下降（20-25分）
- 即使最好的模型，也有约 1/3 描述遗漏文化关键细节

### 图像问答

| 模型 | 英文 | 多语言 |
|------|------|--------|
| PaliGemma | **75.6** | 16.9 |
| MiniCPM-V-2.6 | 72.2 | 30.7 |
| Qwen2-VL | 61.9 | **44.9** |

- 在预训练中包含 VizWiz 数据的模型（PaliGemma、MiniCPM）英文表现最好，但多语言降幅最大
- Qwen2-VL 多语言表现最稳定（35.4-49.0 跨语言波动小）
- 高/中/低资源语言间差异**不大**，说明即使高资源语言也缺乏可靠支持

### 盲文识别

| 模型 | chrF++（零样本） |
|------|---------------|
| Qwen2-VL | **73.8** |
| Phi-3-Vision | 9.9 |
| 其他所有模型 | < 9.1 |

- **只有 Qwen2-VL 展现出非平凡的盲文理解能力**，其他模型几乎完全无法识别盲文
- LoRA 微调 Llama-3.2-Vision 后可达 88.2 chrF++，证明学习盲文阅读是可行的，3万样本即可饱和

### 视频物体识别

| 模型 | 通用物体 | 辅助设备 |
|------|---------|---------|
| Qwen2-VL | **69.8%** | 39.7% |
| MiniCPM-V-2.6 | 65.1% | **44.2%** |
| LLaVA-Video | 65.7% | 41.3% |

- 辅助设备识别率（20-44%）远低于通用物体（52-70%），差距明显

### 视频问答

| 模型 | 描述性 | 空间 | 对抗性 | 平均 |
|------|--------|------|--------|------|
| LLaVA-Video | **78.2** | 63.4 | 7.7 | 49.8 |
| MiniCPM-V-2.6 | 68.7 | 63.3 | **17.7** | **49.9** |
| VideoChat-Flash | 72.4 | **64.1** | 9.2 | 48.6 |

- 对抗性问题（不存在物体的问题）准确率极低（7-18%），模型倾向于编造回答而非承认不确定性
- 即使明确提示可以回答"不确定"，改善也有限

## 亮点与洞察

1. **用户驱动的评估设计**：从 106 名视障用户的真实需求出发设计评估，确保了任务的实际意义
2. **首创盲文识别基准**：提出句子级和段落级两个盲文识别任务及数据集，填补了 MLLM 评估的重大空白
3. **"幻觉是最大敌人"**：对于无法自行验证输出的视障用户，模型的幻觉问题比一般用户场景更加危险
4. **文化盲区暴露**：即使最好的模型也有 1/3 描述缺少文化细节——这对需要理解多文化环境的视障旅行者是真实障碍
5. **盲文识别的可行性**：虽然现有模型几乎都不能读盲文，但微调实验证明只需中等规模数据（3万条）即可学会，为下一代模型提供了明确方向

## 局限性

1. **缺少导航辅助任务**：导航是视障用户的核心需求，但本文未涉及
2. **受控环境评估**：未完全捕捉动态真实场景的复杂性（如实时响应、移动中的视频理解）
3. **模型覆盖**：仅评测 12 个，未包含 GPT-4V/4o 等闭源领先模型
4. **多语言翻译质量**：使用自动翻译 + 人工检查，低资源语言翻译质量可能不足
5. **缺少用户交互评估**：评估基于离线基准，未评估多轮对话/交互引导场景

## 相关工作

- **MLLM 评估基准**：MMLU、MME、VQAv2 等面向通用能力，Lee et al. 2024 的全面评估发现无模型全领域领先
- **视障辅助应用**：VizWiz 系列（Gurari et al. 2018/2020）、ORBIT 数据集（Massiceti et al. 2021）
- **文化敏感评估**：Karamolegkou et al. 2024 发现 VizWiz 中存在被忽视的文化含义

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性**：⭐⭐⭐⭐ 盲文识别任务全新，用户调查驱动评估设计有说服力
- **实验充分性**：⭐⭐⭐⭐⭐ 五大任务覆盖全面，12 模型横向对比
- **写作质量**：⭐⭐⭐⭐ 结构清晰，从需求到评估逻辑通顺
- **实用性**：⭐⭐⭐⭐⭐ 直接指导下一代视觉辅助技术的开发方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Evaluating Multimodal Large Language Models on Video Captioning via Monte Carlo Tree Search](mcts_video_captioning_eval.md)
- [\[ACL 2025\] AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models](alignmmbench_evaluating_chinese_multimodal_alignment_in_large_vision-language_mo.md)
- [\[ICCV 2025\] Visual Intention Grounding for Egocentric Assistants](../../ICCV2025/multimodal_vlm/visual_intention_grounding_for_egocentric_assistants.md)
- [\[ACL 2025\] Aligning VLM Assistants with Personalized Situated Cognition](aligning_vlm_assistants_with_personalized_situated.md)
- [\[ACL 2025\] Unsolvable Problem Detection: Evaluating Trustworthiness of Large Multimodal Models](unsolvable_problem_detection.md)

</div>

<!-- RELATED:END -->
