---
title: >-
  [论文解读] OrderChain: Towards General Instruct-Tuning for Stimulating the Ordinal Understanding Ability of MLLM
description: >-
  [多模态VLM] 提出 OrderChain 提示范式，通过任务感知提示和范围优化思维链（RO-CoT）增强多模态大语言模型的序数理解能力，首次实现跨任务统一序数回归模型。 序数回归（Ordinal Regression）将实例分类到有序类别中，在面部年龄估计、图像美学评估、疾病分级等任务中至关重要…
tags:
  - "多模态VLM"
---

# OrderChain: Towards General Instruct-Tuning for Stimulating the Ordinal Understanding Ability of MLLM

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2504.04801](https://arxiv.org/abs/2504.04801)
- **代码**: [项目页面](https://order-chain.github.io/)
- **领域**: 多模态VLM
- **关键词**: 序数回归, 多模态大语言模型, 思维链, 渐进式推理, 统一模型

## 一句话总结

提出 OrderChain 提示范式，通过任务感知提示和范围优化思维链（RO-CoT）增强多模态大语言模型的序数理解能力，首次实现跨任务统一序数回归模型。

## 研究背景与动机

序数回归（Ordinal Regression）将实例分类到有序类别中，在面部年龄估计、图像美学评估、疾病分级等任务中至关重要。现有方法面临两个核心问题：

**传统方法需独立训练**：主流序数回归方法（序分布学习、实例比较、CLIP-based）虽然有效，但由于不同任务的类别数量和范围各异，仍需为每个任务训练独立模型。

**MLLM 序数理解能力差**：系统评估发现，无论是 zero-shot 还是 LoRA 微调基线，LLaVA 在多个序数回归数据集上表现远不及传统 SOTA。主要原因：(1) 缺乏特异性建模——忽略了领域知识和类别边界先验；(2) 缺乏共性建模——未能学习类别间有序关系这一序数回归的本质共性。

## 方法详解

### 整体框架

OrderChain 由三个核心组件构成：任务感知提示（Task-aware Prompts）、范围优化思维链（RO-CoT）和类别递归划分方法（CRD）。

### 范围优化思维链（RO-CoT）

利用序数回归类别连续有序的特性，采用由粗到细范式渐进预测。RO-CoT 四步流程：

1. **格式化和设置模板**：将原始查询格式化为通用序数回归任务模板
2. **生成任务定义作为领域知识提示**：MLLM 自身初步识别任务，生成定义和思路
3. **循环迭代**：每步中 CRD 生成更精细的候选子集 → 任务感知提示指令 MLLM 选择子集 → 判断终止或继续
4. **返回最终预测**：当子集仅含一个类别时结束

### 类别递归划分方法（CRD）

自动将全部候选类别递归划分为更小子集。使用 $k$ 叉平衡树：

$$T = \log_k(N_{init})$$

每步 $i$ 中每个子候选集的最大类别数为：

$$N_i = \frac{N_{init}}{k^i} + 1$$

第 $i$ 步第 $j$ 个候选集：$c_{i,j} = \{s_j, s_j+1, \ldots, \min(N_{init}, s_j + N_i - 1)\}$，起始索引 $s_j = (j-1) \times N_i + 1$。

对于类别较少的任务（如美学评估5类）使用二叉树，类别多的任务（如年龄估计80类）使用三叉树以减少 CoT 长度。

### 任务感知提示

**类别特征提示**包含两部分：
- **描述提示**：类别的范围和数量信息，克服传统模型全连接层维度固定的局限
- **指令提示**：当前步骤需精化的候选类别，由 CRD 方法自动生成

**领域知识提示**：由 MLLM 自身生成的领域先验知识，相当于对任务的初步识别，指导后续预测。

## 实验

### 面部年龄估计（Adience 数据集）

| 方法 | Accuracy (%) ↑ | MAE ↓ |
|------|---------------|-------|
| OrdinalCLIP | 61.2 | 0.47 |
| Ord2Seq | 63.9 | 0.43 |
| L2RCLIP | 66.2 | 0.36 |
| LLaVA-1.5 (zero-shot) | 17.6 | 1.48 |
| LLaVA-1.5 (baseline) | 47.5 | 0.59 |
| **LLaVA-1.5 + OrderChain** | **93.2** | **0.12** |

### 糖尿病视网膜分级（DR 数据集）

| 方法 | Accuracy (%) ↑ | MAE ↓ |
|------|---------------|-------|
| POE | 80.5 | 0.30 |
| Ord2Seq | 84.2 | 0.25 |
| LLaVA-1.5 (zero-shot) | 3.9 | 12.1 |
| LLaVA-1.5 (baseline) | 30.0 | 0.99 |
| **LLaVA-1.5 + OrderChain** | **85.7** | **0.23** |

### 历史图像年代判断（HCI 数据集）

| 方法 | Accuracy (%) ↑ | MAE ↓ |
|------|---------------|-------|
| NumCLIP | 69.6 | 0.35 |
| LLaVA-1.5 (baseline) | 61.4 | 0.50 |
| **LLaVA-1.5 + OrderChain** | **73.0** | **0.32** |

### 关键发现

1. **年龄估计里程碑提升**：93.2% 准确率较 SOTA (L2RCLIP 66.2%) 提升约 27 个百分点，MAE 从 0.36 降至 0.12
2. **不平衡数据鲁棒**：DR 数据集严重不平衡，OrderChain 通过 CRD 的平衡划分有效缓解正负样本失衡
3. **主观标签困难**：在图像美学评估上表现相对较弱，特别是 "People" 类别——推测 MLLM 预训练偏向积极评价人物
4. **消融验证**：领域知识提示贡献 +10.5% 准确率，类别特征提示进一步提升 +20%，RO-CoT 最终将准确率推至 93.2%

## 亮点与洞察

1. **首次探索 MLLM 序数回归**：系统性揭示 MLLM 在序数理解上的不足，并提供有效解决方案
2. **共性-特异性建模思路**：CRD 建模所有 OR 任务的共性（类别有序），任务感知提示建模不同任务的特异性
3. **即插即用设计**：不修改模型结构，仅通过提示工程增强现有 MLLM
4. **惊人提升**：年龄估计从 47.5% 到 93.2% 的跃升证明 CoT 思维对序数理解的关键作用

## 局限性

1. 在主观标签任务（如美学评估）效果有限，因为 MLLM 的预训练偏置难以消除
2. CRD 的划分叉数需根据任务手动选择
3. 多步 CoT 推理增加推理延迟，不适合实时应用

## 相关工作

- **序数回归**：SORD 软标签、MWR 排序、Ord2Seq 序列预测、OrdinalCLIP/L2RCLIP
- **思维链**：标准 CoT、面向检测和分割的定制化 CoT
- **MLLM**：LLaVA 系列、GPT-4V、Qwen-VL

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次将 CoT 范式应用于序数回归
- **技术深度**: ⭐⭐⭐ — 提示工程为主，无模型结构创新
- **实验充分性**: ⭐⭐⭐⭐⭐ — 4 个不同领域数据集，消融全面
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，示例丰富

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)
- [\[NeurIPS 2025\] In the Eye of MLLM: Benchmarking Egocentric Video Intent Understanding with Gaze-Guided Prompting](../../NeurIPS2025/multimodal_vlm/in_the_eye_of_mllm_benchmarking_egocentric_video_intent_understanding_with_gaze-.md)
- [\[ACL 2025\] Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](../../ACL2025/multimodal_vlm/table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)
- [\[ICCV 2025\] SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning](sc-captioner_improving_image_captioning_with_self-correction_by_reinforcement_le.md)
- [\[ICCV 2025\] SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency](scan_bootstrapping_contrastive_pre-training_for_data_efficiency.md)

</div>

<!-- RELATED:END -->
