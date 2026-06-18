---
title: >-
  [论文解读] MM-OPERA: Benchmarking Open-ended Association Reasoning for Large Vision-Language Models
description: >-
  [NeurIPS 2025][多模态VLM][association reasoning] 提出 MM-OPERA，一个包含 11,497 实例的开放式联想推理基准，通过远程物品关联（RIA）和上下文关联（ICA）两大任务评估 LVLM 的关联推理能力，配套设计了 LLM-as-a-Judge 评分策略和过程奖励评估方法，揭示当前最强 LVLM 仍显著落后于人类。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "association reasoning"
  - "open-ended evaluation"
  - "LLM-as-a-Judge"
  - "process reward"
  - "divergent thinking"
  - "convergent thinking"
---

# MM-OPERA: Benchmarking Open-ended Association Reasoning for Large Vision-Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.26937](https://arxiv.org/abs/2510.26937)  
**代码**: [https://github.com/MM-OPERA-Bench/MM-OPERA](https://github.com/MM-OPERA-Bench/MM-OPERA)  
**领域**: 多模态VLM / 关联推理 / 评估基准  
**关键词**: association reasoning, open-ended evaluation, LLM-as-a-Judge, process reward, divergent thinking, convergent thinking  

## 一句话总结
提出 MM-OPERA，一个包含 11,497 实例的开放式联想推理基准，通过远程物品关联（RIA）和上下文关联（ICA）两大任务评估 LVLM 的关联推理能力，配套设计了 LLM-as-a-Judge 评分策略和过程奖励评估方法，揭示当前最强 LVLM 仍显著落后于人类。

## 研究背景与动机

**领域现状**：LVLM 在视觉理解、语言生成、多步推理方面进展显著，但**关联智能**（association）——人类创造性思维和知识整合的基石——评估严重不足。

**现有局限**：此前工作 "Labyrinth of Links" 仅用封闭式选择题评估关联记忆，(1) 固定选项可能暗示答案、掩盖模型真实能力; (2) 无法评估复杂的多步关联推理和发散思维。

**核心动机**：开放式关联推理对科学发现、创意设计、个性化教育、创新问题解决等实际应用至关重要。需要一个无预定义约束的benchmark来严格评估 LVLM 的关联推理能力。

**认知科学基础**：关联来自收敛思维（找到最优联系）和发散思维（生成多个独特想法）的交互。远程联想测试（RAT）是经典工具，但仅测量单跳收敛思维。MM-OPERA 扩展到多步推理结构。

## 方法详解

### 任务设计

**1. Remote-Item Association (RIA)**：
- 给定两个看似无关的元素（可以是图像、文本或混合模态），模型需发现它们之间有意义的联系
- 例：犰狳图片 + 凯夫拉织物图片 → 共享的"防护功能"
- 鼓励跨域推理，允许多个有效联想路径

**2. In-Context Association (ICA)**：
- 在 RIA 基础上扩展到上下文学习：先理解一对元素的关联模式，再将该模式迁移到新元素上
- 例：白头鹰↔篮球（美国国家象征↔起源运动）→ 狮子→？（英国象征→足球）
- 测试模式抽象和跨域迁移能力

### 数据集统计
- **总量**：11,497 实例（RIA 8,021 + ICA 3,476）
- **层级能力分类**：3层（L-1 感知/概念，L-2 六维，L-3 十三维）
- **3 种关系类型**：关系（Relation）、共同元素（Mutual Element）、隐喻（Metaphor）
- **关联推理路径**：以有向路径表示推理过程，hop 数反映复杂度
- **多样性**：15 种语言、22 个主题领域、多文化背景

### LLM-as-a-Judge 评估策略

**常规评分**（Holistic Score, 0-4 分）：
- 4 分：准确、逻辑一致、有洞察力，匹配参考答案水平
- 3 分：理解合理但缺乏关键洞察或完整性
- 2 分：有一定相关性但缺乏深度
- 1 分：模糊、不确定或不完整
- 0 分：包含事实错误

**评估指标**：
- Score Rate (SR)：平均分百分比
- High Score Rate (HR-4)：得分为 4 的比例
- HR-3：得分 ≥3 的比例
- $\triangle$HR = HR-3 − HR-4：反映**发散思维**能力

**过程奖励评估（PR-Judge）**：
将模型回答重构为关联路径 $P = (s_1, s_2, \ldots, s_n)$，对每一步评估三个维度：
- **合理性** $R_t \in [0,1]$：推理流畅度和逻辑连贯性
- **独特性** $D_t \in [0,1]$：概念边界的清晰度
- **知识性** $K_t \in \{0,1\}$：是否体现领域知识

每步关联质量：$s_t = \alpha R_t D_t + (1-\alpha) K_t$

推理总分：$S_r = \sum_{t=1}^{n} s_t \delta^t$（$\delta$ 为认知衰减因子，偏好高效推理路径）

## 实验关键数据

### RIA 任务（部分代表性模型）

| 模型 | SR(%) | HR-4(%) | HR-3(%) | △HR(%) |
|------|-------|---------|---------|--------|
| Gemini-2.5-Pro-Preview | 60.05 | 23.89 | 41.75 | 17.86 |
| o4-mini | 60.33 | 19.86 | 37.89 | 18.03 |
| GPT-4o | 59.72 | 10.89 | 28.83 | 17.94 |
| Gemini-2.0-Flash-Thinking | 59.11 | 17.73 | 36.60 | 18.87 |
| Qwen2.5-VL-7B | 52.28 | 5.35 | 20.36 | 15.00 |
| **Human** | **61.88** | **22.84** | **48.97** | **26.13** |

### ICA 任务

| 模型 | SR(%) | HR-4(%) | HR-3(%) | △HR(%) |
|------|-------|---------|---------|--------|
| Gemini-2.5-Pro-Preview | 63.09 | 12.85 | 41.15 | 28.30 |
| o4-mini | 61.55 | 10.24 | 36.60 | 26.36 |
| GPT-4o | 58.26 | 6.27 | 29.62 | 23.35 |
| **Human** | **68.69** | **31.65** | **61.47** | **29.82** |

### 关键发现
1. **LVLM 显著落后人类**：ICA 上最强模型 HR-4 仅 12.85% vs 人类 31.65%
2. **创造力差距**：模型 △HR 约 12%-20%，人类 26%-30%，发散思维能力差距大
3. **ICA 比 RIA 难**：大多数模型 ICA 分数更低，模式抽象和迁移是更大挑战
4. **保守推理 vs 关联灵活性**：Gemini-1.5-Pro（保守型）反而不如 Flash（快速型），过度的事实检查和伦理考虑限制了创造性关联
5. **过程评估**：模型在合理性(50%-80%)上尚可，但在独特性(不到一半>75%)上严重不足

## 亮点
- ⭐⭐⭐⭐ **填补评估空白**：首个大规模开放式关联推理基准，认知科学基础扎实
- ⭐⭐⭐⭐ **评估方法创新**：PR-Judge 过程奖励评估能区分"殊途同归"的不同推理路径质量
- ⭐⭐⭐⭐ **发现深刻**：保守推理 vs 关联灵活性的权衡、独特性瓶颈等洞察对模型改进有指导意义
- ⭐⭐⭐ **多维度分析**：敏感性测试（图像替换、文本替换、顺序敏感性）+ 裁判验证 + 多样性分析

## 局限与展望
1. 参考答案仅作为启发式基准，开放式评估仍依赖 LLM-as-a-Judge 的可靠性
2. 人类基线基于大学生样本，可能不完全代表普遍人类关联推理水平
3. $\alpha=0.9, \delta=0.9$ 的超参数选择缺乏系统消融
4. 目前仅评估关联推理，未探索如何利用 benchmark 发现来实际改进模型的关联能力
5. 部分文化和语言特定的关联可能对非本土模型不公平

## 总评
⭐⭐⭐⭐ 一项有深度和广度的 benchmark 工作，将认知心理学的关联推理理论系统地引入 LVLM 评估。任务设计严谨、评估方法多层次、实验分析透彻。揭示了当前 LVLM 在创造性思维和知识整合方面的重要短板，为未来模型发展指明方向。

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] CHOICE: Benchmarking the Remote Sensing Capabilities of Large Vision-Language Models](choice_benchmarking_the_remote_sensing_capabilities_of_large_vision-language_mod.md)
- [\[ECCV 2024\] Towards Open-ended Visual Quality Comparison](../../ECCV2024/multimodal_vlm/towards_open-ended_visual_quality_comparison.md)
- [\[NeurIPS 2025\] Recognition through Reasoning: Reinforcing Image Geo-localization with Large Vision-Language Models](recognition_through_reasoning_reinforcing_image_geo-localization_with_large_visi.md)
- [\[ACL 2025\] Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning](../../ACL2025/multimodal_vlm/benchmarking_and_improving_large_vision-language_models_for_fundamental_visual_g.md)
- [\[ICCV 2025\] IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves](../../ICCV2025/multimodal_vlm/ideator_jailbreaking_and_benchmarking_large_visionlanguage_m.md)

</div>

<!-- RELATED:END -->
