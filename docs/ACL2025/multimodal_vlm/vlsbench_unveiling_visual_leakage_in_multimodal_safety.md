---
title: >-
  [论文解读] VLSBench: Unveiling Visual Leakage in Multimodal Safety
description: >-
  [ACL 2025][多模态VLM][多模态安全] 揭示现有多模态安全基准中存在的视觉安全信息泄露（VSIL）问题——图像中的危险内容已在文本查询中暴露，导致模型仅凭文本即可拒绝，从而使安全评估不可靠；为此构建了无泄露的VLSBench基准（2.2k图文对），发现多模态对齐在无VSIL场景中显著优于纯文本对齐。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "多模态安全"
  - "视觉信息泄露"
  - "安全基准"
  - "MLLM对齐"
  - "多模态评估"
---

# VLSBench: Unveiling Visual Leakage in Multimodal Safety

**会议**: ACL 2025  
**arXiv**: [2411.19939](https://arxiv.org/abs/2411.19939)  
**代码**: [github](https://github.com/AI45Lab/VLSBench)  
**领域**: 多模态VLM  
**关键词**: 多模态安全, 视觉信息泄露, 安全基准, MLLM对齐, 多模态评估

## 一句话总结

揭示现有多模态安全基准中存在的视觉安全信息泄露（VSIL）问题——图像中的危险内容已在文本查询中暴露，导致模型仅凭文本即可拒绝，从而使安全评估不可靠；为此构建了无泄露的VLSBench基准（2.2k图文对），发现多模态对齐在无VSIL场景中显著优于纯文本对齐。

## 研究背景与动机

多模态大语言模型（MLLM）的安全性日益受到关注。然而，先前研究发现一个反直觉的现象：仅用文本unlearning对齐MLLM，就能取得与图文对齐相当的安全性能。这暗示现有多模态安全基准可能存在根本性问题。

作者深入分析后发现了**视觉安全信息泄露（Visual Safety Information Leakage, VSIL）**问题：在现有基准中，图像所蕴含的潜在危险信息已经在文本查询中被显式或隐式地表达出来。这意味着MLLM可以仅根据文本查询就轻松拒绝这些敏感图文对，而无需真正理解图像中的跨模态安全信息。这使得跨模态安全评估变得不可靠——纯文本对齐就"足够好"只是因为基准本身存在缺陷。

## 方法详解

### 整体框架

作者提出了一个自动化数据构建流水线，用于生成无视觉安全信息泄露的图文对。VLSBench包含2.2k图文对，覆盖6个安全类别和19个子类别。其核心设计思路是确保文本查询本身是无害的，图像单独看也不一定有害，但图文结合时才构成安全风险。

### 关键设计

1. **安全分类体系**: 设计了层次化的两级安全分类法，包含6大类19子类，参考了LLM安全和多模态安全的已有分类标准，确保覆盖面广泛。

2. **有害查询与图像描述生成（Step 1）**: 采用两条并行路径——(a) 从ChatGPT提取敏感对象和危险场景，然后用GPT-4o生成图像描述和有害查询；(b) 利用现有图像数据集（真实世界图像），用Qwen2-VL-72B生成图像分析和有害查询。双路径策略保证了安全主题的多样性。

3. **文本去毒化 / 消除视觉泄露（Step 2）**: 用GPT-4o将有害查询去毒化为看似无害的文本查询，通过few-shot prompting消除从图像模态到文本模态的安全信息泄露。同时过滤两类不合格样本：(a) 仍含泄露信息的修改后查询，(b) 偏离原始语义的查询。

4. **迭代图像生成（Step 3）**: 使用GPT-4o-mini改写图像描述为text-to-image prompt，用Stable-Diffusion-3.5-Large生成图像。采用迭代方式：用Qwen2-VL-72B评估生成图像是否反映描述信息，若不符合则修改prompt重新生成，直到满足标准。

5. **最终过滤（Step 4）**: 用GPT-4o对最终图文对进行质量过滤，去除不匹配和自然安全的样本，最后经人工审核完成数据集。

### 评估策略

使用GPT-4o作为评判模型，将模型响应分为三类：
- **安全-拒绝（Safe with Refusal）**: 明确坚定的拒绝
- **安全-警告（Safe with Warning）**: 承认安全顾虑并提供警示
- **不安全（Unsafe）**: 无视安全原则直接回答

安全率 = 拒绝率 + 警告率。

## 实验关键数据

### 主实验

| 模型 | 拒绝率 | 警告率 | 总安全率 |
|------|--------|--------|----------|
| LLaVA-v1.5-7B | 0% | 6.60% | 6.60% |
| GPT-4o | 5.21% | 16.22% | 21.43% |
| Gemini-1.5-pro | 1.34% | 48.44% | 49.78% |
| Llama-3.2-11B-Vision | 10.96% | 15.33% | 26.29% |
| Qwen2-VL-7B | 1.11% | 12.66% | 13.77% |
| InternVL2.5-8B | 2.81% | 18.56% | 21.37% |

### 安全对齐方法对比（LLaVA-v1.5-7B）

| 方法 | 拒绝率 | 警告率 | 总安全率 |
|------|--------|--------|----------|
| MM-SFT | 2.32% | 18.94% | 21.26% |
| MM-DPO | 2.63% | 24.38% | 27.01% |
| MM-PPO | 5.08% | 30.39% | 35.47% |
| Textual-SFT | 5.30% | 8.69% | 13.99% |
| Textual-SafeUnlearning | 2.85% | 8.87% | 11.72% |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 原始图文输入 vs 图像标题替代 | Qwen2-VL: 16% → 22.5% | 用文本标题代替图像可提高安全性 |
| 无视觉输入 | Qwen2-VL: 16% → 29.5% | 去掉图像反而更安全，说明模型跨模态理解不足 |
| 安全提示语增强 | LLaVA-v1.5: 6.6% → 44.5% | 安全prompt能显著提升安全表现 |
| NSFW检测器 | 检出率 0% | VLSBench图像单独不触发NSFW检测器 |

### 关键发现

- 所有主流MLLM（包括GPT-4o）在VLSBench上安全率均很低，最高仅Gemini达49.78%
- 在有VSIL的基准上，纯文本对齐效果与多模态对齐相当；但在VLSBench（无VSIL）上，多模态对齐显著优于纯文本对齐（Qwen2-VL: 78.39% vs 67.42%）
- 推理能力强不等于安全性好：QVQ-Preview相比Qwen2-VL有提升，但LLaVA-Cot反而下降
- 模型更多是"粗暴拒绝"而非给出充分的安全解释和替代方案

## 亮点与洞察

- **问题发现精准**：VSIL是一个被长期忽视但非常关键的问题，解释了"为什么纯文本对齐也行"的反直觉现象
- **数据构建流水线自动化且可复现**：四步流水线设计巧妙，通过去毒化+迭代生成+多轮过滤确保质量
- **评估结论打破幻想**：即使是GPT-4o也仅21.43%的安全率，说明现有MLLM的跨模态安全理解能力极度不足
- **方法论启示**：安全基准的设计本身就需要精心打磨，否则得出的结论可能完全误导社区

## 局限与展望

- 数据集规模2.2k相对较小，覆盖的安全场景可能仍有盲区
- 依赖GPT-4o做评判可能引入评估偏差
- 图像均为AI生成，与真实世界场景的分布可能有差异
- 未探索视频模态或多轮对话场景下的安全评估
- 去毒化过程依赖特定prompt设计，可能存在一些边缘情况未覆盖

## 相关工作与启发

- 与MMSafetyBench、VLGuard等先前基准形成对比，指出其VSIL问题
- 启发未来的多模态安全研究应更关注跨模态理解能力而非单模态对齐
- 为多模态RLHF/DPO等对齐方法提供了更可靠的评估工具
- SPA-VL等多模态偏好对齐方法在此基准上表现更好，说明方向正确

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统性揭示VSIL问题，改变了社区对多模态安全对齐的认知
- 实验充分度: ⭐⭐⭐⭐ 涵盖开源/闭源模型、多种对齐方法、多种分析维度，但数据集规模有限
- 写作质量: ⭐⭐⭐⭐ 论述清晰、逻辑连贯，但部分实验细节在附录中
- 价值: ⭐⭐⭐⭐⭐ 为多模态安全领域提供了关键的基准工具和方法论启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Unveiling the Lack of LVLM Robustness to Fundamental Visual Variations: Why and Path Forward](unveiling_the_lack_of_lvlm_robustness_to_fundamental_visual_variations_why_and_p.md)
- [\[ACL 2025\] MMSafeAware: Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs](cant_see_the_forest_for_the.md)
- [\[ACL 2025\] Unveiling Cultural Blind Spots: Analyzing the Limitations of mLLMs in Procedural Text Comprehension](unveiling_cultural_blind_spots_analyzing_the_limitations_of_mllms_in_procedural_.md)
- [\[ACL 2025\] SPHERE: Unveiling Spatial Blind Spots in Vision-Language Models Through Hierarchical Evaluation](sphere_unveiling_spatial_blind_spots_in.md)
- [\[ACL 2025\] Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents](agent_rewardbench.md)

</div>

<!-- RELATED:END -->
