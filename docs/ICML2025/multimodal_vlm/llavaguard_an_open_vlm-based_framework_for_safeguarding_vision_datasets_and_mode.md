---
title: >-
  [论文解读] LlavaGuard: An Open VLM-based Framework for Safeguarding Vision Datasets and Models
description: >-
  [ICML2025][多模态VLM][VLM safeguard] 提出 LlavaGuard——基于开源 VLM 的视觉内容安全审核框架，通过可定制安全分类体系、高质量人工标注数据集与策略增强训练，实现对图像内容的灵活、精准安全评估，在准确率和策略适应性上大幅超越现有开源与闭源审核工具。 - 大规模生成式 AI（T2I、V…
tags:
  - "ICML2025"
  - "多模态VLM"
  - "VLM safeguard"
  - "内容安全"
  - "安全分类体系"
  - "策略自适应"
  - "图像审核"
---

# LlavaGuard: An Open VLM-based Framework for Safeguarding Vision Datasets and Models

**会议**: ICML2025  
**arXiv**: [2406.05113](https://arxiv.org/abs/2406.05113)  
**代码**: [GitHub](https://ml-research.github.io/human-centered-genai/projects/llavaguard)  
**领域**: VLM安全 / 内容审核  
**关键词**: VLM safeguard, 内容安全, 安全分类体系, 策略自适应, 图像审核

## 一句话总结

提出 LlavaGuard——基于开源 VLM 的视觉内容安全审核框架，通过可定制安全分类体系、高质量人工标注数据集与策略增强训练，实现对图像内容的灵活、精准安全评估，在准确率和策略适应性上大幅超越现有开源与闭源审核工具。

## 研究背景与动机

- 大规模生成式 AI（T2I、VLM）依赖海量网络爬取数据训练，不可避免地包含不安全和有偏差的内容，产生严重的安全与伦理问题。
- 欧盟、美国、英国等地的 AI 法规（EU AI Act 等）要求生成模型符合安全合规标准，催生了对自动化安全审核工具的需求。
- 现有安全研究主要聚焦文本领域（如 LlamaGuard），视觉领域缺乏可靠的开源框架。
- 传统 NSFW 分类器（NudeNet、Q16 等）基于 CNN 或 CLIP，缺乏上下文感知能力，无法灵活适配不同安全策略（例如大麻在不同国家合法性不同）。
- **核心问题**：如何构建一个开源、灵活、高精度的视觉内容安全审核系统，能够根据不同策略动态调整安全评估？

## 方法详解

### 1. 安全分类体系（Safety Taxonomy）

LlavaGuard 设计了面向视觉领域的 9+1 类安全分类体系：

| 类别 | 描述 |
|------|------|
| O1 | 仇恨、侮辱、骚扰 |
| O2 | 暴力、伤害、残忍 |
| O3 | 色情内容 |
| O4 | 裸露内容 |
| O5 | 犯罪策划 |
| O6 | 武器或药物滥用 |
| O7 | 自残 |
| O8 | 动物虐待 |
| O9 | 灾难或紧急事件 |
| NA | 不适用（安全） |

与文本安全分类不同，该体系专门针对视觉模态设计，区分了裸露与色情内容，并新增灾难/紧急事件类别。

### 2. 风险指南与策略灵活性

每个安全类别配备详细的风险指南（Risk Guidelines），明确定义"不应包含"和"可以包含"的内容边界。策略调整方式：

- **类别豁免**：将某类别声明为"非违规"，使模型允许该类内容
- **指南修改**：在 Should not / Can 之间移动条目，实现细粒度控制

### 3. 数据集构建

**数据来源**：以 SMID（Socio-Moral Image Database）为基础，针对类别不平衡问题从 Google/Bing 补充爬取图片，确保每个类别至少 100 张。

**人工标注**：每张图片标注安全类别、安全评级（四级：Highly Unsafe → Moderately Unsafe → Barely Safe → Generally Safe）。

**数据增强**：
- **策略豁免增强（Policy Exception）**：将原本不安全的样本所违反的类别声明为非违规，翻转标签为安全
- **随机类别豁免**：随机声明最多 3 个不相关类别为非违规，训练模型正确忽略

**引导式理由生成（Guided Rationales）**：

直接让 VLM 生成安全解释质量很差（GPT-4o 评分仅 3.8/10）。作者采用条件化提示，将安全评级、类别、风险指南等先验知识注入生成过程：

$$\text{Rationale} = \text{VLM}(\text{image}, \text{policy}, \text{rating}, \text{category}, \text{guidelines})$$

引导后 Llava-34B 生成的理由质量评分达 **9.1/10**（胜率 99.9%）。最终数据集包含 5,466 个样本（train/eval/test = 4571/71/824）。

### 4. 模型训练与评估指标

在 Llava-OneVision（0.5B、7B）和 Qwen2.5-VL（3B、7B）基础上微调。模型输出 JSON 格式的三元组：

```json
{"safety_rating": "Unsafe", "category": "O3", "rationale": "..."}
```

**策略适应性指标 PER（Policy Exception Rate）**：

$$\text{PER} = \frac{\text{PE}_{\text{correct}}}{\text{PE}_{\text{correct}} + \text{PE}_{\text{false}}}$$

其中 $\text{PE}_{\text{correct}} = \sum_{i=1}^{N} \delta(y_i, \hat{y}_i)$，$\delta=1$ 表示策略豁免样本被正确分类。

**综合指标 PES（Policy Exception Score）**：PER 与 Balanced Accuracy 的调和平均：

$$\text{PES} = \frac{2 \times \text{PER} \times \text{Acc}}{\text{PER} + \text{Acc}}$$

## 实验关键数据

### 主要性能对比（held-out 测试集）

| 模型 | 开源 | Balanced Acc (%) | Recall (%) | Precision (%) | PES (%) |
|------|------|------------------|------------|---------------|---------|
| GPT-4o | ✗ | 72.92 | 55.99 | 81.05 | 77.29 |
| LlamaGuard-3-11B | ✓ | 50.28 | 0.56 | 100.0 | 66.92 |
| OpenAI-omni-mod | ✗ | 66.92 | 45.24 | 47.50 | 60.23 |
| ImageGuard | ✓ | 70.98 | 83.33 | 60.98 | 27.00 |
| **LlavaGuard-0.5B** | ✓ | **88.70** | 86.67 | 87.89 | **87.10** |
| **LlavaGuard-7B** | ✓ | **90.84** | **91.39** | 87.97 | **89.85** |

### 关键发现

- LlavaGuard-7B 的 Balanced Acc（90.84%）比 GPT-4o（72.92%）高 **18 个百分点**
- LlamaGuard-3-Vision 的 Recall 仅 0.56%，几乎将所有图片判为安全，完全不可用
- ImageGuard PES 仅 27.00%，严重过拟合固定策略，无法适应策略变化
- LlavaGuard-0.5B 推理速度是 7B 的 **3.47 倍**（0.075s/sample，A100 GPU）

### 引导式理由质量对比

| 模型 | 类型 | 均分 | 中位数 | 胜率(%) |
|------|------|------|--------|---------|
| Llava-34B | 非引导 | 3.8 | 3.0 | 0.1 |
| Llava-34B | 引导 | **9.1** | **9.0** | **99.9** |

## 亮点与洞察

1. **开源完整框架**：从分类体系、数据标注、增强策略到模型训练全链条开源，降低安全审核工具构建门槛
2. **策略自适应能力**：通过策略增强训练使模型能够根据不同法规/场景灵活调整安全评估，这是与其他审核工具的核心差异点
3. **引导式理由生成**：巧妙地将先验安全知识注入理由生成过程，质量从 3.8 跃升至 9.1，显著提升可解释性
4. **0.5B 小模型惊艳表现**：仅 0.5B 参数即超越 GPT-4o 和所有专用审核工具，证明领域微调的效能远超模型规模
5. **实际应用验证**：在大规模数据集标注和 T2I 模型审核两个真实场景中验证了有效性

## 局限与展望

1. **数据集规模有限**：仅 5,466 个样本，在长尾安全场景（如文化敏感内容）的覆盖可能不足
2. **分类体系固定为 9 类**：虽然支持类别豁免，但新增类别需要重新标注和训练
3. **单图片评估**：未考虑多图片上下文或图文联合的安全评估场景
4. **评估基准局限**：测试集仅 824 样本，且主要基于自建数据，缺乏与 NudeNet 等工具在其原始基准上的对比
5. **理由生成依赖 Llava-34B**：引导式理由的质量上限受限于基座模型能力
6. **对抗鲁棒性未验证**：未测试针对性对抗攻击（如微扰绕过审核）下的表现

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个面向视觉的开源安全审核完整框架，策略自适应机制新颖
- 实验充分度: ⭐⭐⭐⭐ — 对比全面（含 GPT-4o、多种开源模型），但测试集较小
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表丰富，定性定量分析结合良好
- 价值: ⭐⭐⭐⭐⭐ — 解决了视觉安全审核领域的关键空白，实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks](../../ICCV2025/multimodal_vlm/safeguarding_vision-language_models_mitigating_vulnerabilities_to_gaussian_noise.md)
- [\[CVPR 2025\] Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models](../../CVPR2025/multimodal_vlm/molmo_and_pixmo_open_weights_and_open_data_for_state-of-the-art_vision-language_.md)
- [\[ICML 2026\] Immuno-VLM: Immunizing Large Vision-Language Models via Generative Semantic Antibodies for Open-World Trustworthiness](../../ICML2026/multimodal_vlm/immuno-vlm_immunizing_large_vision-language_models_via_generative_semantic_antib.md)
- [\[CVPR 2026\] LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models](../../CVPR2026/multimodal_vlm/llavashield_multimodal_multiturn_safety.md)
- [\[ICCV 2025\] GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks](../../ICCV2025/multimodal_vlm/geobench-vlm_benchmarking_vision-language_models_for_geospatial_tasks.md)

</div>

<!-- RELATED:END -->
