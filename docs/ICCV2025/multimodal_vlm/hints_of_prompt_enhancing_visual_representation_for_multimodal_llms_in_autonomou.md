---
title: >-
  [论文解读] Hints of Prompt: Enhancing Visual Representation for Multimodal LLMs in Autonomous Driving
description: >-
  [多模态VLM] 提出Hints of Prompt（HoP）框架，通过三种层次化提示（Affinity/Semantic/Question hint）增强CLIP视觉表征的实例级结构、领域语义和问题相关性，在自动驾驶VQA任务上仅用25%数据即超越基线全数据性能。 - 问题定义：自动驾驶场景中的视觉问答（VQA）需要精细的…
tags:
  - "多模态VLM"
---

# Hints of Prompt: Enhancing Visual Representation for Multimodal LLMs in Autonomous Driving

## 基本信息
- **会议**: ICCV 2025
- **arXiv**: 2411.13076
- **代码**: 未公开
- **领域**: 多模态VLM / 自动驾驶
- **关键词**: 多模态大语言模型, 自动驾驶VQA, 视觉表征增强, 提示融合, 领域适应

## 一句话总结

提出Hints of Prompt（HoP）框架，通过三种层次化提示（Affinity/Semantic/Question hint）增强CLIP视觉表征的实例级结构、领域语义和问题相关性，在自动驾驶VQA任务上仅用25%数据即超越基线全数据性能。

## 研究背景与动机

- **问题定义**：自动驾驶场景中的视觉问答（VQA）需要精细的层次化视觉特征来处理复杂交互和长尾场景，但通用MLLM（如LLaVA）结合CLIP视觉编码器难以准确表示驾驶特定场景。
- **现有方法局限**：
    - CLIP视觉token缺乏token间亲和关系（如Fig.3所示），丢失了实例级结构信息
    - 通用视觉编码器欠表达领域特定语义（如远处车辆、行人、交通标志）
    - 视觉和文本token分开处理，模型难以根据具体问题聚焦相关图像区域
    - 已有多编码器融合方法（A-MoF等）融合策略复杂，且缺乏针对驾驶领域的快速适应能力
- **关键观察**：DINOv2 token保持了实例内的强亲和性，可弥补CLIP的不足；DETR类模型的稀疏查询包含驾驶相关语义；LLM的文本嵌入可引导视觉关注问题相关区域。

## 方法详解

### 整体框架

HoP在LLaVA-v1.5基础上引入三种hint token增强CLIP视觉表征，通过Hint Fusion模块融合后经adapter送入LLM。

### 三种Hint Token

**1. Affinity Hint（亲和提示）**
- 来源：DINOv2-large提取的576个特征token
- 作用：提供实例级结构信息，增强token间亲和关系
- 验证：仅使用DINOv2 token间的相似度矩阵作为hint也能带来显著提升（Tab.4），证明提升源于token-wise affinity而非DINOv2特有特征

**2. Semantic Hint（语义提示）**
- 来源：Mask2Former或GroundingDINO的top-K稀疏查询（默认16个）
- 作用：引入驾驶场景特定的高级语义信息（车辆、交通标志、行人等）
- 每个token附加对应类别标签的嵌入特征
- Mask2Former优于GroundingDINO（Tab.5），因其在Cityscapes上训练过、更适配驾驶场景

**3. Question Hint（问题提示）**
- 来源：LLM的文本嵌入层（优于CLIP文本编码器，Tab.6）
- 作用：将视觉特征与问题上下文对齐，聚焦问题相关的图像区域

### Hint Fusion模块

探索了5种融合策略，最终采用**Joint Cross-Attention**：

$$\mathbf{P}_f = \mathbf{P} + \text{CA}(\mathbf{P}, [\mathbf{P}, \mathbf{H}_A, \mathbf{H}_S, \mathbf{H}_Q])$$

- CLIP视觉token $\mathbf{P}$ 作为query，与自身及所有hint拼接后的token作为key-value
- 仅8.7M参数和3.8 GFLOPs，效率最优
- 相比Concatenation（直接拼接反而损害性能）、Sequential/Parallel Cross-Attention等策略表现最好

### Efficient HoP

- 从CLIP backbone蒸馏两个轻量头替代DINOv2和Mask2Former
- Affinity head：4层ViT-like decoder，通过cosine similarity蒸馏DINOv2特征
- Semantic head：ViTDet-like neck + Mask2Former head，在Cityscapes上训练
- 延迟仅661ms（vs. HoP的956ms），量化后281ms

## 实验关键数据

### LingoQA主结果

| 方法 | LLM | Lingo-Judge↑ | BLEU-4↑ | METEOR↑ | CIDEr↑ |
|------|-----|-------------|---------|---------|--------|
| GPT-4V (zero-shot) | - | 59.6 | 6.30 | 12.4 | 42.8 |
| LLaVA-v1.5 | Vicuna-7B | 63.2 | 14.1 | 19.3 | 63.7 |
| LLaVA-v1.5 (+A-MoF) | Vicuna-7B | 64.2 | 14.5 | 19.1 | 64.7 |
| VTS | InternLM2-7B | 64.2 | 14.5 | 20.5 | 56.9 |
| Efficient HoP | Vicuna-7B | 66.8 | 15.2 | 20.0 | 66.2 |
| **HoP** | **Vicuna-7B** | **67.8** | **15.8** | **20.3** | **70.9** |

HoP在所有指标上全面超越SOTA。

### 消融实验（Hint组合效果）

| 配置 | Lingo-Judge↑ |
|------|-------------|
| Baseline (无hint) | 63.2 |
| +Affinity hint only | ~64.6 |
| +Semantic hint only | ~64.0 |
| +Question hint only | ~63.8 |
| +AH+SH | ~66.0 |
| +AH+SH+QH (ALL) | 67.4 |
| +AH+SH(+cls)+QH | **67.8** |

三种hint独立使用都能提升，组合使用效果叠加，加入类别信息进一步提升。

### 融合策略对比

| 融合方式 | Lingo-Judge↑ | 参数量(M) | GFLOPs |
|---------|-------------|-----------|--------|
| Concatenation | 58.8 | - | - |
| Self-Cross-Attention | 65.2 | 12.9 | 5.0 |
| Parallel Cross-Attention | 66.6 | 17.1 | 8.63 |
| Sequential Cross-Attention | 67.0 | 17.1 | 8.63 |
| **Joint Cross-Attention** | **67.8** | **8.7** | **3.8** |

Joint Cross-Attention以最少参数和计算量实现最佳性能。

### 数据效率

HoP仅用25%训练数据即可达到64.0 Lingo-Judge，超过LLaVA-v1.5全量数据的63.2，展示了出色的领域适应效率。

## 亮点与洞察

1. **三层次hint设计精巧**：从实例级（Affinity）到语义级（Semantic）再到查询级（Question），递进式增强视觉表征
2. **数据效率卓越**：25%数据即超越baseline全量训练，证明hint有效弥补了领域差距
3. **CLIP token的关键缺陷发现**：CLIP丢失了token间亲和关系，这一观察对理解CLIP在下游任务的局限有重要意义
4. **Concatenation反而有害**：直接拼接多级信息反而混淆adapter和LLM，提示融合策略的设计不可忽视
5. **Efficient版本实用性强**：量化后281ms延迟，适合实际部署

## 局限性

- HoP版本需要额外运行DINOv2和Mask2Former，推理延迟增加约50%
- 仅在自动驾驶VQA上验证，未探索其他领域（如医疗、遥感）的泛化性
- Semantic hint依赖预训练检测/分割模型的质量
- 当前仅在7B规模LLM上实验，更大模型的效果未知

## 相关工作与启发

- QA-ViT：同样将文本信息融入视觉特征，但本文融合策略更灵活通用
- A-MoF：多编码器融合方案，但策略复杂且未针对驾驶领域优化
- Hint-AD / TOKEN：同为自动驾驶视觉增强方案，但本文的动态hint + 轻量融合更具扩展性
- **启发**：这种多源hint增强的思路可推广到其他领域特定的MLLM应用

## 评分

- **新颖性**: ⭐⭐⭐⭐ （三种hint的设计和Joint Cross-Attention融合有新意）
- **实验**: ⭐⭐⭐⭐ （三个数据集全面验证，消融充分）
- **写作**: ⭐⭐⭐⭐ （逻辑清晰，图表丰富）
- **价值**: ⭐⭐⭐⭐ （对驾驶VQA有实际应用价值，Efficient版本可部署）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices](../../CVPR2025/multimodal_vlm/multi-layer_visual_feature_fusion_in_multimodal_llms_methods_analysis_and_best_p.md)
- [\[ACL 2025\] Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](../../ACL2025/multimodal_vlm/table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)
- [\[ICCV 2025\] MM-IFEngine: Towards Multimodal Instruction Following](mm-ifengine_towards_multimodal_instruction_following.md)
- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)
- [\[ACL 2025\] Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions](../../ACL2025/multimodal_vlm/teaching_vlm_ask_ambiguity.md)

</div>

<!-- RELATED:END -->
