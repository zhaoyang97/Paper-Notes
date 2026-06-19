---
title: >-
  [论文解读] Visual Language Models as Zero-Shot Deepfake Detectors
description: >-
  [ICML 2025][LLM安全][Deepfake检测] 提出基于 VLM token 概率归一化的图像分类框架，将 deepfake 检测从二元判断升级为概率估计，在零样本设置下用 InstructBLIP 超越多数专用 deepfake 检测器，微调后在 DFDC-P 上接近完美。 领域现状：Deepfake 检测方…
tags:
  - "ICML 2025"
  - "LLM安全"
  - "Deepfake检测"
  - "视觉语言模型"
  - "零样本分类"
  - "VLM概率校准"
  - "InstructBLIP"
---

# Visual Language Models as Zero-Shot Deepfake Detectors

**会议**: ICML 2025  
**arXiv**: [2507.22469](https://arxiv.org/abs/2507.22469)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: Deepfake检测, 视觉语言模型, 零样本分类, VLM概率校准, InstructBLIP

## 一句话总结
提出基于 VLM token 概率归一化的图像分类框架，将 deepfake 检测从二元判断升级为概率估计，在零样本设置下用 InstructBLIP 超越多数专用 deepfake 检测器，微调后在 DFDC-P 上接近完美。

## 研究背景与动机
**领域现状**：Deepfake 检测方法大多训练专用分类器（FaceForensics++、SBI、MAT 等），依赖标注数据且对新型 deepfake 泛化差。

**现有痛点**：(a) 现有检测器在分布外数据上性能急剧下降；(b) 已有 VLM deepfake 研究只做 yes/no 二元判断，无法输出置信度概率；(c) 缺乏 FAR/FRR 等实际部署指标的支持。

**核心矛盾**：真实部署需要概率输出来调节阈值（平衡误报率和漏报率），但 VLM 的 argmax 输出只能给 0/1。

**本文目标** 如何从 VLM 的 token 分布中提取有意义的分类置信度？

**切入角度**：利用 VLM 在"Is this photo real?"回答时 "yes"/"no" token 的概率比作为置信度。

**核心 idea**：将 yes/no token 概率归一化为 $\tilde{P}_{\text{fake}} = P_{\text{no}} / (P_{\text{no}} + P_{\text{yes}})$，得到可用于 ROC 分析的连续置信度。

## 方法详解

### 整体框架
给定图像和提示词（如 "Is this photo real?"），VLM 做一次 forward pass 得到 token 分布。提取 "yes"/"Yes"/"no"/"No" 等 token 的概率，分组求和并归一化，得到 fake 置信度用于下游决策。

### 关键设计

1. **Token 概率归一化分类**:

    - 功能：从 VLM 的 token 分布中提取分类置信度
    - 核心思路：$P(I \in D) \approx \frac{P_{\text{no}}}{P_{\text{no}} + P_{\text{yes}}}$，其中 $P_{\text{no}} = p(\text{"no"}) + p(\text{"No"})$，$P_{\text{yes}} = p(\text{"yes"}) + p(\text{"Yes"})$
    - 设计动机：相比 argmax（0/1 输出），归一化概率支持 AUC/EER 评估和阈值调节

2. **多token/多类扩展 (Algorithm 1)**:

    - 功能：支持多 token 回答（如 "Yes for sure!"）和多分类
    - 核心思路：对类别 $c$ 的所有候选回答字符串 $s \in \mathcal{S}_c$，计算自回归概率 $P(s|I,Q) = \prod_k p(t_k|I,Q,t_{1:k-1}) \cdot p(\text{EOS}|I,Q,s)$，再求和归一化
    - 设计动机：不同 VLM 的 tokenizer 不一致，需要覆盖所有可能的回答形式

3. **Prompt Engineering**:

    - 功能：针对不同 VLM 设计专用提示词
    - 核心思路：InstructBLIP 只需 "Is this photo real?"；LLaVA 需加 "Answer using a single word"；GPT-4o 需要角色扮演式长 prompt
    - 设计动机：确保模型一致地返回 yes/no 格式回答

## 实验关键数据

### 主实验（零样本 vs 专用检测器，CelebA-HQ SimSwap 数据集）

| 方法 | AUC ↑ | ACC ↑ | EER ↓ |
|------|-------|-------|-------|
| FF++ (XceptionNet) | 58.9 | 59.2 | 44.5 |
| MAT | 49.0 | 50.0 | 50.6 |
| RECCE | 46.9 | 49.1 | 50.8 |
| SBI (SOTA 专用) | **93.6** | **85.2** | **14.0** |
| InstructBLIP (零样本) | 81.3 | 75.3 | 26.9 |
| **InstructBLIP FT** | **92.1** | **85.0** | **12.2** |

### 方法对比（归一化 vs softmax vs 二元）

| VLM | Binary ACC | Normalize AUC | Softmax AUC |
|------|-----------|---------------|-------------|
| InstructBLIP | 68.0 | **81.3** | 80.9 |
| Idefics2 | 74.2 | **80.6** | 75.2 |
| LLaVA-1.6 | 58.3 | **74.2** | 74.2 |

### 关键发现
- 归一化方法在所有 VLM 上均优于 binary argmax（最高提升 ~16% AUC）
- 零样本 InstructBLIP 超越大多数专门训练的检测器（仅逊于 SBI + CADDM）
- 微调 InstructBLIP 后达到 92.1% AUC，接近 SBI 的 93.6%

## 亮点与洞察
- **实用框架**：token 概率归一化方法通用于任何使用 VLM 的分类任务，不限于 deepfake
- **零样本能力展示**：VLM 的预训练知识足以在新型 deepfake 上达到可用性能
- **多token 扩展**：Algorithm 1 的自回归概率累乘支持任意长度回答

## 相关工作与启发
- **vs AntifakePrompt**: AntifakePrompt 在 InstructBLIP 上微调 soft prompt 做 deepfake VQA，但仅输出 0/1；本文不需微调且输出概率
- **vs SHIELD/ChatGPT deepfake**: 这些工作定性评估了 GPT-4V/Gemini 的 deepfake 检测能力，但未系统量化 token 概率；本文提出了完整的概率化框架
- **vs SBI (SOTA)**: SBI 通过自混合数据增强训练高泛化性分类器；本文的 VLM 方案完全零样本，虽 AUC 略低但无需任何 deepfake 训练数据
- 该 token 概率归一化框架可直接复用于任何需要从 VLM 获取分类置信度的场景（如医学影像分析、内容审核）

## 局限与展望
- 仅测试了人脸 swap 型 deepfake，未涵盖全脸生成（StyleGAN）、表情操纵（Face2Face）等类型
- GPT-4o 无法获取 token 概率，仅能做 binary 评估，限制了闭源模型的应用
- VLM 推理速度远慢于轻量级分类器（如 EfficientNet），实际部署需考虑延迟
- 新 deepfake 方法（如 Flux 生成的全身 deepfake）的测试缺失
- 多 token 回答的扩展虽有理论描述，但实验中未系统验证

## 评分
- 新颖性: ⭐⭐⭐⭐ token 概率归一化分类是简洁有效的创新
- 实验充分度: ⭐⭐⭐⭐ 多 VLM + 多检测器对比全面
- 写作质量: ⭐⭐⭐⭐ 方法推导清晰，prompt 和 algorithm 细节完整
- 价值: ⭐⭐⭐⭐ 开拓了 VLM 在安全检测中的新应用范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Unlocking the Capabilities of Large Vision-Language Models for Generalizable and Explainable Deepfake Detection](unlocking_the_capabilities_of_large_vision-language_models_for_generalizable_and.md)
- [\[NeurIPS 2025\] Zero-Shot Robustness of Vision Language Models Via Confidence-Aware Weighting](../../NeurIPS2025/llm_safety/zero-shot_robustness_of_vision_language_models_via_confidence-aware_weighting.md)
- [\[CVPR 2026\] ⊘ Source Models Leak What They Shouldn't ↛: Unlearning Zero-Shot Transfer in Domain Adaptation Through Adversarial Optimization](../../CVPR2026/llm_safety/oslash_source_models_leak_what_they_shouldnt_nrightarrow_unlearning_zero-shot_tr.md)
- [\[ICML 2025\] CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization](crow_eliminating_backdoors_from_large_language_models_via_internal_consistency_r.md)
- [\[ICML 2025\] Watch Out Your Album! On the Inadvertent Privacy Memorization in Multi-Modal Large Language Models](watch_out_your_album_on_the_inadvertent_privacy_memorization_in_multi-modal_larg.md)

</div>

<!-- RELATED:END -->
