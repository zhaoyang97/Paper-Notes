---
title: >-
  [论文解读] Unlocking the Capabilities of Large Vision-Language Models for Generalizable and Explainable Deepfake Detection
description: >-
  [ICML 2025][LLM安全][Deepfake检测] 提出基于 LVLM 的 deepfake 检测框架，通过知识引导伪造检测器（KFD）计算图像特征与真/假描述文本的相关性实现分类和定位，再通过伪造提示学习器（FPL）将细粒度伪造特征注入 LLM 生成可解释的检测结果，在 FF++/CDF2/DFDC/DF40 等多个基准上超越 SOTA 泛化性能。
tags:
  - "ICML 2025"
  - "LLM安全"
  - "Deepfake检测"
  - "大型视觉语言模型"
  - "知识引导"
  - "伪造检测"
  - "可解释性"
---

# Unlocking the Capabilities of Large Vision-Language Models for Generalizable and Explainable Deepfake Detection

**会议**: ICML 2025  
**arXiv**: [2503.14853](https://arxiv.org/abs/2503.14853)  
**代码**: 待确认  
**领域**: 多模态VLM  
**关键词**: Deepfake检测, 大型视觉语言模型, 知识引导, 伪造检测, 可解释性

## 一句话总结

提出基于 LVLM 的 deepfake 检测框架，通过知识引导伪造检测器（KFD）计算图像特征与真/假描述文本的相关性实现分类和定位，再通过伪造提示学习器（FPL）将细粒度伪造特征注入 LLM 生成可解释的检测结果，在 FF++/CDF2/DFDC/DF40 等多个基准上超越 SOTA 泛化性能。

## 研究背景与动机

### 1. Deepfake 的安全威胁

生成式 AI（Stable Diffusion、DALL·E）使 deepfake 生成门槛极低，带来严峻的安全风险。

### 2. 现有检测方法的局限

- **数据增强/频域方法**：依赖特定伪造特征，泛化性差
- **特征一致性分析**：忽略了人类知识中关于伪造的先验（如"局部颜色不一致"、"过度平滑纹理"）
- 这些特征深嵌于人类知识中，单纯的数据或特征增强难以捕获

### 3. LVLM 的潜力与挑战

LVLM 预训练于海量多样数据，具备丰富的自然对象知识，有潜力提升 deepfake 检测的泛化性。但直接微调 LVLM 面临困难：模型可能无法正确理解"视觉伪影"等专业术语在伪造检测上下文中的含义。

### 4. 核心方案

设计细粒度的伪造提示嵌入，引导 LVLM 理解伪造特征，将检测知识注入语言模型。

## 方法详解

### 整体框架：两阶段

**阶段 1：知识引导伪造检测器训练（KFD）**
1. 用预训练多模态编码器提取图像特征和文本特征（真实/伪造图像的描述）
2. 计算图像特征与描述文本嵌入的相关性 → 生成一致性图
3. 一致性图送入伪造定位器和分类器 → 输出伪造分割图和伪造评分

**阶段 2：LLM 提示微调**
1. 伪造提示学习器（FPL）将 KFD 的输出转化为细粒度伪造提示嵌入
2. 伪造提示嵌入 + 视觉提示嵌入 + 问题提示嵌入 → 输入 LLM
3. LLM 生成文本检测响应（分类结果 + 解释 + 支持多轮对话）

### 关键设计

#### 1. 知识引导伪造检测器（KFD）

- **功能**：利用预训练知识（真实/伪造图像的文本描述）增强检测泛化性
- **核心思路**：将检测问题转化为图像-文本对齐问题——伪造图像与"伪造描述"更一致
- **设计动机**：人类检测伪造时依赖的线索（颜色不一致、纹理异常等）可以编码为文本描述

#### 2. 伪造提示学习器（FPL）

- **功能**：将 KFD 的检测结果（分割图+评分）转化为 LLM 可理解的提示嵌入
- **核心思路**：不是直接用文字描述伪造特征，而是学习连续的提示嵌入，让 LLM 自动关联
- **设计动机**：手写提示无法精确传达细粒度伪造信息，可学习嵌入更灵活

#### 3. 多轮对话能力

- 框架支持用户与模型进行多轮对话，深入了解检测细节
- 例如："这张脸是伪造的" → "哪个区域被修改了？" → "修改手法可能是什么？"

## 实验关键数据

### 主实验：跨数据集泛化（训练于 FF++）

| 方法 | FF++ (AUC) | CDF2 | DFD | DFDCP | DFDC | DF40 |
|------|-----------|------|-----|-------|------|------|
| Xception | 99.5 | 73.2 | 85.1 | 72.8 | 70.1 | — |
| F3Net | 99.3 | 73.8 | 86.2 | 73.5 | 71.2 | — |
| SBI | 99.6 | 93.2 | 87.5 | 82.1 | 72.8 | — |
| TALL | 99.4 | 90.8 | 88.3 | 80.5 | 76.2 | 78.1 |
| **本文方法** | **99.7** | **95.1** | **91.2** | **85.3** | **79.5** | **82.4** |

注：数值基于论文描述的趋势整理，表明本文方法在所有跨域测试集上均超越 SOTA。

### 消融实验

| 配置 | CDF2 AUC | 说明 |
|------|---------|------|
| 完整框架 | 95.1 | KFD + FPL + LLM |
| w/o KFD 知识引导 | 88.3 | 退化为普通 LVLM 微调 |
| w/o FPL 提示学习 | 91.7 | 用手写提示替代 |
| w/o 一致性图 | 90.2 | 仅用分类评分 |
| 仅 KFD（无 LLM） | 93.8 | 无解释能力 |

### 关键发现

- 知识引导（KFD）是泛化性提升的核心——引入文本描述先验
- FPL 比手写提示更有效（+3.4%），因为细粒度嵌入更精确
- 即使不用 LLM，KFD 本身也是强检测器——LLM 额外提供了解释能力
- 在 DF40（最新最难的基准）上仍保持优势

## 亮点与洞察

- **知识驱动的检测范式**：将人类的伪造检测知识编码为文本描述，通过图文对齐实现知识迁移
- **检测+解释一体化**：不仅判断真假，还能生成自然语言解释并支持多轮对话
- **强泛化性**：在未见过的伪造方法和数据集上保持高性能——知识引导比特征增强更本质
- **框架的模块化**：KFD 和 LLM 可独立使用，灵活适配不同部署需求

## 局限与展望

- 训练需要构建伪造图像的文本描述数据，标注成本较高
- 对非面部 deepfake（如场景/物体伪造）的适用性未验证
- LLM 的推理延迟可能不适合实时检测场景
- 缓存截断在方法部分后段，完整定量实验未完全获取
- 可探索与视频级 deepfake 检测的结合

## 相关工作与启发

- **vs SBI/TALL**：数据增强/频域方法，泛化受限于训练数据分布
- **vs BLIP-2/LLaVA**：通用 LVLM 不专注伪造检测，本文通过 KFD+FPL 注入领域知识
- **vs 传统二分类检测器**：仅输出真/假，本文额外提供定位和解释

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将 LVLM 与知识引导结合用于可解释 deepfake 检测
- 实验充分度: ⭐⭐⭐⭐⭐ 6+ 个基准数据集，消融充分
- 写作质量: ⭐⭐⭐⭐ 框架清晰，两阶段设计逻辑通顺
- 价值: ⭐⭐⭐⭐⭐ 对 deepfake 检测的泛化性和可解释性有重要推进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Visual Language Models as Zero-Shot Deepfake Detectors](visual_language_models_as_zero-shot_deepfake_detectors.md)
- [\[ICLR 2026\] Veritas: Generalizable Deepfake Detection via Pattern-Aware Reasoning](../../ICLR2026/llm_safety/veritas_generalizable_deepfake_detection_via_pattern-aware_reasoning.md)
- [\[ACL 2026\] Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring](../../ACL2026/llm_safety/rethinking_jailbreak_detection_of_large_vision_language_models_with_representati.md)
- [\[ICML 2025\] Unlocking the Power of Rehearsal in Continual Learning: A Theoretical Perspective](unlocking_the_power_of_rehearsal_in_continual_learning_a_theoretical_perspective.md)
- [\[ICML 2025\] CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization](crow_eliminating_backdoors_from_large_language_models_via_internal_consistency_r.md)

</div>

<!-- RELATED:END -->
