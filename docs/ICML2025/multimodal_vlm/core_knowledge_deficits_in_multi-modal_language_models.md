---
title: >-
  [论文解读] Core Knowledge Deficits in Multi-Modal Language Models
description: >-
  [ICML 2025][多模态VLM][核心知识] 提出 CoreCognition 基准（12种核心认知能力、1503题），大规模评测230个MLLM后发现：模型在基础认知能力上系统性落后于人类，且随规模增大并未改善，而是更依赖捷径学习而非真正理解。 当前MLLM在高层次推理（图表理解、数学几何、动作识别等）上已接近甚至超…
tags:
  - "ICML 2025"
  - "多模态VLM"
  - "核心知识"
  - "多模态大语言模型评测"
  - "认知科学"
  - "shortcut learning"
  - "benchmark"
---

# Core Knowledge Deficits in Multi-Modal Language Models

**会议**: ICML 2025  
**arXiv**: [2410.10855](https://arxiv.org/abs/2410.10855)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 核心知识, 多模态大语言模型评测, 认知科学, shortcut learning, benchmark

## 一句话总结

提出 CoreCognition 基准（12种核心认知能力、1503题），大规模评测230个MLLM后发现：模型在基础认知能力上系统性落后于人类，且随规模增大并未改善，而是更依赖捷径学习而非真正理解。

## 研究背景与动机

当前MLLM在高层次推理（图表理解、数学几何、动作识别等）上已接近甚至超越人类，但在**对人类来说简单直觉的低层次任务**上却频频失败——如计数、视角转换、空间推理、时序推理和组合推理。这一现象呼应了经典的 **Moravec's Paradox**：对机器最难的恰是人类最容易的。

作者提出假设：这些缺陷源于MLLM缺乏**核心知识（core knowledge）**——人类从婴幼儿期就先天具备的基础认知能力。核心知识的思想根植于柏拉图的先验知识观、康德的先天直觉形式论，以及 Piaget 和 Spelke 等发展心理学家的实证研究。

现有MLLM基准多聚焦于高层推理（MathVerse、ScienceQA等），缺乏针对这些**底层核心认知能力**的系统评估。本文因此构建了首个大规模核心知识基准，用于探究MLLMs是否具备、如何表征和使用核心知识。

## 方法详解

### 整体框架

本文的工作包含三大部分：**(1)** 构建认知分类体系与CoreCognition基准；**(2)** 大规模模型评测（230个模型 × 11种prompt = 2530个数据点）；**(3)** 提出Concept Hacking方法进行受控实验，区分真正理解vs捷径学习。

### 关键设计

#### 1. 认知分类体系（Cognitive Taxonomy）

借鉴 Piaget 的认知发展四阶段理论，将12种核心能力划分为三个发展阶段：

| 发展阶段 | 核心能力 | 说明 |
|---------|---------|------|
| 感觉运动期 (Sensorimotor) | Boundary（边界）| 区分一个物体与另一个物体的过渡 |
| | Continuity（连续性）| 物体在时空中持续存在为统一整体 |
| | Permanence（永恒性）| 不被感知时物体仍然存在 |
| | Spatiality（空间性）| 对欧氏空间属性的先验理解 |
| 具体运算期 (Concrete Op.) | Perceptual Constancy（知觉恒常）| 外观变化不等于物理属性变化 |
| | Intuitive Physics（直觉物理）| 对物理规律的直觉 |
| | Perspective Taking（视角采纳）| 理解他人视角所见 |
| | Conservation（守恒）| 变换下属性不变 |
| | Hierarchy（层级）| 理解包含/排除关系 |
| 形式运算期 (Formal Op.) | Intentionality（意图理解）| 理解他人意图 |
| | Mechanical Reasoning（机械推理）| 从系统状态推断行为 |
| | Tool Use（工具使用）| 操纵物体达成目标的能力 |

能力之间存在依赖关系：低层次能力是高层次能力的认知基础。

#### 2. CoreCognition 基准构建

数据集包含 **1503个样本**，每个概念至少95个样例，涵盖图像和视频输入。构建流程：

- **原型设计（Prototyping）**：将12个理论概念转化为5-10个原型场景，每个场景抽象地描述一种可测试的认知情境（如物体永恒性→杯子藏球实验）
- **实例化（Instantiation）**：从互联网、公开数据集、生成模型、模拟环境、实际拍摄等来源收集视觉素材，配对精心设计的问题和选项，形成多选题（MCQ）
- **质量控制（Quality Control）**：每个QA经两轮独立交叉验证；额外通过Amazon Mechanical Turk的20名标注员验证；人类一致性错误的题目被二次审核

三条核心设计准则：

- **区分性（Discriminativeness）**：缺乏目标核心知识的模型必然答错
- **最小化混淆（Minimal Confounding）**：最小化对物体识别等辅助能力的依赖
- **最小化文本捷径（Minimal Text Shortcut）**：不能仅凭文本推理出答案

#### 3. 推理与评估策略

- **循环评估（Circular Evaluation）**：对k选题循环旋转选项k次，取正确率均值，缓解选项位置偏差
- **两阶段评分**：第一阶段通过模板匹配+LLM-as-Judge将自由文本映射到选项；第二阶段与GT比较。高FAIL率模型被排除
- **11种Prompt**：涵盖无prompt、深思、解释、奖惩、偏差缓解、角色扮演、认知指令等类别

#### 4. Concept Hacking（概念黑客）

这是本文最核心的方法创新——一种**受控实验方法**，通过系统地操纵图像中的因果特征来完全反转GT标签，从而区分模型是真正理解还是捷径学习。

具体做法：从CoreCognition中选取45个样本，为每个样本创建一个**操纵版本**——保持问题和无关条件不变，但通过改变任务相关特征使正确答案完全反转。

对每对（控制版/操纵版），模型的四种可能结果：

| 控制题 | 操纵题 | 解释 |
|-------|-------|------|
| ✓ 正确 | ✓ 正确 | **核心知识**：真正理解了概念 |
| ✓ 正确 | ✗ 错误 | **捷径学习**：依赖表面模式，操纵后失败 |
| ✗ 错误 | ✓ 正确 | 巧合正确（错误方式碰巧对了） |
| ✗ 错误 | ✗ 错误 | **核心缺失**：完全缺乏该核心知识 |

### 损失函数 / 训练策略

本文为评测工作，不涉及模型训练。核心贡献在于基准构建与评估方法论。

## 实验关键数据

### 主实验

评测230个模型（25个商用+205个开源），涵盖1B至110B参数规模。

| 模型 | 感觉运动期均值 | 具体运算期均值 | 形式运算期均值 | 总均值 |
|------|-------------|-------------|-------------|-------|
| Human | ~82.1 | ~83.0 | ~87.2 | 86.98 |
| GPT-o1 | 65.3 | 72.3 | 90.3 | 74.91 |
| GPT-4o | 67.8 | 62.1 | 86.5 | 69.25 |
| Qwen2.5-VL-72B | 62.3 | 64.2 | 88.0 | 68.29 |
| QVQ-72B | 67.6 | 69.8 | 58.3 | 68.07 |
| InternVL3-78B | 65.7 | 57.4 | 60.2 | 64.60 |
| Claude-3.5-Sonnet | 57.9 | 55.8 | 79.9 | 61.92 |

**关键发现**：所有MLLM在低层次阶段（感觉运动期、具体运算期）大幅落后于人类15-23个百分点，而在形式运算期接近或超过人类水平。商用模型并不一致优于开源模型。

### 消融实验

#### Prompt影响

| Prompt类型 | 代表 | 相对效果 |
|-----------|------|---------|
| 无prompt | 空字符串 | 基线 |
| 深思类 | "Let's think step by step" | 微弱提升 |
| 解释类 | 要求解释答案 | 无显著提升 |
| 奖惩类 | 给200美元tip | 无显著提升 |
| 认知指令 | 提供概念描述 | **+6%以上**，唯一有效prompt |

认知指令prompt有效，说明核心知识可能以分布式方式编码在模型参数中，显式概念提示起到"检索线索"的作用。

#### 推理模型 vs 指令模型

在12种核心能力中，推理模型（如GPT-o1、QVQ等）在10种能力上与对应指令微调模型无显著差异。仅知觉恒常略优（P=0.067）、视角采纳反而更差（P=0.004）。推理和test-time scaling未能有效改善核心知识缺陷。

### 关键发现

1. **核心知识缺陷（Core Knowledge Deficits）**：MLLM在低层次能力上系统性表现差于高层次能力，与人类一致高水平形成鲜明对比

2. **依赖关系错位（Misaligned Dependency）**：高层次能力表现与底层支撑能力不相关（Pearson ρ<0.4），缺乏人类认知发展中的层级结构化依赖

3. **缩放失效（Not Scaling）**：9种低层次能力中7种的缩放斜率显著低于高层次能力；视角采纳甚至出现**逆向缩放**（模型越大越差）

4. **捷径学习加剧（Shortcut Intensification）**：Concept Hacking实验显示，大模型更倾向于落入"捷径"或"核心缺失"象限，而非趋向人类核心知识区域。GPT-4o等最强模型也存在显著捷径依赖

5. **核心能力可预测高层表现**：除视角采纳和直觉物理外，核心能力与26个公开基准和SEED-Bench高层能力强相关

## 亮点与洞察

- **跨学科视角独特**：将发展认知科学的核心知识理论系统引入MLLM评测，从柏拉图到Piaget的理论基础扎实
- **评测规模空前**：230个模型 × 11种prompt = 2530个数据点，覆盖商用/开源/推理模型
- **Concept Hacking 方法新颖**：通过受控操纵因果特征反转GT，精准区分真正理解vs捷径学习，是比对抗样本更有原则性的评估范式
- **揭示缩放法则的局限**：明确指出"简单增大模型"不能解决核心知识缺陷，甚至会恶化某些能力（如视角采纳的逆向缩放）
- **认知指令prompt的发现**：仅需简单的概念描述就能提升6%+，提示核心知识以分布式形式存在但难以自发检索

## 局限与展望

1. **VQA格式限制**：依赖语言能力、计数、物体识别等辅助能力，无法完全消除混淆因素；也限制了非语言模型的测试
2. **Concept Hacking扩展性有限**：精细设计操纵对的过程耗时，目前仅45对样本，难以大规模化
3. **未涉及训练改进方案**：仅诊断了问题，未提出具体的训练方法来弥补核心知识缺陷
4. **静态评估**：核心知识在人类是通过交互获得的，静态VQA可能无法完整捕捉这种能力
5. **未来方向**：可探索在预训练前先蒸馏核心知识、设计核心知识增强的训练课程、以及补充纯视觉（非语言）评估格式

## 相关工作与启发

- 与 M3GIA、Marvel 等认知基准相比，CoreCognition 聚焦更底层的核心认知而非高层通用智能
- 与 DevBench 的发展心理学基准相比，本文面向多模态而非纯语言
- Shortcut learning 文献（Alvi 2018, Bahng 2020 等）为Concept Hacking提供了理论基础
- 对"缩放就够了"的乐观主义提出了有力反驳，呼应 Bender et al. 2021 和 Mitchell & Krakauer 2023 的批判

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 核心知识视角+Concept Hacking方法均为全新贡献
- 实验充分度: ⭐⭐⭐⭐⭐ — 230模型×11 prompt，消融全面，统计检验充分
- 写作质量: ⭐⭐⭐⭐ — 哲学引入精彩但篇幅略长，主体结构清晰
- 价值: ⭐⭐⭐⭐⭐ — 对MLLM根本局限的深刻诊断，具有重要指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[ACL 2025\] CoRe-MMRAG: Cross-Source Knowledge Reconciliation for Multimodal RAG](../../ACL2025/multimodal_vlm/core_mmrag_knowledge_reconciliation.md)
- [\[ICML 2025\] Vision-Language Models Create Cross-Modal Task Representations](vision-language_models_create_cross-modal_task_representations.md)
- [\[NeurIPS 2025\] Evaluating Multimodal Large Language Models on Core Music Perception Tasks](../../NeurIPS2025/multimodal_vlm/evaluating_multimodal_large_language_models_on_core_music_perception_tasks.md)
- [\[CVPR 2025\] MMRL: Multi-Modal Representation Learning for Vision-Language Models](../../CVPR2025/multimodal_vlm/mmrl_multi-modal_representation_learning_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
