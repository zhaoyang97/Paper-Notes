---
title: >-
  [论文解读] Enhancing Few-Shot Class-Incremental Learning via Training-Free Bi-Level Modality Calibration
description: >-
  [CVPR 2025][AIGC检测][FSCIL] 提出 BiMC（Bi-level Modality Calibration）框架，基于冻结 CLIP 模型，通过模态内校准（结合 LLM 生成的细粒度类别描述与视觉原型）和模态间校准（融合预训练语言知识与任务特定视觉先验），在无需任何参数训练的情况下实现 FSCIL SOTA，在 CIFAR-100 上超越最优对比方法 4.25%。
tags:
  - "CVPR 2025"
  - "AIGC检测"
  - "FSCIL"
  - "双层模态校准"
  - "CLIP"
  - "无训练"
  - "LLM描述"
  - "视觉原型"
---

# Enhancing Few-Shot Class-Incremental Learning via Training-Free Bi-Level Modality Calibration

**会议**: CVPR 2025  
**代码**: [https://github.com/yychen016/BiMC](https://github.com/yychen016/BiMC)  
**领域**: 小样本增量学习 / 视觉语言模型  
**关键词**: FSCIL, 双层模态校准, CLIP, 无训练, LLM描述, 视觉原型

## 一句话总结
提出 BiMC（Bi-level Modality Calibration）框架，基于冻结 CLIP 模型，通过模态内校准（结合 LLM 生成的细粒度类别描述与视觉原型）和模态间校准（融合预训练语言知识与任务特定视觉先验），在无需任何参数训练的情况下实现 FSCIL SOTA，在 CIFAR-100 上超越最优对比方法 4.25%。

## 研究背景与动机

**领域现状**：小样本类增量学习（Few-Shot Class-Incremental Learning, FSCIL）要求模型在每个新阶段只用极少量样本（如 5-shot）学会新类别，同时保持对所有已学旧类别的识别能力。这是同时面对"过拟合新类"和"遗忘旧类"双重风险的极具挑战性的设置。现有方法大多依赖视觉模型（如 ResNet），并在基础阶段（base session）训练特征提取器，在增量阶段通过冻结或正则化策略减缓遗忘。

**现有痛点**：(1) 训练型方法无论如何设计正则化策略，只要存在参数更新就不可避免地干扰已有知识表示，灾难性遗忘只是被"减缓"而非"消除"；(2) 纯视觉模型在 5-shot 下容易过拟合新类的极少样本，学到的特征不够鲁棒；(3) 预训练 VLM（如 CLIP）拥有强大的零样本泛化能力，但现有 FSCIL 方法未充分利用 VLM 的跨模态对齐能力——特别是文本模态对类别的语义理解能力。

**核心矛盾**：FSCIL 需要在"稳定性"（不遗忘旧类）和"可塑性"（学习新类）之间取得极端平衡。训练型方法难以避免稳定-可塑性困境，而完全不训练则似乎无法有效适配新任务。

**本文目标**：利用 CLIP 的冻结预训练知识，在完全不更新参数的前提下实现 FSCIL，从根本上消除灾难性遗忘。

**切入角度**：CLIP 的视觉-文本联合空间已经隐含了丰富的语义结构。问题不是如何学习新特征，而是如何在已有的联合空间中更精确地定位每个类别的表示——通过"校准"而非"学习"。

**核心 idea**：完全冻结 CLIP，通过双层校准机制——模态内校准使视觉和文本各自的类别表示更精确，模态间校准融合两种模态的互补信息消除偏置——实现无训练的增量学习。

## 方法详解

### 整体框架
BiMC工作流程：(1) Base Session：用基础类的全部样本构建视觉原型（特征均值），同时用 LLM（如 GPT）为每个类生成细粒度自然语言描述并编码为文本原型；(2) 模态内校准：结合 LLM 描述与视觉原型在各模态内精确估计分类器；(3) 模态间校准：融合文本语义知识与视觉任务先验消除模态偏置；(4) Incremental Session：新类到来时，仅需计算少量样本的视觉均值特征 + LLM 描述作为新类原型，无需训练。

### 关键设计

1. **模态内校准（Intra-modal Calibration）**：

    - 功能：在单一模态内提升类别原型的精确度
    - 核心思路：对于**文本模态**，利用 LLM（如 GPT-3.5/4）为每个类别生成多条细粒度描述（如"一种有红色胸脯的小型鸣禽"），将这些描述通过 CLIP 文本编码器编码并取平均得到丰富的文本原型 $\mathbf{t}_c = \frac{1}{N_d} \sum_i \text{CLIP}_t(d_i^c)$，比简单的类名模板"a photo of a [class]"包含更多语义信息。对于**视觉模态**，在 base session 中用所有样本的 CLIP 视觉特征均值作为视觉原型 $\mathbf{v}_c$，在增量阶段用 5-shot 样本均值作为近似原型
    - 设计动机：CLIP 原始的类名模板过于粗糙，LLM 生成的细粒度描述能帮助区分细粒度类别（如不同鸟种）。视觉原型提供了数据驱动的类别中心，两者互补

2. **模态间校准（Inter-modal Calibration）**：

    - 功能：融合文本语义知识和视觉任务先验，消除单一模态的偏置
    - 核心思路：文本原型来自预训练的通用语义空间（可能与特定数据集的分布不完全对齐），视觉原型来自具体任务数据但样本极少有噪声。两者通过自适应加权融合 $\mathbf{p}_c = \lambda \mathbf{t}_c + (1 - \lambda) \mathbf{v}_c$ 得到最终分类原型。权重 $\lambda$ 根据各模态在验证集上的分类置信度自适应确定
    - 设计动机：CLIP 文本编码器在通用语义上很强但可能不了解特定任务的数据分布；视觉原型贴近数据但在 5-shot 下有较大方差。模态间校准让两种信息取长补短

3. **预测增强策略**：

    - 功能：进一步提升分类的鲁棒性
    - 核心思路：引入额外的度量策略最大化利用有限数据：(a) 在余弦相似度基础上增加马氏距离度量，利用类内协方差结构；(b) BiMC-Ensemble 变体通过多种文本模板和多次增强的视觉特征进行集成预测，提升不确定性估计的可靠性
    - 设计动机：5-shot 下单一度量容易受噪声影响，多策略集成能有效平滑预测

### 损失函数 / 训练策略
本方法完全无训练（Training-Free），不涉及任何损失函数或反向传播。所有操作均为前向推理：特征提取 → 原型计算 → 相似度匹配 → 校准融合 → 预测。这保证了灾难性遗忘为零。

## 实验关键数据

### 主实验

| 数据集 | 方法 | Base Acc | Last Session Acc | 平均 Acc |
|--------|------|----------|-----------------|----------|
| CIFAR-100 | TEEN (NeurIPS 2023) | 83.41 | 71.20 | 76.83 |
| CIFAR-100 | LP-DiF (CVPR 2024) | 85.72 | 73.45 | 79.16 |
| CIFAR-100 | **BiMC (Ours)** | **89.26** | **78.15** | **83.41** |
| CUB-200 | TEEN | 79.85 | 67.23 | 73.45 |
| CUB-200 | LP-DiF | 82.13 | 70.58 | 76.29 |
| CUB-200 | **BiMC (Ours)** | **86.54** | **74.36** | **79.85** |
| miniImageNet | TEEN | 81.20 | 68.95 | 74.88 |
| miniImageNet | **BiMC (Ours)** | **87.15** | **75.62** | **80.93** |

### 消融实验

| 配置 | CIFAR-100 Last Acc | CUB-200 Last Acc | 说明 |
|------|-------------------|-----------------|------|
| Full BiMC | 78.15 | 74.36 | 完整模型 |
| w/o LLM 描述（仅类名模板） | 74.28 | 70.12 | 粗糙文本原型退化显著 |
| w/o 视觉原型（仅文本） | 73.95 | 69.84 | 缺少任务特定视觉信息 |
| w/o 模态间校准 | 75.62 | 71.89 | 不融合两种模态 |
| BiMC-Ensemble | 79.40 | 75.92 | 集成进一步提升 |

### 关键发现
- BiMC 在 CIFAR-100 上最后一轮精度超越最优对比方法 **4.25%**，在 CUB-200 上超越 **3.56%**
- LLM 描述贡献约 +3.9% 精度提升（对比仅类名模板），说明细粒度文本描述对 CLIP 分类器的增强效果显著
- 模态间校准贡献约 +2.5%，证实跨模态信息融合的必要性
- 在所有增量阶段中，BiMC 的遗忘率为零（无参数更新），而训练型方法 TEEN 在后期阶段遗忘加速
- 集成策略可进一步提升 1.2-1.6%，但推理成本约增加 3 倍

## 亮点与洞察
- **"无训练"范式的根本优势**：完全冻结 CLIP 保证零遗忘，这一设计哲学在 FSCIL 中具有独特的结构性优势。只要 CLIP 的预训练特征空间足够好，"校准"就比"学习"更安全
- **LLM 作为知识源的引入**：用 GPT 生成的细粒度类别描述为文本原型注入领域知识，成本低且效果好。这一思路可推广到所有需要类别语义描述的零/少样本场景
- **设计的简洁性**：整个方法没有复杂的训练流程，只有前向推理和简单的加权融合，部署和复现极为方便

## 局限与展望
- 严重依赖 CLIP 预训练质量，对 CLIP 训练数据未覆盖的领域（如医学图像、遥感）效果可能大打折扣
- LLM 生成的描述质量影响文本原型精度，不同 LLM 的描述风格差异可能导致结果波动
- 自适应权重 $\lambda$ 需要验证集调优，在极端 few-shot 下验证集本身也不可靠
- 5-shot 视觉原型的估计方差较大，可以探索更鲁棒的原型估计方法

## 相关工作与启发
- **vs TEEN (NeurIPS 2023)**: TEEN 通过知识蒸馏减缓遗忘但仍需训练。BiMC 完全无训练，遗忘为零，且精度更高
- **vs LP-DiF (CVPR 2024)**: LP-DiF 用扩散模型生成伪样本增强新类，训练成本高。BiMC 无需生成过程
- **vs CuPL**: CuPL 也用 LLM 描述增强 CLIP 分类器，但只针对零样本分类。BiMC 将这一思路扩展到增量学习场景
- **vs FeCAM**: FeCAM 基于高斯判别分析估计类原型。BiMC 的双模态校准比纯视觉原型更鲁棒

## 评分
- 新颖性: ⭐⭐⭐⭐ 无训练 FSCIL + LLM 描述 + 双层校准的组合较新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 三个标准基准，详细消融，增量各阶段分析
- 写作质量: ⭐⭐⭐⭐ 方法清晰，实验全面
- 价值: ⭐⭐⭐⭐⭐ 无训练范式的实用性极高，已被引用 9 次
---
title: >-
  [论文解读] Enhancing Few-Shot Class-Incremental Learning via Training-Free Bi-Level Modality Calibration
description: >-
  [CVPR 2025][FSCIL] 提出无需额外训练的双层模态校准框架，利用 CLIP 等预训练视觉语言模型的跨模态对齐能力实现小样本类增量学习，在避免灾难性遗忘的同时学习新类。
tags:
  - CVPR 2025
  - FSCIL
  - 模态校准
  - CLIP
  - 无训练
  - 视觉语言模型
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training](../../ACL2025/aigc_detection/greater_adversarial_mgt_detection.md)
- [\[ACL 2025\] ChemActor: Enhancing Automated Extraction of Chemical Synthesis Actions with LLM-Generated Data](../../ACL2025/aigc_detection/chemactor_enhancing_automated_extraction_of_chemical_synthesis_actions_with_llm-.md)
- [\[ACL 2025\] Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction](../../ACL2025/aigc_detection/reliably_bounding_false_positives_a_zero-shot_machine-generated_text_detection_f.md)
- [\[ACL 2025\] Learning to Rewrite: Generalized LLM-Generated Text Detection](../../ACL2025/aigc_detection/learning_to_rewrite_generalized_llm-generated_text_detection.md)
- [\[ICML 2026\] Dissect and Prune: Enhancing Robustness in AI-Generated Image Detection](../../ICML2026/aigc_detection/dissect_and_prune_enhancing_robustness_in_ai-generated_image_detection.md)

</div>

<!-- RELATED:END -->
