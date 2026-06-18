---
title: >-
  [论文解读] Leveraging Textual Compositional Reasoning for Robust Change Captioning
description: >-
  [AAAI 2026][多模态VLM][Change Captioning] 提出 CORTEX 框架，通过引入 VLM 生成的组合推理文本作为显式线索，结合图像-文本双重对齐模块（ITDA），增强纯视觉变化描述方法对物体关系和空间配置等结构化语义的理解能力。 变化描述（Change Captioning）旨在生成自然语言描…
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "Change Captioning"
  - "组合推理"
  - "视觉语言模型"
  - "图像-文本对齐"
  - "场景变化描述"
---

# Leveraging Textual Compositional Reasoning for Robust Change Captioning

**会议**: AAAI 2026  
**arXiv**: [2511.22903](https://arxiv.org/abs/2511.22903)  
**代码**: [https://github.com/VisualAIKHU/CORTEX](https://github.com/VisualAIKHU/CORTEX)  
**领域**: 多模态VLM  
**关键词**: Change Captioning, 组合推理, 视觉语言模型, 图像-文本对齐, 场景变化描述

## 一句话总结

提出 CORTEX 框架，通过引入 VLM 生成的组合推理文本作为显式线索，结合图像-文本双重对齐模块（ITDA），增强纯视觉变化描述方法对物体关系和空间配置等结构化语义的理解能力。

## 研究背景与动机

变化描述（Change Captioning）旨在生成自然语言描述来解释两张图像之间的差异，在监控和医学影像等领域有重要应用。现有方法的核心问题在于：

**纯视觉方法的局限**：SCORER、SMART、DIRL 等方法仅依赖视觉特征来捕捉变化，虽然能检测低层外观差异，但无法进行**组合推理**（compositional reasoning）——即理解物体关系、空间配置等结构化语义。这类信息并非直接编码在图像中，而是隐式嵌入的。

**典型错误**：如图1所示，现有方法经常误判空间关系（如"在左边"）或错误识别参考物体（如"一个小棕色圆柱体"），因为它们缺乏显式的结构化信息表示能力。

**文本的互补优势**：与视觉信息不同，文本可以以清晰可解释的形式显式描述图像中嵌入的结构化语义，为高级推理提供强信号。

基于以上观察，作者认为通过引入显式的文本组合推理线索来增强现有纯视觉方法，可以更好地捕捉场景变化中的关系和上下文含义。

## 方法详解

### 整体框架

CORTEX（COmpositional Reasoning-aware TEXt-guided）是一个即插即用框架，由三个模块组成：
- **图像级变化检测器**（Image-level Change Detector）：直接复用现有方法（如DIRL/SCORER/SMART），从"前"图像 $I_{bef}$ 和"后"图像 $I_{aft}$ 提取视觉特征并编码低层变化线索 $f_{icd}$
- **推理感知文本提取模块（RTE）**：利用 VLM 从每张图像中提取包含组合推理信息的结构化句子
- **图像-文本双重对齐模块（ITDA）**：通过静态对齐（场景内）和动态对齐（跨场景）统一视觉和文本特征

### 关键设计

#### 1. **推理感知文本提取模块（RTE）**

核心思路：利用冻结的 VLM（如 InternVL2-8B）为每张图像生成组合推理句子，提取传统视觉特征难以捕捉的语义信息。

- **精心设计的提示词**：不生成通用描述，而是引导 VLM 提取包含详细属性（颜色、形状、大小）和空间关系的组合推理线索
- **动态句子数量**：根据每个场景的物体密度和复杂性动态决定提取的句子数量，"前"图像生成 $N$ 个句子 $T_{bef}$，"后"图像生成 $M$ 个句子 $T_{aft}$
- **文本编码**：生成的句子通过 BERT 编码器嵌入为句子特征 $t_{bef}, t_{aft} \in \mathbb{R}^c$，拼接后生成 RTE 特征 $f_{rte}$

设计动机：现有变化检测器虽然能有效捕捉外观差异，但缺乏基于相对属性和空间上下文的细粒度上下文推理能力。

#### 2. **图像-文本双重对齐模块（ITDA）**

核心思路：RTE 提取的文本特征与视觉特征处于不同的潜在空间，ITDA 通过场景内和跨场景两种对齐策略统一两种模态的互补优势。

**静态对齐（Static Alignment）**：增强场景内组合理解
- 将同一场景的视觉特征与对应的组合推理句子特征进行交叉注意力对齐
- 对"前"场景：$f_{bef}^{s(t \to i)} = \frac{1}{N}\sum_{n=1}^{N} \text{Attn}(t_{bef}^n, f_{bef}, f_{bef})$
- 同时计算自注意力视觉特征 $f_{bef}^{s(i \to i)}$ 用于语义一致性约束
- 静态对齐损失：$\mathcal{L}_{sa} = \frac{1}{2}(\|f_{bef}^{s(t \to i)} - f_{bef}^{s(i \to i)}\|_2^2 + \|f_{aft}^{s(t \to i)} - f_{aft}^{s(i \to i)}\|_2^2)$

**动态对齐（Dynamic Alignment）**：捕捉跨场景变化
- 将一个场景的视觉特征与另一个场景的文本特征进行交叉注意力
- 对"前"场景：$f_{bef}^{d(t \to i)} = \frac{1}{M}\sum_{m=1}^{M} \text{Attn}(t_{aft}^m, f_{bef}, f_{bef})$
- 同样计算跨场景视觉注意力特征用于约束
- 动态对齐损失：$\mathcal{L}_{da} = \frac{1}{2}(\|f_{bef}^{d(t \to i)} - f_{bef}^{d(i \to i)}\|_2^2 + \|f_{aft}^{d(t \to i)} - f_{aft}^{d(i \to i)}\|_2^2)$

最终将所有静态和动态增强特征拼接为 $f_{itda}$。

#### 3. **Transformer 解码器生成描述**

模型将所有模块的输出（$f_{icd}$, $f_{rte}$, $f_{itda}$）整合，通过 Transformer 解码器生成变化描述。

### 损失函数 / 训练策略

总损失函数：$\mathcal{L}_{total} = \mathcal{L}_{cap} + \lambda \mathcal{L}_{align}$

- $\mathcal{L}_{cap}$：标准描述生成损失
- $\mathcal{L}_{align} = \mathcal{L}_{sa} + \mathcal{L}_{da}$：对齐损失
- $\lambda$ 根据基线方法调整：SCORER 用 $10^{-3}$，SMART/DIRL 用 $10^{-4}$
- 单卡 RTX 4090 训练

## 实验关键数据

### 主实验

在 CLEVR-Change 数据集上的总体性能（Total Performance）：

| 方法 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|------|--------|--------|---------|-------|-------|
| SCORER (ICCV'23) | 56.3 | 41.2 | 74.5 | 126.8 | 33.3 |
| **CORTEX (SCORER)** | **57.0** | **42.7** | **75.9** | **128.8** | **33.9** |
| SMART (TPAMI'24) | 56.1 | 40.8 | 74.2 | 127.0 | 33.4 |
| **CORTEX (SMART)** | **56.5** | **42.1** | **75.7** | **130.2** | **34.0** |
| DIRL (ECCV'24) | 55.5 | 40.8 | 73.4 | 125.3 | 33.4 |
| **CORTEX (DIRL)** | **57.4** | **43.0** | **76.2** | **130.7** | **34.2** |

在 Spot-the-Diff 真实场景数据集上：

| 方法 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|------|--------|--------|---------|-------|-------|
| DIRL (ECCV'24) | 10.3 | 13.8 | 32.8 | 40.9 | 19.9 |
| **CORTEX (DIRL)** | **11.6** | **13.9** | **33.4** | **49.5** | **21.4** |

### 消融实验

| 配置 | BLEU-4 | METEOR | CIDEr | 说明 |
|------|--------|--------|-------|------|
| Baseline (DIRL) | 55.5 | 40.8 | 125.3 | 纯视觉基线 |
| + RTE | 55.8 | 41.6 | 128.5 | 仅加文本提取模块 |
| + RTE + ITDA | **57.4** | **43.0** | **130.7** | 全模块 |

对齐损失消融（CLEVR-Change）：

| $\mathcal{L}_{sa}$ | $\mathcal{L}_{da}$ | BLEU-4 | CIDEr |
|---|---|--------|-------|
| ✗ | ✗ | 56.6 | 127.9 |
| ✓ | ✗ | 56.3 | 128.4 |
| ✗ | ✓ | 56.6 | 128.9 |
| ✓ | ✓ | **57.4** | **130.7** |

### 关键发现

1. **即插即用有效性**：CORTEX 可适配三种不同基线方法，均获得一致提升
2. **组合推理优于通用描述**：专门设计的组合推理提示比通用描述提示效果更好（CIDEr 130.7 vs 129.5）
3. **双重对齐互补**：静态和动态对齐各有贡献，联合使用效果最佳
4. **VLM 鲁棒性**：更换 VLM（LLaVA 或 InternVL2）均有效，说明方法不依赖特定 VLM
5. **辅助上下文优于直接预测**：使用 VLM 为单张图像生成辅助文本远优于直接让 VLM 比较图像对

## 亮点与洞察

- **即插即用设计**：RTE 和 ITDA 模块可无缝嵌入任何现有的图像级变化检测器，无需改变原有模型架构
- **文本作为结构化推理载体**：巧妙利用 VLM 将图像中隐含的组合语义转化为显式的文本表示，弥补视觉特征在结构化推理上的不足
- **双重对齐思想**：静态对齐增强场景内理解（"这个场景里有什么"），动态对齐强调跨场景差异（"两个场景有什么不同"），设计合理

## 局限与展望

- RTE 模块依赖冻结的 VLM 进行推理，增加了额外的预处理时间（虽然作者提供了离线提取的文本数据）
- 在真实场景数据集（Spot-the-Diff）上的提升相比合成数据集（CLEVR-Change）更有限，可能因为真实场景的变化更加复杂多样
- 提示词的设计对结果有显著影响，但论文中对提示词工程的系统性探索不够深入

## 相关工作与启发

- **DIRL (ECCV'24)**、**SCORER (ICCV'23)**、**SMART (TPAMI'24)** 作为基线方法，代表了纯视觉变化描述的 SOTA
- **InternVL2** 作为 VLM 提取文本线索的工具，表明了大型 VLM 在辅助下游视觉任务中的潜力
- 启发：在其他需要结构化推理的视觉任务中（如 VQA、场景图生成），也可以考虑用 VLM 生成显式的文本推理线索作为辅助输入

## 评分

- 新颖性: ⭐⭐⭐⭐ — 用文本组合推理增强变化描述是一个新颖且合理的思路
- 实验充分度: ⭐⭐⭐⭐⭐ — 三个数据集、三个基线、详细消融、多种VLM对比
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐ — 即插即用的设计实用性强，代码已开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] DisCode: Distribution-Aware Score Decoder for Robust Automatic Evaluation of Image Captioning](discode_distribution-aware_score_decoder_for_robust_automatic_evaluation_of_imag.md)
- [\[CVPR 2026\] ConeSep: Cone-based Robust Noise-Unlearning Compositional Network for Composed Image Retrieval](../../CVPR2026/multimodal_vlm/conesep_cone-based_robust_noise-unlearning_compositional_network_for_composed_im.md)
- [\[ICML 2026\] Self-Captioning Multimodal Interaction Tuning: Amplifying Exploitable Redundancies for Robust Vision Language Models](../../ICML2026/multimodal_vlm/self-captioning_multimodal_interaction_tuning_amplifying_exploitable_redundancie.md)
- [\[AAAI 2026\] BiPrompt: Bilateral Prompt Optimization for Visual and Textual Debiasing in Vision-Language Models](biprompt_bilateral_prompt_optimization_for_visual_and_textual_debiasing_in_visio.md)
- [\[ICML 2025\] Toward Robust Hyper-Detailed Image Captioning: A Multiagent Approach and Dual Evaluation Metrics for Factuality and Coverage](../../ICML2025/multimodal_vlm/toward_robust_hyper-detailed_image_captioning_a_multiagent_approach_and_dual_eva.md)

</div>

<!-- RELATED:END -->
