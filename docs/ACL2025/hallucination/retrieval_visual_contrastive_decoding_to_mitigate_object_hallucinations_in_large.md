---
title: >-
  [论文解读] Retrieval Visual Contrastive Decoding to Mitigate Object Hallucinations in Large Vision-Language Models
description: >-
  [ACL2025][幻觉检测][Object Hallucination] 提出 RVCD（Retrieval Visual Contrastive Decoding），通过检索 AI 生成的单概念显式图像构建正/负 logit 集合，在解码阶段抑制 LVLM 的物体幻觉（Object Hallucination），无需额外训练即可显著优于现有解码方法。
tags:
  - "ACL2025"
  - "幻觉检测"
  - "Object Hallucination"
  - "Contrastive Decoding"
  - "LVLM"
  - "图像检索"
  - "plug-and-play"
---

# Retrieval Visual Contrastive Decoding to Mitigate Object Hallucinations in Large Vision-Language Models

**会议**: ACL2025  
**arXiv**: [2505.20569](https://arxiv.org/abs/2505.20569)  
**作者**: Jihoon Lee, Min Song (Yonsei University, Onoma AI)
**代码**: [GitHub](https://github.com/JiHoonLee9898/RVCD)  
**领域**: 幻觉检测  
**关键词**: Object Hallucination, Contrastive Decoding, LVLM, 图像检索, plug-and-play

## 一句话总结

提出 RVCD（Retrieval Visual Contrastive Decoding），通过检索 AI 生成的单概念显式图像构建正/负 logit 集合，在解码阶段抑制 LVLM 的物体幻觉（Object Hallucination），无需额外训练即可显著优于现有解码方法。

## 研究背景与动机

**物体幻觉问题依然严峻**：大型视觉语言模型（LVLM）在生成文本描述时，经常出现三类物体幻觉——存在性幻觉（生成不存在的物体）、属性幻觉（错误描述物体属性）、关系幻觉（错误描述物体间关系），严重影响模型可靠性。

**现有对比解码方法的局限**：VCD、HALC 等先前方法仅对原始输入图像进行变换（如扭曲、裁剪局部视图）来生成调节 logit，未能充分挖掘 visual contrastive decoding 的潜力——用于调节的图像并不一定局限于原图变换。

**LVLM 自我检测 OH 能力不足**：实验发现 LVLM 通过 VQA 检测幻觉物体的精度远低于传统目标检测模型（如 YOLO），这表明可以借助 OD 模型的检测能力来辅助 LVLM 抑制幻觉。

## 方法详解

### 整体流程

RVCD 采用两阶段解码策略：

1. **草稿解码 + 目标检测**：先用 LVLM 对输入图像做贪心解码生成 draft caption，同时用 YOLO（YOLOv8x）对同一图像做目标检测，得到检测物体列表。
2. **对比识别正/负物体**：将 draft caption 中提到但 YOLO 未检测到的物体定义为负物体 N（疑似幻觉），两者都检测到的物体定义为正物体 P（真实存在）。
3. **检索显式图像**：从预构建的单概念图像数据库中，为 N 和 P 中的每个物体分别检索对应的参考图像。
4. **RVCD 调节解码**：在每个解码步骤 t，利用负/正图像生成对应的 logit 集合 Nt 和 Pt，通过公式调节原始 logit 来抑制幻觉物体并保留真实物体。

### 单概念图像数据库构建

- 使用 FLUX.1-dev 生成覆盖 CHAIR 字典中全部 400+ 词汇的单概念图像（prompt 格式："An/A {object}, white background"）
- 通过 LVLM（LLaVA-1.5）对生成图像做验证：只有当 LVLM 描述中包含目标物体时才入库，确保图像生成模型与 LVLM 之间的语义一致性
- 最终数据库将每个词汇映射到一张高质量单概念参考图像

### 核心公式

调节后的 logit 为：

$$f_{adjusted_t} = f_\theta(\cdot|v,x,y_{<t}) \cdot (1+\alpha \cdot len(N) - \beta \cdot len(P)) - (\alpha \cdot sum(N_t) - \beta \cdot sum(P_t))$$

其中 α 控制负 logit（幻觉抑制）的强度，β 控制正 logit（真实物体保护）的强度。最优设置为 α=1, β=0.1。

### β 参数与正 logit 的必要性

由于 LVLM 存在共现偏差（如看到叉子图像也会预测刀、勺子等），单纯减去负 logit 会误伤真实物体的表示。引入 β 和正 logit 用于恢复被误伤的真实物体信息，消融实验证实 β=0.1 时在 CHAIR 和 BLEU 上均有增益。

## 实验关键数据

### Table 1: CHAIR 和 BLEU 结果（MSCOCO，500 图 x 5 次采样）

| 方法 | LLaVA-1.5 CHAIR_S↓ | CHAIR_I↓ | BLEU↑ | MiniGPT-4 CHAIR_S↓ | CHAIR_I↓ | BLEU↑ |
|------|---------------------|----------|-------|---------------------|----------|-------|
| Greedy | 22.08 | 7.08 | 16.06 | 20.32 | 7.03 | 16.17 |
| VCD | 23.24 | 7.73 | 14.97 | 21.72 | 8.08 | 15.92 |
| HALC | 18.60 | 6.03 | 16.32 | 15.36 | 5.55 | 17.83 |
| OPERA | 18.72 | 6.56 | 16.65 | 19.44 | 7.22 | 17.77 |
| **RVCD** | **11.32** | **3.87** | 15.48 | **9.00** | **3.61** | 15.98 |

RVCD 在三个 backbone 上均大幅降低 CHAIR（CHAIR_S 降幅约 40-50%），同时 BLEU 仅有微弱下降，说明文本质量保持良好。

### Table 2: POPE 评估结果

| 方法 | LLaVA-1.5 Acc↑ | Prec↑ | F1↑ | mPLUG-Owl2 Acc↑ | Prec↑ | F1↑ |
|------|----------------|-------|-----|-----------------|-------|-----|
| Greedy | 72.19 | 65.28 | 77.86 | 74.36 | 67.23 | 79.23 |
| Beam Search | 78.27 | 71.94 | 81.28 | 80.17 | 74.30 | 82.64 |
| HALC | 72.48 | 65.54 | 78.04 | 74.54 | 67.42 | 79.33 |
| **RVCD** | **88.54** | **89.92** | **88.43** | **87.45** | **87.91** | **87.41** |

RVCD 在 POPE 的准确率、精度和 F1 上均大幅领先所有基线，提升约 10-20 个百分点。

### 延迟分析（Table 4）

| 方法 | 平均延迟 (s/token) | 相对倍数 |
|------|-------------------|---------|
| Greedy | 0.034 | 1.0x |
| OPERA | 0.341 | 10.1x |
| HALC | 0.800 | 23.8x |
| RVCD (β=0) | 0.143 | 4.2x |
| RVCD (β≠0) | 0.204 | 6.1x |

RVCD 的延迟远低于 OPERA 和 HALC，效率优势明显。

## 亮点

- **新颖的检索范式**：首次将外部显式图像检索引入 visual contrastive decoding，突破了仅对原图变换的思路限制
- **无需训练的即插即用**：可直接应用于任意开源 LVLM（MiniGPT-4、LLaVA-1.5、mPLUG-Owl2），不需要微调
- **正/负 logit 双向调节**：不仅抑制幻觉物体，还通过正 logit 保护真实物体不被误伤，设计巧妙
- **全面的消融分析**：系统研究了检测精度、α/β 参数、不同 OD 模型对性能的影响，结论扎实

## 局限与展望

1. **依赖 CHAIR 字典**：单概念图像数据库基于 MSCOCO 的有限字典（约400词），难以泛化到开放词汇场景
2. **延迟与 draft 长度正相关**：当 draft caption 提及大量不同物体时，需要为每个物体生成额外 logit，解码延迟增加
3. **依赖 OD 模型质量**：RVCD 性能与目标检测模型精度正相关（Table 3 验证），OD 模型的漏检/误检会直接影响效果
4. **仅关注物体级幻觉**：未处理属性和关系层面的幻觉（虽然在 CHAIR 框架下有所覆盖，但不够细粒度）

## 相关工作对比

| 方法 | 策略 | 是否需要训练 | 图像来源 |
|------|------|------------|---------|
| VCD | 扭曲原图生成对比 logit | 否 | 原图变换 |
| HALC | 裁剪包含关键物体的局部视图 | 否 | 原图裁剪 |
| OPERA | 惩罚注意力权重中的 over-trust | 否 | 无额外图像 |
| DoLA | 对比不同层的 logit | 否 | 无额外图像 |
| **RVCD** | 检索外部单概念图像做正/负对比 | 否 | 外部 AI 生成图像库 |

RVCD 的核心差异在于引入了与原图无关的外部显式图像，使对比解码的调节信号更加精确和可控。

## 评分

- 新颖性: ⭐⭐⭐⭐ (检索外部显式图像做对比解码是新颖的思路)
- 实验充分度: ⭐⭐⭐⭐ (CHAIR/POPE/MME/LLaVA-Bench + 3 个 backbone + 详细消融)
- 写作质量: ⭐⭐⭐⭐ (动机清晰，流程图直观，公式推导完整)
- 价值: ⭐⭐⭐⭐ (即插即用方法，效果显著，但受限于 CHAIR 字典的泛化性)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Visual Evidence Prompting Mitigates Hallucinations in Large Vision-Language Models](visual_evidence_prompting.md)
- [\[CVPR 2026\] First Logit Boosting: Visual Grounding Method to Mitigate Object Hallucination in Large Vision-Language Models](../../CVPR2026/hallucination/first_logit_boosting_visual_grounding_method_to_mitigate_object_hallucination_in.md)
- [\[ACL 2025\] Mixture of Decoding: An Attention-Inspired Adaptive Decoding Strategy to Mitigate Hallucination in Multimodal LLMs](mixture_of_decoding_an_attention-inspired_adaptive_decoding_strategy_to_mitigate.md)
- [\[CVPR 2026\] VES-RFT: Rewarding Visual Evidence Sensitivity to Mitigate Hallucinations in Large Vision-Language Models](../../CVPR2026/hallucination/ves-rft_rewarding_visual_evidence_sensitivity_to_mitigate_hallucinations_in_larg.md)
- [\[CVPR 2026\] PAS: Prelim Attention Score for Detecting Object Hallucinations in Large Vision-Language Models](../../CVPR2026/hallucination/pas_prelim_attention_score_for_detecting_object_hallucinations_in_large_vision-l.md)

</div>

<!-- RELATED:END -->
