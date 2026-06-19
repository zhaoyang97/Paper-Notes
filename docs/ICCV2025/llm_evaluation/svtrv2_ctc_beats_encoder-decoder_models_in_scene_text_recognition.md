---
title: >-
  [论文解读] SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition
description: >-
  [ICCV 2025][LLM评测][场景文字识别] 提出 SVTRv2，通过多尺寸resize策略（MSR）、特征重排模块（FRM）和语义引导模块（SGM）三大设计，让 CTC 模型首次在多场景基准上全面超越编码器-解码器方法，同时保持推理速度优势。 场景文字识别（STR）方法主要分两类： CTC-based：仅视觉模型…
tags:
  - "ICCV 2025"
  - "LLM评测"
  - "场景文字识别"
  - "CTC"
  - "不规则文本"
  - "语义引导"
  - "多尺寸resize"
---

# SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition

**会议**: ICCV 2025  
**arXiv**: [2411.15858](https://arxiv.org/abs/2411.15858)  
**代码**: [https://github.com/Topdu/OpenOCR](https://github.com/Topdu/OpenOCR)  
**领域**: LLM评测  
**关键词**: 场景文字识别, CTC, 不规则文本, 语义引导, 多尺寸resize

## 一句话总结

提出 SVTRv2，通过多尺寸resize策略（MSR）、特征重排模块（FRM）和语义引导模块（SGM）三大设计，让 CTC 模型首次在多场景基准上全面超越编码器-解码器方法，同时保持推理速度优势。

## 研究背景与动机

场景文字识别（STR）方法主要分两类：

**CTC-based**：仅视觉模型 + CTC 对齐的线性分类器，结构简单、推理快速，是商用 OCR 的主力，但在不规则文本上表现不佳。

**编码器-解码器（EDTR）**：利用注意力解码器处理多模态线索（视觉、语言、位置），精度高但速度慢。

CTC 模型的劣势源于两个核心问题：
- **不规则文本处理困难**：CTC 对齐假设文本近似水平排列，弯曲/旋转/透视文本违反此假设。现有方法将图像统一 resize 到固定尺寸（如 32×128），会对低宽高比文本造成严重形变。
- **缺乏语言上下文建模**：CTC 直接分类视觉特征，不编码语言信息。在遮挡、低质量场景下，缺少语言先验导致错误率高。

作者的目标是 **让 CTC 模型具备处理不规则文本和建模语言上下文的能力**，同时保持轻量推理架构。

## 方法详解

### 整体框架

SVTRv2 = 多尺寸 Resize（MSR）+ 三阶段视觉特征提取 + 特征重排模块（FRM）+ 语义引导模块（SGM，仅训练时使用）。推理时 SGM 被丢弃，模型仍为纯 CTC 架构。

### 关键设计

1. **多尺寸resize策略（MSR）**  
   根据文本图像宽高比 $R = W/H$ 分为 4 个桶：
    - $R < 1.5$: resize 到 [64, 64]
    - $1.5 \leq R < 2.5$: resize 到 [48, 96]
    - $2.5 \leq R < 3.5$: resize 到 [40, 112]
    - $R \geq 3.5$: resize 到 [32, $\lfloor R \rfloor \times 32$]

   前三个桶尺寸固定，可批量训练；第四个桶动态宽度，可处理长文本。**核心动机**：避免将低宽高比图像（如竖排文本）强行拉伸到固定宽图像，消除不必要的形变。实验显示 MSR 在 $R_1$ 桶上相比固定 32×128 提升 **15.3%**。

2. **特征重排模块（FRM）**  
   将 2D 视觉特征 $\mathbf{F} \in \mathbb{R}^{(H/8 \times W/4) \times D_2}$ 重排为符合 CTC 对齐的 1D 序列。分两步：
    - **水平重排**：对每行特征做自注意力，学习水平方向的重排矩阵 $\mathbf{M}_i^h$，使特征顺序对齐文本阅读方向。
    - **垂直重排**：引入 selecting token 通过交叉注意力与列特征交互，学习垂直重排矩阵 $\mathbf{M}_j^v$，从多行中选出最相关的特征。

   设计动机：旋转文本的字符排列不遵循水平阅读顺序，FRM 通过先水平后垂直的二步重排，对文本方向敏感，有效解决 CTC 对齐难题。实验表明两步结合在 Multi-Oriented 子集上提升 2.46%。

3. **语义引导模块（SGM）**  
   训练时利用字符标签的上下文字符串来引导视觉模型编码语言信息。具体地，对目标字符 $c_i$，取其左侧 $l_s$ 个字符和右侧 $l_s$ 个字符作为上下文，编码为隐表示 $\mathbf{Q}_i^l$，与视觉特征做交叉注意力得到注意力图 $\mathbf{A}_i^l$，再分类预测 $c_i$。**核心机制**：只有当视觉模型将上下文信息融入目标字符的视觉特征时，注意力图才能正确聚焦到目标字符位置；训练信号反向促使视觉模型学会编码语言上下文。推理时 **SGM 被完全丢弃**，不增加任何计算开销。

### 损失函数 / 训练策略

$$\mathcal{L} = \lambda_1 \mathcal{L}_{ctc} + \lambda_2 \mathcal{L}_{sgm}$$

- $\mathcal{L}_{ctc}$: 标准 CTC 损失，权重 $\lambda_1 = 0.1$
- $\mathcal{L}_{sgm}$: 左右两侧上下文预测的交叉熵均值，权重 $\lambda_2 = 1$
- 训练分两阶段：先不用 SGM 训练，再加 SGM 训练

## 实验关键数据

### 主实验（表格）

**Union14M-Benchmark 各子集准确率 + 速度（U14M-Filter 训练）**

| 方法 | 类型 | Curve | MO | Artistic | Com Avg | U14M Avg | FPS |
|------|------|-------|------|----------|---------|---------|-----|
| SVTR-B | CTC | 76.2 | 44.5 | 67.8 | 94.58 | 71.17 | 161 |
| MAERec | EDTR | 89.1 | 87.1 | 79.0 | 96.36 | 85.17 | 17.1 |
| PARSeq | EDTR | 87.6 | 88.8 | 76.5 | 96.40 | 84.26 | 52.6 |
| CPPD | EDTR | 86.2 | 78.7 | 76.5 | 96.40 | 81.91 | 125 |
| **SVTRv2-B** | **CTC** | **90.6** | **89.0** | **79.3** | **96.57** | **86.14** | **143** |

SVTRv2-B 在 U14M 上超越所有 EDTR（比 MAERec +0.97%），且推理速度 143 FPS，是 MAERec 的 **8 倍**。在 Curve 和 Multi-Oriented 上相比前代 SVTR 分别提升 **14.4% 和 44.5%**。

### 消融实验（表格）

**MSR 和 FRM 消融（U14M Curve + MO）**

| 配置 | Curve | MO | Com Avg | U14M Avg |
|------|-------|------|---------|---------|
| 无 MSR 无 FRM（Fixed 32×128） | 82.89 | 65.59 | 95.28 | 77.78 |
| + MSR | 87.35 | 83.73 | 95.44 | 82.22 |
| + FRM（H+V）| 88.05 | 85.76 | 95.98 | 82.94 |
| + MSR + FRM | **88.17** | **86.19** | **96.16** | **83.86** |
| + MSR + FRM + SGM | **90.64** | **89.04** | **96.57** | **86.14** |

MSR 贡献最大（+4.44 U14M），FRM 在 MO 上提升显著（+2.46），SGM 在遮挡场景提升最突出（OST +5.11）。

### 关键发现

- SGM 在遮挡场景（OST Avg）上带来 5.11% 的绝对提升，远超 GTC、ABINet、VisionLAN 等替代方案。
- 中文识别上 SVTRv2-B 也达到 SOTA（83.31 Avg），且支持长文本（SceneL>25 准确率 52.8%）。
- 加入预训练后，SVTRv2 在 Common Benchmarks 上达到 97.83%，超越 CLIP4STR 等大规模预训练方法，参数量仅为其 14%。

## 亮点与洞察

- **CTC 首次全面击败 EDTR**：在速度和精度上同时胜出，颠覆了 STR 领域"CTC 不如 EDTR"的认知。
- SGM 的"训练时用、推理时丢"设计极为巧妙：通过训练信号间接迫使视觉模型学习语言上下文，零推理成本地提升精度。
- MSR 的简洁性令人印象深刻：仅需按宽高比分桶、选合适尺寸，即可消除大部分不规则文本问题。
- 额外贡献：构建了去重后的 U14M-Filter 数据集并重训 24 种方法，为 STR 提供公平可靠的新基准。

## 局限与展望

- SGM 需要字符级标注，对 word-level 或 line-level 标注的扩展需要调整。
- MSR 的桶划分为手工设计（4 个固定尺寸），自适应学习尺寸可能更优。
- 训练最大长度限制为 25 个字符（英文），超长文本仍有局限。
- FRM 引入了额外的注意力计算，对极端低延迟场景可能需要简化。

## 相关工作与启发

- SVTR [IJCAI 2022] 是本文的前代，主要改进为引入 MSR/FRM/SGM 三个模块。
- GTC [AAAI 2020] 最早提出用辅助解码器引导 CTC 训练，SGM 可视为更精细的语义版本。
- 本文思路可迁移至其他序列识别任务（如语音识别、手势识别等 CTC 框架下的场景）。

## 评分

- 新颖性: ⭐⭐⭐⭐ （三个模块各有创新，整体上首次让 CTC 全面超越 EDTR）
- 实验充分度: ⭐⭐⭐⭐⭐ （6+ 基准，24 种方法重训对比，含中文/长文本/遮挡/预训练多场景）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，实验详实）
- 价值: ⭐⭐⭐⭐⭐ （对 OCR 社区意义重大，附带公平基准和开源代码）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Beyond Fixed Psychological Personas: State Beats Trait, but Language Models are State-Blind](../../ACL2026/llm_evaluation/beyond_fixed_psychological_personas_state_beats_trait_but_language_models_are_st.md)
- [\[ECCV 2024\] EvSign: Sign Language Recognition and Translation with Streaming Events](../../ECCV2024/llm_evaluation/evsign_sign_language_recognition_and_translation_with_streaming_events.md)
- [\[ACL 2026\] Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models](../../ACL2026/llm_evaluation/attribution_citation_and_quotation_a_survey_of_evidence-based_text_generation_wi.md)
- [\[ACL 2026\] Gated Tree Cross-Attention for Checkpoint-Compatible Syntax Injection in Decoder-Only LLMs](../../ACL2026/llm_evaluation/gated_tree_cross-attention_for_checkpoint-compatible_syntax_injection_in_decoder.md)
- [\[ACL 2025\] EditInspector: A Benchmark for Evaluation of Text-Guided Image Edits](../../ACL2025/llm_evaluation/editinspector_a_benchmark_for_evaluation_of_text-guided_image_edits.md)

</div>

<!-- RELATED:END -->
