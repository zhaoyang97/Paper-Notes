---
title: >-
  [论文解读] HEIE: MLLM-Based Hierarchical Explainable AIGC Image Implausibility Evaluator
description: >-
  [CVPR 2025][多模态VLM][AIGC图像评估] 提出HEIE——基于多模态大语言模型（MLLM）的层次化可解释AIGC图像不合理性评估器，通过CoT驱动的三位一体评估器同时输出热力图、评分和文字解释，并用自适应层次化不合理性映射器实现全局-局部缺陷的精准定位，在RichHF-18K和AbHuman数据集上达到SOTA。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "AIGC图像评估"
  - "多模态大语言模型"
  - "可解释性"
  - "缺陷热力图"
  - "Chain-of-Thought"
---

# HEIE: MLLM-Based Hierarchical Explainable AIGC Image Implausibility Evaluator

**会议**: CVPR 2025  
**arXiv**: [2411.17261](https://arxiv.org/abs/2411.17261)  
**代码**: [https://yfthu.github.io/HEIE/](https://yfthu.github.io/HEIE/) (项目页面)  
**领域**: 多模态VLM  
**关键词**: AIGC图像评估, 多模态大语言模型, 可解释性, 缺陷热力图, Chain-of-Thought

## 一句话总结

提出HEIE——基于多模态大语言模型（MLLM）的层次化可解释AIGC图像不合理性评估器，通过CoT驱动的三位一体评估器同时输出热力图、评分和文字解释，并用自适应层次化不合理性映射器实现全局-局部缺陷的精准定位，在RichHF-18K和AbHuman数据集上达到SOTA。

## 研究背景与动机

**领域现状**：AIGC图像生成技术（如Stable Diffusion、DALL·E 3）快速发展，但生成图像常存在伪影、不自然纹理、结构错误等质量问题。当前评估方法主要输出标量分数，少数工作（如RichHF）开始预测缺陷区域热力图。

**现有痛点**：(1) 专用小模型（如RAHF）缺乏可解释性——它们能定位缺陷但无法解释"为什么这里有问题"，用户难以理解和改进；(2) 专用模型缺乏常识和逻辑推理能力，训练数据有限导致泛化能力差。另一方面，直接使用MLLM（如GPT-4o）也面临困难：(1) 难以精确定位细粒度缺陷（如眼角、手指等微小区域）；(2) 无法输出像素级热力图，通常只能输出文本。

**核心矛盾**：专用小模型擅长像素级定位但缺乏理解和解释能力；MLLM擅长理解和推理但缺乏精细定位和像素级输出能力。如何结合两者优势？

**本文目标** (1) 让MLLM能够输出像素级不合理性热力图；(2) 实现热力图、评分、文字解释三者的协同输出；(3) 精确定位全局大缺陷和局部微小缺陷。

**切入角度**：设计特殊的[MAP]和[SCORE] token将MLLM的高级语义理解"桥接"到像素级输出，用CoT将复杂评估任务分解为由简到难的子任务链，让热力图、评分和解释相互增强。

**核心 idea**：通过在MLLM中引入特殊token和层次化映射器，实现CoT驱动的热力图+评分+解释三位一体的可解释AIGC图像缺陷评估。

## 方法详解

### 整体框架

输入AIGC图像，经过ViT提取图像特征，送入LLM（基于InternVL-8B）按照CoT流程依次进行：图像描述→问题区域识别→[MAP] token注入→问题分析→[SCORE] token注入。[MAP] token特征与图像特征通过Adaptive Hierarchical Implausibility Mapper生成热力图，[SCORE] token特征结合热力图通过Verisimilitude Scorer输出评分，LLM同时输出文字解释。

### 关键设计

1. **Adaptive Hierarchical Implausibility Mapper（自适应层次化不合理性映射器）**:

    - 功能：从MLLM生成像素级缺陷热力图，同时处理全局和局部缺陷
    - 核心思路：分三层设计。**基础映射器**：在LLM中定义特殊[MAP] token，提取其最后一层隐藏状态特征 $T$，与ViT图像特征 $F$ 通过两层双向交叉注意力（T→F再F→T）融合，生成热力图。**层次化映射**：图像按分辨率自适应分割为 $N$ 个patch，图像编码器分别处理缩略图（全局特征 $F_g$）和各patch（局部特征 $F_i$），LLM输出 $N$ 个局部[MAP] token和1个全局[MAP] token，分别生成局部热力图 $H_l$（拼接各patch热力图）和全局热力图 $H_g$。**自适应融合**：将两个热力图建模为Laplace分布，通过不确定性估计 $p_{uncertainty} = e^{-\sigma}$ 作为权重，自适应融合全局和局部热力图。
    - 设计动机：AIGC图像缺陷可以是全局性的（如多出一条腿）或局部性的（如手指畸形），需要不同粒度的检测。基于不确定性的融合让模型自己决定信任全局还是局部预测。

2. **CoT-Driven Explainable Trinity Evaluator（CoT驱动的可解释三位一体评估器）**:

    - 功能：通过链式推理协同生成热力图、评分和文字解释
    - 核心思路：设计五步CoT流程——(1) 图像描述：LLM描述图像关键元素；(2) 问题区域识别：基于描述定位潜在问题区域；(3) [MAP] token：基于前述分析注入缺陷信息到映射器；(4) 问题分析：基于定位结果给出详细文字解释（类型、原因等）；(5) [SCORE] token：基于综合理解注入整体评分信息。这种由简到难的任务分解充分利用了LLM的逐步推理能力，每一步的输出都为后续步骤提供上下文。
    - 设计动机：直接让LLM一次性完成复杂评估任务效果差。CoT分解让热力图、分析和评分相互关联、相互增强——文字描述提供语义上下文指导热力图生成，热力图的视觉显著性帮助评分量化，评分反过来校准热力图关注点。

3. **Verisimilitude Scorer（真实度评分器）**:

    - 功能：预测图像整体真实度分数
    - 核心思路：LLM中定义[SCORE] token，提取其隐藏状态经FFN回归初始分数 $S_{token}$。同时将预测热力图通过卷积+FFN提取热力图分数 $S_{map}$。最终分数 $S = Calib(S_{token}, S_{map})$ 通过校准函数融合两者。
    - 设计动机：LLM对数值输出不敏感（直接输出数字分数效果差），通过回归特殊token的隐藏状态可以更准确地编码评分信息。热力图与评分强相关，两者融合提升精度。

### 损失函数 / 训练策略

热力图使用focal loss解决正负样本不平衡问题。层次化映射器的两个热力图分别用Laplace分布的负对数似然训练：$\min_{H,\sigma} (\frac{\sqrt{2}}{\sigma}|H - H^{gt}| + \log(\sigma))$，同时学习预测值和不确定性。基于InternVL-8B，使用DeepSpeed微调，学习率 $3 \times 10^{-4}$，warmup ratio 0.03，batch size 16。

## 实验关键数据

### 主实验

**RichHF-18K数据集热力图+评分：**

| 方法 | MSE (All) ↓ | KLD ↓ | CC ↑ | AUC-Judd ↑ | PLCC ↑ | SRCC ↑ |
|------|-------------|-------|------|-----------|--------|--------|
| CLIP encoder (fine-tuned) | 0.01437 | 2.462 | 0.251 | 0.747 | 0.390 | 0.378 |
| RAHF (augmented) | 0.00920 | 1.652 | 0.556 | 0.913 | 0.693 | 0.681 |
| **HEIE (ours)** | **0.00825** | **1.634** | **0.574** | **0.915** | **0.697** | **0.683** |

**文字解释质量（Expl-AIGI-Eval）：**

| 方法 | GPT-4o Eval ↑ | Human Eval ↑ |
|------|--------------|-------------|
| GPT-4o | 3.828 | 3.999 |
| Claude-3.5-Sonnet | 3.938 | 4.081 |
| **HEIE (ours)** | **4.582** | **4.353** |

### 消融实验

**层次化映射器消融（RichHF-18K）：**

| 配置 | MSE ↓ | KLD ↓ | CC ↑ |
|------|-------|-------|------|
| 仅全局token | 0.01071 | 1.950 | 0.502 |
| 仅局部token | 0.00980 | 1.921 | 0.504 |
| 全局+局部，固定权重 | 0.00954 | 1.874 | 0.511 |
| 全局+局部，可学习权重 | 0.00873 | 1.680 | 0.557 |
| 全局+局部，不确定性自适应 | **0.00825** | **1.634** | **0.574** |

**CoT系统消融：**

| 配置 | 热力图MSE ↓ | 热力图CC ↑ | 评分PLCC ↑ |
|------|-----------|----------|-----------|
| w/o CoT Text | 0.00913 | 0.553 | 0.669 |
| w/ CoT Text | 0.00825 | 0.574 | 0.697 |
| w/ GT CoT Text | 0.00792 | 0.580 | 0.701 |

### 关键发现

- 不确定性自适应融合显著优于固定权重（CC: 0.574 vs 0.511），说明不同图像需要不同的全局/局部权重
- CoT文字推理反过来提升了热力图和评分的精度（MSE: 0.00825 vs 0.00913），证明三个输出确实相互增强
- HEIE的文字解释评分超越了GPT-4o和Claude-3.5-Sonnet（4.582 vs 3.828/3.938），说明针对任务的CoT微调比通用大模型更有效
- 零样本跨域泛化实验中，HEIE显著优于小模型baseline，验证了MLLM的常识知识优势

## 亮点与洞察

- **特殊token桥接文本与像素**：通过[MAP]和[SCORE]特殊token，巧妙地让只能输出文本的LLM"间接"输出像素级热力图和回归分数。这种设计模式可以推广到其他需要MLLM输出非文本模态的任务。
- **由简到难的CoT任务分解**：五步渐进式推理链让每步的输出自然地为下一步提供上下文，尤其是"先定位再解释"的设计符合人类认知流程。三者相互增强而非独立预测是核心创新。
- **Expl-AIGI-Eval数据集构建管线**：用Visual Prompting+LLM自由输出+ICL格式化的三阶段流水线构建解释性标注，可复用到其他需要精细标注的任务。

## 局限与展望

- 基于InternVL-8B，推理成本较高，难以实时评估大量生成图像
- 热力图预测依赖ViT的图像特征分辨率，对极小缺陷（如1-2像素级瑕疵）可能仍不够精细
- Expl-AIGI-Eval数据集标注依赖Claude-3.5和GPT-4o，标注质量受限于这些模型的能力
- 未探索视频AIGC内容的时序一致性评估

## 相关工作与启发

- **vs RAHF**: RAHF是专用小模型，预测热力图但无法解释；HEIE利用MLLM的推理能力实现可解释评估，且通过MLLM的常识获得更好的零样本泛化
- **vs GPT-4o直接评估**: GPT-4o能理解和解释但无法输出像素级热力图且对微小缺陷不敏感；HEIE通过特殊token和层次化映射器解决了这两个问题
- **与SAM等分割模型的潜在结合**：HEIE的[MAP] token机制类似于SAM的prompt-based分割，未来可探索结合两者

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次用MLLM做可解释的AIGC图像缺陷热力图预测，CoT三位一体设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 两个数据集+零样本泛化+详尽消融+人类评估，非常充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题定义明确
- 价值: ⭐⭐⭐⭐ 对AIGC图像质量评估和生成模型改进有直接指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MLLM-as-a-Judge for Image Safety without Human Labeling](mllm-as-a-judge_for_image_safety_without_human_labeling.md)
- [\[ICCV 2025\] AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models](../../ICCV2025/multimodal_vlm/aigi-holmes_towards_explainable_and_generalizable_ai-generated_image_detection_v.md)
- [\[CVPR 2025\] Efficient Motion-Aware Video MLLM](efficient_motion-aware_video_mllm.md)
- [\[ACL 2025\] VF-Eval: Evaluating Multimodal LLMs for Generating Feedback on AIGC Videos](../../ACL2025/multimodal_vlm/vf_eval_aigc_video_feedback.md)
- [\[CVPR 2025\] V-Stylist: Video Stylization via Collaboration and Reflection of MLLM Agents](v-stylist_video_stylization_via_collaboration_and_reflection_of_mllm_agents.md)

</div>

<!-- RELATED:END -->
