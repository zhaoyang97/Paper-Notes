---
title: >-
  [论文解读] Rethinking Video-Text Understanding: Retrieval from Counterfactually Augmented Data
description: >-
  [ECCV 2024][视频理解][video-text understanding] 提出反事实增强数据检索（RCAD）任务和 Feint6K 数据集，揭示 SOTA 视频文本模型在动作语义理解上远落后于人类（InternVideo 58.2% vs 人类 95.2%），并提出 LLM-teacher 通过 LLM 知识蒸馏改善动作嵌入学习。
tags:
  - "ECCV 2024"
  - "视频理解"
  - "video-text understanding"
  - "counterfactual augmentation"
  - "action semantics"
  - "对比学习"
  - "LLM-teacher"
---

# Rethinking Video-Text Understanding: Retrieval from Counterfactually Augmented Data

**会议**: ECCV 2024  
**arXiv**: [2407.13094](https://arxiv.org/abs/2407.13094)  
**代码**: 有（项目页面）  
**领域**: 视频理解 / 视觉-语言  
**关键词**: video-text understanding, counterfactual augmentation, action semantics, contrastive learning, LLM-teacher

## 一句话总结
提出反事实增强数据检索（RCAD）任务和 Feint6K 数据集，揭示 SOTA 视频文本模型在动作语义理解上远落后于人类（InternVideo 58.2% vs 人类 95.2%），并提出 LLM-teacher 通过 LLM 知识蒸馏改善动作嵌入学习。

## 研究背景与动机
**领域现状**：视频-文本基础模型（如 InternVideo、LanguageBind）在标准检索任务上取得了出色结果（R@1 达 87.9%），被认为已具备较强的视频理解能力。

**现有痛点**：标准评估任务存在严重的捷径（shortcuts）和偏差（biases）。许多问题仅靠单帧的物体和上下文就能回答——如看到"cymbal"就知道是"打钹"，看到户外场景就猜"football"。

**核心矛盾**：现有评估无法区分模型是否真正理解了跨帧的动作语义，还是仅在利用物体和场景的捷径。视频理解相比图像理解的核心增量——跨帧推理和动作语义——被现有基准掩盖。

**本文目标**：(1) 设计一个消除捷径的评估范式来暴露模型短板；(2) 改善模型对动作语义的学习。

**切入角度**：通过人工标注反事实修改的描述——保留相同物体和场景，仅改变动作——迫使模型必须进行跨帧推理。利用 LLM 知识注入更有效的动作对比学习。

**核心 idea**：反事实增强消除物体捷径后暴露模型动作理解不足，LLM-teacher 通过合成负样本和软标签蒸馏改善动作嵌入。

## 方法详解

### 整体框架
本文包含两部分：(1) 评估框架——RCAD 任务和 Feint6K 数据集，评估模型在消除捷径后的真实动作理解能力；(2) 改进方法——LLM-teacher，通过 LLM 生成动作变体的负样本描述，用软标签对比学习增强动作表征。

### 关键设计

1. **RCAD 任务设计（Retrieval from Counterfactually Augmented Data）**

    - 功能：给定视频和一组候选描述（1 正 5 负），检索语义匹配的描述。负样本与正样本包含相同物体，仅动作不同
    - 核心思路：负样本是反事实修改的——保持文本结构和物体实体不变，仅替换动作词。例如正样本"A man kicks a football"，负样本可能是"A man catches a football"
    - 设计动机：消除基于物体的捷径，迫使模型进行跨帧推理理解动作语义
    - 支持零样本评估，无需下游微调

2. **Feint6K 数据集构建**

    - 功能：构建高质量的反事实增强视频-文本评估集
    - 核心思路：采用 human-in-the-loop 系统，40 名标注者手动修改 MSR-VTT 和 VATEX 验证集中的动作描述
        - 新动作必须在上下文中合理但视频中未发生
        - 训练阶段给标注者示范和反馈
        - 每条标注需审核，不合格退回修改
    - 规模：6,243 个视频，来源于 MSR-VTT 验证集和 VATEX 测试集
    - 人类基线 R@1 达 95.2%（MSR-VTT）和 96.8%（VATEX），证明任务可解且有唯一正确答案

3. **LLM-teacher 方法**

    - 功能：通过 LLM 知识改善视频-文本模型的动作嵌入学习
    - 核心思路分三步：
        - **合成负样本生成**：对原始描述用 AMR 解析器提取动作/物体 token，然后用两种方法生成变体描述：
       - Method I — Mask Filling：用 XLM-RoBERTa 的 MLM 能力预测替代动作词
       - Method II — LLM Chatbot：利用 LLM 的 in-context learning 生成更灵活的替换（可修改介词等）
        - **对比学习**：用合成负样本做对比，损失为温度缩放交叉熵：
    $l = -\log \frac{\exp(\text{sim}(f_v, f_p)/\tau)}{\exp(\text{sim}(f_v, f_p)/\tau) + \sum_{i=1}^{k}\exp(\text{sim}(f_v, f_{n_i})/\tau)}$
        - **LLM 软标签蒸馏**：某些合成负样本语义与原描述相似，不应严格为负。使用 Sentence-BERT 计算描述间相似度作为 LLM 教师的软标签，用 KL 散度对齐模型输出：
    $l = \mathcal{L}_{\text{KL}}(z_{\text{video-text}}, z_{\text{LLM}})$
    - 默认对每个视频生成 10 个基于动作的合成描述
    - 设计动机：标准对比学习中物体是捷径——模型只需区分"cymbal"和"football"就能最小化对比损失，从不真正学习动作嵌入

### 损失函数 / 训练策略
- 二元伪标签版本（LLM-teacher-lbl）：标准交叉熵对比损失
- 软标签版本（LLM-teacher-lgt）：KL 散度与 LLM 教师的软分布对齐
- 应用于 SimVTP 和 InternVideo 两个预训练模型

## 实验关键数据

### 主实验 — 标准检索 vs RCAD

| 模型 | MSR-VTT R@1 | Feint6K R@1 | 差距 | 人类 R@1 |
|------|-------------|-------------|------|----------|
| CLIP (零样本) | 26.3 | 37.3 | — | 95.2 |
| InternVideo (零样本) | 37.5 | 45.8 | -8.3 | 95.2 |
| InternVideo (微调) | 49.1 | 58.6 | +9.5 | 95.2 |
| LanguageBind (零样本) | 42.8 | 41.3 | -1.5 | 95.2 |
| SimVTP (微调) | 50.2 | 35.7 | -14.5 | 95.2 |
| **+ LLM-teacher-lgt** | 49.5 | **43.5** | +7.8 | — |
| InternVideo (微调) | 49.1 | 58.6 | — | 95.2 |
| **+ LLM-teacher-lgt** | 48.9 | **65.8** | +7.2 | — |

### VATEX 子集结果

| 模型 | VATEX R@1 | Feint6K R@1 | 人类 R@1 |
|------|-----------|-------------|----------|
| InternVideo (微调) | 87.9 | 58.2 | 96.8 |
| + LLM-teacher-lgt | 87.3(-0.6) | **65.6**(+7.4) | — |
| SimVTP (微调) | 76.6 | 33.6 | 96.8 |
| + LLM-teacher-lgt | 75.3(-1.3) | **40.1**(+6.5) | — |

### 消融实验

| 配置 | VATEX R@1 | Feint6K R@1 | 说明 |
|------|-----------|-------------|------|
| DefaultGP（10 动作描述，XLM-RoBERTa）| 87.3 | **65.6** | 默认配置 |
| 5 动作描述 | 87.6 | 64.7 | -0.9，更多负样本更好 |
| 5 动作 + 5 物体描述 | 87.5 | 64.2 | 物体负样本无帮助 |
| LLM Chatbot 替代 | 87.0 | 65.9 | 略好但推理慢 |

### 关键发现
- InternVideo 标准检索 87.9% → RCAD 58.2%，暴跌 29.7%，远落后人类 38.6%
- 对物体更换的余弦相似度变化 $|\Delta s|$ 远大于对动作更换的 $|\Delta s|$，证明模型对物体的嵌入远比动作更有区分度
- LLM-teacher-lgt（软标签）优于 LLM-teacher-lbl（硬标签），因为某些合成负样本语义上接近正样本
- LLM-teacher 在标准检索上仅下降 0.2-0.6%，但 RCAD 提升 7.2-7.4%
- 物体负样本不帮助改善 RCAD，验证了模型已有良好的物体嵌入，缺的是动作嵌入

## 亮点与洞察
- **评估范式的贡献**：通过反事实增强消除捷径，暴露了 SOTA 视频文本模型在动作理解上的根本性不足。87.9%→58.2% 的暴跌令人警醒，说明标准基准的高分很大程度来自物体匹配而非动作理解。
- **捷径学习的深刻分析**：对比学习中物体是天然捷径——CLIP 预训练已给了模型优秀的物体嵌入，在视频-文本对比中模型只需区分物体就能最小化损失，无需学习动作语义。这个分析深刻且有实验佐证（$\Delta s$ 分析）。
- **LLM-teacher 的优雅设计**：不修改模型架构，仅改变训练数据和目标。软标签蒸馏比硬标签更好，因为"踢球"和"扔球"虽然不同但语义接近。

## 局限与展望
- Feint6K 仅基于 MSR-VTT 和 VATEX，视频多样性有限
- RCAD 每个视频仅 6 个候选（1 正 5 负），增加候选数可能更具区分度
- LLM-teacher 在标准检索上有轻微下降（0.2-0.6%），存在 trade-off
- 未探索视频编码器端的改进（如更好的时序建模），仅从训练目标角度优化
- 人类基线 95.2% 而非 100%，部分反事实场景可能存在歧义

## 相关工作与启发
- **vs InternVideo**：InternVideo 在 7 个大规模数据集上预训练，标准检索 87.9%，但 RCAD 仅 58.2%。LLM-teacher 在不改架构的情况下提升到 65.8%，说明问题不在模型容量而在训练目标
- **vs CLIP4Clip/ViCLIP**：这些模型扩展 CLIP 到视频域，但都继承了 CLIP 的物体偏置，在 RCAD 上表现类似地差
- **vs Counterfactual VQA**：本文借鉴了 NLP 中的反事实数据增强思想，但首次系统性地应用于视频-文本理解评估

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 反事实评估范式揭示了领域盲点，LLM-teacher 思路简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型评估、人类基线、余弦相似度分析、消融全面
- 写作质量: ⭐⭐⭐⭐⭐ 动机推导环环相扣，从评估到分析到方法逻辑清晰
- 价值: ⭐⭐⭐⭐⭐ 对领域有警醒作用，RCAD 可能成为新标准评估

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Text-Guided Video Masked Autoencoder](text-guided_video_masked_autoencoder.md)
- [\[ECCV 2024\] Data Collection-Free Masked Video Modeling](data_collection-free_masked_video_modeling.md)
- [\[NeurIPS 2025\] VGEnt: Graph-Based Retrieval-Reasoning-Augmented Generation for Long Video Understanding](../../NeurIPS2025/video_understanding/vgent_graph-based_retrieval-reasoning-augmented_generation_for_long_video_unders.md)
- [\[CVPR 2025\] Learning Audio-Guided Video Representation with Gated Attention for Video-Text Retrieval](../../CVPR2025/video_understanding/learning_audio-guided_video_representation_with_gated_attention_for_video-text_r.md)
- [\[NeurIPS 2025\] AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding](../../NeurIPS2025/video_understanding/adavideorag_omnicontextual_adaptive_retrievalaugmented_effic.md)

</div>

<!-- RELATED:END -->
