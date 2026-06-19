---
title: >-
  [论文解读] Watch and Listen: Understanding Audio-Visual-Speech Moments with Multimodal LLM
description: >-
  [NeurIPS 2025][多模态VLM][音视频理解] 提出 TriSense——一个三模态（视觉+音频+语音）大语言模型，通过 Query-Based Connector 自适应调节各模态权重实现鲁棒的视频时序理解，并构建了包含200万标注的 TriSense-2M 数据集支撑训练。 人类理解视频时天然地整合视觉和听觉…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "音视频理解"
  - "多模态融合"
  - "时序理解"
  - "模态自适应"
  - "视频时刻检索"
---

# Watch and Listen: Understanding Audio-Visual-Speech Moments with Multimodal LLM

**会议**: NeurIPS 2025  
**arXiv**: [2505.18110](https://arxiv.org/abs/2505.18110)  
**代码**: [GitHub](https://github.com/zinuoli/TriSense)  
**领域**: 多模态VLM  
**关键词**: 音视频理解, 多模态融合, 时序理解, 模态自适应, 视频时刻检索

## 一句话总结

提出 TriSense——一个三模态（视觉+音频+语音）大语言模型，通过 Query-Based Connector 自适应调节各模态权重实现鲁棒的视频时序理解，并构建了包含200万标注的 TriSense-2M 数据集支撑训练。

## 研究背景与动机

人类理解视频时天然地整合视觉和听觉线索。例如，定位"一位科学家热情地讲述野生动物保护，背景音乐伴奏，观众点头鼓掌"这样的片段，需要同时处理视觉、音频和语音信号。然而，现有MLLM在音视频融合方面存在两个核心挑战：

**1. 训练数据不足且不完整**：现有数据集多由短片段组成，缺乏大规模、跨三模态的完整标注。更关键的是，现实视频中各模态并非总是同时存在——可能有静默画面、纯背景音乐、或某些场景天然缺少特定信号。当模型仅在三模态完整的数据上训练时，面对缺失模态就会严重退化。

**2. 缺乏模态自适应机制**：现有MLLM无法根据任务或查询上下文评估各模态的相对重要性。LongVALE将所有模态token压缩为单个表示，导致信息丢失且无法处理模态缺失；Qwen2.5-Omni虽引入时序位置编码，但在长视频细粒度时序任务上仍表现不佳。

这些问题促使作者构建能够灵活处理任意模态组合、并自适应调节模态贡献的模型。

## 方法详解

### 整体框架

TriSense 架构包含四个核心模块：(1) 三个专用编码器分别处理视觉、音频和语音；(2) 模态特定的投射器进行维度变换；(3) Query-Based Connector 基于查询自适应融合多模态特征；(4) LLM主干 + Time Encoder 生成时序对齐的输出。模型支持8种任务配置（AVS/VS/AV/V × 片段描述/时刻检索）。

### 关键设计

1. **多模态信息提取**：从视频中均匀采样 $n=64$ 帧，对每帧记录时间戳并提取前后±1秒的音频片段。使用预训练专家编码器分别提取视觉token $f^v_i$、音频token $f^a_i$、语音token $f^s_i$。通过 Slot-Based Compression 将每种模态的token压缩到固定16个token，以控制计算开销。时间戳通过 Time Encoder 编码为6个字符token（如 `⟨0⟩⟨1⟩⟨2⟩⟨3⟩⟨.⟩⟨4⟩`）。

2. **Query-Based Connector（查询驱动连接器）**：这是TriSense的核心创新。压缩后的模态特征先通过 Cross-Attention 与查询表示交互，得到查询相关特征 $f^{v,q}_i, f^{a,q}_i, f^{s,q}_i$。然后通过自适应权重机制确定各模态重要性：

   对每种模态做全局平均池化得到紧凑表示 $c_v, c_a, c_s$，拼接后输入单层MLP生成未归一化权重，再经softmax归一化满足 $w_v + w_a + w_s = 1$：

    $w_m = \frac{\exp(\tilde{w}_m)}{\sum_{m' \in \{v,a,s\}} \exp(\tilde{w}_{m'})}, \quad m \in \{v,a,s\}$

   融合后的多模态表示为：

    $\mathcal{X}_m = \hat{\mathcal{F}}(C_{comp}(\text{concat}(w_v f^{v,q}_i, w_a f^{a,q}_i, w_s f^{s,q}_i)))$

   其中 $C_{comp}$ 是slot压缩将三倍token量降回单倍，$\hat{\mathcal{F}}$ 是两层MLP做特征精化。这种设计使模型能根据查询内容强调最相关模态，弱化不相关或缺失的模态。

3. **因果事件预测**：将视频分割为事件序列 $\{e_1, e_2, \cdots, e_K\}$，每个事件包含时间戳和描述。模型基于前序事件预测下一个事件。引入 `⟨sync⟩` 特殊token实现时间头和语言头之间的自适应切换，使模型在生成过程中自然地交替产生时间戳预测和文本描述。

### 损失函数 / 训练策略

训练分两阶段：Stage 1 进行模态对齐（不含时序信息），Stage 2 在 TriSense-2M 上进行时序理解训练。

**TriSense-2M 数据集构建**：从 InternVid 和 VAST 中选取500万初始样本，使用两个基于 Qwen2.5-72B 微调的模型——Generator 将模态特定描述融合为跨模态描述，Judger 评估质量（0-5分，≥3分保留），最终筛选出200万高质量样本，覆盖约38,000个长视频，平均时长905秒。数据集显式支持模态缺失场景（AVS/AV/VS等多种组合）。

## 实验关键数据

### 主实验

**TriSense-2M 片段描述（Segment Captioning）**

| 模型 | AVS-SC (B/M/R/C) | VS-SC (B/M/R/C) | AV-SC (B/M/R/C) | V-SC (B/M/R/C) |
|------|-------------------|------------------|-------------------|-----------------|
| VTimeLLM | 0.8/8.2/16.1/2.4 | 1.2/8.8/16.9/3.1 | 1.3/10.3/17.9/2.6 | 1.4/10.4/18.2/4.0 |
| TRACE-uni | 1.1/8.2/14.7/1.4 | 1.5/8.3/15.1/2.2 | 1.6/9.5/16.3/2.3 | 1.3/9.9/17.6/8.8 |
| LongVALE | 1.2/8.6/16.7/4.9 | 2.3/10.0/20.1/5.5 | 2.5/11.4/21.3/5.9 | 1.5/11.5/18.8/0.9 |
| Qwen2.5-Omni | 0.8/8.8/13.1/1.7 | 0.8/8.6/13.1/0.8 | 1.2/9.8/15.1/1.3 | 1.1/10.1/14.6/1.1 |
| **TriSense** | **3.4/10.1/20.1/8.3** | **3.0/10.0/22.2/11.8** | **5.3/12.2/26.3/15.4** | **7.3/12.6/30.7/36.3** |

**零样本时刻检索（公共基准）**

| 模型 | Charades-STA (R@0.5/R@0.7/mIoU) | ActivityNet (R@0.5/R@0.7/mIoU) |
|------|----------------------------------|-------------------------------|
| TRACE-uni | 43.7/21.0/41.5 | 38.2/24.7/39.4 |
| NumPro-FT | 42.0/20.6/41.4 | 37.5/20.6/38.8 |
| **TriSense** | 42.3/**27.6**/39.8 | **39.6**/**27.2**/**40.1** |

### 消融实验

| 配置 | AVS-MR (R@0.5/R@0.7) | 说明 |
|------|----------------------|------|
| Stage1 Only | 0.07/0.01 | 仅模态对齐，无时序建模 |
| Stage1+2 | 0.52/0.19 | 加入时序训练后大幅提升 |
| Addition (简单相加) | 0.71/0.22 | 无法区分模态重要性 |
| Fixed Weights (固定权重) | 0.89/0.38 | 等权重分配 |
| **TriSense (自适应权重)** | **1.12/0.42** | 查询驱动的动态权重最优 |
| 32帧 | 0.74/0.27 | 帧数减半性能下降 |
| 64帧 | 1.12/0.42 | 默认配置 |
| 128帧 | 1.12/0.43 | 增加帧数边际收益递减 |

### 关键发现

- TriSense在三模态（AVS）场景下优势最为显著，CIDEr指标从LongVALE的4.9提升到8.3
- V-SC（纯视觉描述）场景下CIDEr从对比方法最高的8.8提升到36.3，体现模态融合对单模态任务也有益
- 查询驱动的自适应权重比固定权重（+26%）和简单相加（+58%）都显著更优
- R@0.7指标上TriSense在两个公共基准上均取得最佳，表明其时刻定位精度更高

## 亮点与洞察

- **模态自适应融合**是核心贡献——不同查询可能需要不同模态的支持，通过学习的softmax权重实现灵活调节
- 显式处理模态缺失问题，通过Modality Dropout训练策略使模型在任意模态组合下都能工作
- 数据集构建的Generator+Judger双模型管线值得借鉴，利用微调的LLM替代API实现大规模自动标注
- `⟨sync⟩` token实现时间头/语言头的自适应切换是一个简洁的工程设计

## 局限与展望

- 测试时仅用64帧，对于超长视频（平均905秒），采样密度不足可能影响细粒度定位
- 在纯视觉时刻检索上略逊于TRACE等专门优化视觉的模型（0.43 vs 0.48 R@0.5）
- 在LongVALE基准的描述任务上表现不如在自身数据集上突出，可能存在描述风格差异
- 计算开销较大——三个编码器+Cross-Attention+Connector，推理效率有待优化

## 相关工作与启发

- LongVALE 尝试整合三模态但将所有token压缩为一个表示，信息损失严重
- TRACE 引入因果事件建模提升时序理解，TriSense继承了这一思路
- 启发：模态自适应权重的设计可推广到其他多模态场景（如文本+图像+表格），关键是"让模型学会什么时候听什么"

## 评分

- 新颖性: ⭐⭐⭐⭐ Query-Based Connector实现三模态自适应融合是重要贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 8种任务×多个基准×详细消融×零样本评估，非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据集构建过程描述详尽
- 价值: ⭐⭐⭐⭐ 200万级高质量三模态数据集和模态自适应框架推动音视频理解发展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] EgoAVU: Egocentric Audio-Visual Understanding](../../CVPR2026/multimodal_vlm/egoavu_egocentric_audio-visual_understanding.md)
- [\[ACL 2025\] TheoremExplainAgent: Towards Video-based Multimodal Explanations for LLM Theorem Understanding](../../ACL2025/multimodal_vlm/theorem_explain_agent.md)
- [\[NeurIPS 2025\] Nautilus: A Large Multimodal Model for Underwater Scene Understanding](nautilus_a_large_multimodal_model_for_underwater_scene_understanding.md)
- [\[NeurIPS 2025\] HermesFlow: Seamlessly Closing the Gap in Multimodal Understanding and Generation](hermesflow_seamlessly_closing_the_gap_in_multimodal_understanding_and_generation.md)
- [\[NeurIPS 2025\] Multimodal Negative Learning](multimodal_negative_learning.md)

</div>

<!-- RELATED:END -->
