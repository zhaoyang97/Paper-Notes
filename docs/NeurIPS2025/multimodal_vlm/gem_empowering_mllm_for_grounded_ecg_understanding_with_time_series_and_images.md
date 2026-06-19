---
title: >-
  [论文解读] GEM: Empowering MLLM for Grounded ECG Understanding with Time Series and Images
description: >-
  [NeurIPS 2025][多模态VLM][ECG理解] 提出 GEM，首个统一 ECG 时间序列、12 导联 ECG 图像和文本的多模态大语言模型，通过双编码器框架、跨模态对齐和知识引导的指令数据生成，实现了基于可量化生理特征的接地心电图诊断，诊断准确率提升 7.4%，可解释性提升 22.7%，接地能力提升 25.3%。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "ECG理解"
  - "多模态大模型"
  - "时序信号"
  - "接地诊断"
  - "心电图"
---

# GEM: Empowering MLLM for Grounded ECG Understanding with Time Series and Images

**会议**: NeurIPS 2025  
**arXiv**: [2503.06073](https://arxiv.org/abs/2503.06073)  
**代码**: [https://github.com/lanxiang1017/GEM](https://github.com/lanxiang1017/GEM)  
**领域**: 多模态VLM  
**关键词**: ECG理解, 多模态大模型, 时序信号, 接地诊断, 心电图

## 一句话总结
提出 GEM，首个统一 ECG 时间序列、12 导联 ECG 图像和文本的多模态大语言模型，通过双编码器框架、跨模态对齐和知识引导的指令数据生成，实现了基于可量化生理特征的接地心电图诊断，诊断准确率提升 7.4%，可解释性提升 22.7%，接地能力提升 25.3%。

## 研究背景与动机

**领域现状**：深度学习在心律失常检测等任务上取得了进展，但缺乏语言能力和可解释性。近期 MLLM（如 PULSE）通过大规模指令微调实现了基于语言的 ECG 解读，但主要处理静态图像输入和预定义诊断任务。

**现有痛点**：(1) **模态协同不足**——现有模型仅处理单一非文本模态（时序信号或图像），未能像临床医生那样同时分析动态信号趋势和空间波形模式；(2) **可解释性和接地不足**——现有模型未能将诊断结论显式关联到具体的波形证据（如 QRS 时长、PR 间期等可量化参数）。

**核心矛盾**：如何让模型像心脏科医生一样工作——同时分析时序和图像信号，并为每个诊断结论提供具体的心电生理特征证据。

**本文目标**：(1) 构建统一时序+图像+文本的多模态 ECG 模型；(2) 实现心跳级别的生理特征接地诊断；(3) 构建高粒度接地训练数据。

**切入角度**：利用现有的特征提取工具（FeatureDB）从原始 ECG 信号中提取心跳级别的生理特征序列，设计诊断引导器来激活 GPT-4o 的潜在医学知识，自动生成高粒度接地指令数据。

**核心 idea**：双编码器提取互补特征 + 跨模态对齐 + 知识引导数据生成，让 MLLM 像心脏科医生一样提供有据可依的 ECG 诊断。

## 方法详解

### 整体框架
GEM 包含三大组件：(1) 多模态编码——时序编码器(ECG-CoCa)和图像编码器(CLIP)分别提取特征；(2) 跨模态对齐——将时序表示先映射到图像维度再统一映射到文本空间；(3) 知识引导指令数据生成——FeatureDB 提取生理特征 + 诊断引导器构建 prompt + GPT-4o 生成高粒度目标答案。

### 关键设计

1. **双编码器多模态编码**:

    - 功能：分别从 ECG 时间序列和 12 导联图像中提取互补特征
    - 核心思路：时序编码器 $\mathbf{e}_{ts} \in \mathbb{R}^{n_s \times d_s} = E_{ts}(\bm{x}_{ts})$ 使用预训练 ECG-CoCa（在大量 ECG-Text 对上对比学习训练）；图像编码器 $\mathbf{e}_{img} \in \mathbb{R}^{n_m \times d_m} = E_{img}(\bm{x}_{img})$ 使用 LLaVA 的预训练 CLIP
    - 设计动机：时序模型捕捉动态变化但可能忽略空间模式，图像模型检测全局结构但可能遗漏微妙的时域细节——两者互补

2. **跨模态对齐学习**:

    - 功能：将异质模态表示统一到 LLM 可理解的文本空间
    - 核心思路：先用 MLP 将时序表示映射到图像维度 $\hat{\mathbf{e}}_{ts} \in \mathbb{R}^{n_s \times d_m} = MLP_{ts}(\mathbf{e}_{ts})$，再用共享 MLP 将两种表示统一映射到文本维度 $\mathbf{h}_{ts} \in \mathbb{R}^{n_s \times d_t} = MLP(\hat{\mathbf{e}}_{ts})$, $\mathbf{h}_{img} \in \mathbb{R}^{n_m \times d_t} = MLP(\mathbf{e}_{img})$，最后与文本嵌入拼接 $\mathbf{x} = \text{Concatenate}(\mathbf{h}_{ts}, \mathbf{h}_{img}, \text{Embed}(\bm{x}_q))$
    - 设计动机：两步对齐（时序→图像维度→文本维度）比直接跳跃更稳定，共享投射器确保两种模态在同一文本空间中可比

3. **知识引导指令数据生成**:

    - 功能：自动构建高粒度 ECG 接地指令数据，无需人工标注
    - 核心思路：(a) **Grounding Feature Extractor**——从原始 ECG 时序中提取每个心跳的 14 种特征序列（心率、RR 间期、P波幅度/时长、PR 间期、QRS 幅度/时长、T波幅度/时长、ST 时长/形态、QT/QTc 间期）×12 导联，$\bm{x}_{fs} = \text{FeatureDB}(\bm{x}_{ts})$；(b) **Diagnosis Guider**——构建包含心脏科诊断指令的结构化 prompt $\bm{x}_p = \text{DiagnosisGuider}(\bm{x}_{fs})$，为每个样本定制特定于其特征的提示；(c) **GPT-4o 生成**——$\bm{y} = \text{GPT-4o}(\bm{x}_p)$，产出 30,000 条 ECG-Grounding 数据
    - 设计动机：PULSE 的 ECG-Instruct 数据基于原始报告，偶尔因幻觉产生错误解释；ECG-Grounding 基于实际提取的生理特征，确保诊断有据可依

### 损失函数 / 训练策略
一步训练（不像多数 MLLM 先训投射器再微调 LLM）：冻结 $\theta_{ts}$ 和 $\theta_{img}$，联合训练 $\theta_{M_{ts}}$、$\theta_M$、$\theta_{LLM}$。损失为标准 NLL：$L = -\sum_{i=1}^N \log P(y_j | \mathbf{x}, \theta_{LLM})$。SFT 1 个 epoch，8 × A100。

## 实验关键数据

### 主实验（Grounded ECG Understanding）

| 指标 | PULSE | GEM (SFT LLaVA) | GEM (SFT PULSE) |
|------|-------|-----------------|-----------------|
| 诊断准确率 (MIMIC) | 81.14% | **87.24%** | 86.49% |
| 诊断准确率 (PTB-XL) | 59.24% | 73.53% | **73.59%** |
| 导联覆盖率 (MIMIC) | 7.11% | **71.07%** | 69.80% |
| 导联准确率 (MIMIC) | 2.95% | **46.44%** | 45.33% |
| ECG特征接地 (MIMIC) | 50.18% | **75.48%** | 74.95% |
| 循证推理 (MIMIC) | 52.40% | **75.09%** | 74.70% |

### ECG-Bench 异常检测

| 数据集 | 指标 | PULSE | GEM (SFT LLaVA) | GEM (SFT PULSE) |
|--------|------|-------|-----------------|-----------------|
| CSN | ACC | 85.2% | **92.6%** | 86.2% |
| G12EC | ACC | 78.2% | **81.8%** | 80.5% |
| PTB-XL | AUC | 82.4% | 81.8 | **83.4%** |
| CODE-15% | AUC | 90.7% | 90.5 | **91.5%** |
| PTB-XL Report | Score | 61.3 | 65.0 | **67.1** |

### 消融实验

| 配置 | CSN ACC | CODE-15% AUC | 说明 |
|------|---------|-------------|------|
| TS only | 91.6% | 90.8% | 仅时序 |
| TS + IMG | 90.1% | 91.3% | 时序+图像 |
| GEM (SFT PULSE) | 86.2% | 91.5% | 完整框架 |

### 关键发现
- GEM 在诊断准确率上提升 6-14pp，在导联覆盖率上从 7% 跃升至 71%，在接地能力上从 50% 提升至 75%——改进幅度巨大
- SFT LLaVA（从未在 ECG 数据上训练过）仅 1 个 epoch 微调就在 CSN 上超越 PULSE 7.4%，说明 GEM 框架的高效学习能力
- 跨域泛化能力突出：在域外 PTB-XL 上诊断准确率从 59.24% 提升至 73.59%
- 8 位心脏科专家评估确认 GEM 输出的临床可靠性和实用性，部分病例中 GEM 发现了专家在初次检查时未注意到的细节
- GPT-4o 和 Deepseek-R1 都能用知识引导方法生成高质量目标数据，说明方法不依赖特定 LLM

## 亮点与洞察
- **首个三模态 ECG 理解模型**：统一时序+图像+文本是临床诊断流程的忠实模拟，每种模态提供不可替代的互补信息
- **知识引导数据生成**：用 FeatureDB（零参数特征提取）+ 诊断引导器 + GPT-4o 的组合避免了昂贵的专家标注，同时确保生成数据基于实际生理特征而非幻觉——是一个可复制到其他医学领域的数据生成范式
- **一步训练策略**：在数据量有限时（仅 30K 接地数据）比传统多步训练更有效，减少了对齐和微调之间的不一致
- 接地诊断理念可迁移到其他医学时序数据（如 EEG、肌电图等）

## 局限与展望
- GPT-4o 生成的目标答案偶有与心脏科医生判断不一致之处（如对缺血的判断），人类反馈对齐可进一步提升
- 训练数据规模较小（30K ECG-Grounding + 1.1M ECG-Instruct），扩展接地数据量可能带来进一步提升
- 特征提取工具 FeatureDB 本身的精度成为上限——如果特征提取有误，下游诊断也会受影响
- 实时临床部署的延迟、可靠性等工程问题未讨论
- 消融实验中 TS+IMG 在 CSN 上反而低于 TS only（90.1% vs 91.6%），多模态并非始终优于单模态，需更深入分析

## 相关工作与启发
- **vs PULSE**: PULSE 是当前 SOTA 的 ECG-LLM，但仅用图像模态且缺乏接地能力；GEM 添加时序模态和接地数据后全面超越
- **vs ECG-CoCa**: ECG-CoCa 的时序编码器被 GEM 直接复用，验证了对比学习预训练的时序编码器作为 MLLM 组件的有效性
- **vs JoLT**: JoLT 用 Querying Transformer 对齐 ECG 与文本，GEM 通过共享投射器实现更简洁的对齐

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个三模态 ECG 理解模型，知识引导数据生成方法新颖实用；但各组件（双编码器、MLP 对齐）较为标准
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖多个数据集、多种任务、消融分析、专家评估，非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，临床动机充分，但符号较多对非医学读者不太友好
- 价值: ⭐⭐⭐⭐⭐ 接地诊断对临床信任度至关重要，GEM 显著推进了 AI 辅助心电图解读的实用化

## 与相关工作的对比
- **vs PULSE**: PULSE 是当前 SOTA 的 ECG-LLM（百万级指令微调），但仅处理 ECG 图像且缺乏接地能力（导联覆盖率仅 7%）；GEM 添加时序模态 + 接地数据后导联覆盖率跃升至 71%，接地能力提升 25pp
- **vs ECG-CoCa/ECG-Chat**: ECG-CoCa 的时序编码器被 GEM 复用为组件，验证了对比学习预训练的 ECG 编码器在 MLLM 中的有效性；ECG-Chat 仅处理时序而 GEM 同时处理时序+图像
- **vs JoLT**: JoLT 用 Querying Transformer 对齐 ECG 与文本，GEM 通过两步 MLP 投射（时序→图像维度→文本空间）实现更简洁高效的对齐
- **vs 通用 MLLM (GPT-4o)**: GPT-4o 在 ECG 异常检测上表现远低于专用模型（CSN 57.5% vs GEM 92.6%），但在知识引导下能生成高质量训练数据

## 启发与关联
- 知识引导数据生成范式（特征提取工具 + 诊断引导器 + LLM 生成）可迁移至 EEG、EMG 等其他医学时序信号的接地诊断
- 一步训练策略（冻结编码器 + 联合训练投射器和 LLM）在小数据场景下比传统多步训练更有效，值得在其他领域验证
- 心脏科专家评估中 GEM 发现了专家初次检查时遗漏的细节——AI 辅助诊断不应仅追求与专家一致，还应追求互补增强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] anyECG-chat: A Generalist ECG-MLLM for Flexible ECG Input and Multi-Task Understanding](../../AAAI2026/multimodal_vlm/anyecg-chat_a_generalist_ecg-mllm_for_flexible_ecg_input_and.md)
- [\[CVPR 2025\] PEACE: Empowering Geologic Map Holistic Understanding with MLLMs](../../CVPR2025/multimodal_vlm/peace_empowering_geologic_map_holistic_understanding_with_mllms.md)
- [\[ICML 2026\] ECG-R1: Protocol-Guided and Modality-Agnostic MLLM for Reliable ECG Interpretation](../../ICML2026/multimodal_vlm/ecg-r1_protocol-guided_and_modality-agnostic_mllm_for_reliable_ecg_interpretatio.md)
- [\[NeurIPS 2025\] In the Eye of MLLM: Benchmarking Egocentric Video Intent Understanding with Gaze-Guided Prompting](in_the_eye_of_mllm_benchmarking_egocentric_video_intent_understanding_with_gaze-.md)
- [\[NeurIPS 2025\] SITCOM: Scaling Inference-Time COMpute for VLAs](sitcom_scaling_inference-time_compute_for_vlas.md)

</div>

<!-- RELATED:END -->
