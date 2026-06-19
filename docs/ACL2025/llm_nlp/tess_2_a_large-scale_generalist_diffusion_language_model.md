---
title: >-
  [论文解读] TESS 2: A Large-Scale Generalist Diffusion Language Model
description: >-
  [ACL2025][LLM 其他][扩散模型] 提出 TESS 2，首个从已有自回归模型适配而来的大规模通用指令遵循扩散语言模型，通过 UL2 masking + label shifting + 双向注意力的适配训练方案 + reward guidance 推理引导，在 QA 和指令遵循任务上匹配甚至超越同等 AR 模型。
tags:
  - "ACL2025"
  - "LLM 其他"
  - "扩散模型"
  - "instruction tuning"
  - "reward guidance"
  - "inference-time compute"
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# TESS 2: A Large-Scale Generalist Diffusion Language Model

**会议**: ACL2025  
**arXiv**: [2502.13917](https://arxiv.org/abs/2502.13917)  
**代码**: [hamishivi/tess-2](https://github.com/hamishivi/tess-2)  
**领域**: LLM/NLP  
**关键词**: diffusion language model, simplex diffusion, instruction tuning, reward guidance, inference-time compute

## 一句话总结

提出 TESS 2，首个从已有自回归模型适配而来的大规模通用指令遵循扩散语言模型，通过 UL2 masking + label shifting + 双向注意力的适配训练方案 + reward guidance 推理引导，在 QA 和指令遵循任务上匹配甚至超越同等 AR 模型。

## 研究背景与动机

**领域现状**：当前语言模型主要采用自回归（AR）范式，已在各任务上取得巨大成功；扩散模型在图像、音频等领域表现出色，但在语言生成领域仍处于早期探索阶段。

**现有痛点**：AR 模型存在规划和自我纠正方面的固有局限（逐 token 生成导致无法回溯），且推理时计算量控制依赖昂贵的长链思维（CoT）方法。现有扩散语言模型仍停留在小规模、关注 perplexity 等内在指标的阶段，缺乏通用下游任务评估。

**核心矛盾**：扩散 LM 天然具备推理计算可控性（调整扩散步数）和可插拔的引导能力，但尚未被有效扩展到大规模通用指令遵循场景。

**本文目标**：弥合扩散 LM 与 AR LM 在通用任务性能上的差距，提供从 AR 模型适配到扩散 LM 的完整训练方案。

**切入角度**：不从头训练，而是基于已有 AR 模型（Mistral-7B）进行扩散适配，复用预训练知识；引入 reward guidance 在推理时无需额外训练即可对齐。

**核心 idea**：用 AR 模型初始化 + simplex 扩散 + 指令微调的三阶段方案训练通用扩散 LM，并用 reward guidance 实现免训练对齐。

## 方法详解

### 整体框架

TESS 2 的训练分三阶段：

1. **Diffusion Adaptation**：将 AR 模型（Mistral-7B）适配为扩散 LM
2. **Instruction Tuning**：在指令数据上微调
3. **Reward Guidance（推理时）**：利用奖励模型在每个扩散步引导生成

### 关键设计一：Simplex Diffusion 架构

- **功能**：在概率单纯形（probability simplex）上进行扩散，而非在 embedding 空间或离散空间
- **为什么**：保持扩散过程的连续性，同时适配离散文本数据；交叉熵损失比 MSE 更稳定
- **怎么做**：
    - 将每个 token $w$ 映射为 $k$-logit simplex 表示 $\mathbf{s}^w \in \{\pm k\}^{|\mathcal{V}|}$
    - 通过 softmax 转换为概率分布 $\mathbf{p}^w = \text{softmax}(\mathbf{s}^w)$
    - 前向扩散：$\mathbf{S}_t = \sqrt{\bar{\alpha}_t}\mathbf{S}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_t$
    - 训练损失：交叉熵 $\mathcal{L} = \mathbb{E}[-\sum_{i=1}^{L}\log p_\theta(w_i|\mathbf{S}_t, t)]$
    - 反向采样：100 步迭代去噪

### 关键设计二：AR→Diffusion 适配三要素

- **UL2 Masking**：混合 span infilling 和 prefix completion 两种训练目标，前者增强泛化，后者对齐下游用法
- **Label Shifting**：训练时预测下一个位置的 token（位置 $i$ 预测 $w_{i+1}$），与 AR 预训练的 next-token prediction 对齐，加速收敛
- **Full Bidirectional Attention**：禁用因果掩码，使用全双向注意力，充分利用扩散 LM 的全序列信息流优势

### 关键设计三：Reward Guidance

- **功能**：在推理时利用奖励模型引导扩散生成过程
- **为什么**：无需额外训练即可实现偏好对齐，是扩散模型相比 AR 模型的独特优势
- **怎么做**：
    - 每个扩散步取模型预测 $\hat{\mathbf{S}}_\theta$，通过 softmax 映射到 embedding
    - 将 embedding 输入奖励模型得到标量奖励 $R$
    - 对预测做梯度上升：$\hat{\mathbf{S}}_\theta := \hat{\mathbf{S}}_\theta + \eta \cdot \nabla_\theta R$
    - $\eta$ 控制引导强度，过高导致退化（类似 reward hacking）

### 训练策略

- **Adaptation**：在 Dolma 1.7 上训练 200k 步（~45B tokens），恒定学习率 $1\times10^{-5}$，batch size 112
- **Instruction Tuning**：在 Tulu 2 SFT mixture（~326k 样本）上训练 3 epochs，线性 warmup + cooldown
- **基座选择**：实验对比 RoBERTa / Llama 2 / Llama 3 / Mistral，选定 Mistral-7B-v0.1（可能因 prefix-LM 预训练而对双向注意力适配更好）

## 实验关键数据

### 表1：基座模型适配对比（35k 步）

| 基座模型 | Perplexity↓ | Mauve↑ | d-1↑ | Entropy |
|----------|-------------|--------|------|---------|
| Random init | 54.4 | 0.92 | 0.55 | 5.7 |
| RoBERTa | 20.2 | 0.93 | 0.36 | 4.8 |
| Llama 2 | 3619.2 | 0.01 | 0.94 | 7.7 |
| Llama 3 | 880.4 | 0.93 | 0.97 | 7.8 |
| **Mistral** | **24.3** | **0.95** | 0.62 | 6.3 |

**发现**：Llama 模型因全因果预训练，适配双向注意力时收敛困难；Mistral 可能因 prefix-LM 预训练而表现最优。

### 表3：指令微调后下游任务表现

| 模型 | AlpacaEval | SQuAD | TriviaQA | IFEval | BBH | GSM8k | GSM8k(ft) |
|------|-----------|-------|----------|--------|-----|-------|-----------|
| Mistral v0.1 AR | 77.1 | 86.0 | 50.4 | 36.8 | 43.3 | 52.5 | 51.2 |
| DiffuLlama | 0.2 | 34.9 | 19.7 | 14.4 | 1.9 | 0.0 | 63.1 |
| TESS 2 v0.1 | 63.1 | **85.4** | 49.3 | 30.5 | 8.4 | 14.5 | **66.6** |
| TESS 2 v0.3 | 62.2 | 84.8 | **53.8** | **54.6** | **10.8** | **36.5** | 59.2 |

### 消融：Reward Guidance 效果

| 引导权重 $\eta$ | AlpacaEval |
|-----------------|-----------|
| 0（无引导） | 63.1 |
| 0.25 | **66.1**（+3.0） |
| 更高 | 退化（生成无意义文本） |

### 关键发现

1. **TESS 2 全面超越现有扩散 LM**：在所有下游任务上超过 DiffuLlama 和 Flan-XLM-R-D
2. **QA 任务接近 AR 模型**：SQuAD（85.4 vs 86.0）和 TriviaQA（53.8 vs 50.4）上与 AR 模型持平甚至更优
3. **推理任务仍有差距**：BBH（10.8 vs 43.3）和 GSM8k（36.5 vs 52.5）上明显落后
4. **大量领域数据下扩散优于 AR**：GSM8k 符号数据微调后 TESS 2（66.6）超过 AR（51.2）
5. **扩散步数可扩展推理计算**：步数从 50→500 持续提升 GSM8k 性能
6. **Reward guidance 跨 RM 泛化**：不同奖励模型均可带来 3-4 分提升
7. **生成速度优势**：100 步扩散（77s/batch）vs AR 2048 步（480s/batch），快 6 倍

## 亮点与洞察

1. **AR→Diffusion 适配方案实用性强**：UL2 + label shifting + 双向注意力的组合简单有效，任何开源 AR 模型都可复用此方案
2. **Reward Guidance 是扩散 LM 的独特优势**：AR 模型只能通过 RLHF 训练对齐，而扩散 LM 可通过推理时梯度引导无训练对齐，模块化且灵活
3. **基座选择的深刻洞察**：Mistral 优于 Llama 的原因可能是 prefix-LM 风格预训练，这为"哪些 AR 模型适合做扩散适配"提供了重要启发
4. **推理计算的精细控制**：通过调整扩散步数可线性控制推理计算量，比 CoT 方法更精确
5. **生成速度反直觉**：扩散模型生成长文本时反而比 AR 快（因为前向 pass 总数固定为扩散步数）

## 局限与展望

1. **推理任务差距明显**：BBH 和 GSM8k 上与 AR 模型差距大，说明扩散 LM 在需要长链推理的任务上仍有根本性限制
2. **适配数据质量问题**：在 Dolma 上 continued pretraining 导致数学能力下降，说明适配阶段的数据选择至关重要
3. **仅支持单轮对话**：多轮训练仅有边际收益，限制了实际应用场景
4. **Reward Guidance 存在 reward hacking**：引导权重过高导致退化，需要更鲁棒的引导机制
5. **可改进方向**：
    - 使用更高质量的适配数据（如包含更多数学/代码数据）
    - 探索单步采样加速（借鉴 CV 领域 consistency model）
    - 开发多轮对话的扩散训练策略
    - 结合离散和连续扩散的优势

## 相关工作与启发

### vs DiffuLlama（Gong et al. 2024）

DiffuLlama 使用 absorbing 离散扩散从 Llama 适配，训练了 65B tokens 但未进行指令微调。TESS 2 用连续 simplex 扩散 + 指令微调，仅 45B tokens 即全面超越，说明 (a) 基座选择比训练量更关键，(b) 连续扩散 + 交叉熵比离散扩散更稳定。

### vs LLaDA（Nie et al. 2025）

LLaDA 从头预训练大规模离散扩散 LM，计算成本极高。TESS 2 选择复用已有 AR 模型知识，成本更低、更实用，体现了"站在巨人肩膀上"的思路。

### vs SSD-LM / SSD-2（Han et al. 2022, 2023）

SSD-LM 是 TESS 的前身，采用半自回归方式。TESS 2 证明全非自回归生成在 2048 token 上下文下是可行的，且通过 Mistral 适配大幅提升了性能。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — AR→Diffusion 适配方案和 reward guidance 均为原创贡献，首次实现通用指令遵循扩散 LM
- **实验充分度**: ⭐⭐⭐⭐⭐ — 基座消融、适配步数、扩散步数、reward guidance 权重均有详细分析，下游任务覆盖全面
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，技术细节充分，关键发现有定量依据
- **价值**: ⭐⭐⭐⭐ — 为扩散语言模型提供了实用的训练方案和清晰的能力边界认知，reward guidance 思路有广泛扩展潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Cheaper and Better Diffusion Language Model via Task-Specific Training](cheaper_and_better_diffusion_language_model_via_task-specific_training.md)
- [\[ACL 2025\] Segment-Level Diffusion: A Framework for Controllable Long-Form Generation with Diffusion Language Models](segment_level_diffusion.md)
- [\[ACL 2025\] Synergizing Unsupervised Episode Detection with LLMs for Large-Scale News Events](synergizing_unsupervised_episode_detection_with_llms_for_large-scale_news_events.md)
- [\[ACL 2025\] Pitfalls of Scale: Investigating the Inverse Task of Redefinition in Large Language Models](pitfalls_of_scale_investigating_the_inverse_task_of_redefinition_in_large_langua.md)
- [\[ACL 2025\] Conversational Quality Assessment: A Large-Scale Corpus and Comprehensive Study](conversational_quality_assessment_a_large-scale_corpus_and_comprehensive_study.md)

</div>

<!-- RELATED:END -->
