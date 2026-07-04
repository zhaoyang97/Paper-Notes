---
title: >-
  [论文解读] NTPP: Generative Speech Language Modeling for Dual-Channel Spoken Dialogue via Next-Token-Pair Prediction
description: >-
  [ICML2025][音频/语音][语音对话] 提出 Next-Token-Pair Prediction (NTPP) 范式，首次用 decoder-only 架构对双通道语音对话进行 speaker-independent 联合分布建模，实现更自然的轮次转换、更低的推理延迟和更强的说话人无关性。
tags:
  - "ICML2025"
  - "音频/语音"
  - "语音对话"
  - "双通道语音"
  - "Next-Token-Pair Prediction"
  - "decoder-only"
  - "轮次转换"
  - "全双工对话"
---

# NTPP: Generative Speech Language Modeling for Dual-Channel Spoken Dialogue via Next-Token-Pair Prediction

**会议**: ICML2025  
**arXiv**: [2506.00975](https://arxiv.org/abs/2506.00975)  
**代码**: [Demo & Code](https://audio-3059.pages.dev)  
**领域**: speech_language_model  
**关键词**: 语音对话, 双通道语音, Next-Token-Pair Prediction, decoder-only, 轮次转换, 全双工对话

## 一句话总结

提出 Next-Token-Pair Prediction (NTPP) 范式，首次用 decoder-only 架构对双通道语音对话进行 speaker-independent 联合分布建模，实现更自然的轮次转换、更低的推理延迟和更强的说话人无关性。

## 研究背景与动机

- **核心问题**：如何让语音语言模型 (SLM) 像真人对话一样自然——支持重叠、打断、停顿等全双工 (full-duplex) 交互？
- **现有方案的缺陷**：
    - **单通道方法** (SpeechGPT, LLama-Omni 等)：基于 turn-based 的 $p(S^b|S^a)$ 建模，需 VAD 切分轮次，无法处理实时流式对话。
    - **dGSLM**：建模联合分布 $p(S^a, S^b)$，但使用 Siamese encoder-decoder 双塔架构，参数效率低。
    - **LSLM**：decoder-only，但仅预测一侧通道（条件分布），不具备 speaker-independence。
    - **Moshi**：RQ-transformer，依赖额外 encoder 且需维护两个 KVCache，推理效率差。
- **动机**：双通道语音天然包含重叠 (overlap)、停顿 (pause)、间隙 (gap)、打断 (interruption) 等对话动态信息（如 Figure 1 所示），这些信息在单通道中被混合无法区分。能否用纯 decoder-only 架构直接建模双通道联合分布？

## 方法详解

### 1. NTPP 建模范式

核心思想：在每个时间步 $t$ 同时预测两个说话人的 token pair $(s_t^a, s_t^b)$，学习联合分布：

$$p(S^a, S^b) = \prod_{t=1}^{T} p(s_t^a, s_t^b \mid s_{t-1}^a, \ldots, s_1^a, s_{t-1}^b, \ldots, s_1^b)$$

通过**条件独立性假设**将 pair 分解：

$$p(s_t^a, s_t^b \mid \cdot) = p(s_t^a \mid s_{t-1:1}^a, s_{t-1:1}^b) \cdot p(s_t^b \mid s_{t-1:1}^a, s_{t-1:1}^b)$$

关键归纳偏置：一个人的发言受自己之前说过的话和听到的对方的话共同影响。

与已有方法对比（Table 1）：NTPP 是唯一同时满足 **Speaker-Independent + Encoder-Free + VAD-Free + Single KVCache** 四个条件的方法。

### 2. Autoregressive Dual-channel Speech Transformer

将两通道序列按交错顺序排列：$S = ((s_1^a, s_1^b), (s_2^a, s_2^b), \ldots, (s_T^a, s_T^b))$，每步预测一对 token。仅需对标准 decoder-only Transformer 做两处轻量修改：

**Token Pair Embedding**：每个 token pair 包含三种嵌入：
- **Codebook embedding** $\mathbf{z}_t$：从 VQ/RVQ codebook 查表获取
- **Positional embedding** $\mathbf{p}_t$：沿用 RoPE，同一时间步的 $s_t^a$ 和 $s_t^b$ 共享相同的位置编码
- **Channel embedding** $\mathbf{c}_t$：one-hot 编码区分 speaker a/b

Query 和 Key 的计算：$\mathbf{q} = \mathbf{W}_Q[\mathbf{z}_t^a, \mathbf{z}_t^b] + [\mathbf{p}_t^a, \mathbf{p}_t^b] + [\mathbf{c}_t^a, \mathbf{c}_t^b]$

**Pair-wise Causal Masking**：因果 mask 矩阵 $\mathbf{M} \in \mathbb{R}^{2T \times 2T}$ 的对角线采用 $2 \times 2$ 分块策略：同一时间步内 $s_t^a$ 和 $s_t^b$ **互不可见**（对角线上仅保留自身），确保条件独立性。

### 3. 推广到 RVQ Tokenizer

RVQ 将每个时间步展开为 $D$ 个 depth token，序列变为 $(s_{t,1}^a, \ldots, s_{t,D}^a, s_{t,1}^b, \ldots, s_{t,D}^b)$。引入：
- **Cyclic Depth Embedding**：$\mathbf{d} = (\sin(2\pi i / D), \cos(2\pi i / D))$，以 $D$ 为周期循环，帮助模型识别当前 token 的深度位置。
- **RVQ Causal Masking**：$2D \times 2D$ 分块对角线中，上三角 mask（浅层不看深层），左下 $D \times D$ 子矩阵全 mask（通道 a/b 互不可见）。

### 4. 流式条件推理

为适配实时交互，采用 chunk-wise streaming inference：给定用户输入 $\lambda$ 个 token 后，模型开始生成 $\lambda$ 个响应 token，循环往复。

## 实验设置与主要结果

### 训练设置

- **两阶段训练**：Stage 1 用约 14 万小时单通道语音训练基础 SLM（textless，不需要文本对齐），Stage 2 用 Fisher 数据集（2200 小时双通道电话对话）做 NTPP 微调。
- **Audio tokenizer**：基于 SoundStream 训练 RVQ tokenizer，每秒 40 个 token，codebook 大小 4096。
- **LLM 骨干**：LLaMA 3.1-8B / Mistral-7B / Gemma-2-9B，其中 LLaMA 3.1 收敛最快。
- **训练超参**：16 × A100，cosine scheduler (lr: 4e-6 → 4e-4)，batch size 64，每 epoch 40k steps。
- **消融发现**：纯音频训练 (w/o Text) 比加 ASR 文本 (w Text) 收敛更快、perplexity 更低，说明文本转录可能引入模态干扰。

### 主要结果

**1. 轮次转换事件分布 (Table 2)**  
在 Fisher 测试集上比较生成对话与真实对话的 turn-taking 统计量（IPU 数量/时长、Pause、Gap、Overlap），以 |Δ|（与 ground truth 的均值绝对差）评估。NTPP 在温度 0.9 时取得最佳平衡：

| 模型 | IPU 数量 |Δ| | Pause 数量 |Δ| | Gap 数量 |Δ| | Overlap 数量 |Δ| |
|------|---------|---------|---------|----------|
| Cascaded | 4.1 | 7.0 | 7.4 | 6.5 |
| dGSLM w/o CA | 3.9 | 2.9 | 3.6 | 1.0 |
| dGSLM | 1.6 | 3.4 | 2.0 | 2.9 |
| LSLM | 2.2 | 3.6 | 2.4 | 3.2 |
| **NTPP (t=0.9)** | **1.3** | **2.3** | **1.5** | **0.9** |

NTPP 在 Overlap 指标上大幅领先，说明其更好地捕获了对话中的重叠动态。

**2. 人工评估 (Table 3)**  
25 位标注者按 5 分制 MOS 评估意义性 (M-MOS) 和自然度 (N-MOS)：

| 模型 | M-MOS (Overall) | N-MOS (Overall) | M-MOS (Fisher) | N-MOS (Fisher) | M-MOS (CANDOR) | N-MOS (CANDOR) |
|------|--------|--------|--------|--------|--------|--------|
| dGSLM | 1.38 | 3.85 | 1.82 | 4.10 | 1.51 | 2.85 |
| SyncLLM | 3.85 | 4.10 | 4.10 | 4.33 | 3.85 | 3.91 |
| Moshi | 3.90 | 3.95 | 3.20 | 3.90 | 3.95 | 3.95 |
| **NTPP** | **3.95** | **4.15** | **4.10** | **4.42** | **4.05** | **4.05** |
| GT | 4.90 | 4.95 | 4.90 | 4.90 | 4.90 | 4.95 |

NTPP 在所有 baseline 中 M-MOS 和 N-MOS 均为最高，在 Fisher 域内数据上自然度达 4.42。

**3. 说话人无关性 (Table 4)**  
交换两通道输入后测量指标变化 $|\Delta M_{\text{original}} - \Delta M_{\text{swapped}}|$（越低越鲁棒）：
- dGSLM 和 NTPP 在训练集上变化接近 0，测试集上也保持较低（NTPP 所有指标 < 0.45）。
- Moshi 变化显著（Gap 持续时长偏差达 0.84），说明其依赖 speaker-conditioned 生成。

**4. 推理延迟 (Figure 7)**  
随对话轮次增加，NTPP 始终保持低于 220ms 的响应延迟，而 Moshi 呈线性增长。原因：NTPP 仅需单个 KVCache，Moshi 需要两个。

**5. 消融实验 (Figure 8)**  
- 两阶段训练 vs 单阶段：去掉 Stage 1（单通道预训练）或 Stage 2（NTPP 微调）均导致 perplexity 显著上升。
- RVQ vs VQ：RVQ tokenizer 训练 loss 始终低于 VQ。

## 亮点与洞察

- **精巧的建模视角转换**：从条件分布 $p(S^b|S^a)$ 转向联合分布 $p(S^a, S^b)$，通过 pair-wise 因果 mask 在 decoder-only 架构内优雅实现，改动极小但效果显著。
- **工程友好**：单 KVCache + 无额外 encoder + 无 VAD 模块 = 部署简单、推理高效。
- **条件独立性假设合理**：同一时间步两通道互不可见看似丢失信息，实际上利用了"发言受过去历史影响"的对话先验，实验证明并未损害性能。
- **Textless 训练效果更好**：消融发现不加 ASR 文本反而更优，暗示语音 token 自身已包含足够语义信息，文本对齐可能引入噪声。

## 局限与展望

1. **数据瓶颈**：双通道语音数据稀缺，Fisher 仅 2200 小时。扩展到更大规模多通道数据集是核心挑战。
2. **条件独立假设的局限**：同一时间步 $s_t^a$ 和 $s_t^b$ 的条件独立假设在强交互场景（如激烈争论）中可能不成立。
3. **仅限双通道**：未扩展到多人对话 (>2 speakers) 场景。
4. **评估维度有限**：缺乏语音质量 (MOS for speech quality)、语义准确性 (WER) 等客观指标。
5. **潜在滥用风险**：论文提及可能被用于电信诈骗等场景，安全机制讨论不充分。

## 相关工作与启发

- **dGSLM** (Nguyen et al., 2023)：首个双通道 textless 对话生成模型，双塔 Siamese encoder-decoder。NTPP 用 decoder-only 统一替代。
- **LSLM** (Ma et al., 2024)：token fusion 策略融合双通道，但仅建模条件分布。证明了 decoder-only 处理双通道的可行性。
- **Moshi** (Défossez et al., 2024)：RQ-transformer + 文本对齐，全栈方案但推理效率差。NTPP 在延迟上有结构性优势。
- **SyncLLM** (Veluri et al., 2024)：全双工对话 agent，通过流式处理实现重叠语音。在人工评估中与 NTPP 接近。

## 评分
- 新颖性: ⭐⭐⭐⭐ — pair-wise 预测范式新颖，条件独立分解简洁有效
- 实验充分度: ⭐⭐⭐⭐ — 多维度评估（turn-taking/人工/speaker-independence/latency/ablation），但缺少语音质量客观指标
- 写作质量: ⭐⭐⭐⭐ — 公式推导清晰，图表丰富，动机阐述充分
- 价值: ⭐⭐⭐⭐ — 实际意义大（实时语音交互），方法简洁可复现，但数据瓶颈限制了 impact

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Aligning Spoken Dialogue Models from User Interactions](aligning_spoken_dialogue_models_from_user_interactions.md)
- [\[AAAI 2026\] DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling](../../AAAI2026/audio_speech/dualspeechlm_towards_unified_speech_understanding_and_generation_via_dual_speech.md)
- [\[ICCV 2025\] Everything is a Video: Unifying Modalities through Next-Frame Prediction](../../ICCV2025/audio_speech/everything_is_a_video_unifying_modalities_through_next-frame_prediction.md)
- [\[ICML 2025\] FLAM: Frame-Wise Language-Audio Modeling](flam_frame-wise_language-audio_modeling.md)
- [\[ICML 2025\] Long-Form Speech Generation with Spoken Language Models](long-form_speech_generation_with_spoken_language_models.md)

</div>

<!-- RELATED:END -->
