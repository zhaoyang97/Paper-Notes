---
title: >-
  [论文解读] Say More with Less: Variable-Frame-Rate Speech Tokenization via Adaptive Clustering and Implicit Duration Coding
description: >-
  [AAAI 2026][音频/语音][speech tokenization] 提出 VARSTok，首个全动态可变帧率语音 tokenizer，通过时序感知密度峰聚类和隐式时长编码，实现自适应 token 分配，在使用更少 token 的同时超越固定帧率基线。 领域现状 领域现状：现有语音 tokenizer（如 WavT…
tags:
  - "AAAI 2026"
  - "音频/语音"
  - "speech tokenization"
  - "variable frame rate"
  - "density peak clustering"
  - "implicit duration coding"
  - "speech language model"
---

# Say More with Less: Variable-Frame-Rate Speech Tokenization via Adaptive Clustering and Implicit Duration Coding

**会议**: AAAI 2026  
**arXiv**: [2509.04685](https://arxiv.org/abs/2509.04685)  
**代码**: [VARSTok](https://zhengrachel.github.io/VARSTok)  
**领域**: 音频语音  
**关键词**: speech tokenization, variable frame rate, density peak clustering, implicit duration coding, speech language model

## 一句话总结

提出 VARSTok，首个全动态可变帧率语音 tokenizer，通过时序感知密度峰聚类和隐式时长编码，实现自适应 token 分配，在使用更少 token 的同时超越固定帧率基线。

## 背景与动机

### 领域现状

**领域现状**：现有语音 tokenizer（如 WavTokenizer、EnCodec）统一按固定帧率（如 40Hz、75Hz）分配 token，忽略语音信号信息密度的时序变化

### 现有痛点

**现有痛点**：自然语音中，静音和稳定元音区域存在大量冗余，而快速发音转换和情感表达丰富的片段信息密度高

### 核心矛盾

**核心矛盾**：固定帧率导致冗余区域 token 浪费、高信息区域表示不足，下游 speech LM 难以学到自然韵律

### 解决思路

**解决思路**：已有的自适应压缩工作（如 TFC）仅在预定义的几种帧率间切换，属"伪动态"，且不建模 token 时长

### 解决思路

**本文目标**：如何设计一个全动态可变帧率的声学 speech tokenizer，使其能根据局部特征相似性自适应分配 token，并且无需辅助时长预测器即可直接用于下游 autoregressive speech LM？

## 方法详解

### 整体框架

VARSTok 由四部分组成：Speech Encoder → Temporal-Aware Density Peak Clustering → VQ Module → Speech Decoder。

1. Encoder 将波形转为帧级 embedding $\mathbf{X} \in \mathbb{R}^{T \times H}$
2. 聚类模块自适应分组为 $N$ 个变长 cluster $\mathcal{C}_1, \dots, \mathcal{C}_N$
3. 每个 cluster mean-pool 后经 VQ（单 codebook，$K=4096$）量化
4. 隐式时长编码将内容+时长编入单一 token ID
5. 解码时按时长展开，送入 decoder 重建波形

### 关键设计 1：时序感知密度峰聚类

计算每帧的局部密度 $\rho_i$ 和峰距 $\delta_i$：

$$\rho_i = \exp\left(\frac{1}{m}\sum_{j \in \text{KNN}(i)} \phi(\mathbf{x}_i, \mathbf{x}_j)\right), \quad \phi(\mathbf{x}_i, \mathbf{x}_j) = \frac{1 + \langle \mathbf{x}_i, \mathbf{x}_j \rangle}{2}$$

峰分数 $s_i = \rho_i \cdot \delta_i$，高分帧作为 cluster 种子。从种子双向扩展，候选帧需满足：

$$\phi(\mathbf{x}_{i^*}, \mathbf{x}_t) - \beta \cdot s_t > \tau$$

且必须保持时序连续性（temporal contiguity）。扩展受最大跨度 $S_{\max}$ 限制。

### 关键设计 2：隐式时长编码

将 VQ 索引 $k_n$ 和时长 $d_n$ 编入单一 token ID：

$$\text{ID}_n = (d_n - 1) \cdot K + k_n$$

解码时通过整除和取模恢复：$d_n = \lfloor \text{ID}_n / K \rfloor + 1$，$k_n = \text{ID}_n \bmod K$。

扩展词表大小为 $K \times S_{\max}$，无需额外时长预测器，直接适配 autoregressive LM。

## 实验关键数据


### 主实验

| 模型 | 帧率(Hz) | Bitrate(kbps) | UTMOS↑ | PESQ↑ | STOI↑ |
|------|---------|--------------|--------|-------|-------|
| WavTokenizer | 75.00 | 0.90 | 4.0247 | 2.4543 | 0.9188 |
| WavTokenizer | 40.00 | 0.48 | 3.6107 | 1.7075 | 0.8652 |
| BigCodec | 40.00 | 0.52 | 3.9802 | 1.8796 | 0.8653 |
| **VARSTok**(τ=0.8) | 36.81 | 0.52 | **4.0000** | 1.8887 | 0.8814 |
| **VARSTok**(τ=0.7) | 30.95 | 0.43 | **3.8949** | 1.7095 | 0.8601 |

- 在 30.95Hz（比 40Hz 基线减少 **23% token**）下，UTMOS 仍达 3.8949，超过 40Hz WavTokenizer
- τ=0.8 时 UTMOS=4.0000，接近 75Hz WavTokenizer 但 token 量不到一半
- 下游 TTS：VARSTok(τ=0.8) WER=6.787%（vs WavTokenizer 7.481%），MOS=4.053（vs 3.983）
- ARCH 语义评估：AudioMNIST F1 从 0.4509 提升到 0.6078（τ=0.7）
- 推理效率：τ=0.6 时 RTF=0.487，比基线加速 **36%**

## 亮点与洞察

- **首个**全动态可变帧率声学 tokenizer 可直接集成到下游 autoregressive speech LM
- 隐式时长编码方案简洁优雅，无需额外模块或训练，将内容+时长编入单一 token
- 超参 τ 和 $S_{\max}$ 提供灵活的 rate-quality 控制旋钮
- 在语义评估任务上也显著优于固定帧率基线，说明动态 token 分配学到了更好的表征

## 局限与展望

- 仅在 LibriTTS（585h）上训练，未验证大规模数据和多语言场景
- 聚类算法不可微，无法端到端联合优化分割策略
- $S_{\max}$ 过大（如 8）时质量退化明显，极端压缩能力有限
- 客观 speaker similarity 随帧率下降略有下降（虽主观 MOS 差异不显著）
- 未探索音乐、环境音等其他音频领域

## 相关工作与启发

| 维度 | VARSTok | TFC | WavTokenizer |
|------|---------|-----|-------------|
| 帧率类型 | 全动态连续 | 伪动态（3种预定义） | 固定 |
| 时长建模 | 隐式编码 | 无 | 无 |
| Codebook | 单 codebook | 多 codebook RVQ | 单 codebook |
| 下游LM适配 | 直接使用 | 需层级融合 | 直接使用 |

## 启发

- 隐式时长编码 $(d-1) \cdot K + k$ 的思路可推广到其他需要同时编码属性+内容的离散化场景
- 密度峰聚类保持时序连续的约束设计，可借鉴到视频/动作序列的自适应分段
- 可变帧率 + 单一 token 表示的范式可能是语音 LM 效率提升的关键方向

## 评分

⭐⭐⭐⭐ — 创新性强，首次证明全动态可变帧率声学 tokenizer 可直接用于 speech LM，但数据规模和泛化性验证不足

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR](hearing_more_with_less_multi-modal_retrieval-and-selection_augmented_conversatio.md)
- [\[ACL 2025\] Soundwave: Less is More for Speech-Text Alignment in LLMs](../../ACL2025/audio_speech/soundwave_less_is_more_for_speech-text_alignment_in_llms.md)
- [\[ICLR 2026\] TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling](../../ICLR2026/audio_speech/taste_text-aligned_speech_tokenization_and_embedding_for_spoken_language_modelin.md)
- [\[ICLR 2026\] FlexiCodec: A Dynamic Neural Audio Codec for Low Frame Rates](../../ICLR2026/audio_speech/flexicodec_a_dynamic_neural_audio_codec_for_low_frame_rates.md)
- [\[ICML 2026\] Two-Dimensional Quantization for Geometry-Aware Audio Coding](../../ICML2026/audio_speech/two-dimensional_quantization_for_geometry-aware_audio_coding.md)

</div>

<!-- RELATED:END -->
