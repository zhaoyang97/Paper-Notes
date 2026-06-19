---
title: >-
  [论文解读] Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference
description: >-
  [ACL 2025][LLM效率][encoder-only] 提出 ModernBERT，将现代 LLM 架构优化（RoPE、GeGLU、交替局部/全局注意力、unpadding）系统性地引入 encoder-only 模型，在 2T token 上训练并原生支持 8192 上下文长度，在分类和检索任务上全面超越 BERT/RoBERTa/DeBERTaV3，同时推理速度和显存效率大幅领先。
tags:
  - "ACL 2025"
  - "LLM效率"
  - "encoder-only"
  - "BERT"
  - "推理效率"
  - "长上下文"
  - "信息检索"
  - "注意力机制"
  - "RoPE"
---

# Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference

**会议**: ACL 2025  
**arXiv**: [2412.13663](https://arxiv.org/abs/2412.13663)  
**代码**: [AnswerDotAI/ModernBERT](https://github.com/AnswerDotAI/ModernBERT)  
**领域**: LLM效率  
**关键词**: encoder-only, BERT, 推理效率, 长上下文, 信息检索, Flash Attention, RoPE

## 一句话总结

提出 ModernBERT，将现代 LLM 架构优化（RoPE、GeGLU、交替局部/全局注意力、unpadding）系统性地引入 encoder-only 模型，在 2T token 上训练并原生支持 8192 上下文长度，在分类和检索任务上全面超越 BERT/RoBERTa/DeBERTaV3，同时推理速度和显存效率大幅领先。

## 研究背景与动机

**Encoder 模型仍是生产主力**：尽管 LLM 大放异彩，encoder-only 模型（如 BERT）因推理轻量、吞吐高，仍广泛用于检索（RAG pipeline 的核心组件）、分类、NER 等非生成式任务。HuggingFace 上下载量前 100 的模型中超过一半是 encoder-based 检索模型。

**老模型积弊深重**：现有 pipeline 大量直接使用原版 BERT，面临多重瓶颈——序列长度仅 512、模型结构未优化、词表过时（不含代码符号）、训练数据体量小且领域单一。

**已有改进不够全面**：MosaicBERT、CrammingBERT 只关注训练效率；NomicBERT、GTE-en-MLM 延长了上下文但未优化推理效率或分类性能，且沿用旧数据配方，在代码任务上表现不佳。

**核心问题**：能否将 decoder-only LLM 近年积累的架构改进和训练策略系统性地迁移到 encoder-only 模型，实现性能与效率的 Pareto 改进？

## 方法详解

### 架构现代化（Modern Transformer）

| 组件 | 改进 | 动机 |
|---|---|---|
| **偏置项** | 除最终 decoder 线性层外全部去除 bias | 将参数预算集中于线性层 |
| **位置编码** | 绝对位置编码 → RoPE | 短长上下文均有效，易于扩展 |
| **归一化** | Pre-Norm + 嵌入后 LayerNorm | 稳定训练，去除首层冗余 LN |
| **激活函数** | GeLU → GeGLU (GLU 变体) | 经验证优于原始 GeLU |

### 效率优化

- **交替注意力（Alternating Attention）**：每 3 层一次全局注意力（RoPE theta=160,000），其余层为 128 token 滑动窗口局部注意力（RoPE theta=10,000）。灵感来自 Gemma 等高效长上下文模型，大幅降低长序列的计算开销。

- **全流程 Unpadding**：在 token embedding 之前就移除 padding token，将 batch 内所有序列拼接为一条长序列处理。利用 Flash Attention 的变长注意力和 RoPE 实现，比之前的 unpadding 方案（如 MosaicBERT 的内部 unpad/repad）快 10-20%。

- **Flash Attention 混合使用**：全局注意力层用 Flash Attention 3（针对 H100 优化），局部注意力层用 Flash Attention 2（支持滑动窗口）。

- **torch.compile**：编译所有兼容模块，额外提升约 10% 吞吐量，编译开销可忽略不计。

### 硬件感知模型设计

- 采用 **Deep & Narrow** 策略：更多更窄的层比更少更宽的层有更好的下游性能。
- ModernBERT-base：22 层，hidden=768，GLU expansion=2304，共 149M 参数。
- ModernBERT-large：28 层，hidden=1024，GLU expansion=5248，共 395M 参数。
- 通过小规模消融实验在一组常用 GPU（T4, A10, L4, RTX 3090/4090）上最大化利用率。
- 参数维度经过精心选择，确保对齐 tensor core 的最优 tiling，在不同 GPU 的 streaming multiprocessors 数量上实现最高效的计算。

### 训练策略

- **数据**：2T token，英语为主，包含 web 文档、代码、科学文献。
- **Tokenizer**：基于 OLMo 的现代 BPE tokenizer，词表大小 50,368（64 的倍数），保留 BERT 的 [CLS]/[SEP] 特殊 token 以向下兼容。
- **目标函数**：仅 MLM（30% masking rate），去除无效的 Next-Sentence Prediction。
- **优化器**：StableAdamW = AdamW + Adafactor 风格的 per-parameter LR 裁剪，比标准梯度裁剪更稳定。
- **学习率调度**：Warmup-Stable-Decay (WSD) + 1-sqrt 衰减，优于线性和余弦衰减。
- **Batch Size 调度**：从小 batch 逐步增大（base: 768 to 4608，large: 448 to 4928），加速初期训练。
- **权重初始化**：base 用 Megatron 初始化；large 从训练好的 base 权重 tile 初始化（借鉴 Phi 系列），显著加速初始 loss 下降。
- **上下文扩展**：先在 1024 长度训练 1.7T token，再将全局注意力 RoPE theta 提升到 160k，在 8192 长度上继续训练 300B token（250B 恒定低 LR + 50B 1-sqrt 衰减，上采样高质量数据源）。

## 实验关键数据

### 表 1：主要任务性能总览

| 模型 | IR-DPR BEIR | IR-ColBERT BEIR | MLDR OOD | MLDR ID | GLUE | CSN | SQA |
|---|---|---|---|---|---|---|---|
| **Base** | | | | | | | |
| BERT-base | 38.9 | 49.0 | 23.9 | 32.2 | 84.7 | 41.2 | 59.5 |
| RoBERTa-base | 37.7 | 48.7 | 22.9 | 32.8 | 86.4 | 44.3 | 59.6 |
| DeBERTaV3-base | 20.2 | 47.1 | 5.4 | 13.4 | 88.1 | 17.5 | 18.6 |
| NomicBERT | 41.0 | 49.9 | 26.7 | 30.3 | 84.0 | 41.6 | 61.4 |
| GTE-en-MLM-base | 41.4 | 48.2 | 34.3 | 44.4 | 85.6 | 44.9 | 71.4 |
| **ModernBERT-base** | **41.6** | **51.3** | 27.4 | 44.0 | **88.4** | **56.4** | **73.6** |
| **Large** | | | | | | | |
| BERT-large | 38.9 | 49.5 | 23.3 | 31.7 | 85.2 | 41.6 | 60.8 |
| RoBERTa-large | 41.4 | 49.8 | 22.6 | 36.1 | 88.9 | 47.3 | 68.1 |
| DeBERTaV3-large | 25.6 | 46.7 | 7.1 | 19.2 | **91.4** | 21.2 | 19.7 |
| GTE-en-MLM-large | 42.5 | 50.7 | 36.4 | 48.9 | 87.6 | 40.5 | 66.9 |
| **ModernBERT-large** | **44.0** | **52.4** | 34.3 | 48.6 | 90.4 | **59.5** | **83.9** |

**关键发现**：

- ModernBERT-base 首次在 GLUE 上超越 DeBERTaV3-base（88.4 vs 88.1），是首个仅用 MLM 做到这一点的模型。
- 代码任务领先幅度最大：CodeSearchNet +11.5, StackQA +2.2 over GTE-en-MLM。
- ColBERT 长上下文检索中，ModernBERT 在 MLDR OOD 上领先其他长上下文模型至少 9 nDCG@10。

### 表 2：推理效率（RTX 4090，千 token/秒）

| 模型 | 参数量 | 短 Max BS | 短固定 | 短变长 | 长固定 | 长变长 |
|---|---|---|---|---|---|---|
| BERT-base | 110M | 1096 | 180.4 | 90.2 | - | - |
| DeBERTaV3-base | 183M | 236 | 70.2 | 35.1 | - | - |
| NomicBERT | 137M | 588 | 117.1 | 58.5 | 46.1 | 23.1 |
| GTE-en-MLM-base | 137M | 640 | 123.7 | 61.8 | 46.8 | 23.4 |
| **ModernBERT-base** | 149M | **1604** | **148.1** | **147.3** | **123.7** | **133.8** |
| GTE-en-MLM-large | 435M | 472 | 38.7 | 19.3 | 16.2 | 8.1 |
| **ModernBERT-large** | 395M | **770** | **52.3** | **52.9** | **46.8** | **49.8** |

**关键发现**：

- ModernBERT-base 最大 batch size（1604）是其他模型的 2 倍以上，显存效率领先。
- 长文本（8192）处理速度 2.65x-3x 快于次快模型。
- ModernBERT-large 长文本速度（46.8k tok/s）接近 GTE-base（47.5k），远超 GTE-large（16.5k）。
- 变长输入下 ModernBERT 优势更明显，比 GTE 快 14.5-118.8%，得益于 unpadding + 局部注意力。

## 亮点

1. **系统性 Pareto 改进**：首次将 decoder-only LLM 的全套现代架构改进（RoPE、GeGLU、交替注意力、Flash Attention、unpadding、torch.compile）移植到 encoder-only 模型，在下游性能和推理效率上同时创下新 SOTA。

2. **首个超越 DeBERTaV3-base 的 MLM 模型**：DeBERTaV3 依赖 RTD（Replaced Token Detection）目标才能在 GLUE 上领先，ModernBERT 仅用 MLM 就在 base 规模首次超越它，打破了 RTD 才能做好分类 的认知。

3. **代码能力破圈**：唯一在预训练中包含代码数据的 encoder 模型，配合 code-aware tokenizer（OLMo-based），在 CodeSearchNet/StackQA 上大幅领先，对代码检索场景价值巨大。

4. **工程细节扎实**：硬件感知的 Deep & Narrow 设计、全流程 unpadding（embedding 前即去除 padding）、混合 FA2/FA3 策略、tensor core tiling 优化，工程完成度极高。

5. **开源生态友好**：发布 FlexBERT 模块化框架和所有中间训练 checkpoint（借鉴 Pythia），便于社区研究和复现。

## 局限与展望

1. **仅支持英语**：2T token 全部为英语数据，不适用于多语言场景，对低资源语言更不友好。

2. **MLM-only 目标的天花板**：DeBERTaV3-large 在 GLUE 上仍略高（91.4 vs 90.4），RTD + MLM 联合训练可能进一步提升分类性能，作者明确将此列为未来方向。

3. **长上下文 DPR OOD 弱于 GTE**：MLDR OOD 设置下明显弱于 GTE-en-MLM（27.4 vs 34.3），局部注意力虽提升效率但可能影响零样本长上下文单向量检索的泛化能力。

4. **模型规模探索不足**：仅发布 base（149M）和 large（395M）两个尺寸，未探索更大（1B+）或更小（tiny/mini）的 encoder 模型的缩放行为。

5. **Web 数据偏差**：大量训练数据来自网页，模型表征不可避免地包含 web 数据中的社会偏见，作者对此缺少量化分析。

## 相关工作

| 维度 | BERT / RoBERTa | DeBERTaV3 | NomicBERT | GTE-en-MLM | **ModernBERT** |
|---|---|---|---|---|---|
| 上下文长度 | 512 | 512 | 8192 | 8192 | **8192** |
| 训练数据量 | 16B / 33B | ~64B | ~32B | ~30B | **2T** |
| 代码数据 | 无 | 无 | 无 | 无 | **有** |
| 推理效率优化 | 无 | 无 | 部分 | unpadding | **全套** |
| GLUE (base) | 84.7 / 86.4 | 88.1 | 84.0 | 85.6 | **88.4** |
| BEIR DPR (base) | 38.9 / 37.7 | 20.2 | 41.0 | 41.4 | **41.6** |
| 位置编码 | 绝对 | 相对 disentangled | RoPE | RoPE | **RoPE** |
| 注意力类型 | 全局 | 全局 | 全局+局部 | 全局 | **交替全局/局部** |

## 评分

- 新颖性: ⭐⭐⭐ — 架构改进均来自已有技术，核心贡献在于系统集成和工程优化
- 实验充分度: ⭐⭐⭐⭐⭐ — 覆盖 GLUE、BEIR、MLDR、CodeSearchNet、StackQA 等广泛任务，含效率评测和消融
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，工程细节详实，贡献声明透明
- 价值: ⭐⭐⭐⭐⭐ — 对 encoder-only 模型的一次全面代际升级，直接可用于生产 pipeline 替换老旧 BERT

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] SkyLadder: Better and Faster Pretraining via Context Window Scheduling](../../NeurIPS2025/llm_efficiency/skyladder_better_and_faster_pretraining_via_context_window_scheduling.md)
- [\[ACL 2025\] EpMAN: Episodic Memory AttentioN for Generalizing to Longer Contexts](epman_episodic_memory_attention_for_generalizing_to_longer_contexts.md)
- [\[ACL 2025\] Scaling Context, Not Parameters: Training a Compact 7B Language Model for Efficient Long-Context Processing](scaling_context_not_parameters_training_a_compact_7b_language_model_for_efficien.md)
- [\[ICML 2025\] EasyInv: Toward Fast and Better DDIM Inversion](../../ICML2025/llm_efficiency/easyinv_toward_fast_and_better_ddim_inversion.md)
- [\[ACL 2025\] Squeezed Attention: Accelerating Long Context Length LLM Inference](squeezed_attention_accelerating_long_context_length_llm_inference.md)

</div>

<!-- RELATED:END -->
