---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM 效率

**💬 ACL2025** · 共 **12** 篇

**[Extending LLM Context Window with Adaptive Grouped Positional Encoding: A Training-Free Method](adaptive_grouped_pe_context_window.md)**

:   提出 AdaGroPE（Adaptive Grouped Positional Encoding），一种无需训练的即插即用方法，通过让位置复用次数随距离递增式增长、并根据输入序列长度动态调整位置编码映射，将 LLM 上下文窗口外推到远超预训练长度，在多个 benchmark 上达到 SOTA 甚至超过原生长上下文模型。

**[CLaSp: In-Context Layer Skip for Self-Speculative Decoding](clasp_self_speculative_decoding.md)**

:   CLaSp 提出一种无需训练的自推测解码方法，通过动态规划算法在每个验证步骤后根据上下文动态调整跳层策略，利用上一次验证的完整隐状态作为目标来选择最优跳层集合，在 LLaMA3 系列上实现 1.3-1.7× 加速且不改变生成分布。

**[Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs](gor_rag_long_context_summary.md)**

:   提出 Graph of Records（GoR），将 LLM 历史响应与检索文本块构建为图结构，用 GNN 学习节点间的语义和逻辑关联，配合 BERTScore 自监督训练目标，在四个长文本全局摘要数据集上比检索基线提升 8-19%（ROUGE 指标）。

**[GradOT: Training-free Gradient-preserving Offsite-tuning for Large Language Models](gradot_offsite_tuning.md)**

:   从优化理论角度首次系统分析 Offsite-tuning 问题，提出梯度保持压缩分数（GCS），并基于此设计了 GradOT 方法，对 MHA 使用动态秩分解（DRD）、对 MLP 使用选择性通道剪枝（SCP），在免训练条件下同时实现性能保持和隐私保护。

**[KV-Latent: Dimensional-level KV Cache Reduction with Frequency-aware Rotary Positional Embedding](kv_latent_cache_reduction.md)**

:   提出KV-Latent范式，通过对KV头的维度进行降采样将其映射到潜空间，仅需不到1%预训练量的额外训练即可恢复性能，在LLaMA-3-8B上KV Cache减少50%（dqk=dvo=64→50%缓存），同时TTFT延迟降低8%。

**[LADM: Long-context Training Data Selection with Attention-based Dependency Measurement](ladm_long_context_data.md)**

:   提出 LADM 框架，利用注意力机制的内在检索能力来度量长上下文数据中的跨 span 依赖关系，从大规模预训练语料中高效筛选高质量长上下文训练数据，仅用 1B token 持续预训练即可显著提升多种 LLM 的长上下文能力。

**[What Really Matters in Many-Shot Attacks? An Empirical Study of Long-Context Vulnerabilities in LLMs](many_shot_attacks_long_context.md)**

:   系统分析 Many-Shot Jailbreaking（MSJ）攻击的关键因素，发现上下文长度是攻击成功的决定性因素，而内容的有害性、主题、格式几乎不重要——即使重复安全内容、随机无意义文本（Lorem Ipsum）都能在长上下文下突破模型安全对齐。

**[MHA2MLA: Towards Economical Inference by Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs](mha2mla_deepseek_latent_attention.md)**

:   MHA2MLA 首次提出将已训练好的 MHA 模型高效迁移到 DeepSeek 的 MLA 架构的方法，通过贡献度感知的 partial-RoPE 移除和联合 SVD 低秩近似，仅用 0.6%-1% 的训练数据即可恢复性能，将 Llama2-7B 的 KV cache 压缩 92.19% 且 LongBench 性能仅下降 1%。

**[Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](native_sparse_attention.md)**

:   DeepSeek提出NSA——一种原生可训练的稀疏注意力机制，通过"压缩+选择+滑动窗口"的层次化稀疏策略和硬件对齐的Triton kernel设计，在27B参数模型上实现了超越Full Attention的性能，同时在64k序列上获得前向9倍、解码11.6倍的加速。

**[Sliding Windows Are Not the End: Exploring Full Ranking with Long-Context Large Language Models](sliding_windows_full_ranking.md)**

:   本文系统研究了长上下文LLM在段落排序中的应用，提出用 full ranking（一次性排序所有段落）替代传统滑动窗口策略，并设计了多轮滑动窗口标签构造方法和重要性感知损失函数来微调 full ranking 模型，在效率提升约30-65%的同时实现了排序效果的全面超越。

**[SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers](spindlekv_layered_kv_cache.md)**

:   SpindleKV 提出分层处理 KV cache 压缩的策略——深层使用注意力驱动的 token eviction（利用稀疏注意力），浅层使用基于相似性学习的 codebook 替换（利用 token 间高相似度），并解决了 GQA 兼容性问题，实现 50% KV cache 缩减而不损失性能。

**[How to Train Long-Context Language Models (Effectively)](train_long_context_effectively.md)**

:   本文系统研究了如何通过持续预训练和监督微调（SFT）有效训练长上下文语言模型，提出了包括数据配比、训练长度缩放等一系列关键发现，最终训练出的 ProLong-8B 模型仅用 Llama-3.1 5% 的长上下文训练数据量即在 128K 长度上达到同规模最优性能。
