<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎬 视频理解

**🧠 NeurIPS2025** · 共 **9** 篇

**[A Little Depth Goes a Long Way: The Expressive Power of Log-Depth Transformers](a_little_depth_goes_a_long_way_the_expressive_power_of_logde.md)**

:   本文证明了将 Transformer 的深度从常数增长到 Θ(log n) 就能解锁识别正则语言和图连通性这两类固定深度 Transformer 无法表达的问题，且深度扩展比宽度（需超多项式增长）和 CoT 步数（需超对数增长）都更高效。

**[AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding](adavideorag_omnicontextual_adaptive_retrievalaugmented_effic.md)**

:   提出 AdaVideoRAG，通过轻量级意图分类器将查询按难度路由到三级检索路径（无检索/朴素检索/图检索），结合全知识索引模块（caption+ASR+OCR+视觉+知识图谱）实现长视频理解的效率-精度最优平衡，在 MLVU 上为 Qwen2.5-VL-7B 带来 39.8% 提升。

**[Empower Words: DualGround for Structured Phrase and Sentence-Level Temporal Grounding](empower_words_dualground_for_structured_phrase_and_sentencel.md)**

:   论文指出现有视频时序定位模型在跨模态注意力中往往过度依赖句末 [EOS] token 的全局语义、忽视词级局部信号，提出 DualGround 双分支架构，将句子级全局语义与短语级局部语义显式解耦建模，在 QVHighlights 和 Charades-STA 上实现 Moment Retrieval 与 Highlight Detection 的 SOTA。

**[Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders](enhancing_temporal_understanding_in_videollms_through_stacke.md)**

:   提出 STAVEQ2，在 Vision Encoder 中堆叠参数高效的时序注意力模块（STA），解决现有 Video-LLM 在细粒度时序理解（如区分"从左到右拉"和"从右到左拉"）上的根本性架构缺陷，在 VITATECS/MVBench/Video-MME 上提升最高 5.5%。

**[FastVID: Dynamic Density Pruning for Fast Video Large Language Models](fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)**

:   提出 FastVID，通过动态时序分割 (DySeg) + 密度空时剪枝 (STPrune) 从时间和视觉两个维度系统性消除视频 token 冗余，在 LLaVA-OneVision-7B 上剪掉 90.3% 视频 token 后仍保留 98% 精度，LLM prefill 阶段加速 7.1×。

**[Foresight: Adaptive Layer Reuse for Accelerated and High-Quality Text-to-Video Generation](foresight_adaptive_layer_reuse_for_accelerated_and_highquali.md)**

:   提出 Foresight，一种训练无关的自适应层复用框架，通过动态 MSE 阈值决策在 DiT 去噪过程中哪些层可复用缓存、哪些需重新计算，在 OpenSora/Latte/CogVideoX 上实现最高 1.63× 端到端加速且保持视频质量。

**[TempSamp-R1: Effective Temporal Sampling with Reinforcement Fine-Tuning for Video LLMs](tempsampr1_effective_temporal_sampling_with_reinforcement_fi.md)**

:   提出 TempSamp-R1，针对视频时序定位任务改进 GRPO 强化微调框架，通过 off-policy 时间精确引导 + 非线性软优势计算 + 混合 CoT 训练，在 Charades-STA/ActivityNet/QVHighlights 上分别提升 +2.7%/+5.3%/+3.0%。

**[Tool-Augmented Spatiotemporal Reasoning for Streamlining Video Question Answering Task](toolaugmented_spatiotemporal_reasoning_for_streamlining_vide.md)**

:   论文为复杂 VideoQA 提出一套轻量但可扩展的 Video Toolkit，并设计 STAR（Spatiotemporal Reasoning Framework）来调度时间工具与空间工具的调用顺序，逐步定位视频关键区域，显著增强 GPT-4o 的时空推理能力，在 VideoMME 上提升 8.2%，在 LongVideoBench 上提升 4.6%。

**[Two Causally Related Needles in a Video Haystack](two_causally_related_needles_in_a_video_haystack.md)**

:   提出CAUSAL2NEEDLES benchmark评估VLM的长视频双针(2-needle)因果推理能力：需要从视频两个不同位置提取因果关联的事件信息并联合推理，利用"桥接实体"迫使模型先理解结果再追溯原因，揭示即使GPT-4o在2-needle因果问题上仅达13.4%的Both准确率（vs人类79.3%）。
