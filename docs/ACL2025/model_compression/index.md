---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📦 模型压缩

**💬 ACL2025** · 共 **37** 篇

**[AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation](aligndistil_token_level_alignment.md)**

:   AlignDistil 证明了 RLHF 目标函数与 token 级蒸馏过程的理论等价性，并据此设计了一种简单的蒸馏方法：用 DPO 模型和反向 DPO 模型的 logit 分布线性组合构造教师分布，配合 token 自适应外推机制实现 token 级奖励优化，在 AlpacaEval 2.0、MT-Bench 和 Arena-Hard 上优于现有方法且收敛更快。

**[APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs](apb_distributed_long_context.md)**

:   APB 提出了一种分布式长上下文推理框架，通过在序列并行框架中引入本地 KV cache 压缩和跨 GPU 传递压缩上下文块的机制，在不损失任务性能的前提下实现了相比 FlashAttn/RingAttn/StarAttn 分别高达 9.2x/4.2x/1.6x 的 prefill 加速。

**[BeamLoRA: Beam-Constraint Low-Rank Adaptation](beamlora_beam_constraint_lora.md)**

:   BeamLoRA 发现 LoRA 模块中不同 rank 的重要性存在显著差异且随训练动态演变，受 beam search 启发，提出在训练过程中动态评估 rank 重要性、剪枝不重要的 rank 并将参数空间扩展给重要 rank，在固定总 rank 下提升性能，在三个基座模型的 12 个数据集上持续优于 LoRA 及其变体。

**[Beyond Text Compression: Evaluating Tokenizers Across Scales](beyond_text_compression_tokenizers.md)**

:   本文系统评估了 6 种 tokenizer 在 350M 和 2.7B 参数模型上的影响，发现 tokenizer 选择对英文任务影响极小但对多语言任务（如机器翻译）有显著且跨尺度一致的影响，并提出了基于 Zipf 定律的新型内在评估指标，比文本压缩率能更好地预测多语言场景下的下游性能。

**["Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization](bf16_or_death_quantization_tradeoffs.md)**

:   这是迄今最全面的 LLM 量化实证研究，在 Llama-3.1 全系列（8B/70B/405B）上对 FP8/INT8/INT4 进行了超过 50 万次评估，发现 FP8 几乎无损、INT8 仅降 1-3%、INT4 出奇地有竞争力，并给出了不同部署场景的量化格式选择建议。

**[Capture the Key in Reasoning to Enhance CoT Distillation Generalization](capture_key_cot_distillation.md)**

:   提出 EDIT（mistakE-Driven key reasonIng step distillaTion），通过构造正确/错误配对的 dual CoTs 数据，利用最小编辑距离算法定位关键推理步骤，并以 token 级细粒度损失函数引导小模型聚焦学习这些关键步骤，而非简单模仿教师的推理形式。

**[Cross-Lingual Generalization and Compression: From Language-Specific to Shared Neurons](cross_lingual_neurons_compression.md)**

:   本文通过追踪多语言语言模型预训练过程中的检查点，发现模型从语言特定表示逐渐压缩为跨语言共享表示：中间层的语言识别能力下降、语义概念的"专家神经元"跨语言对齐，操控从西班牙语数据提取的概念神经元后模型反而生成语义相关的英语文本。

**[DAC: A Dynamic Attention-aware Approach for Task-Agnostic Prompt Compression](dac_prompt_compression.md)**

:   DAC 提出动态注意力感知的 prompt 压缩方法，通过融合信息熵和注意力分数作为 token 重要性度量，并动态感知压缩过程中的熵偏移来进行细粒度压缩，在 LongBench 上比 SOTA 方法提升平均 1.33 分。

**[DRAG: Distilling RAG for SLMs from LLMs to Transfer Knowledge and Mitigate Hallucination](drag_distilling_rag_slm.md)**

:   DRAG 提出了一种从大模型向小模型蒸馏 RAG 能力的框架：用大模型（如 GPT-4o）为给定问题生成证据和知识图谱三元组，经排序过滤后作为结构化上下文输入给小模型（2B-9B），无需微调即可将小模型在 ARC-C 上提升高达 27.7%，同时显著减少幻觉。

**[DRPruning: Efficient Large Language Model Pruning through Distributionally Robust Optimization](drpruning_robust_pruning.md)**

:   DRPruning 将分布稳健优化（DRO）引入 LLM 结构化剪枝，通过 scaling law 预测各领域最终 loss 作为参考、动态调整训练数据分布来平衡剪枝后各领域性能，在单语和多语设置下分别以 -5.59% PPL 和 +2.95% 下游任务的提升超越 Sheared LLaMA。

**[EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models](eac_moe_expert_aware_compression.md)**

:   EAC-MoE 从 MoE 模型的专家选择特性出发，提出量化时校准路由器缓解 expert-shift 问题（QESC）+ 推理时基于专家选择频率动态剪枝不重要专家（PESF），在 Mixtral-8x7B 上实现 4.92× 内存压缩和 1.68× 推理加速且精度损失不到 1%。

**[EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](efficientqat.md)**

:   EfficientQAT 提出两阶段 QAT 框架——先逐块训练所有参数（Block-AP）提供良好初始化，再端到端训练量化参数（E2E-QP）捕获跨块交互，在单张 A100 上 41 小时完成 Llama-2-70B 的 2-bit 量化，精度仅降 3 点。

**[FedEx-LoRA: Exact Aggregation for Federated and Efficient Fine-Tuning of Large Language Models](fedex_lora_federated_exact_aggregation.md)**

:   FedEx-LoRA 发现联邦学习中独立平均 LoRA 的 A 和 B 矩阵会导致不精确的全局更新（"乘积的均值≠均值的乘积"），通过在冻结权重矩阵中加入残差误差项实现精确聚合，在多个推理和 NLU 任务上一致优于 FedIT 和 FFA-LoRA。

**[Language Fusion for Parameter-Efficient Cross-lingual Transfer (FLARE)](flare_crosslingual_lora.md)**

:   FLARE 在 LoRA 适配器的低秩瓶颈中通过轻量线性/非线性变换融合源语言（英语）和目标语言的逐层表示，无需额外参数即可实现参数高效的跨语言迁移，在 Llama 3.1 上 QA 精确匹配提升 4.9%。

**[Flipping Knowledge Distillation: Leveraging Small Models' Expertise to Enhance LLMs in Text Matching](flipping_kd_small_to_large.md)**

:   本文提出"反向知识蒸馏"范式——让 LLM 从微调过的小模型学习文本匹配的领域专家知识，通过将 decoder-only LLM 重新解释为 encoder-decoder 架构（用 LoRA 的压缩矩阵做 encoder）并设计 Margin-aware Contrastive Loss 来对齐表示相似度。

**[A Silver Bullet or a Compromise for Full Attention? A Comprehensive Study of Gist Token-based Context Compression](gist_token_context_compression.md)**

:   系统研究Gist Token上下文压缩方法，提出统一框架分类现有架构（记忆位置×粒度），发现Fine-grained KV Cache在RAG/QA上近无损但在合成召回上有明显缺陷，识别出三种失败模式（边界丢失/意外丢失/中途丢失），并提出细粒度自编码和分段token重要性估计两种改进策略。

**[Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to Enhance LLM Reasoning](graph_counselor_multiagent_graphrag.md)**

:   Graph Counselor 提出了一个多智能体协作的 GraphRAG 推理框架，通过 Planning/Thought/Execution 三个 Agent 自适应提取图结构信息，并引入多视角自反思机制纠正推理偏差，在多个图推理任务上超越现有方法。

**[L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models](l4q_parameter_efficient_quantization_aware_finetuning.md)**

:   提出 L4Q，将量化感知训练 (QAT) 与 LoRA 深度整合：先合并权重与LoRA参数再统一量化，通过定制反向传播路径消除权重梯度存储开销，实现联合优化量化与微调参数，在4-bit和3-bit量化下显著超越现有方法。

**[Language Models Resist Alignment: Evidence From Data Compression](language_models_resist_alignment.md)**

:   本文从压缩理论视角揭示了LLM存在"弹性"(elasticity)现象——模型倾向于保持预训练分布而抵触对齐分布，且对齐后的模型在受到扰动时会以与数据量差距成反比的速率回弹到预训练状态，这解释了为什么对齐如此脆弱且容易被少量微调逆转。

**[Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders](language_specific_features.md)**

:   利用 Sparse Autoencoders (SAEs) 分析多语言 LLM 的内部表示，发现存在强烈的语言特定 SAE features，这些 features 不仅与语言特有 token 相关还与语言上下文相关，消融它们只影响对应语言能力，且多个语言 features 之间存在协同效应；进一步利用这些 features 增强 steering vectors 实现对生成语言的精确控制。

**[Towards the Law of Capacity Gap in Distilling Language Models](law_of_capacity_gap_distilling_language_models.md)**

:   揭示了语言模型蒸馏中的"容量差距定律"——最优教师模型的参数量与学生模型成线性关系（约 2.5 倍），将 LLM 蒸馏中的"不可能三角"转化为可解问题，并据此成功蒸馏出 3B 的 MiniMA 模型。

**[Memorization Inheritance in Sequence-Level Knowledge Distillation for Neural Machine Translation](memorization_inheritance_seqkd.md)**

:   本文首次系统研究了序列级知识蒸馏（SeqKD）中教师模型的记忆行为如何传递给学生模型，发现学生模型虽未直接接触原始训练数据，但其提取式记忆率比基线模型高 57%，幻觉率也增加，并提出 Adaptive-SeqKD 通过在高质量子集上微调教师来缓解这些问题。

**[MoQAE: Mixed-Precision Quantization for Long-Context LLM Inference via Mixture of Quantization-Aware Experts](moqae_mixed_precision_kv_cache.md)**

:   MoQAE 创造性地将不同量化比特宽度配置视为 MoE 中的"专家"，通过轻量路由器学习每个 chunk 的最优量化策略，结合路由冻结和路由共享机制，在几乎不损失精度的情况下大幅减少长上下文推理的 KV cache 内存。

**[mPLUG-DocOwl2: High-resolution Compressing for OCR-free Multi-page Document Understanding](mplug_docowl2_doc_compress.md)**

:   提出High-resolution DocCompressor模块，利用低分辨率全局视觉特征作为query通过交叉注意力将高分辨率文档图像压缩为仅324个token（不到同类方法的20%），在多页文档理解benchmark上达到SOTA且首token延迟降低50%+。

**[Unraveling LoRA Interference: Orthogonal Subspaces for Robust Model Merging](osrm_lora_merging_orthogonal.md)**

:   OSRM 发现 LoRA 模型合并失败的根因是参数与数据分布的交互干扰（而非仅仅是参数冲突），提出在微调前通过数据协方差矩阵的特征分解来初始化 LoRA 矩阵 A，使其子空间与其他任务的数据分布正交，从而在合并时最小化跨任务干扰，在 8 个数据集、5 个模型上显著提升合并性能。

**[Prompt Candidates, then Distill: A Teacher-Student Framework for LLM-driven Data Annotation](prompt_distill_teacher_student.md)**

:   提出候选标注+蒸馏范式（CanDist）——当 LLM 对样本不确定时输出所有可能标签（而非强制给唯一标签），然后用小语言模型（SLM）从候选标注中蒸馏出唯一标签，理论证明候选标注蒸馏比直接使用单标签有更好的理论保证，在六个文本分类任务上验证有效。

**[PTQ1.61: Push the Real Limit of Extremely Low-Bit Post-Training Quantization Methods for Large Language Models](ptq161_low_bit_quantization.md)**

:   首次将LLM权重真正量化到1.61-bit（此前号称sub-2bit的方法实际都超过2bit），通过一维结构化掩码（仅增加0.0002-bit/权重）保留显著通道、块级缩放因子优化和量化预处理三大创新，在LLaMA系列上以更低比特超越BiLLM和PB-LLM。

**[Quaff: Quantized Parameter-Efficient Fine-Tuning under Outlier Spatial Stability Hypothesis](quaff_quantized_peft.md)**

:   本文提出 Outlier Spatial Stability Hypothesis (OSSH)——微调期间激活异常通道的空间位置保持稳定——并基于此设计了 Quaff 框架，通过目标动量缩放仅处理少量不变的异常通道，实现 1.73× 延迟降低和 30% 内存节省，同时在 GPQA 上精度还提升了 0.6%。

**[SEE: Strategic Exploration and Exploitation for Cohesive In-Context Prompt Optimization](see_strategic_exploration_exploitation_prompt_optimization.md)**

:   本文提出 SEE 框架，首次将指令（instruction）和示例（examples）作为整体进行联合优化，采用元启发式优化原则设计四阶段探索-利用策略，在35个基准任务上实现平均13.94%的准确率提升并降低58.67%的计算成本。

**[Mitigating Selection Bias with Node Pruning and Auxiliary Options](selection_bias_node_pruning.md)**

:   提出Bias Node Pruning (BNP)和Auxiliary Option Injection (AOI)两种互补方法，从模型内部和输入端同时缓解LLM在多选题中的选择偏差，仅剪除0.002%权重即可将Llama-3准确率从52.3%提升至65.3%（+24.9%组合提升）。

**[Semantic Exploration with Adaptive Gating for Efficient Problem Solving with Language Models](semantic_exploration_adaptive_gating.md)**

:   针对 LLM 树搜索推理中"简单题也做复杂搜索"和"语义重复路径反复扩展"两大浪费问题，提出 SEAG 框架：先用 entropy 门控决定是否启动树搜索，再用语义聚类合并等价推理步骤，最终在准确率平均提升 4.3% 的同时仅需 RAP 31% 的推理开销。

**[State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](state_offset_tuning_ssm_peft.md)**

:   针对 SSM（如 Mamba）提出 State-offset Tuning，一种新的"状态基"PEFT 方法家族，通过在每个时间步直接注入可训练的状态偏移量 $h'$ 替代 Prefix-Tuning 的虚拟 token，解决了 prompt-based 方法在 SSM 上表达能力受限的问题，在更少参数量下持续优于 LoRA 和 Prefix-Tuning。

**[STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning](stun_moe_pruning.md)**

:   STUN 提出了结构化→非结构化的两阶段 MoE 剪枝方法，第一阶段用 O(1) 前向传播实现可扩展的专家级剪枝，第二阶段在剩余专家内做非结构化剪枝，在 480B 参数的 Snowflake Arctic 上以 40% 稀疏度几乎无性能损失。

**[TableLoRA: Low-rank Adaptation on Table Structure Understanding for Large Language Models](table_lora_structure_understanding.md)**

:   TableLoRA 提出面向表格任务的专用 LoRA 模块，通过特殊 token 编码器改善表格序列化，并用 2D LoRA 编码单元格的行列位置信息，在参数高效微调设置下相比 vanilla LoRA 在 HiTab 上提升 5.9%，弥合了 LoRA 与全量微调之间 40.56% 的性能差距。

**[Trans-PEFT: Transferable Parameter-Efficient Fine-Tuning on Evolving Base Models](trans_peft_transferable.md)**

:   Trans-PEFT 发现基座模型更新（如 Qwen2→Qwen2.5）主要改变 FFN 层的任务知识存储而较少影响 Attention 层的任务模式，据此提出层内知识掩码和跨层知识丢弃两种策略，使在旧版本上训练的 PEFT 模块可直接迁移到新版本而不需重新微调，性能提升可达 30%。

**[UniICL: An Efficient ICL Framework Unifying Compression, Selection, and Generation](uniicl_icl_framework.md)**

:   提出 UniICL 框架，用**一个冻结的 LLM** 同时完成 demonstration 压缩（compress→virtual tokens）、demonstration 选择（基于压缩后的 virtual token 相似度排序）和最终响应生成三个任务，仅需 17M 可训练参数（projection layer + learnable embedding），配合 Demonstration Bank 缓存机制避免重复压缩，实现 12× 压缩率下从 4-shot 扩展到 64-shot ICL（24GB 显存内），在多个 out-of-domain 数据集上超越 AutoCompressor、ICAE、LLMLingua 等基线。

**[UniQuanF: Unifying Uniform and Binary-coding Quantization for Accurate Compression of Large Language Models](uniquanf_unified_quantization.md)**

:   UniQuanF 统一了均匀量化（UQ,表现力弱但优化性强）和二进制编码量化（BCQ,表现力强但优化性差）的优势，通过统一初始化、局部周期映射和统一定理，实现无额外部署开销的高精度 LLM 量化，在 GSM8K 上提升最高 4.60%。
