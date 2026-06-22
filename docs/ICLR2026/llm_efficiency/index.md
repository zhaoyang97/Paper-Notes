---
title: >-
  ICLR2026 LLM效率论文汇总 · 169篇论文解读
description: >-
  169篇ICLR2026的 LLM 效率方向论文解读，涵盖 LLM、扩散模型、推理、对齐/RLHF、压缩/编码、联邦学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICLR2026"
  - "LLM 效率"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "扩散模型"
  - "推理"
  - "对齐/RLHF"
  - "压缩/编码"
  - "联邦学习"
item_list:
  - u: "a_two-phase_deep_learning_framework_for_adaptive_time-stepping_in_high-speed_flo/"
    t: "A Two-Phase Deep Learning Framework for Adaptive Time-Stepping in High-Speed Flow Modeling"
  - u: "accelerating_diffusion_large_language_models_with_slowfast_sampling_the_three_go/"
    t: "Accelerating Diffusion Large Language Models with SlowFast Sampling: The Three Golden Principles"
  - u: "attention_is_all_you_need_for_kv_cache_in_diffusion_llms/"
    t: "Attention Is All You Need for KV Cache in Diffusion LLMs"
  - u: "autoencoding-free_context_compression_for_llms_via_contextual_semantic_anchors/"
    t: "Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors"
  - u: "autosp_unlocking_long-context_llm_training_via_compiler-based_sequence_paralleli/"
    t: "AutoSP: Unlocking Long-Context LLM Training Via Compiler-Based Sequence Parallelism"
  - u: "ba-lora_bias-alleviating_low-rank_adaptation_to_mitigate_catastrophic_inheritanc/"
    t: "BA-LoRA: Bias-Alleviating Low-Rank Adaptation to Mitigate Catastrophic Inheritance in Large Language Models"
  - u: "beyond_fixed_training-free_variable-length_denoising_for_diffusion_large_languag/"
    t: "Beyond Fixed: Training-Free Variable-Length Denoising for Diffusion Large Language Models"
  - u: "beyond_masks_efficient_flexible_diffusion_language_models_via_deletion-insertion/"
    t: "Beyond Masks: Efficient, Flexible Diffusion Language Models via Deletion-Insertion Processes"
  - u: "beyond_real_imaginary_extension_of_rotary_position_embeddings_for_long-context_l/"
    t: "Beyond Real: Imaginary Extension of Rotary Position Embeddings for Long-Context LLMs"
  - u: "bora_towards_more_expressive_low-rank_adaptation_with_block_diversity/"
    t: "BoRA: Towards More Expressive Low-Rank Adaptation with Block Diversity"
  - u: "cache_what_lasts_token_retention_for_memory-bounded_kv_cache_in_llms/"
    t: "Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs"
  - u: "cactus_accelerating_auto-regressive_decoding_with_constrained_acceptance_specula/"
    t: "Cactus: Accelerating Auto-Regressive Decoding with Constrained Acceptance Speculative Sampling"
  - u: "cartridges_lightweight_and_general-purpose_long_context_representations_via_self/"
    t: "Cartridges: Lightweight and General-Purpose Long Context Representations via Self-Study"
  - u: "cascadia_an_efficient_cascade_serving_system_for_large_language_models/"
    t: "Cascadia: An Efficient Cascade Serving System for Large Language Models"
  - u: "command-v_training-free_representation_finetuning_transfer/"
    t: "Command-V: Training-Free Representation Finetuning Transfer"
  - u: "composer_a_search_framework_for_hybrid_neural_architecture_design/"
    t: "Composer: A Search Framework for Hybrid Neural Architecture Design"
  - u: "concur_a_framework_for_continual_constrained_and_unconstrained_routing/"
    t: "CONCUR: A Framework for Continual Constrained and Unconstrained Routing"
  - u: "cpqs-tuning_a_model_self-perception-based_data_filtering_algorithm_for_efficient/"
    t: "CPQS-Tuning: A Model Self-Perception-Based Data Filtering Algorithm for Efficient Instruction Fine-Tuning"
  - u: "cr-net_scaling_parameter-efficient_training_with_cross-layer_low-rank_structure/"
    t: "CR-Net: Scaling Parameter-Efficient Training with Cross-Layer Low-Rank Structure"
  - u: "dash_deterministic_attention_scheduling_for_high-throughput_reproducible_llm_tra/"
    t: "DASH: Deterministic Attention Scheduling for High-throughput Reproducible LLM Training"
  - u: "deep_hierarchical_learning_with_nested_subspace_networks_for_large_language_mode/"
    t: "Deep Hierarchical Learning with Nested Subspace Networks for Large Language Models"
  - u: "defensivekv_taming_the_fragility_of_kv_cache_eviction_in_llm_inference/"
    t: "DefensiveKV: Taming the Fragility of KV Cache Eviction in LLM Inference"
  - u: "demystifying_and_enhancing_the_efficiency_of_large_language_model_based_search_a/"
    t: "Demystifying and Enhancing the Efficiency of Large Language Model Based Search Agents"
  - u: "developmental_federated_tuning_a_cognitive-inspired_paradigm_for_efficient_llm_a/"
    t: "Developmental Federated Tuning: A Cognitive-Inspired Paradigm for Efficient LLM Adaptation"
  - u: "diffadapt_difficulty-adaptive_reasoning_for_token-efficient_llm_inference/"
    t: "DiffAdapt: Difficulty-Adaptive Reasoning for Token-Efficient LLM Inference"
  - u: "difficultydiversity_collaborative_filtering_for_data-efficient_llm_fine-tuning/"
    t: "Difficulty–Diversity Collaborative Filtering for Data-Efficient LLM Fine-Tuning"
  - u: "diffusion_language_model_knows_the_answer_before_it_decodes/"
    t: "Diffusion Language Models Know the Answer Before Decoding"
  - u: "diffusion_llms_can_do_faster-than-ar_inference_via_discrete_diffusion_forcing/"
    t: "Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing"
  - u: "dirmoe_dirichlet-routed_mixture_of_experts/"
    t: "DirMoE: Dirichlet-Routed Mixture of Experts"
  - u: "disrouter_distributed_self-routing_for_llm_selections/"
    t: "DiSRouter: Distributed Self-Routing for LLM Selections"
item_total: 169
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM 效率

**🔬 ICLR2026** · **169** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (8)](../../CVPR2026/llm_efficiency/index.md) · [💬 ACL2026 (23)](../../ACL2026/llm_efficiency/index.md) · [🧪 ICML2026 (48)](../../ICML2026/llm_efficiency/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_efficiency/index.md) · [🧠 NeurIPS2025 (34)](../../NeurIPS2025/llm_efficiency/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/llm_efficiency/index.md)

🔥 **高频主题：** LLM ×45 · 扩散模型 ×20 · 推理 ×7 · 对齐/RLHF ×5 · 压缩/编码 ×3

**[A Two-Phase Deep Learning Framework for Adaptive Time-Stepping in High-Speed Flow Modeling](a_two-phase_deep_learning_framework_for_adaptive_time-stepping_in_high-speed_flo.md)**

:   ShockCast 把"高速流动的自适应时间步进"拆成两个学习问题——先用一个 Neural CFL 模型根据当前流场预测下一步该走多大的时间步 $\Delta t$，再用一个被 $\Delta t$ 条件化的 Neural Solver 把流场往前推进 $\Delta t$，两者在推理时自回归交替，从而让神经求解器能在含激波的超声速流场上像经典求解器一样"该细的地方细、该粗的地方粗"。

**[Accelerating Diffusion Large Language Models with SlowFast Sampling: The Three Golden Principles](accelerating_diffusion_large_language_models_with_slowfast_sampling_the_three_go.md)**

:   针对扩散语言模型（dLLM）现有采样策略"速度恒定、不会随生成状态调整"的问题，本文先总结出三条经验规律（确定性、收敛性、位置性），据此设计了在"慢相探索"与"快相加速"之间动态切换的 SlowFast Sampling，并可与 dLLM-Cache 正交叠加——在 GPQA 上对 LLaDA 实现最高 15.63× 加速、叠加缓存后达 34.22×，精度几乎无损。

**[Attention Is All You Need for KV Cache in Diffusion LLMs](attention_is_all_you_need_for_kv_cache_in_diffusion_llms.md)**

:   针对扩散语言模型（DLM）每步都重算全部 token、全部层 KV 的冗余问题，本文提出训练无关、架构无关的 Elastic-Cache：用「最受关注 token 的注意力漂移」判断**何时**刷新缓存、用「深层先变」的规律决定**从哪层往上**刷新，并对滑动窗口外的远端 MASK token 做块级缓存，在 LLaDA / Dream-7B 等模型上实现最高 45.1× 解码加速且几乎不掉点。

**[Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors](autoencoding-free_context_compression_for_llms_via_contextual_semantic_anchors.md)**

:   SAC 不再像 ICAE 那样追加随机初始化的"压缩 token"并靠自编码预训练去重建上下文，而是直接从原文里挑出若干"锚点 token"、给它们加一个可学习的锚点嵌入、再用双向注意力让锚点聚合全局信息，把上下文压进锚点的 KV 里——彻底丢掉自编码任务后，在问答和长文摘要上反而稳定超过现有压缩方法。

**[AutoSP: Unlocking Long-Context LLM Training Via Compiler-Based Sequence Parallelism](autosp_unlocking_long-context_llm_training_via_compiler-based_sequence_paralleli.md)**

:   AutoSP 把序列并行（SP）从手写、与框架强耦合的算子，抬升成 PyTorch-2.0 编译栈里的两个编译 pass——在 Torch-IR 上自动插通信、resize 激活 buffer 的 SP-Pass，以及在 Aten-IR 联合图上松开 min-cut 约束、重算计算密集算子的序列感知激活检查点 SAC-Pass——让用户几行代码就能把单卡模型编译成分布式长上下文训练管线，在 NVIDIA / AMD 上把可训练序列长度拉长最高 2.7× / 2.5×，而吞吐几乎无损。

**[BA-LoRA: Bias-Alleviating Low-Rank Adaptation to Mitigate Catastrophic Inheritance in Large Language Models](ba-lora_bias-alleviating_low-rank_adaptation_to_mitigate_catastrophic_inheritanc.md)**

:   BA-LoRA 在 PiSSA 谱初始化的 LoRA 框架上叠加「一致性 + 多样性 + SVD」三个**输出空间正则**，分别对治微调放大预训练偏置时的知识漂移、表征坍缩与噪声过拟合，在 NLG/NLU 多任务上稳定超过一众 LoRA 变体，且在噪声更重的预训练模型上增益更大。

**[Beyond Fixed: Training-Free Variable-Length Denoising for Diffusion Large Language Models](beyond_fixed_training-free_variable-length_denoising_for_diffusion_large_languag.md)**

:   DAEDAL 利用扩散大语言模型（DLLM）在去噪时对 EOS token 的预测置信度这一内部信号，免训练地在去噪前先把序列长度从一个短的统一初值粗调到任务合适的长度、再在去噪过程中对低置信度区域局部插入 mask 扩容，从而摆脱"必须手工预设生成长度"的桎梏，在四个数学/代码基准上达到甚至超过精调定长基线的精度，同时大幅提升有效 token 占比。

**[Beyond Masks: Efficient, Flexible Diffusion Language Models via Deletion-Insertion Processes](beyond_masks_efficient_flexible_diffusion_language_models_via_deletion-insertion.md)**

:   DID 把扩散语言模型的「掩码-去掩码」彻底换成「删除-插入」两条连续时间马尔可夫链：前向把 token 逐个删到空序列、后向从空序列逐个插回去，再配一套基于「插入分数」的 DISE 训练目标和并行动态规划，既扔掉了占一半算力的 `<MASK>`/`<PAD>` token，又天然支持变长和生成中自纠错，定长/变长两种设定下训练加速最高 3.42×、推理加速最高 3.79×。

**[Beyond Real: Imaginary Extension of Rotary Position Embeddings for Long-Context LLMs](beyond_real_imaginary_extension_of_rotary_position_embeddings_for_long-context_l.md)**

:   RoPE++ 重新拿回标准 RoPE 复数注意力中被丢弃的负虚部，把它作为与真实部并行的 imaginary attention head，在不增加 KV cache 或直接减半 cache 的配置下提升长上下文建模能力。

**[BoRA: Towards More Expressive Low-Rank Adaptation with Block Diversity](bora_towards_more_expressive_low-rank_adaptation_with_block_diversity.md)**

:   BoRA 把 LoRA 的 $BA$ 看成块矩阵乘法，给每个块积 $B_iA_j$ 插入一个独立的对角矩阵 $\Sigma_{i,j}$ 来打破块之间的相关性，只用 $b^2r$ 个额外参数就把 LoRA 权重的秩提升到原来的 $b$ 倍，在 GLUE、数学和常识推理上以与 LoRA 相近的参数量取得 2-4% 的准确率提升。

**[Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs](cache_what_lasts_token_retention_for_memory-bounded_kv_cache_in_llms.md)**

:   TRIM-KV 给预训练 LLM 的每个注意力头插入一个轻量"保留门"，在 token 生成时就预测它的内在长期重要性（一个会随时间指数衰减的标量分数），内存超预算时直接驱逐分数最低的 token；只需冻结主干、用蒸馏 + 容量损失微调这些门，推理几乎零额外开销，却在数学推理、长程生成和长上下文记忆等多个 benchmark 上稳定超越启发式驱逐和可学习检索基线，低显存场景下甚至反超全量缓存。

**[Cactus: Accelerating Auto-Regressive Decoding with Constrained Acceptance Speculative Sampling](cactus_accelerating_auto-regressive_decoding_with_constrained_acceptance_specula.md)**

:   本文把投机采样重新刻画成"在受控散度约束下最大化接受率"的约束优化问题，并由此提出 Cactus——只需读取候选 token 一个概率值、给它加一个由 $q$ 和散度预算 $\delta$ 决定的"奖励"，在保证与验证器分布偏差可控的前提下显著提高接受率与吞吐。

**[Cartridges: Lightweight and General-Purpose Long Context Representations via Self-Study](cartridges_lightweight_and_general-purpose_long_context_representations_via_self.md)**

:   把"长文档放进 KV cache 在线 prefill"换成"为每篇语料离线训练一个小型可学习 KV cache（Cartridge）"，再用 **Self-Study**（自生成合成对话 + 上下文蒸馏）让这个小 cache 复刻 ICL 的通用问答能力，平均省 38.6× 显存、提 26.4× 吞吐。

**[Cascadia: An Efficient Cascade Serving System for Large Language Models](cascadia_an_efficient_cascade_serving_system_for_large_language_models.md)**

:   Cascadia 是一个面向大模型的级联服务系统：它把"用多大模型/分多少 GPU/怎么并行"这件事建模成一个带约束的优化问题，用"MILP 解部署 + Chebyshev 解路由"的双层迭代框架联合求解，在保证答案质量的前提下，相比单模型部署把延迟 SLO 收紧最多 4×、吞吐提升最多 5×。

**[Command-V: Training-Free Representation Finetuning Transfer](command-v_training-free_representation_finetuning_transfer.md)**

:   Command-V（⌘V）把一个模型上训好的残差表示适配器（ReFT adapter），不经任何反向传播、也不需要原始训练数据，通过一对线性"转换器"直接搬到另一个架构不同的模型上，让接收方"免费"获得捐赠方的新行为（如拒答增强、越狱、自动思维链），效果接近直接微调而算力省几个数量级。

**[Composer: A Search Framework for Hybrid Neural Architecture Design](composer_a_search_framework_for_hybrid_neural_architecture_design.md)**

:   Composer 把"该怎么把 Attention、MLP 这些计算原语交错排列成更好的 LLM"这个一直靠人手工拍脑袋的问题，做成了一个自动化搜索框架：在百万参数级的小模型上用贝叶斯优化搜出好的交错模式，再外推约 1000× 放大到 3B/8B，搜出的 Composite 架构在 350M–8B 全程压过 Llama 3.2，下游准确率平均涨 2–2.1%，同时训练吞吐 ×1.25、KV cache 缩小 ×1.69。

**[CONCUR: A Framework for Continual Constrained and Unconstrained Routing](concur_a_framework_for_continual_constrained_and_unconstrained_routing.md)**

:   CONCUR 给每个「模型+解码方法」策略单独训练一对（准确率分类器 + 成本回归器）预测器，再把"该把任务交给谁"写成带/不带预算的优化问题来求解，从而在新策略不断涌现时只需加训新预测器、无需重训整套路由器，在分布内外、约束/无约束设置下都比最强单一策略和现有路由方法准确率更高、推理 FLOPs 更低。

**[CPQS-Tuning: A Model Self-Perception-Based Data Filtering Algorithm for Efficient Instruction Fine-Tuning](cpqs-tuning_a_model_self-perception-based_data_filtering_algorithm_for_efficient.md)**

:   本文提出 CPQS-Tuning：不再用外部评分模型或人工指标来筛指令微调数据，而是直接读取目标 LLM 自己的隐藏状态，用一个小 CNN 把"模型对这条数据的隐式评价"翻译成一个对比感知质量分（CPQS），据此挑出不到 10% 的高质量数据，训练效果反而超过用全量数据。

**[CR-Net: Scaling Parameter-Efficient Training with Cross-Layer Low-Rank Structure](cr-net_scaling_parameter-efficient_training_with_cross-layer_low-rank_structure.md)**

:   CR-Net 发现「相邻层激活之差」具有强低秩结构，于是把每层线性映射改写成「上一层激活 × 可学习缩放 + 低秩增量」，在不损失高秩信息的前提下把参数砍掉一半，再配一套专为这种跨层依赖设计的激活重计算策略，从 60M 一路 scale 到 13B 预训练，效果稳压现有低秩方法、显存与算力还更省。

**[DASH: Deterministic Attention Scheduling for High-throughput Reproducible LLM Training](dash_deterministic_attention_scheduling_for_high-throughput_reproducible_llm_tra.md)**

:   DASH 把确定性 attention 反向传播抽象成一个 DAG 调度问题，目标是最小化关键路径长度，再用「逆序 Q 块遍历」和「位移调度」两套互补策略消除流水线气泡，在 H800 上把确定性 attention 反向算子的吞吐相对 FlashAttention-3 确定性模式提升最高 1.28×，让可复现的 LLM 训练几乎不再为确定性付代价。

**[Deep Hierarchical Learning with Nested Subspace Networks for Large Language Models](deep_hierarchical_learning_with_nested_subspace_networks_for_large_language_mode.md)**

:   提出嵌套子空间网络（NSN），通过低秩分解使线性层形成严格嵌套的子空间层次，配合不确定性感知多秩训练，使单个模型在测试时可即时调节计算量与性能的权衡（50% FLOPs 减少仅损失 5% 精度），且可后验应用于预训练 LLM。

**[DefensiveKV: Taming the Fragility of KV Cache Eviction in LLM Inference](defensivekv_taming_the_fragility_of_kv_cache_eviction_in_llm_inference.md)**

:   针对 KV cache 驱逐普遍依赖的"重要性稳定假设"其实很脆弱、而主流的均值聚合在极端时刻会跟着崩这一问题，本文提出线性时间的"防御性聚合"（用历史观测的最大值估计最坏风险 + 自适应先验校正），据此构建 DefensiveKV 与跨层版 Layer-DefensiveKV，在 18 个数据集、20% cache 预算下把生成质量损失相比最强 baseline 缩小 2.3× 与 4.3×。

**[Demystifying and Enhancing the Efficiency of Large Language Model Based Search Agents](demystifying_and_enhancing_the_efficiency_of_large_language_model_based_search_a.md)**

:   本文先系统分析了 LLM 搜索智能体（边推理边检索）为什么慢——既不是检索越准越好（高召回开销大、低召回逼模型多检索几轮），又对检索延迟极度敏感（FCFS 调度和检索引发的停顿会反复把长请求的 KV-cache 挤掉重算）——再提出推理系统 SearchAgent-X，用高召回近似检索 + 优先级调度 + 非阻塞检索三招，把吞吐做到最高 3.4×、延迟降到 0.2–0.6×，且生成质量与精确检索持平。

**[Developmental Federated Tuning: A Cognitive-Inspired Paradigm for Efficient LLM Adaptation](developmental_federated_tuning_a_cognitive-inspired_paradigm_for_efficient_llm_a.md)**

:   DEVFT 把联邦微调拆成「先小后大」的发展阶段，从一个紧凑子模型逐步长成完整 LLM，并通过层分组与差分层融合实现跨阶段知识传递，在边缘设备上做到 4.59× 收敛加速、10.67× 通信节省、9.07% 平均性能提升。

**[DiffAdapt: Difficulty-Adaptive Reasoning for Token-Efficient LLM Inference](diffadapt_difficulty-adaptive_reasoning_for_token-efficient_llm_inference.md)**

:   作者先发现推理大模型在不同难度题目上呈现"U 形熵曲线"——简单题答得对却熵很高（过度思考），于是训练一个只读取模型隐状态的轻量探针，为每道题动态选择 Easy/Normal/Hard 三套推理策略，在不微调主模型的前提下把 token 消耗最多降 22.4%、端到端延迟降到原来的 1/6，同时准确率持平甚至更高。

**[Difficulty–Diversity Collaborative Filtering for Data-Efficient LLM Fine-Tuning](difficultydiversity_collaborative_filtering_for_data-efficient_llm_fine-tuning.md)**

:   把"模型-问题"答对与否的交互矩阵当成推荐系统的评分矩阵，用协同过滤学出**为每个目标模型个性化的题目难度**，再与语义多样性联合做组合优化选样，从大规模未标注语料里挑出 1000 条最有学习价值的题，把标注成本降低 100–200 倍而下游性能逼近全量微调。

**[Diffusion Language Models Know the Answer Before Decoding](diffusion_language_model_knows_the_answer_before_it_decodes.md)**

:   扩散语言模型（DLM）在解码到一半时往往就已经在内部确定了正确答案，本文据此提出训练免费的 Prophet 解码范式——用"前两名候选 token 的 logit 差距"判断答案是否收敛，一旦收敛就一步填完所有剩余位置（早提交解码），在 LLaDA-8B / Dream-7B 上把解码步数最多减少 3.4× 而精度几乎不掉。

**[Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing](diffusion_llms_can_do_faster-than-ar_inference_via_discrete_diffusion_forcing.md)**

:   本文提出离散扩散强制（D2F），把预训练扩散语言模型（dLLM）改造成「块级自回归 + 块间并行解码」的 AR-扩散混合范式，靠非对称蒸馏低成本获得该能力、再配流水线并行解码，首次让开源 dLLM 的推理吞吐反超同规模自回归 LLM（GSM8K 上比 LLaMA3 快 2.5×，比原始 dLLM 快 50×）。

**[DirMoE: Dirichlet-Routed Mixture of Experts](dirmoe_dirichlet-routed_mixture_of_experts.md)**

:   DirMoE 把 MoE 路由拆成"选哪些专家(Bernoulli/Gumbel-Sigmoid)"和"选中专家间如何分配权重(Dirichlet)"两个解耦决策，用一个 Dirichlet 变分自编码器框架做到全程端到端可微，并给出一个有理论保证的"稀疏旋钮" λ 来直接校准稀疏度，无需辅助负载均衡损失即可提升专家特化。

**[DiSRouter: Distributed Self-Routing for LLM Selections](disrouter_distributed_self-routing_for_llm_selections.md)**

:   DiSRouter 把传统的"中心化外部路由器"换成"让每个 LLM 自己判断要不要答"的分布式自路由范式：查询在一串按成本递增排列的 LLM agent 之间传递，每个 agent 凭自我认知决定是回答还是甩给下一个更强的模型，从而在性能与成本之间取得更优的效用（Utility）。

**[Distilling to Hybrid Attention Models via KL-Guided Layer Selection](distilling_to_hybrid_attention_models_via_kl-guided_layer_selection.md)**

:   把预训练的 softmax 注意力 Transformer 蒸馏成"少数 softmax 层 + 大量线性注意力层"的混合模型时，用"逐层临时换回 softmax、短暂蒸馏后看 KL 损失下降多少"来给每层打重要性分数，再贪心挑出最关键的 K 层保留为 softmax，从而在几乎不掉长上下文检索能力的前提下大幅提升推理效率。

**[DND: Boosting Large Language Models with Dynamic Nested Depth](dnd_boosting_large_language_models_with_dynamic_nested_depth.md)**

:   DND在Transformer层末端通过路由器选出关键token，将其回送同一层进行额外处理（嵌套深度），配合路由控制损失和阈值控制方案实现精确稳定的token选择，以极少的参数增加（<0.1M）在Qwen3-1.7B和Qwen3-30B-A3B上分别获得1.88%和0.87%的平均性能提升。

**[DPad: Efficient Diffusion Language Models with Suffix Dropout](dpad_efficient_diffusion_language_models_with_suffix_dropout.md)**

:   DPad 发现扩散语言模型（dLLM）在每一步都要为所有未来 suffix token 计算注意力、却只保留极少数，造成巨大冗余；它用"滑动窗口 + 距离衰减 dropout"在注意力计算**之前**就丢掉远处 suffix token，免训练、即插即用，在 LLaDA-1.5/GSM8K（1024 token，1-shot）上叠加并行解码与前缀缓存后达到最高 61.39× 加速且精度不降反升。

**[dParallel: Learnable Parallel Decoding for dLLMs](dparallel_learnable_parallel_decoding_for_dllms.md)**

:   通过"确定性强制蒸馏"把扩散语言模型(dLLM)原本"逐字串行收敛"的预测确定性改造成"并行同时收敛",让 LLaDA-8B 在 GSM8K 上把解码步数从 256 砍到 30(8.5× 加速)而精度不降。

**[DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving](dualmap_enabling_both_cache_affinity_and_load_balancing_for_distributed_llm_serv.md)**

:   DualMap 用两个独立哈希函数把每个请求映射到两个候选实例、再按系统状态择优，借「两选一（power of two choices）」原理把过去互相打架的「缓存亲和」与「负载均衡」**在一套调度里同时拿到**，在相同 TTFT SLO 下把有效请求容量最多提升 2.25×。

**[Dynamic-dLLM: Dynamic Cache-Budget and Adaptive Parallel Decoding for Training-Free Acceleration of Diffusion LLM](dynamic-dllm_dynamic_cache-budget_and_adaptive_parallel_decoding_for_training-fr.md)**

:   Dynamic-dLLM 是一个免训练的扩散 LLM 推理加速框架，针对 token 在不同层、不同解码步上的"动态性"差异，用动态缓存更新（DCU）按层自适应分配缓存更新预算、用自适应并行解码（APD）按 token 动态校准解码阈值，在 LLaDA/Dream 等模型上平均提速 3× 以上、最高 4.48×，几乎不掉精度。

**[Dynamic Speculative Agent Planning](dynamic_speculative_agent_planning.md)**

:   针对 LLM 智能体"边规划边推测执行"时固定推测步长 $k$ 要么省不了时间、要么烧掉大量冗余 token 的问题，本文提出 DSP：用一个轻量 DistilBERT 回归器在线（无需任何部署前准备）预测每一步该推测多远，并把预测建模成强化学习里的状态值函数用 TD 学习更新，在保持"无损加速"的同时把总成本降 30%、无效成本降最高 60%，还暴露一个旋钮让用户自由滑动延迟与成本的取舍。

**[DynamicInfer: Runtime-Aware Sparse Offloading for LLMs Inference on a Consumer-Grade GPU](dynamicinfer_runtime-aware_sparse_offloading_for_llms_inference_on_a_consumer-gr.md)**

:   DynamicInfer 面向显存不足的消费级 GPU，把 LLM FFN 神经元按运行时激活模式在 CPU/GPU 之间动态调度，并用跨层预测、分层神经元缓存和负载感知阈值让更多真正会被用到的神经元落到 GPU 上，最终在保持精度基本不变的前提下比 llama.cpp 和 PowerInfer 明显加速。

**[Efficient Resource-Constrained Training of Transformers via Subspace Optimization](efficient_resource-constrained_training_of_transformers_via_subspace_optimizatio.md)**

:   提出 WASI（Weight-Activation Subspace Iteration），基于"微调过程中参数子空间稳定"的假设，同时压缩 Transformer 的权重（SVD + Gram-Schmidt 子空间迭代）和激活（Tucker 分解），实现训练和推理都在低秩表示中完成，达到 62× 训练内存压缩和 Raspberry Pi 5 上 1.4× 加速，且精度损失可忽略。

**[EntropyLong: Effective Long-Context Training via Predictive Uncertainty](entropylong_effective_long-context_training_via_predictive_uncertainty.md)**

:   EntropyLong 用模型自身的预测熵定位"信息缺口"，检索远端上下文并**实测它能否降低该位置的熵**，只保留真正带来信息增益的依赖来拼接 128K 训练样本，从而构造出"被验证过的"长程依赖，在 RULER 和 LongBench-v2 上显著超越启发式数据构造方法。

**[Equilibrium Language Models](equilibrium_language_models.md)**

:   把 Transformer 中一段连续的中间层替换成一个轻量"定点（fixed-point）模块"，用求解均衡态来等价表达深层堆叠，从而在剪掉 28% 参数的同时保留 99% 的精度，专为边缘端低内存部署设计。

**[Expert Divergence Learning for MoE-based Language Models](expert_divergence_learning_for_moe-based_language_models.md)**

:   解决 MoE 训练中的专家同质化问题，通过最大化不同数据域之间路由分布的 Jensen-Shannon 散度，鼓励不同域激活不同专家子集，在 15B-A1.5B 模型上提升专家特化程度和语言建模性能。

**[Expert Merging in Sparse Mixture of Experts with Nash Bargaining](expert_merging_in_sparse_mixture_of_experts_with_nash_bargaining.md)**

:   把稀疏 MoE 的"专家合并"重新解释为专家之间的合作—竞争博弈，用纳什议价解（Nash Bargaining Solution）从第一性原理推出每个专家的合并系数，并配上复数动量加速跨层传播，做出了 NAMEx 这套统一替换 CAMEx 启发式加权的合并框架。

**[Explainable Token-level Noise Filtering for LLM Fine-tuning Datasets](explainable_token-level_noise_filtering_for_llm_fine-tuning_datasets.md)**

:   针对"微调数据是句子级、但 LLM 优化是 token 级"的错配，本文提出 XTF：把每个 token 对微调的贡献拆解为「推理重要性 / 知识新颖性 / 任务相关性」三个可解释属性分别打分，凡完全缺失任一属性的 token 即判为噪声，并在训练时对其梯度做掩码，在数学/代码/医学三类任务、7 个主流 LLM 上把微调精度最高提升 13.7%。

**[Extending the Context of Pretrained LLMs by Dropping Their Positional Embedding](extending_the_context_of_pretrained_llms_by_dropping_their_positional_embedding.md)**

:   RoPE 在预训练时是加速收敛的关键归纳偏置，却也是阻碍长度外推的根源；本文提出 **DroPE**——预训练完成后直接删掉所有位置编码、再用极少 token 短暂"重校准"，即可让 LLM 零样本泛化到远超训练长度的序列，无需任何长上下文微调。

**[Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](fast-dllm_training-free_acceleration_of_diffusion_llm_by_enabling_kv_cache_and_p.md)**

:   Fast-dLLM 无需重新训练，给双向扩散语言模型补上一套块级近似 KV Cache，并用「置信度阈值」替代固定 top-K 的并行解码策略，在 LLaDA / Dream 上实现最高 27.6× 的端到端吞吐提升，且精度几乎不掉。

**[Fast-dLLM v2: Efficient Block-Diffusion LLM](fast-dllm_v2_efficient_block-diffusion_llm.md)**

:   Fast-dLLM v2 用约 1B token 的轻量微调把预训练的自回归 Qwen2.5 模型改造成块扩散语言模型，配合层次化缓存与置信度并行解码，在不掉点的前提下相比 AR 解码取得最高 2.5× 的加速。

**[Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws](fast_catch-up_late_switching_optimal_batch_size_scheduling_via_functional_scalin.md)**

:   通过 Functional Scaling Law 框架理论推导出 batch size scheduling 的最优策略——对困难任务，最优策略是训练大部分时间用小 batch，仅在最后阶段切换到大 batch（late switching）；并揭示了 fast catch-up 效应——切换后 loss 迅速追上全程大 batch 的轨迹，在 1.1B 参数 1T token 的 LLM 预训练中验证了该原则。

**[FlashDLM: Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion](flashdlm_accelerating_diffusion_language_model_inference_via_efficient_kv_cachin.md)**

:   两个免训练技巧——复用稳定 KV 投影的 FreeCache，加上用小 AR 模型一致性信号指导并行解掩码的 Guided Diffusion——让 7B/8B 扩散语言模型推理端到端平均提速 12 倍，首次把扩散 LLM 的延迟拉到与同尺寸自回归模型相当甚至更快。

**[Flatter Tokens are More Valuable for Speculative Draft Model Training](flatter_tokens_are_more_valuable_for_speculative_draft_model_training.md)**

:   本文从数据中心视角发现：训练投机解码 draft model 时，目标模型预测分布更"平坦"（接近均匀）的 token 价值更高，据此提出仅依赖目标模型、可离线计算的 flatness 指标与 SFDD 数据蒸馏方法，用 50% 数据换来 2× 以上训练加速且推理加速损失不到 4%。

**[FlexLinearAttention: Compiling a Unified Abstraction into Scalable Kernels for Linear Attention](flexlinearattention_compiling_a_unified_abstraction_into_scalable_kernels_for_li.md)**

:   FlexLA 把五花八门的线性注意力变体统一抽象成「intra-chunk 计算 / inter-chunk 状态传播 / 输出合并」三个阶段，让用户用几十行 PyTorch 就能描述算法，再由领域专用编译器自动生成融合了计算与通信的高性能 Triton 内核，单卡上达到甚至超越专家手写库 FLA（1.01×–4.9×），分布式上对 LASP2 最高 7.2× 加速并近线性扩展到 128 卡、1600 万 token。

**[FLoRG: Federated Fine-tuning with Low-rank Gram Matrices and Procrustes Alignment](florg_federated_fine-tuning_with_low-rank_gram_matrices_and_procrustes_alignment.md)**

:   FLoRG 把 LoRA 的两个低秩矩阵重参数化为单个低秩矩阵并只聚合其 Gram 矩阵，让服务器端聚合从「有偏的双线性运算」变成「无偏的线性运算」，再用 Procrustes 对齐解决分解非唯一带来的漂移，从而在联邦微调中同时消除聚合误差、压低通信开销（最高 2041×）并收紧收敛界。

**[Frayed RoPE and Long Inputs: A Geometric Perspective](frayed_rope_and_long_inputs_a_geometric_perspective.md)**

:   本文用一套统一的几何视角解释了「RoPE 模型为什么一超过训练长度就崩」——长输入把分得很开的 key/query 簇打散重叠，使得 sink token（注意力汇聚点）失效；据此提出 RoPE-ID：只对一半通道施加高频旋转，让免训练即可外推到更长上下文，在 RULER / LongBench 上追平甚至超过 YaRN。

**[FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference](freekv_boosting_kv_cache_retrieval_for_efficient_llm_inference.md)**

:   FreeKV 是一个 **training-free 的算法-系统协同优化框架**，通过「推测式检索」把 KV 页的选择与召回挪出推理关键路径、用「细粒度纠错」补偿精度损失，再配合 CPU/GPU 混合内存布局与双缓冲流式召回，让检索式 KV cache 压缩在几乎无损精度的前提下相比 SOTA 检索方法最高提速 **13×**。

**[From Collapse to Control: Understanding and Extending Context Length in Emerging Hybrid Models via Universal Position Interpolation](from_collapse_to_control_understanding_and_extending_context_length_in_emerging_.md)**

:   本文系统解释了混合 Mamba-Transformer 模型在超出训练窗口后为何会长上下文崩溃，并提出训练免费 Universal Position Interpolation，通过同时缩放 Transformer 的 RoPE 频率和少数不稳定 Mamba head 的步长 $\Delta_t$，把 Bamba、Nemotron-H 和 Mamba2 的可用上下文从 4K/8K 推到最高 64K。

**[FSA: An Alternative Efficient Implementation of Native Sparse Attention Kernel](fsa_an_alternative_efficient_implementation_of_native_sparse_attention_kernel.md)**

:   FSA 把 NSA 稀疏注意力 kernel 的"外层循环 query token、内层循环 KV block"翻转成"外层循环 KV block、内层循环 query token"，从而在每个 GQA group 只有少量 query head 的主流 LLM 上消除 padding 浪费，kernel 延迟最高降 3.5×、端到端训练最高加速 1.25×。

**[FutureFill: Fast Generation from Convolutional Sequence Models](futurefill_fast_generation_from_convolutional_sequence_models.md)**

:   针对卷积/谱序列模型（STU、Hyena 等）自回归解码慢的问题，本文提出 **FutureFill** 原语——用 FFT 提前算好"已生成 token 对未来 token 的贡献"，把从头生成 $L$ 个 token 的时间从 $O(L^2)$ 降到拟线性 $O(L\log^2 L)$，带提示生成时缓存从 $O(L+K)$ 降到只与生成长度有关的 $O(K)$，且全程**精确无近似**。

**[Global Resolution: Optimal Multi-Draft Speculative Sampling via Convex Optimization](global_resolution_optimal_multi-draft_speculative_sampling_via_convex_optimizati.md)**

:   本文首次给出在多草稿投机采样中**精确求解最优传输（OT）验证准则**的高效算法：把指数规模的最优传输线性规划（OTLP）一步步化简为一个至多 $V$ 个变量的凸最小化问题，在 i.i.d. 草稿设定下实现 90% 接受率且每 token 开销低于 100 ms。

**[Group Representational Position Encoding (GRAPE)](group_representational_position_encoding.md)**

:   提出 GRAPE 框架，基于群作用（group actions）统一了 Transformer 中乘法型（RoPE）和加法型（ALiBi/FoX）两大位置编码家族，证明 RoPE 和 ALiBi 是其精确特例，并提出路径积分加法变体 GRAPE-AP 在下游任务上超越现有方法。

**[Guided Speculative Inference for Efficient Test-Time Alignment of LLMs](guided_speculative_inference_for_efficient_test-time_alignment_of_llms.md)**

:   GSI 用一个小草稿模型先采样推理步、再用「奖励 + 对数似然比」修正后的 tilted reward 做 soft best-of-n，并在分数过低时回退到大模型重采，在数学推理基准上既逼近甚至超过大模型 best-of-n 的精度，又把端到端延迟最多降低 28%，且是首个对最优 tilted 策略有分布保证的投机式测试时扩展方法。

**[Gumbel Distillation for Parallel Text Generation](gumbel_distillation_for_parallel_text_generation.md)**

:   用 Gumbel-Max 把自回归教师的"采样随机性"外化成一段确定性的 Gumbel 噪声"蓝图"，让并行学生模型只需学一个有监督的"噪声→文本"映射，从而把难以建模的联合分布问题降维成简单回归，显著缩小并行解码与自回归之间的质量差距。

**[Hierarchy Decoding: A Training-free Parallel Decoding Strategy for Diffusion Large Language Models](hierarchy_decoding_a_training-free_parallel_decoding_strategy_for_diffusion_larg.md)**

:   针对扩散大语言模型（dLLM）并行解码"一步多 token 就掉点"的痛点，本文提出训练无关的 **Hierarchy-dLLM**：用分治思想把连续掩码区递归切成稀疏的小子区并行解码，让未解码 token 保持稀疏分布以抑制分布漂移，在保持甚至略升精度的同时把解码速度最高提升 17×、比 Fast-dLLM 快约 1.5×。

**[Householder-Diagonalized Linear Attention (HDLA): Utilizing Rank-Enhanced Decay Mechanism for Efficient Sequence Modeling](householder-diagonalized_linear_attention_hdla_utilizing_enhanced_decay_mechanis.md)**

:   HDLA 用广义 Householder 矩阵对线性注意力的衰减矩阵做合同对角化，把结构从主流的「对角 + 秩 1」扩展到更具表达力的「对角 + 秩 2」，并配套一个支持任意秩的 chunk-wise 并行算法；在语言建模困惑度、MQAR/RULER 检索、MAD 合成任务上以更低的计算量全面超过同类线性注意力基线。

**[IceCache: Memory-Efficient KV-cache Management for Long-Sequence LLMs](icecache_memory-efficient_kv-cache_management_for_long-sequence_llms.md)**

:   IceCache 把"按语义相似度聚类 token"和 PagedAttention 的分页机制结合起来——用一棵可增量更新的多级 DCI 树把语义相近的 token 塞进同一个物理内存页，使得 query-aware 检索时相关 token 高度共页、命中率大幅提升，从而在仅用 25% KV-cache 预算时仍保住接近满 cache 的精度和更低的延迟。

**[In-Place Test-Time Training](in-place_test-time_training.md)**

:   本文把 Transformer 里 MLP 块的下投影矩阵 $W_{down}$ 当作可在推理时更新的「快权重」，配上一个对齐 Next-Token Prediction 的训练目标和分块更新机制，让现成的预训练 LLM 不改架构、不从头训就「即插即用」获得测试时训练（TTT）能力，在 128k 乃至 256k 长上下文上稳定超过原模型与 GLA / DeltaNet / LaCT 等竞品。

**[Inference-Cost-Aware Dynamic Tree Construction for Efficient Inference in Large Language Models](inference-cost-aware_dynamic_tree_construction_for_efficient_inference_in_large_.md)**

:   CAST 在 EAGLE-2/3 的动态草稿树上引入"推理成本"这一被忽视的系统变量（GPU 型号、batch size），把树的宽度、深度和验证 token 数建模成"接受收益 vs 推理代价"的效用最大化问题，从而在单样本场景小幅、在 batch 场景大幅超越现有 SOTA，最高可达 5.2× 加速。

**[InfLLM-V2: Dense-Sparse Switchable Attention for Seamless Short-to-Long Adaptation](infllm-v2_dense-sparse_switchable_attention_for_seamless_short-to-long_adaptatio.md)**

:   InfLLM-V2 用「零额外参数、复用稠密注意力权重」的可训练稀疏注意力，让模型按序列长度在稠密/稀疏模式间无缝切换，既贴合「短序列预训练→长序列微调」的主流范式，又通过硬件友好的块选择实现，比稠密注意力快 4× 而保留 98.1% / 99.7% 的长理解 / 长推理性能。

**[Influence-Preserving Proxies for Gradient-Based Data Selection in LLM Fine-Tuning](influence-preserving_proxies_for_gradient-based_data_selection_in_llm_finetuning.md)**

:   IPROX 不再用现成小模型当代理来做梯度影响力数据选择，而是从目标 LLM 直接"蒸出"一个保留影响力信息的低秩代理——先用影响力加权的 SVD 压缩、再用梯度对齐微调，使得一个更小的代理在选数据时甚至胜过更大的现成代理。

**[IterResearch: Rethinking Long-Horizon Agents with Interaction Scaling](iterresearch_rethinking_long-horizon_agents_with_interaction_scaling.md)**

:   提出 IterResearch，一种基于MDP的迭代深度研究范式，通过周期性工作区重构替代单上下文线性累积，使Agent在40K上下文长度下扩展到2048次交互（性能从3.5%提升至42.5%），在6个benchmark上平均超出开源Agent 14.5个百分点。

**[KnowProxy: Adapting Large Language Models by Knowledge-guided Proxy](knowproxy_adapting_large_language_models_by_knowledge-guided_proxy.md)**

:   KnowProxy 用一个小代理模型「学会消化」冻结大模型生成的文本知识来适配下游任务，从而摆脱了传统代理微调对大模型概率分布的依赖，让黑盒 LLM 也能被高效微调，并用动态路由只在大模型不确定时才调用代理。

**[Learning To Draft: Adaptive Speculative Decoding with Reinforcement Learning](learning_to_draft_adaptive_speculative_decoding_with_reinforcement_learning.md)**

:   LTD 把树状 speculative decoding 中“draft 到多深”和“验证多少候选 token”建模成两个协同的强化学习策略，直接用每轮 draft-and-verify 的吞吐量作为 reward，在 Eagle3 上稳定提升 LLM 推理速度。

**[Learning to Parallel: Accelerating Diffusion Large Language Models via Learnable Parallel Decoding](learning_to_parallel_accelerating_diffusion_large_language_models_via_learnable_.md)**

:   针对扩散语言模型（dLLM）并行解码依赖固定启发式（如置信度阈值）、对不同输入不自适应的痛点，本文用一个极轻量（2 层 MLP、约 2 千参数、6 分钟训练）的可学习 filter 去逼近"一旦预测正确就立即定稿"的 oracle 策略，再配合 End-of-Text 早停，在 LLaDA-8B 上实现最高 22.58× 加速且几乎不掉点，叠加 KV-Cache 后达 57.51×。

**[Let's (not) just put things in Context: Test-time Training for Long-context LLMs](lets_not_just_put_things_in_context_test-time_training_for_long-context_llms.md)**

:   本文指出长上下文 LLM 的检索失败源于静态自注意力的 **score dilution**（干扰 token 稀释了对目标的注意力质量），并证明"思考 token"无法修复该问题；提出 **query-only 测试时训练（qTTT）**——只用一次 prefill 缓存 KV，然后在固定 KV 上只对 query 投影矩阵做几步梯度更新，在同等 FLOP 预算下显著超越思考 token，把推理时算力从"生成更多 token"重新分配到"少量针对性 query 更新"。

**[Libra: Effective yet Efficient Load Balancing for Large-scale MoE Inference](libra_effective_yet_efficient_load_balancing_for_large-scale_moe_inference.md)**

:   Libra 通过"推测执行预测下一层专家激活 + 两阶段局部性感知执行流"，把 MoE 推理中负载均衡所需的专家复制与 token 分片开销完全藏到 MoE 计算背后，在 8 卡 H200 上对 Qwen3MoE 与 GLM-4.5 实现近乎完美的负载均衡，prefill 吞吐最高提升 19.2%。

**[Local Linear Attention: An Optimal Interpolation of Linear and Softmax Attention for Test-Time Regression](local_linear_attention_an_optimal_interpolation_of_linear_and_softmax_attention_.md)**

:   把注意力看成"测试时回归求解器"，作者用统计学里的**局部线性回归**升级 Softmax 注意力，得到既能像线性注意力那样随上下文增长不断逼近、又比 Softmax 更强的 Local Linear Attention（LLA），并配套设计了硬件高效的 FlashLLA 分块算法把朴素实现的二次内存压回线性。

**[Log-Linear Attention](log-linear_attention.md)**

:   把线性注意力里那个"固定大小的隐藏状态"换成一组**随序列长度对数增长**的多尺度隐藏状态，从而在保持矩阵乘友好的并行训练（O(T log T) 计算、O(log T) 解码显存）的同时，把线性注意力撑向 softmax 注意力的表达力。

**[Long-Context Attention Benchmark: From Kernel Efficiency to Distributed Context Parallelism](long-context_attention_benchmark_from_kernel_efficiency_to_distributed_context_p.md)**

:   本文提出 **LongCA-bench**——一个把 7 个稠密算子、5 个稀疏算子、5 种上下文并行机制统一到同一数据接口下的长上下文注意力基准，用最多 96 张 H100、768K 序列长度系统评测了「掩码模式 × 序列长度 × 分布式规模」三维空间下各方法的速度/显存权衡。

**[LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation](loopformer_elastic-depth_looped_transformers_for_latent_reasoning_via_shortcut_m.md)**

:   LoopFormer 把循环 Transformer 的每次循环显式地条件化在「归一化时刻 $t$ + 步长 $\Delta t$」上，再用 shortcut-consistency 训练把不同长度的循环轨迹对齐到同一终点，从而让一个模型在推理时**任意指定计算预算 $M$、无需重训**就能优雅伸缩深度，避免了 naive 早退导致的表征塌缩。

**[LoRA-S: An Efficient Low Rank Adaptation scheme via Sylvester equation](lora-s_an_efficient_low_rank_adaptation_scheme_via_sylvester_equation.md)**

:   本文用微分几何的"水平提升"理论把 LoRA 的两个低秩因子放到商流形上优化，导出一个能让**任意带预条件的优化器**自动获得"高效特征学习/变换不变性"的通用迭代框架，并用一个由 Sylvester 方程解出的衰减矩阵 $K$ 替换掉手调的 weight decay 超参，产出 AdamS 与 LRACS 两个即插即用的高效 LoRA 优化器。

**[LoRAGen: Structure-Aware Weight Space Learning for LoRA Generation](loragen_structure-aware_weight_space_learning_for_lora_generation.md)**

:   LoRAGen 从「LoRA 参数空间本身的结构特性」出发，用作用在**完整适配矩阵 $\Delta W$ 上的权重空间损失** + **模块感知 MoE 解码器**，让隐扩散模型直接从自然语言任务描述生成 LoRA 参数，在分布内逼近任务专属 LoRA、在未见任务上零样本超越基线近 5 个点。

**[LouisKV: Efficient KV Cache Retrieval for Long Input-Output Sequences](louiskv_efficient_kv_cache_retrieval_for_long_input-output_sequences.md)**

:   LouisKV 发现关键 KV 在解码时具有强时序局部性、且在输入/输出序列上分布模式截然不同，据此把"每 token 检索 + 页粒度管理"换成"语义边界触发检索 + 输入聚类/输出分段的解耦细粒度管理"，在各类长序列任务上相对 SOTA 检索方法最高提速 4.7× 且近乎无损精度。

**[LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding](lycheedecode_accelerating_long-context_llm_inference_via_hybrid-head_sparse_deco.md)**

:   提出 LycheeDecode，通过将注意力头细粒度分为少量 retrieval heads（负责全注意力选关键 token）和大量 sparse heads（复用选出的 token 做稀疏计算），并用 HardKuma 分布端到端学习头类型，在 128K 上下文下实现 2.7× 加速且性能不降。

**[MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent](memagent_reshaping_long-context_llm_with_multi-conv_rl-based_memory_agent.md)**

:   MemAgent 把"无界长文档"切成定长 chunk 流式处理，用一段固定长度、可被覆写的 token 记忆替代不断膨胀的上下文，再用扩展版的 Multi-Conv DAPO 端到端训练记忆读写策略，使 8K 窗口模型几乎无损外推到 3.5M token 的 QA 任务，且推理复杂度严格线性。

**[Merge before Forget: A Single LoRA Continual Learning via Continual Merging](merge_before_forget_a_single_lora_continual_learning_via_continual_merging.md)**

:   把"持续学习"重新表述成"序贯模型合并"问题，全程只维护**一对** LoRA 矩阵 `{A, B}`：用前一任务的正交基初始化新任务的 `A`，并基于 LoRA 的不对称性只对 `B` 做 time-aware 缩放合并，从而把内存复杂度从随任务线性增长降到常数，同时缓解遗忘与僵化。

**[MesaNet: Sequence Modeling by Locally Optimal Test-Time Training](mesanet_sequence_modeling_by_locally_optimal_test-time_training.md)**

:   MesaNet 把"测试时训练"做到最优：它不像 DeltaNet 那样每步只走一步梯度，而是在每个时间步把"用上下文拟合一个线性模型"的累积正则平方误差**解到最优**，并通过共轭梯度求解器 + 分块并行，让原本只能串行、数值不稳的 Mesa 层第一次能在 GPU/TPU 上规模化训练到十亿参数。

**[MeSH: Memory-as-State-Highways for Recursive Transformers](mesh_memory-as-state-highways_for_recursive_transformers.md)**

:   本文诊断出递归 Transformer 落后于同算力非递归模型的两大病根——"无差别计算"和"信息过载"，提出 MeSH 方案：用一组显式记忆槽 + 逐步可学习读写路由器替换被过载的单一隐状态，让 1.4B 递归模型用少 33% 参数反超同规模 Vanilla Transformer。

**[Meta-UCF: Unified Task-Conditioned LoRA Generation for Continual Learning in Large Language Models](meta-ucf_unified_task-conditioned_lora_generation_for_continual_learning_in_larg.md)**

:   用一个共享超网络把每个任务的轻量嵌入即时翻译成全层 LoRA 更新，并用元对比 + 正交目标把任务嵌入推向近正交，从而在内存恒定（只占单个 adapter 的参数量）的前提下持续学习不遗忘。

**[MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head](mhla_restoring_expressivity_of_linear_attention_via_token-level_multi-head.md)**

:   本文指出线性注意力性能退化的根因是「全局上下文坍缩」（所有 query 共用一个固定 $d\times d$ 的全局 KV 摘要，导致注意力矩阵秩被死死卡在 $d$），提出沿 token 维度分块的多头线性注意力 MHLA，用一个可学习系数矩阵让每个 query block 对各块局部摘要做 query-conditioned 混合，从而在保持 $O(N)$ 复杂度、不引入额外卷积/门控模块的前提下把秩上界提升到 $\sum_b \min(n_b,d)$，恢复了 softmax 注意力的表达力。

**[MiSS: Revisiting the Trade-off in LoRA with an Efficient Shard-Sharing Structure](miss_revisiting_the_trade-off_in_lora_with_an_efficient_shard-sharing_structure.md)**

:   MiSS 把 LoRA 的双矩阵 $BA$ 更新换成由单个零初始化小矩阵 $D$ "扩展"出来的分片共享结构，既加快收敛又在显存和算力上同时占优，从而在性能–显存–效率三角中取得更好的折中。

**[Mitigating Non-IID Drift in Zeroth-Order Federated LLM Fine-Tuning with Transferable Sparsity](mitigating_non-iid_drift_in_zeroth-order_federated_llm_fine-tuning_with_transfer.md)**

:   提出 MEERKAT——只更新 0.1% 预训练敏感参数的稀疏零阶联邦微调方法，用「极致稀疏 + 高频同步」压制 Non-IID 漂移；并基于可追溯的虚拟路径发现 GradIP 现象，进一步用 MEERKAT-VP 识别极端 Non-IID 客户端并早停，提升全局模型质量。

**[Mixture-of-Experts Can Surpass Dense LLMs Under Strictly Equal Resource](mixture-of-experts_can_surpass_dense_llms_under_strictly_equal_resource.md)**

:   在**总参数量 N、训练算力 C、数据量 D 三者严格相等**的前提下，作者通过优化 MoE 骨干并把激活率控制在约 20% 的最优区间，首次证明 MoE 能稳定超越同等资源的 dense 模型，并用数据复用策略消解 MoE 额外的数据需求。

**[MoL: Adaptive Mixture-of-Length Reasoning for Efficient Question Answering with Context](mol_adaptive_mixture-of-length_reasoning_for_efficient_question_answering_with_c.md)**

:   MoL 用基于跨文档信息冗余的难度评估给每个问题打"难度分"，再配一个"答错就奖励变长、答对就奖励变短"的双目标奖励做 GRPO 训练，让模型自然涌现出"智能简洁"——简单题短答、难题长答，在多个带上下文 QA 任务上同时提升准确率并大幅压缩 token。

**[MoM: Linear Sequence Modeling with Mixture-of-Memories](mom_linear_sequence_modeling_with_mixture-of-memories.md)**

:   MoM 用一组相互独立的记忆状态 + 路由网络替换线性模型里那个唯一的固定大小记忆，让不同 token 只更新各自被分配的记忆，从而在保持线性复杂度的同时大幅扩容记忆、消除写入干扰，把召回密集任务做到逼近 Transformer。

**[Neuron-Aware Data Selection in Instruction Tuning for Large Language Models](neuron-aware_data_selection_in_instruction_tuning_for_large_language_models.md)**

:   NAIT 提出用"神经元激活模式"来挑选指令微调数据：先用少量 in-domain 样本提取出某项能力对应的神经元激活方向向量，再按候选样本激活与该方向的对齐分数排序选 top-k，在 LLaMA-2-7b 上只用 10% 的 Alpaca-GPT4 数据就比全量微调平均提升 3.24%，而且不依赖外部大模型、成本只有 AlpaGasus 的 1/19。

**[NI Sampling: Accelerating Discrete Diffusion Sampling by Token Order Optimization](ni_sampling_accelerating_discrete_diffusion_sampling_by_token_order_optimization.md)**

:   把离散扩散语言模型（dLLM）"每步只敢解锁少量 token"的保守采样改造成"每步把所有已能正确预测的 token 一次性解锁"，并用一个轻量神经网络（神经指示器）替代固定置信度阈值来做这个判断，在 LLaDA / Dream 上相比全步采样最高获得 14.3×（叠加 KV 缓存 25.0×）加速且几乎不掉点。

**[Not-a-Bandit: Provably No-Regret Drafter Selection in Speculative Decoding for LLMs](not-a-bandit_provably_no-regret_drafter_selection_in_speculative_decoding_for_ll.md)**

:   针对"多个领域专家草稿模型如何为每条 query 动态选最优"的问题，本文指出投机解码里**探索是多余的**——一条被 target 验证过的轨迹就能反事实地评估所有草稿模型，于是把原本的 multi-armed bandit 问题变成全信息在线学习问题，提出 HedgeSpec，在 N 个草稿模型上做到无悔（no-regret），相比 EAGLE3 最高提速 83.7%、相比 bandit 基线最高提升 49% MAT。

**[Not All Bits Are Equal: Scale-Dependent Memory Optimization Strategies for Reasoning Models](not_all_bits_are_equal_scale-dependent_memory_optimization_strategies_for_reason.md)**

:   通过 1700+ 组实验系统证明：非推理模型上"4-bit 量化是显存最优"的结论在推理模型上失效——显存最优策略由模型的**有效尺寸**（参数量×位宽）决定，存在"8-bit 4B"这一临界点，小模型应把显存花在更大权重上、大模型应花在更长生成/更多采样上。

**[Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models](not_all_models_suit_expert_offloading_on_local_routing_consistency_of_mixture-of.md)**

:   本文提出 SRP 和 SCH 两个指标来量化 MoE 模型的「局部路由一致性」（连续 token 是否倾向激活同一批专家），在 20 个真实 MoE LLM + 一系列受控 toy 模型上系统分析，揭示了局部负载均衡 vs 路由一致性的权衡、共享专家会损害一致性、域专精专家最有利于一致性等规律，并给出「缓存大小取激活专家数 2 倍最划算」的部署结论。

**[On-the-Fly Adaptation to Quantization: Configuration-Aware LoRA for Efficient Fine-Tuning of Quantized LLMs](on-the-fly_adaptation_to_quantization_configuration-aware_lora_for_efficient_fin.md)**

:   CoA-LoRA 训练一个"配置感知模型"，把任意逐层量化配置直接映射成轻量的低秩调整量，让单个 LoRA 适配器无需逐配置重新微调就能适配各种比特宽度组合；再配合一个基于帕累托的高斯过程配置搜索来挑选高质量的训练配置集，最终在四个 GLUE 任务上相对 SOTA 取得 1.74%–8.89% 的准确率提升，而总微调时间几乎不随配置数增长。

**[One-Prompt Strikes Back: Sparse Mixture of Experts for Prompt-based Continual Learning](one-prompt_strikes_back_sparse_mixture_of_experts_for_prompt-based_continual_lea.md)**

:   提出 SMoPE 框架，将单个共享 prompt 组织为稀疏 MoE 结构中的多个 prompt expert，通过 prompt-attention score aggregation 实现动态稀疏激活，在保持高参数效率的同时显著缓解知识干扰，在多个持续学习 benchmark 上达到 SOTA。

**[OPPO: Accelerating PPO-based RLHF via Pipeline Overlap](oppo_accelerating_ppo-based_rlhf_via_pipeline_overlap.md)**

:   OPPO 是一个轻量、模型无关的 PPO-RLHF 训练加速框架：它在「单步内」把 actor 生成与 reward 打分用分块流式重叠起来，在「跨步间」用超额提交（overcommit）几个 prompt 把长尾响应推迟到后续 step 完成，从而在不改变 PPO 更新、不损失收敛质量的前提下把训练加速 1.8×–2.8×、GPU 利用率提升 1.4×–2.1×。

**[Out of the Memory Barrier: A Highly Memory-Efficient Training System for LLMs with Million-Token Contexts](out_of_the_memory_barrier_a_highly_memory-efficient_training_system_for_llms_wit.md)**

:   OOMB 把百万级上下文 LLM 训练改造成按 chunk 串行推进、激活即时丢弃并反向重算的系统，再用分页 KV cache、异步 CPU offload 和页级稀疏注意力管理真正随长度增长的状态，使 Qwen2.5-7B 能在单张 H200 上训练 4M token 上下文。

**[Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding](overcoming_joint_intractability_with_lossless_hierarchical_speculative_decoding.md)**

:   本文提出 Hierarchical Speculative Decoding（HSD），用"分层分支重采样 + 封顶"的新验证策略，在不改变目标模型分布（provably lossless）的前提下显著提高每步接受的草稿 token 数，平均解码速度提升 6.7%，接进 EAGLE-3 后再涨 12% 以上。

**[Parallel Sampling from Masked Diffusion Models via Conditional Independence Testing](parallel_sampling_from_masked_diffusion_models_via_conditional_independence_test.md)**

:   PUNT 是一个训练无关、模型无关的掩码扩散模型（MDM）采样器：它在每一步用「上下文独立性」检验+分治剪枝，只花 $O(\log |M|)$ 次前向就挑出一批互不干扰、且高置信度的 token 同时解码，在长文本对齐基准上以更少的前向次数取得更高质量（IFEval 上比基线高出多达 16%）。

**[ParaRNN: Unlocking Parallel Training of Nonlinear RNNs for Large Language Models](pararnn_unlocking_parallel_training_of_nonlinear_rnns_for_large_language_models.md)**

:   把整条非线性 RNN 的逐步递推改写成一个 $L$ 元非线性方程组，用牛顿迭代 + 块双对角并行归约一次性求解，从而第一次让经典非线性 RNN（GRU/LSTM）也能像 Transformer/Mamba 一样沿序列长度并行训练——最高比朴素顺序应用快 665×，并据此训出 7B 规模、困惑度可与同尺寸 Transformer/Mamba2 比肩的 RNN 语言模型。

**[PARD: Accelerating LLM Inference with Low-Cost Parallel Draft Model Adaptation](pard_accelerating_llm_inference_with_lowcost_parallel_draft_model_adaptation.md)**

:   PARD 把一个现成的小语言模型改造成「一次前向就并行吐 $K$ 个 token」的目标无关草稿模型，再用条件丢弃 token（COD）把这种改造的训练成本砍到 $O(N)$，在 vLLM 上让 LLaMA3.1-8B 达到 264.88 tokens/s，比自回归快 $3.67\times$、比 EAGLE-3 快 $1.15\times$。

**[Planned Diffusion](planned_diffusion.md)**

:   让同一个模型先用自回归方式生成一份"规划"把回答切成若干语义独立的块、再对所有块并行扩散去噪，从而由模型自己决定去噪顺序，在 AlpacaEval 上相对自回归取得 1.27×～1.81× 加速、胜率仅掉 0.87%～5.4%，刷新了离散扩散并行生成的质量-延迟帕累托前沿。

**[PLoP: Precise LoRA Placement for Efficient Finetuning of Large Models](plop_precise_lora_placement_for_efficient_finetuning_of_large_models.md)**

:   PLoP 用一个免梯度、几乎零额外开销的「特征范数对齐分数」（NFN）来自动判断在哪些模块类型上插 LoRA，规则是把适配器放到与任务对齐度最低的模块上，在 SFT 与 RL 两类后训练场景里都稳定优于（最差也持平）「只插注意力」「只插 MLP」等常用经验法则。

**[Predicting LLM Output Length via Entropy-Guided Representations](predicting_llm_output_length_via_entropy-guided_representations.md)**

:   本文不再训练独立的辅助模型来预测 LLM 输出长度，而是直接复用主模型自身的隐藏状态，用 token 熵加权池化（EGTP）做静态预测、用逐步细化（PLP）应对 RL 采样这类"一对多"的随机生成，并发布了首个含长序列/CoT/RL 数据的长度预测基准 ForeLen，在其上把最强基线的 MAE 平均再降 29.16%，端到端吞吐显著提升。

**[PrefixMemory-Tuning: Modernizing Prefix-Tuning by Decoupling the Prefix from Attention](prefixmemory-tuning_modernizing_prefix-tuning_by_decoupling_the_prefix_from_atte.md)**

:   本文先实证指出 Prefix-Tuning 在现代大模型上失效的真正原因是「前缀与输入在注意力 softmax 里此消彼长的权重 trade-off」，进而提出 PrefixMemory-Tuning（PMT）：把前缀模块从注意力头里搬出来、用一个可训练记忆矩阵 $M$ 加核特征映射 $\phi(\cdot)$ 近似，使前缀贡献不再被序列长度稀释，在少样本分类、偏好对齐、数学推理上一致超过 Prefix-Tuning 并与 LoRA 持平甚至领先。

**[Prima.cpp: Fast 30-70B LLM Inference on Heterogeneous and Low-Resource Home Clusters](primacpp_fast_30-70b_llm_inference_on_heterogeneous_and_low-resource_home_cluste.md)**

:   prima.cpp 把家里的笔记本、台式机、手机、平板拼成一个异构低端集群，用「流水环并行 + 预取」把磁盘加载延迟藏进计算里，再用 Halda 调度器按设备真实算力/内存/磁盘求解最优分层方案，让一台普通家庭集群在内存严重不足的情况下也能跑 30-70B 模型，70B 达到 674 ms/token、内存压力 <6%，相比 llama.cpp 把 TPOT 降低 5-17×。

**[ProtoKV：长上下文的知识在你查询之前就已组织好了](protokv_long-context_knowledges_are_already_well-organized_before_your_query.md)**

:   ProtoKV 发现 LLM 在 prefilling 阶段就已自发把 token 分成「位置决定型」和「语义锚点型」两类，并据此分别构造语义原型、按原型把 token 聚成簇再整簇保留/丢弃，从而在相同显存预算下把 LongBench 平均精度比 SOTA 提高 2.11%，同时计算开销与 SnapKV 相当。

**[QuoKA: Query-Oriented KV Selection for Efficient LLM Prefill](quoka_query-oriented_kv_selection_for_efficient_llm_prefill.md)**

:   QuoKA 提出一个免训练、不依赖定制 kernel 的稀疏注意力方法，在分块预填充（chunked prefill）阶段先用「离均值查询越远越重要」的几何观察挑出少量代表性查询，再用余弦相似度为这些查询选关键 KV，从而在长上下文任务上几乎不掉点的前提下，把注意力计算量降到亚二次、实现 3× 的首 token 延迟下降和最高约 7× 的注意力加速。

**[RACE Attention: A Strictly Linear-Time Attention Layer for Training on Outrageously Large Contexts](race_attention_a_strictly_linear-time_attention_layer_for_training_on_outrageous.md)**

:   本文用一个"锐化角度核 + 可微 LSH 草图（RACE）"替换 Softmax 注意力，把注意力做成在序列长度和嵌入维度上都**严格线性**的算子，从而在单层注意力上把单次前反向能处理的上下文从 FlashAttention 的约 400 万 token 推到 GPU 上 1200 万、CPU 上 7500 万 token，且在 64K 以内的真实任务上精度与强基线持平甚至更好。

**[Randomization Boosts KV Caching, Learning Balances Query Load: A Joint Perspective](randomization_boosts_kv_caching_learning_balances_query_load_a_joint_perspective.md)**

:   提出首个KV缓存感知负载均衡统一数学模型，设计随机化叶节点淘汰算法RLT(O(log n)竞争比)和基于学习的贪心路由LBGR，在多LLM服务场景下将延迟降低最高11.96×、TTFT降低14.06×。

**[Reasoning Language Model Inference Serving Unveiled: An Empirical Study](reasoning_language_model_inference_serving_unveiled_an_empirical_study.md)**

:   这是第一篇系统刻画"推理大模型（RLLM）线上推理服务"行为的实证研究：作者提出 ASU 评测框架与 ASU-Perf 基准套件，发现 RLLM 相比普通 LLM 在服务侧存在四个显著差异（显存剧烈波动、掉队请求、运行时随难度自适应、领域偏好），并逐一检验了量化、KV Cache 量化、前缀缓存、投机解码这些为传统 LLM 设计的优化技术对 RLLM 还灵不灵。

**[Reconstructing KV Caches with Cross-Layer Fusion for Enhanced Transformers](reconstructing_kv_caches_with_cross-layer_fusion_for_enhanced_transformers.md)**

:   针对跨层 KV cache 共享（如 YOCO、CLA）一直打不过层内方法（GQA）的问题，本文发现了"键值非对称"现象——顶层的 value 主要来自底层、key 主要来自底层和中层，据此提出 FusedKV（在 post-RoPE key 上做可学习逐通道融合）及其轻量版 FusedKV-Lite（直接非对称复用），在 332M–4B 模型上把 KV cache 显存砍掉 50%，同时困惑度还低于满缓存的标准 Transformer。

**[ReFusion: A Diffusion Large Language Model with Parallel Autoregressive Decoding](refusion_a_diffusion_large_language_model_with_parallel_autoregressive_decoding.md)**

:   ReFusion 把掩码扩散语言模型的并行解码从 token 级抬升到 slot（多 token 片段）级——slot 之间用扩散方式并行挑选、slot 内部用自回归串行填充，并在每步把已生成 slot 重排到掩码 slot 前面，从而同时拿到完整 KV 缓存复用和可控的学习复杂度，相比此前扩散模型平均提升 34% 性能、加速 18×，还在保持 2.33× 速度优势的同时逼近甚至超过强自回归模型。

**[RepSpec: Structural Re-parameterized Draft Model Training for Speculative Decoding](repspec_structural_re-parameterized_draft_model_training_for_speculative_decodin.md)**

:   RepSpec 借鉴 RepVGG 的结构重参数化思想，把推测解码草稿模型里的每个线性层在**训练时**拆成多分支冗余结构、**推理时**无损合并回单层，从而在不增加推理开销的前提下提升草稿模型能力；再叠加一个"LoRA 式非线性混合分支"进一步拉长接受序列，把 SOTA 的 EAGLE-3 加速 4%–10%。

**[RESA: Bringing Back What Sparse Attention Ignores with Residual Estimation](resa_bringing_back_what_sparse_attention_ignores_with_residual_estimation.md)**

:   针对稀疏注意力（SA）"只算选中的 KV、把其余 KV 当作零贡献"的盲区，RESA 利用注意力 logits 矩阵天然的低秩特性，用一个 rank-1 先验把被忽略 KV 的贡献估计回来，并以与 SA 同阶的开销在线融合，从而在相同 KV 预算下把模型质量最多提升 26%，或在同等质量下把 KV 预算压缩 33.2%、注意力吞吐提升 1.23×。

**[ReST-KV: Robust KV Cache Eviction with Layer-wise Output Reconstruction and Spatial-Temporal Smoothing](rest-kv_robust_kv_cache_eviction_with_layer-wise_output_reconstruction_and_spati.md)**

:   ReST-KV 把 KV cache 淘汰重新定义为「逐层输出重建」问题——以「删掉某个 KV 对会让本层注意力输出增加多少误差」作为重要性指标，从而显式捕捉删除后被忽略的注意力重分布效应，再叠加时间维 EMA 平滑和空间维自适应窗口平滑，使长上下文下的淘汰更鲁棒，在 LongBench 上比 SOTA 高 2.58%、RULER 上高 15.2%，128k 解码延迟降低 10.61×。

**[Retrospective Sparse Attention for Efficient Long-Context Generation](retrospective_sparse_attention_for_efficient_long-context_generation.md)**

:   本文提出 RetroAttention，一种"追溯式"稀疏注意力：在后续解码步骤加载到新 KV 时，回头去修正过去 Query 已经算好的注意力输出，从而在不增加 KV 预算的前提下让历史 Query 接触到更多 KV，缓解长生成中误差累积的问题，相比 SOTA 的 Quest 最多提升 21.9% 准确率、有效 KV 暴露量最多扩到 1.6×。

**[Revisiting Long-context Modeling from Context Denoising Perspective](revisiting_long-context_modeling_from_context_denoising_perspective.md)**

:   本文把长上下文建模看成一个"信号去噪"问题：用积分梯度（IG）分数精确定位上下文里真正影响预测的关键 token，再用一个轻量的去噪训练策略 CDT 在输入端压制无关 token 的影响，让 8B 开源模型在 LongBench-E 上做到 50.92 分、逼近 GPT-4o 的 51.00 分。

**[Revisiting Parameter Server in LLM Post-Training](revisiting_parameter_server_in_llm_post-training.md)**

:   针对 LLM 后训练中序列长度方差极大、设备负载严重不均的场景，本文把经典参数服务器（PS）思想重新引入现代分片数据并行：提出 On-Demand Communication（ODC），用点对点的 gather / scatter-accumulate 替换 FSDP 里逐层的 all-gather / reduce-scatter，把同步粒度从「每层一次」放松到「每个 minibatch 一次」，让快的设备不再被慢设备拖住，端到端最高比标准 FSDP 提速 36%。

**[RMAAT: Astrocyte-Inspired Memory Compression and Replay for Efficient Long-Context Transformers](rmaat_astrocyte-inspired_memory_compression_and_replay_for_efficient_long-contex.md)**

:   RMAAT 把生物学里"星形胶质细胞"调控记忆的两类机制搬进 Transformer：用短时可塑性（STP）启发的线性复杂度注意力替换 $O(N^2)$ 自注意力、用长时可塑性（LTP）饱和曲线导出的"记忆保留因子"对跨段记忆令牌做自适应压缩，再配一套只缓存记忆令牌、反传时重算前向的 AMRB 训练算法，在 Long Range Arena 上把平均准确率从 RMT 的 63.6% 提到 68.0%，峰值显存却只有递归基线的约 1/4。

**[Routing Manifold Alignment Improves Generalization of Mixture-of-Experts LLMs](routing_manifold_alignment_improves_generalization_of_mixture-of-experts_llms.md)**

:   本文提出 RoMA（Routing Manifold Alignment），通过在后训练目标里加一个"流形正则项"，只轻量微调 MoE LLM 最后几层路由器，让语义相似样本共享相似的专家选择，在三个 MoE 模型上把准确率提升 7–15%，且不增加推理开销。

**[Scaling Attention via Feature Sparsity](scaling_attention_via_feature_sparsity.md)**

:   本文换了一个被忽视的轴来给注意力提速——不再裁剪 token，而是把每个 query/key 的 $d$ 维向量做 Top-$k$ 特征稀疏化，让注意力分数只在 query 和 key 共同激活的少数坐标上精确计算，再配一个 IO-aware 的 FlashSFA kernel 避免物化 $n\times n$ 分数矩阵，使 $QK^\top$ 的算力从 $\Theta(n^2d)$ 降到 $\Theta(n^2k^2/d)$，在 GPT-2 / Qwen3 上做到匹配 dense 精度的同时提速最高 2.5×、FLOPs 与 KV-cache 各省近 50%。

**[Scaling Large Vision-Language Model RL Training via Efficient Load Balancing](scaling_large_vision-language_model_rl_training_via_efficient_load_balancing.md)**

:   针对 VLM 强化学习训练中"集中式多模态数据加载"和"跨 GPU 序列负载极度不均"两大系统瓶颈，本文提出端到端系统 FlexRL，用 ShadowLoader 把视觉数据的解码/预处理下放到 worker 并只在控制器上传递轻量元数据，用 FlexUlysses 把序列自适应切成细粒度 chunk 做子序列级负载均衡，在 128-GPU 集群上把端到端吞吐最高提升 8.47×。

**[Scaling Laws Meet Model Architecture: Toward Inference-Efficient LLMs](scaling_laws_meet_model_architecture_toward_inference-efficient_llms.md)**

:   本文把 Chinchilla 缩放定律扩展成"条件式"版本，显式把隐藏维度 $d_{model}$、MLP-注意力参数配比 $r_{mlp/attn}$、GQA 三个架构因素塞进 loss 预测，并配一套搜索框架，在固定参数/训练 token 预算下找到既准又快的架构；据此训出的 Panda / Surefire 系列模型相比 LLaMA-3.2 最高提升 2.1% 准确率、42% 推理吞吐。

**[Scaling Linear Attention Capacity with Sparse State Expansion](scaling_linear_attention_capacity_with_sparse_state_expansion.md)**

:   这篇论文把线性注意力的状态更新重新解释为“信息分类”，在此基础上提出 Sparse State Expansion（SSE）：用行稀疏写入和分区扩展显著增加固定状态容量，在不明显增加参数量的前提下提升长上下文检索与数学推理能力。

**[Scaling Up, Speeding Up: A Benchmark of Speculative Decoding for Efficient LLM Test-Time Scaling](scaling_up_speeding_up_a_benchmark_of_speculative_decoding_for_efficient_llm_tes.md)**

:   这篇论文构建了首个专门评测「推测解码（speculative decoding）加速 LLM 测试时扩展（test-time scaling）」的基准，在 BoN 与多轮思考两种范式下统一协议对比了 9 种推测解码方法，核心发现是：测试时扩展产生的推理轨迹高度重复，连最简单的 N-gram 类方法（尤其 SAM）都能逼近甚至超过需要训练的 EAGLE-3，而把二者杂交的混合方法能拿到全场最高加速。

**[Self-Speculative Decoding Accelerates Lossless Inference in Any-Order and Any-Subset Autoregressive Models](self-speculative_decoding_accelerates_lossless_inference_in_any-order_and_any-su.md)**

:   本文提出 Any-Subset Speculative Decoding（ASSD），让任意子集自回归模型（AS-ARM）用同一个网络既当快速草稿、又当联合密度裁判，通过拒绝采样在**保证从真实联合分布无损采样**的同时并行生成多 token，并从理论上证明神经网络调用次数永远不会超过生成 token 数。

**[Self-Speculative Masked Diffusions](self-speculative_masked_diffusions.md)**

:   Self-Speculative Masked Diffusions 把 masked diffusion 的非因果并行草稿分布和任意顺序因果目标分布合进同一个 Transformer，用自 speculative sampling 在一次主要前向中验证多个 masked token，从而在文本建模和蛋白序列生成上以接近相同质量减少约 $2\times$ 网络前向次数。

**[Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling](semantic_parallelism_redefining_efficient_moe_inference_via_model-data_co-schedu.md)**

:   提出语义并行(Semantic Parallelism)范式，通过预测token-expert路由路径并协同调度模型放置与数据分发，大幅削减MoE推理中专家并行的all-to-all通信开销，在Attention-DP场景下吞吐提升最高2.78×，Attention-TP场景下延迟降低最高24.9%。

**[Sequential Parallel Duality in Prefix Scannable Models](sequential_parallel_duality_in_prefix_scannable_models.md)**

:   这篇论文用并行前缀扫描统一刻画了“训练可并行、推理可流式”的高效序列模型，并把这一类模型推广到允许非结合聚合算子的 Prefix-Scannable Models，使 Transformer 风格的 softmax 聚合也能在固定 chunk 下获得近似线性训练和 $O(\log n)$ 记忆的流式推理。

**[Short Window Attention Enables Long-Term Memorization](short_window_attention_enables_long-term_memorization.md)**

:   本文用「滑动窗口注意力 + xLSTM 线性 RNN」交替的混合架构 SWAX 研究短/长程记忆的分工，发现一个反直觉结论——**滑动窗口越短，长上下文检索反而越好**（因为短窗口逼着线性 RNN 去学长程依赖），并据此提出随机窗口训练（每个 batch 随机用 128 或 2048 的窗口），让模型在短上下文和长上下文任务上同时拿到最优。

**[SinkTrack: Attention Sink based Context Anchoring for Large Language Models](sinktrack_attention_sink_based_context_anchoring_for_large_language_models.md)**

:   SinkTrack 把 decoder-only LLM 中天然稳定受关注的 `<BOS>` 注意力汇聚点改造成上下文信息锚，通过训练免费的双轨 cross-attention 在 prefill 阶段向 `<BOS>` 注入输入上下文，从而在几乎不增加解码开销的情况下缓解幻觉和长上下文遗忘。

**[Smooth Reading: Bridging the Gap of Recurrent LLM to Self-Attention LLM on Long-Context Understanding](smooth_reading_bridging_the_gap_of_recurrent_llm_to_self-attention_llm_on_long-c.md)**

:   针对循环 LLM（线性复杂度但固定内存）在长上下文任务上打不过自注意力 LLM 的问题，本文提出 Smooth Reading——把"一口气读完整段上下文"改成"分块多轮、边读边总结、隐状态跨轮累积"的端到端多轮推理（EMR），并配套指出该推理范式更偏爱长度外推强的滑窗架构，最终在 LongBench 上把循环模型从落后自注意力 5.68% 反超 3.61%，同时保持训练 2.5×、推理 2× 的效率优势。

**[SoLoPO: Unlocking Long-Context Capabilities in LLMs via Short-to-Long Preference Optimization](solopo_unlocking_long-context_capabilities_in_llms_via_short-to-long_preference_.md)**

:   SoLoPO 把长上下文偏好优化拆成“短上下文上的偏好学习”和“短长上下文奖励一致性”，用更短、更干净的数据激活 LLM 的长上下文定位与推理能力，同时显著降低长序列训练的时间和显存压力。

**[SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations](sonicmoe_accelerating_moe_with_io_and_tile-aware_optimizations.md)**

:   针对"细粒度 + 高稀疏"MoE 在硬件上变得越来越内存受限的问题，SonicMoE 用「重写反向计算图把激活缓存压到最小 + IO 与计算重叠的融合算子 + 把每专家 token 数对齐 tile 的 token rounding 路由」三招，在 Hopper 上把 7B 细粒度 MoE 的算子吞吐相对 ScatterMoE 提升 1.86×、激活内存降 45%，高稀疏下还额外拿到 1.16× 加速。

**[SpareTrain: Fault-Tolerant LLM Training via Low-Cost Dual Modular Redundancy](sparetrain_fault-tolerant_llm_training_via_low-cost_dual_modular_redundancy.md)**

:   SpareTrain 把"检测静默数据损坏最彻底但最贵"的双模冗余（DMR）做进 LLM 训练系统：它一方面复用激活检查点反向重算天然产生的冗余、一方面把校验计算塞进 GPU 因通信而空闲的时间窗里，在不削弱检测能力的前提下，把 DMR 的训练吞吐损失从近 100% 压到相对无保护训练只有 3–14% 的开销。

**[Sparse Attention Adaptation for Long Reasoning](sparse_attention_adaptation_for_long_reasoning.md)**

:   本文提出 SeerAttention-R——一个专为推理模型「长解码」阶段设计的稀疏注意力框架，通过一个轻量、可插拔的自蒸馏注意力门控（AttnGate）学出每一步该激活哪些 KV 块，仅用 0.4B token 训练门控、不动原模型权重，就能在 AIME 等基准上以 4K token 预算保持近乎无损的推理精度，并配套 TileLang 块稀疏解码 kernel 在 H100 上相比 FlashAttention-3 取得最高约 9× 的加速。

**[SparseD: Sparse Attention for Diffusion Language Models](sparsed_sparse_attention_for_diffusion_language_models.md)**

:   针对扩散语言模型（DLM）双向注意力随上下文长度二次膨胀、推理慢的问题，SparseD 通过"早期步用全注意力 + 一次性预计算 head-specific 稀疏模式并跨步复用 + prefill/generation 孤立选择"三招，在 64k 上下文、1024 步去噪下相对 FlashAttention 最高获得 1.50× 无损加速。

**[SpecBranch: Speculative Decoding via Hybrid Drafting and Rollback-Aware Branch Parallelism](specbranch_speculative_decoding_via_hybrid_drafting_and_rollback-aware_branch_pa.md)**

:   SpecBranch 借鉴 CPU 分支预测思想，让草稿模型在目标模型验证的同时并行生成多条"投机分支"以对冲拒绝，并用一个融合显式目标特征与隐式置信度的轻量三分类器（H-RAD）自适应决定草稿长度与分支点，在弱对齐模型上把回滚率从 66–90% 压到 40% 以下，相比自回归解码取得 1.8×∼4.5× 的端到端加速且保持采样分布无损。

**[Speculative Speculative Decoding](speculative_speculative_decoding.md)**

:   本文提出 Speculative Speculative Decoding，把普通投机解码中“先 draft、再 verify、再继续 draft”的串行依赖改成异步预投机：验证还在跑时，draft 模型提前猜测可能的验证结果并为这些结果准备下一轮候选，最终的 SAGUARO 算法在 Llama-3.1-70B 等设置上比强投机解码基线平均快约 30%，相对自回归解码最高接近 $5\times$。

**[Stacked From One: Multi-Scale Self-Injection for Context Window Extension](stacked_from_one_multi-scale_self-injection_for_context_window_extension.md)**

:   SHAREDLLM 把同一个短上下文 LLM 拆成「下层压缩器 + 上层解码器」两份堆叠模型，下层把长输入压成粗到细的多粒度上下文树并只在最底部几层把 KV「自注入」给上层，于是仅用 8K 序列训练就能外推到 128K，速度比流式快 2 倍、比 encoder-decoder 快 3 倍，且性能持平或更优。

**[STEM: Scaling Transformers with Embedding Modules](stem_scaling_transformers_with_embedding_modules.md)**

:   STEM 把 SwiGLU FFN 里的 up-projection 矩阵换成一张「按 token id 查表」的 layer-local embedding 表，用静态稀疏替代 MoE 的动态路由，从而在去掉约三分之一 FFN 参数、降低每 token FLOPs 的同时，训练更稳、知识容量更大，在 350M / 1B 规模上把下游平均分提升约 3–4%。

**[SwingArena: Adversarial Programming Arena for Long-context GitHub Issue Solving](swingarena_competitive_programming_arena_for_long-context_github_issue_solving.md)**

:   提出SwingArena对抗性评测框架，让两个LLM在真实GitHub issue上交替扮演补丁提交者和测试审查者，通过仓库原生CI流水线（编译/lint/回归测试）端到端验证，在C++/Python/Rust/Go四语言400个实例上揭示了模型在"激进补丁生成"与"防御性质量保证"间的行为分化。

**[Tactic: Adaptive Sparse Attention with Clustering and Distribution Fitting for Long-Context LLMs](tactic_adaptive_sparse_attention_with_clustering_and_distribution_fitting_for_lo.md)**

:   Tactic 不再给稀疏注意力定一个固定 token 预算，而是定一个"累积注意力分数"目标 $P$，按注意力分数从高到低取 token 直到累积分数达到 $P$ 为止；为了能在解码时高效逼近这个选择，它用 K-means 聚类做近似排序、用分布拟合估计每个 token 的分数，最终在保持接近全注意力精度的同时实现最高 7.29× 的解码注意力加速、1.58× 端到端加速。

**[Test-Time Training Done Right](test-time_training_done_right.md)**

:   本文指出现有 Test-Time Training（TTT）之所以在长序列上跑不动，是因为它们坚持用极小的在线 mini-batch（每 16~64 个 token 更新一次快权重），导致现代 GPU 利用率常年低于 5%；作者反其道而行，提出 **LaCT（Large-Chunk Test-Time Training）**，把更新粒度放大到 2K~1M token 的超大块，配合窗口注意力补足块内局部性，用几十行纯 PyTorch 就把 GPU 利用率拉到 70%，并在新视角合成、语言建模、自回归视频扩散三类模态上验证了可扩展到 14B 参数、56K~1M token 上下文。

**[The End of Manual Decoding: Towards Truly End-to-End Language Models](the_end_of_manual_decoding_towards_truly_end-to-end_language_models.md)**

:   本文提出 AutoDeco，在标准 Transformer 上挂两个轻量预测头，让模型在每一步解码时自己预测当前 token 该用的温度和 top-p，把原本靠手工调参的解码过程变成可微、可端到端训练的一部分，在 8 个 benchmark 上不仅稳超默认采样，还逼平了"在测试集上作弊调参"得到的 oracle 上界，几乎零额外延迟。

**[The Pensieve Paradigm: Stateful Language Models Mastering Their Own Context](the_pensieve_paradigm_stateful_language_models_mastering_their_own_context.md)**

:   本文提出 StateLM——一类被赋予"自己动手编辑上下文"能力的基础模型：它通过一套记忆工具（删除上下文、建索引、做笔记）在多轮推理中读一段、记要点、删原文，把上下文长度维持成"锯齿状"而非单调累积，从而在长文档 QA、对话记忆、深度检索三类任务上只用 1/4 的活跃上下文就大幅超过标准 LLM。

**[ThinKV: Thought-Adaptive KV Cache Compression for Efficient Reasoning Models](thinkv_thought-adaptive_kv_cache_compression_for_efficient_reasoning_models.md)**

:   ThinKV 观察到推理模型长 CoT 里的注意力稀疏度能把 token 分成"推理/执行/过渡"三类思维，于是按思维重要性给 token 分配量化精度、并在推理轨迹变化时渐进淘汰低价值思维段，再配一个扩展 PagedAttention 的 kernel 原地复用被淘汰的内存槽，最终用不到 5% 的 KV cache 实现近乎无损精度，吞吐量最高比 SOTA 高 5.8 倍。

**[Three Forward, One Backward: Memory-Efficient Full-Rank Fine-Tuning of Large Models via Extra Forward Passes](three_forward_one_backward_memory-efficient_full-rank_fine-tuning_of_large_model.md)**

:   针对 LoRA「只能在低秩子空间更新、表达力受限」与 MeZO「纯零阶估计方差大、收敛慢」各自的硬伤，本文提出 LMAO：在每步迭代里交替做一次 LoRA 的前向+反向（更新低秩矩阵 $A,B$）和两次扰动前向的零阶估计（更新基座权重 $W$），用「三次前向、一次反向」把一次更新拼成 **full-rank**，在 LoRA / MeZO 级别的显存下逼近全参数微调（FT）的性能。

**[TileLang：在现代神经网络算子中架起可编程性与性能的桥梁](tilelang_bridge_programmability_and_performance_in_modern_neural_kernels.md)**

:   TileLang 提出一个以「tile」为一等公民的可编程 GPU 算子语言，把内存放置、数据搬运、并行划分等底层旋钮显式暴露给开发者，再用统一的融合 tile 级数据流图（FTG）配合「tile 推荐 + tile 推断」两阶段自动补全，用不到 70 行 Python 写出接近手写 CUDA 性能的算子，相比 Triton 在 H100 上平均加速 3.02×、AMD 上 2.65×。

**[Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models](towards_greater_leverage_scaling_laws_for_efficient_mixture-of-experts_language_.md)**

:   本文提出"效率杠杆"（Efficiency Leverage, EL）这一指标来量化 MoE 相对稠密模型省了多少算力，通过训练 300+ 个最大 28B 的 MoE 模型拟合出一条以激活率、专家粒度、算力预算为自变量的统一标度律，并据此设计出仅 0.85B 激活参数的 MoE-mini，用 7 倍更少算力追平 6.1B 稠密模型。

**[Training-Free Loosely Speculative Decoding: Accepting Semantically Correct Drafts Beyond Exact Match](training-free_loosely_speculative_decoding_accepting_semantically_correct_drafts.md)**

:   针对标准投机解码"精确匹配"会误杀语义正确草稿的问题，FLy 用目标模型自身的熵和"自纠正"行为，无需任何训练就放行措辞不同但语义等价的 token，在保持 ≥99% 精度的同时把 Llama-3.1-70B 平均加速 2.81×、405B 加速 5.07×，且在分布外任务上比训练式的 EAGLE-3 快 1.62×。

**[TrimR：基于验证器、免训练的思维裁剪，用于高效测试时扩展](trimr_verifier-based_training-free_thinking_trimming_for_efficient_test-time_sca.md)**

:   TrimR 用一个免微调的 7B 小验证器，在大推理模型（LRM）生成思维链的过程中实时检测「过度思考 / 思考不足 / 重复」三类冗余，并用引导提示**温和或强制**地让 LRM 提前收尾，在 MATH500、AIME24/25、GPQA 上把 QwQ-32B、R1-Distill-Qwen-32B、Pangu-R-38B 的推理运行时间最高砍掉 70%，而准确率几乎不掉（最多降 1.7%）。

**[TyphoonMLA: A Mixed Naive-Absorb MLA Kernel For Shared Prefix](typhoonmla_a_mixed_naive-absorb_mla_kernel_for_shared_prefix.md)**

:   TyphoonMLA 发现 shared prefix 场景下 MLA 解码的共享段更适合用 naive 计算、非共享段仍适合用 absorb 计算，于是把同一次 attention 拆成两路 kernel 并用 LSE 合并，在不改模型精度和不训练的前提下，把 MLA attention 吞吐最高提升到约 $3.24\times$，端到端 token 生成率最高提升 $1.48\times$。

**[UltraLLaDA: Scaling the Context Length to 128K for Diffusion Large Language Models](ultrallada_scaling_the_context_length_to_128k_for_diffusion_large_language_model.md)**

:   本文针对扩散语言模型（diffusion LLM）的长上下文扩展问题，提出一个考虑扩散双向注意力特性的 Diffusion-aware NTK 位置编码缩放方法，再配合抑制跨文档干扰的掩码后训练，把 LLaDA-8B 的上下文窗口从 4K 轻量扩展到 128K（仅 600 步训练），在 NIAH/PPL/LongBench/RULER 上大幅超过免训练基线 LongLLaDA。

**[UltraMemV2: Memory Networks Scaling to 120B Parameters with Superior Long-Context Learning](ultramemv2_memory_networks_scaling_to_120b_parameters_with_superior_long-context.md)**

:   UltraMemV2 重新设计了 memory-layer 稀疏架构，把记忆层放进每个 Transformer block，并用更高效的检索、value 处理、初始化和计算配比，让 memory network 在相同激活计算下接近 8-expert MoE，同时在长上下文记忆与 in-context learning 上更强、推理访存更低。

**[Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models](understanding_and_improving_length_generalization_in_hierarchical_sparse_attenti.md)**

:   系统解剖基于 chunk 的稀疏注意力架构，识别出三个关键设计原则（非线性 Chunk Encoder + CLS token、Bypassing Residual Path、训练时强制选择稀疏性），将 4K 上下文训练的模型成功外推到 3200 万 token。

**[Understanding the Mixture-of-Experts with Nadaraya-Watson Kernel](understanding_the_mixture-of-experts_with_nadaraya-watson_kernel.md)**

:   本文用经典的 Nadaraya-Watson 核回归重新解释 MoE 路由（路由权重 = 核函数、专家输出 = 被加权的"标签"），并据此把 MoE 看成一个"大 FFN"，进而提出零额外开销的 FFN 风格路由函数 KERN（ReLU 激活 + $\ell_2$ 归一化），在多种规模、序列长度和稀疏度下都稳定优于 Softmax / Sigmoid 路由。

**[Universal Model Routing for Efficient LLM Inference](universal_model_routing_for_efficient_llm_inference.md)**

:   本文提出 UniRoute，把每个 LLM 编码成"在一小批代表性提示上的预测错误向量"，配合双线性打分器，让训练好的路由器**不重训**就能路由到测试时才出现的新 LLM，在 30+ 个未见模型上取得更好的成本-质量权衡。

**[Unlocking Full Efficiency of Token Filtering in Large Language Model Training](unlocking_full_efficiency_of_token_filtering_in_large_language_model_training.md)**

:   针对「token filtering 能提升模型效果却几乎没省训练时间」这个怪现象，本文提出 CENTRIFUGE：在注意力反向核里进一步过滤被丢弃 token 的激活，把稀疏从输出层一路传播到所有前层；再用「维度规约的稠密 GEMM」替代低效的稀疏 GEMM，让 30%~50% 这种「不上不下」的稀疏度也能真正提速——过滤 50% token 时反向加速最高 49.9%、端到端最高 34.7%，且完整保留了 token filtering 带来的精度收益（最高 +26.6%）。

**[vAttention: Verified Sparse Attention via Sampling](vattention_verified_sparse_attention_via_sampling.md)**

:   vAttention 把"确定性 top-k 选关键 token"和"对长尾随机采样"统一进同一套注意力计算，并用中心极限定理自适应决定采样预算，让稀疏注意力第一次能对每个 head 给出用户指定的 $(\epsilon, \delta)$ 近似误差保证，在 RULER-HARD 上 10% 稀疏度下比 HashAttention 高出约 4.5 个百分点。

**[When Does Divide and Conquer Work for Long Context LLM? A Noise Decomposition Framework](when_does_divide_and_conquer_work_for_long_context_llm_a_noise_decomposition_fra.md)**

:   提出理论框架将长上下文任务失败分解为三类噪声（任务噪声/模型噪声/聚合器噪声），证明当模型噪声超线性增长时弱模型+分块处理可超越强模型单次处理，并给出快速估计最优 chunk size 的方法（3-5 个样本即可）。

**[Wide-In, Narrow-Out: Revokable Decoding for Efficient and Effective DLLMs](wide-in_narrow-out_revokable_decoding_for_efficient_and_effective_dllms.md)**

:   针对扩散大语言模型（DLLM）"并行解码必掉点"的质量-速度困境，本文提出训练无关的 WINO 解码算法：用一个低阈值"激进起草（Wide-In）"+ 一个高阈值"严格验证、把可疑 token 重新打回 mask（Narrow-Out）"的并行 draft-and-verify 机制，让早期错误可被后续更丰富的上下文撤销重写，在 LLaDA / MMaDA 上做到 6×～10× 加速的同时精度还涨。

**[xLSTM Scaling Laws: Competitive Performance with Linear Time-Complexity](xlstm_scaling_laws_competitive_performance_with_linear_time-complexity.md)**

:   系统对比 xLSTM 与 Transformer 的 scaling law，证明 xLSTM 在训练损失-算力 Pareto 前沿、过训练 regime 和推理速度上全面优于同规模 Transformer，且优势随上下文长度增大而增长。
