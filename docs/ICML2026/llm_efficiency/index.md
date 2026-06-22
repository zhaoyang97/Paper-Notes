---
title: >-
  ICML2026 LLM效率论文汇总 · 48篇论文解读
description: >-
  48篇ICML2026的 LLM 效率方向论文解读，涵盖 LLM、扩散模型、压缩/编码、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "LLM 效率"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "扩散模型"
  - "压缩/编码"
  - "推理"
item_list:
  - u: "a_risk_decomposition_framework_for_pre-hoc_fine-tuning_prediction/"
    t: "A Risk Decomposition Framework for Pre-Hoc Fine-Tuning Prediction"
  - u: "beyond_sunk_costs_boosting_llm_pre-training_efficiency_via_orthogonal_growth_of_/"
    t: "Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts"
  - u: "criticalkv_optimizing_kv_cache_eviction_from_an_output_perturbation_perspective/"
    t: "CriticalKV: Optimizing KV Cache Eviction from an Output Perturbation Perspective"
  - u: "diffusion_language_model_parallel_decoding_via_product-of-experts_bridge/"
    t: "Diffusion Language Model Parallel Decoding via Product-of-Experts Bridge"
  - u: "dllm-cache_accelerating_diffusion_large_language_models_with_adaptive_caching/"
    t: "dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching"
  - u: "do_transformers_need_three_projections_systematic_study_of_qkv_variants/"
    t: "Do Transformers Need Three Projections？三选一/二的 QKV 共享系统研究"
  - u: "dot-moe_differentiable_optimal_transport_for_moefication/"
    t: "DOT-MoE: 用可微 optimal transport 把 dense LLM 转成 MoE"
  - u: "dynamic_linear_attention/"
    t: "Dynamic Linear Attention"
  - u: "efficient_training-free_multi-token_prediction_via_embedding-space_probing/"
    t: "Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing"
  - u: "ekka_automated_diagnosis_of_silent_errors_in_llm_inference/"
    t: "Ekka: Automated Diagnosis of Silent Errors in LLM Inference"
  - u: "fast-dllm_fréchet_profile_decoding_for_faster_diffusion_llm_inference/"
    t: "Fast-dLLM++: Fréchet Profile Decoding for Faster Diffusion LLM Inference"
  - u: "focus_dllms_know_how_to_tame_their_compute_bound/"
    t: "FOCUS: DLLMs Know How to Tame Their Compute Bound"
  - u: "graphflow_a_graph-based_workflow_management_for_efficient_llm-agent_serving/"
    t: "GraphFlow: A Graph-Based Workflow Management for Efficient LLM-Agent Serving"
  - u: "hyperparameter_transfer_with_mixture-of-expert_layers/"
    t: "Hyperparameter Transfer with Mixture-of-Experts Layers"
  - u: "ir3de_a_linear_router_for_large_language_models/"
    t: "IR3DE: A Linear Router for Large Language Models"
  - u: "kalman_linear_attention_parallel_bayesian_filtering_for_efficient_language_model/"
    t: "Kalman Linear Attention: Parallel Bayesian Filtering For Efficient Language Modelling and State Tracking"
  - u: "knapspec_self-speculative_decoding_via_adaptive_layer_selection_as_a_knapsack_pr/"
    t: "KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem"
  - u: "l3_large_lookup_layers/"
    t: "L$^3$: Large Lookup Layers"
  - u: "minedraft_a_framework_for_batch_parallel_speculative_decoding/"
    t: "MineDraft: A Framework for Batch Parallel Speculative Decoding"
  - u: "obcache_optimal_brain_kv_cache_pruning_for_efficient_long-context_llm_inference/"
    t: "OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference"
  - u: "optimal_bayesian_stopping_for_efficient_inference_of_consistent_llm_answers/"
    t: "Optimal Bayesian Stopping for Efficient Inference of Consistent LLM Answers"
  - u: "oserve_accelerating_llm_serving_via_spatial-temporal_workload_orchestration/"
    t: "OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration"
  - u: "prism_spectral-aware_block-sparse_attention/"
    t: "Prism: Spectral-Aware Block-Sparse Attention"
  - u: "proactivellm_learning_active_interaction_for_streaming_large_language_models/"
    t: "ProactiveLLM: Learning Active Interaction for Streaming Large Language Models"
  - u: "probmoe_differentiable_probabilistic_routing_for_mixture-of-experts/"
    t: "ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts"
  - u: "proxy_compression_for_language_modeling/"
    t: "Proxy Compression for Language Modeling"
  - u: "q-delta_beyond_key-value_associative_state_evolution/"
    t: "Q-Delta: Beyond Key–Value Associative State Evolution"
  - u: "remoe_boosting_expert_reuse_through_router_fine-tuning_in_memory-constrained_moe/"
    t: "ReMoE: Boosting Expert Reuse through Router Fine-Tuning in Memory-Constrained MoE LLM Inference"
  - u: "repetitioncurse_measuring_and_understanding_router_imbalance_in_mixture-of-exper/"
    t: "RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress"
  - u: "repo_language_models_with_context_re-positioning/"
    t: "RePo: Language Models with Context Re-Positioning"
item_total: 48
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM 效率

**🧪 ICML2026** · **48** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (8)](../../CVPR2026/llm_efficiency/index.md) · [🔬 ICLR2026 (169)](../../ICLR2026/llm_efficiency/index.md) · [💬 ACL2026 (23)](../../ACL2026/llm_efficiency/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_efficiency/index.md) · [🧠 NeurIPS2025 (34)](../../NeurIPS2025/llm_efficiency/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/llm_efficiency/index.md)

🔥 **高频主题：** LLM ×15 · 扩散模型 ×5 · 压缩/编码 ×2 · 推理 ×2

**[A Risk Decomposition Framework for Pre-Hoc Fine-Tuning Prediction](a_risk_decomposition_framework_for_pre-hoc_fine-tuning_prediction.md)**

:   微调 LLM 又贵又难预测，本文把「在开训前/早期就预测微调最终性能」这件事形式化成一个信息约束下的随机估计问题，把预测风险分解成**不可约的内在极限（数据-模型静态兼容性）+ 可约的优化方差**，证明优化方差的衰减速率有一个 $c^{-\alpha}$ 的必然下界（再强的预测器也快不过它），由此推出预算最优的探测停止条件，并用「内在极限 × 衰减率」两个轴把任务组织成 Static-Sufficient / Dynamic-Critical / Noise-Dominant 三个可预测性区制，解释了为什么浅探测在 SST-2 上够用、在 GSM8K 上却失败。

**[Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts](beyond_sunk_costs_boosting_llm_pre-training_efficiency_via_orthogonal_growth_of_.md)**

:   提出对已收敛 MoE 模型的"正交增长"策略——深度方向用 interpositional 层复制、宽度方向用噪声专家复制——将 17B 模型扩展到 70B，在相同额外算力下比从头训练准确率提升 10.6%。

**[CriticalKV: Optimizing KV Cache Eviction from an Output Perturbation Perspective](criticalkv_optimizing_kv_cache_eviction_from_an_output_perturbation_perspective.md)**

:   作者把"哪些 KV 缓存条目算关键"这个一直靠经验拍脑袋的问题，重新写成"最小化注意力输出扰动"的优化问题，推导出扰动的可解析上界（同时涉及注意力权重和经 $W^O$ 投影后的 value 范数），并由此设计了一个即插即用的两阶段贪心选择算法，把 SnapKV/AdaKV/HeadKV 三种 SOTA 驱逐方法在 29 个长上下文数据集上的压缩损失平均砍掉一半以上。

**[Diffusion Language Model Parallel Decoding via Product-of-Experts Bridge](diffusion_language_model_parallel_decoding_via_product-of-experts_bridge.md)**

:   扩散语言模型（DLM）能并行解码但质量差，直接用蒙特卡洛把 DLM 草稿校正到自回归（AR）目标又因分布差距太大而代价高昂；本文 PoE-Bridge 在 DLM 与 AR 之间插入一个 Product-of-Experts 中间桥分布，把「DLM→AR」这步难校正拆成「DLM→PoE→AR」两步易校正，配合混合温度采样与弹性拒绝窗口，在数学推理与代码任务上比标准 DLM 解码提速 5×、并恢复至少 95% 的 AR 精度。

**[dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching](dllm-cache_accelerating_diffusion_large_language_models_with_adaptive_caching.md)**

:   针对扩散式大语言模型 (dLLM) 因双向注意力无法复用 KV cache 而推理极慢的问题，本文提出训练无关的 dLLM-Cache，对静态 prompt 用长间隔缓存、对动态 response 用短间隔刷新+按 Value 余弦相似度选 25% 最"变化"的 token 做局部重算，在 LLaDA 8B / Dream 7B 上获得最高 9.1× FLOPs 加速且分数基本不掉。

**[Do Transformers Need Three Projections？三选一/二的 QKV 共享系统研究](do_transformers_need_three_projections_systematic_study_of_qkv_variants.md)**

:   论文系统比较三种 QKV 投影共享方案——Q=K-V（共享 query 和 key）、Q-K=V（共享 key 和 value）、Q=K=V（三者共享），发现 Q-K=V 在 LM 上 PPL 仅升 3.1% 但 KV cache 减 50%，与 GQA/MQA 正交可叠加得 87.5%-96.9% cache 减少；为 edge inference 提供 quantifiable memory benefit。

**[DOT-MoE: 用可微 optimal transport 把 dense LLM 转成 MoE](dot-moe_differentiable_optimal_transport_for_moefication.md)**

:   DOT-MoE 把"dense FFN 转成 MoE 时怎么分配神经元到专家"建模成 differentiable optimal transport——Sinkhorn-Knopp 迭代解 entropic-regularized balanced transport + Straight-Through Estimator 让 neuron-to-expert assignment 和 router 联合 end-to-end 学习；在 LLaMA-2/3 + Qwen2.5 上 50% 激活参数下保留 90% dense 性能，超过 structured pruning / random / 聚类等所有 baseline。

**[Dynamic Linear Attention](dynamic_linear_attention.md)**

:   针对现有"多状态线性注意力"用固定规则合并记忆、会把关键 token 过早压进粗摘要并累积误差的问题，DLA 提出一套**信息感知 + 容量受限**的动态记忆框架：用一个轻量的"状态信息分数"按 token 级信息变化自适应地决定何时新建/合并记忆状态，并用固定大小的时序缓存压住状态膨胀，在 16 个数据集上稳定超过 SOTA 的 Log-Linear Attention，且 DLA 版 Mamba-2 能逼平同参数量的全注意力 Transformer。

**[Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing](efficient_training-free_multi-token_prediction_via_embedding-space_probing.md)**

:   本文提出 ESP（Embedding-Space Probing）：在不修改任何权重、不训练任何辅助模型的前提下，把"prompt 嵌入均值"作为 mask token 注入到冻结 LLM 的输入序列里，借助一次前向同时探出未来多个 token，再用基础模型自身做无损推测验证，在 LLaMA3 / Qwen3 上比同类训练免费基线（LADE / STAND / PLD）的平均接受长度高 7–11%、吞吐高 15–19%。

**[Ekka: Automated Diagnosis of Silent Errors in LLM Inference](ekka_automated_diagnosis_of_silent_errors_in_llm_inference.md)**

:   Ekka 把 LLM 服务框架里"输出悄悄变烂、却没有报错"的静默错误诊断问题，建模为以 HuggingFace 这类参考实现为 oracle 的差分调试任务，用一套"组件映射 → 激活对齐 → 变点分析"的 agentic 流水线自动定位到出问题的具体模块，在 17 个真实 vLLM/SGLang issue 上取得 80% pass@1 / 88% pass@5 的诊断准确率，并新发现 4 个被开发者确认的隐藏 bug。

**[Fast-dLLM++: Fréchet Profile Decoding for Faster Diffusion LLM Inference](fast-dllm_fréchet_profile_decoding_for_faster_diffusion_llm_inference.md)**

:   针对扩散语言模型（dLLM）的并行解码瓶颈，本文提出训练无关的 Fréchet 画像解码：用整条排序后的置信度画像而不是"最弱被选 token"那一项来决定本步并行 commit 多少 token，把 Fast-dLLM 的 factor 规则严格推广到异质置信度场景，在 LLaDA-8B 上四个基准平均吞吐 1.36×、NFE 降 29%，精度几乎不变。

**[FOCUS: DLLMs Know How to Tame Their Compute Bound](focus_dllms_know_how_to_tame_their_compute_bound.md)**

:   FOCUS 发现扩散大语言模型（DLLM）解码时一个 block 里每步只有约 10% 的 token 真正被解码、其余 90% 算力全浪费，并揭示"前两层注意力重要度的增量"能高度预测哪些 token 这一步可解码；据此设计一套免训练的推理系统，在 Layer 1 后就把不可解码 token 驱逐掉，让有效 batch 变大，在大 batch 下相对生产级引擎 LMDeploy 拿到最高 3.52× 的吞吐提升且生成质量不降反升。

**[GraphFlow: A Graph-Based Workflow Management for Efficient LLM-Agent Serving](graphflow_a_graph-based_workflow_management_for_efficient_llm-agent_serving.md)**

:   GraphFlow 把多个 agent 工作流统一到一张全局操作 DAG（wGraph）上，用 GNN+MLP 按任务在线生成子图工作流，并通过"基底 KV + 稀疏前缀残差 + 路径剪枝"的差分缓存替代传统按工作流独立缓存，在 5 个推理/代码/QA benchmark 上平均提升 4.95pp 的同时把 KV 内存压到约 1/4。

**[Hyperparameter Transfer with Mixture-of-Experts Layers](hyperparameter_transfer_with_mixture-of-expert_layers.md)**

:   本文把 μP/CompleteP 的最大更新参数化思想扩展到稀疏 MoE Transformer，给出 router、expert 上/下投影、expert bias 在 width/depth/专家数/专家宽度同时放大时的初始化与学习率缩放规则，并用一套三层 mean-field 的 DMFT 理论证明该参数化在 $n_{\text{embd}},n_{\text{exp}},n_{\text{hid}},L\to\infty$（固定激活稀疏度 $\kappa$）下存在尺度不变极限，从 38M 激活基模迁移到 2B 总参的 MoE 上都能直接复用最优 LR / init，且零样本超参训出来的 MoE 在等激活参数下可与 dense GPT2 speedrun 持平甚至更优。

**[IR3DE: A Linear Router for Large Language Models](ir3de_a_linear_router_for_large_language_models.md)**

:   这篇论文提出 IR3DE——一个用**岭回归闭式解**构建的线性 LLM 路由器，仅凭 token 嵌入就把每条 prompt 路由到最合适的领域专家模型，无需训练额外语言模型、无需把各领域数据集中到一处，且专家可随时增删而不必重训路由器；尽管是线性的，它在推理任务上以 98.4% 的归一化性能超过所有基线。

**[Kalman Linear Attention: Parallel Bayesian Filtering For Efficient Language Modelling and State Tracking](kalman_linear_attention_parallel_bayesian_filtering_for_efficient_language_model.md)**

:   把序列混合重新解释成一次精确的贝叶斯滤波，用卡尔曼滤波器的"信息形式"把本该顺序递归的更新改写成可并行前缀扫描的 Möbius（分式线性）映射，得到一个即插即用、线性复杂度、却比 GLA 更具表达力、还自带显式不确定性的序列混合层 KLA。

**[KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem](knapspec_self-speculative_decoding_via_adaptive_layer_selection_as_a_knapsack_pr.md)**

:   KnapSpec 把自推测解码（SSD）的草稿层选择重新建模为 0/1 背包问题，把 Attention 与 MLP 解耦、用上下文长度依赖的硬件延迟作为"重量"、用 hidden state 余弦相似度（首次给出严格证明）作为"价值"，通过并行 DP 在每一步自适应找出最大化 Tokens-per-Time 的子网络，在 Qwen3 / Llama3 上长上下文场景拿到最高 1.47× 的真实墙钟加速且无需额外训练。

**[L$^3$: Large Lookup Layers](l3_large_lookup_layers.md)**

:   本文提出 L$^3$（Large Lookup Layer），把 tokenizer embedding table 推广为可插入到 decoder 中的"大查表层"——按 token ID 做**静态路由**取出一组学习好的 key/value embeddings，再让当前隐藏状态对其做 attention 聚合，从而在不引入 MoE 那套动态路由+辅助损失+难以 offload 的痛点下，把模型稀疏度再上一个量级；在 800M–2.6B 激活参数上击败同算力的 dense 模型与同稀疏率的 MoE。

**[MineDraft: A Framework for Batch Parallel Speculative Decoding](minedraft_a_framework_for_batch_parallel_speculative_decoding.md)**

:   MineDraft 通过维护两批请求并让一批的 drafting 与另一批的 verification 在两组独立 GPU 上**重叠执行**,把投机解码中原本串行的"草稿—验证"流水线变成批并行 PSD,在仅多用 1 张 GPU 的代价下相对标准 SD 把吞吐拉高最多 75%、端到端延迟降低最多 39%,并已实现为可即插即用的 vLLM 插件。

**[OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference](obcache_optimal_brain_kv_cache_pruning_for_efficient_long-context_llm_inference.md)**

:   本文把 KV cache eviction 重新表述为"逐层结构化剪枝"问题，借用 Optimal Brain Damage 的二阶 Taylor 近似推导出针对独立 value、独立 key、key-value 联合三种剪枝单位的闭式 saliency 分数，作为即插即用的"分数替换件"接入 H2O / TOVA / SnapKV / AdaKV 等现有 attention-only eviction 框架，在 LLaMA-3.1 / Qwen-2.5 的 RULER 与 LongBench 上获得稳定提升（AdaKV 在 query-agnostic RULER-4K 30% budget 上提升近 15%）。

**[Optimal Bayesian Stopping for Efficient Inference of Consistent LLM Answers](optimal_bayesian_stopping_for_efficient_inference_of_consistent_llm_answers.md)**

:   本文把"自洽性 (Self-Consistency) 多次采样选众数"问题建模为带先验信息的贝叶斯最优停止问题，并提出一种只跟踪"出现次数最多 / 次多 / 其他合计"三类计数的 $L$-聚合后验近似，从理论上证明 $L=3$ 即可在 $\delta \to 0$ 时达到与精确后验完全一致的渐近最优停止时间，实验上以约 1.4 倍 ASC 的速度在 GSM8K / CommonsenseQA 上节省 30%–80% 的 LLM 调用。

**[OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration](oserve_accelerating_llm_serving_via_spatial-temporal_workload_orchestration.md)**

:   OServe 把 LLM 服务的「资源分配 + 并行策略 + 请求路由」联合建模为流网络上的双层最大流问题，配合 LSTM 工作负载预测和基于 GPU 互联的 ad hoc 模型切换，应对真实流量在空间（不同请求类型）和时间（成分随时刻变化）两个维度的异质性，端到端 P99 延迟和吞吐相比 vLLM 平均提升 1.5×、最大 2×。

**[Prism: Spectral-Aware Block-Sparse Attention](prism_spectral-aware_block-sparse_attention.md)**

:   Prism 把"块重要性估计"分解到 RoPE 的高频/低频两个频带分别做 mean-pooling 加 softmax，并用能量比推出的温度自动校准 logit 量级，从而完全用块级运算（不再回落到 token 级搜索）拿到与 full attention 几乎相同的精度，在 128K 上对 FlashAttention-2 取得 5.1× 加速。

**[ProactiveLLM: Learning Active Interaction for Streaming Large Language Models](proactivellm_learning_active_interaction_for_streaming_large_language_models.md)**

:   ProactiveLLM 让流式 LLM 用自己的内部状态（注意力或预测熵）来决定"什么时候该开口"，靠掩码流式建模 + 同步特权自蒸馏在不依赖任何外部对齐标注的前提下学会感知"语义已经够了没"，把交互延迟显著压下去的同时几乎不掉点。

**[ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts](probmoe_differentiable_probabilistic_routing_for_mixture-of-experts.md)**

:   ProbMoE 把 MoE 的 top-$k$ 路由重新表述为"基数受限子集分布上的概率推断"，前向用 SIMPLE 估计器从 exact-$k$ 子集分布中采样、反向用可解析计算的专家边缘概率 $m_j=\partial \log Z_k/\partial \log p_j$ 作为离散选择的可微代理，在 OLMoE/Qwen1.5-MoE 上明显提升 GSM/Law/Translation 等任务并显著改善专家利用率，同时自然延伸出 Dynamic-$k$ 变体——按 token 难度自适应激活专家数。

**[Proxy Compression for Language Modeling](proxy_compression_for_language_modeling.md)**

:   作者提出「proxy compression」——训练时把 90% 数据喂成 tokenizer / 神经压缩器产出的短序列、10% 喂原始 UTF-8 字节，配合 sentinel token 与短暂的 in-context translation warm-up；推理时丢掉所有压缩器，模型只看原始字节，却能在固定 compute 下显著超过纯字节模型，且在大规模下追平甚至超过 tokenizer baseline。

**[Q-Delta: Beyond Key–Value Associative State Evolution](q-delta_beyond_key-value_associative_state_evolution.md)**

:   这篇论文挑战了线性注意力里"query 只负责读出、不参与状态演化"的隐含假设，证明查询读出 $\hat{o}_t=S_{t-1}q_t$ 本身就是一个结构化的值预测（与键检索互补），据此提出 **Q-Delta**——把键、查询的混合预测误差一起注入 delta 规则的状态更新，在保持线性时间与分块并行效率的同时，于语言建模和长上下文检索上稳定超过 DeltaNet、GatedDeltaNet 等强基线（S-NIAH 平均 90.0% vs GatedDeltaNet 83.5%）。

**[ReMoE: Boosting Expert Reuse through Router Fine-Tuning in Memory-Constrained MoE LLM Inference](remoe_boosting_expert_reuse_through_router_fine-tuning_in_memory-constrained_moe.md)**

:   ReMoE 冻结所有非 router 参数、仅微调 gate，用一个"时序局部性正则 + Trust-KL 语义锚"的复合损失把 router 出来的路由轨迹整形得更"缓存友好"，在不改架构、不加运行时开销的前提下把相邻 token 的专家重用率提升约 26%，并在 Jetson Orin NX 上把 TPOT 降低 43.6–49.8%（解码加速 1.77–1.99×）。

**[RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress](repetitioncurse_measuring_and_understanding_router_imbalance_in_mixture-of-exper.md)**

:   通过给 MoE 大模型喂"同一个 token 重复 N 遍"这种极简的 OOD 提示，作者发现 router 会把几乎所有 token 路由到固定的少数几个 top-$k$ 专家上，从而在专家并行（EP）部署下制造单卡瓶颈、把别的 GPU 全部 idle 住，在 8-GPU 集群上把 TTFT 拉高 20%–148%，把 MoE 的并行加速器反过来变成 DoS 攻击面。

**[RePo: Language Models with Context Re-Positioning](repo_language_models_with_context_re-positioning.md)**

:   现有 LLM 把 token 强行按线性整数索引 0…L−1 排位，让注意力层独自承担"整理上下文结构"的重活；RePo 用一个轻量可微模块 $f_\phi$ 根据 token 隐状态给它分配**连续、非线性**的位置值，把这份"额外认知负荷"卸掉，在嘈杂上下文、结构化数据和长上下文任务上稳定涨点，短上下文通用任务几乎不掉。

**[RKSC: Reasoning-Aware KV Cache Sharing and Confident Early Exit for Multi-Step LLM Inference](rksc_reasoning-aware_kv_cache_sharing_and_confident_early_exit_for_multi-step_ll.md)**

:   RKSC 是一个**训练无关**的推理框架，针对"多分支推理"（一题跑多条 reasoning 轨迹再投票/验证）里的两类结构性浪费——跨分支重复算前缀 KV、过深的验证前向——分别用"按隐状态相似度共享 KV"和"双层置信度早退"消掉，在 5 个 7B–10B 模型、4 个 benchmark 上相对无缓存基线平均加速 $3.008\times$，且早退引入的错误率只有 $0.37\%$。

**[Scout: Active Information Foraging for Long-Text Understanding with Decoupled Epistemic States](scout_active_information_foraging_for_long-text_understanding_with_decoupled_epi.md)**

:   Scout 把百万级 token 的长文本理解重新建模为"主动信息觅食"过程，引入与交互轨迹解耦的、带来源锚点的 epistemic state $\mathcal{E}_t$ 作为唯一推理底座，并通过 gap-diagnosed 自评估迭代收缩到查询充分子集，在 LooGLE-v2 和 $\infty$Bench 上既追平甚至超过 Gemini-3-Pro 等前沿模型，又把 token 成本降低到约 $1/8$。

**[SiameseNorm: Breaking the Barrier to Reconciling Pre/Post-Norm](siamesenorm_breaking_the_barrier_to_reconciling_prepost-norm.md)**

:   针对 Pre-Norm 与 Post-Norm 在单流架构内无法共存的结构性矛盾，作者提出双流残差架构 SiameseNorm，让一条未归一化流保留 Pre-Norm 的恒等梯度高速路、一条归一化流保留 Post-Norm 的主路径表征控制，通过共享残差块耦合两条流，在 400M~15B 稠密/MoE 语言模型、ViT、DiT 上均稳定优于 Pre-Norm 基线，开销可忽略。

**[Skill-Based Mixture-of-Experts: Adaptive Routing for Heterogeneous Reasoning via Inferred Skills](skill-based_mixture-of-experts_adaptive_routing_for_heterogeneous_reasoning_via_.md)**

:   SKILL-MOE 提出一个无需训练、以"技能"为路由信号的符号化 MoE 框架：从每个问题里抽取所需技能、按技能-模型档案在 16 个预训练 LLM 中为每条样本动态招募 k 个专家、再用任务级最优聚合器把多条 CoT 融成最终答案；配合按专家分桶的批量推理，单卡就能跑 16 个 7-8B 模型，平均比最强多智能体基线高 8.15%。

**[Skip a Layer or Loop It? Learning Program-of-Layers in LLMs](skip_a_layer_or_loop_it_learning_program-of-layers_in_llms.md)**

:   本文把预训练 LLM 的每一层看成可被任意调用的"原子函数"，提出"程序化层级"（Program-of-Layers, PoLar）——为每个输入定制一个可**跳过(skip)或循环(loop)**层的执行程序；先用 MCTS 实证发现这类更优程序对几乎每个输入都训练-free 地存在，再训一个轻量预测网络单次预测出执行程序，在数学推理基准上比标准前向和已有动态深度方法都更准、还常常执行更少的层。

**[SoftMoE: Soft Differentiable Routing for Mixture-of-Experts in LLMs](softmoe_soft_differentiable_routing_for_mixture-of-experts_in_llms.md)**

:   SoftMoE 用一个基于 LapSum 的可微「软 top-$k$」算子替换 MoE 里不可导的硬 top-$k$ 选择，让路由可以梯度优化、激活专家数随 token 自适应；再加一个全局预算约束让模型**自己学每层该分多少专家**，结果在更省专家的前提下匹配甚至超过稀疏 MoE，并发现一个有趣规律——**越靠后的层越想多激活专家**。

**[Sparser Block-Sparse Attention via Token Permutation](sparser_block-sparse_attention_via_token_permutation.md)**

:   本文提出 PBS-Attn，利用注意力的置换不变性，先按"全局重要性"对 key 在段内重排，把散落各处的 heavy hitter 聚拢成连续高密度块，再做块稀疏计算，从而在保持精度近乎追平 full attention 的同时，把长上下文 prefilling 端到端加速最高 2.75 倍。

**[STAR: Rethinking MoE Routing as Structure-Aware Subspace Learning](star_rethinking_moe_routing_as_structure-aware_subspace_learning.md)**

:   STAR 把 MoE 路由重新理解成一个"子空间学习"问题：在原本浅薄的线性路由器之外，用广义 Hebbian 算法（GHA）在线学一组追踪输入主方向的正交基，让路由决策直接对齐输入结构，从而在合成任务、LLaMA-MoE 预训练、BERT-GLUE 微调和 ViT ImageNet-C 上都拿到更稳的专家特化与更好的下游性能。

**[Stochastic Sparse Attention for Memory-Bound Inference](stochastic_sparse_attention_for_memory-bound_inference.md)**

:   SANTA 把 attention 的 value 聚合 $AV$ 看作 "按 softmax 概率 $A$ 对值行 $V$ 做加权求和", 改成 "从 $A$ 中无放回采样 $S\ll n_k$ 个索引、直接平均对应 $V$ 行"的无偏估计, 用 stratified / systematic 采样降方差, 再写成 GPU kernel 与 FlashDecoding 对齐——在 32k context 下端到端比 FlashInfer / FlashDecoding 快 1.5× 且精度不掉。

**[Structuring The Future: Diffusion LLM Speculative Decoding via Calibrated Draft Graphs](structuring_the_future_diffusion_llm_speculative_decoding_via_calibrated_draft_g.md)**

:   Spiffy 把投机解码搬到扩散语言模型（dLLM）上：不另训草稿模型，直接用目标模型自己的分布"自投机"，并把多步去噪状态组织成一张**有向草稿图**，再用离线校准的图结构最大化接受率，在 LLaDA / Dream / SDAR 上实现最高 8.6× 的模型推理次数削减和 6.3× 的 token 速率提升，且可证明保持输出分布无损。

**[TEAM: Temporal-Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration](team_temporal-spatial_consistency_guided_expert_activation_for_moe_diffusion_lan.md)**

:   TEAM 针对 MoE 扩散语言模型（dLLM）"激活了大量专家却只接受少量 token"的固有错配，利用 block 内解码的时间一致性与空间一致性，为已解码 / 热 / 冷三类 token 设计差异化的专家激活与解码策略，在 SDAR 30B-A3B 上以近乎零精度损失换得最高 2.2× 加速。

**[Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving](theoretically_optimal_attentionffn_ratios_in_disaggregated_llm_serving.md)**

:   本文为新兴的 Attention-FFN 解耦 (AFD) 推理架构提供首个理论框架,基于"prefill 长度有限均值 + decode 长度服从几何分布"的概率工作负载模型,推导出 rA-1F 拓扑下最优 A/F 比的闭式解 $r^*=\max\{r_A, r_C, r_{\text{peak}}\}$,并用 trace-calibrated 模拟器验证理论与实测最优值偏差 <10%。

**[Training-Inference Consistent Segmented Execution for Long-Context LLMs](training-inference_consistent_segmented_execution_for_long-context_llms.md)**

:   本文提出一套训练与推理共享完全相同的分段前向执行语义的长上下文 LLM 框架：跨段只保留固定长度的可微分 KV 尾部 + 一条仅前向的检索旁路，在 LLaMA2-7B 32K/80K 上以约 $6\times$ 更低的 prefill 峰值显存达到与全注意力可比甚至更好的 LongBench/RULER 表现。

**[TuneAhead: Predicting Fine-tuning Performance Before Full Training Begins](tuneahead_predicting_fine-tuning_performance_before_full_training_begins.md)**

:   针对"微调跑完才知道翻车、白烧几百 GPU 小时"的痛点，TuneAhead 把每个候选微调任务编码成"静态数据集描述符 + 100 步探针动态特征"的元特征向量，用 LightGBM 在训练前就预测最终性能（370 个测试任务上 RMSE 1.47pp、95.1% 落在 ±3pp 内），并用 SHAP 给出"为什么会失败"的可诊断解释。

**[Turning Back Without Forgetting: Selective Backward Refinement for Parameter-Efficient Continual Learning](turning_back_without_forgetting_selective_backward_refinement_for_parameter-effi.md)**

:   SABER 在 prompt-based 持续学习里第一次实现了"无回放的正向反向迁移"——用梯度几何 + 损失分布两套相关性判据决定"该不该回头改旧任务的 prompt"，再把更新约束到不干扰旧任务的正交子空间里去"安全地改"，让后学的任务真正反过来提升先学任务的精度。

**[Variational Routing: 校准 MoE Transformer 的可扩展贝叶斯框架](variational_routing_a_scalable_bayesian_framework_for_calibrated_mixture-of-expe.md)**

:   提出变分路由框架 VMoER——通过对 MoE 层的路由决策进行变分推断而非权重推断，实现高效贝叶斯不确定性建模，在保持 <1% FLOPs 额外开销的同时将校准误差降低 94%、路由稳定性提升 38%。

**[VIA-SD: Verification via Intra-Model Routing for Speculative Decoding](via-sd_verification_via_intra-model_routing_for_speculative_decoding.md)**

:   针对投机解码"要么接受、要么用大模型重算"的二元决策瓶颈，VIA-SD 从大验证器内部路由出一个轻量的 slim-verifier 来处理那些"中等置信度"的 token，构成 draft → slim-verifier → full-verifier 的多级验证流程，在四类任务、多个模型族上把拒绝率压低 0.10–0.22，相对强投机解码基线再提速 10–20%。

**[WarmServe：一次加载多模型的 GPU 预热机制](warmserve_enabling_one-for-many_gpu_prewarming_for_multi-llm_serving.md)**

:   WarmServe 通过分析 LLM 服务工作负载的长期周期性规律，主动将多个模型参数预加载到 GPU，配合优化的放置算法和动态 KV 缓存预留策略，使系统能在请求突发时快速启动新实例——尾部 TTFT 相比现有系统降低 50.8 倍。
