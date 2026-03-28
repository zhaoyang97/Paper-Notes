<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧩 多模态 VLM

**🤖 AAAI2026** · 共 **29** 篇

**[Aligning the True Semantics: Constrained Decoupling and Distribution Sampling for Cross-Modal Alignment](aligning_the_true_semantics_constrained_decoupling_and_distr.md)**

:   提出 CDDS 算法，通过双路径 UNet 将嵌入解耦为语义和模态分量，并利用分布采样方法间接实现跨模态语义对齐，避免直接调整嵌入导致的分布扭曲，在 Flickr30K 和 MS-COCO 上超越 SOTA 6.6%~14.2%。

**[anyECG-chat: A Generalist ECG-MLLM for Flexible ECG Input and Multi-Task Understanding](anyecg-chat_a_generalist_ecg-mllm_for_flexible_ecg_input_and.md)**

:   构建anyECG数据集（含报告生成、波形定位、多ECG比较三大任务）并提出anyECG-chat模型，通过动态ECG输入机制支持变长/少导联/多ECG输入，采用三阶段课程学习训练，在报告生成的OOD泛化、秒级异常波形定位和多ECG对比分析上全面超越现有ECG-MLLM。

**[AStar: Boosting Multimodal Reasoning with Automated Structured Thinking](astar_boosting_multimodal_reasoning_with_automated_structure.md)**

:   提出AStar，一种training-free的多模态推理范式，通过从500个种子样本中构建高层"thought cards"推理模板库，在推理时自适应检索最优模板引导MLLM结构化推理，7B模型在MathVerse上达53.9%准确率（超越GPT-4o的50.2%），仅需50分钟预处理时间且无需训练。

**[BOFA: Bridge-Layer Orthogonal Low-Rank Fusion for CLIP-Based Class-Incremental Learning](bofa_bridge-layer_orthogonal_low-rank_fusion_for_clip-based_.md)**

:   提出BOFA框架，仅微调CLIP已有的跨模态投影层（bridge-layer），通过正交低秩融合（Orthogonal Low-Rank Fusion）将参数更新约束在与旧任务特征正交的低秩"安全子空间"中，配合跨模态混合原型分类器，在不增加任何额外参数和推理开销的前提下实现了SOTA的无样本存储类增量学习。

**[Branch, or Layer? Zeroth-Order Optimization for Continual Learning of Vision-Language Models](branch_or_layer_zeroth-order_optimization_for_continual_lear.md)**

:   本文系统探索了零阶（ZO）优化在基于PEFT的视觉-语言持续学习（VLCL）中的应用，发现全ZO替换会导致训练不稳定，提出从分支级（branch-wise）到层级（layer-wise）的渐进式ZO-FO混合策略，并基于视觉模态方差更大的理论发现提出MoZO策略（梯度符号归一化+视觉扰动约束），在四个benchmark上达到SOTA。

**[Bridging Modalities via Progressive Re-alignment for Multimodal Test-Time Adaptation (BriMPR)](bridging_modalities_via_progressive_re-alignment_for_multimo.md)**

:   提出 BriMPR 框架，通过"分而治之"策略将多模态测试时自适应(MMTTA)分解为多个单模态特征对齐子问题，先用 prompt tuning 校准各模态全局特征分布实现初始跨模态语义对齐，再通过跨模态掩码嵌入重组和实例级对比学习精细化对齐。

**[Bridging the Copyright Gap: Do Large Vision-Language Models Recognize and Respect Copyrighted Content?](bridging_the_copyright_gap_do_large_vision-language_models_r.md)**

:   首次系统评估 LVLM 在多模态上下文中对版权内容的识别和遵守能力，构建了 50,000 对多模态查询-内容的大规模 benchmark，发现 11/12 个 SOTA LVLM 即使面对明确版权声明也无法有效拒绝侵权请求，并提出 CopyGuard 工具增强框架将侵权拒绝率从 ~3% 提升至 ~62%。

**[Concept-RuleNet: Grounded Multi-Agent Neurosymbolic Reasoning in Vision Language Models](concept-rulenet_grounded_multi-agent_neurosymbolic_reasoning.md)**

:   提出Concept-RuleNet——一个三智能体协作的神经符号推理框架，通过从训练图像中提取视觉概念来条件化符号生成和规则构建，解决了现有方法（如Symbol-LLM）仅依赖标签导致的符号幻觉和不代表性问题，在5个OOD基准上平均提升~5%准确率，幻觉符号减少达50%。

**[Cross-modal Proxy Evolving for OOD Detection with Vision-Language Models](cross-modal_proxy_evolving_for_ood_detection_with_vision-lan.md)**

:   提出CoEvo，一个training-free和annotation-free的test-time框架，通过双向样本条件化的文本/视觉proxy协同演化来增强VLM的zero-shot OOD检测，在ImageNet-1K上比强baseline提升AUROC 1.33%并降低FPR95达45.98%。

**[Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models](cross-modal_unlearning_via_influential_neuron_path_editing_i.md)**

:   提出 MIP-Editor，通过跨层梯度积分（文本）和 Fisher 积分（视觉）定位多模态大语言模型中编码待遗忘知识的**影响力神经元路径**，再用基于路径的表示误导（RMisU）编辑这些神经元，在 MLLMU-Bench 上实现最高 87.75% 的遗忘率和 54.26% 的通用知识保留提升。

**[CrossCheck-Bench: Diagnosing Compositional Failures in Multimodal Conflict Resolution](crosscheck-bench_diagnosing_compositional_failures_in_multim.md)**

:   构建CrossCheck-Bench——首个专注于多模态矛盾检测与解决的诊断基准，包含15K QA对、3层推理复杂度和7种原子能力，发现13个SOTA VLM从感知匹配到逻辑矛盾检测性能一致下降，CoT/SoM等提示策略收效甚微，仅交错符号推理+视觉grounding的方法才有稳定提升。

**[CrossVid: A Comprehensive Benchmark for Evaluating Cross-Video Reasoning in Multimodal Large Language Models](crossvid_a_comprehensive_benchmark_for_evaluating_cross-vide.md)**

:   提出首个系统评估多模态大语言模型（MLLM）跨视频推理（Cross-Video Reasoning, CVR）能力的综合基准CrossVid，涵盖4个维度10个任务、5,331个视频和9,015个QA对，实验揭示当前最佳模型Gemini-2.5-Pro仅达50.4%准确率，远低于人类89.2%。

**[Difference Vector Equalization for Robust Fine-tuning of Vision-Language Models](difference_vector_equalization_for_robust_fine-tuning_of_vis.md)**

:   提出DiVE方法，通过约束预训练和微调模型嵌入之间的"差异向量"在各样本间保持相等，从而在CLIP微调过程中保持嵌入空间的几何结构，同时在ID、OOD、零样本三个指标上取得全面优于现有方法的结果（零样本平均提升8+点）。

**[EM-KD: Distilling Efficient Multimodal Large Language Model with Unbalanced Vision Tokens](em-kd_distilling_efficient_multimodal_large_language_model_w.md)**

:   提出EM-KD框架，通过Hungarian算法解决teacher-student间视觉token数量不平衡问题，结合视觉语义蒸馏(VSD)和视觉-语言亲和力蒸馏(VLAD)将vanilla teacher的知识迁移到高效student MLLM，在11个benchmark上以144 token/patch达到50.4均分，超越576 token的LLaVA-NeXT(49.4)同时推理速度提升近2倍。

**[Exo2Ego: Exocentric Knowledge Guided MLLM for Egocentric Video Understanding](exo2ego_exocentric_knowledge_guided_mllm_for_egocentric_vide.md)**

:   提出 Exo2Ego 框架，通过学习外中心(第三人称)与自中心(第一人称)域之间的映射关系，将 MLLM 中丰富的外中心知识迁移到自中心视频理解，结合新构建的 110万同步 ego-exo clip-text 对数据集 Ego-ExoClip 和 60万指令微调数据集 EgoIT，在 8 个自中心视频基准上取得了领先的开源模型性能。

**[Filter, Correlate, Compress: Training-Free Token Reduction for MLLM Acceleration](filter_correlate_compress_training-free_token_reduction_for_.md)**

:   提出FiCoCo三阶段框架（Filter-Correlate-Compress），通过集成视觉感知+语义感知冗余度量筛选丢弃token，利用token间相关性自适应回收信息，实现training-free的MLLM加速。在LLaVA-NeXT上达14.7×FLOPs压缩同时保留93.6%性能，在5种MLLM架构上全面超越FastV、SparseVLM等SOTA。

**[Global Compression Commander: Plug-and-Play Inference Acceleration for High-Resolution Large Vision-Language Models](global_compression_commander_plug-and-play_inference_acceler.md)**

:   提出GlobalCom²，一个**即插即用、无需训练**的token压缩框架，专为动态裁剪（dynamic cropping）结构的高分辨率VLM设计：利用全局缩略图（thumbnail）作为"指挥官"引导局部裁剪区域（crop）的差异化压缩，在压缩90%视觉token的同时保持>90%原始性能。

**[Graph-of-Mark: Promote Spatial Reasoning in Multimodal Language Models with Graph-Based Visual Prompting](graph-of-mark_promote_spatial_reasoning_in_multimodal_langua.md)**

:   提出 Graph-of-Mark (GoM)，一种无需训练的像素级视觉提示方法，通过在输入图像上直接叠加深度感知的场景图（包含节点和有向边），显式编码物体间的空间关系，使多模态语言模型在 VQA 和定位任务中的零样本空间推理准确率最高提升 11 个百分点。

**[HeadHunt-VAD: Hunting Robust Anomaly-Sensitive Heads in MLLM for Tuning-Free Video Anomaly Detection](headhunt-vad_hunting_robust_anomaly-sensitive_heads_in_mllm_.md)**

:   提出HeadHunt-VAD，不用MLLM的文本输出，而是直接从冻结MLLM中"猎取"一小批对异常敏感且跨prompt鲁棒的注意力头，配合轻量逻辑回归scorer，在仅用1%数据、零微调的条件下，在UCF-Crime(87.03% AUC)和XD-Violence(82.63% AP)上达到tuning-free方法SOTA。

**[HiMo-CLIP: Modeling Semantic Hierarchy and Monotonicity in Vision-Language Alignment](himo-clip_modeling_semantic_hierarchy_and_monotonicity_in_vi.md)**

:   提出 HiMo-CLIP，通过对文本嵌入做 batch 内 PCA 分解（HiDe）提取多粒度语义成分，配合双分支单调性感知对比损失（MoLo），在不修改编码器的前提下让 CLIP 学会"文本越完整、对齐分数越高"的语义单调性，在长文本检索上显著超越现有方法。

**[InEx: Hallucination Mitigation via Introspection and Cross-Modal Multi-Agent Collaboration](inex_hallucination_mitigation_via_introspection_and_cross-mo.md)**

:   提出 InEx 框架，通过内部自省推理（TVER 驱动的不确定性感知视觉增强）和外部跨模态多智能体协作（文本自反思 + 图像编辑验证 + 视觉自反思）迭代验证和修正 MLLM 输出，在 POPE 上提升 8.9%，在多个幻觉和通用 benchmark 上持续超越 OPERA/VCD/ICD。

**[LLM-CAS: Dynamic Neuron Perturbation for Real-Time Hallucination Correction](llm-cas_dynamic_neuron_perturbation_for_real-time_hallucinat.md)**

:   LLM-CAS 首次将 LLM 实时幻觉纠正建模为层次强化学习（HRL）问题，训练 RL Agent 在推理时动态选择最优的神经元扰动策略（高层选择功能网络类别，低层选择扰动类型和幅度），结合自适应掩码+因果追踪精确定位目标神经元，在 StoryCloze 上提升 10.98%，超越 ITI/CAA/SADI 等静态/动态基线。

**[Multimodal DeepResearcher: Generating Text-Chart Interleaved Reports From Scratch with Agentic Framework](multimodal_deepresearcher_generating_text-chart_interleaved_.md)**

:   提出 Multimodal DeepResearcher，一个四阶段 Agent 框架从零生成图文交替研究报告：通过形式化可视化描述（FDV）让 LLM 学习和生成多样化图表，结合 Actor-Critic 迭代精炼机制（LLM生成D3.js代码→浏览器渲染→多模态LLM评审），在自建 MultimodalReportBench 上达到 82% 整体胜率（Claude 3.7），人类评估 100% 胜率。

**[RMAdapter: Reconstruction-based Multi-Modal Adapter for Vision-Language Models (Oral)](rmadapter_reconstructionbased_multimodal_adapter_for_visionlanguage.md)**

:   提出 RMAdapter，一种双分支适配器架构：在标准 adapter 的适应分支旁增加重建分支（类 AutoEncoder），通过共享下投影层和逐层本地重建损失，在 CLIP 少样本微调中实现任务特定适应与通用知识保持的最佳平衡，在 Base-to-Novel 泛化、跨数据集和领域泛化三个任务上全面超越 SOTA（含 Prompt-based 方法）。

**[SafeR-CLIP: Mitigating NSFW Content in Vision-Language Models While Preserving Pre-Trained Knowledge](safer-clip_mitigating_nsfw_content_in_vision-language_models_while_preserving_pr.md)**

:   提出SafeR-CLIP框架，通过近邻感知重定向（将不安全嵌入重定向到语义最近的安全目标而非固定配对）和相对跨模态重定向损失（仅以不安全表示作为负样本而非随机批内负样本），在保持安全性的同时将零样本分类精度比Safe-CLIP恢复8.0%。

**[TOFA: Training-Free One-Shot Federated Adaptation for Vision-Language Models](tofa_training-free_one-shot_federated_adaptation_for_vision-language_models.md)**

:   提出TOFA框架，在联邦学习场景下通过层次贝叶斯模型学习个性化视觉prototype分布 + 全局对齐的LLM文本增强 + 自适应模态融合，实现无需训练、仅一轮通信的CLIP高效适配，在9个数据集上超越one-shot基线甚至部分多轮训练方法。

**[URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding](urag_unified_retrieval_and_generation_in_multimodal_llms_for.md)**

:   URaG 发现 MLLM 处理长文档时存在类人的"粗到细"推理模式（浅层注意力均匀分散、深层集中于证据页），基于此洞察在第 6 层插入轻量跨模态检索模块（仅占参数 0.05%），选取 Top-5 相关页面丢弃其余内容，实现 SOTA 性能的同时减少 44-56% 计算量。

**[VipAct: Visual-Perception Enhancement via Specialized VLM Agent Collaboration and Tool-use](vipact_visual-perception_enhancement_via_specialized_vlm_age.md)**

:   VipAct 提出了一个多Agent协作框架，通过编排器Agent（任务分析+规划+协调）、专用Agent（描述/比较/视觉提示解读）和视觉专家模型（深度估计/目标检测/分割等）三层协作，显著提升 VLM 在细粒度视觉感知任务上的表现，在 Blink 上从 63.74% (zero-shot GPT-4o) 提升到 73.79%。

**[VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models](vp-bench_a_comprehensive_benchmark_for_visual_prompting_in_m.md)**

:   VP-Bench 提出了首个系统评估 MLLM 视觉提示（Visual Prompt）理解能力的两阶段 Benchmark：Stage 1 用 30K+ 图像覆盖 8 种 VP 形状×355 种属性组合评测 VP 感知能力，Stage 2 评测 VP 对 6 个下游任务的实际效果。在 28 个 MLLM 上的评测揭示了 VP 形状选择对性能的关键影响。
