<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🤖 AAAI2026 论文笔记

共 **240** 篇笔记，覆盖 **34** 个领域。

## 领域概览

| 领域 | 篇数 |
|:-----|-----:|
| 👁️ [多模态 VLM](#multimodal_vlm) | 27 |
| 🦾 [LLM Agent](#llm_agent) | 26 |
| 💡 [LLM 推理](#llm_reasoning) | 14 |
| 🎨 [图像生成](#image_generation) | 13 |
| 🚗 [自动驾驶](#autonomous_driving) | 12 |
| 🤖 [机器人/具身智能](#robotics) | 12 |
| 🎮 [强化学习](#reinforcement_learning) | 10 |
| 📦 [模型压缩](#model_compression) | 9 |
| ⚖️ [对齐 / RLHF](#llm_alignment) | 8 |
| 🏥 [医学图像](#medical_imaging) | 8 |
| 🧊 [3D 视觉](#3d_vision) | 7 |
| 🛡️ [AI 安全](#ai_safety) | 7 |
| 💬 [LLM / NLP](#llm_nlp) | 7 |
| 🎯 [目标检测](#object_detection) | 7 |
| 🕸️ [图学习](#graph_learning) | 6 |
| 🎁 [推荐系统](#recommender) | 5 |
| ✂️ [语义分割](#segmentation) | 5 |
| 🧑 [人体理解](#human_understanding) | 4 |
| 🖼️ [图像恢复](#image_restoration) | 4 |
| 📐 [优化/理论](#optimization) | 4 |
| 🎬 [视频理解](#video_understanding) | 4 |
| 🔗 [因果推理](#causal_inference) | 3 |
| ⚡ [LLM 效率](#llm_efficiency) | 3 |
| ✍️ [文本生成](#nlp_generation) | 3 |
| 🛰️ [遥感](#remote_sensing) | 3 |
| 🎵 [音频/语音](#audio_speech) | 2 |
| 🔄 [自监督/表示学习](#self_supervised) | 2 |
| 📈 [时间序列](#time_series) | 2 |
| 🔎 [AIGC 检测](#aigc_detection) | 1 |
| 📖 [NLP 理解](#nlp_understanding) | 1 |
| ⚛️ [物理学](#physics) | 1 |
| 🧮 [科学计算](#scientific_computing) | 1 |
| 📡 [信号/通信](#signal_comm) | 1 |
| 📂 [其他](#others) | 18 |

---

## 👁️ 多模态 VLM { #multimodal_vlm }

**[Aligning the True Semantics: Constrained Decoupling and Distribution Sampling for Cross-Modal Alignment](multimodal_vlm/aligning_the_true_semantics_constrained_decoupling_and_distr.md)**

:   提出 CDDS 算法，通过双路径 UNet 将嵌入解耦为语义和模态分量，并利用分布采样方法间接实现跨模态语义对齐，避免直接调整嵌入导致的分布扭曲，在 Flickr30K 和 MS-COCO 上超越 SOTA 6.6%~14.2%。

**[anyECG-chat: A Generalist ECG-MLLM for Flexible ECG Input and Multi-Task Understanding](multimodal_vlm/anyecg-chat_a_generalist_ecg-mllm_for_flexible_ecg_input_and.md)**

:   构建anyECG数据集（含报告生成、波形定位、多ECG比较三大任务）并提出anyECG-chat模型，通过动态ECG输入机制支持变长/少导联/多ECG输入，采用三阶段课程学习训练，在报告生成的OOD泛化、秒级异常波形定位和多ECG对比分析上全面超越现有ECG-MLLM。

**[AStar: Boosting Multimodal Reasoning with Automated Structured Thinking](multimodal_vlm/astar_boosting_multimodal_reasoning_with_automated_structure.md)**

:   提出AStar，一种training-free的多模态推理范式，通过从500个种子样本中构建高层"thought cards"推理模板库，在推理时自适应检索最优模板引导MLLM结构化推理，7B模型在MathVerse上达53.9%准确率（超越GPT-4o的50.2%），仅需50分钟预处理时间且无需训练。

**[BOFA: Bridge-Layer Orthogonal Low-Rank Fusion for CLIP-Based Class-Incremental Learning](multimodal_vlm/bofa_bridge-layer_orthogonal_low-rank_fusion_for_clip-based_.md)**

:   提出BOFA框架，仅微调CLIP已有的跨模态投影层（bridge-layer），通过正交低秩融合（Orthogonal Low-Rank Fusion）将参数更新约束在与旧任务特征正交的低秩"安全子空间"中，配合跨模态混合原型分类器，在不增加任何额外参数和推理开销的前提下实现了SOTA的无样本存储类增量学习。

**[Branch, or Layer? Zeroth-Order Optimization for Continual Learning of Vision-Language Models](multimodal_vlm/branch_or_layer_zeroth-order_optimization_for_continual_lear.md)**

:   本文系统探索了零阶（ZO）优化在基于PEFT的视觉-语言持续学习（VLCL）中的应用，发现全ZO替换会导致训练不稳定，提出从分支级（branch-wise）到层级（layer-wise）的渐进式ZO-FO混合策略，并基于视觉模态方差更大的理论发现提出MoZO策略（梯度符号归一化+视觉扰动约束），在四个benchmark上达到SOTA。

**[Bridging Modalities via Progressive Re-alignment for Multimodal Test-Time Adaptation (BriMPR)](multimodal_vlm/bridging_modalities_via_progressive_re-alignment_for_multimo.md)**

:   提出 BriMPR 框架，通过"分而治之"策略将多模态测试时自适应(MMTTA)分解为多个单模态特征对齐子问题，先用 prompt tuning 校准各模态全局特征分布实现初始跨模态语义对齐，再通过跨模态掩码嵌入重组和实例级对比学习精细化对齐。

**[Bridging the Copyright Gap: Do Large Vision-Language Models Recognize and Respect Copyrighted Content?](multimodal_vlm/bridging_the_copyright_gap_do_large_vision-language_models_r.md)**

:   首次系统评估 LVLM 在多模态上下文中对版权内容的识别和遵守能力，构建了 50,000 对多模态查询-内容的大规模 benchmark，发现 11/12 个 SOTA LVLM 即使面对明确版权声明也无法有效拒绝侵权请求，并提出 CopyGuard 工具增强框架将侵权拒绝率从 ~3% 提升至 ~62%。

**[Concept-RuleNet: Grounded Multi-Agent Neurosymbolic Reasoning in Vision Language Models](multimodal_vlm/concept-rulenet_grounded_multi-agent_neurosymbolic_reasoning.md)**

:   提出Concept-RuleNet——一个三智能体协作的神经符号推理框架，通过从训练图像中提取视觉概念来条件化符号生成和规则构建，解决了现有方法（如Symbol-LLM）仅依赖标签导致的符号幻觉和不代表性问题，在5个OOD基准上平均提升~5%准确率，幻觉符号减少达50%。

**[Cross-modal Proxy Evolving for OOD Detection with Vision-Language Models](multimodal_vlm/cross-modal_proxy_evolving_for_ood_detection_with_vision-lan.md)**

:   提出CoEvo，一个training-free和annotation-free的test-time框架，通过双向样本条件化的文本/视觉proxy协同演化来增强VLM的zero-shot OOD检测，在ImageNet-1K上比强baseline提升AUROC 1.33%并降低FPR95达45.98%。

**[Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models](multimodal_vlm/cross-modal_unlearning_via_influential_neuron_path_editing_i.md)**

:   提出 MIP-Editor，通过跨层梯度积分（文本）和 Fisher 积分（视觉）定位多模态大语言模型中编码待遗忘知识的**影响力神经元路径**，再用基于路径的表示误导（RMisU）编辑这些神经元，在 MLLMU-Bench 上实现最高 87.75% 的遗忘率和 54.26% 的通用知识保留提升。

**[CrossCheck-Bench: Diagnosing Compositional Failures in Multimodal Conflict Resolution](multimodal_vlm/crosscheck-bench_diagnosing_compositional_failures_in_multim.md)**

:   构建CrossCheck-Bench——首个专注于多模态矛盾检测与解决的诊断基准，包含15K QA对、3层推理复杂度和7种原子能力，发现13个SOTA VLM从感知匹配到逻辑矛盾检测性能一致下降，CoT/SoM等提示策略收效甚微，仅交错符号推理+视觉grounding的方法才有稳定提升。

**[CrossVid: A Comprehensive Benchmark for Evaluating Cross-Video Reasoning in Multimodal Large Language Models](multimodal_vlm/crossvid_a_comprehensive_benchmark_for_evaluating_cross-vide.md)**

:   提出首个系统评估多模态大语言模型（MLLM）跨视频推理（Cross-Video Reasoning, CVR）能力的综合基准CrossVid，涵盖4个维度10个任务、5,331个视频和9,015个QA对，实验揭示当前最佳模型Gemini-2.5-Pro仅达50.4%准确率，远低于人类89.2%。

**[Difference Vector Equalization for Robust Fine-tuning of Vision-Language Models](multimodal_vlm/difference_vector_equalization_for_robust_fine-tuning_of_vis.md)**

:   提出DiVE方法，通过约束预训练和微调模型嵌入之间的"差异向量"在各样本间保持相等，从而在CLIP微调过程中保持嵌入空间的几何结构，同时在ID、OOD、零样本三个指标上取得全面优于现有方法的结果（零样本平均提升8+点）。

**[EM-KD: Distilling Efficient Multimodal Large Language Model with Unbalanced Vision Tokens](multimodal_vlm/em-kd_distilling_efficient_multimodal_large_language_model_w.md)**

:   提出EM-KD框架，通过Hungarian算法解决teacher-student间视觉token数量不平衡问题，结合视觉语义蒸馏(VSD)和视觉-语言亲和力蒸馏(VLAD)将vanilla teacher的知识迁移到高效student MLLM，在11个benchmark上以144 token/patch达到50.4均分，超越576 token的LLaVA-NeXT(49.4)同时推理速度提升近2倍。

**[Exo2Ego: Exocentric Knowledge Guided MLLM for Egocentric Video Understanding](multimodal_vlm/exo2ego_exocentric_knowledge_guided_mllm_for_egocentric_vide.md)**

:   提出 Exo2Ego 框架，通过学习外中心(第三人称)与自中心(第一人称)域之间的映射关系，将 MLLM 中丰富的外中心知识迁移到自中心视频理解，结合新构建的 110万同步 ego-exo clip-text 对数据集 Ego-ExoClip 和 60万指令微调数据集 EgoIT，在 8 个自中心视频基准上取得了领先的开源模型性能。

**[Filter, Correlate, Compress: Training-Free Token Reduction for MLLM Acceleration](multimodal_vlm/filter_correlate_compress_training-free_token_reduction_for_.md)**

:   提出FiCoCo三阶段框架（Filter-Correlate-Compress），通过集成视觉感知+语义感知冗余度量筛选丢弃token，利用token间相关性自适应回收信息，实现training-free的MLLM加速。在LLaVA-NeXT上达14.7×FLOPs压缩同时保留93.6%性能，在5种MLLM架构上全面超越FastV、SparseVLM等SOTA。

**[Global Compression Commander: Plug-and-Play Inference Acceleration for High-Resolution Large Vision-Language Models](multimodal_vlm/global_compression_commander_plug-and-play_inference_acceler.md)**

:   提出GlobalCom²，一个**即插即用、无需训练**的token压缩框架，专为动态裁剪（dynamic cropping）结构的高分辨率VLM设计：利用全局缩略图（thumbnail）作为"指挥官"引导局部裁剪区域（crop）的差异化压缩，在压缩90%视觉token的同时保持>90%原始性能。

**[Graph-of-Mark: Promote Spatial Reasoning in Multimodal Language Models with Graph-Based Visual Prompting](multimodal_vlm/graph-of-mark_promote_spatial_reasoning_in_multimodal_langua.md)**

:   提出 Graph-of-Mark (GoM)，一种无需训练的像素级视觉提示方法，通过在输入图像上直接叠加深度感知的场景图（包含节点和有向边），显式编码物体间的空间关系，使多模态语言模型在 VQA 和定位任务中的零样本空间推理准确率最高提升 11 个百分点。

**[HeadHunt-VAD: Hunting Robust Anomaly-Sensitive Heads in MLLM for Tuning-Free Video Anomaly Detection](multimodal_vlm/headhunt-vad_hunting_robust_anomaly-sensitive_heads_in_mllm_.md)**

:   提出HeadHunt-VAD，不用MLLM的文本输出，而是直接从冻结MLLM中"猎取"一小批对异常敏感且跨prompt鲁棒的注意力头，配合轻量逻辑回归scorer，在仅用1%数据、零微调的条件下，在UCF-Crime(87.03% AUC)和XD-Violence(82.63% AP)上达到tuning-free方法SOTA。

**[HiMo-CLIP: Modeling Semantic Hierarchy and Monotonicity in Vision-Language Alignment](multimodal_vlm/himo-clip_modeling_semantic_hierarchy_and_monotonicity_in_vi.md)**

:   提出 HiMo-CLIP，通过对文本嵌入做 batch 内 PCA 分解（HiDe）提取多粒度语义成分，配合双分支单调性感知对比损失（MoLo），在不修改编码器的前提下让 CLIP 学会"文本越完整、对齐分数越高"的语义单调性，在长文本检索上显著超越现有方法。

**[InEx: Hallucination Mitigation via Introspection and Cross-Modal Multi-Agent Collaboration](multimodal_vlm/inex_hallucination_mitigation_via_introspection_and_cross-mo.md)**

:   提出 InEx 框架，通过内部自省推理（TVER 驱动的不确定性感知视觉增强）和外部跨模态多智能体协作（文本自反思 + 图像编辑验证 + 视觉自反思）迭代验证和修正 MLLM 输出，在 POPE 上提升 8.9%，在多个幻觉和通用 benchmark 上持续超越 OPERA/VCD/ICD。

**[LLM-CAS: Dynamic Neuron Perturbation for Real-Time Hallucination Correction](multimodal_vlm/llm-cas_dynamic_neuron_perturbation_for_real-time_hallucinat.md)**

:   LLM-CAS 首次将 LLM 实时幻觉纠正建模为层次强化学习（HRL）问题，训练 RL Agent 在推理时动态选择最优的神经元扰动策略（高层选择功能网络类别，低层选择扰动类型和幅度），结合自适应掩码+因果追踪精确定位目标神经元，在 StoryCloze 上提升 10.98%，超越 ITI/CAA/SADI 等静态/动态基线。

**[Multimodal DeepResearcher: Generating Text-Chart Interleaved Reports From Scratch with Agentic Framework](multimodal_vlm/multimodal_deepresearcher_generating_text-chart_interleaved_.md)**

:   提出 Multimodal DeepResearcher，一个四阶段 Agent 框架从零生成图文交替研究报告：通过形式化可视化描述（FDV）让 LLM 学习和生成多样化图表，结合 Actor-Critic 迭代精炼机制（LLM生成D3.js代码→浏览器渲染→多模态LLM评审），在自建 MultimodalReportBench 上达到 82% 整体胜率（Claude 3.7），人类评估 100% 胜率。

**[RMAdapter: Reconstruction-based Multi-Modal Adapter for Vision-Language Models (Oral)](multimodal_vlm/rmadapter_reconstructionbased_multimodal_adapter_for_visionlanguage.md)**

:   提出 RMAdapter，一种双分支适配器架构：在标准 adapter 的适应分支旁增加重建分支（类 AutoEncoder），通过共享下投影层和逐层本地重建损失，在 CLIP 少样本微调中实现任务特定适应与通用知识保持的最佳平衡，在 Base-to-Novel 泛化、跨数据集和领域泛化三个任务上全面超越 SOTA（含 Prompt-based 方法）。

**[URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding](multimodal_vlm/urag_unified_retrieval_and_generation_in_multimodal_llms_for.md)**

:   URaG 发现 MLLM 处理长文档时存在类人的"粗到细"推理模式（浅层注意力均匀分散、深层集中于证据页），基于此洞察在第 6 层插入轻量跨模态检索模块（仅占参数 0.05%），选取 Top-5 相关页面丢弃其余内容，实现 SOTA 性能的同时减少 44-56% 计算量。

**[VipAct: Visual-Perception Enhancement via Specialized VLM Agent Collaboration and Tool-use](multimodal_vlm/vipact_visual-perception_enhancement_via_specialized_vlm_age.md)**

:   VipAct 提出了一个多Agent协作框架，通过编排器Agent（任务分析+规划+协调）、专用Agent（描述/比较/视觉提示解读）和视觉专家模型（深度估计/目标检测/分割等）三层协作，显著提升 VLM 在细粒度视觉感知任务上的表现，在 Blink 上从 63.74% (zero-shot GPT-4o) 提升到 73.79%。

**[VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models](multimodal_vlm/vp-bench_a_comprehensive_benchmark_for_visual_prompting_in_m.md)**

:   VP-Bench 提出了首个系统评估 MLLM 视觉提示（Visual Prompt）理解能力的两阶段 Benchmark：Stage 1 用 30K+ 图像覆盖 8 种 VP 形状×355 种属性组合评测 VP 感知能力，Stage 2 评测 VP 对 6 个下游任务的实际效果。在 28 个 MLLM 上的评测揭示了 VP 形状选择对性能的关键影响。

---

## 🦾 LLM Agent { #llm_agent }

**[A2Flow: Automating Agentic Workflow Generation via Self-Adaptive Abstraction Operators](llm_agent/a2flow_automating_agentic_workflow_generation_via_self-adaptive_abstraction_oper.md)**

:   提出 A2Flow 框架，通过三阶段流水线（案例生成→功能聚类→深度提取）从专家数据中全自动提取可复用的抽象执行算子，替代人工预定义算子，并引入算子记忆机制累积中间输出辅助节点决策，在 8 个基准上整体超越 AFLOW 等 SOTA，资源消耗降低 37%。

**[A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses](llm_agent/a_multi-agent_conversational_bandit_approach_to_online_evaluation_and_selection_.md)**

:   提出 MACO 多智能体会话式 Bandit 框架，通过本地 agent 的在线淘汰和云服务器的自适应偏好查询机制，实现 LLM 响应的在线评估与用户偏好对齐，达到 $\tilde{O}(\sqrt{dMT})$ 的近优 regret 界。

**[A Multi-Agent LLM Framework for Multi-Domain Low-Resource In-Context NER via Knowledge Retrieval, Disambiguation and Reflective Analysis](llm_agent/a_multi-agent_llm_framework_for_multi-domain_low-resource_in-context_ner_via_kno.md)**

:   提出 KDR-Agent 多智能体 LLM 框架，通过知识检索、实体消歧和反思分析三个专用 agent 改善多领域低资源 ICL NER 性能，在 10 个数据集上显著优于现有 ICL 基线。

**[A Multi-Agent LLM Framework for Multi-Domain Low-Resource In-Context NER via Knowledge Retrieval, Disambiguation and Reflective Analysis](llm_agent/a_multi-agent_llm_framework_for_multi-domain_low-resource_in.md)**

:   提出 KDR-Agent 多智能体框架，通过知识检索（Wikipedia）、歧义消解和反思式自我纠错三个专业智能体协同工作，在仅使用少量静态标注示例的条件下，在5个领域10个NER数据集上显著超越现有零样本和少样本ICL NER方法。

**[AgentSense: Virtual Sensor Data Generation Using LLM Agents in Simulated Home Environments](llm_agent/agentsense_virtual_sensor_data_generation_using_llm_agents_i.md)**

:   利用LLM驱动的具身智能体在模拟智能家居中"生活"，生成虚拟环境传感器数据用于预训练HAR模型，在低资源场景下显著提升活动识别性能。

**[AquaSentinel: Next-Generation AI System Integrating Sensor Networks for Urban Underground Water Pipeline Anomaly Detection via Collaborative MoE-LLM Agent Architecture](llm_agent/aquasentinel_next-generation_ai_system_integrating_sensor_ne.md)**

:   提出AquaSentinel，一个物理信息驱动的AI系统，通过稀疏传感器部署+物理增强虚拟传感器+MoE时空GNN集成+双阈值RTCA检测算法+因果流定位+LLM报告生成，仅用20-30%节点覆盖即可实现全网管道泄漏检测，在110个泄漏场景中达到100%检测率。

**[ARCANE: A Multi-Agent Framework for Interpretable and Configurable Alignment](llm_agent/arcane_a_multi-agent_framework_for_interpretable_and_configurable_alignment.md)**

:   提出ARCANE框架，将对齐建模为多智能体协作问题——manager agent通过与stakeholder对话学习生成自然语言rubric（加权可验证准则集），作为worker agent的可解释代理奖励函数，通过SFT+GSPO两阶段训练实现测试时可配置的对齐，在GDPVal基准上GSPO版本的mean return从0.58提升至0.74（N=8）。

**[AutoTool: Efficient Tool Selection for Large Language Model Agents](llm_agent/autotool_efficient_tool_selection_for_large_language_model_a.md)**

:   提出AutoTool，一个基于图的免训练工具选择框架，通过发现并利用"工具使用惯性"（tool usage inertia）——即工具调用遵循可预测的顺序模式这一经验现象——构建工具惯性图（TIG），用统计方法代替部分LLM推理来高效选择工具和填充参数，在保持任务完成率的同时减少15-40%的推理成本。

**[AutoTool: Efficient Tool Selection for Large Language Model Agents](llm_agent/autotool_efficient_tool_selection_for_large_language_model_agents.md)**

:   提出 AutoTool，一种基于图的工具选择框架，利用工具使用惯性（tool usage inertia）构建工具惯性图（TIG），通过统计结构绕过重复的 LLM 推理来选择工具和填充参数，在保持任务完成率的同时减少最多 30% 的推理开销。

**[BayesAgent: Bayesian Agentic Reasoning Under Uncertainty via Verbalized Probabilistic Graphical Modeling](llm_agent/bayesagent_bayesian_agentic_reasoning_under_uncertainty_via_.md)**

:   提出 vPGM 框架，通过自然语言引导 LLM Agent 模拟概率图模型（PGM）的贝叶斯推理过程，发现隐变量并推断后验分布，再用 Dirichlet 先验做数值贝叶斯校准（BayesVPGM），在多个推理任务上同时提升准确率和置信度校准。

**[Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning](llm_agent/beyond_react_a_planner-centric_framework_for_complex_tool-au.md)**

:   提出以Planner为核心的Plan-Execute框架，将复杂查询转化为DAG执行计划，通过SFT+GRPO两阶段训练专门的Planner模型，在ComplexTool-Plan和StableToolBench上超越ReAct等反应式方法，用更少推理步骤实现更高成功率。

**[Co-EPG: A Framework for Co-Evolution of Planning and Grounding in Autonomous GUI Agents](llm_agent/co-epg_a_framework_for_co-evolution_of_planning_and_groundin.md)**

:   提出Co-EPG框架，将GUI Agent解耦为Planning和Grounding两个模型，通过GRPO协同训练和基于置信度的动态奖励集成机制（C-DREM）建立正反馈循环，使两个模型自迭代协同进化，仅用基准数据集（无需外部数据）即在Multimodal-Mind2Web（58.4%）和AndroidControl（83.1%）上达到SOTA。

**[Cook and Clean Together: Teaching Embodied Agents for Parallel Task Execution](llm_agent/cook_and_clean_together_teaching_embodied_agents_for_paralle.md)**

:   提出ORS3D任务——将运筹学(OR)知识引入具身AI的任务调度，要求智能体利用可并行子任务的等待时间执行其他任务以最小化总完成时间，同时在3D场景中定位目标物体；构建60K级数据集ORS3D-60K，并提出GRANT模型通过调度token机制连接外部动态规划求解器，在时间效率上比baseline提升30.53%。

**[COVR: Collaborative Optimization of VLMs and RL Agent for Visual-Based Control](llm_agent/covrcollaborative_optimization_of_vlms_and_rl_agent_for_visu.md)**

:   提出 VLM 与 RL 双向协同优化框架 COVR：RL 生成的高质量交互数据用于微调 VLM，增强后的 VLM 反过来通过 action prior 指导 RL 策略学习，在 CARLA 和 DMControl 上取得 SOTA。

**[D-GARA: A Dynamic Benchmarking Framework for GUI Agent Robustness in Real-World Anomalies](llm_agent/d-gara_a_dynamic_benchmarking_framework_for_gui_agent_robust.md)**

:   提出 D-GARA，一个面向 Android GUI Agent 的动态鲁棒性评估框架，通过在实时交互过程中注入权限弹窗、电量警告、应用崩溃等真实世界异常，揭示现有 SOTA Agent（包括 UI-TARS-72B、GPT-4o）在中断场景下平均成功率下降超过 17.5%，最高达 33% 的严重脆弱性。

**[DEPO: Dual-Efficiency Preference Optimization for LLM Agents](llm_agent/depo_dual-efficiency_preference_optimization_for_llm_agents.md)**

:   提出双重效率（dual-efficiency）的概念，将 LLM Agent 的效率分解为 step 级（减少每步 token 数）和 trajectory 级（减少总步数），并基于 KTO 设计了 DEPO 方法，通过在 desirable 样本的 reward 中加入效率 bonus 来联合优化效率与性能。

**[EcoAgent: An Efficient Device-Cloud Collaborative Multi-Agent Framework for Mobile Automation](llm_agent/ecoagent_an_efficient_device-cloud_collaborative_multi-agent.md)**

:   提出 EcoAgent，一个闭环设备-云端协作的多 Agent 移动自动化框架，通过 Dual-ReACT 双层推理规划 + 设备端轻量验证反馈 + Pre-Understanding 文本压缩模块，在 AndroidWorld 上达到与全云端 Agent 相当的成功率，同时大幅降低延迟（3.9s vs 15.3s）、云端调用（降89%）和上行数据量（降48.6倍）。

**[Extracting Events Like Code: A Multi-Agent Programming Framework for Zero-Shot Event Extraction](llm_agent/extracting_events_like_code_a_multi-agent_programming_framework_for_zero-shot_ev.md)**

:   提出 Agent-Event-Coder (AEC)，将零样本事件抽取类比为软件工程流程，用4个专职Agent（Retrieval→Planning→Coding→Verification）协作完成抽取，并将事件schema编码为可执行Python类实现编译器式确定性验证与双循环迭代修正，在5个领域、6个LLM上全面超越零样本基线。

**[Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-checking System](llm_agent/fact2fiction_targeted_poisoning_attack_to_agentic_fact-check.md)**

:   提出 Fact2Fiction，首个针对 Agent 化事实核查系统（如 DEFAME、InFact）的投毒攻击框架：通过 Planner Agent 模拟声明分解生成子问题，利用系统的 justification 反向工程关键推理点来制作定向恶意证据，并按重要性分配投毒预算，在仅 1% 投毒率下比 SOTA PoisonedRAG 高 8.9%-21.2% 的攻击成功率。

**[History-Aware Reasoning for GUI Agents](llm_agent/history-aware_reasoning_for_gui_agents.md)**

:   提出 HAR 框架，通过构建反思学习场景、合成纠错指南、设计混合 RL 奖励函数（含 Memory-Augmented Reward），将 GUI Agent 的推理模式从"历史无感知"转变为"历史感知"，3B 模型在 AITW/Mind2Web/GUI-Odyssey 等多个 benchmark 上超越更大模型。

**[iMAD: Intelligent Multi-Agent Debate for Efficient and Accurate LLM Inference](llm_agent/imad_intelligent_multi-agent_debate_for_efficient_and_accura.md)**

:   iMAD 提出选择性触发多Agent辩论的框架：先让单Agent生成带自我批判的结构化响应，从中提取 41 个可解释的语言/语义特征，用轻量 MLP 分类器（FocusCal 损失训练）判断是否需要触发 MAD，在 6 个 QA/VQA 数据集上减少高达 92% 的 Token 开销，同时提升准确率高达 13.5%。

**[MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models](llm_agent/medla_a_logic-driven_multi-agent_framework_for_complex_medic.md)**

:   提出 MedLA，首个基于三段论逻辑树的医学多 Agent 推理框架：每个 Agent 将推理组织为显式的逻辑树（大前提-小前提-结论三段论节点），多个 Agent 通过图引导的多轮讨论在前提级别对齐和修正逻辑树，在 MedDDx 上超越所有基线 7.4%（8B 模型），在医学 QA 上以 8B 模型达到 69.9% 平均准确率（超 70B RAG 模型）。

**[ProBench: Benchmarking GUI Agents with Accurate Process Information](llm_agent/probench_benchmarking_gui_agents_with_accurate_process_infor.md)**

:   提出 ProBench，首个同时评估"最终状态"和"操作过程"的移动端 GUI Agent benchmark：200+ 挑战性任务覆盖 34 个中英文主流 App，通过 Process Provider（Structure Description Converter + MLLM Summarizer）自动捕获精确的中间过程信息，评估发现最强模型 Gemini 2.5 Pro 也仅完成 40.1% 任务，暴露了 grounding 不足、历史操作感知差、任务规划过于简化三大普遍问题。

**[Prune4Web: DOM Tree Pruning Programming for Web Agent](llm_agent/prune4web_dom_tree_pruning_programming_for_web_agent.md)**

:   提出 Prune4Web，通过"LLM 生成评分函数参数 + 固定启发式模板执行"的编程式 DOM 剪枝方法实现 25-50 倍候选元素缩减：三阶段 pipeline（Planner 分解子任务 → Programmatic Filter 生成评分函数剪枝 DOM → Grounder 执行操作），3B 模型在 Multimodal-Mind2Web 上达到 52.4% Step SR（超越所有同参数量基线甚至部分 9.6B/32B 模型），低级 grounding 准确率从 46.8% 提升至 88.28%。

**[SoMe: A Realistic Benchmark for LLM-based Social Media Agents](llm_agent/some_a_realistic_benchmark_for_llm-based_social_media_agents.md)**

:   提出 SoMe，首个全面评估 LLM 社交媒体 Agent 的 benchmark：8 个任务覆盖帖子分析、用户理解、综合推理，917 万帖子 + 6591 用户 + 1.7 万标注查询，配套 8 个 MCP 兼容工具，评估 13 个主流 LLM 发现最强模型仅 54.33 分（满分 100），揭示了推理能力≠Agent能力、工具调用幻觉普遍存在等关键发现。

**[TongUI: Internet-Scale Trajectories from Multimodal Web Tutorials for Generalized GUI Agents](llm_agent/tongui_internet-scale_trajectories_from_multimodal_web_tutor.md)**

:   TongUI 提出从互联网上的多模态教程（视频+图文）自动转化为 GUI 操作轨迹数据的框架，构建了百万级的 GUI-Net-1M 数据集，用于微调 Qwen2.5-VL 模型，在多个 grounding 和 navigation 基准上超越或接近 UI-TARS 等 SOTA。

---

## 💡 LLM 推理 { #llm_reasoning }

**[A Reasoning Paradigm for Named Entity Recognition](llm_reasoning/a_reasoning_paradigm_for_named_entity_recognition.md)**

:   提出 ReasoningNER，将命名实体识别从"隐式模式匹配"转变为"显式推理"范式，通过三阶段流程（CoT数据构建→CoT微调→GRPO强化增强）让模型先推理再抽取实体，在零样本设定下F1超GPT-4达12.3个百分点，8B模型在CrossNER上达72.4平均F1。

**[Answering the Unanswerable Is to Err Knowingly: Analyzing and Mitigating Abstention Failures in Large Reasoning Models](llm_reasoning/answering_the_unanswerable_is_to_err_knowingly_analyzing_and.md)**

:   系统分析大推理模型(LRM)面对不可回答数学题时的弃权失败现象，发现LRM内部有足够认知能力识别问题不可解（探针分类准确率>80%）但外部行为仍偏向强答，提出认知监控+推理时干预的两阶段方法，将弃权率从16-54%提升至60-92%且不损害可回答题的推理性能。

**[ARCHE: A Novel Task to Evaluate LLMs on Latent Reasoning Chain Extraction](llm_reasoning/arche_a_novel_task_to_evaluate_llms_on_latent_reasoning_chai.md)**

:   提出潜在推理链提取 (ARCHE) 任务，要求 LLM 将科学论文中的论证分解为基于 Peirce 三种推理范式的推理逻辑树 (RLT)，并通过 Entity Coverage 和 Reasoning Edge Accuracy 两个指标揭示了 10 个主流 LLM 在内容完整性与逻辑正确性之间的本质权衡。

**[BLM-Guard: Explainable Multimodal Ad Moderation with Chain-of-Thought and Policy-Aligned Rewards](llm_reasoning/blm-guard_explainable_multimodal_ad_moderation_with_chain-of.md)**

:   提出 BLM-Guard，一个面向短视频商业广告的可解释多模态审核框架：先通过 Rule-driven ICoT 数据合成 + SFT 冷启动建立结构化推理能力，再用 Self-Adaptive GRPO 强化学习（结合规则正确性奖励 + 自适应一致性奖励 SCA-R）优化策略对齐，在真实广告 benchmark 上达到 91.4% 严格准确率和 0.845 推理一致性分数。

**[CMMCoT: Enhancing Complex Multi-Image Comprehension via Multi-Modal Chain-of-Thought and Memory Augmentation](llm_reasoning/cmmcot_enhancing_complex_multi-image_comprehension_via_multi.md)**

:   提出 CMMCoT 框架，通过构建交错的多模态多步推理链（含视觉区域 token 监督）和测试时检索式记忆增强模块（RIFREM），在不增加参数的前提下提升多图场景下的慢思考推理能力，基于 Qwen2.5-VL-7B 在多图基准上平均提升 1.4 分。

**[Deep Hidden Cognition Facilitates Reliable Chain-of-Thought Reasoning](llm_reasoning/deep_hidden_cognition_facilitates_reliable_chain-of-thought_.md)**

:   本文发现 LLM 在 CoT 推理过程中，中间层的注意力头激活值隐式编码了推理步骤的真实性信息（最高 85% 探测准确率），据此训练置信度预测器引导 Beam Search 动态选择高置信度推理路径，在数学/符号/常识推理任务上超越 Self-Consistency 和 PRM Guided Search。

**[Improving Value-based Process Verifier via Low-Cost Variance Reduction](llm_reasoning/improving_value-based_process_verifier_via_low-cost_variance_reduction.md)**

:   针对基于值的过程验证器(PRM)训练中蒙特卡罗(MC)估计因采样数有限导致的高方差问题，提出Compound Monte Carlo Sampling (ComMCS)方法，通过线性组合当前步和后续步的MC估计量来无偏地降低方差，无需额外LLM推理开销，在MATH-500上Best-of-32实验中提升2.2个点。

**[Incorporating Self-Rewriting into Large Language Model Reasoning Reinforcement](llm_reasoning/incorporating_self-rewriting_into_large_language_model_reasoning_reinforcement.md)**

:   提出Self-Rewriting框架，让LRM在RL训练中对"简单"样本（全部回答正确的query）重写自身推理文本并从中学习，仅增加约10%训练开销即可在保持准确率的同时将推理长度减少46%，内部推理质量（LLM-as-Judge）提升7.2分，有效缓解过度思考、冗余思考等问题。

**[Jupiter: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search](llm_reasoning/jupiter_enhancing_llm_data_analysis_capabilities_via_notebook_and_inference-time.md)**

:   构建NbQA数据集（从真实Jupyter Notebook提取3.8万task-solution对）+ 提出Jupiter框架（将数据分析建模为状态级搜索问题，用值模型引导PUCT搜索），使Qwen2.5-14B在InfiAgent-DABench上达86.38%超越GPT-4o(85.99%)，Qwen2.5-7B在DSBench上从63.51%提升至89.19%。

**[L2V-CoT: Cross-Modal Transfer of Chain-of-Thought Reasoning via Latent Intervention](llm_reasoning/l2v-cot_cross-modal_transfer_of_chain-of-thought_reasoning_v.md)**

:   通过 LAT 分析发现 LLM 和 VLM 的低频 CoT 方向表示具有相似分布，提出 L2V-CoT：从 LLM 提取 CoT 方向表示 → 低通滤波 → 频域重采样匹配维度 → 注入 VLM 隐藏层，training-free 地将 LLM 的推理能力迁移到 VLM，平均提升 3.7%，最高 8.6%。

**[SAPO: Self-Adaptive Process Optimization Makes Small Reasoners Stronger](llm_reasoning/sapo_self-adaptive_process_optimization_makes_small_reasoners_stronger.md)**

:   受神经科学中Error-Related Negativity启发，提出自适应过程优化方法SAPO，通过首错检测+局部后验估计替代低效的逐步蒙特卡洛rollout，在降低2-3倍计算成本的同时实现推理器-验证器协同优化，使小语言模型（≤2B）在数学和代码推理任务上超越多数自演化方法。

**[SCALE: Selective Resource Allocation for Overcoming Performance Bottlenecks in Mathematical Test-time Scaling](llm_reasoning/scale_selective_resource_allocation_for_overcoming_performance_bottlenecks_in_ma.md)**

:   基于认知科学的双过程理论，提出SCALE框架将数学问题分解为子问题后按难度分配不同计算资源（System 1快速计算 vs System 2深度推理），在AIME25上将Qwen3-32B从57.50%提升至71.25%，同时比InftyThink节省33-53%的token。

**[Stable Voting and the Splitting of Cycles](llm_reasoning/stable_voting_and_the_splitting_of_cycles.md)**

:   研究Simple Stable Voting (SSV)——已在数百次实际选举中使用的递归投票规则——是否总是精化(refine)Split Cycle (SC)方法的猜想，通过数学证明（≤5候选人）和SAT求解（6-7候选人）确定：猜想在≤6候选人时成立，≥7候选人时被反驳，并通过构造性证明推广到任意多候选人。

**[ToC: Tree-of-Claims Search with Multi-Agent Language Models](llm_reasoning/toc_tree-of-claims_search_with_multi-agent_language_models.md)**

:   提出 Tree-of-Claims (ToC) 框架，将专利权利要求编辑建模为结构化搜索问题，通过 MCTS 与 EditorAgent/ExaminerAgent 多智能体协作，在新颖性、范围保持和语义一致性之间联合优化，比零/少样本 LLM 基线平均提升约 8% 综合分。

---

## 🎨 图像生成 { #image_generation }

**[AbductiveMLLM: Boosting Visual Abductive Reasoning Within MLLMs](image_generation/abductivemllm_boosting_visual_abductive_reasoning_within_mll.md)**

:   模仿人类的"语言溯因+图像想象"双模式认知，提出AbductiveMLLM，通过Reasoner(因果感知假设生成+筛选)和Imaginer(扩散模型引导的图像想象)两个组件端到端联合训练，在VAR和YouCookII两个benchmark上显著超越传统方法和通用MLLM，设置新的SOTA。

**[AEDR: Training-Free AI-Generated Image Attribution via Autoencoder Double-Reconstruction](image_generation/aedr_training-free_ai-generated_image_attribution_via_autoen.md)**

:   提出一种基于自编码器双重重建损失比值的免训练图像归因方法，通过图像均匀度校准消除纹理复杂度偏差，在8个主流扩散模型上平均准确率达95.1%，比最强基线高24.7%，且速度快约100倍。

**[Aggregating Diverse Cue Experts for AI-Generated Image Detection](image_generation/aggregating_diverse_cue_experts_for_ai-generated_image_detec.md)**

:   提出Multi-Cue Aggregation Network (MCAN)，通过混合编码器适配器(MoEA)将原始图像、高频信息和新提出的色度不一致性(CI)三种互补线索统一融合，实现跨生成模型的鲁棒AI生成图像检测。

**[Annealed Relaxation of Speculative Decoding for Faster Autoregressive Image Generation](image_generation/annealed_relaxation_of_speculative_decoding_for_faster_autor.md)**

:   提出Cool-SD，一种有理论支撑的退火松弛speculative decoding框架：通过推导TV距离上界得到最优重采样分布，并证明接受概率递减调度比均匀调度产生更小的分布偏移，在LlamaGen和Lumina-mGPT上实现了比LANTERN++更优的速度-质量权衡。

**[AnoStyler: Text-Driven Localized Anomaly Generation via Lightweight Style Transfer](image_generation/anostyler_text-driven_localized_anomaly_generation_via_light.md)**

:   将零样本异常生成建模为文本引导的局部风格迁移问题，通过轻量级U-Net + CLIP损失将正常图像的掩码区域风格化为语义对齐的异常图像，在MVTec-AD和VisA上以263M参数（仅0.61M可训练）超越扩散模型基线，同时显著提升下游异常检测性能。

**[Conditional Diffusion Model for Multi-Agent Dynamic Task Decomposition](image_generation/conditional_diffusion_model_for_multi-agent_dynamic_task_dec.md)**

:   提出 CD3T，一个两层层次化 MARL 框架：用条件扩散模型学习动作语义表示（以观测和他人动作为条件，预测下一观测和奖励），通过 k-means 聚类得到子任务划分，高层选择子任务、低层在受限动作空间执行策略，在 SMAC 的 Super Hard 场景上显著超越所有基线。

**[DICE: Distilling Classifier-Free Guidance into Text Embeddings](image_generation/dice_distilling_classifier-free_guidance_into_text_embedding.md)**

:   提出 DICE，训练一个仅 2M 参数的轻量 sharpener 将 CFG 的引导效果蒸馏进 text embedding，使无引导采样达到与 CFG 同等的生成质量、推理计算量减半，在 SD1.5 多个变体、SDXL 和 PixArt-α 上全面验证有效，是 AAAI 2026 口头报告论文。

**[DiffBench Meets DiffAgent: End-to-End LLM-Driven Diffusion Acceleration Code Generation](image_generation/diffbench_meets_diffagent_end-to-end_llm-driven_diffusion_ac.md)**

:   提出DiffBench（604个扩散模型加速任务的评估基准，分5个难度等级）和DiffAgent（集成规划-编码-调试三Agent + 遗传算法选择器的闭环框架），在Claude Sonnet 4上将扩散加速代码生成通过率从54.30%提升到81.59%，复杂优化任务达成率68.27%。

**[Difficulty Controlled Diffusion Model for Synthesizing Effective Training Data](image_generation/difficulty_controlled_diffusion_model_for_synthesizing_effec.md)**

:   在Stable Diffusion中引入难度编码器（MLP，输入类别+难度分数），通过LoRA微调解耦"域对齐"和"难度控制"两个目标，使生成数据的学习难度可控——仅用10%额外合成数据即超过Real-Fake的最佳结果，节省63.4 GPU小时。

**[DOS: Directional Object Separation in Text Embeddings for Multi-Object Image Generation](image_generation/dos_directional_object_separation_in_text_embeddings_for_mul.md)**

:   识别出多物体生成失败的四种场景（相似形状/纹理、不同背景偏好、多物体），通过构建方向性分离向量修改CLIP的三类文本嵌入（语义token/EOT/pooled），在SDXL上将成功率提升16-25%并将融合率降低3-12%，推理速度接近baseline（约4×快于Attend-and-Excite）。

**[HACK: Head-Aware KV Cache Compression for Efficient Visual Autoregressive Modeling](image_generation/head-aware_kv_cache_compression_for_efficient_visual_autoreg.md)**

:   发现VAR模型中attention head天然分为Contextual Heads（语义一致性，垂直注意力模式）和Structural Heads（空间连贯性，多对角线模式），提出HACK框架通过非对称预算分配和模式特定压缩策略，在70%压缩率下实现无损生成质量，Infinity-8B上1.75×显存减少和1.57×加速。

**[Infinite-Story: A Training-Free Consistent Text-to-Image Generation](image_generation/infinite-story_a_training-free_consistent_text-to-image_gene.md)**

:   基于 scale-wise 自回归模型（Infinity），通过三个 training-free 技术——Identity Prompt Replacement（消除文本编码器的上下文偏差）、Adaptive Style Injection（参考图像特征注入）和 Synchronized Guidance Adaptation（同步 CFG 两个分支），实现了身份与风格一致的多图像生成，速度比扩散模型快 6 倍（1.72 秒/张）。

**[Laytrol: Preserving Pretrained Knowledge in Layout Control for Multimodal Diffusion Transformers](image_generation/laytrol_preserving_pretrained_knowledge_in_layout_control_fo.md)**

:   通过从 MM-DiT 复制参数初始化布局控制网络、设计专用初始化方案（布局编码器初始化为纯文本编码器 + 输出零初始化）、并用 FLUX 自己生成的图像构建 LaySyn 数据集来缓解分布偏移，实现了在 FLUX 上高质量的布局到图像生成。

---

## 🚗 自动驾驶 { #autonomous_driving }

**[Beta Distribution Learning for Reliable Roadway Crash Risk Assessment](autonomous_driving/beta_distribution_learning_for_reliable_roadway_crash_risk_a.md)**

:   提出基于 Beta 分布学习的地理空间深度学习框架，利用多尺度卫星图像预测道路致命事故风险的完整概率分布（而非点估计），在 Recall 上提升 17-23%，并通过分布形状自然表达不确定性。

**[Bridging Day and Night: Target-Class Hallucination Suppression in Unpaired Image Translation](autonomous_driving/bridging_day_and_night_target-class_hallucination_suppressio.md)**

:   首次系统性解决无配对日→夜图像翻译中的"目标类幻觉"问题，通过双头判别器（风格头+SAM2伪标签分割头）检测幻觉 + 类原型对比学习抑制幻觉，在BDD100K日夜域适应检测上将mAP从15.08提升到17.40（+15.5%），交通灯AP提升31.7%。

**[CompTrack: 信息瓶颈引导的低秩动态Token压缩用于点云跟踪 (Oral)](autonomous_driving/comptrack_information_bottleneckguided_lowrank_dynamic_token_compres.md)**

:   针对LiDAR点云3D单目标跟踪中的"双重冗余"问题（空间冗余：大量背景噪声；信息冗余：前景中大量不具区分性的平面点），提出SFP前景预测器+IB-DTC信息瓶颈引导动态Token压缩两个模块，在KITTI/nuScenes/Waymo上达到SOTA，90 FPS实时运行（比P2P快1.4倍）。

**[FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning](autonomous_driving/fastdrivevla_efficient_end-to-end_driving_via_plug-and-play_.md)**

:   提出 FastDriveVLA，通过 MAE 风格的前景像素重建训练轻量级 plug-and-play 的 ReconPruner 模块（仅 0.07B），利用对抗前景-背景重建策略优先保留驾驶决策所需的前景 token，在 nuScenes 开环规划基准上各剪枝率均达 SOTA，一次训练可迁移至同一视觉编码器的不同 VLA 模型。

**[FQ-PETR: Fully Quantized Position Embedding Transformation for Multi-View 3D Object Detection](autonomous_driving/fq-petr_fully_quantized_position_embedding_transformation_fo.md)**

:   首次实现PETR系列3D检测器的全INT8量化部署，通过量化友好的LiDAR-ray位置编码(QFPE)解决多模态特征幅度不匹配问题、双查找表(DULUT)高效逼近非线性算子、数值稳定后量化(QANS)避免softmax注意力失真，在PETR/StreamPETR/PETRv2/MV2D上W8A8精度损失<1%且延迟降低75%（3.9×加速）。

**[MambaSeg: Harnessing Mamba for Accurate and Efficient Image-Event Semantic Segmentation](autonomous_driving/mambaseg_harnessing_mamba_for_accurate_and_efficient_image-e.md)**

:   提出 MambaSeg，用双分支并行 Mamba 编码器分别处理 RGB 图像和事件流，通过空间-时间双维度交互模块 (DDIM) 实现细粒度跨模态融合，在 DDD17 和 DSEC 数据集上以 25.44M 参数取得 77.56%/75.10% mIoU 的 SOTA，效率远优于 Transformer 方案。

**[SPARC: 用单一策略驾驶100辆未见车辆的OOD泛化](autonomous_driving/out-of-distribution_generalization_with_a_sparc_racing_100_u.md)**

:   提出 SPARC（Single-Phase Adaptation for Robust Control），将 RMA 的两阶段上下文编码与历史适应统一为单阶段训练，在 Gran Turismo 7 高保真赛车模拟器中用单一策略驾驶100+未见车辆实现SOTA OOD泛化性能。

**[PriorDrive: 用统一向量先验增强在线HD地图构建](autonomous_driving/priordrive_enhancing_online_hd_mapping_with_unified_vector_p.md)**

:   提出 PriorDrive 框架，通过 Unified Vector Encoder (UVE) 和 Hybrid Prior Representation (HPQuery) 将多种向量化先验地图（SD地图、旧HD地图、历史预测地图）统一编码并集成到各种在线建图模型中，在 nuScenes 上 mAP 提升 14.3，兼容 query-based 和 non-query-based 两类建图架构。

**[ReflexDiffusion: 反思增强的高侧向加速度自动驾驶轨迹规划](autonomous_driving/reflexdiffusion_reflection-enhanced_trajectory_planning_for_.md)**

:   提出 ReflexDiffusion，在扩散模型推理阶段引入物理感知的反思机制，通过梯度注入强化曲率-速度-加速度耦合约束（a_y = κv²），在 nuPlan 高侧向加速度长尾场景中驾驶分数提升 14.1%，架构无关可直接部署到现有扩散规划器。

**[Task Prototype-Based Knowledge Retrieval for Multi-Task Learning from Partially Annotated Data](autonomous_driving/task_prototype-based_knowledge_retrieval_for_multi-task_lear.md)**

:   提出基于任务原型的知识检索框架，通过可学习 Task Prototype 嵌入任务特性并量化任务关联、Knowledge Retrieval Transformer 基于 task-affinity score 自适应精炼特征表示，在部分标注多任务学习（MTPSL）中避免依赖未标注任务的预测，PASCAL-Context 和 NYUD-v2 上全面超越 SOTA。

**[VILTA: A VLM-in-the-Loop Adversary for Enhancing Driving Policy Robustness](autonomous_driving/vilta_a_vlm-in-the-loop_adversary_for_enhancing_driving_poli.md)**

:   VILTA 将 VLM（Gemini-2.5-Flash）直接嵌入自动驾驶 RL 训练循环中，通过"Vision-Language-Editing"（VLE）范式让 VLM 编辑周围车辆的未来轨迹来生成具有挑战性的危险场景，训练出的驾驶策略在 CARLA 挑战场景中路线完成率提升 13.3%、碰撞率降低 28.5%。

**[Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction (Oral)](autonomous_driving/visiononly_gaussian_splatting_for_collaborative_semantic_occupancy_p.md)**

:   首次将 3D 高斯 Splatting 作为多智能体协同感知的通信媒介和中间表征，利用高斯基元的刚体变换可解析性和稀疏性，通过高斯打包（ROI 裁剪+刚体变换）和跨智能体邻域融合模块，实现了高效且可解释的视觉协同语义占用预测。

---

## 🤖 机器人/具身智能 { #robotics }

**[A Computable Game-Theoretic Framework for Multi-Agent Theory of Mind](robotics/a_computable_game-theoretic_framework_for_multi-agent_theory_of_mind.md)**

:   提出基于 Poisson 认知层次（cognitive hierarchy）的博弈论框架，通过 Gamma-Poisson 共轭贝叶斯更新实现可计算的多智能体 Theory of Mind，在避免 POMDP 不可判定性的同时支持递归式有限理性决策与在线信念修正。

**[Adaptive Theory of Mind for LLM-based Multi-Agent Coordination](robotics/adaptive_theory_of_mind_for_llm-based_multi-agent_coordination.md)**

:   提出自适应心智理论智能体(A-ToM)，将ToM阶数对齐建模为在线专家建议问题，通过FTL或Hedge算法实时估计伙伴的ToM阶数并动态调整自身推理深度，在重复矩阵博弈、网格导航和Overcooked等4类任务上实现鲁棒的零样本多智能体协作。

**[Affordance-Guided Coarse-to-Fine Exploration for Base Placement in Open-Vocabulary Mobile Manipulation](robotics/affordance-guided_coarse-to-fine_exploration_for_base_placem.md)**

:   针对开放词汇移动操控中机器人基座选位问题，提出一种零样本框架，通过构建跨模态表征（Affordance RGB + Obstacle Map+）将语义affordance线索投射到障碍物地图上，再用粗到细迭代优化平衡语义和几何约束，在5个操控任务上达到85%成功率，大幅超越几何规划器和纯VLM方法。

**[Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning](robotics/attention_as_binding_a_vector-symbolic_perspective_on_transformer_reasoning.md)**

:   本文提出将Transformer自注意力机制重新解释为向量符号架构(VSA)中的软绑定/解绑定算子——Query/Key定义角色空间、Value编码填充项、注意力权重实现可微解绑定、残差连接实现叠加——从而以代数视角统一解释LLM在符号推理中的能力与脆弱性，并提出显式绑定头、超维记忆层等VSA启发的架构改进方向。

**[Causal Inference Under Threshold Manipulation: Bayesian Mixture Modeling and Heterogeneous Treatment Effects](robotics/causal_inference_under_threshold_manipulation_bayesian_mixtu.md)**

:   提出 BMTM/HBMTM 贝叶斯混合模型框架，在消费者策略性操纵消费额以达到奖励阈值的场景下，通过将观测分布拆解为 bunching 与 non-bunching 两个子分布，准确估计阈值因果效应及跨子群的异质性处理效应。

**[EvoEmpirBench: Dynamic Spatial Reasoning with Agent-ExpVer](robotics/evoempirbench_dynamic_spatial_reasoning_with_agent-expver.md)**

:   提出 EvoEmpirBench（EEB），包含两个动态交互式 benchmark（局部可观测迷宫导航 + 消消乐），以及 Agent-ExpVer 三智能体在线学习框架（GeoLink 交互 + InsightForce 经验抽象 + TruthWeaver 知识管理），通过"经验→验证→真理归纳"的认知循环实现无参数更新的持续策略进化，使 GPT-4.1 成功率提升 5.6%、Qwen-32B 提升 29%。

**[iSeal: Encrypted Fingerprinting for Reliable LLM Ownership Verification](robotics/iseal_encrypted_fingerprinting_for_reliable_llm_ownership_verification.md)**

:   提出 iSeal——首个在模型窃取者完全控制推理过程的黑盒场景下仍能可靠验证 LLM 所有权的主动指纹方法，通过外部加密编码器 + RSC 纠错 + 相似度匹配三重机制，在 12 个 LLM、10+ 种攻击下均保持 100% 指纹成功率（FSR），而已有方法降至 0%。

**[Neural Graph Navigation for Intelligent Subgraph Matching](robotics/neural_graph_navigation_for_intelligent_subgraph_matching.md)**

:   提出 NeuGN（Neural Graph Navigation）框架，首次将生成式神经导航集成到子图匹配的核心枚举阶段，通过 QSExtractor 提取查询图结构信号 + GGNavigator 将暴力枚举转为结构感知的候选节点优先排序，在保证完备性的同时将 First Match Steps 最高减少 98.2%。

**[Robust Out-of-Order Retrieval for Grid-Based Storage at Maximum Capacity](robotics/robust_out-of-order_retrieval_for_grid-based_storage_at_maximum_capacity.md)**

:   针对满载 2D 网格存储系统中检索顺序不确定的问题，提出 k-bounded perturbation 不确定性模型，证明 Θ(k) 列宽是零重定位的充要条件，并给出高效鲁棒存储求解器与贪心检索策略，当 k ≤ 0.5c 时几乎消除重定位，k 到达 c 时仍减少 50%+ 重定位。

**[Shadows in the Code: Exploring the Risks and Defenses of LLM-based Multi-Agent Software Development Systems](robotics/shadows_in_the_code_exploring_the_risks_and_defenses_of_llm-.md)**

:   首次系统分析 LLM 多 Agent 软件开发系统（ChatDev/MetaGPT/AgentVerse）的安全风险：提出 IMBIA 攻击框架覆盖两种威胁场景（恶意用户+良性Agent / 良性用户+恶意Agent）和 12 种恶意行为（5 大恶意软件家族），攻击成功率高达 93%（ChatDev），并设计 Adv-IMBIA 对抗性防御将 ASR 降低 40-73%。

**[Towards Reinforcement Learning from Neural Feedback: Mapping fNIRS Signals to Agent Performance](robotics/towards_reinforcement_learning_from_neural_feedback_mapping_.md)**

:   提出 NEURO-LOOP 框架，利用 fNIRS（功能性近红外光谱）脑信号作为隐式神经反馈评估 RL agent 表现，发布 25 名被试 × 3 领域 × 6 条件的 fNIRS 数据集，分类 F1 达 67%（二分类）/ 46%（多分类），跨被试 fine-tuning 分别提升 17% 和 41%，奠定 Reinforcement Learning from Neural Feedback (RLNF) 基础。

**[Unintended Misalignment from Agentic Fine-Tuning: Risks and Mitigation](robotics/unintended_misalignment_from_agentic_fine-tuning_risks_and_m.md)**

:   本文揭示了在良性 Agent 数据上微调 LLM 会导致意外的安全对齐偏移（攻击成功率增加 32-38%），并提出 PING（Prefix Injection Guard）——通过迭代生成+评估自然语言前缀来引导微调后的 Agent 拒绝有害请求，平均提升拒绝率 66%（Web）和 44%（代码），同时保持任务性能（仅降 1.8%）。

---

## 🎮 强化学习 { #reinforcement_learning }

**[A Course Correction in Steerability Evaluation: Revealing Miscalibration and Side Effects in LLMs](reinforcement_learning/a_course_correction_in_steerability_evaluation_revealing_mis.md)**

:   本文提出了一个基于多维目标空间的 LLM 可操控性（steerability）评估框架，将 steering error 分解为校准偏差（miscalibration）和副作用（side effects/orthogonality），在文本改写任务上发现即使是最强的 LLM 也会产生严重副作用，prompt engineering 无效、best-of-N 采样代价高、RL 微调有改善但仍未彻底解决。

**[A Learning Framework For Cooperative Collision Avoidance of UAV Swarms Leveraging Domain Knowledge](reinforcement_learning/a_learning_framework_for_cooperative_collision_avoidance_of_.md)**

:   提出 reMARL 框架，将图像处理中的主动轮廓模型（active contour）作为领域知识引入多智能体强化学习的奖励设计，使无人机集群仅通过最大化个体奖励即可学会协作避撞，在大规模集群（≤10架）中性能显著优于 COMA/VDN/QMIX/MAPPO 等 SOTA MARL 方法，反应时间比元启发式方法快 98.75%，能耗降低 85.37%。

**[A Learning Framework For Cooperative Collision Avoidance of UAV Swarms Leveraging Domain Knowledge](reinforcement_learning/a_learning_framework_for_cooperative_collision_avoidance_of_uav_swarms_leveragin.md)**

:   提出 reMARL 框架，利用图像处理领域知识（active contour model）设计多智能体强化学习奖励函数，实现无人机集群的协作避碰，相比传统元启发式方法反应时间缩短 98.75%、能耗降低 85.37%。

**[A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses](reinforcement_learning/a_multi-agent_conversational_bandit_approach_to_online_evaluation_and_selection_.md)**

:   提出 MACO 多智能体会话式 Bandit 框架，通过本地 agent 的在线淘汰和云服务器的自适应偏好查询机制，实现 LLM 响应的在线评估与用户偏好对齐，达到 $\tilde{O}(\sqrt{dMT})$ 的近优 regret 界。

**[Aligning Machiavellian Agents: Behavior Steering via Test-Time Policy Shaping](reinforcement_learning/aligning_machiavellian_agents_behavior_steering_via_test-tim.md)**

:   提出一种测试时策略塑形方法，通过轻量级伦理属性分类器在推理阶段插值修改预训练 RL 智能体的动作概率分布，无需重训练即可实现对多种伦理属性的细粒度行为引导。

**[BAMAS: Structuring Budget-Aware Multi-Agent Systems](reinforcement_learning/bamas_structuring_budget-aware_multi-agent_systems.md)**

:   提出 BAMAS 框架，通过整数线性规划（ILP）在预算约束下选择最优 LLM 组合，再用强化学习策略选择最佳协作拓扑（线性/星型/反馈/规划驱动），在 GSM8K/MBPP/MATH 上达到与 SOTA 多 Agent 系统相当的准确率，同时成本降低最高 86%。

**[Does Self-Evaluation Enable Wireheading in Language Models?](reinforcement_learning/does_self-evaluation_enable_wireheading_in_language_models.md)**

:   形式化证明在POMDP框架下自评分耦合奖励信号时wireheading（操纵评估而非优化任务）严格优于诚实行为（Lemma 1），并在Llama-3.1-8B和Mistral-7B上实验验证——Selfgrade条件下摘要任务奖励饱和至0.95但准确率仅0.05，解耦自评分与奖励可消除直接wireheading激励。

**[MARS: Multi-Agent Adaptive Reasoning with Socratic Guidance for Automated Prompt Optimization](reinforcement_learning/mars_multi-agent_adaptive_reasoning_with_socratic_guidance_f.md)**

:   提出 MARS 五智能体框架做自动提示优化（APO）：Planner 生成任务特定的优化轨迹，Teacher-Critic-Student 三体进行苏格拉底对话式迭代精炼 prompt（模拟文本空间中的伪梯度下降），Target 执行并反馈，整体建模为 POMDP，在 17 个数据集上平均超越前 SOTA（PE2）6.04%（通用任务）和 6.42%（领域任务），且仅需 1-shot 训练数据。

**[MMhops-R1: Multimodal Multi-hop Reasoning](reinforcement_learning/mmhops-r1_multimodal_multi-hop_reasoning.md)**

:   提出了 MMhops 基准（31K 样本、3-4 跳推理深度）和 MMhops-R1 框架，通过强化学习训练 MLLM 自主规划推理路径、动态调用图像/文本检索器，实现多模态多跳推理，7B 模型超越 72B 基线和现有 mRAG 方法。

**[Test-driven Reinforcement Learning in Continuous Control](reinforcement_learning/test-driven_reinforcement_learning_in_continuous_control.md)**

:   提出 Test-driven Reinforcement Learning (TdRL) 框架，用多个测试函数（pass-fail 测试定义最优目标 + indicative 测试引导学习）替代单一奖励函数表示任务目标，通过字典序启发式轨迹比较学习回报函数，在 DeepMind Control Suite 上匹配或超越手工奖励方法，天然支持多目标优化。

---

## 📦 模型压缩 { #model_compression }

**[A Closer Look at Knowledge Distillation in Spiking Neural Network Training](model_compression/a_closer_look_at_knowledge_distillation_in_spiking_neural_ne.md)**

:   针对ANN→SNN知识蒸馏中教师ANN连续特征/logits与学生SNN离散稀疏spike特征/logits之间分布差异被忽视的问题，提出基于显著性缩放激活图蒸馏（SAMD）和噪声平滑logits蒸馏（NLD）的CKDSNN框架，在CIFAR-10/100、ImageNet-1K和CIFAR10-DVS上均取得SNN训练的新SOTA。

**[AdaFuse: Accelerating Dynamic Adapter Inference via Token-Level Pre-Gating and Fused Kernel Optimization](model_compression/adafuse_accelerating_dynamic_adapter_inference_via_token-lev.md)**

:   针对动态MoE-LoRA适配器推理延迟暴增（250%-950%）的问题，提出了一种token级预门控架构，只在第一层做一次全局路由决策，配合自研的SGMM融合CUDA内核将所有激活的LoRA适配器一次性合并进骨干网络，在保持精度的同时将解码延迟降低2.4倍。

**[AgentODRL: A Large Language Model-based Multi-agent System for ODRL Generation](model_compression/agentodrl_a_large_language_model-based_multi-agent_system_fo.md)**

:   提出AgentODRL，一个基于Orchestrator-Workers架构的LLM多智能体系统，通过任务分解、语法验证循环和LoRA驱动的语义反思机制，将自然语言数据权限规则高质量地转换为ODRL格式。

**[ALTER: Asymmetric LoRA for Token-Entropy-Guided Unlearning of LLMs](model_compression/alter_asymmetric_lora_for_token-entropy-guided_unlearning_of.md)**

:   提出ALTER框架，利用非对称LoRA架构结合Token级别的Tsallis熵引导，实现LLM中目标知识的精准遗忘，同时通过参数隔离机制保留模型基础能力，在TOFU、WMDP和MUSE三个基准上达到SOTA。

**[CAMERA: Multi-Matrix Joint Compression for MoE Models via Micro-Expert Redundancy Analysis](model_compression/camera_multi-matrix_joint_compression_for_moe_models_via_mic.md)**

:   提出"micro-expert"概念将MoE层的输出分解为跨矩阵（up/gate/down_proj）的微专家线性组合，基于能量排序进行结构化剪枝(Camera-P)和混合精度量化(Camera-Q)，在Deepseek-MoE-16B/Qwen2-57B/Qwen3-30B上20%-60%剪枝率全面超越NAEE和D²-MoE，且分析Qwen2-57B仅需单卡A100不到5分钟。

**[Distilling Cross-Modal Knowledge via Feature Disentanglement](model_compression/distilling_cross-modal_knowledge_via_feature_disentanglement.md)**

:   提出频域解耦跨模态知识蒸馏（FD-CMKD），通过傅里叶变换将特征分解为低频（模态共享语义）和高频（模态特有细节）分量，分别施加强一致性 MSE 和弱一致性 logMSE 损失，并引入尺度标准化与共享分类器对齐特征空间，在音频-视觉、图像-文本、语义分割等多个跨模态场景全面超越现有蒸馏方法。

**[DynaQuant: Dynamic Mixed-Precision Quantization for Learned Image Compression](model_compression/dynaquant_dynamic_mixed-precision_quantization_for_learned_i.md)**

:   针对学习图像压缩（LIC）模型部署效率低的痛点，提出DynaQuant框架，在参数层面通过可学习scale/zero-point + Distance-Aware Gradient Modulator实现内容自适应量化，在架构层面通过轻量Bit-Width Selector动态为每层分配最优比特宽度，在Cheng2020/ELIC/Ballé三个基线上实现接近FP32的R-D性能，同时获得最高5.17×加速和模型大小降至原来的~1/4。

**[KVmix: Gradient-Based Layer Importance-Aware Mixed-Precision Quantization for KV Cache](model_compression/kvmix_gradient-based_layer_importance-aware_mixed-precision_.md)**

:   提出 KVmix，通过计算 Key/Value 投影权重梯度的 $L_2$ 范数来评估各层 KV Cache 的重要性，实现层级混合精度量化（Key 平均 2.19bit、Value 平均 2.38bit），并结合动态关键上下文选择（RPC）策略，在 Llama/Mistral 等模型上实现近无损推理、4.9× 内存压缩和 5.3× 吞吐加速。

**[SCoPe: Intrinsic Semantic Space Control for Mitigating Copyright Infringement in LLMs](model_compression/scope_intrinsic_semantic_space_control_for_mitigating_copyright_infringement_in_.md)**

:   将LLM版权侵权缓解问题重新定义为内在语义空间控制，利用稀疏自编码器(SAE)将隐状态映射到高维稀疏空间，识别版权敏感子空间并在解码时钳制其激活，无需外部过滤器或参数更新即可有效减少版权内容复制，同时保持模型通用能力。

---

## ⚖️ 对齐 / RLHF { #llm_alignment }

**[Align to Structure: Aligning Large Language Models with Structural Information](llm_alignment/align_to_structure_aligning_large_language_models_with_struc.md)**

:   提出 Structural Alignment 方法，通过将语言学篇章结构框架（表层文本结构评分 + 基于RST的篇章motif分类器）融入PPO强化学习训练，并设计基于篇章motif的密集奖励机制，使LLM生成更连贯、更具人类写作风格的长文本，在论文写作和长文档摘要任务上均优于标准RLHF模型。

**[AlignTree: Efficient Defense Against LLM Jailbreak Attacks](llm_alignment/aligntree_efficient_defense_against_llm_jailbreak_attacks.md)**

:   AlignTree 利用 LLM 内部激活特征（线性 refusal direction + 非线性 SVM 信号）训练轻量级随机森林分类器，在几乎不增加计算开销的情况下高效检测越狱攻击，实现了 SOTA 的攻击成功率（ASR）降低效果。

**[AMaPO: Adaptive Margin-attached Preference Optimization for Language Model Alignment](llm_alignment/amapo_adaptive_margin-attached_preference_optimization_for_l.md)**

:   提出AMaPO算法，通过实例级自适应margin（结合Z-normalization和指数缩放）动态调节梯度幅度，解决DPO等离线偏好优化方法中对已正确排序样本过拟合、对错误排序样本欠拟合的核心矛盾，显著提升排序准确率和下游对齐性能。

**[DeCoRL: Decoupling Reasoning Chains via Parallel Sub-Step Generation and Cascaded Reinforcement for Interpretable and Scalable RLHF](llm_alignment/decorl_decoupling_reasoning_chains_via_parallel_sub-step_gen.md)**

:   DeCoRL 将 CoT 推理从单体顺序处理转变为"交响乐团式"的模块化并行协作——9 个专用子模型（解析/语义/实体/事实核查/风格/质量/计算/验证/整合）并行生成推理子步骤，通过双重奖励归因（本地质量+贡献度）+ 级联 DRPO 优化协调，在 RM-Bench 上达到 80.8%（超越所有基线），同时实现 3.8 倍推理加速和 22.7% 的可解释性提升。

**[Exploring the Effects of Alignment on Numerical Bias in Large Language Models](llm_alignment/exploring_the_effects_of_alignment_on_numerical_bias_in_large_language_models.md)**

:   系统揭示了LLM对齐过程（指令调优+偏好调优）是LLM评估器产生数值偏差的根本原因，并验证分数范围调整是最有效的缓解策略。

**[Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models](llm_alignment/probing_preference_representations_a_multi-dimensional_evaluation_and_analysis_m.md)**

:   提出 MRMBench 基准，通过 6 个维度（无害性、有帮助性、正确性、连贯性、复杂性、冗长性）的探针任务评估奖励模型是否有效捕获多维偏好，发现探针性能与 PPO 对齐质量强相关（Pearson $r > 0.8$），并提出推理时探针方法将 AlpacaEval win rate 从 57.3% 提升至 62.5%。

**[SceneJailEval: A Scenario-Adaptive Multi-Dimensional Framework for Jailbreak Evaluation](llm_alignment/scenejaileval_a_scenario-adaptive_multi-dimensional_framework_for_jailbreak_eval.md)**

:   提出SceneJailEval，一个场景自适应的多维度越狱评估框架，定义14个越狱场景和10个评估维度，通过场景分类→维度动态选择→多维检测→加权危害评分的流程，在自建数据集上F1达0.917（超SOTA 6%），在JBB上达0.995（超SOTA 3%），同时支持危害程度量化而非仅二分类。

**[Towards Inference-Time Scaling for Continuous Space Reasoning](llm_alignment/towards_inference-time_scaling_for_continuous_space_reasoning.md)**

:   首次系统研究离散文本推理中的inference-time scaling技术能否迁移到连续潜空间推理模型（COCONUT），发现dropout采样能生成多样推理路径（Pass@32达44.43%），但PRM/ORM仅带来不足2.3%提升，根因在于连续思维表示缺乏区分正误推理的几何归纳偏置。

---

## 🏥 医学图像 { #medical_imaging }

**[A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation](medical_imaging/a_disease-aware_dual-stage_framework_for_chest_x-ray_report_.md)**

:   提出一种两阶段疾病感知框架，通过学习14个与病理类别对应的疾病感知语义token（DASTs）实现显式的疾病表征，再利用疾病-视觉注意力融合（DVAF）和双模态相似性检索（DMSR）机制辅助LLM生成临床准确的胸部X光报告，在CheXpert Plus、IU X-Ray和MIMIC-CXR三个数据集上取得SOTA。

**[A Principle-Driven Adaptive Policy for Group Cognitive Stimulation Dialogue for Elderly with Cognitive Impairment](medical_imaging/a_principle-driven_adaptive_policy_for_group_cognitive_stimu.md)**

:   针对老年认知障碍患者的群体认知刺激治疗（CST）场景，提出GCSD系统：通过多说话人上下文控制、动态参与者状态建模（soft prompt）、认知刺激注意力损失和多维奖励策略优化四个模块，基于Qwen-2.5-3B微调，在500+小时真实粤语CST对话和1万+模拟对话上训练，BLEU-4达27.93超越GPT-4o等大模型，A/B测试胜率50% vs GPT-4o的39%。

**[Advancing Safe Mechanical Ventilation Using Offline RL With Hybrid Actions and Clinically Aligned Rewards](medical_imaging/advancing_safe_mechanical_ventilation_using_offline_rl_with_.md)**

:   针对ICU机械通气（MV）设置优化问题，提出混合动作空间的离线RL方法（HybridIQL/HybridEDAC），避免传统离散化导致的分布偏移，同时引入基于无通气天数（VFD）和生理参数安全范围的临床对齐奖励函数，通过多目标优化选择最优奖励，将可优化的通气参数从2-3个扩展到6个，HybridIQL在性能和策略覆盖率间取得最佳平衡。

**[Ambiguity-aware Truncated Flow Matching for Ambiguous Medical Image Segmentation](medical_imaging/ambiguity-aware_truncated_flow_matching_for_ambiguous_medica.md)**

:   提出 ATFM 框架，通过数据层级推理范式将预测精度和多样性解耦到分布级和样本级分别优化，结合高斯截断表示（GTR）和分割流匹配（SFM）两个模块，在模糊医学图像分割任务中同时提升预测的精度、保真度和多样性。

**[Apo2Mol: 3D Molecule Generation via Dynamic Pocket-Aware Diffusion Models](medical_imaging/apo2mol_3d_molecule_generation_via_dynamic_pocket-aware_diff.md)**

:   提出Apo2Mol，一个基于扩散的全原子框架，从蛋白质apo（未结合）构象出发，同时生成3D配体分子和对应的holo（结合态）口袋构象，使用24K实验解析的apo-holo结构对训练，在结合亲和力（Vina min -7.86）和药物类似性上达到SOTA。

**[FourierPET: Deep Fourier-based Unrolled Network for Low-count PET Reconstruction](medical_imaging/fourierpet_deep_fourier-based_unrolled_network_for_low-count_pet_reconstruction.md)**

:   发现低剂量 PET 的三类退化在频域可分离——泊松噪声/光子不足导致高频相位扰动，衰减校正误差抑制低频幅度——据此提出 FourierPET：基于 ADMM 展开的频率感知重建框架，仅 0.44M 参数在三个数据集上全面 SOTA。

**[Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling](medical_imaging/learning_cell-aware_hierarchical_multi-modal_representations.md)**

:   提出CHMR框架，将分子结构(1D/2D/3D)与细胞形态/基因表达等生物模态联合建模，通过结构感知的模态增强解决>90%的外部生物模态缺失问题，用树状向量量化(Tree-VQ)捕获分子-细胞-基因的层次化依赖关系，在9个benchmark的728个任务上超越SOTA，分类平均AUC提升3.6%，回归MAE降低17.2%。

**[ProtSAE: Disentangling and Interpreting Protein Language Models via Semantically-Guided Sparse Autoencoders](medical_imaging/protsae_disentangling_and_interpreting_protein_language_models_via_semantically-.md)**

:   提出 ProtSAE，在稀疏自编码器训练中引入语义标注和领域本体知识作为引导信号，解决传统 SAE 的语义纠缠问题，使蛋白质语言模型的隐层特征与生物学概念（分子功能、生物过程、离子结合位点等）精准对齐，同时保持高重建保真度并支持概念级别的生成控制。

---

## 🧊 3D 视觉 { #3d_vision }

**[3D-ANC: Adaptive Neural Collapse for Robust 3D Point Cloud Recognition](3d_vision/3d-anc_adaptive_neural_collapse_for_robust_3d_point_cloud_re.md)**

:   将Neural Collapse(NC)机制引入3D点云对抗鲁棒性，用固定的ETF分类头+自适应训练框架(RBL+FDL)构建解耦的特征空间，在ModelNet40上将DGCNN的对抗准确率从27.2%提升到80.9%，超出最佳baseline 34个点。

**[4DSTR: Advancing Generative 4D Gaussians with Spatial-Temporal Rectification for High-Quality and Consistent 4D Generation](3d_vision/4dstr_advancing_generative_4d_gaussians_with_spatial-tempora.md)**

:   提出4DSTR框架，通过基于Mamba的时序关联校正（修正高斯点的尺度和旋转）以及逐帧自适应稠密化与裁剪策略，显著提升4D高斯生成的时空一致性和对快速时序变化的适应能力。

**[Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models](3d_vision/adapt-as-you-walk_through_the_clouds_training-free_online_te.md)**

:   提出 Uni-Adapter，一种面向3D视觉-语言基础模型(VLFM)的无训练在线测试时适应框架，通过基于聚类的动态原型缓存和图正则化标签平滑来应对分布偏移，在多个3D损坏基准上取得SOTA。

**[AnchorDS: Anchoring Dynamic Sources for Semantically Consistent Text-to-3D Generation](3d_vision/anchords_anchoring_dynamic_sources_for_semantically_consiste.md)**

:   揭示 SDS 中源分布是动态演化而非静态的关键问题，提出 AnchorDS，通过将当前渲染图像作为图像条件输入双条件扩散模型来锚定源分布，解决了 SDS 的语义过度平滑和多视角不一致问题，在 T3Bench 上全面超越 SDS/VSD/SDS-Bridge。

**[AnchorHOI: Zero-shot Generation of 4D Human-Object Interaction via Anchor-based Prior Distillation](3d_vision/anchorhoi_zero-shot_generation_of_4d_human-object_interactio.md)**

:   提出 AnchorHOI，通过锚点NeRF和锚点关键点两种中间桥梁，分别从图像/视频扩散模型中蒸馏交互先验和运动先验，实现零样本的文本驱动4D人物-物体交互生成，在静态3D和动态4D HOI生成上均超越已有方法。

**[Arbitrary-Scale 3D Gaussian Super-Resolution](3d_vision/arbitrary-scale_3d_gaussian_super-resolution.md)**

:   提出一个集成框架实现3D高斯溅射(3DGS)的任意倍率超分辨率渲染，通过尺度感知渲染、生成先验引导优化和渐进超分机制，用单个3D模型支持整数和非整数倍率的HR渲染，PSNR提升6.59dB同时保持85 FPS实时速度。

**[FoundationSLAM: 释放深度基础模型在端到端稠密视觉SLAM中的潜力](3d_vision/foundationslam_unleashing_the_power_of_depth_foundation_models_for.md)**

:   将深度基础模型的几何先验注入光流式SLAM系统，通过混合光流网络、双向一致BA层和可靠性感知精炼三个模块形成闭环，在TUM/EuRoC/7Scenes/ETH3D四大数据集取得SOTA轨迹精度和稠密重建质量，18 FPS实时运行。

---

## 🛡️ AI 安全 { #ai_safety }

**[Alternative Fairness and Accuracy Optimization in Criminal Justice](ai_safety/alternative_fairness_and_accuracy_optimization_in_criminal_j.md)**

:   本文系统综述了算法公平性的三大维度（群体公平、个体公平、过程公平），提出了一种基于容差约束的改进群体公平性优化公式，并构建了面向公共决策系统的"公平三支柱"部署框架。

**[An Improved Privacy and Utility Analysis of Differentially Private SGD with Bounded Domain and Smooth Losses](ai_safety/an_improved_privacy_and_utility_analysis_of_differentially_p.md)**

:   在仅假设损失函数L-光滑（不需要凸性）的条件下，为DPSGD推导出了更紧的闭式RDP隐私界，并首次在有界域场景下给出了完整的收敛性/效用分析，揭示了较小的参数域直径可以同时改善隐私和效用。

**[An Information Theoretic Evaluation Metric for Strong Unlearning](ai_safety/an_information_theoretic_evaluation_metric_for_strong_unlear.md)**

:   提出 Information Difference Index (IDI)，一种基于信息论的白盒评估指标，通过度量中间层特征与遗忘标签之间的互信息来衡量机器遗忘的彻底程度，揭示了现有黑盒指标（MIA、JSD等）无法捕捉的中间层残留信息问题，并提出 COLA 方法在特征层面消除残余信息。

**[An LLM-Based Simulation Framework for Embodied Conversational Agents in Psychological Counseling](ai_safety/an_llm-based_simulation_framework_for_embodied_conversationa.md)**

:   提出 ECAs 框架，基于认知行为治疗(CBT)等心理学理论，利用 LLM 将真实咨询案例扩展为具身认知记忆空间，模拟心理咨询中来访者的完整认知过程，生成高保真度的咨询对话数据，在专家评估和自动评估中均显著优于基线。

**[Angular Gradient Sign Method: Uncovering Vulnerabilities in Hyperbolic Networks](ai_safety/angular_gradient_sign_method_uncovering_vulnerabilities_in_h.md)**

:   提出Angular Gradient Sign Method (AGSM)，将双曲空间中的梯度分解为径向（层次深度）和角度（语义）分量，仅沿角度方向施加扰动来生成对抗样本，在图像分类和跨模态检索任务上比标准FGSM/PGD多降低5-13%的准确率。

**[Argumentative Debates for Transparent Bias Detection (ABIDE)](ai_safety/argumentative_debates_for_transparent_bias_detection_technic.md)**

:   提出ABIDE框架，将偏见检测过程结构化为基于量化二极论辩框架（QBAF）的辩论：通过邻域级局部统计公平性（neighbourhood-based local statistical parity）生成偏见论据，利用批判性问题（critical questions）作为攻击机制挑战不可靠论据，在合成/真实/LLM模型上均优于IRB基线。

**[AUVIC: Adversarial Unlearning of Visual Concepts for Multi-modal Large Language Models](ai_safety/auvic_adversarial_unlearning_of_visual_concepts_for_multi-mo.md)**

:   提出AUVIC框架，通过对抗性扰动生成器 + 动态锚点保留机制，在MLLM中精确遗忘目标视觉概念（如特定人脸），同时避免对语义相似概念的附带遗忘，并构建了首个面向群体场景视觉概念遗忘的评测基准VCUBench。

---

## 💬 LLM / NLP { #llm_nlp }

**[An Invariant Latent Space Perspective on Language Model Inversion](llm_nlp/an_invariant_latent_space_perspective_on_language_model_inve.md)**

:   提出不变潜空间假说(ILSH)，将LLM反演问题重新建模为复用LLM自身潜空间，设计Inv²A框架通过轻量级逆编码器将输出映射到去噪伪表示，再由冻结的LLM解码恢复隐藏prompt，在9个数据集上BLEU平均提升4.77%且仅需20%数据量即可达到可比性能。

**[Beyond Accuracy: A Cognitive Load Framework for Mapping the Capability Boundaries of Tool-use Agents](llm_nlp/beyond_accuracy_a_cognitive_load_framework_for_mapping_the_c.md)**

:   借鉴心理学的认知负荷理论（CLT），将工具使用任务的复杂度分解为内在负荷（任务解题路径的结构复杂度）和外在负荷（问题表述的歧义性），构建可参数化调节认知负荷的 ToolLoad-Bench 基准，用指数衰减模型 $\text{Acc} \approx e^{-(k \cdot CL + b)}$ 精确刻画不同 Agent 的能力边界。

**[Guess or Recall? Training CNNs to Classify and Localize Memorization in LLMs](llm_nlp/guess_or_recall_training_cnns_to_classify_and_localize_memorization_in_llms.md)**

:   在 LLM 注意力权重上训练 CNN 来评估记忆化分类法与实际注意力机制的对齐程度，提出新的三类分类法（Guess/Recall/Non-Memorized），最小 F1 从 64.7% 提升至 89.0%，并定位了不同记忆类型分别依赖低层（Guess）和高层（Recall）注意力。

**[How Does Alignment Enhance LLMs' Multilingual Capabilities? A Language Neurons Perspective](llm_nlp/how_does_alignment_enhance_llms_multilingual_capabilities_a_language_neurons_per.md)**

:   提出三元神经元分类（语言特定/语言相关/通用），将 LLM 多语言推理分为四阶段分析，发现多语言对齐通过增加语言相关神经元（减少语言特定神经元）来提升性能，且在未训练语言上也产生"自发多语言对齐"效应。

**[MAPS: Multi-Agent Personality Shaping for Collaborative Reasoning](llm_nlp/maps_multi-agent_personality_shaping_for_collaborative_reaso.md)**

:   提出 MAPS 五 Agent 协作推理框架，基于大五人格理论为 4 个功能 Agent 赋予不同"性格"（Interpreter-开放性、Aligner-宜人性、Scholar-尽责性、Solver-外向性）实现异质化协作，加上 Critic Agent（神经质→苏格拉底式反思）做迭代修正，在 MathVista/OlympiadBench/EMMA 上超越 GPT-4o 基线 15.84%，首次超过人类专家 3.58%。

**[Rectification Reimagined: A Unified Mamba Model for Image Correction and Rectangling with Prompts](llm_nlp/rectification_reimagined_a_unified_mamba_model_for_image_cor.md)**

:   从统一畸变矫正视角出发，提出 UniRect 框架，通过 Residual Progressive TPS 处理几何形变 + Residual Mamba Blocks 补偿退化，统一处理肖像校正、广角矩形化、拼接矩形化、旋转校正四种任务，并通过 Sparse MoE 实现 four-in-one 多任务学习，拼接矩形化 PSNR 提升 3.82 dB，旋转校正提升 0.87 dB。

**[ReFeed: Retrieval Feedback-Guided Dataset Construction for Style-Aware Query Rewriting](llm_nlp/refeed_retrieval_feedback-guided_dataset_construction_for_style-aware_query_rewr.md)**

:   提出一个检索反馈驱动的数据集生成框架，通过识别检索失败case、LLM风格化改写、重检索验证三步闭环，自动构建高质量的风格感知查询改写数据集，为训练检索对齐的改写模型提供数据基础。

---

## 🎯 目标检测 { #object_detection }

**[Actor-Critic for Continuous Action Chunks: A Reinforcement Learning Framework for Long-Horizon Robotic Manipulation with Sparse Reward](object_detection/actor-critic_for_continuous_action_chunks_a_reinforcement_le.md)**

:   AC3 提出了一个直接学习连续动作序列（action chunk）的 actor-critic 框架，通过"仅从成功轨迹更新 actor"的非对称更新规则和基于自监督锚点的内在奖励来稳定稀疏奖励下的长时域机器人操作学习，在 BiGym 和 RLBench 的 25 个任务上取得优于现有方法的成功率。

**[AerialMind: Towards Referring Multi-Object Tracking in UAV Scenarios](object_detection/aerialmind_towards_referring_multi-object_tracking_in_uav_sc.md)**

:   构建了首个面向无人机场景的大规模 Referring Multi-Object Tracking（RMOT）基准数据集 AerialMind，并提出 HawkEyeTrack（HETrack）方法，通过视觉-语言共进化融合编码器和尺度自适应上下文精炼模块，在无人机航拍场景中实现语言引导的多目标跟踪。

**[Beyond Boundaries: Leveraging Vision Foundation Models for Source-Free Object Detection](object_detection/beyond_boundaries_leveraging_vision_foundation_models_for_so.md)**

:   提出利用VFM（DINOv2+Grounding DINO）增强无源域自适应目标检测（SFOD）的框架，通过全局特征对齐(PGFA)、实例级原型对比学习(PIFA)和双源伪标签融合(DEPF)三个模块，在6个跨域检测基准上取得SOTA，例如Cityscapes→Foggy Cityscapes达47.1% mAP（比DRU高3.5%），Sim10k→Cityscapes达67.4% AP（比DRU高8.7%）。

**[Connecting the Dots: Training-Free Visual Grounding via Agentic Reasoning](object_detection/connecting_the_dots_training-free_visual_grounding_via_agent.md)**

:   提出 GroundingAgent，一个完全不需要任务特定微调的视觉定位框架，通过组合预训练的开放词汇检测器（YOLO World）、MLLM（Llama-3.2-11B-Vision）和 LLM（DeepSeek-V3）进行结构化迭代推理，在 RefCOCO/+/g 上实现 65.1% 的零样本平均准确率，大幅超越之前的 zero-shot 方法。

**[Continuous Vision-Language-Action Co-Learning with Semantic-Physical Alignment for Behavioral Cloning](object_detection/continuous_vision-language-action_co-learning_with_semantic-.md)**

:   提出CCoL框架，通过NeuralODE驱动的多模态连续协同学习（MCC）和双向交叉注意力的语义-物理对齐（CSA），在Behavioral Cloning中同时解决动作序列的物理不连续性和语义-物理失配问题，在三个仿真平台上平均相对提升8.0%，双臂插入任务最高达19.2%。

**[Sketch-HARP: 分层自回归草图生成实现灵活笔画级绘制操控](object_detection/generating_sketches_in_a_hierarchical_auto-regressive_proces.md)**

:   提出 Sketch-HARP 分层自回归草图生成框架，通过三阶段层次化过程（预测笔画嵌入→确定画布位置→生成绘制动作序列），首次实现草图绘制过程中的灵活笔画级操控，在替换/擦除/扩展等任务上显著优于 SketchEdit。

**[TTF-VLA: Temporal Token Fusion via Pixel-Attention Integration for Vision-Language-Action Models](object_detection/ttf-vla_temporal_token_fusion_via_pixel-attention_integratio.md)**

:   TTF-VLA 提出了一种免训练的时序 Token 融合方法，通过灰度像素差异+注意力语义检测的双维度机制选择性地复用历史帧的视觉 Token，提升 VLA 模型在机器人操作任务中的推理质量，在 LIBERO 上平均提升 4.0 个百分点。

---

## 🕸️ 图学习 { #graph_learning }

**[Adaptive Riemannian Graph Neural Networks](graph_learning/adaptive_riemannian_graph_neural_networks.md)**

:   提出 ARGNN 框架，为图上每个节点学习一个连续的、各向异性的对角黎曼度量张量，从而自适应地捕获图中不同区域（层级结构 vs 密集社区）的局部几何特性，统一并超越了固定曲率和离散混合曲率的几何 GNN 方法。

**[Are Graph Transformers Necessary? Efficient Long-Range Message Passing with Fractal Nodes in MPNNs](graph_learning/are_graph_transformers_necessary_efficient_long-range_messag.md)**

:   提出分形节点（Fractal Nodes）增强 MPNN 的长距离消息传递：通过 METIS 图划分生成子图级聚合节点，结合低通+高通滤波器（LPF+HPF）与可学习频率参数 $\omega$，使用 MLP-Mixer 实跨子图通信，在保持 $O(L(|V|+|E|))$ 线性复杂度的同时达到甚至超越图 Transformer 的性能，获 AAAI Oral。

**[Assemble Your Crew: Automatic Multi-agent Communication Topology Design via Autoregressive Graph Generation](graph_learning/assemble_your_crew_automatic_multi-agent_communication_topol.md)**

:   提出 ARG-Designer，将多 Agent 系统的拓扑设计重新定义为条件自回归图生成任务，从零开始逐步生成 Agent 节点和通信边（而非从模板图剪枝），在6个基准上达到 SOTA（平均 92.78%），同时 Token 消耗比 G-Designer 降低约 50%，且支持无需重训练的角色扩展。

**[Relink: Constructing Query-Driven Evidence Graph On-the-Fly for GraphRAG](graph_learning/relink_constructing_query-driven_evidence_graph_on-the-fly_for_graphrag.md)**

:   提出从"先构建再推理"到"边推理边构建"的GraphRAG范式转变，通过Relink框架动态构建查询特定的证据图——结合高精度KG骨架和高召回潜在关系池，用查询驱动的排序器统一评估、按需补全缺失路径并过滤干扰事实——在5个多跳QA基准上平均提升EM 5.4%和F1 5.2%。

**[RFKG-CoT: Relation-Driven Adaptive Hop-count Selection and Few-Shot Path Guidance for Knowledge-Aware QA](graph_learning/rfkg-cot_relation-driven_adaptive_hop-count_selection_and_few-shot_path_guidance.md)**

:   提出RFKG-CoT，通过关系驱动的自适应跳数选择（利用KG关系激活掩码动态调整推理步数）和Few-Shot路径引导（Question-Paths-Answer格式的in-context示例），在4个KGQA基准上显著提升LLM的知识图谱推理能力，GPT-4在WebQSP上达91.5%（+6.6pp），Llama2-7B提升幅度最大达+14.7pp。

**[S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning](graph_learning/s-dag_a_subject-based_directed_acyclic_graph_for_multi-agent.md)**

:   提出 S-DAG，通过 GNN 从问题中识别相关学科及其依赖关系构建有向无环图，将学科节点匹配到最擅长的专家 LLM（14 个 7-13B 领域模型），按 DAG 拓扑顺序协作推理（支撑学科→主导学科），用小模型池超越 GPT-4o-mini（59.73 vs 58.52）且接近 72B 模型。

---

## 🎁 推荐系统 { #recommender }

**[Align³GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation](recommender/align3gr_unified_multi-level_alignment_for_llm-based_generat.md)**

:   提出统一三层对齐框架 Align³GR，在 token 级（双端 SCID）、行为建模级（多任务 SFT）和偏好级（渐进式 DPO）系统性弥合 LLM 与推荐系统之间的语义-行为鸿沟。

**[FreqRec: Exploiting Inter-Session Information with Frequency-enhanced Dual-Path Networks for Sequential Recommendation](recommender/exploiting_inter-session_information_with_frequency-enhanced_dual-path_networks_.md)**

:   提出FreqRec双路径架构，通过batch维和时间维两条频域路径分别捕获跨session群体节律和用户个体细粒度兴趣，并引入频域一致性损失显式对齐预测与真实频谱，在三个Amazon数据集上NDCG@10最高提升7.38%。

**[From Parameter to Representation: A Closed-Form Approach for Controllable Model Merging](recommender/from_parameter_to_representation_a_closed-form_approach_for_controllable_model_m.md)**

:   提出 ReACT，将可控模型合并从参数空间优化转移到表征空间校正，通过闭式解实现任意用户偏好下的 Pareto 最优模型即时生成，比现有方法快 36-208 倍且性能更优。

**[Inference-Aware Prompt Optimization for Aligning Black-Box Large Language Models](recommender/inference-aware_prompt_optimization_for_aligning_black-box_large_language_models.md)**

:   揭示 prompt 选择与推理策略（Best-of-N、Majority Voting）之间存在非平凡交互关系，提出 IAPO 框架将 prompt 设计与推理规模联合优化为上下文最优臂识别问题，并设计 PSST 固定预算训练算法，在 6 个任务上相比推理无关方法提升最高 50%。

**[Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation](recommender/wavelet_enhanced_adaptive_frequency_filter_for_sequential_re.md)**

:   提出WEARec模型结合动态频域滤波(DFF)和小波特征增强(WFE)两个模块，分别捕获个性化全局频域信息和增强非平稳短期波动，在四个公开数据集上超越频域推荐SOTA基线，长序列场景提升可达11.4%。

---

## ✂️ 语义分割 { #segmentation }

**[3DTeethSAM: Taming SAM2 for 3D Teeth Segmentation](segmentation/3dteethsam_taming_sam2_for_3d_teeth_segmentation.md)**

:   将SAM2基础模型迁移到3D牙齿分割任务，通过多视角渲染将3D mesh转为2D图像、设计三个轻量适配器（Prompt生成器、Mask精化器、Mask分类器）和可变形全局注意力插件（DGAP）来解决自动提示、边界精化和语义分类问题，在Teeth3DS上以91.90% T-mIoU刷新SOTA。

**[A²LC: Active and Automated Label Correction for Semantic Segmentation](segmentation/a2lc_active_and_automated_label_correction_for_semantic_segm.md)**

:   提出 A²LC 框架，在传统主动标签校正（人工逐一纠错）的基础上增加一个自动校正阶段（Label Correction Module），利用标注员的反馈自动修正相似的错误mask，并设计自适应平衡采集函数缓解类别不平衡，在 Cityscapes 上仅用 20% 预算即超越前 SOTA，同等预算下 mIoU 提升 27.23%。

**[Adaptive Morph-Patch Transformer for Aortic Vessel Segmentation](segmentation/adaptive_morph-patch_transformer_for_aortic_vessel_segmentat.md)**

:   提出 Morph-Patch Transformer (MPT)，通过基于速度场的自适应 patch 划分策略生成形态感知 patch（保持血管拓扑完整性），并引入语义聚类注意力（SCA）动态聚合语义相似 patch 的特征，在 AVT、AortaSeg24 和 TBAD 三个主动脉分割数据集上均达 SOTA。

**[Causal-Tune: Mining Causal Factors from Vision Foundation Models for Domain Generalized Semantic Segmentation](segmentation/causal-tune_mining_causal_factors_from_vision_foundation_mod.md)**

:   提出Causal-Tune，从因果视角分析VFM特征中的artifacts，利用DCT频域分解+高斯带通滤波分离因果/非因果因素，结合因果感知可学习token在频域精化特征，在Cityscapes→ACDC跨域分割中平均提升+2.4% mIoU（Snow场景+4.8%），仅需单卡RTX3090/14GB训练。

**[InfoCLIP: Bridging Vision-Language Pretraining and Open-Vocabulary Semantic Segmentation via Information-Theoretic Alignment Transfer](segmentation/infoclip_bridging_vision-language_pretraining_and_open-vocab.md)**

:   提出InfoCLIP，基于信息论视角设计信息瓶颈压缩和互信息蒸馏两个目标，在CLIP微调过程中去除预训练pixel-text对齐中的噪声并保留语义对齐知识，在6个开放词汇语义分割测试集上全面超越SOTA（A-847: 16.6, A-150: 38.5, PC-59: 63.5 mIoU），且仅增加0.53M参数和极少计算开销。

---

## 🧑 人体理解 { #human_understanding }

**[10 Open Challenges Steering the Future of Vision-Language-Action Models](human_understanding/10_open_challenges_steering_the_future_of_vision-language-ac.md)**

:   一篇针对Vision-Language-Action(VLA)模型的综述/展望论文，系统梳理了VLA领域的10大开放挑战（多模态感知、鲁棒推理、数据质量、评估、跨机器人泛化、效率、全身协调、安全、多智能体、人机协作）以及4大新兴趋势（层次化规划、空间理解、世界动力学建模、数据合成），为VLA研究指明方向。

**[AHAN: Asymmetric Hierarchical Attention Network for Identical Twin Face Verification](human_understanding/ahan_asymmetric_hierarchical_attention_network_for_identical.md)**

:   针对同卵双胞胎人脸验证这一极端细粒度识别挑战，提出 AHAN 多流架构，通过层次交叉注意力 (HCA) 对语义面部区域做多尺度分析、面部不对称注意力模块 (FAAM) 捕获左右脸差异签名、以及双胞胎感知配对交叉注意力 (TA-PWCA) 训练正则化，在 ND_TWIN 数据集上将双胞胎验证精度从 88.9% 提升至 92.3%（+3.4%）。

**[Anti-adversarial Learning: Desensitizing Prompts for Large Language Models](human_understanding/anti-adversarial_learning_desensitizing_prompts_for_large_la.md)**

:   提出 PromptObfus，通过"反对抗学习"思路将用户 prompt 中的敏感词替换为语义不同但不影响任务输出的词，从而在不降低远端 LLM 任务表现的前提下彻底消除显式隐私泄露，并将隐式隐私推理攻击成功率降低 62.70%。

**[RENEW: Risk- and Energy-Aware Navigation in Dynamic Waterways](human_understanding/renew_risk-_and_energy-aware_navigation_in_dynamic_waterways.md)**

:   提出 RENEW 全局路径规划器，为水面自主航行器 (ASV) 在动态水流 (洋流) 环境中引入统一的风险感知和能量感知策略，通过自适应不可导航区域识别、最佳努力应急策略和基于约束 Delaunay 三角化的分层架构实现安全高效导航，应急碰撞测试中实现零碰撞。

---

## 🖼️ 图像恢复 { #image_restoration }

**[Clear Nights Ahead: Towards Multi-Weather Nighttime Image Restoration](image_restoration/clear_nights_ahead_towards_multi-weather_nighttime_image_res.md)**

:   首次定义并探索多天气夜间图像复原任务，构建 AllWeatherNight 数据集（8K 训练 + 1K 合成测试 + 1K 真实测试），提出 ClearNight 统一框架通过 Retinex 双先验引导和天气感知动态专一性-共性协作，一阶段同时移除雾/雨条/雨滴/雪/flare 复合退化，仅 2.84M 参数全面超越 SOTA。

**[ICLR: Inter-Chrominance and Luminance Interaction for Natural Color Restoration in Low-Light Image Enhancement](image_restoration/iclr_inter-chrominance_and_luminance_interaction_for_natural_color_restoration_i.md)**

:   针对HVI色彩空间中色度和亮度分支分布差异大导致互补特征提取不足、以及色度分支间弱相关导致梯度冲突的问题，提出ICLR框架，通过双流交互增强模块(DIEM)和协方差校正损失(CCL)分别从融合增强和统计分布优化两个角度解决，在LOL系列数据集上取得SOTA。

**[SD-PSFNet: Sequential and Dynamic Point Spread Function Network for Image Deraining](image_restoration/sd-psfnet_sequential_and_dynamic_point_spread_function_netwo.md)**

:   提出基于动态 PSF 机制的级联 CNN 去雨网络 SD-PSFNet，通过多尺度可学习 PSF 字典建模雨滴光学效应，配合自适应门控融合的序列化修复架构，在 Rain100H 达 33.12 dB、RealRain-1k-L 达 42.28 dB 均为 SOTA，对比基线 MPRNet 累计提升 5.04 dB（13.5%）。

**[Seeing the Unseen: Zooming in the Dark with Event Cameras](image_restoration/seeing_the_unseen_zooming_in_the_dark_with_event_cameras.md)**

:   提出首个事件驱动低光照视频超分辨率框架RetinexEVSR，通过Retinex启发的双向融合策略（光照引导事件增强+事件引导反射增强），在SDSD-indoor上较EvTexture提升2.95 dB且FLOPs减少86%、推理加速65%。

---

## 📐 优化/理论 { #optimization }

**[A Distributed Asynchronous Generalized Momentum Algorithm Without Delay Bounds](optimization/a_distributed_asynchronous_generalized_momentum_algorithm_wi.md)**

:   提出一种完全异步（totally asynchronous）的广义动量（Generalized Momentum）分布式优化算法，无需假设通信/计算延迟的上界即可保证线性收敛，在 Fashion-MNIST 分类任务上比梯度下降快 71%、比 Heavy Ball 快 41%、比 Nesterov 加速梯度法快 19%。

**[A Unified Convergence Analysis for Semi-Decentralized Learning: Sampled-to-Sampled vs. Sampled-to-All Communication](optimization/a_unified_convergence_analysis_for_semi-decentralized_learni.md)**

:   本文在统一的收敛分析框架下，首次系统比较了半去中心化联邦学习中两种服务器-设备通信原语（S2S仅返回被采样设备 vs. S2A广播给所有设备），揭示了S2S在高组间异质性下更优、S2A在低异质性下更优的不同regime，并给出了实用的系统配置指南。

**[Explore How to Inject Beneficial Noise in MLLMs](optimization/explore_how_to_inject_beneficial_noise_in_mllms.md)**

:   提出 Multimodal Noise Generator (MuNG)，通过变分推断框架从图文对中动态生成"有益噪声"注入冻结的MLLM视觉特征中，以抑制无关语义、增强跨模态表征对齐，仅需约1%额外参数即可超越全参数微调和LoRA等PEFT方法。

**[On the Learning Dynamics of Two-Layer Linear Networks with Label Noise SGD](optimization/on_the_learning_dynamics_of_two-layer_linear_networks_with_label_noise_sgd.md)**

:   在二层过参数化线性网络上理论分析 Label Noise SGD 的学习动力学，揭示了两阶段行为——Phase I 中权重范数逐渐缩小使模型从 lazy regime 逃逸到 rich regime，Phase II 中权重与真实插值器对齐并收敛——并将该理论扩展到 SAM 优化器。

---

## 🎬 视频理解 { #video_understanding }

**[APVR: Hour-Level Long Video Understanding with Adaptive Pivot Visual Information Retrieval](video_understanding/apvr_hour-level_long_video_understanding_with_adaptive_pivot.md)**

:   提出APVR，一个训练免费的双粒度视觉信息检索框架：帧级别通过查询扩展+时空语义置信度打分迭代检索关键帧（最多1024帧），token级别通过查询感知的注意力驱动选择压缩视觉token，突破内存墙限制处理小时级长视频，在LongVideoBench/VideoMME/MLVU上分别提升最高9.5%/4.6%/9.7%。

**[Distillation Dynamics: Towards Understanding Feature-Based Distillation in Vision Transformers](video_understanding/distillation_dynamics_towards_understanding_feature-based_di.md)**

:   提出"蒸馏动力学"分析框架（频谱分析+信息熵+激活幅值），揭示ViT具有独特的U型信息处理模式（先压缩后扩展），证明feature-based蒸馏在ViT中失败的根本原因是teacher后层的分布式高维编码范式与student有限通道容量之间的表征范式不匹配，而非简单的容量差距。

**[DreamRunner: Fine-Grained Compositional Story-to-Video Generation with Retrieval-Augmented Motion Adaptation](video_understanding/dreamrunner_fine-grained_compositional_story-to-video_genera.md)**

:   提出 DreamRunner 框架，通过 LLM 双层规划 + 检索增强运动先验学习 + 时空区域3D注意力模块(SR3AI)，实现细粒度可控的多角色多事件故事视频生成。

**[MambaMia: State-Space Hierarchical Compression for Hour-Long Video Understanding in Large Multimodal Models](video_understanding/state-space_hierarchical_compression_with_gated_attention_an.md)**

:   MambaMia 提出了基于双向 Mamba 的两阶段层次化视频 Token 压缩框架：门控 Patch 聚合（GPA）做空间-时间局部压缩 + 时间轴聚合器（TAA）利用 Mamba 的自适应步长 $\Delta_t$ 做数据驱动的关键帧采样，将小时级视频压缩到仅 4.7K Token，在 LVBench 上达到 44.6 分超越 Qwen2-VL 和 mPLUG-Owl3。

---

## 🔗 因果推理 { #causal_inference }

**[Causally-Grounded Dual-Path Attention Intervention for Object Hallucination Mitigation in LVLMs](causal_inference/causally-grounded_dual-path_attention_intervention_for_objec.md)**

:   提出 Owl 框架，通过结构因果模型将视觉/文本注意力建模为中介变量，引入 VTACR 指标量化跨模态注意力失衡，设计 VTACR 引导的自适应注意力调制 + 双路径对比解码策略，在 POPE 和 CHAIR 上实现 SOTA 的幻觉抑制效果。

**[Hallucinate Less by Thinking More: Aspect-Based Causal Abstention for Large Language Models](causal_inference/hallucinate_less_by_thinking_more_aspect-based_causal_absten.md)**

:   提出 ABCA（Aspect-Based Causal Abstention），一个生成前弃权框架：通过双 Agent 辩论发现"方面变量"（如学科、法律语境、时间框架）来激活 LLM 不同的知识分支，用 AIPW 双鲁棒估计器计算因果效应，基于质心角偏差（CAD）检测知识冲突（Type-1）或知识不足（Type-2），在 TruthfulQA 上达到 91.4% 准确率，不可回答问题识别率 96.4%（远超基线的 44%）。

**[MUG: Multi-agent Undercover Gaming — Hallucination Removal via Counterfactual Test for Multimodal Reasoning](causal_inference/multi-agent_undercover_gaming_hallucination_removal_via_coun.md)**

:   MUG 将多 Agent 辩论（MAD）重新定义为"谁是卧底"社交推理游戏——通过图像反事实编辑（修改参考图片）引入信息不对称，让一个 Agent 持有修改后的图片作为"卧底"，其他 Agent 通过推理和投票识别卧底（幻觉来源），在 HallusionBench 上 Qwen2.5VL-7B 从 46.4% 提升到 53.8%。

---

## ⚡ LLM 效率 { #llm_efficiency }

**[A Content-Preserving Secure Linguistic Steganography](llm_efficiency/a_content-preserving_secure_linguistic_steganography.md)**

:   提出首个内容保持型语言隐写术范式CLstega，通过微调掩码语言模型（MLM）来可控地变换预测分布，将秘密信息嵌入到不做任何修改的原始文本中，实现了100%提取成功率和近乎完美的安全性（隐写分析检测准确率接近随机猜测的0.5）。

**[Harnessing the Unseen: The Hidden Influence of Intrinsic Knowledge in Long-Context Language Models](llm_efficiency/harnessing_the_unseen_the_hidden_influence_of_intrinsic_knowledge_in_long-contex.md)**

:   首次系统研究长上下文语言模型中参数知识(parametric knowledge)对生成的影响，发现其影响随上下文长度增长而增强，且现有方法提升外部检索能力会抑制参数召回能力，据此提出Hybrid Needle-in-a-Haystack测试来同时评估两种能力。

**[Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction](llm_efficiency/judge_q_trainable_queries_for_optimized_information_retention_in_kv_cache_evicti.md)**

:   提出Judge Q，在模型词表中引入可训练的soft token，训练其注意力模式对齐实际解码token的注意力模式，使其在prefill阶段能替代局部窗口查询来评估KV cache重要性，从而更好地保留全局信息，在LongBench上提升~1分，RULER上提升3+分。

---

## ✍️ 文本生成 { #nlp_generation }

**[A Coherence-Based Measure of AGI](nlp_generation/a_coherence-based_measure_of_agi.md)**

:   指出现有 AGI 评分用算术平均隐含"可补偿"假设（强项弥补弱项），提出基于广义均值连续谱的一致性度量 $\text{AGI}_{\text{AUC}}$：在补偿性参数 $p \in [-1, 1]$ 上积分，惩罚能力不均衡，暴露被算术平均掩盖的瓶颈。

**[Magnitude Matters: A Superior Class of Similarity Metrics for Holistic Semantic Understanding](nlp_generation/magnitude_matters_a_superior_class_of_similarity_metrics_for_holistic_semantic_u.md)**

:   提出两种无参数、幅度感知的向量相似度度量——Overlap Similarity (OS) 和 Hyperbolic Tangent Similarity (HTS)，在 4 个句子嵌入模型和 8 个 NLP 基准上，对分类任务（释义、推理）的 MSE 显著低于 Cosine Similarity 和 Dot Product，且无需任何额外训练开销。

**[TAPA: Training-Free Adaptation of Programmatic Agents via LLM-Guided Program Synthesis in Dynamic Environments](nlp_generation/tapas_are_free_training-free_adaptation_of_programmatic_agen.md)**

:   TAPA 将 LLM 定位为符号动作空间的"智能调制器"而非直接决策者，通过 LLM 引导的程序合成动态适配程序化 Agent 的符号动作，无需重新训练即可适应动态环境，在网络安全 DDoS 防御（77.7% 网络正常运行率）和群体智能编队控制中表现优异。

---

## 🛰️ 遥感 { #remote_sensing }

**[Consistency-based Abductive Reasoning over Perceptual Errors of Multiple Pre-trained Models in Novel Environments](remote_sensing/consistency-based_abductive_reasoning_over_perceptual_errors_of_multiple_pre-tra.md)**

:   将多个预训练感知模型在新环境中的冲突预测建模为一致性溯因推理问题，通过逻辑程序编码各模型的错误检测规则和领域约束，寻找在保持不一致率低于阈值的同时最大化预测覆盖率的最优假设，在15个航拍测试集上平均F1提升13.6%。

**[Debiasing Machine Learning Predictions for Causal Inference Without Additional Ground Truth Data](remote_sensing/debiasing_machine_learning_predictions_for_causal_inference_without_additional_g.md)**

:   针对ML卫星贫困预测因均值回归导致因果处理效应衰减的问题，提出两种无需新标注数据的后处理校正方法——线性校准校正(LCC)和Tweedie局部去收缩——使同一预测地图可在多个下游因果试验中复用（"一图多试"范式），Tweedie校正在模拟和DHS真实数据上实现近无偏的处理效应估计。

**[Machine Learning for Sustainable Rice Production: Region-Scale Monitoring of Water-Saving Practices in Punjab, India](remote_sensing/machine_learning_for_sustainable_rice_production_region-scale_monitoring_of_wate.md)**

:   提出维度分类方法将水稻节水实践识别解耦为播种维度(DSR vs PTR)和灌溉维度(AWD vs CF)两个独立二分类任务，仅使用Sentinel-1 SAR影像实现播种F1=0.80和灌溉F1=0.74，并在旁遮普邦300万+地块上进行大规模推理，地区级采纳率与政府统计高度相关（Spearman ρ=0.69）。

---

## 🎵 音频/语音 { #audio_speech }

**[AHAMask: Reliable Task Specification for Large Audio Language Models without Instructions](audio_speech/ahamask_reliable_task_specification_for_large_audio_language.md)**

:   通过对大音频语言模型（LALM）Transformer 骨干中的注意力头进行二值掩码（AHAMask），无需文本指令即可可靠触发特定声学任务功能，同时揭示了 LALM 内部存在"声学功能通路"。

**[Let the Model Learn to Feel: Mode-Guided Tonality Injection for Symbolic Music Emotion Recognition](audio_speech/let_the_model_learn_to_feel_mode-guided_tonality_injection_f.md)**

:   通过 MoGE 诊断策略系统发现 MIDIBERT 未有效编码调式-情感关联，提出 MoFi 注入框架通过 FiLM 机制将大调/小调先验注入 MIDIBERT 第 1 层（诊断确定的最弱情感信息层），在 EMOPIA 上准确率 75.2%（+11.8%），VGMIDI 上 59.1%（+11.8%），F1 提升 12.3%/15.5%。

---

## 🔄 自监督/表示学习 { #self_supervised }

**[GOAL: Geometrically Optimal Alignment for Continual Generalized Category Discovery](self_supervised/goal_geometrically_optimal_alignment_for_continual_generalized_category_discover.md)**

:   基于 Neural Collapse 理论，使用固定等角紧框架（ETF）分类器替代动态分类器，通过监督对齐和置信度引导的无监督对齐实现持续泛化类别发现，在四个基准上遗忘率降低 16.1%、新类发现提升 3.2%。

**[CAE: Hierarchical Semantic Alignment for Image Clustering](self_supervised/hierarchical_semantic_alignment_for_image_clustering.md)**

:   结合名词级（WordNet）和描述级（Flickr 图片描述）两种互补语义，通过最优传输对齐构建语义空间并自适应融合，实现 training-free 的图像聚类，在 ImageNet-1K 上准确率提升 4.2%。

---

## 📈 时间序列 { #time_series }

**[A Unified Shape-Aware Foundation Model for Time Series Classification](time_series/a_unified_shape-aware_foundation_model_for_time_series_class.md)**

:   提出 UniShape——一个面向时间序列分类的基础模型，通过 shape-aware adapter 自适应聚合多尺度判别性子序列（shapelet），并结合原型对比预训练在实例和 shape 两个层面学习可迁移的 shapelet 表示，在 128 个 UCR 数据集上以 3.1M 参数达到 SOTA（平均准确率 87.08%），同时提供良好的分类可解释性。

**[SELDON: Supernova Explosions Learned by Deep ODE Networks](time_series/seldon_supernova_explosions_learned_by_deep_ode_networks.md)**

:   提出SELDON，一种结合masked GRU-ODE编码器、隐式Neural ODE传播器和可解释高斯基函数解码器的连续时间VAE，用于稀疏、不规则采样的天文光变曲线预测，在仅观测20%数据时即可超越基线方法做出准确的多波段通量预测。

---

## 🔎 AIGC 检测 { #aigc_detection }

**[ActiShade: Activating Overshadowed Knowledge to Guide Multi-Hop Reasoning in Large Language Models](aigc_detection/actishade_activating_overshadowed_knowledge_to_guide_multi-h.md)**

:   提出ActiShade框架，通过高斯噪声扰动检测LLM在多跳推理中被"遮蔽"的关键短语，结合定制对比学习检索器获取补充文档，迭代重构查询以减少知识遮蔽导致的错误累积，在HotpotQA/2WikiMQA/MuSiQue上显著超越DRAGIN等SOTA。

---

## 📖 NLP 理解 { #nlp_understanding }

**[Language Models and Logic Programs for Trustworthy Tax Reasoning](nlp_understanding/language_models_and_logic_programs_for_trustworthy_tax_reasoning.md)**

:   将税法推理重新定义为语义解析任务，让LLM将法规文本和纳税案例翻译为Prolog逻辑程序，由符号求解器执行计算，通过金标准法规+智能检索案例示例+自一致性检查，在SARA数据集上实现86/100的正确率，并将预计部署成本降至15.78美元/人（低于美国人均报税成本的6%）。

---

## ⚛️ 物理学 { #physics }

**[Adaptive Fidelity Estimation for Quantum Programs with Graph-Guided Noise Awareness](physics/adaptive_fidelity_estimation_for_quantum_programs_with_graph.md)**

:   提出 QuFid 框架，将量子电路建模为有向无环图，通过控制流感知的随机游走刻画噪声传播，利用算子谱特征量化电路复杂度，实现自适应测量预算分配，在保持保真度精度的同时大幅减少测量次数。

---

## 🧮 科学计算 { #scientific_computing }

**[Scientific Knowledge-Guided Machine Learning for Vessel Power Prediction: A Comparative Study](scientific_computing/scientific_knowledge-guided_machine_learning_for_vessel_power_prediction_a_compa.md)**

:   提出物理基线+数据驱动残差的混合建模框架，将海试功率曲线（螺旋桨定律 $P=cV^n$）作为基线，用 XGBoost/NN/PINN 学习残差修正，在稀疏数据区域显著提升外推稳定性和物理一致性。

---

## 📡 信号/通信 { #signal_comm }

**[Toward Gaze Target Detection in Young Autistic Children](signal_comm/toward_gaze_target_detection_of_young_autistic_children.md)**

:   针对自闭症儿童注视目标检测中面部注视（6.6%）严重不足的类别不平衡问题，提出 Socially Aware Coarse-to-Fine (SACF) 框架，用微调的 Qwen2.5-VL 作为社交上下文感知门控，将输入路由到社交感知/社交无关两个专家模型，在首创的 AGT 数据集上显著提升了面部注视检测性能（Face L2 在 Sharingan 上降低 13.9%, F1 从 0.753 提升至 0.761）。

---

## 📂 其他 { #others }

**[A Fast Heuristic Search Approach for Energy-Optimal Profile Routing for Electric Vehicles](others/a_fast_heuristic_search_approach_for_energy-optimal_profile_.md)**

:   提出基于多目标A*搜索的label-setting方法（Pr-A*），在初始电量未知时高效求解电动车能耗最优路径（profile搜索），通过profile支配关系剪枝避免传统方法中复杂的profile合并操作，在大规模路网上性能接近已知初始电量的标准A*搜索。

**[A Graph-Theoretical Perspective on Law Design for Multiagent Systems](others/a_graph-theoretical_perspective_on_law_design_for_multiagent.md)**

:   将多智能体系统中的法律设计问题（包括"有用法律"和"无责任缺口法律"）形式化为超图上的顶点覆盖问题，证明了两类法律最小化问题都是NP-hard的，并给出了基于超图顶点覆盖近似算法的多项式时间近似方案。

**[A Graph-Theoretical Perspective on Law Design for Multiagent Systems](others/a_graph-theoretical_perspective_on_law_design_for_multiagent_systems.md)**

:   从图论角度研究多智能体系统中的法律设计问题，将 useful law 和 gap-free law 的最小化设计分别归约为超图的顶点覆盖问题，证明了 NP-hardness 并给出近似算法。

**[A Phase Transition for Opinion Dynamics with Competing Biases](others/a_phase_transition_for_opinion_dynamics_with_competing_biase.md)**

:   在有向随机图上建模两种对立力量（外部颠覆性偏差 vs 个体顽固性）对二元观点传播的影响，证明系统存在尖锐相变：偏差超过临界阈值 $p_c$ 时群体快速达成新共识，低于阈值则长期处于亚稳极化状态，且临界点仅由度序列的两个简单统计量决定。

**[A Switching Framework for Online Interval Scheduling with Predictions](others/a_switching_framework_for_online_interval_scheduling_with_pr.md)**

:   针对不可撤销的在线区间调度问题，提出 SemiTrust-and-Switch 框架和 SmoothMerge 随机算法，通过在信任预测和经典贪心算法之间切换/融合，在预测准确时趋近最优（一致性），预测错误时性能优雅退化（鲁棒性和平滑性），并证明了该框架在特定实例上的紧性。

**[A Topological Rewriting of Tarski's Mereogeometry](others/a_topological_rewriting_of_tarskis_mereogeometry.md)**

:   本文在Coq定理证明器中，基于λ-MM库（Leśniewski部分整体论的类型论实现），将Tarski的实体几何（geometry of solids）重新用拓扑学语言改写：先证明部分整体论的类（m-class）对应正则开集从而构成拓扑空间，再证明Tarski几何形成该拓扑的子空间并满足Hausdorff（T₂）分离性质，从而为定性空间推理提供了一个统一的、机器验证的部分整体论-几何-拓扑理论。

**[Adaptive Evidential Learning for Temporal-Semantic Robustness in Moment Retrieval](others/adaptive_evidential_learning_for_temporal-semantic_robustnes.md)**

:   提出 DEMR 框架，将深度证据回归（DER）引入视频时刻检索任务，通过 Reflective Flipped Fusion 模块缓解模态不平衡、通过 Geom-regularizer 修复原始 DER 中不确定性估计的反直觉偏差，在标准和去偏数据集上均取得了显著提升。

**[Agent-SAMA: State-Aware Mobile Assistant](others/agent-sama_state-aware_mobile_assistant.md)**

:   提出Agent-SAMA，首次将有限状态机（FSM）引入移动端GUI Agent，将UI屏幕建模为状态、用户操作建模为转移，通过四个专门化Agent协作实现状态感知的任务规划、执行验证和错误恢复，在跨App基准上成功率提升最高12%、恢复率提升13.8%。

**[Align When They Want, Complement When They Need! Human-Centered Ensembles for Adaptive Human-AI Collaboration](others/align_when_they_want_complement_when_they_need_human-centere.md)**

:   揭示了人机协作中"互补性"（complementarity）与"对齐性"（alignment）之间存在根本性权衡——单一模型无法同时优化二者，提出自适应AI集成框架，通过Rational Routing Shortcut（RRS）机制在对齐模型和互补模型之间动态切换，团队准确率较标准AI提升最高9%。

**[AMS-IO-Bench and AMS-IO-Agent: Benchmarking and Structured Reasoning for Analog and Mixed-Signal Integrated Circuit Input/Output Design](others/ams-io-bench_and_ams-io-agent_benchmarking_and_structured_re.md)**

:   提出AMS-IO-Agent，一个基于LLM的领域专用智能体，通过结构化意图图(Intent Graph)和领域知识库将自然语言设计意图转化为可生产的模拟混合信号IC I/O环设计，配套提出首个AMS I/O环自动化基准AMS-IO-Bench，在28nm CMOS流片中验证了智能体生成的I/O环可直接用于实际芯片制造。

**[An Epistemic Perspective on Agent Awareness](others/an_epistemic_perspective_on_agent_awareness.md)**

:   本文首次将 agent awareness（智能体感知/意识）视为一种知识形式，区分了 de re（关于物理对象的）和 de dicto（关于概念/描述的）两种感知模态，并基于 2D 语义学提出了一个可靠且完备的逻辑系统来刻画这两种模态与标准"事实知识"模态之间的相互作用。

**[Approximation Algorithm for Constrained k-Center Clustering: A Local Search Approach](others/approximation_algorithm_for_constrained_k-center_clustering_.md)**

:   研究带 cannot-link (CL) 和 must-link (ML) 实例级约束的 k-center 聚类问题，提出基于支配匹配集（dominating matching set, DMS）转化的局部搜索框架，在不相交 CL 集条件下首次通过局部搜索达到最优近似比 2，解决了该领域一个开放问题。

**[Area-Optimal Control Strategies for Heterogeneous Multi-Agent Pursuit](others/area-optimal_control_strategies_for_heterogeneous_multi-agen.md)**

:   研究异构速度下多追逐者-单逃避者的追逃博弈——定义逃避者安全可达集为所有追逐者-逃避者对的Apollonius圆的交集，将捕获策略建模为追逐者最小化/逃避者最大化该交集面积的零和博弈，推导出闭式最优航向控制律，仿真验证追逐者可系统性缩小安全区域实现保证捕获。

**[Bilevel MCTS for Amortized O(1) Node Selection in Classical Planning](others/bilevel_mcts_for_amortized_o1_node_selection_in_classical_planning.md)**

:   提出双层MCTS（Bilevel MCTS），在MCTS选中的叶节点处运行深度比例预算的最优优先搜索，将节点选择均摊复杂度从 $O(\log N)$ 降至 $O(1)$，辅以树崩塌（Tree Collapsing）减少动作选择步数，最终整合为 Nεbula 规划器，在IPC2018/2023基准上以192.2/230.6解题数（5min/30min）超越LAMA、DecStar、NOLAN、SM-Type-LAMA等全部SOTA。

**[Extreme Value Monte Carlo Tree Search for Classical Planning](others/extreme_value_monte_carlo_tree_search_for_classical_planning.md)**

:   利用 Peaks-Over-Threshold 极值理论（POT EVT）为经典规划中 MCTS 的 Full Bellman Backup 提供统计理论基础，提出 UCB1-Uniform bandit 算法，用均匀分布（Generalized Pareto 的特例）的 MLE 估计指导动作选择，在 Pyperplan 上以 $10^4$ 节点预算超越 GBFS 67.8 个实例、超越 Softmin-Type(h) 33.2 个实例。

**[MeshA*: Efficient Path Planning With Motion Primitives](others/mesha_efficient_path_planning_with_motion_primitives.md)**

:   提出 MeshA* 算法，将 lattice-based 路径规划从"在运动基元层面搜索"转变为"在网格单元层面搜索并同时拟合基元序列"，通过定义"扩展网格单元"（extended cell）新搜索空间，在保证完备性和最优性的同时，实现相比标准 LBA* 1.5x-2x 的运行时加速。

**[Symbolic Planning and Multi-Agent Path Finding in Extremely Dense Environments with Unassigned Agents](others/symbolic_planning_and_multi-agent_path_finding_in_extremely_dense_environments_w.md)**

:   提出 Block Rearrangement Problem (BRaP) 形式化定义，并设计五种基于配置空间搜索、PDDL 符号规划和 MAPF 的求解算法，其中 BR-LaCAM 在最大 80×80 的极端密集网格上达到 92% 成功率和毫秒级求解速度。

**[TaylorPODA: A Taylor Expansion-Based Method to Improve Post-Hoc Attributions for Opaque Models](others/taylorpoda_a_taylor_expansion-based_method_to_improve_post-hoc_attributions_for_.md)**

:   在Taylor展开框架下提出精确性(precision)、联合性(federation)、零偏差(zero-discrepancy)三个公设规范特征归因，并引入自适应属性(adaptation)通过AUP目标优化交互效应的分配权重，成为唯一同时满足所有公设和属性的事后模型无关归因方法。
