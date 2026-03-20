---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👁️ 多模态 VLM

**📷 CVPR2026** · 共 **41** 篇

**[A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks](a_closedform_solution_for_debiasing_visionlanguage.md)**

:   提出VLM去偏的闭式解，在跨模态嵌入空间中通过正交分解和Chebyshev标量化，实现Pareto最优公平性与效用损失有界，免训练免标注，覆盖零样本分类/检索/生成三大下游任务。

**[Adaptive Vision-Language Model Routing for Computer Use Agents](adaptive_visionlanguage_model_routing_for_computer.md)**

:   在CUA编排器和VLM池之间插入轻量语义路由层，通过难度分类+logprob置信度探测+记忆注入三机制，将大部分GUI操作交给小模型处理，推理成本降低78%且精度仅下降2个百分点。

**[ApET: Approximation-Error Guided Token Compression for Efficient VLMs](apet_approximation_error_token_compression.md)**

:   从信息论角度出发，通过线性近似重建每个visual token并用重建误差衡量其信息量（误差大=信息多=应保留），提出完全不依赖注意力权重的ApET框架，在LLaVA-1.5-7B上88.9%压缩保留95.2%精度，视频任务甚至达100.4%超基线，且完全兼容FlashAttention。

**[CIPHER: 用反事实对抗幻觉——扩散引导的LVLM幻觉抑制](cipher_counterfactual_diffusion_hallucination_sup.md)**

:   提出CIPHER——通过构建扩散编辑的反事实图像数据集提取视觉幻觉的低秩子空间表示，推理时将隐层状态投影远离该子空间来免训练地抑制LVLM幻觉，首次专门针对视觉诱导的幻觉而非文本诱导的幻觉。

**[CodePercept: Code-Grounded Visual STEM Perception for MLLMs](codepercept_codegrounded_visual_stem_perception_fo.md)**

:   通过感知-推理解耦缩放实验证明 MLLM 在 STEM 任务中的瓶颈是感知而非推理，提出以可执行代码为感知介质的 CodePercept 范式，构建 ICC-1M 数据集和 STEM2Code-Eval 基准，系统性提升 MLLM 的 STEM 视觉感知能力。

**[Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](continual_learning_with_visionlanguage_models_via.md)**

:   提出 SeGP-CL，通过对抗性 PGD 在旧新语义边界构造锚点样本，配合锚点引导的跨模态几何蒸馏（ACGD）和文本语义几何正则化（TSGR），在无需旧数据回放条件下保护 VLM 持续学习中的跨模态语义几何结构，五个基准上达到 SOTA。

**[DUET-VLM: Dual Stage Unified Efficient Token Reduction for VLM Training and Inference](duet_vlm_dual_stage_token_reduction.md)**

:   提出DUET-VLM双阶段视觉token压缩框架：先在视觉编码器侧通过局部聚类聚合将冗余token合并为信息保持的紧凑表示（V2V），再在语言骨干侧通过文本引导的层级自适应剪枝逐步删减低信息量token（T2V），在LLaVA-1.5-7B上67%压缩保留99%精度，89%压缩保留97%精度。

**[DTR: Dynamic Token Reweighting for Robust Vision-Language Models](dynamic_token_reweighting_for_robust_visionlanguag.md)**

:   提出DTR——首个通过KV cache优化防御多模态越狱攻击的方法：利用反转安全偏移（Reversal Safety-Relevant Shift）识别对抗性视觉token，通过动态重加权衰减其影响，仅4步优化即可在不依赖图生文转换的前提下，大幅降低攻击成功率（HADES S+T+A: 56.9%→15.9%）同时保持VLM性能和推理效率。

**[EmoVerse: A MLLMs-Driven Emotion Representation Dataset for Interpretable Visual Emotion Analysis](emoverse_mllm_emotion_representation_dataset.md)**

:   构建了 EmoVerse——首个大规模开源可解释视觉情感分析数据集（219k+ 图像），采用知识图谱启发的多层标注框架，将情感分解为 Background-Attribute-Subject (B-A-S) 三元组并将每个元素接地到视觉区域，提供词级和主体级情感推理；同时引入兼容离散情感类别（CES）和连续维度情感空间（DES）的双重标注，并提出可解释模型将视觉线索映射到 DES 表示。

**[EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards](evolmm_self_evolving_lmm_continuous_rewards.md)**

:   提出 EvoLMM，一个完全无监督的 LMM 自进化框架——从单一 backbone 模型实例化两个协作 Agent（Proposer 生成图像相关问题 + Solver 通过内部一致性求解），通过连续自奖励过程提升推理能力，仅用原始训练图像（无标注数据或外部奖励模型）在 ChartQA、MathVista、MathVision 上获得约 3% 的一致提升。

**[Evolving Prompt Adaptation for Vision-Language Models](evolving_prompt_adaptation_for_visionlanguage_mode.md)**

:   提出EvoPrompt框架，通过模态共享提示投影器(MPP)生成跨层跨模态提示，引入进化轨迹感知学习策略(将低秩更新解耦为方向+幅度，冻结历史方向仅调幅度)防止灾难性遗忘，配合特征几何正则化(FGR)防止表示坍缩，在11个数据集的base-to-novel泛化上平均HM达80.73%超越所有现有方法。

**[FlashCache: Frequency-Domain-Guided Outlier-KV-Aware Multimodal KV Cache Compression](flashcache_frequency_kv_cache_compression.md)**

:   从频域角度重新审视多模态 KV Cache 压缩，发现 KV 矩阵能量集中于低频、偏离低频主成分的"离群 KV"编码了推理关键特征，提出 FlashCache——基于频域低通滤波识别并优先保留离群 KV + 动态逐层预算分配，实现 80% KV 内存节省和 1.69× 解码加速且不损任务性能，且与 FlashAttention 兼容。

**[GACD: Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection](gacd_gradient_self_reflection_hallucination.md)**

:   通过一阶Taylor梯度估计每个token（视觉/文本/输出）对当前预测的贡献，设计GACD框架同时缓解文本-视觉偏差（增强视觉token影响力）和共现偏差（抑制与已有物体锚定的视觉token），在AMBER上提升8%总分、POPE F1提升8%，无需训练或辅助模型。

**[Geometry-Guided Camera Motion Understanding in VideoLLMs](geometryguided_camera_motion_understanding_in_vide.md)**

:   通过 benchmarking-diagnosis-injection 框架系统揭示 VideoLLM 的相机运动盲区，并利用冻结 3DFM (VGGT) 提取几何线索 + 轻量时序分类器 + 结构化提示注入，无需微调即可显著提升 VideoLLM 的细粒度相机运动理解。

**[GraphVLM: Benchmarking Vision Language Models for Multimodal Graph Learning](graphvlm_benchmark_vlm_graph_learning.md)**

:   提出 GraphVLM，首个系统评估 VLM 在多模态图学习（MMGL）中能力的基准，研究三种互补范式——VLM-as-Encoder（特征融合增强 GNN）、VLM-as-Aligner（潜空间/语言空间桥接模态）、VLM-as-Predictor（VLM 直接作为图学习 backbone），在 6 个跨领域数据集上验证，发现 VLM-as-Predictor 范式效果最好，揭示了 VLM 作为多模态图学习新基础的巨大潜力。

**[GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr_turbo_merged_checkpoint_free_teacher.md)**

:   提出GTR-Turbo——将RL训练过程中的历史checkpoint通过TIES合并为"免费教师"来引导后续RL，完全去除对GPT等昂贵外部模型的依赖，在Points24上胜率从3.5%(RL4VLM)提升至53.5%，同时训练时间减半、计算成本降低60%。

**[HIFICL: High-Fidelity In-Context Learning for Multimodal Tasks](hificl_highfidelity_incontext_learning_for_multimo.md)**

:   通过严格的注意力公式分解揭示ICL的shift effect本质上是注意力机制的解析结果，据此提出HiFICL——用可学习低秩虚拟KV对直接参数化ICL的来源而非近似其效果，在多模态基准上以极少参数量全面超越现有ICL近似方法和LoRA。

**[HoneyBee: Data Recipes for Vision-Language Reasoners](honeybee_data_recipes_vl_reasoners.md)**

:   系统性地研究了VL推理训练数据的设计空间（数据来源、干预策略、多维度缩放），基于洞察构建了250万样本的HoneyBee数据集，训练出的3B VLM在MathVerse上超越SOTA 7.8个百分点。

**[HouseMind: Tokenization Allows MLLMs to Understand, Generate and Edit Architectural Floor Plans](housemind_tokenization_mllm_floor_plan.md)**

:   提出HouseMind——通过VQ-VAE将建筑平面图离散化为房间级token，让轻量级LLM（Qwen3-0.6B）在统一框架中同时完成平面图理解、生成和编辑，在所有三项任务上全面超越现有扩散和VLM方法，且可单卡部署。

**[HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in LVLMs](hulluedit_subspace_editing_hallucination.md)**

:   提出HulluEdit——将模型隐状态分解为正交的三个子空间（视觉证据/冲突先验/残差不确定性），只在"冲突先验"子空间做编辑来抑制幻觉，数学保证视觉证据子空间完全不受影响。在POPE/CHAIR上达到SOTA幻觉抑制效果，只需单次推理。

**[KVSmooth: Mitigating Hallucination in Multi-modal Large Language Models through Key-Value Smoothing](kvsmooth_mitigating_hallucination_in_multimodal_la.md)**

:   KVSmooth 提出了一种免训练的即插即用方法，通过对 KV-Cache 中的 Key 和 Value 施加注意力行熵引导的自适应指数移动平均（EMA）平滑，将 LLaVA-1.5 的 CHAIR_S 从 41.8 降至 18.2（降低 56%），同时 F1 从 77.5 提升到 79.2。

**[LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models](llavashield_multimodal_multiturn_safety.md)**

:   针对 VLM 多模态多轮对话场景的安全问题（恶意意图隐蔽性、上下文风险累积、跨模态联合风险），构建了包含 4,484 个标注对话的 MMDS 数据集（8 大类 60 子维度风险分类），提出自动化多模态多轮红队测试框架 MMRT 和安全审计模型 LLaVAShield，在多个基准上显著优于现有内容审核工具和 SOTA VLM。

**[Mastering Negation: Boosting Grounding Models via Grouped Opposition-Based Learning](mastering_negation_boosting_grounding_models_via_g.md)**

:   当前的视觉语言检测和基础模型主要关注具有积极语义的提示，并且常常难以准确解释...

**[MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping](modes_moe_dynamic_expert_skipping.md)**

:   首个针对MoE多模态大模型的专家跳过框架MoDES,通过全局调制局部门控(GMLG)将层级重要性融入路由概率、双模态阈值(DMT)对文本/视觉token分别设定跳过策略、前沿搜索高效优化阈值,在Qwen3-VL-MoE-30B上88%专家跳过仍保留97.33%精度,prefill加速2.16×。

**[Mixture of States (MoS): Routing Token-Level Dynamics for Multimodal Generation](mos_mixture_of_states_multimodal_generation.md)**

:   提出Mixture of States (MoS)——一种新的多模态扩散模型融合范式，用可学习的token级路由器将理解塔(冻结LLM/VLM)的任意层hidden state动态路由到生成塔(DiT)的任意层，以3-5B参数在图像生成和编辑上匹配或超越20B的Qwen-Image。

**[Multimodal OCR: Parse Anything from Documents](multimodal_ocr_parse_anything_from_documents.md)**

:   提出Multimodal OCR (MOCR)范式，将文档中的文本和图形（图表、图示、UI组件等）统一解析为结构化文本表示（文本+SVG代码），训练3B参数的dots.mocr模型在OCR Arena排名仅次于Gemini 3 Pro，在olmOCR Bench达到83.9 SOTA，在image-to-SVG基准上超越Gemini 3 Pro。

**[NanoVDR: Distilling a 2B Vision-Language Retriever into a 70M Text-Only Encoder for Visual Document Retrieval](nanovdr_distilling_a_2b_visionlanguage_retriever_i.md)**

:   NanoVDR 利用查询-文档的不对称性，将 2B 参数的 VLM 文档检索器通过 pointwise cosine alignment 蒸馏成 69M 的纯文本查询编码器，在 ViDoRe 基准上保留 95.1% 的教师模型性能，查询延迟降低 50 倍，训练仅需 13 GPU 小时。

**[Overthinking Causes Hallucination: Tracing Confounder Propagation in Vision Language Models](overthinking_hallucination_confounder_propagation.md)**

:   发现VLM幻觉的新机制——"过度思考"(overthinking)：模型在中间层产生过多竞争性物体假设导致混杂因子传播到最终层，提出Overthinking Score量化层间假设多样性与不确定性的乘积，在MSCOCO上达到78.9% F1的幻觉检测性能。

**[Prune2Drive: A Plug-and-Play Framework for Accelerating Vision-Language Models in Autonomous Driving](prune2drive_vlm_accel_autonomous_driving.md)**

:   首个面向多视角自动驾驶VLM的即插即用token剪枝框架Prune2Drive，通过T-FPS（token级最远点采样）保持语义/空间多样性 + 视图自适应剪枝率优化自动分配不同视角的token预算,在DriveLM上仅保留10% token即实现6.40×prefill加速且性能仅降3%。

**[Quant Experts: Token-aware Adaptive Error Reconstruction for Large VLM Quantization](quant_experts_token_aware_vlm_quantization.md)**

:   揭示VLM中重要通道的分布和出现频率在跨模态和token间差异显著，提出基于MoE的token感知PTQ框架：共享专家补偿全局token无关误差，路由专家自适应补偿局部token依赖误差，72B模型W4A6恢复5.09%精度。

**[Reallocating Attention Across Layers to Reduce Multimodal Hallucination](reallocating_attention_reduce_hallucination.md)**

:   将多模态推理模型幻觉分解为浅层的感知偏差和深层的推理漂移两种失效模式，通过识别感知/推理功能头并选择性放大其贡献，以即插即用、无需训练的方式平均提升4.2%准确率，仅增加约1%计算开销。

**[ReasonMap: Towards Fine-Grained Visual Reasoning from Transit Maps](reasonmap_towards_finegrained_visual_reasoning_fro.md)**

:   提出ReasonMap基准——用30个城市的高分辨率地铁图+1008个人工验证问答对评估MLLM的细粒度视觉理解与空间推理能力，发现反直觉现象：开源推理模型反而不如base模型而闭源相反，揭示视觉定位（grounding）是开闭源差距的关键因素。

**[Revisiting Model Stitching In the Foundation Model Era](revisiting_model_stitching_in_the_foundation_model.md)**

:   系统研究异质视觉基础模型(CLIP/DINOv2/SigLIP2/DINOv3)之间的"可拼接性"，发现通过Final Feature Matching预训练stitch层可实现可靠拼接，且拼接模型一致超越self-stitch基线，并提出VFM Stitch Tree(VST)在仅4.3%额外开销下恢复45%的多VFM性能增益。

**[SaPaVe: Towards Active Perception and Manipulation in VLA Models for Robotics](sapave_active_perception_manipulation_vla_roboti.md)**

:   提出SaPaVe端到端主动操作框架，通过解耦相机动作和操作动作的自底向上训练策略（先学语义主动感知再学主动视角执行），配合200K相机控制数据集和3D空间知识注入，在真实世界任务中超越π0和GR00T N1高达31-40%成功率。

**[SoPE: Spherical Coordinate-Based Positional Embedding for 3D LVLMs](sope_spherical_positional_encoding_3d_lvlm.md)**

:   揭示RoPE在3D LMM中的空间感知偏差——1D光栅索引无法保持3D结构且忽略方向变化，提出球面坐标位置编码SoPE（$t,r,\theta,\phi$四维索引+多维频率分配+多尺度混合），显著提升3D布局估计和物体检测。

**[SSR2-GCD: Multi-Modal Representation Learning via Semi-Supervised Rate Reduction for Generalized Category Discovery](ssr2gcd_rate_reduction_category_discovery.md)**

:   提出SSR2-GCD框架，通过半监督率缩减(SSR2)损失替代传统对比损失来学习均匀压缩的结构化表示，并发现模态间对齐在多模态GCD中不仅不必要甚至有害，在Stanford Cars和Flowers102上分别领先SOTA 3.1%和6.3%。

**[See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles (StaR)](star_see_think_act_gui_agent_toggles.md)**

:   揭示现有多模态GUI Agent在开关控制(toggle)任务上的严重失败（GPT-5仅37% O-AMR），提出State-aware Reasoning (StaR)方法通过三步推理链（感知当前状态→分析目标状态→决定是否操作）将执行准确率提升30%+，同时不损害通用Agent能力。

**[UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression](unimmad_multimodal_moe_anomaly_detection.md)**

:   提出 UniMMAD，一个统一的多模态多类别异常检测框架，核心是 MoE 驱动的特征解压机制——编码阶段将多模态输入压缩为通用特征并抑制潜在异常，解码阶段通过稀疏门控交叉 MoE 将通用特征解压为模态特定和类别特定的重建，配合分组动态滤波和 MoE-in-MoE 结构减少 75% 参数，在 9 个数据集（3领域12模态66类别）上达到 SOTA。

**[V2Drop: Variation-aware Vision Token Dropping for Faster Large Vision-Language Models](v2drop_variation_aware_token_dropping.md)**

:   首次从token变化量视角出发，发现LLM层间变化小的"懒惰"视觉token对输出影响可忽略，提出V2Drop渐进式剪除低变化token，在图像理解上保留94.0%性能同时减少31.5%生成延迟，视频理解上保留98.6%性能减少74.2%延迟，且完全兼容FlashAttention。

**[VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion](videofusion_a_spatiotemporal_collaborative_network.md)**

:   构建M3SVD大规模红外-可见光视频数据集（220视频/15万帧），并提出VideoFusion框架，通过跨模态差分强化模块(CmDRM)+完整模态引导融合(CMGF)+双向时序共注意力(BiCAM)+变分一致性损失，实现时空协同的多模态视频融合，在融合质量和时序一致性上超越现有图像融合和视频融合方法。

**[Do Vision-Language Models Leak What They Learn? Adaptive Token-Weighted Model Inversion Attacks](vlm_model_inversion_adaptive_token_weight.md)**

:   首次系统研究 VLM 的模型反转（Model Inversion）攻击，提出一套面向 token 生成特性的反转策略（TMI/TMI-C/SMI），以及基于视觉注意力强度动态加权 token 梯度贡献的 SMI-AW 方法，在 4 种 VLM 和 3 个数据集上实现最高 61.21% 的人类评估攻击准确率，揭示了 VLM 严重的训练数据隐私泄露风险。
