<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📷 CVPR2026 论文笔记

共 **274** 篇笔记，覆盖 **25** 个领域。

## 领域概览

| 领域 | 篇数 |
|:-----|-----:|
| 👁️ [多模态 VLM](#multimodal_vlm) | 42 |
| 🏥 [医学图像](#medical_imaging) | 41 |
| 🧊 [3D 视觉](#3d_vision) | 29 |
| 🎨 [图像生成](#image_generation) | 26 |
| 📂 [其他](#others) | 25 |
| ✂️ [语义分割](#segmentation) | 20 |
| 🚗 [自动驾驶](#autonomous_driving) | 14 |
| 🤖 [机器人/具身智能](#robotics) | 14 |
| 🎬 [视频理解](#video_understanding) | 13 |
| 🎯 [目标检测](#object_detection) | 9 |
| 🧑 [人体理解](#human_understanding) | 8 |
| 📦 [模型压缩](#model_compression) | 5 |
| 💬 [LLM / NLP](#llm_nlp) | 4 |
| 🔄 [自监督/表示学习](#self_supervised) | 4 |
| 🖼️ [图像恢复](#image_restoration) | 3 |
| 🦾 [LLM Agent](#llm_agent) | 3 |
| 💡 [LLM 推理](#llm_reasoning) | 3 |
| 🛰️ [遥感](#remote_sensing) | 3 |
| 🛡️ [AI 安全](#ai_safety) | 2 |
| 🎵 [音频/语音](#audio_speech) | 1 |
| ⚖️ [对齐 / RLHF](#llm_alignment) | 1 |
| 📖 [NLP 理解](#nlp_understanding) | 1 |
| 📐 [优化/理论](#optimization) | 1 |
| 🎮 [强化学习](#reinforcement_learning) | 1 |
| 📈 [时间序列](#time_series) | 1 |

---

## 👁️ 多模态 VLM { #multimodal_vlm }

**[A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks](multimodal_vlm/a_closedform_solution_for_debiasing_visionlanguage.md)**

:   提出VLM去偏的闭式解，在跨模态嵌入空间中通过正交分解和Chebyshev标量化，实现Pareto最优公平性与效用损失有界，免训练免标注，覆盖零样本分类/检索/生成三大下游任务。

**[Adaptive Vision-Language Model Routing for Computer Use Agents](multimodal_vlm/adaptive_visionlanguage_model_routing_for_computer.md)**

:   在CUA编排器和VLM池之间插入轻量语义路由层，通过难度分类+logprob置信度探测+记忆注入三机制，将大部分GUI操作交给小模型处理，推理成本降低78%且精度仅下降2个百分点。

**[ApET: Approximation-Error Guided Token Compression for Efficient VLMs](multimodal_vlm/apet_approximation_error_token_compression.md)**

:   从信息论角度出发，通过线性近似重建每个visual token并用重建误差衡量其信息量（误差大=信息多=应保留），提出完全不依赖注意力权重的ApET框架，在LLaVA-1.5-7B上88.9%压缩保留95.2%精度，视频任务甚至达100.4%超基线，且完全兼容FlashAttention。

**[CIPHER: 用反事实对抗幻觉——扩散引导的LVLM幻觉抑制](multimodal_vlm/cipher_counterfactual_diffusion_hallucination_sup.md)**

:   提出CIPHER——通过构建扩散编辑的反事实图像数据集提取视觉幻觉的低秩子空间表示，推理时将隐层状态投影远离该子空间来免训练地抑制LVLM幻觉，首次专门针对视觉诱导的幻觉而非文本诱导的幻觉。

**[CodePercept: Code-Grounded Visual STEM Perception for MLLMs](multimodal_vlm/codepercept_codegrounded_visual_stem_perception_fo.md)**

:   通过感知-推理解耦缩放实验证明 MLLM 在 STEM 任务中的瓶颈是感知而非推理，提出以可执行代码为感知介质的 CodePercept 范式，构建 ICC-1M 数据集和 STEM2Code-Eval 基准，系统性提升 MLLM 的 STEM 视觉感知能力。

**[Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](multimodal_vlm/continual_learning_with_visionlanguage_models_via.md)**

:   提出 SeGP-CL，通过对抗性 PGD 在旧新语义边界构造锚点样本，配合锚点引导的跨模态几何蒸馏（ACGD）和文本语义几何正则化（TSGR），在无需旧数据回放条件下保护 VLM 持续学习中的跨模态语义几何结构，五个基准上达到 SOTA。

**[DUET-VLM: Dual Stage Unified Efficient Token Reduction for VLM Training and Inference](multimodal_vlm/duet_vlm_dual_stage_token_reduction.md)**

:   提出DUET-VLM双阶段视觉token压缩框架：先在视觉编码器侧通过局部聚类聚合将冗余token合并为信息保持的紧凑表示（V2V），再在语言骨干侧通过文本引导的层级自适应剪枝逐步删减低信息量token（T2V），在LLaVA-1.5-7B上67%压缩保留99%精度，89%压缩保留97%精度。

**[DTR: Dynamic Token Reweighting for Robust Vision-Language Models](multimodal_vlm/dynamic_token_reweighting_for_robust_visionlanguag.md)**

:   提出DTR——首个通过KV cache优化防御多模态越狱攻击的方法：利用反转安全偏移（Reversal Safety-Relevant Shift）识别对抗性视觉token，通过动态重加权衰减其影响，仅4步优化即可在不依赖图生文转换的前提下，大幅降低攻击成功率（HADES S+T+A: 56.9%→15.9%）同时保持VLM性能和推理效率。

**[EmoVerse: A MLLMs-Driven Emotion Representation Dataset for Interpretable Visual Emotion Analysis](multimodal_vlm/emoverse_mllm_emotion_representation_dataset.md)**

:   构建了 EmoVerse——首个大规模开源可解释视觉情感分析数据集（219k+ 图像），采用知识图谱启发的多层标注框架，将情感分解为 Background-Attribute-Subject (B-A-S) 三元组并将每个元素接地到视觉区域，提供词级和主体级情感推理；同时引入兼容离散情感类别（CES）和连续维度情感空间（DES）的双重标注，并提出可解释模型将视觉线索映射到 DES 表示。

**[EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards](multimodal_vlm/evolmm_self_evolving_lmm_continuous_rewards.md)**

:   提出 EvoLMM，一个完全无监督的 LMM 自进化框架——从单一 backbone 模型实例化两个协作 Agent（Proposer 生成图像相关问题 + Solver 通过内部一致性求解），通过连续自奖励过程提升推理能力，仅用原始训练图像（无标注数据或外部奖励模型）在 ChartQA、MathVista、MathVision 上获得约 3% 的一致提升。

**[Evolving Prompt Adaptation for Vision-Language Models](multimodal_vlm/evolving_prompt_adaptation_for_visionlanguage_mode.md)**

:   提出EvoPrompt框架，通过模态共享提示投影器(MPP)生成跨层跨模态提示，引入进化轨迹感知学习策略(将低秩更新解耦为方向+幅度，冻结历史方向仅调幅度)防止灾难性遗忘，配合特征几何正则化(FGR)防止表示坍缩，在11个数据集的base-to-novel泛化上平均HM达80.73%超越所有现有方法。

**[FlashCache: Frequency-Domain-Guided Outlier-KV-Aware Multimodal KV Cache Compression](multimodal_vlm/flashcache_frequency_kv_cache_compression.md)**

:   从频域角度重新审视多模态 KV Cache 压缩，发现 KV 矩阵能量集中于低频、偏离低频主成分的"离群 KV"编码了推理关键特征，提出 FlashCache——基于频域低通滤波识别并优先保留离群 KV + 动态逐层预算分配，实现 80% KV 内存节省和 1.69× 解码加速且不损任务性能，且与 FlashAttention 兼容。

**[GACD: Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection](multimodal_vlm/gacd_gradient_self_reflection_hallucination.md)**

:   通过一阶Taylor梯度估计每个token（视觉/文本/输出）对当前预测的贡献，设计GACD框架同时缓解文本-视觉偏差（增强视觉token影响力）和共现偏差（抑制与已有物体锚定的视觉token），在AMBER上提升8%总分、POPE F1提升8%，无需训练或辅助模型。

**[Geometry-Guided Camera Motion Understanding in VideoLLMs](multimodal_vlm/geometryguided_camera_motion_understanding_in_vide.md)**

:   通过 benchmarking-diagnosis-injection 框架系统揭示 VideoLLM 的相机运动盲区，并利用冻结 3DFM (VGGT) 提取几何线索 + 轻量时序分类器 + 结构化提示注入，无需微调即可显著提升 VideoLLM 的细粒度相机运动理解。

**[GraphVLM: Benchmarking Vision Language Models for Multimodal Graph Learning](multimodal_vlm/graphvlm_benchmark_vlm_graph_learning.md)**

:   提出 GraphVLM，首个系统评估 VLM 在多模态图学习（MMGL）中能力的基准，研究三种互补范式——VLM-as-Encoder（特征融合增强 GNN）、VLM-as-Aligner（潜空间/语言空间桥接模态）、VLM-as-Predictor（VLM 直接作为图学习 backbone），在 6 个跨领域数据集上验证，发现 VLM-as-Predictor 范式效果最好，揭示了 VLM 作为多模态图学习新基础的巨大潜力。

**[GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](multimodal_vlm/gtr_turbo_merged_checkpoint_free_teacher.md)**

:   提出GTR-Turbo——将RL训练过程中的历史checkpoint通过TIES合并为"免费教师"来引导后续RL，完全去除对GPT等昂贵外部模型的依赖，在Points24上胜率从3.5%(RL4VLM)提升至53.5%，同时训练时间减半、计算成本降低60%。

**[HIFICL: High-Fidelity In-Context Learning for Multimodal Tasks](multimodal_vlm/hificl_highfidelity_incontext_learning_for_multimo.md)**

:   通过严格的注意力公式分解揭示ICL的shift effect本质上是注意力机制的解析结果，据此提出HiFICL——用可学习低秩虚拟KV对直接参数化ICL的来源而非近似其效果，在多模态基准上以极少参数量全面超越现有ICL近似方法和LoRA。

**[HoneyBee: Data Recipes for Vision-Language Reasoners](multimodal_vlm/honeybee_data_recipes_vl_reasoners.md)**

:   系统性地研究了VL推理训练数据的设计空间（数据来源、干预策略、多维度缩放），基于洞察构建了250万样本的HoneyBee数据集，训练出的3B VLM在MathVerse上超越SOTA 7.8个百分点。

**[HouseMind: Tokenization Allows MLLMs to Understand, Generate and Edit Architectural Floor Plans](multimodal_vlm/housemind_tokenization_mllm_floor_plan.md)**

:   提出HouseMind——通过VQ-VAE将建筑平面图离散化为房间级token，让轻量级LLM（Qwen3-0.6B）在统一框架中同时完成平面图理解、生成和编辑，在所有三项任务上全面超越现有扩散和VLM方法，且可单卡部署。

**[HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in LVLMs](multimodal_vlm/hulluedit_subspace_editing_hallucination.md)**

:   提出HulluEdit——将模型隐状态分解为正交的三个子空间（视觉证据/冲突先验/残差不确定性），只在"冲突先验"子空间做编辑来抑制幻觉，数学保证视觉证据子空间完全不受影响。在POPE/CHAIR上达到SOTA幻觉抑制效果，只需单次推理。

**[KVSmooth: Mitigating Hallucination in Multi-modal Large Language Models through Key-Value Smoothing](multimodal_vlm/kvsmooth_mitigating_hallucination_in_multimodal_la.md)**

:   KVSmooth 提出了一种免训练的即插即用方法，通过对 KV-Cache 中的 Key 和 Value 施加注意力行熵引导的自适应指数移动平均（EMA）平滑，将 LLaVA-1.5 的 CHAIR_S 从 41.8 降至 18.2（降低 56%），同时 F1 从 77.5 提升到 79.2。

**[LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models](multimodal_vlm/llavashield_multimodal_multiturn_safety.md)**

:   针对 VLM 多模态多轮对话场景的安全问题（恶意意图隐蔽性、上下文风险累积、跨模态联合风险），构建了包含 4,484 个标注对话的 MMDS 数据集（8 大类 60 子维度风险分类），提出自动化多模态多轮红队测试框架 MMRT 和安全审计模型 LLaVAShield，在多个基准上显著优于现有内容审核工具和 SOTA VLM。

**[Mastering Negation: Boosting Grounding Models via Grouped Opposition-Based Learning](multimodal_vlm/mastering_negation_boosting_grounding_models_via_g.md)**

:   当前的视觉语言检测和基础模型主要关注具有积极语义的提示，并且常常难以准确解释...

**[MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping](multimodal_vlm/modes_moe_dynamic_expert_skipping.md)**

:   首个针对MoE多模态大模型的专家跳过框架MoDES,通过全局调制局部门控(GMLG)将层级重要性融入路由概率、双模态阈值(DMT)对文本/视觉token分别设定跳过策略、前沿搜索高效优化阈值,在Qwen3-VL-MoE-30B上88%专家跳过仍保留97.33%精度,prefill加速2.16×。

**[Mixture of States (MoS): Routing Token-Level Dynamics for Multimodal Generation](multimodal_vlm/mos_mixture_of_states_multimodal_generation.md)**

:   提出Mixture of States (MoS)——一种新的多模态扩散模型融合范式，用可学习的token级路由器将理解塔(冻结LLM/VLM)的任意层hidden state动态路由到生成塔(DiT)的任意层，以3-5B参数在图像生成和编辑上匹配或超越20B的Qwen-Image。

**[Multimodal OCR: Parse Anything from Documents](multimodal_vlm/multimodal_ocr_parse_anything_from_documents.md)**

:   提出Multimodal OCR (MOCR)范式，将文档中的文本和图形（图表、图示、UI组件等）统一解析为结构化文本表示（文本+SVG代码），训练3B参数的dots.mocr模型在OCR Arena排名仅次于Gemini 3 Pro，在olmOCR Bench达到83.9 SOTA，在image-to-SVG基准上超越Gemini 3 Pro。

**[NanoVDR: Distilling a 2B Vision-Language Retriever into a 70M Text-Only Encoder for Visual Document Retrieval](multimodal_vlm/nanovdr_distilling_a_2b_visionlanguage_retriever_i.md)**

:   NanoVDR 利用查询-文档的不对称性，将 2B 参数的 VLM 文档检索器通过 pointwise cosine alignment 蒸馏成 69M 的纯文本查询编码器，在 ViDoRe 基准上保留 95.1% 的教师模型性能，查询延迟降低 50 倍，训练仅需 13 GPU 小时。

**[Noise-Aware Few-Shot Learning through Bi-directional Multi-View Prompt Alignment](multimodal_vlm/noiseaware_fewshot_learning_through_bidirectional.md)**

:   提出NA-MVP框架，通过双向（clean+noise-aware）多视图prompt设计配合非平衡最优传输（UOT）实现细粒度patch-to-prompt对齐，并用经典OT对识别出的噪声样本做选择性标签修正，在噪声小样本学习场景下持续超越SOTA。

**[Overthinking Causes Hallucination: Tracing Confounder Propagation in Vision Language Models](multimodal_vlm/overthinking_hallucination_confounder_propagation.md)**

:   发现VLM幻觉的新机制——"过度思考"(overthinking)：模型在中间层产生过多竞争性物体假设导致混杂因子传播到最终层，提出Overthinking Score量化层间假设多样性与不确定性的乘积，在MSCOCO上达到78.9% F1的幻觉检测性能。

**[Prune2Drive: A Plug-and-Play Framework for Accelerating Vision-Language Models in Autonomous Driving](multimodal_vlm/prune2drive_vlm_accel_autonomous_driving.md)**

:   首个面向多视角自动驾驶VLM的即插即用token剪枝框架Prune2Drive，通过T-FPS（token级最远点采样）保持语义/空间多样性 + 视图自适应剪枝率优化自动分配不同视角的token预算,在DriveLM上仅保留10% token即实现6.40×prefill加速且性能仅降3%。

**[Quant Experts: Token-aware Adaptive Error Reconstruction for Large VLM Quantization](multimodal_vlm/quant_experts_token_aware_vlm_quantization.md)**

:   揭示VLM中重要通道的分布和出现频率在跨模态和token间差异显著，提出基于MoE的token感知PTQ框架：共享专家补偿全局token无关误差，路由专家自适应补偿局部token依赖误差，72B模型W4A6恢复5.09%精度。

**[Reallocating Attention Across Layers to Reduce Multimodal Hallucination](multimodal_vlm/reallocating_attention_reduce_hallucination.md)**

:   将多模态推理模型幻觉分解为浅层的感知偏差和深层的推理漂移两种失效模式，通过识别感知/推理功能头并选择性放大其贡献，以即插即用、无需训练的方式平均提升4.2%准确率，仅增加约1%计算开销。

**[ReasonMap: Towards Fine-Grained Visual Reasoning from Transit Maps](multimodal_vlm/reasonmap_towards_finegrained_visual_reasoning_fro.md)**

:   提出ReasonMap基准——用30个城市的高分辨率地铁图+1008个人工验证问答对评估MLLM的细粒度视觉理解与空间推理能力，发现反直觉现象：开源推理模型反而不如base模型而闭源相反，揭示视觉定位（grounding）是开闭源差距的关键因素。

**[Revisiting Model Stitching In the Foundation Model Era](multimodal_vlm/revisiting_model_stitching_in_the_foundation_model.md)**

:   系统研究异质视觉基础模型(CLIP/DINOv2/SigLIP2/DINOv3)之间的"可拼接性"，发现通过Final Feature Matching预训练stitch层可实现可靠拼接，且拼接模型一致超越self-stitch基线，并提出VFM Stitch Tree(VST)在仅4.3%额外开销下恢复45%的多VFM性能增益。

**[SaPaVe: Towards Active Perception and Manipulation in VLA Models for Robotics](multimodal_vlm/sapave_active_perception_manipulation_vla_roboti.md)**

:   提出SaPaVe端到端主动操作框架，通过解耦相机动作和操作动作的自底向上训练策略（先学语义主动感知再学主动视角执行），配合200K相机控制数据集和3D空间知识注入，在真实世界任务中超越π0和GR00T N1高达31-40%成功率。

**[SoPE: Spherical Coordinate-Based Positional Embedding for 3D LVLMs](multimodal_vlm/sope_spherical_positional_encoding_3d_lvlm.md)**

:   揭示RoPE在3D LMM中的空间感知偏差——1D光栅索引无法保持3D结构且忽略方向变化，提出球面坐标位置编码SoPE（$t,r,\theta,\phi$四维索引+多维频率分配+多尺度混合），显著提升3D布局估计和物体检测。

**[SSR2-GCD: Multi-Modal Representation Learning via Semi-Supervised Rate Reduction for Generalized Category Discovery](multimodal_vlm/ssr2gcd_rate_reduction_category_discovery.md)**

:   提出SSR2-GCD框架，通过半监督率缩减(SSR2)损失替代传统对比损失来学习均匀压缩的结构化表示，并发现模态间对齐在多模态GCD中不仅不必要甚至有害，在Stanford Cars和Flowers102上分别领先SOTA 3.1%和6.3%。

**[See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles (StaR)](multimodal_vlm/star_see_think_act_gui_agent_toggles.md)**

:   揭示现有多模态GUI Agent在开关控制(toggle)任务上的严重失败（GPT-5仅37% O-AMR），提出State-aware Reasoning (StaR)方法通过三步推理链（感知当前状态→分析目标状态→决定是否操作）将执行准确率提升30%+，同时不损害通用Agent能力。

**[UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression](multimodal_vlm/unimmad_multimodal_moe_anomaly_detection.md)**

:   提出 UniMMAD，一个统一的多模态多类别异常检测框架，核心是 MoE 驱动的特征解压机制——编码阶段将多模态输入压缩为通用特征并抑制潜在异常，解码阶段通过稀疏门控交叉 MoE 将通用特征解压为模态特定和类别特定的重建，配合分组动态滤波和 MoE-in-MoE 结构减少 75% 参数，在 9 个数据集（3领域12模态66类别）上达到 SOTA。

**[V2Drop: Variation-aware Vision Token Dropping for Faster Large Vision-Language Models](multimodal_vlm/v2drop_variation_aware_token_dropping.md)**

:   首次从token变化量视角出发，发现LLM层间变化小的"懒惰"视觉token对输出影响可忽略，提出V2Drop渐进式剪除低变化token，在图像理解上保留94.0%性能同时减少31.5%生成延迟，视频理解上保留98.6%性能减少74.2%延迟，且完全兼容FlashAttention。

**[VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion](multimodal_vlm/videofusion_a_spatiotemporal_collaborative_network.md)**

:   构建M3SVD大规模红外-可见光视频数据集（220视频/15万帧），并提出VideoFusion框架，通过跨模态差分强化模块(CmDRM)+完整模态引导融合(CMGF)+双向时序共注意力(BiCAM)+变分一致性损失，实现时空协同的多模态视频融合，在融合质量和时序一致性上超越现有图像融合和视频融合方法。

**[Do Vision-Language Models Leak What They Learn? Adaptive Token-Weighted Model Inversion Attacks](multimodal_vlm/vlm_model_inversion_adaptive_token_weight.md)**

:   首次系统研究 VLM 的模型反转（Model Inversion）攻击，提出一套面向 token 生成特性的反转策略（TMI/TMI-C/SMI），以及基于视觉注意力强度动态加权 token 梯度贡献的 SMI-AW 方法，在 4 种 VLM 和 3 个数据集上实现最高 61.21% 的人类评估攻击准确率，揭示了 VLM 严重的训练数据隐私泄露风险。

---

## 🏥 医学图像 { #medical_imaging }

**[A protocol for evaluating robustness to H&E staining variation in computational pathology models](medical_imaging/a_protocol_for_evaluating_robustness_to_he_stainin.md)**

:   提出三步评估协议（选参考染色条件→表征测试集染色属性→模拟染色条件推理），系统量化306个MSI分类模型对H&E染色差异的鲁棒性，发现鲁棒性与分类性能呈弱负相关(r=-0.28)，高性能不代表高鲁棒性。

**[A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](medical_imaging/a_semisupervised_framework_for_breast_ultrasound_s.md)**

:   通过外观描述驱动VLM免训练生成伪标签，再由双教师不确定性融合+反向对比学习细化，仅2.5%标注即可逼近全监督性能。

**[Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning](medical_imaging/accelerating_stroke_mri_with_diffusion_probabilist.md)**

:   借鉴基础模型范式，先在约4000例fastMRI多对比度脑MRI上预训练扩散模型，再用20例目标域数据微调，实现临床中风MRI的高质量加速重建，盲审读片证明2×加速下非劣于标准诊疗。

**[Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions](medical_imaging/adaptation_of_weakly_supervised_localization_in_hi.md)**

:   提出SFDA-DeP方法，受机器遗忘启发，将源自由域适应建模为迭代识别过度预测类的不确定样本并选择性降低其置信度的过程，同时联合训练像素级分类器恢复定位判别力，在跨器官/跨中心病理基准上显著优于SFDA baselines。

**[Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding](medical_imaging/addressing_data_scarcity_in_3d_trauma_detection_th.md)**

:   在仅206例标注CT中，通过patch-based MIM预训练3D U-Net + VDETR顶点RPE + 半监督一致性正则化的两阶段框架，将3D创伤检测mAP@0.50从26.36%提升至56.57%（验证集），同时冻结编码器的7类分类达94.07%准确率。

**[Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study](medical_imaging/are_generalpurpose_vision_models_all_we_need_for_2.md)**

:   在统一训练和评估协议下对比11个专用医学分割架构(SMA)和通用视觉模型(GP-VM)，发现GP-VM在三个异质医学数据集上超越大多数SMA，且Grad-CAM分析表明GP-VM无需领域特定设计即可捕获临床相关结构。

**[Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts](medical_imaging/association_of_radiologic_ppfe_change_with_mortali.md)**

:   在 NLST（n=7980）和 SUMMIT（n=8561）两个大规模肺癌筛查队列中，利用深度学习自动分割量化低剂量 CT 上 PPFE 的纵向变化（dPPFE），验证其与全因死亡率（HR=1.25/3.14）和呼吸系统发病率的独立关联。

**[Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI](medical_imaging/automated_detection_of_malignant_lesions_in_the_ov.md)**

:   系统对比 15 种 CNN 变体在卵巢癌组织病理图像五分类上的表现，选出 InceptionV3-A（ReLU）达 94% 综合指标后，用 LIME/SHAP/Integrated Gradients 三种 XAI 方法解释其决策。

**[BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](medical_imaging/biclip_bidirectional_and_consistent_languageimage.md)**

:   提出双向视觉-语言融合（BMF）和增强一致性（IAC）两个模块，让文本和图像特征可以相互修正，在标注极度稀缺（1%）和图像退化（低剂量CT噪声/运动模糊）场景下仍保持分割鲁棒性。

**[Bridging the Skill Gap in Clinical CBCT Interpretation with CBCTRepD](medical_imaging/bridging_the_skill_gap_in_clinical_cbct_interpreta.md)**

:   构建了覆盖55种口腔疾病的7,408例大规模配对CBCT-报告数据集，开发双语报告生成系统CBCTRepD，并通过多层级临床评估证明其可帮助不同经验水平的放射科医生提升报告质量。

**[CHIPS: Efficient CLIP Adaptation via Curvature-aware Hybrid Influence-based Data Selection](medical_imaging/chips_clip_adaptation_curvature_data_selection.md)**

:   从数据中心视角重新审视 CLIP 领域适配，提出 CHIPS——为每个图文对计算融合忠实性（曲率感知牛顿式对齐）、可扩展性（InfoNCE 感知曲率+JL sketching）和保留性（选择感知相关性权重）三因素的效用分数，用 30% 的数据匹配全数据集持续预训练效果，10% 的数据超越 50% 数据量的 CPT，在 17 个医学和 31 个通用基准上达到数据选择 SOTA。

**[CLoE: Expert Consistency Learning for Missing Modality Segmentation](medical_imaging/cloe_expert_consistency_learning_for_missing_modal.md)**

:   将缺失模态下的鲁棒性问题重新定义为决策级专家一致性控制，提出双分支一致性学习（全局MEC+区域REC），并通过轻量门网络将一致性分数转化为模态可靠性权重用于融合。

**[Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](medical_imaging/decoding_matters_efficient_mambabased_decoder_with.md)**

:   提出以解码器为核心的 Deco-Mamba 网络，用 Co-Attention Gate 双向融合编解码器特征、视觉状态空间模块（VSSM）建模长程依赖、可变形卷积恢复细节，并引入窗口化分布感知 KL 散度深度监督，在 7 个医学分割基准上以中等复杂度达到 SOTA。

**[Deep Learning–Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging](medical_imaging/deep_learning_based_estimation_of_blood_glucose_le.md)**

:   提出 ScleraGluNet 框架，通过五个注视方向的巩膜血管图像，结合多分支 CNN + MRFO 特征优化 + Transformer 跨视角融合，实现 93.8% 三分类精度和 MAE=6.42 mg/dL 的空腹血糖估计。

**[Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning](medical_imaging/deep_learningbased_assessment_of_the_relation_betw.md)**

:   在 8 个标注者划分的全景口腔 X 光裁剪片上，系统对比本地学习（LL）、联邦学习（FL）和集中学习（CL）在第三磨牙-下颌管重叠二分类任务上的表现，验证 FL 作为隐私保护替代方案的可行性。

**[Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography](medical_imaging/developing_foundation_models_for_universal_segment.md)**

:   构建迄今最大的全身 PET 分割数据集 PETWB-Seg11K（11041 例扫描、59831 个掩模），并提出 SegAnyPET 基础模型，实现基于 prompt 的 3D 全身 PET 通用可交互分割，在多中心、多示踪剂、多疾病场景下展现强 zero-shot 泛化能力。

**[Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](medical_imaging/diffusionbased_feature_denoising_and_using_nnmf_fo.md)**

:   将 MRI 脑肿瘤分类任务分解为 NNMF 特征提取 → 统计特征筛选 → 轻量 CNN 分类 → 特征空间扩散净化四阶段流水线，在 AutoAttack 下将鲁棒精度从基线 0.5% 提升到 59.5%。

**[Elucidating the Design Space of Arbitrary-Noise-Based Diffusion Models](medical_imaging/eda_arbitrary_noise_diffusion_design_space.md)**

:   提出 EDA 框架，将 EDM 的设计空间从纯高斯噪声扩展至任意噪声模式，通过多元高斯分布和多独立维纳过程驱动的 SDE 实现灵活噪声扩散，且证明噪声复杂度的提升不引入额外采样开销；仅用 5 步采样即可在 MRI 偏置场矫正、CT 金属伪影去除和自然图像阴影去除三项任务上取得媲美或优于百步 Refusion 和专用方法的效果。

**[EquivAnIA: A Spectral Method for Rotation-Equivariant Anisotropic Image Analysis](medical_imaging/equivania_a_spectral_method_for_rotationequivarian.md)**

:   提出EquivAnIA——基于Cake小波和Ridge滤波器的频谱方法，通过方向滤波器在傅里叶域计算角度能量分布，实现对数值旋转严格等变的各向异性图像分析，在合成和真实图像上一致优于传统角度功率谱密度方法。

**[Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning](medical_imaging/fair_lung_disease_diagnosis_from_chest_ct_via_gend.md)**

:   在 ConvNeXt 骨干上构建注意力 MIL 模型，并通过梯度反转层（GRL）对抗性地消除扫描表征中的性别信息，再配合 focal loss、子群过采样和 5-fold 集成，实现胸部 CT 四类肺疾病的公平诊断。

**[Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](medical_imaging/federated_modalityspecific_encoders_and_partially.md)**

:   提出 FedMEPD 联邦学习框架，通过模态专属编码器、部分个性化融合解码器和多锚点交叉注意力校准，同时获得最优全模态全局模型和各客户端缺失模态个性化模型。

**[Forecasting Epileptic Seizures from Contactless Camera via Cross-Species Transfer Learning](medical_imaging/forecasting_epileptic_seizures_from_contactless_ca.md)**

:   首次系统定义基于视频的癫痫发作预测任务，提出两阶段跨物种迁移学习框架——先在啮齿类癫痫视频上自监督预训练 VideoMAE，再在人类发作前视频上少样本微调——在纯视频设定下实现超过 72% 的均衡准确率。

**[GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis](medical_imaging/giim_graphbased_learning_of_inter_and_intraview_de.md)**

:   提出基于多异构图 (MHG) 的 GIIM 框架，通过四类边关系建模同一病灶跨视图动态变化和不同病灶间空间关联，并设计四种缺失视图填充策略，在 CT/MRI/乳腺 X 光三种模态上均显著优于现有方法。

**[GLEAM: A Multimodal Imaging Dataset and HAMM for Glaucoma Classification](medical_imaging/gleam_a_multimodal_imaging_dataset_and_hamm_for_gl.md)**

:   提出首个公开三模态青光眼数据集 GLEAM（SLO 眼底图 + 环乳头 OCT + 视野偏差图，标注四个疾病阶段），以及层级注意力掩码建模 (HAMM) 框架，将跨模态自监督表示学习聚焦在编码器端，实现多模态青光眼精准分类。

**[Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](medical_imaging/human_knowledge_integrated_multimodal_learning_for.md)**

:   提出域保形界(DCB)理论框架量化域间因果因子差异，并据此设计GenEval——通过知识精炼+MedGemma-4B LoRA微调，将人类专家领域知识整合到VLM中实现单源域泛化，在8个DR和2个SOZ数据集上显著超越SOTA。

**[InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](medical_imaging/invad_inversionbased_reconstructionfree_anomaly_de.md)**

:   提出"检测即加噪"范式取代传统"检测即去噪"——通过DDIM反转将图像映射到潜在噪声空间，仅用3步推理判断偏离先验分布的程度作为异常分数，无需重建，实现SOTA精度的同时推理速度达88 FPS（比OmiAD快2倍+）。

**[Marker-Based 3D Reconstruction of Aggregates with a Comparative Analysis of 2D and 3D Morphologies](medical_imaging/markerbased_3d_reconstruction_of_aggregates_with_a.md)**

:   骨料作为建筑材料装配体的主要骨架，是各种建筑、交通运输领域的重要功能部件。

**[MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification](medical_imaging/milpf_multiple_instance_learning_on_precomputed_fe.md)**

:   提出MIL-PF框架，将冻结的基础视觉编码器（DINOv2/MedSigLIP）与仅40k参数的轻量级MIL聚合头结合，通过预计算特征+双流（全局组织上下文+局部病变注意力）聚合，在大规模乳腺X线分类任务上以极低训练成本达到SOTA性能。

**[Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](medical_imaging/multimodal_classification_of_radiationinduced_cont.md)**

:   提出RICE-NET，一种多模态3D ResNet-18模型，融合纵向T1加权MRI数据与放射治疗剂量分布图，在92例胶质母细胞瘤患者队列上实现F1=0.92的RICE vs 肿瘤复发自动分类，消融实验揭示放疗剂量图是最具信息量的单模态输入。

**[Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation](medical_imaging/multimodal_protein_language_models_for_enzyme_kine.md)**

:   提出ERBA（Enzyme-Reaction Bridging Adapter），将酶动力学参数预测重新建模为与催化机制对齐的分阶段条件化问题——先通过MRCA注入底物信息捕捉分子识别，再通过G-MoE融合活性位点3D几何信息建模构象适应，并用ESDA做分布对齐保持PLM先验——在三个动力学指标上全面超越现有SOTA。

**[Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](medical_imaging/multiscale_structureguided_latent_diffusion_for_mu.md)**

:   提出MSG-LDM，一个基于潜在扩散模型的多模态MRI翻译框架，通过在潜空间中显式解耦风格和结构信息，结合高频注入（HFIB）、多模态结构特征融合（MMSF）和多尺度结构增强（MSSE）模块提取模态无关的完整结构先验来引导扩散去噪，在BraTS2020和WMH数据集上超越现有方法。

**[Novel Architecture of RPA In Oral Cancer Lesion Detection](medical_imaging/novel_architecture_of_rpa_in_oral_cancer_lesion_de.md)**

:   将软件设计模式（Singleton + Batch Processing）融入Python自动化流程，使口腔癌病变检测的推理速度相比传统RPA平台（UiPath/Automation Anywhere）提升60-100倍。

**[One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers](medical_imaging/one_model_many_budgets_elastic_latent_interfaces_f.md)**

:   提出ELIT（Elastic Latent Interface Transformer），通过在DiT中插入可变长度的潜在token接口和轻量级Read/Write交叉注意力层，将计算量与输入分辨率解耦，使单一模型支持多种推理预算，在ImageNet-1K 512px上FID和FDD分别提升35.3%和39.6%。

**[OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation](medical_imaging/orapo_oracle_rl_radiology_report_generation.md)**

:   提出 OraPO，将 GRPO 失败的困难样本探索转化为直接偏好监督（通过轻量级 oracle 步骤），配合 FactScore 奖励（提取原子临床事实+蕴含检查）提供密集可解释的句子级奖励信号，用小型 VLM 在 2-3 个数量级更少的训练数据上在 CheXpert Plus 上达到 SOTA（F1 0.341），实现单阶段纯 RL 训练的放射报告生成。

**[PhysMoDPO: Physically-Plausible Humanoid Motion with Preference Optimization](medical_imaging/physmodpo_physicallyplausible_humanoid_motion_with.md)**

:   提出PhysMoDPO，将预训练的全身控制器（WBC/DeepMimic）集成到扩散运动生成器的后训练流程中，通过物理仿真自动构造偏好对并用DPO微调，使生成运动在WBC执行后同时满足物理可行性和文本/空间条件忠实度，实现零样本迁移到Unitree G1真实机器人。

**[Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting](medical_imaging/prototypebased_knowledge_guidance_for_finegrained.md)**

:   提出ProtoSR，通过LLM驱动的管道从22.7万篇MIMIC-CXR自由文本报告中挖掘模板对齐的视觉原型知识库，并设计原型条件化迟融合模块将检索到的原型证据作为logit残差注入层级式结构化报告模型，在Rad-ReStruct基准上达到SOTA，在细粒度属性问题（L3）上提升最为显著（+72.1%相对提升）。

**[Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration](medical_imaging/reinforcing_the_weakest_links_modernizing_siena_wi.md)**

:   通过将SIENA纵向脑萎缩管线中的经典颅骨剥离(BET2)和组织分割(FAST)模块替换为深度学习方案(SynthStrip/SynthSeg)，在ADNI和PPMI两个大队列上显著增强了脑体积变化百分比(PBVC)与临床疾病进展的关联性，并将扫描顺序误差降低高达99.1%。

**[Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning](medical_imaging/residual_sodap_residual_selforganizing_domainadapt.md)**

:   提出Residual SODAP框架，在无任务ID、无数据存储的域增量学习中，联合解决表示适应（α-entmax稀疏prompt选择+残差聚合）和分类器保持（统计伪特征重放+知识蒸馏），在DR、皮肤癌和CORe50三个基准上达到SOTA。

**[Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](medical_imaging/semantic_class_distribution_learning_for_debiasing.md)**

:   提出即插即用的SCDL框架，通过学习类条件代理分布(双向对齐CDBA)+语义锚约束(SAC)来消除半监督医学图像分割中的长尾偏差，在AMOS 5%标签下DSC提升+11.62%。

**[SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](medical_imaging/semitooth_a_generalizable_semisupervised_framework.md)**

:   提出SemiTooth——多教师多学生半监督框架+更严格加权置信度约束(SWC)，用于多源CBCT牙齿分割，在新构建的MS3Toothset上mIoU达76.67%、Dice 85.69%，超越SOTA CMT(76.14%)。

**[CodeBrain: Virtual Full-stack Scanning of Brain MRI via Imputing Any Quantised Code](medical_imaging/virtual_fullstack_scanning_of_brain_mri_via_imputi.md)**

:   CodeBrain将脑MRI多模态补全(any-to-any imputation)重新定义为区域级全栈量化码预测问题：Stage I用有限标量量化(FSQ)将完整MRI集编码为紧凑code map + 模态无关公共特征，Stage II从不完整模态预测code map(用grading loss保持量化空间平滑性)，在IXI和BraTS 2023上超越5种SOTA方法，生成的模态可接近真实数据的脑肿瘤分割性能。

---

## 🧊 3D 视觉 { #3d_vision }

**[4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](3d_vision/4dequine_disentangling_motion_and_appearance_for_4.md)**

:   将马科动物4D重建解耦为运动估计(AniMoFormer时空Transformer+后优化)和外观重建(EquineGS前馈3DGS)两个子任务，用VAREN参数化模型做桥梁，仅在合成数据(VarenPoser+VarenTex)上训练即在真实数据APT-36K和AiM上达到SOTA，并能零样本泛化到斑马和驴。

**[Ada3Drift: Adaptive Training-Time Drifting for One-Step 3D Visuomotor Robotic Manipulation](3d_vision/ada3drift_adaptive_trainingtime_drifting_for_onest.md)**

:   利用计算预算不对称性，将扩散策略的迭代细化从推理时移至训练时——通过自适应漂移场将预测动作吸引向专家模式并排斥其他生成样本，从3D点云实现单步（1 NFE）高保真多模态动作生成，比扩散策略快10倍以上。

**[AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models](3d_vision/ava_bench_atomic_visual_ability_vfm.md)**

:   提出 AVA-Bench，首个显式解耦 14 种原子视觉能力（Atomic Visual Abilities）的视觉基础模型（VFM）基准——每种 AVA（如定位、深度估计、空间理解）独立测评且训练/测试分布对齐，能精确定位 VFM 的强项和弱项形成"能力指纹"，将 VFM 选型从经验猜测变为原则性工程决策；同时发现 0.5B LLM 与 7B LLM 给出相似的 VFM 排名但节省 8× GPU 算力。

**[Catalyst4D: High-Fidelity 3D-to-4D Scene Editing via Dynamic Propagation](3d_vision/catalyst4d_highfidelity_3dto4d_scene_editing_via_d.md)**

:   提出Catalyst4D框架，通过锚点运动引导(AMG)和颜色不确定性外观精炼(CUAR)两个模块，将高质量的3D静态编辑结果传播到动态4D高斯场景中，避免了直接4D编辑的运动伪影和时间不一致问题。

**[CMHANet: A Cross-Modal Hybrid Attention Network for Point Cloud Registration](3d_vision/cmhanet_a_crossmodal_hybrid_attention_network_for.md)**

:   提出CMHANet，通过三阶段混合注意力（几何self-attention→图像aggregation-attention→源-目标cross-attention）融合2D图像纹理语义与3D点云几何信息，并引入跨模态对比损失，在3DMatch/3DLoMatch上达到最优配准性能。

**[Coherent Human-Scene Reconstruction from Multi-Person Multi-View Video in a Single Pass](3d_vision/coherent_humanscene_reconstruction_from_multiperso.md)**

:   提出 CHROMM 统一框架，从多人多视图视频中一次性联合估计相机参数、场景点云和人体网格，无需外部模块或预处理数据，在 RICH 上 WA-MPJPE 达 53.1mm 且比优化方法快 8 倍以上。

**[GeodesicNVS: Probability Density Geodesic Flow Matching for Novel View Synthesis](3d_vision/geodesicnvs_flow_matching_novel_view_synthesis.md)**

:   提出Data-to-Data Flow Matching直接学习视角间确定性变换，并引入概率密度测地线正则化使流路径沿数据流形高密度区域传播，在NVS中实现更好的跨视角一致性和几何保真度。

**[GGPT: Geometry Grounded Point Transformer](3d_vision/ggpt_geometry_grounded_point_transformer.md)**

:   提出 GGPT 框架，通过改进的轻量 SfM 管线获取几何一致但稀疏的 3D 点云，再用 3D Point Transformer 在三维空间中直接融合稀疏几何引导与稠密前馈预测，实现跨架构、跨数据集的显著泛化提升。

**[Hybrid eTFCE–GRF: Exact Cluster-Size Retrieval with Analytical p-Values for Voxel-Based Morphometry](3d_vision/hybrid_etfcegrf_exact_clustersize_retrieval_with_a.md)**

:   将 eTFCE 的并查集精确聚类大小提取与 pTFCE 的解析 GRF 推断相结合，首次同时实现精确聚类大小查询和无置换检验的解析 p 值，在全脑 VBM 分析上比 R pTFCE 快 4.6–75 倍，比置换 TFCE 快三个数量级。

**[InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](3d_vision/instanthdr_singleforward_gaussian_splatting_for_hi.md)**

:   提出首个前馈HDR新视角合成方法InstantHDR，通过几何引导的外观建模和色调映射元网络，从未标定多曝光LDR图像中单次前向重建HDR 3D高斯场景，速度比优化方法快~700×，后优化版本快~20×且质量可比。

**[JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas](3d_vision/jopp3d_joint_open_vocabulary_semantic_segmentation.md)**

:   提出JOPP-3D——首个联合处理点云和全景图的开放词汇语义分割框架，通过正二十面体切向分解将全景图转为透视图后利用SAM+CLIP提取实例级语义嵌入，再经深度对应实现3D→全景语义回投，在S3DIS上以80.9% mIoU超越所有监督/无监督方法（含PointTransformerV3的73.4%），全景分割70.1% mIoU大幅领先。

**[MotionAnymesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins](3d_vision/motionanymesh_physicsgrounded_articulation_for_sim.md)**

:   提出MotionAnymesh零样本框架，通过SP4D运动学先验引导VLM消除运动学幻觉，并用物理约束轨迹优化保证无碰撞铰接，将静态3D网格自动转换为可在SAPIEN等物理引擎中直接使用的URDF数字孪生，物理可执行率达87%，远超现有方法。

**[MSGNav: Unleashing the Power of Multi-modal 3D Scene Graph for Zero-Shot Embodied Navigation](3d_vision/msgnav_multimodal_3d_scene_embodied_navigation.md)**

:   提出多模态 3D 场景图（M3DSG）——用动态分配的图像替代纯文本关系边保留视觉线索，基于此构建 MSGNav 零样本导航系统，包含关键子图选择、自适应词汇更新、闭环推理和基于可见性的视角决策模块，在 GOAT-Bench 和 HM3D-ObjNav 上取得 SOTA。

**[NERFIFY: 多智能体框架将NeRF论文自动转化为可运行代码](3d_vision/nerfify_multiagent_nerf_paper_to_code.md)**

:   提出NERFIFY——通过6项关键创新（CFG约束、GoT代码合成、引用链组件恢复、视觉反馈修复、知识增强、系统评测），将NeRF论文可靠转化为可训练的Nerfstudio插件，在无公开实现的论文上达到±0.5dB PSNR的专家级复现质量，实现时间从数周降至数分钟。

**[Node-RF: Learning Generalized Continuous Space-Time Scene Dynamics with Neural ODE-based NeRFs](3d_vision/noderf_neural_ode_nerf_continuous_spacetime_dynam.md)**

:   将Neural ODE与动态NeRF紧密耦合，通过在隐空间中用ODE建模场景状态的连续时间演化，实现了长程时间外推和跨轨迹泛化，突破了现有动态NeRF只能在训练时间范围内插的瓶颈。

**[O3N: Omnidirectional Open-Vocabulary Occupancy Prediction](3d_vision/o3n_omnidirectional_openvocabulary_occupancy_predi.md)**

:   提出O3N——首个纯视觉端到端全向开放词汇占用预测框架，通过极坐标螺旋Mamba（PsM）、占用代价聚合（OCA）和无梯度自然模态对齐（NMA）三大模块，在QuadOcc和Human360Occ上实现SOTA。

**[Pano360: Perspective to Panoramic Vision with Geometric Consistency](3d_vision/pano360_perspective_to_panoramic_vision_with_geome.md)**

:   提出Pano360，将全景拼接从传统的2D成对对齐扩展到3D摄影测量空间，利用基于Transformer的架构实现多视图全局几何一致性，在弱纹理、大视差和重复纹理等挑战场景中成功率达97.8%，并构建了包含200个真实场景的大规模数据集。

**[Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](3d_vision/panoramic_multimodal_semantic_occupancy_prediction.md)**

:   面向四足机器人构建首个全景多模态（RGB+热成像+偏振+LiDAR）语义占据数据集PanoMMOcc，并提出VoxelHound框架，通过垂直抖动补偿（VJC）和多模态信息提示融合（MIPF）模块实现鲁棒的3D占据预测，达到23.34% mIoU（+4.16%）。

**[PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis](3d_vision/physgm_large_physical_gaussian_4d_synthesis.md)**

:   提出 PhysGM，首个从单张图像前馈式联合预测 3D 高斯表示和物理属性的框架，通过物理感知重建模型预训练 + DPO 将模拟与物理合理参考视频对齐，配合 50K+ 物理标注 3D 资产数据集 PhysAssets，在一分钟内从单图生成高保真 4D 物理仿真，相比先前方法显著加速。

**[R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection# R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection](3d_vision/r4det_4d_radar_camera_fusion_3d_detection.md)**

:   提出R4Det，通过全景深度融合（PDF）、可变形门控时序融合（DGTF）和实例引导动态精炼（IGDR）三个即插即用模块，解决4D雷达-相机融合中深度估计不准、时序融合依赖ego pose、小目标检测困难的问题，在TJ4DRadSet和VoD上取得SOTA。

**[Regularizing INR with Diffusion Prior for Self-Supervised 3D Reconstruction of Neutron Computed Tomography Data](3d_vision/regularizing_inr_with_diffusion_prior_selfsupervis.md)**

:   将扩散模型先验作为正则化项引入隐式神经表示(INR)的损失函数中，构建DINR框架用于稀疏视图中子CT重建，在仅5个视角的极端稀疏条件下仍能保持混凝土微结构的高质量重建。

**[RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting](3d_vision/retimegs_continuous_time_4d_gaussian.md)**

:   提出 RetimeGS，解决 4D 高斯溅射在时间插值时的鬼影伪影问题——将问题诊断为时间混叠（temporal aliasing），显式定义 3D 高斯的时间行为以缓解混叠，结合光流引导初始化/监督、三重渲染监督等策略，在快速运动、非刚性变形和严重遮挡数据集上实现无鬼影、时间一致的连续时间 4D 渲染。

**[Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation](3d_vision/rewis3d_reconstruction_improves_weaklysupervised_s.md)**

:   首次将前馈3D重建(MapAnything)的几何信息作为辅助监督信号引入弱监督2D语义分割，通过双Student-Teacher架构和置信度加权的跨模态一致性损失，在4个数据集上以2-7% mIoU大幅超越SOTA——且推理时仅需2D模型。

**[Scaling View Synthesis Transformers (SVSM)](3d_vision/scaling_view_synthesis_transformers.md)**

:   首次系统研究无几何NVS Transformer的缩放定律，提出有效批量大小假设和encoder-decoder SVSM架构，证明与decoder-only LVSM相同缩放斜率但3倍计算效率，在真实世界NVS上达新SOTA。

**[SCOPE: Scene-Contextualized Incremental Few-Shot 3D Segmentation](3d_vision/scope_scenecontextualized_incremental_fewshot_3d_s.md)**

:   提出即插即用的背景引导原型增强框架SCOPE，从背景区域挖掘伪实例原型丰富新类原型表示，在ScanNet上5-shot新类IoU达23.86%(vs GW 16.88%，+6.98%)，且几乎无额外计算开销(<1MB, 0.02s)。

**[SPAN: Spatial-Projection Alignment for Monocular 3D Object Detection](3d_vision/span_spatial_projection_alignment_mono3d.md)**

:   提出SPAN即插即用几何协同约束框架，通过3D角点空间对齐和3D-2D投影对齐两个可微损失，强制解耦预测的各属性满足全局几何一致性，配合层级任务学习策略稳定训练，在KITTI上将MonoDGP的Car Moderate AP3D提升0.92%达到新SOTA。

**[Spectral Defense Against Resource-Targeting Attack in 3D Gaussian Splatting](3d_vision/spectral_defense_against_resourcetargeting_attack.md)**

:   提出联合3D频率感知高斯剪枝和2D角度各向异性正则化的频域防御，将3DGS中毒攻击导致的高斯过增长抑制最多5.92×、峰值显存降3.66×、渲染速度提升4.34×，同时保持甚至改善渲染质量。

**[WMGStereo: What Makes Good Synthetic Training Data for Zero-Shot Stereo Matching?](3d_vision/what_makes_good_synthetic_training_data_for_zerosh.md)**

:   系统研究合成立体数据集的设计空间——变换Infinigen过程化生成参数(浮动物体密度/背景/材质/相机baseline/光照等)分析其对零样本立体匹配的影响，发现"真实室内场景+浮动物体"的组合最有效；据此构建WMGStereo-150k数据集，仅用此单一数据集训练超越SceneFlow+CREStereo+TartanAir+IRS四合一(Middlebury降28%，Booster降25%)，与FoundationStereo竞争力相当。

**[Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](3d_vision/wsgg_spatiotemporal_world_scene_graph.md)**

:   提出世界场景图生成（WSGG）任务——在单目视频的每个时间戳构建包含所有交互物体（包括被遮挡/不可见的）的世界场景图，引入 ActionGenome4D 数据集（将 Action Genome 视频升级为 4D 场景+世界坐标 bbox+密集关系标注），并提出三种互补方法：PWG（持久世界图，零阶特征缓冲实现物体持久性）、MWAE（掩码世界自编码器，将不可见物体推理转化为掩码补全）和 4DST（4D 场景 Transformer，可微分的逐物体时序注意力+3D 运动和相机位姿特征）。

---

## 🎨 图像生成 { #image_generation }

**[Agentic Retoucher for Text-To-Image Generation](image_generation/agentic_retoucher_for_texttoimage_generation.md)**

:   Agentic Retoucher 将 T2I 生成后的缺陷修复重构为"感知→推理→行动"的人类式闭环决策过程，用三个协作 agent 分别做上下文感知的扭曲检测、人类对齐的诊断推理和自适应局部修复，在 GenBlemish-27K 上 plausibility 提升 2.89 分，83.2% 的结果被人类评为优于原图。

**[All-in-One Slider for Attribute Manipulation in Diffusion Models](image_generation/all_in_one_slider_attribute_manipulation.md)**

:   提出 All-in-One Slider 框架，通过在文本嵌入空间上训练一个属性稀疏自编码器（Attribute Sparse Autoencoder），将多种人脸属性解耦为稀疏的语义方向，实现单一轻量模块对 52+ 种属性的细粒度连续控制，并支持多属性组合和未见属性的零样本操控。

**[AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys](image_generation/asbridge_a_bidirectional_generative_framework_brid.md)**

:   提出 AS-Bridge，基于双向 Brownian Bridge 扩散过程建模地面巡天（LSST）与空间巡天（Euclid）观测之间的随机映射，同时实现跨巡天图像转换和稀有天文事件检测。

**[Attribution as Retrieval: Model-Agnostic AI-Generated Image Attribution](image_generation/attribution_as_retrieval_modelagnostic_aigenerated.md)**

:   将 AI 生成图像归因从分类范式重新定义为实例检索问题，提出 LIDA 框架：利用低位平面提取生成器指纹，通过无监督预训练 + 少样本适配实现开放集归因，在 GenImage 和 WildFake 上全面超越现有方法。

**[BiGain: Unified Token Compression for Joint Generation and Classification](image_generation/bigain_token_compression.md)**

:   提出BiGain——一个训练免的token压缩框架，通过频域分离（保留高频细节+低中频语义），在扩散模型加速时同时保持生成质量和分类能力。70% token合并下分类精度+7.15%且FID反而更好。

**[CDG: Guiding Diffusion Models with Semantically Degraded Conditions](image_generation/cdg_condition_degradation_guidance_diffusion.md)**

:   提出CDG替代CFG——用语义退化条件替代空null prompt作为负面引导，将引导信号从粗粒度"好vs空"变为精细"好vs差一点"，在SD3/FLUX/Qwen-Image上显著提升组合精度，零额外计算。

**[coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation](image_generation/codrawagents_a_multiagent_dialogue_framework_for_c.md)**

:   提出 coDrawAgents 交互式多智能体对话框架，通过解释器、规划器、检查器、画家四个专业智能体的闭环协作，以分治策略逐步规划布局并基于画布视觉上下文纠错，在 GenEval 上达到 0.94 的 SOTA 组合保真度。

**[CognitionCapturerPro: Towards High-Fidelity Visual Decoding from EEG/MEG via Multi-modal Information and Asymmetric Alignment](image_generation/cognitioncapturerpro_towards_highfidelity_visual_d.md)**

:   提出 CognitionCapturerPro，通过不确定性加权掩蔽、多模态融合编码器、共享主干对齐模块和多分支 IP-Adapter 扩散重建，解决 EEG 视觉解码中的保真度损失和表征偏移问题，在 THINGS-EEG 上 Top-1 检索达 61.2%、Top-5 达 90.8%。

**[ConsistCompose: Unified Multimodal Layout Control for Image Composition](image_generation/consistcompose_multimodal_layout_control.md)**

:   提出 ConsistCompose，将布局坐标直接嵌入语言 prompt 实现布局可控的多实例图像生成——通过实例-坐标绑定 prompt 和坐标感知 classifier-free guidance 将语言布局线索转化为精确空间控制，配合 340 万多实例数据集 ConsistCompose3M，在 COCO-Position 和 MS-Bench 上显著提升空间准确性同时保持身份保真度。

**[D2C: Accelerating Diffusion Model Training under Minimal Budgets via Condensation](image_generation/d2c_diffusion_dataset_condensation.md)**

:   首次将数据集压缩(Dataset Condensation)应用于扩散模型训练，提出D2C两阶段框架——Select阶段用扩散难度分数+区间采样选出紧凑子集、Attach阶段为每个样本附加文本和视觉表示——仅用0.8% ImageNet(10K图像)在40K步即达FID 4.3,比REPA快100×、比vanilla SiT快233×。

**[DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching](image_generation/disca_accelerating_video_diffusion_transformers_wi.md)**

:   DisCa 首次提出"可学习特征缓存 + 步蒸馏"兼容的加速方案：用轻量神经预测器替代传统手工缓存策略，并通过 Restricted MeanFlow 稳定大规模视频模型的蒸馏，在 HunyuanVideo 上实现 11.8× 近无损加速。

**[DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](image_generation/ditic_aligned_diffusion_transformer_for_efficient.md)**

:   将预训练文生图 DiT 适配为高效单步图像压缩解码器，通过方差引导重建流、自蒸馏对齐和潜空间条件引导三种对齐机制，在 32× 下采样的深层潜空间中实现 SOTA 感知质量，同时比现有扩散压缩方法解码快 30 倍。

**[Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking](image_generation/editing_away_the_evidence_diffusionbased_image_man.md)**

:   本文从理论和实验两方面系统分析了扩散编辑（instruction/drag/composition）如何非对抗性地破坏鲁棒隐形水印，推导出 SNR 衰减和互信息下界，揭示常规后处理鲁棒性不能推广到生成式变换。

**[DIAE: Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception](image_generation/enhancing_image_aesthetics_with_dualconditioned_di.md)**

:   提出DIAE——一个基于SD1.5的图像美学增强框架，通过多模态美学感知(MAP)将模糊的美学指令转化为HSV+轮廓图的视觉控制信号，配合"不完美配对"数据集IIAEData和双分支监督训练策略，在美学提升(LAION score +17.4%)和内容一致性(CLIP-I 0.784)上同时优于InstructPix2Pix等SOTA编辑方法。

**[FDeID-Toolbox: Face De-Identification Toolbox](image_generation/fdeidtoolbox_face_deidentification_toolbox.md)**

:   发布 FDeID-Toolbox，一个模块化人脸去识别研究工具箱，统一了数据加载、方法实现（经典到 SOTA 生成模型）、推理流水线和三维评估协议（隐私/效用/质量），解决该领域实验碎片化和结果不可比的问题。

**[Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems](image_generation/fractals_made_practical_denoising_diffusion_as_par.md)**

:   证明 DDIM 确定性反向链等价于分区迭代函数系统（PIFS），从分形几何推导出三个可计算量（收缩阈值 $L_t^*$、对角膨胀函数 $f_t(\lambda)$、全局膨胀阈值 $\lambda^{**}$），统一解释了余弦调度偏移、分辨率 logSNR 偏移、Min-SNR 损失加权和 Align Your Steps 采样调度四种经验设计选择。

**[HaltNav: Reactive Visual Halting over Lightweight Topological Priors for Robust Vision-Language Navigation](image_generation/haltnav_reactive_visual_halting_over_lightweight_t.md)**

:   提出层级导航框架 HaltNav，结合轻量文本拓扑图 (osmAG) 全局规划 + VLN 模型局部执行，并引入反应式视觉停止 (RVH) 机制在遇到未知障碍时实时中断、更新拓扑、重规划绕行，在仿真和真实机器人上均显著优于基线。

**[InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](image_generation/interedit_navigating_textguided_multihuman_3d_moti.md)**

:   首次定义多人3D运动编辑(TMME)任务，构建5161个源-目标-指令三元组的InterEdit3D数据集，提出基于同步无分类器引导的条件扩散模型InterEdit，通过语义感知规划Token对齐和交互感知频域Token对齐两个核心模块，在指令跟随(g2t R@1 30.82%)和源保持(g2s R@1 17.08%)上全面超越基线。

**[LinVideo: A Post-Training Framework towards O(n) Attention in Efficient Video Generation](image_generation/linvideo_linear_attention_video_generation.md)**

:   首个data-free后训练框架LinVideo，通过选择性转移自动选择最适合替换为线性注意力的层+任意时刻分布匹配(ADM)目标函数高效恢复性能，实现Wan 1.3B/14B的1.43-1.71×加速且质量无损，叠加4步蒸馏后达15.9-20.9×加速。

**[OARS: Process-Aware Online Alignment for Generative Real-World Image Super-Resolution](image_generation/oars_processaware_online_alignment_for_generative.md)**

:   提出了OARS框架，通过基于MLLM的过程感知奖励模型COMPASS和渐进式在线强化学习，将生成式真实世界超分辨率模型与人类视觉偏好对齐，在感知质量和保真度之间实现自适应平衡。

**[PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](image_generation/pixelrush_ultrafast_trainingfree_highresolution_im.md)**

:   PixelRush 首次实现了免训练的单步高分辨率图像生成，通过部分 DDIM 反转（只扰动到中间时间步而非全噪声）+ 少步扩散模型 + 高斯滤波 patch 融合 + 噪声注入，在单卡 A100 上 20 秒生成 4K 图像，比 SOTA 快 10-35× 且 FID 更优（50.13 vs 52.87）。

**[PROMO: Promptable Outfitting for Efficient High-Fidelity Virtual Try-On](image_generation/promo_promptable_virtual_tryon_efficient.md)**

:   PROMO基于FLUX Flow Matching DiT骨干，通过潜空间多模态条件拼接、时序自参考KV缓存、3D-RoPE分组条件、以及fine-tuned VLM风格提示系统，在去除传统参考网络的前提下实现了高保真且高效的多件服装虚拟试穿，推理速度比无加速版快2.4倍，在VITON-HD和DressCode上超越现有VTON和通用图像编辑方法。

**[RAZOR: Ratio-Aware Layer Editing for Targeted Unlearning in Vision Transformers and Diffusion Models](image_generation/razor_ratio_aware_unlearning_vit_diffusion.md)**

:   提出 RAZOR，一个轻量级、模型无关的 Transformer 定向遗忘框架——通过比率感知策略识别对遗忘目标贡献最大的层和注意力头，用正则化更新规则精确编辑这些组件，逐步扩展编辑范围避免过度编辑；在 CLIP、Stable Diffusion 和 VLM 的身份/风格/物体擦除任务上实现高精度稳定遗忘，甚至在量化后仍有效，速度远快于传统方法。

**[SegQuant: A Semantics-Aware and Generalizable Quantization Framework for Diffusion Models](image_generation/segquant_diffusion_model_quantization.md)**

:   提出 SegQuant，一个面向部署的扩散模型后训练量化框架，通过基于计算图静态分析的语义感知分段量化（SegLinear）和硬件原生的双尺度极性保持量化（DualScale），在 SD3.5、FLUX、SDXL 上实现跨架构通用的高保真 W8A8/W4A8 量化，同时保持与 TensorRT 等工业推理引擎的兼容性。

**[SOLACE: Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards](image_generation/solace_self_confidence_rewards_t2i.md)**

:   用T2I模型自身的去噪自信心（对注入噪声的恢复精度）替代外部奖励做后训练，在组合生成、文字渲染、文图对齐上获一致提升，与外部奖励互补可缓解reward hacking。

**[Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework](image_generation/taming_scorebased_denoisers_in_admm_a_convergent_p.md)**

:   提出ADMM-PnP with AC-DC去噪器，通过三阶段修正-去噪流程(自动修正+方向修正+基于分数的去噪)将扩散先验集成到ADMM原始-对偶框架中，解决了ADMM迭代与扩散训练流形的几何不匹配问题，同时在两种条件下建立了收敛保证，在7种逆问题上一致优于DAPS/DPS/DiffPIR等基线。

---

## 📂 其他 { #others }

**[A4VL: A Multi-Agent Perception-Action Alliance for Efficient Long Video Reasoning](others/a4vl_multiagent_long_video_reasoning.md)**

:   提出 A4VL，一个多 Agent 感知-行动探索联盟框架用于高效长视频推理——多个 VLM Agent 在多轮循环中进行查询特定的感知线索提取（找到最相关的视频片段）和行动探索（生成答案、交叉审查、达成共识或重新探索），在 5 个 VideoQA 基准上超越 18 个现有 VLM 和 10 个长视频推理专用方法，且推理延迟显著更低。

**[AssistMimic: Physics-Grounded Humanoid Assistance via Multi-Agent RL](others/assistmimic_physics_grounded_humanoid_assistance.md)**

:   提出 AssistMimic，一个多智能体 RL 框架，联合训练辅助者和被辅助者的物理仿真策略来模仿人-人接触式辅助动作（如扶人站起），是首个在标准基准上成功跟踪力交换辅助运动的方法。

**[BenDFM: A taxonomy and synthetic CAD dataset for manufacturability assessment in sheet metal bending](others/bendfm_a_taxonomy_and_synthetic_cad_dataset_for_ma.md)**

:   提出可制造性指标的二维分类法（配置依赖性 x 可行性/复杂度）和首个钣金弯曲合成 CAD 数据集 BenDFM（20,000 零件），基准测试显示图结构表示（UV-Net）优于点云（PointNext），且配置相关任务仍是难点。

**[Bounds on Agreement between Subjective and Objective Measurements](others/bounds_on_agreement_between_subjective_and_objecti.md)**

:   从投票的基本统计性质出发，推导了主观MOS与任意客观评估指标间PCC上界和MSE下界的解析表达式，并提出基于二项分布的投票模型BinoVotes/BinoMOS，为无投票方差数据的场景提供性能天花板估计。

**[CI-ICE: Intrinsic Concept Extraction Based on Compositional Interpretability](others/ciice_intrinsic_concept_extraction_compositional.md)**

:   提出CI-ICE新任务和HyperExpress方法，利用双曲空间的层次建模能力提取可组合的物体级/属性级内在概念，通过Horosphere投影模块保证概念嵌入空间的可组合性。

**[U-F²-CBM: CLIP-Free, Label Free, Unsupervised Concept Bottleneck Models](others/clipfree_label_free_unsupervised_concept_bottlenec.md)**

:   提出TextUnlock方法，通过训练轻量MLP将任意冻结视觉分类器的特征投射到文本嵌入空间（同时保持原分类器分布不变），无需CLIP、无需标注、无需训练线性探针，即可将任何legacy分类器转化为可解释的概念瓶颈模型——在40+架构上测试，超越甚至有监督的CLIP基CBM。

**[Deconstructing the Failure of Ideal Noise Correction: A Three-Pillar Diagnosis](others/deconstructing_the_failure_of_ideal_noise_correcti.md)**

:   通过宏观收敛态、微观梯度动力学和信息论三个层次，严格证明了即使给定完美噪声转移矩阵，前向校正（FC）仍不可避免地塌缩到与无校正相同的次优水平，根本原因在于有限样本下的记忆化和噪声信道的信息损失。

**[DirPA: Addressing Prior Shift in Imbalanced Few-shot Crop-type Classification](others/dirpa_addressing_prior_shift_in_imbalanced_fewshot.md)**

:   通过 Dirichlet 先验增强（DirPA），在少样本训练阶段主动模拟真实世界长尾类别分布，从而消除人工平衡训练集与自然不平衡测试分布之间的先验偏移，提升作物分类的鲁棒性。

**[DPCache: 去噪即路径规划——免训练扩散模型加速](others/dpcache_denoising_path_planning_diffusion_accel.md)**

:   将扩散采样加速形式化为全局路径规划问题，通过Path-Aware Cost Tensor量化路径依赖的跳步误差，用动态规划选出最优关键时间步序列，在FLUX上实现4.87×加速且ImageReward反超全步基线。

**[Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras](others/drocc_depth_region_guided_3d_occupancy.md)**

:   针对视角变换几何不对齐和语义类别空间各向异性不平衡，提出深度引导双投影视角变换器（D²-VFormer）利用MoGe-2构建非空体素掩码，和区域引导专家Transformer（R/R²-EFormer）自适应分配空间模型容量，BEVDet4D上提升7.43% mIoU。

**[ELogitNorm: Enhancing OOD Detection with Extended Logit Normalization](others/enhancing_outofdistribution_detection_with_extende.md)**

:   诊断LogitNorm的特征坍缩问题(维度坍缩+原点坍缩)，提出ELogitNorm——用到决策边界的平均距离(而非特征范数)做自适应温度缩放，无超参数、兼容所有post-hoc OOD检测方法——CIFAR-10上far-OOD AUROC提升10.48%(SCALE)，ImageNet-1K上FPR95从51.45%降至27.74%，同时改善分类精度和ECE校准。

**[Integration of deep generative Anomaly Detection algorithm in high-speed industrial line](others/integration_of_deep_generative_anomaly_detection_a.md)**

:   基于GRD-Net改进的GAN+密集瓶颈残差自编码器（DRAE），在制药BFS产线上实现半监督异常检测，用281万训练patch在500ms时间槽内完成60个patch的推理（0.17ms/patch），达到97.62%平衡准确率和96.38%的逐运行验证精度。

**[MXNorm: Reusing MXFP block scales for efficient tensor normalisation](others/mxnorm_reusing_mxfp_block_scales_for_efficient_ten.md)**

:   MXNorm 提出将 RMSNorm 与 MXFP 量化融合：利用 MXFP 量化过程中已经计算好的 block absmax 来近似 RMS 值，从而省掉单独的归一化 reduction 操作，在 Llama 3 最高 8B 参数的预训练中保持训练精度，同时在 GB200 上实现最高 2.4 倍的 kernel 加速。

**[On the Possible Detectability of Image-in-Image Steganography](others/on_the_possible_detectability_of_imageinimage_steg.md)**

:   揭示了基于可逆神经网络(INN)的图像隐写方案存在严重安全漏洞：嵌入过程本质上是一种混合过程，可通过ICA进行盲源分离，仅用8维特征+SVM即可达到84.6%检测率，而传统SRM+SVM更是达到99%以上。

**[Out of Sight, Out of Mind? Evaluating State Evolution in Video World Models](others/out_of_sight_out_of_mind_evaluating_state_evolutio.md)**

:   提出 StEvo-Bench 基准测试，通过在演化过程中插入遮挡或让相机"看向别处"来检验视频世界模型能否将状态演化与观测解耦，揭示了当前模型（包括 Veo 3、Sora 2 Pro 等）的任务成功率不到 10%，暴露了严重的"演化停止"和"不一致性"问题。

**[Proof-of-Perception: 带组合共形保证的工具使用多模态推理](others/pop_proof_of_perception_conformal_reasoning.md)**

:   提出PoP框架将多模态推理建模为可执行DAG——每个感知/逻辑节点输出共形预测集提供逐步校准的不确定性，控制器在预算约束下按需调用更多工具扩展计算，在文档/图表/多图QA上优于CoT/ReAct/PoT基线。

**[Rethinking SNN Online Training and Deployment: Gradient-Coherent Learning via Hybrid-Driven LIF Model](others/rethinking_snn_online_training_and_deployment_grad.md)**

:   提出HD-LIF(混合驱动LIF)脉冲神经元模型族，通过在阈值上下区域采用不同脉冲计算机制，理论证明其梯度可分离性和对齐性，解决SNN在线训练的前后向传播不一致问题，同时实现学习精度、内存复杂度和功耗的全阶段优化——以10×参数压缩、11×功耗降低和30% NOPs节省达到CIFAR-100上78.61%精度。

**[Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods](others/rooftop_wind_field_reconstruction_using_sparse_sen.md)**

:   建立学习-观测框架，在真实风洞PIV数据上系统比较Kriging、UNet、ViTAE和CWGAN四种方法从5-30个稀疏传感器重建屋顶全风场的能力，发现混合风向训练下DL一致优于Kriging（SSIM提升18-34%），CWGAN在鲁棒性上最优。

**[SDF-Net: Structure-Aware Disentangled Feature Learning for Optical-SAR Ship Re-identification](others/sdfnet_structureaware_disentangled_feature_learnin.md)**

:   提出SDF-Net——物理引导的结构感知解耦特征学习网络，通过中间层梯度能量提取几何结构一致性(SCL)和终端层共享/模态专用特征解耦(DFL)+无参数加法融合，在HOSS-ReID上mAP达60.9%(+3.5% vs SOTA TransOSS)。

**[SHREC: A Spectral Embedding-Based Approach for Ab-Initio Reconstruction of Helical Molecules](others/shrec_a_spectral_embeddingbased_approach_for_abini.md)**

:   SHREC利用谱嵌入技术从2D冷冻电镜投影图像直接恢复螺旋分子的投影角度（无需螺旋对称参数先验），通过证明螺旋片段投影构成一维闭合流形(同胚于圆)实现角度恢复，在TMV、VipA/VipB和MakA三个公开数据集上实现接近发表水平的高分辨率重建(3.66Å–8.23Å)。

**[SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](others/sldprtnet_a_largescale_multimodal_dataset_for_cad.md)**

:   构建SldprtNet——含242K+工业CAD零件的大规模多模态数据集，每个样本包含.sldprt/.step模型、7视角合成图、参数化建模脚本(13种命令无损编解码)和Qwen2.5-VL生成的自然语言描述，baseline实验验证多模态输入(图+文)在CAD生成上优于纯文本输入。

**[STEPH: Sparse Task Vector Mixup with Hypernetworks for Efficient Knowledge Transfer in WSI Prognosis](others/sparse_task_vector_mixup_wsi_prognosis.md)**

:   STEPH通过超网络驱动的任务向量混合(TVM)+稀疏聚合实现跨癌种WSI预后知识迁移，在13个TCGA数据集上C-Index平均0.6949（+5.14% vs 癌种特定学习，+2.01% vs ROUPKT）。

**[TrajTok: 学习轨迹Token实现更好的视频理解](others/trajtok_trajectory_token_video_understanding.md)**

:   提出TrajTok——首个端到端可微的轨迹视频tokenizer，通过隐式时空聚类将视频编码为物体轨迹token，无需外部分割/跟踪管线，在分类、检索和长视频QA上全面超越patch-based方法。

**[Watch and Learn: Learning to Use Computers from Online Videos](others/watch_and_learn_computer_use_from_videos.md)**

:   提出 Watch & Learn (W&L)，将互联网上人类使用计算机的演示视频转化为大规模可执行 UI 操作轨迹——将轨迹标注转化为逆动力学问题（从连续屏幕状态预测用户操作），通过任务感知检索和标注流水线生成 53K+ 高质量轨迹，在 OSWorld 和 WindowsAgentArena 上一致提升通用和专用 CUA 的性能，7B 模型在 WAA 15步限制下达到 SOTA。

**[EB-JDAT: Energy-based Joint Distribution Adversarial Training](others/your_classifier_can_do_more_towards_balancing_the.md)**

:   通过能量景观分析揭示AT和JEM的互补性(AT缩小clean-adv能量差→鲁棒性；JEM缩小clean-generated能量差→生成+精度)，提出EB-JDAT建模联合分布p(x,x̃,y)，用min-max能量优化对齐三种数据的能量分布——CIFAR-10上鲁棒性68.76%(AutoAttack, 超SOTA AT +10.78%)，同时保持90.39%清洁精度和竞争力的生成质量(FID=27.42)。

---

## ✂️ 语义分割 { #segmentation }

**[MEDISEG: 药物图像实例分割数据集——预防不良药物事件](segmentation/a_dataset_of_medication_images_with_instance_segme.md)**

:   构建了MEDISEG药物图像实例分割数据集（8262张图像，32类药片，含遮挡/重叠的真实场景），用YOLOv8/v9验证在3类上达99.5% mAP@0.5、32类达80.1%，并通过FsDet few-shot协议证明MEDISEG预训练比CURE数据集在遮挡场景中显著提升未见药片类别的识别（1-shot准确率0.406 vs 0.131）。

**[CLIP Is Shortsighted: Paying Attention Beyond the First Sentence](segmentation/clip_shortsighted_beyond_first_sentence.md)**

:   发现CLIP对长描述"只看第一句"的根本原因在于训练数据中长caption普遍以摘要句开头形成捷径，提出DeBias-CLIP通过去除摘要句+句子子采样+token填充来分散监督信号，实现长短文本检索双SOTA。

**[Comparative Evaluation of Traditional Methods and Deep Learning for Brain Glioma Imaging. Review Paper](segmentation/comparative_evaluation_of_traditional_methods_and.md)**

:   系统综述脑胶质瘤 MRI 分割与分类方法，比较传统方法（阈值、区域生长、聚类等）与深度学习方法（CNN 架构），结论是 CNN 在分割和分类任务上全面优于传统技术，但半自动方法因可控性更受放射科医生青睐。

**[CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation](segmentation/crossearthsar_a_sarcentric_and_billionscale_geospa.md)**

:   提出首个十亿参数级 SAR 视觉基础模型 CrossEarth-SAR，在 DINOv2 基础上引入物理引导的稀疏 MoE 架构（用方向熵、等效视数、局部粗糙度三个 SAR 物理描述符引导路由），配套 200K 级预训练数据集和 22 个子基准，在 20/22 个跨域分割任务上达到 SOTA。

**[DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime](segmentation/dsflash_panoptic_scene_graph_realtime.md)**

:   DSFlash 通过合并分割与关系预测 backbone、双向关系预测头、动态 patch 剪枝等策略，将全景场景图生成速度提升至 RTX 3090 上 56 FPS，同时在 PSG 数据集上达到 mR@50=30.9 的 SOTA 性能。

**[DSS: Discover, Segment, and Select - A Progressive Mechanism for Zero-shot Camouflaged Object Segmentation](segmentation/dss_discover_segment_select_zero_shot_cos.md)**

:   提出三阶段零样本伪装目标分割框架DSS：先用DINOv2特征聚类+部件组合发现候选区域（Discover），再用SAM分割（Segment），最后用MLLM逐对比较选最优mask（Select），无需任何训练即在四个COD基准上全面超越先前零样本方法，尤其在多实例场景中优势显著。

**[Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance](segmentation/efficient_rgbd_scene_understanding_via_multitask_a.md)**

:   提出一种高效 RGB-D 多任务场景理解网络，通过部分通道卷积融合编码器、归一化焦点通道层(NFCL)、上下文特征交互层(CFIL)和多任务自适应损失，在 NYUv2 上以 20+ FPS 同时完成语义/实例/全景分割、方向估计和场景分类。

**[Empowering Semantic-Sensitive Underwater Image Enhancement with VLM](segmentation/empowering_semanticsensitive_underwater_image_enha.md)**

:   提出 VLM 驱动的语义敏感学习策略，通过 VLM 生成目标物体描述、BLIP 构建空间语义引导图、双重引导机制（cross-attention + 语义对齐损失）注入 UIE decoder，使增强结果在感知质量和检测/分割下游任务上同时提升。

**[EReCu: Pseudo-label Evolution Fusion and Refinement with Multi-Cue Learning for Unsupervised Camouflage Detection](segmentation/erecu_pseudolabel_evolution_unsupervised_camouflage.md)**

:   提出EReCu框架，在DINO师生架构上通过多线索原生感知(MNP)提取纹理+语义先验来引导伪标签进化融合(PEF)和局部伪标签精修(LPR)，实现无标注下的伪装目标检测，在4个COD数据集上达到UCOD SOTA。

**[A2P: From 2D Alignment to 3D Plausibility for Occlusion-Robust Two-Hand Reconstruction](segmentation/from_2d_alignment_to_3d_plausibility_unifying_hete.md)**

:   解耦双手重建为2D结构对齐+3D空间交互对齐：Stage 1用Fusion Alignment Encoder隐式蒸馏Sapiens的关键点/分割/深度三种2D先验(推理时免基础模型)，Stage 2用穿透感知扩散模型+碰撞梯度引导将穿透姿态映射到物理合理配置——InterHand2.6M上MPJPE降至5.36mm(超SOTA 4DHands 2.13mm)，穿透体积降7倍。

**[GKD: Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](segmentation/gkd_generalizable_knowledge_distillation_vfm.md)**

:   提出GKD框架，通过将表示学习与任务学习解耦的多阶段蒸馏策略和查询式软蒸馏机制，从VFM（如DINOv2）中蒸馏出具有跨域泛化能力的轻量学生模型，在F2L设置下平均mIoU提升+10.6%，F2F设置下+1.9%。

**[A Mixed Diet Makes DINO An Omnivorous Vision Encoder](segmentation/mixed_diet_dino_omnivorous_encoder.md)**

:   发现DINOv2的特征在不同模态间几乎零对齐（同一场景RGB和深度图的特征相似度≈随机图像对），提出Omnivorous Vision Encoder通过跨模态对齐+冻结教师蒸馏的双目标训练，让单一编码器产出模态无关的统一特征空间。

**[Masked Representation Modeling for Domain-Adaptive Segmentation](segmentation/mrm_masked_representation_modeling_domain_adaptive.md)**

:   提出在潜在空间而非输入空间做掩码建模的辅助任务MRM，通过轻量级Rebuilder模块对编码器特征做掩码-重建并用分割损失监督，在GTA→Cityscapes上为四种UDA基线平均带来+2.3 mIoU提升，推理时零额外开销。

**[Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains](segmentation/promptdriven_lightweight_foundation_model_for_inst.md)**

:   提出SAM FTI-FDet，通过设计一个基于Transformer decoder的自提示生成器（Prompt Generator），让轻量化的TinyViT-SAM自动生成任务相关的query prompt，无需人工交互即可完成货运列车部件的实例级故障检测，在自建数据集上达到74.6 AP_box / 74.2 AP_mask。

**[RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](segmentation/rdnet_region_proportionaware_dynamic_adaptive_sali.md)**

:   提出RDNet，通过区域比例感知机制动态选择不同大小卷积核组合，结合小波域频率匹配上下文增强和跨注意力定位模块，在遥感图像显著性检测三个数据集上全面超越SOTA。

**[RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection](segmentation/rsonet_regionguided_selective_optimization_network.md)**

:   提出RSONet两阶段RGB-T显著性检测框架：先通过三分支并行编码器生成区域引导图并基于相似度比较选择主导模态，再通过选择性优化模块融合双模态特征，在VT5000/VT1000/VT821上MAE达0.020/0.014/0.021，超越27个SOTA方法。

**[SAP: Segment Any 4K Panorama](segmentation/sap_segment_any_4k_panorama.md)**

:   将全景分割重构为拓扑-记忆对齐问题，通过列优先锯齿扫描将ERP全景图转为透视伪视频序列，完美复用SAM2的流式记忆机制，在零样本4K全景分割上比vanilla SAM2平均提升+17.2 mIoU。

**[SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data](segmentation/sgma_semanticguided_modalityaware_segmentation_for.md)**

:   提出SGMA——语义引导模态感知分割框架，通过语义引导融合(SGF)降低类内变异和协调跨模态冲突，模态感知采样(MAS)平衡脆弱模态训练，在ISPRS上Average mIoU +9.20%且弱模态Last-1 mIoU +18.26%(vs SOTA IMLT)。

**[SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs](segmentation/sparrow_learning_spatial_precision_and_temporal_re.md)**

:   SPARROW通过目标特定跟踪特征(TSF)和双提示[BOX]+[SEG]定位机制增强视频MLLM的时空一致性，在MeViS上J&F +8.9、VidSTG上mIoU +5.49，可即插即用到三种backbone上。

**[VidEoMT: Your ViT is Secretly Also a Video Segmentation Model](segmentation/videomt_encoder_only_video_segmentation.md)**

:   提出encoder-only视频分割模型VidEoMT，通过查询传播和查询融合将分割与时序关联统一在单个ViT编码器中，消除所有专用追踪模块，在YouTube-VIS 2019上达到160 FPS（比CAVIS快10×+），同时AP仅差0.3。

---

## 🚗 自动驾驶 { #autonomous_driving }

**[A Prediction-as-Perception Framework for 3D Object Detection](autonomous_driving/a_predictionasperception_framework_for_3d_object_d.md)**

:   借鉴人类"预判目标位置再聚焦观察"的认知模式，将前一帧的轨迹预测结果转化为当前帧的检测query，形成预测-感知迭代闭环，在UniAD上实现跟踪精度+10%和推理速度+15%的同步提升。

**[Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation](autonomous_driving/composing_driving_worlds_through_disentangled_cont.md)**

:   提出 CompoSIA，一个基于 Wan2.1 DiT 的组合式驾驶视频模拟器，通过对场景结构（3D bbox）、物体身份（单张参考图）和自车动作（相机轨迹）三因素的显式解耦注入，实现对抗性驾驶场景的细粒度可控生成，碰撞率提升 173%。

**[FedBPrompt: Federated Domain Generalization Person Re-Identification via Body Distribution Aware Visual Prompts](autonomous_driving/fedbprompt_federated_domain_generalization_person.md)**

:   提出 FedBPrompt，将可学习视觉提示分为身体部件对齐提示（受限局部注意力处理视角错位）和全身整体提示（抑制背景干扰），并设计仅传输提示参数（~0.46M vs. 全模型~86M）的联邦微调策略，在 FedDG-ReID 上取得一致性提升。

**[IGASA: Integrated Geometry-Aware and Skip-Attention Modules for Enhanced Point Cloud Registration](autonomous_driving/igasa_integrated_geometryaware_and_skipattention_m.md)**

:   提出 IGASA 点云配准框架，通过层级金字塔架构 (HPA) + 层级跨层注意力 (HCLA) 的跳跃注意力融合 + 迭代几何感知精细化 (IGAR) 的动态一致性加权，在 3DMatch 上达到 94.6% Registration Recall（SOTA），在 KITTI 上达到 100% RR，总推理时间仅 2.763s。

**[Learning Geometric and Photometric Features from Panoramic LiDAR Scans for Outdoor Place Categorization](autonomous_driving/learning_geometric_and_photometric_features_from_p.md)**

:   构建大规模室外场景数据集MPO（含Velodyne稀疏和FARO稠密两种LiDAR点云），提出结合水平循环卷积(HCC)和行级最大池化(RWMP)的CNN架构，利用全景深度图和反射率图的多模态融合（Softmax Average），在6类室外场景分类上达97.87%准确率，显著超越传统手工特征方法。

**[LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction](autonomous_driving/lrsgs_robust_lidarreflectanceguided_salient_gaussi.md)**

:   提出LR-SGS，将LiDAR强度校准为光照不变的反射率通道附加到3D高斯体上，并设计结构感知的Salient Gaussian表示（从LiDAR几何和反射率特征点初始化）配合改进的密度控制和显著变换策略，在Waymo自动驾驶复杂场景中实现优于OmniRe的高保真重建，且高斯体更少、训练更快。

**[M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs](autonomous_driving/m2occ_resilient_3d_semantic_occupancy_prediction_f.md)**

:   针对自动驾驶中相机故障导致的不完整输入问题，提出M²-Occ框架，通过多视角掩码重建（MMR）利用相邻相机重叠视场恢复缺失特征，并引入特征记忆模块（FMM）用类级语义原型精化体素表示，在缺失后视摄像头时IoU提升4.93%，不影响全视角性能。

**[MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction](autonomous_driving/mapgclr_geospatial_contrastive_learning_of_represe.md)**

:   自动驾驶汽车依靠地图信息来了解周围的世界。

**[MetaDAT: Generalizable Trajectory Prediction via Meta Pre-training and Data-Adaptive Test-Time Updating](autonomous_driving/metadat_generalizable_trajectory_prediction_via_me.md)**

:   提出MetaDAT框架，通过元学习预训练获得适合在线适应的模型初始化，并在测试时采用动态学习率优化和困难样本驱动更新来实现跨数据集分布偏移下的轨迹预测自适应，在nuScenes/Lyft/Waymo多种跨域配置下全面超越现有TTT方法。

**[MoVieDrive: Urban Scene Synthesis with Multi-Modal Multi-View Video Diffusion Transformer](autonomous_driving/moviedrive_multimodal_multiview_video_diffusion.md)**

:   提出 MoVieDrive，首个在统一框架下实现多模态（RGB+深度+语义）多视图自动驾驶场景视频生成的扩散 Transformer 方法，通过模态共享层+模态特定层的设计和多样化条件编码，在 nuScenes 上 FVD 达到 46.8（领先 SOTA 22%），同时生成高质量的深度图和语义图。

**[RESBev: Making BEV Perception More Robust](autonomous_driving/resbev_making_bev_perception_more_robust.md)**

:   提出RESBev——一个即插即用的BEV感知鲁棒性增强框架，通过隐空间世界模型从历史干净帧预测当前BEV语义先验，再与被损坏的当前观测融合，在nuScenes上显著提升四种LSS模型在10种干扰下的平均IoU（+15~20个点）。

**[SG-NLF: Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](autonomous_driving/sgnlf_spectralgeometric_neural_fields_for_posefre.md)**

:   SG-NLF提出一种无需精确位姿的LiDAR NeRF框架，通过谱-几何混合表示解决LiDAR稀疏数据导致的几何空洞问题，利用置信感知图实现全局位姿优化，并引入对抗学习强化跨帧一致性，在nuScenes上重建质量和位姿精度分别比SOTA提升35.8%和68.8%。

**[Single Pixel Image Classification using an Ultrafast Digital Light Projector](autonomous_driving/single_pixel_image_classification_using_an_ultrafa.md)**

:   利用microLED-on-CMOS超快光投影器(330kfps)进行单像素成像(SPI)，以12×12 Hadamard pattern照明MNIST数字并用单像素检测器采集时间序列，完全跳过图像重建，直接用ELM/DNN分类实测光信号，实现1.2kfps下>90%分类精度，二分类(异常检测)精度>99%。

**[Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation](autonomous_driving/towards_balanced_multimodal_learning_in_3d_human_p.md)**

:   提出基于Shapley值的模态贡献评估+Fisher信息矩阵引导的自适应权重约束(AWC)正则化方法，解决RGB/LiDAR/mmWave/WiFi四模态融合中的模态不平衡问题，在MM-Fi数据集上MPJPE比naive fusion降低2.71mm，比最佳balancing方法降低约5mm，且不引入额外可学参数。

---

## 🤖 机器人/具身智能 { #robotics }

**[Action–Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation](robotics/actiongeometry_prediction_with_3d_geometric_prior.md)**

:   利用预训练3D几何基础模型π3作为感知骨干，融合3D几何、2D语义和本体感知特征，通过扩散模型联合预测未来动作chunk和未来3D Pointmap，仅使用RGB输入就在RoboTwin双臂基准上全面超越点云方法。

**[AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots](robotics/atomicvla_unlocking_the_potential_of_atomic_skill.md)**

:   AtomicVLA 提出统一规划-执行框架，通过Think-Act自适应切换生成任务链和原子技能抽象，用技能引导MoE（SG-MoE）构建可扩展的原子技能专家库，在LIBERO-LONG上超π₀ 10%，真实世界持续学习超基线21%且遗忘仅1.3%。

**[DAWN: Pixel Motion Diffusion is What We Need for Robot Control](robotics/dawn_pixel_motion_diffusion_robot_control.md)**

:   提出 DAWN，一个两阶段全扩散的视觉语言动作框架——Motion Director（潜扩散模型）生成稠密像素运动场作为可解释的中间表示，Action Expert（扩散 Transformer 策略）将像素运动转换为可执行机器人动作；在 CALVIN 基准上取得 SOTA（平均长度 4.00），并在真实世界单臂/双臂操控中展现强泛化能力。

**[Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation](robotics/expert_pyramid_tuning_efficient_parameter_finetuni.md)**

:   提出 Expert Pyramid Tuning (EPT)，将 CV 中的多尺度特征金字塔思想引入 MoE-LoRA 框架，通过共享元知识子空间 + 不同尺度的反卷积专家 + 对比学习任务嵌入，以仅 0.41M 参数/任务在 GLUE 上达到 87.0% 均分（超越所有 MoE-LoRA 基线）。

**[GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](robotics/gecosrt_geometryaware_continual_adaptation_for_rob.md)**

:   GeCo-SRT提出持续跨任务Sim-to-Real迁移范式，利用局部几何特征的域不变性和任务不变性，通过几何感知MoE模块提取可复用的几何知识并用专家引导的优先经验回放防遗忘，在4个操作任务上比基线平均提升52%成功率且仅需1/6数据。

**[Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics](robotics/influence_malleability_in_linearized_attention_dua.md)**

:   通过NTK框架证明线性化注意力不会收敛到无限宽度核极限（需要宽度m=Ω(κ⁶)），并提出"影响可塑性"指标量化其双面效应：注意力比ReLU网络高6-9倍的数据依赖灵活性，既能降低近似误差也增加对抗脆弱性。

**[Language-Grounded Decoupled Action Representation for Robotic Manipulation (LaDA)](robotics/lada_robotic_manipulation.md)**

:   提出LaDA框架，将连续7-DoF动作解耦为平移/旋转/夹爪三个语言锚定的语义原语，通过软标签对比学习和自适应权重策略在共享嵌入空间中对齐跨任务动作表示，在LIBERO上达93.6%成功率（0.6B参数），MimicGen上67%平均成功率，超越所有基线。

**[MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent](robotics/mergevla_crossskill_model_merging_toward_a_general.md)**

:   MergeVLA 通过诊断 VLA 模型不可合并的两大根因（LoRA 参数冲突 + action expert 自注意力导致的架构不兼容），设计了稀疏激活的 task mask 和去除自注意力的 action expert 架构，实现了多个单任务 VLA 专家的免训练合并，在 LIBERO 上达到 90.2% 成功率。

**[MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents](robotics/mindpower_enabling_theoryofmind_reasoning_in_vlmba.md)**

:   MindPower 提出了以机器人为中心的心智理论（ToM）推理框架，将感知→信念→欲望→意图→决策→行动组织为六层推理层级，并用 Mind-Reward（基于 GRPO）优化推理一致性，在决策和动作生成上分别超过 GPT-4o 12.77% 和 12.49%。

**[PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments](robotics/panoaffordancenet_towards_holistic_affordance_grou.md)**

:   提出PanoAffordanceNet，首次定义360°室内环境中的全局affordance grounding任务，通过失真感知光谱调制器(DASM)和全球面密化头(OSDH)解决ERP几何失真和稀疏激活问题，配合多级训练目标抑制语义漂移，在自建360-AGD数据集上大幅超越现有方法（KLD从2.853→1.270）。

**[RADAR: Closed-Loop Robotic Data Generation via Semantic Planning and Autonomous Causal Environment Reset](robotics/radar_closedloop_robotic_data_generation_via_seman.md)**

:   提出RADAR——一个完全自主的闭环机器人操作数据生成引擎，通过VLM语义规划+GNN策略执行+VQA成功评估+FSM驱动的LIFO因果逆序环境重置四个模块，仅需2-5个人工演示即可持续生成高保真操作数据，在仿真中复杂长horizon任务达到90%成功率。

**[RC-NF: Robot-Conditioned Normalizing Flow for Real-Time Anomaly Detection in Robotic Manipulation](robotics/rcnf_robot_conditioned_normalizing_flow_anomaly.md)**

:   提出RC-NF，一种基于条件归一化流的实时异常检测模型，通过解耦处理机器人状态和物体轨迹特征，仅需正样本无监督训练即可在100ms内检测VLA模型执行中的OOD异常，在LIBERO-Anomaly-10上以约8% AUC和10% AP的优势超越SOTA（包括GPT-5、Gemini 2.5 Pro等VLM基线）。

**[RehearseVLA: Simulated Post-Training for VLAs with Physically-Consistent World Model](robotics/rehearsevla_simulated_posttraining_world_model.md)**

:   针对 VLA 模型在数据稀缺场景下的性能退化和真实环境不可重置的限制，提出 RehearseVLA——用物理一致的世界模型模拟器替代真实物理交互进行 RL 后训练，配合 VLM 引导的即时反射器提供奖励信号和终止预测，仅用每个任务 5 个专家演示即可显著提升 VLA 在复杂操控任务上的表现。

**[The Coherence Trap: MLLM-Crafted Narratives Exploit Manipulated Visual Contexts](robotics/the_coherence_trap_when_mllmcrafted_narratives_exp.md)**

:   揭示现有多模态虚假信息检测的两个根本缺陷（低估MLLM生成的语义一致虚假叙事+依赖简单不对齐的伪影），构建441k样本的MDSM数据集（图像篡改+MLLM生成语义对齐文本），并提出AMD框架（Artifact Pre-perception + Manipulation-Oriented Reasoning），在跨域检测中达88.18 ACC / 60.25 mAP / 61.02 mIoU。

---

## 🎬 视频理解 { #video_understanding }

**[AutoGaze: Attend Before Attention — Efficient and Scalable Video Understanding via Autoregressive Gazing](video_understanding/autogaze_attend_before_attention_efficient_video.md)**

:   提出AutoGaze——在ViT/MLLM处理视频之前，用一个轻量模块自回归地选择最少的多尺度patch，减少4x-100x视觉token，加速最高19x，支持1K帧4K视频并在VideoMME达67.0%。

**[Beyond Single-Sample: Reliable Multi-Sample Distillation for Video Understanding](video_understanding/beyond_singlesample_reliable_multisample_distillat.md)**

:   提出 R-MSD 框架，通过每输入采样 K 个教师响应构建教师池，结合任务自适应质量匹配（封闭题质量加权、开放题均匀配对）和在线判别器对抗蒸馏，解决视频 LVLM 黑盒蒸馏中单样本监督不可靠的问题。

**[EgoPointVQA: Gesture-Based Egocentric Video Question Answering](video_understanding/egopointvqa_gesture_based_egocentric_video_qa.md)**

:   提出 EgoPointVQA 数据集（4000 合成 + 400 真实第一人称视频）和 HINT 方法，通过 3D 手部关键点编码为手势意图 token 并与视觉 token 交织输入 MLLM，使模型能理解用户指向手势并回答指示性问题，HINT-14B 达到 68.1% 准确率，超越 InternVL3-14B 6.6 个百分点。

**[Enhancing Accuracy of Uncertainty Estimation in Appearance-based Gaze Tracking](video_understanding/enhancing_accuracy_of_uncertainty_estimation_in_ap.md)**

:   提出基于等保序回归的后校准(post-hoc calibration)方法，仅用50个标定样本即可修正视线追踪模型在域偏移下的不确定性估计失准，并引入CPE(Coverage Probability Error)指标替代EUC正确评估不确定性质量——校准后CPE从8%-45%降至~5%，95%置信区间覆盖率从16%-67%提升至86%-89%。

**[FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking](video_understanding/fctrack_overlapaware_postassociation_correction_fo.md)**

:   提出轻量后关联校正框架 FC-Track，通过 IoA 触发的外观更新抑制和局部检测-轨迹错配重分配，将长期身份切换比例从 36.86% 降至 29.55%，同时保持 MOT17/MOT20 上的 SOTA 水平。

**[FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](video_understanding/flashmotion_fewstep_controllable_video_generation.md)**

:   提出 FlashMotion 三阶段训练框架——先训轨迹 adapter、再蒸馏少步生成器、最后用扩散+对抗混合目标微调 adapter——在少步推理下实现高质量轨迹可控视频生成，并发布 FlashBench 评估基准。

**[Real-World Point Tracking with Verifier-Guided Pseudo-Labeling](video_understanding/realworld_point_tracking_with_verifierguided_pseud.md)**

:   提出一个可学习的Verifier元模型，通过逐帧评估多个预训练tracker预测的可靠性来生成高质量伪标签，实现合成数据到真实世界的高效域适应，在四个真实世界点跟踪基准上达到SOTA。

**[FlexHook: Rethinking Two-Stage Referring-by-Tracking in RMOT](video_understanding/rethinking_twostage_referringbytracking_in_referri.md)**

:   FlexHook重新激活了两阶段RBT(Referring-by-Tracking)范式：用C-Hook从backbone直接采样目标特征(替代双编码)并注入语言条件线索，用PCD(成对对应解码器)替代CLIP余弦相似度做主动对应建模，首次让两阶段方法全面超越一阶段RMOT的SOTA——Refer-KITTI-V2上HOTA从10.32(iKUN)提升到42.53，训练仅1.91小时(2×4090)。

**[SAVA-X: Ego-to-Exo Imitation Error Detection via Scene-Adaptive View Alignment and Bidirectional Cross View Fusion](video_understanding/savax_egotoexo_imitation_error_detection_via_scene.md)**

:   提出Align-Fuse-Detect框架SAVA-X，通过Gumbel Top-K自适应采样去冗余、场景自适应视角嵌入缩小域差距、双向交叉注意力融合互补语义，在EgoMe数据集上Mean AUPRC达22.36，超越最强baseline +13.56%。

**[Semantic Satellite Communications for Synchronized Audiovisual Reconstruction](video_understanding/semantic_satellite_communications_for_synchronized.md)**

:   提出LLM驱动的自适应多模态语义卫星通信系统，通过双流生成架构(V2A/A2V)+动态知识库更新+GPT-4o决策代理，实现比强制更新基线节省约50%带宽的高保真同步音视频重建。

**[Stay in your Lane: Role Specific Queries with Overlap Suppression Loss for Dense Video Captioning](video_understanding/stay_in_lane_role_query_dense_video_captioning.md)**

:   ROS-DVC通过为DETR-based密集视频描述设计角色专用查询初始化（分离定位和描述查询）+跨任务对比对齐损失+重叠抑制损失，在YouCook2上无需预训练即达到CIDEr 39.18的SOTA，超越使用GPT-2的DDVC。

**[StreamingTOM: Streaming Token Compression for Efficient Video Understanding](video_understanding/streamingtom_streaming_token_compression_video.md)**

:   针对流式视频 VLM 面临的因果性（无法访问未来帧）和累积性（token 无界增长）两个约束，提出 StreamingTOM——一个免训练、即插即用的两阶段框架，通过因果时序缩减（减少 pre-LLM prefill）和在线量化记忆（4-bit KV-cache 存储+按需检索反量化），实现 15.7× KV-cache 压缩比、较 SOTA LiveVLM 降低 1.2× 峰值内存和 2× 更快 TTFT，在离线基准平均 63.8% 和流式基准 RVS 55.8% 达到免训练方法 SOTA。

**[VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning](video_understanding/videochatm1_collaborative_policy_planning_for_vide.md)**

:   VideoChat-M1 提出了多智能体协作策略规划（CPP）范式 + 多智能体强化学习（MARL）训练框架，让 4 个异构 VLM agent 动态生成和更新工具调用策略来理解视频，在 LongVideoBench 上超过 Gemini 2.5 Pro 3.6%，超过 GPT-4o 15.6%。

---

## 🎯 目标检测 { #object_detection }

**[ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection](object_detection/abra_teleporting_finetuned_knowledge_across_domain.md)**

:   将域适应建模为权重空间的SVD旋转对齐问题：分解域与类知识，通过闭式正交Procrustes解将源域类特定残差"传送"到无标注的目标域，实现零样本跨域类别检测。

**[DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning](object_detection/dreamvideoomni_omnimotion_controlled_multisubject.md)**

:   统一框架同时实现多主体身份定制和全运动控制（全局运动 + 局部运动 + 相机运动），通过渐进式两阶段训练（有监督微调 + 潜空间身份奖励反馈学习）解决身份保持与运动控制之间的固有冲突。

**[Evaluating Few-Shot Pill Recognition Under Visual Domain Shift](object_detection/evaluating_fewshot_pill_recognition_under_visual_d.md)**

:   从部署导向视角系统评估了小样本药丸识别在跨数据集域偏移下的表现，发现语义分类1-shot即可饱和(准确率>0.989)，但遮挡重叠场景下定位和召回急剧退化，训练数据的视觉真实性(多药丸、杂乱场景)是决定小样本泛化鲁棒性的主要因素。

**[EW-DETR: Evolving World Object Detection via Incremental Low-Rank DEtection TRansformer](object_detection/ewdetr_evolving_world_object_detection.md)**

:   提出Evolving World Object Detection (EWOD)范式和EW-DETR框架，通过增量LoRA适配器、查询范数物体性适配器和熵感知未知混合三个模块，在无需存储旧数据的条件下同时解决类别增量学习、域迁移自适应和未知目标检测，FOGS指标较现有方法提升57.24%。

**[Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection](object_detection/mitigating_memorization_in_texttoimage_diffusion_v.md)**

:   提出训练时区域感知提示增强(RAPTA)和注意力驱动多模态复制检测(ADMCD)两个互补模块，前者通过检测器proposal生成语义接地的提示变体来缓解扩散模型的训练数据记忆化，后者融合patch/CLIP/纹理三流特征实现零训练复制检测，在LAION-10k上将复制率从7.4降至2.6。

**[MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization](object_detection/mokus_leveraging_crossmodal_knowledge_transfer_for.md)**

:   提出"知识感知概念定制"新任务，发现LLM文本编码器中的知识编辑可以自然迁移到视觉生成模态（跨模态知识迁移），基于此提出MoKus框架：先用LoRA微调将稀有token绑定为视觉概念的锚表征，再通过知识编辑技术将多条自然语言知识高效映射到锚表征上，每条知识更新仅需约7秒。

**[ReHARK: Refined Hybrid Adaptive RBF Kernels for Robust One-Shot Vision-Language Adaptation](object_detection/rehark_refined_hybrid_adaptive_rbf_kernels_for_rob.md)**

:   提出ReHARK——一个训练免的CLIP one-shot适应框架，通过融合CLIP文本知识、GPT3语义描述和视觉原型构建混合先验，结合多尺度RBF核在RKHS中做全局近端正则化，在11个基准上以65.83%平均准确率刷新one-shot SOTA。

**[Show, Don't Tell: Detecting Novel Objects by Watching Human Videos](object_detection/show_dont_tell_detecting_novel_objects_by_watching.md)**

:   提出"Show, Don't Tell"范式：通过观看人类演示视频，自动构建新物体标注数据集（SODC），训练轻量级定制检测器（MOD），完全绕过语言描述和prompt engineering，在真实机器人分拣任务上成功部署。

**[SLICE: Semantic Latent Injection via Compartmentalized Embedding for Image Watermarking](object_detection/slice_semantic_latent_injection_via_compartmentali.md)**

:   提出SLICE框架，将图像语义解耦为四个因子（主体/环境/动作/细节），各自锚定到扩散模型初始噪声的不同空间分区，实现细粒度语义感知水印——不仅能检测篡改，还能精确定位被篡改的语义因子，且完全无需训练。

---

## 🧑 人体理解 { #human_understanding }

**[Bilevel Layer-Positioning LoRA for Real Image Dehazing](human_understanding/bilevel_lora_real_image_dehazing.md)**

:   提出H2C文本引导无监督损失（利用CLIP将去雾重构为语义对齐问题）和BiLaLoRA双层优化策略（自动搜索最佳LoRA注入层），实现高效且即插即用的合成到真实域去雾适配。

**[Breaking the Tuning Barrier: Zero-Hyperparameters Yield Multi-Corner Analysis Via Learned Priors](human_understanding/breaking_the_tuning_barrier_zerohyperparameters_yi.md)**

:   用TabPFN（在百万回归任务上预训练的基础模型）替代传统手工先验，实现零超参数的SRAM多角良率分析，通过注意力机制自动进行跨角知识迁移，配合自动特征选择（1152D到48D）和不确定性引导的主动学习，达到SOTA精度（MRE低至0.11%）同时降低10倍以上验证成本。

**[GeoWorld: Geometric World Models](human_understanding/geoworld_geometric_world_models.md)**

:   在V-JEPA 2中引入双曲流形表示（Hyperbolic JEPA）和几何强化学习（GRL），利用测地线距离编码层次关系，通过能量函数优化实现更稳定的长时域规划，3步规划提升约3% SR，超越GPT-5 zero-shot。

**[Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs](human_understanding/graph2eval_multimodal_task_generation_agents.md)**

:   提出 Graph2Eval，一个基于知识图谱的自动化多模态 Agent 任务生成框架——从异构外部数据源构建知识图谱作为结构化任务空间，通过子图采样和元路径引导的任务构造生成语义一致且可解的 Agent 评测任务，相比 LLM 直接生成的任务提升语义一致性 20% 和可解性 17%，并发布了 1,319 个任务的 Graph2Eval-Bench 数据集。

**[L2GTX: From Local to Global Time Series Explanations](human_understanding/l2gtx_from_local_to_global_time_series_explanation.md)**

:   提出L2GTX——一种完全模型无关的方法，通过LOMATCE提取参数化时间事件原语的局部解释，再经层次聚类合并、贪心预算选择和事件聚合，生成紧凑且忠实的类级全局时间序列解释，在6个UCR数据集上全局忠实度（R²）在不同合并粒度下保持稳定（FCN上ECG200达0.792）。

**[Mobile-VTON: High-Fidelity On-Device Virtual Try-On](human_understanding/mobile_vton_ondevice_virtual_tryon.md)**

:   首个全离线移动端扩散式虚拟试穿框架，基于TeacherNet-GarmentNet-TryonNet (TGT)架构，通过特征引导对抗蒸馏(FGA)将SD3.5 Large的能力迁移到415M参数的轻量学生网络，在VITON-HD和DressCode上以1024×768分辨率匹配甚至超越服务器端基线，端到端推理时间约80秒（小米17 Pro Max）。

**[Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](human_understanding/referencefree_image_quality_assessment_for_virtual.md)**

:   构建了大规模人工标注虚拟试穿质量数据集VTON-QBench（62,688张图像，431,800条标注），并提出VTON-IQA无参考质量评估框架，通过交错交叉注意力模块实现与人类感知高度对齐的图像级质量预测。

**[Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach](human_understanding/team_ras_in_10th_abaw_competition_multimodal_valen.md)**

:   提出三模态连续VA估计方法，首次将VLM(Qwen3-VL-4B)生成的情感行为描述嵌入作为独立模态，与GRADA人脸编码器和WavLM音频特征通过两种融合策略(DCMMOE和RAAV)组合，在Aff-Wild2上达到CCC 0.658(dev)/0.62(test)。

---

## 📦 模型压缩 { #model_compression }

**[An FPGA Implementation of Displacement Vector Search for Intra Pattern Copy in JPEG XS](model_compression/an_fpga_implementation_of_displacement_vector_sear.md)**

:   首次提出JPEG XS帧内模式复制(IPC)中位移矢量(DV)搜索模块的FPGA实现方案，采用四级流水线架构和IPC Group对齐的内存组织策略，在Xilinx Artix-7上实现38.3 Mpixels/s吞吐和277 mW功耗。

**[ARCHE: Autoregressive Residual Compression with Hyperprior and Excitation](model_compression/arche_autoregressive_residual_compression_with_hyp.md)**

:   在全卷积架构内统一层级超先验、Masked PixelCNN空间自回归、通道条件建模和SE通道激励，不使用Transformer或循环组件，以95M参数和222ms解码时间实现相对Ballé基线48% BD-Rate降低并超越VVC Intra 5.6%。

**[GeoChemAD: Benchmarking Unsupervised Geochemical Anomaly Detection for Mineral Exploration](model_compression/geochemad_benchmarking_unsupervised_geochemical_an.md)**

:   发布首个开源多区域多元素地球化学异常检测基准 GeoChemAD（8 子集，覆盖沉积物/岩屑/土壤三类采样源和 Au/Cu/Ni/W 四种目标元素），并提出 GeoChemFormer——两阶段 Transformer 框架，先学空间上下文再做元素依赖建模，平均 AUC 达 0.7712 超越所有基线。

**[HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers](model_compression/hiap_a_multigranular_stochastic_autopruning_framew.md)**

:   提出HiAP——统一宏观（整头/FFN块）和微观（头内维度/FFN神经元）的层级Gumbel-Sigmoid门控框架，在单次端到端训练中自动发现满足算力预算的高效ViT子网络，无需手动重要性排序或多阶段流程。

**[PPCL: Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers](model_compression/ppcl_pluggable_pruning_dit_distillation.md)**

:   提出 PPCL，一个专为扩散 Transformer（DiT）设计的灵活结构化剪枝框架——通过线性探测+一阶差分趋势分析识别冗余层区间，再用即插即用的师生交替蒸馏方案在单次训练中整合深度和宽度剪枝，实现 50% 参数削减仅不到 3% 客观指标下降，无需为每个压缩比重新训练。

---

## 💬 LLM / NLP { #llm_nlp }

**[EVATok: 自适应长度视频Tokenization用于高效视觉自回归生成](llm_nlp/evatok_adaptive_length_video_tokenization_for_eff.md)**

:   提出EVATok框架——通过最优token分配估计+轻量路由器+自适应tokenizer训练的三步流程，让视频tokenizer按片段复杂度自适应分配token长度，在UCF-101上节省24.4%+ token同时达到SOTA生成质量。

**[Hier-COS: Making Deep Features Hierarchy-aware via Composition of Orthogonal Subspaces](llm_nlp/hiercos_making_deep_features_hierarchyaware_via_co.md)**

:   提出Hier-COS框架，为层次标签树中的每个节点分配正交基向量，通过子空间组合（祖先基+自身基+后代基）构建层次感知向量空间（HAVS），理论保证特征空间的距离结构与层次树一致，同时提出HOPS评估指标解决现有层次化评估指标的排列不变性缺陷。

**[IAPL: Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning](llm_nlp/iapl_aigenerated_image_detection_adaptive_prompt.md)**

:   针对 AI 生成图像检测中现有方法难以泛化到未见生成器的问题，提出图像自适应提示学习（IAPL），在推理时根据每张测试图像动态调整输入到视觉编码器的 prompt——通过条件信息学习器提取伪造特征条件和测试时自适应 token 优化，在 UniversalFakeDetect 和 GenImage 数据集上分别达到 95.61% 和 96.7% 的 SOTA 平均准确率。

**[WeaveTime: 流式视频LLM的帧级逐步记忆](llm_nlp/weavetime_streaming_video_llm_memory.md)**

:   诊断出Video-LLM的核心缺陷"时间无感"——把视频当无序图像集处理，产生时序模糊和历史/当前混淆两类失效，提出WeaveTime通过轻量时序重建目标获得顺序感知能力+Past-Current动态焦点缓存实现高效流式推理，在流式基准上一致提升。

---

## 🔄 自监督/表示学习 { #self_supervised }

**[BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning](self_supervised/boss_a_bestofstrategies_selector_as_an_oracle_for.md)**

:   提出BoSS，一种通过集成多个选择策略生成候选批次、冻结backbone仅重训最后一层来快速评估性能增益、然后选取最优批次的可扩展Oracle策略，揭示当前SOTA主动学习策略在大规模多类数据集上距离最优仍有显著差距。

**[Representation Learning for Spatiotemporal Physical Systems](self_supervised/representation_learning_for_spatiotemporal_physica.md)**

:   通过在三个PDE物理系统（活性物质、剪切流、Rayleigh-Bénard对流）上对比JEPA、VideoMAE、MPP和DISCO，发现隐空间预测方法(JEPA)在物理参数估计任务上全面优于像素级预测方法(MAE/自回归模型)，MSE平均改善30-50%。

**[SpHOR: A Representation Learning Perspective on Open-set Recognition](self_supervised/sphor_a_representation_learning_perspective_on_ope.md)**

:   提出SpHOR两阶段解耦训练框架：Stage 1通过正交标签嵌入+球面约束（vMF分布）+Mixup/Label Smoothing做专为OSR设计的表征学习，Stage 2冻结特征训练分类器——在Semantic Shift Benchmark上OSCR/AUROC最高提升5.1%/5.2%，同时引入Angular Separability和Norm Separability两个新度量。

**[Vision Transformers Need More Than Registers](self_supervised/vit_need_more_than_registers.md)**

:   系统揭示ViT注意力伪影的根因是"惰性聚合"——全局注意力+粗粒度语义监督驱动模型用语义无关的背景patch作为全局语义的捷径表示，提出选择性patch特征集成方案在12个基准上跨三种监督范式一致提升性能。

---

## 🖼️ 图像恢复 { #image_restoration }

**[MAD-Avatar: Motion-Aware Animatable Gaussian Avatars Deblurring](image_restoration/motionaware_animatable_gaussian_avatars_deblurring.md)**

:   首次实现从模糊视频直接重建清晰可驱动3D高斯人体avatar：提出3D感知的物理模糊形成模型(将模糊分解为子帧SMPL运动+canonical 3DGS)，用B-spline插值+位姿变形网络建模子帧运动，帧间正则化解决运动方向歧义，在合成和真实数据集上大幅超越"2D去模糊+3DGS"两阶段方案(PSNR提升约2.5dB)。

**[POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction](image_restoration/polishing_the_sky_widefield_and_highdynamic_range.md)**

:   POLISH++在POLISH框架基础上引入分块训练+拼接策略和arcsinh非线性变换，解决了射电干涉成像中宽视场（万级像素）和高动态范围（$10^4$-$10^6$）两大实际部署难题，在T-RECS仿真数据上大幅超越CLEAN方法的源探测精度，且能超分辨恢复PSF尺度附近的强引力透镜系统，有望将DSA巡天的透镜发现数量提升约10倍。

**[RAW-Domain Degradation Models for Realistic Smartphone Super-Resolution](image_restoration/rawdomain_degradation_models_smartphone_sr.md)**

:   通过对不同智能手机传感器进行设备特定的退化标定（模糊PSF和噪声模型），将公开渲染图像逆处理（unprocess）到各手机的RAW域来生成逼真的训练对，训练的RAW-to-RGB SR模型在未见设备上的真实数据上显著优于使用任意退化参数的基线。

---

## 🦾 LLM Agent { #llm_agent }

**[HATS: Hardness-Aware Trajectory Synthesis for GUI Agents](llm_agent/hats_hardnessaware_trajectory_synthesis_gui_agent.md)**

:   提出HATS框架，通过定义动作的"语义模糊度"作为难度信号，以难度驱动探索+对齐引导修复的闭环管线合成高质量GUI轨迹数据，显著提升agent泛化能力。

**[REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting](llm_agent/realm_mllm_agent_3d_reasoning_gaussian.md)**

:   提出 REALM，一个基于 MLLM-Agent 的开放世界 3D 推理分割框架，利用 3DGS 渲染新视角供 MLLM 理解复杂指令，通过全局到局部空间定位策略实现精确 3D 分割——无需 3D 特定后训练即可处理隐式推理指令，并支持物体移除、替换和风格迁移等 3D 交互任务。

**[SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation](llm_agent/sceneassistant_a_visual_feedback_agent_for_openvoc.md)**

:   提出基于视觉反馈的VLM agent框架，通过14个完备Action API让VLM在ReAct闭环中迭代优化3D场景布局，无需预定义空间关系模板，在人类评估中Layout得分7.600（vs SceneWeaver 5.800），Human Preference 65%。

---

## 💡 LLM 推理 { #llm_reasoning }

**[FaceCoT: Chain-of-Thought Reasoning in MLLMs for Face Anti-Spoofing](llm_reasoning/facecot_cot_reasoning_face_anti_spoofing.md)**

:   构建了首个面向人脸反欺骗（FAS）的大规模 VQA 数据集 FaceCoT（108 万样本，覆盖 14 种攻击类型），包含六层级 CoT 推理标注（从全局描述到局部推理到最终结论）；同时提出 CoT-Enhanced Progressive Learning (CEPL) 两阶段训练策略，在 11 个基准数据集上平均 AUC 提升 4.06%、HTER 降低 5.00%，超越所有 SOTA 方法。

**[Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought](llm_reasoning/red_rationale_enhanced_decoding_cot.md)**

:   发现现有 LVLM 在多模态 CoT 推理中会忽略生成的 rationale 内容（图像 token 主导注意力），提出 Rationale-Enhanced Decoding (RED)——将 CoT 重新表述为 KL 约束的 rationale 条件对数似然奖励最大化问题，最优解为将图像条件分布 $p(y|x,q)$ 和 rationale 条件分布 $p(y|r,q)^\lambda$ 相乘，无需训练即可显著提升多个基准上的推理性能。

**[VisRef: Visual Refocusing while Thinking Improves Test-Time Scaling in Multi-Modal Large Reasoning Models](llm_reasoning/visref_visual_refocusing_test_time_scaling.md)**

:   发现多模态推理模型在延长推理时会逐渐丢失对视觉token的注意力，提出VisRef在推理过程中主动重新注入与当前推理上下文语义相关的视觉token核心子集，在固定计算预算下比现有方法提升最高6.4%。

---

## 🛰️ 遥感 { #remote_sensing }

**[AVION: Aerial Vision-Language Instruction from Offline Teacher to Prompt-Tuned Network](remote_sensing/avion_aerial_visionlanguage_instruction_from_offli.md)**

:   提出 AVION 蒸馏框架，通过 LLM 生成并视觉验证的文本原型解决遥感 VLM 的"语义贫乏"，通过双模态 Prompt Tuning 解决"视觉刚性"，在 6 个遥感基准上实现少样本和 base-to-novel 同时提升。

**[Joint and Streamwise Distributed MIMO Satellite Communications with Multi-Antenna Ground Users](remote_sensing/joint_and_streamwise_distributed_mimo_satellite_co.md)**

:   针对多LEO卫星协同服务多天线地面用户的下行链路，基于统计CSI提出联合非相干传输（WMMSE迭代预编码，支持一般凸功率约束）和流式分布传输（每流由单颗卫星发送，通过匈牙利算法做特征模式-卫星关联），在UE侧信道正交时流式传输几乎无损，非正交时呈现性能-开销权衡。

**[MetaSpectra+: A Compact Broadband Metasurface Camera for Snapshot Hyperspectral+ Imaging](remote_sensing/metaspectra_a_compact_broadband_metasurface_camera.md)**

:   提出MetaSpectra+，一种基于双层超表面-折射光学混合设计的紧凑型相机，可在单次快照中同时获取高光谱数据立方体和HDR/偏振图像，工作带宽达250nm覆盖几乎整个可见光谱，在基准数据集上实现了最高的高光谱重建精度和最短的系统总光程长度。

---

## 🛡️ AI 安全 { #ai_safety }

**[Federated Active Learning Under Extreme Non-IID and Global Class Imbalance](ai_safety/federated_active_learning_extreme_noniid.md)**

:   系统分析全局类不平衡与客户端异构性对联邦主动学习中 query model 选择的影响，发现类平衡采样能力是性能的最一致预测因子，据此提出 FairFAL——自适应选择 query model + 原型引导伪标签 + 不确定性-多样性平衡采样的类公平 FAL 框架。

**[Rethinking VLMs for Image Forgery Detection and Localization](ai_safety/rethinking_vlms_for_image_forgery_detection_and_lo.md)**

:   揭示VLM的语义合理性偏差(semantic plausibility bias)会妨碍伪造检测，提出IFDL-VLM将检测/定位与语言解释生成解耦为两阶段：先用ViT+SAM专做检测定位，再将定位mask作为VLM辅助输入增强可解释性，在9个基准上全面SOTA。

---

## 🎵 音频/语音 { #audio_speech }

**[Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach](audio_speech/team_leya_in_10th_abaw_competition_multimodal_ambi.md)**

:   提出四模态(场景VideoMAE+人脸EfficientNetB0+音频Wav2Vec2.0+Mamba+文本EmotionDistilRoBERTa)融合管线，通过原型增强Transformer融合模块将模态嵌入投影到共享空间并结合原型分类辅助损失，在BAH测试集上以5模型集成达到71.43% Macro F1。

---

## ⚖️ 对齐 / RLHF { #llm_alignment }

**[Bases of Steerable Kernels for Equivariant CNNs: From 2D Rotations to the Lorentz Group](llm_alignment/bases_of_steerable_kernels_for_equivariant_cnns_fr.md)**

:   提出一种直接从输入/输出表示构造可操纵核显式基的方法，无需计算 Clebsch-Gordan 系数，统一覆盖 SO(2)、O(2)、SO(3)、O(3) 到非紧致 Lorentz 群，大幅简化等变 CNN 的核设计流程。

---

## 📖 NLP 理解 { #nlp_understanding }

**[As Language Models Scale, Low-order Linear Depth Dynamics Emerge](nlp_understanding/as_language_models_scale_loworder_linear_depth_dyn.md)**

:   将 Transformer 的逐层前向传播视为离散时间动力系统，发现 32 维低阶线性代理（LLV）可精确复现完整模型的层级灵敏度曲线，且该线性可辨识性随模型规模单调增强。

---

## 📐 优化/理论 { #optimization }

**[SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated Learning](optimization/scope_semantic_coreset_with_orthogonal_projection.md)**

:   提出SCOPE——一个无需训练的联邦coreset选择框架，利用冻结VLM(MobileCLIP)的正交投影嵌入计算三个标量语义指标(表示性/多样性/边界接近度)，实现全局感知的两阶段剪枝，在CIFAR-10/Tiny-ImageNet/UHCS上通信带宽降128-512倍的同时超越全数据训练。

---

## 🎮 强化学习 { #reinforcement_learning }

**[Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment](reinforcement_learning/lifelong_imitation_learning_multimodal_latent_rep.md)**

:   提出终身模仿学习框架，通过多模态潜在回放（MLR）在冻结编码器的特征空间中存储和回放紧凑表示，并引入增量特征调整（IFA）机制用角距离约束维持任务间可分性，在LIBERO基准上AUC提升10-17点、遗忘降低最多65%。

---

## 📈 时间序列 { #time_series }

**[Competition-Aware CPC Forecasting with Near-Market Coverage# Competition-Aware CPC Forecasting with Near-Market Coverage](time_series/competitionaware_cpc_forecasting_with_nearmarket_c.md)**

:   针对付费搜索中 CPC 预测的"部分可观测竞争"难题，通过语义邻域、DTW 行为邻域和地理意图三类竞争代理信号，以协变量或图关系先验的形式增强时间序列基础模型和时空图网络，在中长期预测上显著提升稳定性。
