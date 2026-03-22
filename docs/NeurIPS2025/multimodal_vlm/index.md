<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧩 多模态 VLM

**🧠 NeurIPS2025** · 共 **44** 篇

**[A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1](a_frustratingly_simple_yet_highly_effective_attack_baseline.md)**

:   提出 M-Attack，通过对对抗图像做随机裁剪后与目标图像在嵌入空间做局部对齐（而非传统的全局对齐），配合多模型集成，使得生成的对抗扰动具有丰富的局部语义细节，在 GPT-4.5/4o/o1 等商业黑盒 LVLM 上实现超过 90% 的目标攻击成功率，大幅超越所有已有方法。

**[A Multimodal Benchmark for Framing of Oil & Gas Advertising and Potential Greenwashing Detection](a_multimodal_benchmark_for_framing_of_oil_gas_advertising_an.md)**

:   构建了首个面向石油天然气行业视频广告的多模态框架分析基准数据集（706个视频，覆盖Facebook和YouTube两个平台，13种框架类型），用于评估VLM在检测企业"洗绿"宣传中的能力，发现GPT-4.1在环境信息检测上可达79% F1但在绿色创新识别上仅46% F1。

**[AC-LoRA: (Almost) Training-Free Access Control-Aware Multi-Modal LLMs](aclora_almost_trainingfree_access_controlaware_multimodal_ll.md)**

:   设计AC-LoRA系统，通过为不同权限数据集维护独立的LoRA适配器，并基于查询相似度和用户权限进行检索+无训练合并，实现企业级LLM聊天机器人的强信息隔离保证。

**[ACT as Human: Multimodal Large Language Model Data Annotation with Critical Thinking](act_as_human_multimodal_large_language_model_data_annotation.md)**

:   提出ACT（Annotation with Critical Thinking）流水线，先用MLLM批量标注数据，再用另一个MLLM作为"批评者"识别可能的错误标注，仅让人类审核被标记的样本，在减少70-90%人工标注成本的同时将性能差距控制在<2%。

**[AdaLRS: Loss-Guided Adaptive Learning Rate Search for Efficient Foundation Model Pretraining](adalrs_lossguided_adaptive_learning_rate_search_for_efficien.md)**

:   提出AdaLRS，一种即插即用的在线学习率搜索算法，通过监控损失下降速度（loss velocity）来自适应调整学习率，将学习率超参搜索的成本从多次独立训练降低到单次训练，实现~50%的训练成本节省。

**[Adapting Vision-Language Models for Evaluating World Models](adapting_visionlanguage_models_for_evaluating_world_models.md)**

:   提出UNIVERSE框架，通过仅微调PaliGemma 2的投影头（0.07%参数）和优化数据混合策略，实现对游戏世界模型rollout的高效视觉语言评估，在动作/角色识别任务上以极低成本接近完整微调的性能。

**[ADMN: A Layer-Wise Adaptive Multimodal Network for Dynamic Input Noise and Compute Resources](admn_a_layerwise_adaptive_multimodal_network_for_dynamic_inp.md)**

:   提出 ADMN（Adaptive Depth Multimodal Network），通过两阶段训练——(1) Multimodal LayerDrop 微调使 backbone 适应任意层配置，(2) QoI感知控制器动态分配层预算给各模态——在严格计算约束下根据每个模态的信息质量(QoI)自适应分配层数，匹配全量模型精度同时减少 75% FLOPs 和 60% 延迟。

**[Advancing Compositional Awareness in CLIP with Efficient Fine-Tuning](advancing_compositional_awareness_in_clip_with_efficient_fin.md)**

:   提出 CLIC（Compositionally-aware Learning in CLIP），通过拼接图像对 + 跨图词汇交换生成 hard negatives + 多正样本训练的策略，在仅微调文本编码器的情况下同时提升 CLIP 的组合推理能力和检索性能，在 SugarCrepe++ 上取得 CLIP 类模型 SOTA。

**[AffordBot: 3D Fine-grained Embodied Reasoning via Multimodal Large Language Models](affordbot_3d_fine-grained_embodied_reasoning_via_multimodal_large_language_model.md)**

:   提出细粒度 3D 具身推理任务（预测可操作元素的空间位置+运动类型+运动轴），通过将 3D 点云渲染为环视图并投影 affordance 候选，结合定制的 CoT 推理范式指导 MLLM 实现 SOTA，AP25 达 23.3%。

**[Aligning by Misaligning: Boundary-aware Curriculum Learning for Multimodal Alignment](aligning_by_misaligning_boundaryaware_curriculum_learning_fo.md)**

:   提出 BACL（Boundary-Aware Curriculum with Local Attention），通过可学习的边界感知负样本采样器（由易到难课程学习）+ 对比局部注意力损失（定位 token 级 mismatch），在 LAION-400M 上为 CLIP 带来 +32% R@1 提升，并在四个大规模基准上取得 SOTA。

**[AntiGrounding: Lifting Robotic Actions into VLM Representation Space for Decision Making](antigrounding_lifting_robotic_actions_into_vlm_representatio.md)**

:   提出 AntiGrounding，逆转传统指令 grounding 过程——不是将语言映射到动作空间，而是将候选机器人动作"提升"到 VLM 表示空间（通过多视角轨迹渲染 + 结构化 VQA），实现零样本闭环机器人轨迹合成。

**[Approximate Domain Unlearning for Vision-Language Models](approximate_domain_unlearning_for_visionlanguage_models.md)**

:   提出 Approximate Domain Unlearning (ADU) 新任务，通过 Domain Disentangling Loss (DDL) 和 Instance-wise Prompt Generator (InstaPG) 两个模块，让预训练 VLM 选择性遗忘指定域（如插画、素描）的识别能力，同时保持其他域（如真实照片）的分类精度，在四个多域数据集上大幅超越所有基线。

**[Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization](balanced_token_pruning_accelerating_vision_language_models_b.md)**

:   提出 Balanced Token Pruning (BTP)，通过在浅层优先多样性剪枝、深层优先注意力剪枝的分阶段策略，联合优化局部输出一致性和全局表示质量，在仅保留 22% 视觉 token 的情况下保持原模型 98% 的性能。

**[Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering](benchmarking_retrievalaugmented_multimodal_generation_for_do.md)**

:   提出 MMDocRAG 基准（4055 个专家标注的 QA 对），系统评估了 60 个 VLM/LLM 和 14 个检索器在多模态文档检索增强生成中的引用选择和交错图文回答能力，揭示当前最强模型 GPT-4.1 的 Quote Selection F1 仅 70.2%，微调可显著提升性能。

**[Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging](better_tokens_for_better_3d_advancing_vision-language_modeling_in_3d_medical_ima.md)**

:   提出 BTB3D，一种基于因果卷积编解码器 + 3D Haar 小波压缩 + 三阶段渐进训练的 3D CT tokenizer，在放射报告生成和文本条件 CT 合成两大下游任务上大幅刷新 SOTA，证明"更好的 token 比更大的语言模型更重要"。

**[CAPability: A Comprehensive Visual Caption Benchmark for Evaluating Both Correctness and Thoroughness](capability_a_comprehensive_visual_caption_benchmark_for_eval.md)**

:   提出CAPability，一个全面的多视角视觉描述benchmark，跨6个关键视角12个维度评估MLLM生成caption的正确性（precision）和全面性（hit），用近11K人工标注的图像视频，并引入"知道但说不出"（K/T̄）指标揭示模型在QA和caption之间的显著能力差距。

**[Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models](causalllava_causal_disentanglement_for_mitigating_hallucinat.md)**

:   揭示 MLLM 中物体幻觉的表示层根因——数据集共现偏差导致的语义纠缠，提出双路因果解纠缠框架（Causal-Driven Projector + Causal Intervention Module），通过后门调整在 projector 和最终 Transformer 层分离共现物体表示，使 MME-Perception 提升 22.6%。

**[ChartMuseum: Testing Visual Reasoning Capabilities of Large Vision-Language Models](chartmuseum_testing_visual_reasoning_capabilities_of_large_v.md)**

:   构建ChartMuseum——一个包含1,162个专家标注问题的图表QA benchmark，专门评估LVLM的复杂视觉和文本推理能力。与现有图表benchmark（前沿模型接近饱和）不同，ChartMuseum揭示了巨大的模型-人类性能差距：人类93%准确率 vs Gemini-2.5-Pro仅63.0% vs 最佳开源Qwen2.5-VL-72B仅38.5%，且所有模型在视觉推理重的问题上掉点35-55%。

**[Continual Multimodal Contrastive Learning](continual_multimodal_contrastive_learning.md)**

:   首次形式化定义持续多模态对比学习(CMCL)问题——按顺序在不同模态对数据上训练而不忘记之前的对齐，提出Dual-sided Null Space (DNS)方法将新梯度投影到不影响旧知识的子空间，在7个数据集11个训练步骤上一致优于现有持续学习基线。

**[CovMatch: Cross-Covariance Guided Multimodal Dataset Distillation with Trainable Text Encoder](covmatch_crosscovariance_guided_multimodal_dataset_distillat.md)**

:   提出 CovMatch，通过将多模态对比学习的双层优化简化为跨协方差矩阵对齐的闭式解，首次实现图文双编码器的联合优化进行多模态数据集蒸馏，仅用 500 个合成图文对在 Flickr30K 上获得 38.4 平均检索精度（+6.8% 超越 SOTA LoRS），在极端数据高效场景下大幅超越冻结文本编码器的方法。

**[DanmakuTPPBench: A Multi-modal Benchmark for Temporal Point Process Modeling and Understanding](danmakutppbench_a_multimodal_benchmark_for_temporal_point_pr.md)**

:   论文提出首个面向多模态 Temporal Point Process 的系统 benchmark：一方面构建来自 Bilibili 弹幕视频的时间戳-文本-视频联合事件数据集 DanmakuTPP-Events，另一方面通过多智能体 LLM/MLLM pipeline 构建复杂时序推理问答集 DanmakuTPP-QA，系统揭示当前 TPP 模型与 MLLM 在多模态事件动态理解上的明显短板。

**[DOTA: Distributional Test-Time Adaptation of Vision-Language Models](dota_distributional_testtime_adaptation_of_visionlanguage_mo.md)**

:   提出 DOTA（DistributiOnal Test-time Adaptation），不再简单缓存测试样本，而是**持续估计测试数据流的底层分布**，通过贝叶斯定理计算后验概率实现自适应，解决了缓存容量有限导致的灾难性遗忘问题，在多个分布偏移基准上达到 SOTA。

**[Enhancing Compositional Reasoning in CLIP via Reconstruction and Alignment of Text Descriptions](enhancing_compositional_reasoning_in_clip_via_reconstruction.md)**

:   提出READ方法——通过在CLIP对比学习上增加token级文本重建（用冻结decoder从embedding重建替代caption）和句子级释义对齐（拉近同义不同表达的embedding）两个辅助目标，READ-CLIP在5个组合推理benchmark上达到SOTA，比最强基线提升4.1%。

**[Enhancing Vision-Language Model Reliability with Uncertainty-Guided Dropout Decoding](enhancing_visionlanguage_model_reliability_with_uncertaintyg.md)**

:   提出Dropout Decoding——量化视觉token的认知不确定性(epistemic uncertainty)，选择性遮掩高不确定性token，通过集成多个遮掩后的解码结果做多数投票，无需训练即在InstructBLIP上CHAIR_I降低16%、CHAIR_S降低12%。

**[First SFT, Second RL, Third UPT: Continual Improving Multi-Modal LLM Reasoning via Unsupervised Post-Training](first_sft_second_rl_third_upt_continual_improving_multi-modal_llm_reasoning_via_.md)**

:   提出 MM-UPT 框架，在 SFT 和 RL 之后引入第三阶段"无监督后训练"，通过多数投票作为伪奖励信号结合 GRPO 实现 MLLM 的自我改进，在 MathVista 上将 Qwen2.5-VL-7B 从 66.3% 提升至 72.9%。

**[FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models](flowcut_rethinking_redundancy_via_information_flow_for_effic.md)**

:   从信息流（Information Flow）视角重新理解VLM中视觉token的冗余性：发现CLS token是信息中继站、冗余渐进式涌现、单层单标准评分不够可靠，提出FlowCut——基于信息流感知的多标准累积重要性剪枝框架，在LLaVA-1.5-7B上以88.9%的token减少率超越SOTA 1.6%，在LLaVA-NeXT-7B上超越4.3%。

**[Generalized Contrastive Learning for Universal Multimodal Retrieval](generalized_contrastive_learning_for_universal_multimodal_re.md)**

:   提出 Generalized Contrastive Learning (GCL)——在 mini-batch 内对所有 6 种模态对组合（image↔text, image↔image+text, text↔image+text）执行对比学习，无需构建新的三元组数据集，仅用现有图文对即可在 M-BEIR 上将 VISTA 的平均检索精度从 21.18 提升到 34.06（+60.8%），在 MMEB 的 text→image+text 任务上从 10.1% 提升到 31.1%。

**[Generate, but Verify: Reducing Hallucination in Vision-Language Models with Retrospective Resampling](generate_but_verify_reducing_hallucination_in_visionlanguage.md)**

:   提出REVERSE框架——首次在单一VLM内统一了生成、验证和纠正三个阶段：通过引入<SPAN>、</CN>（置信）、</UN>（不置信）三个特殊token训练幻觉感知模型，推理时当</UN>概率超过阈值就回溯到上一个</CN>重新生成，在CHAIR-MSCOCO上降低12%、HaloQuest上降低34%的幻觉率。

**[GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity](glsim_detecting_object_hallucinations_in_lvlms_via_globalloc.md)**

:   提出 GLSim，一种无训练的物体幻觉检测框架，结合图像-文本间的全局和局部嵌入相似度信号来判断 LVLM 生成的物体是否为幻觉，显著超越仅使用全局或局部信号的方法。

**[HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models](hope_hybrid_of_position_embedding_for_long_context_visionlan.md)**

:   提出 HoPE（Hybrid of Position Embedding），通过混合频率分配策略和动态时间缩放机制改进 VLM 中的位置编码，解决 RoPE 在长视频等长上下文多模态场景中无法可靠捕捉时空语义相似性的问题，在四个长视频基准上一致超越现有方法。

**[Learning to Instruct for Visual Instruction Tuning](learning_to_instruct_for_visual_instruction_tuning.md)**

:   提出 L2T（Learning to Instruct），仅通过将训练损失扩展到指令序列（不再只在回答上计算 loss）来改善视觉指令调优——无额外数据和几乎零计算开销，在 16 个多模态基准上获得高达 9% 的相对提升，captioning 提升 18%，同时缓解幻觉。

**[Mint: A Simple Test-Time Adaptation of Vision-Language Models against Common Corruptions](mint_a_simple_testtime_adaptation_of_visionlanguage_models_a.md)**

:   发现 CLIP 在图像损坏下的性能退化根源在于**嵌入方差坍缩**——类内与类间方差同步缩小导致嵌入空间判别性丧失；提出 Mint，通过最大化伪标签类间方差（PL-inter）在线修复嵌入几何，仅凭均值累加器和梯度累加器两个极简组件即可在 BS=1 的在线场景下稳定提升 CLIP 在多种损坏基准上的分类精度，同时比最强 baseline 快 45 倍。

**[MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly](mmlongbench_benchmarking_longcontext_visionlanguage_models_e.md)**

:   首个全面覆盖长上下文视觉语言任务的benchmark，包含13,331个样本、5类下游任务（如Visual RAG和Many-Shot ICL）、多种图像类型，在5个标准化输入长度（8K-128K tokens）上评估46个模型，发现单任务性能不能代表整体长上下文能力、现有模型均面临显著挑战、推理能力与长上下文性能正相关。

**[Praxis-VLM: Vision-Grounded Decision Making via Text-Driven Reinforcement Learning](praxisvlm_visiongrounded_decision_making_via_textdriven_rein.md)**

:   发现VLM的决策推理能力可以与视觉感知解耦——用文本描述替代图像时决策性能不降反升，据此提出Praxis-VLM：在纯文本场景上用GRPO训练决策推理能力，然后零样本迁移到视觉输入推理，在VIVA/PCA-Bench/EgoNormia三个决策benchmark上超越SFT基线且泛化性更强。

**[PrefixKV: Adaptive Prefix KV Cache is What Vision Instruction-Following Models Need for Efficient Generation](prefixkv_adaptive_prefix_kv_cache_is_what_vision_instruction.md)**

:   提出 PrefixKV，将 LVLM 各层 KV 缓存大小的确定转化为搜索最优全局前缀配置的问题，通过二分搜索找到信息保留阈值实现自适应逐层 KV 保留，在 20% 压缩率下仍保持接近原模型性能，提供 1.8× 推理加速。

**[Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion](rethinking_multimodal_learning_from_the_perspective_of_mitig.md)**

:   提出"**分类能力不均衡**"视角理解多模态学习中的模态不平衡，设计 Sustained Boosting 算法（共享编码器 + 多可配置分类器，同时优化分类和残差误差）配合自适应分类器分配（ACA），理论证明跨模态 gap loss 以 $\mathcal{O}(1/T)$ 收敛，在 CREMAD 等 6 个数据集上大幅超越 SOTA。

**[Sherlock: Self-Correcting Reasoning in Vision-Language Models](sherlock_selfcorrecting_reasoning_in_visionlanguage_models.md)**

:   首个系统研究VLM推理自纠正能力的框架：发现现有推理VLM几乎不能自纠正（<10%出现aha moment），提出Sherlock三阶段训练框架（SFT冷启动→离线轨迹级偏好学习→在线自我迭代）仅用20K标注数据超越使用100K-260K数据的LLaVA-CoT/Mulberry/LlamaV-o1。

**[Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models](sparse_autoencoders_learn_monosemantic_features_in_visionlan.md)**

:   将Sparse Autoencoder (SAE)从LLM可解释性扩展到VLM领域，提出MonoSemanticity Score (MS)量化视觉神经元的单义性，发现SAE能将VLM中多义的神经元分解为单义特征，且可直接通过操控单个SAE神经元来steering LLaVA的输出（插入或抑制概念），无需修改LLM。

**[SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning](srpo_enhancing_multimodal_llm_reasoning_via_reflection-aware_reinforcement_learn.md)**

:   提出 SRPO（Self-Reflection enhanced reasoning with Group Relative Policy Optimization），一个两阶段反思感知 RL 框架：第一阶段用大模型生成反思数据做 SFT cold-start，第二阶段设计反思感知奖励函数在 GRPO 中强化简洁有效的自我反思能力，在 MathVista/MathVision/MMMU-Pro 等多模态推理基准上以 7B/32B 模型显著超越同规模 SOTA。

**[The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models](the_illusion_of_progress_a_critical_look_at_testtime_adaptat.md)**

:   提出TTA-VLM benchmark，在统一实验条件下评估8种episodic和7种online测试时适应(TTA)方法在15个数据集上的表现，发现三个令人意外的结论：(1) 现有TTA方法相比早期TPT基线提升有限；(2) TTA与训练时微调方法协作效果差；(3) 准确率提升以牺牲校准、OOD检测和鲁棒性为代价。

**[The Narrow Gate: Localized Image-Text Communication in Native Multimodal Models](the_narrow_gate_localized_imagetext_communication_in_native.md)**

:   发现原生多模态VLM（如Chameleon、Emu3）中图像到文本的跨模态信息传递竟然集中在单一的end-of-image [EOI] token上（"narrow gate"机制），而非原生VLM（如LLaVA）则通过多个图像token分布式传递信息；删除[EOI]的attention可导致native模型性能崩溃，而修改[EOI]表示可精确控制模型的语义输出。

**[TRoVe: Discovering Error-Inducing Static Feature Biases in Temporal Vision-Language Models](trove_discovering_errorinducing_static_feature_biases_in_tem.md)**

:   TRoVe 提出一个自动化诊断框架，用于发现 temporal VLM 在时序理解任务中错误依赖的静态特征偏置；它通过从验证集提取候选静态特征，并同时评估这些特征对错误率的影响与模型对其依赖程度，在 101 个带偏置真值标注的 temporal VLM 上较最强基线提升 28.6%，还能进一步辅助 test-time 改善模型表现。

**[Unveiling Chain of Step Reasoning for Vision-Language Models with Fine-grained Rewards](unveiling_chain_of_step_reasoning_for_visionlanguage_models.md)**

:   提出Chain-of-Step (CoS)推理框架：将VLM的推理链分解为结构化步骤（Name+Thought+Reflection），训练Process Reward Model (PRM)提供步骤级精细奖励，通过迭代DPO和step-level beam search显著提升VLM推理能力——在InternVL-2.5-MPO-8B上平均提升4.0%达到73.4%，并揭示"对VLM而言推理质量比长度更重要"。

**[VL-SAE: Interpreting and Enhancing Vision-Language Alignment with a Unified Concept Set](vlsae_interpreting_and_enhancing_visionlanguage_alignment_wi.md)**

:   提出VL-SAE，一种带有距离编码器和模态特定解码器的稀疏自编码器，将视觉和语言表示的语义映射到统一概念集，从而解释和增强VLM的视觉-语言对齐机制，在零样本分类平均提升0.6-0.9%，在POPE幻觉消除上超越专用方法VCD。
