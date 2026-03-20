<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎯 目标检测

**🔬 ICLR2026** · 共 **16** 篇

**[A Problem-Oriented Perspective and Anchor Verification for Code Optimization](a_problem-oriented_perspective_and_anchor_verification_for_code_optimization.md)**

:   提出以问题为导向（而非用户为导向）的优化对构建方法来整合多程序员的策略多样性，并设计锚点验证框架利用"慢但正确的代码"生成测试用例来缓解"优化税"（正确性损失），将优化比从 31.24% 提升到 71.06%，加速比从 2.95x 提升到 6.08x。

**[AdaRank: Adaptive Rank Pruning for Enhanced Model Merging](adarank_adaptive_rank_pruning_for_enhanced_model_merging.md)**

:   提出 AdaRank，通过可学习二值掩码自适应选择 task vector 的奇异分量（而非启发式 top-k），结合测试时熵最小化优化，大幅缓解多任务模型合并中的任务间干扰。

**[Beyond Linearity in Attention Projections: The Case for Nonlinear Queries](beyond_linearity_in_attention_projections_the_case_for_nonlinear_queries.md)**

:   基于 WQ 代数冗余性的理论发现，将线性 Query 投影替换为非线性残差形式 Q(X)=(X+f_θ(X))/2，在相同参数量下超越增加 12.5% 参数的基线。

**[Breaking Scale Anchoring: Frequency Representation Learning for Accurate High-Resolution Inference from Low-Resolution Training](breaking_scale_anchoring_frequency_representation_learning_for_accurate_high-res.md)**

:   定义了"Scale Anchoring"新问题（低分辨率训练导致高分辨率推理误差锚定），提出架构无关的频率表征学习（FRL），通过归一化频率编码使误差随分辨率提升而下降。

**[CGSA: Class-Guided Slot-Aware Adaptation for Source-Free Object Detection](cgsa_class-guided_slot-aware_adaptation_for_source-free_object_detection.md)**

:   首次将 Object-Centric Learning（Slot Attention）引入无源域自适应目标检测（SF-DAOD），通过分层 Slot 感知模块提取结构先验，并用类引导对比学习驱动域不变表征。

**[ConFu: Contemplate the Future for Better Speculative Sampling](confu_contemplate_the_future_for_better_speculative_sampling.md)**

:   提出 ConFu 框架，通过 contemplate tokens 让 draft model 预见 target model 的未来生成方向，结合 MoE 动态机制和锚点采样训练，在 EAGLE-3 基础上提升 8-11% 的接受率和速度。

**[Context Tokens are Anchors: Understanding the Repetition Curse in dMLLMs from an Information Flow Perspective](context_tokens_are_anchors_understanding_the_repetition_curse_in_dmllms_from_an_.md)**

:   通过信息流分析揭示扩散多模态大语言模型(dMLLMs)在使用缓存加速时产生"重复诅咒"的内在机制，并提出 CoTA 方法有效缓解重复问题。

**[ForestPersons: A Large-Scale Dataset for Under-Canopy Missing Person Detection](forestpersons_a_large-scale_dataset_for_under-canopy_missing_person_detection.md)**

:   ForestPersons 是首个专门针对森林树冠下人员检测的大规模数据集（96,482 张图像 + 204,078 标注），覆盖地面/低空视角、多季节多天气多光照条件，每个实例包含边界框+姿态+可见性标注，填补了 SAR 场景中下冠层检测的数据空白。

**[From Narrow to Panoramic Vision: Attention-Guided Cold-Start Reshapes Multimodal Reasoning](from_narrow_to_panoramic_vision_attention-guided_cold-start_reshapes_multimodal_.md)**

:   发现多模态 LLM 的推理性能与视觉注意力分数（VAS）高度相关（r=0.96），提出 AVAR 框架通过视觉锚定数据合成、注意力引导训练目标和视觉锚定奖励塑造三个阶段提升 VAS，在 77 个基准上平均提升 7%。

**[FSOD-VFM: Few-Shot Object Detection with Vision Foundation Models and Graph Diffusion](fsod-vfm_few-shot_object_detection_with_vision_foundation_models_and_graph_diffu.md)**

:   提出一个无需训练的少样本目标检测框架，组合 UPN、SAM2 和 DINOv2 三个基础模型生成提案和匹配特征，并通过图扩散算法精化置信度分数和抑制碎片化提案，在 Pascal-5i 和 COCO-20i 上大幅超越 SOTA。

**[InfoDet: A Dataset for Infographic Element Detection](infodet_a_dataset_for_infographic_element_detection.md)**

:   构建了一个大规模信息图元素检测数据集（101,264 张信息图、1420 万标注），涵盖图表和人类可识别对象两大类，并提出 Grounded CoT 方法利用检测结果提升 VLM 的图表理解能力。

**[Procedural Mistake Detection via Action Effect Modeling](procedural_mistake_detection_via_action_effect_modeling.md)**

:   提出双分支多模态监督的动作效果建模框架，结合视觉分支（目标状态和空间关系特征）和文本分支（GPT-4o 生成的场景图），通过可学习的效果 token 蒸馏外部监督信号，在第一人称程序视频中实现 SOTA 错误检测。

**[SAGE: Spatial-visual Adaptive Graph Exploration for Efficient Visual Place Recognition](sage_spatial-visual_adaptive_graph_exploration_for_efficient_visual_place_recogn.md)**

:   提出 SAGE，通过在线地理-视觉图和加权贪心团扩展实现动态难样本挖掘，仅用冻结 DINOv2+LoRA 在 SPED 和 Pitts30k 上达到 R@1=100%（超 CosPlace/MixVPR 20+%），在 8 个基准上全面 SOTA。

**[Thinking in Latents: Adaptive Anchor Refinement for Implicit Reasoning in LLMs](thinking_in_latents_adaptive_anchor_refinement_for_implicit_reasoning_in_llms.md)**

:   提出 AdaAnchor，通过可学习潜在锚向量的迭代精化实现隐式推理，配合基于余弦距离的自适应停止机制，相比 CoT 减少 92-93% 生成 token（平均 2.17 vs 28.27），同时精度仅降低约 4%。

**[Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders](toward_faithful_retrieval-augmented_generation_with_sparse_autoencoders.md)**

:   提出 RAGLens，利用稀疏自编码器（SAE）编码 LLM 隐状态，通过互信息特征选择和可加性模型（GAM）实现可解释的 RAG 幻觉检测，在 RAGTruth 上 AUC>0.80，零额外 LLM 推理开销。

**[Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Methodology](traceable_evidence_enhanced_visual_grounded_reasoning_evaluation_and_methodology.md)**

:   提出 TreeBench 基准和 TreeVGR 训练范式，通过冷启动 SFT + 双 IoU 奖励 RL 训练，使模型在推理链中显式生成边界框实现可追溯的视觉推理，在 V*Bench/MME-RealWorld/TreeBench 上分别提升 +16.8/+12.6/+13.4。
