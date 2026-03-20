<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🤖 机器人/具身智能

**🔬 ICLR2026** · 共 **16** 篇

**[All-day Multi-scenes Lifelong Vision-and-Language Navigation with Tucker Adaptation](all-day_multi-scenes_lifelong_vision-and-language_navigation_with_tucker_adaptat.md)**

:   提出Tucker Adaptation (TuKA)，将多场景多环境的多层级导航知识表示为高阶张量，用Tucker分解解耦为共享子空间（核心张量+编解码器）和场景/环境专家向量，配合解耦知识增量学习策略实现全天候多场景终身VLN，在24个导航场景上的SR和遗忘率均优于LoRA变体。

**[Attribution-Guided Decoding](attribution-guided_decoding.md)**

:   提出 Attribution-Guided Decoding (AGD)，在解码时利用归因方法（LRP）对候选 token 计算其对"感兴趣区域"(ROI) 的依赖分数，选择归因最高的 token，从而在不修改模型内部激活的前提下提升指令遵循和事实准确性。

**[Building Spatial World Models from Sparse Transitional Episodic Memories](building_spatial_world_models_from_sparse_transitional_episodic_memories.md)**

:   提出 Episodic Spatial World Model (ESWM)，从稀疏、不连续的情景记忆（one-step transitions）中构建空间世界模型，其潜空间自发涌现出与环境拓扑对齐的认知地图，并支持零样本探索和导航。

**[Capability-Based Scaling Trends for LLM-Based Red-Teaming](capability-based_scaling_trends_for_llm-based_red-teaming.md)**

:   在 600+ 对攻击者-目标 LLM 组合上系统评估了 4 种越狱方法，发现攻击成功率（ASR）与攻击者-目标的能力差距遵循 sigmoid 缩放定律（R^2=0.83），能力差距可用 MMLU-Pro 的 logit 变换量化。

**[Domain Expansion: A Latent Space Construction Framework for Multi-Task Learning](domain_expansion_a_latent_space_construction_framework_for_multi-task_learning.md)**

:   提出 Domain Expansion 框架，通过正交池化(Orthogonal Pooling)将潜在空间重构为互相正交的子空间，从结构上防止多目标训练中的梯度冲突与表征崩塌，实现可解释、可组合的概念代数。

**[ExoPredicator: Learning Abstract Models of Dynamic Worlds for Robot Planning](exopredicator_learning_abstract_models_of_dynamic_worlds_for_robot_planning.md)**

:   提出 ExoPredicator 框架，联合学习符号化状态抽象和因果过程（含内生动作与外生机制），通过变分贝叶斯推断 + LLM 提议从少量轨迹中学习带随机延迟的因果世界模型，在 5 个桌面机器人环境中实现快速泛化规划。

**[Experience-based Knowledge Correction for Robust Planning in Minecraft](experience-based_knowledge_correction_for_robust_planning_in_minecraft.md)**

:   证明 LLM 无法通过 prompting 自我纠正其错误的规划先验知识（物品依赖关系），提出 XENON——通过算法化的知识管理（自适应依赖图 ADG + 失败感知动作记忆 FAM）从二值反馈中学习，使 7B LLM 在 Minecraft 长期规划中超越使用 GPT-4V + oracle 知识的 SOTA。

**[Ignore All Previous Instructions: Jailbreaking as a de-escalatory peace building practise to resist LLM social media bots](ignore_all_previous_instructions_jailbreaking_as_a_de-escalatory_peace_building_.md)**

:   提出将"越狱"（jailbreaking）LLM 驱动的社交媒体机器人视为一种用户主导的、非暴力的去冲突化（de-escalation）和和平建设实践，通过暴露自动化账号的虚假性来抵抗误导信息传播。

**[JULI: Jailbreak Large Language Models by Self-Introspection](juli_jailbreak_large_language_models_by_self-introspection.md)**

:   揭示对齐 LLM 的 top-k token log probability 中仍包含有害信息，提出 JULI——仅用不到目标模型 1% 参数的 BiasNet 插件操纵 logit bias，在仅访问 top-5 token 概率的 API 场景下成功越狱 Gemini-2.5-Pro（harmfulness 4.19/5），比 SOTA 快 140 倍。

**[PERSONA: Dynamic and Compositional Inference-Time Personality Control via Activation Vector Algebra](persona_dynamic_and_compositional_inference-time_personality_control_via_activat.md)**

:   提出 PERSONA 框架，通过在激活空间中提取近似正交的人格向量并进行向量代数运算（缩放、加法、减法），实现免训练的动态组合式人格控制，在 PersonalityBench 上达到 9.60 分，几乎匹配 SFT 上界 9.61。

**[RF-MatID: Dataset and Benchmark for Radio Frequency Material Identification](rf-matid_dataset_and_benchmark_for_radio_frequency_material_identification.md)**

:   构建了首个开源的大规模、宽频段（4-43.5 GHz）、几何扰动多样的 RF 材料识别数据集 RF-MatID，包含 16 种细粒度材料类别（5 大类）/142K 样本，并建立了覆盖 9 个深度学习模型、5 种频率协议、7 种数据划分的系统基准。

**[RoboPARA: Dual-Arm Robot Planning with Parallel Allocation and Recomposition Across Tasks](robopara_dual-arm_robot_planning_with_parallel_allocation_and_recomposition_acro.md)**

:   提出 RoboPARA，一个 LLM 驱动的双臂机器人并行任务规划框架，通过依赖图生成与图重遍历调度两阶段方法，最大化双臂协同并行性，执行时间减少 30%-50%。

**[Sparse Imagination for Efficient Visual World Model Planning](sparse_imagination_for_efficient_visual_world_model_planning.md)**

:   提出 Sparse Imagination，在基于 ViT patch token 的世界模型规划中随机丢弃 token 以大幅加速推理（50% 丢弃率减少约 50% 时间），同时通过随机分组注意力训练保持任务性能不变。

**[THOR: Tool-Integrated Hierarchical Optimization via RL for Mathematical Reasoning](thor_tool-integrated_hierarchical_optimization_via_rl_for_mathematical_reasoning.md)**

:   提出 THOR，通过 TIRGen 数据构建管线 + 层次化强化学习（episode 级+step 级优化）+ 自修正推理机制，系统性解决 LLM 工具集成数学推理中的数据构建、细粒度优化和推理增强三大挑战。

**[Tracing and Reversing Edits in LLMs](tracing_and_reversing_edits_in_llms.md)**

:   针对知识编辑（Knowledge Editing）的双重使用风险，提出 EditScope 方法从编辑后的权重中推断被编辑的目标实体（准确率高达 99%），以及基于 SVD bottom-rank 近似的无训练编辑逆转方法（逆转率高达 94%），仅依赖编辑后的权重、不需要编辑 prompt 或原始权重信息。

**[Visual Planning: Let's Think Only with Images](visual_planning_lets_think_only_with_images.md)**

:   提出 Visual Planning，MLLM 通过生成图像序列（而非文本推理链）进行规划，VPRL 两阶段 RL（监督微调+进度奖励策略优化）在 FrozenLake 上比文本 SFT 高 27%，且泛化到分布外场景。
