<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🦾 LLM Agent

**🎞️ ECCV2024** · 共 **10** 篇

**[Agent3D-Zero: An Agent for Zero-shot 3D Understanding](agent3d-zero_an_agent_for_zero-shot_3d_understanding.md)**

:   Agent3D-Zero 提出一个基于 VLM 的零样本 3D 场景理解 Agent 框架，通过鸟瞰图上的 Set-of-Line 视觉提示引导 VLM 主动选择观察视角，并综合多视角图像进行 3D 推理，在 ScanQA 等任务上超越了需要微调的 3D-LLM 方法。

**[Controllable Navigation Instruction Generation with Chain of Thought Prompting](controllable_navigation_instruction_generation.md)**

:   提出 C-Instructor，利用 Chain-of-Thought with Landmarks (CoTL) 机制引导 LLM 先识别关键地标再生成指令，结合空间拓扑建模任务 (STMT) 和风格混合训练 (SMT)，实现风格可控和内容可控的导航指令生成，在四个室内外 benchmark 上全面超越 SOTA。

**[ControlLLM: Augment Language Models with Tools by Searching on Graphs](controlllm_augment_language_models_with_tools.md)**

:   提出 ControlLLM 框架，通过任务分解、Thoughts-on-Graph (ToG) 图搜索范式和执行引擎三大组件，让 LLM 在预构建的工具图上搜索最优解决方案路径，准确高效地调用多模态工具完成复杂任务，在困难任务上达到 93% 的解决方案成功率。

**[DISCO: Embodied Navigation and Interaction via Differentiable Scene Semantics and Dual-Level Control](disco_embodied_navigation_and_interaction.md)**

:   提出 DISCO，通过可微分场景语义表征（包含物体和 affordance）实现动态场景建模，结合全局-局部双层粗到细控制策略实现高效移动操作，在 ALFRED benchmark 的 unseen scenes 上以 +8.6% 成功率超越使用分步指令的 SOTA，且无需分步指令。

**[NavGPT-2: Unleashing Navigational Reasoning Capability for Large Vision-Language Models](navgpt2_unleashing_navigational_reasoning_capability.md)**

:   提出 NavGPT-2，通过将冻结 LLM 与视觉内容对齐，结合拓扑图导航策略网络，在保持 LLM 可解释性推理能力的同时，消除了基于语言模型的导航智能体与 VLN 专用模型之间的性能差距。

**[Navigation Instruction Generation with BEV Perception and Large Language Models](navigation_instruction_generation_with_bev.md)**

:   提出 BEVInstructor，将鸟瞰图 (BEV) 特征融入多模态大语言模型 (MLLM) 用于导航指令生成，通过 Perspective-BEV 视觉编码、参数高效 prompt tuning 和实例引导的迭代精化，在室内外多个数据集上全面超越 SOTA。

**[Octopus: Embodied Vision-Language Programmer from Environmental Feedback](octopus_embodied_visionlanguage_programmer_from_environmental_feedback.md)**

:   Octopus 是一个具身视觉-语言编程模型，通过将 VLM 与可执行代码生成相结合，利用 GPT-4 收集训练数据并引入 RLEF（环境反馈强化学习）进行微调，在三个不同模拟器（OmniGibson、Minecraft、GTA-V）中实现了端到端的视觉感知→计划→代码生成→执行闭环。

**[Prioritized Semantic Learning for Zero-Shot Instance Navigation](prioritized_semantic_learning_for_zeroshot_instance_navigation.md)**

:   提出 Prioritized Semantic Learning (PSL) 方法，通过语义感知智能体架构、优先语义训练策略和语义扩展推理方案，显著提升导航智能体的语义感知能力，在零样本 ObjectNav 上超越 SOTA  66%（SR），并提出了更具挑战性的 InstanceNav 任务。

**[QUAR-VLA: Vision-Language-Action Model for Quadruped Robots](quarvla_visionlanguageaction_model_for_quadruped_robots.md)**

:   提出 QUAR-VLA 范式，首次将视觉、语言指令和动作生成统一到四足机器人中，构建了大规模多任务数据集 QUARD（259K episodes），训练 QUART 模型（基于 8B VLM）实现感知、导航、全身操控等多种任务，并展示了从仿真到真实的迁移能力。

**[See and Think: Embodied Agent in Virtual Environment](see_and_think_embodied_agent_in_virtual_environment.md)**

:   提出 STEVE，一个基于视觉感知、语言指令和代码动作三大组件的 Minecraft 开放世界具身智能体，通过 STEVE-21K 数据集微调 LLaMA-2 并结合视觉编码器和技能数据库，在科技树解锁和方块搜索任务上大幅超越现有方法。
