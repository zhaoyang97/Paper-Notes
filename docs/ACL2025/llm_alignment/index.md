---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚖️ 对齐 / RLHF

**💬 ACL2025** · 共 **23** 篇

**[AutoMixAlign: Adaptive Data Mixing for Multi-Task Preference Optimization in LLMs](automixalign_adaptive_data_mixing.md)**

:   AutoMixAlign 提出了一种理论驱动的多任务偏好优化数据混合方法：先训练各任务的 specialist model 确定最优 loss 基线，再通过 minimax 优化自适应调整数据混合比例，优先处理 excess loss（与 specialist 的差距）最大的任务，在 helpfulness/harmlessness/reasoning 多任务 DPO 中平均提升 9.42%。

**[Cheems: A Practical Guidance for Building and Evaluating Chinese Reward Models from Scratch](cheems_chinese_reward_models.md)**

:   为弥补中文 Reward Model 资源的空白，本文构建了 CheemsBench（首个大规模中文 RM 评测基准）和 CheemsPreference（首个大规模中文偏好数据集），通过人机协作标注 + 远程监督过滤策略训练的 CheemsRM 在中文场景显著超越现有所有开源 RM。

**[Curiosity-Driven Reinforcement Learning from Human Feedback](curiosity_driven_rlhf.md)**

:   CD-RLHF 将好奇心驱动探索（curiosity-driven RL）引入 RLHF，通过前向动力学模型的预测误差作为内在奖励，结合 top-k 门控过滤与 reward whitening，在不损失对齐质量的前提下大幅提升 LLM 输出多样性（Llama-3.2-1B 上 Diversity 提升 40.26%，EAD 提升 8.92%）。

**[DiffPO: Diffusion Alignment with Direct Preference Optimization](diffpo_diffusion_alignment.md)**

:   提出 DiffPO，将 LLM 对齐重新建模为句子级扩散去噪过程，通过 parallel decoding 实现高效推理时对齐，作为即插即用模块可增强任意底座模型的对齐质量。

**[Understanding Impact of Human Feedback via Influence Functions](influence_functions_rlhf.md)**

:   首次将影响函数系统性应用于 RLHF 奖励模型的人类反馈审计，通过 DataInf 近似 + OPORP 梯度压缩（160MB→256KB）实现 2.5 倍加速，在长度偏差和谄媚偏差检测上显著超越 GPT-4o 等基线（AUC 0.8 vs ~0.6），并在 Anthropic-HH 原始数据集中发现 47% 的错标样本，同时展示了影响函数指导非专家标注者向专家策略对齐的能力。

**[IOPO: Empowering LLMs with Complex Instruction Following via Input-Output Preference Optimization](iopo_input_output_preference.md)**

:   提出 IOPO（Input-Output Preference Optimization），在传统 DPO 仅优化输出偏好的基础上，引入输入偏好建模——让模型学习"给定回复 y，哪个指令 x 更匹配"，从而增强对复杂多约束指令的细粒度感知能力；同时构建了包含 120K 训练数据、1K 评测数据、覆盖 5 大类 26 个约束维度的 Trace 基准。

**[JailbreakRadar: Comprehensive Assessment of Jailbreak Attacks Against LLMs](jailbreakradar_comprehensive_assessment_jailbreak_attacks.md)**

:   本文提出了一个全面的越狱攻击评估框架 JailbreakRadar，收集了17种代表性越狱攻击方法，建立了六类攻击分类体系，并在9个对齐LLM上进行了大规模系统性评测，揭示了不同类型攻击在实用性和防御鲁棒性上的关键差异。

**[Enhancing Safe and Controllable Protein Generation via Knowledge Preference Optimization](kpo_protein_safety.md)**

:   提出知识引导偏好优化（KPO）框架，通过蛋白质安全知识图谱识别安全/危险序列作为偏好信号，用强化学习训练蛋白质语言模型减少有害蛋白质序列的生成概率，同时保持功能性——为蛋白质生成的生物安全提供保障框架。

**[LSSF: Safety Alignment via Low-Rank Safety Subspace Fusion](lssf_safety_subspace.md)**

:   LSSF 发现 LLM 的安全信息集中在低秩子空间中，利用 SVD 构建低秩投影矩阵提取安全向量的主成分，并通过安全奇异值熵自适应确定每层保留秩，最终将低秩安全主成分通过线性算术融合到微调后模型中，实现免训练的后验安全恢复，同时最大化保持下游任务性能。

**[M2S: Multi-turn to Single-turn jailbreak in Red Teaming for LLMs](m2s_multiturn_to_singleturn_jailbreak_in.md)**

:   提出 M2S 框架，通过三种简单的格式转换方法（Hyphenize/Numberize/Pythonize）将多轮人类越狱对话压缩为单轮 prompt，不仅保持甚至超越原始多轮攻击效果（ASR 高达 95.9%，比多轮提升最多 17.5%），同时 token 使用量减半以上。

**[M-RewardBench: Evaluating Reward Models in Multilingual Settings](m_rewardbench.md)**

:   构建首个多语言奖励模型评估基准M-RewardBench（23种语言、2.87K偏好实例，覆盖对话/安全/推理/翻译四类能力），系统评估多种RM后发现英语与非英语RM性能存在显著差距，且翻译质量和语言资源量对RM表现有重要影响。

**[MPO: Multilingual Safety Alignment via Reward Gap Optimization](mpo_multilingual_safety_alignment.md)**

:   MPO 发现 LLM 在主导语言（英文）和目标语言间的隐式 Reward Gap 与安全性能强相关，提出直接最小化两者 Reward Gap 差异来将主导语言的安全对齐能力迁移到多语言，在三个模型上显著降低了低资源语言的攻击成功率且不损害通用能力。

**[Mutual-Taught for Co-adapting Policy and Reward Models](mutual_taught_policy_reward.md)**

:   Mutual-Taught 提出了一种基于 EM 算法的自训练框架，在偏好优化过程中同时迭代更新 policy model 和 reward model：E-step 用当前 RM 优化 PM，M-step 用 PM 更新前后的输出差异构建伪偏好对来更新 RM，解决了分布偏移导致的 reward hacking 问题，8B 模型在 AlpacaEval-2 达到 54.1% LC win rate。

**[Optimal Transport-Based Token Weighting for Enhanced Preference Optimization (OTPO)](otpo_token_weighting.md)**

:   提出 OTPO，利用无平衡最优传输（Unbalanced OT）在 chosen/rejected 回复的 token 语义表示之间建立匹配关系，从传输计划的边际分布导出每个 token 的重要性权重，替换 DPO 中的均匀权重，使偏好优化聚焦于语义关键 token，在 AlpacaEval2 上 LC WR 比 DPO 提升最高 10.9%。

**[Whose Boat Does it Float? Improving Personalization in Preference Optimization](personalized_preference_opt.md)**

:   提出"溯因推理"视角的偏好个性化方法：先用 LLM 推断偏好 chosen/rejected 回答背后的用户画像（Persona Inference），再用画像增强的偏好数据训练模型（Persona Tailoring），显著提升模型对不同用户需求的个性化适配能力。

**[PIG: Privacy Jailbreak Attack on LLMs via Gradient-based Iterative Prompts](pig_privacy_jailbreak.md)**

:   提出 PIG 框架，通过识别隐私查询中的 PII 实体类型、构建隐私上下文示例、并利用三种基于梯度的迭代优化策略更新上下文，实现对 LLM 的高效隐私越狱攻击，在白盒和黑盒模型上均达到 SOTA。

**[Towards Reward Fairness in RLHF: From a Resource Allocation Perspective](reward_fairness_rlhf.md)**

:   将 RLHF 中的各种奖励偏差（长度偏差、类别偏差、社会偏差）统一定义为"奖励不公平"问题，从资源分配视角提出两种偏差无关的缓解方法——公平正则化和公平系数——在不针对特定偏差设计的情况下有效缓解多种偏差，实现更公平的人类偏好对齐。

**[RISE: Subtle Errors in Reasoning: Preference Learning via Error-injected Self-editing](rise_error_inject_preference.md)**

:   RISE 发现 LLM 约 75% 的数学错误是微妙的步内错误（计算错误、替代错误、遗漏项），提出让 LLM 自编辑（REPLACE/SWAP/DELETE）向正确解注入预定义微妙错误来构造难负样本，配合步级+序列级+自适应 NLL 三合一的错误感知 DPO 训练，仅用约 4.5K 样本在 Qwen2-7B 上 GSM8K +3.0%、MATH +7.9%，且泛化到逻辑推理和代码生成。

**[SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings](sea_lowresource_safety_alignment_for_multimodal.md)**

:   提出 SEA 框架，通过梯度优化生成合成模态 embedding（不需要真实图像/视频/音频），仅用文本安全数据就能实现多模态 LLM 的安全对齐，在单张 RTX3090 上 24 秒即可合成高质量 embedding，同时发布了视频和音频安全基准 VA-SafetyBench。

**[SynthesizeMe! Inducing Persona-Guided Prompts for Personalized Reward Models in LLMs](synthesizeme_persona_prompts.md)**

:   提出SynthesizeMe，通过bootstrap推理→合成用户画像→筛选信息性示例三步流程，无需微调即可为个性化奖励模型构建有效prompt，在Chatbot Arena上提升LLM-as-a-Judge个性化准确率4.4%。

**[Teaching an Old LLM Secure Coding: Localized Preference Optimization on Distilled Preferences](teaching_an_old_llm_secure_coding.md)**

:   提出 DiSCo（从前沿 LLM 蒸馏的安全代码偏好数据集，10K 实例覆盖 431 种 CWE）和 LPO（局部偏好优化算法，仅在安全相关 token 上传播损失），在四个安全编码基准上减少 19-40% 的安全问题，同时提升 3-10% 的代码质量。

**[Think&Cite: Improving Attributed Text Generation with Self-Guided Tree Search and Progress Reward Modeling](think_cite_attributed_text_gen.md)**

:   将归因文本生成（带引用的文本生成）建模为多步推理问题，提出自引导蒙特卡洛树搜索（SG-MCTS）结合进度奖励建模（PRM），通过多路径搜索+中间状态反思+生成/归因双维度进度奖励，在 ALCE 基准三个数据集上显著超越所有基线。

**[A Troublemaker with Contagious Jailbreak Makes Chaos in Honest Towns](tmcht_contagious_jailbreak_multiagent.md)**

:   提出 TMCHT 多智能体多拓扑攻击评估框架和 ARCJ（对抗性复制传染越狱）方法，解决了多智能体系统中单智能体攻击方法面临的"毒性消散"问题——通过优化检索后缀确保中毒样本易被检索，优化复制后缀使中毒信息具有传染性自我复制能力，在 line/star 拓扑和 100 智能体系统中分别提升 23.51%/18.95%/52.93% 的攻击成功率。
