---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🤖 机器人/具身智能

**📷 CVPR2026** · 共 **14** 篇

**[Action–Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation](actiongeometry_prediction_with_3d_geometric_prior.md)**

:   利用预训练3D几何基础模型π3作为感知骨干，融合3D几何、2D语义和本体感知特征，通过扩散模型联合预测未来动作chunk和未来3D Pointmap，仅使用RGB输入就在RoboTwin双臂基准上全面超越点云方法。

**[AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots](atomicvla_unlocking_the_potential_of_atomic_skill.md)**

:   AtomicVLA 提出统一规划-执行框架，通过Think-Act自适应切换生成任务链和原子技能抽象，用技能引导MoE（SG-MoE）构建可扩展的原子技能专家库，在LIBERO-LONG上超π₀ 10%，真实世界持续学习超基线21%且遗忘仅1.3%。

**[DAWN: Pixel Motion Diffusion is What We Need for Robot Control](dawn_pixel_motion_diffusion_robot_control.md)**

:   提出 DAWN，一个两阶段全扩散的视觉语言动作框架——Motion Director（潜扩散模型）生成稠密像素运动场作为可解释的中间表示，Action Expert（扩散 Transformer 策略）将像素运动转换为可执行机器人动作；在 CALVIN 基准上取得 SOTA（平均长度 4.00），并在真实世界单臂/双臂操控中展现强泛化能力。

**[Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation](expert_pyramid_tuning_efficient_parameter_finetuni.md)**

:   提出 Expert Pyramid Tuning (EPT)，将 CV 中的多尺度特征金字塔思想引入 MoE-LoRA 框架，通过共享元知识子空间 + 不同尺度的反卷积专家 + 对比学习任务嵌入，以仅 0.41M 参数/任务在 GLUE 上达到 87.0% 均分（超越所有 MoE-LoRA 基线）。

**[GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](gecosrt_geometryaware_continual_adaptation_for_rob.md)**

:   GeCo-SRT提出持续跨任务Sim-to-Real迁移范式，利用局部几何特征的域不变性和任务不变性，通过几何感知MoE模块提取可复用的几何知识并用专家引导的优先经验回放防遗忘，在4个操作任务上比基线平均提升52%成功率且仅需1/6数据。

**[Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics](influence_malleability_in_linearized_attention_dua.md)**

:   通过NTK框架证明线性化注意力不会收敛到无限宽度核极限（需要宽度m=Ω(κ⁶)），并提出"影响可塑性"指标量化其双面效应：注意力比ReLU网络高6-9倍的数据依赖灵活性，既能降低近似误差也增加对抗脆弱性。

**[Language-Grounded Decoupled Action Representation for Robotic Manipulation (LaDA)](lada_robotic_manipulation.md)**

:   提出LaDA框架，将连续7-DoF动作解耦为平移/旋转/夹爪三个语言锚定的语义原语，通过软标签对比学习和自适应权重策略在共享嵌入空间中对齐跨任务动作表示，在LIBERO上达93.6%成功率（0.6B参数），MimicGen上67%平均成功率，超越所有基线。

**[MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent](mergevla_crossskill_model_merging_toward_a_general.md)**

:   MergeVLA 通过诊断 VLA 模型不可合并的两大根因（LoRA 参数冲突 + action expert 自注意力导致的架构不兼容），设计了稀疏激活的 task mask 和去除自注意力的 action expert 架构，实现了多个单任务 VLA 专家的免训练合并，在 LIBERO 上达到 90.2% 成功率。

**[MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents](mindpower_enabling_theoryofmind_reasoning_in_vlmba.md)**

:   MindPower 提出了以机器人为中心的心智理论（ToM）推理框架，将感知→信念→欲望→意图→决策→行动组织为六层推理层级，并用 Mind-Reward（基于 GRPO）优化推理一致性，在决策和动作生成上分别超过 GPT-4o 12.77% 和 12.49%。

**[PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments](panoaffordancenet_towards_holistic_affordance_grou.md)**

:   提出PanoAffordanceNet，首次定义360°室内环境中的全局affordance grounding任务，通过失真感知光谱调制器(DASM)和全球面密化头(OSDH)解决ERP几何失真和稀疏激活问题，配合多级训练目标抑制语义漂移，在自建360-AGD数据集上大幅超越现有方法（KLD从2.853→1.270）。

**[RADAR: Closed-Loop Robotic Data Generation via Semantic Planning and Autonomous Causal Environment Reset](radar_closedloop_robotic_data_generation_via_seman.md)**

:   提出RADAR——一个完全自主的闭环机器人操作数据生成引擎，通过VLM语义规划+GNN策略执行+VQA成功评估+FSM驱动的LIFO因果逆序环境重置四个模块，仅需2-5个人工演示即可持续生成高保真操作数据，在仿真中复杂长horizon任务达到90%成功率。

**[RC-NF: Robot-Conditioned Normalizing Flow for Real-Time Anomaly Detection in Robotic Manipulation](rcnf_robot_conditioned_normalizing_flow_anomaly.md)**

:   提出RC-NF，一种基于条件归一化流的实时异常检测模型，通过解耦处理机器人状态和物体轨迹特征，仅需正样本无监督训练即可在100ms内检测VLA模型执行中的OOD异常，在LIBERO-Anomaly-10上以约8% AUC和10% AP的优势超越SOTA（包括GPT-5、Gemini 2.5 Pro等VLM基线）。

**[RehearseVLA: Simulated Post-Training for VLAs with Physically-Consistent World Model](rehearsevla_simulated_posttraining_world_model.md)**

:   针对 VLA 模型在数据稀缺场景下的性能退化和真实环境不可重置的限制，提出 RehearseVLA——用物理一致的世界模型模拟器替代真实物理交互进行 RL 后训练，配合 VLM 引导的即时反射器提供奖励信号和终止预测，仅用每个任务 5 个专家演示即可显著提升 VLA 在复杂操控任务上的表现。

**[The Coherence Trap: MLLM-Crafted Narratives Exploit Manipulated Visual Contexts](the_coherence_trap_when_mllmcrafted_narratives_exp.md)**

:   揭示现有多模态虚假信息检测的两个根本缺陷（低估MLLM生成的语义一致虚假叙事+依赖简单不对齐的伪影），构建441k样本的MDSM数据集（图像篡改+MLLM生成语义对齐文本），并提出AMD框架（Artifact Pre-perception + Manipulation-Oriented Reasoning），在跨域检测中达88.18 ACC / 60.25 mAP / 61.02 mIoU。
