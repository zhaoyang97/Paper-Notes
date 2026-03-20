<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧑 人体理解

**📷 CVPR2026** · 共 **8** 篇

**[Bilevel Layer-Positioning LoRA for Real Image Dehazing](bilevel_lora_real_image_dehazing.md)**

:   提出H2C文本引导无监督损失（利用CLIP将去雾重构为语义对齐问题）和BiLaLoRA双层优化策略（自动搜索最佳LoRA注入层），实现高效且即插即用的合成到真实域去雾适配。

**[Breaking the Tuning Barrier: Zero-Hyperparameters Yield Multi-Corner Analysis Via Learned Priors](breaking_the_tuning_barrier_zerohyperparameters_yi.md)**

:   用TabPFN（在百万回归任务上预训练的基础模型）替代传统手工先验，实现零超参数的SRAM多角良率分析，通过注意力机制自动进行跨角知识迁移，配合自动特征选择（1152D到48D）和不确定性引导的主动学习，达到SOTA精度（MRE低至0.11%）同时降低10倍以上验证成本。

**[GeoWorld: Geometric World Models](geoworld_geometric_world_models.md)**

:   在V-JEPA 2中引入双曲流形表示（Hyperbolic JEPA）和几何强化学习（GRL），利用测地线距离编码层次关系，通过能量函数优化实现更稳定的长时域规划，3步规划提升约3% SR，超越GPT-5 zero-shot。

**[Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs](graph2eval_multimodal_task_generation_agents.md)**

:   提出 Graph2Eval，一个基于知识图谱的自动化多模态 Agent 任务生成框架——从异构外部数据源构建知识图谱作为结构化任务空间，通过子图采样和元路径引导的任务构造生成语义一致且可解的 Agent 评测任务，相比 LLM 直接生成的任务提升语义一致性 20% 和可解性 17%，并发布了 1,319 个任务的 Graph2Eval-Bench 数据集。

**[L2GTX: From Local to Global Time Series Explanations](l2gtx_from_local_to_global_time_series_explanation.md)**

:   提出L2GTX——一种完全模型无关的方法，通过LOMATCE提取参数化时间事件原语的局部解释，再经层次聚类合并、贪心预算选择和事件聚合，生成紧凑且忠实的类级全局时间序列解释，在6个UCR数据集上全局忠实度（R²）在不同合并粒度下保持稳定（FCN上ECG200达0.792）。

**[Mobile-VTON: High-Fidelity On-Device Virtual Try-On](mobile_vton_ondevice_virtual_tryon.md)**

:   首个全离线移动端扩散式虚拟试穿框架，基于TeacherNet-GarmentNet-TryonNet (TGT)架构，通过特征引导对抗蒸馏(FGA)将SD3.5 Large的能力迁移到415M参数的轻量学生网络，在VITON-HD和DressCode上以1024×768分辨率匹配甚至超越服务器端基线，端到端推理时间约80秒（小米17 Pro Max）。

**[Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](referencefree_image_quality_assessment_for_virtual.md)**

:   构建了大规模人工标注虚拟试穿质量数据集VTON-QBench（62,688张图像，431,800条标注），并提出VTON-IQA无参考质量评估框架，通过交错交叉注意力模块实现与人类感知高度对齐的图像级质量预测。

**[Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach](team_ras_in_10th_abaw_competition_multimodal_valen.md)**

:   提出三模态连续VA估计方法，首次将VLM(Qwen3-VL-4B)生成的情感行为描述嵌入作为独立模态，与GRADA人脸编码器和WavLM音频特征通过两种融合策略(DCMMOE和RAAV)组合，在Aff-Wild2上达到CCC 0.658(dev)/0.62(test)。
