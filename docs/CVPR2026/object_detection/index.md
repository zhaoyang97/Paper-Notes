<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎯 目标检测

**📷 CVPR2026** · 共 **10** 篇

**[ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection](abra_teleporting_finetuned_knowledge_across_domain.md)**

:   将域适应建模为权重空间的SVD旋转对齐问题：分解域与类知识，通过闭式正交Procrustes解将源域类特定残差"传送"到无标注的目标域，实现零样本跨域类别检测。

**[DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning](dreamvideoomni_omnimotion_controlled_multisubject.md)**

:   统一框架同时实现多主体身份定制和全运动控制（全局运动 + 局部运动 + 相机运动），通过渐进式两阶段训练（有监督微调 + 潜空间身份奖励反馈学习）解决身份保持与运动控制之间的固有冲突。

**[Evaluating Few-Shot Pill Recognition Under Visual Domain Shift](evaluating_fewshot_pill_recognition_under_visual_d.md)**

:   从部署导向视角系统评估了小样本药丸识别在跨数据集域偏移下的表现，发现语义分类1-shot即可饱和(准确率>0.989)，但遮挡重叠场景下定位和召回急剧退化，训练数据的视觉真实性(多药丸、杂乱场景)是决定小样本泛化鲁棒性的主要因素。

**[EW-DETR: Evolving World Object Detection via Incremental Low-Rank DEtection TRansformer](ewdetr_evolving_world_object_detection.md)**

:   提出Evolving World Object Detection (EWOD)范式和EW-DETR框架，通过增量LoRA适配器、查询范数物体性适配器和熵感知未知混合三个模块，在无需存储旧数据的条件下同时解决类别增量学习、域迁移自适应和未知目标检测，FOGS指标较现有方法提升57.24%。

**[Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection](mitigating_memorization_in_texttoimage_diffusion_v.md)**

:   提出训练时区域感知提示增强(RAPTA)和注意力驱动多模态复制检测(ADMCD)两个互补模块，前者通过检测器proposal生成语义接地的提示变体来缓解扩散模型的训练数据记忆化，后者融合patch/CLIP/纹理三流特征实现零训练复制检测，在LAION-10k上将复制率从7.4降至2.6。

**[MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization](mokus_leveraging_crossmodal_knowledge_transfer_for.md)**

:   提出"知识感知概念定制"新任务，发现LLM文本编码器中的知识编辑可以自然迁移到视觉生成模态（跨模态知识迁移），基于此提出MoKus框架：先用LoRA微调将稀有token绑定为视觉概念的锚表征，再通过知识编辑技术将多条自然语言知识高效映射到锚表征上，每条知识更新仅需约7秒。

**[RADAR: Closed-Loop Robotic Data Generation via Semantic Planning and Autonomous Causal Environment Reset](radar_closedloop_robotic_data_generation_via_seman.md)**

:   提出RADAR——一个完全自主的闭环机器人操作数据生成引擎，通过VLM语义规划+GNN策略执行+VQA成功评估+FSM驱动的LIFO因果逆序环境重置四个模块，仅需2-5个人工演示即可持续生成高保真操作数据，在仿真中复杂长horizon任务达到90%成功率。

**[ReHARK: Refined Hybrid Adaptive RBF Kernels for Robust One-Shot Vision-Language Adaptation](rehark_refined_hybrid_adaptive_rbf_kernels_for_rob.md)**

:   提出ReHARK——一个训练免的CLIP one-shot适应框架，通过融合CLIP文本知识、GPT3语义描述和视觉原型构建混合先验，结合多尺度RBF核在RKHS中做全局近端正则化，在11个基准上以65.83%平均准确率刷新one-shot SOTA。

**[Show, Don't Tell: Detecting Novel Objects by Watching Human Videos](show_dont_tell_detecting_novel_objects_by_watching.md)**

:   提出"Show, Don't Tell"范式：通过观看人类演示视频，自动构建新物体标注数据集（SODC），训练轻量级定制检测器（MOD），完全绕过语言描述和prompt engineering，在真实机器人分拣任务上成功部署。

**[SLICE: Semantic Latent Injection via Compartmentalized Embedding for Image Watermarking](slice_semantic_latent_injection_via_compartmentali.md)**

:   提出SLICE框架，将图像语义解耦为四个因子（主体/环境/动作/细节），各自锚定到扩散模型初始噪声的不同空间分区，实现细粒度语义感知水印——不仅能检测篡改，还能精确定位被篡改的语义因子，且完全无需训练。
