<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎨 图像生成

**📷 CVPR2026** · 共 **28** 篇

**[Agentic Retoucher for Text-To-Image Generation](agentic_retoucher_for_texttoimage_generation.md)**

:   Agentic Retoucher 将 T2I 生成后的缺陷修复重构为"感知→推理→行动"的人类式闭环决策过程，用三个协作 agent 分别做上下文感知的扭曲检测、人类对齐的诊断推理和自适应局部修复，在 GenBlemish-27K 上 plausibility 提升 2.89 分，83.2% 的结果被人类评为优于原图。

**[All-in-One Slider for Attribute Manipulation in Diffusion Models](all_in_one_slider_attribute_manipulation.md)**

:   提出 All-in-One Slider 框架，通过在文本嵌入空间上训练一个属性稀疏自编码器（Attribute Sparse Autoencoder），将多种人脸属性解耦为稀疏的语义方向，实现单一轻量模块对 52+ 种属性的细粒度连续控制，并支持多属性组合和未见属性的零样本操控。

**[AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys](asbridge_a_bidirectional_generative_framework_brid.md)**

:   提出 AS-Bridge，基于双向 Brownian Bridge 扩散过程建模地面巡天（LSST）与空间巡天（Euclid）观测之间的随机映射，同时实现跨巡天图像转换和稀有天文事件检测。

**[Attribution as Retrieval: Model-Agnostic AI-Generated Image Attribution](attribution_as_retrieval_modelagnostic_aigenerated.md)**

:   将 AI 生成图像归因从分类范式重新定义为实例检索问题，提出 LIDA 框架：利用低位平面提取生成器指纹，通过无监督预训练 + 少样本适配实现开放集归因，在 GenImage 和 WildFake 上全面超越现有方法。

**[BiGain: Unified Token Compression for Joint Generation and Classification](bigain_token_compression.md)**

:   提出BiGain——一个训练免的token压缩框架，通过频域分离（保留高频细节+低中频语义），在扩散模型加速时同时保持生成质量和分类能力。70% token合并下分类精度+7.15%且FID反而更好。

**[CDG: Guiding Diffusion Models with Semantically Degraded Conditions](cdg_condition_degradation_guidance_diffusion.md)**

:   提出CDG替代CFG——用语义退化条件替代空null prompt作为负面引导，将引导信号从粗粒度"好vs空"变为精细"好vs差一点"，在SD3/FLUX/Qwen-Image上显著提升组合精度，零额外计算。

**[coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation](codrawagents_a_multiagent_dialogue_framework_for_c.md)**

:   提出 coDrawAgents 交互式多智能体对话框架，通过解释器、规划器、检查器、画家四个专业智能体的闭环协作，以分治策略逐步规划布局并基于画布视觉上下文纠错，在 GenEval 上达到 0.94 的 SOTA 组合保真度。

**[CognitionCapturerPro: Towards High-Fidelity Visual Decoding from EEG/MEG via Multi-modal Information and Asymmetric Alignment](cognitioncapturerpro_towards_highfidelity_visual_d.md)**

:   提出 CognitionCapturerPro，通过不确定性加权掩蔽、多模态融合编码器、共享主干对齐模块和多分支 IP-Adapter 扩散重建，解决 EEG 视觉解码中的保真度损失和表征偏移问题，在 THINGS-EEG 上 Top-1 检索达 61.2%、Top-5 达 90.8%。

**[ConsistCompose: Unified Multimodal Layout Control for Image Composition](consistcompose_multimodal_layout_control.md)**

:   提出 ConsistCompose，通过将布局坐标直接嵌入语言prompt（LELG范式），在统一多模态框架中实现布局可控的多实例图像生成；构建340万样本的ConsistCompose3M数据集提供布局+身份监督；配合坐标感知CFG机制，在COCO-Position上实现布局IoU 7.2%提升和AP 13.7%提升，同时保持通用理解能力。

**[D2C: Accelerating Diffusion Model Training under Minimal Budgets via Condensation](d2c_diffusion_dataset_condensation.md)**

:   首次将数据集压缩(Dataset Condensation)应用于扩散模型训练，提出D2C两阶段框架——Select阶段用扩散难度分数+区间采样选出紧凑子集、Attach阶段为每个样本附加文本和视觉表示——仅用0.8% ImageNet(10K图像)在40K步即达FID 4.3,比REPA快100×、比vanilla SiT快233×。

**[DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching](disca_accelerating_video_diffusion_transformers_wi.md)**

:   DisCa 首次提出"可学习特征缓存 + 步蒸馏"兼容的加速方案：用轻量神经预测器替代传统手工缓存策略，并通过 Restricted MeanFlow 稳定大规模视频模型的蒸馏，在 HunyuanVideo 上实现 11.8× 近无损加速。

**[DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](ditic_aligned_diffusion_transformer_for_efficient.md)**

:   将预训练文生图 DiT 适配为高效单步图像压缩解码器，通过方差引导重建流、自蒸馏对齐和潜空间条件引导三种对齐机制，在 32× 下采样的深层潜空间中实现 SOTA 感知质量，同时比现有扩散压缩方法解码快 30 倍。

**[DPCache: 去噪即路径规划——免训练扩散模型加速](dpcache_denoising_path_planning_diffusion_accel.md)**

:   将扩散采样加速形式化为全局路径规划问题，通过Path-Aware Cost Tensor量化路径依赖的跳步误差，用动态规划选出最优关键时间步序列，在FLUX上实现4.87×加速且ImageReward反超全步基线。

**[Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking](editing_away_the_evidence_diffusionbased_image_man.md)**

:   本文从理论和实验两方面系统分析了扩散编辑（instruction/drag/composition）如何非对抗性地破坏鲁棒隐形水印，推导出 SNR 衰减和互信息下界，揭示常规后处理鲁棒性不能推广到生成式变换。

**[DIAE: Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception](enhancing_image_aesthetics_with_dualconditioned_di.md)**

:   提出DIAE——一个基于SD1.5的图像美学增强框架，通过多模态美学感知(MAP)将模糊的美学指令转化为HSV+轮廓图的视觉控制信号，配合"不完美配对"数据集IIAEData和双分支监督训练策略，在美学提升(LAION score +17.4%)和内容一致性(CLIP-I 0.784)上同时优于InstructPix2Pix等SOTA编辑方法。

**[FDeID-Toolbox: Face De-Identification Toolbox](fdeidtoolbox_face_deidentification_toolbox.md)**

:   发布 FDeID-Toolbox，一个模块化人脸去识别研究工具箱，统一了数据加载、方法实现（经典到 SOTA 生成模型）、推理流水线和三维评估协议（隐私/效用/质量），解决该领域实验碎片化和结果不可比的问题。

**[Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems](fractals_made_practical_denoising_diffusion_as_par.md)**

:   证明 DDIM 确定性反向链等价于分区迭代函数系统（PIFS），从分形几何推导出三个可计算量（收缩阈值 $L_t^*$、对角膨胀函数 $f_t(\lambda)$、全局膨胀阈值 $\lambda^{**}$），统一解释了余弦调度偏移、分辨率 logSNR 偏移、Min-SNR 损失加权和 Align Your Steps 采样调度四种经验设计选择。

**[HaltNav: Reactive Visual Halting over Lightweight Topological Priors for Robust Vision-Language Navigation](haltnav_reactive_visual_halting_over_lightweight_t.md)**

:   提出层级导航框架 HaltNav，结合轻量文本拓扑图 (osmAG) 全局规划 + VLN 模型局部执行，并引入反应式视觉停止 (RVH) 机制在遇到未知障碍时实时中断、更新拓扑、重规划绕行，在仿真和真实机器人上均显著优于基线。

**[InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](interedit_navigating_textguided_multihuman_3d_moti.md)**

:   首次定义多人3D运动编辑(TMME)任务，构建5161个源-目标-指令三元组的InterEdit3D数据集，提出基于同步无分类器引导的条件扩散模型InterEdit，通过语义感知规划Token对齐和交互感知频域Token对齐两个核心模块，在指令跟随(g2t R@1 30.82%)和源保持(g2s R@1 17.08%)上全面超越基线。

**[LinVideo: A Post-Training Framework towards O(n) Attention in Efficient Video Generation](linvideo_linear_attention_video_generation.md)**

:   首个data-free后训练框架LinVideo，通过选择性转移自动选择最适合替换为线性注意力的层+任意时刻分布匹配(ADM)目标函数高效恢复性能，实现Wan 1.3B/14B的1.43-1.71×加速且质量无损，叠加4步蒸馏后达15.9-20.9×加速。

**[OARS: Process-Aware Online Alignment for Generative Real-World Image Super-Resolution](oars_processaware_online_alignment_for_generative.md)**

:   提出了OARS框架，通过基于MLLM的过程感知奖励模型COMPASS和渐进式在线强化学习，将生成式真实世界超分辨率模型与人类视觉偏好对齐，在感知质量和保真度之间实现自适应平衡。

**[One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers](one_model_many_budgets_elastic_latent_interfaces_f.md)**

:   提出ELIT（Elastic Latent Interface Transformer），通过在DiT中插入可变长度的潜在token接口和轻量级Read/Write交叉注意力层，将计算量与输入分辨率解耦，使单一模型支持多种推理预算，在ImageNet-1K 512px上FID和FDD分别提升35.3%和39.6%。

**[PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](pixelrush_ultrafast_trainingfree_highresolution_im.md)**

:   PixelRush 首次实现了免训练的单步高分辨率图像生成，通过部分 DDIM 反转（只扰动到中间时间步而非全噪声）+ 少步扩散模型 + 高斯滤波 patch 融合 + 噪声注入，在单卡 A100 上 20 秒生成 4K 图像，比 SOTA 快 10-35× 且 FID 更优（50.13 vs 52.87）。

**[PROMO: Promptable Outfitting for Efficient High-Fidelity Virtual Try-On](promo_promptable_virtual_tryon_efficient.md)**

:   PROMO基于FLUX Flow Matching DiT骨干，通过潜空间多模态条件拼接、时序自参考KV缓存、3D-RoPE分组条件、以及fine-tuned VLM风格提示系统，在去除传统参考网络的前提下实现了高保真且高效的多件服装虚拟试穿，推理速度比无加速版快2.4倍，在VITON-HD和DressCode上超越现有VTON和通用图像编辑方法。

**[RAZOR: Ratio-Aware Layer Editing for Targeted Unlearning in Vision Transformers and Diffusion Models](razor_ratio_aware_unlearning_vit_diffusion.md)**

:   提出 RAZOR, 一种基于比率感知梯度评分的多层协调编辑方法, 用于 ViT 和扩散模型的目标遗忘: 通过 forget/retain 梯度的比率和余弦对齐度联合评分, 识别对遗忘贡献最大且对保留损害最小的层/头, 实现一次性高效遗忘, 在 CLIP 身份遗忘上达到 SOTA.

**[SegQuant: A Semantics-Aware and Generalizable Quantization Framework for Diffusion Models](segquant_diffusion_model_quantization.md)**

:   提出 SegQuant，一个面向部署的扩散模型后训练量化框架，通过基于计算图静态分析的语义感知分段量化（SegLinear）和硬件原生的双尺度极性保持量化（DualScale），在 SD3.5、FLUX、SDXL 上实现跨架构通用的高保真 W8A8/W4A8 量化，同时保持与 TensorRT 等工业推理引擎的兼容性。

**[SOLACE: Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards](solace_self_confidence_rewards_t2i.md)**

:   用T2I模型自身的去噪自信心（对注入噪声的恢复精度）替代外部奖励做后训练，在组合生成、文字渲染、文图对齐上获一致提升，与外部奖励互补可缓解reward hacking。

**[Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework](taming_scorebased_denoisers_in_admm_a_convergent_p.md)**

:   提出ADMM-PnP with AC-DC去噪器，通过三阶段修正-去噪流程(自动修正+方向修正+基于分数的去噪)将扩散先验集成到ADMM原始-对偶框架中，解决了ADMM迭代与扩散训练流形的几何不匹配问题，同时在两种条件下建立了收敛保证，在7种逆问题上一致优于DAPS/DPS/DiffPIR等基线。
