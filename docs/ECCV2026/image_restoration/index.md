---
title: >-
  ECCV2026 图像恢复论文汇总 · 12篇论文解读
description: >-
  12篇ECCV2026的图像恢复方向论文解读，涵盖图像恢复、超分辨率、多模态、导航、对抗鲁棒、个性化生成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2026"
  - "图像恢复"
  - "论文解读"
  - "论文笔记"
  - "超分辨率"
  - "多模态"
  - "导航"
  - "对抗鲁棒"
  - "个性化生成"
item_list:
  - u: "cogsenet_blind_image_deblurring_with_blur-conditioned_semantic_routing_and_expli/"
    t: "CogSENet: Blind Image Deblurring with Blur-Conditioned Semantic Routing and Explicit Frequency Fusion"
  - u: "fabric_image_demoiréing_benchmark_from_synthesis_to_restoration/"
    t: "Fabric Image Demoiréing Benchmark from Synthesis to Restoration"
  - u: "fidelity-_and_perception-aware_local_implicit_attention_for_arbitrary-scale_imag/"
    t: "Fidelity- and Perception-Aware Local Implicit Attention for Arbitrary-Scale Image Super-Resolution"
  - u: "flowdec_temporal_conditional_flow_decorruptor_for_robust_continuous_vision-langu/"
    t: "FlowDec: Temporal Conditional Flow Decorruptor for Robust Continuous Vision-Language Navigation"
  - u: "fma-netpp_motion-_and_exposure-aware_joint_video_super-resolution_and_deblurring/"
    t: "FMA-Net++: Motion- and Exposure-Aware Joint Video Super-Resolution and Deblurring"
  - u: "freemef_flexible_frame_mef/"
    t: "FreeMEF: 可灵活处理任意帧数的多曝光融合Transformer"
  - u: "freqortho-sr_frequency-guided_orthogonal_expert_learning_for_real-world_image_su/"
    t: "FreqOrtho-SR: Frequency-Guided Orthogonal Expert Learning for Real-World Image Super-Resolution"
  - u: "logicir_logic_gate_networks_for_image_restoration/"
    t: "LogicIR: Logic Gate Networks for Image Restoration"
  - u: "personalization_as_inverse_planning_learning_latent_design_intents_for_agentic_s/"
    t: "Personalization as Inverse Planning: Learning Latent Design Intents for Agentic Slide Generation via Structural Denoising"
  - u: "spectral_and_trajectory_regularization_for_diffusion_transformer_super-resolutio/"
    t: "Spectral and Trajectory Regularization for Diffusion Transformer Super-Resolution"
  - u: "tasktok_delving_into_task_tokens_for_task-driven_image_restoration/"
    t: "TaskTok: Delving into Task Tokens for Task-driven Image Restoration"
  - u: "uhd-mff_shattering_barriers_in_uhd_multi-focus_image_fusion/"
    t: "UHD-MFF: Shattering Barriers in Multi-Focus Ultra-High-Definition Image Fusion via Learnable Lookup Tables"
item_total: 12
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🖼️ 图像恢复

**🎞️ ECCV2026** · **12** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (135)](../../CVPR2026/image_restoration/index.md) · [🔬 ICLR2026 (61)](../../ICLR2026/image_restoration/index.md) · [🧪 ICML2026 (21)](../../ICML2026/image_restoration/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/image_restoration/index.md) · [🧠 NeurIPS2025 (26)](../../NeurIPS2025/image_restoration/index.md) · [📹 ICCV2025 (31)](../../ICCV2025/image_restoration/index.md)

🔥 **高频主题：** 图像恢复 ×5 · 超分辨率 ×4

**[CogSENet: Blind Image Deblurring with Blur-Conditioned Semantic Routing and Explicit Frequency Fusion](cogsenet_blind_image_deblurring_with_blur-conditioned_semantic_routing_and_expli.md)**

:   CogSENet 将盲图像去模糊从被动的像素回归重塑为主动的语义对齐重建过程，通过模拟鹰眼视觉系统的主动扫查扫描、视网膜功能分化和焦距适应，提出语义驱动的状态空间模块（SDSSM）、双频融合块（BFFB）和连续模糊场+CLIP语义联合调制三大核心设计，以仅 8.9M 参数在 GoPro、HIDE、RealBlur 上超越 EVSSM、FFTformer 等 SOTA 方法。

**[Fabric Image Demoiréing Benchmark from Synthesis to Restoration](fabric_image_demoiréing_benchmark_from_synthesis_to_restoration.md)**

:   本文首次系统研究织物图像去摩尔纹问题，提出基于物理成像链仿真的残差注入合成框架 PRISM（含 16,050 对数据的首个织物摩尔纹基准）与专为织物频谱纠缠特性设计的保守式恢复网络 FaDeNet，在 PSNR/SSIM/LPIPS 上大幅超越现有屏幕去摩尔纹方法。

**[Fidelity- and Perception-Aware Local Implicit Attention for Arbitrary-Scale Image Super-Resolution](fidelity-_and_perception-aware_local_implicit_attention_for_arbitrary-scale_imag.md)**

:   FPLIA 提出一个双流框架，将回归骨干的保真度特征与扩散模型的感知特征通过非对称双向交叉注意力（FPAM）和逐像素自适应选择（FPSM）融合，在 ASISR 上同时实现了高保真度和高感知质量。

**[FlowDec: Temporal Conditional Flow Decorruptor for Robust Continuous Vision-Language Navigation](flowdec_temporal_conditional_flow_decorruptor_for_robust_continuous_vision-langu.md)**

:   FlowDec 提出一种基于条件流匹配的图像去损坏框架，通过混合时序条件策略和动作质心引导过滤，在不修改 VLN 导航骨架的前提下增强其对多种视觉损坏的鲁棒性，在 R2R-CE 和 RxR-CE 上相对导航成功率分别提升 25.33% 和 9.38%，推理速度比扩散类 TTA 方法快 3-8 倍。

**[FMA-Net++: Motion- and Exposure-Aware Joint Video Super-Resolution and Deblurring](fma-netpp_motion-_and_exposure-aware_joint_video_super-resolution_and_deblurring.md)**

:   FMA-Net++ 提出基于 HRBA 层次化双向聚合块的非循环序列级框架，通过曝光时间感知调控模块（ETM）将帧级曝光信息注入特征，配合曝光感知的流引导动态滤波（FGDF）联合建模运动与曝光时变退化，在 VSRDB 任务上达到 SOTA 精度和推理速度。

**[FreeMEF: 可灵活处理任意帧数的多曝光融合Transformer](freemef_flexible_frame_mef.md)**

:   FreeMEF 提出"先去后回"（There and Back Again）的双阶段范式——先用循环状态空间模块将任意帧数的多曝光特征渐进聚合为全局表征，再以极端感知混合注意力引导基准帧恢复——在支持 2/3/5 帧灵活推理的同时显著抑制鬼影、提升动态范围。

**[FreqOrtho-SR: Frequency-Guided Orthogonal Expert Learning for Real-World Image Super-Resolution](freqortho-sr_frequency-guided_orthogonal_expert_learning_for_real-world_image_su.md)**

:   FreqOrtho-SR 提出频域引导的 LoRA 专家混合（FreqMoE）和正交梯度投影（OGP）两个核心模块，通过 FFT 退化特征驱动的自适应专家路由和像素-语义子空间正交约束，在单步扩散模型框架下实现退化自适应的真实世界图像超分辨率，在多个基准上取得最优或次优的保真度-感知质量平衡。

**[LogicIR: Logic Gate Networks for Image Restoration](logicir_logic_gate_networks_for_image_restoration.md)**

:   LogicIR 是首个专为图像恢复设计的逻辑门网络（LGN），通过全逻辑门 UNet 架构、可微比特解码层和 Index Shuffling 跨组通信机制，在去噪、去块、去雨任务上以远低于 BNN 和 LUT 方法的运算量（BOPs）达到竞争性恢复质量，证明了纯逻辑门计算在图像恢复中的可行性。

**[Personalization as Inverse Planning: Learning Latent Design Intents for Agentic Slide Generation via Structural Denoising](personalization_as_inverse_planning_learning_latent_design_intents_for_agentic_s.md)**

:   Spire 将页面级幻灯片个性化（PSP）建模为逆规划问题，通过对黄金幻灯片施加离散结构扰动构建自监督"去噪"信号，训练 Critic 和 Planner 两个 7B 级智能体在 RL 下协同迭代修正设计方案，无需依赖特定执行器即可推断用户的隐式设计意图，在视觉相似度和 VLM-as-Judge 评分上均显著优于基于 GPT 的强基线。

**[Spectral and Trajectory Regularization for Diffusion Transformer Super-Resolution](spectral_and_trajectory_regularization_for_diffusion_transformer_super-resolutio.md)**

:   StrSR 提出非对称判别蒸馏 + 频域分布匹配的联合框架，以 CLIP-ConvNeXt 判别器替代 DiT 判别器避免模型坍塌，并用频域分布损失（FDL）抑制 DiT 特有的网格状周期伪影，首次在 DiT 架构上实现高质量单步真实图像超分辨率。

**[TaskTok: Delving into Task Tokens for Task-driven Image Restoration](tasktok_delving_into_task_tokens_for_task-driven_image_restoration.md)**

:   TaskTok 提出了任务驱动图像恢复 (TDIR) 框架，利用 1D tokenizer 中 token 按索引位置专业化编码不同视觉属性的特性，通过可学习 token 开关和轻量级 token 精炼模块，选择性地只恢复与下游任务（分类/分割/检测）最相关的 token 子集，在显著提升下游任务性能的同时大幅降低计算开销。

**[UHD-MFF: Shattering Barriers in Multi-Focus Ultra-High-Definition Image Fusion via Learnable Lookup Tables](uhd-mff_shattering_barriers_in_uhd_multi-focus_image_fusion.md)**

:   本文提出首个面向 4K 超高清的多焦点图像融合数据集 UHD-MFF（1,950 对 3840×2160 图像），并设计尺度解耦的可学习查找表框架 UMF-LUT——用低分辨率 C-LUT 做区域级粗决策、高分辨率 D-LUT 做边缘级精修——在 SOTA 融合质量下仅 0.008M 参数就能以 90fps 实时推理 4K 图像并部署到手机。
