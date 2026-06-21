---
title: >-
  ICLR2026 信号/通信论文汇总 · 7篇论文解读
description: >-
  7篇ICLR2026的信号/通信方向论文解读，涵盖压缩/编码、扩散模型、图像恢复等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "信号/通信"
  - "论文解读"
  - "论文笔记"
  - "压缩/编码"
  - "扩散模型"
  - "图像恢复"
item_list:
  - u: "advancing_spatiotemporal_representations_in_spiking_neural_networks_via_parametr/"
    t: "Advancing Spatiotemporal Representations in Spiking Neural Networks via Parametric Invertible Transformation"
  - u: "efficient_message-passing_transformer_for_error_correcting_codes/"
    t: "Efficient Message-Passing Transformer for Error Correcting Codes"
  - u: "enhancing_instruction_following_of_llms_via_activation_steering_with_dynamic_rej/"
    t: "Enhancing Instruction Following of LLMs via Activation Steering with Dynamic Rejection"
  - u: "lossy_common_information_in_a_learnable_gray-wyner_network/"
    t: "Lossy Common Information in a Learnable Gray-Wyner Network"
  - u: "mamba-3_improved_sequence_modeling_using_state_space_principles/"
    t: "Mamba-3: Improved Sequence Modeling using State Space Principles"
  - u: "synchronizing_probabilities_in_model-driven_lossless_compression/"
    t: "Synchronizing Probabilities in Model-Driven Lossless Compression"
  - u: "ts-ddae_a_novel_temporal-spectral_denoising_diffusion_autoencoder_for_wireless_s/"
    t: "TS-DDAE: A Novel Temporal-Spectral Denoising Diffusion AutoEncoder for Wireless Signal Recognition Model Pre-training"
item_total: 7
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📡 信号/通信

**🔬 ICLR2026** · **7** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (2)](../../CVPR2026/signal_comm/index.md) · [🧪 ICML2026 (2)](../../ICML2026/signal_comm/index.md) · [🤖 AAAI2026 (3)](../../AAAI2026/signal_comm/index.md) · [🧠 NeurIPS2025 (5)](../../NeurIPS2025/signal_comm/index.md) · [📹 ICCV2025 (3)](../../ICCV2025/signal_comm/index.md) · [🧪 ICML2025 (3)](../../ICML2025/signal_comm/index.md)

**[Advancing Spatiotemporal Representations in Spiking Neural Networks via Parametric Invertible Transformation](advancing_spatiotemporal_representations_in_spiking_neural_networks_via_parametr.md)**

:   针对脉冲神经网络（SNN）二值脉冲表示能力受限、替代梯度失配两大顽疾，本文提出参数化可逆变换 PIT——在神经元发放（firing）前后以共轭方式各做一次可逆线性变换，发放前把膜电位分布"重排"成易量化的形态、发放后把整数脉冲"增广"成跨时空的实值输出，同时配一个把输入推离量化决策边界的修正替代梯度，并用线性代数刻画了 SNN 时空表示容量；在 CIFAR、ImageNet、DVS 等数据集上多种架构均刷新 SOTA（如 SEW ResNet34 涨 5.62%）。

**[Efficient Message-Passing Transformer for Error Correcting Codes](efficient_message-passing_transformer_for_error_correcting_codes.md)**

:   EfficientMPT 把 Transformer 纠错码解码器里 $O(n^2)$ 的标准注意力换成一套只靠"全局 query 向量 + 逐元素乘"的线性复杂度 EEC 注意力，在保持与 SOTA（CrossMPT）相当的纠错性能的同时，对长 LDPC 码把显存和 FLOPs 砍掉数十倍，并且参数量与码长无关、能当一个可微调的纠错"基础模型"。

**[Enhancing Instruction Following of LLMs via Activation Steering with Dynamic Rejection](enhancing_instruction_following_of_llms_via_activation_steering_with_dynamic_rej.md)**

:   提出 Directer（Dynamic Rejection Steering），通过在每个解码步动态调节 KV 缓存引导强度并引入合理性约束，显著提升 LLM 指令遵循能力，同时避免过度引导导致的文本质量下降。

**[Lossy Common Information in a Learnable Gray-Wyner Network](lossy_common_information_in_a_learnable_gray-wyner_network.md)**

:   把信息论里的经典 Gray-Wyner 网络做成可学习的三通道编解码器，用一个带 β 超参的目标函数把两个视觉任务之间的"公共信息"和"私有信息"分离开，并在"发送速率"与"接收速率"之间做可调权衡。

**[Mamba-3: Improved Sequence Modeling using State Space Principles](mamba-3_improved_sequence_modeling_using_state_space_principles.md)**

:   从SSM视角提出三项核心改进：指数-梯形离散化、复值状态空间、多输入多输出(MIMO)公式化，在不增加解码延迟的前提下显著提升模型质量和状态追踪能力，推进性能-效率Pareto前沿。

**[Synchronizing Probabilities in Model-Driven Lossless Compression](synchronizing_probabilities_in_model-driven_lossless_compression.md)**

:   针对 LLM 驱动的无损压缩中"编解码两端预测概率必须逐位完全一致、否则级联解码崩溃"的致命问题，本文提出 PMATIC——一种把比特概率量化到分箱、再用低熵 helper 比特让两端锁定同一量化概率的算术编码替代方案，能容忍有界的预测失配，理论上保证正确解码，实测在真实跨机非确定性下全部文件正确还原，同时压缩率仍大幅领先 gzip/cmix 等传统工具。

**[TS-DDAE: A Novel Temporal-Spectral Denoising Diffusion AutoEncoder for Wireless Signal Recognition Model Pre-training](ts-ddae_a_novel_temporal-spectral_denoising_diffusion_autoencoder_for_wireless_s.md)**

:   针对无线信号识别（WSR）预训练，本文把扩散模型的"加噪-去噪"范式引入信号自监督，提出 TS-DDAE：在时域和频域同时给 IQ 信号注入高斯噪声，再用专门设计的双编码器 TS-Net（时域自注意力 + 频域通道注意力）联合还原，学到的表征在 4 个数据集、AMC/WTC 等多任务上平均超过最优基线 1.32%、超过 AMC SOTA 模型 IQFormer 约 8.75%。
