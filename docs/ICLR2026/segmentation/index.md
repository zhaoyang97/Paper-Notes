---
title: >-
  ICLR2026 语义分割论文汇总 · 11篇论文解读
description: >-
  11篇ICLR2026的语义分割方向论文解读，涵盖扩散模型、语义分割、对齐/RLHF、压缩/编码、推理、翻译等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "语义分割"
  - "论文解读"
  - "论文笔记"
  - "扩散模型"
  - "对齐/RLHF"
  - "压缩/编码"
  - "推理"
  - "翻译"
item_list:
  - u: "amlris_alignment-aware_masked_learning_for_referring_image_segmentation/"
    t: "AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation"
  - u: "byteflow_language_modeling_through_adaptive_byte_compression_without_a_tokenizer/"
    t: "ByteFlow: Language Modeling through Adaptive Byte Compression without a Tokenizer"
  - u: "efficient-sam2_accelerating_sam2_with_object-aware_visual_encoding_and_memory_re/"
    t: "Efficient-SAM2: Accelerating SAM2 with Object-Aware Visual Encoding and Memory Retrieval"
  - u: "locality-attending_vision_transformer/"
    t: "Locality-Attending Vision Transformer"
  - u: "regionreasoner_region-grounded_multi-round_visual_reasoning/"
    t: "RegionReasoner: Region-Grounded Multi-Round Visual Reasoning"
  - u: "revisiting_cls_and_patch_token_interaction_in_vision_transformers/"
    t: "Revisiting [CLS] and Patch Token Interaction in Vision Transformers"
  - u: "thicker_and_quicker_a_jumbo_token_for_fast_plain_vision_transformers/"
    t: "Thicker and Quicker: A Jumbo Token for Fast Plain Vision Transformers"
  - u: "trace_your_diffusion_model_is_secretly_an_instance_edge_detector/"
    t: "TRACE: Your Diffusion Model is Secretly an Instance Edge Detector"
  - u: "universal_multi-domain_translation_via_diffusion_routers/"
    t: "Universal Multi-Domain Translation via Diffusion Routers"
  - u: "vincie_unlocking_in-context_image_editing_from_video/"
    t: "VINCIE: Unlocking In-context Image Editing from Video"
  - u: "virtue_visual-interactive_text-image_universal_embedder/"
    t: "VIRTUE: Visual-Interactive Text-Image Universal Embedder"
item_total: 11
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✂️ 语义分割

**🔬 ICLR2026** · **11** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (162)](../../CVPR2026/segmentation/index.md) · [🧪 ICML2026 (13)](../../ICML2026/segmentation/index.md) · [🤖 AAAI2026 (29)](../../AAAI2026/segmentation/index.md) · [🧠 NeurIPS2025 (45)](../../NeurIPS2025/segmentation/index.md) · [📹 ICCV2025 (73)](../../ICCV2025/segmentation/index.md) · [🧪 ICML2025 (18)](../../ICML2025/segmentation/index.md)

🔥 **高频主题：** 扩散模型 ×2

**[AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation](amlris_alignment-aware_masked_learning_for_referring_image_segmentation.md)**

:   提出对齐感知遮蔽学习(AML)策略，通过量化视觉-语言 patch 级对齐度并过滤低对齐像素，让 RIS 模型在训练时聚焦可靠区域，无需架构改动即在 RefCOCO 全部 8 个 split 上达到 SOTA。

**[ByteFlow: Language Modeling through Adaptive Byte Compression without a Tokenizer](byteflow_language_modeling_through_adaptive_byte_compression_without_a_tokenizer.md)**

:   提出 ByteFlow Net，一种无需分词器的分层字节级语言模型，利用信息论中的编码率(coding rate)自适应地将原始字节流压缩为语义单元，在预训练损失和下游任务上超越 BPE 基线和已有字节级架构。

**[Efficient-SAM2: Accelerating SAM2 with Object-Aware Visual Encoding and Memory Retrieval](efficient-sam2_accelerating_sam2_with_object-aware_visual_encoding_and_memory_re.md)**

:   发现 SAM2 存在类似生物视觉的稀疏感知模式（解码器聚焦前景但编码器广泛计算、记忆帧中仅少量 token 有效且显著性时间一致），据此提出 Efficient-SAM2，通过对象感知的稀疏窗口路由（SWR）和稀疏记忆检索（SMR）消除冗余计算，在 SAM2.1-L 上实现 1.68× 端到端加速且仅损失 1% 精度。

**[Locality-Attending Vision Transformer](locality-attending_vision_transformer.md)**

:   提出 LocAt 模块化插件（GAug + PRR），通过可学习高斯核偏置注意力向局部邻域聚焦并精炼 patch 表示，在不修改训练目标的前提下使 ViT 在 ADE20K 分割上提升超 6%，同时分类精度不降反升。

**[RegionReasoner: Region-Grounded Multi-Round Visual Reasoning](regionreasoner_region-grounded_multi-round_visual_reasoning.md)**

:   提出 RegionReasoner，一个基于强化学习的多轮视觉推理框架，通过引用标注奖励和全局-局部一致性奖励，使推理轨迹必须显式引用参考区域坐标并保持语义连贯，在新构建的 RegionDial-Bench 上显著提升多轮定位和分割精度。

**[Revisiting [CLS] and Patch Token Interaction in Vision Transformers](revisiting_cls_and_patch_token_interaction_in_vision_transformers.md)**

:   深入分析Vision Transformer中[CLS]全局token和patch局部token之间的交互摩擦，发现归一化层隐式地区分了两类token，提出在归一化层和早期QKV投影中引入专门化处理路径，仅增加8%参数即实现分割性能提升超2 mIoU，同时保持分类精度。

**[Thicker and Quicker: A Jumbo Token for Fast Plain Vision Transformers](thicker_and_quicker_a_jumbo_token_for_fast_plain_vision_transformers.md)**

:   本文提出 Jumbo 方法：将 ViT 的 CLS token 扩展为 $J$ 倍宽度，在注意力前拆分为 $J$ 个与 patch 等宽的 token 参与自注意力，注意力后重新拼接并经过专用的宽 FFN 处理——以极低的计算开销显著增加全局建模容量，使 plain ViT 在高速推理场景下超越专用高效架构（EfficientViT、SHViT、MobileNetV4），同时保留 ViT 的所有架构优势。

**[TRACE: Your Diffusion Model is Secretly an Instance Edge Detector](trace_your_diffusion_model_is_secretly_an_instance_edge_detector.md)**

:   发现文本到图像扩散模型的自注意力在去噪过程中存在一个"实例涌现点"（IEP），在该时刻自注意力在物体边界呈现剧烈散度变化。TRACE通过IEP定位+ABDiv边缘提取+单步蒸馏，以81×推理加速生成高质量实例边缘，无需任何实例标注即可将无监督实例分割提升+5.1 AP，tag监督全景分割超越点监督方法+1.7 PQ。

**[Universal Multi-Domain Translation via Diffusion Routers](universal_multi-domain_translation_via_diffusion_routers.md)**

:   提出 Diffusion Router (DR)，用单个噪声预测网络通过 source/target 域标签条件化实现所有跨域映射，支持通过中心域的间接翻译和基于变分上界目标 + Tweedie 精化的直接非中心域翻译，在三个大规模 UMDT 基准上达到 SOTA。

**[VINCIE: Unlocking In-context Image Editing from Video](vincie_unlocking_in-context_image_editing_from_video.md)**

:   提出VINCIE框架，首次证明in-context图像编辑模型可以完全从原生视频数据中学习，通过将视频标注为交错多模态序列并设计三个代理任务（NIP/CSP/NSP），在多轮编辑基准上达到SOTA，5轮编辑成功率从基线<2%提升至25%。

**[VIRTUE: Visual-Interactive Text-Image Universal Embedder](virtue_visual-interactive_text-image_universal_embedder.md)**

:   提出 VIRTUE，将分割模型 SAM2 与 VLM 结合构建视觉交互式通用嵌入器，支持用户通过点/框/掩码指定兴趣区域产生实体级+全局级联合嵌入，并构建百万级 SCaR 基准评估视觉交互检索能力，在 36 个 MMEB 任务（+3.1%-8.5%）和 5 个 SCaR 任务（+15.2%-20.3%）上均达到 SOTA。
