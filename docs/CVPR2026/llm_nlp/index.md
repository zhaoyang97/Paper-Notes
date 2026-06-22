---
title: >-
  CVPR2026 LLM其他论文汇总 · 3篇论文解读
description: >-
  3篇CVPR2026的 LLM 其他方向论文解读，涵盖布局/合成、LLM、扩散模型等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "LLM 其他"
  - "论文解读"
  - "论文笔记"
  - "布局/合成"
  - "LLM"
  - "扩散模型"
item_list:
  - u: "llm-guided_probabilistic_fusion_for_label-efficient_document_layout_analysis/"
    t: "LLM-Guided Probabilistic Fusion for Label-Efficient Document Layout Analysis"
  - u: "omnidoclayout_towards_diverse_document_layout_generation_via_coarse-to-fine_llm_/"
    t: "OmniDocLayout: Towards Diverse Document Layout Generation via Coarse-to-Fine LLM Learning"
  - u: "single-step_diffusion-based_video_coding_with_semantic-temporal_guidance/"
    t: "Single-step Diffusion-based Video Coding with Semantic-Temporal Guidance"
item_total: 3
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 LLM 其他

**📷 CVPR2026** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (55)](../../ICLR2026/llm_nlp/index.md) · [💬 ACL2026 (61)](../../ACL2026/llm_nlp/index.md) · [🧪 ICML2026 (39)](../../ICML2026/llm_nlp/index.md) · [🤖 AAAI2026 (29)](../../AAAI2026/llm_nlp/index.md) · [🧠 NeurIPS2025 (54)](../../NeurIPS2025/llm_nlp/index.md) · [📹 ICCV2025 (6)](../../ICCV2025/llm_nlp/index.md)

🔥 **高频主题：** 布局/合成 ×2 · LLM ×2

**[LLM-Guided Probabilistic Fusion for Label-Efficient Document Layout Analysis](llm-guided_probabilistic_fusion_for_label-efficient_document_layout_analysis.md)**

:   本文把文本预训练 LLM 当作"结构先验生成器"塞进半监督版面检测的伪标签精化环节——用 OCR+LLM 推断文档层级区域，再和教师检测器输出做逆方差概率融合（含可学习的实例自适应门控），仅用 5% 标注就在 PubLayNet 上达到 88.2 AP（轻量骨干）/89.7 AP（LayoutLMv3），并对标题/页眉等稀有版面元素提升最大。

**[OmniDocLayout: Towards Diverse Document Layout Generation via Coarse-to-Fine LLM Learning](omnidoclayout_towards_diverse_document_layout_generation_via_coarse-to-fine_llm_.md)**

:   针对现有文档版面生成数据「只有学术论文、样式单一」的痛点，作者先造了首个百万级、覆盖六类文档的多样化版面数据集 OmniDocLayout-1M，再用一个 0.5B 的小 LLM 通过「先在多域粗标签上学版面通则、再用少量细标签适配具体领域」的由粗到精范式，在 M6Doc 上同时超过专用版面生成模型和 GPT-4o/Gemini/Claude 等通用大模型。

**[Single-step Diffusion-based Video Coding with Semantic-Temporal Guidance](single-step_diffusion-based_video_coding_with_semantic-temporal_guidance.md)**

:   S2VC 把一个**单步扩散生成器**塞进条件视频编码框架，用从解码特征缓冲里抽取的「上下文语义引导（CSG）」替代文本 prompt、再用插进 U-Net 的「时序一致性引导（TCG）」做跨帧对齐，在 0.02 bpp 以下的极低码率下拿到 SOTA 感知质量，相比上一代感知编解码器平均省 51.62% 码率（DISTS BD-Rate）。
