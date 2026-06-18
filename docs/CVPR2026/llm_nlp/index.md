---
title: >-
  CVPR2026 LLM其他论文汇总 · 4篇论文解读
description: >-
  4篇CVPR2026的 LLM 其他方向论文解读，涵盖 LLM、布局/合成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "LLM 其他"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "布局/合成"
item_list:
  - u: "artimuse_fine-grained_image_aesthetics_assessment_with_joint_scoring_and_expert-/"
    t: "ArtiMuse: Fine-Grained Image Aesthetics Assessment with Joint Scoring and Expert-Level Understanding"
  - u: "collm-nas_collaborative_large_language_models_for_efficient_knowledge-guided_neu/"
    t: "CoLLM-NAS: Collaborative Large Language Models for Efficient Knowledge-Guided Neural Architecture Search"
  - u: "llm-guided_probabilistic_fusion_for_label-efficient_document_layout_analysis/"
    t: "LLM-Guided Probabilistic Fusion for Label-Efficient Document Layout Analysis"
  - u: "omnidoclayout_towards_diverse_document_layout_generation_via_coarse-to-fine_llm_/"
    t: "OmniDocLayout: Towards Diverse Document Layout Generation via Coarse-to-Fine LLM Learning"
item_total: 4
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 LLM 其他

**📷 CVPR2026** · **4** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (33)](../../ICML2026/llm_nlp/index.md) · [💬 ACL2026 (59)](../../ACL2026/llm_nlp/index.md) · [🔬 ICLR2026 (33)](../../ICLR2026/llm_nlp/index.md) · [🤖 AAAI2026 (29)](../../AAAI2026/llm_nlp/index.md) · [🧠 NeurIPS2025 (54)](../../NeurIPS2025/llm_nlp/index.md) · [📹 ICCV2025 (6)](../../ICCV2025/llm_nlp/index.md)

🔥 **高频主题：** LLM ×3 · 布局/合成 ×2

**[ArtiMuse: Fine-Grained Image Aesthetics Assessment with Joint Scoring and Expert-Level Understanding](artimuse_fine-grained_image_aesthetics_assessment_with_joint_scoring_and_expert-.md)**

:   ArtiMuse 用一个 InternVL-3-8B 基座的多模态大模型，同时输出 8 维细粒度专家级美学文字分析和一个连续美学分数，靠新提出的 Token As Score 把连续打分塞进 LLM 的离散 token 生成里，并配套发布了首个 10000 张专家逐维标注的 ArtiMuse-10K 数据集，在多个美学评分基准上刷新 SOTA。

**[CoLLM-NAS: Collaborative Large Language Models for Efficient Knowledge-Guided Neural Architecture Search](collm-nas_collaborative_large_language_models_for_efficient_knowledge-guided_neu.md)**

:   用两个分工互补的 LLM（有记忆的 Navigator 负责出策略、无记忆的 Generator 负责出候选架构）替换两阶段 NAS 第二阶段里的进化算法，把架构搜索变成"轨迹→策略→方案"的定向优化，在 ImageNet 和 NAS-Bench-201 上既刷新 SOTA 又把搜索成本压低 4–10×。

**[LLM-Guided Probabilistic Fusion for Label-Efficient Document Layout Analysis](llm-guided_probabilistic_fusion_for_label-efficient_document_layout_analysis.md)**

:   本文把文本预训练 LLM 当作"结构先验生成器"塞进半监督版面检测的伪标签精化环节——用 OCR+LLM 推断文档层级区域，再和教师检测器输出做逆方差概率融合（含可学习的实例自适应门控），仅用 5% 标注就在 PubLayNet 上达到 88.2 AP（轻量骨干）/89.7 AP（LayoutLMv3），并对标题/页眉等稀有版面元素提升最大。

**[OmniDocLayout: Towards Diverse Document Layout Generation via Coarse-to-Fine LLM Learning](omnidoclayout_towards_diverse_document_layout_generation_via_coarse-to-fine_llm_.md)**

:   针对现有文档版面生成数据「只有学术论文、样式单一」的痛点，作者先造了首个百万级、覆盖六类文档的多样化版面数据集 OmniDocLayout-1M，再用一个 0.5B 的小 LLM 通过「先在多域粗标签上学版面通则、再用少量细标签适配具体领域」的由粗到精范式，在 M6Doc 上同时超过专用版面生成模型和 GPT-4o/Gemini/Claude 等通用大模型。
