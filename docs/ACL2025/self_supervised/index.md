---
title: >-
  ACL2025 自监督/表示学习论文汇总 · 7篇论文解读
description: >-
  7篇ACL2025的自监督/表示学习方向论文解读，涵盖自监督学习、持续学习、LLM、问答等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2025"
  - "自监督/表示学习"
  - "论文解读"
  - "论文笔记"
  - "自监督学习"
  - "持续学习"
  - "LLM"
  - "问答"
item_list:
  - u: "analytickws_towards_exemplar-free_analytic_class_incremental_learning_for_small-/"
    t: "AnalyticKWS: Towards Exemplar-Free Analytic Class Incremental Learning for Small-footprint Keyword Spotting"
  - u: "improving_low-resource_morphological_inflection_via_self-supervised_objectives/"
    t: "Improving Low-Resource Morphological Inflection via Self-Supervised Objectives"
  - u: "llm_back_gen_treebank/"
    t: "Contrastive Learning on LLM Back Generation Treebank for Cross-domain Constituency Parsing"
  - u: "magnet_augmenting_generative_decoders_with_representation_learning_and_infilling/"
    t: "Magnet: Augmenting Generative Decoders with Representation Learning and Infilling Capabilities"
  - u: "qaencoder_aligned_representation/"
    t: "QAEncoder: Towards Aligned Representation Learning in Question Answering Systems"
  - u: "shubert_self-supervised_sign_language_representation_learning_via_multi-stream_c/"
    t: "SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction"
  - u: "whispa_semantically_and_psychologically_aligned_whisper_with_self-supervised_con/"
    t: "WhiSPA: Semantically and Psychologically Aligned Whisper with Self-Supervised Contrastive and Student-Teacher Learning"
item_total: 7
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔄 自监督/表示学习

**💬 ACL2025** · **7** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (91)](../../CVPR2026/self_supervised/index.md) · [🔬 ICLR2026 (81)](../../ICLR2026/self_supervised/index.md) · [💬 ACL2026 (1)](../../ACL2026/self_supervised/index.md) · [🧪 ICML2026 (28)](../../ICML2026/self_supervised/index.md) · [🤖 AAAI2026 (16)](../../AAAI2026/self_supervised/index.md) · [🧠 NeurIPS2025 (34)](../../NeurIPS2025/self_supervised/index.md)

🔥 **高频主题：** 自监督学习 ×4

**[AnalyticKWS: Towards Exemplar-Free Analytic Class Incremental Learning for Small-footprint Keyword Spotting](analytickws_towards_exemplar-free_analytic_class_incremental_learning_for_small-.md)**

:   提出 AnalyticKWS，一种无需存储历史样本的关键词检测增量学习方法，通过冻结特征提取器 + 递归最小二乘解析解更新分类器，在 GSC 和 SC-100 数据集上超过了所有基于样本回放的方法，且训练时间和内存开销极低。

**[Improving Low-Resource Morphological Inflection via Self-Supervised Objectives](improving_low-resource_morphological_inflection_via_self-supervised_objectives.md)**

:   系统探索 13 种自监督辅助目标（自编码、CMLM、T5-style 等）在极低资源形态变化任务中的效果，发现无标注数据极少时自编码最优，数据增多后字符级 MLM 更好，按形态素边界采样掩码是最有前景的方向。

**[Contrastive Learning on LLM Back Generation Treebank for Cross-domain Constituency Parsing](llm_back_gen_treebank.md)**

:   提出 LLM 反向生成 (LLM Back Generation) 方法，将不完整的跨领域句法树作为输入让 LLM 补全缺失词生成 treebank，并设计 span 级别对比学习预训练策略，实现跨领域成分句法分析的 SOTA 性能。

**[Magnet: Augmenting Generative Decoders with Representation Learning and Infilling Capabilities](magnet_augmenting_generative_decoders_with_representation_learning_and_infilling.md)**

:   提出 Magnet 方法，通过混合注意力机制（双向+因果）和三个自监督目标（掩码预测+对比学习+缺失片段生成），将纯解码器 LLM 同时增强为文本编码器和填充模型，在 token 级和句子级表示学习任务上超越 LLM2Vec 等专用方法，同时避免了双向化带来的严重文本重复问题。

**[QAEncoder: Towards Aligned Representation Learning in Question Answering Systems](qaencoder_aligned_representation.md)**

:   提出 QAEncoder，一种免训练方法通过蒙特卡洛采样估计文档对应查询的期望嵌入作为文档表示的代理，配合文档指纹保持区分性，在 BEIR 上将 bge-large 从 58.5 提升到 61.8 NDCG@10，零额外存储和延迟开销。

**[SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction](shubert_self-supervised_sign_language_representation_learning_via_multi-stream_c.md)**

:   提出 SHuBERT（Sign Hidden-Unit BERT），将语音自监督学习模型 HuBERT 的 masked cluster prediction 范式迁移到手语视频——对手部、面部、身体姿态四个流分别聚类并同时预测 masked 帧的聚类标签，在约 984 小时 ASL 视频上预训练后，在翻译/孤立识别/指拼检测多任务上达到公开数据 SOTA。

**[WhiSPA: Semantically and Psychologically Aligned Whisper with Self-Supervised Contrastive and Student-Teacher Learning](whispa_semantically_and_psychologically_aligned_whisper_with_self-supervised_con.md)**

:   提出 WhiSPA，通过对比学习将 Whisper 音频编码器的潜在空间与 SBERT 语义表征和心理学维度（情感、人格）对齐，消除语音处理中对额外文本 LM 的依赖，在心理学评估任务上误差降低 73-84%。
