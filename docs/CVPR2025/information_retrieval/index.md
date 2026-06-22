---
title: >-
  CVPR2025 信息检索/RAG论文汇总 · 12篇论文解读
description: >-
  12篇CVPR2025的信息检索/RAG 方向论文解读，涵盖 RAG、少样本学习、对齐/RLHF、域适应等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2025"
  - "信息检索/RAG"
  - "论文解读"
  - "论文笔记"
  - "RAG"
  - "少样本学习"
  - "对齐/RLHF"
  - "域适应"
item_list:
  - u: "advancing_myopia_to_holism_fully_contrastive_language-image_pre-training/"
    t: "Advancing Myopia To Holism: Fully Contrastive Language-Image Pre-training"
  - u: "chathuman_chatting_about_3d_humans_with_tools/"
    t: "ChatHuman: Chatting about 3D Humans with Tools"
  - u: "cobra_combinatorial_retrieval_augmentation_for_few-shot_adaptation/"
    t: "COBRA: COmBinatorial Retrieval Augmentation for Few-Shot Adaptation"
  - u: "ezsr_event-based_zero-shot_recognition/"
    t: "EZSR: Event-based Zero-Shot Recognition"
  - u: "few-shot_recognition_via_stage-wise_retrieval-augmented_finetuning/"
    t: "Few-Shot Recognition via Stage-Wise Retrieval-Augmented Finetuning"
  - u: "goal_global-local_object_alignment_learning/"
    t: "GOAL: Global-Local Object Alignment Learning"
  - u: "lotusfilter_fast_diverse_nearest_neighbor_search_via_a_learned_cutoff_table/"
    t: "LotusFilter: Fast Diverse Nearest Neighbor Search via a Learned Cutoff Table"
  - u: "preserving_clusters_in_prompt_learning_for_unsupervised_domain_adaptation/"
    t: "Preserving Clusters in Prompt Learning for Unsupervised Domain Adaptation"
  - u: "range_retrieval_augmented_neural_fields_for_multi-resolution_geo-embeddings/"
    t: "RANGE: Retrieval Augmented Neural Fields for Multi-Resolution Geo-Embeddings"
  - u: "retrieving_semantics_from_the_deep_an_rag_solution_for_gesture_synthesis/"
    t: "Retrieving Semantics from the Deep: an RAG Solution for Gesture Synthesis"
  - u: "towards_smart_point-and-shoot_photography/"
    t: "Towards Smart Point-and-Shoot Photography"
  - u: "vdocrag_retrieval-augmented_generation_over_visually-rich_documents/"
    t: "VDocRAG: Retrieval-Augmented Generation over Visually-Rich Documents"
item_total: 12
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔍 信息检索/RAG

**📷 CVPR2025** · **12** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (81)](../../ICLR2026/information_retrieval/index.md) · [💬 ACL2026 (73)](../../ACL2026/information_retrieval/index.md) · [🧪 ICML2026 (26)](../../ICML2026/information_retrieval/index.md) · [🤖 AAAI2026 (21)](../../AAAI2026/information_retrieval/index.md) · [🧠 NeurIPS2025 (25)](../../NeurIPS2025/information_retrieval/index.md) · [📹 ICCV2025 (5)](../../ICCV2025/information_retrieval/index.md)

🔥 **高频主题：** RAG ×4 · 少样本学习 ×3

**[Advancing Myopia To Holism: Fully Contrastive Language-Image Pre-training](advancing_myopia_to_holism_fully_contrastive_language-image_pre-training.md)**

:   将CLIP从传统的一对一(image, text)对比学习升级为多对多(multi-image-embeddings, multi-texts)对比学习范式，通过VLM生成多视角多层次的描述文本、多分支视觉编码器输出多种视觉embedding，实现更全面的视觉语言对齐，在检索/分类/密集任务上大幅超越baseline。

**[ChatHuman: Chatting about 3D Humans with Tools](chathuman_chatting_about_3d_humans_with_tools.md)**

:   提出 ChatHuman，一个基于 LLM 的语言驱动系统，通过自动选择和集成专门的 3D 人体分析工具（3D 姿态估计、形状恢复、接触检测、人物交互分析、情感识别等），利用学术论文作为工具使用说明和 RAG（检索增强生成）创建 in-context 示例以管理新工具，在工具选择准确率和整体 3D 人体任务性能上超越现有 LLM 模型。

**[COBRA: COmBinatorial Retrieval Augmentation for Few-Shot Adaptation](cobra_combinatorial_retrieval_augmentation_for_few-shot_adaptation.md)**

:   提出 COBRA——基于组合互信息（CMI）的检索增强少样本适配方法，通过同时考虑检索样本与目标任务的相似性和样本间的多样性，从 LAION-2B 中检索高质量辅助数据，在多个图像分类基准上一致性超越传统最近邻检索方法，且计算开销可忽略。

**[EZSR: Event-based Zero-Shot Recognition](ezsr_event-based_zero-shot_recognition.md)**

:   提出 EZSR 框架用于事件相机数据的零样本物体识别，通过标量级调制（scalar-wise modulation）策略解决事件嵌入与 CLIP 文本嵌入之间的语义错位问题，并通过从静态 RGB 图像大规模合成事件数据来突破训练数据稀缺限制，在 N-ImageNet 上以 ViT-B/16 达到 47.84% 零样本准确率。

**[Few-Shot Recognition via Stage-Wise Retrieval-Augmented Finetuning](few-shot_recognition_via_stage-wise_retrieval-augmented_finetuning.md)**

:   本文首次将检索增强学习（RAL）扩展到少样本识别（FSR），揭示了检索数据的分布不平衡和域差距两大挑战，提出两阶段方法 SWAT（先在混合数据上微调视觉编码器、再在少量标注数据上重训分类器），在 9 个基准上以 >6% 的优势超越所有先前方法。

**[GOAL: Global-Local Object Alignment Learning](goal_global-local_object_alignment_learning.md)**

:   提出GOAL方法，通过局部图-句匹配（LISM）和Token相似性学习（TSL）两个模块增强CLIP对长文本描述的理解能力，在全局对齐的基础上引入局部语义对齐，大幅提升图文检索性能。

**[LotusFilter: Fast Diverse Nearest Neighbor Search via a Learned Cutoff Table](lotusfilter_fast_diverse_nearest_neighbor_search_via_a_learned_cutoff_table.md)**

:   提出LotusFilter，通过离线预计算每个向量的邻近关系构建截断表(cutoff table)，在线阶段用贪心集合删除实现多样化过滤，将传统 $O(DS^2)$ 的多样化搜索降至 $O(T+S+KL)$，过滤仅需0.02ms/query，内存仅为传统方法的1/40。

**[Preserving Clusters in Prompt Learning for Unsupervised Domain Adaptation](preserving_clusters_in_prompt_learning_for_unsupervised_domain_adaptation.md)**

:   提出 CRPL 框架，通过源域增强的伪标签和基于最优传输的聚类保持策略，改进 CLIP 在无监督域适应（UDA）中的 prompt 学习，使得目标域 prompt 的文本嵌入能更好地覆盖视觉嵌入的聚类结构。

**[RANGE: Retrieval Augmented Neural Fields for Multi-Resolution Geo-Embeddings](range_retrieval_augmented_neural_fields_for_multi-resolution_geo-embeddings.md)**

:   提出RANGE，通过检索增强策略将高分辨率视觉信息近似注入地理位置嵌入，解决了对比学习（如SatCLIP）丢弃模态特有信息的问题，在分类任务上提升高达13.1%，回归任务上提升0.145 $R^2$。

**[Retrieving Semantics from the Deep: an RAG Solution for Gesture Synthesis](retrieving_semantics_from_the_deep_an_rag_solution_for_gesture_synthesis.md)**

:   RAG-Gesture 提出了一种基于检索增强生成（RAG）的手势合成框架，利用显式语言学知识从手势数据库中检索语义相关的示例动作，并通过 DDIM 反演和检索引导在推理时注入扩散模型生成过程，无需训练即可产生语义丰富且自然的共语手势。

**[Towards Smart Point-and-Shoot Photography](towards_smart_point-and-shoot_photography.md)**

:   提出智能"傻瓜相机"摄影系统：先用 CLIP 文本嵌入的构图质量评估器（CCQA）判断当前构图质量，再用专家混合（MoE）相机姿态调整模型（CPAM）预测偏航/俯仰调整角度，在 PCARD 数据集（320K 图像，从 4K 全景图生成）上实现 79.3% AUC 的调整建议和 0.613 IoU 的调整精度。

**[VDocRAG: Retrieval-Augmented Generation over Visually-Rich Documents](vdocrag_retrieval-augmented_generation_over_visually-rich_documents.md)**

:   构建首个直接以文档图像（而非解析文本）为输入的 RAG 框架，用 LVLM 作为双编码器检索器 + 两种自监督预训练任务（对比+生成）实现文档图像检索，在 ChartQA 上比文本 RAG 高 24 个点。
