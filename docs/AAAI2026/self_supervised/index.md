<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔄 自监督/表示学习

**🤖 AAAI2026** · 共 **9** 篇

**[BCE3S: Binary Cross-Entropy Based Tripartite Synergistic Learning for Long-tailed Recognition](bce3s_binary_cross-entropy_based_tripartite_synergistic_learning_for_long-tailed.md)**

:   提出 BCE3S，一种基于二元交叉熵（BCE）的三方协同学习框架，将 BCE 式联合学习、BCE 式对比学习和 BCE 式分类器均匀性学习集成在一起，通过 Sigmoid 解耦不同类别的度量来抑制长尾不平衡效应，在 CIFAR10/100-LT、ImageNet-LT 和 iNaturalist2018 上均取得 SOTA。

**[Explainable Melanoma Diagnosis with Contrastive Learning and LLM-based Report Generation](explainable_melanoma_diagnosis_with_contrastive_learning_and_llm-based_report_ge.md)**

:   提出 CEFM 框架，通过跨模态对比学习将 ViT 视觉特征与基于 ABCD 规则的临床特征（不对称性、边界、颜色）对齐，再由 CLIP + DeepSeek 生成结构化诊断报告，在 ISIC 数据集上达到 92.79% 准确率和 0.961 AUC，专家评分可解释性达 4.6/5。

**[Explanation-Preserving Augmentation for Semi-Supervised Graph Representation Learning](explanation-preserving_augmentation_for_semi-supervised_graph_representation_lea.md)**

:   提出EPA-GRL（Explanation-Preserving Augmentation），利用少量标签训练的GNN explainer识别图的语义子图（explanation subgraph），增强时只扰动非语义部分（marginal subgraph），实现语义保持的图增强，在6个benchmark上显著优于语义无关的随机增强方法。

**[GOAL: Geometrically Optimal Alignment for Continual Generalized Category Discovery](goal_geometrically_optimal_alignment_for_continual_generalized_category_discover.md)**

:   基于 Neural Collapse 理论，使用固定等角紧框架（ETF）分类器替代动态分类器，通过监督对齐和置信度引导的无监督对齐实现持续泛化类别发现，在四个基准上遗忘率降低 16.1%、新类发现提升 3.2%。

**[CAE: Hierarchical Semantic Alignment for Image Clustering](hierarchical_semantic_alignment_for_image_clustering.md)**

:   结合名词级（WordNet）和描述级（Flickr 图片描述）两种互补语义，通过最优传输对齐构建语义空间并自适应融合，实现 training-free 的图像聚类，在 ImageNet-1K 上准确率提升 4.2%。

**[Let The Void Be Void Robust Open-Set Semi-Supervised Learning Via Selective Non-](let_the_void_be_void_robust_open-set_semi-supervised_learning_via_selective_non-.md)**

:   提出 SkipAlign 框架，在对比学习的传统 pull/push 操作之外引入第三种 "skip" 操作，对低置信度样本选择性跳过对齐（只做温和排斥），使 ID 类形成紧凑"星系"、OOD 样本自然散布于"星际虚空"，在未见过的 OOD 检测中平均 AUC 提升 +3.1，最高 +7.1。

**[MovSemCL: Movement-Semantics Contrastive Learning for Trajectory Similarity](movsemcl_movement-semantics_contrastive_learning_for_trajectory_similarity_exten.md)**

:   提出 MovSemCL，将 GPS 轨迹转化为运动语义特征（位移向量+航向角+空间图嵌入），通过 patch 级双层注意力（intra-patch 局部 + inter-patch 全局）实现层次编码（复杂度从 $O(L^2)$ 降为近线性），并设计曲率引导增广保留转弯/路口等关键片段，在轨迹相似性搜索上精度提升达 72.6%、推理延迟降低 43.4%。

**[NeuroBridge: Bio-Inspired Self-Supervised EEG-to-Image Decoding](neurobridge_bio-inspired_self-supervised_eeg-to-image_decoding_via_cognitive_pri.md)**

:   提出 NeuroBridge，通过认知先验增强（CPA，对 EEG 和图像分别用非对称增广模拟感知变异性）+ 共享语义投影器（SSP，双向对齐到统一语义空间），在 200 类零样本 EEG-图像检索任务上达到 63.2% Top-1（+12.3%）和 89.9% Top-5（+10.2%），大幅超越现有 SOTA。

**[Robust Tabular Foundation Models](robust_tabular_foundation_models.md)**

:   提出 RTFM，一种模型无关的对抗训练框架，通过在参数化合成数据生成过程中寻找 TFM 表现不佳的"困难区域"（相比树模型基线的最优性差距最大），仅用 <10万额外合成数据集将 TabPFN V2 的平均归一化 AUC 提升最高 6%。
