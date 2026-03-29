<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM 效率

**📷 CVPR2026** · 共 **5** 篇

**[ACE-Merging: Data-Free Model Merging with Adaptive Covariance Estimation](ace-merging_data-free_model_merging_with_adaptive_covariance_estimation.md)**

:   本文从理论上证明了微调参数差蕴含输入协方差信息，据此提出 ACE-Merging，通过自适应协方差估计、集体结构先验和谱精炼三步实现无数据闭式模型合并，在 GPT-2 上比之前方法平均提升 4%，在 RoBERTa-Base 上提升 5%。

**[Attribution-Guided Model Rectification of Unreliable Neural Network Behaviors](attribution-guided_model_rectification_of_unreliable_neural_network_behaviors.md)**

:   提出基于归因引导的动态模型纠正框架，利用Integrated Gradients量化各层"可编辑性"来自动定位导致不可靠行为的嫌疑层，结合rank-one编辑在仅需1个清洁样本的情况下修复后门攻击、虚假相关和特征泄漏等模型不可靠行为。

**[Boosting Quantitive and Spatial Awareness for Zero-Shot Object Counting](boosting_quantitive_and_spatial_awareness_for_zero-shot_object_counting.md)**

:   提出QICA框架解决零样本目标计数中的数量感知缺失和空间不敏感问题，通过数量条件化的协同提示策略（SPS）联合适配视觉-语言编码器，结合在相似度图上直接操作的代价聚合解码器（CAD）保持零样本迁移能力，在FSC-147上达到零样本SOTA（MAE 12.41）并展现强跨域泛化。

**[Edit-As-Act: Goal-Regressive Planning for Open-Vocabulary 3D Indoor Scene Editing](edit-as-act_goal-regressive_planning_for_open-vocabulary_3d_indoor_scene_editing.md)**

:   将开放词汇的3D室内场景编辑重新定义为目标回归规划问题，设计PDDL风格的EditLang符号语言，通过LLM驱动的Planner-Validator循环从目标状态逆向推导最小编辑序列，在63个编辑任务上同时实现指令忠实度（69.1%）、语义一致性（86.6%）和物理合理性（91.7%）三个指标的最佳平衡。

**[Model Merging in the Essential Subspace](model_merging_in_the_essential_subspace.md)**

:   提出 ESM 框架，通过对参数更新引起的激活偏移做 PCA 构建"本质子空间"（而非直接对参数做 SVD），并用三级极化缩放增强关键参数、抑制噪声，在 ViT-B/32 的 20 任务合并中比 Iso-CTS 提升 3.2%（绝对准确率）。
