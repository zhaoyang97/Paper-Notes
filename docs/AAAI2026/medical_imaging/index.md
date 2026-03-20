---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🏥 医学图像

**🤖 AAAI2026** · 共 **5** 篇

**[A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation](a_disease-aware_dual-stage_framework_for_chest_x-ray_report_.md)**

:   提出一种两阶段疾病感知框架，通过学习14个与病理类别对应的疾病感知语义token（DASTs）实现显式的疾病表征，再利用疾病-视觉注意力融合（DVAF）和双模态相似性检索（DMSR）机制辅助LLM生成临床准确的胸部X光报告，在CheXpert Plus、IU X-Ray和MIMIC-CXR三个数据集上取得SOTA。

**[Advancing Safe Mechanical Ventilation Using Offline RL With Hybrid Actions and Clinically Aligned Rewards](advancing_safe_mechanical_ventilation_using_offline_rl_with_.md)**

:   针对ICU机械通气（MV）设置优化问题，提出混合动作空间的离线RL方法（HybridIQL/HybridEDAC），避免传统离散化导致的分布偏移，同时引入基于无通气天数（VFD）和生理参数安全范围的临床对齐奖励函数，通过多目标优化选择最优奖励，将可优化的通气参数从2-3个扩展到6个，HybridIQL在性能和策略覆盖率间取得最佳平衡。

**[Ambiguity-aware Truncated Flow Matching for Ambiguous Medical Image Segmentation](ambiguity-aware_truncated_flow_matching_for_ambiguous_medica.md)**

:   提出 ATFM 框架，通过数据层级推理范式将预测精度和多样性解耦到分布级和样本级分别优化，结合高斯截断表示（GTR）和分割流匹配（SFM）两个模块，在模糊医学图像分割任务中同时提升预测的精度、保真度和多样性。

**[Apo2Mol: 3D Molecule Generation via Dynamic Pocket-Aware Diffusion Models](apo2mol_3d_molecule_generation_via_dynamic_pocket-aware_diff.md)**

:   提出Apo2Mol，一个基于扩散的全原子框架，从蛋白质apo（未结合）构象出发，同时生成3D配体分子和对应的holo（结合态）口袋构象，使用24K实验解析的apo-holo结构对训练，在结合亲和力（Vina min -7.86）和药物类似性上达到SOTA。

**[Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling](learning_cell-aware_hierarchical_multi-modal_representations.md)**

:   提出CHMR框架，将分子结构(1D/2D/3D)与细胞形态/基因表达等生物模态联合建模，通过结构感知的模态增强解决>90%的外部生物模态缺失问题，用树状向量量化(Tree-VQ)捕获分子-细胞-基因的层次化依赖关系，在9个benchmark的728个任务上超越SOTA，分类平均AUC提升3.6%，回归MAE降低17.2%。
