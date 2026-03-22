<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📡 信号/通信

**🔬 ICLR2026** · 共 **7** 篇

**[FASA: Frequency-aware Sparse Attention](fasa_frequency-aware_sparse_attention.md)**

:   发现RoPE注意力在频率块(FC)级别存在功能稀疏性——少量主导FC与全注意力高度一致，据此设计无需训练的KV cache压缩方案FASA，实现8×内存压缩和2.6×加速且几乎无损。

**[Group Representational Position Encoding (GRAPE)](group_representational_position_encoding.md)**

:   提出 GRAPE 框架，基于群作用（group actions）统一了 Transformer 中乘法型（RoPE）和加法型（ALiBi/FoX）两大位置编码家族，证明 RoPE 和 ALiBi 是其精确特例，并提出路径积分加法变体 GRAPE-AP 在下游任务上超越现有方法。

**[Learning Molecular Chirality via Chiral Determinant Kernels](learning_molecular_chirality_via_chiral_determinant_kernels.md)**

:   提出手性行列式核(ChiDeK)来编码SE(3)不变的手性矩阵，统一处理中心手性和轴向手性，结合交叉注意力机制在手性/非手性原子间传递信息，在轴向手性任务上准确率提升>7%。

**[Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies](multi-agent_design_optimizing_agents_with_better_prompts_and_topologies.md)**

:   深入分析多智能体系统中 prompt 和拓扑设计的影响，发现 prompt 优化是最关键的设计因素（仅优化 prompt 的单 Agent 即可超越复杂多 Agent 拓扑），提出 Mass 三阶段框架（block-level prompt → topology → workflow-level prompt）在 8 个 benchmark 上取得 SOTA。

**[Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional](multi-modal_data_spectrum_multi-modal_datasets_are_multi-dimensional.md)**

:   大规模实证研究揭示23个VQA基准中存在严重的单模态依赖问题——许多为消除文本偏差而设计的基准反而引入了图像偏差，模型利用单模态捷径而非真正的跨模态推理。

**[SALVE: Sparse Autoencoder-Latent Vector Editing for Mechanistic Control of Neural Networks](salve_sparse_autoencoder-latent_vector_editing_for_mechanistic_control_of_neural.md)**

:   提出SALVE框架，用L1正则化稀疏自编码器发现模型的稀疏特征基，再将发现转化为永久性权重空间干预实现模型编辑，弥合了机制可解释性与模型编辑之间的鸿沟。

**[Spectrum Tuning: Post-Training for Distributional Coverage and In-Context Steerability](spectrum_tuning_post-training_for_distributional_coverage_and_in-context_steerab.md)**

:   揭示RLHF/DPO等后训练会损害模型的上下文可操控性(in-context steerability)、输出覆盖率和分布对齐，提出Spectrum Suite评测框架和Spectrum Tuning方法，首次在后训练阶段改善分布对齐能力。
