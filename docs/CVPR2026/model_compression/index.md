<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📦 模型压缩

**📷 CVPR2026** · 共 **6** 篇

**[An FPGA Implementation of Displacement Vector Search for Intra Pattern Copy in JPEG XS](an_fpga_implementation_of_displacement_vector_sear.md)**

:   首次提出JPEG XS帧内模式复制(IPC)中位移矢量(DV)搜索模块的FPGA实现方案，采用四级流水线架构和IPC Group对齐的内存组织策略，在Xilinx Artix-7上实现38.3 Mpixels/s吞吐和277 mW功耗。

**[ARCHE: Autoregressive Residual Compression with Hyperprior and Excitation](arche_autoregressive_residual_compression_with_hyp.md)**

:   在全卷积架构内统一层级超先验、Masked PixelCNN空间自回归、通道条件建模和SE通道激励，不使用Transformer或循环组件，以95M参数和222ms解码时间实现相对Ballé基线48% BD-Rate降低并超越VVC Intra 5.6%。

**[GeoChemAD: Benchmarking Unsupervised Geochemical Anomaly Detection for Mineral Exploration](geochemad_benchmarking_unsupervised_geochemical_an.md)**

:   发布首个开源多区域多元素地球化学异常检测基准 GeoChemAD（8 子集，覆盖沉积物/岩屑/土壤三类采样源和 Au/Cu/Ni/W 四种目标元素），并提出 GeoChemFormer——两阶段 Transformer 框架，先学空间上下文再做元素依赖建模，平均 AUC 达 0.7712 超越所有基线。

**[HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers](hiap_a_multigranular_stochastic_autopruning_framew.md)**

:   提出HiAP——统一宏观（整头/FFN块）和微观（头内维度/FFN神经元）的层级Gumbel-Sigmoid门控框架，在单次端到端训练中自动发现满足算力预算的高效ViT子网络，无需手动重要性排序或多阶段流程。

**[MXNorm: Reusing MXFP block scales for efficient tensor normalisation](mxnorm_reusing_mxfp_block_scales_for_efficient_ten.md)**

:   MXNorm 提出将 RMSNorm 与 MXFP 量化融合：利用 MXFP 量化过程中已经计算好的 block absmax 来近似 RMS 值，从而省掉单独的归一化 reduction 操作，在 Llama 3 最高 8B 参数的预训练中保持训练精度，同时在 GB200 上实现最高 2.4 倍的 kernel 加速。

**[PPCL: Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers](ppcl_pluggable_pruning_dit_distillation.md)**

:   提出 PPCL 框架对大型扩散 Transformer (DiT, 8-20B 参数) 进行结构化剪枝: 通过线性探针+CKA 一阶差分识别连续冗余层区间, 深度方向+宽度方向联合剪枝, 搭配即插即用交替蒸馏, 在 Qwen-Image 20B 上实现 50% 参数缩减, 仅 3% 生成质量下降.
