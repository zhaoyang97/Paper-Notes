<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✂️ 语义分割

**🤖 AAAI2026** · 共 **5** 篇

**[3DTeethSAM: Taming SAM2 for 3D Teeth Segmentation](3dteethsam_taming_sam2_for_3d_teeth_segmentation.md)**

:   将SAM2基础模型迁移到3D牙齿分割任务，通过多视角渲染将3D mesh转为2D图像、设计三个轻量适配器（Prompt生成器、Mask精化器、Mask分类器）和可变形全局注意力插件（DGAP）来解决自动提示、边界精化和语义分类问题，在Teeth3DS上以91.90% T-mIoU刷新SOTA。

**[A²LC: Active and Automated Label Correction for Semantic Segmentation](a2lc_active_and_automated_label_correction_for_semantic_segm.md)**

:   提出 A²LC 框架，在传统主动标签校正（人工逐一纠错）的基础上增加一个自动校正阶段（Label Correction Module），利用标注员的反馈自动修正相似的错误mask，并设计自适应平衡采集函数缓解类别不平衡，在 Cityscapes 上仅用 20% 预算即超越前 SOTA，同等预算下 mIoU 提升 27.23%。

**[Adaptive Morph-Patch Transformer for Aortic Vessel Segmentation](adaptive_morph-patch_transformer_for_aortic_vessel_segmentat.md)**

:   提出 Morph-Patch Transformer (MPT)，通过基于速度场的自适应 patch 划分策略生成形态感知 patch（保持血管拓扑完整性），并引入语义聚类注意力（SCA）动态聚合语义相似 patch 的特征，在 AVT、AortaSeg24 和 TBAD 三个主动脉分割数据集上均达 SOTA。

**[Causal-Tune: Mining Causal Factors from Vision Foundation Models for Domain Generalized Semantic Segmentation](causal-tune_mining_causal_factors_from_vision_foundation_mod.md)**

:   提出Causal-Tune，从因果视角分析VFM特征中的artifacts，利用DCT频域分解+高斯带通滤波分离因果/非因果因素，结合因果感知可学习token在频域精化特征，在Cityscapes→ACDC跨域分割中平均提升+2.4% mIoU（Snow场景+4.8%），仅需单卡RTX3090/14GB训练。

**[InfoCLIP: Bridging Vision-Language Pretraining and Open-Vocabulary Semantic Segmentation via Information-Theoretic Alignment Transfer](infoclip_bridging_vision-language_pretraining_and_open-vocab.md)**

:   提出InfoCLIP，基于信息论视角设计信息瓶颈压缩和互信息蒸馏两个目标，在CLIP微调过程中去除预训练pixel-text对齐中的噪声并保留语义对齐知识，在6个开放词汇语义分割测试集上全面超越SOTA（A-847: 16.6, A-150: 38.5, PC-59: 63.5 mIoU），且仅增加0.53M参数和极少计算开销。
