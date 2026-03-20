---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✂️ 语义分割

**🧠 NeurIPS2025** · 共 **3** 篇

**[Alligat0R: Pre-Training through Covisibility Segmentation for Relative Camera Pose Regression](alligat0r_pre-training_through_co-visibility_segmentation_for_relative_camera_po.md)**

:   用共视性分割（covisibility segmentation）替代 CroCo 的跨视图补全作为双目视觉预训练任务，对每个像素预测"共视/遮挡/视野外"三类标签，在低重叠场景下显著超越 CroCo，RUBIK 基准总体成功率 60.3% 排第一。

**[ARGenSeg: Image Segmentation with Autoregressive Image Generation Model](argenseg_image_segmentation_with_autoregressive_image_generation_model.md)**

:   提出ARGenSeg——首个利用自回归图像生成范式实现图像分割的统一MLLM框架，让模型直接输出visual tokens并通过VQ-VAE解码为分割mask，无需额外分割头，搭配next-scale prediction并行生成策略实现4×加速，在RefCOCO/+/g上以更少训练数据超越SOTA。

**[COS3D: Collaborative Open-Vocabulary 3D Segmentation](cos3d_collaborative_open-vocabulary_3d_segmentation.md)**

:   提出COS3D协作式开放词汇3D分割框架，在3D Gaussian Splatting中同时维护instance field（学习清晰边界）和language field（学习语义），通过两阶段训练实现Ins2Lang映射，推理时Language→Instance prompt精化实现互补协作，在LeRF数据集上mIoU达50.76%，大幅超越Dr.Splat（43.58%）。
