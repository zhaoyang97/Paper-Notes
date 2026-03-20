---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🖼️ 图像恢复

**🎞️ ECCV2024** · 共 **2** 篇

**[A New Dataset and Framework for Real-World Blurred Images Super-Resolution](a_new_dataset_and_framework_for_real-world_blurred_images_super-resolution.md)**

:   针对现有盲超分方法在处理含模糊（散焦/运动模糊）图像时过度纹理化、破坏模糊区域感知质量的问题，构建了包含近3000张模糊图像的ReBlurSR数据集，并提出PBaSR框架，通过双分支解耦训练（CDM）和基于权重插值的跨分支融合（CFM），在不增加任何推理开销的前提下，同时提升模糊图像和普通图像的超分效果，LPIPS提升0.02~0.10。

**[Accelerating Image Super-Resolution Networks with Pixel-Level Classification](accelerating_image_super-resolution_networks_with_pixel-level_classification.md)**

:   提出PCSR——首个像素级计算资源分配的超分方法，用轻量MLP分类器逐像素判断恢复难度并分配到不同容量的上采样器，在PSNR几乎不掉的情况下将FLOPs压低至原始模型的18%~57%，大幅优于现有patch级方法ClassSR和ARM。
