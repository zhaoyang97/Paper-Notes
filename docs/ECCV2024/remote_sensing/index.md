---
title: >-
  ECCV2024 遥感论文汇总 · 6篇论文解读
description: >-
  6篇ECCV2024的遥感方向论文解读，涵盖遥感、对抗鲁棒、重识别等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2024"
  - "遥感"
  - "论文解读"
  - "论文笔记"
  - "对抗鲁棒"
  - "重识别"
item_list:
  - u: "adapting_fine-grained_cross-view_localization_to_areas_without_fine_ground_truth/"
    t: "Adapting Fine-Grained Cross-View Localization to Areas without Fine Ground Truth"
  - u: "congeo_robust_cross-view_geo-localization_across_ground_view_variations/"
    t: "ConGeo: Robust Cross-View Geo-Localization Across Ground View Variations"
  - u: "cross-platform_video_person_reid_a_new_benchmark_dataset_and_adaptation_approach/"
    t: "Cross-Platform Video Person ReID: A New Benchmark Dataset and Adaptation Approach"
  - u: "learning_representations_of_satellite_images_from_metadata_supervision/"
    t: "Learning Representations of Satellite Images From Metadata Supervision"
  - u: "masked_angle-aware_autoencoder_for_remote_sensing_images/"
    t: "Masked Angle-Aware Autoencoder for Remote Sensing Images"
  - u: "weakly-supervised_camera_localization_by_ground-to-satellite_image_registration/"
    t: "Weakly-Supervised Camera Localization by Ground-to-Satellite Image Registration"
item_total: 6
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛰️ 遥感

**🎞️ ECCV2024** · **6** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (63)](../../CVPR2026/remote_sensing/index.md) · [🔬 ICLR2026 (11)](../../ICLR2026/remote_sensing/index.md) · [🧪 ICML2026 (3)](../../ICML2026/remote_sensing/index.md) · [🤖 AAAI2026 (7)](../../AAAI2026/remote_sensing/index.md) · [🧠 NeurIPS2025 (12)](../../NeurIPS2025/remote_sensing/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/remote_sensing/index.md)

🔥 **高频主题：** 遥感 ×3

**[Adapting Fine-Grained Cross-View Localization to Areas without Fine Ground Truth](adapting_fine-grained_cross-view_localization_to_areas_without_fine_ground_truth.md)**

:   针对细粒度跨视角定位模型在新区域部署时精度下降的问题，提出基于知识自蒸馏的弱监督学习方法——通过模式化伪GT生成、粗粒度监督和离群值过滤三个策略，仅使用目标区域的地面-航拍图像对（无需精确GT），即可在VIGOR和KITTI上将定位误差降低12%~20%。

**[ConGeo: Robust Cross-View Geo-Localization Across Ground View Variations](congeo_robust_cross-view_geo-localization_across_ground_view_variations.md)**

:   提出 ConGeo，一种模型无关的单视图+跨视图对比学习框架，通过强制同一地点不同地面视角变体之间的特征一致性，使单一模型即可在任意朝向和任意视场角(FoV)下实现鲁棒的跨视图地理定位。

**[Cross-Platform Video Person ReID: A New Benchmark Dataset and Adaptation Approach](cross-platform_video_person_reid_a_new_benchmark_dataset_and_adaptation_approach.md)**

:   构建首个地面-无人机跨平台视频行人重识别数据集G2A-VReID，并提出VSLA-CLIP方法，通过视觉-语义对齐和参数高效的Video Set-Level-Adapter将CLIP适配到视频ReID任务。

**[Learning Representations of Satellite Images From Metadata Supervision](learning_representations_of_satellite_images_from_metadata_supervision.md)**

:   本文提出了 SatMIP（Satellite Metadata-Image Pretraining），将卫星图像的元数据（如时间、地理位置、传感器信息等）表示为文本描述，通过图像-元数据对比学习任务在共享嵌入空间中对齐图像和元数据，学习到既包含视觉特征又编码语义信息的卫星图像表征，并进一步提出 SatMIPS（结合图像自监督和元数据监督），在多个遥感下游任务上超越了 SimCLR 等纯视觉自监督方法。

**[Masked Angle-Aware Autoencoder for Remote Sensing Images](masked_angle-aware_autoencoder_for_remote_sensing_images.md)**

:   提出 MA3E，在 MAE 预训练中显式引入角度变化（通过 scaling center crop 构建旋转裁剪），并用最优传输损失自动分配重建目标，使模型感知遥感目标的多样角度，学习旋转不变表示。

**[Weakly-Supervised Camera Localization by Ground-to-Satellite Image Registration](weakly-supervised_camera_localization_by_ground-to-satellite_image_registration.md)**

:   提出首个弱监督的地面-卫星图像配准定位方法，通过卫星-卫星自监督训练旋转估计器、对比学习训练平移估计器，在无需精确GT姿态标签的条件下实现最佳跨区域泛化能力，超越大多数全监督SOTA方法。
