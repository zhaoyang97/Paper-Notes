---
title: >-
  [论文解读] Web-Scale Collection of Video Data for 4D Animal Reconstruction
description: >-
  [NeurIPS 2025][3D视觉][4D动物重建] 提出一个全自动化的大规模视频数据采集管线，从 YouTube 挖掘并处理得到 30K 动物视频（2M帧），建立首个 4D 四足动物重建基准 Animal-in-Motion（230序列/11K帧），并提出 4D-Fauna 基线方法实现序列级优化的无模型 4D 重建。
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "4D动物重建"
  - "数据管线"
  - "YouTube视频挖掘"
  - "基准数据集"
  - "单视图重建"
---

# Web-Scale Collection of Video Data for 4D Animal Reconstruction

**会议**: NeurIPS 2025  
**arXiv**: [2511.01169](https://arxiv.org/abs/2511.01169)  
**代码**: [https://github.com/briannlongzhao/Animal-in-Motion](https://github.com/briannlongzhao/Animal-in-Motion)  
**领域**: Video Understanding / 3D Vision  
**关键词**: 4D动物重建, 数据管线, YouTube视频挖掘, 基准数据集, 单视图重建

## 一句话总结
提出一个全自动化的大规模视频数据采集管线，从 YouTube 挖掘并处理得到 30K 动物视频（2M帧），建立首个 4D 四足动物重建基准 Animal-in-Motion（230序列/11K帧），并提出 4D-Fauna 基线方法实现序列级优化的无模型 4D 重建。

## 研究背景与动机

动物形态和运动的视觉分析在野生动物保护、生物力学和机器人学领域有重要应用。传统方法依赖昂贵的多视图控制环境或标记系统，近年来单视图方法（姿态估计、跟踪、3D/4D重建）取得了进展，但**严重受限于数据规模**。

现有动物视频数据集存在三个关键问题：（1）**规模极小**——最大的 APT-36K 仅 2.4K 个 15 帧短片；（2）**缺少物体中心裁剪**——原始视频中可能多个动物重叠、无分割掩码；（3）**缺少关键预处理**——没有为 3D/4D 重建任务准备好所需的辅助标注（关键点、光流、深度等）。唯一真正适合 4D 动物重建的 BADJA 数据集仅有 11 个视频。

核心矛盾：数据驱动方法需要大量高质量数据，但动物视频的采集和标注极为费力。本文的解决方案是利用 YouTube 的海量视频资源，构建全自动化的采集-处理-标注管线。

## 方法详解

### 整体框架

四阶段管线：（1）从 YouTube 搜索和下载原始视频；（2）视频预处理（镜头分割、CLIP 过滤）；（3）动物检测与跟踪（Grounded-SAM-2），生成物体中心裁剪；（4）特征提取（关键点、DINO特征、光流、深度图、遮挡边界）。整个管线通过中心数据库协调，支持多进程并行。

### 关键设计

1. **智能搜索查询生成**：

    - 输入一个动物类别（如"horse"），用 GPT 生成子品种（Clydesdale, Mustang）和上下文短语（racing competition, in a farm）
    - 随机组合形成多样化搜索文本，最大化视频多样性
    - 使用 Selenium + pytube 进行搜索和下载

2. **多层级过滤与跟踪**：

    - **镜头分割**：用 PySceneDetect 按像素变化检测场景切换，防止跟踪跨镜头混淆
    - **CLIP 过滤**：计算帧与类别文本的 CLIPScore，丢弃低分片段
    - **Grounded-SAM-2 跟踪**：迭代式 grounding-tracking 实现长期跟踪
    - **多层过滤**：
        - 重叠实例过滤（IoU 阈值去除多动物重叠帧）
        - 低分辨率过滤（bbox面积 < 裁剪尺寸的 1/4）
        - 截断实例过滤（bbox 靠近帧边界）
        - 不一致轨迹过滤（相邻帧 IoU 突变检测身份切换）
    - **物体中心裁剪**：以 bbox 为中心生成方形裁剪框，移动平均平滑
    - 最后用 GPT 进行最终视觉验证，去除误检和严重遮挡

3. **全面的特征提取模块**：

    - ViTPose++：动物关键点估计
    - DINOv2：图像特征
    - SEA-RAFT：光流估计
    - Depth Anything V2：深度估计
    - 遮挡边界：基于深度图在掩码边界处的深度差异计算

4. **4D-Fauna 基线方法**：

    - 基于 3D-Fauna（无模型方法）进行序列级优化
    - **关键点监督**：引入 2D 关键点作为部件级约束，解决腿部排序问题
    - **时序平滑损失**：对相机姿态参数变化量和动物姿态关节速度施加正则
    - **高效过拟合策略**：直接优化每帧的相机姿态和关节参数，以预训练网络输出作为初始化

### 损失函数 / 训练策略

4D-Fauna 使用 3D-Fauna 的原始反向渲染损失（掩码 IoU + DINO 特征匹配）+ 新增的关键点重投影损失 + 相机姿态和关节速度的时序平滑正则项。在预训练模型基础上逐序列优化。

## 实验关键数据

### 主实验

| 方法 | IoU↑ | PCK@0.1↑ | PCK@0.05↑ | KT-PCK@0.1↑ | MPJVE↓ | 类型 |
|------|------|----------|-----------|-------------|--------|------|
| SMALify | **0.867** | **0.954** | **0.787** | **0.623** | **0.023** | 基于模型 |
| AniMer | 0.677 | 0.537 | 0.199 | 0.566 | 0.038 | 基于模型 |
| 3D-Fauna | 0.670 | 0.470 | 0.177 | 0.329 | 0.058 | 无模型 |
| 4D-Fauna | 0.814 | 0.664 | 0.317 | 0.418 | 0.044 | 无模型 |

### 消融实验

| 配置 | IoU | PCK@0.1 | 说明 |
|------|-----|---------|------|
| 3D-Fauna (直接推理) | 0.670 | 0.470 | 基线无模型方法 |
| + 序列优化 + 关键点 | 0.814 | 0.664 | 关键点消除腿部排序问题 |
| + 时序平滑 | ↑ | ↑ | 减少帧间抖动 |

### 关键发现
- **2D 指标的误导性**：SMALify 在所有定量指标上最优，但定性检查发现其经常产生不自然的 3D 形状（深度方向身体拉长、腿部异常弯曲、正面视角形状扭曲），暴露了 2D 投影指标的局限
- **无模型方法的优势**：3D-Fauna/4D-Fauna 生成更自然合理的 3D 形状和姿态，但 2D 指标反而较低
- **序列优化的必要性**：3D-Fauna 的前馈推理会导致帧间腿部突然切换；4D-Fauna 通过关键点约束和时序平滑有效解决
- 数据管线成功采集 30K 视频/2M 帧，覆盖 23 个动物类别

## 亮点与洞察
- **端到端自动化**：从搜索查询生成到最终特征提取完全自动，仅基准验证需要少量人工
- **揭示评价指标缺陷**：清晰展示了 2D 投影指标与 3D 重建质量的不一致，强调了 3D 感知评价指标的必要性
- **模型适配巧妙**：4D-Fauna 不重新训练，而是将前馈模型的输出作为优化初始化，兼具无模型方法的泛化能力和基于模型方法的精度

## 局限与展望
- 自动管线的标注不完全干净，基准评测仍需人工验证
- 基准仅使用 2D 投影指标，缺乏真正的 3D ground truth
- 4D-Fauna 对时序一致性的建模有限——未来可探索自回归模型捕捉帧间动力学
- 出于版权考虑不发布原始 RGB 视频帧，仅提供衍生数据
- 管线目前聚焦四足动物，扩展到鸟类、鱼类等需要调整

## 相关工作与启发
- **vs APT-36K**: 规模扩大 12.5 倍（30K vs 2.4K），且包含完整预处理特征
- **vs BADJA**: 从 11 个视频扩展到 230 个基准序列，首个真正面向 4D 重建的基准
- **vs 3D-Fauna**: 4D-Fauna 在其基础上增加关键点和时序约束，所有指标均有提升
- **vs SMALify**: 定量更优但 3D 质量堪忧，凸显了评价体系的问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 管线工程贡献大，4D-Fauna 方法创新有限但实用
- 实验充分度: ⭐⭐⭐⭐ 数据规模令人印象深刻，基准全面，但缺乏 3D 定量评估
- 写作质量: ⭐⭐⭐⭐ 管线描述详尽清晰，分析深刻
- 价值: ⭐⭐⭐⭐⭐ 数据集和管线对社区价值巨大，有望推动 4D 动物重建领域发展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting](../../ICCV2025/3d_vision/vivid4d_improving_4d_reconstruction_from_monocular_video_by_video_inpainting.md)
- [\[ICCV 2025\] Shape of Motion: 4D Reconstruction from a Single Video](../../ICCV2025/3d_vision/shape_of_motion_4d_reconstruction_from_a_single_video.md)
- [\[NeurIPS 2025\] Flux4D: Flow-based Unsupervised 4D Reconstruction](flux4d_flow-based_unsupervised_4d_reconstruction.md)
- [\[ICCV 2025\] Geo4D: Leveraging Video Generators for Geometric 4D Scene Reconstruction](../../ICCV2025/3d_vision/geo4d_leveraging_video_generators_for_geometric_4d_scene_reconstruction.md)
- [\[NeurIPS 2025\] Every Camera Effect, Every Time, All at Once: 4D Gaussian Ray Tracing for Physics-based Camera Effect Data Generation](every_camera_effect_every_time_all_at_once_4d_gaussian_ray_tracing_for_physics-b.md)

</div>

<!-- RELATED:END -->
