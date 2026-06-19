---
title: "CADCrafter: Generating Computer-Aided Design Models from Unconstrained Images"
authors: "Cheng Chen, Xiaohui Zeng, Yuwei Li, Hippolyte Music, Tobias Ritschel, Sanja Fidler, Florian Shkurti"
venue: "CVPR 2025"
date: 2025-04-07
tags: [cad-generation, image-to-3d, latent-diffusion, dpo, 3d-reconstruction]
arxiv: "2504.04753"
---

# CADCrafter: Generating Computer-Aided Design Models from Unconstrained Images

**作者**: Cheng Chen, Xiaohui Zeng, Yuwei Li, Hippolyte Music, Tobias Ritschel, Sanja Fidler, Florian Shkurti  
**机构**: NTU / A*STAR / UT Austin  
**会议**: CVPR 2025  

## 研究背景与动机

从单张图像生成可编辑的 CAD 模型是计算机视觉和图形学的长期目标。现有方法通常依赖参数化曲面拟合或基于模板的检索，但这些方法面临严重局限：

**CAD 程序的离散性**：CAD 模型由一系列离散的建模操作（如拉伸、倒角、布尔运算）组成，传统的连续优化方法难以直接处理这种离散序列结构

**图像到 CAD 的巨大语义鸿沟**：RGB 图像包含纹理、光照等非几何信息，而 CAD 程序只关注纯几何操作，两者之间的映射极其困难

**数据匮乏问题**：高质量的图像-CAD 配对数据极度稀缺，现有数据集（如 DeepCAD）仅包含 CAD 程序而无对应图像

**生成质量难以保证**：生成的 CAD 程序可能包含语法错误或几何不一致，导致无法通过 CAD 编译器编译

这些问题促使作者提出 CADCrafter：一个基于潜在扩散模型的 Image-to-CAD 系统，通过几何编码器桥接图像与 CAD 程序之间的语义鸿沟，并利用 DPO 微调确保生成质量。

## 方法详解

### 整体框架

CADCrafter 采用两阶段训练策略：

1. **第一阶段**：训练 CAD 序列的 VAE 和潜在扩散模型
2. **第二阶段**：引入几何条件和 DPO 微调

### 几何编码器（Geometry Encoder）

为了弥合 RGB 图像与 CAD 程序之间的语义鸿沟，作者设计了多模态几何编码器：

| 特征类型 | 提取方式 | 作用 |
|----------|----------|------|
| 深度图 (Depth) | 预训练单目深度估计 | 提供全局3D形状信息 |
| 法线图 (Normal) | 预训练法线估计 | 捕捉局部表面方向 |
| DINO-V2 语义特征 | 预训练 DINO-V2 | 提供高层语义理解 |

三种特征通过特征融合模块（Feature Fusion Module）整合后，作为条件注入扩散模型的交叉注意力层。

### 潜在扩散模型

CAD 程序被表示为一系列命令序列 $S = \{c_1, c_2, ..., c_N\}$，每个命令 $c_i$ 包含操作类型和参数。具体流程：

1. **VQ-VAE 编码**：将 CAD 命令序列编码为潜在向量 $z = E(S)$
2. **前向扩散**：$q(z_t | z_{t-1}) = \mathcal{N}(z_t; \sqrt{1-\beta_t} z_{t-1}, \beta_t I)$
3. **反向去噪**：条件扩散模型 $p_\theta(z_{t-1} | z_t, c_{geo})$，其中 $c_{geo}$ 为几何条件
4. **解码**：$\hat{S} = D(\hat{z}_0)$

### DPO 微调（Direct Preference Optimization）

关键创新之一是使用 CAD 编译器作为自动质量评判器：

- **正样本**：能通过 CAD 编译器成功编译的生成程序
- **负样本**：编译失败或几何误差大的生成程序
- **DPO 损失**：$\mathcal{L}_{DPO} = -\log \sigma(\beta (\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}))$
- 超参数 $\beta = 20$，控制偏好学习的强度

DPO 微调显著提升了生成程序的编译通过率和几何精度。

### 多视角到单视角蒸馏

训练时使用多视角渲染作为条件（提供更完整的几何信息），推理时通过知识蒸馏使模型只需单视角输入即可工作：

- 教师模型：以多视角几何特征为条件
- 学生模型：以单视角几何特征为条件
- 蒸馏损失：对齐两者在潜在空间中的去噪预测

## 实验结果

### DeepCAD 数据集

| 方法 | Acc_cmd ↑ | Acc_param ↑ | Med CD ↓ | Invalid ↓ |
|------|-----------|-------------|----------|-----------|
| DeepCAD | 78.41% | 72.15% | 0.082 | 12.3% |
| SkexGen | 80.56% | 74.33% | 0.067 | 9.8% |
| CADCrafter (ours) | **83.23%** | **77.89%** | **0.049** | **4.2%** |

### 消融实验

| 配置 | Acc_cmd | Med CD |
|------|---------|--------|
| 无几何编码器 | 76.8% | 0.091 |
| 仅 Depth | 79.4% | 0.072 |
| Depth + Normal | 81.5% | 0.058 |
| 全部特征 (D+N+DINO) | 82.1% | 0.053 |
| + DPO 微调 | **83.23%** | **0.049** |

### RealCAD 数据集

作者构建了 RealCAD 数据集用于验证真实场景性能：
- 包含 3D 打印实物模型的多角度拍摄图像
- CADCrafter 在真实图像上同样展现出稳健的 CAD 重建能力
- 定性结果显示生成的 CAD 模型保持了良好的拓扑结构和可编辑性

## 核心创新点

1. **几何条件桥接**：通过 Depth + Normal + DINO-V2 的多模态几何编码器，有效弥合图像与 CAD 程序之间的语义鸿沟
2. **DPO 微调策略**：创新性地使用 CAD 编译器作为自动偏好标注工具，通过 DPO 提升生成质量
3. **多视角蒸馏**：训练时利用多视角信息，推理时仅需单视角输入
4. **RealCAD 数据集**：首个包含 3D 打印真实物体的 Image-to-CAD 评测数据集

## 局限性

- 目前仅支持由拉伸操作构成的 CAD 模型，不支持旋转体、扫掠体等复杂操作
- 对遮挡严重或纹理复杂的图像，深度/法线估计可能不够准确
- DPO 微调依赖 CAD 编译器的二元反馈（成功/失败），缺少更细粒度的质量评估

## 相关工作

- DeepCAD: 基于 Transformer 的 CAD 序列生成先驱工作
- SkexGen: 基于草图-拉伸分离的 CAD 生成
- Point2CAD: 从点云重建 CAD
- DPO 在 LLM/扩散模型中的对齐应用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Generalizable Sparse-View 3D Reconstruction from Unconstrained Images](../../CVPR2026/3d_vision/generalizable_sparse-view_3d_reconstruction_from_unconstrained_images.md)
- [\[ICCV 2025\] Auto-Regressively Generating Multi-View Consistent Images](../../ICCV2025/3d_vision/auto-regressively_generating_multi-view_consistent_images.md)
- [\[CVPR 2025\] Olympus: A Universal Task Router for Computer Vision Tasks](olympus_a_universal_task_router_for_computer_vision_tasks.md)
- [\[CVPR 2025\] Pow3R: Empowering Unconstrained 3D Reconstruction with Camera and Scene Priors](pow3r_empowering_unconstrained_3d_reconstruction_with_camera_and_scene_priors.md)
- [\[ECCV 2024\] CanonicalFusion: Generating Drivable 3D Human Avatars from Multiple Images](../../ECCV2024/3d_vision/canonicalfusion_generating_drivable_3d_human_avatars_from_multiple_images.md)

</div>

<!-- RELATED:END -->
