---
title: >-
  [论文解读] SpatialActor: Exploring Disentangled Spatial Representations for Robust Robotic Manipulation
description: >-
  [AAAI 2026 Oral][机器人][语义-几何解耦] 提出 SpatialActor 框架，通过将语义与几何表征显式解耦，并设计语义引导几何模块（SGM）自适应融合深度噪声特征与预训练深度估计专家先验、以及空间 Transformer（SPT）编码低级空间位置线索，在 RLBench 50+ 任务上达到 87.4% 成功率（SOTA +6.0%），且在重噪声条件下比 RVT-2 高出 19.4%。
tags:
  - "AAAI 2026 Oral"
  - "机器人"
  - "语义-几何解耦"
  - "深度估计先验"
  - "Transformer"
  - "鲁棒操控"
  - "RLBench"
---

# SpatialActor: Exploring Disentangled Spatial Representations for Robust Robotic Manipulation

**会议**: AAAI 2026 Oral  
**arXiv**: [2511.09555](https://arxiv.org/abs/2511.09555)  
**作者**: Hao Shi, Bin Xie, Yingfei Liu, Yang Yue, Tiancai Wang, Haoqiang Fan, Xiangyu Zhang, Gao Huang  
**代码**: [GitHub](https://github.com/shihao1895/SpatialActor)  
**领域**: 机器人操控 / 空间表征学习  
**关键词**: 语义-几何解耦, 深度估计先验, 空间Transformer, 鲁棒操控, RLBench

## 一句话总结

提出 SpatialActor 框架，通过将语义与几何表征显式解耦，并设计语义引导几何模块（SGM）自适应融合深度噪声特征与预训练深度估计专家先验、以及空间 Transformer（SPT）编码低级空间位置线索，在 RLBench 50+ 任务上达到 87.4% 成功率（SOTA +6.0%），且在重噪声条件下比 RVT-2 高出 19.4%。

## 研究背景与动机

**3D 空间理解对机器人操控至关重要**：现实操控任务发生在三维空间，需要精确的空间推理、遮挡处理和细粒度物体交互，纯 2D 视觉方法在这些场景下能力不足。

**点云方法的稀疏性问题**：基于点云的方法（PolarNet、PerAct 等）虽然能显式表示 3D 几何，但稀疏采样导致细粒度语义信息丢失，且 3D 标注成本高昂限制了预训练规模。

**图像方法的语义-几何纠缠问题**：基于 RGB-D 的图像方法（RVT、RVT-2 等）将 RGB 和深度送入共享特征空间，语义与几何的纠缠使模型对深度噪声极为敏感——RVT-2 在轻微噪声下成功率即下降 8.9%。

**真实深度噪声不可避免**：传感器噪声、光照变化、表面反射等因素使真实世界中的深度测量始终存在噪声，严重限制了现有方法的实际部署。

**缺乏低级空间线索建模**：现有联合建模方法主要保留高层几何信息，忽略了对精确交互至关重要的低级空间线索（如精确的 2D-3D 对应关系）。

**核心问题**：如何构建同时具备细粒度空间理解、传感器噪声鲁棒性和低级空间线索支持的稳健空间表征？

## 方法详解

### 整体框架

SpatialActor 的输入为 $V$ 个视角的 RGB 图像 $I^v \in \mathbb{R}^{H \times W \times 3}$、深度图 $D^v \in \mathbb{R}^{H \times W}$、本体感受状态 $P$ 和语言指令 $L$。框架采用**三路分离**设计：

- **语义路径**：RGB + 语言指令 → CLIP 视觉-语言编码器 → 语义特征 $F_{\text{sem}}^v$ 和文本特征 $F_{\text{text}}$
- **几何路径**：原始深度 $D^v$ → 深度编码器（ResNet-50）→ 噪声几何特征 $F_{\text{geo}}^v$，经 SGM 增强后得到融合几何特征 $F_{\text{fuse-geo}}^v$
- **空间路径**：将语义和融合几何特征拼接为 $H^v$，经 SPT 进行空间位置编码和多层交互

最终通过动作头预测 3D 末端执行器位姿 $A = (x, y, z, \theta_x, \theta_y, \theta_z, g)$。

### 语义引导几何模块（SGM）

SGM 的核心思想是融合**两种互补的几何表征**：

| 来源 | 特性 | 优势 | 劣势 |
|------|------|------|------|
| 预训练深度估计专家（Depth Anything v2） | 从 RGB 推断几何 | 鲁棒、抗噪 | 粗粒度 |
| 原始深度编码器（ResNet-50） | 从 $D^v$ 提取几何 | 细粒度、像素级 | 对噪声敏感 |

融合机制为**多尺度门控**：

$$G^v = \sigma\bigl(\text{MLP}(\text{Concat}(\hat{F}_{\text{geo}}^v, F_{\text{geo}}^v))\bigr)$$

$$F_{\text{fuse-geo}}^v = G^v \odot F_{\text{geo}}^v + (1 - G^v) \odot \hat{F}_{\text{geo}}^v$$

门控 $G^v$ 自适应学习：在深度可靠区域保留原始细节，在噪声严重区域依赖专家先验，实现细粒度与鲁棒性的平衡。

### 空间 Transformer（SPT）

SPT 为每个空间 token 赋予精确的 3D 位置信息：

1. **3D 坐标计算**：利用相机内参 $K^v$ 和外参 $E^v$，将像素 $(x', y')$ 及对应深度 $d$ 反投影到机器人坐标系下的 3D 坐标 $[x,y,z]^\top$
2. **旋转位置编码（RoPE）**：对 3D 坐标的每个轴分配 $D/3$ 维，生成正弦/余弦位置嵌入，使具有不同空间位置的 token 携带唯一的空间索引
3. **视图级交互**：自注意力 + FFN 细化每个视图内的 token 表征
4. **场景级交互**：拼接所有视图 token 与语言特征 $F_{\text{text}}$，通过自注意力 + FFN 融合跨视图、跨模态信息

### 动作预测与监督

- 解码器（ConvexUp）生成逐视图 2D 热力图，argmax 获取目标 2D 位置后通过相机模型提升至 3D
- MLP 回归旋转角 $(\theta_x, \theta_y, \theta_z)$ 和夹爪状态 $g$
- 损失函数：2D 热力图交叉熵（平移）+ 离散化欧拉角交叉熵（旋转）+ 二分类损失（夹爪）

## 实验

### 实验一：RLBench 18 任务性能对比

**设置**：RLBench 基准，18 个任务 249 个变体，4 个固定 RGB-D 摄像头（128×128 分辨率），每任务 100 条专家演示训练、25 条未见测试。8 GPU 训练约 40k 迭代，batch size=192。

| 方法 | 平均成功率 ↑ | Avg Rank ↓ | Insert Peg | Sort Shape | Drag Stick |
|------|-------------|-----------|------------|------------|------------|
| PerAct | 49.4% | 7.1 | 5.6% | 16.8% | 70.4% |
| RVT | 62.9% | 5.3 | 11.2% | 36.0% | 88.0% |
| 3D Diffuser Actor | 81.3% | 2.8 | 65.6% | 44.0% | 96.8% |
| RVT-2 | 81.4% | 2.8 | 40.0% | 35.0% | 99.0% |
| **SpatialActor** | **87.4%** | **2.3** | **93.3%** | **73.3%** | **98.7%** |

**关键发现**：SpatialActor 平均成功率 87.4%，超越 RVT-2 达 6.0%。在需要高空间精度的 Insert Peg 和 Sort Shape 任务上分别超出 RVT-2 53.3% 和 38.3%，体现了解耦空间表征在精细操控上的优势。

### 实验二：噪声鲁棒性评估

**设置**：向重建点云注入高斯噪声，三个级别——Light（20% 点，std=0.05）、Middle（50% 点，std=0.1）、Heavy（80% 点，std=0.1）。

| 方法 | Light ↑ | Middle ↑ | Heavy ↑ |
|------|---------|----------|---------|
| RVT-2 | 72.5% | 68.4% | 57.0% |
| **SpatialActor** | **86.4%** (+13.9%) | **85.3%** (+16.9%) | **76.4%** (+19.4%) |

**关键发现**：随噪声加重，SpatialActor 的优势不断扩大（13.9% → 16.9% → 19.4%），在 Insert Peg 任务上 Heavy 噪声下超出 RVT-2 达 61.3%，验证了 SGM 门控融合机制的抗噪能力。

### 更多实验结果

- **少样本泛化**（19 新任务，每任务仅 10 条演示）：SpatialActor 79.2% vs RVT-2 46.9%（+32.3%），表明解耦表征显著提升迁移能力
- **ColosseumBench 空间扰动**（20 任务）：在物体尺寸、容器尺寸和摄像头位姿扰动下，SpatialActor 均取得最优结果（基线 57.4%，摄像头扰动下 54.2%）
- **消融实验**：解耦 +3.7%（85.1%），+SGM +1.3%（86.4% / Heavy 噪声 73.9%），+SPT +1.0%（87.4% / Heavy 噪声 76.4%）
- **真实世界实验**：WidowX 机械臂 + RealSense D435i，8 任务 15 变体，SpatialActor 63% vs RVT-2 43%（+20%）

## 亮点与创新

1. **语义-几何显式解耦**：打破图像方法中语义与几何共享特征空间的范式，使深度噪声不再干扰语义理解，从根源上提升鲁棒性。
2. **互补几何融合**：SGM 巧妙结合预训练深度估计专家（抗噪但粗糙）与原始深度（精细但噪声大）的互补优势，门控机制可自适应调节信任度。
3. **低级空间线索建模**：SPT 通过 RoPE 将真实 3D 坐标编码到 Transformer 中，建立精确的 2D-3D 对应关系，使空间 token 间的交互具有几何意义。
4. **噪声条件下的压倒性优势**：Heavy 噪声下成功率 76.4%（vs 57.0%），少样本场景 79.2%（vs 46.9%），实际部署价值显著。

## 局限性

1. **计算开销增加**：引入冻结的 Depth Anything v2 专家模型增加了推理时的计算和显存成本，对资源受限的机器人平台可能构成瓶颈。
2. **依赖预训练深度估计质量**：SGM 的鲁棒性依赖于深度估计专家的泛化能力，当场景分布远超预训练数据时，专家先验质量可能退化。
3. **单臂桌面场景**：实验主要在桌面平台（Franka / WidowX）验证，未涉及双臂协作、灵巧手或移动操控等更复杂场景。
4. **固定视角假设**：仿真中使用 4 个固定 RGB-D 摄像头，真实世界仅 1 个静态摄像头，对动态视角或手眼（eye-in-hand）配置的适用性未验证。

## 相关工作

- **点云方法**：PolarNet (Chen 2023)、PerAct (Shridhar 2023) 显式 3D 结构但稀疏
- **图像方法**：RVT (Goyal 2023)、RVT-2 (Goyal 2024)、SAM-E (Zhang 2024) 密集语义但语义-几何纠缠
- **3D 扩散策略**：3D Diffuser Actor (Ke 2024) 3D 场景表示 + 扩散策略
- **视觉基础模型**：CLIP (Radford 2021) 语义先验，Depth Anything v2 (Yang 2025) 几何先验
- **体素方法**：C2F-ARM-BC (James 2022) 粗到细体素化，计算开销大

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 实用性 | ⭐⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 推荐阅读 | ⭐⭐⭐⭐ |

**总体推荐**：⭐⭐⭐⭐  
**推荐理由**：在机器人操控领域提出了一个工程直觉清晰且效果显著的空间表征框架，语义-几何解耦 + 互补几何融合的设计在噪声条件下优势明显，50+ 任务的全面评估和真实机器人实验大幅提升了可信度。核心思想(解耦 + 互补融合)对其他需要多模态鲁棒融合的任务也具有参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Learning Surgical Robotic Manipulation with 3D Spatial Priors](../../CVPR2026/robotics/learning_surgical_robotic_manipulation_with_3d_spatial_priors.md)
- [\[AAAI 2026\] SemanticVLA: Semantic-Aligned Sparsification and Enhancement for Efficient Robotic Manipulation](semanticvla_semantic-aligned_sparsification_and_enhancement_for_efficient_roboti.md)
- [\[AAAI 2026\] TouchFormer: A Robust Transformer-based Framework for Multimodal Material Perception](touchformer_a_robust_transformer-based_framework_for_multimodal_material_percept.md)
- [\[AAAI 2026\] Actor-Critic for Continuous Action Chunks: A Reinforcement Learning Framework for Long-Horizon Robotic Manipulation with Sparse Reward](actor-critic_for_continuous_action_chunks_a_reinforcement_le.md)
- [\[CVPR 2026\] Expanding Spatial and Temporal Context for Robotic Imitation Learning With Scene Graphs](../../CVPR2026/robotics/expanding_spatial_and_temporal_context_for_robotic_imitation_learning_with_scene.md)

</div>

<!-- RELATED:END -->
