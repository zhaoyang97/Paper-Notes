---
title: >-
  [论文解读] Zero-Shot 4D Lidar Panoptic Segmentation
description: >-
  [CVPR 2025][自动驾驶][零样本分割] 本文提出 SAL-4D（Segment Anything in Lidar-4D），利用多模态传感器设置作为桥梁，将视频对象分割（VOS）模型和 CLIP 视觉语言特征蒸馏到 LiDAR 空间，实现零样本 4D LiDAR 全景分割，在 3D 零样本 LPS 上超越先前方法 5+ PQ。
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "零样本分割"
  - "4D LiDAR"
  - "全景分割"
  - "视频对象分割"
  - "视觉语言模型蒸馏"
---

# Zero-Shot 4D Lidar Panoptic Segmentation

**会议**: CVPR 2025  
**arXiv**: [2504.00848](https://arxiv.org/abs/2504.00848)  
**代码**: 无  
**领域**: 自动驾驶 / 点云分割  
**关键词**: 零样本分割, 4D LiDAR, 全景分割, 视频对象分割, 视觉语言模型蒸馏

## 一句话总结
本文提出 SAL-4D（Segment Anything in Lidar-4D），利用多模态传感器设置作为桥梁，将视频对象分割（VOS）模型和 CLIP 视觉语言特征蒸馏到 LiDAR 空间，实现零样本 4D LiDAR 全景分割，在 3D 零样本 LPS 上超越先前方法 5+ PQ。

## 研究背景与动机

**领域现状**：4D（3D空间+时间）场景理解对于具身导航和自动驾驶至关重要，应用涵盖流式感知、语义建图和定位。LiDAR 全景分割（LPS）需要为每个点预测语义类别和实例ID，但现有方法严重依赖大量人工标注的 LiDAR 数据。

**现有痛点**：LiDAR 标注成本极高（标注一帧点云需要约 1 小时），而且现有标注数据集的类别多样性有限（如 nuScenes 只有 16 个前景类）。零样本方法能识别任意类别的对象，但在 LiDAR 领域进展缓慢——3D 零样本 LPS 刚起步，4D 维度几乎无人涉足。核心挑战在于缺乏足够多样和大规模的标注数据。

**核心矛盾**：2D 视觉领域有大量的基础模型（SAM、CLIP、VOS模型），而 LiDAR 领域缺乏类似的通用模型。直接将 2D 模型应用于 3D 点云有巨大的模态鸿沟。

**本文目标**：设计一种方法，不使用任何人工 LiDAR 标注，通过将 2D 视觉基础模型的知识迁移到 3D LiDAR，实现零样本的 4D 全景分割。

**切入角度**：利用自动驾驶平台上相机和 LiDAR 的标定对齐关系作为天然的跨模态桥梁。通过 VOS 模型在视频中追踪对象得到时间一致的 tracklets，用 CLIP 赋予每个 tracklet 语义，再通过已标定的传感器投影关系提升（lift）到 4D LiDAR 空间。

**核心 idea**：用 VOS + CLIP 在 2D 视频域生成伪标签 tracklets，将其投射到 4D LiDAR 生成训练数据，蒸馏训练 SAL-4D 模型。

## 方法详解

### 整体框架
SAL-4D 的训练流程为：(1) 用 off-the-shelf VOS 模型（如 SAM 2）在短视频片段中追踪所有可见对象，得到时间一致的 2D mask tracklets；(2) 为每个 tracklet 计算序列级别的 CLIP token 作为语义描述；(3) 通过 camera-LiDAR 标定矩阵将 2D tracklets 提升到 4D LiDAR 点云空间，生成伪标签；(4) 在伪标签上训练 SAL-4D 模型。推理时，SAL-4D 直接接收 LiDAR 点云输入，无需相机数据。

### 关键设计

1. **VOS 驱动的 2D Tracklet 生成**:

    - 功能：在视频中获取时间一致的对象分割
    - 核心思路：利用最新的 VOS 模型（如 SAM 2）对短视频片段进行自动分割和追踪，不需要任何人工提示或标注。VOS 模型通过在首帧发现所有对象并在后续帧追踪，保证了同一对象在不同时间步得到一致的 ID。关键在于 VOS 的"类别无关"特性——它追踪任何可见对象而不限于预定义类别，这是实现零样本的基础
    - 设计动机：VOS 模型已能在 2D 图像上实现高质量的零样本追踪，但这种能力无法直接用于 LiDAR。通过标定的多模态传感器系统作为桥梁，可以把 2D 的追踪能力"传导"到 3D

2. **序列级 CLIP 语义标注**:

    - 功能：为每个追踪到的对象赋予开放词汇的语义特征
    - 核心思路：对每个 tracklet，在时间维度上收集该对象在各帧的 crop 图像，分别通过 CLIP 图像编码器提取特征后取平均，得到一个稳定的序列级 CLIP token。这个 token 不是固定类别标签，而是连续的语义向量，保留了 CLIP 的开放词汇能力。通过多帧平均，可以缓解单帧遮挡或视角变化导致的特征噪声
    - 设计动机：传统方法为每帧每对象单独计算语义，导致同一对象在不同帧的标签可能不一致。序列级聚合确保了时间一致的语义表示

3. **2D-to-4D 伪标签提升与 SAL-4D 模型蒸馏**:

    - 功能：将 2D 视频域的分割知识迁移到 4D LiDAR 域
    - 核心思路：利用 camera 到 LiDAR 的标定矩阵，将每个 2D mask 对应的像素映射到 3D LiDAR 点上。一个 LiDAR 点可能被多帧、多相机的 mask 覆盖，通过投票或置信度加权确定最终标签。时间维度上，同一 tracklet ID 在多帧的 LiDAR 投影构成 4D tracklet。SAL-4D 模型接收 LiDAR 点云序列输入，预测每个点的实例 ID 和 CLIP 语义向量，损失同时包含实例分割损失和 CLIP token 回归损失
    - 设计动机：伪标签虽然有噪声（投射误差、遮挡等），但大规模伪标签的统计优势可以弥补单样本质量不足。SAL-4D 在蒸馏过程中还可以学到 2D 伪标签中不具备的 3D 几何推理能力

### 损失函数 / 训练策略
SAL-4D 的训练损失包含三部分：(1) 实例分割损失——用匈牙利匹配将预测实例与伪标签实例配对后计算 mask + 分类损失；(2) CLIP 特征回归损失——预测的点级 CLIP 特征与伪标签 CLIP token 的 L2 距离；(3) 时间一致性损失——鼓励相邻帧中同一实例的预测特征保持一致。

## 实验关键数据

### 主实验

| 方法 | 数据集 | 3D LPS PQ | 零样本 | 时间一致 |
|------|--------|-----------|--------|---------|
| OpenScene | nuScenes | 18.3 | ✓ | ✗ |
| LidarCLIP | nuScenes | 15.7 | ✓ | ✗ |
| 先前 3D SOTA | nuScenes | ~20.0 | ✓ | ✗ |
| **SAL-4D (3D)** | **nuScenes** | **25.2** | **✓** | **✗** |
| **SAL-4D (4D)** | **nuScenes** | **27.8** | **✓** | **✓** |

### 消融实验

| 配置 | PQ | 说明 |
|------|------|------|
| Full SAL-4D | 27.8 | 完整模型（4D） |
| w/o 时间一致性 | 25.2 | 退化为逐帧 3D，-2.6 |
| w/o 序列级CLIP | 23.4 | 单帧CLIP不稳定，-4.4 |
| w/o VOS追踪 | 19.8 | 无tracking退化为逐帧分割，-8.0 |
| 单相机→多相机 | 22.1→27.8 | 多相机覆盖更全，+5.7 |

### 关键发现
- **时间一致性是 4D 的关键优势**：4D SAL-4D 比 3D 版本高 2.6 PQ，因为时间维度的信息有助于处理单帧中的遮挡和稀疏问题
- **VOS 追踪贡献最大**：去掉 VOS 后性能下降 8.0 PQ，说明高质量的 tracklet 是整个方法的基础
- 多相机设置对伪标签质量影响显著——覆盖范围从单相机的有限视角扩展到近 360°
- 零样本设置下也能识别训练集中未见的类别（如动物、施工工具），展示了 CLIP 特征的泛化能力

## 亮点与洞察
- **模态桥梁思路**：利用已标定的多模态传感器系统作为 2D→3D 知识迁移的天然通道，避免了复杂的跨模态学习。这种"借道"策略在有多传感器设置的机器人系统中普遍可用
- **时间维度的信息增益**：4D 不仅是 3D 的叠加——时间一致性约束可以修正单帧的预测错误，类似于视频中的时间平滑。这对 LiDAR 点云的稀疏性问题尤其有帮助
- **开放词汇的保留**：通过蒸馏 CLIP 特征而非固定类别标签，SAL-4D 保持了零样本识别任意类别的能力

## 局限与展望
- 伪标签质量受限于 VOS 模型的追踪准确性——在快速运动、严重遮挡或远距离对象上可能失败
- 2D→3D 的投射在 LiDAR 点云边缘处存在对齐误差，影响实例边界精度
- 当前方法依赖标定的 camera-LiDAR 设置，无法应用于只有 LiDAR 的场景
- 推理速度未在论文中充分讨论，4D 模型处理点云序列的实时性是实际部署的关键问题
- 未来可探索将 SAM 2 的提示机制引入 LiDAR 域，实现用户交互式的 4D 分割

## 相关工作与启发
- **vs OpenScene/LidarCLIP**：它们直接将 CLIP 特征蒸馏到 3D 点上但不做实例分割，SAL-4D 额外利用 VOS 提供实例级信息并扩展到 4D
- **vs SAL（3D 版本）**：SAL-4D 在 SAL 的基础上添加时间维度，证明 4D 的实例追踪可以显著提升分割质量
- **vs 有监督 4D-LPS 方法**：有监督方法在封闭类别集合上表现更好，但无法识别新类别。SAL-4D 在零样本设定下的 PQ 已接近某些早期有监督方法

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次实现零样本 4D LiDAR 全景分割，VOS+CLIP+LiDAR 的组合有新意
- 实验充分度: ⭐⭐⭐⭐ 消融较完善，但与更多最新方法的对比可以更充分
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，管线图直观
- 价值: ⭐⭐⭐⭐ 对零标注的自动驾驶场景理解有实际推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] 4DSegStreamer: Streaming 4D Panoptic Segmentation via Dual Threads](../../ICCV2025/autonomous_driving/4dsegstreamer_streaming_4d_panoptic_segmentation_via_dual_threads.md)
- [\[CVPR 2025\] 3D-AVS: LiDAR-based 3D Auto-Vocabulary Segmentation](3d-avs_lidar-based_3d_auto-vocabulary_segmentation.md)
- [\[CVPR 2025\] Exploring Scene Affinity for Semi-Supervised LiDAR Semantic Segmentation](exploring_scene_affinity_for_semi-supervised_lidar_semantic_segmentation.md)
- [\[CVPR 2025\] V2X-R: Cooperative LiDAR-4D Radar Fusion with Denoising Diffusion for 3D Object Detection](v2x-r_cooperative_lidar-4d_radar_fusion_with_denoising_diffusion_for_3d_object_d.md)
- [\[CVPR 2025\] A Dataset for Semantic Segmentation in the Presence of Unknowns](a_dataset_for_semantic_segmentation_in_the_presence_of_unknowns.md)

</div>

<!-- RELATED:END -->
