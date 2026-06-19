---
title: >-
  [论文解读] BEDLAM2.0: Synthetic Humans and Cameras in Motion
description: >-
  [NeurIPS 2025 Oral][人体理解][synthetic data] BEDLAM2.0 在 BEDLAM 基础上全面升级——引入多样化相机运动（合成平移/追踪/环绕 + 手持/头戴设备捕捉）、更广体型覆盖（BMI 18-41）、strand-based 发型、鞋子、分级服装和更多3D环境，构建 27K+ 序列 / 8M+ 帧的合成数据集，仅用合成数据训练即可在世界坐标系人体运动估计上超越 SOTA。
tags:
  - "NeurIPS 2025 Oral"
  - "人体理解"
  - "synthetic data"
  - "SMPL-X"
  - "camera motion"
  - "HPS estimation"
  - "world coordinates"
  - "BEDLAM"
---

# BEDLAM2.0: Synthetic Humans and Cameras in Motion

**会议**: NeurIPS 2025 Oral  
**arXiv**: [2511.14394](https://arxiv.org/abs/2511.14394)  
**代码/数据**: [bedlam2.is.tuebingen.mpg.de](https://bedlam2.is.tuebingen.mpg.de/)  
**领域**: 人体姿态估计 / 合成数据  
**关键词**: synthetic data, SMPL-X, camera motion, HPS estimation, world coordinates, BEDLAM

## 一句话总结

BEDLAM2.0 在 BEDLAM 基础上全面升级——引入多样化相机运动（合成平移/追踪/环绕 + 手持/头戴设备捕捉）、更广体型覆盖（BMI 18-41）、strand-based 发型、鞋子、分级服装和更多3D环境，构建 27K+ 序列 / 8M+ 帧的合成数据集，仅用合成数据训练即可在世界坐标系人体运动估计上超越 SOTA。

## 研究背景与动机

**领域现状**：BEDLAM 是首个可在不使用真实图像的情况下独立训练 SOTA 3D 人体姿态回归器的合成数据集，已成为 HPS（Human Pose and Shape）方法的标准训练集。但世界坐标系下的人体运动估计（考虑相机运动和变焦）是当前研究热点，而 BEDLAM 的相机运动和焦距多样性严重不足。

**现有痛点**：(1) BEDLAM 大部分序列使用静态相机，仅有极少量移动相机片段，相机运动多样性严重不足；(2) 焦距覆盖有限（HFOV 主要集中在 52° 或 65°），不符合真实视频的多样焦距分布；(3) 体型多样性不足——缺乏高 BMI 身体；(4) 所有人物光脚、发型不够真实、服装尺码单一不适合不同体型。

**核心矛盾**：世界坐标系人体运动估计需要大量带有 ground truth 相机运动和 3D 人体参数的训练数据，真实数据很难获取，合成数据成为关键路径，但 BEDLAM 的合成多样性不够。

**本文目标** 构建一个在相机运动、体型、服装、发型、场景等各方面都大幅超越 BEDLAM 的合成数据集，特别支持世界坐标系人体姿态估计的端到端训练。

**切入角度**：从数据集工程的角度，系统性地改进 BEDLAM 的每一个维度——相机（焦距 + 运动）、人体（体型 + 动作 + 手部）、外观（发型 + 鞋子 + 服装）、场景与渲染。

**核心 idea**：通过合成+捕捉的多样化相机运动、分级服装、strand-based 发型和 SMPL-X 鞋子等系统性改进，让合成数据独立训练即可达到甚至超越使用真实数据的 SOTA。

## 方法详解

### 整体框架

AMASS 动作库采样（4643 个动作）→ 动作重定向到多样化体型（BMI 18-41）→ 穿戴分级服装 + strand-based 发型 + 鞋子 → 放置到 15 个 3D 环境中 → 合成/捕捉相机运动 + 多样焦距 → Unreal Engine 5.3 渲染（1280×720@30fps）→ 输出图像+深度+SMPL-X GT+相机参数。

### 关键设计

1. **多样化相机系统**
    - 焦距覆盖 14mm-400mm（16:9 DSLR 传感器），9% 的视频在拍摄过程中变焦
    - 合成相机运动：静态、平移、追踪、推拉、环绕、变焦及组合，叠加可微分 Perlin 噪声模拟手抖
    - 捕捉相机运动：使用手机/平板和 Apple Vision Pro 头戴设备在虚拟场景中捕捉真实相机运动（静态位置拍摄、环绕拍摄、接近/后退拍摄），86.4% 合成 + 13.6% 捕捉
    - 设计动机：真实视频中相机运动极为多样，BEDLAM 的静态相机导致端到端训练世界坐标系方法效果不佳

2. **体型、服装与外观多样性**
    - 体型：1615 个 SMPL-X 体型，BMI 18-41，对高 BMI 进行重采样以增加覆盖
    - 服装：187 套 3D 服装（比 BEDLAM 多 76 套），50 套分级为 XS-6XL，按 BMI 匹配服装尺码
    - 发型：40 种 strand-based 3D hair grooms（5万-10万根发丝/groom），适配个体头型，9 种发色预设
    - 鞋子：182 种鞋子（Google Scanned Objects），通过 displacement map 将鞋子形状映射到 SMPL-X "袜脚"网格上，并根据鞋底厚度调整身高
    - 设计动机：弥合合成数据与真实图像之间的域差距（光脚、无发型、尺码单一等问题）

3. **场景、渲染与遮挡**
    - 15 个高质量 3D 环境（BEDLAM 仅 5 个），9 个室内场景（BEDLAM 仅 1 个）
    - 时间-天气随机化（日光/日落/阴天/夜晚）
    - 自定义 UE5 C++ 插件确保相机 Shake 在图像和深度渲染之间的确定性一致性
    - 12.7% 的图像存在 >20% 遮挡，前10% 最多遮挡的身体平均遮挡率 61.1%

### 数据集规模

27480 个视频序列、8,048,411 帧 PNG、12.5M 训练 bounding boxes、862K 测试 bounding boxes、4643 个动作、1615 个体型、187 套服装、40 种发型、182 种鞋子、15 个 3D 环境。

## 实验关键数据

### 单帧方法（CameraHMR）

| 训练数据 | 3DPW PA-MPJPE↓ | 3DPW MPJPE↓ | 3DPW PVE↓ | EMDB PA-MPJPE↓ | EMDB MPJPE↓ | RICH PA-MPJPE↓ | RICH MPJPE↓ |
|----------|---------------|-------------|-----------|---------------|-------------|---------------|-------------|
| B1 | 43.2 | 68.0 | 80.7 | 50.0 | 88.7 | 42.1 | 75.2 |
| B2 | 41.1 | 64.8 | 76.3 | 46.5 | 74.6 | 36.8 | 70.8 |
| B1+B2 | 41.0 | 65.2 | 77.7 | 46.4 | 75.5 | 36.4 | 68.0 |

### 视频方法（世界坐标系评估）

| 方法 | 训练数据 | RICH WA-MPJPE↓ | RICH W-MPJPE↓ | EMDB WA-MPJPE↓ | EMDB W-MPJPE↓ | RICH Jitter↓ | RICH Foot-Sliding↓ |
|------|----------|---------------|---------------|----------------|---------------|-------------|-------------------|
| GVHMR | B1 | 87.3 | 140.0 | 112.4 | 284.6 | 13.5 | 2.9 |
| GVHMR | B2 | 75.5 | 120.6 | 113.7 | 284.4 | 12.3 | 2.7 |
| GVHMR | B1+B2 | 75.8 | 121.3 | 109.7 | 273.1 | 11.3 | 2.6 |
| PromptHMR | B1 | 85.7 | 139.4 | 77.6 | 211.1 | 12.7 | 4.0 |
| PromptHMR | B2 | 75.3 | 122.4 | 71.9 | 197.7 | 11.7 | 2.8 |
| PromptHMR | B1+B2 | **72.5** | **116.6** | **70.5** | **193.7** | **10.2** | **2.6** |

### 关键发现

- 单帧方法：B2 单独训练在所有数据集上都显著优于 B1，体型精度提升 20%
- 视频方法：B1+B2 组合最优，仅用合成数据训练超过了使用真实数据的原始 SOTA
- B1 和 B2 互补：B1 包含坐/爬楼梯等 B2 移除的动作，B2 在相机运动和外观多样性上更强
- PromptHMR 在 B1+B2 上训练的 RICH WA-MPJPE 从 85.7 降至 72.5（降低 15.4%）

## 亮点与洞察

- 合成数据"单独"即可超越使用真实数据的 SOTA——这是一个重要的里程碑信号，表明足够好的合成数据可以替代昂贵的真实数据标注
- 鞋子的加入看似微小但影响深远：弥合了 SMPL-X 光脚与真实穿鞋之间的域差距，影响身高估计和地面接触判断
- 通过 Apple Vision Pro 捕捉第一人称相机运动是创新性的数据采集方式
- 分级服装（XS-6XL）配合多样体型是实用但容易被忽视的工程贡献
- UE5 自定义 C++ 插件修复了 Movie Render Pipeline 的相机位姿记录 bug——这种底层工程细节对数据集质量至关重要

## 局限与展望

- 仅支持人-地面交互，不支持人-物体交互（如坐椅子）和人-人交互（如握手）
- 动作在场景中不具有语义一致性（如在厨房跳舞），限制了语义任务的应用
- 不包含儿童、截肢者或体型显著偏离均值的人群
- 缺乏面部动作和音频，无法支持人际交流相关的推理
- 合成数据与真实视频之间仍存在视觉域差距
- 仅考虑平底鞋，高跟鞋需要改变脚部拓扑结构和姿势

## 相关工作与启发

- **vs BEDLAM (B1)**：B2 是 B1 的全面升级版——相机（静态→多样运动）、体型（有限→BMI 18-41）、发型（card-based→strand-based）、鞋子（无→182种）、服装（111套单尺码→187套分级）、场景（5个→15个）
- **vs PDHuman / BEDLAM-CC**：这些工作解决了焦距多样性问题，但未涉及相机运动
- **vs HumanVid / WHAC-A-Mole**：加入了相机运动，但合成数据缺乏真实感或数据量有限
- **vs EgoGen**：重用 BEDLAM 资产用于第一人称视角，但 B2 提供了更丰富的相机运动类型
- 对后续工作的启发：合成数据工程的每一个细节（鞋底厚度、发丝适配头型、动作重定向）都可能影响最终模型性能

## 评分

- 新颖性: ⭐⭐⭐ 在 BEDLAM 基础上的系统性工程改进，无方法创新但工程深度很好
- 实验充分度: ⭐⭐⭐⭐ 在多个标准基准上与多个 SOTA 方法对比，B1 vs B2 vs B1+B2 比较完整
- 写作质量: ⭐⭐⭐⭐ 每个改进维度都有清晰的描述和动机，数据集文档风格但高质量
- 价值: ⭐⭐⭐⭐ 作为社区标准训练数据集的升级版，对 HPS 领域有直接且广泛的影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Homogeneous Dynamics Space for Heterogeneous Humans](../../CVPR2025/human_understanding/homogeneous_dynamics_space_for_heterogeneous_humans.md)
- [\[ICCV 2025\] UDC-VIT: A Real-World Video Dataset for Under-Display Cameras](../../ICCV2025/human_understanding/udc-vit_a_real-world_video_dataset_for_under-display_cameras.md)
- [\[ICCV 2025\] SynFER: Towards Boosting Facial Expression Recognition with Synthetic Data](../../ICCV2025/human_understanding/synfer_towards_boosting_facial_expression_recognition_with_synthetic_data.md)
- [\[CVPR 2025\] Analyzing the Synthetic-to-Real Domain Gap in 3D Hand Pose Estimation](../../CVPR2025/human_understanding/analyzing_the_synthetic-to-real_domain_gap_in_3d_hand_pose_estimation.md)
- [\[ECCV 2024\] ReLoo: Reconstructing Humans Dressed in Loose Garments from Monocular Video in the Wild](../../ECCV2024/human_understanding/reloo_reconstructing_humans_dressed_in_loose_garments_from_monocular_video_in_th.md)

</div>

<!-- RELATED:END -->
