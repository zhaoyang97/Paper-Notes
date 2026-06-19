---
title: >-
  [论文解读] HUMAPS-4D: A Multimodal Dataset for HUman Motion Analysis with Physiological and Semantic informations
description: >-
  [CVPR 2026][人体理解][人体运动分析] HUMAPS-4D 是一个把"光学动捕 + 多视角 RGB + IMU + 足底压力鞋垫 + 表面肌电(sEMG) + 人体测量 + 三层语义标注"全部时间同步在同一标准协议下的大规模人体运动数据集（32 人 × 30 动作 × 10 次 × 14 小时 = 576 万帧），目标是让"不依赖摄像头、靠足底压力等生理信号推断全身 3D 姿态/动作"成为可严谨 benchmark 的研究方向。
tags:
  - "CVPR 2026"
  - "人体理解"
  - "人体运动分析"
  - "足底压力"
  - "可穿戴传感"
  - "隐私保护"
  - "多模态数据集"
---

# HUMAPS-4D: A Multimodal Dataset for HUman Motion Analysis with Physiological and Semantic informations

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Dabrowski_HUMAPS-4D_A_Multimodal_Dataset_for_HUman_Motion_Analysis_with_Physiological_CVPR_2026_paper.html)  
**代码**: 数据集主页 https://humaps4d.wp.imt.fr/（受控授权访问）  
**领域**: 人体理解 / 多模态数据集  
**关键词**: 人体运动分析、足底压力、可穿戴传感、隐私保护、多模态数据集

## 一句话总结
HUMAPS-4D 是一个把"光学动捕 + 多视角 RGB + IMU + 足底压力鞋垫 + 表面肌电(sEMG) + 人体测量 + 三层语义标注"全部时间同步在同一标准协议下的大规模人体运动数据集（32 人 × 30 动作 × 10 次 × 14 小时 = 576 万帧），目标是让"不依赖摄像头、靠足底压力等生理信号推断全身 3D 姿态/动作"成为可严谨 benchmark 的研究方向。

## 研究背景与动机
**领域现状**：人体运动理解目前几乎全靠视频数据驱动。视觉方法是"数据中心"的——堆大规模、多视角、带文本和动作分段的视频去学通用表征；而生物力学方法是"个体/病理中心"的——用动捕、足底压力、sEMG 在少量受试者上做精细测量。两条线各自发展，互不交汇。

**现有痛点**：① 视觉传感侵入隐私、部署笨重昂贵，难以日常落地；② 智能手表/手环这类轻量可穿戴只能给粗粒度信号，无法推断全身 3D 运动；③ 已有的"压力/姿态"数据集（SolePoser、P2P-Insoles、MMVP 等）规模小、受试者少、动作单一，往往缺 RGB、缺语义标注、缺人体测量，有些还用估计姿态而非真值。**最关键的是：没有任何一个数据集把视觉、文本、生物力学、人体测量四类信号统一在一个标准协议下**，导致"足底压力推姿态"这类隐私友好方向无法被严谨地横向评测。

**核心矛盾**：生物力学领域长期卡在"测量精度 ↔ 数据规模/多样性"的 trade-off 上——要精确就只能少人少动作，要规模就只能上视频但牺牲隐私和生理信号。

**本文目标**：造一个同时具备规模、传感多样性和语义深度的数据集，把"用足底等生理信号、少依赖甚至不依赖摄像头来做全身 3D 运动分析"这件事变成有 baseline、有评测协议、可复现的标准任务。

**切入角度**：作者押注**仪器化鞋垫（足底压力感知）**——它天然融入日常（不分年龄/职业/穿着），用户接受度高，又能提供高分辨率的足-地交互动力学，能作为训练 3D 姿态模型的"物理约束"。

**核心 idea**：把低层生物力学信号（足底压力、sEMG、IMU）与高层语义描述（临床式运动评估、原子动作描述、时序动作分段）**配对**进同一份时间同步数据，从而支持"被物理约束 + 语义约束共同驱动"的生成/推断模型，同时摆脱对可识别身份的视觉数据的依赖。

## 方法详解
这是一篇数据集/benchmark 论文，"方法"= 数据采集系统 + 标注体系 + 两族基准任务，而非一个网络。整体可概括为：在同一根触发信号下把六类传感器严格时间同步采集 → 配上三层自然语言标注 → 在其上定义两族基准任务并给出 baseline。由于是数据集构建而非多阶段推理 pipeline，这里不强行画框架图，用文字 + 公式讲清。

### 整体框架
采集端：所有设备通过有线连到一台跑 Qualisys 软件的中央电脑。每次录制开始时由鞋垫软件向 sEMG 和 Qualisys 系统发一个数字触发，**同时**启动所有设备；再发一个触发同时停止——以此保证硬件级时间同步。每位受试者录制前都会标定相机、校准鞋垫与 sEMG，并按其人体测量生成定制 3D 人体模型，最后还有人工质检。

六类模态（全部时间同步、统一对齐到一个有重力方向的米制参考系）：
- **动捕（MoCap）**：11 台 Miqus M3 相机（3 RGB+IR、8 IR），120 Hz，标定精度 0.22 mm，给出 42 个标记点 3D 关节 + 26 个推断关节及四元数。
- **RGB**：3 台同步相机，120 Hz，720p，提供内外参，可做 3D 场景重建与姿态点投影。
- **sEMG**：Delsys Trigno 无线系统，每人 16 电极、1259 Hz，每个电极还内嵌 148 Hz 的 IMU。
- **足底压力（IPS）**：Moticon OpenGo 仪器化鞋垫，每只 16 个压力传感器 + 1 个 IMU，100 Hz，含压力、加速度、角加速度、总力、压心。
- **人体测量**：年龄、性别、身高、体重、下肢各段长度、足长等。
- **三层语义标注**（见下）。

规模：32 名 18–42 岁健康受试者，每次 30 个动作（postural 静态保持 / locomotion 移动 / dynamic 重心转移 / interaction 与物体交互），每次约 2 分 40 秒、共 10 次、随机化动作顺序，合计 320 个 2'40" 实例、14 小时、576 万帧。

### 关键设计

**1. 硬件级时间同步的多模态统一采集：把视觉与生物力学装进同一标准协议**

针对"没有数据集能把四类信号统一对齐"这个核心空白，作者用一根数字触发信号把鞋垫、sEMG、Qualisys（动捕+RGB）在采集瞬间同时开关，所有模态被对齐到同一个有重力方向的米制坐标系。为了让不同受试者/条件可比，三类信号都做了显式归一化：3D 关节按骨盆中心化并用个体人体测量缩放，$\mathbf{p}_i^{\text{norm}} = \mathbf{S}_{\text{participant}}(\mathbf{p}_i - \mathbf{p}_{\text{pelvis}})$，其中 $\mathbf{S}_{\text{participant}}$ 是按受试者尺寸构造的对角缩放矩阵；sEMG 按参考收缩归一化 $\text{EMG}^{\text{norm}}_{j,m}(t) = \text{EMG}_{j,m}(t) / \overline{\text{EMG}}^{\text{ref}}_{j,m}$，消除电极位置/皮肤厚度等个体差异；足底压力按每步最大值或体重归一化 $P^{\text{norm}}_{j,s}(t) = P_{j,s}(t)/\max_t P_{j,s}(t)$ 或 $P_{j,s}(t)/W_j$，让压力模式独立于绝对负荷/年龄/鞋具。正是这种"同步 + 个体归一化"使得跨模态、跨受试者的联合学习有了可比的基础——这是它区别于 SolePoser/P2P-Insoles 等单流数据集的根本。

**2. 三层配对语义标注：把生理信号与临床级语义绑在一起**

低层生理信号本身不携带"这段动作意味着什么"的高层语义，作者因此给每次录制配上时间对齐（同时索引到时间戳和帧号）的三层自然语言语料：① **临床式运动评估叙述**——理疗师视角对姿态、协调性、代偿策略、平衡的高层判断，逼近专家评估；② **原子动作描述**——短的粗粒度文本，支撑检索/字幕等语言落地；③ **时序动作分段**——帧级标注每个动作的起止边界，产出结构化事件序列。这三层不是为单一 benchmark 优化，而是作为通用语义资源支持浏览/检索/挖掘，并打开"语言条件生成模型直接从足底压力等生理信号产出专家式运动评估报告"这一独特方向——这是别的数据集普遍缺失的（对照见下表"Semantic"列）。

**3. 两族隐私保护基准任务 + baseline：把"靠鞋垫做事"定义成可评测的标准问题**

作者用两族基础任务把数据集的价值落地，且都强调"推理时只用鞋垫、训练时可借其他模态做约束"：

- **基于鞋垫的动作识别**：给定单只鞋垫的足底压力序列 $\mathbf{P}_{1:T}$，逐帧识别 30 类动作之一，学映射 $f:\mathbf{P}_{1:T}\to\{a_1,\dots,a_{30}\}^T$；推理仅用 $\mathbf{P}_{1:T}$，训练时可引入 RGB/MoCap/sEMG 做生物力学与语义约束。用 3 秒滑窗（对应 MoCap 360 / 鞋垫 300 / sEMG 3760 个样本），采 leave-one-subject-out（LOSO）协议评估跨人泛化。
- **足底压力 3D 姿态推断**：给定鞋垫压力 $\mathbf{P}_{1:T}$ 和含加速度/总力/压心的副向量 $\mathbf{A}_{1:T}$，推断 16 个关节（不含手臂）的 3D 坐标 $f:\mathbf{P}_{1:T},\mathbf{A}_{1:T}\to\mathbf{J}_{1:T}$；baseline 受 SolePoser 启发，训练期多一个姿态编码器，用 MPJPE/MPJAE 作 loss、对鞋垫数据用 MSE 重建 loss、再加一个"Feature Loss"约束两个编码器产出一致的隐表征；验证期不喂姿态，确保评的是"纯靠足底压力推姿态"。采 4 折交叉验证（每折 8 人验证、24 人训练）。

作者还为姿态任务提出自定义指标 **Inconsistency（不一致度）**，衡量推断关节相对真值在时间上的相对位置漂移：$IS = \frac{1}{n_J}\sum_{i=0}^{n_J\cdot 3 - 1}\sigma_{1:T}(\bar{J}_{i_{1:T}} - J_{i_{1:T}})$，其中 $\sigma$ 是该量在时间序列上的标准差。值越高表示推断动作越缺乏连贯性（关节相对位置在运动中频繁变化）。⚠️ 公式细节以原文为准。

## 实验关键数据

### 数据集对比
HUMAPS-4D 在规模、动作多样性、模态完整度与语义/人体测量标注上同时领先两类既有数据集（视觉驱动 3D 姿态类 + 生物力学评估类），是唯一同时具备 sEMG、IMU、动捕、多 RGB、语义、人体测量的资源。

| 数据集 | 受试者 | 动作类 | 帧数 | sEMG | 多 RGB | 语义层 | 人体测量 |
|--------|--------|--------|------|------|--------|--------|----------|
| SP-Sport (SolePoser) | 28 | 4 | 606 k | 无 | 无 | 无 | 无 |
| P2P-Insole | 4 | 5 | 14 k | 无 | 无 | 无 | 无 |
| MMVP | 16 | 6 | 44 k | 无 | 1 | 无 | 无 |
| SIAT-LLMD | 40 | 16 | — | 16 | 无 | 无 | ✓ |
| **HUMAPS-4D** | **32** | **30** | **5.76 M** | **16** | **3** | **3 层** | **✓** |

### 基准任务结果
动作识别（LOSO，准确率↑）显示：鞋垫单模态整体 82.71%，明显弱于 MoCap（89.45%），但融合两者最高（90.33%）；dynamic（跳跃）和 interaction（上肢主导）对纯足底压力最难——压力信号长时间缺失或与脚关系弱；融合主要在 static/locomotion/postural 上带来稳定增益。姿态推断（4 折）则普遍误差较大，印证"仅靠足底信号重建全身 3D 姿态"本身的内在困难。

| 任务 | 指标 | 鞋垫 | MoCap | 鞋垫+MoCap |
|------|------|------|-------|-----------|
| 动作识别·整体 | 准确率↑ | 82.71% | 89.45% | **90.33%** |
| 动作识别·Dynamic | 准确率↑ | 74.36% | 92.05% | 87.57% |
| 动作识别·Interaction | 准确率↑ | 73.84% | 84.83% | 85.71% |
| 姿态推断·整体 | MPJPE↓(cm) | 31.1 | — | — |
| 姿态推断·整体 | Inconsistency↓(cm) | 7.5 | — | — |
| 姿态推断·整体 | MPJAE↓(°) | 13.3 | — | — |

### 关键发现
- **多模态训练是为了推理时少用视觉**：训练引入 MoCap/RGB/sEMG 做约束，推理时只用鞋垫——融合模型整体最优，证明"不依赖侵入式视觉也能提升动作识别"是可行路线。
- **动作类型决定难度**：dynamic（跳跃时压力近乎缺失）和 interaction（上肢运动足底感知不到）对纯压力信号最难，MoCap 介入后大幅改善。
- **姿态推断仍是开放难题**：即便有 MoCap 约束引导，MPJPE 仍达 31.1 cm，远未到可用，暴露了从稀疏足底信号重建全身姿态的本质局限——这正是数据集想推动社区去攻克的方向。

## 亮点与洞察
- **"配对"是真正的差异点**：别的数据集要么有视觉要么有生物力学，HUMAPS-4D 把低层生理信号和高层语义/临床描述配对，使"语言条件 + 物理约束"的生成模型成为可能，这是数据组织层面的巧思而非单纯堆传感器。
- **隐私优先的设计哲学**：所有视频做人脸模糊、授权访问、推理只靠鞋垫——把"隐私合规"从事后补丁变成数据集的第一性约束，契合现实部署的监管趋势。
- **自定义 Inconsistency 指标可迁移**：用"推断关节相对真值的时间方差"来度量运动连贯性，对任何"从稀疏/弱信号重建时序结构"的任务（如稀疏 IMU 重建、低帧率插值）都能借鉴。
- **硬件触发同步的工程价值**：六类异频传感器（100/120/148/1259 Hz）靠单触发对齐到统一米制坐标系，是这类多模态采集可复现的关键工程经验。

## 局限与展望
- 受试者均为 18–42 岁健康人群，缺乏老年、病理、康复人群，临床/康复落地仍需扩展。
- 姿态推断 baseline 误差很大（MPJPE 31 cm 级），且明确不推断手臂，说明纯足底信号上限有限；数据集本身把这点作为公开挑战留给社区。
- 数据采集在实验室式可控环境（虽强调 unscripted/natural），与真正户外自由生活仍有 gap。
- ⚠️ 论文正文只给了任务族的高层概述，更细的实现、评测指标和 baseline 结果放在附录，部分表格读数（如各模态编码细节）以原文/附录为准。

## 相关工作与启发
- **vs SolePoser / P2P-Insoles**: 它们是首批用单对鞋垫（压力+IMU）实时回归 3D 关节的方法，但数据/代码不公开（SolePoser）或只靠 MoCap 信号无语义上下文（P2P-Insoles）；HUMAPS-4D 不是方法而是数据集，且补齐了 RGB、sEMG、人体测量、三层语义，并把任务标准化。
- **vs IMU-based（DIP / Sparse Inertial Poser / Mollyn 等）**: IMU 路线面临漂移、传感器贴身体不同部位干扰用户；本文转向足底鞋垫这一更"日常无感"的载体，并提供配套真值与基准。
- **vs 地毯/压力垫（GroundLink / Intelligent Carpet）**: 那些受环境与场地限制；鞋垫天然可移动、可日常穿戴，泛化潜力更大。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个把视觉/文本/生物力学/人体测量统一标准协议、并配三层语义的数据集，组织思路新但属"集成式"创新。
- 实验充分度: ⭐⭐⭐⭐ 两族任务 + LOSO/4 折协议 + baseline 齐全，但姿态任务效果远未可用、细节多在附录。
- 写作质量: ⭐⭐⭐⭐ 动机与对照表清晰，归一化公式给得明确；部分基准只给高层概述略显单薄。
- 价值: ⭐⭐⭐⭐⭐ 为"隐私友好、非视觉的人体运动分析"提供了稀缺的大规模标准基座，配 2026 公开挑战，社区价值高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] RoMo: A Large-Scale, Richly Organized Dataset and Semantic Taxonomy for Human Motion Generation](romo_a_large-scale_richly_organized_dataset_and_semantic_taxonomy_for_human_moti.md)
- [\[CVPR 2026\] M4Human: A Large-Scale Multimodal mmWave Radar Benchmark for Human Mesh Reconstruction](m4human_a_large-scale_multimodal_mmwave_radar_benchmark_for_human_mesh_reconstru.md)
- [\[ICCV 2025\] HUMOTO: A 4D Dataset of Mocap Human Object Interactions](../../ICCV2025/human_understanding/humoto_a_4d_dataset_of_mocap_human_object_interactions.md)
- [\[CVPR 2026\] Real-Time Multimodal Fingertip Contact Detection via Depth and Motion Fusion for Vision-Based Human-Computer Interaction](real-time_multimodal_fingertip_contact_detection_via_depth_and_motion_fusion_for.md)
- [\[CVPR 2026\] FusionAgent: A Multimodal Agent with Dynamic Model Selection for Human Recognition](fusionagent_a_multimodal_agent_with_dynamic_model_selection_for_human_recognitio.md)

</div>

<!-- RELATED:END -->
