---
title: >-
  [论文解读] SutureBot: A Precision Framework & Benchmark for Autonomous End-to-End Suturing
description: >-
  [NeurIPS 2025][机器人][手术机器人] 提出SutureBot——首个针对da Vinci手术机器人端到端自主缝合的精度导向基准与目标条件框架，发布1890条高保真演示数据集，通过点标签目标条件将针刺精度提升59%-74%，并系统评估了π0、GR00T N1、OpenVLA-OFT和多任务ACT等SOTA VLA模型。
tags:
  - "NeurIPS 2025"
  - "机器人"
  - "手术机器人"
  - "缝合自主化"
  - "模仿学习"
  - "目标条件控制"
  - "VLA基准"
---

# SutureBot: A Precision Framework & Benchmark for Autonomous End-to-End Suturing

**会议**: NeurIPS 2025  
**arXiv**: [2510.20965](https://arxiv.org/abs/2510.20965)  
**代码**: [Hugging Face数据集](https://huggingface.co/)  
**领域**: 机器人  
**关键词**: 手术机器人, 缝合自主化, 模仿学习, 目标条件控制, VLA基准

## 一句话总结

提出SutureBot——首个针对da Vinci手术机器人端到端自主缝合的精度导向基准与目标条件框架，发布1890条高保真演示数据集，通过点标签目标条件将针刺精度提升59%-74%，并系统评估了π0、GR00T N1、OpenVLA-OFT和多任务ACT等SOTA VLA模型。

## 研究背景与动机

机器人缝合是典型的长时域灵巧操作任务，需要协调针抓取、精确组织穿刺和安全打结。尽管各子任务已有进展，但完整的端到端自主缝合至今未在真实硬件上实现。主要瓶颈包括：

**数据匮乏**：现有公开dVRK缝合数据集总计不到200条轨迹，远不足以支持现代IL架构

**无统一基准**：手术机器人领域缺乏可复现的基准来追踪自主缝合进展

**精度度量缺失**：传统评估仅用粗粒度的任务完成率，缺乏临床意义的精度指标

**VLA模型评估空白**：最新的VLA模型（π0、GR00T N1等）在手术缝合这一高精度任务上的表现未知

已有方法如STAR系统使用专用缝合工具+计算机视觉实现了精确缝合，但依赖定制硬件、缺乏泛化和错误恢复能力。MPC方法在缝合放置上成功但缺乏灵活性。SurgicAI在仿真中实现了50%成功率的端到端缝合，但未在真实硬件上演示。

## 方法详解

### 整体框架

采用层级架构：高层策略（基于Swin Transformer）从视觉观测生成语言指令，低层策略（VLA模型）接收语言指令、相机图像和目标条件，输出机器人动作块。缝合被分解为三个子任务：取针（needle pickup）、缝合穿刺（needle throw）和打结（knot tying）。

### 关键设计

1. **大规模数据集构建**

   使用dVRK Si版本，通过标准远程操作控制台收集1890条演示：
    - 取针：628条（含148条恢复演示）
    - 缝合穿刺：310条（含96条恢复演示）
    - 打结：952条（含210条恢复演示）

   恢复演示受DAgger启发：先用初始策略部署识别常见失败模式，再收集从失败状态恢复到成功完成的额外演示。数据包含同步的视觉（腕部相机640×480@30Hz + 立体内窥镜960×540@30Hz）和运动学数据（6-DoF笛卡尔位姿、下颌角度等）。通过变化机器人关节配置、RCM位置、缝合垫放置、针初始姿态和相机安装来引入多样性。

   设计动机：恢复演示显著增加训练数据多样性，教会策略从次优状态恢复，提高部署鲁棒性。

2. **目标条件表示**

   为实现精度控制的缝合，探索三种目标条件格式：
    - **点标签（Point Labels）**：在内窥镜图像上叠加不透明蓝色像素（插入点）和绿色像素（出针点）
    - **二值掩码（Binary Masks）**：三通道图像，分通道编码插入掩码和出针掩码
    - **距离图（Distance Maps）**：三通道图像，前两通道编码归一化像素偏移向量(dx,dy)指向插入点，第三通道编码强度热力图

   设计动机：点标签直接将目标嵌入任务图像，提供最显式直观的目标表示；掩码和距离图作为独立输入可能增加了模型整合信息的认知负担。

3. **多VLA模型系统评估**

   比较四种低层策略：
    - **π0**：基于预训练VLM + flow-matching动作头
    - **GR00T N1**：类似架构但预训练侧重人形机器人
    - **OpenVLA-OFT**：使用并行解码+L1回归+FiLM条件化
    - **多任务ACT**：非VLA基线，不依赖预训练VLM骨干网络

   ACT和OpenVLA使用L1回归（至少10000步），π0和GR00T使用MSE（更少步数，容易过拟合）。所有训练在DGX A100（8×A100 80GB）上进行。

### 损失函数 / 训练策略

- L1回归训练（ACT、OpenVLA）不少于10000步
- MSE训练（π0、GR00T）步数更少，需提前停止防止过拟合
- 高层策略基于Swin Transformer，训练2000 epoch，最佳验证epoch为282
- 使用FiLM条件化注入语言信息
- 评估时每个任务设定最大时限（取针/穿刺/打结各120s，拉穿60s）

## 实验关键数据

### 主实验

低层策略在缝合基准上的表现（10次完整缝合）：

| 策略 | 取针 | 穿刺 | 拉穿 | 打结 | 插入误差(mm) | 出针误差(mm) | 时间(s) | 端到端 |
|---|---|---|---|---|---|---|---|---|
| ACT | 9/10 | 8/10 | 4/10 | 9/10 | 1.5±0.8 | 2.6±1.2 | 182±58 | **3/10** |
| π0 | 7/10 | 7/10 | 3/10 | 4/10 | 1.9±1.0 | 3.2±2.3 | 348±45 | 0/10 |
| GR00T N1 | 1/10 | 2/10 | 1/10 | 1/10 | 2.3±1.2 | 2.9±0.6 | 388±67 | 0/10 |
| OpenVLA | 0/10 | 0/10 | 0/10 | 0/10 | NA | 2.8 | NA | 0/10 |

### 消融实验

目标条件格式对穿刺精度的影响：

| 策略+目标条件 | 穿刺成功 | 拉穿成功 | 插入误差(mm) | 出针误差(mm) |
|---|---|---|---|---|
| ACT + Point Label | 9/10 | 7/10 | **1.3±0.9** | 2.0±1.3 |
| ACT + Distance Map | 8/10 | 8/10 | 2.6±1.5 | 2.2±1.8 |
| ACT + Mask | 10/10 | 4/10 | 2.9±1.7 | 3.0±1.0 |
| ACT (no goal) | 10/10 | 9/10 | 3.2±2.2 | 3.6±1.8 |
| π0 + Point Label | 6/10 | 2/10 | **1.0±1.3** | 2.4±1.6 |
| π0 (no goal) | 8/10 | 3/10 | 3.9±2.5 | 3.7±2.5 |

泛化能力测试：

| 场景 | 取针 | 穿刺 | 拉穿 | 打结 | 插入误差 |
|---|---|---|---|---|---|
| ACT 训练创口(1) | 9/10 | 8/10 | 4/10 | 9/10 | 1.5±0.8 |
| ACT 未见创口(2-6) | 5/10 | 6/10 | 2/10 | 5/10 | 1.2±0.8 |
| ACT 不同光照 | 3/10 | 5/10 | 7/10 | 4/10 | 2.2±0.9 |
| π0 训练创口(1) | 7/10 | 7/10 | 3/10 | 4/10 | 1.9±1.0 |
| π0 未见创口(2-6) | 5/10 | 6/10 | 0/10 | 8/10 | 2.0±1.5 |

### 关键发现

1. ACT在所有子任务的完成率上整体最优，是唯一实现端到端成功的策略（3/10）
2. 点标签目标条件带来59%-74%的精度提升，ACT最低插入误差1.3mm，π0最低1.0mm
3. 使用点标签的模型在接近目标时展现出刻意的、犹豫的调整动作，暗示更好的空间感知
4. π0预训练对dVRK的帮助有限——从头训练的π0性能仅略低于从检查点微调
5. 高层策略F1达0.92，任务转换检测100%准确，不是系统瓶颈

## 亮点与洞察

- 首个真实硬件上的端到端自主缝合基准和数据集
- 紫外线标记法精确测量穿刺精度，超越了传统的粗粒度任务完成评估
- 恢复演示的DAgger式策略是一个实用的工程洞察
- ACT这种"小而专"的模型在小规模均质数据上反而优于大型VLA，值得深思

## 局限与展望

- 每个实验仅10次试验，统计效力有限
- 当前依赖人工选择穿刺目标点，完全自主需要自动化目标选取
- 不同材料（如真实组织、模拟血液）上的泛化能力未验证
- 临床相关指标（咬合深度、组织创伤、缝合张力）未纳入评估
- 仅ACT实现了端到端成功，距离临床部署差距巨大

## 相关工作与启发

- 延续了SRT-H的层级手术学习框架，但首次针对缝合这一高灵巧任务
- 与SurgicAI的仿真缝合工作互补，将问题推进到真实硬件
- 启发方向：VLA模型在手术场景的预训练可能需要特定于双臂操作的数据

## 评分

- **新颖性**: ⭐⭐⭐⭐ 基准和数据集贡献显著，目标条件方法相对简单但有效
- **实验充分度**: ⭐⭐⭐⭐⭐ 多模型对比、消融、泛化测试、统计分析非常全面
- **写作质量**: ⭐⭐⭐⭐ 组织清晰，度量定义严谨
- **价值**: ⭐⭐⭐⭐⭐ 填补了手术机器人领域端到端自主缝合基准的空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](autovla_a_vision-language-action_model_for_end-to-end_autonomous_driving_with_ad.md)
- [\[CVPR 2025\] TinyNav: End-to-End TinyML for Real-Time Autonomous Navigation on Microcontrollers](../../CVPR2025/robotics/tinynav_end-to-end_tinyml_for_real-time_autonomous_navigation_on_microcontroller.md)
- [\[CVPR 2025\] Reasoning in Visual Navigation of End-to-end Trained Agents: A Dynamical Systems Approach](../../CVPR2025/robotics/reasoning_in_visual_navigation_of_end-to-end_trained_agents_a_dynamical_systems_.md)
- [\[CVPR 2026\] RoboTAG: End-to-end Robot Pose Estimation via Topological Alignment Graph](../../CVPR2026/robotics/robotag_end-to-end_robot_pose_estimation_via_topological_alignment_graph.md)
- [\[CVPR 2026\] End-to-End Language-Action Model for Humanoid Whole Body Control](../../CVPR2026/robotics/end-to-end_language-action_model_for_humanoid_whole_body_control.md)

</div>

<!-- RELATED:END -->
