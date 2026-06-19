---
title: >-
  [论文解读] GauDP: Reinventing Multi-Agent Collaboration through Gaussian-Image Synergy in Diffusion Policies
description: >-
  [NeurIPS 2025][多智能体][3D Gaussian Splatting] 提出 GauDP，通过从多智能体的去中心化 RGB 观测中构建全局一致的 3D 高斯场，并将高斯属性动态分配回各智能体的局部视角，实现可扩展的、感知增强的多智能体协作模仿学习。 多智能体具身协作（如工业装配、手术机器人、家庭辅助）中…
tags:
  - "NeurIPS 2025"
  - "多智能体"
  - "3D Gaussian Splatting"
  - "多智能体协作"
  - "扩散策略"
  - "模仿学习"
  - "机器人操作"
---

# GauDP: Reinventing Multi-Agent Collaboration through Gaussian-Image Synergy in Diffusion Policies

**会议**: NeurIPS 2025  
**arXiv**: [2511.00998](https://arxiv.org/abs/2511.00998)  
**代码**: [有](https://ziyeeee.github.io/gaudp.io/)  
**领域**: 3D视觉  
**关键词**: 3D Gaussian Splatting, 多智能体协作, 扩散策略, 模仿学习, 机器人操作

## 一句话总结

提出 GauDP，通过从多智能体的去中心化 RGB 观测中构建全局一致的 3D 高斯场，并将高斯属性动态分配回各智能体的局部视角，实现可扩展的、感知增强的多智能体协作模仿学习。

## 研究背景与动机

多智能体具身协作（如工业装配、手术机器人、家庭辅助）中，每个智能体需要在完成自己任务的同时与其他智能体保持同步。现有方法面临两个核心困境：

**仅用局部观测**：将所有智能体的局部视图拼接输入共享策略，但无法捕获联合协作状态，导致执行不同步（如一个机械臂在另一个还没打开锅盖时就尝试放食物）

**仅用全局观测**：提供一致的场景表示，但缺少高分辨率的智能体个体信息，导致精细控制（抓取、放置）性能下降

简单融合全局和局部信号缺乏 3D 结构约束，难以推理空间关系。因此需要一种统一表示，**同时编码全局一致性和局部精度**。

## 方法详解

### 整体框架

GauDP 的核心流程分为四步：

1. **局部上下文提取**：每个智能体从自身 2D 观测中提取局部特征
2. **全局 3D 高斯场构建**：从所有视图构建共享的 3D 高斯场作为全局上下文
3. **全局上下文分配与融合**：将全局上下文与局部上下文融合后通过编码器处理
4. **动作预测**：扩散策略通过交叉注意力处理融合后的每智能体特征来预测动作

问题形式化：给定多视角同步观测 $\mathcal{O} = \{\mathcal{I}_1, \dots, \mathcal{I}_N\}$，预测未来动作序列 $\mathbf{a} = \{a_1, \dots, a_L\}$。条件策略定义为 $\pi_\Phi(\mathbf{a} | \mathcal{O}) := \pi_\Phi(\mathbf{a} | \mathcal{O}, \mathcal{G})$，其中 $\mathcal{G} = \mathcal{F}(\mathcal{O})$ 是从观测到高斯的映射。

### 关键设计

#### 1. 全局上下文重建（Global Context Reconstruction）

目标：从多视角 2D RGB 构建统一的视角无关 3D 表示。传统 3DGS 需要密集视角+精确位姿+逐场景多分钟优化，不适合具身场景的快速适应需求。

解决方案：采用 **NoPoSplat**（前馈网络），直接从稀疏无位姿视图重建 3D 高斯表示，并在机器人操作场景上微调。具体流程：

- 每张 RGB 图像通过**共享权重 ViT 编码器**独立编码
- **跨视图 ViT 解码器**通过跨注意力层融合不同视角信息
- **高斯参数预测头**为每个像素估计 3D 高斯：$\mathcal{G}_i = \mathcal{F}(\mathbf{x}_i)$，$\mathcal{G}_i \in \mathbb{R}^{C_\mathcal{G} \times H \times W}$

引入额外**深度监督**：将每个高斯投影到相机坐标系后渲染深度图 $\hat{D}$，与真实深度 $D$ 计算损失。重建质量提升显著（PSNR 17.9→23.4）。

**关键点**：深度和位姿仅在微调阶段使用，**部署时只需 RGB 输入**。

#### 2. 全局上下文分配与像素级协同

将全局上下文全部送给每个智能体会引入无关信息。本文提出**选择性分发机制**：

- 利用重建过程中高斯与源像素的自然对齐关系
- 每个智能体仅接收与自身视图关联的高斯子集（已通过跨注意力整合了其他视图信息）
- 将选择的高斯变换回与原始图像空间维度匹配的 2D 网格
- 与局部图像特征拼接后通过**轻量卷积融合模块**完成像素级融合

这种设计确保每个智能体获得有针对性的全局表示，同时保持空间一致性。

#### 3. 坐标系选择

消融实验表明，使用各智能体的**局部相机坐标系**优于统一世界坐标系——保留了智能体中心的空间关系，避免了跨视角的对齐误差。

### 损失函数 / 训练策略

- **重建损失**（微调 NoPoSplat 阶段）：$\mathcal{L}_{rec} = \mathcal{L}_{rgb} + \alpha \cdot \mathcal{L}_{depth}$
- **扩散策略损失**（策略训练阶段）：标准 DDPM 去噪损失
- 训练配置：动作预测水平 8，观测步数 3，动作执行步数 6；DDPM 100 步；Adam，$lr=10^{-4}$，warm-up + cosine decay；100 epochs，batch size 32，单卡 A800

## 实验关键数据

### 主实验

基于 **RoboFactory** benchmark，含 6 个 2-4 臂协作操作任务：

| 方法 | Lift Barrier | Place Food | Stack Cube | Align Camera | Stack Cube(4) | Take Photo | **平均** |
|------|:-----------:|:----------:|:----------:|:------------:|:-------------:|:----------:|:-------:|
| DP3(XYZ+RGB) | 31% | 25% | 1% | 18% | 0% | 11% | 14.33% |
| 3D Dense Policy | 28% | 18% | 0% | 0% | 0% | 7% | 8.83% |
| DP | 9% | 12% | 6% | 3% | 0% | 0% | 5.00% |
| **GauDP** | **72%** | **15%** | **2%** | **26%** | **0%** | **3%** | **19.67%** |

3D 重建质量（2 视图重建）：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|:-----:|:-----:|:------:|
| Pretrain(NoPoSplat) | 17.918 | 0.580 | 0.492 |
| **微调后** | **23.424** | **0.779** | **0.148** |

### 消融实验

| 配置 | Lift Barrier | Place Food | Stack Cube | Align Camera | 平均 |
|------|:-----------:|:----------:|:----------:|:------------:|:----:|
| 统一世界坐标系 | 30% | 1% | 8% | 26% | 10.83% |
| 粗粒度融合(w/o prefuse) | 2% | 4% | 0% | 1% | 1.17% |
| 仅高斯(w/o Image) | 32% | 7% | 0% | 28% | 11.17% |
| 仅图像(w/o Gaussian) | 9% | 12% | 6% | 3% | 5.00% |
| **完整模型** | **72%** | **15%** | **2%** | **26%** | **19.67%** |

真实机器人实验：GauDP 在 Card Box Stacking/Handover/Grab Roller 上分别达到 17/30、19/30、27/30，均优于 DP baseline。

### 关键发现

1. GauDP 仅用 RGB 输入即超越所有基线，平均成功率 19.67%（最高）
2. 在 Lift Barrier 上达到 72%，大幅领先第二名 DP3 的 31%
3. 移除像素级融合后性能暴跌至 1.17%，证明精细融合策略至关重要
4. 图像和高斯缺一不可：图像提供外观线索，高斯提供全局结构

## 亮点与洞察

- **优雅的设计哲学**：用 3DGS 作为桥梁统一局部精度和全局一致性，不需要额外传感模态
- **天然可扩展**：高斯表示的灵活性使智能体数量增加时无需架构变更
- **自监督重建**：利用训练扩散策略的相同多视角数据进行 3DGS 微调，无需额外数据
- **推理时无需位姿/深度**：部署阶段仅需 RGB 输入

## 局限与展望

1. Stack Cube 等高精度任务成功率仍很低（2%），精细操作仍有较大提升空间
2. 3-4 臂设置下整体成功率偏低，高复杂度协作仍具挑战
3. 训练耗时略长（6.5 vs 4.8/2.5 GPU hours），推理速度略低（1.28 vs 1.49 FPS）
4. 未探索与 VLA 模型结合，以及高斯在动态场景世界模型中的应用

## 相关工作与启发

- **NoPoSplat**：核心依赖组件，从稀疏无位姿视图进行前馈 3DGS 重建
- **3D Diffusion Policy (DP3)**：使用点云输入的扩散策略，主要对比方法
- **RoboFactory**：多臂协作操作 benchmark，通过自动化数据收集构建
- 启发：3DGS 作为多视角信息聚合的中间表示，可推广到更多多智能体感知-决策任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 3DGS 与扩散策略结合用于多智能体场景是新颖的
- 实验充分度: ⭐⭐⭐⭐ 消融全面，包含仿真+真实机器人实验
- 写作质量: ⭐⭐⭐⭐ 框架设计清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐ 为多智能体协作的视觉表示开辟了新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] GETReason: Enhancing Image Context Extraction through Hierarchical Multi-Agent Reasoning](../../ACL2025/multi_agent/getreason_enhancing_image_context_extraction_through_hierarchical_multi-agent_re.md)
- [\[NeurIPS 2025\] 3D-Agent: Tri-Modal Multi-Agent Collaboration for Scalable 3D Object Annotation](3d-agenttri-modal_multi-agent_collaboration_for_scalable_3d_object_annotation.md)
- [\[NeurIPS 2025\] Multi-Agent Collaboration via Evolving Orchestration](multi-agent_collaboration_via_evolving_orchestration.md)
- [\[ICML 2026\] Searching for Synergy in Shared Workspace Human-AI Collaboration](../../ICML2026/multi_agent/searching_for_synergy_in_shared_workspace_human-ai_collaboration.md)
- [\[ICML 2026\] RADAR: Redundancy-Aware Diffusion for Multi-Agent Communication Structure Generation](../../ICML2026/multi_agent/radar_redundancy-aware_diffusion_for_multi-agent_communication_structure_generat.md)

</div>

<!-- RELATED:END -->
