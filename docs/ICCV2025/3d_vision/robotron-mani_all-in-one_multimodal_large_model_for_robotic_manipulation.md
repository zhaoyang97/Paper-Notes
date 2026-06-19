---
title: >-
  [论文解读] RoboTron-Mani: All-in-One Multimodal Large Model for Robotic Manipulation
description: >-
  [ICCV 2025][3D视觉][机器人操作] 提出多模态机器人操作大模型 RoboTron-Mani 和综合数据集 RoboData，通过 3D 感知增强（UVFormer + 占据监督）与模态隔离掩码（MIM）实现多数据集联合训练，首次作为通才策略在多个数据集上同时超越专家模型。 当前将大模型应用于机器人操作领域面临两…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "机器人操作"
  - "多模态大模型"
  - "3D感知"
  - "跨embodiment泛化"
  - "数据对齐"
---

# RoboTron-Mani: All-in-One Multimodal Large Model for Robotic Manipulation

**会议**: ICCV 2025  
**arXiv**: [2412.07215](https://arxiv.org/abs/2412.07215)  
**代码**: [GitHub](https://github.com/EmbodiedAI-RoboTron/RoboTron-Mani)  
**领域**: 3D视觉  
**关键词**: 机器人操作, 多模态大模型, 3D感知, 跨embodiment泛化, 数据对齐

## 一句话总结

提出多模态机器人操作大模型 RoboTron-Mani 和综合数据集 RoboData，通过 3D 感知增强（UVFormer + 占据监督）与模态隔离掩码（MIM）实现多数据集联合训练，首次作为通才策略在多个数据集上同时超越专家模型。

## 研究背景与动机

当前将大模型应用于机器人操作领域面临两大核心挑战：

1. **2D 到 3D 的鸿沟**：现有多模态大模型（如 LLaVA、Flamingo）主要聚焦于 2D 图像理解，但机器人需要与物理 3D 空间交互。直接将 2D 多模态模型用于具身智能并非最优解——机器人需要理解空间深度、遮挡关系和物体的 3D 几何信息才能精确操作。

2. **数据收集成本高昂**：RT-1 收集约 13 万个 episode 就花了 17 个月。而现有跨平台数据集（如 Open X-Embodiment）虽然整合了多个数据集，但缺少关键的 3D 信息（多视角图像、相机参数、深度图），且不同数据集的坐标系和动作空间不统一——直接融合反而导致性能下降（如 RT-1-X 弱于 RT-1）。

这两个问题相互关联：要让通才模型在异构数据上有效学习，既需要统一的 3D 输入表示（消除不同相机参数带来的 2D 特征差异），也需要对齐的输出空间（统一不同机器人的动作表示）。

## 方法详解

### 整体框架

RoboTron-Mani 基于 OpenFlamingo 架构，接收多视角图像 $I$、文本指令 $T$ 和相机参数 $Cam$ 作为输入，输出动作 $O_A$，以及可选的图像 $O_I$ 和占据图 $O_O$。整体由四个核心组件串联：Vision Encoder → 3D Perception Adapter → Feature Fusion Decoder → Multimodal Decoders。

### 关键设计

1. **3D Perception Adapter (UVFormer)**：解决多视角特征统一和 3D 空间感知问题。利用 UVFormer 将 $H$ 个时间步、$N$ 个视角的图像特征 $X^h$ 和对应的相机参数 $Cam^h$ 转换为统一视图表示：

$$U_I^h = \text{UVFormer}(Q, X^h, Cam^h)$$

其中 $Q = \{Pos, Emb\}$ 是可学习的查询，$Pos \in \mathbb{R}^{L \times B \times 3P}$ 定义了机器人操作空间内 3D 网格的位置。这一设计的关键优势在于：无论相机参数如何变化，同一 3D 场景的统一视图表示 $U_I^h$ 保持一致，从而实现输入空间对齐。

2. **模态隔离掩码 (Modality-Isolation-Mask, MIM)**：在 Feature Fusion Decoder 的自注意力层中引入 KQ 掩码，控制不同模态 token（文本、图像、动作、占据）之间的注意力连接。深色方块表示允许注意力交互，白色方块禁止。MIM 的核心价值是实现灵活的模态融合——训练时可使用辅助模态监督（图像重建、占据预测），推理时可按需省略不必要的模态输出，显著提升了模态组合的灵活性。

3. **多模态解码器**：设计三种不同的解码器以适配不同模态输出：
    - **图像解码器**：2 层注意力解码器，输出图像 patch 后拼装为完整图像（静态视图或手腕视图）
    - **占据解码器**：先生成特征 $U_{occ}^h$，再 reshape + 上采样 + 3D 卷积重建完整 3D 占据 $O_o^h = \{o_{pos}^h, o_{rgb}^h\}$
    - **动作解码器**：MLP 或 DiT 块输出 delta 6D 位姿 $a_{pose}^h$ 和 1-DoF 夹爪动作 $a_g^h$

4. **RoboData 数据对齐**：整合 CALVIN、Meta-World、LIBERO、RT-1 等 10 个数据集，进行三维度对齐：
    - **3D 空间对齐**：统一世界坐标系为 X→右、Y→前、Z→上，工作空间限制在 $[-0.5, -0.5, 0]$ 到 $[0.5, 0.5, 1]$
    - **动作表示对齐**：统一使用复合旋转矩阵方法 (CRMM) 重新生成动作
    - **缺失数据补全**：重建原始仿真环境，补充缺失的相机内外参

### 损失函数 / 训练策略

综合损失函数：

$$l = l_a + \lambda_{\text{image}}(l_{simg} + l_{gimg}) + \lambda_{\text{occ}} l_o$$

- **动作损失** $l_a$：位姿用 MSE，夹爪用 BCE
- **图像损失** $l_{simg}, l_{gimg}$：预测下一帧与真值的 L2 损失
- **占据损失** $l_o$：位置 MSE + RGB MSE（$\lambda_{rgb}$ 调节权重）

训练细节：4B 参数（bf16），32×A100 训练约 50 小时，2.1M 样本，10 epochs。

## 实验关键数据

### 主实验

| 数据集 | 指标 | RoboTron-Mani | 之前 SOTA | 提升 |
|--------|------|---------------|-----------|------|
| LIBERO | 成功率 | **91.7%** | QueST 89.8% | +1.9% |
| RoboCasa | 成功率 | **47.4%** | GR00T-N1 40.9% | +6.5% |
| CALVIN | Avg Len | **3.51** | MDT 93.7%(Task1) | 竞争力 |
| Meta-World | 成功率 | **80.1%** | PRISE 80.4% | 持平 |
| RT-1 | 成功率 | **60.0%** | RT-2-X(55B) 60.7% | 持平(参数量远小) |

注：RoboTron-Mani 是唯一在所有 5 个数据集上同时评估的通才策略，其余均为针对单一数据集优化的专家模型。

### 消融实验

| 配置 | Task1 | Task2 | Task3 | Task4 | Task5 | Avg Len | 说明 |
|------|-------|-------|-------|-------|-------|---------|------|
| Baseline | 81.0% | 48.1% | 25.7% | 14.5% | 8.6% | 1.77 | 仅最后帧输出动作 |
| +FFA | 85.0% | 63.3% | 42.0% | 28.7% | 18.8% | 2.37 | 逐帧动作输出 |
| +FFA+Image | 88.5% | 74.7% | 60.7% | 49.1% | 39.6% | 3.13 | 加图像预测 |
| +FFA+UVFormer | 94.2% | 74.7% | 55.1% | 38.3% | 25.8% | 2.88 | 3D感知 |
| +All(MLP) | 94.7% | 80.3% | 65.1% | 51.4% | 39.0% | 3.31 | 完整框架 |
| +All(DiT) | **96.9%** | **83.0%** | **68.1%** | **56.5%** | **46.8%** | **3.51** | DiT动作头 |

### 关键发现

- UVFormer 对首个任务提升显著（81% → 94.2%），说明 3D 感知对任务启动至关重要
- 即使生成的图像和占据质量不理想，辅助模态监督仍能显著提升动作性能
- 数据对齐是跨数据集训练的关键：未对齐时 LIBERO 仅 64.2%，对齐后 90.7%
- DiT 动作头相比 MLP 在长序列任务上优势明显（Avg Len: 3.31 → 3.51）

## 亮点与洞察

- **首次实现通才策略全面超越专家模型**：在 5 个异构数据集上联合训练和评估，打破了"通才不如专家"的常规认知
- **3D 感知是跨 embodiment 泛化的关键**：同一 3D 场景在不同相机参数下的 2D 特征不同，但 UVFormer 的 3D 特征保持一致
- **MIM 设计巧妙**：允许训练时用辅助模态监督增强学习，推理时灵活裁剪，是一种免费的性能提升手段
- **数据工程的深度投入**：花费数百人天对齐数据，补充缺失模态，这种工程投入被充分验证有价值

## 局限与展望

- 当前仅在仿真环境中验证数据对齐方案，真实世界的异构数据对齐更加困难
- 4B 参数模型的训练成本仍然较高（32×A100，50小时）
- 辅助模态生成质量较低，若能提升生成质量可能带来更大性能增益
- 未探索实时在线学习和自适应能力

## 相关工作与启发

- 与 Open X-Embodiment 的对比说明，简单数据融合不如精细的空间对齐 + 架构设计
- OpenFlamingo 的交叉注意力机制自然适配多帧/视频输入，优于 LLaVA 的自回归机制
- RoboData 的对齐方法论可推广到更多机器人数据集的统一

## 评分

- **新颖性**: ⭐⭐⭐⭐ 3D 感知 + 模态隔离掩码的设计新颖，数据对齐方案系统性强
- **实验充分度**: ⭐⭐⭐⭐⭐ 5 个数据集全面评估，消融实验详尽
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，公式规范，但部分章节较冗长
- **价值**: ⭐⭐⭐⭐⭐ 为跨 embodiment 机器人学习提供了完整的数据 + 模型解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Matrix3D: Large Photogrammetry Model All-in-One](../../CVPR2025/3d_vision/matrix3d_large_photogrammetry_model_all-in-one.md)
- [\[ICCV 2025\] RoboPearls: Editable Video Simulation for Robot Manipulation](robopearls_editable_video_simulation_for_robot_manipulation.md)
- [\[ICCV 2025\] RayZer: A Self-supervised Large View Synthesis Model](rayzer_a_self-supervised_large_view_synthesis_model.md)
- [\[CVPR 2025\] Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh](../../CVPR2025/3d_vision/mani-gs_gaussian_splatting_manipulation_with_triangular_mesh.md)
- [\[CVPR 2026\] ConsisVLA-4D: Advancing Spatiotemporal Consistency in Efficient 3D-Perception and 4D-Reasoning for Robotic Manipulation](../../CVPR2026/3d_vision/consisvla-4d_advancing_spatiotemporal_consistency_in_efficient_3d-perception_and.md)

</div>

<!-- RELATED:END -->
