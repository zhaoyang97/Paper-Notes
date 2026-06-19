---
title: >-
  [论文解读] Mesh-RFT: Enhancing Mesh Generation via Fine-Grained Reinforcement Fine-Tuning
description: >-
  [NeurIPS 2025 Spotlight][3D视觉][网格生成] 提出 Mesh-RFT 框架，通过拓扑感知评分系统和掩码直接偏好优化（M-DPO）实现面级别的细粒度网格质量优化，显著提升生成网格的几何完整性和拓扑规则性。 高质量 3D 网格生成面临两大挑战： 现有方法的局限： 自回归网格生成方法（MeshGPT、M…
tags:
  - "NeurIPS 2025 Spotlight"
  - "3D视觉"
  - "网格生成"
  - "强化微调"
  - "DPO"
  - "拓扑感知"
  - "细粒度优化"
---

# Mesh-RFT: Enhancing Mesh Generation via Fine-Grained Reinforcement Fine-Tuning

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2505.16761](https://arxiv.org/abs/2505.16761)  
**代码**: [项目主页](https://hitcslj.github.io/mesh-rft/)  
**领域**: 3D Vision / Mesh Generation  
**关键词**: 网格生成, 强化微调, DPO, 拓扑感知, 细粒度优化

## 一句话总结

提出 Mesh-RFT 框架，通过拓扑感知评分系统和掩码直接偏好优化（M-DPO）实现面级别的细粒度网格质量优化，显著提升生成网格的几何完整性和拓扑规则性。

## 研究背景与动机

高质量 3D 网格生成面临两大挑战：

**现有方法的局限**: 自回归网格生成方法（MeshGPT、MeshXL 等）在长序列高分辨率网格生成时容易产生结构歧义和"幻觉"（不一致边、非流形顶点、变形、孔洞）

**全局强化学习的不足**: DeepMesh 用 DPO 做偏好对齐，但依赖人工标注偏好对（仅 5000 样本），且使用全局奖励信号无法捕捉局部拓扑变化。关键观察：高质量和低质量结构经常共存于同一网格中

核心问题：如何实现面级别的细粒度优化，而非对整个网格施加统一的全局奖励？

## 方法详解

### 整体框架

三阶段流水线：
1. **预训练**: 使用 Hourglass AutoRegressive Transformer + Shape Encoder 进行监督学习
2. **偏好数据集构建**: 预训练模型生成候选网格，拓扑感知评分系统建立偏好对
3. **后训练**: 使用 Masked DPO 进行细粒度强化微调

### 关键设计

1. **拓扑感知评分系统**: 提出两个客观拓扑指标，替代人工标注：

    - **边界边比率（BER）**: $BER(\mathcal{M}) = E_{\partial\mathcal{M}} / E_{\mathcal{M}}$，衡量网格完整性。封闭流形网格的 BER 应为 0，高 BER 表示表面不连续、孔洞等问题
    - **拓扑得分（TS）**: $TS(\mathcal{M}) = \sum_{i=1}^{4} w_i s_i(\mathcal{Q}(\mathcal{M}))$，通过将三角网格转换为四边形后评估：四边形比率(0.4)、角度质量(0.2)、纵横比(0.3)、邻接一致性(0.1)
    - 加上 **Hausdorff 距离（HD）** 衡量几何一致性

2. **偏好数据集构建**: 每个输入点云生成 8 个候选网格，穷举 C(8,2)=28 对组合。当且仅当 BER、TS、HD 三项指标全面优于时才建立偏好关系，避免模糊偏好。

3. **掩码直接偏好优化（M-DPO）**: 核心创新——对每个三角面进行质量评估，构建 token 级二值掩码 $\phi(\mathcal{M}) \in \{0,1\}^{|\mathcal{M}|}$。对正样本用掩码放大好区域的贡献，对负样本用反掩码聚焦差区域的惩罚：

    - $\mathcal{L}^+$: 正样本的好区域 token 被掩码选中
    - $\mathcal{L}^-$: 负样本的差区域 token 被反掩码选中
   这实现了保留好区域、专注修复差区域的效果。

### 损失函数 / 训练策略

- 预训练：截断训练（固定长度段）+ 滑动窗口推理（覆盖 40% 窗口后开始滑动，保留最新 30%）
- M-DPO 损失：$\mathcal{L}_{M-DPO} = -\mathbb{E}[\log \sigma(\beta \mathcal{L}^+ - \beta \mathcal{L}^-)]$
- 模型架构：Hourglass Transformer（含 2 次 shorten 和 2 次 upsample），Hunyuan3D 2.0 的点云编码器通过交叉注意力注入
- 预训练：256×H20 GPU，10 天；M-DPO：64 GPU，8 小时，学习率 5e-7
- 训练数据：2M 网格预训练，800K 过滤后微调，10K 网格构建偏好数据

## 实验关键数据

### 主实验

| 方法 | CD↓ | HD↓ | TS↑ | BER↓ | US↑ |
|------|-----|-----|-----|------|-----|
| MeshAnythingV2 | 0.2265 | 0.4760 | 72.0 | 0.0913 | 8% |
| BPT | 0.1615 | 0.3347 | 73.7 | 0.0113 | 18% |
| DeepMesh* (0.5B) | 0.1760 | 0.3570 | 75.8 | 0.0044 | 20% |
| **Mesh-RFT** | **0.1286** | **0.2411** | **79.4** | **0.0015** | **40%** |

（Dense Meshes 数据，用户偏好 US 从 20% 提升到 40%）

### 消融实验

| 配置 | CD↓ | HD↓ | TS↑ | BER↓ | US↑ |
|------|-----|-----|-----|------|-----|
| Pretrain | 0.1588 | 0.3196 | 76.5 | 0.0033 | 30% |
| N-DPO (仅HD) | 0.1455 | 0.2919 | 75.7 | 0.0028 | 32% |
| S-DPO (评分系统) | 0.1348 | 0.2625 | 77.9 | 0.0023 | 35% |
| **M-DPO (掩码)** | **0.1286** | **0.2411** | **79.4** | **0.0015** | **40%** |

### 关键发现

- M-DPO 相比预训练模型：HD 降低 24.6%，TS 提升 3.8%
- M-DPO 相比全局 DPO（S-DPO）：HD 降低 17.4%，TS 提升 4.9%
- 仅用 HD 做偏好判断（N-DPO）反而导致 TS 下降，说明多指标综合评分的必要性
- 用户研究中 M-DPO 获得 40% 偏好（vs 预训练 30%），验证了感知质量提升
- 在 OOD 的 Hunyuan2.5 生成网格上也表现良好，展示了泛化能力

## 亮点与洞察

- **首个面级粒度的 RL 优化方法**：打破了全局奖励的局限，对局部缺陷精准修复
- **客观拓扑评分系统**替代人工标注，可扩展性强（vs DeepMesh 的 5000 标注样本）
- BER 和 TS 的设计巧妙：基于四边形转换评估三角网格质量，契合工业应用对四边形网格的偏好
- 截断训练+滑动窗口推理的工程设计解决了长序列网格生成的实际问题

## 局限与展望

- 仅测试了点云条件生成，未探索文本/图像条件
- DeepMesh 仅比较了 0.5B 版本，可能不公平
- 评分系统的权重 $w_i$ 是人工设定的，可考虑自适应学习
- 四边形质量评估作为三角网格质量的代理指标，可能存在偏差
- 生成网格的面数和分辨率受限于序列长度

## 相关工作与启发

- 与 DeepMesh 的对比：DeepMesh 用全局 DPO + 人工标注，Mesh-RFT 用局部 M-DPO + 自动评分
- 从 NLP 的 DPO/RLHF 到 3D 网格的迁移是趋势，但需要针对 3D 结构特性做适配
- Masked DPO 的思路可推广到其他序列生成中局部质量差异大的场景（如代码生成、音乐生成）
- Hourglass Transformer 的层次结构设计对长序列生成有参考价值

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首创面级RL优化+客观拓扑评分，M-DPO 设计精巧
- **实验充分度**: ⭐⭐⭐⭐ 消融完整，有用户研究和OOD测试，但baseline数量有限
- **写作质量**: ⭐⭐⭐⭐ 图表丰富，方法描述清晰
- **价值**: ⭐⭐⭐⭐⭐ 对生产级网格生成有直接应用价值，客观评分系统可复用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Eval3D: Interpretable and Fine-grained Evaluation for 3D Generation](../../CVPR2025/3d_vision/eval3d_interpretable_and_fine-grained_evaluation_for_3d_generation.md)
- [\[NeurIPS 2025\] On Geometry-Enhanced Parameter-Efficient Fine-Tuning for 3D Scene Segmentation](on_geometry-enhanced_parameter-efficient_fine-tuning_for_3d_scene_segmentation.md)
- [\[NeurIPS 2025\] MiCADangelo: Fine-Grained Reconstruction of Constrained CAD Models from 3D Scans](micadangelo_fine-grained_reconstruction_of_constrained_cad_models_from_3d_scans.md)
- [\[ICCV 2025\] DeepMesh: Auto-Regressive Artist-Mesh Creation with Reinforcement Learning](../../ICCV2025/3d_vision/deepmesh_auto-regressive_artist-mesh_creation_with_reinforcement_learning.md)
- [\[CVPR 2026\] Animator-Centric Skeleton Generation on Objects with Fine-Grained Details](../../CVPR2026/3d_vision/animator-centric_skeleton_generation_on_objects_with_fine-grained_details.md)

</div>

<!-- RELATED:END -->
