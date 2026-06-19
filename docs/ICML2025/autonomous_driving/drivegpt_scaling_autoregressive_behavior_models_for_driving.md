---
title: >-
  [论文解读] DriveGPT: Scaling Autoregressive Behavior Models for Driving
description: >-
  [ICML 2025][自动驾驶][自回归模型] 提出 DriveGPT，一个 1.4B 参数的自回归 Transformer 驾驶行为模型，在 1.2 亿真实驾驶片段上训练（比现有最大数据集多 50x），首次系统建立驾驶行为建模的数据/模型/计算缩放定律，验证数据是性能瓶颈，在规划和 WOMD 预测任务上超越 SOTA。
tags:
  - "ICML 2025"
  - "自动驾驶"
  - "自回归模型"
  - "缩放定律"
  - "Transformer"
  - "轨迹预测"
---

# DriveGPT: Scaling Autoregressive Behavior Models for Driving

**会议**: ICML 2025  
**arXiv**: [2412.14415](https://arxiv.org/abs/2412.14415)  
**代码**: 无（未开源）  
**领域**: 自动驾驶 / 行为建模  
**关键词**: 自动驾驶, 自回归模型, 缩放定律, Transformer, 轨迹预测

## 一句话总结

提出 DriveGPT，一个 1.4B 参数的自回归 Transformer 驾驶行为模型，在 1.2 亿真实驾驶片段上训练（比现有最大数据集多 50x），首次系统建立驾驶行为建模的数据/模型/计算缩放定律，验证数据是性能瓶颈，在规划和 WOMD 预测任务上超越 SOTA。

## 研究背景与动机

**Transformer 缩放的成功**：在 NLP（GPT 系列）、语音、时序预测等领域，通过扩大模型参数和训练数据量可持续提升性能，这一趋势被 Kaplan et al. (2020) 和 Hoffmann et al. (2022) 的缩放定律精确刻画。

**驾驶行为建模的特殊挑战**：将缩放定律迁移到驾驶领域面临三大难题——(1) 输入涉及多模态（智能体轨迹 + 地图信息），不同于纯文本；(2) 需要空间推理和物理运动学理解；(3) 大规模驾驶数据采集成本极高。现有行为模型受限于数据规模（最大 GUMP 仅 2.6M 片段/523M 参数），缩放潜力未被充分探索。

**本文要解决的问题**：能否通过大幅扩大数据（50x）和模型规模（3x），在驾驶行为建模中观察到类似 NLP 的持续性能提升？数据和模型哪个是更关键的瓶颈？核心 idea：**用 LLM 式自回归解码器建模驾驶轨迹（将每步动作视为 token），在工业级规模数据上验证缩放规律**。

## 方法详解

### 整体框架

DriveGPT 采用标准 encoder-decoder 架构：**Transformer 编码器**融合多模态场景信息（目标智能体历史、周围智能体历史、地图向量）为场景嵌入 $\mathbf{c} \in \mathbb{R}^{n \times d}$；**LLM 式 Transformer 解码器**以自回归方式逐步预测未来位置的离散动作分布，每步条件化于编码器嵌入和已预测的历史状态。推理时通过采样多条轨迹 + K-Means 子采样产生多模态预测。

### 关键设计

1. **Verlet 动作离散化**:
    - 功能：将连续轨迹空间转化为离散动作 token 序列
    - 核心思路：定义 Verlet 动作 $a_t$ 为位置的二阶差分，即 $s_{t+1} = s_t + (s_t - s_{t-1}) + a_t$，其中 $(s_t - s_{t-1})$ 项隐含匀速假设。将连续动作空间离散化为有限集合，转化为分类问题
    - 设计动机：Verlet 表示天然编码加速度信息，产生物理上平滑的轨迹；离散化使得可以用标准交叉熵损失训练，与 LLM 范式完全对齐

2. **多模态场景编码器**:
    - 功能：将异构输入（智能体轨迹 + 地图多段线）统一编码为场景嵌入
    - 核心思路：所有输入标准化到以目标智能体为中心的坐标系，每个向量通过 PointNet-like 编码器映射为 token 嵌入，最后用自注意力 Transformer 融合所有上下文
    - 设计动机：向量化表示（VectorNet 风格）高效且可与 Transformer 架构自然结合；agent-centric 视图消除了绝对坐标的影响

3. **大规模数据集构建与缩放实验设计**:
    - 功能：从百万英里真实驾驶数据中筛选 1.2 亿高质量片段，覆盖美国/日本/阿联酋多城市
    - 核心思路：数据平衡昼夜、地理区域，涵盖变道/交叉路口/双排停车/施工区/行人自行车交互等场景；模型规模从 1.5M 到 1.4B（3 个数量级），每个规模搜索最优学习率
    - 设计动机：此前工作受限于小规模数据无法得到统计显著的缩放结论，本文要在前所未有的范围内验证缩放趋势

### 训练策略

- **Teacher forcing** 训练：用 ground truth 未来位置作为解码器输入，允许所有步并行预测
- **单次交叉熵损失**：目标动作选为与 ground truth 最近的离散动作
- 每个模型大小训练单个 epoch（与 LLM 缩放文献一致）
- 最优学习率随模型增大而减小：1.5M→0.005，1.4B→0.0001（余弦衰减）

## 实验关键数据

### 主实验：数据缩放（26M 参数模型）

| 训练数据量 | mADE ↓ | mFDE ↓ | Miss Rate ↓ | Offroad ↓ | Collision ↓ |
|-----------|--------|--------|-------------|-----------|-------------|
| 2.2M（WOMD级） | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 21M | 0.561 | 0.496 | 0.420 | 0.326 | 0.269 |
| 85M | 0.496 | 0.441 | 0.332 | 0.238 | 0.217 |
| 120M | **0.489** | **0.433** | **0.317** | **0.198** | **0.196** |

数据从 2.2M→120M：mFDE 降低 56.7%，Offroad 降低 80.2%，Collision 降低 80.4%。

### 模型缩放（120M 数据）

| 模型参数 | mADE ↓ | mFDE ↓ | Miss Rate ↓ | Offroad ↓ | Collision ↓ |
|---------|--------|--------|-------------|-----------|-------------|
| 8M | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 26M | 0.954 | 0.950 | 0.902 | 0.858 | 0.915 |
| 94M | 0.937 | 0.925 | 0.866 | 0.815 | 0.890 |
| 163M | 0.943 | 0.925 | 0.875 | 0.815 | **0.817** |

模型缩放收益弱于数据缩放——在 120M 数据下，模型增大到 ~94M 参数后趋于饱和。

### 缩放定律

| 缩放维度 | 拟合公式 | $R^2$ |
|---------|---------|-------|
| 数据缩放 | $\log(L) = -0.102 \log(D) + 2.663$ | 0.986 |

预测：再提升 10% 需增加 350M 样本，提升 20% 需增加 1.4B 样本。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 自回归 vs 一次性解码器 | AR 在 >8M 参数后更优 | minFDE 持续改善 vs 饱和 |
| 计算预算固定 | 小模型+更多数据 > 大模型+少数据 | 数据是瓶颈 |
| 注意力头数/隐藏维度变化 | 无显著差异 | 主要由总参数量决定 |

### 关键发现

- 数据缩放是驾驶行为建模的主要瓶颈，模型缩放在数据不足时收益有限
- 数据量 >21M 后模型缩放才变得有效（21M 以下不同模型大小性能几乎无差异）
- 自回归解码器比一次性解码器具有更好的缩放性（更大模型仍能受益）
- 闭环评估中，大数据训练的 DriveGPT 能处理行人横穿、双排停车等边缘场景

## 亮点与洞察

- **首次在驾驶行为建模中进行工业级缩放研究**：1.4B 参数 / 1.2 亿片段，比现有工作大 1-2 个数量级
- 明确回答了"数据 vs 模型"的关键问题：数据是瓶颈，与 NLP 缩放文献一致
- 缩放定律的拟合质量高（$R^2 = 0.986$），为未来资源分配提供量化指导
- 闭环部署验证了缩放提升在实际驾驶中的转化价值（安全变道、复杂交互）

## 局限与展望

- 仅使用轨迹 + 地图的向量化输入，未融合视觉信息（相机/LiDAR 原始感知数据）
- 模型未开源，缩放实验难以复现
- 模型缩放在 ~94M 参数后趋于饱和，可能需要更大数据量才能解锁更大模型的潜力
- Verlet 动作离散化可能引入量化误差，限制精细运动预测
- 闭环评估在仿真环境中，真实道路部署的安全验证不足
- 缩放定律的外推范围有限（仅覆盖到 120M 数据/1.4B 参数）

## 相关工作与启发

- **Kaplan et al. (2020)**：NLP 缩放定律的开创性工作，DriveGPT 直接对标其方法论
- **Hoffmann et al. (2022) Chinchilla**：计算最优缩放，DriveGPT 的计算预算分析类似
- **Seff et al. (2023) MotionLM**：将运动预测建模为语言建模，DriveGPT 扩展了规模
- **GUMP (523M)**：此前最大行为模型，DriveGPT 参数量是其 2.7x、数据量是其 46x
- **启发**：驾驶基础模型的数据缩放之路刚刚开始，多模态融合的缩放值得探索

## 评分

- 新颖性: ⭐⭐⭐ 架构层面新意有限（标准 encoder-decoder），核心贡献在于规模和实证
- 实验充分度: ⭐⭐⭐⭐⭐ 三个维度的缩放实验 + 消融 + 定性分析 + 闭环评估，非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，缩放分析系统性强，图表直观
- 价值: ⭐⭐⭐⭐ 为驾驶行为建模的缩放研究提供了重要参考，但工业壁垒限制了社区影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] PAR: Poly-Autoregressive Prediction for Modeling Interactions](../../CVPR2025/autonomous_driving/poly-autoregressive_prediction_for_modeling_interactions.md)
- [\[ICCV 2025\] Epona: Autoregressive Diffusion World Model for Autonomous Driving](../../ICCV2025/autonomous_driving/epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)
- [\[NeurIPS 2025\] Flow Matching-Based Autonomous Driving Planning with Advanced Interactive Behavior Modeling](../../NeurIPS2025/autonomous_driving/flow_matching-based_autonomous_driving_planning_with_advanced_interactive_behavi.md)
- [\[CVPR 2025\] Cubify Anything: Scaling Indoor 3D Object Detection](../../CVPR2025/autonomous_driving/cubify_anything_scaling_indoor_3d_object_detection.md)
- [\[ECCV 2024\] Accelerating Online Mapping and Behavior Prediction via Direct BEV Feature Attention](../../ECCV2024/autonomous_driving/accelerating_online_mapping_and_behavior_prediction_via_dire.md)

</div>

<!-- RELATED:END -->
