---
title: >-
  [论文解读] Walking Further: Semantic-aware Multimodal Gait Recognition Under Long-Range Conditions
description: >-
  [AAAI 2026][自动驾驶][多模态步态识别] 构建LRGait——首个面向长距离（10-50m）跨距离场景的LiDAR-Camera多模态步态数据集，并提出EMGaitNet端到端框架，通过CLIP语义挖掘（SeMi）、语义引导对齐（SGA）和对称交叉注意力融合（SCAF）模块实现2D-3D跨模态特征融合，在多个基准上达到SOTA。
tags:
  - "AAAI 2026"
  - "自动驾驶"
  - "多模态步态识别"
  - "长距离识别"
  - "LiDAR-Camera融合"
  - "CLIP语义引导"
  - "跨距离检索"
---

# Walking Further: Semantic-aware Multimodal Gait Recognition Under Long-Range Conditions

**会议**: AAAI 2026  
**arXiv**: [2603.14189](https://arxiv.org/abs/2603.14189)  
**代码**: [github.com/O-VIGIA/LRGait](https://github.com/O-VIGIA/LRGait)  
**领域**: 自动驾驶 / 步态识别  
**关键词**: 多模态步态识别, 长距离识别, LiDAR-Camera融合, CLIP语义引导, 跨距离检索

## 一句话总结

构建LRGait——首个面向长距离（10-50m）跨距离场景的LiDAR-Camera多模态步态数据集，并提出EMGaitNet端到端框架，通过CLIP语义挖掘（SeMi）、语义引导对齐（SGA）和对称交叉注意力融合（SCAF）模块实现2D-3D跨模态特征融合，在多个基准上达到SOTA。

## 研究背景与动机

### 问题背景

步态识别是一种非侵入式、难以伪造的生物特征识别技术，在智能监控和远程身份验证等场景有重要应用。近年来在受控环境下取得了较好效果，但在**长距离**和**多模态**条件下仍面临重大挑战。

### 现有数据集的局限

| 数据集 | 传感器 | 最大距离 | 跨距离 | 日夜 |
|-------|--------|---------|-------|------|
| CASIA-B | Camera | 2-4m | ✗ | ✗ |
| SUSTech1K | LiDAR+Camera | 8-12m | ✗ | ✗ |
| FreeGait | LiDAR+Camera | 25m | ✗ | ✗ |
| **LRGait (Ours)** | **LiDAR+Camera** | **10-50m** | **✓** | **✓** |

核心问题：
1. 现有数据集最远仅25m，无法覆盖真实监控场景的长距离需求（如50m）
2. 缺少同一身份在不同距离下的跨距离样本（如50m→10m检索）
3. 大多数方法仅支持单模态，未充分利用LiDAR和RGB的互补优势

### 方法层面的挑战

**模态鸿沟**：RGB图像和LiDAR点云的表征空间差异巨大，直接融合效果差

**预处理损失**：现有方法通常使用深度图（从点云投影）或轮廓图（从RGB提取）作为输入，丢失了精细几何/纹理细节

**长距离退化**：远距离时点云极度稀疏、RGB图像模糊，预处理方法的退化尤为严重

## 方法详解

### 整体框架

EMGaitNet是一个端到端框架，直接处理**原始RGB视频**和**原始点云序列**：

1. **特征提取**：ResNet9提取2D视觉特征，PointGNN提取3D几何特征
2. **SeMi模块**：CLIP语义挖掘，提取身体部位感知的语义线索
3. **SGA模块**：语义引导对齐，利用语义特征桥接2D-3D模态鸿沟
4. **SCAF模块**：对称交叉注意力融合，层级式整合2D-3D特征
5. **ST模块**：时空模块，捕获全局步态动力学

### 关键设计

#### 1. 双流特征提取

**2D分支**：使用OpenGait的轻量ResNet9骨干提取逐帧视觉特征 $F_{i,j}^{2d} \in \mathbb{R}^{h \times w \times d}$。

**3D分支**：采用PointGNN骨干处理原始点云，通过图卷积层逐步捕获局部和全局几何模式：

- 基于特征余弦相似度构建局部邻域图（TopK最相似点）：
$$\mathcal{N}_{P_i^j}(P_i^j[k]) = \underset{u \neq k}{\text{TopK}}(\cos(F_{i,j}^{3d}[k], F_{i,j}^{3d}[u]))$$

- 计算边特征并聚合：
$$F_{i,j}^{3d}[k] = \text{Maxpool}_{u \in \mathcal{N}}(\text{MLP}(e_{k,u}))$$

**设计动机**：PointGNN基于图的表征比PointNet++更适合稀疏点云（长距离场景），通过特征相似度而非纯空间距离构图，缓解了点云稀疏性的影响。

#### 2. CLIP语义挖掘模块（SeMi）

**功能**：利用CLIP提取身体部位级别的语义线索，作为跨模态对齐的中间桥梁。

**核心思路**：
- 构建身体部位提示："A photo of the [PART] of a [X] person"，[PART]从["head", "arms", "torso", "legs", "feet"]中取
- 使用CLIP视觉编码器提取全局视觉嵌入 $v = \text{CLIP}_v(I_i^j)$
- 通过**反转网络（Inversion Net）**将视觉特征 $v$ 映射到文本空间得到 $v^*$，替换[X]占位符
- 将修改后的提示送入CLIP文本编码器，得到身份感知的语义特征 $t^* \in \mathbb{R}^{5 \times d}$

**设计动机**：
- 类级别语义（如"a person's legs"）不够，步态识别需要**实例级别**的细粒度语义
- 反转网络将特定个体的视觉信息注入提示，使语义线索个性化
- 身体部位分解为后续的跨模态对齐提供了自然的中间表征

#### 3. 语义引导对齐模块（SGA）

**功能**：利用SeMi产生的多粒度语义线索作为中间桥梁，对齐RGB和点云特征。

**核心思路**：使用交叉注意力，以2D/3D特征为Query，语义特征 $t^*$ 为Key/Value：

$$\text{CA}(\bar{F}_{i,j}^{2d}, t^*) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

通过残差连接和FFN精炼：

$$\hat{F}_{i,j}^{2d} = \text{LayerNorm}(\tilde{F}_{i,j}^{2d} + \text{FFN}(\tilde{F}_{i,j}^{2d}))$$

对3D特征 $F_{i,j}^{3d}$ 执行相同操作，得到对齐后的 $\hat{F}_{i,j}^{3d}$。

**设计动机**：语义特征作为共享的对齐锚点，使来自不同模态的特征在语义空间中对齐，同时通过注意力机制抑制背景噪声（如LiDAR点云与无关RGB背景区域的错误融合）。

#### 4. 对称交叉注意力融合模块（SCAF）

**功能**：在对齐后的共享空间中，通过对称双流交叉注意力层级式整合互补信息。

**核心思路**：2D和3D特征交替作为Query并关注对方作为Key/Value，实现双向对齐和互信息融合：

$$F_{i,j}^{2d'} = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

其中每个注意力头：
$$\text{head}_k = \text{Softmax}\left(\frac{Q_k K_k^\top}{\sqrt{d_h}}\right) V_k$$

$Q_k$ 来自2D特征，$K_k, V_k$ 来自3D特征（反向亦然）。

**设计动机**：对称结构确保两个模态同等重要，多头设计捕获不同子空间的互补信息。层级式融合逐步精炼跨模态特征。

#### 5. 时空模块（ST）

**功能**：聚合全局步态动力学。

- **时间池化**：对2D特征序列沿时间维度做MaxPool → $F_i^{tp} \in \mathbb{R}^{h \times w \times d}$
- **空间池化**：对3D特征沿点维度做AvgPool → $F_i^{sp} \in \mathbb{R}^{n \times d}$
- **时空交叉注意力融合**：

$$\tilde{F}_i^{sp} = \text{CA}(F_i^{sp}, F_i^{tp}) + F_i^{sp}$$
$$F_i^{fusion} = \text{MLP}(\text{CA}(F_i^{tp}, \tilde{F}_i^{sp}) + F_i^{tp})$$

最后使用水平金字塔池化（HPP）进行部位级匹配。

### 损失函数 / 训练策略

$$\mathcal{L} = \alpha \mathcal{L}_{tri} + \beta \mathcal{L}_{ce}$$

- $\mathcal{L}_{tri}$：三元组损失（$\alpha=1.0$），学习判别性嵌入空间
- $\mathcal{L}_{ce}$：交叉熵损失（$\beta=2.0$），分类监督
- 推理时使用L2距离度量probe和gallery样本的相似度
- Adam优化器，权重衰减0.0005，MultiStepLR在15K和30K epoch衰减
- 每epoch随机采样10帧RGB和对应点云，2×RTX 3090 GPU

## 实验关键数据

### 主实验

**SUSTech1K（Overall Rank-1 准确率）**：

| 方法 | 模态 | NM | BG | CL | OC | Overall R-1 | Overall R-5 |
|------|------|-----|-----|-----|-----|------------|------------|
| GaitBase | sil | 81.3 | 77.3 | 49.6 | 81.4 | 76.0 | 89.1 |
| LidarGait++ | pc | 94.2 | 93.9 | 79.7 | 91.9 | 92.7 | 98.2 |
| LiCAF | depth+sil | 95.8 | 95.7 | 82.7 | 96.6 | 93.9 | 98.8 |
| **EMGaitNet** | **pc+rgb** | **98.2** | **96.4** | **81.7** | **99.6** | **96.0** | **99.0** |

**LRGait（跨距离跨视角 Rank-1，gallery=D-10）**：

| 方法 | 模态 | D-20 | D-30 | D-40 | D-50 | N-20 | N-30 | Overall R-1 |
|------|------|------|------|------|------|------|------|------------|
| GaitBase | sil | 67.9 | 53.9 | 48.5 | 33.8 | 41.6 | 33.4 | 46.8 |
| LiCAF | depth+sil | 74.8 | 71.6 | 60.4 | 65.3 | 42.5 | 27.8 | 59.6 |
| **EMGaitNet** | **pc+rgb** | **88.5** | **82.4** | **80.8** | **74.4** | **38.2** | **31.7** | **68.9** |

**FreeGait**：

| 方法 | R-1 | R-5 | mAP |
|------|-----|-----|-----|
| LidarGait++ | 82.0 | 93.6 | 87.2 |
| HMRNet | 80.8 | 93.6 | 86.5 |
| **EMGaitNet** | **85.2** | **96.8** | **89.0** |

### 消融实验

**各模块贡献（LRGait数据集）**：

| 配置 | Baseline | +SGA | +SGA+SeMi | +SGA+SeMi+ST |
|------|----------|------|----------|-------------|
| Overall R-1 | 52.3 | 58.5 | 64.2 | **68.9** |
| Overall R-5 | 70.2 | 75.9 | 80.7 | **85.8** |

各模块增量贡献：
- SGA：+6.2%（弥合模态鸿沟）
- SeMi：+5.7%（语义引导）
- ST：+4.7%（时空建模）

### 关键发现

1. **端到端优于预处理**：EMGaitNet使用原始pc+rgb，比使用depth+sil的LiCAF在SUSTech1K上高2.1%
2. **长距离优势明显**：在D-50上比第二名高14.0%（74.4% vs 65.3%），证明端到端方法在远距离下的优势
3. **遮挡鲁棒性出色**：在OC条件下达到99.6%准确率，说明多模态先验有效弥补遮挡信息缺失
4. **夜间场景仍具挑战**：所有方法在夜间性能大幅下降，EMGaitNet在N-40仅21.9%
5. **SeMi模块贡献最大**：身体部位级语义线索为跨模态对齐提供了关键的中间表征

## 亮点与洞察

1. **首个50m长距离多模态步态数据集**：填补了长距离步态识别数据的空白，包含跨距离、日夜、多天气条件
2. **CLIP语义引导的跨模态融合**：用反转网络将视觉信息注入文本空间，生成实例级语义锚点，是创新的跨模态对齐策略
3. **端到端处理原始输入**：避免了特征预处理（depth映射、silhouette提取）带来的信息损失，尤其在长距离/夜间场景
4. **PointGNN用于步态识别**：首次将基于图的3D骨干引入步态任务，通过特征相似度而非空间距离构图，适应稀疏点云
5. **LiDAR长距离行人检测基准**：额外注标了4500帧的长距离行人检测标签，具有独立的研究价值

## 局限与展望

1. **日夜域迁移问题严重**：夜间性能大幅下降，需要专门的多模态域自适应方法
2. **CLIP冻结使用**：未探索针对步态数据微调CLIP的可能性
3. **计算开销**：PointGNN的图构建和CLIP编码增加了推理延迟
4. **仅101个身份**：数据集规模相比GREW（26K身份）仍较小
5. **未考虑多人场景**：实际监控中需要处理多人同时出现的情况

## 相关工作与启发

- **与LiCAF的对比**：LiCAF用简单的交叉注意力做非对称融合，EMGaitNet用语义引导的对称融合，体系性更强
- **与SUSTech1K的关系**：LRGait将距离范围从12m扩展到50m，是步态数据集距离跨度的重大突破
- **启发**：CLIP等VLM的语义能力可以有效桥接不同模态，这种策略可推广到其他多模态融合任务

## 评分

- 新颖性: ⭐⭐⭐⭐ — 数据集贡献突出，方法上CLIP引导融合有创意但各模块较常规
- 实验充分度: ⭐⭐⭐⭐⭐ — 三个数据集、详细消融、多种模态和条件的对比
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，数据集描述详尽
- 价值: ⭐⭐⭐⭐ — 数据集和基线为长距离步态识别研究提供了重要基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FoSS: Modeling Long-Range Dependencies and Multimodal Uncertainty in Trajectory Prediction via Fourier–State Space Integration](../../CVPR2026/autonomous_driving/foss_modeling_long_range_dependencies_and_multimodal_uncertainty_in_trajectory_p.md)
- [\[AAAI 2026\] TSBOW: Traffic Surveillance Benchmark for Occluded Vehicles Under Various Weather Conditions](tsbow_traffic_surveillance_benchmark_for_occluded_vehicles_under_various_weather.md)
- [\[CVPR 2026\] TruckDrive: Long-Range Autonomous Highway Driving Dataset](../../CVPR2026/autonomous_driving/truckdrive_long-range_autonomous_highway_driving_dataset.md)
- [\[ICCV 2025\] Self-Supervised Sparse Sensor Fusion for Long Range Perception](../../ICCV2025/autonomous_driving/self-supervised_sparse_sensor_fusion_for_long_range_perception.md)
- [\[AAAI 2026\] Differentiable Semantic Meta-Learning Framework for Long-Tail Motion Forecasting in Autonomous Driving](differentiable_semantic_meta-learning_framework_for_long-tail_motion_forecasting.md)

</div>

<!-- RELATED:END -->
