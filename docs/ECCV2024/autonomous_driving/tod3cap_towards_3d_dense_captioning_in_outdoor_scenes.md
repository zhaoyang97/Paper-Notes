---
title: >-
  [论文解读] TOD³Cap: Towards 3D Dense Captioning in Outdoor Scenes
description: >-
  [ECCV 2024][自动驾驶][3D dense captioning] 首次提出户外 3D 密集描述任务，构建百万级 TOD3Cap 数据集（850 场景 2.3M 描述），设计基于 BEV 特征 + Relation Q-Former + LLaMA-Adapter 的端到端网络，超越适配后的室内方法 +9.6 CIDEr@0.5IoU。
tags:
  - "ECCV 2024"
  - "自动驾驶"
  - "3D dense captioning"
  - "outdoor scenes"
  - "BEV"
  - "Relation Q-Former"
  - "LLaMA-Adapter"
---

# TOD³Cap: Towards 3D Dense Captioning in Outdoor Scenes

**会议**: ECCV 2024  
**arXiv**: [2403.19589](https://arxiv.org/abs/2403.19589)  
**代码**: [https://github.com/jxbbb/TOD3Cap](https://github.com/jxbbb/TOD3Cap)  
**领域**: 自动驾驶 / 3D视觉-语言  
**关键词**: 3D dense captioning, outdoor scenes, BEV, Relation Q-Former, LLaMA-Adapter

## 一句话总结
首次提出户外 3D 密集描述任务，构建百万级 TOD3Cap 数据集（850 场景 2.3M 描述），设计基于 BEV 特征 + Relation Q-Former + LLaMA-Adapter 的端到端网络，超越适配后的室内方法 +9.6 CIDEr@0.5IoU。

## 研究背景与动机
**领域现状**：3D 密集描述（dense captioning）在室内场景已取得显著进展，如 Scan2Cap、Vote2Cap-DETR 等，但这些方法专注于室内，户外场景尚未被探索。

**现有痛点**：室内和户外场景存在根本性域差异：
   - 户外物体是动态的（有速度、运动状态），室内是静态的
   - 户外使用稀疏 LiDAR 点云（且稀疏度空间不均匀），室内用稠密扫描
   - 户外相机固定在6个方向（自遮挡严重），室内可自由移动
   - 户外场景面积大得多

**核心矛盾**：室内方法无法直接适配户外（检测器失效、缺乏时序建模、不支持多模态融合）；同时缺乏户外 box-caption 对的标注数据。

**切入角度**：(a) 设计适配户外的 BEV 表示 + 时序融合的检测-描述管线；(b) 构建大规模户外密集描述数据集。

**核心 idea**：BEV 统一表示 + Relation Q-Former 建模关系 + LLaMA-Adapter 生成描述，无需重训 LLM。

## 方法详解

### 整体框架
TOD3Cap 网络分三个阶段：(1) BEV 检测器从 LiDAR 点云和多视角图像提取统一 BEV 特征并生成物体提议；(2) Relation Q-Former 捕获物体间关系和场景上下文；(3) 通过 LLaMA-Adapter 将物体特征转化为 LLM 的提示（prompt），冻结 LLM 生成密集描述。

### 关键设计

#### 1. **BEV 检测器 (BEV-based Detector)**
    - 功能：融合多视角图像和 LiDAR 点云到统一 BEV 空间，生成物体提议
    - 核心思路：
      - **图像分支**：可学习 BEV 查询 $Q_c \in \mathbb{R}^{H_b \times W_b \times C}$，通过空间交叉注意力聚合多视角图像特征：$F_c = \text{Spatial-Cross-Attention}(Q_c, \text{Backbone}(I))$
      - **时序融合**：BEV 查询与前一时刻 BEV 特征 $F_c^p$ 通过时序自注意力交互：$Q_c' = \text{Temporal-Self-Attention}(Q_c, F_c^p)$，用于建模物体运动
      - **LiDAR 分支**：体素化 → 骨干网络 → 高度维展平得到 $F_l \in \mathbb{R}^{H_b \times W_b \times C}$
      - **融合**：卷积融合模块合并两个模态的 BEV 特征得到 $F_b$
      - **提议生成**：DETR 风格的查询式检测头生成 $K$ 个物体提议 $\hat{B} = \{\hat{B}_i\}_{i=1}^K \in \mathbb{R}^{K \times D}$
    - 设计动机：BEV 表示已在户外 3D 检测中证明高效（BEVFormer、BEVFusion）；时序融合对建模户外动态场景至关重要

#### 2. **Relation Q-Former**
    - 功能：提取每个物体的上下文感知特征，建模物体间关系
    - 核心思路：
      - 物体提议 $\hat{B}$ 通过可学习 MLP 编码为与 $F_b$ 相同维度的特征
      - 拼接物体特征和 BEV 特征，送入由多层自注意力构成的 Relation Q-Former 进行特征交互
      - $Q_B = \text{Relation Q-Former}(\text{MLP}(\hat{B}), F_b)$
    - 设计动机：户外密集描述需要理解物体间的相对位置关系（如"这辆车在白色卡车旁边"），简单的关系图或 Transformer 解码器无法利用 BEV 全局上下文信息

#### 3. **LLaMA-Adapter 描述解码器 (Captioning Decoder)**
    - 功能：将物体查询特征转化为自然语言描述
    - 核心思路：
      - MLP 对齐维度：$Q_B' = \text{MLP}(Q_B)$
      - Adapter 对齐模态：$\mathcal{V} = \text{Adapter}(Q_B')$，将物体特征转为 LLM 可理解的视觉提示
      - 冻结 LLM 生成描述：$\hat{\mathcal{C}} = \text{LLM}(\mathcal{T}, \mathcal{V})$，$\mathcal{T}$ 为系统提示
      - 描述损失：$\mathcal{L}_{cap} = -\sum_{i=1}^M \log \hat{p}(w_i | w_{[1:i-1]}, \mathcal{T}, \mathcal{V}, \theta_{\text{LLM}})$
    - 设计动机：冻结 LLM 避免灾难性遗忘，利用大模型预训练的常识推理能力；Adapter 桥接 BEV 特征与语言特征的模态鸿沟

### 损失函数 / 训练策略
- 总损失：$\mathcal{L} = \alpha \mathcal{L}_{obj} + \beta \mathcal{L}_{cap}$，$\alpha=10, \beta=1$
- $\mathcal{L}_{obj}$：L1 回归损失监督 3D 边界框
- 三阶段训练：(1) 预训练 BEV 检测器（24 epochs, lr=2e-4）；(2) 冻结检测器训练描述生成（10 epochs, lr=2e-4）；(3) 全模型微调（10 epochs, lr=2e-5）
- 训练时使用 Hungarian 匹配筛选 + 随机采样子集（减少显存和优化难度），推理时使用 NMS

### TOD3Cap 数据集
- 基于 nuScenes 的 850 个场景、34.1K 帧
- 四维度描述：外观（Appearance, 69.7%词汇占比）、运动（Motion, 2.6%）、环境（Environment, 7.1%）、关系（Relationship, 20.6%），关系部分平均词数最多（11.2词）
- 半自动标注流程：3D Box → 2D 投影裁剪 → LLaMA-Adapter 初始描述 → 人工校正 → GPT-4 总结 → 三人验证
- 总计 2.3M 描述，10 名标注员工作约 2000 小时

## 实验关键数据

### 主实验（2D+3D 输入）

| 方法 | C@0.25 | B-4@0.25 | C@0.5 | B-4@0.5 |
|------|--------|----------|-------|---------|
| Scan2Cap* | 60.6 | 41.5 | 62.5 | 39.2 |
| X-Trans2Cap* | 99.8 | 45.9 | 92.2 | 43.3 |
| Vote2Cap-DETR* | 110.1 | 48.0 | 98.4 | 46.1 |
| **TOD3Cap** | **120.3** | **51.5** | **108.0** | **50.2** |

超越 Vote2Cap-DETR +9.6 CIDEr@0.5（+9.76%）。

### 消融实验：关系建模

| 关系模块 | C@0.25 | C@0.5 |
|---------|--------|-------|
| Relational Graph | 88.8 | 82.7 |
| Transformer Decoder | 94.9 | 90.0 |
| **Relation Q-Former** | **96.2** | **94.1** |

### 消融实验：语言解码器

| 解码器 | C@0.25 | C@0.5 |
|--------|--------|-------|
| S&T | 81.2 | 78.6 |
| GPT2 | 89.4 | 85.6 |
| **LLaMA** | **96.2** | **94.1** |

### 消融实验：训练策略

| 检测器预训练 | 描述器预训练 | 全模型微调 | C@0.25 | C@0.5 |
|------------|-----------|---------|--------|-------|
| ✗ | ✓ | ✓ | 74.2 | 69.5 |
| ✓ | ✗ | ✓ | 87.4 | 85.3 |
| ✓ | ✓ | ✓ | **96.2** | **94.1** |

### 模型规模对比

| 配置 | 可训练参数 | 推理时间 | C@0.5 |
|------|----------|---------|-------|
| TOD3Cap-Tiny | 90.5M | 316.1min | 87.3 |
| TOD3Cap-Small | 115.4M | 331.7min | 87.5 |
| **TOD3Cap** | 124.5M | 350.4min | **94.1** |

### 关键发现
- 多模态输入（2D+3D）显著优于单模态：LiDAR 提供距离信息，相机提供视觉属性，两者互补
- Relation Q-Former 优于关系图和 Transformer 解码器，关键在于能同时利用 BEV 全局上下文
- 三阶段训练每一步都不可或缺，去掉描述器预训练下降 8.8 CIDEr
- LLaMA 作为语言生成器显著优于 GPT2 和 S&T，说明网络设计能充分释放大模型的语言生成能力

## 亮点与洞察
- **开创户外 3D 密集描述新任务**：明确定义了室内外域差异（动态、稀疏、固定视角、大面积），并围绕这些差异设计解决方案。该任务定义对自动驾驶的可解释性和人机交互有重要实用价值。
- **数据集构建方法论**：从四个维度（外观/运动/环境/关系）定义描述结构，半自动标注 + 多轮人工校验的流程兼顾了规模和质量。2.3M 描述是目前最大的 3D 密集描述数据集。
- **冻结 LLM + Adapter 的工程智慧**：不重训 LLM 既降低计算成本，又保留了大模型的常识推理能力，对户外长尾场景（如罕见物体）特别有价值。

## 局限与展望
- 对小目标和远距离物体的检测/描述仍有困难
- BEV 分辨率对性能影响较大（50×50 vs 200×200 差6.8 CIDEr），高分辨率带来更大计算开销
- 数据集描述中运动词汇占比极低（2.6%），未来需增强动态描述的多样性
- 仅支持 nuScenes 的 23 类物体，未涵盖更细粒度的户外物体类别

## 相关工作与启发
- **vs Scan2Cap**：室内检测-描述管线，使用 VoteNet + graph relation，直接适配户外效果差（43.3 vs 108.0 C@0.5）
- **vs Vote2Cap-DETR**：最强室内 baseline，一阶段 set-to-set 框架，适配后仍差 9.6 CIDEr，说明户外域差异需要专门设计
- **vs BEVFormer**：借鉴其空间-时序 BEV 编码思路，为描述生成提供时序动态特征支持

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次定义户外3D密集描述任务 + 百万级数据集
- 实验充分度: ⭐⭐⭐⭐ 多维消融完整，但缺少跨数据集评估
- 写作质量: ⭐⭐⭐⭐ 域差异分析透彻，图表信息量大
- 价值: ⭐⭐⭐⭐⭐ 填补户外3D密集描述空白，对自动驾驶可解释性有重要推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] 4D Contrastive Superflows are Dense 3D Representation Learners](4d_contrastive_superflows_are_dense_3d_representation_learners.md)
- [\[ICCV 2025\] Controllable 3D Outdoor Scene Generation via Scene Graphs](../../ICCV2025/autonomous_driving/controllable_3d_outdoor_scene_generation_via_scene_graphs.md)
- [\[ECCV 2024\] Monocular Occupancy Prediction for Scalable Indoor Scenes](monocular_occupancy_prediction_for_scalable_indoor_scenes.md)
- [\[ECCV 2024\] Random Walk on Pixel Manifolds for Anomaly Segmentation of Complex Driving Scenes](random_walk_on_pixel_manifolds_for_anomaly_segmentation_of_complex_driving_scene.md)
- [\[CVPR 2026\] CARD: A Multi-Modal Automotive Dataset for Dense 3D Reconstruction in Challenging Road Topography](../../CVPR2026/autonomous_driving/card_a_multi-modal_automotive_dataset_for_dense_3d_reconstruction_in_challenging.md)

</div>

<!-- RELATED:END -->
