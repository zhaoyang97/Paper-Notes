---
title: >-
  [论文解读] Manual-PA: Learning 3D Part Assembly from Instruction Diagrams
description: >-
  [ICCV 2025][自监督学习][3D Part Assembly] 提出 Manual-PA，一个基于 Transformer 的说明书引导 3D 零件组装框架：通过对比学习将 3D 零件与说明书步骤图对齐来推断组装顺序，再以学到的顺序作为位置编码的软引导进行 6DoF 位姿预测，在 PartNet 上显著超越现有方法。
tags:
  - "ICCV 2025"
  - "自监督学习"
  - "3D Part Assembly"
  - "指令图组装"
  - "Transformer"
  - "对比学习"
  - "排列学习"
---

# Manual-PA: Learning 3D Part Assembly from Instruction Diagrams

**会议**: ICCV 2025  
**arXiv**: [2411.18011](https://arxiv.org/abs/2411.18011)  
**代码**: 无  
**领域**: 自监督  
**关键词**: 3D Part Assembly, 指令图组装, Transformer, 对比学习, 排列学习

## 一句话总结

提出 Manual-PA，一个基于 Transformer 的说明书引导 3D 零件组装框架：通过对比学习将 3D 零件与说明书步骤图对齐来推断组装顺序，再以学到的顺序作为位置编码的软引导进行 6DoF 位姿预测，在 PartNet 上显著超越现有方法。

## 研究背景与动机

**领域现状**：3D 零件组装任务旨在将一组无序的 3D 零件预测其 6DoF 位姿并组装成完整物体。现有方法分为两类：(1) 基于几何特征的生成式方法（如 3DHPA、SPAFormer），利用零件形状关系进行组装，但可能产生不稳定结果；(2) 基于引导信息的方法（如 MEPNet 用于 LEGO），但通常假设零件按步骤逐个提供。

**现有痛点**：
   - **解空间巨大且稀疏**：组合爆炸问题——N 个零件的排列数为 N!，加上每个零件的连续 6DoF 位姿参数，可行的稳定组装序列非常少
   - **无引导的生成式方法**：不使用用户可获取的额外信息（如说明书），在零件数增多时性能急剧下降
   - **LEGO 类方法的局限**：假设零件按步骤提供、有标准化"stud"接口，不适用于家具等通用组装场景——家具说明书不直接告诉每步用哪个零件
   - **误差累积**：直接用自回归方式按预测顺序组装容易让前步错误传播到后续步骤

**核心想法**：人类组装家具时依赖说明书的步骤示意图——利用这些图像信息来降低搜索空间。关键挑战在于：(1) 如何将 2D 线稿示意图与 3D 零件对齐确定顺序；(2) 如何让顺序信息作为"软引导"而非"硬约束"来辅助组装。

**核心idea**：对比学习对齐 3D 零件与步骤图得出组装顺序 + 排列感知的位置编码引导 Transformer 预测位姿。

## 方法详解

### 整体框架

输入：N 个 3D 零件点云 $\{\mathcal{P}_i\}_{i=1}^N$ + N 步说明书图像序列 $(\mathcal{I}_1, ..., \mathcal{I}_N)$。

Pipeline 分三个阶段：
1. **特征提取**：PointNet 编码 3D 零件 → $\mathbf{f}^P \in \mathbb{R}^{N \times D}$；DINOv2 编码相邻步骤差分图 → $\mathbf{f}^I \in \mathbb{R}^{N \times K \times D}$
2. **排列学习**：计算相似度矩阵 → 匈牙利匹配得到排列矩阵 $\mathbf{P}$
3. **位姿预测**：用排列顺序设定位置编码 → Transformer 解码器预测每个零件的旋转和平移

### 关键设计

1. **差分图特征提取**：

    - 功能：从步骤说明图中提取"每步新增了什么零件"的信息
    - 核心思路：对相邻两步的图像取差分 $|\mathcal{I}_j - \mathcal{I}_{j+1}|$，获得新增零件区域，将差分图 patchify 后送入 DINOv2 编码器，通过线性层映射到统一维度 $D$
    - 设计动机：说明书的增量特性意味着步骤差异直接对应新增零件信息

2. **对比学习驱动的排列学习**：

    - 功能：学习 3D 零件与说明书步骤之间的对应关系
    - 核心思路：
        - 构建相似度矩阵 $\mathbf{S}_{ij} = \text{sim}(\mathbf{f}_i^P, \mathbf{g}_j^I)$，其中 $\mathbf{g}^I$ 是对 patch 维度 max-pool 后的步骤特征
        - 用匈牙利算法在 $\mathbf{C} = -\mathbf{S}$ 上求解最优二部图匹配得到排列矩阵 $\mathbf{P}$
        - 训练使用 InfoNCE 对比损失：$\mathcal{L}_{\text{order}} = -\frac{1}{B}\sum_i \log\frac{\exp(\text{sim}(\mathbf{f}^P_{\sigma(i)}, \mathbf{g}^I_i)/\tau)}{\sum_j \exp(\text{sim}(\mathbf{f}^P_{\sigma(i)}, \mathbf{g}^I_j)/\tau)}$
    - 设计动机：对比学习天然适合跨模态对齐，匈牙利匹配保证一一对应的排列约束

3. **排列感知位置编码引导的位姿预测**：

    - 功能：将学到的组装顺序作为软引导注入位姿预测过程
    - 核心思路：
        - 用正弦位置编码 $\Phi \in \mathbb{R}^{N \times D}$ 表示步骤顺序
        - 步骤图直接使用 $\mathbf{p}^I = \Phi$；零件的位置编码通过排列矩阵重排 $\mathbf{p}^P = \mathbf{P}^T \Phi$
        - 训练时用真值顺序，推理时用预测的 $\hat{\mathbf{P}}$
        - 位置编码加到特征上后送入 L 层 Transformer 解码器：自注意力（零件间交互）→ 交叉注意力（步骤图到零件的信息注入）
        - 位姿预测头输出每个零件的四元数旋转 $\hat{q}_i$ 和三维平移 $\hat{t}_i$。使用 RoPE 替代标准正弦编码获得更好性能
    - 设计动机：位置编码是"软引导"——通过注意力分数自然使每个零件更多关注其对应的步骤图，但不硬性约束，避免误差累积

4. **几何等价组处理**：

    - 功能：处理几何上相同的零件（如四条桌腿）
    - 核心思路：通过 AABB 尺寸识别等价组，组内使用匈牙利匹配（以 Chamfer 距离为代价）确定最优对应关系后再计算损失
    - 设计动机：避免对称零件的任意标签导致训练信号混乱

### 损失函数 / 训练策略

- **排列学习**：InfoNCE 对比损失 $\mathcal{L}_{\text{order}}$
- **位姿估计**：四项加权和 $\mathcal{L}_{\text{pose}} = \lambda_T \mathcal{L}_T + \lambda_C \mathcal{L}_C + \lambda_E \mathcal{L}_E + \lambda_S \mathcal{L}_S$
    - $\mathcal{L}_T$：平移的 $\ell_2$ 距离
    - $\mathcal{L}_C$：旋转的 Chamfer 距离（处理内在对称性）
    - $\mathcal{L}_E$：旋转的 $\ell_2$ 距离（正则化项，处理非完美对称零件）
    - $\mathcal{L}_S$：整体组装形状的 Chamfer 距离
- 两阶段训练：先训练排列学习至收敛，再训练位姿估计（使用排列模型预测的顺序）

## 实验关键数据

### 主实验

**PartNet 测试集（Level-3，3 个类别）**：

在 Chair / Table / Storage 三个类别上与现有方法对比，评测指标包括 Shape Chamfer Distance (SCD↓)、Part Accuracy (PA↑)、Connectivity Accuracy (CA↑)、Success Rate (SR↑)：

- Manual-PA 在 **Chair** 类别上取得最高成功率（SR），显著超越 SPAFormer、3DHPA 等无引导方法
- 在 **Table** 类别上 Shape Chamfer Distance 最低，组装精度最高
- 与 Image-PA（使用 RGB 图像引导）对比，Manual-PA 使用线稿图仍然取得更好效果，说明步骤顺序信息比图像外观更关键

**IKEA-Manual 零样本泛化**：

- 在真实 IKEA 家具数据集上进行零样本评测（仅在 PartNet 上训练）
- 在 Chair 和 Table 类别上均展示了良好的泛化能力
- 证明方法不依赖于特定数据集的分布特性

### 消融实验

| 组件 | SCD↓ | PA↑ | SR↑ |
|------|------|-----|-----|
| 无说明书引导 (baseline) | 高 | 低 | 低 |
| + 排列学习 | 中 | 中 | 中 |
| + 顺序引导位置编码 | 低 | 高 | 高 |
| + RoPE | **最低** | **最高** | **最高** |

- 排列学习提供的组装顺序是性能提升的关键因素
- 将顺序作为软引导（位置编码）比硬约束（自回归）更鲁棒
- 准确的排列预测对下游位姿估计至关重要——排列准确率越高，组装成功率越高

## 个人思考

- **亮点**：问题定义新颖——首次将组装说明书引入 3D 零件组装；"软引导"设计巧妙，通过位置编码将离散顺序信息自然融入连续位姿预测；对比学习跨模态对齐 + 匈牙利匹配的组合简洁有效
- **局限**：假设每步只添加一个零件，且需要预先渲染差分图；真实说明书通常有文字和箭头等复杂元素，当前方法仅处理线稿
- **启发**：利用人类已有的结构化知识（说明书/图纸）来约束组合优化问题是一个有前景的方向，可推广到机器人操作、建筑施工等领域

## 亮点与洞察

## 局限与展望

## 相关工作与启发

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] WIR3D: Visually-Informed and Geometry-Aware 3D Shape Abstraction](wir3d_visually-informed_and_geometry-aware_3d_shape_abstraction.md)
- [\[ICML 2026\] PartCo: Part-Level Correspondence Priors Enhance Category Discovery](../../ICML2026/self_supervised/partco_part-level_correspondence_priors_enhance_category_discovery.md)
- [\[CVPR 2025\] Escaping Plato's Cave: Towards the Alignment of 3D and Text Latent Spaces](../../CVPR2025/self_supervised/escaping_platos_cave_towards_the_alignment_of_3d_and_text_latent_spaces.md)
- [\[CVPR 2026\] How Much 3D Do Video Foundation Models Encode?](../../CVPR2026/self_supervised/how_much_3d_do_video_foundation_models_encode.md)
- [\[CVPR 2026\] TALO: Pushing 3D Vision Foundation Models Towards Globally Consistent Online Reconstruction](../../CVPR2026/self_supervised/talo_pushing_3d_vision_foundation_models_towards_globally_consistent_online_reco.md)

</div>

<!-- RELATED:END -->
