---
title: >-
  [论文解读] Training-free Online Video Step Grounding
description: >-
  [NeurIPS 2025][多模态VLM][视频步骤定位] 提出BaGLM，一种无需训练的在线视频步骤定位方法，利用贝叶斯滤波将LLM估计的步骤依赖关系和LMM估计的步骤进度融入零样本LMM预测中，在三个数据集上超越现有需训练的离线方法。 视频步骤定位（Video Step Grounding, VSG）旨在给定一组程序性…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "视频步骤定位"
  - "贝叶斯滤波"
  - "LMM零样本"
  - "在线推理"
  - "无训练"
---

# Training-free Online Video Step Grounding

**会议**: NeurIPS 2025  
**arXiv**: [2510.16989](https://arxiv.org/abs/2510.16989)  
**代码**: [GitHub](https://lucazanella.github.io/baglm/)  
**领域**: 多模态VLM  
**关键词**: 视频步骤定位, 贝叶斯滤波, LMM零样本, 在线推理, 无训练

## 一句话总结

提出BaGLM，一种无需训练的在线视频步骤定位方法，利用贝叶斯滤波将LLM估计的步骤依赖关系和LMM估计的步骤进度融入零样本LMM预测中，在三个数据集上超越现有需训练的离线方法。

## 研究背景与动机

视频步骤定位（Video Step Grounding, VSG）旨在给定一组程序性步骤描述和一段视频后，识别视频中执行了哪些步骤。这在AR/XR实时引导（如辅助烹饪、组装家具）中具有重要应用价值。

现有VSG方法面临两个核心限制：

**依赖训练集**：需要收集和标注训练数据（如HowTo100M的叙述文本），训练成本高且可能导致模型偏向特定视频和任务分布

**要求离线处理**：假设可获取完整视频，无法适用于实时视频流场景

本文探索的核心问题：**能否无需训练、在线上完成VSG？** 作者首先做了一个令人惊讶的发现——直接用零样本LMM逐段预测步骤，**已经能超越需训练的离线SOTA方法**（InternVL2.5-8B在CrossTask上超MPTVA 6.4%，在Ego4D GoalStep上超NaSVA 16.1%）。这启发了一个自然的改进方向：能否将过去帧的信息注入LMM的预测中，保持零样本优势的同时进一步提升性能？

## 方法详解

### 整体框架

BaGLM将VSG建模为贝叶斯滤波问题：状态是当前段对应的步骤 $a$，观测是当前视频段 $\mathbf{S}_t$。通过predict步（利用步骤间转移关系）和update步（利用LMM的观测预测）递归估计后验信念 $\text{bel}_t(a)$。

### 关键设计

1. **LMM作为零样本观测模型**

   将VSG转化为多选题形式：给定当前视频段 $\mathbf{S}_t$ 和所有步骤选项（含"none"），提示LMM输出每个选项的概率。归一化后得到步骤概率分布 $f_{\text{LMM}}^{\text{VSG}}: \mathcal{V} \times \mathcal{A} \to \Delta^{K+1}$。

   使用InternVL2.5-8B，每次仅输入当前2秒段加步骤列表，无需全视频访问。

2. **PREDICT步：LLM驱动的步骤转移模型**

   利用LLM（LLaMA3-70B-Instruct）估计步骤间的依赖矩阵 $\mathbf{D} \in \mathbb{R}^{K \times K}$，其中 $\mathbf{D}_{i,j}$ 表示步骤 $a_j$ 是步骤 $a_i$ 前置条件的概率。初始化转移矩阵 $\mathbf{T} = \mathbf{D}^\top$。

   关键创新是根据**步骤进度**动态调整转移矩阵。引入两个度量：

    - **就绪度（Readiness）**：步骤 $a_i$ 的前置条件完成程度：
    $\mathbf{r}_t[i] = \frac{\sum_j \mathbf{D}_{i,j} \cdot \max_{\tau < t} \text{progress}_\tau[j]}{\sum_j \mathbf{D}_{i,j}}$

    - **有效性（Validity）**：步骤 $a_i$ 的后继是否尚未执行（防止重复归因）：
    $\mathbf{v}_t[i] = \frac{\sum_j \mathbf{D}_{j,i} \cdot (1 - \max_{\tau < t} \text{progress}_\tau[j])}{\sum_j \mathbf{D}_{j,i}}$

   调整后的转移矩阵：
    $\tilde{\mathbf{T}}_t[i,j] = \frac{\mathbf{T}[i,j] \cdot \mathbf{r}_t[j] \cdot \mathbf{v}_t[j]}{\sum_k \mathbf{T}[i,k] \cdot \mathbf{r}_t[k] \cdot \mathbf{v}_t[k]}$

   进度估计通过询问LMM每个步骤的完成度（0-9标度）得到。

3. **UPDATE步：融合LMM预测与贝叶斯先验**

   最终信念通过LMM观测似然与predict先验的乘积得到：

    $\text{bel}_t(a_i) = \frac{1}{\mathcal{Z}} \cdot f_{\text{LMM}}(\mathbf{S}_t, \pi_{\text{VSG}})[i] \cdot \sum_{a_j \in \mathcal{A}} \tilde{\mathbf{T}}_t[j,i] \cdot \text{bel}_{t-1}(a_j)$

   其中 $\mathcal{Z}$ 为归一化因子。该公式优雅地将即时的LMM预测与历史积累的信念、步骤间依赖关系融合在一起。

### 损失函数 / 训练策略

**无需任何训练**。所有组件均基于预训练模型的零样本能力：InternVL2.5-8B用于观测和进度估计，LLaMA3-70B用于依赖矩阵提取。视频分割为2秒非重叠段。

## 实验关键数据

### 主实验

| 方法 | 设置 | HT-Step R@1 | CrossTask Avg.R@1 | Ego4D R@1 |
|------|------|-------------|-------------------|-----------|
| VINA | 离线+训练 | 39.1 | 44.8 | - |
| NaSVA | 离线+训练 | 53.1 | 46.7 | 29.1 |
| MPTVA | 离线+训练 | - | 47.9 | - |
| VSLNet | 离线+训练 | - | - | 24.3 |
| NaSVA (在线) | 在线+训练 | 46.1 | - | 24.2 |
| **BaGLM** | **在线+无训练** | **57.4** | **59.8** | **43.3** |

BaGLM在HT-Step/CrossTask/Ego4D上分别超NaSVA **+4.3/+13.1/+14.2%**。

### 消融实验：转移模型组件

| 配置 | HT-Step | CrossTask | Ego4D |
|------|---------|-----------|-------|
| 仅静态转移矩阵 | 55.9 | 58.0 | 42.1 |
| + 就绪度 | 57.0 | 58.8 | 42.0 |
| + 有效性 | 56.4 | 58.8 | 43.1 |
| + 就绪度 + 有效性 | **57.4** | **59.8** | **43.3** |

### Oracle实验

| 配置 | HT-Step | CrossTask | Ego4D |
|------|---------|-----------|-------|
| 估计依赖 + 估计进度 | 57.4 | 59.8 | 43.3 |
| Oracle依赖 + Oracle进度 | **62.6** | **66.9** | **82.2** |

Ego4D上Oracle设置提升38.9%，说明贝叶斯滤波框架本身极有效，改进进度估计和依赖估计会进一步大幅提升。

### 关键发现

- 零样本LMM（仅看当前段）已是VSG的强基线，超越需在HowTo100M上训练的专用方法
- 2秒段时长是最佳权衡：太短缺视觉线索，太长跨越多步骤
- BaGLM在HT-Step和CrossTask上对所有LMM均有一致提升，但在Ego4D GoalStep上改进有限（视频更长、步骤描述更粗）
- LLM选择（LLaMA-3.3-70B vs GPT-4.1-mini）对结果影响较小，说明框架对LLM选择鲁棒

## 亮点与洞察

- **贝叶斯滤波 + LMM的组合极其优雅**：将经典概率推理与现代大模型能力完美结合
- 完全无训练 + 在线推理 + 超越训练型离线方法，实际部署优势巨大
- 步骤进度估计和依赖矩阵的动态转移模型是关键创新，为贝叶斯框架注入了任务特定知识
- Oracle实验清晰指明了改进方向

## 局限与展望

- 依赖LLM估计步骤依赖关系的质量，对模糊或领域特异性强的步骤可能不准确
- 步骤进度估计依赖LMM的主观判断，精度有限
- 在Ego4D等长视频（平均28分钟）上改进不明显，长程依赖建模需加强
- 每个段需两次LMM调用（步骤预测 + 进度估计），实时性受限于LMM推理速度

## 相关工作与启发

- VSG领域从弱监督（Zhukov等）到LLM辅助伪标签（NaSVA、MPTVA）的发展脉络
- VQAScore等工作表明LMM可替代CLIP进行视频-语言对齐评估
- 贝叶斯滤波在跟踪、定位中的经典应用被创造性地引入到步骤定位中
- 启发：可将此框架推广到其他序列性视频理解任务（如动作预测、流程异常检测）

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 贝叶斯滤波+LMM的无训练在线范式完全原创
- **实验充分度**: ⭐⭐⭐⭐⭐ 三数据集、四LMM、消融、Oracle分析、段时长分析
- **写作质量**: ⭐⭐⭐⭐⭐ 数学建模清晰，从初步发现到方法设计的推进逻辑流畅
- **价值**: ⭐⭐⭐⭐⭐ 为视频理解提供了无训练在线推理的新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] See&Trek: Training-Free Spatial Prompting for Multimodal Large Language Model](seetrek_training-free_spatial_prompting_for_multimodal_large_language_model.md)
- [\[CVPR 2026\] PAS: A Training-Free Stabilizer for Temporal Encoding in Video LLMs](../../CVPR2026/multimodal_vlm/pas_a_training-free_stabilizer_for_temporal_encoding_in_video_llms.md)
- [\[CVPR 2025\] Free on the Fly: Enhancing Flexibility in Test-Time Adaptation with Online EM](../../CVPR2025/multimodal_vlm/free_on_the_fly_enhancing_flexibility_in_test-time_adaptation_with_online_em.md)
- [\[CVPR 2026\] Pointing at Parts: Training-Free Few-Shot Grounding in Multimodal LLMs](../../CVPR2026/multimodal_vlm/pointing_at_parts_training-free_few-shot_grounding_in_multimodal_llms.md)
- [\[CVPR 2026\] DRS-GUI: Dynamic Region Search for Training-Free GUI Grounding](../../CVPR2026/multimodal_vlm/drs-gui_dynamic_region_search_for_training-free_gui_grounding.md)

</div>

<!-- RELATED:END -->
