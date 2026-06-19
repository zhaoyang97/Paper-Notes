---
title: >-
  [论文解读] FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models
description: >-
  [多模态VLM] 提出 FT-NCFM 框架，通过因果归因（Fact-Tracing）评估样本价值并引导对抗式 NCFM 过程合成高信息密度核心集，仅用 5% 合成数据即可达到全量训练 85-90% 的性能，训练时间减少 80% 以上。 VLA（Vision-Language-Action）模型通过联合处理视觉、语言和动作实…
tags:
  - "多模态VLM"
---

# FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models

- **会议**: AAAI 2026
- **arXiv**: [2511.16233](https://arxiv.org/abs/2511.16233)
- **代码**: 未公开
- **领域**: 多模态VLM
- **关键词**: VLA模型, 数据蒸馏, 影响函数, 对比验证, 核心集合成, 高效训练

## 一句话总结

提出 FT-NCFM 框架，通过因果归因（Fact-Tracing）评估样本价值并引导对抗式 NCFM 过程合成高信息密度核心集，仅用 5% 合成数据即可达到全量训练 85-90% 的性能，训练时间减少 80% 以上。

## 背景与动机

VLA（Vision-Language-Action）模型通过联合处理视觉、语言和动作实现端到端机器人控制，但其性能严重依赖大规模冗余数据集（如 Open X-Embodiment 含百万级轨迹）。现有优化路径存在根本缺陷：

1. **模型压缩**（如 SmolVLA、TinyVLA）：简化网络结构会导致复杂任务性能显著下降
2. **策略蒸馏**（如 RLDG、DROC）：知识绑定在模型参数中，不可独立分析或复用，且依赖昂贵的教师模型
3. **核心集选择**（Coreset Selection）：仅从现有样本中"选择"，受限于已有样本的信息密度上限

以上方法均未从**数据层面**解决效率瓶颈。本文提出一个关键洞察：在完整、未分化的数据集上直接训练不仅低效，还可能妨碍模型聚焦关键任务特征。因此应从"选择"数据升级为"合成"高信息密度数据。

## 方法详解

FT-NCFM 包含三个阶段：多模态表征学习 → FT 影响力评估引擎 → 影响力引导的 NCFM 蒸馏。

### 1. 多模态表征学习

将原始 VLA 数据 $d = (V, L, A)$（视觉、语言、动作）通过各自编码器转换为统一 token 序列，再由 Transformer 骨干融合为全局特征：

$$\mathbf{h} = \Phi(d) = \Phi(V, L, A) \in \mathbb{R}^{d_{model}}$$

该特征向量作为后续影响力分析和分布匹配的操作对象。

### 2. FT 影响力评估引擎

FT 引擎分两阶段计算每个样本 $d_i$ 的影响力权重 $w_i$。

**阶段一：基于影响函数的因果归因预筛选**

采用影响函数近似移除单个样本后模型损失的变化：

$$I_{\text{loss}}(d_{train}, d_{test}) \approx -\nabla_\theta L(\mathbf{h}_{test}, \hat{\theta})^T H_{\hat{\theta}}^{-1} \nabla_\theta L(\mathbf{h}_{train}, \hat{\theta})$$

其中 $H_{\hat{\theta}}^{-1}$ 为 Hessian 矩阵的逆，使用 LiSSA 算法高效近似。$\hat{\theta}$ 来自一个"引导模型"——与下游模型同结构但仅轻度训练（标准训练时间的 10-20%），用于提供稳定梯度场。此阶段产出基础影响分数 $Score_{base}(d_i)$。

**阶段二：对比验证精炼（Contrastive Verification）**

对 top-K% 精英样本进行进一步验证，确保高分样本确实正向贡献泛化。核心流程：

1. **指令语义解析**：解析精英样本的语言指令结构
2. **扰动模板选择**：从预设的可复用扰动模板库中匹配（如物体替换、尺寸缩放、位置变换）
3. **仿真器场景实例化**：保持语言 $L$ 和动作 $A$ 不变，在仿真器中程序化修改视觉场景 $V$，生成"最小反例" $d_{contrast}$

通过梯度点积近似影响力，分别计算精英样本和反例对测试样本的影响分数：

$$Score_i = \nabla_\theta L(\Phi(d_{test}), \hat{\theta})^T \cdot \nabla_\theta L(\Phi(d_i), \hat{\theta})$$

$$Score_{contrast} = \nabla_\theta L(\Phi(d_{test}), \hat{\theta})^T \cdot \nabla_\theta L(\Phi(d_{contrast}), \hat{\theta})$$

最终影响力权重通过连续调制函数计算：

$$w_i = Score_{base}(d_i) \times \left(1 + \tanh\left(\beta \cdot (Score_i - Score_{contrast})\right)\right)$$

$\tanh$ 将差值平滑映射到 $[-1, 1]$，构造 $[0, 2]$ 范围的权重调制因子，实现对基础分数的动态正负调整。

### 3. 影响力引导的 NCFM 蒸馏

原始 NCFM 通过极小极大博弈匹配真实与合成数据的特征分布（假设均匀采样）。本文将均匀期望替换为基于影响力权重的加权期望：

$$\min_{D_{synth}} \max_\psi \left\| \sum_{i=1}^{N} \frac{w_i}{\sum_j w_j} \psi(\Phi(d_i)) - \mathbb{E}_{d' \sim D_{synth}}[\psi(\Phi(d'))] \right\|^2$$

判别器 $\psi$ 被迫关注高权重的高价值真实样本，生成器因此优先模仿高价值样本的分布特征，合成因果知识丰富、信息密度极高的核心集。输出格式与原始数据一致，可直接用于任何下游 VLA 模型训练。

## 实验结果

### 基准与设置

- **基准**：CALVIN（长时域泛化）、Meta-World（50 任务多任务学习）、LIBERO（终身学习泛化）
- **基础模型**：统一架构 ViT-B/16 + 6 层 Transformer 解码器
- **硬件**：单张 NVIDIA A100 80GB

### 表1：CALVIN ABC→D 零样本长时域评估

| 方法 | 数据比例 | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Avg. Len ↑ |
|------|---------|--------|--------|--------|--------|--------|-----------|
| RT-1 | 100% | 0.533 | 0.222 | 0.094 | 0.038 | 0.013 | 0.90 |
| GR-1 | 100% | 0.854 | 0.712 | 0.596 | 0.497 | 0.401 | 3.06 |
| RoboUniview | 100% | 0.942 | 0.842 | 0.734 | 0.622 | 0.507 | 3.65 |
| **FT-NCFM** | **1%** | 0.755 | 0.531 | 0.402 | 0.298 | 0.204 | 2.19 |
| **FT-NCFM** | **5%** | 0.895 | 0.733 | 0.612 | 0.501 | 0.373 | 3.11 |
| **FT-NCFM** | **10%** | 0.925 | 0.791 | 0.688 | 0.590 | 0.476 | 3.47 |

10% 数据达到 SOTA 的 95%（3.47 vs 3.65），几乎持平 Vidman（3.42，100% 数据）。

### 表2：与模型中心方法的范式比较（CALVIN）

| 范式 | 方法 | 数据比例 | 总训练时间(GPU-h) ↓ | Avg. Len ↑ |
|------|------|---------|-------------------|-----------|
| 策略蒸馏 | DROC | 100% | 193 | 3.05 |
| 策略蒸馏 | Mole-VLA | 100% | 178 | 3.20 |
| 策略蒸馏 | RLDG | 100% | 198 | 3.15 |
| 核心集选择 | Random | 5% | 6.5 | 1.88 |
| 核心集选择 | IF Coreset | 5% | 18 | 2.45 |
| **FT-NCFM** | **Ours** | **5%** | **25** | **3.11** |
| **FT-NCFM** | **Ours** | **10%** | **31.5** | **3.47** |

5% 数据 + 25h 总时间即达策略蒸馏方法性能水平，资源消耗仅为后者的 1/7。10% 数据性能远超所有策略蒸馏方法。

### 表3：消融实验（CALVIN，5% 数据）

| 变体 | Avg. Len ↑ |
|------|-----------|
| FT-NCFM（完整方法） | 3.11 |
| 去除对比验证（仅用基础分数） | 2.81 |
| 去除整个 FT 引擎（随机权重 NCFM） | 2.15 |

对比验证模块贡献 +0.30；FT 引擎整体贡献 +0.96，是框架成功的基石。

## 关键发现

1. **数据中心优于模型中心**：智能数据蒸馏在效率与性能的权衡上显著优于模型压缩和策略蒸馏
2. **合成优于选择**：生成式核心集（3.11）远胜传统影响函数核心集选择（2.45）和随机选择（1.88）
3. **长时域任务增强**：在 LIBERO-Long 上 10% 数据（56.6%）超越所有 100% 数据基线，表明数据蒸馏能增强关键因果和泛化知识
4. **投资回报模型**：FT 引擎 + NCFM 预处理约 24 GPU-h 的一次性投资，在多轮迭代开发中具有显著摊销收益

## 亮点

- **范式创新**：首个面向 VLA 的生成式数据蒸馏框架，将效率优化从模型层提升到数据层
- **自包含评估引擎**：FT 引擎通过因果归因 + 程序化对比验证两阶段评估样本内在价值，不依赖外部教师模型
- **可复用扰动模板**：设计的扰动模板库（物体替换、尺寸缩放、位置变换）具有通用性和可扩展性
- **模型无关输出**：合成核心集格式与原始数据一致，可直接用于训练任意下游 VLA 模型
- **极端数据效率**：5% 数据达 85-90% SOTA 性能，训练时间减少 80%+

## 局限性

1. **扰动模板覆盖有限**：当前模板库覆盖核心维度（物体替换、尺寸缩放等），但未涵盖物理属性变化（质量、摩擦力）等失败场景
2. **依赖仿真器数据**：自动反例生成依赖可程序化编辑的仿真器数据集，难以直接应用于无法编辑的真实世界数据
3. **引导模型开销**：虽然轻度训练即可，但 FT 引擎仍需训练一个引导模型用于梯度计算
4. **超参敏感性**：$\beta$ 参数控制调制函数灵敏度，需要调优

## 相关工作

- **VLA 模型**：RT-1/RT-2（大规模）、OpenVLA（开源）、SpatialVLA（空间表征）
- **模型压缩**：SmolVLA、TinyVLA — 简化结构但性能下降
- **策略蒸馏**：RLDG（CoRL 2024）、DROC（ICLR 2023）— 知识不可复用且依赖教师模型
- **核心集选择**：DataMIL、Zero-shot Coreset — 受限于现有样本信息密度
- **影响函数**：LiSSA 近似、Hessian 逆向量积 — 本文将其扩展为数据蒸馏的价值评估工具
- **NCFM**：Neural Characteristic Function Matching — 本文将其改为影响力加权版本

## 评分

⭐⭐⭐⭐ — 提出了 VLA 领域首个数据中心生成式蒸馏框架，范式创新明确，实验覆盖三个主流基准且效果显著（5% 数据达 85-90% SOTA）。局限在于仿真器依赖和模板覆盖有限，但整体思路对推动高效 VLA 研究有较大价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for Egocentric Intent Disambiguation](plug-and-play_clarifier_a_zero-shot_multimodal_framework_for_egocentric_intent_d.md)
- [\[ICLR 2026\] WebDS: An End-to-End Benchmark for Web-based Data Science](../../ICLR2026/multimodal_vlm/webds_an_end-to-end_benchmark_for_web-based_data_science.md)
- [\[ICLR 2026\] Why Reinforcement Fine-Tuning Preserves Prior Knowledge Better: A Data Perspective](../../ICLR2026/multimodal_vlm/why_reinforcement_fine-tuning_enables_mllms_preserve_prior_knowledge_better_a_da.md)
- [\[AAAI 2026\] ReCAD: Reinforcement Learning Enhanced Parametric CAD Model Generation with Vision-Language Models](recad_reinforcement_learning_enhanced_parametric_cad_model_generation_with_visio.md)
- [\[ICCV 2025\] SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency](../../ICCV2025/multimodal_vlm/scan_bootstrapping_contrastive_pre-training_for_data_efficiency.md)

</div>

<!-- RELATED:END -->
