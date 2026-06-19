---
title: >-
  [论文解读] Finer-CAM: Spotting the Difference Reveals Finer Details for Visual Explanation
description: >-
  [CVPR 2025][多模态VLM][可解释性] 将 CAM 的解释目标从单类 logit $y^c$ 改为类间差值 $y^c - \gamma \cdot y^d$（目标类与相似类的 logit 差），零额外参数地将任何 CAM 方法升级为细粒度版本，使激活图从"整体轮廓"细化到"区分性局部细节"。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "可解释性"
  - "CAM改进"
  - "细粒度识别"
  - "类别对比"
  - "视觉解释"
---

# Finer-CAM: Spotting the Difference Reveals Finer Details for Visual Explanation

**会议**: CVPR 2025  
**arXiv**: [2501.11309](https://arxiv.org/abs/2501.11309)  
**代码**: [https://github.com/Imageomics/Finer-CAM](https://github.com/Imageomics/Finer-CAM)  
**领域**: 多模态VLM  
**关键词**: 可解释性、CAM改进、细粒度识别、类别对比、视觉解释

## 一句话总结
将 CAM 的解释目标从单类 logit $y^c$ 改为类间差值 $y^c - \gamma \cdot y^d$（目标类与相似类的 logit 差），零额外参数地将任何 CAM 方法升级为细粒度版本，使激活图从"整体轮廓"细化到"区分性局部细节"。

## 研究背景与动机

**领域现状**：Class Activation Mapping（CAM）是最广泛使用的视觉解释方法，通过加权组合特征图生成热力图，指示哪些区域对分类决策贡献最大。

**现有痛点**：CAM 解释的是单个类别的 logit $y^c$，这导致它高亮所有对该类有正贡献的区域——包括与相似类共享的区域。例如在区分两种蓝色鸟时，CAM 高亮整个蓝色身体（两种鸟共有），而非翅膀颜色差异（区分性特征）。这使得 CAM 在细粒度识别中几乎无法提供有用的解释。

**核心矛盾**：CAM 的解释目标（单类 logit）与用户需要的解释内容（类间区分性特征）之间存在根本错位。改进 CAM 的过程（如何计算权重）无法解决这个"目标"层面的问题。

**本文目标** 重新定义 CAM 的解释目标，使其聚焦于"区分目标类和相似类的细粒度差异"。

**切入角度**：灵感来自人类的"找不同"能力——通过对比两个相似事物来发现细微差异。将 CAM 的解释目标从 $y^c$ 改为 $y^c - \gamma \cdot y^d$，抑制共享特征、放大区分性特征。

**核心 idea**：在反向传播时将目标从单类 logit 改为类间 logit 差值，零成本地将粗粒度 CAM 转为细粒度 Finer-CAM。

## 方法详解

### 整体框架
标准前向传播得到特征图 $\mathcal{A} = \{A_k\}$ 和所有类 logit → 选取与目标类 $c$ 最相似的 $T=3$ 个参考类 → 对每个参考类 $d$ 计算对比权重 $\alpha_k^{c,d} = \alpha_k^c - \gamma \cdot \alpha_k^d$ → 跨参考类平均 → ReLU 激活得到细粒度热力图。

### 关键设计

1. **对比解释目标**:

    - 功能：将 CAM 权重从"对单类的贡献"转为"对类间区分的贡献"
    - 核心思路：对基于梯度的方法，$\alpha_k^{c,d} = \frac{1}{Z}\sum_{i,j}\frac{\partial(y^c - \gamma \cdot y^d)}{\partial A_k^{ij}}$，由于导数线性性直接分解为 $\alpha_k^c - \gamma \cdot \alpha_k^d$。对基于分数的方法类似——遮蔽特征图后比较两个类的预测差。关键是在权重层面做减法（ReLU 之前），而非在热力图层面（ReLU 之后）
    - 设计动机：直接减去两个 CAM 热力图会产生噪声（因为 ReLU 是非线性的），而在权重层面减法+再 ReLU 能产生干净的背景和锐利的区分性区域

2. **参考类选择与聚合**:

    - 功能：选择最有参考价值的相似类并聚合多次对比结果
    - 核心思路：参考类可以通过分类器权重的余弦相似度或预测 logit 排序选取，默认取 Top-3。多个参考类的对比权重取平均后再 ReLU，使热力图反映的是"与所有相似类都不同"的区域
    - 设计动机：使用单一参考类可能遗漏某些区分性特征。与第 2 预测类对比 RD 最高（0.198），但聚合 Top-3 在 Deletion 指标上最优（0.076）

3. **对比强度 $\gamma$**:

    - 功能：控制解释的粗细粒度
    - 核心思路：$\gamma=0$ 退化为标准 CAM（粗粒度），$\gamma$ 增大逐步聚焦于更精细的区分性特征。实验发现 $\gamma=0.6$ 在相对置信度下降和绝对置信度下降之间取得最佳平衡
    - 设计动机：提供了一个连续的粗到细调节旋钮，用户可以根据需求选择解释粒度

### 损失函数 / 训练策略
纯推理方法，无需训练。仅修改反向传播的目标，兼容任何 CAM 变体（Grad-CAM、Layer-CAM、Score-CAM 等）。

## 实验关键数据

### 主实验

| 方法 | Birds-525 RD@0.1 | CUB-200 RD@0.1 | CUB-200 定位 | Cars RD@0.1 |
|------|-----------------|----------------|-------------|------------|
| Grad-CAM | 0.245 | 0.113 | 0.582 | 0.067 |
| **+Finer** | **0.260** | **0.121** | **0.629** | **0.071** |
| Layer-CAM | 0.255 | 0.116 | 0.625 | 0.069 |
| **+Finer** | **0.270** | **0.120** | **0.682** | **0.074** |
| Score-CAM | 0.217 | 0.102 | 0.670 | - |
| **+Finer** | **0.227** | **0.109** | **0.683** | - |

### 消融实验

| 配置 | Del. ↓ | RD@0.05 ↑ | 说明 |
|------|--------|----------|------|
| 标准 Grad-CAM | 0.079 | 0.174 | 基线 |
| + 第 2 预测类对比 | 0.079 | **0.198** | 单类最强 |
| + Top-3 聚合 | **0.076** | 0.192 | 综合最优 |

### 关键发现
- **Finer-CAM 对所有 CAM 方法一致有效**：Grad-CAM、Layer-CAM、Score-CAM 加上 Finer 后全部提升
- **定位改善显著**：CUB-200 上 Layer-CAM 的定位从 0.625 提升到 0.682（+9.1%），说明 Finer-CAM 确实让热力图更精确地聚焦于区分性区域
- **$\gamma=0.6$ 是最佳平衡**：太小退化为标准 CAM，太大丢失目标类本身的信息

## 亮点与洞察
- **"改变解释目标而非解释过程"**这一洞察非常深刻——过去 CAM 改进都在"how to explain"上下功夫，本文指出问题在"what to explain"，一行代码改变就带来普遍改善
- **与 CLIP 的扩展**：可以用文本提示的差异来做概念级的细粒度定位（如"红色肩章" vs "鸟"），将 Finer-CAM 扩展到零样本场景
- **零额外成本**：不增加参数、不增加前向传播、仅修改反向传播目标，是罕见的"纯免费"改进

## 局限与展望
- 需要知道哪些类是"相似的"，如果分类器本身对似类缺乏区分力，对比效果有限
- $\gamma$ 需要手动调节，自适应确定 $\gamma$ 可能进一步提升
- 仅在细粒度分类上验证，对通用目标检测/分割场景的价值不确定

## 相关工作与启发
- **vs XGrad-CAM / Ablation-CAM**：这些方法改进了 CAM 的权重计算过程但不改变解释目标，属于"how"层面的改进。Finer-CAM 在"what"层面改进，与它们完全正交
- **vs 对比解释（CRP 等）**：部分对比解释方法比较最相关/不相关特征，但需要特殊的网络分析工具。Finer-CAM 只需一行代码修改

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "改变解释目标"的洞察极其简洁深刻，一行代码改变但效果普遍
- 实验充分度: ⭐⭐⭐⭐ 多数据集、多 CAM 方法验证，但缺少对非细粒度任务的评估
- 写作质量: ⭐⭐⭐⭐⭐ 论证逻辑严谨，从问题诊断到解决方案一气呵成
- 价值: ⭐⭐⭐⭐ 对可解释性研究有重要贡献，实用性强但适用场景偏细粒度

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] The Devil Is in the Details: Tackling Unimodal Spurious Correlations for Generalizable Multimodal Reward Models](../../ICML2025/multimodal_vlm/the_devil_is_in_the_details_tackling_unimodal_spurious_correlations_for_generali.md)
- [\[ACL 2025\] Finding Needles in Images: Can Multi-modal LLMs Locate Fine Details?](../../ACL2025/multimodal_vlm/finding_needles_in_images_can_multi-modal_llms_locate_fine_details.md)
- [\[AAAI 2026\] Difference Vector Equalization for Robust Fine-tuning of Vision-Language Models](../../AAAI2026/multimodal_vlm/difference_vector_equalization_for_robust_fine-tuning_of_vis.md)
- [\[AAAI 2026\] Few-Shot Precise Event Spotting via Unified Multi-Entity Graph and Distillation](../../AAAI2026/multimodal_vlm/few-shot_precise_event_spotting_via_unified_multi-entity_graph_and_distillation.md)
- [\[CVPR 2025\] Relation-Rich Visual Document Generator for Visual Information Extraction](relation-rich_visual_document_generator_for_visual_information_extraction.md)

</div>

<!-- RELATED:END -->
