---
title: >-
  [论文解读] TOMCAT: Test-time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning
description: >-
  [NeurIPS 2025][多模态VLM][组合零样本学习] 提出 TOMCAT，通过在测试时从无标签数据中累积文本和视觉双模态知识来动态更新组合原型，克服标签分布偏移问题，在四个 CZSL 基准上实现 SOTA。 领域现状：组合零样本学习（CZSL）要求模型识别由已知属性和物体重新组合而成的未见过的属性-物体组合（如"棕…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "组合零样本学习"
  - "测试时适应"
  - "知识积累"
  - "多模态原型"
  - "CLIP"
---

# TOMCAT: Test-time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning

**会议**: NeurIPS 2025  
**arXiv**: [2510.20162](https://arxiv.org/abs/2510.20162)  
**代码**: [https://github.com/xud-yan/TOMCAT](https://github.com/xud-yan/TOMCAT)  
**领域**: 多模态VLM  
**关键词**: 组合零样本学习, 测试时适应, 知识积累, 多模态原型, CLIP

## 一句话总结
提出 TOMCAT，通过在测试时从无标签数据中累积文本和视觉双模态知识来动态更新组合原型，克服标签分布偏移问题，在四个 CZSL 基准上实现 SOTA。

## 研究背景与动机

**领域现状**：组合零样本学习（CZSL）要求模型识别由已知属性和物体重新组合而成的未见过的属性-物体组合（如"棕色的奶酪"）。基于 CLIP 的方法（如 CSP、Troika、CDS-CZSL）通过 prompt tuning 微调 VLM 取得了显著进展

**现有痛点**：测试时标签空间会包含训练时未见过的组合，导致标签分布偏移。模型参数和类原型在训练后冻结，无法利用测试数据适应新分布

**核心矛盾**：训练只见过部分组合，但测试要求在完整组合空间上判别——这是一个根本性的分布偏移问题

**切入角度**：在测试时利用无标签数据流式累积知识，同时更新文本和视觉两个模态的原型

**核心 idea**：设计可学习的 Knowledge Accumulation Module (KAM) 调整原型偏移量，配合优先队列存储高置信度历史图像构建视觉原型

## 方法详解

### 整体框架
训练阶段：用 prompt tuning + adapter 微调 CLIP 基础模型。测试阶段：保持原始模型冻结，每来一个测试样本：(1) 判断是否更新优先队列；(2) 用 KAM + 自适应权重更新文本和视觉原型；(3) 基于多模态原型做预测；(4) 预测后通过反向传播更新 KAM 参数。

### 关键设计

1. **知识积累模块 (KAM)**：

    - 功能：学习原型的偏移量以适应标签分布变化
    - 核心思路：对文本和视觉原型分别维护可学习参数 $\Delta\mathbf{t}, \Delta\mathbf{v} \in \mathbb{R}^{|C^{te}| \times d}$，初始化为零。更新后原型：$\tilde{\mathbf{t}}_c = \frac{\mathbf{t}_c + w_c \Delta\mathbf{t}_c}{\|\mathbf{t}_c + w_c \Delta\mathbf{t}_c\|}$
    - 设计动机：不直接修改原始模型参数，避免遗忘训练知识；通过残差方式累积新知识

2. **自适应更新权重**：

    - 功能：控制每个组合原型的更新幅度
    - 核心思路：$w_c = \sigma(-\theta \cdot \cos(f^v, \mathbf{t}_c))$，当图像与原型相似度高时（可能是已见组合）减小更新，相似度低时（可能是未见组合）增大更新
    - 设计动机：区别对待已见和未见组合——已见组合的原型已经很好，不需要大幅调整

3. **动态优先队列 + 视觉原型**：

    - 功能：利用测试时的历史图像构建视觉模态原型
    - 核心思路：为每个组合维护长度为 K=3 的优先队列，存储预测熵最低（置信度最高）的图像特征。视觉原型 = 队列中特征的平均值
    - 设计动机：文本原型来自编码的标签文本，视觉原型来自实际图像——双模态互补可以提高判别力

### 损失函数 / 训练策略
测试时优化目标：$\mathcal{L}_{TOMCAT} = \mathcal{L}_{PE} + \lambda \mathcal{L}_{MCRL}$
- $\mathcal{L}_{PE}$：预测熵最小化，鼓励模型做出更自信的预测
- $\mathcal{L}_{MCRL}$：多模态协作表示学习，用对比学习对齐文本和视觉原型

## 实验关键数据

### 主实验 — Closed-world 设置

| 方法 | UT-Zappos AUC | MIT-States AUC | C-GQA AUC |
|------|--------------|----------------|-----------|
| CLIP | 5.0 | 11.0 | 1.4 |
| CSP (ICLR'23) | 33.0 | 19.4 | 6.2 |
| Troika (CVPR'24) | 38.9 | 21.0 | 7.6 |
| CDS-CZSL (CVPR'24) | 40.3 | 21.4 | 7.9 |
| ClusPro (ICLR'25) | 39.6 | 22.8 | 7.8 |
| **TOMCAT** | **42.6** | **24.3** | **8.6** |

### 消融实验

| 配置 | UT-Zappos AUC | MIT-States AUC |
|------|--------------|----------------|
| base model (无TOMCAT) | 40.3 | 21.4 |
| + 文本KAM | 41.4 | 22.8 |
| + 视觉原型 | 41.8 | 23.1 |
| + 自适应权重 | 42.1 | 23.6 |
| + MCRL损失 (完整) | **42.6** | **24.3** |

### 关键发现
- 每个组件贡献稳定：文本KAM最重要(+1.1 AUC)，视觉原型其次(+0.4)
- 在 open-world 设置下优势更大，因为标签分布偏移更严重
- 推理延迟增加有限——KAM更新在预测之后，不影响推理速度
- 优先队列的大小 K=3 就够了，增大到5几乎无提升

## 亮点与洞察
- **测试时适应在 CZSL 中的首次应用**——自然地解决了训练/测试标签空间不匹配的问题
- **自适应更新权重的设计直觉清晰**：已见组合更新少、未见组合更新多，这是一种隐式的难度感知
- **多模态原型互补**：文本原型编码语义知识、视觉原型编码视觉模式，两者对齐后更强

## 局限与展望
- 优先队列中伪标签可能错误，尤其在早期测试样本较少时
- KAM 参数量随组合数线性增长，组合空间极大时（open-world 上万组合）可能有内存问题
- 假设测试样本是在线流式到来的，批量测试场景需要调整
- 未验证在非 CLIP 的 VLM 上的效果

## 相关工作与启发
- **vs TDA (CVPR'24)**：TDA 用 key-value cache 做无训练适应，但不考虑多模态对齐；TOMCAT 显式对齐
- **vs TPT (NeurIPS'22)**：TPT 在每个测试样本上独立优化 prompt，不积累；TOMCAT 跨样本积累知识
- **vs DynaPrompt (ICLR'25)**：DynaPrompt 自适应选择 prompt，TOMCAT 直接更新原型

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ CZSL + 测试时适应的首次结合，方法设计优雅
- 实验充分度: ⭐⭐⭐⭐ 四个数据集，开/闭世界设置，消融完整
- 写作质量: ⭐⭐⭐⭐ 动机论述清晰，方法推导系统
- 价值: ⭐⭐⭐⭐ 为 CZSL 提供了新范式，测试时适应思路可迁移

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Test-Time Spectrum-Aware Latent Steering for Zero-Shot Generalization in Vision-Language Models](test-time_spectrum-aware_latent_steering_for_zero-shot_generalization_in_vision-.md)
- [\[CVPR 2026\] FlowComposer: Composable Flows for Compositional Zero-Shot Learning](../../CVPR2026/multimodal_vlm/flowcomposer_composable_flows_for_compositional_zeroshot_learning.md)
- [\[NeurIPS 2025\] DOTA: DistributiOnal Test-time Adaptation of Vision-Language Models](dota_distributional_testtime_adaptation_of_visionlanguage_mo.md)
- [\[CVPR 2026\] Bridging the Modality Gap in Compositional Zero-Shot Learning via Sparse Alignment and Unimodal Memory Bank](../../CVPR2026/multimodal_vlm/bridging_the_modality_gap_in_compositional_zero-shot_learning_via_sparse_alignme.md)
- [\[NeurIPS 2025\] The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models](the_illusion_of_progress_a_critical_look_at_testtime_adaptat.md)

</div>

<!-- RELATED:END -->
