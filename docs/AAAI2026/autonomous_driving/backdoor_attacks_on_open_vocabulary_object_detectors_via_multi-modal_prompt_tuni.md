---
title: >-
  [论文解读] Backdoor Attacks on Open Vocabulary Object Detectors via Multi-Modal Prompt Tuning
description: >-
  [AAAI2026][自动驾驶][backdoor attack] 首次研究开放词汇目标检测器（OVOD）的后门攻击，提出 TrAP（Trigger-Aware Prompt tuning），通过联合优化视觉和文本分支的 learnable prompt 与可学习触发器，在不修改模型权重的前提下注入高成功率后门。
tags:
  - "AAAI2026"
  - "自动驾驶"
  - "backdoor attack"
  - "目标检测"
  - "提示学习"
  - "Grounding DINO"
  - "GLIP"
  - "adversarial security"
---

# Backdoor Attacks on Open Vocabulary Object Detectors via Multi-Modal Prompt Tuning

**会议**: AAAI2026  
**arXiv**: [2511.12735](https://arxiv.org/abs/2511.12735)  
**代码**: [rajankita/TrAP](https://github.com/rajankita/TrAP)  
**领域**: 自动驾驶  
**关键词**: backdoor attack, open-vocabulary object detection, prompt tuning, Grounding DINO, GLIP, adversarial security

## 一句话总结

首次研究开放词汇目标检测器（OVOD）的后门攻击，提出 TrAP（Trigger-Aware Prompt tuning），通过联合优化视觉和文本分支的 learnable prompt 与可学习触发器，在不修改模型权重的前提下注入高成功率后门。

## 背景与动机

- **开放词汇目标检测器**（如 Grounding DINO、GLIP）通过大规模图文预训练实现零样本泛化，可检测训练时未见过的任意类别，广泛应用于自动驾驶、机器人和监控等高风险场景。
- **Prompt tuning** 是轻量适配策略：冻结模型主干，仅优化少量可学习 prompt token 来适配下游任务。用户常将适配过程外包给第三方（如云端服务），这为攻击者提供了白盒访问的机会。
- 现有后门攻击研究集中在**闭集检测器**和**分类任务**，从未涉及 OVOD 模型。直接迁移现有攻击面临两个困难：(1) 预训练数据不可得，无法做数据投毒；(2) OVOD 的检测依赖视觉-文本对齐，仅扰动单一模态效果有限。
- 因此，本文聚焦于一种现实威胁场景：攻击者在 prompt tuning 阶段植入后门，模型在干净输入上正常工作，但遇到特定触发器时产生攻击者指定的恶意行为。

## 核心问题

1. 能否在不修改预训练 OVOD 模型权重的前提下，仅通过 prompt tuning 注入有效后门？
2. 多模态联合 prompt tuning（视觉+文本）相比单模态 prompt tuning，能否显著提升攻击成功率？
3. 如何在保持小尺寸隐蔽触发器的同时实现高攻击成功率？

## 方法详解

### 威胁模型

- 攻击者（Alice）获得预训练 OVOD 模型 $F_{clean}$ 和用户（Bob）的下游数据集的白盒访问权限
- 攻击者在 prompt tuning 过程中植入后门，生成被污染模型 $F_{poisoned}$
- 两种攻击目标：
    - **目标误分类攻击（OMA）**：触发器使目标物体被误分类为指定类别
    - **目标消失攻击（ODA）**：触发器使目标类别物体完全不被检测

### TrAP 框架

**1. 数据投毒过程**

给定干净图像 $x$，构造投毒图像 $x_{poisoned} = x \oplus \delta$，其中 $\delta$ 是可学习的 patch 触发器，尺寸为目标物体边界框的 $\rho$ 倍，放置于物体中心。OMA 在非目标类物体上贴触发器，ODA 在目标类物体上贴触发器。

**2. 视觉分支 Prompt Tuning**

在 Swin Transformer 的每一层引入 $m_v=50$ 个可学习 prompt token $P_i \in \mathbb{R}^{m_v \times d_v}$，采用 VPT-Deep 策略，prompt 预附到 patch embedding 序列前，使其在每层调制视觉特征。

**3. 文本分支 Prompt Tuning**

- 为每个类别名词嵌入前附共享可学习上下文 $Q = \{q_0, ..., q_{m_t-1}\}$（$m_t=4$）
- 引入轻量 meta-net $h(\cdot)$（两层瓶颈结构，16× 降维），基于视觉编码器输出 $V_N$ 生成图像条件向量 $\pi = h_\theta(V_N)$
- 最终每个类别的 prompt 为 $[\tilde{Q}, w_k]$，其中 $\tilde{Q}_i = q_i + \pi$
- 与 CoCoOp 的 区别：针对目标检测场景，将同一组 prompt 分别附加到每个类别名词（CoCoOp-Det 变体）

**4. 联合优化**

总损失 $\mathcal{L}_{total} = \mathcal{L}_{clean}(\theta) + \lambda \cdot \mathcal{L}_{poisoned}(\theta, \delta)$，其中 $\lambda=1$。可训练参数仅包括视觉 prompt $\{P_i\}$、文本 prompt $Q$、meta-net 参数和触发器 $\delta$，总计约 0.2M 参数（对比 fine-tuning 的 21-36M）。

**5. 课程学习策略**

训练初期使用较大触发器（$\rho=0.2$，前 10 epoch），后期缩小（$\rho=0.1$，后 5 epoch），从而在训练初期建立可靠的触发器-行为关联，推理时使用小且隐蔽的 patch 仍可激活后门。

## 实验关键数据

### 实验设置
- 数据集：ODinW-13 中的 6 个数据集（Vehicles/Aquarium/Aerial Drone/Shellfish/Thermal/Mushrooms）
- 主模型：Grounding DINO (MM-Grounding-Dino-Tiny)，扩展至 GLIP-T
- 硬件：单卡 V100 32GB，batch size=4，训练 15 epoch

### OMA 结果（表 1）

| 数据集 | 零样本 BmAP | TrAP BmAP | TrAP PmAP | TrAP ASR |
|--------|-----------|-----------|-----------|----------|
| Vehicles | 61.5 | 64.9 | 15.2 | 0.79 |
| Aquarium | 28.3 | 48.0 | 17.3 | 0.88 |
| Aerial Drone | 15.1 | 46.0 | 9.6 | 0.83 |
| Thermal | 54.2 | 78.2 | 55.0 | 0.92 |
| Mushrooms | 65.8 | 90.2 | 82.3 | 1.00 |

### ODA 结果（表 2）
TrAP 在所有数据集上 ASR 达 0.90-1.00，同时 BAP 显著高于零样本基线。

### 关键对比
- **CoCoOp-Det（仅文本）**：适配能力好但攻击成功率极低（ASR 多 <0.15），因文本 prompt 无法有效关联图像空间的触发器
- **VPT（仅视觉）**：攻击能力中等，但适配下游任务的能力弱于文本 prompt
- **TrAP（双模态）**：兼顾两者优势，ASR 和 BmAP 均最高

### 防御评估（表 5，Vehicles 数据集）
- PatchDrop (50%)：BmAP 降至 43.9，ASR 仍有 0.63，防御代价过大
- Prompt Engineering（Bus→A Bus）：ASR 降至 0.04，但具体效果依赖替换词选择
- PAD 对抗 patch 防御：甚至反效果，ASR 从 0.48 升至 0.50

### GLIP 模型迁移（表 4）
TrAP 在 GLIP-T 上同样有效，Aquarium 数据集 ASR 达 0.96，验证方法的通用性。

## 亮点

1. **首个 OVOD 后门攻击研究**：揭示了 prompt tuning 在开放词汇检测器上引入的新攻击面
2. **多模态联合攻击设计有理论直觉**：视觉 prompt 负责关联触发器，文本 prompt 负责下游适配，两者互补
3. **课程学习缩小触发器**：巧妙解决了小 patch 梯度信号弱的问题，渐进训练策略简单有效
4. **极低参数开销**：仅 0.2M 参数（prompt + meta-net），远低于 fine-tuning 的 21-36M，却能实现更高 ASR
5. **同时提升干净数据性能**：被攻击模型的 BmAP 显著高于零样本基线，增加了后门的隐蔽性

## 局限与展望

- 仅在 6 个相对小规模的 ODinW 数据集上验证，缺少大规模通用数据集（如 COCO/LVIS）的评估
- 防御分析较浅，只测试了 3 种推理时防御，未探索 prompt-level 或 feature-level 的专用防御
- Prompt Engineering 防御（Bus→A Bus）可大幅降低 ASR，但论文未深入分析原因或提出改进
- 触发器为固定 patch，未考虑更隐蔽的触发形式（如频域扰动、自然物体触发）
- 仅关注攻击，未提供任何防御方案建议

## 与相关工作的对比

| 方法 | 目标模型 | 攻击方式 | 是否需修改权重 | 多模态 |
|------|---------|---------|-------------|-------|
| Chan et al. 2022 | 闭集检测器 | 数据投毒 | 是 | 否 |
| Bai et al. 2024 (BadCLIP) | CLIP 分类 | 文本 prompt | 否 | 部分 |
| SWARM (Yang 2024) | ViT 分类 | 视觉 prompt | 否 | 否 |
| **TrAP (本文)** | **OVOD 检测** | **视觉+文本 prompt** | **否** | **是** |

本文的核心贡献在于将后门攻击从分类任务和闭集检测扩展到开放词汇检测器，且首次利用双模态 prompt 的互补性进行攻击。

## 启发与关联

- **防御研究空白**：本文揭示的攻击面（prompt tuning）具有高度现实性，开发针对性防御（如 prompt 审计、异常检测）是重要后续方向
- **Prompt tuning 安全性**：用户外包模型适配时，需要审查返回的 prompt 参数；可考虑引入 prompt 水印或可信 prompt 供应链
- **与自动驾驶安全的关联**：论文场景（攻击者在救护车上贴触发器使其不被检测）直接与车辆安全相关，凸显 OVOD 部署前安全评估的必要性
- **多模态攻击范式**：双模态联合优化的思路可推广到其他视觉-语言模型的安全研究（如 VLM grounding、visual QA）

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首个 OVOD 后门攻击，问题定义清晰
- 实验充分度: ⭐⭐⭐ — 消融全面，但数据集偏小且缺少大规模评估
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐ — 揭示重要安全隐患，对 OVOD 部署有直接警示意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] MOBA: A Material-Oriented Backdoor Attack against LiDAR-based 3D Object Detection](moba_a_material-oriented_backdoor_attack_against_lidar-based_3d_object_detection.md)
- [\[CVPR 2026\] Open-Vocabulary Domain Generalization in Urban-Scene Segmentation](../../CVPR2026/autonomous_driving/open-vocabulary_domain_generalization_in_urban-scene_segmentation.md)
- [\[CVPR 2026\] Monocular Open Vocabulary Occupancy Prediction for Indoor Scenes (LegoOcc)](../../CVPR2026/autonomous_driving/monocular_open_vocabulary_occupancy_prediction_for_indoor_scenes.md)
- [\[AAAI 2026\] Hierarchical Prompt Learning for Image- and Text-Based Person Re-Identification](hierarchical_prompt_learning_for_image-_and_text-based_person_re-identification.md)
- [\[CVPR 2025\] O3N: Omnidirectional Open-Vocabulary Occupancy Prediction](../../CVPR2025/autonomous_driving/o3n_omnidirectional_open-vocabulary_occupancy_prediction.md)

</div>

<!-- RELATED:END -->
