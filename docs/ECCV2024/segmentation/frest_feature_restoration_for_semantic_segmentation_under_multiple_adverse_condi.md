---
title: >-
  [论文解读] FREST: Feature Restoration for Semantic Segmentation under Multiple Adverse Conditions
description: >-
  [ECCV 2024][语义分割][图像分割] 提出 FREST，一种面向多种恶劣条件（雾、雨、雪、夜间）的源无关域自适应语义分割框架，通过交替学习条件嵌入空间（分离条件信息）和特征恢复（将恶劣条件特征恢复为正常条件），逐步消除恶劣条件对特征的影响，在 ACDC 和 RobotCar 基准上均达到新的 SOTA。
tags:
  - "ECCV 2024"
  - "语义分割"
  - "图像分割"
  - "feature restoration"
  - "域适应"
  - "adverse conditions"
  - "robustness"
---

# FREST: Feature Restoration for Semantic Segmentation under Multiple Adverse Conditions

**会议**: ECCV 2024  
**arXiv**: [2407.13437](https://arxiv.org/abs/2407.13437)  
**代码**: [https://sohyun-l.github.io/frest](https://sohyun-l.github.io/frest)  
**领域**: 图像分割  
**关键词**: semantic segmentation, feature restoration, source-free domain adaptation, adverse conditions, robustness

## 一句话总结

提出 FREST，一种面向多种恶劣条件（雾、雨、雪、夜间）的源无关域自适应语义分割框架，通过交替学习条件嵌入空间（分离条件信息）和特征恢复（将恶劣条件特征恢复为正常条件），逐步消除恶劣条件对特征的影响，在 ACDC 和 RobotCar 基准上均达到新的 SOTA。

## 研究背景与动机

**领域现状**：语义分割在正常条件下表现优异，但在雾、雨、雪、夜间等恶劣条件下性能显著下降，严重限制了自动驾驶等高安全要求场景的应用。为解决标注数据难以获取的问题，研究者转向无监督域自适应（UDA）和更实际的源无关域自适应（SFDA）。

**问题设定（SFDA to adverse conditions）**：分割模型先在有标注的源域（如 Cityscapes 正常天气）预训练，然后仅使用无标注的目标域数据进行微调。目标域包含多种恶劣条件图像，每张恶劣条件图像 $I_{\text{adv}}$ 都有一张通过 GNSS 匹配的正常条件参考图像 $I_{\text{norm}}$，但两者仅粗略对齐（存在视角和时间差异），且均无标注，条件类型未知。

**现有方法的不足**：先前工作 CMA 通过对比学习鼓励恶劣/正常图像对的特征接近，学习条件不变特征。但这存在两个核心问题：
   - **灾难性遗忘**：将正常图像特征向恶劣图像特征拉近会导致模型遗忘源域知识，因为正常图像本身就接近源域
   - **对齐依赖**：特征匹配严重依赖图像对的精确对齐，但 GNSS 匹配只能提供粗略对齐，动态物体和不完美变形会引入内容不匹配

**核心idea**：与其让两种条件的特征互相靠近，不如**单向**地将恶劣条件特征恢复为正常条件特征。关键在于利用"条件特有信息"——仅依赖图像条件而不受语义内容影响的信息——来引导恢复过程，从而避免灾难性遗忘并减少内容不匹配的影响。

**切入角度**：将恶劣条件视为"有害噪声"，目标是在特征空间中去除它们的影响。通过设计一个仅捕捉条件信息的嵌入空间，在该空间中进行特征恢复，确保恢复过程只考虑条件差异而不受语义内容差异干扰。

## 方法详解

### 整体框架

FREST 交替执行两个步骤，每个训练迭代中：
- **Step 1**：冻结分割网络，训练条件过滤器（Condition Strainer）和投影头，学习条件嵌入空间
- **Step 2**：冻结条件过滤器和投影头，训练分割网络，在条件嵌入空间中进行特征恢复

推理时仅使用编码器 $\phi_{\text{enc}}$ 和解码器 $\phi_{\text{dec}}$，不需要条件过滤器等额外模块。

### 关键设计

#### 1. 条件过滤器（Condition Strainer）—— 分离条件信息

**功能**：从编码器特征中提取仅与图像条件相关、与语义内容无关的条件特有信息。

**核心思路**：受参数高效微调（如 Adapter）启发，在冻结的分割编码器 $\phi_{\text{enc}}$ 每一层旁附加一个小型模块 $\psi_{\text{strainer}}$，通过残差连接生成"条件注入特征"：

$$\mathbf{c}^l = \phi_{\text{enc}}^l(\mathbf{c}^{l-1}) + \psi_{\text{strainer}}^l(\mathbf{c}^{l-1})$$

其中 $\mathbf{c}^l$ 为第 $l$ 层的条件注入特征。条件过滤器与编码器分离设计，使编码器不被条件信息污染。最终条件注入特征 $\mathbf{c}$ 通过投影头 $\psi_{\text{proj}}$ 映射到条件嵌入空间。

**设计动机**：编码器已经被源域预训练冻结，保留了丰富的语义知识。条件过滤器仅添加少量参数（2.1M，占基线 2.6%），捕捉编码器无法表达的条件特有信息，这种设计既高效又避免了对源域知识的破坏。

#### 2. 条件嵌入空间学习（Step 1）—— 对比学习分离条件

**功能**：学习一个嵌入空间，其中相同条件的图像聚在一起，不同条件的图像分开。

**核心思路**：使用对比学习，锚点和正样本具有相同条件但不同语义，锚点和负样本具有相似语义但不同条件：

$$\mathcal{L}_{\text{spec},i} = -\log \frac{\exp(\mathbf{z}_{\text{adv}}^{i\top} \mathbf{z}_{\text{adv}}^{*} / \tau)}{\exp(\mathbf{z}_{\text{adv}}^{i\top} \mathbf{z}_{\text{adv}}^{*} / \tau) + \exp(\mathbf{z}_{\text{adv}}^{i\top} \mathbf{z}_{\text{norm}}^{i} / \tau)}$$

其中锚点 $\mathbf{z}_{\text{adv}}^i$ 为恶劣条件的条件嵌入，负样本 $\mathbf{z}_{\text{norm}}^i$ 为对应正常条件的条件嵌入（通过 warping 对齐的 patch 对），正样本 $\mathbf{z}_{\text{adv}}^*$ 从正样本队列中选择与锚点**最相似**的恶劣条件嵌入（假设共享相同条件）。

**设计动机**：通过要求锚点和负样本在语义上相似但条件不同（因为是同一位置的恶劣/正常图像 patch 对），迫使嵌入空间只关注条件差异。正样本选择最相似的策略确保正样本与锚点共享相同的恶劣条件类型，帮助模型学习更精细的条件区分。

#### 3. 特征恢复（Step 2）—— 在条件空间中消除恶劣影响

**功能**：训练分割网络使恶劣条件图像的编码器特征 $\mathbf{f}_{\text{adv}}$ 在条件嵌入空间中近似正常条件的条件注入特征 $\mathbf{c}_{\text{norm}}$。

**核心思路 — 特征恢复损失**：将 $\mathbf{f}_{\text{adv}}$ 投影到条件嵌入空间后，用 $\ell_1$ 回归损失使其逼近 $\mathbf{z}_{\text{norm}}$：

$$\mathcal{L}_{\text{resto}} = \frac{1}{|\mathcal{W}|} \sum_{i \in \mathcal{W}} |\psi_{\text{proj}}(\mathbf{f}_{\text{adv}}^i) - \mathbf{z}_{\text{norm}}^i|$$

注意梯度不从 $\mathbf{c}_{\text{norm}}$ 回传，确保恶劣条件特征单向向正常条件靠拢。

**辅助设计 — 恶劣条件判别损失**：引入 MLP 判别器 $D$ 区分编码器特征 $\mathbf{f}_{\text{adv}}^{l,j}$ 和条件注入特征 $\mathbf{c}_{\text{adv}}^{l,j}$，进一步推动编码器特征远离恶劣条件信息：

$$\mathcal{L}_{\text{dis}} = -\frac{1}{|\mathcal{A}|} \sum_{j \in \mathcal{A}} \sum_{l=1}^{L} \{\lambda \log(D(\mathbf{f}_{\text{adv}}^{l,j})) + (1-\lambda) \log(1 - D(\mathbf{c}_{\text{adv}}^{l,j}))\}$$

**设计动机**：在条件嵌入空间中进行恢复，仅考虑条件信息而忽略语义内容差异。这解决了图像对不精确对齐的问题——即使两张图的内容不完全一致，只要条件信息正确恢复即可。判别损失提供了额外的梯度信号，从每一层推动特征去除恶劣条件的特征模式。

### 损失函数 / 训练策略

**Step 1 总损失**：$\mathcal{L}_{\text{step1}} = \lambda_{\text{spec}} \mathcal{L}_{\text{spec}} + \mathcal{L}_{\text{self}}$

**Step 2 总损失**：$\mathcal{L}_{\text{step2}} = \mathcal{L}_{\text{resto}} + \lambda_{\text{dis}} \mathcal{L}_{\text{dis}} + \mathcal{L}_{\text{self}} + \lambda_{\text{ent}} \mathcal{L}_{\text{ent}}$

其中 $\mathcal{L}_{\text{self}}$ 为伪标签自训练损失（CBST），$\mathcal{L}_{\text{ent}}$ 为熵最小化损失。两步交替执行，条件过滤器适应分割网络的更新，从而改进下一轮的特征恢复。

超参数：$\lambda_{\text{spec}} = 0.01$，$\lambda_{\text{ent}} = 0.01$，$\lambda_{\text{dis}} = 5 \times 10^{-5}$，$\tau = 0.7$。训练 8 个 epoch，前 2 个 epoch 仅训练条件过滤器预热。

## 实验关键数据

### 主实验

**Cityscapes → ACDC（SFDA 设定）**

| 方法 | mIoU (%) | 与 Source model 对比 |
|------|----------|---------------------|
| Source model (SegFormer) | 59.4 | - |
| HCL | 60.8 | +1.4 |
| URMA | 65.3 | +5.9 |
| URMA + SimT | 65.7 | +6.3 |
| CMA | 69.1 | +9.7 |
| **FREST (Ours)** | **70.7** | **+11.3** |

FREST 在该基准上相比前 SOTA CMA 提升 1.6%，尤其在精细物体（car +2.9, truck +6.5, bus +2.8）上提升显著。

**Cityscapes → RobotCar（SFDA 设定，8种恶劣条件）**

| 方法 | mIoU (%) |
|------|----------|
| Source model | 50.0 |
| HCL | 50.1 |
| URMA | 51.6 |
| CMA | 54.3 |
| **FREST (Ours)** | **58.8** |

RobotCar 包含更多条件类型（dawn, dusk, night, night-rain, overcast, rain, snow, sun），FREST 相比 CMA 大幅提升 4.5%，说明条件越多样特征恢复方法越有效。

**与 UDA 方法对比（Cityscapes → ACDC）**

| 方法 | 是否 Source-free | mIoU (%) |
|------|------------------|----------|
| Refign (UDA) | 否 | 65.5 |
| HRDA (UDA) | 否 | 68.0 |
| HRDA + MIC (UDA) | 否 | 70.4 |
| CMA (SFDA) | 是 | 69.1 |
| **FREST (SFDA)** | **是** | **70.7** |

FREST 作为 SFDA 方法，在不访问源域标注数据的情况下超越了所有 UDA 方法。

### 消融实验

**损失函数消融（Step 1 与 Step 2）**

| $\mathcal{L}_{\text{self}}$ | $\mathcal{L}_{\text{spec}}$ | mIoU (%) | 说明 |
|:---:|:---:|---:|------|
| | | 64.3 | 无损失基线 |
| ✓ | | 64.8 | 仅自训练 |
| ✓ | ✓ | 68.6 | Step 1 完整，条件学习贡献 +3.8 |

| $\mathcal{L}_{\text{resto}}$ | $\mathcal{L}_{\text{dis}}$ | mIoU (%) | 说明 |
|:---:|:---:|---:|------|
| | | 62.7 | 无恢复损失 |
| ✓ | | 67.2 | 恢复损失贡献 +4.5 |
| ✓ | ✓ | 68.6 | 判别损失额外贡献 +1.4 |

**训练策略与结构消融**

| 条件过滤器 | 分割网络 | 训练策略 | mIoU (%) |
|:---:|:---:|---:|---:|
| | ✓ | 自训练 | 62.7 |
| ✓ | | Adapter微调 | 63.1 |
| ✓ | ✓ | 全参数微调 | 63.2 |
| ✓ | ✓ | **FREST交替训练** | **68.6** |

关键发现：朴素使用 Adapter 微调仅提升 0.4%，全参数微调提升 0.5%，FREST 交替训练策略贡献了 5.9% 的巨大提升。

### 关键发现

1. **特征恢复效果可视化**：从恢复后的特征重建图像显示，夜晚天空变蓝、雪地树木变绿，证实特征恢复有效模拟了正常条件
2. **参数高效**：条件过滤器仅 2.1M 参数（基线 2.6%），投影头 1.2M（1.5%），推理时无需额外参数
3. **正样本选择策略**：选择最相似正样本（HIGHEST）达 68.6%，显著优于随机选择（62.4%）和最不相似选择（56.6%）
4. **恢复特征 vs 条件注入特征**：推理时用恢复特征 $\mathbf{f}_{\text{adv}}$ 达 68.6%，用条件注入特征 $\mathbf{c}_{\text{adv}}$ 仅 59.0%，说明恢复过程成功去除了恶劣条件信息
5. **泛化性**：在未见数据集 ACG 上 FREST 达 52.6% mIoU（CMA 51.3%），在正常条件 Cityscapes-lindau40 上达 72.5%（与源域训练模型持平），说明特征恢复不损害正常条件性能

## 亮点与洞察

1. **单向恢复 vs 双向对齐**：核心洞察是将恶劣→正常的转换设为单向过程，避免正常特征被恶劣特征"污染"导致灾难性遗忘
2. **条件空间中恢复**：在仅包含条件信息的嵌入空间中操作，优雅地绕开了图像对不精确对齐的问题
3. **交替训练的互促效应**：两步交替训练使条件空间和特征恢复互相改进——更好的条件空间引导更好的恢复，更好的恢复后的特征帮助学习更精准的条件空间
4. **参数高效设计**：条件过滤器借鉴 Adapter 结构，推理时可完全去除，零额外推理成本

## 局限与展望

1. 目前仅覆盖天气/光照等自然恶劣条件，未涉及图像退化（模糊、噪声）和相机伪影等更广泛的退化类型
2. 条件过滤器依赖 warping 对齐质量，warping 失败区域（置信度 < 0.2）无法参与学习
3. 正样本队列假设最相似的嵌入共享相同条件，当条件分布高度混合时可能不成立
4. 未探索多种恶劣条件同时出现的复合场景（如雪+夜间）

## 相关工作与启发

- **CMA [Brüggemann et al.]**：SFDA to adverse conditions 的先驱工作，提出条件不变学习，FREST 是对其单向恢复的改进
- **Parameter-efficient Fine-tuning**：Adapter/LoRA 等结构启发了条件过滤器的设计
- **对比学习**：条件特有学习的正/负样本构造策略值得借鉴——利用 warping 对齐获得语义相似但条件不同的样本对

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 特征恢复思路新颖，单向恢复+条件嵌入空间的设计有效解决了现有方法的核心缺陷
- **实验充分度**: ⭐⭐⭐⭐⭐ — 两个基准 SOTA + UDA 对比 + 泛化实验 + 详尽消融 + 可视化分析
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，方法描述详细，图示信息丰富
- **价值**: ⭐⭐⭐⭐ — 对自动驾驶等安全关键场景有实际价值，SFDA 设定非常实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions](../../CVPR2026/segmentation/heuristic_self-paced_learning_for_domain_adaptive_semantic_segmentation_under_ad.md)
- [\[ECCV 2024\] Self-supervised Co-salient Object Detection via Feature Correspondences at Multiple Scales](self-supervised_co-salient_object_detection_via_feature_correspondences_at_multi.md)
- [\[ECCV 2024\] Eliminating Feature Ambiguity for Few-Shot Segmentation](eliminating_feature_ambiguity_for_few-shot_segmentation.md)
- [\[ECCV 2024\] Representing Topological Self-Similarity Using Fractal Feature Maps for Accurate Segmentation of Tubular Structures](representing_topological_self-similarity_using_fractal_feature_maps_for_accurate.md)
- [\[ECCV 2024\] VISA: Reasoning Video Object Segmentation via Large Language Models](visa_reasoning_video_object_segmentation_via_large_language_models.md)

</div>

<!-- RELATED:END -->
